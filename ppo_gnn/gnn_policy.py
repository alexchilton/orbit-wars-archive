"""GNN policy network for Orbit Wars with factored action distribution.

The action is decomposed into three sequential decisions:

    1. Source selection — pick which owned planet to launch from (or noop).
       Logits: one per planet + one noop logit. Masked to owned planets with ships.

    2. Target selection — pick which planet to send the fleet to, conditioned
       on the chosen source. Logits computed from (source_embed, target_embed,
       edge_feat, global_context). Masked to exclude the source planet itself,
       and optionally masked by sun intersection.

    3. Fraction selection — pick what fraction of ships to send {25%, 50%, 75%, 100%},
       conditioned on source and target.

Joint log-probability:
    log p(action) = log p(source) + log p(target | source) + log p(fraction | source, target)

When noop is selected, target/fraction log-probs are 0.

The backbone is a 2-layer GAT (Graph Attention Network) with edge-conditioned
attention, switchable to GraphSAGE via a flag.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sun_geometry import compute_sun_edge_features_batch

NODE_DIM = 28
EDGE_DIM = 6  # distance, travel_time, angle_sin, angle_cos, sun_intersects, sun_clearance
FRACTION_BUCKETS = [0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# GAT layer (edge-conditioned, multi-head)
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """GAT layer with edge features in both attention AND messages.

    Standard GAT only uses edges for attention weights. This version also
    incorporates edge features into the value/message, allowing the model
    to learn spatial reasoning like "planet j is close AND has lots of ships".
    """
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, num_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.dropout = dropout

        # Query/Key projections for attention
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.a_tgt = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.a_edge = nn.Linear(edge_dim, num_heads, bias=False)

        # Edge-to-message projection: edge features become part of the value
        self.edge_val = nn.Linear(edge_dim, out_dim, bias=False)

        self.norm = nn.LayerNorm(out_dim)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_tgt.unsqueeze(0))

    def forward(self, h: torch.Tensor, edge_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, N, in_dim)
            edge_feats: (B, N, N, edge_dim)
        Returns:
            (B, N, out_dim)
        """
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.num_heads, self.head_dim)  # (B, N, H, D)

        # Attention scores
        score_src = (Wh * self.a_src).sum(-1)  # (B, N, H)
        score_tgt = (Wh * self.a_tgt).sum(-1)  # (B, N, H)
        score_edge = self.a_edge(edge_feats)    # (B, N, N, H)

        attn = score_src.unsqueeze(2) + score_tgt.unsqueeze(1) + score_edge
        attn = F.leaky_relu(attn, 0.2)
        attn = F.softmax(attn, dim=2)  # normalize over j (neighbors)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        # Value = neighbor node embedding + edge embedding
        edge_v = self.edge_val(edge_feats).view(B, N, N, self.num_heads, self.head_dim)
        Wh_msg = Wh.unsqueeze(1) + edge_v  # (B, N_i, N_j, H, D) — message from j to i

        # Aggregate: weighted sum of messages
        out = (attn.unsqueeze(-1) * Wh_msg).sum(dim=2)  # (B, N, H, D)
        out = out.reshape(B, N, -1)  # (B, N, out_dim)

        return self.norm(out + self.residual(h))


# ---------------------------------------------------------------------------
# GraphSAGE layer (mean aggregator, edge-conditioned)
# ---------------------------------------------------------------------------

class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, edge_feats: torch.Tensor) -> torch.Tensor:
        B, N, _ = h.shape
        gate = self.edge_gate(edge_feats).squeeze(-1)  # (B, N, N)
        # Weighted mean of neighbors
        h_neigh = (gate.unsqueeze(-1) * h.unsqueeze(1)).sum(dim=2)  # (B, N, D_in)
        h_neigh = h_neigh / gate.sum(dim=2, keepdim=True).clamp(min=1.0)
        out = F.relu(self.W_self(h) + self.W_neigh(h_neigh))
        return self.norm(out)


# ---------------------------------------------------------------------------
# Full policy
# ---------------------------------------------------------------------------

class OrbitWarsGNNPolicy(nn.Module):
    """GNN policy with factored action head for Orbit Wars.

    Args:
        hidden_dim: Hidden dimension for node embeddings. 64 or 128.
        use_gat: If True, use GAT layers. If False, use GraphSAGE.
        num_gat_heads: Number of attention heads for GAT.
        mask_sun_targets: If True, hard-mask targets where path crosses sun.
        sun_safety_margin: Extra radius around sun for danger zone.
        ship_speed: Default ship speed for travel time computation.
        separate_critic: If True, use a separate GNN backbone for the value
            function so policy and value gradients don't interfere.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        use_gat: bool = True,
        num_gat_heads: int = 4,
        mask_sun_targets: bool = False,
        sun_safety_margin: float = 2.0,
        ship_speed: float = 6.0,
        separate_critic: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_gat = use_gat
        self.mask_sun_targets = mask_sun_targets
        self.sun_safety_margin = sun_safety_margin
        self.ship_speed = ship_speed
        self.separate_critic = separate_critic

        # Encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(NODE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(EDGE_DIM, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        # GNN layers
        edge_enc_dim = hidden_dim // 2
        if use_gat:
            self.gnn1 = GATLayer(hidden_dim, hidden_dim, edge_enc_dim, num_gat_heads)
            self.gnn2 = GATLayer(hidden_dim, hidden_dim, edge_enc_dim, num_gat_heads)
        else:
            self.gnn1 = GraphSAGELayer(hidden_dim, hidden_dim, edge_enc_dim)
            self.gnn2 = GraphSAGELayer(hidden_dim, hidden_dim, edge_enc_dim)

        # Global context (mean pool -> project)
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)

        # --- Separate critic backbone (own GNN, not shared with policy) ---
        if separate_critic:
            self.critic_node_encoder = nn.Sequential(
                nn.Linear(NODE_DIM, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.critic_edge_encoder = nn.Sequential(
                nn.Linear(EDGE_DIM, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
            )
            if use_gat:
                self.critic_gnn1 = GATLayer(hidden_dim, hidden_dim, edge_enc_dim, num_gat_heads)
                self.critic_gnn2 = GATLayer(hidden_dim, hidden_dim, edge_enc_dim, num_gat_heads)
            else:
                self.critic_gnn1 = GraphSAGELayer(hidden_dim, hidden_dim, edge_enc_dim)
                self.critic_gnn2 = GraphSAGELayer(hidden_dim, hidden_dim, edge_enc_dim)
            self.critic_global_proj = nn.Linear(hidden_dim, hidden_dim)

        # Source head: per-node logit + noop logit
        self.source_head = nn.Linear(hidden_dim, 1)
        self.noop_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Bias noop down so the agent prefers acting early in training
        with torch.no_grad():
            self.noop_head[-1].bias.fill_(-2.0)

        # Target head: (source_embed, target_embed, edge_enc, global) -> 1
        target_in = hidden_dim + hidden_dim + edge_enc_dim + hidden_dim
        self.target_head = nn.Sequential(
            nn.Linear(target_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Fraction head: (source_embed, target_embed, global) -> 4
        frac_in = hidden_dim + hidden_dim + hidden_dim
        self.fraction_head = nn.Sequential(
            nn.Linear(frac_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(FRACTION_BUCKETS)),
        )

        # Value head: global context -> scalar
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def compute_edge_features(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute raw 6-dim edge features from planet positions.

        Args:
            positions: (B, N, 2)

        Returns:
            (B, N, N, 6): distance, travel_time, angle_sin, angle_cos, sun_intersects, sun_clearance
        """
        B, N, _ = positions.shape
        board_diag = math.sqrt(100**2 + 100**2)

        p1 = positions.unsqueeze(2)  # (B, N, 1, 2)
        p2 = positions.unsqueeze(1)  # (B, 1, N, 2)
        diff = p2 - p1  # (B, N, N, 2)

        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)  # (B, N, N, 1)
        dist_norm = dist / board_diag
        travel_time = dist / self.ship_speed

        angle = torch.atan2(diff[..., 1:2], diff[..., 0:1])  # (B, N, N, 1)
        angle_sin = torch.sin(angle)
        angle_cos = torch.cos(angle)

        sun_inter, sun_clear = compute_sun_edge_features_batch(
            positions, safety_margin=self.sun_safety_margin,
        )

        return torch.cat([
            dist_norm,
            travel_time,
            angle_sin,
            angle_cos,
            sun_inter.unsqueeze(-1),
            sun_clear.unsqueeze(-1),
        ], dim=-1)

    def _encode(
        self, node_features: torch.Tensor, positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run node encoder + GNN layers + global pooling (policy backbone).

        Returns:
            h: (B, N, hidden_dim) — node embeddings after message passing
            edge_enc: (B, N, N, hidden_dim//2) — encoded edge features
            g: (B, hidden_dim) — global context vector
        """
        h = self.node_encoder(node_features)
        raw_edges = self.compute_edge_features(positions)
        edge_enc = self.edge_encoder(raw_edges)

        h = self.gnn1(h, edge_enc)
        h = self.gnn2(h, edge_enc)

        g = h.mean(dim=1)  # (B, hidden_dim)
        g = F.relu(self.global_proj(g))

        return h, edge_enc, g

    def _critic_encode(
        self, node_features: torch.Tensor, positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run critic backbone (separate GNN) and return global context.

        Falls back to shared backbone if separate_critic is False.
        """
        if not self.separate_critic:
            _, _, g = self._encode(node_features, positions)
            return g

        h = self.critic_node_encoder(node_features)
        raw_edges = self.compute_edge_features(positions)
        edge_enc = self.critic_edge_encoder(raw_edges)

        h = self.critic_gnn1(h, edge_enc)
        h = self.critic_gnn2(h, edge_enc)

        g = h.mean(dim=1)
        g = F.relu(self.critic_global_proj(g))
        return g

    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        owned_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass returning all logits needed for action sampling.

        Args:
            node_features: (B, N, 10)
            positions: (B, N, 2)
            owned_mask: (B, N) — 1.0 for planets the agent owns with ships > 0

        Returns:
            source_logits: (B, N+1) — last index is noop
            all_target_logits: (B, N, N) — target logits for every possible source
            all_fraction_logits: (B, N, N, 4) — fraction logits for every (source, target)
        """
        B, N, _ = node_features.shape
        h, edge_enc, g = self._encode(node_features, positions)

        # --- Source logits ---
        src_logits = self.source_head(h).squeeze(-1)  # (B, N)
        noop_logit = self.noop_head(g)  # (B, 1)
        # Mask: only owned planets are valid sources
        src_logits = src_logits.masked_fill(owned_mask == 0, -1e4)
        source_logits = torch.cat([src_logits, noop_logit], dim=-1)  # (B, N+1)

        # --- Target logits for all sources ---
        # h_s: (B, N_src, 1, D), h_t: (B, 1, N_tgt, D)
        h_s = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_t = h.unsqueeze(1).expand(-1, N, -1, -1)
        g_exp = g.unsqueeze(1).unsqueeze(1).expand(-1, N, N, -1)
        target_input = torch.cat([h_s, h_t, edge_enc, g_exp], dim=-1)  # (B, N, N, D_target)
        all_target_logits = self.target_head(target_input).squeeze(-1)  # (B, N, N)

        # Mask self-loops (can't send to yourself)
        self_mask = torch.eye(N, device=h.device, dtype=torch.bool).unsqueeze(0)
        all_target_logits = all_target_logits.masked_fill(self_mask, -1e4)

        # Optional: mask sun-blocked paths
        if self.mask_sun_targets:
            sun_inter, _ = compute_sun_edge_features_batch(
                positions, safety_margin=self.sun_safety_margin,
            )
            all_target_logits = all_target_logits.masked_fill(sun_inter.bool(), -1e4)

        # --- Fraction logits for all (source, target) pairs ---
        frac_input = torch.cat([h_s, h_t, g_exp], dim=-1)  # (B, N, N, 3*D)
        all_fraction_logits = self.fraction_head(frac_input)  # (B, N, N, 4)

        return source_logits, all_target_logits, all_fraction_logits

    def bc_forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        owned_mask: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
        num_planets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Efficient forward for BC/AWR training — avoids computing all N^2 target logits.

        Only computes target logits for the specific source row, and fraction logits
        for the specific (source, target) pair. Much faster than full forward().

        Args:
            source_idx: (B,) — source planet index (clamped for noop)
            target_idx: (B,) — target planet index (clamped for noop)
            num_planets: (B,) — actual planet count per sample (for masking padding)

        Returns:
            source_logits: (B, N+1)
            target_logits_for_src: (B, N) — target logits for the given source only
            fraction_logits: (B, 4) — fraction logits for the given (source, target) only
        """
        B, N, _ = node_features.shape
        h, edge_enc, g = self._encode(node_features, positions)

        # Build valid mask (planets that actually exist, not zero-padding)
        if num_planets is not None:
            idx = torch.arange(N, device=node_features.device).unsqueeze(0)  # (1, N)
            valid_mask = (idx < num_planets.unsqueeze(1)).float()  # (B, N)
        else:
            valid_mask = torch.ones(B, N, device=node_features.device)

        # Source logits: mask both unowned and padded planets
        src_logits = self.source_head(h).squeeze(-1)
        noop_logit = self.noop_head(g)
        source_mask = (owned_mask == 0) | (valid_mask == 0)
        src_logits = src_logits.masked_fill(source_mask, -1e4)
        source_logits = torch.cat([src_logits, noop_logit], dim=-1)  # (B, N+1)

        # Target logits for the specific source only
        src_clamped = source_idx.clamp(0, N - 1).long()
        batch_idx = torch.arange(B, device=h.device)
        h_s = h[batch_idx, src_clamped]  # (B, D)
        h_s_exp = h_s.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)
        edge_for_src = edge_enc[batch_idx, src_clamped]  # (B, N, edge_dim)
        g_exp = g.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)
        target_input = torch.cat([h_s_exp, h, edge_for_src, g_exp], dim=-1)  # (B, N, D_target)
        target_logits = self.target_head(target_input).squeeze(-1)  # (B, N)

        # Mask self-loop and padded planets
        target_logits[batch_idx, src_clamped] = -1e4
        target_logits = target_logits.masked_fill(valid_mask == 0, -1e4)

        if self.mask_sun_targets:
            sun_inter, _ = compute_sun_edge_features_batch(
                positions, safety_margin=self.sun_safety_margin,
            )
            sun_mask = sun_inter[batch_idx, src_clamped]  # (B, N)
            target_logits = target_logits.masked_fill(sun_mask.bool(), -1e4)

        # Fraction logits for specific (source, target) pair
        tgt_clamped = target_idx.clamp(0, N - 1).long()
        h_t = h[batch_idx, tgt_clamped]  # (B, D)
        frac_input = torch.cat([h_s, h_t, g], dim=-1)  # (B, 3*D)
        fraction_logits = self.fraction_head(frac_input)  # (B, 4)

        return source_logits, target_logits, fraction_logits

    def get_value(self, node_features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Compute state value only (for critic updates).

        Returns:
            (B,) value estimates
        """
        g = self._critic_encode(node_features, positions)
        return self.value_head(g).squeeze(-1)

    def evaluate_action(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        owned_mask: torch.Tensor,
        action_source: torch.Tensor,
        action_target: torch.Tensor,
        action_fraction: torch.Tensor,
        action_is_noop: torch.Tensor,
        noop_penalty: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log-prob, entropy, and value for a batch of known actions.

        Used by PPO and AWR for computing the policy gradient.

        Args:
            action_source: (B,) — source planet index (ignored if noop)
            action_target: (B,) — target planet index (ignored if noop)
            action_fraction: (B,) — fraction bucket index (ignored if noop)
            action_is_noop: (B,) — 1.0 if noop
            noop_penalty: logit penalty subtracted from noop — MUST match
                the value used during rollout so PPO ratios are consistent.

        Returns:
            log_prob: (B,)
            entropy: (B,)
            value: (B,)
        """
        B, N, _ = node_features.shape
        source_logits, all_target_logits, all_fraction_logits = self.forward(
            node_features, positions, owned_mask,
        )
        # Use critic backbone for value (separate from policy if configured)
        g_critic = self._critic_encode(node_features, positions)
        value = self.value_head(g_critic).squeeze(-1)

        # Apply same noop penalty as rollout so log-probs match
        if noop_penalty != 0.0:
            source_logits = source_logits.clone()
            source_logits[:, N] -= noop_penalty

        # Source log-prob
        source_dist = torch.distributions.Categorical(logits=source_logits)
        # For noop, the "source" is index N (the noop slot)
        effective_source = torch.where(
            action_is_noop.bool(),
            torch.full_like(action_source, N),
            action_source,
        )
        log_p_source = source_dist.log_prob(effective_source)  # (B,)
        entropy_source = source_dist.entropy()

        # Target log-prob (only for non-noop)
        # Gather the target logits for the selected source
        src_idx = action_source.clamp(0, N - 1).long()
        target_logits_for_src = all_target_logits[torch.arange(B), src_idx]  # (B, N)
        target_dist = torch.distributions.Categorical(logits=target_logits_for_src)
        log_p_target = target_dist.log_prob(action_target.clamp(0, N - 1).long())
        entropy_target = target_dist.entropy()

        # Fraction log-prob (only for non-noop)
        frac_logits = all_fraction_logits[torch.arange(B), src_idx, action_target.clamp(0, N - 1).long()]  # (B, 4)
        frac_dist = torch.distributions.Categorical(logits=frac_logits)
        log_p_fraction = frac_dist.log_prob(action_fraction.clamp(0, 3).long())
        entropy_fraction = frac_dist.entropy()

        # Joint log-prob: noop only uses source, launch uses all three
        is_launch = (1.0 - action_is_noop)
        log_prob = log_p_source + is_launch * (log_p_target + log_p_fraction)
        entropy = entropy_source + is_launch * (entropy_target + entropy_fraction)

        return log_prob, entropy, value

    def sample_action(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        owned_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, int, int, bool, float, float]:
        """Sample a single action for inference (batch size 1).

        Returns:
            source_idx, target_idx, fraction_idx, is_noop, log_prob, value
        """
        with torch.no_grad():
            nf = node_features.unsqueeze(0) if node_features.dim() == 2 else node_features
            pos = positions.unsqueeze(0) if positions.dim() == 2 else positions
            om = owned_mask.unsqueeze(0) if owned_mask.dim() == 1 else owned_mask
            N = nf.shape[1]

            source_logits, all_target_logits, all_fraction_logits = self.forward(nf, pos, om)
            g_critic = self._critic_encode(nf, pos)
            value = self.value_head(g_critic).squeeze(-1).item()

            # Sample source (or noop)
            source_dist = torch.distributions.Categorical(logits=source_logits.squeeze(0))
            if deterministic:
                src = source_logits.squeeze(0).argmax().item()
            else:
                src = source_dist.sample().item()
            log_p_src = source_dist.log_prob(torch.tensor(src)).item()

            if src == N:  # noop
                return 0, 0, 0, True, log_p_src, value

            # Sample target
            tgt_logits = all_target_logits[0, src]  # (N,)
            target_dist = torch.distributions.Categorical(logits=tgt_logits)
            if deterministic:
                tgt = tgt_logits.argmax().item()
            else:
                tgt = target_dist.sample().item()
            log_p_tgt = target_dist.log_prob(torch.tensor(tgt)).item()

            # Sample fraction
            frac_logits = all_fraction_logits[0, src, tgt]  # (4,)
            frac_dist = torch.distributions.Categorical(logits=frac_logits)
            if deterministic:
                frac = frac_logits.argmax().item()
            else:
                frac = frac_dist.sample().item()
            log_p_frac = frac_dist.log_prob(torch.tensor(frac)).item()

            log_prob = log_p_src + log_p_tgt + log_p_frac
            return src, tgt, frac, False, log_prob, value
