"""Edge-based transformer policy for Orbit Wars.

Instead of the factored source→target→fraction approach, this architecture:
1. Heuristic scorer generates all source→target candidate edges
2. Filters to top-K (default 192) candidates
3. Each candidate edge gets rich features (source + target + edge + heuristic)
4. Transformer with self-attention scores all K candidates simultaneously
5. Policy picks top-3 edges per turn (multi-action native)
6. Each selected edge also gets a fraction prediction

This solves the noop-99 problem by making multi-action the default: the
agent always picks exactly 3 actions (with masking for invalid ones).

The self-attention lets candidates "see" each other, enabling coordination
like "don't send two fleets to the same planet" or "attack from two sides."
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sun_geometry import (
    BOARD_DIAGONAL,
    SUN_RADIUS,
    SUN_X,
    SUN_Y,
    closest_approach_to_sun,
    sun_intersects_path,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CANDIDATES = 192
MAX_ACTIONS = 5
FRACTION_BUCKETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
NUM_FRACTIONS = len(FRACTION_BUCKETS)

# Phase/macro strategy indices
PHASE_EXPAND = 0      # target neutrals
PHASE_ATTACK = 1      # target enemies
PHASE_DEFEND = 2      # reinforce own planets
PHASE_ACCUMULATE = 3  # noop, build ships
NUM_PHASES = 4
# Feature indices for edge ownership flags (src(28) + tgt(28) + spatial(6) + heuristic offset)
FEAT_IS_NEUTRAL = 64
FEAT_IS_ENEMY = 65
FEAT_IS_FRIENDLY = 66

# Per-edge heuristic feature count
HEURISTIC_DIM = 12  # see _compute_heuristic_features

# Per-edge input = source_node(28) + target_node(28) + spatial(6) + heuristic(12)
EDGE_INPUT_DIM = 74


# ---------------------------------------------------------------------------
# Heuristic edge scorer — filters ~4000 edges to top-K
# ---------------------------------------------------------------------------

def compute_candidate_edges(
    planets: list,
    fleets: list,
    player_id: int,
    num_players: int,
    step: int,
    max_steps: int,
    max_candidates: int = MAX_CANDIDATES,
    min_ships_to_send: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Score all possible source→target edges and return top-K candidates.

    Args:
        planets: Raw planet list from observation [[id, owner, x, y, orbit_vel, ships, prod], ...]
        fleets: Raw fleet list from observation
        player_id: Agent's player ID
        num_players: Total players (2 or 4)
        step: Current game step
        max_steps: Max steps in game
        max_candidates: Maximum edges to keep (default 192)
        min_ships_to_send: Minimum ships on source to consider

    Returns:
        edge_features: (K, EDGE_INPUT_DIM) — features for each candidate edge
        edge_indices: (K, 2) — [source_planet_idx, target_planet_idx] for each edge
        edge_mask: (K,) — 1.0 for valid edges, 0.0 for padding
        num_valid: int — number of actual valid edges (before padding)
    """
    N = len(planets)

    # Precompute per-planet info
    planet_ships = [float(p[5]) for p in planets]
    planet_prod = [float(p[6]) for p in planets]
    planet_owner = [int(p[1]) for p in planets]
    planet_x = [float(p[2]) for p in planets]
    planet_y = [float(p[3]) for p in planets]
    planet_orbit_vel = [float(p[4]) for p in planets]

    # Identify owned planets with enough ships
    my_planets = [i for i in range(N)
                  if planet_owner[i] == player_id and planet_ships[i] >= min_ships_to_send]

    if not my_planets:
        # No valid sources — return empty padded tensor
        return (
            torch.zeros(max_candidates, EDGE_INPUT_DIM),
            torch.zeros(max_candidates, 2, dtype=torch.long),
            torch.zeros(max_candidates),
            0,
        )

    # Precompute inbound ships per planet
    inbound_friendly = [0.0] * N
    inbound_enemy = [0.0] * N
    for f in fleets:
        f_owner = int(f[1])
        f_ships = float(f[6])
        f_x, f_y = float(f[2]), float(f[3])
        # Find closest planet as destination estimate
        best_dist = float('inf')
        best_idx = 0
        for i in range(N):
            d = math.sqrt((f_x - planet_x[i])**2 + (f_y - planet_y[i])**2)
            if d < best_dist:
                best_dist = d
                best_idx = i
        if f_owner == player_id:
            inbound_friendly[best_idx] += f_ships
        else:
            inbound_enemy[best_idx] += f_ships

    # Global stats
    my_total_ships = sum(planet_ships[i] for i in range(N) if planet_owner[i] == player_id)
    my_total_prod = sum(planet_prod[i] for i in range(N) if planet_owner[i] == player_id)
    total_prod = sum(planet_prod[i] for i in range(N) if planet_owner[i] >= 0)
    step_progress = step / max(max_steps, 1)

    # Score all edges
    scored_edges = []
    for src_idx in my_planets:
        sx, sy = planet_x[src_idx], planet_y[src_idx]
        src_ships = planet_ships[src_idx]

        for tgt_idx in range(N):
            if tgt_idx == src_idx:
                continue

            tx, ty = planet_x[tgt_idx], planet_y[tgt_idx]
            tgt_owner = planet_owner[tgt_idx]
            tgt_ships = planet_ships[tgt_idx]
            tgt_prod = planet_prod[tgt_idx]

            # Basic geometry
            dist = math.sqrt((tx - sx)**2 + (ty - sy)**2)
            if dist < 1e-6:
                continue

            # Sun check (soft — just penalizes score, doesn't eliminate)
            sun_blocked = sun_intersects_path(sx, sy, tx, ty)

            # Heuristic score (higher = more promising)
            score = 0.0

            # Distance preference (closer is better, but not too close)
            dist_score = max(0, 1.0 - dist / BOARD_DIAGONAL) * 2.0
            score += dist_score

            # Production value of target
            prod_score = tgt_prod / 5.0
            score += prod_score

            # Fleet ratio — can we take it?
            defense = tgt_ships + inbound_enemy[tgt_idx] if tgt_owner != player_id else tgt_ships
            takeover_ratio = src_ships / max(defense + 1, 1)
            if tgt_owner < 0:  # neutral
                score += min(takeover_ratio, 2.0) * 1.5
            elif tgt_owner != player_id:  # enemy
                score += min(takeover_ratio, 2.0) * 2.0
            else:  # own planet — never include
                # v131/shunlite approach: defense = hold ships, don't shuffle
                continue

            # Sun penalty
            if sun_blocked:
                score -= 3.0

            # Static planet bonus (shunlite: orbiting planets are harder to hit)
            orbit_vel = abs(planet_orbit_vel[tgt_idx])
            if orbit_vel < 0.1:  # static planet
                score += 1.0
            elif orbit_vel > 1.0:  # fast orbiter
                score -= 0.5

            # Max ETA filter — don't include edges across the entire map
            travel_time = dist / 6.0
            step_progress_local = step / max(max_steps, 1)
            max_eta = 14 if step_progress_local < 0.1 else (20 if step_progress_local < 0.4 else 30)
            if travel_time > max_eta:
                continue

            # Production value relative to defense cost
            if defense > 0:
                roi = tgt_prod / (defense + 1)
                score += roi * 2.0

            scored_edges.append((score, src_idx, tgt_idx))

    # Diverse candidate selection: quota-based, not strict top-K
    # ~40% neutral, ~40% enemy, ~20% overall top (fills gaps)
    neutral_edges = [(s, src, tgt) for s, src, tgt in scored_edges if planet_owner[tgt] < 0]
    enemy_edges = [(s, src, tgt) for s, src, tgt in scored_edges
                   if planet_owner[tgt] >= 0 and planet_owner[tgt] != player_id]

    neutral_edges.sort(key=lambda x: x[0], reverse=True)
    enemy_edges.sort(key=lambda x: x[0], reverse=True)

    n_neutral = min(len(neutral_edges), max_candidates * 2 // 5)
    n_enemy = min(len(enemy_edges), max_candidates * 2 // 5)

    top_edges = []
    seen = set()
    for e in neutral_edges[:n_neutral]:
        top_edges.append(e)
        seen.add((e[1], e[2]))
    for e in enemy_edges[:n_enemy]:
        if (e[1], e[2]) not in seen:
            top_edges.append(e)
            seen.add((e[1], e[2]))
    # Fill remaining slots from overall top
    all_sorted = sorted(scored_edges, key=lambda x: x[0], reverse=True)
    for e in all_sorted:
        if len(top_edges) >= max_candidates:
            break
        if (e[1], e[2]) not in seen:
            top_edges.append(e)
            seen.add((e[1], e[2]))
    num_valid = len(top_edges)

    # Build feature tensors
    edge_features = torch.zeros(max_candidates, EDGE_INPUT_DIM)
    edge_indices = torch.zeros(max_candidates, 2, dtype=torch.long)
    edge_mask = torch.zeros(max_candidates)

    # Build node features once (reused across edges)
    node_feats_raw = _build_node_features_compact(
        planets, fleets, player_id, num_players, step, max_steps,
        planet_ships, planet_prod, planet_owner, planet_x, planet_y,
        planet_orbit_vel, inbound_friendly, inbound_enemy,
        my_total_ships, my_total_prod, total_prod,
    )

    for k, (heur_score, src_idx, tgt_idx) in enumerate(top_edges):
        src_feats = node_feats_raw[src_idx]   # (28,)
        tgt_feats = node_feats_raw[tgt_idx]   # (28,)

        # Spatial/edge features (6 dims)
        sx, sy = planet_x[src_idx], planet_y[src_idx]
        tx, ty = planet_x[tgt_idx], planet_y[tgt_idx]
        dist = math.sqrt((tx - sx)**2 + (ty - sy)**2)
        dist_norm = dist / BOARD_DIAGONAL
        travel_time = dist / 6.0  # ship_speed
        angle = math.atan2(ty - sy, tx - sx)
        sun_blocked = sun_intersects_path(sx, sy, tx, ty)
        sun_clear = closest_approach_to_sun(sx, sy, tx, ty) / BOARD_DIAGONAL

        spatial = torch.tensor([
            dist_norm, travel_time / 25.0,
            math.sin(angle), math.cos(angle),
            float(sun_blocked), min(sun_clear, 1.0),
        ])

        # Heuristic features (12 dims)
        tgt_owner = planet_owner[tgt_idx]
        tgt_ships = planet_ships[tgt_idx]
        tgt_prod = planet_prod[tgt_idx]
        src_ships = planet_ships[src_idx]
        defense = tgt_ships + (inbound_enemy[tgt_idx] if tgt_owner != player_id else 0)

        heuristic = torch.tensor([
            min(heur_score / 10.0, 1.0),                      # normalized heuristic score
            min(src_ships / max(defense + 1, 1), 3.0) / 3.0,  # takeover ratio
            float(tgt_owner < 0),                              # is neutral
            float(tgt_owner >= 0 and tgt_owner != player_id),  # is enemy
            float(tgt_owner == player_id),                     # is friendly (reinforce)
            tgt_prod / 5.0,                                    # target production
            min(inbound_friendly[tgt_idx] / 100.0, 1.0),      # friendly reinforcements to target
            min(inbound_enemy[tgt_idx] / 100.0, 1.0),         # enemy threats to target
            min(defense / 100.0, 1.0),                         # total defense
            min(src_ships / 100.0, 1.0),                       # source fleet available
            step_progress,                                     # game phase
            float(k) / max(num_valid - 1, 1),                  # rank in heuristic ordering
        ])

        edge_features[k] = torch.cat([src_feats, tgt_feats, spatial, heuristic])
        edge_indices[k] = torch.tensor([src_idx, tgt_idx])
        edge_mask[k] = 1.0

    return edge_features, edge_indices, edge_mask, num_valid


def _build_node_features_compact(
    planets, fleets, player_id, num_players, step, max_steps,
    planet_ships, planet_prod, planet_owner, planet_x, planet_y,
    planet_orbit_vel, inbound_friendly, inbound_enemy,
    my_total_ships, my_total_prod, total_prod,
) -> torch.Tensor:
    """Build the same 28-dim node features used by the GNN policy.

    Returns: (N, 28) tensor
    """
    N = len(planets)
    feats = torch.zeros(N, 28)
    step_progress = step / max(max_steps, 1)

    # Per-player aggregates
    enemy_total_prod = 0.0
    my_planet_count = 0
    total_owned = 0
    my_garrison = my_total_ships
    my_fleets_intransit = 0.0
    total_ships_global = sum(planet_ships)
    strongest_enemy = 0.0

    per_player_ships = {}
    for i in range(N):
        o = planet_owner[i]
        if o >= 0:
            total_owned += 1
            per_player_ships[o] = per_player_ships.get(o, 0) + planet_ships[i]
            if o == player_id:
                my_planet_count += 1
            else:
                enemy_total_prod += planet_prod[i]

    for f in fleets:
        fo = int(f[1])
        fs = float(f[6])
        per_player_ships[fo] = per_player_ships.get(fo, 0) + fs
        total_ships_global += fs
        if fo == player_id:
            my_fleets_intransit += fs

    for pid, ships in per_player_ships.items():
        if pid != player_id:
            strongest_enemy = max(strongest_enemy, ships)

    my_total_all = per_player_ships.get(player_id, 0)

    for i in range(N):
        x_n = planet_x[i] / 100.0
        y_n = planet_y[i] / 100.0
        is_mine = float(planet_owner[i] == player_id)
        is_enemy = float(planet_owner[i] >= 0 and planet_owner[i] != player_id)
        is_neutral = float(planet_owner[i] < 0)
        log_ships = math.log1p(planet_ships[i]) / 6.5
        prod_n = planet_prod[i] / 10.0
        ships_i = planet_ships[i]

        # Threat ratio
        threat_ratio = min(inbound_enemy[i] / max(ships_i, 1), 3.0) / 3.0
        friendly_log = math.log1p(inbound_friendly[i]) / 6.5
        enemy_log = math.log1p(inbound_enemy[i]) / 6.5
        future_prod = (planet_prod[i] * (max_steps - step) / max(max_steps, 1))
        prod_adv = min(max((my_total_prod - enemy_total_prod) / 20.0, -1.0), 1.0)

        # Nearest enemy distance
        nearest_enemy_dist = 200.0
        nearest_enemy_eta = 50.0
        for j in range(N):
            if planet_owner[j] >= 0 and planet_owner[j] != player_id:
                d = math.sqrt((planet_x[j] - planet_x[i])**2 +
                              (planet_y[j] - planet_y[i])**2)
                if d < nearest_enemy_dist:
                    nearest_enemy_dist = d

        # Surplus
        surplus = min(max((ships_i - inbound_enemy[i]) / 6.5, -1.0), 1.0)

        # Neighbors within 30 units
        friendly_near = 0
        enemy_near = 0
        for j in range(N):
            if j == i:
                continue
            d = math.sqrt((planet_x[j] - planet_x[i])**2 +
                          (planet_y[j] - planet_y[i])**2)
            if d <= 30.0:
                if planet_owner[j] == player_id:
                    friendly_near += 1
                elif planet_owner[j] >= 0:
                    enemy_near += 1

        nearest_friendly = 200.0
        for j in range(N):
            if j != i and planet_owner[j] == player_id:
                d = math.sqrt((planet_x[j] - planet_x[i])**2 +
                              (planet_y[j] - planet_y[i])**2)
                if d < nearest_friendly:
                    nearest_friendly = d

        is_frontline = float(nearest_enemy_dist < nearest_friendly) if is_mine else 0.0

        # Count inbound fleets
        inbound_f_count = 0
        inbound_e_count = 0
        for f in fleets:
            f_x, f_y = float(f[2]), float(f[3])
            d = math.sqrt((f_x - planet_x[i])**2 + (f_y - planet_y[i])**2)
            if d < 40.0:  # rough proximity
                if int(f[1]) == player_id:
                    inbound_f_count += 1
                else:
                    inbound_e_count += 1

        local_balance = 0.0
        total_near = friendly_near + enemy_near
        if total_near > 0:
            local_balance = (friendly_near - enemy_near) / total_near

        feats[i] = torch.tensor([
            x_n, y_n,                                               # 0-1: position
            is_mine, is_enemy, is_neutral,                          # 2-4: ownership
            log_ships, prod_n,                                      # 5-6: ships, production
            threat_ratio, friendly_log, enemy_log,                  # 7-9: inbound
            step_progress,                                          # 10: game phase
            future_prod / 10.0,                                     # 11: future production value
            prod_adv,                                               # 12: production advantage
            min(nearest_enemy_dist / 100.0, 1.0),                   # 13: nearest enemy
            min(nearest_enemy_eta / 50.0, 1.0),                     # 14: enemy fleet ETA
            surplus,                                                # 15: surplus ratio
            planet_orbit_vel[i] / 4.0,                              # 16: orbit velocity
            num_players / 4.0,                                      # 17: game mode
            my_planet_count / max(total_owned, 1),                  # 18: planet fraction
            my_total_all / max(total_ships_global, 1),              # 19: military share
            strongest_enemy / max(total_ships_global, 1),           # 20: enemy strongest
            min(friendly_near / 5.0, 1.0),                          # 21: friendly neighbors
            min(enemy_near / 5.0, 1.0),                             # 22: enemy neighbors
            local_balance,                                          # 23: local force balance
            is_frontline,                                           # 24: frontline flag
            min(inbound_f_count / 5.0, 1.0),                       # 25: inbound friendly count
            min(inbound_e_count / 5.0, 1.0),                       # 26: inbound enemy count
            my_fleets_intransit / max(my_total_all, 1),             # 27: fleet ratio
        ])

    return feats


# ---------------------------------------------------------------------------
# Transformer encoder for candidate edges
# ---------------------------------------------------------------------------

class EdgeTransformerBlock(nn.Module):
    """Single transformer block with self-attention over candidate edges."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, K, d_model) — K candidate edges
            mask: (B, K) — True for PADDED positions (to be ignored)
        """
        # Build float attention mask instead of bool key_padding_mask
        # Using large negative instead of -inf prevents NaN in softmax backward
        if mask is not None:
            B, K_seq = mask.shape
            # attn_mask: (B*nheads, K, K) — additive mask
            n_heads = self.attn.num_heads
            float_mask = torch.zeros(B, K_seq, K_seq, device=x.device, dtype=x.dtype)
            # Mask out padded KEY positions for all queries
            pad_cols = mask.unsqueeze(1).expand(-1, K_seq, -1)  # (B, K, K)
            float_mask = float_mask.masked_fill(pad_cols, -1e4)
            # Expand for multi-head: (B, K, K) -> (B*nheads, K, K)
            float_mask = float_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
            float_mask = float_mask.reshape(B * n_heads, K_seq, K_seq)
            attn_out, _ = self.attn(x, x, x, attn_mask=float_mask, need_weights=False)
        else:
            attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# ---------------------------------------------------------------------------
# Edge-based policy network
# ---------------------------------------------------------------------------

class EdgePolicy(nn.Module):
    """Edge-based transformer policy for Orbit Wars.

    Architecture:
        1. Edge encoder: project EDGE_INPUT_DIM → d_model
        2. N transformer blocks with self-attention over K candidates
        3. Selection head: score each edge (pick top-3)
        4. Fraction head: 4-way fraction per edge
        5. Value head: global pooling → scalar value

    Separate critic encoder ensures policy/value don't compete for features.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_actions: int = MAX_ACTIONS,
        separate_critic: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_actions = max_actions
        self.separate_critic = separate_critic

        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(EDGE_INPUT_DIM, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Transformer blocks
        self.transformer = nn.ModuleList([
            EdgeTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Selection head: score per candidate edge
        self.selection_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        # Noop head: should we do nothing instead?
        # Input: global context from mean pool
        self.noop_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        with torch.no_grad():
            self.noop_head[-1].bias.fill_(-3.0)  # Discourage noop

        # Fraction head: 4 buckets per candidate
        self.fraction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, NUM_FRACTIONS),
        )

        # Phase/macro strategy head — decides expand/attack/defend/accumulate
        self.phase_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, NUM_PHASES),
        )
        # Bias toward EXPAND at init so it doesn't start passive
        with torch.no_grad():
            self.phase_head[-1].bias.copy_(torch.tensor([1.0, 0.5, -0.5, -1.0]))

        # Critic (separate encoder)
        if separate_critic:
            self.critic_encoder = nn.Sequential(
                nn.Linear(EDGE_INPUT_DIM, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            self.critic_transformer = nn.ModuleList([
                EdgeTransformerBlock(d_model, n_heads, dropout)
                for _ in range(n_layers)
            ])

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode(
        self, edge_features: torch.Tensor, edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode candidate edges through transformer.

        Args:
            edge_features: (B, K, EDGE_INPUT_DIM)
            edge_mask: (B, K) — 1.0 for valid, 0.0 for padding

        Returns:
            (B, K, d_model) — contextualized edge embeddings
        """
        # Clamp features to prevent extreme values causing NaN in attention
        edge_features = edge_features.clamp(-50.0, 50.0)
        h = self.edge_encoder(edge_features)  # (B, K, d_model)
        h = h.clamp(-100.0, 100.0)

        # key_padding_mask: True = ignore (padded)
        pad_mask = (edge_mask < 0.5)  # (B, K) — True for padding

        for block in self.transformer:
            h = block(h, mask=pad_mask)
            h = h.clamp(-100.0, 100.0)

        return h

    def _critic_encode(
        self, edge_features: torch.Tensor, edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode for critic with separate backbone."""
        if not self.separate_critic:
            return self._encode(edge_features, edge_mask)

        edge_features = edge_features.clamp(-50.0, 50.0)
        h = self.critic_encoder(edge_features)
        h = h.clamp(-100.0, 100.0)
        pad_mask = (edge_mask < 0.5)
        for block in self.critic_transformer:
            h = block(h, mask=pad_mask)
            h = h.clamp(-100.0, 100.0)
        return h

    def forward(
        self,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            edge_features: (B, K, EDGE_INPUT_DIM)
            edge_mask: (B, K) — 1.0 for valid, 0.0 for padding

        Returns:
            noop_logit: (B, 1) — binary act/noop logit (positive = noop)
            edge_logits: (B, K) — per-edge scores (masked, no noop)
            fraction_logits: (B, K, NUM_FRACTIONS) — fraction per candidate
            value: (B, 1) — state value
        """
        h = self._encode(edge_features, edge_mask)  # (B, K, d_model)

        # Edge selection logits (K classes, no noop)
        edge_logits = self.selection_head(h).squeeze(-1)  # (B, K)
        # Mask invalid edges to -inf
        edge_logits = edge_logits.masked_fill(edge_mask < 0.5, float('-inf'))

        # Binary noop logit from global context
        valid_count = edge_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        g = (h * edge_mask.unsqueeze(-1)).sum(dim=1) / valid_count  # (B, d_model)
        noop_logit = self.noop_head(g)  # (B, 1)

        # Phase head (inactive for now — enable via self.use_phase = True)
        phase_logits = self.phase_head(g)  # (B, NUM_PHASES)
        if getattr(self, 'use_phase', False):
            phase_probs = F.softmax(phase_logits, dim=-1)
            is_neutral = edge_features[:, :, FEAT_IS_NEUTRAL]
            is_enemy = edge_features[:, :, FEAT_IS_ENEMY]
            is_friendly = edge_features[:, :, FEAT_IS_FRIENDLY]
            PHASE_BONUS = 1.5
            phase_bias = (
                phase_probs[:, 0:1] * is_neutral * PHASE_BONUS
                + phase_probs[:, 1:2] * is_enemy * PHASE_BONUS
                + phase_probs[:, 2:3] * is_friendly * PHASE_BONUS
                - phase_probs[:, 3:4] * PHASE_BONUS
            )
            edge_logits = edge_logits + phase_bias

        # Fraction logits per candidate
        frac_logits = self.fraction_head(h)  # (B, K, NUM_FRACTIONS)

        # Value from separate critic
        h_c = self._critic_encode(edge_features, edge_mask)
        valid_count_c = edge_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        g_c = (h_c * edge_mask.unsqueeze(-1)).sum(dim=1) / valid_count_c
        value = self.value_head(g_c)  # (B, 1)

        return noop_logit, edge_logits, frac_logits, value

    def sample_action(
        self,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[List[int], List[int], bool, float, float]:
        """Sample up to max_actions edges from candidates.

        Uses binary noop decision first, then edge-only softmax if acting.

        Args:
            edge_features: (K, EDGE_INPUT_DIM) — unbatched
            edge_mask: (K,) — 1.0 for valid

        Returns:
            selected_edges: List of candidate indices
            fractions: List of fraction bucket indices
            is_noop: True if noop was selected
            log_prob: Total log probability of the action
            value: State value estimate
        """
        # Add batch dim
        ef = edge_features.unsqueeze(0)  # (1, K, D)
        em = edge_mask.unsqueeze(0)      # (1, K)

        with torch.no_grad():
            noop_logit, edge_logits, frac_logits, value = self.forward(ef, em)

        noop_logit = noop_logit[0, 0]    # scalar
        edge_logits = edge_logits[0]     # (K,)
        frac_logits = frac_logits[0]     # (K, NUM_FRACTIONS)
        value_scalar = value[0, 0].item()

        # Guard NaN
        if torch.isnan(noop_logit):
            noop_logit = torch.tensor(0.0)
        if torch.isnan(edge_logits).any():
            edge_logits = torch.nan_to_num(edge_logits, nan=0.0)
        if torch.isnan(frac_logits).any():
            frac_logits = torch.nan_to_num(frac_logits, nan=0.0)

        K = edge_mask.shape[0]
        total_log_prob = 0.0
        dev = edge_features.device

        # Step 1: Binary noop decision (positive logit = noop)
        noop_prob = torch.sigmoid(noop_logit)
        if deterministic:
            do_noop = noop_prob > 0.5
        else:
            do_noop = torch.bernoulli(noop_prob).bool()
            # Log prob of the noop/act decision
            if do_noop:
                total_log_prob += torch.log(noop_prob + 1e-8).item()
            else:
                total_log_prob += torch.log(1.0 - noop_prob + 1e-8).item()

        if do_noop:
            if deterministic:
                total_log_prob = torch.log(noop_prob + 1e-8).item()
            return [], [], True, total_log_prob, value_scalar

        # Step 2: Sample edges from edge-only softmax (no noop competing)
        selected = []
        fractions = []
        remaining_mask = edge_mask.clone()

        for action_idx in range(self.max_actions):
            masked_logits = edge_logits.clone()
            masked_logits = masked_logits.masked_fill(remaining_mask < 0.5, float('-inf'))

            # Check if any valid edges remain
            if (remaining_mask > 0.5).sum() == 0:
                break

            if deterministic:
                chosen = masked_logits.argmax().item()
            else:
                dist = torch.distributions.Categorical(logits=masked_logits)
                chosen = dist.sample().item()
                total_log_prob += dist.log_prob(torch.tensor(chosen).to(dev)).item()

            selected.append(chosen)
            remaining_mask[chosen] = 0.0

            # Sample fraction for this edge
            f_logits = frac_logits[chosen]
            if deterministic:
                frac_idx = f_logits.argmax().item()
            else:
                f_dist = torch.distributions.Categorical(logits=f_logits)
                frac_idx = f_dist.sample().item()
                total_log_prob += f_dist.log_prob(torch.tensor(frac_idx).to(dev)).item()

            fractions.append(frac_idx)

        is_noop = len(selected) == 0
        return selected, fractions, is_noop, total_log_prob, value_scalar

    def evaluate_action(
        self,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
        action_edges: torch.Tensor,
        action_fractions: torch.Tensor,
        action_counts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-prob and entropy for batched actions.

        Uses split architecture: binary noop decision + edge-only softmax.

        Args:
            edge_features: (B, K, EDGE_INPUT_DIM)
            edge_mask: (B, K) — 1.0 for valid
            action_edges: (B, max_actions) — candidate indices chosen (-1 for unused slots)
            action_fractions: (B, max_actions) — fraction bucket indices (-1 for unused)
            action_counts: (B,) — how many edges were actually selected (0 = noop)

        Returns:
            log_prob: (B,) — total log probability
            entropy: (B,) — total entropy
            value: (B,) — state value
        """
        B, K, _ = edge_features.shape
        dev = edge_features.device

        noop_logit, edge_logits, frac_logits, value = self.forward(edge_features, edge_mask)
        # noop_logit: (B, 1), edge_logits: (B, K), frac_logits: (B, K, NUM_FRACTIONS)

        total_log_prob = torch.zeros(B, device=dev)
        total_entropy = torch.zeros(B, device=dev)

        # Step 1: Binary noop log-prob (positive logit = noop)
        is_noop = (action_counts == 0).float()  # (B,)
        noop_logit_flat = noop_logit.squeeze(-1)  # (B,)
        # log p(noop) = log sigmoid(logit), log p(act) = log sigmoid(-logit)
        noop_lp = F.logsigmoid(noop_logit_flat) * is_noop + \
                  F.logsigmoid(-noop_logit_flat) * (1 - is_noop)
        total_log_prob = total_log_prob + noop_lp

        # Noop entropy: -p*log(p) - (1-p)*log(1-p)
        noop_prob = torch.sigmoid(noop_logit_flat)
        noop_ent = -(noop_prob * F.logsigmoid(noop_logit_flat) +
                      (1 - noop_prob) * F.logsigmoid(-noop_logit_flat))
        total_entropy = total_entropy + noop_ent

        # Step 2: Edge selection + fraction log-probs (only for non-noop samples)
        has_any_action = (action_counts > 0).float()  # (B,)

        remaining_mask = edge_mask.clone()  # (B, K)

        for action_idx in range(self.max_actions):
            edge_chosen = action_edges[:, action_idx]     # (B,)
            frac_chosen = action_fractions[:, action_idx]  # (B,)
            has_action = (action_counts > action_idx).float()  # (B,)

            if has_action.sum() < 1:
                break

            # Edge-only softmax (no noop in this distribution)
            masked_edge = edge_logits.clone()
            masked_edge = masked_edge.masked_fill(remaining_mask < 0.5, float('-inf'))
            masked_edge = torch.where(
                torch.isnan(masked_edge),
                torch.full_like(masked_edge, -1e6),
                masked_edge,
            )

            edge_dist = torch.distributions.Categorical(logits=masked_edge, validate_args=False)
            edge_idx_clamped = edge_chosen.clamp(0, K - 1)
            lp = edge_dist.log_prob(edge_idx_clamped)
            ent = edge_dist.entropy()

            total_log_prob = total_log_prob + lp * has_action
            total_entropy = total_entropy + ent * has_action

            # Fraction log-prob
            chosen_frac_logits = frac_logits[
                torch.arange(B, device=dev), edge_idx_clamped,
            ]  # (B, NUM_FRACTIONS)
            chosen_frac_logits = torch.where(
                torch.isnan(chosen_frac_logits),
                torch.full_like(chosen_frac_logits, -1e6),
                chosen_frac_logits,
            )
            f_dist = torch.distributions.Categorical(logits=chosen_frac_logits, validate_args=False)
            f_lp = f_dist.log_prob(frac_chosen.clamp(0, NUM_FRACTIONS - 1))
            f_ent = f_dist.entropy()

            total_log_prob = total_log_prob + f_lp * has_action
            total_entropy = total_entropy + f_ent * has_action

            # Update remaining mask
            if action_idx < self.max_actions - 1:
                for b in range(B):
                    if has_action[b] > 0.5 and edge_chosen[b] >= 0:
                        remaining_mask[b, edge_chosen[b].clamp(0, K-1)] = 0.0

        return total_log_prob, total_entropy, value.squeeze(-1)

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        def _count(module):
            return sum(p.numel() for p in module.parameters())

        result = {
            'edge_encoder': _count(self.edge_encoder),
            'transformer': _count(self.transformer),
            'selection_head': _count(self.selection_head),
            'noop_head': _count(self.noop_head),
            'fraction_head': _count(self.fraction_head),
            'value_head': _count(self.value_head),
        }
        if self.separate_critic:
            result['critic_encoder'] = _count(self.critic_encoder)
            result['critic_transformer'] = _count(self.critic_transformer)
        result['total'] = sum(p.numel() for p in self.parameters())
        return result
