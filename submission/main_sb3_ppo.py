"""SB3 MaskablePPO Transformer agent for Orbit Wars.

Imports compute_candidate_edges and sun geometry from ppo_gnn/ — single source of truth.
Loads policy weights from sb3_policy_weights.pt + VecNormalize stats.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn

from ppo_gnn.edge_policy import (
    compute_candidate_edges,
    solve_intercept,
    FRACTION_BUCKETS,
    NUM_FRACTIONS,
    MAX_ACTIONS,
    EDGE_INPUT_DIM,
)
from ppo_gnn.sb3_constants import (
    SB3_MAX_CANDIDATES,
    NUM_CHOICES,
    NOOP_ACTION,
)
from ppo_gnn.sun_geometry import sun_intersects_path

# ---------------------------------------------------------------------------
# Constants (inference-only, not in shared modules)
# ---------------------------------------------------------------------------

INNER_ORBIT_THRESHOLD = 48.0
MAX_CANDIDATES = SB3_MAX_CANDIDATES


# ---------------------------------------------------------------------------
# Policy network (inference-only, mirrors SB3 CandidateTransformerExtractor + MLP)
# ---------------------------------------------------------------------------


class EdgeTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        if mask is not None:
            B, K = mask.shape
            n_heads = self.attn.num_heads
            float_mask = torch.zeros(B, K, K, device=x.device, dtype=x.dtype)
            pad_cols = mask.unsqueeze(1).expand(-1, K, -1)
            float_mask = float_mask.masked_fill(pad_cols, -1e4)
            float_mask = float_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
            float_mask = float_mask.reshape(B * n_heads, K, K)
            attn_out, _ = self.attn(x, x, x, attn_mask=float_mask, need_weights=False)
        else:
            attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class PolicyNetwork(nn.Module):
    """Standalone policy network matching SB3 PointerNetPolicy architecture.

    Per-candidate scoring: Transformer → scorer(128→64→10) per candidate → noop bias.
    """

    def __init__(self, d_model=128, n_heads=4, n_layers=3, n_candidates=MAX_CANDIDATES, edge_dim=EDGE_INPUT_DIM):
        super().__init__()
        self.n_candidates = n_candidates
        self.edge_dim = edge_dim
        self.d_model = d_model

        # Feature extractor (matches CandidateTransformerExtractor)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList([
            EdgeTransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        # Per-candidate scorer (matches PointerNetPolicy.candidate_scorer)
        self.candidate_scorer = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, NUM_FRACTIONS),
        )
        # Learned noop bias
        self.noop_logit = nn.Parameter(torch.zeros(1))

    def forward(self, obs_flat, action_mask):
        B = obs_flat.shape[0]
        x = obs_flat.view(B, self.n_candidates, self.edge_dim)
        pad_mask = (x.abs().sum(dim=-1) < 1e-6)
        x = x.clamp(-50.0, 50.0)
        x = self.edge_encoder(x)
        x = x.clamp(-100.0, 100.0)
        for block in self.blocks:
            x = block(x, mask=pad_mask)

        # Per-candidate scoring: (B, 48, 10)
        scores = self.candidate_scorer(x)
        # Flatten to (B, 480), append noop → (B, 481)
        flat = scores.reshape(B, -1)
        noop = self.noop_logit.expand(B, 1)
        slot_logits = torch.cat([flat, noop], dim=-1)

        # Tile for 5 slots → (B, 5, 481)
        logits = slot_logits.unsqueeze(1).expand(-1, MAX_ACTIONS, -1)
        mask_reshaped = action_mask.view(B, MAX_ACTIONS, NUM_CHOICES)
        logits = logits.masked_fill(~mask_reshaped, -1e8)

        return logits


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_MODEL = None
_DEVICE = None
_OBS_MEAN = None
_OBS_VAR = None
_CLIP_OBS = 10.0


def _load_model():
    global _MODEL, _DEVICE, _OBS_MEAN, _OBS_VAR, _CLIP_OBS
    if _MODEL is not None:
        return

    _DEVICE = torch.device("cpu")
    _MODEL = PolicyNetwork()

    try:
        _dir = os.path.dirname(__file__)
    except NameError:
        _dir = "/kaggle_simulations/agent"

    # Load VecNormalize stats
    norm_path = os.path.join(_dir, "vec_normalize_stats.npz")
    if os.path.exists(norm_path):
        stats = np.load(norm_path)
        _OBS_MEAN = stats["mean"]
        _OBS_VAR = stats["var"]
        _CLIP_OBS = float(stats["clip_obs"][0])

    weights_path = os.path.join(_dir, "sb3_policy_weights.pt")
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")

    # Extract sub-dicts by SB3 prefix
    def extract_prefix(prefix):
        p = prefix + "."
        return {k[len(p):]: v for k, v in state.items() if k.startswith(p)}

    fe_state = extract_prefix("features_extractor")

    _MODEL.edge_encoder.load_state_dict(
        {k.replace("edge_encoder.", ""): v for k, v in fe_state.items()
         if k.startswith("edge_encoder.")})

    block_state = {k: v for k, v in fe_state.items() if k.startswith("blocks.")}
    blocks_sd = _MODEL.blocks.state_dict()
    for k in blocks_sd:
        full_key = f"blocks.{k}"
        if full_key in block_state:
            blocks_sd[k] = block_state[full_key]
    _MODEL.blocks.load_state_dict(blocks_sd)

    # Per-candidate scorer (PointerNetPolicy)
    scorer_state = extract_prefix("candidate_scorer")
    _MODEL.candidate_scorer.load_state_dict(scorer_state)

    # Noop logit
    noop_key = "noop_logit"
    if noop_key in state:
        _MODEL.noop_logit.data.copy_(state[noop_key])

    _MODEL.eval()


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------


def agent(obs, config=None):
    _load_model()

    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])
    player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
    step = obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)
    angular_velocity = obs.get("angular_velocity", 0.0) if isinstance(obs, dict) else getattr(obs, "angular_velocity", 0.0)

    if not planets or not any(int(p[1]) == player for p in planets):
        return []

    # Detect num_players from planet ownership
    owners = set(int(p[1]) for p in planets if int(p[1]) >= 0)
    num_players = max(len(owners), 2)

    ef, edge_indices, em, num_valid = compute_candidate_edges(
        planets=planets, fleets=fleets, player_id=player,
        num_players=num_players, step=step, max_steps=500,
        max_candidates=MAX_CANDIDATES, angular_velocity=angular_velocity,
    )
    ef = torch.nan_to_num(ef, nan=0.0, posinf=1.0, neginf=-1.0)
    obs_np = ef.numpy().flatten().astype(np.float32)

    # Apply VecNormalize (must match training normalization)
    if _OBS_MEAN is not None:
        obs_np = (obs_np - _OBS_MEAN) / np.sqrt(_OBS_VAR + 1e-8)
        obs_np = np.clip(obs_np, -_CLIP_OBS, _CLIP_OBS)

    obs_flat = torch.from_numpy(obs_np.astype(np.float32)).unsqueeze(0)  # (1, 3552)

    # Build action mask
    nv = min(num_valid, MAX_CANDIDATES)
    single_mask = np.zeros(NUM_CHOICES, dtype=bool)
    for i in range(nv):
        single_mask[i * NUM_FRACTIONS: (i + 1) * NUM_FRACTIONS] = True
    single_mask[NOOP_ACTION] = True
    mask = np.tile(single_mask, MAX_ACTIONS)
    mask_t = torch.from_numpy(mask).unsqueeze(0)

    with torch.inference_mode():
        logits = _MODEL(obs_flat, mask_t)  # (1, MAX_ACTIONS, NUM_CHOICES)
        actions_idx = logits.argmax(dim=-1).squeeze(0).numpy()  # (MAX_ACTIONS,)

    # Decode actions into game commands
    actions = []
    committed = {}

    for slot_action in actions_idx:
        slot_action = int(slot_action)
        if slot_action == NOOP_ACTION:
            continue

        cand_idx = slot_action // NUM_FRACTIONS
        frac_idx = slot_action % NUM_FRACTIONS
        if cand_idx >= nv:
            continue

        src_pidx = edge_indices[cand_idx, 0].item()
        tgt_pidx = edge_indices[cand_idx, 1].item()
        src_p, tgt_p = planets[src_pidx], planets[tgt_pidx]

        if int(src_p[1]) != player:
            continue

        sx, sy = float(src_p[2]), float(src_p[3])
        tx, ty = float(tgt_p[2]), float(tgt_p[3])
        tgt_planet_r = float(tgt_p[4])
        tgt_orbital_r = math.hypot(tx - 50.0, ty - 50.0)
        omega = angular_velocity if (tgt_orbital_r + tgt_planet_r < INNER_ORBIT_THRESHOLD) else 0.0

        src_fleet = int(float(src_p[5]))
        already = committed.get(src_pidx, 0)
        base_reserve = max(5, int(src_fleet * 0.15))
        available = src_fleet - base_reserve - already
        if available < 5:
            continue

        ships = max(1, int(available * FRACTION_BUCKETS[frac_idx]))
        if ships < 5:
            continue

        ix, iy = solve_intercept(sx, sy, tx, ty, omega, ships)

        # Sun avoidance at action level (belt-and-suspenders with candidate filter)
        if sun_intersects_path(sx, sy, ix, iy):
            dx, dy = ix - sx, iy - sy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 1e-6:
                mid_x, mid_y = (sx + ix) / 2, (sy + iy) / 2
                sun_dist = math.sqrt((mid_x - 50.0) ** 2 + (mid_y - 50.0) ** 2)
                if sun_dist < 1e-6:
                    continue
                perp_x = -(50.0 - mid_y) / sun_dist
                perp_y = (50.0 - mid_x) / sun_dist
                offset = 14.0 / dist
                ix2 = ix + perp_x * offset * dist * 0.3
                iy2 = iy + perp_y * offset * dist * 0.3
                if sun_intersects_path(sx, sy, ix2, iy2):
                    continue
                ix, iy = ix2, iy2

        angle = math.atan2(iy - sy, ix - sx)
        actions.append([int(src_p[0]), angle, ships])
        committed[src_pidx] = already + ships

    return actions
