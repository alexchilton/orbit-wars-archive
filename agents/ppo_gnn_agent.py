"""GNN agent wrapper for Orbit Wars — compatible with test_agent.py harness.

Usage:
    python test_agent.py --agent ppo_gnn_agent.py --opponent submission/main_s2_shun_combined.py --games 20 --swap
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from ppo_gnn.gnn_policy import FRACTION_BUCKETS, OrbitWarsGNNPolicy
from ppo_gnn.replay_parser import _build_node_features
from ppo_gnn.sun_geometry import sun_intersects_path

# Auto-detect mode from number of players (set on first call)
_model_2p: OrbitWarsGNNPolicy | None = None
_model_4p: OrbitWarsGNNPolicy | None = None

CACHE_DIR = Path(__file__).parent / "ppo_gnn" / "cache"

# Noop penalty: subtract from the noop logit to discourage idle play.
# The model heavily over-noops due to training data distribution (~32% noops).
NOOP_PENALTY = 4.0

# Idle surplus threshold: if we have many more ships than needed, force action
IDLE_SURPLUS_RATIO = 3.0  # force action if we have 3x enemy's total ships


def _get_model(num_players: int) -> OrbitWarsGNNPolicy:
    global _model_2p, _model_4p

    if num_players == 2:
        if _model_2p is None:
            _model_2p = OrbitWarsGNNPolicy(hidden_dim=64, use_gat=True)
            ckpt = CACHE_DIR / "checkpoint_awr_2p.pt"
            if not ckpt.exists():
                ckpt = CACHE_DIR / "checkpoint_bc_2p.pt"
            _model_2p.load_state_dict(torch.load(str(ckpt), weights_only=True))
            _model_2p.eval()
        return _model_2p
    else:
        if _model_4p is None:
            _model_4p = OrbitWarsGNNPolicy(hidden_dim=64, use_gat=True)
            ckpt = CACHE_DIR / "checkpoint_awr_4p.pt"
            if not ckpt.exists():
                ckpt = CACHE_DIR / "checkpoint_bc_4p.pt"
            _model_4p.load_state_dict(torch.load(str(ckpt), weights_only=True))
            _model_4p.eval()
        return _model_4p


def _sample_with_noop_penalty(model, nf, pos, owned, noop_penalty=NOOP_PENALTY):
    """Sample action with noop logit penalized to encourage play."""
    with torch.no_grad():
        nf_b = nf.unsqueeze(0) if nf.dim() == 2 else nf
        pos_b = pos.unsqueeze(0) if pos.dim() == 2 else pos
        om_b = owned.unsqueeze(0) if owned.dim() == 1 else owned
        N = nf_b.shape[1]

        source_logits, all_target_logits, all_fraction_logits = model(nf_b, pos_b, om_b)

        # Apply noop penalty
        source_logits[0, N] -= noop_penalty

        # Sample source
        source_dist = torch.distributions.Categorical(logits=source_logits.squeeze(0))
        src = source_dist.sample().item()

        if src == N:  # noop even after penalty
            return 0, 0, 0, True

        # Sample target (stochastic for diversity)
        tgt_logits = all_target_logits[0, src]
        tgt = torch.distributions.Categorical(logits=tgt_logits).sample().item()

        # Sample fraction
        frac_logits = all_fraction_logits[0, src, tgt]
        frac = torch.distributions.Categorical(logits=frac_logits).sample().item()

        return src, tgt, frac, False


def agent(obs, config):
    """Kaggle-compatible agent function."""
    planets = obs["planets"]
    fleets = obs.get("fleets", [])
    player = obs["player"]

    if not planets:
        return []

    # Detect number of players from planet ownership
    owners = set(int(p[1]) for p in planets if int(p[1]) >= 0)
    num_players = max(len(owners), 2)

    model = _get_model(num_players)
    nf, pos, owned = _build_node_features(planets, fleets, player, num_players)

    src, tgt, frac, is_noop = _sample_with_noop_penalty(model, nf, pos, owned)

    if is_noop:
        return []

    # Convert to Kaggle action format
    src_planet = planets[src]
    tgt_planet = planets[tgt]

    # Check sun intersection — skip if path crosses sun
    sx, sy = float(src_planet[2]), float(src_planet[3])
    tx, ty = float(tgt_planet[2]), float(tgt_planet[3])
    if sun_intersects_path(sx, sy, tx, ty):
        return []  # don't launch through the sun

    angle = math.atan2(ty - sy, tx - sx)
    ships = int(float(src_planet[5]) * FRACTION_BUCKETS[frac])

    if ships < 1:
        return []

    return [[int(src_planet[0]), angle, ships]]
