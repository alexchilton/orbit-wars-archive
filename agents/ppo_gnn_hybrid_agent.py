"""Hybrid residual agent: GNN model + shun_combined heuristic.

The heuristic is the default actor. The GNN model proposes an override only
when the model's value estimate gives it high confidence. This is the
"residual policy" approach from Checkpoint 014.

Decision logic per step:
    1. Get heuristic action from shun_combined
    2. Get GNN model's proposed action
    3. If model's value confidence is high enough AND model chose to act,
       use the model's action. Otherwise, use the heuristic's.

This makes the GNN a selective override — it only takes control when it
believes it can do better than the safe heuristic baseline.

Usage:
    python test_agent.py --agent ppo_gnn_hybrid_agent.py --opponent submission/main_release_candidate_v3_antidogpile_position.py --games 20 --swap
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from ppo_gnn.gnn_policy import FRACTION_BUCKETS, OrbitWarsGNNPolicy
from ppo_gnn.replay_parser import _build_node_features
from ppo_gnn.sun_geometry import sun_intersects_path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model override threshold: the model must have value estimate this much above
# the baseline expectation (0.0 = neutral) to be allowed to override.
VALUE_CONFIDENCE_THRESHOLD = 0.1

# Noop penalty applied to model's source logits
NOOP_PENALTY = 4.0

# Fallback: use heuristic if model has no planets or fails
ALWAYS_FALLBACK_ON_ERROR = True

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------

_heuristic: Optional[Callable] = None
_model_2p: Optional[OrbitWarsGNNPolicy] = None
_model_4p: Optional[OrbitWarsGNNPolicy] = None
_call_count = 0

CACHE_DIR = Path(__file__).parent / "ppo_gnn" / "cache"
HEURISTIC_PATH = Path(__file__).parent / "submission" / "main_s2_shun_combined.py"


def _load_heuristic() -> Callable:
    global _heuristic
    if _heuristic is None:
        p = HEURISTIC_PATH.resolve()
        module_name = f"_heuristic_{p.stem}"
        spec = importlib.util.spec_from_file_location(module_name, p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        _heuristic = mod.agent
    return _heuristic


def _get_model(num_players: int) -> OrbitWarsGNNPolicy:
    global _model_2p, _model_4p

    if num_players == 2:
        if _model_2p is None:
            _model_2p = OrbitWarsGNNPolicy(hidden_dim=64, use_gat=True)
            # Prefer PPO > AWR > BC checkpoint
            for name in ["checkpoint_ppo_2p.pt", "checkpoint_awr_2p.pt", "checkpoint_bc_2p.pt"]:
                ckpt = CACHE_DIR / name
                if ckpt.exists():
                    _model_2p.load_state_dict(torch.load(str(ckpt), weights_only=True))
                    break
            _model_2p.eval()
        return _model_2p
    else:
        if _model_4p is None:
            _model_4p = OrbitWarsGNNPolicy(hidden_dim=64, use_gat=True)
            for name in ["checkpoint_ppo_4p.pt", "checkpoint_awr_4p.pt", "checkpoint_bc_4p.pt"]:
                ckpt = CACHE_DIR / name
                if ckpt.exists():
                    _model_4p.load_state_dict(torch.load(str(ckpt), weights_only=True))
                    break
            _model_4p.eval()
        return _model_4p


# ---------------------------------------------------------------------------
# Model action proposal
# ---------------------------------------------------------------------------

def _model_propose(model, planets, fleets, player, num_players):
    """Get the model's proposed action and value estimate.

    Returns:
        (action_list, value, is_noop)
        action_list is in Kaggle format: [[planet_id, angle, ships]] or []
    """
    nf, pos, owned = _build_node_features(planets, fleets, player, num_players)

    with torch.no_grad():
        nf_b = nf.unsqueeze(0)
        pos_b = pos.unsqueeze(0)
        om_b = owned.unsqueeze(0)
        N = nf_b.shape[1]

        source_logits, all_target_logits, all_fraction_logits = model(
            nf_b, pos_b, om_b,
        )
        h, _, g = model._encode(nf_b, pos_b)
        value = model.value_head(g).squeeze(-1).item()

        # Apply noop penalty
        source_logits[0, N] -= NOOP_PENALTY

        # Use stochastic sampling for exploration diversity
        src = torch.distributions.Categorical(logits=source_logits.squeeze(0)).sample().item()

        if src == N:
            return [], value, True

        tgt = torch.distributions.Categorical(logits=all_target_logits[0, src]).sample().item()
        frac = torch.distributions.Categorical(logits=all_fraction_logits[0, src, tgt]).sample().item()

    src_planet = planets[src]
    tgt_planet = planets[tgt]
    sx, sy = float(src_planet[2]), float(src_planet[3])
    tx, ty = float(tgt_planet[2]), float(tgt_planet[3])

    # Sun safety
    if sun_intersects_path(sx, sy, tx, ty):
        return [], value, True

    angle = math.atan2(ty - sy, tx - sx)
    ships = int(float(src_planet[5]) * FRACTION_BUCKETS[frac])

    if ships < 1:
        return [], value, True

    return [[int(src_planet[0]), angle, ships]], value, False


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------

def agent(obs, config):
    """Hybrid agent: heuristic baseline with GNN model overrides."""
    global _call_count
    _call_count += 1

    planets = obs.get("planets", [])
    if not planets:
        return []

    player = obs.get("player", 0)

    # Detect mode
    owners = set(int(p[1]) for p in planets if int(p[1]) >= 0)
    num_players = max(len(owners), 2)

    # Always get heuristic action (our safe baseline)
    heuristic_fn = _load_heuristic()
    try:
        heuristic_action = heuristic_fn(obs, config)
    except Exception:
        heuristic_action = []

    # Get model's proposal
    try:
        model = _get_model(num_players)
        fleets = obs.get("fleets", [])
        model_action, model_value, model_noop = _model_propose(
            model, planets, fleets, player, num_players,
        )
    except Exception:
        if ALWAYS_FALLBACK_ON_ERROR:
            return heuristic_action
        raise

    # Decision: should the model override the heuristic?
    #
    # Conditions for model override:
    # 1. Model proposed an action (not noop)
    # 2. Model's value estimate is above the confidence threshold
    #    (positive value = model thinks we're in a winning position,
    #     so it has learned something useful about this state)
    # 3. Heuristic also proposed an action (if heuristic says noop,
    #    the model shouldn't force a launch — the heuristic knows
    #    about local tactics we might not)

    model_should_override = (
        not model_noop
        and model_value > VALUE_CONFIDENCE_THRESHOLD
        and len(heuristic_action) > 0
    )

    if model_should_override:
        return model_action
    else:
        return heuristic_action
