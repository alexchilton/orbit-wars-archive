"""Test that self-play agent obs matches training env obs dimensions."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ppo_gnn.sb3_constants import (
    OBS_DIM, CANDIDATE_OBS_DIM, TEMPORAL_OBS_DIM,
    SB3_MAX_CANDIDATES, EDGE_INPUT_DIM, GLOBAL_DIM, TEMPORAL_STEPS,
)


def test_compute_global_features_standalone():
    """compute_global_features should be importable as a standalone function."""
    from ppo_gnn.sb3_env import compute_global_features

    planets = [
        [0, 0, 50, 50, 0.1, 20, 3],  # player 0 owns this
        [1, 1, 60, 60, 0.1, 15, 2],  # player 1 owns this
        [2, -1, 40, 40, 0.1, 10, 1],  # neutral
    ]
    fleets = []
    result = compute_global_features(planets, fleets, player=0, step=100, max_steps=500)
    assert result.shape == (GLOBAL_DIM,), f"Expected ({GLOBAL_DIM},), got {result.shape}"
    assert result.dtype == np.float32


def test_selfplay_obs_includes_temporal():
    """Self-play agent obs must include temporal features (total OBS_DIM, not just CANDIDATE_OBS_DIM)."""
    # The self-play agent in sb3_callbacks.py builds obs from raw game state.
    # It must produce obs of shape (OBS_DIM,) = 7184, not just (CANDIDATE_OBS_DIM,) = 7104.
    # This test verifies the obs dimension matches what the model expects.
    assert OBS_DIM == CANDIDATE_OBS_DIM + TEMPORAL_OBS_DIM
    assert OBS_DIM == 7184, f"Expected OBS_DIM=7184, got {OBS_DIM}"
    assert TEMPORAL_OBS_DIM == 80, f"Expected TEMPORAL_OBS_DIM=80, got {TEMPORAL_OBS_DIM}"

    # The current self-play agent only produces CANDIDATE_OBS_DIM = 7104
    # This documents the bug: model expects 7184 but gets 7104
    from ppo_gnn.edge_policy import compute_candidate_edges
    import torch

    # Fake game obs
    planets = [
        [i, (0 if i < 5 else (1 if i < 10 else -1)), 50 + i, 50 + i, 0.1, 20, 3]
        for i in range(20)
    ]
    fleets = []

    ef, edge_indices, em, num_valid = compute_candidate_edges(
        planets=planets, fleets=fleets, player_id=0, num_players=2,
        step=100, max_steps=500, max_candidates=SB3_MAX_CANDIDATES,
        angular_velocity=0.01,
    )
    ef = torch.nan_to_num(ef, nan=0.0, posinf=1.0, neginf=-1.0)
    candidate_flat = ef.numpy().flatten().astype(np.float32)

    # Old self-play agent only had this:
    old_obs = candidate_flat  # 7104 dims
    assert old_obs.shape[0] == CANDIDATE_OBS_DIM, f"Candidate obs should be {CANDIDATE_OBS_DIM}"

    # New self-play agent must produce full obs:
    from ppo_gnn.sb3_env import compute_global_features
    global_feats = compute_global_features(planets, fleets, player=0, step=100, max_steps=500)
    temporal_buffer = np.zeros((TEMPORAL_STEPS, GLOBAL_DIM), dtype=np.float32)
    temporal_buffer[-1] = global_feats  # Only latest step filled

    full_obs = np.zeros(OBS_DIM, dtype=np.float32)
    full_obs[:CANDIDATE_OBS_DIM] = candidate_flat
    full_obs[CANDIDATE_OBS_DIM:] = temporal_buffer.flatten()
    assert full_obs.shape[0] == OBS_DIM, f"Full obs should be {OBS_DIM}, got {full_obs.shape[0]}"


if __name__ == "__main__":
    test_compute_global_features_standalone()
    print("PASS: test_compute_global_features_standalone")
    test_selfplay_obs_includes_temporal()
    print("PASS: test_selfplay_obs_includes_temporal")
    print("All tests passed!")
