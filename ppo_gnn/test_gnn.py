"""Unit tests for the GNN policy, sun geometry, and replay parser."""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest
import torch

from .gnn_policy import FRACTION_BUCKETS, NODE_DIM, OrbitWarsGNNPolicy
from .sun_geometry import (
    closest_approach_to_sun,
    compute_sun_edge_features_batch,
    sun_clearance_normalized,
    sun_intersects_path,
)


# ---------------------------------------------------------------------------
# Sun geometry tests
# ---------------------------------------------------------------------------

class TestSunGeometry:
    def test_direct_path_through_sun(self):
        """Path from (20,50) to (80,50) goes straight through sun at (50,50)."""
        assert sun_intersects_path(20, 50, 80, 50) is True
        dist = closest_approach_to_sun(20, 50, 80, 50)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_path_far_from_sun(self):
        """Path along top edge, far from sun."""
        assert sun_intersects_path(0, 95, 100, 95) is False
        dist = closest_approach_to_sun(0, 95, 100, 95)
        assert dist > 40

    def test_path_grazing_sun(self):
        """Path that just barely misses the sun (distance ~ sun_radius)."""
        # Path from (30, 40) to (70, 40) — closest approach = 10 (sun_y=50, path_y=40)
        dist = closest_approach_to_sun(30, 40, 70, 40)
        assert dist == pytest.approx(10.0, abs=0.1)
        # With default safety margin of 2, this intersects (10 < 10 + 2)
        assert sun_intersects_path(30, 40, 70, 40) is True
        # With no safety margin, this doesn't intersect (10 == 10, not < 10)
        assert sun_intersects_path(30, 40, 70, 40, safety_margin=0.0) is False

    def test_clearance_normalized(self):
        c = sun_clearance_normalized(20, 50, 80, 50)
        assert c == pytest.approx(0.0, abs=0.01)
        c2 = sun_clearance_normalized(0, 95, 100, 95)
        assert c2 > 0.3

    def test_batch_computation(self):
        positions = torch.tensor([
            [[20.0, 50.0], [80.0, 50.0], [10.0, 90.0]],
        ])
        sun_inter, sun_clear = compute_sun_edge_features_batch(positions)
        assert sun_inter.shape == (1, 3, 3)
        assert sun_clear.shape == (1, 3, 3)
        # (0,1) path goes through sun
        assert sun_inter[0, 0, 1].item() == 1.0
        # (0,2) path should not cross sun (top-left to top-right-ish)
        # Self-loop should be far (same point)
        assert sun_inter[0, 0, 0].item() == 0.0  # same point, dist=0 but segment is 0-length


# ---------------------------------------------------------------------------
# GNN policy tests
# ---------------------------------------------------------------------------

class TestGNNPolicy:
    @pytest.fixture
    def model_gat(self):
        return OrbitWarsGNNPolicy(hidden_dim=64, use_gat=True)

    @pytest.fixture
    def model_sage(self):
        return OrbitWarsGNNPolicy(hidden_dim=64, use_gat=False)

    def test_forward_shapes_small(self, model_gat):
        """Test with 7 planets (2P game)."""
        B, N = 2, 7
        nf = torch.randn(B, N, NODE_DIM)
        pos = torch.rand(B, N, 2) * 100
        owned = torch.zeros(B, N)
        owned[:, 0] = 1.0  # own planet 0
        owned[:, 1] = 1.0  # own planet 1

        src_logits, tgt_logits, frac_logits = model_gat(nf, pos, owned)
        assert src_logits.shape == (B, N + 1)  # N planets + noop
        assert tgt_logits.shape == (B, N, N)
        assert frac_logits.shape == (B, N, N, len(FRACTION_BUCKETS))

    def test_forward_shapes_large(self, model_gat):
        """Test with 28 planets (4P game)."""
        B, N = 2, 28
        nf = torch.randn(B, N, NODE_DIM)
        pos = torch.rand(B, N, 2) * 100
        owned = torch.zeros(B, N)
        owned[:, :5] = 1.0

        src_logits, tgt_logits, frac_logits = model_gat(nf, pos, owned)
        assert src_logits.shape == (B, N + 1)
        assert tgt_logits.shape == (B, N, N)
        assert frac_logits.shape == (B, N, N, 4)

    def test_sage_same_shapes(self, model_sage):
        """GraphSAGE variant produces same shapes as GAT."""
        B, N = 2, 12
        nf = torch.randn(B, N, NODE_DIM)
        pos = torch.rand(B, N, 2) * 100
        owned = torch.zeros(B, N)
        owned[:, 0] = 1.0

        src_logits, tgt_logits, frac_logits = model_sage(nf, pos, owned)
        assert src_logits.shape == (B, N + 1)
        assert tgt_logits.shape == (B, N, N)
        assert frac_logits.shape == (B, N, N, 4)

    def test_source_masking(self, model_gat):
        """Unowned planets should have -inf source logits."""
        B, N = 1, 7
        nf = torch.randn(B, N, NODE_DIM)
        pos = torch.rand(B, N, 2) * 100
        owned = torch.zeros(B, N)
        owned[0, 2] = 1.0  # only planet 2 is owned

        src_logits, _, _ = model_gat(nf, pos, owned)
        # Planets 0,1,3,4,5,6 should be masked
        for i in [0, 1, 3, 4, 5, 6]:
            assert src_logits[0, i].item() < -1e3
        # Planet 2 should not be masked
        assert src_logits[0, 2].item() > -1e3
        # Noop should not be masked
        assert src_logits[0, N].item() > -1e3

    def test_self_loop_masked(self, model_gat):
        """Target logits: can't send to yourself (diagonal masked)."""
        B, N = 1, 7
        nf = torch.randn(B, N, NODE_DIM)
        pos = torch.rand(B, N, 2) * 100
        owned = torch.ones(B, N)

        _, tgt_logits, _ = model_gat(nf, pos, owned)
        for i in range(N):
            assert tgt_logits[0, i, i].item() < -1e3

    def test_value_head(self, model_gat):
        B, N = 3, 10
        nf = torch.randn(B, N, NODE_DIM)
        pos = torch.rand(B, N, 2) * 100
        v = model_gat.get_value(nf, pos)
        assert v.shape == (B,)

    def test_sample_action(self, model_gat):
        N = 7
        nf = torch.randn(N, NODE_DIM)
        pos = torch.rand(N, 2) * 100
        owned = torch.zeros(N)
        owned[0] = 1.0
        owned[3] = 1.0

        src, tgt, frac, is_noop, log_prob, value = model_gat.sample_action(nf, pos, owned)
        assert isinstance(src, int)
        assert isinstance(tgt, int)
        assert isinstance(frac, int)
        assert isinstance(is_noop, bool)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        if not is_noop:
            assert src in [0, 3]  # must be an owned planet
            assert 0 <= frac < len(FRACTION_BUCKETS)

    def test_evaluate_action(self, model_gat):
        B, N = 4, 10
        nf = torch.randn(B, N, NODE_DIM)
        pos = torch.rand(B, N, 2) * 100
        owned = torch.ones(B, N)
        source = torch.tensor([0, 1, 2, 3])
        target = torch.tensor([1, 2, 3, 4])
        fraction = torch.tensor([0, 1, 2, 3])
        is_noop = torch.tensor([0.0, 0.0, 0.0, 1.0])

        log_prob, entropy, value = model_gat.evaluate_action(
            nf, pos, owned, source, target, fraction, is_noop,
        )
        assert log_prob.shape == (B,)
        assert entropy.shape == (B,)
        assert value.shape == (B,)
        # Log probs should be negative
        assert (log_prob <= 0).all()
        # Entropy should be non-negative
        assert (entropy >= 0).all()

    def test_factored_logprob_decomposition(self, model_gat):
        """Joint log-prob should equal sum of components."""
        N = 10
        nf = torch.randn(1, N, NODE_DIM)
        pos = torch.rand(1, N, 2) * 100
        owned = torch.ones(1, N)

        src_logits, tgt_logits, frac_logits = model_gat(nf, pos, owned)

        # Pick a specific action
        src_idx = 3
        tgt_idx = 7
        frac_idx = 2

        # Component log-probs
        src_dist = torch.distributions.Categorical(logits=src_logits[0])
        log_p_src = src_dist.log_prob(torch.tensor(src_idx)).item()

        tgt_dist = torch.distributions.Categorical(logits=tgt_logits[0, src_idx])
        log_p_tgt = tgt_dist.log_prob(torch.tensor(tgt_idx)).item()

        frac_dist = torch.distributions.Categorical(logits=frac_logits[0, src_idx, tgt_idx])
        log_p_frac = frac_dist.log_prob(torch.tensor(frac_idx)).item()

        expected_joint = log_p_src + log_p_tgt + log_p_frac

        # Evaluate via model method
        actual_joint, _, _ = model_gat.evaluate_action(
            nf, pos, owned,
            torch.tensor([src_idx]),
            torch.tensor([tgt_idx]),
            torch.tensor([frac_idx]),
            torch.tensor([0.0]),
        )

        assert actual_joint.item() == pytest.approx(expected_joint, abs=1e-4)

    def test_param_count_under_budget(self, model_gat):
        total = sum(p.numel() for p in model_gat.parameters())
        assert total < 500_000, f"Model has {total} params, exceeds 500k budget"
        print(f"  GAT model params: {total:,}")

    def test_param_count_sage(self, model_sage):
        total = sum(p.numel() for p in model_sage.parameters())
        assert total < 500_000
        print(f"  SAGE model params: {total:,}")


# ---------------------------------------------------------------------------
# Replay parser tests
# ---------------------------------------------------------------------------

class TestReplayParser:
    def test_angle_to_target(self):
        from .replay_parser import _angle_to_target
        # Planet 0 at (20, 50), Planet 1 at (80, 50) — angle 0 from planet 0 should find planet 1
        planets = [
            [0, 0, 20.0, 50.0, 2.0, 10, 3],
            [1, 1, 80.0, 50.0, 2.0, 10, 3],
            [2, -1, 50.0, 90.0, 2.0, 5, 2],
        ]
        idx = _angle_to_target(20.0, 50.0, 0.0, planets, source_id=0)
        assert idx == 1  # planet 1 is directly east

        # Angle pointing toward planet 2 (at 50,90 from 20,50 → atan2(40,30) ≈ 0.93)
        angle_to_p2 = math.atan2(90.0 - 50.0, 50.0 - 20.0)
        idx2 = _angle_to_target(20.0, 50.0, angle_to_p2, planets, source_id=0)
        assert idx2 == 2

    def test_fraction_to_bucket(self):
        from .replay_parser import _fraction_to_bucket
        assert _fraction_to_bucket(25, 100) == 0   # 0.25
        assert _fraction_to_bucket(50, 100) == 1   # 0.50
        assert _fraction_to_bucket(75, 100) == 2   # 0.75
        assert _fraction_to_bucket(100, 100) == 3  # 1.00
        assert _fraction_to_bucket(60, 100) == 1   # closer to 0.50 than 0.75

    def test_parse_real_replay(self):
        """Test parsing a real replay file if available."""
        replay_dir = os.path.join(os.path.dirname(__file__), "..", "kaggle_replays")
        if not os.path.exists(replay_dir):
            pytest.skip("No kaggle_replays directory")

        from .replay_parser import parse_all_replays
        # Just parse one subdirectory to keep the test fast
        subdirs = [d for d in os.listdir(replay_dir) if os.path.isdir(os.path.join(replay_dir, d))]
        if not subdirs:
            pytest.skip("No replay subdirectories")

        first_subdir = os.path.join(replay_dir, subdirs[0])
        files = [f for f in os.listdir(first_subdir) if f.endswith(".json")]
        if not files:
            pytest.skip("No JSON files in first subdirectory")

        from .replay_parser import parse_replay
        transitions = parse_replay(os.path.join(first_subdir, files[0]))
        assert len(transitions) > 0
        t = transitions[0]
        assert t.node_features.shape[1] == 10
        assert t.positions.shape[1] == 2
        assert t.mode in ("2p", "4p")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
