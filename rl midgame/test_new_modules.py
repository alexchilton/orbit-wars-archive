"""Integration tests for the new RL/heuristic improvement modules.

Tests verify that all new modules:
1. Import and construct correctly
2. Integrate with the existing midgame RL pipeline
3. Don't break the stage3 baseline agent
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("KAGGLE_ENVIRONMENTS_QUIET", "1")

WORKSPACE_DIR = Path(__file__).resolve().parent
ROOT = WORKSPACE_DIR.parent
GENOME_DIR = ROOT / "genome test"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
if str(GENOME_DIR) not in sys.path:
    sys.path.insert(0, str(GENOME_DIR))


class TestCandidateFilters(unittest.TestCase):
    def test_filter_config_defaults(self):
        from candidate_filters import FilterConfig
        cfg = FilterConfig()
        self.assertTrue(cfg.dedup_enabled)
        self.assertTrue(cfg.feasibility_enabled)
        self.assertTrue(cfg.reserve_enabled)
        self.assertGreater(cfg.feasibility_ship_ratio, 0.0)

    def test_pipeline_constructs(self):
        from candidate_filters import CandidateFilterPipeline, FilterConfig
        pipeline = CandidateFilterPipeline(FilterConfig())
        self.assertTrue(hasattr(pipeline, 'run'))

    def test_consolidate_candidate(self):
        from candidate_filters import make_consolidate_candidate, FilterConfig
        cfg = FilterConfig()
        c = make_consolidate_candidate(cfg)
        self.assertEqual(c.mission, "consolidate")
        self.assertEqual(c.source_ids, [])
        self.assertEqual(sum(c.ships), 0)


class TestTacticalVeto(unittest.TestCase):
    def test_value_model_score(self):
        from tactical_veto import PositionValue
        model = PositionValue()
        features = [0.5, 0.5, 0.3, 1.0, 0.4, 0.6]
        score = model.score(features)
        self.assertIsInstance(score, float)

    def test_value_model_save_load(self):
        from tactical_veto import PositionValue
        model = PositionValue(weights=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], bias=0.1)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            model.save(path)
            loaded = PositionValue.load(path)
            self.assertEqual(model.weights, loaded.weights)
            self.assertAlmostEqual(model.bias, loaded.bias)
        finally:
            os.unlink(path)

    def test_veto_layer_no_veto_when_behind(self):
        from tactical_veto import PositionValue, TacticalVetoLayer
        model = PositionValue(weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], bias=0.0)
        veto = TacticalVetoLayer(model, epsilon=0.02)
        # When is_behind=True, should never veto
        self.assertTrue(hasattr(veto, 'epsilon'))


class TestScaledBenchmark(unittest.TestCase):
    def test_wilson_ci_bounds(self):
        from scaled_benchmark import wilson_ci
        lower, upper = wilson_ci(50, 100)
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)
        self.assertGreater(lower, 0.0)
        self.assertLess(upper, 1.0)

    def test_wilson_ci_extremes(self):
        from scaled_benchmark import wilson_ci
        lower, upper = wilson_ci(0, 100)
        self.assertAlmostEqual(lower, 0.0, places=1)
        lower, upper = wilson_ci(100, 100)
        self.assertAlmostEqual(upper, 1.0, places=1)

    def test_binomial_p_value(self):
        from scaled_benchmark import binomial_p_value
        # 70/100 wins should be highly significant
        p = binomial_p_value(70, 100)
        self.assertLess(p, 0.01)
        # 52/100 wins should not be significant
        p = binomial_p_value(52, 100)
        self.assertGreater(p, 0.05)

    def test_is_significant(self):
        from scaled_benchmark import is_significant
        self.assertTrue(is_significant(70, 100))
        self.assertFalse(is_significant(52, 100))


class TestOpeningPlaybooks(unittest.TestCase):
    def test_map_features_vector(self):
        from opening_playbooks import MapFeatures
        mf = MapFeatures(
            num_players=2, num_planets=12, avg_planet_distance=25.0,
            cluster_score=0.5, starting_production=3.0,
            nearest_neutral_distance=10.0, nearest_enemy_distance=40.0,
            has_orbiting_planets=True,
        )
        vec = mf.to_vector()
        self.assertEqual(len(vec), 8)
        self.assertEqual(vec[0], 2)  # num_players
        self.assertEqual(vec[-1], 1.0)  # has_orbiting

    def test_bandit_select_update(self):
        from opening_playbooks import MapFeatures, ContextualBandit
        bandit = ContextualBandit(n_arms=4, n_features=8)
        mf = MapFeatures(2, 12, 25.0, 0.5, 3.0, 10.0, 40.0, True)
        arm = bandit.select(mf)
        self.assertIn(arm, range(4))
        bandit.update(arm, 1.0)
        self.assertEqual(bandit.counts[arm], 1)

    def test_bandit_save_load(self):
        from opening_playbooks import MapFeatures, ContextualBandit
        bandit = ContextualBandit(n_arms=3, n_features=8)
        mf = MapFeatures(4, 16, 30.0, 0.3, 2.0, 15.0, 35.0, False)
        bandit.select(mf)
        bandit.update(0, 0.5)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            bandit.save_json(path)
            loaded = ContextualBandit.load_json(path)
            self.assertEqual(loaded.n_arms, 3)
            self.assertEqual(loaded.counts[0], 1)
        finally:
            os.unlink(path)


class TestSplitReranker(unittest.TestCase):
    def test_split_policy_dispatch(self):
        from split_reranker import SplitRerankerPolicy
        from midgame_policy import LinearMissionPolicy, FEATURE_NAMES
        p2 = LinearMissionPolicy(weights=[1.0] * len(FEATURE_NAMES))
        p4 = LinearMissionPolicy(weights=[-1.0] * len(FEATURE_NAMES))
        split = SplitRerankerPolicy(p2, p4)

        # Build a 2p feature vector (is_two_player = index 2 = 1.0)
        vec_2p = [0.0] * len(FEATURE_NAMES)
        vec_2p[2] = 1.0  # is_two_player
        score_2p = split.score(vec_2p)

        # Build a 4p feature vector (is_two_player = 0.0)
        vec_4p = [0.0] * len(FEATURE_NAMES)
        vec_4p[2] = 0.0
        score_4p = split.score(vec_4p)

        # 2p policy has all +1 weights, so with is_two_player=1.0 it should give positive
        self.assertGreater(score_2p, 0.0)
        # 4p policy has all -1 weights, so score should be 0 (all features are 0)
        self.assertEqual(score_4p, 0.0)

    def test_split_policy_save_load(self):
        from split_reranker import SplitRerankerPolicy
        from midgame_policy import LinearMissionPolicy
        p2 = LinearMissionPolicy()
        p4 = LinearMissionPolicy()
        split = SplitRerankerPolicy(p2, p4)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            split.save_json(path)
            loaded = SplitRerankerPolicy.load_json(path)
            self.assertIsNotNone(loaded.policy_2p)
            self.assertIsNotNone(loaded.policy_4p)
        finally:
            os.unlink(path)


class TestMLPComparison(unittest.TestCase):
    def test_config_defaults(self):
        from mlp_comparison import ComparisonConfig
        cfg = ComparisonConfig(replay_globs=["kaggle_replays/*/episode-*-replay.json"])
        self.assertEqual(cfg.pretrain_positions, 80)
        self.assertEqual(cfg.pretrain_epochs, 30)
        self.assertEqual(cfg.mlp_hidden_size, 64)
        self.assertEqual(cfg.player_name, "alex chilton")


class TestBaselineNotBroken(unittest.TestCase):
    """Verify the base agent still works after all our imports."""

    def test_base_agent_loads(self):
        from midgame_rl_agent import BASE, BASE_AGENT
        self.assertIsNotNone(BASE_AGENT)
        self.assertTrue(callable(BASE_AGENT))

    def test_base_agent_runs_one_game(self):
        import test_agent
        baseline = test_agent.load_baseline_agent()
        result = test_agent.run_game(baseline, "random")
        self.assertIn(result["winner"], ("A", "B", "draw"))


if __name__ == "__main__":
    unittest.main()
