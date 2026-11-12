from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

WORKSPACE_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from midgame_features import FEATURE_NAMES, build_mission_feature_bundle  # noqa: E402
from midgame_policy import DecisionSample, LinearMissionPolicy, MLPMissionPolicy, load_policy_json  # noqa: E402
from midgame_rl_agent import EpisodeRecorder, MidgameRLConfig, MidgameRLDecisionLogic  # noqa: E402


class RLMidgameTests(unittest.TestCase):
    def _fake_logic(self) -> SimpleNamespace:
        target = SimpleNamespace(id=7, owner=1, production=4, ships=26)
        state = SimpleNamespace(
            step=42,
            remaining_steps=158,
            is_opening=False,
            is_very_late=False,
            num_players=2,
            player=0,
            planets_by_id={7: target},
        )
        return SimpleNamespace(
            state=state,
            modes={
                "my_total": 90,
                "enemy_total": 85,
                "owner_prod": {0: 11, 1: 10},
                "is_ahead": True,
                "is_behind": False,
                "is_dominating": False,
                "is_finishing": False,
            },
            enemy_priority={1: 1.1},
        )

    def test_feature_bundle_matches_feature_names(self) -> None:
        logic = self._fake_logic()
        mission = SimpleNamespace(
            score=18.0,
            source_ids=[1],
            target_id=7,
            angles=[0.2],
            etas=[11],
            ships=[17],
            needed=17,
            mission="attack",
        )
        bundle = build_mission_feature_bundle(logic, mission, base_value=21.0, existing_moves=1, turn_launch_cap=3)
        self.assertEqual(len(bundle.vector), len(FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(value) for value in bundle.vector))
        self.assertEqual(bundle.metadata["target_id"], 7)
        self.assertEqual(bundle.metadata["mission"], "attack")

    def test_policy_update_changes_weights(self) -> None:
        policy = LinearMissionPolicy()
        before = list(policy.weights)
        sample = DecisionSample(
            feature_vectors=[
                [1.0] + [0.0] * (len(FEATURE_NAMES) - 1),
                [0.0] * len(FEATURE_NAMES),
            ],
            chosen_index=0,
            probabilities=[0.5, 0.5],
            metadata={"step": 50},
        )
        policy.update([sample], reward=1.0, learning_rate=0.2)
        self.assertNotEqual(before, policy.weights)
        self.assertGreater(policy.weights[0], 0.0)

    def test_policy_roundtrip(self) -> None:
        policy = LinearMissionPolicy(weights=[0.1] * len(FEATURE_NAMES), temperature=0.7)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.json"
            policy.save_json(path)
            loaded = LinearMissionPolicy.load_json(path)
        self.assertEqual(policy.weights, loaded.weights)
        self.assertAlmostEqual(policy.temperature, loaded.temperature)

    def test_mlp_policy_update_and_roundtrip(self) -> None:
        policy = MLPMissionPolicy(hidden_size=8, seed=7)
        before = policy.average_abs_weight()
        sample = DecisionSample(
            feature_vectors=[
                [1.0] + [0.0] * (len(FEATURE_NAMES) - 1),
                [0.0] * len(FEATURE_NAMES),
            ],
            chosen_index=0,
            probabilities=[0.5, 0.5],
            metadata={"step": 50},
        )
        policy.update([sample], reward=1.0, learning_rate=0.1)
        self.assertNotEqual(before, policy.average_abs_weight())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mlp_policy.json"
            policy.save_json(path)
            loaded = load_policy_json(path)
        self.assertIsInstance(loaded, MLPMissionPolicy)
        self.assertEqual(policy.hidden_size, loaded.hidden_size)

    def test_episode_recorder_resets_and_pops(self) -> None:
        recorder = EpisodeRecorder()
        recorder.start_turn(player=0, step=0)
        recorder.record(0, DecisionSample(feature_vectors=[[0.0]], chosen_index=0, probabilities=[1.0]))
        recorder.start_turn(player=0, step=1)
        self.assertEqual(len(recorder.pop(0)), 1)

        recorder.start_turn(player=0, step=0)
        self.assertEqual(recorder.pop(0), [])

    def test_rl_window_can_be_opened_during_opening(self) -> None:
        logic = MidgameRLDecisionLogic.__new__(MidgameRLDecisionLogic)
        logic.state = SimpleNamespace(
            step=12,
            is_opening=True,
            is_very_late=False,
            num_players=2,
        )
        logic.modes = {
            "my_total": 60.0,
            "enemy_total": 58.0,
            "is_finishing": False,
        }
        logic.rl_config = MidgameRLConfig(
            activation_turn=0,
            max_turn=40,
            min_candidates=1,
            contested_only=False,
            allow_opening=True,
        )
        self.assertTrue(logic._rl_window_open(2))

    def test_rl_window_force_flag_bypasses_phase_gating(self) -> None:
        logic = MidgameRLDecisionLogic.__new__(MidgameRLDecisionLogic)
        logic.state = SimpleNamespace(
            step=8,
            is_opening=True,
            is_very_late=False,
            num_players=4,
        )
        logic.modes = {
            "my_rank": 4,
            "is_dominating": False,
            "is_cleanup": False,
        }
        logic.rl_config = MidgameRLConfig(
            activation_turn=0,
            max_turn=20,
            min_candidates=1,
            contested_only=True,
            force_rl_window=True,
        )
        self.assertTrue(logic._rl_window_open(1))


if __name__ == "__main__":
    unittest.main()
