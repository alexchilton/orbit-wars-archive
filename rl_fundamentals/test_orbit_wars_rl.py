from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from fundamental_env import SimplifiedPlanetEnv
from orbit_wars_rl import ActionEncoder, OrbitWarsGCN, evaluate_mode, train_mode


class OrbitWarsRLSmokeTests(unittest.TestCase):
    def test_action_encoder_round_trip_and_noop(self) -> None:
        num_planets = 7
        action = ActionEncoder.encode(2, 5, 3, num_planets)
        source, target, fraction = ActionEncoder.decode(action, num_planets)
        self.assertEqual((source, target), (2, 5))
        self.assertGreater(fraction, 0.0)
        self.assertEqual(ActionEncoder.decode(ActionEncoder.noop_index(num_planets), num_planets), (None, None, 0.0))

    def test_observation_mask_has_valid_actions(self) -> None:
        env = SimplifiedPlanetEnv("two_player", seed=7)
        obs = env.build_gcn_observation(0)
        valid_count = sum(
            1
            for source in obs["action_mask"]
            for target in source
            for allowed in target
            if allowed
        )
        self.assertGreater(valid_count, 0)

    def test_short_gcn_training_run_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_mode("two_player", episodes=8, eval_episodes=4, seed=7, output_dir=Path(tmpdir))
            self.assertIn("evaluation", result)
            self.assertTrue((Path(tmpdir) / "two_player_gcn_policy.pt").exists())
            self.assertTrue((Path(tmpdir) / "two_player_gcn_summary.json").exists())
            metrics = evaluate_mode("two_player", Path(tmpdir) / "two_player_gcn_policy.pt", episodes=2, seed=17)
            self.assertIn("win_rate", metrics)

    def test_model_forward_matches_action_space(self) -> None:
        env = SimplifiedPlanetEnv("two_player", seed=3)
        obs = env.build_gcn_observation(0)
        model = OrbitWarsGCN()
        import torch

        logits, value = model(
            torch.tensor([obs["node_features"]], dtype=torch.float32),
            torch.tensor([obs["global_features"]], dtype=torch.float32),
            torch.tensor([obs["positions"]], dtype=torch.float32),
            torch.tensor([obs["velocities"]], dtype=torch.float32),
            torch.tensor([obs["valid_mask"]], dtype=torch.float32),
            torch.ones((1, ActionEncoder.total_actions(env.max_planets)), dtype=torch.bool),
        )
        self.assertEqual(logits.shape[-1], ActionEncoder.total_actions(env.max_planets))
        self.assertEqual(value.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
