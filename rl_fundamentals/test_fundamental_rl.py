from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from fundamental_env import SimplifiedPlanetEnv
from train_q_policy import TabularQPolicy, evaluate_policy, train_mode


class FundamentalRLTests(unittest.TestCase):
    def test_two_player_state_encoding_is_stable(self) -> None:
        env = SimplifiedPlanetEnv("two_player", seed=7)
        state = env.encode_state(0)
        self.assertEqual(len(state), 8)
        self.assertTrue(all(isinstance(value, int) for value in state))

    def test_four_player_heuristic_episode_terminates(self) -> None:
        env = SimplifiedPlanetEnv("four_player", seed=11)
        for _ in range(200):
            actions = {}
            for player in range(env.num_players):
                if env._player_alive(player):
                    actions[player] = env.heuristic_action(player)
            result = env.step(actions)
            if result.done:
                break
        else:
            self.fail("Heuristic episode did not terminate")

        self.assertIsNotNone(result.winner)
        self.assertIn(result.winner, range(env.num_players))

    def test_short_training_run_emits_policy(self) -> None:
        result = train_mode(
            mode="two_player",
            episodes=40,
            eval_episodes=8,
            seed=5,
            learning_rate=0.2,
            gamma=0.9,
            epsilon_start=0.2,
            epsilon_end=0.05,
        )
        policy = result["policy"]
        self.assertIsInstance(policy, TabularQPolicy)
        self.assertGreater(len(policy.q_values), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.json"
            policy.save_json(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("q_values", payload)
        self.assertIn("visit_counts", payload)

    def test_policy_can_be_evaluated(self) -> None:
        env = SimplifiedPlanetEnv("two_player", seed=3)
        policy = TabularQPolicy(actions=env.action_names())
        metrics = evaluate_policy("two_player", policy, episodes=6, seed=19)
        self.assertIn("win_rate", metrics)
        self.assertIn("avg_reward", metrics)


if __name__ == "__main__":
    unittest.main()
