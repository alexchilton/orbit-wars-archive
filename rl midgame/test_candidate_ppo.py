from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from midgame_policy import DecisionSample, FEATURE_NAMES, PPOMissionPolicy, load_policy_json  # noqa: E402


class CandidatePPOPolicyTests(unittest.TestCase):
    def test_choose_and_metadata_round_trip(self) -> None:
        policy = PPOMissionPolicy(hidden_size=16, seed=7)
        vectors = [
            [1.0] + [0.0] * (len(FEATURE_NAMES) - 1),
            [0.0] * len(FEATURE_NAMES),
        ]
        choice = policy.choose(vectors, explore=False)
        self.assertIn(choice.index, (0, 1))
        meta = policy.decision_metadata(vectors, choice)
        self.assertIn("old_log_prob", meta)
        self.assertIn("value_estimate", meta)

    def test_update_and_reload(self) -> None:
        policy = PPOMissionPolicy(hidden_size=16, seed=7)
        before = policy.average_abs_weight()
        sample = DecisionSample(
            feature_vectors=[
                [1.0] + [0.0] * (len(FEATURE_NAMES) - 1),
                [0.0] * len(FEATURE_NAMES),
            ],
            chosen_index=0,
            probabilities=[0.5, 0.5],
            metadata={"old_log_prob": -0.69, "value_estimate": 0.0},
        )
        metrics = policy.update([sample], reward=1.0, learning_rate=3e-4)
        self.assertIn("loss", metrics)
        self.assertNotEqual(before, policy.average_abs_weight())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ppo_policy.json"
            policy.save_json(path)
            loaded = load_policy_json(path)
        self.assertIsInstance(loaded, PPOMissionPolicy)


if __name__ == "__main__":
    unittest.main()
