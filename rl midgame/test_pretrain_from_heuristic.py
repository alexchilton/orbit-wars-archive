from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


WORKSPACE_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from midgame_features import FEATURE_NAMES  # noqa: E402
from midgame_policy import MLPMissionPolicy  # noqa: E402
from pretrain_from_heuristic import build_heuristic_sample, pretrain_policy  # noqa: E402


class HeuristicPretrainTests(unittest.TestCase):
    def test_build_heuristic_sample_targets_best_candidate(self) -> None:
        target = SimpleNamespace(id=7, owner=1, production=4, ships=26)
        logic = SimpleNamespace(
            state=SimpleNamespace(
                step=42,
                remaining_steps=158,
                is_opening=False,
                is_very_late=False,
                num_players=2,
                player=0,
                planets_by_id={7: target},
            ),
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
            _turn_launch_cap=lambda: 3,
            _mission_can_commit=lambda mission, **kwargs: True,
            _mission_blocks_target=lambda mission: True,
        )
        missions = [
            SimpleNamespace(score=18.0, source_ids=[1], target_id=7, angles=[0.2], etas=[11], ships=[17], needed=17, mission="attack"),
            SimpleNamespace(score=12.0, source_ids=[2], target_id=7, angles=[0.1], etas=[9], ships=[12], needed=12, mission="attack"),
        ]

        sample = build_heuristic_sample(logic, missions, top_k=4)

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample["target_index"], 1)
        self.assertEqual(sample["heuristic_best_idx"], 0)
        self.assertEqual(sample["candidate_count"], 2)
        self.assertEqual(len(sample["vectors"]), 3)
        self.assertEqual(len(sample["vectors"][0]), len(FEATURE_NAMES))
        self.assertEqual(sample["vectors"][0][-1], 1.0)

    def test_pretrain_updates_mlp_policy(self) -> None:
        policy = MLPMissionPolicy(hidden_size=8, seed=7)
        vectors = [
            [1.0] + [0.0] * (len(FEATURE_NAMES) - 1),
            [0.0] * len(FEATURE_NAMES),
        ]
        before_margin = policy.score(vectors[0]) - policy.score(vectors[1])

        result = pretrain_policy(
            policy,
            samples=[{
                "vectors": vectors,
                "target_index": 0,
            }],
            epochs=8,
            learning_rate=0.05,
        )

        after_margin = policy.score(vectors[0]) - policy.score(vectors[1])
        self.assertGreater(after_margin, before_margin)
        self.assertGreaterEqual(result["final_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
