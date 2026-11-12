from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from replay_midgame_experiment import compute_step_metrics, select_replay_candidates  # noqa: E402


def _step(step: int, planets, rewards=None):
    rewards = rewards or [0, 0]
    obs = {
        "planets": planets,
        "fleets": [],
        "step": step,
    }
    return [
        {"action": [], "observation": obs, "reward": rewards[0], "status": "ACTIVE"},
        {"action": [], "observation": {"player": 1}, "reward": rewards[1], "status": "ACTIVE"},
    ]


class ReplayMidgameTests(unittest.TestCase):
    def test_compute_step_metrics_counts_planets_and_fleets(self) -> None:
        observation = {
            "planets": [
                [0, 0, 0, 0, 1, 40, 2],
                [1, 1, 0, 0, 1, 30, 3],
            ],
            "fleets": [
                [0, 0, 0, 0, 0, 0, 5],
                [1, 1, 0, 0, 0, 1, 10],
            ],
        }
        metrics = compute_step_metrics(observation, player_index=0, num_agents=2, step=12)
        self.assertEqual(metrics.step, 12)
        self.assertEqual(metrics.target_ships, 45.0)
        self.assertEqual(metrics.production_by_player, [2.0, 3.0])
        self.assertEqual(metrics.target_rank, 1)

    def test_select_replay_candidates_finds_ahead_then_drop(self) -> None:
        replay = {
            "info": {"EpisodeId": 1, "TeamNames": ["alex chilton", "opponent"]},
            "rewards": [-1, 1],
            "steps": [
                _step(0, [[0, 0, 0, 0, 1, 10, 1], [1, 1, 0, 0, 1, 10, 1]]),
                _step(1, [[0, 0, 0, 0, 1, 10, 1], [1, 1, 0, 0, 1, 10, 1]]),
                _step(2, [[0, 0, 0, 0, 1, 60, 1], [1, 1, 0, 0, 1, 30, 1]]),
                _step(3, [[0, 0, 0, 0, 1, 20, 1], [1, 1, 0, 0, 1, 70, 1]]),
            ],
        }
        tmp = WORKSPACE_DIR / "_tmp_replay_candidate.json"
        tmp.write_text(__import__("json").dumps(replay), encoding="utf-8")
        try:
            candidates = select_replay_candidates([tmp], "alex chilton", min_step=2, max_step=2, horizon=1, max_candidates=1)
        finally:
            tmp.unlink(missing_ok=True)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].start_step, 2)
        self.assertGreater(candidates[0].historical_margin_drop, 0.0)

    def test_select_replay_candidates_can_keep_multiple_starts_per_replay(self) -> None:
        replay = {
            "info": {"EpisodeId": 2, "TeamNames": ["alex chilton", "opponent"]},
            "rewards": [-1, 1],
            "steps": [
                _step(0, [[0, 0, 0, 0, 1, 20, 1], [1, 1, 0, 0, 1, 20, 1]]),
                _step(1, [[0, 0, 0, 0, 1, 70, 1], [1, 1, 0, 0, 1, 30, 1]]),
                _step(2, [[0, 0, 0, 0, 1, 65, 1], [1, 1, 0, 0, 1, 35, 1]]),
                _step(3, [[0, 0, 0, 0, 1, 25, 1], [1, 1, 0, 0, 1, 75, 1]]),
                _step(4, [[0, 0, 0, 0, 1, 60, 1], [1, 1, 0, 0, 1, 40, 1]]),
                _step(5, [[0, 0, 0, 0, 1, 15, 1], [1, 1, 0, 0, 1, 85, 1]]),
            ],
        }
        tmp = WORKSPACE_DIR / "_tmp_replay_multi_candidate.json"
        tmp.write_text(__import__("json").dumps(replay), encoding="utf-8")
        try:
            candidates = select_replay_candidates(
                [tmp],
                "alex chilton",
                min_step=1,
                max_step=4,
                horizon=1,
                max_candidates=4,
                candidates_per_replay=2,
            )
        finally:
            tmp.unlink(missing_ok=True)
        self.assertEqual(len(candidates), 2)
        self.assertEqual([candidate.start_step for candidate in candidates], [4, 2])


if __name__ == "__main__":
    unittest.main()
