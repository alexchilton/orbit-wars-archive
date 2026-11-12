from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from synthetic_opening_experiment import mutate_step_state  # noqa: E402


class SyntheticOpeningTests(unittest.TestCase):
    def test_mutate_step_state_changes_ships_but_preserves_shape(self) -> None:
        step_state = [
            {
                "observation": {
                    "player": 0,
                    "step": 8,
                    "planets": [
                        [0, 0, 10.0, 10.0, 1.0, 40, 3],
                        [1, -1, 20.0, 20.0, 1.0, 25, 2],
                    ],
                    "fleets": [
                        [0, 1, 30.0, 30.0, 0.4, 1, 12],
                    ],
                }
            },
            {
                "observation": {
                    "player": 1,
                    "step": 8,
                    "planets": [
                        [0, 0, 10.0, 10.0, 1.0, 40, 3],
                        [1, -1, 20.0, 20.0, 1.0, 25, 2],
                    ],
                    "fleets": [
                        [0, 1, 30.0, 30.0, 0.4, 1, 12],
                    ],
                }
            },
        ]
        before = copy.deepcopy(step_state)
        after = mutate_step_state(step_state, __import__("random").Random(7))

        self.assertEqual(len(after), len(before))
        self.assertEqual(len(after[0]["observation"]["planets"]), len(before[0]["observation"]["planets"]))
        self.assertEqual(len(after[0]["observation"]["fleets"]), len(before[0]["observation"]["fleets"]))
        self.assertNotEqual(after[0]["observation"]["planets"][0][5], before[0]["observation"]["planets"][0][5])
        self.assertGreaterEqual(after[0]["observation"]["fleets"][0][6], 1)


if __name__ == "__main__":
    unittest.main()
