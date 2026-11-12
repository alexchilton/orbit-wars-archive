from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = ROOT / "snapshots" / "stage4_leaderboard_search_base.py"
MODULE_NAME = "_test_stage4_leaderboard_search_base"


def load_stage4_base():
    if MODULE_NAME in sys.modules:
        return sys.modules[MODULE_NAME]
    spec = importlib.util.spec_from_file_location(MODULE_NAME, BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


base = load_stage4_base()


def make_planet(
    planet_id: int,
    owner: int,
    ships: int,
    production: int,
    x: float,
    y: float,
    radius: float = 1.0,
):
    return base.Planet(planet_id, owner, x, y, radius, ships, production)


class Stage4LeaderboardSearchBaseTests(unittest.TestCase):
    def make_logic(self):
        logic = base.DecisionLogic.__new__(base.DecisionLogic)
        logic.used_donor_ids = set()
        logic.committed_ships = defaultdict(int)
        logic.targeted_planet_ids = set()
        logic.planned_commitments = defaultdict(list)
        logic.expired = lambda: False
        return logic

    def test_build_modes_marks_runaway_recovery(self) -> None:
        logic = self.make_logic()
        logic.state = SimpleNamespace(
            planets=[
                make_planet(1, 0, 42, 3, 85.0, 85.0),
                make_planet(2, 1, 130, 8, 12.0, 12.0),
                make_planet(3, 2, 70, 4, 82.0, 14.0),
                make_planet(4, 3, 25, 2, 16.0, 82.0),
            ],
            fleets=[],
            player=0,
            num_players=4,
            step=64,
        )
        logic._is_4p_crowded = lambda: False
        logic._four_player_safe_neutral_count = lambda: 1
        logic._four_player_stage_ready = lambda: True

        modes = logic._build_modes()

        self.assertTrue(modes["is_behind"])
        self.assertTrue(modes["four_player_recovery_mode"])
        self.assertEqual(modes["runaway_owner"], 1)
        self.assertTrue(modes["four_player_pivot_ready"])

    def test_frontline_reserve_uses_crowded_floor(self) -> None:
        logic = self.make_logic()
        source = make_planet(1, 0, 50, 3, 80.0, 80.0)
        enemy = make_planet(2, 1, 30, 3, 62.0, 80.0)
        logic.state = SimpleNamespace(
            num_players=4,
            step=24,
            enemy_planets=[enemy],
        )
        logic.modes = {"is_ahead": False, "four_player_crowded": True}

        reserve = logic._frontline_reserve(source)

        self.assertGreaterEqual(
            reserve,
            int(source.ships * base.FOUR_PLAYER_CROWDED_FRONT_RESERVE_RATIO),
        )

    def test_position_multiplier_penalizes_long_tight_neutral(self) -> None:
        logic = self.make_logic()
        target = make_planet(9, -1, 6, 3, 68.0, 50.0)
        logic.state = SimpleNamespace(
            num_players=4,
            step=36,
        )
        logic.modes = {"is_dominating": False, "four_player_crowded": True}
        logic.reaction_map = {target.id: (16, 17)}

        mult = logic._four_player_neutral_position_multiplier(target, 16)

        self.assertLess(mult, 0.75)


if __name__ == "__main__":
    unittest.main()
