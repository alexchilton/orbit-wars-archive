from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = ROOT / "snapshots" / "stage3_search_base.py"
MODULE_NAME = "_test_stage3_search_base"


def load_stage3_base():
    if MODULE_NAME in sys.modules:
        return sys.modules[MODULE_NAME]
    spec = importlib.util.spec_from_file_location(MODULE_NAME, BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


base = load_stage3_base()


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


class Stage3SearchBaseTests(unittest.TestCase):
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
        logic._four_player_safe_neutral_count = lambda: 1
        logic._four_player_stage_ready = lambda: True

        modes = logic._build_modes()

        self.assertTrue(modes["is_behind"])
        self.assertTrue(modes["four_player_recovery_mode"])
        self.assertEqual(modes["runaway_owner"], 1)
        self.assertTrue(modes["four_player_pivot_ready"])

    def test_turn_launch_cap_relaxes_in_recovery_mode(self) -> None:
        logic = self.make_logic()
        logic.state = SimpleNamespace(num_players=4, step=80)
        logic.modes = {
            "is_cleanup": False,
            "four_player_recovery_mode": True,
            "four_player_pivot_ready": False,
            "is_behind": True,
            "is_dominating": False,
        }
        logic._mtmr_opening_active = lambda: False
        logic._shun_opening_active = lambda: False
        logic._four_player_pressure_active = lambda: False

        self.assertEqual(logic._turn_launch_cap(), base.FOUR_PLAYER_RECOVERY_LAUNCH_CAP)

    def test_recovery_mode_still_builds_attack_missions_with_many_neutrals(self) -> None:
        logic = self.make_logic()
        source = make_planet(1, 0, 24, 3, 80.0, 80.0)
        target = make_planet(10, 1, 8, 4, 30.0, 30.0)
        neutral_a = make_planet(20, -1, 5, 3, 70.0, 70.0)
        neutral_b = make_planet(21, -1, 5, 3, 68.0, 74.0)
        neutral_c = make_planet(22, -1, 5, 2, 72.0, 66.0)
        logic.state = SimpleNamespace(
            enemy_planets=[target],
            neutral_planets=[neutral_a, neutral_b, neutral_c],
            my_planets=[source],
            planets_by_id={source.id: source, target.id: target},
            num_players=4,
            is_early=True,
            is_opening=True,
            step=52,
            player=0,
            remaining_steps=220,
        )
        logic.modes = {
            "four_player_recovery_mode": True,
            "four_player_pivot_ready": False,
            "four_player_safe_neutral_count": 3,
            "is_dominating": False,
            "runaway_owner": 1,
        }
        logic._mtmr_opening_active = lambda: False
        logic._shun_opening_active = lambda: False
        logic._available_my_planets = lambda: [source]
        logic._planet_surplus = lambda donor: 12
        logic._settle_plan = lambda donor, target, surplus, mission, max_turn=35: base.CapturePlan(
            target=target,
            ships=10,
            angle=0.0,
            eta=7,
            eval_turn=7,
            required_ships=9,
        )
        logic._candidate_time_valid = lambda target, eta: True
        logic._opening_mission_allowed = lambda target, plan, mission: True
        logic._target_value = lambda target, eta, mission: 120.0
        logic._score_mission = lambda value, ships, eta, target, mission: 4.0

        missions = []
        logic._build_attack_missions(missions)

        self.assertEqual(len(missions), 1)
        self.assertEqual(missions[0].target_id, target.id)


if __name__ == "__main__":
    unittest.main()
