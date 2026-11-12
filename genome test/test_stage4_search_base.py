from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = ROOT / "snapshots" / "stage4_search_base.py"
MODULE_NAME = "_test_stage4_search_base"


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


class Stage4SearchBaseTests(unittest.TestCase):
    def make_logic(self):
        logic = base.DecisionLogic.__new__(base.DecisionLogic)
        logic.used_donor_ids = set()
        logic.committed_ships = defaultdict(int)
        logic.targeted_planet_ids = set()
        logic.planned_commitments = defaultdict(list)
        logic.expired = lambda: False
        return logic

    def test_build_modes_marks_dogpile_pressure(self) -> None:
        logic = self.make_logic()
        logic.state = SimpleNamespace(
            planets=[
                make_planet(1, 0, 40, 3, 80.0, 80.0),
                make_planet(2, 1, 52, 4, 62.0, 80.0),
                make_planet(3, 2, 46, 3, 80.0, 62.0),
                make_planet(4, 3, 28, 2, 16.0, 14.0),
            ],
            fleets=[],
            player=0,
            num_players=4,
            step=36,
            my_planets=[make_planet(1, 0, 40, 3, 80.0, 80.0)],
            enemy_planets=[
                make_planet(2, 1, 52, 4, 62.0, 80.0),
                make_planet(3, 2, 46, 3, 80.0, 62.0),
                make_planet(4, 3, 28, 2, 16.0, 14.0),
            ],
        )
        logic._four_player_safe_neutral_count = lambda: 2
        logic._four_player_stage_ready = lambda: True

        modes = logic._build_modes()

        self.assertTrue(modes["four_player_dogpile_mode"])
        self.assertEqual(modes["pressure_owner_ids"], [1, 2])

    def test_turn_launch_cap_tightens_in_dogpile_mode(self) -> None:
        logic = self.make_logic()
        logic.state = SimpleNamespace(num_players=4, step=44)
        logic.modes = {
            "is_cleanup": False,
            "four_player_dogpile_mode": True,
            "four_player_recovery_mode": False,
            "four_player_pivot_ready": False,
            "is_behind": True,
            "is_dominating": False,
        }
        logic._mtmr_opening_active = lambda: False
        logic._shun_opening_active = lambda: False
        logic._four_player_pressure_active = lambda: True

        self.assertEqual(logic._turn_launch_cap(), base.FOUR_PLAYER_DOGPILE_LAUNCH_CAP)

    def test_duel_frontline_reserve_appears_when_ahead(self) -> None:
        logic = self.make_logic()
        frontline = make_planet(1, 0, 50, 3, 80.0, 80.0)
        enemy = make_planet(2, 1, 40, 3, 64.0, 80.0)
        logic.state = SimpleNamespace(
            num_players=2,
            enemy_planets=[enemy],
        )
        logic.modes = {
            "is_ahead": True,
            "is_dominating": False,
        }

        self.assertGreater(logic._duel_frontline_reserve(frontline), 0)

    def test_duel_finishing_triggers_earlier_than_stage3_base(self) -> None:
        logic = self.make_logic()
        logic.state = SimpleNamespace(
            planets=[
                make_planet(1, 0, 78, 9, 82.0, 82.0),
                make_planet(2, 1, 42, 7, 18.0, 18.0),
            ],
            fleets=[],
            player=0,
            num_players=2,
            step=92,
            my_planets=[make_planet(1, 0, 78, 9, 82.0, 82.0)],
            enemy_planets=[make_planet(2, 1, 42, 7, 18.0, 18.0)],
        )

        modes = logic._build_modes()

        self.assertTrue(modes["is_finishing"])

    def test_duel_turn_launch_cap_tightens_when_ahead(self) -> None:
        logic = self.make_logic()
        logic.state = SimpleNamespace(num_players=2, step=44)
        logic.modes = {
            "is_ahead": True,
            "is_dominating": False,
            "is_finishing": False,
        }
        logic._mtmr_opening_active = lambda: False
        logic._shun_opening_active = lambda: False

        self.assertEqual(logic._turn_launch_cap(), base.DUEL_PROTECT_LAUNCH_CAP)

    def test_orbiting_counterpost_candidate_prefers_safe_orbiting_neutral(self) -> None:
        logic = self.make_logic()
        logic.state = SimpleNamespace(
            num_players=4,
            enemy_planets=[make_planet(9, 1, 18, 3, 62.0, 62.0)],
        )
        logic._is_safe_neutral = lambda target: True

        self.assertTrue(
            logic._orbiting_counterpost_candidate(make_planet(7, -1, 6, 4, 54.0, 54.0))
        )
        self.assertFalse(
            logic._orbiting_counterpost_candidate(make_planet(8, -1, 6, 4, 88.0, 88.0))
        )

    def test_dogpile_attack_prefers_backline_donor(self) -> None:
        logic = self.make_logic()
        backline = make_planet(1, 0, 28, 3, 86.0, 86.0)
        frontline = make_planet(2, 0, 28, 3, 58.0, 58.0)
        target = make_planet(10, 1, 8, 4, 52.0, 52.0)
        other_enemy = make_planet(11, 2, 10, 3, 62.0, 58.0)
        neutral_a = make_planet(20, -1, 5, 3, 72.0, 72.0)
        neutral_b = make_planet(21, -1, 5, 3, 74.0, 68.0)
        neutral_c = make_planet(22, -1, 5, 2, 70.0, 66.0)
        logic.state = SimpleNamespace(
            enemy_planets=[target, other_enemy],
            neutral_planets=[neutral_a, neutral_b, neutral_c],
            my_planets=[frontline, backline],
            planets_by_id={backline.id: backline, frontline.id: frontline, target.id: target},
            num_players=4,
            is_early=True,
            is_opening=True,
            step=44,
            player=0,
            remaining_steps=220,
        )
        logic.modes = {
            "four_player_dogpile_mode": True,
            "four_player_recovery_mode": False,
            "four_player_pivot_ready": False,
            "four_player_safe_neutral_count": 3,
            "is_dominating": False,
            "pressure_owner_ids": [1, 2],
        }
        logic._mtmr_opening_active = lambda: False
        logic._shun_opening_active = lambda: False
        logic._available_my_planets = lambda: [frontline, backline]
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

        self.assertGreaterEqual(len(missions), 1)
        self.assertEqual(missions[0].source_ids, [backline.id])


if __name__ == "__main__":
    unittest.main()
