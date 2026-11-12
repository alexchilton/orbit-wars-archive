import math
import unittest
from collections import defaultdict
from types import SimpleNamespace

import orbit_wars_agent as agent_mod


def make_planet(
    planet_id: int,
    owner: int,
    ships: int,
    production: int,
    x: float,
    y: float,
    radius: float = 1.0,
):
    return agent_mod.Planet(planet_id, owner, x, y, radius, ships, production)


class OrbitWarsAgentTests(unittest.TestCase):
    def make_logic(self):
        logic = agent_mod.DecisionLogic.__new__(agent_mod.DecisionLogic)
        logic.used_donor_ids = set()
        logic.committed_ships = defaultdict(int)
        logic.targeted_planet_ids = set()
        logic.planned_commitments = defaultdict(list)
        logic.expired = lambda: False
        return logic

    def test_opening_capture_allows_static_high_production_neutral(self):
        logic = self.make_logic()
        target = make_planet(10, -1, 7, 5, 92.0, 92.0)
        plan = agent_mod.CapturePlan(
            target=target,
            ships=10,
            angle=0.0,
            eta=8,
            eval_turn=8,
            required_ships=10,
        )
        logic.state = SimpleNamespace(
            is_opening=True,
            is_early=True,
            player=0,
            step=8,
        )
        logic.reaction_map = {target.id: (8, 15)}

        self.assertGreater(
            logic._opening_capture_confidence(target, plan, "expand"),
            agent_mod.OPENING_MIN_CONFIDENCE,
        )
        self.assertTrue(logic._opening_mission_allowed(target, plan, "expand"))

    def test_opening_capture_rejects_slow_low_value_orbiting_neutral(self):
        logic = self.make_logic()
        target = make_planet(11, -1, 9, 1, 60.0, 60.0)
        plan = agent_mod.CapturePlan(
            target=target,
            ships=12,
            angle=0.0,
            eta=13,
            eval_turn=13,
            required_ships=10,
        )
        logic.state = SimpleNamespace(
            is_opening=True,
            is_early=True,
            player=0,
            step=8,
        )
        logic.reaction_map = {target.id: (13, 13)}

        self.assertFalse(logic._opening_mission_allowed(target, plan, "expand"))

    def test_shun_opening_rejects_early_hostile_attacks(self):
        logic = self.make_logic()
        target = make_planet(12, 1, 10, 4, 90.0, 90.0)
        plan = agent_mod.CapturePlan(
            target=target,
            ships=12,
            angle=0.0,
            eta=7,
            eval_turn=7,
            required_ships=11,
        )
        logic.state = SimpleNamespace(
            is_opening=True,
            is_early=True,
            player=0,
            step=6,
            num_players=4,
        )
        logic.reaction_map = {target.id: (7, 12)}

        self.assertFalse(logic._opening_mission_allowed(target, plan, "attack"))

    def test_duel_opening_rejects_early_hostile_attacks_while_good_neutrals_remain(self):
        logic = self.make_logic()
        target = make_planet(12, 1, 10, 4, 90.0, 90.0)
        neutrals = [
            make_planet(20, -1, 7, 4, 80.0, 80.0),
            make_planet(21, -1, 6, 3, 78.0, 74.0),
        ]
        plan = agent_mod.CapturePlan(
            target=target,
            ships=12,
            angle=0.0,
            eta=7,
            eval_turn=7,
            required_ships=11,
        )
        logic.state = SimpleNamespace(
            is_opening=True,
            is_early=True,
            player=0,
            step=10,
            num_players=2,
            neutral_planets=neutrals,
        )
        logic.modes = {"is_ahead": False}
        logic.reaction_map = {
            target.id: (7, 12),
            20: (6, 11),
            21: (7, 13),
        }

        self.assertFalse(logic._duel_opening_pivot_ready())
        self.assertFalse(logic._opening_mission_allowed(target, plan, "attack"))

    def test_duel_opening_allows_hostile_pivot_after_growth_window(self):
        logic = self.make_logic()
        target = make_planet(13, 1, 9, 4, 85.0, 85.0)
        neutrals = [make_planet(22, -1, 6, 3, 76.0, 76.0)]
        plan = agent_mod.CapturePlan(
            target=target,
            ships=12,
            angle=0.0,
            eta=8,
            eval_turn=8,
            required_ships=11,
        )
        logic.state = SimpleNamespace(
            is_opening=True,
            is_early=True,
            player=0,
            step=26,
            num_players=2,
            neutral_planets=neutrals,
        )
        logic.modes = {"is_ahead": False}
        logic.reaction_map = {
            target.id: (8, 14),
            22: (7, 12),
        }

        self.assertTrue(logic._duel_opening_pivot_ready())
        self.assertTrue(logic._opening_mission_allowed(target, plan, "attack"))

    def test_build_modes_uses_duel_logic_in_two_player(self):
        logic = self.make_logic()
        logic.state = SimpleNamespace(
            planets=[
                make_planet(1, 0, 60, 5, 90.0, 90.0),
                make_planet(2, 1, 20, 5, 10.0, 10.0),
            ],
            fleets=[agent_mod.Fleet(1, 0, 0.0, 0.0, 0.0, 1, 20)],
            player=0,
            num_players=2,
            step=80,
        )

        modes = logic._build_modes()

        self.assertTrue(modes["is_ahead"])
        self.assertTrue(modes["is_dominating"])
        self.assertFalse(modes["is_behind"])

    def test_build_modes_uses_rank_logic_in_four_player(self):
        logic = self.make_logic()
        logic.state = SimpleNamespace(
            planets=[
                make_planet(1, 0, 30, 3, 90.0, 90.0),
                make_planet(2, 1, 50, 3, 10.0, 10.0),
                make_planet(3, 2, 45, 3, 90.0, 10.0),
                make_planet(4, 3, 20, 3, 10.0, 90.0),
            ],
            fleets=[],
            player=0,
            num_players=4,
            step=80,
        )

        modes = logic._build_modes()

        self.assertTrue(modes["is_behind"])
        self.assertFalse(modes["is_ahead"])
        self.assertEqual(modes["my_rank"], 3)

    def test_build_modes_detects_cleanup_when_far_ahead_late(self):
        logic = self.make_logic()
        logic.state = SimpleNamespace(
            planets=[
                make_planet(1, 0, 400, 8, 90.0, 90.0),
                make_planet(2, 0, 350, 7, 80.0, 80.0),
                make_planet(3, 0, 300, 6, 70.0, 70.0),
                make_planet(4, 1, 90, 1, 15.0, 15.0),
                make_planet(5, 1, 80, 1, 20.0, 12.0),
                make_planet(6, 1, 70, 1, 12.0, 20.0),
            ],
            fleets=[],
            player=0,
            num_players=2,
            step=220,
        )

        modes = logic._build_modes()

        self.assertTrue(modes["is_cleanup"])
        self.assertEqual(modes["enemy_planet_count"], 3)

    def test_cleanup_target_value_boosts_last_enemy_planets(self):
        logic = self.make_logic()
        target = make_planet(30, 1, 120, 1, 20.0, 20.0)
        logic.state = SimpleNamespace(
            remaining_steps=80,
            is_late=False,
            is_opening=False,
            is_early=False,
            num_players=2,
            player=0,
            enemy_planets=[target],
            neutral_planets=[],
            comet_planet_ids=set(),
        )
        logic.indirect_wealth_map = defaultdict(float)
        logic.modes = {
            "is_finishing": False,
            "is_cleanup": True,
            "is_behind": False,
            "is_dominating": False,
            "owner_planet_counts": {1: 1},
        }

        boosted = logic._target_value(target, eta=6, mission="attack")

        self.assertGreater(boosted, 400.0)

    def test_reinforcement_needed_returns_zero_when_planet_already_holds(self):
        home = make_planet(1, 0, 40, 4, 20.0, 20.0)
        state = SimpleNamespace(
            planets=[home],
            planets_by_id={home.id: home},
            fleets=[],
            player=0,
            max_speed=agent_mod.DEFAULT_MAX_SHIP_SPEED,
        )
        world = agent_mod.ProjectedWorld(state, predictor=SimpleNamespace(), deadline=None)

        self.assertEqual(
            world.reinforcement_needed_to_hold_until(home.id, arrival_turn=5, hold_until=18),
            0,
        )

    def test_build_reinforce_missions_skips_zero_need_transfers(self):
        logic = self.make_logic()
        target = make_planet(10, 0, 100, 5, 20.0, 20.0)
        donor = make_planet(11, 0, 80, 4, 24.0, 20.0)
        enemy = make_planet(12, 1, 100, 5, 38.0, 20.0)
        logic.state = SimpleNamespace(
            my_planets=[target, donor],
            enemy_planets=[enemy],
            planets_by_id={10: target, 11: donor, 12: enemy},
            remaining_steps=100,
        )
        logic.world = SimpleNamespace(
            hold_status=lambda _pid, _horizon: {"fall_turn": None},
        )
        logic._available_my_planets = lambda excluded_ids=None: [p for p in [target, donor] if p.id not in set(excluded_ids or [])]
        logic._planet_surplus = lambda _planet: 20
        logic._settle_plan = lambda *args, **kwargs: agent_mod.CapturePlan(
            target=target,
            ships=7,
            angle=0.0,
            eta=3,
            eval_turn=6,
            required_ships=0,
        )

        missions = []
        logic._build_reinforce_missions(missions)

        self.assertEqual(missions, [])

    def test_frontier_keep_reserves_extra_garrison_near_enemy(self):
        logic = self.make_logic()
        planet = make_planet(10, 0, 60, 3, 40.0, 40.0)
        enemy = make_planet(11, 1, 30, 4, 58.0, 40.0)
        logic.state = SimpleNamespace(enemy_planets=[enemy])

        self.assertEqual(logic._frontier_keep(planet), 18)

    def test_hostile_reinforcement_margin_accounts_for_nearby_enemy_support(self):
        logic = self.make_logic()
        target = make_planet(10, 1, 20, 4, 70.0, 70.0)
        support = make_planet(11, 1, 21, 3, 78.0, 70.0)
        logic.state = SimpleNamespace(player=0, enemy_planets=[target, support])
        logic._plan_shot = lambda source, _target, _ships: (0.0, 6 if source.id == support.id else 3)

        self.assertEqual(logic._hostile_reinforcement_margin(target, arrival_turn=2), 5)

    def test_build_enemy_priority_prefers_closer_frontier_opponent(self):
        logic = self.make_logic()
        my_frontier = make_planet(1, 0, 40, 3, 50.0, 50.0)
        my_rear = make_planet(2, 0, 40, 3, 85.0, 85.0)
        enemy_close = make_planet(10, 1, 25, 3, 60.0, 50.0)
        enemy_mid = make_planet(11, 2, 35, 3, 74.0, 50.0)
        enemy_far = make_planet(12, 3, 45, 3, 95.0, 50.0)
        logic.state = SimpleNamespace(
            my_planets=[my_frontier, my_rear],
            enemy_planets=[enemy_close, enemy_mid, enemy_far],
            enemy_fleets=[],
        )
        logic.modes = {"owner_strength": {1: 25, 2: 35, 3: 45}}

        priority = logic._build_enemy_priority()

        self.assertGreater(priority[1], priority[2])
        self.assertGreater(priority[2], priority[3])

    def test_commit_missions_uses_lookahead_to_unlock_better_follow_up(self):
        logic = self.make_logic()
        logic.state = SimpleNamespace(
            planets_by_id={
                1: make_planet(1, 0, 20, 3, 90.0, 90.0),
                2: make_planet(2, 0, 20, 3, 10.0, 10.0),
            },
            player=0,
            step=30,
        )

        missions = [
            agent_mod.MissionOption(
                score=10.0,
                source_ids=[1],
                target_id=100,
                angles=[0.0],
                etas=[4],
                ships=[5],
                needed=5,
                mission="expand",
            ),
            agent_mod.MissionOption(
                score=8.0,
                source_ids=[2],
                target_id=100,
                angles=[0.0],
                etas=[4],
                ships=[5],
                needed=5,
                mission="expand",
            ),
            agent_mod.MissionOption(
                score=7.0,
                source_ids=[1],
                target_id=101,
                angles=[0.0],
                etas=[4],
                ships=[5],
                needed=5,
                mission="expand",
            ),
        ]

        moves = logic._commit_missions(missions)

        self.assertEqual([(move.source_id, move.target_id) for move in moves], [(2, 100), (1, 101)])

    def test_validate_intercept_accepts_safe_static_hit(self):
        source = make_planet(1, 0, 50, 3, 20.0, 20.0)
        target = make_planet(2, 1, 20, 3, 35.0, 20.0)
        predictor = agent_mod.PositionPredictor(
            current_step=0,
            angular_velocity=0.0,
            initial_planets=[source, target],
            current_planets=[source, target],
        )
        solver = agent_mod.InterceptSolver(predictor)

        shot = solver.solve_intercept(
            source.x,
            source.y,
            20,
            target,
            current_step=0,
            angular_velocity=0.0,
            initial_planets=[source, target],
            from_radius=source.radius,
        )

        self.assertIsNotNone(shot)
        angle, eta = shot
        hit_turn = solver._validate_intercept(
            source.x,
            source.y,
            source.radius,
            angle,
            20,
            target,
            eta + 1,
        )
        self.assertIsNotNone(hit_turn)
        self.assertLessEqual(abs(hit_turn - eta), 1)

    def test_validate_intercept_rejects_sun_crossing_path(self):
        source = make_planet(1, 0, 50, 3, 20.0, 50.0)
        target = make_planet(2, 1, 20, 3, 80.0, 50.0)
        predictor = agent_mod.PositionPredictor(
            current_step=0,
            angular_velocity=0.0,
            initial_planets=[source, target],
            current_planets=[source, target],
        )
        solver = agent_mod.InterceptSolver(predictor)

        hit_turn = solver._validate_intercept(
            source.x,
            source.y,
            source.radius,
            0.0,
            20,
            target,
            20,
        )

        self.assertIsNone(hit_turn)

    def test_validate_intercept_rejects_non_target_planet_collision(self):
        source = make_planet(1, 0, 50, 1, 52.9, 60.0)
        target = make_planet(2, 1, 20, 1, 95.0, 5.0)
        blocker = make_planet(3, -1, 8, 2, 58.0, 60.0)
        predictor = agent_mod.PositionPredictor(
            current_step=0,
            angular_velocity=0.0,
            initial_planets=[source, target, blocker],
            current_planets=[source, target, blocker],
        )
        solver = agent_mod.InterceptSolver(predictor)

        hit_turn = solver._validate_intercept(
            source.x,
            source.y,
            source.radius,
            0.0,
            1000,
            target,
            4,
        )

        self.assertIsNone(hit_turn)

    def test_validate_intercept_rejects_non_target_planet_sweep(self):
        source = make_planet(1, 0, 50, 1, 52.9, 60.0)
        target = make_planet(2, 1, 20, 1, 95.0, 5.0)
        sweeper = make_planet(3, -1, 8, 3, 70.0, 50.0)
        predictor = agent_mod.PositionPredictor(
            current_step=0,
            angular_velocity=math.pi / 2,
            initial_planets=[source, target, sweeper],
            current_planets=[source, target, sweeper],
        )
        solver = agent_mod.InterceptSolver(predictor)

        hit_turn = solver._validate_intercept(
            source.x,
            source.y,
            source.radius,
            0.0,
            1000,
            target,
            4,
        )

        self.assertIsNone(hit_turn)

    def test_solve_intercept_drops_unverified_last_guess(self):
        target = make_planet(2, 1, 20, 3, 40.0, 40.0)
        predictor = SimpleNamespace(
            current_step=0,
            predict_target_pos=lambda _planet, _step: (40.0, 40.0),
            target_can_move=lambda _planet: True,
        )
        solver = agent_mod.InterceptSolver(predictor)
        solver._estimate_to_position = lambda *args, **kwargs: (0.0, 3)
        solver._validate_intercept = lambda *args, **kwargs: None
        solver._search_safe_intercept = lambda *args, **kwargs: None

        shot = solver.solve_intercept(
            10.0,
            10.0,
            20,
            target,
            current_step=0,
            angular_velocity=0.0,
            initial_planets=[],
            from_radius=1.0,
        )

        self.assertIsNone(shot)


if __name__ == "__main__":
    unittest.main()
