import unittest
from types import SimpleNamespace

import planetwars_orbit_agent as mod


class PlanetWarsOrbitAgentTests(unittest.TestCase):
    def test_build_send_buckets_scales_and_keeps_full_surplus(self):
        self.assertEqual(mod.build_send_buckets(0), [])
        self.assertEqual(mod.build_send_buckets(5), [2, 4, 5])
        self.assertEqual(mod.build_send_buckets(18), [7, 13, 18])

    def test_choose_capture_step_combines_sources_when_needed(self):
        logic = mod.DecisionLogic.__new__(mod.DecisionLogic)
        logic.expired = lambda: False
        logic.state = SimpleNamespace(player=0, remaining_steps=100)
        logic._score_capture_step = lambda target, mission, eval_turn, launches: 100.0 - sum(l.eta for l in launches)
        target = SimpleNamespace(id=9, production=4, ships=12, owner=-1)
        options = [
            mod.LaunchOption(1, 9, 8, 0.0, 4, "expand"),
            mod.LaunchOption(2, 9, 7, 0.0, 5, "expand"),
            mod.LaunchOption(3, 9, 6, 0.0, 8, "expand"),
        ]
        logic._candidate_launches = lambda _target, _mission: list(options)
        logic._capture_progress = lambda _target_id, _eval_turn, launches: (
            sum(launch.ships for launch in launches) >= 15,
            sum(launch.ships for launch in launches),
        )

        step = logic._choose_capture_step(target, "expand")

        self.assertIsNotNone(step)
        self.assertEqual([launch.source_id for launch in step.launches], [1, 2])
        self.assertEqual(step.ships_sent, 15)

    def test_choose_move_set_avoids_source_conflicts(self):
        logic = mod.DecisionLogic.__new__(mod.DecisionLogic)
        logic.expired = lambda: False
        logic._evaluate_step_set = lambda steps: sum(step.score for step in steps)
        steps = [
            mod.StepPlan("expand", 10, [mod.LaunchOption(1, 10, 8, 0.0, 4, "expand")], 4, 8, 5.0, True),
            mod.StepPlan("expand", 11, [mod.LaunchOption(1, 11, 7, 0.0, 5, "expand")], 5, 7, 4.0, True),
            mod.StepPlan("attack", 12, [mod.LaunchOption(2, 12, 9, 0.0, 6, "attack")], 6, 9, 3.0, True),
        ]

        chosen = logic._choose_move_set(steps)

        self.assertEqual([step.target_id for step in chosen], [10, 12])


if __name__ == "__main__":
    unittest.main()
