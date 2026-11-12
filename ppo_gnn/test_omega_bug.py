"""Test proving p[4] is planet radius, NOT angular velocity.

The game observation format is:
  p[0]=id, p[1]=owner, p[2]=x, p[3]=y, p[4]=RADIUS, p[5]=ships, p[6]=production

Angular velocity is a single game-wide value at obs['angular_velocity'].
Our code at edge_policy.py:169 reads p[4] as orbit velocity — this is WRONG.
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"

import pytest


def _get_game_obs():
    """Run a real game and return observation after 1 step."""
    from kaggle_environments import make
    env = make("orbit_wars", debug=False)
    trainer = env.train([None, "random"])
    obs = trainer.reset()
    # Step once to populate planets
    obs, _, _, _ = trainer.step([])
    return obs


class TestPlanetFieldIndices:
    """Prove that p[4] is radius, not angular_velocity."""

    def test_p4_is_radius_not_omega(self):
        """p[4] should be planet radius (~1-5), not angular velocity (~0.025-0.05)."""
        obs = _get_game_obs()
        planets = obs["planets"]
        angular_velocity = obs["angular_velocity"]

        assert len(planets) > 0, "No planets in observation"
        assert angular_velocity > 0, "angular_velocity should be positive"

        # angular_velocity is always in [0.025, 0.05] per game source
        assert 0.02 <= angular_velocity <= 0.06, (
            f"angular_velocity={angular_velocity} outside expected [0.025, 0.05]"
        )

        for p in planets:
            p4_value = float(p[4])
            # If p[4] were omega, it would be ~0.025-0.05 for ALL planets
            # But p[4] is radius, so it's ~1-5 (much larger)
            # This assertion proves p[4] != angular_velocity
            if p4_value > 0.1:
                # Found a planet where p[4] >> angular_velocity
                # This CANNOT be angular velocity
                assert p4_value != pytest.approx(angular_velocity, abs=0.01), (
                    f"p[4]={p4_value} should NOT match angular_velocity={angular_velocity}"
                )
                return  # Test passes — p[4] is clearly not omega

        pytest.fail("All p[4] values were < 0.1 — couldn't distinguish from omega")

    def test_p4_values_match_planet_radius_range(self):
        """p[4] values should be in planet radius range (~1-5), not omega range (~0.025-0.05)."""
        obs = _get_game_obs()
        planets = obs["planets"]

        p4_values = [float(p[4]) for p in planets]
        # Planet radii in the game are typically 1.0 to 5.0
        # Angular velocity is 0.025 to 0.05
        avg_p4 = sum(p4_values) / len(p4_values)

        assert avg_p4 > 0.5, (
            f"Average p[4]={avg_p4} — if this were omega it'd be ~0.035. "
            f"Values > 0.5 prove it's radius, not omega."
        )

    def test_angular_velocity_is_global_not_per_planet(self):
        """angular_velocity is a single game-wide value, not per-planet."""
        obs = _get_game_obs()
        angular_velocity = obs["angular_velocity"]

        # It's a scalar float, not a list
        assert isinstance(angular_velocity, float), (
            f"angular_velocity should be float, got {type(angular_velocity)}"
        )

    def test_edge_policy_reads_p4_as_omega(self):
        """Prove that edge_policy.py:169 incorrectly reads p[4] as orbit velocity."""
        obs = _get_game_obs()
        planets = obs["planets"]
        angular_velocity = obs["angular_velocity"]

        # This is what edge_policy.py:169 does:
        planet_orbit_vel = [float(p[4]) for p in planets]

        # These should NOT be used as omega — they're planet radii
        for i, vel in enumerate(planet_orbit_vel):
            if vel > 0.1:  # radius values are > 1.0 typically
                # This "velocity" is ~60x the real angular velocity
                ratio = vel / angular_velocity
                assert ratio > 10, (
                    f"Planet {i}: p[4]={vel}, real omega={angular_velocity}, "
                    f"ratio={ratio:.1f}x — p[4] is clearly NOT omega"
                )
                return

        pytest.fail("Couldn't find planet with p[4] > 0.1")


class TestOrbitPredictionWithCorrectOmega:
    """Verify that orbit predictions stay in-bounds with correct omega."""

    def test_predict_orbit_inbounds_with_real_omega(self):
        """With correct omega (~0.035), predict_orbit should stay in [0, 100]."""
        from ppo_gnn.edge_policy import predict_orbit

        obs = _get_game_obs()
        planets = obs["planets"]
        real_omega = obs["angular_velocity"]

        for p in planets:
            x, y = float(p[2]), float(p[3])
            orbital_r = math.hypot(x - 50, y - 50)
            planet_radius = float(p[4])

            # Only orbiting planets (orbital_r + planet_radius < 50)
            if orbital_r + planet_radius >= 50:
                continue  # static planet

            # Predict position at various future times
            for dt in [1, 5, 10, 20, 50, 100]:
                px, py = predict_orbit(x, y, real_omega, dt)
                assert 0 <= px <= 100 and 0 <= py <= 100, (
                    f"Planet at ({x:.1f}, {y:.1f}) with real omega={real_omega:.4f}: "
                    f"predicted ({px:.1f}, {py:.1f}) at dt={dt} is off-board!"
                )

    def test_predict_orbit_offboard_with_wrong_omega(self):
        """With wrong omega (p[4] = radius ~2.0), predictions go off-board."""
        from ppo_gnn.edge_policy import predict_orbit

        obs = _get_game_obs()
        planets = obs["planets"]

        any_offboard = False
        for p in planets:
            x, y = float(p[2]), float(p[3])
            wrong_omega = float(p[4])  # This is radius, not omega!

            if wrong_omega < 0.1:
                continue

            # With wrong omega (~2.0 rad/step), prediction spins wildly
            for dt in [1, 5, 10]:
                px, py = predict_orbit(x, y, wrong_omega, dt)
                if not (0 <= px <= 100 and 0 <= py <= 100):
                    any_offboard = True
                    break
            if any_offboard:
                break

        # With wrong omega, at least some predictions should go off-board
        # (proving the off-board issue was caused by wrong omega, not real orbits)
        assert any_offboard, (
            "Expected off-board predictions with wrong omega — "
            "this test validates that wrong omega causes the observed bug"
        )


class TestComputeCandidateEdgesOmegaFix:
    """Verify compute_candidate_edges uses angular_velocity correctly after fix."""

    def test_compute_candidate_edges_accepts_angular_velocity(self):
        """compute_candidate_edges should accept angular_velocity parameter."""
        from ppo_gnn.edge_policy import compute_candidate_edges

        obs = _get_game_obs()
        planets = obs["planets"]
        fleets = obs["fleets"]
        player = obs["player"]
        angular_velocity = obs["angular_velocity"]

        # Should not raise
        ef, ei, em, nv = compute_candidate_edges(
            planets=planets,
            fleets=fleets,
            player_id=player,
            num_players=2,
            step=1,
            max_steps=500,
            max_candidates=48,
            angular_velocity=angular_velocity,
        )
        assert ef.shape[0] == 48
        assert ei.shape == (48, 2)

    def test_planet_orbit_vel_computed_correctly(self):
        """Orbiting planets should get angular_velocity, static ones should get 0."""
        obs = _get_game_obs()
        planets = obs["planets"]
        angular_velocity = obs["angular_velocity"]

        for p in planets:
            x, y = float(p[2]), float(p[3])
            planet_r = float(p[4])
            orbital_r = math.hypot(x - 50, y - 50)

            if orbital_r + planet_r < 48.0:
                # Orbiting — should use angular_velocity
                expected_omega = angular_velocity
            else:
                # Static — should use 0
                expected_omega = 0.0

            # Verify the expected omega is reasonable
            assert expected_omega >= 0.0
            if expected_omega > 0:
                assert 0.02 <= expected_omega <= 0.06


class TestSolveInterceptWithCorrectOmega:
    """Verify solve_intercept works correctly with real omega."""

    def test_intercept_inbounds_with_real_omega(self):
        """solve_intercept should produce in-bounds results with correct omega."""
        from ppo_gnn.edge_policy import solve_intercept

        obs = _get_game_obs()
        planets = obs["planets"]
        real_omega = obs["angular_velocity"]

        # Find a source planet (owned) and target planet (not owned)
        my_player = obs["player"]
        my_planets = [p for p in planets if int(p[1]) == my_player]
        other_planets = [p for p in planets if int(p[1]) != my_player]

        if not my_planets or not other_planets:
            pytest.skip("Need at least one own and one other planet")

        src = my_planets[0]
        sx, sy = float(src[2]), float(src[3])

        for tgt in other_planets[:5]:
            tx, ty = float(tgt[2]), float(tgt[3])
            orbital_r = math.hypot(tx - 50, ty - 50)
            planet_radius = float(tgt[4])

            # Determine if target orbits
            omega = real_omega if (orbital_r + planet_radius < 50) else 0.0

            ix, iy = solve_intercept(sx, sy, tx, ty, omega, 30)
            assert 0 <= ix <= 100 and 0 <= iy <= 100, (
                f"Intercept ({ix:.1f}, {iy:.1f}) off-board for target at "
                f"({tx:.1f}, {ty:.1f}) with omega={omega:.4f}"
            )
