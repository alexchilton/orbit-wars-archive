"""
tests/test_physics.py — Unit tests for physics_sim.py

Tests the forward physics engine in isolation: fleet speed, travel time,
orbit prediction, combat resolution, production, and fleet destination
inference. No kaggle env. No agent. Pure synthetic states.

Run:
    cd /Users/alexchilton/DataspellProjects/orbit_wars
    python -m pytest tests/test_physics.py -v
"""
import math
import sys
import pytest

sys.path.insert(0, '/Users/alexchilton/DataspellProjects/orbit_wars')
sys.path.insert(0, '/Users/alexchilton/DataspellProjects/orbit_wars/submission')

from physics_sim import (
    SUN_X, SUN_Y,
    fleet_speed,
    travel_time,
    predict_orbit,
    infer_fleet_dest,
    step_state,
    predict,
    apply_moves,
    copy_state,
)


# ── State factory helpers ──────────────────────────────────────────────────────

def make_planet(pid, x, y, owner=-1, ships=10, prod=1, radius=5, is_orb=False):
    r = math.hypot(x - SUN_X, y - SUN_Y)
    return (pid, {
        'x': x, 'y': y, 'owner': owner, 'ships': float(ships),
        'prod': float(prod), 'radius': float(radius),
        'is_orb': is_orb, 'is_comet': False,
        'r': r, 'theta': math.atan2(y - SUN_Y, x - SUN_X),
    })


def make_fleet(owner, ships, target_pid, eta):
    return {'owner': owner, 'ships': float(ships), 'target_pid': target_pid, 'eta': float(eta)}


def make_state(planets, fleets=None, me=0, step=0, omega=0.03):
    return {
        'planets': dict(planets),
        'fleets': list(fleets or []),
        'step': step, 'omega': omega, 'me': me,
    }


# ── fleet_speed ────────────────────────────────────────────────────────────────

class TestFleetSpeed:

    def test_zero_or_one_ship_gives_speed_one(self):
        assert abs(fleet_speed(0) - 1.0) < 1e-9
        assert abs(fleet_speed(1) - 1.0) < 1e-9

    def test_speed_increases_monotonically_with_ships(self):
        speeds = [fleet_speed(n) for n in [1, 10, 100, 500, 999]]
        assert speeds == sorted(speeds)

    def test_speed_never_reaches_max(self):
        assert fleet_speed(999) < 6.0

    def test_speed_positive_for_all_reasonable_inputs(self):
        for n in [0, 1, 2, 50, 1000]:
            assert fleet_speed(n) > 0


# ── travel_time ────────────────────────────────────────────────────────────────

class TestTravelTime:

    def test_ten_units_one_ship_is_ten_steps(self):
        t = travel_time(0, 0, 10, 0, ships=1)
        assert abs(t - 10.0) < 1e-6

    def test_farther_distance_takes_longer(self):
        assert travel_time(0, 0, 100, 0, 1) > travel_time(0, 0, 10, 0, 1)

    def test_larger_fleet_is_faster(self):
        assert travel_time(0, 0, 100, 0, 500) < travel_time(0, 0, 100, 0, 1)

    def test_zero_ships_returns_large_sentinel(self):
        assert travel_time(0, 0, 50, 0, 0) >= 999.0

    def test_distance_zero_returns_zero(self):
        assert travel_time(5, 5, 5, 5, 10) == 0.0


# ── predict_orbit ──────────────────────────────────────────────────────────────

class TestPredictOrbit:

    def test_dt_zero_returns_same_position(self):
        x0, y0 = SUN_X + 30, SUN_Y
        nx, ny = predict_orbit(x0, y0, omega=0.03, dt=0)
        assert abs(nx - x0) < 1e-9
        assert abs(ny - y0) < 1e-9

    def test_orbital_radius_is_preserved(self):
        x0, y0 = SUN_X + 30, SUN_Y
        r_before = math.hypot(x0 - SUN_X, y0 - SUN_Y)
        for dt in [1, 10, 100]:
            nx, ny = predict_orbit(x0, y0, omega=0.03, dt=dt)
            r_after = math.hypot(nx - SUN_X, ny - SUN_Y)
            assert abs(r_before - r_after) < 1e-6, f"Radius changed at dt={dt}"

    def test_full_orbit_returns_to_start(self):
        x0, y0 = SUN_X + 30, SUN_Y
        omega = 0.03
        # Use exact float steps to avoid integer truncation error
        full = 2 * math.pi / omega
        nx, ny = predict_orbit(x0, y0, omega, full)
        assert math.hypot(nx - x0, ny - y0) < 1e-6


# ── step_state: combat ─────────────────────────────────────────────────────────

class TestCombat:

    def test_attacker_captures_neutral(self):
        """10 attacking vs 5 neutral garrison → attacker owns with 5 ships."""
        planets = [make_planet(0, 30, 50, owner=-1, ships=5, prod=0)]
        state = make_state(planets, [make_fleet(owner=0, ships=10, target_pid=0, eta=1.0)])
        step_state(state)
        p = state['planets'][0]
        assert p['owner'] == 0
        assert abs(p['ships'] - 5.0) < 1e-6

    def test_defender_holds_with_superior_garrison(self):
        """10 attacking vs 20 defending → defender holds with 10 remaining."""
        planets = [make_planet(0, 30, 50, owner=0, ships=20, prod=0)]
        state = make_state(planets, [make_fleet(owner=1, ships=10, target_pid=0, eta=1.0)])
        step_state(state)
        p = state['planets'][0]
        assert p['owner'] == 0
        assert abs(p['ships'] - 10.0) < 1e-6

    def test_reinforcement_adds_to_garrison(self):
        """Same-owner fleet arriving → garrison grows, ownership unchanged."""
        planets = [make_planet(0, 30, 50, owner=0, ships=5, prod=0)]
        state = make_state(planets, [make_fleet(owner=0, ships=8, target_pid=0, eta=1.0)])
        step_state(state)
        p = state['planets'][0]
        assert p['owner'] == 0
        assert abs(p['ships'] - 13.0) < 1e-6

    def test_fleet_does_not_arrive_early(self):
        """Fleet with ETA=3 must not affect planet before step 3."""
        planets = [make_planet(0, 30, 50, owner=-1, ships=5, prod=0)]
        state = make_state(planets, [make_fleet(owner=0, ships=20, target_pid=0, eta=3.0)])
        step_state(state)  # step 1
        assert state['planets'][0]['owner'] == -1, "Should not arrive at step 1"
        step_state(state)  # step 2
        assert state['planets'][0]['owner'] == -1, "Should not arrive at step 2"
        step_state(state)  # step 3
        assert state['planets'][0]['owner'] == 0, "Should arrive at step 3"


# ── step_state: production ─────────────────────────────────────────────────────

class TestProduction:

    def test_owned_planet_produces_each_step(self):
        planets = [make_planet(0, 30, 50, owner=0, ships=10, prod=3)]
        state = make_state(planets)
        step_state(state)
        assert abs(state['planets'][0]['ships'] - 13.0) < 1e-6

    def test_neutral_planet_does_not_produce(self):
        planets = [make_planet(0, 30, 50, owner=-1, ships=10, prod=3)]
        state = make_state(planets)
        step_state(state)
        assert abs(state['planets'][0]['ships'] - 10.0) < 1e-6

    def test_production_compounds_over_steps(self):
        planets = [make_planet(0, 30, 50, owner=0, ships=0, prod=5)]
        state = make_state(planets)
        for _ in range(4):
            step_state(state)
        assert abs(state['planets'][0]['ships'] - 20.0) < 1e-6


# ── predict ────────────────────────────────────────────────────────────────────

class TestPredict:

    def test_predict_matches_manual_steps(self):
        """predict(state, N) must equal N calls to step_state."""
        planets = [
            make_planet(0, 30, 50, owner=0, ships=10, prod=2),
            make_planet(1, 70, 50, owner=1, ships=6,  prod=1),
        ]
        state = make_state(planets)
        manual = copy_state(state)
        for _ in range(5):
            step_state(manual)
        predicted = predict(state, 5)
        assert abs(predicted['planets'][0]['ships'] - manual['planets'][0]['ships']) < 1e-6
        assert abs(predicted['planets'][1]['ships'] - manual['planets'][1]['ships']) < 1e-6

    def test_predict_does_not_mutate_original(self):
        planets = [make_planet(0, 30, 50, owner=0, ships=10, prod=2)]
        state = make_state(planets)
        ships_before = state['planets'][0]['ships']
        predict(state, 10)
        assert abs(state['planets'][0]['ships'] - ships_before) < 1e-6

    def test_predict_step_counter_advances(self):
        planets = [make_planet(0, 30, 50, owner=0, ships=10, prod=1)]
        state = make_state(planets, step=5)
        result = predict(state, 7)
        assert result['step'] == 12


# ── apply_moves ────────────────────────────────────────────────────────────────

class TestApplyMoves:

    def test_deducts_ships_from_source(self):
        planets = [
            make_planet(0, 30, 50, owner=0, ships=20, prod=0),
            make_planet(1, 70, 50, owner=-1, ships=5, prod=0),
        ]
        state = make_state(planets)
        new = apply_moves(state, [(0, 1, 10)])
        assert abs(new['planets'][0]['ships'] - 10.0) < 1e-6

    def test_creates_fleet_with_correct_target(self):
        planets = [
            make_planet(0, 30, 50, owner=0, ships=20, prod=0),
            make_planet(1, 70, 50, owner=-1, ships=5, prod=0),
        ]
        state = make_state(planets)
        new = apply_moves(state, [(0, 1, 10)])
        assert len(new['fleets']) == 1
        assert new['fleets'][0]['target_pid'] == 1
        assert abs(new['fleets'][0]['ships'] - 10.0) < 1e-6

    def test_leaves_at_least_one_ship_on_source(self):
        planets = [
            make_planet(0, 30, 50, owner=0, ships=5, prod=0),
            make_planet(1, 70, 50, owner=-1, ships=5, prod=0),
        ]
        state = make_state(planets)
        new = apply_moves(state, [(0, 1, 5)])  # try to send all
        assert new['planets'][0]['ships'] >= 1.0


# ── infer_fleet_dest ───────────────────────────────────────────────────────────

class TestInferFleetDest:

    def test_fleet_heading_directly_at_static_planet(self):
        """Fleet at (50,50) heading east (angle=0) → should infer planet at (100,50)."""
        planets = dict([
            make_planet(0,   0, 50),   # behind fleet — wrong direction
            make_planet(1, 100, 50),   # directly ahead
        ])
        dest = infer_fleet_dest(50, 50, fangle=0.0, from_pid=0, planets=planets, ships=10)
        assert dest == 1

    def test_fleet_heading_north_infers_northern_planet(self):
        """Fleet at (50,30) heading north (angle=π/2) → planet at (50,80)."""
        planets = dict([
            make_planet(0, 50, 10),   # south — wrong direction
            make_planet(1, 50, 80),   # north — correct
        ])
        dest = infer_fleet_dest(50, 30, fangle=math.pi / 2, from_pid=0, planets=planets, ships=10)
        assert dest == 1
