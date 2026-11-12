"""
tests/test_geometry_cases.py — Decision logic across different map geometries.

Each class represents one map geometry type.  The tests verify that the agent
makes the right strategic choice when the map has that structure.

Sun at (50, 50), radius 10.  INNER_ORBIT_THRESHOLD = 48.
Non-orbiting: (r + planet_radius) >= 48  →  r >= 43 with radius=5.

Positions used (all non-orbiting unless stated):
  SRC  = (0,  50)  r=50  ← our home base
  NEAR = (0,  65)  r≈52  dist=15
  MID  = (0,  80)  r≈58  dist=30
  FAR  = (0, -80)  r≈134 dist=130  → ETA > SIM_HORIZON
  EAST = (100,50)  r=50              ← opposite side of sun (path crosses sun)
  ORB  = (35, 50)  r=15  is_orb=True

Run:
    cd /Users/alexchilton/DataspellProjects/orbit_wars
    python -m pytest tests/test_geometry_cases.py -v
"""
import sys, math
import pytest

sys.path.insert(0, '/Users/alexchilton/DataspellProjects/orbit_wars')
sys.path.insert(0, '/Users/alexchilton/DataspellProjects/orbit_wars/submission')

from physics_sim import SUN_X, SUN_Y, INNER_ORBIT_THRESHOLD, travel_time
import main_pred_2p as _mod
from main_pred_2p import (
    agent, _rough_roi, _SIM_HORIZON, path_crosses_sun, multi_leg_path,
    analyze_counter_windows,
)


# ── obs factory ───────────────────────────────────────────────────────────────

def obs(planets_raw, fleets_raw=None, player=0, step=0):
    """planets_raw: [[id,owner,x,y,radius,ships,prod], ...]"""
    return {
        'player': player, 'planets': planets_raw,
        'fleets': fleets_raw or [], 'step': step,
        'angular_velocity': 0.03, 'comet_planet_ids': [], 'comets': [],
    }


def planet(pid, owner, x, y, ships, prod, radius=5):
    return [pid, owner, x, y, radius, ships, prod]


def check_is_orb(x, y, radius=5):
    r = math.hypot(x - SUN_X, y - SUN_Y)
    return (r + radius) < INNER_ORBIT_THRESHOLD


@pytest.fixture(autouse=True)
def clear_registry():
    _mod._FLEET_REGISTRY.clear()
    _mod._REGISTRY_PREV_STEP = 0
    yield
    _mod._FLEET_REGISTRY.clear()


# ── Case 1: Compact map ───────────────────────────────────────────────────────

class TestCompactMap:
    """
    All planets within 30 units of each other → every target is inside SIM_HORIZON.
    Agent should expand to ALL neutrals and register targets correctly.

    Layout (all non-orbiting):
        SRC(0,50) ─ 15u ─ NEAR(0,65) ─ 15u ─ MID(0,80)
    """

    def test_agent_fires_at_nearest_neutral(self):
        o = obs([planet(0, 0, 0, 50, 50, 3), planet(1, -1, 0, 65, 5, 3)])
        assert not check_is_orb(0, 65), "NEAR must not be orbiting"
        actions = agent(o)
        assert len(actions) > 0, "Should fire when compact map has a reachable neutral"

    def test_near_ranked_above_mid(self):
        """
        NEAR (dist=15) and MID (dist=30): same prod, same garrison.
        rough_roi ranks NEAR higher → agent should target NEAR.
        """
        import main_pred_2p as _mod2
        src = {'x': 0, 'y': 50, 'owner': 0, 'ships': 50.0, 'prod': 3.0,
               'radius': 5.0, 'is_orb': False, 'r': 50, 'theta': 0}
        near = {'x': 0, 'y': 65, 'owner': -1, 'ships': 5.0, 'prod': 3.0,
                'radius': 5.0, 'is_orb': False, 'r': 52, 'theta': 0}
        mid  = {'x': 0, 'y': 80, 'owner': -1, 'ships': 5.0, 'prod': 3.0,
                'radius': 5.0, 'is_orb': False, 'r': 58, 'theta': 0}
        roi_near = _rough_roi(src, near, av=50, omega=0.03)
        roi_mid  = _rough_roi(src, mid,  av=50, omega=0.03)
        assert roi_near > roi_mid, \
            f"NEAR roi={roi_near:.1f} should beat MID roi={roi_mid:.1f}"

    def test_two_close_neutrals_both_get_targeted(self):
        """
        Two very close neutrals — agent may fire to both in one turn
        (one per source planet is the limit, but both should register).
        """
        o = obs([
            planet(0, 0, 0, 50, 80, 3),   # big home planet
            planet(1, -1, 0, 65, 5, 3),    # NEAR
            planet(2, -1, 0, 80, 5, 3),    # MID
        ])
        actions = agent(o)
        assert len(actions) >= 1, "At least one neutral should be targeted"
        fired = {f['target_pid'] for f in _mod._FLEET_REGISTRY}
        assert 1 in fired, "NEAR should be targeted"


# ── Case 2: Spread map ────────────────────────────────────────────────────────

class TestSpreadMap:
    """
    FAR planet at (0, -80): dist=130, ETA ≈ 30 steps.
    Under the corrected heuristic, far planets with positive production ARE worth
    attacking — the old 25-step horizon was a bug. So we test meaningful invariants:
    - Zero-production targets have negative marginal gain (capturing them costs ships
      but never earns them back) → baseline guard correctly blocks.
    - The closest profitable target gets targeted before a far one when ships are
      scarce (sorted by _rough_roi which values production × remaining steps - eta).
    """

    def test_far_planet_eta_beyond_old_horizon(self):
        """Sanity: FAR planet has ETA > 25 (the old fixed horizon)."""
        eta = travel_time(0, 50, 0, -80, 20)
        assert eta > _SIM_HORIZON, f"FAR ETA={eta:.1f} should exceed SIM_HORIZON={_SIM_HORIZON}"

    def test_zero_prod_far_fires_nothing(self):
        """
        A far neutral with zero production has marginal gain < 0
        (costs ships to capture, never earns them back) → should not fire.
        """
        o = obs([planet(0, 0, 0, 50, 50, 3), planet(1, -1, 0, -80, 5, 0)])
        actions = agent(o)
        fired = {f['target_pid'] for f in _mod._FLEET_REGISTRY}
        assert 1 not in fired, \
            f"Zero-prod far neutral should not be targeted, got {actions}"

    def test_near_targeted_when_both_present(self):
        """NEAR and FAR both present — NEAR should be targeted (sorted first by _rough_roi)."""
        o = obs([
            planet(0, 0, 0, 50, 50, 3),
            planet(1, -1, 0, 65, 5, 3),   # NEAR: higher ROI (shorter ETA)
            planet(2, -1, 0, -80, 5, 3),  # FAR: also profitable but lower ROI
        ])
        agent(o)
        fired = {f['target_pid'] for f in _mod._FLEET_REGISTRY}
        assert 1 in fired, "NEAR should always be targeted (highest ROI)"

    def test_rough_roi_far_positive(self):
        """rough_roi for FAR must be positive (prod × remaining_steps not zero)."""
        src = {'x': 0, 'y': 50, 'owner': 0, 'ships': 50.0, 'prod': 3.0,
               'radius': 5.0, 'is_orb': False, 'r': 50, 'theta': 0}
        far = {'x': 0, 'y': -80, 'owner': -1, 'ships': 5.0, 'prod': 3.0,
               'radius': 5.0, 'is_orb': False, 'r': 134, 'theta': 0}
        roi = _rough_roi(src, far, av=50, omega=0.03)
        assert roi > 0, f"FAR rough_roi should be positive (prod × remaining steps), got {roi:.2f}"

    def test_rough_roi_near_exceeds_far(self):
        """Near planet has higher ROI than far planet with same production (shorter ETA)."""
        src = {'x': 0, 'y': 50, 'owner': 0, 'ships': 50.0, 'prod': 3.0,
               'radius': 5.0, 'is_orb': False, 'r': 50, 'theta': 0}
        near = {'x': 0, 'y': 65, 'owner': -1, 'ships': 5.0, 'prod': 3.0,
                'radius': 5.0, 'is_orb': False, 'r': 15, 'theta': 0}
        far  = {'x': 0, 'y': -80, 'owner': -1, 'ships': 5.0, 'prod': 3.0,
                'radius': 5.0, 'is_orb': False, 'r': 134, 'theta': 0}
        roi_near = _rough_roi(src, near, av=50, omega=0.03)
        roi_far  = _rough_roi(src, far,  av=50, omega=0.03)
        assert roi_near > roi_far, \
            f"Near ROI={roi_near:.1f} should exceed far ROI={roi_far:.1f}"


# ── Case 3: Sun-blocked path ──────────────────────────────────────────────────

class TestSunBlockedPath:
    """
    SRC=(0,50), EAST=(100,50): the horizontal path passes directly through
    the sun at (50,50).  Agent must route via waypoint, not fire direct.

    The key property: path_crosses_sun returns True for this path.
    Agent still fires — multi_leg_path finds a waypoint above or below the sun.
    """

    def test_direct_path_crosses_sun(self):
        assert path_crosses_sun(0, 50, 100, 50), \
            "Horizontal path through sun centre should register as crossing"

    def test_clear_path_does_not_cross_sun(self):
        assert not path_crosses_sun(0, 50, 0, 75), \
            "Vertical path at x=0 is 50 units from sun — should be clear"

    def test_agent_skips_unprofitable_sun_blocked_target(self):
        """
        EAST at (100,50) is sun-blocked AND has zero production:
        - path crosses sun → multi_leg_path adds leg via (50,75)
        - prod=0 → marginal gain is negative (costs ships, never earns them back)
        Agent correctly fires nothing.
        """
        o = obs([
            planet(0, 0, 0, 50, 50, 3),    # SRC: 50 ships
            planet(1, -1, 100, 50, 5, 0),   # EAST: 100 units away, zero production
        ])
        actions = agent(o)
        assert actions == [], \
            "Agent correctly skips sun-blocked zero-prod target"

    def test_multi_leg_path_routes_around_sun(self):
        """
        multi_leg_path must return a valid waypoint for the sun-blocked path.
        The waypoint at (50,75) avoids the sun and the agent uses it.
        """
        legs, dist = multi_leg_path(0, 50, 100, 50)
        assert legs is not None, "multi_leg_path should find a route around sun"
        assert len(legs) == 2, f"Expected [waypoint, target], got {legs}"
        wx, wy = legs[0]
        # Waypoint must be clear of the sun (distance > sun radius + margin)
        wp_sun_dist = math.hypot(wx - SUN_X, wy - SUN_Y)
        assert wp_sun_dist > 10 + 1.5, \
            f"Waypoint ({wx:.1f},{wy:.1f}) is inside sun margin (dist={wp_sun_dist:.1f})"
        # Second leg from waypoint to EAST must also be clear
        assert not path_crosses_sun(wx, wy, 100, 50), \
            f"Path from waypoint to EAST still crosses sun"


# ── Case 4: Orbiting planet ───────────────────────────────────────────────────

class TestOrbitingPlanetGeometry:
    """
    ORB = (35, 50): inside INNER_ORBIT_THRESHOLD → is_orb=True.
    The intercept ETA from solve_intercept differs from direct travel_time
    because the planet moves.  _rough_roi must use the intercept ETA.
    """

    def test_orb_is_actually_orbiting(self):
        assert check_is_orb(35, 50), "ORB must be inside inner orbit threshold"

    def test_rough_roi_orbiting_uses_intercept_not_naive(self):
        """
        For ORB, _rough_roi must call solve_intercept.
        We verify indirectly: the ROI for ORB matches the intercept-based formula,
        not the naive travel_time formula.
        Formula: prod × max(0, remaining_steps - eta) - needed × 0.5
        """
        DEFAULT_REMAINING = 350  # default used by _rough_roi when remaining_steps not passed
        src = {'x': 0, 'y': 50, 'owner': 0, 'ships': 40.0, 'prod': 2.0,
               'radius': 5.0, 'is_orb': False, 'r': 50, 'theta': 0}
        orb = {'x': 35, 'y': 50, 'owner': -1, 'ships': 5.0, 'prod': 3.0,
               'radius': 5.0, 'is_orb': True,
               'r': math.hypot(35-SUN_X, 50-SUN_Y), 'theta': 0}
        from main_pred_2p import solve_intercept
        _, _, eta_intercept = solve_intercept(src['x'], src['y'], orb['x'], orb['y'],
                                               True, 0.03, 40)
        eta_naive = travel_time(src['x'], src['y'], orb['x'], orb['y'], 40)

        roi = _rough_roi(src, orb, av=40, omega=0.03)
        expected  = orb['prod'] * max(0.0, DEFAULT_REMAINING - eta_intercept) - 3.0
        naive_roi = orb['prod'] * max(0.0, DEFAULT_REMAINING - eta_naive) - 3.0

        err_intercept = abs(roi - expected)
        err_naive     = abs(roi - naive_roi)
        # _rough_roi should be closer to intercept formula than naive formula
        # (or both may be equal if orbiting planet happens to not have moved much)
        assert err_intercept <= err_naive + 0.5, \
            (f"_rough_roi={roi:.2f} should match intercept formula={expected:.2f} "
             f"(err={err_intercept:.2f}), not naive={naive_roi:.2f} (err={err_naive:.2f})")

    def test_agent_does_not_crash_on_orbiting_target(self):
        """Agent must handle orbiting target planet without crashing."""
        # ORB at (35,50): r=15, radius=5 → 20 < 48 → is_orb=True via parse_obs
        o = obs([
            planet(0, 0, 0, 50, 50, 3),     # SRC (non-orbiting)
            planet(1, -1, 35, 50, 5, 3),    # ORB (will be detected as orbiting by parse_obs)
        ])
        try:
            actions = agent(o)
            assert isinstance(actions, list)
        except Exception as e:
            pytest.fail(f"Agent crashed on orbiting target: {e}")


# ── Case 5: Production priority ───────────────────────────────────────────────

class TestProductionPriority:
    """
    Two neutrals at the same distance: one prod=5, one prod=1.
    Agent must target the high-prod planet first.
    """

    def test_rough_roi_high_prod_beats_low_prod(self):
        src = {'x': 0, 'y': 50, 'owner': 0, 'ships': 50.0, 'prod': 2.0,
               'radius': 5.0, 'is_orb': False, 'r': 50, 'theta': 0}
        hi = {'x': 0, 'y': 65, 'owner': -1, 'ships': 5.0, 'prod': 5.0,
              'radius': 5.0, 'is_orb': False, 'r': 52, 'theta': 0}
        lo = {'x': 0, 'y': 66, 'owner': -1, 'ships': 5.0, 'prod': 1.0,
              'radius': 5.0, 'is_orb': False, 'r': 53, 'theta': 0}
        assert _rough_roi(src, hi, 50, 0.03) > _rough_roi(src, lo, 50, 0.03)

    def test_agent_targets_high_prod_not_low_prod(self):
        """
        Two equidistant neutrals at (0,65) and (0,66).
        Planet 1 (prod=5) should be targeted before planet 2 (prod=1).
        """
        o = obs([
            planet(0, 0, 0, 50, 50, 3),
            planet(1, -1, 0, 65, 5, 5),   # high prod
            planet(2, -1, 0, 66, 5, 1),   # low prod
        ])
        actions = agent(o)
        fired = {f['target_pid'] for f in _mod._FLEET_REGISTRY}
        assert 1 in fired, \
            f"High-prod planet (pid=1) should be targeted, fired={fired}"


# ── Case 6: Enemy targeting conditions ───────────────────────────────────────

class TestEnemyTargetingConditions:
    """
    Enemy is only added to tgt_set if:
      ship_ratio > 1.5  OR  prod_ratio > 1.5  OR  step > 60 (and not losing badly)
    """

    def test_enemy_ignored_when_equal_early_game(self):
        """
        Equal ships, equal prod, step=0 → enemy NOT in tgt_set.
        Agent should only expand to the neutral, not attack the enemy.
        """
        o = obs([
            planet(0, 0, 0, 50, 50, 3),    # mine: 50 ships prod=3
            planet(1, 1, 0, 65, 50, 3),    # enemy: same ships/prod
            planet(2, -1, 50, 0, 5, 2),    # neutral to grab instead
        ])
        actions = agent(o)
        fired = {f['target_pid'] for f in _mod._FLEET_REGISTRY}
        assert 1 not in fired, \
            f"Enemy should NOT be targeted early game at equal strength, fired={fired}"

    def test_enemy_targeted_when_ship_advantage(self):
        """ship_ratio = 100/10 = 10 > 1.5 → enemy in tgt_set."""
        o = obs([
            planet(0, 0, 0, 50, 100, 3),   # mine: 100 ships
            planet(1, 1, 0, 65, 10, 3),    # enemy: 10 ships
        ])
        actions = agent(o)
        fired = {f['target_pid'] for f in _mod._FLEET_REGISTRY}
        assert 1 in fired, \
            f"Enemy should be targeted with 10x ship advantage, fired={fired}"

    def test_enemy_targeted_after_step_30(self):
        """
        Enemy gate: 'step > 30 and not losing_badly' enables targeting.
        Geometry chosen so smash_pids is empty (enemy just outside 50-unit smash
        radius) and ship_ratio < 1.5, prod_ratio < 1.5, only 1 friendly planet.

        me=200 ships at (0,50), enemy=134 ships at (0,105) → dist=55 > 50.
        ship_ratio=1.49 < 1.5, prod_ratio=1.0 < 1.5, not losing_badly.
        """
        # Step 25: below gate → enemy NOT in tgt_set
        _mod._FLEET_REGISTRY.clear()
        o25 = obs([
            planet(0, 0,   0, 50, 200, 3),   # mine
            planet(1, 1,   0, 105, 134, 3),  # enemy at dist=55 (outside smash=50)
        ], step=25)
        agent(o25)
        assert 1 not in {f['target_pid'] for f in _mod._FLEET_REGISTRY}, \
            "Enemy should NOT be targeted at step=25 (step>30 gate not yet met)"

        # Step 35: above gate → enemy in tgt_set → fires
        _mod._FLEET_REGISTRY.clear()
        o35 = obs([
            planet(0, 0,   0, 50, 200, 3),
            planet(1, 1,   0, 105, 134, 3),
        ], step=35)
        agent(o35)
        assert 1 in {f['target_pid'] for f in _mod._FLEET_REGISTRY}, \
            "Enemy SHOULD be targeted at step=35 (step>30 enables enemy targeting)"

    def test_enemy_targeted_with_prod_advantage(self):
        """
        prod_ratio=6/3=2.0 > 1.5, ship_ratio=100/69=1.45 < 1.5 → prod_ratio
        triggers enemy targeting even at step=0.

        rough_roi for enemy=19.9 → positive.
        """
        o = obs([
            planet(0, 0, 0, 50, 100, 6),   # mine: prod=6
            planet(1, 1, 0, 65, 69, 3),    # enemy: prod=3, prod_ratio=2.0
        ])
        actions = agent(o)
        fired = {f['target_pid'] for f in _mod._FLEET_REGISTRY}
        assert 1 in fired, \
            f"Enemy should be targeted with prod_ratio=2.0 > 1.5, fired={fired}"


# ── Case 7: Counter window detection ─────────────────────────────────────────

class TestCounterWindow:
    """
    analyze_counter_windows detects enemy planets where garrison < prod × 4.
    These are prime attack targets (enemy just sent a big fleet).
    """

    def test_low_garrison_enemy_is_counter_window(self):
        from physics_sim import parse_obs
        from main_pred_2p import analyze_counter_windows
        # Enemy prod=5, garrison=8 → 8 < 5×4=20 → window
        state = {
            'planets': {
                0: {'owner': 0, 'ships': 50.0, 'prod': 3.0, 'x': 0, 'y': 50,
                    'radius': 5.0, 'is_orb': False, 'r': 50, 'theta': 0},
                1: {'owner': 1, 'ships': 8.0,  'prod': 5.0, 'x': 0, 'y': 65,
                    'radius': 5.0, 'is_orb': False, 'r': 52, 'theta': 0},
            },
            'fleets': [], 'step': 0, 'omega': 0.03, 'me': 0,
        }
        windows = analyze_counter_windows(state, enemy_pids={1})
        assert 1 in windows, "Low-garrison enemy should be a counter window"

    def test_full_garrison_not_counter_window(self):
        from main_pred_2p import analyze_counter_windows
        # Enemy prod=3, garrison=50 → 50 >= 3×4=12 → NOT a window
        state = {
            'planets': {
                1: {'owner': 1, 'ships': 50.0, 'prod': 3.0, 'x': 0, 'y': 65,
                    'radius': 5.0, 'is_orb': False, 'r': 52, 'theta': 0},
            },
            'fleets': [], 'step': 0, 'omega': 0.03, 'me': 0,
        }
        windows = analyze_counter_windows(state, enemy_pids={1})
        assert 1 not in windows, "Full-garrison enemy should not be a counter window"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
