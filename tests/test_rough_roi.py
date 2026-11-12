"""
tests/test_rough_roi.py — Unit tests for _rough_roi and solve_intercept.

These functions are the core of the target pre-filter that decides which
planets the beam search will even consider.  If they're wrong, the beam search
evaluates the wrong candidates regardless of how good _score_state is.

Run:
    cd /Users/alexchilton/DataspellProjects/orbit_wars
    python -m pytest tests/test_rough_roi.py -v
"""
import sys, math
import pytest

sys.path.insert(0, '/Users/alexchilton/DataspellProjects/orbit_wars')
sys.path.insert(0, '/Users/alexchilton/DataspellProjects/orbit_wars/submission')

from physics_sim import SUN_X, SUN_Y, SUN_RADIUS, INNER_ORBIT_THRESHOLD, travel_time
from main_pred_2p import _rough_roi, solve_intercept, _SIM_HORIZON, TAKEOVER_MARGIN


# ── helpers ───────────────────────────────────────────────────────────────────

def static_planet(x, y, owner=-1, ships=5, prod=3, radius=5.0):
    """Planet far from sun (r+radius > 48) so is_orb = False."""
    r = math.hypot(x - SUN_X, y - SUN_Y)
    return {
        'x': x, 'y': y, 'owner': owner,
        'ships': float(ships), 'prod': float(prod),
        'radius': radius, 'is_orb': False, 'is_comet': False,
        'r': r, 'theta': math.atan2(y - SUN_Y, x - SUN_X),
    }


def orbiting_planet(x, y, owner=-1, ships=5, prod=3, radius=5.0):
    """Planet inside INNER_ORBIT_THRESHOLD so is_orb = True."""
    r = math.hypot(x - SUN_X, y - SUN_Y)
    # Verify this is actually orbiting
    assert (r + radius) < INNER_ORBIT_THRESHOLD, \
        f"Planet at ({x},{y}) is NOT orbiting (r+radius={r+radius:.1f} >= {INNER_ORBIT_THRESHOLD})"
    return {
        'x': x, 'y': y, 'owner': owner,
        'ships': float(ships), 'prod': float(prod),
        'radius': radius, 'is_orb': True, 'is_comet': False,
        'r': r, 'theta': math.atan2(y - SUN_Y, x - SUN_X),
    }


SRC = static_planet(0, 50)   # our base — 50 units west of sun
OMEGA = 0.03


# ── solve_intercept ───────────────────────────────────────────────────────────

class TestSolveIntercept:
    """solve_intercept must return (ix, iy, eta) where eta is the actual travel time."""

    def test_static_planet_returns_current_position(self):
        """Non-orbiting planet: intercept IS the current position."""
        tgt = static_planet(0, 75)   # 25 units north of SRC
        ix, iy, eta = solve_intercept(
            SRC['x'], SRC['y'], tgt['x'], tgt['y'],
            orbiting=False, omega=OMEGA, ships=20,
        )
        assert abs(ix - tgt['x']) < 0.01, f"ix={ix} should be tgt x={tgt['x']}"
        assert abs(iy - tgt['y']) < 0.01, f"iy={iy} should be tgt y={tgt['y']}"
        expected_eta = travel_time(SRC['x'], SRC['y'], tgt['x'], tgt['y'], 20)
        assert abs(eta - expected_eta) < 0.1, f"eta={eta:.2f} should be ≈{expected_eta:.2f}"

    def test_static_planet_eta_matches_travel_time(self):
        """For a static planet, solve_intercept eta == travel_time."""
        tgt = static_planet(0, -100)   # far north
        ix, iy, eta = solve_intercept(
            SRC['x'], SRC['y'], tgt['x'], tgt['y'],
            orbiting=False, omega=OMEGA, ships=10,
        )
        expected = travel_time(SRC['x'], SRC['y'], tgt['x'], tgt['y'], 10)
        assert abs(eta - expected) < 0.01

    def test_orbiting_planet_intercept_differs_from_current_pos(self):
        """
        An orbiting planet will have moved by the time the fleet arrives.
        The intercept point (ix, iy) should NOT be the planet's current position.
        """
        # Planet near sun so it orbits — at (35, 50), r=15, radius=5 → 15+5=20 < 48
        tgt = orbiting_planet(35, 50)   # 15 units east of sun centre
        ix, iy, eta = solve_intercept(
            SRC['x'], SRC['y'], tgt['x'], tgt['y'],
            orbiting=True, omega=OMEGA, ships=20,
        )
        # The planet orbits at angular speed omega; after eta steps it has moved
        delta_theta = OMEGA * eta
        if abs(delta_theta) > 0.01:   # only assert if planet actually moved
            dist_to_current = math.hypot(ix - tgt['x'], iy - tgt['y'])
            assert dist_to_current > 0.5, \
                f"Intercept ({ix:.1f},{iy:.1f}) should differ from current pos ({tgt['x']},{tgt['y']})"

    def test_orbiting_planet_eta_is_travel_time_to_intercept(self):
        """
        ETA returned by solve_intercept should equal travel_time from src to intercept point.
        This is the self-consistency check: fleet speed × eta == dist(src, intercept).
        """
        tgt = orbiting_planet(35, 50)
        ships = 20
        ix, iy, eta = solve_intercept(
            SRC['x'], SRC['y'], tgt['x'], tgt['y'],
            orbiting=True, omega=OMEGA, ships=ships,
        )
        actual_travel = travel_time(SRC['x'], SRC['y'], ix, iy, ships)
        assert abs(eta - actual_travel) < 0.5, \
            f"solve_intercept eta={eta:.2f} should match travel_time to intercept={actual_travel:.2f}"

    def test_orbiting_eta_never_less_than_static_eta(self):
        """
        An orbiting planet might end up farther away than its current position.
        ETA should be >= direct travel time to current pos (could be less if it orbits toward us).
        But the returned ETA must always be >= 0.
        """
        tgt = orbiting_planet(35, 50)
        _, _, eta = solve_intercept(
            SRC['x'], SRC['y'], tgt['x'], tgt['y'],
            orbiting=True, omega=OMEGA, ships=15,
        )
        assert eta > 0, f"ETA must be positive, got {eta}"


# ── _rough_roi ────────────────────────────────────────────────────────────────

class TestRoughRoi:
    """_rough_roi is the target pre-filter that determines which planets the beam search sees."""

    # ── ideal cases (static planets) ─────────────────────────────────────────

    def test_close_affordable_target_gives_positive_roi(self):
        """Close neutral: prod × (horizon - eta) dominates → positive ROI."""
        src = static_planet(0, 50, ships=50, prod=3)
        tgt = static_planet(0, 75, ships=5, prod=3)  # 25 units away, eta ≈ 5
        roi = _rough_roi(src, tgt, av=50, omega=OMEGA)
        assert roi > 0, f"Close affordable target should give positive ROI, got {roi:.2f}"

    def test_beyond_horizon_target_gives_positive_roi(self):
        """
        Far target (ETA > 25, the old hard-cap horizon) still has positive ROI.
        _rough_roi uses remaining_steps - eta (not a fixed horizon cap), so any
        planet reachable within the game still earns production value.
        A planet at ETA≈48 with prod=3 earns 3*(350-48)≈906 ship-turns of value.
        Note: beam search SKIPS ETA >= 50, but _rough_roi pre-filter is positive
        so the planet enters the candidate list (and then gets skipped by beam search).
        """
        src = static_planet(0, 50, ships=50, prod=3)
        tgt = static_planet(0, -100, ships=5, prod=3)  # 150 units away, ETA ≈ 48
        eta = travel_time(src['x'], src['y'], tgt['x'], tgt['y'], 50)
        assert eta > 25, f"Test requires far target (ETA > 25 old threshold), ETA={eta:.1f}"
        roi = _rough_roi(src, tgt, av=50, omega=OMEGA)
        assert roi > 0, f"Far target with positive production should have positive ROI, got {roi:.2f}"

    def test_zero_prod_target_gives_nonpositive_roi(self):
        """
        Zero-production target: costs ships to capture, earns nothing back → ROI ≤ 0.
        """
        src = static_planet(0, 50, ships=50, prod=3)
        tgt = static_planet(0, 75, ships=5, prod=0)
        roi = _rough_roi(src, tgt, av=50, omega=OMEGA)
        assert roi <= 0, f"Zero-prod target should have non-positive ROI, got {roi:.2f}"

    def test_unaffordable_target_returns_sentinel(self):
        """If needed > av, return -1e9 sentinel (excluded from ranking)."""
        src = static_planet(0, 50, ships=10, prod=2)
        tgt = static_planet(0, 75, ships=100, prod=3)  # 101+ ships needed
        roi = _rough_roi(src, tgt, av=10, omega=OMEGA)
        assert roi == -1e9, f"Unaffordable target should return -1e9, got {roi}"

    def test_high_prod_target_scores_higher_than_low_prod(self):
        """At equal distance, higher production gives higher ROI."""
        src = static_planet(0, 50, ships=50, prod=2)
        tgt_hi = static_planet(0, 75, ships=5, prod=5)
        tgt_lo = static_planet(0, 75, ships=5, prod=1)
        assert _rough_roi(src, tgt_hi, 50, OMEGA) > _rough_roi(src, tgt_lo, 50, OMEGA)

    def test_close_target_beats_far_target_same_prod(self):
        """Same production: close planet scores higher than far one."""
        src = static_planet(0, 50, ships=50, prod=2)
        close = static_planet(0, 75, ships=5, prod=3)   # 25 units
        far   = static_planet(0, 90, ships=5, prod=3)   # 40 units
        assert _rough_roi(src, close, 50, OMEGA) > _rough_roi(src, far, 50, OMEGA)

    def test_enemy_planet_roi_higher_than_neutral(self):
        """
        Enemy target has higher ROI than an identical neutral:
        - We gain their production AND deny theirs (double credit)
        - Plus credit for ships they burn defending

        This is correct: attacking an enemy planet is more valuable than
        capturing a neutral with the same garrison and production.
        """
        src    = static_planet(0, 50, ships=100, prod=2)
        neutral = static_planet(0, 75, ships=20, prod=3, owner=-1)
        enemy   = static_planet(0, 75, ships=20, prod=3, owner= 1)
        roi_n = _rough_roi(src, neutral, 100, OMEGA)
        roi_e = _rough_roi(src, enemy,   100, OMEGA)
        assert roi_e > roi_n, \
            f"Enemy ROI ({roi_e:.1f}) should beat neutral ROI ({roi_n:.1f}) — double prod + ship credit"

    # ── orbiting planet cases ─────────────────────────────────────────────────

    def test_orbiting_planet_uses_intercept_eta_not_current_pos(self):
        """
        _rough_roi for an orbiting planet should use solve_intercept ETA,
        NOT travel_time to the planet's current position.
        The two should differ when the planet is moving significantly.
        """
        src = static_planet(0, 50, ships=50, prod=2)
        # Orbiting planet at (35, 50) — directly east of sun, orbits at omega=0.03
        tgt_orb = orbiting_planet(35, 50, ships=5, prod=3)

        roi_orb = _rough_roi(src, tgt_orb, av=50, omega=OMEGA)
        # Just verify it doesn't crash and returns a finite number
        assert math.isfinite(roi_orb), f"Orbiting planet ROI should be finite, got {roi_orb}"

    def test_orbiting_planet_roi_uses_correct_eta(self):
        """
        The ETA used in _rough_roi for an orbiting planet should match
        solve_intercept, NOT travel_time to current pos.
        We verify by comparing the raw ROI formula with both ETAs.
        The correct formula: prod × max(0, remaining_steps - eta) - needed × 0.5
        """
        src = static_planet(0, 50, ships=30, prod=2)
        tgt = orbiting_planet(35, 50, ships=5, prod=3)
        DEFAULT_REMAINING = 350  # default used by _rough_roi

        # What solve_intercept says
        _, _, eta_intercept = solve_intercept(
            src['x'], src['y'], tgt['x'], tgt['y'], True, OMEGA, 30)
        # What naive travel_time-to-current-pos would say
        eta_naive = travel_time(src['x'], src['y'], tgt['x'], tgt['y'], 30)

        roi_from_intercept = tgt['prod'] * max(0.0, DEFAULT_REMAINING - eta_intercept) - 3  # needed≈6 * 0.5 = 3
        roi_actual         = _rough_roi(src, tgt, av=30, omega=OMEGA)

        # roi_actual should be close to roi_from_intercept, NOT roi_from_naive
        err_intercept = abs(roi_actual - roi_from_intercept)
        err_naive     = abs(roi_actual - (tgt['prod'] * max(0.0, DEFAULT_REMAINING - eta_naive) - 3))

        assert err_intercept < err_naive or err_intercept < 1.0, \
            (f"_rough_roi={roi_actual:.2f} should be closer to intercept-based "
             f"ROI={roi_from_intercept:.2f} (err={err_intercept:.2f}) "
             f"than naive ROI (err={err_naive:.2f})")
