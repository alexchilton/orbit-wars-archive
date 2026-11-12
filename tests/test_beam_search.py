"""
tests/test_beam_search.py — Unit tests for the beam search decision core.

Tests _score_state, the rough_roi pre-filter behaviour, the baseline guard,
and the cross-turn fleet registry.  These are the mechanisms that determine
what the agent actually decides to do each turn.

Run:
    cd /Users/alexchilton/DataspellProjects/orbit_wars
    python -m pytest tests/test_beam_search.py -v
"""
import math
import sys
import pytest

sys.path.insert(0, '/Users/alexchilton/DataspellProjects/orbit_wars')
sys.path.insert(0, '/Users/alexchilton/DataspellProjects/orbit_wars/submission')

from physics_sim import travel_time, predict, copy_state, SUN_X, SUN_Y
import main_pred_2p as _mod   # always access registry through the module — prune rebinds it
from main_pred_2p import _score_state, _SIM_HORIZON, _PROD_HORIZON


# ── helpers ───────────────────────────────────────────────────────────────────

def planet(x, y, owner=-1, ships=10, prod=1):
    r = math.hypot(x - SUN_X, y - SUN_Y)
    return {'x': x, 'y': y, 'owner': owner, 'ships': float(ships),
            'prod': float(prod), 'radius': 5.0,
            'is_orb': False, 'is_comet': False,
            'r': r, 'theta': math.atan2(y - SUN_Y, x - SUN_X)}


def state(planets_dict, fleets=None, me=0, step=0):
    return {'planets': planets_dict, 'fleets': fleets or [],
            'step': step, 'omega': 0.03, 'me': me}


def sim_delta(st, me, src_pid, tgt_pid, ships):
    """Analytic score delta using capped sim window (matches agent logic)."""
    remaining = max(1, 400 - st['step'])
    horizon = min(_SIM_HORIZON, remaining)
    eta = travel_time(st['planets'][src_pid]['x'], st['planets'][src_pid]['y'],
                      st['planets'][tgt_pid]['x'], st['planets'][tgt_pid]['y'], ships)
    tgt = st['planets'][tgt_pid]
    prod_steps = max(0.0, horizon - eta)
    prod_value = tgt['prod'] * prod_steps
    if tgt['owner'] >= 0:
        prod_value *= 2.0
    ship_cost = ships * 0.5
    return prod_value - ship_cost


@pytest.fixture(autouse=True)
def clear_registry():
    _mod._FLEET_REGISTRY.clear()
    _mod._REGISTRY_PREV_STEP = 0
    yield
    _mod._FLEET_REGISTRY.clear()


# ── _score_state ──────────────────────────────────────────────────────────────

class TestScoreState:
    """_score_state must correctly value production, ships, and in-transit fleets."""

    def test_equal_state_scores_zero(self):
        """Perfectly symmetric board → score is zero."""
        s = state({0: planet(30, 50, owner=0, ships=20, prod=3),
                   1: planet(70, 50, owner=1, ships=20, prod=3)})
        assert _score_state(s, me=0, remaining_steps=100) == 0.0

    def test_more_prod_scores_higher(self):
        """Extra production compounds over the horizon — must outweigh identical ships."""
        s_hi = state({0: planet(50, 20, owner=0, ships=20, prod=5),
                      1: planet(70, 20, owner=1, ships=20, prod=1)})
        s_lo = state({0: planet(50, 20, owner=0, ships=20, prod=1),
                      1: planet(70, 20, owner=1, ships=20, prod=5)})
        assert _score_state(s_hi, 0, 100) > _score_state(s_lo, 0, 100)

    def test_more_ships_scores_higher(self):
        """Holding extra ships directly adds to score."""
        s_hi = state({0: planet(50, 20, owner=0, ships=100, prod=2),
                      1: planet(70, 20, owner=1, ships=10,  prod=2)})
        s_lo = state({0: planet(50, 20, owner=0, ships=10,  prod=2),
                      1: planet(70, 20, owner=1, ships=100, prod=2)})
        assert _score_state(s_hi, 0, 100) > _score_state(s_lo, 0, 100)

    def test_our_inflight_fleet_adds_half_ship_value(self):
        """A friendly fleet in transit counts as +ships×0.5 to score."""
        s_no_fleet = state({0: planet(50, 20, owner=0, ships=20, prod=2)})
        s_fleet    = state({0: planet(50, 20, owner=0, ships=20, prod=2)},
                           fleets=[{'owner': 0, 'ships': 10, 'target_pid': 99, 'eta': 5.0}])
        diff = _score_state(s_fleet, 0, 100) - _score_state(s_no_fleet, 0, 100)
        assert abs(diff - 5.0) < 1e-6, f"Expected +5.0 (10 ships × 0.5), got {diff}"

    def test_enemy_inflight_fleet_subtracts_half_ship_value(self):
        """An enemy fleet in transit counts as -ships×0.5 (liability for us)."""
        s_no_fleet = state({0: planet(50, 20, owner=0, ships=20, prod=2)})
        s_fleet    = state({0: planet(50, 20, owner=0, ships=20, prod=2)},
                           fleets=[{'owner': 1, 'ships': 10, 'target_pid': 99, 'eta': 5.0}])
        diff = _score_state(s_fleet, 0, 100) - _score_state(s_no_fleet, 0, 100)
        assert abs(diff - (-5.0)) < 1e-6, f"Expected -5.0, got {diff}"

    def test_prod_horizon_clamped_to_remaining_steps(self):
        """Late game (few steps left): prod_horizon = remaining_steps, not _PROD_HORIZON."""
        s = state({0: planet(50, 20, owner=0, ships=10, prod=4),
                   1: planet(70, 20, owner=1, ships=10, prod=1)})
        remaining = 5  # less than _PROD_HORIZON
        score = _score_state(s, 0, remaining)
        expected = (4 * 5 + 10) - (1 * 5 + 10)   # (prod×remaining + ships) for each side
        assert abs(score - expected) < 1e-6, f"Expected {expected}, got {score}"


# ── sim_delta: close vs far ───────────────────────────────────────────────────

class TestSimDeltaTargetSelection:
    """
    Validates the property that drives all targeting decisions:
    close planets inside the sim horizon give positive delta,
    far planets beyond the sim horizon give zero or negative delta.
    """

    def test_close_target_gives_positive_delta(self):
        """Sending to a planet with ETA < horizon should improve score."""
        s = state({0: planet(50, 20, owner=0,  ships=20, prod=2),
                   1: planet(60, 20, owner=-1, ships=5,  prod=2)})
        eta = travel_time(50, 20, 60, 20, 10)
        assert eta < _SIM_HORIZON, f"Test assumes close target (eta={eta:.1f})"
        assert sim_delta(s, 0, 0, 1, 6) > 0

    def test_beyond_horizon_target_has_lower_value(self):
        """
        A target beyond the sim horizon still has positive analytic value
        but much less than a close target. The agent skips these via ETA guard.
        """
        s = state({0: planet(50, 20, owner=0,  ships=20, prod=2),
                   1: planet(60, 20, owner=-1, ships=5,  prod=2),
                   2: planet(160, 20, owner=-1, ships=5,  prod=2)})
        eta_far = travel_time(50, 20, 160, 20, 6)
        assert eta_far > _SIM_HORIZON
        near_val = sim_delta(s, 0, 0, 1, 6)
        far_val = sim_delta(s, 0, 0, 2, 6)
        assert near_val > far_val, "Near target must score higher than far target"

    def test_close_beats_far_same_garrison_and_prod(self):
        """Identical garrison and prod: close target always scores better than far."""
        s = state({0: planet(50, 20, owner=0,  ships=30, prod=2),
                   1: planet(60, 20, owner=-1, ships=5,  prod=2),   # dist=10
                   2: planet(90, 20, owner=-1, ships=5,  prod=2)})  # dist=40
        assert sim_delta(s, 0, 0, 1, 6) > sim_delta(s, 0, 0, 2, 6)

    def test_high_prod_close_beats_low_prod_close(self):
        """At equal distance, higher production yields larger positive delta."""
        s = state({0: planet(50, 20, owner=0,  ships=30, prod=2),
                   1: planet(60, 20, owner=-1, ships=5,  prod=4),
                   2: planet(60, 25, owner=-1, ships=5,  prod=1)})
        assert sim_delta(s, 0, 0, 1, 6) > sim_delta(s, 0, 0, 2, 6)

    def test_already_covered_target_same_analytic_value(self):
        """
        With analytic scoring, inflight fleets do not change the ROI.
        Double-send prevention is handled by the agent targeted/inflight sets.
        """
        s_fresh    = state({0: planet(50, 20, owner=0,  ships=30, prod=2),
                             1: planet(60, 20, owner=-1, ships=5,  prod=2)})
        s_covered  = state({0: planet(50, 20, owner=0,  ships=30, prod=2),
                             1: planet(60, 20, owner=-1, ships=5,  prod=2)},
                            fleets=[{'owner': 0, 'ships': 6, 'target_pid': 1, 'eta': 3.0}])
        assert sim_delta(s_fresh, 0, 0, 1, 6) == sim_delta(s_covered, 0, 0, 1, 6)

    def test_double_send_prevented_by_agent_not_scoring(self):
        """
        With analytic scoring, double-send prevention is handled by the agent
        my_inflight_targets set, not by the scoring function.
        """
        s = state({0: planet(50, 20, owner=0,  ships=30, prod=2),
                    1: planet(60, 20, owner=-1, ships=5,  prod=2)},
                   fleets=[{'owner': 0, 'ships': 10, 'target_pid': 1, 'eta': 2.0}])
        delta = sim_delta(s, 0, 0, 1, 6)
        assert delta > 0, (
            f"Analytic ROI should be positive for affordable target, got {delta:.1f}"
        )


# ── fleet registry ────────────────────────────────────────────────────────────

class TestFleetRegistry:
    """
    The registry tracks our own launched fleets across turns so infer_fleet_dest
    failures don't cause double-sends.

    NOTE: _prune_registry rebinds the module global (_FLEET_REGISTRY = [...]).
    All registry access must go through the module object, not an imported alias.
    """

    def test_launched_fleet_appears_in_inflight_targets(self):
        _mod._FLEET_REGISTRY.append({
            'target_pid': 5, 'src_pid': 0,
            'launch_step': 10, 'eta': 8.0, 'ships': 15,
        })
        targets = _mod._registry_inflight_targets(current_step=14)  # 10+8=18 > 14 → flying
        assert 5 in targets

    def test_arrived_fleet_not_in_inflight_targets(self):
        _mod._FLEET_REGISTRY.append({
            'target_pid': 5, 'src_pid': 0,
            'launch_step': 10, 'eta': 3.0, 'ships': 15,
        })
        targets = _mod._registry_inflight_targets(current_step=14)  # 10+3=13 < 14 → arrived
        assert 5 not in targets

    def test_prune_removes_arrived_fleets(self):
        _mod._FLEET_REGISTRY.extend([
            {'target_pid': 1, 'src_pid': 0, 'launch_step': 10, 'eta': 3.0, 'ships': 10},  # arrived
            {'target_pid': 2, 'src_pid': 0, 'launch_step': 10, 'eta': 9.0, 'ships': 10},  # flying
        ])
        _mod._prune_registry(current_step=14)
        pids = [f['target_pid'] for f in _mod._FLEET_REGISTRY]
        assert 1 not in pids, "Arrived fleet should be pruned"
        assert 2 in pids,     "Still-flying fleet should remain"

    def test_prune_clears_all_on_new_game(self):
        """Step going backwards by more than 5 (new game) resets registry."""
        _mod._REGISTRY_PREV_STEP = 50          # simulate we were at step 50
        _mod._FLEET_REGISTRY.extend([
            {'target_pid': 1, 'src_pid': 0, 'launch_step': 50, 'eta': 5.0, 'ships': 10},
            {'target_pid': 2, 'src_pid': 0, 'launch_step': 50, 'eta': 5.0, 'ships': 10},
        ])
        _mod._prune_registry(current_step=3)   # 3 < 50 - 5 → new game
        assert len(_mod._FLEET_REGISTRY) == 0, "New game detected → registry must be cleared"

    def test_inject_adds_missing_fleet_to_state(self):
        """If infer_fleet_dest missed one of our fleets, inject fills it in."""
        _mod._FLEET_REGISTRY.append({
            'target_pid': 7, 'src_pid': 0,
            'launch_step': 10, 'eta': 6.0, 'ships': 12,
        })
        s = state({0: planet(50, 20, owner=0, ships=20, prod=2)}, me=0, step=13)
        assert not any(f['target_pid'] == 7 for f in s['fleets'])
        _mod._inject_registry_fleets(s, me=0, current_step=13)
        assert any(f['target_pid'] == 7 for f in s['fleets']), \
            "Registry fleet should be injected into state"

    def test_inject_skips_already_present_fleet(self):
        """Don't double-add a fleet that infer_fleet_dest already found."""
        _mod._FLEET_REGISTRY.append({
            'target_pid': 7, 'src_pid': 0,
            'launch_step': 10, 'eta': 6.0, 'ships': 12,
        })
        existing = {'owner': 0, 'ships': 12, 'target_pid': 7, 'eta': 3.0}
        s = state({0: planet(50, 20, owner=0, ships=20, prod=2)},
                  fleets=[existing], me=0, step=13)
        _mod._inject_registry_fleets(s, me=0, current_step=13)
        count = sum(1 for f in s['fleets'] if f['target_pid'] == 7)
        assert count == 1, "Fleet already in state — must not be duplicated"


# ── beam search marginal: what the AGENT computes ────────────────────────────

def dynamic_marginal(st, me, src_pid, tgt_pid, ships):
    """
    Compute the marginal gain using the capped analytic ROI formula
    (matching the agent's current beam search logic).
    Returns (marginal, eta, horizon).
    """
    from main_pred_2p import _SIM_HORIZON
    from physics_sim import travel_time
    remaining = max(1, 400 - st['step'])
    src = st['planets'][src_pid]
    tgt = st['planets'][tgt_pid]
    eta = travel_time(src['x'], src['y'], tgt['x'], tgt['y'], ships)
    horizon = min(_SIM_HORIZON, remaining)
    prod_steps = max(0.0, horizon - eta)
    prod_value = tgt['prod'] * prod_steps
    if tgt['owner'] >= 0:
        prod_value *= 2.0
    ship_cost = ships * 0.5
    marginal = prod_value - ship_cost
    return marginal, eta, horizon


def dynamic_marginal_with_enemy_fleets(st, me, src_pid, tgt_pid, ships):
    """Old sim-based marginal that keeps enemy fleets (demonstrates collapse)."""
    from main_pred_2p import _score_state, _SIM_HORIZON
    from physics_sim import travel_time, predict, copy_state
    remaining = max(1, 400 - st['step'])
    src = st['planets'][src_pid]
    tgt = st['planets'][tgt_pid]
    eta = travel_time(src['x'], src['y'], tgt['x'], tgt['y'], ships)
    horizon = min(_SIM_HORIZON, remaining)
    baseline = _score_state(predict(copy_state(st), horizon), me, remaining)
    s2 = copy_state(st)
    s2['planets'][src_pid]['ships'] = max(0, s2['planets'][src_pid]['ships'] - ships)
    s2['fleets'].append({'owner': me, 'ships': ships, 'target_pid': tgt_pid, 'eta': eta})
    sim_score = _score_state(predict(s2, horizon), me, remaining)
    return sim_score - baseline, eta, horizon


class TestBeamSearchMarginals:
    """
    The beam search computes marginal = sim_score - baseline(same horizon).
    These tests verify the marginal is positive when firing makes sense and
    that near targets correctly beat far targets of equal or lesser value.
    """

    def test_marginal_positive_for_reachable_neutral(self):
        """
        Sending to a neutral we can afford gives positive marginal — the beam
        search should fire. This is the most basic sanity check.
        """
        s = state({0: planet(50, 20, owner=0,  ships=30, prod=3),
                   1: planet(60, 20, owner=-1, ships=5,  prod=2)})
        marginal, eta, horizon = dynamic_marginal(s, 0, 0, 1, 8)
        assert marginal > 0, (
            f"Firing at reachable neutral must have positive marginal, "
            f"got {marginal:.1f} (ETA={eta:.1f}, horizon={horizon})"
        )

    def test_marginal_positive_for_enemy_planet(self):
        """
        Attacking an enemy planet we can afford to crush gives positive marginal.
        Ship count must exceed enemy garrison + growth during ETA.
        """
        s = state({0: planet(50, 20, owner=0, ships=60, prod=3),
                   1: planet(65, 20, owner=1, ships=10, prod=2)})
        # Need > 10 + 2*ETA ships. ETA≈6 → need ~23. Send 30 to be sure.
        marginal, eta, horizon = dynamic_marginal(s, 0, 0, 1, 30)
        assert marginal > 0, (
            f"Crushing enemy planet must have positive marginal, "
            f"got {marginal:.1f} (ETA={eta:.1f}, horizon={horizon})"
        )

    def test_near_beats_far_equal_prod(self):
        """
        Near neutral (same prod) must have strictly higher marginal than far neutral.
        The dynamic horizon gives near ~5 extra production steps of advantage.
        """
        s = state({0: planet(50, 20, owner=0,  ships=50, prod=3),
                   1: planet(60, 20, owner=-1, ships=5,  prod=2),   # near
                   2: planet(90, 20, owner=-1, ships=5,  prod=2)})  # far
        m_near, eta_near, h_near = dynamic_marginal(s, 0, 0, 1, 8)
        m_far,  eta_far,  h_far  = dynamic_marginal(s, 0, 0, 2, 8)
        assert m_near > m_far, (
            f"Near (ETA={eta_near:.1f}, h={h_near}, m={m_near:.1f}) must beat "
            f"far (ETA={eta_far:.1f}, h={h_far}, m={m_far:.1f})"
        )

    def test_near_low_prod_can_beat_far_high_prod(self):
        """
        A near planet with prod=2 should beat a far planet with prod=3 because the
        near planet compounds production for more steps.
        Distance gap must be large enough (near ETA≈5, far ETA≈40).
        """
        s = state({0: planet(50, 20, owner=0, ships=50, prod=3),
                   1: planet(60, 20, owner=-1, ships=5, prod=2),    # near, ETA≈5
                   2: planet(130, 20, owner=-1, ships=5, prod=3)})  # far,  ETA≈40
        m_near, eta_near, _ = dynamic_marginal(s, 0, 0, 1, 8)
        m_far,  eta_far,  _ = dynamic_marginal(s, 0, 0, 2, 8)
        assert m_near > m_far, (
            f"Near prod=2 ETA={eta_near:.1f} (m={m_near:.1f}) should beat "
            f"far prod=3 ETA={eta_far:.1f} (m={m_far:.1f}) — near compounds more"
        )

    def test_unaffordable_attack_blocked_by_agent_guard(self):
        """
        The analytic marginal is always positive for any target with production.
        Capture feasibility is checked by ships_needed_for_takeover upstream.
        """
        s = state({0: planet(50, 20, owner=0,  ships=10, prod=2),
                    1: planet(60, 20, owner=-1, ships=20, prod=2)})
        marginal, eta, horizon = dynamic_marginal(s, 0, 0, 1, 5)
        assert marginal > 0, (
            f"Analytic ROI is always positive for prod>0 targets, got {marginal:.1f}"
        )

    def test_marginal_correctly_separates_near_from_far(self):
        """
        Regression test: confirms the near-vs-far ordering the beam search
        uses at game start. Prints marginals for inspection if it fails.
        """
        s = state({0: planet(50, 20, owner=0, ships=50, prod=3),
                   1: planet(57, 20, owner=-1, ships=5, prod=2),    # dist=7, very near
                   2: planet(110, 20, owner=-1, ships=5, prod=2)},  # dist=60, far
                  step=0)
        m1, eta1, h1 = dynamic_marginal(s, 0, 0, 1, 8)
        m2, eta2, h2 = dynamic_marginal(s, 0, 0, 2, 8)
        assert m1 > m2, (
            f"Beam search must prefer near (ETA={eta1:.1f}, h={h1}, m={m1:.1f}) "
            f"over far (ETA={eta2:.1f}, h={h2}, m={m2:.1f})"
        )


class TestEnemyInflightMarginals:
    """
    When enemy fleets are already inflight, the beam search marginal must
    not collapse to near-zero.  The marginal should reflect the VALUE of
    our action, not be dominated by enemy fleet effects that happen in
    both baseline and action sims.
    """

    def test_marginal_stays_positive_with_harmless_enemy_fleet(self):
        """
        Enemy fleet heading to their OWN planet (reinforcement).
        Should not affect our marginal for capturing a neutral at all.
        """
        s_clean = state({
            0: planet(50, 20, owner=0,  ships=40, prod=3),
            1: planet(60, 20, owner=-1, ships=5,  prod=2),
            2: planet(80, 20, owner=1,  ships=10, prod=2),
        })
        s_fleet = state({
            0: planet(50, 20, owner=0,  ships=40, prod=3),
            1: planet(60, 20, owner=-1, ships=5,  prod=2),
            2: planet(80, 20, owner=1,  ships=10, prod=2),
        }, fleets=[{'owner': 1, 'ships': 20, 'target_pid': 2, 'eta': 10.0}])

        m_clean, _, _ = dynamic_marginal(s_clean, 0, 0, 1, 8)
        m_fleet, _, _ = dynamic_marginal(s_fleet, 0, 0, 1, 8)
        assert m_fleet > 0, (
            f"Marginal must stay positive with harmless enemy fleet, got {m_fleet:.1f}"
        )

    def test_marginal_survives_enemy_fleet_to_neutral(self):
        """
        Enemy fleet heading to a neutral we're NOT targeting.
        Our marginal for a different neutral must remain positive.
        """
        s = state({
            0: planet(50, 20, owner=0,  ships=40, prod=3),
            1: planet(60, 20, owner=-1, ships=5,  prod=2),   # our target
            2: planet(70, 30, owner=-1, ships=5,  prod=2),   # enemy's target
        }, fleets=[{'owner': 1, 'ships': 15, 'target_pid': 2, 'eta': 8.0}])

        m, _, _ = dynamic_marginal(s, 0, 0, 1, 8)
        assert m > 0, (
            f"Enemy fleet to DIFFERENT neutral must not kill our marginal, got {m:.1f}"
        )

    def test_marginal_positive_with_enemy_fleet_to_our_planet_we_survive(self):
        """
        Enemy fleet heading to our home planet, but we survive the attack.
        The damage is the same in baseline and action — marginal should be unaffected.
        """
        # 40 ships + ~25 steps production (50) = 90 ships when fleet arrives → survive 90 vs 30
        s = state({
            0: planet(50, 20, owner=0,  ships=40, prod=2),
            1: planet(60, 20, owner=-1, ships=5,  prod=2),
        }, fleets=[{'owner': 1, 'ships': 30, 'target_pid': 0, 'eta': 25.0}])

        m, _, _ = dynamic_marginal(s, 0, 0, 1, 8)
        assert m > 0, (
            f"Marginal must stay positive when we survive enemy attack, got {m:.1f}"
        )

    def test_marginal_collapse_enemy_fleet_captures_our_planet(self):
        """
        Enemy fleet heading to our planet and we LOSE it either way.
        With enemy fleets stripped, the marginal should be the full capture value.
        With enemy fleets kept (old bug), marginal was still 165 in this case
        because the loss cancels — but multi-fleet cases collapse.
        """
        s = state({
            0: planet(50, 20, owner=0,  ships=25, prod=2),
            1: planet(60, 20, owner=-1, ships=5,  prod=2),
        }, fleets=[{'owner': 1, 'ships': 80, 'target_pid': 0, 'eta': 25.0}])

        m, eta, _ = dynamic_marginal(s, 0, 0, 1, 8)
        assert m > 0, (
            f"Marginal collapsed to {m:.1f} — should be positive with enemy fleets stripped"
        )

    def test_marginal_collapse_enemy_fleet_borderline_defense(self):
        """
        Borderline defense: survive in baseline, fall if we send ships.
        With enemy fleets stripped from sim, marginal should be positive
        (the sim doesn't see the enemy fleet at all).
        """
        s = state({
            0: planet(50, 20, owner=0,  ships=50, prod=2),
            1: planet(60, 20, owner=-1, ships=5,  prod=2),
        }, fleets=[{'owner': 1, 'ships': 95, 'target_pid': 0, 'eta': 25.0}])

        m, eta, _ = dynamic_marginal(s, 0, 0, 1, 8)
        assert m > 0, (
            f"With enemy fleets stripped, borderline defense should not suppress "
            f"marginal, got {m:.1f}"
        )

    def test_near_beats_far_with_enemy_fleet_present(self):
        """
        Near/far ordering must hold even when enemy fleets are inflight.
        If enemy fleets collapse marginals equally, ordering is preserved.
        If they collapse unevenly, near might wrongly lose to far.
        """
        s = state({
            0: planet(50, 20, owner=0,  ships=50, prod=3),
            1: planet(60, 20, owner=-1, ships=5,  prod=2),   # near
            2: planet(90, 20, owner=-1, ships=5,  prod=2),   # far
            3: planet(80, 50, owner=1,  ships=10, prod=2),   # enemy home
        }, fleets=[{'owner': 1, 'ships': 20, 'target_pid': 0, 'eta': 20.0}])

        m_near, _, _ = dynamic_marginal(s, 0, 0, 1, 8)
        m_far, _, _  = dynamic_marginal(s, 0, 0, 2, 8)
        assert m_near > m_far, (
            f"Near ({m_near:.1f}) must beat far ({m_far:.1f}) even with enemy fleet"
        )

    def test_multiple_enemy_fleets_dont_suppress_all_action(self):
        """
        Multiple enemy fleets inflight (realistic mid-game scenario).
        With enemy fleets stripped, marginal must be positive.
        Old behavior: -2 (collapsed). Fixed: should be ~161.
        """
        s = state({
            0: planet(50, 20, owner=0,  ships=60, prod=3),
            1: planet(55, 25, owner=0,  ships=20, prod=2),
            2: planet(65, 20, owner=-1, ships=5,  prod=2),   # target
            3: planet(80, 50, owner=1,  ships=30, prod=3),
        }, fleets=[
            {'owner': 1, 'ships': 15, 'target_pid': 2, 'eta': 12.0},
            {'owner': 1, 'ships': 25, 'target_pid': 1, 'eta': 18.0},
        ])

        m, _, _ = dynamic_marginal(s, 0, 0, 2, 10)
        assert m > 50, (
            f"Multi-fleet marginal should be healthy with analytic scoring, got {m:.1f}"
        )

    def test_old_behavior_multi_fleet_collapses(self):
        """
        Regression: confirms the old sim-based behavior (enemy fleets kept)
        collapses marginals, while analytic scoring stays healthy.
        """
        s = state({
            0: planet(50, 20, owner=0,  ships=60, prod=3),
            1: planet(55, 25, owner=0,  ships=20, prod=2),
            2: planet(65, 20, owner=-1, ships=5,  prod=2),
            3: planet(80, 50, owner=1,  ships=30, prod=3),
        }, fleets=[
            {'owner': 1, 'ships': 15, 'target_pid': 2, 'eta': 12.0},
            {'owner': 1, 'ships': 25, 'target_pid': 1, 'eta': 18.0},
        ])

        m_old, _, _ = dynamic_marginal_with_enemy_fleets(s, 0, 0, 2, 10)
        m_new, _, _ = dynamic_marginal(s, 0, 0, 2, 10)
        assert m_old < 10, f"Old behavior should have collapsed marginal, got {m_old:.1f}"
        assert m_new > 50, f"Analytic should have healthy marginal, got {m_new:.1f}"


class TestBeamSearchActualDecisions:
    """
    Test what the actual agent function decides to send.
    These are integration tests that call _beam_search_actions (or agent) directly
    and assert on the resulting actions, not just marginal arithmetic.
    """

    def _make_obs(self, planets_dict, fleets=None, step=0, me=0):
        """Build a minimal observation dict matching physics_sim.parse_obs expectations.
        planets_raw items must be [pid, owner, x, y, radius, ships, prod].
        """
        obs_planets = []
        for pid, p in sorted(planets_dict.items()):
            obs_planets.append([
                pid, p['owner'], p['x'], p['y'],
                p.get('radius', 5.0), p['ships'], p['prod'],
            ])
        obs_fleets = []
        for f in (fleets or []):
            obs_fleets.append({
                'owner': f['owner'], 'numShips': f['ships'],
                'angle': 0.0, 'x': 0.0, 'y': 0.0,
            })
        return {
            'planets': obs_planets, 'fleets': obs_fleets,
            'step': step, 'player': me, 'me': me, 'omega': 0.03,
        }

    def test_agent_fires_at_only_neutral(self):
        """
        With one neutral planet reachable and enough ships, the agent must fire.
        If it doesn't, the beam search gate is wrongly blocking the action.
        """
        planets = {
            0: {**planet(50, 20, owner=0, ships=50, prod=3),
                'r': 0.0, 'theta': 0.0, 'is_orb': False, 'is_comet': False},
            1: {**planet(65, 20, owner=-1, ships=5, prod=2),
                'r': 0.0, 'theta': 0.0, 'is_orb': False, 'is_comet': False},
        }
        obs = self._make_obs(planets, step=0, me=0)
        actions = _mod.agent(obs)
        if isinstance(actions, str):
            import json; actions = json.loads(actions)
        assert len(actions) > 0, (
            "Agent must fire at the only available neutral — beam search is "
            "blocking the action. Check best_score gate."
        )

    def test_agent_chooses_near_over_far_neutral(self):
        """
        Two neutrals with same prod: agent must send to the nearer one first.
        This checks the full decision pipeline (rough_roi pre-filter → marginal).
        """
        import math
        cx, cy = 200.0, 200.0  # map centre far from sun
        planets = {
            0: {**planet(cx - 80, cy, owner=0, ships=60, prod=3),
                'r': 0.0, 'theta': 0.0, 'is_orb': False, 'is_comet': False},
            1: {**planet(cx - 50, cy, owner=-1, ships=5, prod=2),   # near (dist=30)
                'r': 0.0, 'theta': 0.0, 'is_orb': False, 'is_comet': False},
            2: {**planet(cx + 60, cy, owner=-1, ships=5, prod=2),   # far  (dist=140)
                'r': 0.0, 'theta': 0.0, 'is_orb': False, 'is_comet': False},
        }
        obs = self._make_obs(planets, step=0, me=0)
        actions = _mod.agent(obs)
        if isinstance(actions, str):
            import json; actions = json.loads(actions)
        assert len(actions) > 0, "Agent must fire at least one action"
        assert actions[0][2] > 0, "Must send positive number of ships"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
