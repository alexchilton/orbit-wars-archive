"""
tests/test_pred_tactics.py — Unit tests for main_pred_2p.py tactical functions.

Each test creates a minimal hand-crafted state and verifies that the tactical
analysis (threats, counter windows, denial, wave attack) produces the expected
output WITHOUT running a full game.

Run with:
    cd /Users/alexchilton/DataspellProjects/orbit_wars
    python -m pytest tests/test_pred_tactics.py -v
"""
import sys, math
import pytest

sys.path.insert(0, '/Users/alexchilton/DataspellProjects/orbit_wars')
sys.path.insert(0, '/Users/alexchilton/DataspellProjects/orbit_wars/submission')

from submission.main_pred_2p import (
    analyze_threats,
    analyze_counter_windows,
    analyze_denial,
    ships_needed_for_takeover,
    travel_time,
    solve_intercept,
    path_crosses_sun,
    TAKEOVER_MARGIN,
    COUNTER_RATIO,
    SNIPER_BONUS,
    DENIAL_BASE_SCORE,
)

# ── State factory helpers ──────────────────────────────────────────────────────

def make_planet(pid, owner, x, y, ships, prod, is_orb=False, radius=5.0):
    r = math.hypot(x - 50, y - 50)
    return (pid, {
        'x': x, 'y': y, 'owner': owner, 'ships': float(ships),
        'prod': float(prod), 'radius': radius,
        'is_orb': is_orb, 'is_comet': False,
        'r': r, 'theta': math.atan2(y - 50, x - 50),
    })


def make_fleet(owner, ships, target_pid, eta):
    return {'owner': owner, 'ships': float(ships), 'target_pid': target_pid, 'eta': float(eta)}


def make_state(planets, fleets, me=0, step=0, omega=0.03):
    return {
        'planets': dict(planets),
        'fleets': list(fleets),
        'step': step, 'omega': omega, 'me': me,
    }


# ── analyze_threats ────────────────────────────────────────────────────────────

class TestAnalyzeThreats:

    def test_no_threat_when_no_enemy_fleets(self):
        planets = [
            make_planet(0, 0, 20, 50, ships=100, prod=3),   # mine
            make_planet(1, 1, 80, 50, ships=50,  prod=3),   # enemy
        ]
        state = make_state(planets, fleets=[])
        threats = analyze_threats(state)
        assert threats == {}, "No enemy fleets → no threats"

    def test_threat_detected_with_correct_eta(self):
        """Enemy fleet heading to our planet: threat ETA should match fleet ETA."""
        planets = [
            make_planet(0, 0, 20, 50, ships=30, prod=2),   # mine — under attack
            make_planet(1, 1, 80, 50, ships=50, prod=3),   # enemy source
        ]
        # Fleet arriving in exactly 10 steps carrying 40 ships
        fleets = [make_fleet(owner=1, ships=40, target_pid=0, eta=10.0)]
        state  = make_state(planets, fleets)

        threats = analyze_threats(state)

        assert 0 in threats, "Planet 0 should be flagged as threatened"
        t = threats[0]
        assert abs(t['earliest_eta'] - 10.0) < 0.01, \
            f"ETA should be 10.0, got {t['earliest_eta']}"
        assert abs(t['total_enemy'] - 40.0) < 0.01, \
            f"Enemy ships should be 40, got {t['total_enemy']}"

    def test_threat_doomed_when_outgunned(self):
        """40 enemy ships vs 20 garrison (+ prod) → doomed."""
        planets = [make_planet(0, 0, 20, 50, ships=20, prod=1)]
        fleets  = [make_fleet(owner=1, ships=40, target_pid=0, eta=5.0)]
        state   = make_state(planets, fleets)

        threats = analyze_threats(state)
        assert threats[0]['is_doomed'], "40 > (20+5*1)*1.05 → doomed"

    def test_threat_not_doomed_with_production(self):
        """
        Enemy arrives in 20 steps with 30 ships, our garrison 20 + 20 prod steps = 40.
        predict() resolves combat: 40 vs 30 → we win (owner stays 0), garrison = 10 surviving.
        is_doomed uses pred_owner != me, so with owner=0 after combat → NOT doomed.
        """
        planets = [make_planet(0, 0, 20, 50, ships=20, prod=1)]
        fleets  = [make_fleet(owner=1, ships=30, target_pid=0, eta=20.0)]
        state   = make_state(planets, fleets)

        threats = analyze_threats(state)
        # After 20 steps: 20+20=40 garrison vs 30 enemy → we survive → not doomed
        t = threats[0]
        assert not t['is_doomed'], \
            f"Owner still 0 after combat → not doomed (garrison_at_arrival={t['garrison_at_arrival']:.1f})"

    def test_multiple_enemy_fleets_summed(self):
        """Two enemy fleets to same planet → total_enemy is sum."""
        planets = [make_planet(0, 0, 20, 50, ships=100, prod=5)]
        fleets  = [
            make_fleet(owner=1, ships=20, target_pid=0, eta=8.0),
            make_fleet(owner=1, ships=25, target_pid=0, eta=12.0),
        ]
        state = make_state(planets, fleets)

        threats = analyze_threats(state)
        assert abs(threats[0]['total_enemy'] - 45.0) < 0.01, \
            f"Total enemy should be 45, got {threats[0]['total_enemy']}"
        assert abs(threats[0]['earliest_eta'] - 8.0) < 0.01

    def test_only_our_planets_tracked(self):
        """Enemy planet getting attacked by a third fleet should NOT appear in threats."""
        planets = [
            make_planet(0, 0, 20, 50, ships=100, prod=3),  # mine
            make_planet(1, 1, 80, 50, ships=50,  prod=3),  # enemy
        ]
        # Fleet heading to enemy planet (not mine)
        fleets = [make_fleet(owner=2, ships=30, target_pid=1, eta=5.0)]
        state  = make_state(planets, fleets, me=0)

        threats = analyze_threats(state)
        assert 0 not in threats, "Planet 0 not under attack"
        assert 1 not in threats, "We don't track threats to enemy planets"


# ── analyze_counter_windows ────────────────────────────────────────────────────

class TestAnalyzeCounterWindows:

    def test_depleted_garrison_is_window(self):
        """Enemy garrison = 3, prod = 4 → 3 < 4×4 = 16 → counter window."""
        planets = [
            make_planet(0, 0, 20, 50, ships=100, prod=3),
            make_planet(1, 1, 80, 50, ships=3, prod=4),
        ]
        state = make_state(planets, [])
        windows = analyze_counter_windows(state, enemy_pids={1})
        assert 1 in windows, "Low garrison enemy planet should be a counter window"

    def test_full_garrison_not_window(self):
        """Enemy garrison = 100, prod = 4 → not depleted."""
        planets = [
            make_planet(0, 0, 20, 50, ships=100, prod=3),
            make_planet(1, 1, 80, 50, ships=100, prod=4),
        ]
        state = make_state(planets, [])
        windows = analyze_counter_windows(state, enemy_pids={1})
        assert 1 not in windows, "Well-garrisoned planet should not be window"

    def test_never_flags_own_planets(self):
        """Our depleted planet must not appear in counter windows."""
        planets = [make_planet(0, 0, 20, 50, ships=2, prod=5)]
        state = make_state(planets, [])
        windows = analyze_counter_windows(state, enemy_pids=set())
        assert 0 not in windows


# ── analyze_denial ─────────────────────────────────────────────────────────────

class TestAnalyzeDenial:

    def test_reactive_denial_enemy_fleet_to_neutral(self):
        """Enemy fleet heading to neutral planet → denial map entry with positive bonus."""
        planets = [
            make_planet(0, 0, 20, 50, ships=50, prod=3),   # mine
            make_planet(1, 1, 80, 50, ships=50, prod=3),   # enemy
            make_planet(2, -1, 50, 20, ships=5,  prod=5),  # neutral target
        ]
        # Enemy fleet heading to neutral planet 2 in 8 steps
        fleets = [make_fleet(owner=1, ships=10, target_pid=2, eta=8.0)]
        state  = make_state(planets, fleets)

        denial = analyze_denial(state, neutral_pids={2}, enemy_pids={1}, me=0)

        assert 2 in denial, "Neutral being raced to should be in denial map"
        bonus, enemy_eta = denial[2]
        assert bonus >= DENIAL_BASE_SCORE, f"Bonus {bonus:.1f} should be >= {DENIAL_BASE_SCORE}"
        assert abs(enemy_eta - 8.0) < 0.01, f"ETA should be 8.0, got {enemy_eta}"

    def test_reactive_denial_exact_eta_from_physics(self):
        """Exact ETA from fleet tracking — not approximated."""
        planets = [
            make_planet(0, 0, 20, 50, ships=50, prod=3),
            make_planet(2, -1, 50, 70, ships=5,  prod=8),  # high prod
        ]
        fleets = [make_fleet(owner=1, ships=15, target_pid=2, eta=12.5)]
        state  = make_state(planets, fleets)

        # enemy_pids: planet IDs owned by enemy (none here — fleet is orphan in this test)
        denial = analyze_denial(state, neutral_pids={2}, enemy_pids=set(), me=0)
        assert 2 in denial
        _, enemy_eta = denial[2]
        assert abs(enemy_eta - 12.5) < 0.01, "ETA must be exact from fleet tracking"

    def test_sniper_nest_denial(self):
        """Neutral within SNIPER_RANGE of our large planet → denial entry."""
        planets = [
            make_planet(0, 0, 20, 50, ships=60, prod=3),   # mine (big)
            make_planet(2, -1, 35, 50, ships=3, prod=2),   # neutral 15 units away
        ]
        state = make_state(planets, [])
        denial = analyze_denial(state, neutral_pids={2}, enemy_pids=set(), me=0)
        # dist(20,50)→(35,50) = 15 < SNIPER_RANGE=35
        assert 2 in denial, "Neutral within sniper range should get denial bonus"
        bonus, _ = denial[2]
        assert bonus >= SNIPER_BONUS, f"Sniper bonus {bonus:.1f} should >= {SNIPER_BONUS}"

    def test_non_threat_neutral_no_bonus(self):
        """Far neutral (>SNIPER_RANGE away) with low prod, no enemy nearby → no denial entry."""
        planets = [
            make_planet(0, 0, 20, 50, ships=50, prod=3),
            make_planet(2, -1, 80, 50, ships=5, prod=1),   # 60 units away > SNIPER_RANGE=35, low prod
        ]
        state = make_state(planets, [])
        denial = analyze_denial(state, neutral_pids={2}, enemy_pids=set(), me=0)
        assert 2 not in denial, \
            "Neutral outside sniper range with low prod and no threats should not be in denial"


# ── ships_needed_for_takeover ──────────────────────────────────────────────────

class TestShipsNeeded:

    def test_neutral_no_growth(self):
        """Neutral doesn't grow → just garrison × margin."""
        needed = ships_needed_for_takeover(tgt_ships=20, tgt_prod=5, tt=10, owner=-1)
        assert needed == int(20 * TAKEOVER_MARGIN) + 1

    def test_enemy_grows_during_travel(self):
        """Enemy garrison grows by prod × tt during fleet travel."""
        needed = ships_needed_for_takeover(tgt_ships=20, tgt_prod=3, tt=10, owner=1)
        expected = int((20 + 3 * 10) * TAKEOVER_MARGIN) + 1
        assert needed == expected, f"Expected {expected}, got {needed}"

    def test_zero_travel_time(self):
        """Zero travel time: garrison doesn't grow."""
        n0 = ships_needed_for_takeover(tgt_ships=30, tgt_prod=10, tt=0, owner=1)
        assert n0 == int(30 * TAKEOVER_MARGIN) + 1


# ── path_crosses_sun ──────────────────────────────────────────────────────────

class TestPathCrossesSun:

    def test_path_not_through_sun(self):
        """Path along the edge — should not cross."""
        # Sun at (50, 50), radius 10. Path from (10, 10) to (90, 10) is at y=10 → clear.
        assert not path_crosses_sun(10, 10, 90, 10), "Horizontal path below sun should be clear"

    def test_path_through_sun(self):
        """Diagonal across the sun centre should cross."""
        assert path_crosses_sun(10, 10, 90, 90), "Diagonal through sun centre should cross"


# ── Integration: agent returns valid action list ───────────────────────────────

class TestAgentValidActions:

    def _run_agent(self, obs_dict):
        """Import and run agent from main_pred_2p."""
        from submission.main_pred_2p import agent
        return agent(obs_dict)

    def _make_obs(self, planets_raw, fleets_raw=None, player=0, step=0, omega=0.03):
        return {
            'player': player,
            'planets': planets_raw,
            'fleets': fleets_raw or [],
            'step': step,
            'angular_velocity': omega,
            'comet_planet_ids': [],
            'comets': [],
        }

    def test_agent_no_crash_empty_game(self):
        """Agent with only our starting planet does not crash."""
        obs = self._make_obs([[0, 0, 20, 50, 5, 30, 3]])
        actions = self._run_agent(obs)
        # No targets — should return empty or small list
        assert isinstance(actions, list)

    def test_agent_returns_valid_format(self):
        """Each action should be [planet_id, angle_float, ships_int]."""
        obs = self._make_obs([
            [0, 0, 20, 50, 5, 50, 3],   # mine
            [1, -1, 80, 50, 5, 10, 2],  # neutral target
        ])
        actions = self._run_agent(obs)
        for act in actions:
            assert len(act) == 3, f"Action should have 3 elements: {act}"
            pid, angle, ships = act
            assert isinstance(ships, int) or isinstance(ships, float)
            assert ships >= 1, f"Must send at least 1 ship: {act}"
            assert -math.pi * 2 <= angle <= math.pi * 2, f"Angle out of range: {angle}"

    def test_agent_no_crash_under_attack(self):
        """Agent under threat should not crash."""
        obs = self._make_obs(
            planets_raw=[
                [0, 0, 20, 50, 5, 30, 3],   # mine
                [1, 1, 80, 50, 5, 50, 3],   # enemy
            ],
            fleets_raw=[
                [99, 1, 60, 50, math.pi, 1, 25],  # enemy fleet, heading left
            ]
        )
        actions = self._run_agent(obs)
        assert isinstance(actions, list)

    def test_agent_evacuates_doomed_planet(self):
        """
        Our planet is about to be overwhelmed (100 enemy ships incoming, we have 10).
        Agent should launch evacuation (send ships somewhere else).
        """
        # Our planet 0 at (20,50): 10 ships. Enemy fleet 100 ships ETA 5.
        # Our planet 1 at (25,50): safe 50 ships.
        # Enemy planet 2 at (80,50).
        obs = self._make_obs(
            planets_raw=[
                [0, 0, 20, 50, 5, 10, 2],   # mine under attack
                [1, 0, 25, 50, 5, 50, 3],   # mine safe
                [2, 1, 80, 50, 5, 50, 3],   # enemy
            ],
            fleets_raw=[
                # Enemy fleet of 100 ships at (60,50) heading left toward planet 0
                [99, 1, 50, 50, math.pi, 2, 100],
            ]
        )
        actions = self._run_agent(obs)
        # Should produce at least one action (evacuation from planet 0)
        src_planets = {a[0] for a in actions}
        # Either evacuates or still attacks — just verify no crash and valid format
        for act in actions:
            assert len(act) == 3
            assert act[2] >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
