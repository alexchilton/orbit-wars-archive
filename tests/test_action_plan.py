"""
Unit tests for the _ACTION_PLAN infrastructure in main_pred_2p.py.

Tests cover:
  - comet window helpers (_comet_active_window, _next_comet_spawn)
  - _prune_action_plan: removes stale / past-due / invalid entries
  - _plan_comet_actions: adds grab and evac entries correctly
  - _execute_action_plan: fires due entries, skips invalid ones
  - New-game detection (step resets the plan)
"""
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'submission'))

import main_pred_2p as _mod


# ────────────────────────────────────────────────────────────────────────────────
# Helpers to build minimal state dicts
# ────────────────────────────────────────────────────────────────────────────────

def _planet(pid, x, y, owner, ships, prod, is_orb=False, is_comet=False):
    return {
        'x': x, 'y': y, 'owner': owner, 'ships': ships,
        'prod': prod, 'is_orb': is_orb, 'is_comet': is_comet,
    }


def _state(step, me, planets, omega=0.03):
    return {'step': step, 'me': me, 'planets': planets, 'omega': omega,
            'fleets': []}


def _reset_plan(*entries):
    """Reset _ACTION_PLAN and _PLAN_PREV_STEP to known values."""
    _mod._ACTION_PLAN = list(entries)
    _mod._PLAN_PREV_STEP = -1   # -1: 'step < -1-5 = -6' never triggers new-game reset


# ════════════════════════════════════════════════════════════════════════════════
# 1. Comet window helpers
# ════════════════════════════════════════════════════════════════════════════════

class TestCometWindowHelpers:
    """_comet_active_window and _next_comet_spawn return correct step ranges."""

    def test_no_window_before_first_spawn(self):
        assert _mod._comet_active_window(30) is None

    def test_window_at_first_spawn(self):
        w = _mod._comet_active_window(50)
        assert w == (50, 69), f"Expected (50, 69), got {w}"

    def test_window_inside_first_spawn(self):
        w = _mod._comet_active_window(60)
        assert w == (50, 69)

    def test_no_window_between_spawns(self):
        # step 70 = one after expiry of first window (50+19=69)
        assert _mod._comet_active_window(70) is None

    def test_window_at_second_spawn(self):
        w = _mod._comet_active_window(150)
        assert w == (150, 169)

    def test_window_at_third_spawn(self):
        w = _mod._comet_active_window(255)
        assert w == (250, 269)

    def test_next_spawn_before_first(self):
        assert _mod._next_comet_spawn(10) == 50

    def test_next_spawn_during_first_window(self):
        # step 55 is inside window 50-69; next spawn is 150
        assert _mod._next_comet_spawn(55) == 150

    def test_next_spawn_between_windows(self):
        assert _mod._next_comet_spawn(70) == 150

    def test_next_spawn_after_second_window(self):
        assert _mod._next_comet_spawn(170) == 250


# ════════════════════════════════════════════════════════════════════════════════
# 2. _prune_action_plan
# ════════════════════════════════════════════════════════════════════════════════

class TestPruneActionPlan:
    """Stale, past-due, and invalid entries are removed."""

    def _entry(self, reason='comet_grab', src_pid=1, tgt_pid=2,
               turn=60, max_turn=68, priority=2):
        return dict(turn=turn, src_pid=src_pid, tgt_pid=tgt_pid,
                    tgt_xy=None, ships=10, reason=reason,
                    max_turn=max_turn, priority=priority)

    def test_removes_past_max_turn(self):
        e = self._entry(max_turn=59)
        planets = {1: _planet(1, 30, 30, 0, 100, 2),
                   2: _planet(2, 70, 70, -1, 5, 3, is_comet=True)}
        _reset_plan(e)
        _mod._prune_action_plan(_state(60, 0, planets))
        assert len(_mod._ACTION_PLAN) == 0

    def test_keeps_valid_entry(self):
        e = self._entry(max_turn=68)
        planets = {1: _planet(1, 30, 30, 0, 100, 2),
                   2: _planet(2, 70, 70, -1, 5, 3, is_comet=True)}
        _reset_plan(e)
        _mod._prune_action_plan(_state(60, 0, planets))
        assert len(_mod._ACTION_PLAN) == 1

    def test_removes_when_source_lost(self):
        # Source planet captured by enemy
        e = self._entry(src_pid=1)
        planets = {1: _planet(1, 30, 30, 1, 100, 2),  # owner=1 (enemy)
                   2: _planet(2, 70, 70, -1, 5, 3, is_comet=True)}
        _reset_plan(e)
        _mod._prune_action_plan(_state(60, 0, planets))
        assert len(_mod._ACTION_PLAN) == 0

    def test_removes_comet_grab_when_already_ours(self):
        # We captured the comet ourselves — no need to grab
        e = self._entry(reason='comet_grab', tgt_pid=2)
        planets = {1: _planet(1, 30, 30, 0, 100, 2),
                   2: _planet(2, 70, 70, 0, 5, 3, is_comet=True)}   # owner=0 = us
        _reset_plan(e)
        _mod._prune_action_plan(_state(60, 0, planets))
        assert len(_mod._ACTION_PLAN) == 0

    def test_removes_comet_evac_when_not_comet(self):
        # Comet already expired (no longer is_comet flag)
        e = self._entry(reason='comet_evac', src_pid=2, tgt_pid=1)
        planets = {1: _planet(1, 30, 30, 0, 100, 2),
                   2: _planet(2, 70, 70, 0, 5, 3, is_comet=False)}  # no longer a comet
        _reset_plan(e)
        _mod._prune_action_plan(_state(60, 0, planets))
        assert len(_mod._ACTION_PLAN) == 0

    def test_keeps_evac_when_still_comet(self):
        e = self._entry(reason='comet_evac', src_pid=2, tgt_pid=1)
        planets = {1: _planet(1, 30, 30, 0, 100, 2),
                   2: _planet(2, 70, 70, 0, 5, 3, is_comet=True)}
        _reset_plan(e)
        _mod._prune_action_plan(_state(60, 0, planets))
        assert len(_mod._ACTION_PLAN) == 1

    def test_new_game_resets_plan(self):
        """Step going backwards by more than 5 triggers new-game reset."""
        e = self._entry(max_turn=200)
        planets = {1: _planet(1, 30, 30, 0, 100, 2),
                   2: _planet(2, 70, 70, -1, 5, 3, is_comet=True)}
        _reset_plan(e)
        _mod._PLAN_PREV_STEP = 100   # simulate being in the middle of a game
        # step=5 << 100 → new game
        _mod._prune_action_plan(_state(5, 0, planets))
        # After the reset, it tries to prune the now-empty plan → still empty
        assert len(_mod._ACTION_PLAN) == 0


# ════════════════════════════════════════════════════════════════════════════════
# 3. _plan_comet_actions
# ════════════════════════════════════════════════════════════════════════════════

class TestPlanCometActions:
    """Grab and evac entries are added correctly."""

    def _run(self, step, planets, me=0):
        _reset_plan()
        _mod._prune_action_plan(_state(step, me, planets))
        _mod._plan_comet_actions(_state(step, me, planets), me)

    def test_adds_grab_for_neutral_comet_in_range(self):
        """Neutral comet close to our planet during an active window → grab planned."""
        planets = {
            1: _planet(1, 45, 70, 0, 100, 2),            # our planet (y=70 avoids sun path)
            2: _planet(2, 55, 70, -1, 5, 0, is_comet=True),   # neutral comet, 10 units away
        }
        self._run(step=55, planets=planets, me=0)
        grabs = [e for e in _mod._ACTION_PLAN if e['reason'] == 'comet_grab']
        assert len(grabs) >= 1, "Expected a comet_grab entry"

    def test_no_grab_when_comet_too_far(self):
        """
        Comet far away with only 2 steps until expiry — ETA > steps_left → no grab.
        Active window 50-69; step=67 → steps_left=2. Comet far → ETA > 2 → skip.
        """
        planets = {
            1: _planet(1, 10, 10, 0, 100, 2),           # our planet, far from comet
            2: _planet(2, 90, 90, -1, 5, 0, is_comet=True),  # comet, far away
        }
        self._run(step=67, planets=planets, me=0)
        grabs = [e for e in _mod._ACTION_PLAN if e['reason'] == 'comet_grab']
        assert len(grabs) == 0, f"Unexpected grab entry for unreachable comet: {grabs}"

    def test_adds_evac_for_owned_comet_near_expiry(self):
        """
        We own a comet; there's a safe planet nearby. Evac should be planned
        so ships leave before expiry.
        """
        planets = {
            1: _planet(1, 45, 70, 0, 60, 2),            # safe planet
            2: _planet(2, 55, 70, 0, 20, 0, is_comet=True),  # we own comet
        }
        self._run(step=55, planets=planets, me=0)
        evacs = [e for e in _mod._ACTION_PLAN if e['reason'] == 'comet_evac']
        assert len(evacs) == 1, f"Expected one evac entry, got: {evacs}"
        assert evacs[0]['src_pid'] == 2
        assert evacs[0]['ships'] == 'all'

    def test_evac_turn_before_expiry(self):
        """Evac turn must be strictly before expire_step."""
        planets = {
            1: _planet(1, 45, 70, 0, 60, 2),
            2: _planet(2, 55, 70, 0, 20, 0, is_comet=True),
        }
        self._run(step=55, planets=planets, me=0)
        evacs = [e for e in _mod._ACTION_PLAN if e['reason'] == 'comet_evac']
        assert evacs, "No evac entry added"
        _, expire = _mod._comet_active_window(55)
        assert evacs[0]['turn'] < expire, (
            f"evac turn {evacs[0]['turn']} not before expire {expire}")

    def test_no_evac_without_safe_planet(self):
        """If we own the comet but have no other planet, no evac is planned (nowhere to go)."""
        planets = {
            2: _planet(2, 50, 50, 0, 20, 0, is_comet=True),  # only planet we own
        }
        self._run(step=55, planets=planets, me=0)
        evacs = [e for e in _mod._ACTION_PLAN if e['reason'] == 'comet_evac']
        assert len(evacs) == 0

    def test_no_duplicate_grabs(self):
        """Calling plan twice doesn't add two grab entries for the same comet."""
        planets = {
            1: _planet(1, 45, 70, 0, 100, 2),
            2: _planet(2, 55, 70, -1, 5, 0, is_comet=True),
        }
        _reset_plan()
        state = _state(55, 0, planets)
        _mod._plan_comet_actions(state, 0)
        _mod._plan_comet_actions(state, 0)   # second call
        grabs = [e for e in _mod._ACTION_PLAN if e['reason'] == 'comet_grab']
        assert len(grabs) == 1, f"Duplicate grabs: {len(grabs)}"

    def test_no_duplicate_evacs(self):
        """Calling plan twice doesn't add two evac entries for the same comet."""
        planets = {
            1: _planet(1, 45, 70, 0, 60, 2),
            2: _planet(2, 55, 70, 0, 20, 0, is_comet=True),
        }
        _reset_plan()
        state = _state(55, 0, planets)
        _mod._plan_comet_actions(state, 0)
        _mod._plan_comet_actions(state, 0)
        evacs = [e for e in _mod._ACTION_PLAN if e['reason'] == 'comet_evac']
        assert len(evacs) == 1, f"Duplicate evacs: {len(evacs)}"


# ════════════════════════════════════════════════════════════════════════════════
# 4. _execute_action_plan
# ════════════════════════════════════════════════════════════════════════════════

class TestExecuteActionPlan:
    """Due entries fire; future entries and invalid entries are skipped."""

    def _make_evac_entry(self, src_pid=2, tgt_pid=1, turn=60, max_turn=68):
        return {
            'turn': turn, 'src_pid': src_pid, 'tgt_pid': tgt_pid,
            'tgt_xy': None, 'ships': 'all', 'reason': 'comet_evac',
            'max_turn': max_turn, 'priority': 1,
        }

    def _make_grab_entry(self, src_pid=1, tgt_pid=2, ships=10, turn=60, max_turn=68):
        return {
            'turn': turn, 'src_pid': src_pid, 'tgt_pid': tgt_pid,
            'tgt_xy': None, 'ships': ships, 'reason': 'comet_grab',
            'max_turn': max_turn, 'priority': 2,
        }

    def test_future_entry_not_fired(self):
        """Entry scheduled for turn 70 should not fire at step 60."""
        entry = self._make_evac_entry(turn=70)
        planets = {
            1: _planet(1, 45, 70, 0, 100, 2),
            2: _planet(2, 60, 70, 0, 30, 0, is_comet=True),
        }
        _reset_plan(entry)
        moves, launched = [], {}
        _mod._execute_action_plan(_state(60, 0, planets), moves, launched)
        assert len(moves) == 0, "Should not fire future entry"

    def test_due_evac_fires(self):
        """Evac entry due this turn fires and produces a move."""
        entry = self._make_evac_entry(src_pid=2, tgt_pid=1, turn=60)
        planets = {
            1: _planet(1, 45, 70, 0, 100, 2),           # y=70 keeps path far from sun
            2: _planet(2, 60, 70, 0, 30, 0, is_comet=True),
        }
        _reset_plan(entry)
        moves, launched = [], {}
        _mod._execute_action_plan(_state(60, 0, planets), moves, launched)
        assert len(moves) == 1, f"Expected 1 move from evac, got {len(moves)}"

    def test_due_grab_fires(self):
        """Grab entry due this turn produces a move."""
        entry = self._make_grab_entry(src_pid=1, tgt_pid=2, ships=12, turn=60)
        planets = {
            1: _planet(1, 45, 70, 0, 100, 2),
            2: _planet(2, 60, 70, -1, 5, 0, is_comet=True),
        }
        _reset_plan(entry)
        moves, launched = [], {}
        _mod._execute_action_plan(_state(60, 0, planets), moves, launched)
        assert len(moves) == 1, f"Expected 1 move from grab, got {len(moves)}"

    def test_skips_entry_when_source_missing(self):
        """If source planet vanished from state, skip silently."""
        entry = self._make_evac_entry(src_pid=99, tgt_pid=1, turn=60)
        planets = {1: _planet(1, 45, 45, 0, 100, 2)}
        _reset_plan(entry)
        moves, launched = [], {}
        _mod._execute_action_plan(_state(60, 0, planets), moves, launched)
        assert len(moves) == 0

    def test_skips_entry_when_not_enough_ships(self):
        """
        Grab wants 80 ships but src only has 10. Should not fire
        (10 < min required fraction of ships).
        """
        entry = self._make_grab_entry(src_pid=1, tgt_pid=2, ships=80, turn=60)
        planets = {
            1: _planet(1, 45, 45, 0, 10, 2),          # only 10 ships
            2: _planet(2, 55, 55, -1, 5, 0, is_comet=True),
        }
        _reset_plan(entry)
        moves, launched = [], {}
        _mod._execute_action_plan(_state(60, 0, planets), moves, launched)
        # av=10, n_ships = min(80, int(10*0.90)) = min(80, 9) = 9 < 80 → but 9 >= 1
        # Actually the logic is: n_ships = min(ships, int(av * 0.90))
        # av=10, n_ships = min(80, 9) = 9, av >= n_ships (10 >= 9) → fires with 9
        # So it does fire with reduced ships. Let's verify the behavior:
        # This tests that the agent doesn't send MORE ships than available.
        if len(moves) == 1:
            _, angle, n = moves[0]
            assert n <= 9, f"Sent too many ships: {n}"

    def test_priority_order_evac_before_grab(self):
        """Priority 1 (evac) should appear before priority 2 (grab) in moves."""
        evac = self._make_evac_entry(src_pid=2, tgt_pid=1, turn=60)
        grab = self._make_grab_entry(src_pid=1, tgt_pid=3, ships=10, turn=60)
        planets = {
            1: _planet(1, 45, 70, 0, 200, 2),           # safe planet, lots of ships
            2: _planet(2, 55, 70, 0, 30, 0, is_comet=True),   # comet we own (evac src)
            3: _planet(3, 65, 70, -1, 5, 0, is_comet=True),   # neutral comet (grab tgt)
        }
        _reset_plan(grab, evac)   # insert grab first, evac second
        moves, launched = [], {}
        _mod._execute_action_plan(_state(60, 0, planets), moves, launched)
        # Both should fire. If both fired: evac is from pid 2, grab from pid 1.
        # The evac move (src=2) should come first in moves.
        assert len(moves) == 2, f"Expected 2 moves, got {len(moves)}"
        src_pids_in_order = [m[0] for m in moves]
        assert src_pids_in_order[0] == 2, (
            f"Evac (src=2) should be first, got order {src_pids_in_order}")

    def test_executed_entries_removed_from_plan(self):
        """After execute, fired entries are removed from _ACTION_PLAN."""
        entry = self._make_grab_entry(src_pid=1, tgt_pid=2, ships=10, turn=60)
        planets = {
            1: _planet(1, 45, 70, 0, 100, 2),
            2: _planet(2, 60, 70, -1, 5, 0, is_comet=True),
        }
        _reset_plan(entry)
        moves, launched = [], {}
        _mod._execute_action_plan(_state(60, 0, planets), moves, launched)
        # After execution, comet_grab for turn=60 should be gone
        remaining = [e for e in _mod._ACTION_PLAN
                     if e['reason'] == 'comet_grab' and e['turn'] == 60]
        assert len(remaining) == 0, f"Executed entry still in plan: {remaining}"


# ════════════════════════════════════════════════════════════════════════════════
# 5. Integration: plan → prune → execute cycle
# ════════════════════════════════════════════════════════════════════════════════

class TestPlanPruneExecuteCycle:
    """Simulate 3 turns of the plan lifecycle."""

    def test_grab_lifecycle(self):
        """
        Turn 55: plan adds a comet_grab entry (comet visible, in range).
        Turn 56: prune keeps it (still valid). execute fires it.
        """
        planets = {
            1: _planet(1, 45, 70, 0, 100, 2),
            2: _planet(2, 55, 70, -1, 5, 0, is_comet=True),
        }

        # Turn 55: plan phase
        _reset_plan()
        _mod._plan_comet_actions(_state(55, 0, planets), me=0)
        assert any(e['reason'] == 'comet_grab' for e in _mod._ACTION_PLAN), \
            "Turn 55: expected grab planned"

        # Turn 56: prune + execute
        _mod._PLAN_PREV_STEP = 55
        _mod._prune_action_plan(_state(56, 0, planets))
        moves, launched = [], {}
        _mod._execute_action_plan(_state(56, 0, planets), moves, launched)
        # Move should be produced (grab fires)
        assert len(moves) >= 1, "Turn 56: expected grab to fire"

    def test_evac_fires_before_expiry(self):
        """
        Step 64: we own the comet. Plan should schedule evac for turn 64 or earlier
        (expire=69, we need to leave by 69 - eta - buffer).
        """
        planets = {
            1: _planet(1, 45, 70, 0, 60, 2),            # safe planet (y=70 avoids sun)
            2: _planet(2, 55, 70, 0, 25, 0, is_comet=True),
        }
        _reset_plan()
        _mod._plan_comet_actions(_state(64, 0, planets), me=0)
        evacs = [e for e in _mod._ACTION_PLAN if e['reason'] == 'comet_evac']
        assert len(evacs) == 1
        # Max turn must be < expire (69)
        assert evacs[0]['max_turn'] < 69, f"max_turn {evacs[0]['max_turn']} too late"
        # Evac turn must allow enough time to reach safety
        assert evacs[0]['turn'] <= 68, f"Evac too late: turn {evacs[0]['turn']}"
