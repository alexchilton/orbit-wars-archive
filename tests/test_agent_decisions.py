"""
tests/test_agent_decisions.py — Strategic correctness tests for the agent() function.

These tests call agent() with hand-crafted game states and assert that the
decision process makes the RIGHT choice — not just that it doesn't crash.

Planet layout: sun at (50, 50) radius 10.
All planets placed so (r + radius) > 48 → static, non-orbiting.
  - Planet 0 (ours):      (0,  50)  r=50  dist from sun 50
  - Planet 1 (close tgt): (0,  75)  r≈56  dist 25 from planet 0  → ETA ≈ 5 steps
  - Planet 2 (far   tgt): (0, -100) r≈158 dist 150 from planet 0 → ETA ≈ 29 steps (> SIM_HORIZON=25)

Run:
    cd /Users/alexchilton/DataspellProjects/orbit_wars
    python -m pytest tests/test_agent_decisions.py -v
"""
import sys, math
import pytest

sys.path.insert(0, '/Users/alexchilton/DataspellProjects/orbit_wars')
sys.path.insert(0, '/Users/alexchilton/DataspellProjects/orbit_wars/submission')

import main_pred_2p as _mod
from main_pred_2p import agent, travel_time, fleet_speed


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_obs(planets_raw, fleets_raw=None, player=0, step=0, omega=0.03):
    return {
        'player':             player,
        'planets':            planets_raw,
        'fleets':             fleets_raw or [],
        'step':               step,
        'angular_velocity':   omega,
        'comet_planet_ids':   [],
        'comets':             [],
    }


# planet format: [id, owner, x, y, radius, ships, prod]
MY_PLANET     = [0,  0, 0,   50, 5, 50, 3]   # ours at (0,50), 50 ships
CLOSE_NEUTRAL = [1, -1, 0,   75, 5,  5, 3]   # dist 25 from ours
FAR_NEUTRAL   = [2, -1, 0, -100, 5,  5, 3]   # dist 150 from ours
ENEMY_PLANET  = [1,  1, 0,   75, 5, 10, 2]   # enemy at same close position


@pytest.fixture(autouse=True)
def clear_registry():
    _mod._FLEET_REGISTRY.clear()
    _mod._REGISTRY_PREV_STEP = 0
    yield
    _mod._FLEET_REGISTRY.clear()


# ── sanity: verify test geometry ──────────────────────────────────────────────

class TestGeometrySanity:
    """Verify that our test planets have the ETAs we assume."""

    def test_close_eta_inside_sim_horizon(self):
        eta = travel_time(0, 50, 0, 75, 20)
        assert eta < _mod._SIM_HORIZON, f"Close target ETA={eta:.1f} should be inside horizon={_mod._SIM_HORIZON}"

    def test_far_eta_outside_sim_horizon(self):
        eta = travel_time(0, 50, 0, -100, 20)
        assert eta > _mod._SIM_HORIZON, f"Far target ETA={eta:.1f} should exceed horizon={_mod._SIM_HORIZON}"


# ── core decision tests ───────────────────────────────────────────────────────

class TestAgentTargeting:

    def test_agent_fires_at_close_neutral(self):
        """Agent with a nearby neutral should send at least one fleet."""
        obs = _make_obs([MY_PLANET, CLOSE_NEUTRAL])
        actions = agent(obs)
        assert len(actions) > 0, "Agent should fire when there is a reachable neutral"
        assert actions[0][2] >= 1, "Must send at least 1 ship"

    def test_agent_fires_to_close_not_far_neutral(self):
        """
        Two neutrals: one inside SIM_HORIZON, one beyond.
        Agent must choose the close one — the far one yields ≤ 0 sim_delta
        and is blocked by the baseline guard.
        """
        obs = _make_obs([MY_PLANET, CLOSE_NEUTRAL, FAR_NEUTRAL])
        actions = agent(obs)

        assert len(actions) > 0, "Agent should fire at the close neutral"
        # The only target that should be registered is planet 1 (close)
        fired_targets = {f['target_pid'] for f in _mod._FLEET_REGISTRY}
        assert 1 in fired_targets, f"Should target planet 1 (close), registry={fired_targets}"
        assert 2 not in fired_targets, f"Should NOT target planet 2 (far, beyond horizon), registry={fired_targets}"

    def test_agent_attacks_enemy_when_advantaged(self):
        """
        We have 10× the ships of the enemy → ship_ratio > 1.5 → enemy in tgt_set.
        Agent should attack the enemy planet.
        """
        my_big = [0, 0, 0, 50, 5, 100, 5]   # ours: 100 ships, prod 5
        obs = _make_obs([my_big, ENEMY_PLANET])
        actions = agent(obs)

        assert len(actions) > 0, "Agent should fire when decisively ahead"
        fired_targets = {f['target_pid'] for f in _mod._FLEET_REGISTRY}
        assert 1 in fired_targets, f"Should target the enemy planet, registry={fired_targets}"

    def test_agent_attacks_enemy_after_early_game(self):
        """Step > 30 → enemy always added to tgt_set regardless of ratio."""
        obs = _make_obs([MY_PLANET, ENEMY_PLANET], step=35)
        actions = agent(obs)

        assert len(actions) > 0, "Agent should fire after early game"
        fired_targets = {f['target_pid'] for f in _mod._FLEET_REGISTRY}
        assert 1 in fired_targets, f"Should target the enemy planet at step 35, registry={fired_targets}"

    def test_no_double_send_to_already_targeted_planet(self):
        """
        Registry says we already have a fleet en route to planet 1 (close neutral).
        The only remaining open target is planet 2 (far neutral with real production
        value). Agent should NOT double-send to planet 1, but may send to planet 2
        if the marginal gain is positive — which is correct now that _rough_roi
        correctly values far targets.
        """
        # Pre-populate registry: fleet launched 1 step ago, ETA=6, still flying
        _mod._FLEET_REGISTRY.append({
            'target_pid': 1, 'src_pid': 0,
            'launch_step': -1, 'eta': 6.0, 'ships': 6,
        })
        _mod._REGISTRY_PREV_STEP = 0

        obs = _make_obs([MY_PLANET, CLOSE_NEUTRAL, FAR_NEUTRAL])
        agent(obs)

        newly_fired = [f['target_pid'] for f in _mod._FLEET_REGISTRY
                       if f['launch_step'] == 0]   # launched THIS step
        # The only strict invariant: never double-send to an already-targeted planet
        assert 1 not in newly_fired, "Should not double-send to already-targeted planet 1"

    def test_no_fire_at_zero_prod_far_neutral(self):
        """
        A far neutral with zero production should have marginal gain ≤ 0
        (can't earn back the ships used to capture it) → baseline guard blocks it.
        """
        zero_prod_far = [2, -1, 0, -100, 5, 1, 0]   # prod=0, far, needs 2 ships
        obs = _make_obs([MY_PLANET, zero_prod_far])
        actions = agent(obs)

        fired_targets = {f['target_pid'] for f in _mod._FLEET_REGISTRY}
        assert 2 not in fired_targets, \
            f"Should not target zero-prod far planet, registry={fired_targets}"

    def test_agent_does_not_send_to_own_planet(self):
        """Sanity: agent should never target its own planet."""
        my_planet_2 = [2, 0, 0, -100, 5, 30, 2]   # second friendly planet (far)
        obs = _make_obs([MY_PLANET, CLOSE_NEUTRAL, my_planet_2])
        actions = agent(obs)

        # Fleet registry should only contain non-me targets
        for f in _mod._FLEET_REGISTRY:
            assert f['target_pid'] != 0, "Should not target own planet 0"

    def test_avail_threshold_blocks_small_planet(self):
        """
        Planet with only 8 ships is below the av<10 threshold → should not fire
        even with a close neutral available.
        """
        tiny = [0, 0, 0, 50, 5, 8, 3]   # only 8 ships
        obs = _make_obs([tiny, CLOSE_NEUTRAL])
        actions = agent(obs)
        # 8 < 10 → beam search skips this source
        assert len(actions) == 0, \
            f"Planet with 8 ships (< threshold 10) should not fire, got {actions}"
