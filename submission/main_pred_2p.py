"""
main_pred_2p.py — Predictive 2-player agent with physics beam search.

Decision engine uses 1-ply forward simulation via physics_sim.predict()
to score every candidate (source → target) action over a fixed 50-step horizon.
All targets use the SAME horizon so nearby captures (ETA≈2, 48 production steps)
correctly dominate speculative long-range attacks (ETA≈40, only 10 steps).

Architecture:
  1. PHYSICS STATE   — parse_obs + fleet tracking gives exact state each step.
  2. THREAT TRIAGE   — exact ETA + garrison projection → evacuate or reinforce.
  3. BEAM SEARCH     — for every candidate action inject a fleet, simulate 50
                       steps, score resulting state by (prod_advantage × horizon
                       + ship_advantage + in-transit capture value), execute best.
  4. TACTICAL HINTS  — denial urgency, counter windows, wave synchrony add bonus
                       score on top of the simulation so physics drives the choice
                       but tactical context can break ties.
  5. WAVE ATTACK     — second source guaranteed to send to wave targets.

Action format: [planet_id, angle_radians, ships]  (same as denial agent)
"""

import os, sys, math
import numpy as np
import pickle

# ── Physics engine ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from physics_sim import (
    parse_obs, predict, copy_state,
    fleet_speed, travel_time, predict_orbit,
    SUN_X, SUN_Y, SUN_RADIUS, INNER_ORBIT_THRESHOLD,
)

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_SPEED          = 6.0
_MAX_SPEED_MINUS_1 = MAX_SPEED - 1.0
_LOG1000           = math.log(1000.0)

# Defensive / takeover
TAKEOVER_MARGIN     = 1.05   # ships needed = (garrison + growth) × this
DEFENSE_MARGIN      = 1.05   # enemy × this > garrison → doomed
EVAC_FRACTION       = 0.70   # fraction of ships to evacuate
REINFORCE_BUFFER    = 1.5    # arrive this many steps before enemy fleet
COUNTER_RATIO       = 4.0    # garrison < this × prod → counter window

# Tactical hint bonuses (added to sim score to break ties)
DENIAL_BASE_SCORE   = 75.0
DENIAL_PROD_WEIGHT  = 8.0
DENIAL_URGENCY_WT   = 20.0
SNIPER_RANGE        = 35.0
SNIPER_MIN_SHIPS    = 40
SNIPER_BONUS        = 60.0
HIGH_PROD_THRESH    = 4.0
PROACTIVE_BONUS     = 15.0
COUNTER_BONUS       = 50.0

# Beam search parameters
_SIM_HORIZON     = 50   # fixed horizon for all beam search sims — near targets
                        # accumulate more production steps than far ones, so near
                        # correctly wins when production value is otherwise equal
_PROD_HORIZON    = 40   # production valuation window used in state scoring
_MAX_SOURCES     = 6    # top source planets evaluated per step
_MAX_TARGETS     = 8    # top targets pre-filtered by rough ROI per source
_SMASH_RADIUS_SQ = 2500  # 50-unit radius for smash geometry check
_ENEMY_PROD_CREDIT = 1.0  # multiplier: enemy prod counts double (our gain + their loss)
_ENEMY_SHIP_CREDIT = 0.4  # ships enemy burns defending add to our ROI estimate

# ── Stateful fleet registry ────────────────────────────────────────────────────
# We track every fleet WE launch with exact (target_pid, eta, launch_step).
# This survives across agent() calls and is immune to infer_fleet_dest failures.
# Format: {'target_pid': int, 'src_pid': int, 'launch_step': int, 'eta': float, 'ships': int}
_FLEET_REGISTRY: list = []
_REGISTRY_PREV_STEP: int = -1  # detect new game (step goes backwards)


def _prune_registry(current_step: int) -> None:
    """Remove arrived fleets and reset entirely on new-game detection."""
    global _FLEET_REGISTRY, _REGISTRY_PREV_STEP
    # New game: step went backwards (tournament re-uses same module instance)
    if current_step < _REGISTRY_PREV_STEP - 5:
        _FLEET_REGISTRY = []
    _REGISTRY_PREV_STEP = current_step
    _FLEET_REGISTRY = [f for f in _FLEET_REGISTRY
                       if f['launch_step'] + f['eta'] > current_step - 1]


def _registry_inflight_targets(current_step: int) -> set:
    """Return target PIDs that our registry says are still in flight."""
    return {f['target_pid'] for f in _FLEET_REGISTRY
            if f['launch_step'] + f['eta'] > current_step}


def _inject_registry_fleets(state: dict, me: int, current_step: int) -> None:
    """
    Add any fleet in our registry that infer_fleet_dest missed (returned None or
    assigned wrong planet).  Prevents double-counting: skip if target_pid already
    present in state['fleets'] for owner == me.
    """
    present = {fl['target_pid'] for fl in state['fleets'] if fl['owner'] == me}
    for f in _FLEET_REGISTRY:
        if f['target_pid'] in present:
            continue
        remaining_eta = f['launch_step'] + f['eta'] - current_step
        if remaining_eta > 0.5:
            state['fleets'].append({
                'owner':      me,
                'ships':      f['ships'],
                'target_pid': f['target_pid'],
                'eta':        remaining_eta,
            })
            present.add(f['target_pid'])

# ── Comet constants ────────────────────────────────────────────────────────────
_COMET_FIRST_SPAWN    = 50     # step of first comet group spawn
_COMET_SPAWN_INTERVAL = 100    # comets respawn every 100 steps
_COMET_LIFESPAN       = 19     # turns alive before disappearing (ships die with it)
_COMET_EVAC_BUFFER    = 4      # leave comet this many steps before expiry
_COMET_MIN_SHIPS      = 10     # minimum fleet to guarantee capture of comet garrison

# ── Action Plan ─────────────────────────────────────────────────────────────────
# A turn-scheduled queue of future moves. Computed each turn, pruned for validity,
# executed when their scheduled turn arrives. Survives across agent() calls.
#
# Entry schema:
#   turn:     int  — execute on this exact step
#   src_pid:  int  — planet to send ships FROM (must be ours at execution time)
#   tgt_pid:  int | None  — planet to send ships TO (None = use tgt_xy)
#   tgt_xy:   (float, float) | None  — coordinate target (for comets: re-aim each turn)
#   ships:    int | 'all'  — ship count ('all' = send everything available)
#   reason:   str  — 'comet_grab' | 'comet_evac' | 'orbit_wait' | 'denial_prep'
#   max_turn: int  — cancel if not executed by this step (opportunity expired)
#   priority: int  — 1 = safety-critical (runs first), 2 = high, 3 = normal
#
_ACTION_PLAN: list = []
_PLAN_PREV_STEP: int = -1   # for new-game detection


def _comet_active_window(step: int):
    """
    Return (spawn_step, expire_step) for the comet group currently alive at `step`,
    or None if no comet group is active right now.
    """
    for spawn in range(_COMET_FIRST_SPAWN, step + _COMET_SPAWN_INTERVAL, _COMET_SPAWN_INTERVAL):
        expire = spawn + _COMET_LIFESPAN
        if spawn <= step < expire:
            return spawn, expire
    return None


def _next_comet_spawn(step: int) -> int:
    """Step of the next comet group spawn after `step`."""
    for spawn in range(_COMET_FIRST_SPAWN, 10000, _COMET_SPAWN_INTERVAL):
        if spawn > step:
            return spawn
    return 10000


def _prune_action_plan(state: dict) -> None:
    """Remove entries that are stale, past-due, or no longer valid."""
    global _ACTION_PLAN, _PLAN_PREV_STEP
    step = state['step']
    me   = state['me']
    planets = state['planets']

    # New game: reset
    if step < _PLAN_PREV_STEP - 5:
        _ACTION_PLAN = []
    _PLAN_PREV_STEP = step

    kept = []
    for e in _ACTION_PLAN:
        # Past max_turn → cancel
        if step > e['max_turn']:
            continue
        # Source planet must still be ours
        if e['src_pid'] not in planets or planets[e['src_pid']]['owner'] != me:
            continue
        # If comet_grab: target must still be neutral and still a comet
        if e['reason'] == 'comet_grab':
            tpid = e.get('tgt_pid')
            if tpid is not None:
                if tpid not in planets:
                    continue
                if planets[tpid]['owner'] == me:
                    continue   # already ours — no need to grab
        # If comet_evac: source must still be a comet we own
        if e['reason'] == 'comet_evac':
            if not planets.get(e['src_pid'], {}).get('is_comet'):
                continue      # comet expired or no longer a comet
        kept.append(e)
    _ACTION_PLAN = kept


def _plan_comet_actions(state: dict, me: int) -> None:
    """
    Schedule comet_grab and comet_evac entries for all visible comet planets.

    Grab: target neutral comets that we can reach before they expire.
          Send just enough ships — comet garrisons are tiny.

    Evac: for comets WE own, schedule an evacuation flight before expiry so
          ships aren't stranded when the comet disappears.
    """
    step    = state['step']
    planets = state['planets']
    omega   = state.get('omega', 0.03)

    # Current and upcoming comet window
    window = _comet_active_window(step)

    # Find comet planets
    comet_pids = [pid for pid, p in planets.items() if p.get('is_comet')]
    if not comet_pids:
        return

    # Source planets available for grabs
    my_pids = [pid for pid, p in planets.items() if p['owner'] == me and not p.get('is_comet')]

    # Planets already covered by existing plan entries (to avoid duplicates)
    planned_grabs = {e['tgt_pid'] for e in _ACTION_PLAN if e['reason'] == 'comet_grab' and e.get('tgt_pid') is not None}
    planned_evac  = {e['src_pid'] for e in _ACTION_PLAN if e['reason'] == 'comet_evac'}

    for cpid, cp in planets.items():
        if not cp.get('is_comet'):
            continue

        # Determine expiry
        if window:
            spawn_step, expire_step = window
        else:
            # Comet visible but we're between windows — use next spawn
            ns = _next_comet_spawn(step)
            spawn_step, expire_step = ns, ns + _COMET_LIFESPAN

        steps_left = expire_step - step

        # ── Comet grab ───────────────────────────────────────────────────────
        if cp['owner'] != me and cpid not in planned_grabs:
            best_src, best_eta, best_turn = None, 999, step
            for spid in my_pids:
                src = planets[spid]
                n   = max(_COMET_MIN_SHIPS, int(src['ships'] * 0.15))
                eta = travel_time(src['x'], src['y'], cp['x'], cp['y'], n)
                if eta < steps_left and eta < best_eta:
                    best_eta, best_src = eta, spid
                    best_turn = step   # fire immediately (comets move; re-aim each turn)

            if best_src is not None:
                n = max(_COMET_MIN_SHIPS, int(planets[best_src]['ships'] * 0.15))
                _ACTION_PLAN.append({
                    'turn':     best_turn,
                    'src_pid':  best_src,
                    'tgt_pid':  cpid,
                    'tgt_xy':   (cp['x'], cp['y']),
                    'ships':    n,
                    'reason':   'comet_grab',
                    'max_turn': expire_step - 2,
                    'priority': 2,
                })

        # ── Comet evacuation ─────────────────────────────────────────────────
        if cp['owner'] == me and cpid not in planned_evac:
            # Find nearest non-comet friendly planet
            my_safe = [(pid, p) for pid, p in planets.items()
                       if p['owner'] == me and not p.get('is_comet') and pid != cpid]
            if not my_safe:
                continue
            safe_pid, safe_p = min(my_safe,
                key=lambda kv: math.hypot(kv[1]['x'] - cp['x'], kv[1]['y'] - cp['y']))
            evac_eta   = travel_time(cp['x'], cp['y'], safe_p['x'], safe_p['y'], 20)
            evac_turn  = max(step, expire_step - int(evac_eta) - _COMET_EVAC_BUFFER)
            _ACTION_PLAN.append({
                'turn':     evac_turn,
                'src_pid':  cpid,
                'tgt_pid':  safe_pid,
                'tgt_xy':   None,
                'ships':    'all',
                'reason':   'comet_evac',
                'max_turn': expire_step - 1,
                'priority': 1,
            })


def _execute_action_plan(state: dict, moves: list, launched: dict) -> None:
    """
    Fire any action plan entries scheduled for this turn.
    Modifies moves and launched in-place.
    Priority 1 (comet_evac) runs first; then priority 2; then 3.
    """
    global _ACTION_PLAN
    step    = state['step']
    me      = state['me']
    planets = state['planets']
    omega   = state.get('omega', 0.03)

    due = sorted([e for e in _ACTION_PLAN if e['turn'] <= step],
                 key=lambda e: e['priority'])

    for e in due:
        src_pid = e['src_pid']
        if src_pid not in planets or planets[src_pid]['owner'] != me:
            continue

        av = planets[src_pid]['ships'] - launched.get(src_pid, 0)

        # Determine target position
        if e.get('tgt_pid') is not None and e['tgt_pid'] in planets:
            tgt = planets[e['tgt_pid']]
            tx, ty = tgt['x'], tgt['y']
            tgt_is_orb = tgt.get('is_orb', False)
        elif e.get('tgt_xy'):
            tx, ty = e['tgt_xy']
            tgt_is_orb = False
        else:
            continue

        # Ship count
        if e['ships'] == 'all':
            n_ships = max(1, int(av * 0.95))
        else:
            n_ships = min(int(e['ships']), int(av * 0.90))

        if n_ships < 1 or av < n_ships:
            continue

        ix, iy, _ = solve_intercept(planets[src_pid]['x'], planets[src_pid]['y'],
                                     tx, ty, tgt_is_orb, omega, n_ships)
        if path_crosses_sun(planets[src_pid]['x'], planets[src_pid]['y'], ix, iy):
            wp, _ = multi_leg_path(planets[src_pid]['x'], planets[src_pid]['y'], ix, iy)
            if wp is None:
                continue
            ix, iy = wp[0]

        angle = math.atan2(iy - planets[src_pid]['y'], ix - planets[src_pid]['x'])
        moves.append([src_pid, angle, n_ships])
        launched[src_pid] = launched.get(src_pid, 0) + n_ships

        # Register in fleet registry if tgt_pid known
        if e.get('tgt_pid') is not None:
            eta = travel_time(planets[src_pid]['x'], planets[src_pid]['y'], tx, ty, n_ships)
            _FLEET_REGISTRY.append({
                'target_pid':  e['tgt_pid'],
                'src_pid':     src_pid,
                'launch_step': step,
                'eta':         eta,
                'ships':       n_ships,
            })

    # Remove executed entries
    executed_reasons = {'comet_grab', 'comet_evac', 'orbit_wait', 'denial_prep'}
    _ACTION_PLAN = [e for e in _ACTION_PLAN if e['turn'] > step or e['reason'] not in executed_reasons]


# ── Neural adjust (identical to denial agent — proven weights) ─────────────────
np.random.seed(42)
HIDDEN      = 32
INPUT_SIZE  = 18
OUTPUT_SIZE = 10

try:
    with open('v90_weights.pkl', 'rb') as _f:
        _WEIGHTS = pickle.load(_f)
except Exception:
    try:
        with open('v89_weights.pkl', 'rb') as _f:
            _WEIGHTS = pickle.load(_f)
    except Exception:
        _WEIGHTS = {
            'W1': np.random.randn(INPUT_SIZE, HIDDEN) * 0.3,
            'B1': np.zeros(HIDDEN),
            'W2': np.random.randn(HIDDEN, OUTPUT_SIZE) * 0.3,
            'B2': np.zeros(OUTPUT_SIZE),
        }


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def neural_adjust(step, my_pc, my_prod, my_ships, ep, es, ifc, thr):
    x = np.array([
        min(step / 400.0, 1.0), my_pc / 8.0,
        min(my_prod / 20.0, 1.0), min(my_ships / 200.0, 1.0),
        min(ep / 20.0, 1.0) if ep > 0 else 0.0,
        min(es / 200.0, 1.0) if es > 0 else 0.0,
        ifc / 10.0, thr / 100.0,
        min(my_prod / max(ep, 0.1), 3.0) / 3.0,
        min(my_ships / max(es, 1.0), 5.0) / 5.0,
        1.0 if my_pc >= 3 else 0.0,
        1.0 if my_prod > ep else 0.0,
        1.0 if my_ships > es else 0.0,
        min(my_ships / max(es, 1.0), 5.0) / 5.0,
        min(ep / max(my_prod, 0.1), 3.0) / 3.0,
        1.0 if thr > my_ships * 0.2 else 0.0,
        min(ifc / 5.0, 1.0),
        1.0 if step < 50 else 0.0,
    ], dtype=np.float32)
    h = _sigmoid(np.dot(x, _WEIGHTS['W1']) + _WEIGHTS['B1'])
    out = _sigmoid(np.dot(h, _WEIGHTS['W2']) + _WEIGHTS['B2'])
    return {
        'aggression':        0.3 + out[0] * 0.7,
        'defense_sensitivity': 0.2 + out[2] * 0.3,
    }


# ── Geometry helpers (from denial agent, proven correct) ──────────────────────

def _line_seg_min_dist_sq(x1, y1, x2, y2, px, py):
    dx, dy = x2 - x1, y2 - y1
    lsq = dx * dx + dy * dy
    if lsq == 0.0:
        return (x1 - px) ** 2 + (y1 - py) ** 2
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / lsq))
    rx, ry = x1 + t * dx - px, y1 + t * dy - py
    return rx * rx + ry * ry


def path_crosses_sun(x1, y1, x2, y2, margin=1.5):
    return _line_seg_min_dist_sq(x1, y1, x2, y2, SUN_X, SUN_Y) < (SUN_RADIUS + margin) ** 2


def safe_angle(x1, y1, x2, y2):
    direct = math.atan2(y2 - y1, x2 - x1)
    if not path_crosses_sun(x1, y1, x2, y2, margin=1.5):
        return direct
    d = math.hypot(x1 - SUN_X, y1 - SUN_Y)
    if d <= SUN_RADIUS + 1.0:
        return direct
    half = math.asin(min(1.0, (SUN_RADIUS + 1.0) / d))
    to_sun = math.atan2(SUN_Y - y1, SUN_X - x1)
    cw, ccw = to_sun + half, to_sun - half
    def adiff(a):
        dd = (a - direct) % (2 * math.pi)
        return dd if dd < math.pi else 2 * math.pi - dd
    return cw if adiff(cw) < adiff(ccw) else ccw


def multi_leg_path(x1, y1, x2, y2, margin=2.0):
    if not path_crosses_sun(x1, y1, x2, y2, margin):
        return [(x2, y2)], math.hypot(x2 - x1, y2 - y1)
    ring = SUN_RADIUS + 15.0
    best_wp, best_d = None, float('inf')
    for angle in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
        bx = SUN_X + ring * math.cos(angle)
        by = SUN_Y + ring * math.sin(angle)
        if not path_crosses_sun(x1, y1, bx, by, margin) and not path_crosses_sun(bx, by, x2, y2, margin):
            d = math.hypot(bx - x1, by - y1) + math.hypot(x2 - bx, y2 - by)
            if d < best_d:
                best_d, best_wp = d, (bx, by)
    if best_wp:
        return [best_wp, (x2, y2)], best_d
    return None, float('inf')


def solve_intercept(fx, fy, tx, ty, orbiting, omega, ships, iterations=25):
    """Iterative intercept for orbiting planets; direct for static."""
    if not orbiting:
        t = travel_time(fx, fy, tx, ty, ships)
        return tx, ty, t
    t = travel_time(fx, fy, tx, ty, ships)
    ix, iy = tx, ty
    for _ in range(iterations):
        ix, iy = predict_orbit(tx, ty, omega, t)
        t2 = travel_time(fx, fy, ix, iy, ships)
        if abs(t2 - t) < 0.05:
            break
        t = t2
    return ix, iy, t


def optimal_intercept_time(fx, fy, tx, ty, omega, ships, max_delay=25):
    """Find launch delay [0, max_delay] that minimises (delay + travel)."""
    best_arr, best_delay, worse_streak = float('inf'), 0, 0
    for delay in range(max_delay + 1):
        px, py = predict_orbit(tx, ty, omega, float(delay)) if delay > 0 else (tx, ty)
        _, _, tt = solve_intercept(fx, fy, px, py, True, omega, ships)
        arr = delay + tt
        if arr < best_arr:
            best_arr, best_delay, worse_streak = arr, delay, 0
        else:
            worse_streak += 1
            if delay >= 5 and worse_streak >= 4:
                break
    return best_delay, best_arr


def ships_needed_for_takeover(tgt_ships, tgt_prod, tt, owner, margin=None):
    if margin is None:
        margin = TAKEOVER_MARGIN
    # Neutrals (owner == -1) do NOT produce ships — only owned planets do.
    # Enemies continue producing during fleet travel, so we must overbid.
    if owner == -1:
        return int(tgt_ships * margin) + 1
    return int((tgt_ships + tgt_prod * tt) * margin) + 1


def _rough_roi(src: dict, tgt: dict, av: int, omega: float,
               remaining_steps: int = 350) -> float:
    """
    Fast target pre-filter: estimate production value gained minus ship cost.

    Uses the correct intercept ETA (via solve_intercept for orbiting planets)
    so orbiting planets are not artificially favoured by stale-position ETA.

    Value = production_gain_over_rest_of_game - ship_cost
    Enemy targets are worth double (we gain their prod AND deny it from them).

    Returns -1e9 if we can't afford the target.
    """
    fleet_size = max(10, av)
    if tgt.get('is_orb', False):
        _, _, eta_est = solve_intercept(
            src['x'], src['y'], tgt['x'], tgt['y'],
            True, omega, fleet_size,
        )
    else:
        eta_est = travel_time(src['x'], src['y'], tgt['x'], tgt['y'], fleet_size)

    needed = ships_needed_for_takeover(tgt['ships'], tgt['prod'], eta_est, tgt['owner'])
    if needed > av:
        return -1e9

    prod_gain = tgt['prod'] * max(0.0, remaining_steps - eta_est)
    # Enemy: we gain their prod AND deny it — count double
    if tgt['owner'] >= 0:
        prod_gain *= (1.0 + _ENEMY_PROD_CREDIT)
        # Also credit ships enemy must spend to hold/retake
        prod_gain += tgt['ships'] * _ENEMY_SHIP_CREDIT
    return prod_gain - needed * 0.5


def _score_state(future, me, remaining_steps):
    """
    Score a predicted state for beam search selection.

    Value = (my_prod × prod_horizon + my_ships) - (enemy equiv)
            + in-transit fleet ship values
            + expected production from fleets likely to capture

    prod_horizon: clamped so late-game values ships more than production.
    """
    my_prod = ep_prod = my_ships = ep_ships = 0.0
    for p in future['planets'].values():
        if p['owner'] == me:
            my_prod  += p['prod']
            my_ships += p['ships']
        elif p['owner'] >= 0:
            ep_prod  += p['prod']
            ep_ships += p['ships']

    prod_horizon = min(remaining_steps, _PROD_HORIZON)
    score = (my_prod * prod_horizon + my_ships) - (ep_prod * prod_horizon + ep_ships)

    for fl in future['fleets']:
        if fl['owner'] == me:
            # Ships still in transit at end of sim window count at half value
            # (fleet launched to a far planet may not yet have landed).
            score += fl['ships'] * 0.5
        elif fl['owner'] >= 0:
            score -= fl['ships'] * 0.5  # enemy in-transit is a liability for us
    return score


# ── Tactical analysis ──────────────────────────────────────────────────────────

def analyze_threats(state):
    """
    Use exact fleet ETAs from physics_sim to identify planets under attack.

    Returns dict: pid -> {
        earliest_eta, total_enemy, garrison_at_arrival, is_doomed
    }

    garrison_at_arrival is computed by predict(state, N) — includes production
    and any of our own fleets arriving in the same window.
    """
    me = state['me']
    threats = {}

    for pid, planet in state['planets'].items():
        if planet['owner'] != me:
            continue

        enemy_inbound = [fl for fl in state['fleets']
                         if fl['target_pid'] == pid and fl['owner'] != me]
        if not enemy_inbound:
            continue

        earliest_eta = min(fl['eta'] for fl in enemy_inbound)
        total_enemy  = sum(fl['ships'] for fl in enemy_inbound)

        # Predict state at arrival — predict() already resolves combat
        arrival_step = max(1, round(earliest_eta))
        pred = predict(state, arrival_step)
        pred_planet  = pred['planets'].get(pid, planet)
        garrison     = pred_planet['ships']
        pred_owner   = pred_planet['owner']

        # is_doomed: we lose the planet in the simulation (owner changed away from us)
        threats[pid] = {
            'earliest_eta':        earliest_eta,
            'total_enemy':         total_enemy,
            'garrison_at_arrival': garrison,
            'is_doomed':           pred_owner != me,
        }
    return threats


def analyze_counter_windows(state, enemy_pids):
    """
    Enemy planets where garrison is suspiciously low (they just launched a big fleet).
    Heuristic: garrison < COUNTER_RATIO × production (4 steps of production).
    These are prime targets — hit now while they're weak.
    Returns set of planet IDs.
    """
    windows = set()
    for pid in enemy_pids:
        p = state['planets'][pid]
        if p['prod'] > 0 and p['ships'] < p['prod'] * COUNTER_RATIO and p['ships'] < 25:
            windows.add(pid)
    return windows


def analyze_denial(state, neutral_pids, enemy_pids, me):
    """
    Build denial priority map: neutral pid -> (bonus_score, enemy_eta).

    Reactive: enemy fleet already heading to this neutral → race them.
    Proactive: high-prod neutrals enemy can reach within 20 steps where we're closer.
    Sniper-nest: neutrals within SNIPER_RANGE of our large planets.

    Uses exact ETAs from physics_sim fleet tracking.
    """
    denial = {}
    omega  = state['omega']

    # Reactive: enemy fleets heading to a neutral
    for fl in state['fleets']:
        if fl['owner'] == me:
            continue
        tgt_pid = fl['target_pid']
        if tgt_pid not in neutral_pids:
            continue
        tgt = state['planets'][tgt_pid]
        enemy_eta = fl['eta']   # exact from physics_sim
        dbonus = (DENIAL_BASE_SCORE
                  + tgt['prod'] * DENIAL_PROD_WEIGHT
                  + (1.0 / max(enemy_eta, 0.5)) * DENIAL_URGENCY_WT)
        if tgt_pid not in denial or dbonus > denial[tgt_pid][0]:
            denial[tgt_pid] = (dbonus, enemy_eta)

    # Proactive: high-prod neutrals enemy can reach soon
    if enemy_pids:
        ep_list = [state['planets'][ep] for ep in enemy_pids if ep in state['planets']]
        my_list = [state['planets'][pp] for pp in state['planets']
                   if state['planets'][pp]['owner'] == me]
        for pid in neutral_pids:
            if pid in denial:
                continue
            tgt = state['planets'][pid]
            if tgt['prod'] < HIGH_PROD_THRESH:
                continue
            min_ene = min(travel_time(e['x'], e['y'], tgt['x'], tgt['y'], 20) for e in ep_list)
            if min_ene > 20:
                continue
            if my_list:
                min_mine = min(travel_time(p['x'], p['y'], tgt['x'], tgt['y'], 20) for p in my_list)
                if min_mine < min_ene:
                    denial[pid] = (PROACTIVE_BONUS, min_ene)

    # Sniper-nest: neutrals close to our big planets
    big_mine = [state['planets'][pp] for pp in state['planets']
                if state['planets'][pp]['owner'] == me
                and state['planets'][pp]['ships'] >= SNIPER_MIN_SHIPS]
    for pid in neutral_pids:
        tgt = state['planets'][pid]
        for bp in big_mine:
            d = math.hypot(tgt['x'] - bp['x'], tgt['y'] - bp['y'])
            if d <= SNIPER_RANGE:
                sb = SNIPER_BONUS + tgt['prod'] * DENIAL_PROD_WEIGHT
                eta = travel_time(bp['x'], bp['y'], tgt['x'], tgt['y'], 20)
                if pid not in denial or sb > denial[pid][0]:
                    denial[pid] = (sb, eta)
                break

    return denial


def find_wave_targets(state, my_pids, all_target_pids):
    """
    Find targets reachable simultaneously from 2 of our planets within WAVE_SYNC_TOLERANCE.
    Returns dict: tgt_pid -> [(eta, src_pid, avail_ships), ...] sorted by ETA.
    Only includes targets where combined ships can capture.
    """
    omega   = state['omega']
    results = {}

    for tgt_pid in all_target_pids:
        tgt = state['planets'][tgt_pid]
        etas = []
        for src_pid in my_pids:
            src = state['planets'][src_pid]
            if src['ships'] < 10:
                continue
            _, _, eta = solve_intercept(
                src['x'], src['y'], tgt['x'], tgt['y'],
                tgt['is_orb'], omega, int(src['ships']))
            etas.append((eta, src_pid, int(src['ships'] * 0.6)))

        if len(etas) < 2:
            continue
        etas.sort()
        eta1, src1, ships1 = etas[0]
        eta2, src2, ships2 = etas[1]
        if abs(eta1 - eta2) > WAVE_SYNC_TOLERANCE:
            continue

        combined     = ships1 + ships2
        arr_step     = max(1, round((eta1 + eta2) / 2))
        pred         = predict(state, arr_step)
        garrison_fut = pred['planets'][tgt_pid]['ships'] if tgt_pid in pred['planets'] else tgt['ships']
        needed       = int(garrison_fut * TAKEOVER_MARGIN) + 1

        if combined >= needed:
            results[tgt_pid] = etas[:2]

    return results


# ── Main agent ─────────────────────────────────────────────────────────────────

def agent(obs):  # noqa: C901 (complex but necessarily so — game AI)
    # ── Parse exact state ─────────────────────────────────────────────────────
    state = parse_obs(obs)
    me    = state['me']
    step  = state['step']
    omega = state['omega']

    my_pids      = {pid for pid, p in state['planets'].items() if p['owner'] == me}
    enemy_pids   = {pid for pid, p in state['planets'].items()
                    if p['owner'] >= 0 and p['owner'] != me}
    neutral_pids = {pid for pid, p in state['planets'].items() if p['owner'] < 0}

    if not my_pids:
        return []

    my      = [state['planets'][pid] for pid in my_pids]
    enemy   = [state['planets'][pid] for pid in enemy_pids]

    my_ships    = sum(p['ships'] for p in my)
    my_prod     = sum(p['prod'] for p in my)
    enemy_ships = sum(p['ships'] for p in enemy) if enemy else 0.0
    enemy_prod  = sum(p['prod'] for p in enemy)  if enemy else 0.0
    prod_ratio  = my_prod  / max(enemy_prod,  1.0)
    ship_ratio  = my_ships / max(enemy_ships, 1.0)

    # ── Fleet registry: reliable inflight tracking ─────────────────────────
    # Must happen BEFORE building my_inflight_targets so the augmented state
    # is used everywhere (beam search baseline, wave targets, denial check).
    _prune_registry(step)
    _inject_registry_fleets(state, me, step)   # fills gaps left by infer_fleet_dest

    # Merge inference-based targets with our registry for complete coverage
    my_inflight_targets = (
        {fl['target_pid'] for fl in state['fleets'] if fl['owner'] == me}
        | _registry_inflight_targets(step)
    )
    in_flight_count     = sum(1 for fl in state['fleets'] if fl['owner'] == me)

    # ── Tactical analysis ─────────────────────────────────────────────────────
    threats        = analyze_threats(state)
    threats_total  = sum(t['total_enemy'] for t in threats.values())
    doomed_pids    = {pid for pid, t in threats.items() if t['is_doomed']}
    threatened_pids = {pid for pid, t in threats.items() if not t['is_doomed']}
    counter_windows = analyze_counter_windows(state, enemy_pids)
    denial_map     = analyze_denial(state, neutral_pids, enemy_pids, me)

    moves    = []
    targeted = set()
    launched = {}  # src_pid -> ships launched this turn

    # ── Action plan: prune stale, fire safety-critical (priority 1) ──────────
    _prune_action_plan(state)
    _execute_action_plan(state, moves, launched)   # priority-1 comet_evac fires here

    def avail(src_pid):
        return state['planets'][src_pid]['ships'] - launched.get(src_pid, 0)

    def send_move(src_pid, tgt_pid, n_ships):
        """Compute intercept angle and append move. Returns True on success."""
        n_ships = int(n_ships)
        if n_ships < 1:
            return False
        src = state['planets'][src_pid]
        tgt = state['planets'][tgt_pid]
        ix, iy, fleet_eta = solve_intercept(src['x'], src['y'], tgt['x'], tgt['y'],
                                            tgt['is_orb'], omega, n_ships)
        if path_crosses_sun(src['x'], src['y'], ix, iy):
            wp, _ = multi_leg_path(src['x'], src['y'], ix, iy)
            if wp is None:
                return False
            ix, iy = wp[0]  # first waypoint leg
        angle = safe_angle(src['x'], src['y'], ix, iy)
        moves.append([src_pid, angle, n_ships])
        launched[src_pid] = launched.get(src_pid, 0) + n_ships
        # Register this launch so future turns know exactly where this fleet is going
        _FLEET_REGISTRY.append({
            'target_pid': tgt_pid,
            'src_pid':    src_pid,
            'launch_step': step,
            'eta':         fleet_eta,
            'ships':       n_ships,
        })
        return True

    # ── 1. Evacuation — move ships off doomed planets ─────────────────────────
    # Priority order for evacuating ships:
    #   A. Attack: send them at a profitable neutral/enemy they can actually take
    #   B. Reinforce: shore up a threatened (not doomed) friendly that needs ships
    #      and we can arrive before the attacker
    #   C. Retreat: nearest safe (non-doomed, non-threatened) friendly
    #   D. Anywhere non-doomed (last resort)
    # Skip evacuation entirely when we're in a dominant winning position.
    remaining_steps = max(1, 400 - step)
    if not (prod_ratio > 3.0 and ship_ratio > 2.0):
        for pid in doomed_pids:
            av = avail(pid)
            if av < 5:
                continue
            evac = int(av * EVAC_FRACTION)
            if evac < 5:
                continue
            src = state['planets'][pid]
            fired = False

            # ── A. Attack with evacuating ships ──────────────────────────────
            attack_pool = [
                p for p in (neutral_pids | enemy_pids)
                if p not in targeted and p not in my_inflight_targets
            ]
            best_atk, best_roi = None, 0.0
            for tpid in attack_pool:
                roi = _rough_roi(src, state['planets'][tpid], evac, omega)
                if roi > best_roi:
                    best_roi, best_atk = roi, tpid
            if best_atk is not None:
                if send_move(pid, best_atk, evac):
                    targeted.add(best_atk)
                    fired = True

            if fired:
                continue

            # ── B. Reinforce a threatened planet we can reach in time ─────────
            best_rein, best_deficit = None, 0
            for tpid in threatened_pids:
                if tpid == pid or tpid in doomed_pids:
                    continue
                threat    = threats[tpid]
                tgt_p     = state['planets'][tpid]
                deficit   = max(0, int(threat['total_enemy'] * DEFENSE_MARGIN
                                       - threat['garrison_at_arrival']))
                if deficit <= 0 or deficit <= best_deficit:
                    continue
                _, _, our_eta = solve_intercept(
                    src['x'], src['y'], tgt_p['x'], tgt_p['y'],
                    tgt_p['is_orb'], omega, max(1, evac)
                )
                if our_eta < threat['earliest_eta'] - REINFORCE_BUFFER:
                    best_deficit, best_rein = deficit, tpid
            if best_rein is not None:
                send_move(pid, best_rein, evac)
                fired = True

            if fired:
                continue

            # ── C/D. Safe retreat ─────────────────────────────────────────────
            # Only retreat if the evacuated ships have positive ROI somewhere
            # OR if enemy would spend fewer ships conquering an empty planet
            # than they would fighting our garrison.  Staying and fighting
            # costs the enemy `evac` ships; only retreat if that's less valuable
            # than what those ships accomplish elsewhere.
            enemy_conquest_cost = evac  # ships enemy burns taking the planet
            safe_dsts = [dpid for dpid in my_pids
                         if dpid != pid and dpid not in doomed_pids
                         and dpid not in threatened_pids]
            if not safe_dsts:
                safe_dsts = [dpid for dpid in my_pids
                             if dpid != pid and dpid not in doomed_pids]
            if not safe_dsts:
                continue
            best_dst = min(safe_dsts,
                           key=lambda dp: math.hypot(state['planets'][dp]['x'] - src['x'],
                                                     state['planets'][dp]['y'] - src['y']))
            dst_p = state['planets'][best_dst]
            # Check if there's a profitable attack reachable from the destination
            # (i.e., these ships will eventually do something useful there)
            retreat_roi = max(
                (_rough_roi(dst_p, state['planets'][tpid], evac, omega, remaining_steps)
                 for tpid in (neutral_pids | enemy_pids)
                 if tpid not in targeted),
                default=-1e9
            )
            # Retreat if ships can do something useful elsewhere
            # OR if destination has many ships (can reinforce a future attack)
            if retreat_roi > 0 or dst_p['ships'] > evac * 2:
                send_move(pid, best_dst, evac)

    # ── 2. Physics beam search — find best action per source planet ──────────────

    # Smash: enemy planets where we have local ship advantage within 50 units.
    # This mirrors the 131/denial "smash phase" — fires from step 0 if geometry is right.
    smash_pids: set = set()
    for e_pid in enemy_pids:
        e = state['planets'][e_pid]
        nearby_ships = sum(
            p['ships'] for p in state['planets'].values()
            if p['owner'] == me
            and (p['x'] - e['x']) ** 2 + (p['y'] - e['y']) ** 2 < _SMASH_RADIUS_SQ
        )
        if nearby_ships > e['ships'] * 1.25:
            smash_pids.add(e_pid)

    losing_badly = ship_ratio < 0.7 and prod_ratio < 0.7
    can_attack_enemy = (
        bool(smash_pids)                                             # local ship advantage
        or ship_ratio > 1.5 or prod_ratio > 1.5                     # clear overall lead
        or (not losing_badly and len(my_pids) >= 3)                  # 3+ planets, not behind
        or (not losing_badly and step > 30)                          # mid-game, not behind
    )
    if losing_badly:
        can_attack_enemy = bool(smash_pids)   # only opportunistic attacks when badly behind

    if enemy_pids and can_attack_enemy:
        tgt_set = enemy_pids | neutral_pids
    elif counter_windows:
        tgt_set = counter_windows | neutral_pids
    else:
        tgt_set = neutral_pids

    # ── Fixed sim horizon for ALL targets ──────────────────────────────────────
    # Analytic scoring: production value over a fixed horizon minus ship cost.
    # Near targets (low ETA) get more production steps within the horizon,
    # correctly beating far targets of equal prod. No simulation needed —
    # capture feasibility is checked by ships_needed_for_takeover.
    # Enemy fleets are not modeled here; threats are handled by analyze_threats().
    _sim_horizon = min(_SIM_HORIZON, remaining_steps)

    # Top sources ordered by available ships (skip doomed ones)
    src_planets = sorted(
        [pid for pid in my_pids if pid not in doomed_pids],
        key=lambda p: -state['planets'][p]['ships']
    )[:_MAX_SOURCES]

    for src_pid in src_planets:
        av = avail(src_pid)
        if av < 10:
            continue
        src = state['planets'][src_pid]

        open_targets = [p for p in tgt_set
                        if p not in targeted and p not in my_inflight_targets]
        top_tgts = sorted(
            open_targets,
            key=lambda pid: _rough_roi(src, state['planets'][pid], av, omega, remaining_steps),
            reverse=True,
        )[:_MAX_TARGETS]

        best_score  = -1e9
        best_action = None   # (tgt_pid, ships, ix, iy, horizon)

        for tgt_pid in top_tgts:
            tgt = state['planets'][tgt_pid]
            ix, iy, eta = solve_intercept(
                src['x'], src['y'], tgt['x'], tgt['y'],
                tgt['is_orb'], omega, int(av)
            )
            if path_crosses_sun(src['x'], src['y'], ix, iy):
                wp, _ = multi_leg_path(src['x'], src['y'], ix, iy)
                if wp is None:
                    continue

            needed = ships_needed_for_takeover(tgt['ships'], tgt['prod'], eta, tgt['owner'])
            if av < needed:
                continue

            # Ship count: neutrals → exact needed (no waste); enemy → commit enough to crush
            if tgt['owner'] < 0:
                ships = needed
            else:
                ships = max(needed, int(av * 0.65))
                ships = min(ships, int(av * 0.90))
                # Smash targets: always commit enough
                if tgt_pid in smash_pids:
                    ships = max(ships, int(av * 0.80))
            if ships < 1:
                continue

            # ── Analytic marginal: production value over sim window ─────────
            # Mirrors what the sim would compute: planet captured at step `eta`,
            # produces for `horizon - eta` steps.  Near targets (low ETA) get
            # more production steps, correctly beating far targets.
            # No actual simulation needed — capture feasibility already checked
            # by ships_needed_for_takeover above.
            prod_steps = max(0.0, _sim_horizon - eta)
            prod_value = tgt['prod'] * prod_steps
            if tgt['owner'] >= 0:
                prod_value *= 2.0  # enemy: gain + deny
            ship_cost = ships * 0.5
            marginal = prod_value - ship_cost

            # Tactical urgency bonuses
            if tgt_pid in denial_map and tgt['owner'] < 0:
                dbonus, enemy_eta = denial_map[tgt_pid]
                if eta <= enemy_eta:
                    marginal += dbonus * 0.25
            if tgt_pid in counter_windows:
                marginal += COUNTER_BONUS * 0.25

            if marginal > best_score:
                best_score  = marginal
                best_action = (tgt_pid, ships, ix, iy)

        if best_action is None:
            continue

        tgt_pid, ships, ix, iy = best_action

        # Only fire if this action genuinely beats doing nothing for the same period.
        if best_score <= 0:
            continue

        if avail(src_pid) >= ships >= 1:
            if send_move(src_pid, tgt_pid, ships):
                targeted.add(tgt_pid)

    # ── 3. Reinforcement — shore up threatened-but-not-doomed planets ─────────
    # Runs AFTER beam search so attacks get first pick of available ships.
    for pid in threatened_pids:
        threat    = threats[pid]
        tgt_data  = state['planets'][pid]
        eta_enemy = threat['earliest_eta']
        proj_garrison = tgt_data['ships'] + tgt_data['prod'] * max(1, round(eta_enemy))
        deficit   = max(0, threat['total_enemy'] * DEFENSE_MARGIN - proj_garrison)

        for src_pid in sorted(my_pids, key=lambda p: math.hypot(
                state['planets'][p]['x'] - tgt_data['x'],
                state['planets'][p]['y'] - tgt_data['y'])):
            if src_pid == pid or src_pid in doomed_pids:
                continue
            av = avail(src_pid)
            if av < 5:
                continue
            src = state['planets'][src_pid]
            _, _, our_eta = solve_intercept(src['x'], src['y'], tgt_data['x'], tgt_data['y'],
                                            tgt_data['is_orb'], omega, max(1, int(av * 0.5)))
            if our_eta < eta_enemy - REINFORCE_BUFFER:
                send = max(5, min(int(deficit * 1.2), int(av * 0.6)))
                if av >= send:
                    send_move(src_pid, pid, send)
                break

    # ── Action plan: plan future comet actions ────────────────────────────────
    _plan_comet_actions(state, me)

    return moves


if __name__ == '__main__':
    print("main_pred_2p: predictive-baseline 2p agent (imports physics_sim.py)")
