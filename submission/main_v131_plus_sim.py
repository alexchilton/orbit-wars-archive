"""
Orbit Wars - sim: forward-simulator agent
Architecture:
  1. Physics engine  — deterministic N-step game simulation
  2. Evaluator       — score any future state
  3. Action planner  — beam search over candidate moves, picks best first action
  4. Execution       — convert plan to angle/send moves with sun avoidance

Enemy model: greedy-best (Option C) — enemy grabs nearest neutral each turn.
"""
import os
os.environ['KAGGLE_ENVELOPES'] = '0'

import math
import numpy as np
import pickle
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

# ─────────────────────────────── constants ───────────────────────────────────
SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS    = 10.0
INNER_ORBIT_THRESHOLD = 48.0
MAX_SPEED     = 6.0
_MAX_SPEED_MINUS_1 = MAX_SPEED - 1.0
_LOG1000 = math.log(1000.0)

SIM_HORIZON      = int(os.environ.get('SIM_HORIZON', '35'))   # turns to look ahead
SIM_BEAM_WIDTH   = int(os.environ.get('SIM_BEAM_WIDTH', '6'))  # beams kept per round
SIM_CANDIDATES   = int(os.environ.get('SIM_CANDIDATES', '4'))  # targets considered per planet
SNIPER_RANGE     = float(os.environ.get('SIM_SNIPER_RANGE', '35'))
SNIPER_MIN_SHIPS = int(os.environ.get('SIM_SNIPER_MIN_SHIPS', '40'))
SNIPER_DENIAL_BONUS = float(os.environ.get('SIM_SNIPER_DENIAL_BONUS', '60'))

# Evaluator weights
W_PROD_RATIO  = float(os.environ.get('SIM_W_PROD',   '4.0'))
W_SHIP_RATIO  = float(os.environ.get('SIM_W_SHIPS',  '2.0'))
W_PLANET_CNT  = float(os.environ.get('SIM_W_PCNT',   '3.0'))
W_THREAT      = float(os.environ.get('SIM_W_THREAT', '-1.5'))

np.random.seed(42)

# ─────────────────────────── neural net (for tuning) ─────────────────────────
HIDDEN, INPUT_SIZE, OUTPUT_SIZE = 32, 18, 10
try:
    with open('TargetStriker/winning_target_selector_v2_weights.pkl', 'rb') as f:
        TARGET_SELECTOR_WEIGHTS = pickle.load(f)
except Exception:
    TARGET_SELECTOR_WEIGHTS = None

try:
    with open('v90_weights.pkl', 'rb') as f:
        WEIGHTS = pickle.load(f)
    print("sim: Loaded v90 weights")
except Exception:
    try:
        with open('v89_weights.pkl', 'rb') as f:
            WEIGHTS = pickle.load(f)
        print("sim: Loaded v89 weights")
    except Exception:
        WEIGHTS = {
            'W1': np.random.randn(INPUT_SIZE, HIDDEN) * 0.3,
            'B1': np.zeros(HIDDEN),
            'W2': np.random.randn(HIDDEN, OUTPUT_SIZE) * 0.3,
            'B2': np.zeros(OUTPUT_SIZE),
        }

# ──────────────────────────── physics helpers ─────────────────────────────────

def fleet_speed(ships: int) -> float:
    return 1.0 + _MAX_SPEED_MINUS_1 * (math.log(max(ships, 1)) / _LOG1000) ** 1.5

def travel_time(x1, y1, x2, y2, ships: int) -> float:
    dx, dy = x2 - x1, y2 - y1
    if ships <= 0:
        return 999.0
    return math.sqrt(dx*dx + dy*dy) / fleet_speed(ships)

def predict_orbit(x, y, omega, dt):
    theta = math.atan2(y - SUN_Y, x - SUN_X)
    r     = math.hypot(x - SUN_X, y - SUN_Y)
    return SUN_X + r * math.cos(theta + omega * dt), SUN_Y + r * math.sin(theta + omega * dt)

def solve_intercept(fx, fy, tx, ty, orbiting, omega, ships, iterations=25):
    if not orbiting:
        return tx, ty, travel_time(fx, fy, tx, ty, ships)
    t = travel_time(fx, fy, tx, ty, ships)
    for _ in range(iterations):
        ix, iy = predict_orbit(tx, ty, omega, t)
        t2 = travel_time(fx, fy, ix, iy, ships)
        if abs(t2 - t) < 0.05:
            break
        t = t2
    return ix, iy, t

def line_seg_min_dist_sq(x1, y1, x2, y2, px, py):
    dx, dy = x2-x1, y2-y1
    lsq = dx*dx + dy*dy
    if lsq == 0:
        return (x1-px)**2 + (y1-py)**2
    t = max(0.0, min(1.0, ((px-x1)*dx + (py-y1)*dy) / lsq))
    return (x1+t*dx-px)**2 + (y1+t*dy-py)**2

def path_crosses_sun(x1, y1, x2, y2, margin=1.5):
    return line_seg_min_dist_sq(x1, y1, x2, y2, SUN_X, SUN_Y) < (SUN_RADIUS+margin)**2

def safe_angle(x1, y1, x2, y2):
    direct = math.atan2(y2-y1, x2-x1)
    if not path_crosses_sun(x1, y1, x2, y2, 1.5):
        return direct
    # Deflect around sun
    d_sq = (x1-SUN_X)**2 + (y1-SUN_Y)**2
    d = math.sqrt(d_sq)
    if d <= SUN_RADIUS + 1.0:
        return direct
    half = math.asin(min(1.0, (SUN_RADIUS + 1.0) / d))
    to_sun = math.atan2(SUN_Y-y1, SUN_X-x1)
    a1 = (to_sun + half) % (2*math.pi)
    a2 = (to_sun - half) % (2*math.pi)
    t_ang = math.atan2(y2-y1, x2-x1) % (2*math.pi)
    diff1 = min(abs(a1-t_ang), 2*math.pi-abs(a1-t_ang))
    diff2 = min(abs(a2-t_ang), 2*math.pi-abs(a2-t_ang))
    return a1 if diff1 < diff2 else a2

def ships_needed(tgt_ships, tgt_prod, tt, owner, margin=1.05):
    if owner == -1:
        return int(tgt_ships * margin) + 1
    return int((tgt_ships + tgt_prod * tt) * margin) + 1

# ──────────────────────────── simulator ──────────────────────────────────────
# State representation (all values are Python primitives for speed):
#   planets: { id -> {'x','y','owner','ships','prod','is_orb','theta','r'} }
#   fleets:  list of {'owner','ships','tx','ty','eta','target_id'}
#   step:    int
#   me:      player id (0 or 1)

def _make_state(obs_planets, obs_fleets, me, omega, step):
    planets = {}
    for p in obs_planets:
        pid = p['id']
        planets[pid] = {
            'x': p['x'], 'y': p['y'],
            'owner': p['owner'], 'ships': float(p['ships']),
            'prod': p.get('production', p.get('prod', 1)),
            'is_orb': p.get('orbiting', p.get('is_orb', False)),
            'theta': math.atan2(p['y']-SUN_Y, p['x']-SUN_X),
            'r': math.hypot(p['x']-SUN_X, p['y']-SUN_Y),
        }
    # Convert live fleets into simplified ETA-based records
    fleets = []
    for f in obs_fleets.values() if isinstance(obs_fleets, dict) else obs_fleets:
        owner = f['owner']
        ships = float(f['ships'])
        fx, fy = f['x'], f['y']
        # Find nearest planet as destination proxy
        best_pid, best_dsq = None, float('inf')
        for pid, pl in planets.items():
            dsq = (fx-pl['x'])**2 + (fy-pl['y'])**2
            if dsq < best_dsq:
                best_dsq, best_pid = dsq, pid
        if best_pid is None:
            continue
        tgt = planets[best_pid]
        eta = travel_time(fx, fy, tgt['x'], tgt['y'], int(max(ships, 1)))
        fleets.append({'owner': owner, 'ships': ships,
                       'target_id': best_pid, 'eta': eta})
    return {'planets': planets, 'fleets': fleets, 'step': step,
            'me': me, 'omega': omega}

def _step_state(state):
    """Advance state by one turn (mutates and returns)."""
    planets = state['planets']
    fleets  = state['fleets']
    omega   = state['omega']
    me      = state['me']
    step    = state['step']

    # 1. Tick fleet ETAs; collect arrivals
    arrivals = {}   # planet_id -> {owner: total_ships}
    surviving = []
    for fl in fleets:
        fl['eta'] -= 1.0
        if fl['eta'] <= 0.0:
            tid = fl['target_id']
            arrivals.setdefault(tid, {})
            arrivals[tid][fl['owner']] = arrivals[tid].get(fl['owner'], 0) + fl['ships']
        else:
            surviving.append(fl)
    state['fleets'] = surviving

    # 2. Resolve arrivals (simplified: sum friendly, fight enemy)
    for tid, att in arrivals.items():
        if tid not in planets:
            continue
        p = planets[tid]
        for owner, ships in att.items():
            if owner == p['owner']:
                p['ships'] += ships
            else:
                if ships > p['ships']:
                    p['owner']  = owner
                    p['ships'] = ships - p['ships']
                else:
                    p['ships'] -= ships

    # 3. Production
    for p in planets.values():
        if p['owner'] != -1:
            p['ships'] += p['prod']

    # 4. Orbit
    for p in planets.values():
        if p['is_orb']:
            p['theta'] += omega
            p['x'] = SUN_X + p['r'] * math.cos(p['theta'])
            p['y'] = SUN_Y + p['r'] * math.sin(p['theta'])

    state['step'] = step + 1
    return state

def _enemy_moves(state):
    """Greedy enemy: each enemy planet sends to nearest reachable neutral or weak enemy."""
    # Fire every 5 steps to avoid over-expansion in simulation
    if state['step'] % 5 != 0:
        return
    planets = state['planets']
    omega   = state['omega']
    me      = state['me']
    new_fleets = []
    my_planet_ids  = {pid for pid, p in planets.items() if p['owner'] == me}
    for pid, p in planets.items():
        if p['owner'] == me or p['owner'] == -1:
            continue
        owner = p['owner']
        avail = p['ships'] * 0.4   # less aggressive: 40% of ships
        if avail < 8:
            continue
        # Greedy: target nearest neutral or my planet
        best, best_tt = None, float('inf')
        for tpid, tp in planets.items():
            if tp['owner'] == owner:
                continue
            _, _, tt = solve_intercept(p['x'], p['y'], tp['x'], tp['y'],
                                       tp['is_orb'], omega, int(avail))
            score = -tt + (tp['prod'] * 5 if tp['owner'] == -1 else tp['prod'] * 3)
            if best is None or -tt > -best_tt:
                best, best_tt = tpid, tt
        if best is not None:
            new_fleets.append({'owner': owner, 'ships': avail,
                               'target_id': best, 'eta': best_tt})
            p['ships'] -= avail
    state['fleets'].extend(new_fleets)

def _simulate(state, my_moves, horizon):
    """Simulate horizon steps; apply my_moves on step 0, enemy moves every step."""
    s = deepcopy(state)
    # Apply my moves as immediate fleet launches
    planets = s['planets']
    omega   = s['omega']
    me      = s['me']
    for (src_id, tgt_id, ships_to_send) in my_moves:
        if src_id not in planets or tgt_id not in planets:
            continue
        src = planets[src_id]
        tgt = planets[tgt_id]
        if src['ships'] < ships_to_send:
            ships_to_send = src['ships'] * 0.8
        if ships_to_send < 1:
            continue
        _, _, eta = solve_intercept(src['x'], src['y'], tgt['x'], tgt['y'],
                                    tgt['is_orb'], omega, int(ships_to_send))
        s['fleets'].append({'owner': me, 'ships': ships_to_send,
                            'target_id': tgt_id, 'eta': eta})
        src['ships'] -= ships_to_send

    for _ in range(horizon):
        _enemy_moves(s)
        _step_state(s)
    return s

# ──────────────────────────── evaluator ──────────────────────────────────────

def evaluate(state):
    """Score a game state from our perspective."""
    planets = state['planets']
    me      = state['me']

    my_prod, en_prod = 0.0, 0.0
    my_ships, en_ships = 0.0, 0.0
    my_cnt, en_cnt = 0, 0

    for p in planets.values():
        if p['owner'] == me:
            my_prod   += p['prod']
            my_ships  += p['ships']
            my_cnt    += 1
        elif p['owner'] != -1:
            en_prod   += p['prod']
            en_ships  += p['ships']
            en_cnt    += 1

    if en_prod == 0 and en_ships == 0:
        return 1e6   # We won
    if my_prod == 0 and my_ships == 0:
        return -1e6  # We lost

    prod_ratio  = my_prod  / max(en_prod,  0.1)
    ship_ratio  = my_ships / max(en_ships, 0.1)
    planet_diff = my_cnt   - en_cnt

    # Threat exposure: friendly ships threatened by incoming enemy fleets
    threat = sum(fl['ships'] for fl in state['fleets']
                 if fl['owner'] != me and
                 planets.get(fl['target_id'], {}).get('owner') == me)

    score = (W_PROD_RATIO  * prod_ratio
           + W_SHIP_RATIO  * ship_ratio
           + W_PLANET_CNT  * planet_diff
           + W_THREAT      * threat)
    return score

# ──────────────────────────── action planner ─────────────────────────────────

def _candidate_targets(state):
    """For each of our planets, return top SIM_CANDIDATES targets to consider."""
    planets = state['planets']
    omega   = state['omega']
    me      = state['me']

    candidates = {}  # src_id -> [(tgt_id, ships, priority_score), ...]
    for pid, p in planets.items():
        if p['owner'] != me:
            continue
        my_ships = p['ships']
        if my_ships < 5:
            continue
        scored = []
        for tpid, tp in planets.items():
            if tp['owner'] == me:
                continue
            # Use up to 90% of available ships for costing
            _, _, tt = solve_intercept(p['x'], p['y'], tp['x'], tp['y'],
                                       tp['is_orb'], omega, int(max(my_ships * 0.8, 1)))
            need = ships_needed(tp['ships'], tp['prod'], tt, tp['owner'])
            # Only skip if we literally can't afford it
            if need > my_ships * 0.95:
                continue
            # Send enough to hold against counter-attack (enemy fires every ~3 steps)
            hold_buffer = max(int(tp['prod'] * 8), 12)
            send_ships = min(need + hold_buffer, int(my_ships * 0.75))
            send_ships = max(send_ships, need)  # always at least enough to capture
            score = tp['prod'] * 18 - tt * 3.5 + (25 if tp['owner'] == -1 else 35)
            scored.append((tpid, send_ships, score))
        scored.sort(key=lambda x: -x[2])
        if scored:
            candidates[pid] = scored[:SIM_CANDIDATES]
    return candidates

def plan(state):
    """
    Greedy target selection with simulation validation.
    Heuristic scores all candidates, picks best non-conflicting moves.
    Simulation used ONLY to reject catastrophically bad moves (10+ pts worse than idle).
    """
    candidates = _candidate_targets(state)
    if not candidates:
        return []

    step = state.get('step', 0)

    # Flatten to (src_id, tgt_id, ships, h_score) sorted by heuristic
    all_candidates = []
    for src_id, targets in candidates.items():
        for tgt_id, ships, h_score in targets:
            all_candidates.append((src_id, tgt_id, ships, h_score))
    all_candidates.sort(key=lambda x: -x[3])

    # Baseline idle score (simulate doing nothing)
    idle_future  = _simulate(state, [], SIM_HORIZON)
    idle_score   = evaluate(idle_future)

    # Greedy: pick best non-conflicting moves, reject catastrophic ones
    best_moves = []
    used_src   = set()
    used_tgt   = set()
    for src_id, tgt_id, ships, h_score in all_candidates:
        if src_id in used_src or tgt_id in used_tgt:
            continue
        move = [(src_id, tgt_id, ships)]
        future = _simulate(state, move, SIM_HORIZON)
        sc = evaluate(future)
        if step in (5, 10, 20):
            _DEBUG_LOG.append({'step': step, 'CAND': move, 'sim': round(sc,2), 'idle': round(idle_score,2)})
        # Only reject if simulation shows this is genuinely catastrophic
        if sc < idle_score - 15.0:
            continue
        best_moves.append((src_id, tgt_id, ships))
        used_src.add(src_id)
        used_tgt.add(tgt_id)

    return best_moves
    return best_moves

# ──────────────────────────── denial pre-pass ────────────────────────────────

def _build_denial_bonus(state, neutrals_list):
    """Flag neutral planets being targeted by enemy fleets, and sniper nests."""
    planets = state['planets']
    fleets  = state['fleets']
    me      = state['me']
    omega   = state['omega']

    denial = {}  # planet_id -> priority_score

    # Reactive: enemy fleet heading for neutral
    for fl in fleets:
        if fl['owner'] == me:
            continue
        tid = fl['target_id']
        if tid not in planets or planets[tid]['owner'] != -1:
            continue
        tp = planets[tid]
        score = 75 + tp['prod'] * 8 + 20 / max(fl['eta'], 0.5)
        if tid not in denial or score > denial[tid]:
            denial[tid] = score

    # Sniper nest: small neutral close to our large planet
    my_big = [p for p in planets.values() if p['owner'] == me and p['ships'] >= SNIPER_MIN_SHIPS]
    for np_ in neutrals_list:
        for bp in my_big:
            d = math.sqrt((np_['x']-bp['x'])**2 + (np_['y']-bp['y'])**2)
            if d <= SNIPER_RANGE:
                score = SNIPER_DENIAL_BONUS + np_['prod'] * 8
                if np_['id'] not in denial or score > denial[np_['id']]:
                    denial[np_['id']] = score
                break

    return denial

# ──────────────────────────── main agent ─────────────────────────────────────

_MAP_DETECTED = False
_MAP_SPREAD   = 'mid'
_TAKEOVER_MARGIN = 1.05
_DEBUG_LOG = []  # temporary debug collection

def _detect_map(planets_data):
    global _MAP_DETECTED, _MAP_SPREAD
    if _MAP_DETECTED or not planets_data or len(planets_data) < 4:
        return
    # planets_data elements are raw lists [id,owner,x,y,...] or dicts
    if isinstance(planets_data[0], (list, tuple)):
        xs = [p[2] for p in planets_data]
        ys = [p[3] for p in planets_data]
    else:
        xs = [p['x'] if isinstance(p, dict) else getattr(p,'x',50) for p in planets_data]
        ys = [p['y'] if isinstance(p, dict) else getattr(p,'y',50) for p in planets_data]
    spread = max(max(xs)-min(xs), max(ys)-min(ys))
    _MAP_SPREAD = 'tight' if spread < 55 else ('spread' if spread > 78 else 'mid')
    _MAP_DETECTED = True

def agent(obs):
    global _MAP_DETECTED

    if isinstance(obs, dict):
        player           = obs.get('player', 0)
        planets_raw      = obs.get('planets', [])
        fleets_raw       = obs.get('fleets', [])
        step             = obs.get('step', 0)
        omega            = obs.get('angular_velocity', 0.03)
        comet_planet_ids = obs.get('comet_planet_ids', [])
    else:
        player           = getattr(obs, 'player', 0)
        planets_raw      = getattr(obs, 'planets', [])
        fleets_raw       = getattr(obs, 'fleets', [])
        step             = getattr(obs, 'step', 0)
        omega            = getattr(obs, 'angular_velocity', 0.03)
        comet_planet_ids = getattr(obs, 'comet_planet_ids', [])

    _detect_map(planets_raw)

    # ── parse raw list format: planet=[id,owner,x,y,radius,ships,prod] ────
    comet_ids_set = set(comet_planet_ids)

    planets_list = []
    for p in planets_raw:
        # Raw env format is a list: [id, owner, x, y, radius, ships, prod]
        if isinstance(p, (list, tuple)):
            pid, owner, x, y, radius, ships, prod = p[:7]
        else:  # dict/Struct fallback
            pid   = p['id']   if isinstance(p, dict) else p.id
            owner = p['owner'] if isinstance(p, dict) else p.owner
            x     = p['x']    if isinstance(p, dict) else p.x
            y     = p['y']    if isinstance(p, dict) else p.y
            radius= p.get('radius', 5) if isinstance(p, dict) else getattr(p,'radius',5)
            ships = p.get('ships', 0)  if isinstance(p, dict) else getattr(p,'ships',0)
            prod  = p.get('production', p.get('prod', 1)) if isinstance(p, dict) else getattr(p,'prod',1)
        r_sq   = (x - SUN_X)**2 + (y - SUN_Y)**2
        r      = math.sqrt(r_sq)
        is_orb = (r + radius) < INNER_ORBIT_THRESHOLD
        planets_list.append({
            'id': pid, 'owner': owner, 'x': x, 'y': y,
            'radius': radius, 'ships': float(ships), 'prod': float(prod),
            'is_orb': is_orb, 'r_sq': r_sq, 'r': r,
        })

    fleets_norm = {}
    for f in fleets_raw:
        # Raw env format: [id, owner, x, y, angle, from_planet, ships]
        if isinstance(f, (list, tuple)):
            fid, fowner, fx, fy, fangle, ffrom, fships = f[:7]
        else:
            fid    = f['id']    if isinstance(f, dict) else f.id
            fowner = f['owner'] if isinstance(f, dict) else f.owner
            fx     = f['x']    if isinstance(f, dict) else f.x
            fy     = f['y']    if isinstance(f, dict) else f.y
            fangle = f.get('angle', 0) if isinstance(f, dict) else getattr(f,'angle',0)
            ffrom  = f.get('from', None) if isinstance(f, dict) else getattr(f,'from',None)
            fships = f.get('ships', 0) if isinstance(f, dict) else getattr(f,'ships',0)
        fleets_norm[fid] = {
            'id': fid, 'owner': fowner, 'x': fx, 'y': fy,
            'angle': fangle, 'from': ffrom, 'ships': float(fships),
        }

    my      = [p for p in planets_list if p['owner'] == player]
    enemy   = [p for p in planets_list if p['owner'] not in (player, -1)]
    neutrals = [p for p in planets_list if p['owner'] == -1]

    if not my:
        return []

    # ── build simulator state ─────────────────────────────────────────────
    state = _make_state(planets_list, fleets_norm, player, omega, step)

    # ── denial pre-pass: collect planets that need urgent attention ────────
    denial = _build_denial_bonus(state, neutrals)

    # ── if any denial targets exist, inject them as high-priority "must do" moves
    #    by boosting their value in the state evaluator context (done via eval already)
    #    but also force-consider them in candidate generation

    # ── run the planner ───────────────────────────────────────────────────
    best_moves = plan(state)
    if step in (5, 10, 20):
        _DEBUG_LOG.append({'step': step, 'my': len(my), 'best_moves': best_moves,
                           'denial_keys': list(_build_denial_bonus(state,neutrals).keys())[:3]})

    # ── convert plan to engine moves ──────────────────────────────────────
    planets_dict = state['planets']
    moves = []
    targeted = set()
    in_flight_to = set()
    for fl in state['fleets']:
        if fl['owner'] != player:
            continue
        in_flight_to.add(fl['target_id'])

    # Denial overrides: if a high-priority denial target is not already targeted,
    # find the closest friendly planet with ships and add it.
    launched_from = {}  # shared across denial + main passes to prevent double-spend
    denial_sorted = sorted(denial.items(), key=lambda x: -x[1])
    for deny_pid, deny_score in denial_sorted[:3]:
        if deny_pid in in_flight_to or deny_pid in targeted:
            continue
        if deny_pid not in planets_dict:
            continue
        tp = planets_dict[deny_pid]
        # Already ours?
        if tp['owner'] == player:
            continue
        best_src, best_tt = None, float('inf')
        for mp in my:
            mp_avail = mp['ships'] - launched_from.get(mp['id'], 0)
            if mp_avail < 3:
                continue
            _, _, tt = solve_intercept(mp['x'], mp['y'], tp['x'], tp['y'],
                                       tp['is_orb'], omega, int(max(mp_avail*0.5, 1)))
            if tt < best_tt:
                best_tt, best_src = tt, mp
        if best_src is None:
            continue
        src_avail = best_src['ships'] - launched_from.get(best_src['id'], 0)
        avail = src_avail * 0.55
        need  = ships_needed(tp['ships'], tp['prod'], best_tt, tp['owner'])
        if need < 3:
            need = 3
        if avail < need:
            continue
        ix, iy, _ = solve_intercept(best_src['x'], best_src['y'], tp['x'], tp['y'],
                                    tp['is_orb'], omega, int(need))
        if path_crosses_sun(best_src['x'], best_src['y'], ix, iy, 1.5):
            continue
        angle = safe_angle(best_src['x'], best_src['y'], ix, iy)
        moves.append([best_src['id'], angle, int(need)])
        targeted.add(deny_pid)
        launched_from[best_src['id']] = launched_from.get(best_src['id'], 0) + need

    # Main planned moves
    for src_id, tgt_id, ships_send in best_moves:
        if tgt_id in targeted:
            continue
        if src_id not in planets_dict or tgt_id not in planets_dict:
            continue
        src = planets_dict[src_id]
        tgt = planets_dict[tgt_id]
        # Recompute intercept with current position for accurate angle
        ix, iy, tt = solve_intercept(src['x'], src['y'], tgt['x'], tgt['y'],
                                     tgt['is_orb'], omega, int(ships_send))
        if path_crosses_sun(src['x'], src['y'], ix, iy, 1.5):
            continue
        # Use planned ship count (includes hold-buffer). Minimum sanity check only.
        min_need = ships_needed(tgt['ships'], tgt['prod'], tt, tgt['owner'])
        send = max(int(ships_send), min_need)
        cur_avail = src['ships'] - launched_from.get(src_id, 0)
        if cur_avail < min_need or send < 1:
            if step in (5, 10, 20):
                _DEBUG_LOG.append({'step': step, 'FILTER': f"{src_id}->{tgt_id} cur_avail={cur_avail:.0f} need={min_need}"})
            continue
        send = min(send, int(cur_avail))  # don't exceed what we have
        angle = safe_angle(src['x'], src['y'], ix, iy)
        moves.append([src_id, angle, int(send)])
        targeted.add(tgt_id)
        launched_from[src_id] = launched_from.get(src_id, 0) + send

    if step in (5, 10, 20):
        _DEBUG_LOG.append({'step': step, 'final_moves': moves})
    return moves
