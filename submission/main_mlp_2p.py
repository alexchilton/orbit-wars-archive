"""
main_mlp_2p.py — MLP-scored 2-player agent.

Decision engine: for every candidate (source → target) pair, compute a 16-dim
feature vector and score it with a tiny 3-layer MLP (pure numpy, no torch).
The MLP is trained offline on game outcomes (win=1, loss=0).
No forward simulation at inference → same tactical quality, 10× faster per step.

Fallback: if no MLP weights are found, falls back to fast rough-ROI scoring
          (same quality as a naive greedy agent).

Architecture:
  1. PHYSICS STATE   — parse_obs gives exact per-step state.
  2. THREAT TRIAGE   — evacuate doomed planets, reinforce threatened ones.
  3. MLP SCORING     — score all (src, tgt) proposals; execute best per source.
  4. TACTICAL HINTS  — denial urgency and counter windows add bonus to MLP score.

Action format: [planet_id, angle_radians, ships]
"""
from __future__ import annotations

import math
import os
import pickle
import sys

import numpy as np

# ── Physics engine ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from physics_sim import (
    parse_obs,
    fleet_speed, travel_time, predict_orbit,
    SUN_X, SUN_Y, SUN_RADIUS,
)

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_SPEED          = 6.0
_MAX_SPEED_MINUS_1 = MAX_SPEED - 1.0
_LOG1000           = math.log(1000.0)

TAKEOVER_MARGIN    = 1.05
DEFENSE_MARGIN     = 1.05
EVAC_FRACTION      = 0.70
REINFORCE_BUFFER   = 1.5

WAVE_SYNC_TOLERANCE = 3.0
COUNTER_RATIO       = 4.0

DENIAL_BASE_SCORE  = 75.0
DENIAL_PROD_WEIGHT = 8.0
DENIAL_URGENCY_WT  = 20.0
COUNTER_BONUS      = 50.0

# ── MLP loader (pure numpy, no torch at inference) ────────────────────────────

_MLP_MODEL: dict | None = None
_MLP_LOADED = False
_FEATURE_DIM = 16

_MLP_PATHS = [
    os.path.join(_HERE, 'mlp_scorer.pkl'),
    os.path.join(_HERE, '..', 'mlp_scorer.pkl'),
    'mlp_scorer.pkl',
    'submission/mlp_scorer.pkl',
]


def _load_mlp() -> dict | None:
    for path in _MLP_PATHS:
        try:
            with open(path, 'rb') as f:
                m = pickle.load(f)
            if 'layers' in m and 'means' in m:
                return m
        except Exception:
            pass
    return None


def _mlp_score(model: dict, x: np.ndarray) -> float:
    """Pure-numpy forward pass. Returns logit (higher = better action)."""
    h = (x - model['means']) / model['stds']
    for i, lay in enumerate(model['layers']):
        h = h @ lay['W'].T + lay['b']
        if i < len(model['layers']) - 1:
            h = np.maximum(h, 0.0)   # ReLU
    return float(h[0])


# ── Neural adjust (identical to denial agent) ─────────────────────────────────
np.random.seed(42)
HIDDEN      = 32
INPUT_SIZE  = 10
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
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def neural_adjust(step, my_pc, my_prod, my_ships, ep, es, ifc, thr):
    x = np.array([
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
        'aggression':          0.3 + out[0] * 0.7,
        'defense_sensitivity': 0.2 + out[2] * 0.3,
    }


# ── Geometry helpers ───────────────────────────────────────────────────────────

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
        if not path_crosses_sun(x1, y1, bx, by, margin) and \
                not path_crosses_sun(bx, by, x2, y2, margin):
            d = math.hypot(bx - x1, by - y1) + math.hypot(x2 - bx, y2 - by)
            if d < best_d:
                best_d, best_wp = d, (bx, by)
    if best_wp:
        return [best_wp, (x2, y2)], best_d
    return None, float('inf')


def solve_intercept(fx, fy, tx, ty, orbiting, omega, ships, iterations=25):
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


def ships_needed_for_takeover(tgt_ships, tgt_prod, tt, owner, margin=None):
    if margin is None:
        margin = TAKEOVER_MARGIN
    # Neutrals (owner == -1) do NOT produce — only owned planets do.
    if owner == -1:
        return int(tgt_ships * margin) + 1
    return int((tgt_ships + tgt_prod * tt) * margin) + 1


# ── Feature extraction for MLP ────────────────────────────────────────────────

def action_features(state: dict, src_pid: int, tgt_pid: int,
                    eta: float, ships: float, needed: float, me: int) -> np.ndarray:
    """
    16-dim feature vector for a (src, tgt) action proposal.

    Must match train_mlp_scorer.py exactly.
    """
    src = state['planets'][src_pid]
    tgt = state['planets'][tgt_pid]

    my_pids = [pid for pid, p in state['planets'].items() if p['owner'] == me]
    ep_pids = [pid for pid, p in state['planets'].items()
               if p['owner'] >= 0 and p['owner'] != me]
    my_prod  = sum(state['planets'][p]['prod'] for p in my_pids)
    ep_prod  = sum(state['planets'][p]['prod'] for p in ep_pids)
    my_ships = sum(state['planets'][p]['ships'] for p in my_pids)
    ep_ships = sum(state['planets'][p]['ships'] for p in ep_pids)
    avail    = src['ships']

    glob = [
        state['step'] / 400.0,
        my_prod / 20.0,
        ep_prod / 20.0,
        my_ships / 2000.0,
        ep_ships / 2000.0,
        (len(my_pids) - len(ep_pids)) / 25.0,
    ]
    act = [
        src['prod'] / 10.0,
        avail / 500.0,
        tgt['prod'] / 10.0,
        tgt['ships'] / 500.0,
        1.0 if tgt['owner'] >= 0 else 0.0,
        eta / 50.0,
        needed / 500.0,
        ships / 500.0,
        (my_prod - ep_prod) / 20.0,
        (avail - needed) / 500.0,
    ]
    return np.array(glob + act, dtype=np.float32)


# ── Tactical analysis (same as pred agent) ────────────────────────────────────

def analyze_threats(state):
    me = state['me']
    threats = {}
    my_pids = {pid for pid, p in state['planets'].items() if p['owner'] == me}
    for fl in state['fleets']:
        if fl['owner'] == me:
            continue
        tpid = fl['target_pid']
        if tpid not in my_pids:
            continue
        tgt = state['planets'][tpid]
        garrison = tgt['ships'] + tgt['prod'] * round(fl['eta'])
        is_doomed = (fl['ships'] * DEFENSE_MARGIN > garrison)
        if tpid not in threats:
            threats[tpid] = {
                'total_enemy':  0.0,
                'is_doomed':    False,
                'earliest_eta': fl['eta'],
            }
        threats[tpid]['total_enemy'] += fl['ships']
        if fl['eta'] < threats[tpid]['earliest_eta']:
            threats[tpid]['earliest_eta'] = fl['eta']
        threats[tpid]['is_doomed'] |= is_doomed
    return threats


def analyze_counter_windows(state, enemy_pids):
    counter = set()
    for pid in enemy_pids:
        p = state['planets'][pid]
        if p['ships'] < COUNTER_RATIO * p['prod']:
            counter.add(pid)
    return counter


def analyze_denial(state, neutral_pids, enemy_pids, me):
    denial_map = {}
    omega = state['omega']
    for npid in neutral_pids:
        np_p = state['planets'][npid]
        best_ep_eta = float('inf')
        for epid in enemy_pids:
            ep = state['planets'][epid]
            _, _, ep_eta = solve_intercept(ep['x'], ep['y'],
                                           np_p['x'], np_p['y'],
                                           np_p['is_orb'], omega,
                                           max(1, int(ep['ships'] * 0.5)))
            if ep_eta < best_ep_eta:
                best_ep_eta = ep_eta
        if best_ep_eta < float('inf'):
            prod_bonus = DENIAL_BASE_SCORE + np_p['prod'] * DENIAL_PROD_WEIGHT
            urgency    = DENIAL_URGENCY_WT / max(1.0, best_ep_eta)
            denial_map[npid] = (prod_bonus + urgency, best_ep_eta)
    return denial_map


# ── Main agent ─────────────────────────────────────────────────────────────────

def agent(obs):  # noqa: C901
    global _MLP_MODEL, _MLP_LOADED

    # Lazy-load MLP once
    if not _MLP_LOADED:
        _MLP_MODEL = _load_mlp()
        _MLP_LOADED = True

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

    my    = [state['planets'][pid] for pid in my_pids]
    enemy = [state['planets'][pid] for pid in enemy_pids]

    my_ships    = sum(p['ships'] for p in my)
    my_prod     = sum(p['prod'] for p in my)
    enemy_ships = sum(p['ships'] for p in enemy) if enemy else 0.0
    enemy_prod  = sum(p['prod'] for p in enemy)  if enemy else 0.0
    prod_ratio  = my_prod  / max(enemy_prod,  1.0)
    ship_ratio  = my_ships / max(enemy_ships, 1.0)

    my_inflight_targets = {fl['target_pid'] for fl in state['fleets'] if fl['owner'] == me}
    in_flight_count     = sum(1 for fl in state['fleets'] if fl['owner'] == me)

    threats         = analyze_threats(state)
    threats_total   = sum(t['total_enemy'] for t in threats.values())
    doomed_pids     = {pid for pid, t in threats.items() if t['is_doomed']}
    threatened_pids = {pid for pid, t in threats.items() if not t['is_doomed']}
    counter_windows = analyze_counter_windows(state, enemy_pids)
    denial_map      = analyze_denial(state, neutral_pids, enemy_pids, me)

    # neural_adjust weights not available — skip (output not used downstream)
    # adj = neural_adjust(step, len(my_pids), my_prod, my_ships,
    #                     enemy_prod, enemy_ships, in_flight_count, threats_total)

    moves    = []
    targeted = set()
    launched = {}

    def avail(src_pid):
        return state['planets'][src_pid]['ships'] - launched.get(src_pid, 0)

    def send_move(src_pid, tgt_pid, n_ships):
        n_ships = int(n_ships)
        if n_ships < 1:
            return False
        src = state['planets'][src_pid]
        tgt = state['planets'][tgt_pid]
        ix, iy, _ = solve_intercept(src['x'], src['y'], tgt['x'], tgt['y'],
                                     tgt['is_orb'], omega, n_ships)
        if path_crosses_sun(src['x'], src['y'], ix, iy):
            wp, _ = multi_leg_path(src['x'], src['y'], ix, iy)
            if wp is None:
                return False
            ix, iy = wp[0]
        angle = safe_angle(src['x'], src['y'], ix, iy)
        moves.append([src_pid, angle, n_ships])
        launched[src_pid] = launched.get(src_pid, 0) + n_ships
        return True

    # ── 1. Evacuation ─────────────────────────────────────────────────────────
    if not (prod_ratio > 3.0 and ship_ratio > 2.0):
        for pid in doomed_pids:
            av = avail(pid)
            if av < 5:
                continue
            evac = int(av * EVAC_FRACTION)
            if evac < 5:
                continue
            src = state['planets'][pid]
            safe_dsts = [dpid for dpid in my_pids if dpid != pid and dpid not in doomed_pids]
            if not safe_dsts:
                continue
            best_dst = min(safe_dsts,
                           key=lambda dp: math.hypot(state['planets'][dp]['x'] - src['x'],
                                                     state['planets'][dp]['y'] - src['y']))
            send_move(pid, best_dst, evac)

    # ── 2. Reinforcement ──────────────────────────────────────────────────────
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
            _, _, our_eta = solve_intercept(src['x'], src['y'],
                                            tgt_data['x'], tgt_data['y'],
                                            tgt_data['is_orb'], omega, max(1, int(av * 0.5)))
            if our_eta < eta_enemy - REINFORCE_BUFFER:
                send = max(5, min(int(deficit * 1.2), int(av * 0.6)))
                if av >= send:
                    send_move(src_pid, pid, send)
                break

    # ── 3. MLP scoring — evaluate all (src, tgt) pairs ────────────────────────
    if enemy_pids and (ship_ratio > 1.5 or prod_ratio > 1.5 or step > 30):
        tgt_set = enemy_pids | neutral_pids
    else:
        tgt_set = neutral_pids

    # Evaluate ALL planets with enough ships (not just top 3)
    src_planets = sorted(
        [pid for pid in my_pids if pid not in doomed_pids],
        key=lambda p: -state['planets'][p]['ships']
    )

    for src_pid in src_planets:
        av = avail(src_pid)
        if av < 10:
            continue
        src = state['planets'][src_pid]

        open_targets = [p for p in tgt_set
                        if p not in targeted and p not in my_inflight_targets]
        if not open_targets:
            break   # all interesting targets already hit this turn

        best_score  = -1e9
        best_action = None   # (tgt_pid, ships, ix, iy)

        for tgt_pid in open_targets:
            tgt = state['planets'][tgt_pid]
            try:
                ix, iy, eta = solve_intercept(
                    src['x'], src['y'], tgt['x'], tgt['y'],
                    tgt['is_orb'], omega, int(av)
                )
            except Exception:
                continue
            if path_crosses_sun(src['x'], src['y'], ix, iy):
                wp, _ = multi_leg_path(src['x'], src['y'], ix, iy)
                if wp is None:
                    continue

            needed = ships_needed_for_takeover(tgt['ships'], tgt['prod'], eta, tgt['owner'])
            if av < needed:
                continue

            ships = needed if tgt['owner'] < 0 else min(
                max(needed, int(av * 0.65)), int(av * 0.90)
            )
            if ships < 1:
                continue

            # ── Score this proposal ────────────────────────────────────────
            if _MLP_MODEL is not None:
                feats = action_features(state, src_pid, tgt_pid, eta, ships, needed, me)
                score = _mlp_score(_MLP_MODEL, feats)
            else:
                # Fallback: rough production ROI (no sim, no MLP)
                score = tgt['prod'] * max(0.0, 30.0 - eta) - needed * 0.1

            # Tactical urgency bonuses (break score ties deterministically)
            if tgt_pid in denial_map and tgt['owner'] < 0:
                _, enemy_eta = denial_map[tgt_pid]
                if eta <= enemy_eta:
                    score += denial_map[tgt_pid][0] * 0.001  # tiny tie-break
            if tgt_pid in counter_windows:
                score += COUNTER_BONUS * 0.001

            if score > best_score:
                best_score  = score
                best_action = (tgt_pid, ships, ix, iy)

        if best_action is None:
            continue

        tgt_pid, ships, ix, iy = best_action
        if avail(src_pid) >= ships >= 1:
            if send_move(src_pid, tgt_pid, ships):
                targeted.add(tgt_pid)

    return moves


if __name__ == '__main__':
    print("main_mlp_2p: MLP-scored 2p agent")
