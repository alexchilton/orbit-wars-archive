"""
Orbit Wars - prodexpand: production-weighted early expansion
Key changes on top of v131_plus_2p:
- In turns 0-80, heavily weight expansion targets by production (compounds massively)
- Track top-3 highest-production neutral planets as anchors with permanent bonus
- In turns 0-120, also weight hostile planet production for attack scoring
"""
import os
os.environ['KAGGLE_ENVELOPES'] = '0'

import math
import numpy as np
import pickle

SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS = 10.0
MAX_SPEED = 6.0
DECOY_THRESHOLD = int(os.environ.get('V131_DECOY_THRESHOLD', '8'))
INNER_ORBIT_THRESHOLD = 48.0
_MAX_SPEED_MINUS_1 = MAX_SPEED - 1.0
_LOG1000 = math.log(1000.0)

np.random.seed(42)

HIDDEN = 32
INPUT_SIZE = 18
OUTPUT_SIZE = 10
DUEL_PRESSURE_STEP = int(os.environ.get('V131_PRESSURE_STEP', '28'))
DUEL_PRESSURE_PROD_RATIO = float(os.environ.get('V131_PRESSURE_PROD_RATIO', '0.95'))
DUEL_PRESSURE_SHIP_RATIO = float(os.environ.get('V131_PRESSURE_SHIP_RATIO', '0.90'))
DUEL_CLEANUP_STEP = int(os.environ.get('V131_CLEANUP_STEP', '110'))
DUEL_CLEANUP_MAX_ENEMY_PLANETS = 4
DUEL_CLEANUP_PROD_RATIO = float(os.environ.get('V131_CLEANUP_PROD_RATIO', '1.9'))
DUEL_CLEANUP_SHIP_RATIO = float(os.environ.get('V131_CLEANUP_SHIP_RATIO', '1.8'))

# Map-adaptive globals (set on first call)
_MAP_SPREAD = 'mid'  # 'tight', 'mid', 'spread'
_TAKEOVER_MARGIN = float(os.environ.get('V131_TAKEOVER_MARGIN', '1.05'))
_SEND_FRACTION = 0.75  # fraction of available ships to commit
_MAP_DETECTED = False

# Scoring constants (env var overridable)
_SCORE_PROD_WEIGHT = float(os.environ.get('V131_SCORE_PROD_WEIGHT', '18'))
_SCORE_TT_PENALTY = float(os.environ.get('V131_SCORE_TT_PENALTY', '3.5'))
_NEUTRAL_BONUS = float(os.environ.get('V131_NEUTRAL_BONUS', '25'))
_AGGRO_HOSTILE_BONUS = float(os.environ.get('V131_AGGRO_HOSTILE_BONUS', '35'))
_PRESSURE_HOSTILE_BONUS = float(os.environ.get('V131_PRESSURE_HOSTILE_BONUS', '26'))
_CLEANUP_HOSTILE_BONUS = float(os.environ.get('V131_CLEANUP_HOSTILE_BONUS', '55'))

# Send fractions per phase (env var overridable)
_SEND_SMASH = float(os.environ.get('V131_SEND_SMASH', '0.9'))
_SEND_CLEANUP = float(os.environ.get('V131_SEND_CLEANUP', '0.82'))
_SEND_PRESSURE = float(os.environ.get('V131_SEND_PRESSURE', '0.55'))
_SEND_RUSH = float(os.environ.get('V131_SEND_RUSH', '0.8'))
_SEND_AGGRESSIVE = float(os.environ.get('V131_SEND_AGGRESSIVE', '0.55'))
_SEND_DOMINATE = float(os.environ.get('V131_SEND_DOMINATE', '0.5'))

# Comet tuning (env var overridable)
_CURG_MULT = float(os.environ.get('V131_CURG_MULT', '1.12'))
_COMET_LOOKAHEAD = int(os.environ.get('V131_COMET_LOOKAHEAD', '28'))
_COMET_SCORE_BASE = float(os.environ.get('V131_COMET_SCORE_BASE', '100.0'))
_COMET_DIST_PENALTY = float(os.environ.get('V131_COMET_DIST_PENALTY', '2.0'))

# Production-weighted expansion constants
PROD_EXPAND_BOOST = 2.5
ANCHOR_PROD_BONUS = 12.0
MAX_PROD = 6.0
PROD_BONUS_PER_UNIT = 4.0   # Fixed: additive bonus per unit of production

# Module-level state: top-prod anchor planets (set once per game)
_ANCHOR_PLANETS_PE = None


def _early_game_prod_weight(step, planet_prod):
    """Returns score multiplier: highest early game, fades by step 80."""
    phase_factor = max(0.0, 1.0 - step / 80.0)
    return 1.0 + phase_factor * (planet_prod / MAX_PROD) * PROD_EXPAND_BOOST

# Target selector weights (trained on winning games from top agents, v2 with more training)
try:
    with open('TargetStriker/winning_target_selector_v2_weights.pkl', 'rb') as f:
        TARGET_SELECTOR_WEIGHTS = pickle.load(f)
    print("v131: Loaded winning_target_selector_v2 weights from TargetStriker")
except Exception as e:
    print(f"v131: Could not load winning_target_selector_v2 ({e})")
    TARGET_SELECTOR_WEIGHTS = None

try:
    with open('v90_weights.pkl', 'rb') as f:
        WEIGHTS = pickle.load(f)
    print("v131: Loaded v90 weights")
except Exception as e:
    print(f"v131: Could not load v90_weights ({e}), trying v89_weights")
    try:
        with open('v89_weights.pkl', 'rb') as f:
            WEIGHTS = pickle.load(f)
        print("v131: Loaded v89 weights as base")
    except Exception as e2:
        print(f"v131: Could not load v89 weights either ({e2}), using defaults")
        WEIGHTS = {
            'W1': np.random.randn(18, HIDDEN) * 0.3,
            'B1': np.zeros(HIDDEN),
            'W2': np.random.randn(HIDDEN, OUTPUT_SIZE) * 0.3,
            'B2': np.zeros(OUTPUT_SIZE),
        }


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-8)


def neural_adjust(step, my_planet_count, my_prod, my_ships, enemy_prod, enemy_ships,
                  in_flight_count, threats_total):
    x = np.array([
        min(step / 400.0, 1.0),
        my_planet_count / 8.0,
        min(my_prod / 20.0, 1.0),
        min(my_ships / 200.0, 1.0),
        min(enemy_prod / 20.0, 1.0) if enemy_prod > 0 else 0.0,
        min(enemy_ships / 200.0, 1.0) if enemy_ships > 0 else 0.0,
        in_flight_count / 10.0,
        threats_total / 100.0,
        min(my_prod / max(enemy_prod, 0.1), 3.0) / 3.0,
        min(my_ships / max(enemy_ships, 1.0), 5.0) / 5.0,
        1.0 if my_planet_count >= 3 else 0.0,
        1.0 if my_prod > enemy_prod else 0.0,
        1.0 if my_ships > enemy_ships else 0.0,
        min(my_ships / max(enemy_ships, 1.0), 5.0) / 5.0,
        min(enemy_prod / max(my_prod, 0.1), 3.0) / 3.0,
        1.0 if threats_total > my_ships * 0.2 else 0.0,
        min(in_flight_count / 5.0, 1.0),
        1.0 if step < 50 else 0.0,
    ], dtype=np.float32)

    h = sigmoid(np.dot(x, WEIGHTS['W1']) + WEIGHTS['B1'])
    out = sigmoid(np.dot(h, WEIGHTS['W2']) + WEIGHTS['B2'])
    return {
        'aggression': 0.3 + out[0] * 0.7,
        'expand_thresh': 3.0 + out[1] * 4.0,
        'defense_sensitivity': 0.2 + out[2] * 0.3,
        'capture_bonus_scale': 0.5 + out[3] * 1.5,
        'extra_0': out[4],
        'extra_1': out[5],
        'extra_2': out[6],
        'extra_3': out[7],
        'extra_4': out[8],
        'extra_5': out[9],
    }


def batch_norm_affine(x, mean, var, weight, bias, eps=1e-5):
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * weight + bias


def deep_nn_forward(x, w):
    if w is None:
        return 0.5
    x = np.dot(x, w['net.0.weight'].T) + w['net.0.bias']
    x = batch_norm_affine(x, w['net.1.running_mean'], w['net.1.running_var'],
                          w['net.1.weight'], w['net.1.bias'])
    x = np.maximum(x, 0)

    x = np.dot(x, w['net.3.weight'].T) + w['net.3.bias']
    x = batch_norm_affine(x, w['net.4.running_mean'], w['net.4.running_var'],
                          w['net.4.weight'], w['net.4.bias'])
    x = np.maximum(x, 0)

    x = np.dot(x, w['net.6.weight'].T) + w['net.6.bias']
    x = batch_norm_affine(x, w['net.7.running_mean'], w['net.7.running_var'],
                          w['net.7.weight'], w['net.7.bias'])
    x = np.maximum(x, 0)

    x = np.dot(x, w['net.9.weight'].T) + w['net.9.bias']
    x = np.maximum(x, 0)

    x = np.dot(x, w['net.11.weight'].T) + w['net.11.bias']
    x = 1.0 / (1.0 + np.exp(-x))
    return float(x[0])


def target_selector_forward(x, w):
    """Numpy inference for target selector (13 -> 128 -> 128 -> 64 -> 32 -> 1)"""
    if w is None:
        return 0.5
    # Layer 1: Linear + BatchNorm + ReLU + Dropout
    x = np.dot(x, w['net.0.weight'].T) + w['net.0.bias']
    x = batch_norm_affine(x, w['net.1.running_mean'], w['net.1.running_var'],
                          w['net.1.weight'], w['net.1.bias'])
    x = np.maximum(x, 0)

    # Layer 2: Linear + BatchNorm + ReLU + Dropout
    x = np.dot(x, w['net.4.weight'].T) + w['net.4.bias']
    x = batch_norm_affine(x, w['net.5.running_mean'], w['net.5.running_var'],
                          w['net.5.weight'], w['net.5.bias'])
    x = np.maximum(x, 0)

    # Layer 3: Linear + BatchNorm + ReLU + Dropout
    x = np.dot(x, w['net.8.weight'].T) + w['net.8.bias']
    x = batch_norm_affine(x, w['net.9.running_mean'], w['net.9.running_var'],
                          w['net.9.weight'], w['net.9.bias'])
    x = np.maximum(x, 0)

    # Layer 4: Linear + ReLU
    x = np.dot(x, w['net.12.weight'].T) + w['net.12.bias']
    x = np.maximum(x, 0)

    # Layer 5: Linear (output)
    x = np.dot(x, w['net.14.weight'].T) + w['net.14.bias']
    x = 1.0 / (1.0 + np.exp(-x))
    return float(x[0])


def get_target_selector_features(src, tgt, step, n_my_planets, my_prod, enemy_prod):
    """Extract 13 features for target selection model."""
    dist = math.hypot(tgt['x'] - src['x'], tgt['y'] - src['y'])
    return np.array([
        min(src['ships'] / 100.0, 1.0),
        src['prod'] / 5.0,
        1.0 if src.get('is_orb', False) else 0.0,
        tgt['prod'] / 10.0,
        tgt['radius'] / 10.0,
        tgt['ships'] / 200.0,
        dist / 100.0,
        1.0 if tgt['owner'] == -1 else 0.0,
        1.0 if tgt['owner'] >= 0 else 0.0,
        1.0 if tgt.get('is_orb', False) else 0.0,
        step / 400.0,
        min(n_my_planets / 20.0, 1.0),
        my_prod / max(enemy_prod, 1.0),
    ], dtype=np.float32)


def get_deep_nn_features(src, tgt, step, n_my_planets, needed, dist):
    return np.array([
        min(src['ships'] / 100.0, 1.0),
        src['prod'] / 5.0,
        1.0 if src.get('is_orb', False) else 0.0,
        min(tgt['ships'] / 200.0, 1.0),
        tgt['prod'] / 5.0,
        1.0 if tgt.get('is_orb', False) else 0.0,
        min(dist / 100.0, 1.0),
        1.0 if tgt['owner'] == -1 else 0.0,
        step / 400.0,
        min(n_my_planets / 20.0, 1.0),
        min(needed / 100.0, 1.0),
    ], dtype=np.float32)


def fleet_speed(ships: int) -> float:
    if ships <= 0:
        return 1.0
    return 1.0 + _MAX_SPEED_MINUS_1 * (math.log(max(ships, 1)) / _LOG1000) ** 1.5


def travel_time(x1: float, y1: float, x2: float, y2: float, ships: int) -> float:
    dx = x2 - x1
    dy = y2 - y1
    if ships <= 0:
        return 999.0
    return math.sqrt(dx*dx + dy*dy) / fleet_speed(ships)


def line_seg_min_dist_sq(x1: float, y1: float, x2: float, y2: float, px: float, py: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0.0:
        return (x1 - px) ** 2 + (y1 - py) ** 2
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    dx_t = x1 + t * dx - px
    dy_t = y1 + t * dy - py
    return dx_t * dx_t + dy_t * dy_t


def path_crosses_sun(x1: float, y1: float, x2: float, y2: float, margin: float = 1.5) -> bool:
    margin_sq = (SUN_RADIUS + margin) ** 2
    return line_seg_min_dist_sq(x1, y1, x2, y2, SUN_X, SUN_Y) < margin_sq


def predict_orbit(x: float, y: float, omega: float, dt: float):
    theta = math.atan2(y - SUN_Y, x - SUN_X)
    r = math.hypot(x - SUN_X, y - SUN_Y)
    return SUN_X + r * math.cos(theta + omega * dt), SUN_Y + r * math.sin(theta + omega * dt)


def solve_intercept(fx: float, fy: float, tx: float, ty: float, orbiting: bool, omega: float, ships: int, iterations: int = 25):
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


def safe_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    direct = math.atan2(y2 - y1, x2 - x1)
    if not path_crosses_sun(x1, y1, x2, y2, margin=1.5):
        return direct
    d_sq = (x1 - SUN_X) ** 2 + (y1 - SUN_Y) ** 2
    d = math.sqrt(d_sq)
    if d <= SUN_RADIUS + 1.0:
        return direct
    half = math.asin(min(1.0, (SUN_RADIUS + 1.0) / d))
    to_sun = math.atan2(SUN_Y - y1, SUN_X - x1)
    cw = to_sun + half
    ccw = to_sun - half
    def adiff(a):
        dd = (a - direct) % (2 * math.pi)
        return dd if dd < math.pi else 2 * math.pi - dd
    return cw if adiff(cw) < adiff(ccw) else ccw


def is_decoy_fleet(fleet, planet_nearest, planet_radii, planets):
    if fleet['ships'] < DECOY_THRESHOLD:
        return True
    tgt_id = planet_nearest.get(fleet['id'])
    if tgt_id is None:
        return True
    tgt = planets.get(tgt_id)
    if tgt is None:
        return True
    if fleet['ships'] < (tgt['ships'] + 1) * 0.4:
        return True
    return False


def ships_needed_for_takeover(tgt_ships, tgt_prod, tt, owner, margin=None):
    if margin is None:
        margin = _TAKEOVER_MARGIN
    if owner == -1:
        return int(tgt_ships * margin) + 1
    growth = tgt_prod * tt
    return int((tgt_ships + growth) * margin) + 1


def planet_under_threat(p_id, fleets, planets, player, omega, planet_nearest, fleet_target_map):
    incoming = 0
    p = planets[p_id]
    p_x, p_y = p['x'], p['y']
    p_r = math.sqrt((p_x - SUN_X) ** 2 + (p_y - SUN_Y) ** 2)
    is_orbiting_p = (p_r + p['radius']) < INNER_ORBIT_THRESHOLD

    for f in fleets.values():
        if f['owner'] == player:
            continue
        tgt_id = fleet_target_map.get(f['id'])
        if tgt_id is None or tgt_id == p_id:
            continue
        if tgt_id == f['from']:
            continue

        f_x, f_y = f['x'], f['y']
        f_ships = int(f['ships'])

        if is_orbiting_p:
            tt = travel_time(f_x, f_y, p_x, p_y, f_ships)
            ix, iy = predict_orbit(p_x, p_y, omega, tt)
            d_sq = (ix - p_x) ** 2 + (iy - p_y) ** 2
        else:
            dx = f_x - p_x
            dy = f_y - p_y
            d_sq = dx * dx + dy * dy

        if d_sq < 2500:
            incoming += f['ships']
    return incoming


def compute_tangent_points(x1: float, y1: float, margin: float = 2.0):
    d_sq = (x1 - SUN_X) ** 2 + (y1 - SUN_Y) ** 2
    d = math.sqrt(d_sq)
    if d <= SUN_RADIUS + margin:
        return None, None
    half_angle = math.asin(min(1.0, (SUN_RADIUS + margin) / d))
    to_sun = math.atan2(SUN_Y - y1, SUN_X - x1)
    return to_sun + half_angle, to_sun - half_angle


def multi_leg_path(x1: float, y1: float, x2: float, y2: float, margin: float = 2.0):
    if not path_crosses_sun(x1, y1, x2, y2, margin):
        return [(x2, y2)], math.hypot(x2 - x1, y2 - y1)
    beacon_ring = SUN_RADIUS + 15.0
    waypoints = []
    for angle in [0, math.pi/2, math.pi, 3*math.pi/2]:
        bx = SUN_X + beacon_ring * math.cos(angle)
        by = SUN_Y + beacon_ring * math.sin(angle)
        if not path_crosses_sun(x1, y1, bx, by, margin) and not path_crosses_sun(bx, by, x2, y2, margin):
            waypoints.append((bx, by))
    if not waypoints:
        return None, float('inf')
    best_wp = None
    best_dist = float('inf')
    for wx, wy in waypoints:
        d = math.hypot(wx - x1, wy - y1) + math.hypot(x2 - wx, y2 - wy)
        if d < best_dist:
            best_dist = d
            best_wp = (wx, wy)
    if best_wp:
        return [best_wp, (x2, y2)], best_dist
    return None, float('inf')


def estimate_capture_bonus(src_x: float, src_y: float, planet, omega: float, ships: int, scale: float = 1.0) -> float:
    r_sq = (planet['x'] - SUN_X) ** 2 + (planet['y'] - SUN_Y) ** 2
    r = math.sqrt(r_sq)
    if (r + planet['radius']) >= INNER_ORBIT_THRESHOLD:
        return 0.0
    if not path_crosses_sun(src_x, src_y, planet['x'], planet['y'], margin=2.0):
        return 3.0 * scale

    p_x, p_y = planet['x'], planet['y']
    safe_count = 0
    for offset in range(-6, 7):
        fx, fy = predict_orbit(p_x, p_y, omega, offset)
        if not path_crosses_sun(src_x, src_y, fx, fy, margin=2.0):
            safe_count += 1
    return (safe_count / 13.0) * 5.0 * scale


def _detect_map(planets_data):
    """Detect map geometry on first call and set adaptive globals."""
    global _MAP_DETECTED, _MAP_SPREAD, _TAKEOVER_MARGIN, _SEND_FRACTION
    global DUEL_PRESSURE_STEP
    if _MAP_DETECTED or not planets_data or len(planets_data) < 4:
        return
    _MAP_DETECTED = True

    coords = []
    for p in planets_data:
        if len(p) >= 4:
            coords.append((float(p[2]), float(p[3])))
    if len(coords) < 4:
        return

    total_dist = 0.0
    n_pairs = 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            total_dist += (dx * dx + dy * dy) ** 0.5
            n_pairs += 1
    avg_pair_dist = total_dist / max(1, n_pairs)

    if avg_pair_dist < 59.0:
        _MAP_SPREAD = 'tight'
        _TAKEOVER_MARGIN = 1.03  # Less margin needed, targets close
        _SEND_FRACTION = 0.80   # Commit more, can reinforce quickly
        DUEL_PRESSURE_STEP = 22  # Pressure earlier on tight maps
    elif avg_pair_dist > 61.0:
        _MAP_SPREAD = 'spread'
        _TAKEOVER_MARGIN = 1.08  # More margin, targets grow during long travel
        _SEND_FRACTION = 0.70   # More conservative, can't reinforce
        DUEL_PRESSURE_STEP = 32  # Pressure later on spread maps
    else:
        _MAP_SPREAD = 'mid'
        _TAKEOVER_MARGIN = 1.05
        _SEND_FRACTION = 0.75
        DUEL_PRESSURE_STEP = 28


def agent(obs):
    if isinstance(obs, dict):
        player = obs.get('player', 0)
        planets_data = obs.get('planets', [])
        fleets_data = obs.get('fleets', [])
        step = obs.get('step', 0)
        omega = obs.get('angular_velocity', 0.03)
        comets_data = obs.get('comets', [])
        comet_planet_ids = obs.get('comet_planet_ids', [])
    else:
        player = getattr(obs, 'player', 0)
        planets_data = getattr(obs, 'planets', [])
        fleets_data = getattr(obs, 'fleets', [])
        step = getattr(obs, 'step', 0)
        omega = getattr(obs, 'angular_velocity', 0.03)
        comets_data = getattr(obs, 'comets', [])
        comet_planet_ids = getattr(obs, 'comet_planet_ids', [])

    _detect_map(planets_data)

    comet_ids_set = set(comet_planet_ids)

    planets = {}
    for p in planets_data:
        pid, owner, x, y, radius, ships, prod = p[:7]
        r_sq = (x - SUN_X) ** 2 + (y - SUN_Y) ** 2
        r = math.sqrt(r_sq)
        is_comet = pid in comet_ids_set
        planets[pid] = {
            'id': pid, 'owner': owner, 'x': x, 'y': y,
            'radius': radius, 'ships': float(ships), 'prod': float(prod),
            'is_orb': (r + radius) < INNER_ORBIT_THRESHOLD,
            'r_sq': r_sq, 'r': r, 'is_comet': is_comet
        }

    fleets = {}
    fleet_nearest_planet = {}
    for f in fleets_data:
        fleets[f[0]] = {
            'id': f[0], 'owner': f[1], 'x': f[2], 'y': f[3],
            'angle': f[4], 'from': f[5], 'ships': float(f[6])
        }
        best_dist_sq = float('inf')
        best_pid = None
        f_x, f_y = f[2], f[3]
        for pid, p in planets.items():
            dx = f_x - p['x']
            dy = f_y - p['y']
            d_sq = dx * dx + dy * dy
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_pid = pid
        fleet_nearest_planet[f[0]] = best_pid

    my = [p for p in planets.values() if p['owner'] == player]
    if not my:
        return []

    enemy = [p for p in planets.values() if p['owner'] != player and p['owner'] != -1]
    neutrals = [p for p in planets.values() if p['owner'] == -1]

    # PRODEXPAND: initialize anchor planets on first call (top-3 highest-prod neutrals)
    global _ANCHOR_PLANETS_PE
    if _ANCHOR_PLANETS_PE is None and neutrals:
        _sorted_neutrals = sorted(neutrals, key=lambda p: p['prod'], reverse=True)
        _ANCHOR_PLANETS_PE = {p['id'] for p in _sorted_neutrals[:3]}

    my_prod = sum(p['prod'] for p in my)
    my_ships = sum(p['ships'] for p in my)
    enemy_prod = sum(p['prod'] for p in enemy) if enemy else 0
    enemy_ships = sum(p['ships'] for p in enemy) if enemy else 0
    enemy_planet_count = len(enemy)

    prod_ratio = my_prod / enemy_prod if enemy_prod > 0 else 999
    ship_ratio = my_ships / enemy_ships if enemy_ships > 0 else 999

    my_planet_count = len(my)
    neighbor_count = 0
    for t in neutrals:
        t_x, t_y = t['x'], t['y']
        for p in my:
            dx = t_x - p['x']
            dy = t_y - p['y']
            if dx * dx + dy * dy < 1225:
                neighbor_count += 1
                break

    nearby_larger_planets = []
    for src in my:
        src_x, src_y = src['x'], src['y']
        for t in (neutrals + enemy):
            dx = t['x'] - src_x
            dy = t['y'] - src_y
            d_sq = dx * dx + dy * dy
            if d_sq < 1600 and d_sq >= 0:
                d = math.sqrt(d_sq)
                if d < 40 and t['prod'] >= src['prod'] * 0.8 and t['radius'] >= src['radius'] * 0.8:
                    nearby_larger_planets.append((src['id'], t['id'], d))

    real_enemy_fleets = {f_id: f for f_id, f in fleets.items() if f['owner'] != player and not is_decoy_fleet(f, fleet_nearest_planet, None, planets)}

    in_flight_from = set()
    in_flight_to = set()
    in_flight_count = 0
    fleet_target_map = {}
    # v131: Track in-flight ships FROM each planet
    in_flight_ships_from = {}
    
    for f in fleets.values():
        if f['owner'] == player and f['from'] is not None:
            in_flight_from.add(f['from'])
            in_flight_count += 1
            # Track in-flight ships per source planet
            if f['from'] not in in_flight_ships_from:
                in_flight_ships_from[f['from']] = 0
            in_flight_ships_from[f['from']] += f['ships']
            
            best_tgt, best_d_sq = None, float('inf')
            f_x, f_y = f['x'], f['y']
            for p in planets.values():
                if p['id'] == f['from']:
                    continue
                dx = f_x - p['x']
                dy = f_y - p['y']
                d_sq = dx * dx + dy * dy
                if d_sq < best_d_sq:
                    best_d_sq = d_sq
                    best_tgt = p['id']
            if best_tgt:
                in_flight_to.add(best_tgt)
                fleet_target_map[f['id']] = best_tgt

    threats = {}
    threats_total = 0
    for p in planets.values():
        if p['owner'] == player:
            th = planet_under_threat(p['id'], fleets, planets, player, omega, fleet_nearest_planet, fleet_target_map)
            threats[p['id']] = th
            threats_total += th

    smash_targets = set()
    if my_planet_count >= 2:  # Need at least 2 planets before smash
        for e in enemy:
            e_x, e_y = e['x'], e['y']
            nearby_my_ships = 0
            for p in my:
                dx = p['x'] - e_x
                dy = p['y'] - e_y
                if dx * dx + dy * dy < 2500:
                    nearby_my_ships += p['ships']
            if nearby_my_ships > e['ships'] * 1.25:  # Was 0.95 — too eager at parity
                smash_targets.add(e['id'])

    adj = neural_adjust(step, my_planet_count, my_prod, my_ships, enemy_prod, enemy_ships, in_flight_count, threats_total)
    aggression = adj['aggression']
    expand_thresh = adj['expand_thresh']
    defense_sens = adj['defense_sensitivity']
    capture_scale = adj['capture_bonus_scale']

    smash_targets_open = smash_targets - in_flight_to
    if smash_targets_open:
        phase = 'smash'
    elif (
        enemy
        and step >= DUEL_CLEANUP_STEP
        and enemy_planet_count <= DUEL_CLEANUP_MAX_ENEMY_PLANETS
        and my_prod > max(enemy_prod, 1.0) * DUEL_CLEANUP_PROD_RATIO
        and my_ships > max(enemy_ships, 1.0) * DUEL_CLEANUP_SHIP_RATIO
    ):
        phase = 'cleanup'
    elif (
        enemy
        and step >= DUEL_PRESSURE_STEP
        and my_planet_count >= 3
        and my_prod >= max(enemy_prod, 1.0) * DUEL_PRESSURE_PROD_RATIO
        and my_ships >= max(enemy_ships, 1.0) * DUEL_PRESSURE_SHIP_RATIO
    ):
        phase = 'pressure'
    elif my_ships > 120 and my_planet_count < 4 and enemy:
        phase = 'rush'
    elif my_planet_count < 3 or (neighbor_count > 0 and my_planet_count < 5):
        phase = 'expand'
    elif threats and any(t > my_ships * defense_sens for t in threats.values()):
        phase = 'counter_attack'
    elif prod_ratio > 4 and my_ships > 80 and my_planet_count >= 3:
        phase = 'crush'
    elif prod_ratio > 2.0 or ship_ratio > 2.5:
        phase = 'aggressive'
    elif my_prod < enemy_prod * 0.7:
        phase = 'defend'
    elif len(enemy) > 0 and len(my) >= 3 and my_prod > enemy_prod * 1.0:
        phase = 'dominate'
    else:
        phase = 'grow'

    moves = []
    targeted_this_turn = set()
    # v131: Track ships already launched this turn per planet
    launched_this_turn = {}

    # v132: Comet prediction - comets spawn at turns 50, 150, 250, 350, 450 (every 100 turns)
    COMET_SPAWN_INTERVAL = 100
    COMET_FIRST_SPAWN = 50
    COMET_LIFESPAN = 19  # Effective turns before comet expires (19 to leave on turn 20)
    
    def predict_comet_spawn_time(current_step):
        """Predict when the next comet group will spawn"""
        if current_step < COMET_FIRST_SPAWN:
            return COMET_FIRST_SPAWN
        # Next spawn after current step
        future_spawns = [s for s in range(COMET_FIRST_SPAWN, 500, COMET_SPAWN_INTERVAL) if s > current_step]
        return future_spawns[0] if future_spawns else None
    
    def time_until_comet_spawn(current_step):
        """How many turns until next comet spawn"""
        next_spawn = predict_comet_spawn_time(current_step)
        return next_spawn - current_step if next_spawn else 999

    if comets_data:
        for src in my:
            # v131: Calculate available ships (don't double count in-flight)
            in_flight = in_flight_ships_from.get(src['id'], 0)
            already_launched = launched_this_turn.get(src['id'], 0)
            available = src['ships'] - in_flight - already_launched
            
            if available < 15:
                continue
            best_comet = None
            best_comet_score = -1e9
            src_x, src_y = src['x'], src['y']
            
            # v132: Get time until comet spawn
            turns_until_spawn = time_until_comet_spawn(step)
            
            for c in comets_data:
                path = c.get('path', [])
                if not path:
                    continue
                speed = c.get('cometSpeed', 4.0)
                path_index = c.get('path_index', 0)
                
                # v132+CL28: Tighter lookahead (proven better than 30)
                for lookahead in range(0, _COMET_LOOKAHEAD, 2):
                    future_idx = (path_index + lookahead * int(speed)) % len(path)
                    if future_idx >= len(path):
                        break
                    cx, cy = path[future_idx]
                    d = math.hypot(cx - src_x, cy - src_y)
                    if d > 60:
                        continue

                    # v132: Score based on timing - prefer comets that will spawn soon
                    score = _COMET_SCORE_BASE - d * _COMET_DIST_PENALTY + speed * 5.0

                    # CURG: Urgency boost near comet spawn windows (turns 50,150,250,350,450)
                    dist_to_spawn = min((sp - step) % 100 for sp in (50, 150, 250, 350, 450))
                    if dist_to_spawn <= 10:
                        score *= _CURG_MULT
                    # v132: If comet spawn is soon and we're close, boost score
                    if turns_until_spawn <= 10 and d < 40:
                        score += 50
                    # v132: If we're already nearby a comet, encourage staying
                    if d < 20 and available > 15:
                        score += 30
                    
                    if score > best_comet_score:
                        best_comet_score = score
                        best_comet = (cx, cy)
            
            if best_comet and available > 20:
                cx, cy = best_comet
                # v132: For comets, send enough to capture but not so many we can't escape
                # Comet garrison is low, and we need to leave with ships before turn 20
                comet_capture_cost = 8  # Typical comet garrison
                send = max(comet_capture_cost + 2, int(available * 0.25))
                send = min(send, available - 5)  # Keep some ships to escape
                
                if available > send + 10:
                    angle = safe_angle(src_x, src_y, cx, cy)
                    moves.append([src['id'], angle, send])
                    targeted_this_turn.add(f"comet_{cx}_{cy}")
                    if src['id'] not in launched_this_turn:
                        launched_this_turn[src['id']] = 0
                    launched_this_turn[src['id']] += send

    for src in my:
        src_x, src_y = src['x'], src['y']
        
        # v131: Calculate available ships at START of turn for this planet
        in_flight = in_flight_ships_from.get(src['id'], 0)
        already_launched = launched_this_turn.get(src['id'], 0)
        available = src['ships'] - in_flight - already_launched
        
        if available < 10:
            continue

        need_defense = threats.get(src['id'], 0) > src['ships'] * defense_sens

        if need_defense and phase != 'counter_attack':
            continue

        if need_defense and phase == 'counter_attack' and threats.get(src['id'], 0) >= src['ships'] * 0.5:
            continue

        if phase == 'expand':
            nearby_larger = {nl[1] for nl in nearby_larger_planets if nl[0] == src['id']}
            best_target = None
            best_score = -1e9
            for t in neutrals:
                if t['id'] == src['id']:
                    continue
                if t['id'] in in_flight_to or t['id'] in targeted_this_turn:
                    continue
                dx = t['x'] - src_x
                dy = t['y'] - src_y
                d = math.sqrt(dx * dx + dy * dy)
                score = -d * 3 + t['prod'] * 3
                if nearby_larger and t['radius'] < src['radius'] * 0.7 and d > 25:
                    score -= 50
                if score > best_score:
                    best_score = score
                    best_target = t
            if best_target:
                r_sq = best_target['r_sq']
                r = best_target['r']
                is_orbiting = best_target['is_orb']
                ix, iy, tt = solve_intercept(src_x, src_y, best_target['x'], best_target['y'], is_orbiting, omega, int(available))
                if not path_crosses_sun(src_x, src_y, ix, iy, margin=1.5):
                    send = ships_needed_for_takeover(best_target['ships'], best_target['prod'], tt, best_target['owner'])
                    if available >= send:
                        angle = safe_angle(src_x, src_y, ix, iy)
                        moves.append([src['id'], angle, send])
                        targeted_this_turn.add(best_target['id'])
                        if src['id'] not in launched_this_turn:
                            launched_this_turn[src['id']] = 0
                        launched_this_turn[src['id']] += send
                        available -= send
                        if available < 5:
                            break
            elif available > 40:
                decoy_tgt = None
                decoy_score = -1e9
                for t in (enemy + neutrals):
                    if t['id'] == src['id']:
                        continue
                    if t['id'] in targeted_this_turn:
                        continue
                    dx = t['x'] - src_x
                    dy = t['y'] - src_y
                    d = math.sqrt(dx * dx + dy * dy)
                    score = -d + (t['prod'] if t['owner'] != -1 else 0) * 5
                    if nearby_larger and t['radius'] < src['radius'] * 0.7 and d > 25:
                        score -= 50
                    if score > decoy_score:
                        decoy_score = score
                        decoy_tgt = t
                if decoy_tgt and available > 25:
                    send = min(8, int(available * 0.15))
                    if send >= 5:
                        r_sq = decoy_tgt['r_sq']
                        r = decoy_tgt['r']
                        is_orbiting = decoy_tgt['is_orb']
                        ix, iy, tt = solve_intercept(src_x, src_y, decoy_tgt['x'], decoy_tgt['y'], is_orbiting, omega, int(available))
                        if not path_crosses_sun(src_x, src_y, ix, iy, margin=1.5):
                            angle = safe_angle(src_x, src_y, ix, iy)
                            moves.append([src['id'], angle, send])
                            targeted_this_turn.add(decoy_tgt['id'])
                            if src['id'] not in launched_this_turn:
                                launched_this_turn[src['id']] = 0
                            launched_this_turn[src['id']] += send
                            available -= send
                            if available < 10:
                                break

        # v131: Recalculate available after expand phase
        in_flight = in_flight_ships_from.get(src['id'], 0)
        already_launched = launched_this_turn.get(src['id'], 0)
        available = src['ships'] - in_flight - already_launched
        
        if available < 10:
            continue

        if phase == 'counter_attack':
            best_enemy = None
            best_score = -1e9
            for t in enemy:
                if t['id'] in targeted_this_turn:
                    continue
                dx = t['x'] - src_x
                dy = t['y'] - src_y
                d = math.sqrt(dx * dx + dy * dy)
                score = t['ships'] * 0.8 + t['prod'] * 8 - d
                if t['id'] in smash_targets:
                    score += 50
                if score > best_score:
                    best_score = score
                    best_enemy = t
            if best_enemy:
                r_sq = best_enemy['r_sq']
                r = best_enemy['r']
                is_orbiting = best_enemy['is_orb']
                ix, iy, tt = solve_intercept(src_x, src_y, best_enemy['x'], best_enemy['y'], is_orbiting, omega, int(available))
                if not path_crosses_sun(src_x, src_y, ix, iy, margin=1.5):
                    send = int(available * 0.8)
                    send = max(send, int(best_enemy['ships'] * 1.1))
                    send = min(send, int(available * 0.95))
                    if available > send + 3:
                        angle = safe_angle(src_x, src_y, ix, iy)
                        moves.append([src['id'], angle, send])
                        targeted_this_turn.add(best_enemy['id'])
                        if src['id'] not in launched_this_turn:
                            launched_this_turn[src['id']] = 0
                        launched_this_turn[src['id']] += send
                        available -= send

        # v131: Recalculate available for main targeting phase
        in_flight = in_flight_ships_from.get(src['id'], 0)
        already_launched = launched_this_turn.get(src['id'], 0)
        available = src['ships'] - in_flight - already_launched
        
        if available < 10:
            continue

        best_tgt = None
        best_score = -1e9

        if phase == 'smash':
            candidates = [t for t in enemy if t['id'] in smash_targets_open]
        elif phase == 'cleanup':
            candidates = enemy
        elif phase == 'pressure':
            candidates = enemy
        elif phase == 'rush':
            candidates = enemy
        elif phase == 'expand' or phase == 'opportunistic' or phase == 'aggressive' or phase == 'dominate':
            candidates = neutrals if phase not in ('aggressive', 'dominate') else (enemy + neutrals)
        elif phase == 'grow':
            candidates = neutrals
        else:
            candidates = []

        for t in candidates:
            if t['id'] == src['id']:
                continue
            if t['id'] in in_flight_to:
                continue
            if t['id'] in targeted_this_turn:
                continue

            incoming = threats.get(t['id'], 0)
            if incoming > 0:
                continue

            r_sq = t['r_sq']
            r = t['r']
            is_orbiting = t['is_orb']
            d = math.sqrt((t['x'] - src_x)**2 + (t['y'] - src_y)**2)

            ix, iy, tt = solve_intercept(src_x, src_y, t['x'], t['y'], is_orbiting, omega, int(available))

            if path_crosses_sun(src_x, src_y, ix, iy, margin=1.5):
                waypoints, _ = multi_leg_path(src_x, src_y, ix, iy)
                if waypoints is None:
                    continue
                final_x, final_y = waypoints[-1]
                if path_crosses_sun(src_x, src_y, final_x, final_y, margin=1.5):
                    continue

            if is_orbiting:
                planet_future = predict_orbit(t['x'], t['y'], omega, tt)
                to_planet = math.atan2(planet_future[1] - src_y, planet_future[0] - src_x)
                to_target = math.atan2(t['y'] - src_y, t['x'] - src_x)
                diff = abs((to_planet - to_target) % (2 * math.pi))
                if diff > 0.5 and diff < (2 * math.pi - 0.5):
                    continue

            score = t['prod'] * _SCORE_PROD_WEIGHT - tt * _SCORE_TT_PENALTY

            if t['owner'] == -1:
                score += _NEUTRAL_BONUS

            if phase == 'aggressive' and t['owner'] != -1:
                score += _AGGRO_HOSTILE_BONUS - t['ships'] * 0.12

            if phase == 'pressure' and t['owner'] != -1:
                score += _PRESSURE_HOSTILE_BONUS - t['ships'] * 0.07

            if phase == 'cleanup' and t['owner'] != -1:
                score += _CLEANUP_HOSTILE_BONUS - t['ships'] * 0.04

            if phase == 'cleanup' and t['owner'] == -1:
                score -= 25

            if phase == 'dominate' and t['owner'] != -1:
                score += 45 - t['ships'] * 0.08

            if phase == 'dominate' and t['owner'] == -1:
                score += 20

            if is_orbiting:
                score -= 6

            if is_orbiting and d > 50:
                score *= 0.85

            if phase == 'aggressive':
                score *= aggression
            elif phase == 'expand' and my_planet_count < 4:
                score *= (1.0 + aggression * 0.3)

            # PRODEXPAND: production-weighted scoring (additive — no multiplicative distortion)
            if t['owner'] == -1:
                # Additive bonus that tapers off over first 100 turns
                _phase_factor = max(0.0, 1.0 - step / 100.0)
                score += PROD_BONUS_PER_UNIT * t['prod'] * _phase_factor
                # Permanent anchor bonus for top-3 highest-prod neutrals
                if _ANCHOR_PLANETS_PE and t['id'] in _ANCHOR_PLANETS_PE:
                    score += ANCHOR_PROD_BONUS
            elif t['owner'] != -1 and step < 120:
                # Additive bonus for attacking high-prod enemy planets early
                _phase_factor_hostile = max(0.0, 1.0 - step / 120.0)
                score += PROD_BONUS_PER_UNIT * t['prod'] * _phase_factor_hostile

            if score > best_score:
                best_score = score
                best_tgt = (t, ix, iy, tt)

        smash_fallback = False
        if best_tgt is None and phase == 'smash':
            # Smash found no viable enemy target — fall back to neutrals with expand logic
            for t in neutrals:
                if t['id'] == src['id'] or t['id'] in in_flight_to or t['id'] in targeted_this_turn:
                    continue
                incoming = threats.get(t['id'], 0)
                if incoming > 0:
                    continue
                is_orbiting = t['is_orb']
                ix2, iy2, tt2 = solve_intercept(src_x, src_y, t['x'], t['y'], is_orbiting, omega, int(available))
                if path_crosses_sun(src_x, src_y, ix2, iy2, margin=1.5):
                    waypoints, _ = multi_leg_path(src_x, src_y, ix2, iy2)
                    if waypoints is None:
                        continue
                    final_x, final_y = waypoints[-1]
                    if path_crosses_sun(src_x, src_y, final_x, final_y, margin=1.5):
                        continue
                score2 = t['prod'] * _SCORE_PROD_WEIGHT - tt2 * _SCORE_TT_PENALTY + _NEUTRAL_BONUS
                if score2 > best_score:
                    best_score = score2
                    best_tgt = (t, ix2, iy2, tt2)
            if best_tgt is not None:
                smash_fallback = True

        if best_tgt is None:
            continue

        tgt, ix, iy, tt = best_tgt
        tgt_dist = math.sqrt((tgt['x'] - src_x)**2 + (tgt['y'] - src['y'])**2)

        if phase == 'smash' and not smash_fallback:
            send = int(available * _SEND_SMASH)
            send = max(send, ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner']))
        elif phase == 'cleanup':
            send = int(available * _SEND_CLEANUP)
            send = max(send, ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner']))
            send = min(send, int(available * 0.95))
        elif phase == 'pressure':
            send = int(available * _SEND_PRESSURE)
            send = max(send, ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner']))
            send = min(send, int(available * 0.78))
        elif phase == 'rush':
            send = int(available * _SEND_RUSH)
        elif phase == 'aggressive':
            send = int(available * _SEND_AGGRESSIVE)
            send = max(send, ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner']))
            send = min(send, int(available * 0.7))
        elif phase == 'dominate':
            send = int(available * _SEND_DOMINATE)
            send = max(send, ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner']))
            send = min(send, int(available * 0.8))
        elif phase == 'opportunistic':
            send = ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner'])
            send = min(send, int(available * 0.5))
        else:
            send = ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner'])

        # Map-adaptive: scale send by map spread factor
        if _MAP_SPREAD == 'tight' and phase not in ('smash', 'cleanup'):
            send = max(send, int(available * 0.85))  # Commit harder on tight maps
        elif _MAP_SPREAD == 'spread' and phase not in ('smash', 'cleanup', 'rush'):
            send = min(send, int(available * _SEND_FRACTION))  # More conservative on spread

        if available < send:
            continue

        angle = safe_angle(src_x, src_y, ix, iy)
        moves.append([src['id'], angle, send])
        targeted_this_turn.add(tgt['id'])
        if src['id'] not in launched_this_turn:
            launched_this_turn[src['id']] = 0
        launched_this_turn[src['id']] += send

    if phase == 'expand':
        for src in my:
            in_flight = in_flight_ships_from.get(src['id'], 0)
            already_launched = launched_this_turn.get(src['id'], 0)
            available = src['ships'] - in_flight - already_launched
            
            if available < expand_thresh:
                continue
            nearby_larger = [nl for nl in nearby_larger_planets if nl[0] == src['id']]
            if not nearby_larger:
                continue
            candidates = [t for t in (neutrals + enemy)
                          if t['id'] not in targeted_this_turn
                          and t['id'] not in in_flight_to
                          and t['owner'] != player]
            if not candidates:
                continue
            best_tgt = None
            best_score = -1e9
            src_x, src_y = src['x'], src['y']
            for t in candidates:
                dx = t['x'] - src_x
                dy = t['y'] - src_y
                d_sq = dx * dx + dy * dy
                if d_sq > 1600:
                    continue
                d = math.sqrt(d_sq)
                score = t['prod'] * 5 - d
                if t['radius'] >= src['radius'] * 0.8 and t['prod'] >= src['prod'] * 0.8:
                    score += 40
                if score > best_score:
                    best_score = score
                    best_tgt = t
            if best_tgt:
                r_sq = best_tgt['r_sq']
                r = best_tgt['r']
                is_orbiting = best_tgt['is_orb']
                ix, iy, tt = solve_intercept(src_x, src_y, best_tgt['x'], best_tgt['y'], is_orbiting, omega, int(available))
                if not path_crosses_sun(src_x, src_y, ix, iy, margin=1.5):
                    send = ships_needed_for_takeover(best_tgt['ships'], best_tgt['prod'], tt, best_tgt['owner'])
                    if available >= send:
                        angle = safe_angle(src_x, src_y, ix, iy)
                        moves.append([src['id'], angle, send])
                        targeted_this_turn.add(best_tgt['id'])
                        if src['id'] not in launched_this_turn:
                            launched_this_turn[src['id']] = 0
                        launched_this_turn[src['id']] += send

    return moves


if __name__ == '__main__':
    print("v132: v131 with comet prediction and timed capture")
