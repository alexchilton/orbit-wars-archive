"""
validate_sim.py — Physics engine validation harness.

Runs a real game between two simple agents, captures the true game state at
every step, then for each step T runs _simulate(state_T, [], N) and compares
the predicted state at T+N against the actual state at T+N.

Reports per-category mean absolute error so we know exactly where the sim
is wrong before building any planner on top of it.

Usage:
    cd /Users/alexchilton/DataspellProjects/orbit_wars
    python3 validate_sim.py
"""
import sys, importlib.util, math, time, copy
sys.path.insert(0, '.')

import kaggle_environments

# ─── Load sim module (physics engine only, no agent) ─────────────────────────
spec = importlib.util.spec_from_file_location("sim", "submission/main_v131_plus_sim.py")
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)

SUN_X, SUN_Y = 50.0, 50.0

# ─── Simple greedy agent (for generating realistic games to validate against) ─
def greedy_agent(obs):
    """Expand to nearest cheapest neutral. Nothing fancy."""
    if isinstance(obs, dict):
        player = obs.get('player', 0)
        planets_raw = obs.get('planets', [])
        fleets_raw  = obs.get('fleets', [])
        omega = obs.get('angular_velocity', 0.03)
    else:
        player = getattr(obs, 'player', 0)
        planets_raw = getattr(obs, 'planets', [])
        fleets_raw  = getattr(obs, 'fleets', [])
        omega = getattr(obs, 'angular_velocity', 0.03)

    if not planets_raw:
        return []

    my     = [p for p in planets_raw if p[1] == player]
    others = [p for p in planets_raw if p[1] != player]
    if not my or not others:
        return []

    moves = []
    in_flight = {f[5] for f in fleets_raw if f[1] == player}

    for mp in my:
        msrc_id, _, mx, my_y, _, mships, _ = mp[:7]
        if mships < 8:
            continue
        best, best_cost = None, float('inf')
        for tp in others:
            tid, towner, tx, ty, _, tships, tprod = tp[:7]
            if tid in in_flight:
                continue
            d = math.hypot(tx-mx, ty-my_y)
            need = int(tships * 1.1) + 1 if towner == -1 else int(tships * 1.2 + tprod * (d/2)) + 1
            if need < mships * 0.6 and d < best_cost:
                best_cost, best = d, (tid, tx, ty, need)
        if best:
            tid, tx, ty, need = best
            angle = math.atan2(ty - my_y, tx - mx)
            moves.append([msrc_id, angle, need])
    return moves

# ─── State capture ────────────────────────────────────────────────────────────
captured = []   # list of raw obs dicts at each step

def capture_agent(obs):
    """Wraps greedy_agent; captures raw obs each step."""
    if isinstance(obs, dict):
        step = obs.get('step', 0)
        snap = {
            'step': step,
            'player': obs.get('player', 0),
            'omega': obs.get('angular_velocity', 0.03),
            'planets': [list(p) for p in obs.get('planets', [])],
            'fleets':  [list(f) for f in obs.get('fleets', [])],
        }
    else:
        step = getattr(obs, 'step', 0)
        snap = {
            'step': step,
            'player': getattr(obs, 'player', 0),
            'omega': getattr(obs, 'angular_velocity', 0.03),
            'planets': [list(p) for p in getattr(obs, 'planets', [])],
            'fleets':  [list(f) for f in getattr(obs, 'fleets', [])],
        }
    captured.append(snap)
    return greedy_agent(obs)

def capture_agent2(obs):
    return greedy_agent(obs)

# ─── Run a game and capture states ────────────────────────────────────────────
print("Running capture game...", flush=True)
env = kaggle_environments.make('orbit_wars', debug=False)
env.run([capture_agent, capture_agent2])
print(f"Captured {len(captured)} steps.")

# Find first step with planet data
first_data = next((i for i, s in enumerate(captured) if s['planets']), 0)
omega = captured[first_data]['omega']

# ─── Convert raw obs to sim state ────────────────────────────────────────────
def raw_to_sim_state(snap):
    """Parse a captured snapshot into sim's internal state dict."""
    planets_list = []
    for p in snap['planets']:
        pid, owner, x, y, radius, ships, prod = p[:7]
        r_sq = (x - SUN_X)**2 + (y - SUN_Y)**2
        r    = math.sqrt(r_sq)
        is_orb = (r + radius) < sim.INNER_ORBIT_THRESHOLD
        planets_list.append({
            'id': pid, 'owner': owner, 'x': x, 'y': y,
            'radius': radius, 'ships': float(ships), 'prod': float(prod),
            'is_orb': is_orb, 'r': r,
        })

    # For fleets: raw format is [id, owner, x, y, angle, from_planet_id, ships]
    # We know x,y (current position) and angle (direction of travel).
    # We do NOT have the destination planet directly — this is the key problem.
    # Strategy: project each fleet along its angle and find which planet it's heading toward.
    planet_map = {p['id']: p for p in planets_list}
    fleets_norm = {}
    for f in snap['fleets']:
        fid, fowner, fx, fy, fangle, ffrom, fships = f[:7]
        # Project fleet forward along its angle to find best-matching destination planet
        # Try all planets, find which one the angle is pointing at
        best_pid, best_dot = None, -2.0
        for pid, pl in planet_map.items():
            if pid == ffrom:
                continue
            dx = pl['x'] - fx
            dy = pl['y'] - fy
            d = math.hypot(dx, dy)
            if d < 0.1:
                continue
            # Unit vector to planet vs fleet direction
            dot = (dx/d) * math.cos(fangle) + (dy/d) * math.sin(fangle)
            if dot > best_dot:
                best_dot, best_pid = dot, pid
        if best_pid is None or best_dot < 0.5:
            # Can't determine destination confidently — skip
            continue
        tgt = planet_map[best_pid]
        eta = sim.travel_time(fx, fy, tgt['x'], tgt['y'], int(max(fships, 1)))
        fleets_norm[fid] = {
            'id': fid, 'owner': fowner, 'x': fx, 'y': fy,
            'angle': fangle, 'from': ffrom, 'ships': float(fships),
            '_target_id': best_pid, '_eta': eta,
        }

    state = sim._make_state(planets_list, fleets_norm, snap['player'], snap['omega'], snap['step'])
    return state, planet_map

# ─── Validation ───────────────────────────────────────────────────────────────
HORIZONS = [1, 3, 5, 10, 20]
max_horizon = max(HORIZONS)

# We need pairs: (state at step T, actual state at step T+N)
# Only use steps where we have data at T and T+N
usable = [s for s in captured if s['planets']]

print(f"\nUsable steps with planet data: {len(usable)}")
print(f"Step range: {usable[0]['step']} — {usable[-1]['step']}")

results = {h: {'pos_err': [], 'ships_err': [], 'ownership_wrong': 0, 'n': 0} for h in HORIZONS}

step_index = {s['step']: s for s in usable}

for snap in usable:
    T = snap['step']
    for H in HORIZONS:
        if (T + H) not in step_index:
            continue
        actual = step_index[T + H]

        # Build sim state at T
        try:
            state_T, pmap_T = raw_to_sim_state(snap)
        except Exception as e:
            continue

        # Simulate H steps forward (no moves — pure physics)
        from copy import deepcopy
        s = deepcopy(state_T)
        for _ in range(H):
            sim._step_state(s)

        # Compare predicted vs actual
        actual_pmap = {p[0]: p for p in actual['planets']}
        for pid, pred_p in s['planets'].items():
            if pid not in actual_pmap:
                continue
            ap = actual_pmap[pid]  # [id, owner, x, y, radius, ships, prod]
            act_x, act_y, act_ships, act_owner = ap[2], ap[3], ap[5], ap[1]

            pos_err = math.hypot(pred_p['x'] - act_x, pred_p['y'] - act_y)
            ship_err = abs(pred_p['ships'] - act_ships)
            results[H]['pos_err'].append(pos_err)
            results[H]['ships_err'].append(ship_err)
            results[H]['n'] += 1
            if pred_p['owner'] != act_owner:
                results[H]['ownership_wrong'] += 1

# ─── Report ───────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Horizon':>8} | {'Pos MAE':>10} | {'Ships MAE':>10} | {'Owner Wrong':>12} | {'N':>6}")
print("-"*65)
for H in HORIZONS:
    r = results[H]
    if r['n'] == 0:
        print(f"{H:>8} | {'(no data)':>10}")
        continue
    pos_mae   = sum(r['pos_err'])   / len(r['pos_err'])
    ships_mae = sum(r['ships_err']) / len(r['ships_err'])
    own_pct   = 100.0 * r['ownership_wrong'] / r['n']
    print(f"{H:>8} | {pos_mae:>10.3f} | {ships_mae:>10.2f} | {own_pct:>10.1f}% | {r['n']:>6}")

print("="*65)
print("""
Legend:
  Pos MAE    — mean distance error in planet position (orbiting planets)
  Ships MAE  — mean absolute error in ship count (includes production + combat)
  Owner Wrong— % of (planet, step) pairs where owner prediction is wrong
  N          — number of (planet, step) comparison pairs
""")

# ─── Detailed orbit check ─────────────────────────────────────────────────────
print("=== ORBIT CHECK (orbiting planets only, H=10) ===")
H = 10
orbit_errs = []
for snap in usable:
    T = snap['step']
    if (T + H) not in step_index:
        continue
    actual = step_index[T + H]
    actual_pmap = {p[0]: p for p in actual['planets']}
    state_T, _ = raw_to_sim_state(snap)
    # Only orbiting
    orb_ids = [pid for pid, p in state_T['planets'].items() if p['is_orb']]
    if not orb_ids:
        continue
    s = deepcopy(state_T)
    for _ in range(H):
        sim._step_state(s)
    for pid in orb_ids:
        if pid not in actual_pmap:
            continue
        ap = actual_pmap[pid]
        pred_p = s['planets'][pid]
        err = math.hypot(pred_p['x'] - ap[2], pred_p['y'] - ap[3])
        orbit_errs.append((pid, T, err, pred_p['x'], pred_p['y'], ap[2], ap[3]))

if orbit_errs:
    orbit_errs.sort(key=lambda x: -x[2])
    print(f"Orbit position errors (H=10), worst 10:")
    for pid, T, err, px, py, ax, ay in orbit_errs[:10]:
        print(f"  planet {pid} at step {T}: pred=({px:.1f},{py:.1f}) actual=({ax:.1f},{ay:.1f}) err={err:.3f}")
    mean_orb = sum(e[2] for e in orbit_errs) / len(orbit_errs)
    print(f"Mean orbit position error (H=10): {mean_orb:.4f}")
else:
    print("  No orbiting planets found.")

# ─── Fleet ETA check ─────────────────────────────────────────────────────────
print("\n=== FLEET ETA CHECK ===")
print("(Checking whether our fleet destination inference is correct)")
for snap in usable[:3]:  # just first few steps with data
    if not snap['fleets']:
        continue
    actual_pmap = {p[0]: p for p in snap['planets']}
    planet_map2 = {p[0]: (p[2], p[3]) for p in snap['planets']}
    for f in snap['fleets'][:5]:
        fid, fowner, fx, fy, fangle, ffrom, fships = f[:7]
        # Find best-matching destination by angle
        best_pid, best_dot = None, -2.0
        for pid, (px, py) in planet_map2.items():
            if pid == ffrom:
                continue
            dx, dy = px - fx, py - fy
            d = math.hypot(dx, dy)
            if d < 0.1:
                continue
            dot = (dx/d)*math.cos(fangle) + (dy/d)*math.sin(fangle)
            if dot > best_dot:
                best_dot, best_pid = dot, pid
        print(f"  step={snap['step']} fleet {fid} from={ffrom} angle={fangle:.2f} → inferred_dest={best_pid} (dot={best_dot:.3f})")
    break
