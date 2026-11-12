"""
physics_sim.py — Orbit Wars forward physics engine.

Predicts game state N steps ahead:
  - Planet positions (circular orbit for is_orb planets, static for comets/outer)
  - Fleet movement and arrival (both players)
  - Combat resolution on arrival
  - Production each step for owned planets

NO evaluator. NO planner. NO tactics. Pure physics.

Math (fleet_speed, travel_time, predict_orbit) taken directly from
main_v131_plus_denial.py where they are proven correct.
"""

import math
from copy import deepcopy

# ── Constants (match game engine exactly) ─────────────────────────────────────
SUN_X, SUN_Y          = 50.0, 50.0
SUN_RADIUS            = 10.0
INNER_ORBIT_THRESHOLD = 48.0   # (r + radius) < this → orbiting planet
MAX_SPEED             = 6.0
_MAX_SPEED_MINUS_1    = MAX_SPEED - 1.0
_LOG1000              = math.log(1000.0)

# Minimum dot-product confidence to infer a fleet's destination from its angle.
# 0.95 ≈ within ~18° of the planet direction.  Lower = more wrong guesses.
FLEET_DEST_MIN_DOT    = 0.95


# ── Proven math (verbatim from main_v131_plus_denial.py) ──────────────────────

def fleet_speed(ships: int) -> float:
    if ships <= 0:
        return 1.0
    return 1.0 + _MAX_SPEED_MINUS_1 * (math.log(max(ships, 1)) / _LOG1000) ** 1.5


def travel_time(x1: float, y1: float, x2: float, y2: float, ships: int) -> float:
    dx, dy = x2 - x1, y2 - y1
    if ships <= 0:
        return 999.0
    return math.sqrt(dx * dx + dy * dy) / fleet_speed(ships)


def predict_orbit(x: float, y: float, omega: float, dt: float):
    """Return (x, y) after planet has orbited for dt steps."""
    theta = math.atan2(y - SUN_Y, x - SUN_X)
    r     = math.hypot(x - SUN_X, y - SUN_Y)
    return (SUN_X + r * math.cos(theta + omega * dt),
            SUN_Y + r * math.sin(theta + omega * dt))


# ── Fleet destination inference ────────────────────────────────────────────────

def _ray_orbit_arrival(fx: float, fy: float, cos_a: float, sin_a: float,
                       ships: int, planet: dict, omega: float) -> float:
    """
    For an orbiting planet, find the distance d along the ray (fx,fy,cos_a,sin_a)
    that intersects the orbital circle, then return how far the planet will be
    from that intercept point when the fleet arrives.

    Returns the mismatch in UNITS (0.0 = perfect match).
    Returns a large number if the ray misses the orbit or fleet is heading away.
    """
    r = planet['r']
    # Ray vs circle: d^2 + 2*b*d + c = 0
    dx0 = fx - SUN_X
    dy0 = fy - SUN_Y
    b   = dx0 * cos_a + dy0 * sin_a
    c   = dx0**2 + dy0**2 - r**2
    disc = b * b - c
    if disc < 0:
        return 999.0  # ray misses orbital circle entirely
    sqrt_disc = math.sqrt(disc)
    d1 = -b - sqrt_disc
    d2 = -b + sqrt_disc
    # Take the smallest positive distance (forward intersection)
    d_arr = d2 if d1 < 0.5 else d1
    if d_arr < 0.5:
        return 999.0  # fleet is already past the orbit or heading away
    # Intercept point on the orbital circle
    ix = fx + d_arr * cos_a
    iy = fy + d_arr * sin_a
    # When does the fleet arrive?
    eta = d_arr / fleet_speed(max(ships, 1))
    # Where will the planet be at arrival?
    theta_arr = planet['theta'] + omega * eta
    px_arr    = SUN_X + r * math.cos(theta_arr)
    py_arr    = SUN_Y + r * math.sin(theta_arr)
    return math.hypot(px_arr - ix, py_arr - iy)


# Orbiting planet: fleet must arrive within this many units of the planet's
# predicted position for us to call it a match.  ~1.5× the per-step orbital
# arc (omega * r_typical ≈ 0.03 * 40 ≈ 1.2 units) gives a tight but tolerant window.
_ORB_MATCH_DIST = 3.0


def infer_fleet_dest(fx: float, fy: float, fangle: float,
                     from_pid: int, planets: dict,
                     ships: int = 1, omega: float = 0.03):
    """
    Find which planet a mid-flight fleet is heading toward.

    Two strategies, chosen per-planet type:
      • Static planets  — dot-product of flight angle vs direction to planet.
                          Works perfectly because the planet doesn't move.
      • Orbiting planets — geometric ray-orbit intersection: find where the
                          flight path crosses the orbital circle, check if the
                          planet will be ≤ _ORB_MATCH_DIST units from that
                          point when the fleet arrives.  This handles the fact
                          that the fleet's launch angle aimed at the intercept
                          position (where the planet *will be*), not its current
                          position, so pure dot-product under-scores orbiting
                          candidates for fleets already mid-flight.

    If a static planet has dot > 0.95 AND an orbiting planet also matches,
    the static planet wins (higher precision signal).

    Returns dest_pid (int) or None.
    """
    cos_a = math.cos(fangle)
    sin_a = math.sin(fangle)

    best_static_pid, best_static_dot = None, FLEET_DEST_MIN_DOT
    best_orb_pid,    best_orb_err   = None, _ORB_MATCH_DIST

    for pid, p in planets.items():
        if pid == from_pid:
            continue

        if p['is_orb'] and not p['is_comet']:
            err = _ray_orbit_arrival(fx, fy, cos_a, sin_a, ships, p, omega)
            if err < best_orb_err:
                best_orb_err = err
                best_orb_pid = pid
        else:
            dx = p['x'] - fx
            dy = p['y'] - fy
            d  = math.hypot(dx, dy)
            if d < 0.1:
                continue
            dot = (dx / d) * cos_a + (dy / d) * sin_a
            if dot > best_static_dot:
                best_static_dot = dot
                best_static_pid = pid

    # Prefer static if confidently matched; fall back to orbiting.
    if best_static_pid is not None:
        return best_static_pid
    return best_orb_pid


# ── State representation ───────────────────────────────────────────────────────
#
# planets: dict  pid -> {
#   'x', 'y'       — current position
#   'owner'        — int: player id, or -1 neutral
#   'ships'        — float: ship count
#   'prod'         — float: production per step (only when owned)
#   'radius'       — planet radius (used only for is_orb check)
#   'is_orb'       — bool: circles the sun each step
#   'is_comet'     — bool: straight-line mover; we hold it static (can't predict)
#   'r'            — orbital radius (distance from sun)
#   'theta'        — current angle from sun (radians); updated each step if is_orb
# }
#
# fleets: list of {
#   'owner'        — int: player id who sent this fleet
#   'ships'        — float: number of ships
#   'target_pid'   — int: destination planet id
#   'eta'          — float: steps until arrival (countdown; arrives when <= 0)
# }
#
# step:  int  — current game step
# omega: float — angular velocity of orbiting planets (radians/step)
# me:    int  — our player id (0 or 1)


def parse_obs(obs) -> dict:
    """
    Parse a raw Orbit Wars observation into a physics state dict.

    Handles both dict-style and attribute-style obs objects.
    """
    if isinstance(obs, dict):
        player      = obs.get('player', 0)
        planets_raw = obs.get('planets', [])
        fleets_raw  = obs.get('fleets', [])
        step        = obs.get('step', 0)
        omega       = obs.get('angular_velocity', 0.03)
        comet_ids   = set(obs.get('comet_planet_ids', []))
    else:
        player      = getattr(obs, 'player', 0)
        planets_raw = getattr(obs, 'planets', [])
        fleets_raw  = getattr(obs, 'fleets', [])
        step        = getattr(obs, 'step', 0)
        omega       = getattr(obs, 'angular_velocity', 0.03)
        comet_ids   = set(getattr(obs, 'comet_planet_ids', []))

    # ── Parse planets ──────────────────────────────────────────────────────────
    planets = {}
    for p in planets_raw:
        pid, owner, x, y, radius, ships, prod = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
        r        = math.hypot(x - SUN_X, y - SUN_Y)
        is_comet = pid in comet_ids
        # Only planets whose full orbit stays inside INNER_ORBIT_THRESHOLD orbit.
        # Comets pass close to the sun but travel in a straight line — exclude them.
        is_orb   = (r + radius) < INNER_ORBIT_THRESHOLD and not is_comet
        planets[pid] = {
            'x':        x,
            'y':        y,
            'owner':    int(owner),
            'ships':    float(ships),
            'prod':     float(prod),
            'radius':   float(radius),
            'is_orb':   is_orb,
            'is_comet': is_comet,
            'r':        r,
            'theta':    math.atan2(y - SUN_Y, x - SUN_X),
        }

    # ── Parse fleets — infer destination by angle ──────────────────────────────
    # Raw format: [id, owner, x, y, angle, from_planet_id, ships]
    fleets = []
    inferred, skipped = 0, 0
    for f in fleets_raw:
        fid, fowner, fx, fy, fangle, ffrom, fships = (
            f[0], f[1], f[2], f[3], f[4], f[5], float(f[6]))

        dest_pid = infer_fleet_dest(fx, fy, fangle, ffrom, planets,
                                    ships=int(max(fships, 1)), omega=omega)
        if dest_pid is None:
            skipped += 1
            continue

        dest_p = planets[dest_pid]
        if dest_p['is_orb'] and not dest_p['is_comet']:
            # For orbiting destinations use the ray-orbit intersection distance
            # as a more accurate remaining travel time than dist-to-current-pos.
            cos_a = math.cos(fangle)
            sin_a = math.sin(fangle)
            dx0 = fx - SUN_X; dy0 = fy - SUN_Y
            b   = dx0*cos_a + dy0*sin_a
            c   = dx0**2 + dy0**2 - dest_p['r']**2
            disc = b*b - c
            if disc >= 0:
                d1 = -b - math.sqrt(disc); d2 = -b + math.sqrt(disc)
                d_arr = d2 if d1 < 0.5 else d1
                eta = d_arr / fleet_speed(int(max(fships, 1))) if d_arr > 0.5 else 1.0
            else:
                eta = travel_time(fx, fy, dest_p['x'], dest_p['y'], int(max(fships, 1)))
        else:
            eta = travel_time(fx, fy, dest_p['x'], dest_p['y'], int(max(fships, 1)))
        fleets.append({
            'owner':      int(fowner),
            'ships':      fships,
            'target_pid': dest_pid,
            'eta':        max(1.0, eta),
        })
        inferred += 1

    return {
        'planets': planets,
        'fleets':  fleets,
        'step':    step,
        'omega':   omega,
        'me':      player,
        '_fleet_inferred': inferred,
        '_fleet_skipped':  skipped,
    }


def copy_state(state: dict) -> dict:
    return {
        'planets': {pid: dict(p) for pid, p in state['planets'].items()},
        'fleets':  [dict(f) for f in state['fleets']],
        'step':    state['step'],
        'omega':   state['omega'],
        'me':      state['me'],
    }


# ── One-step physics ───────────────────────────────────────────────────────────

def step_state(state: dict) -> dict:
    """
    Advance state by exactly one game turn.  Mutates and returns state.

    Order matches the game engine:
      1. Decrement fleet ETAs; collect arrivals (eta reaches 0)
      2. Resolve all arrivals simultaneously per planet (combat + reinforcement)
      3. Production: every owned planet generates prod ships
      4. Orbit: is_orb planets rotate by omega radians
    """
    planets = state['planets']

    # ── 1. Tick fleets ─────────────────────────────────────────────────────────
    arrivals  = {}   # target_pid -> {owner: total_ships}
    surviving = []
    for fl in state['fleets']:
        fl['eta'] -= 1.0
        if fl['eta'] <= 0.0:
            tid = fl['target_pid']
            if tid not in arrivals:
                arrivals[tid] = {}
            arrivals[tid][fl['owner']] = (
                arrivals[tid].get(fl['owner'], 0.0) + fl['ships'])
        else:
            surviving.append(fl)
    state['fleets'] = surviving

    # ── 2. Resolve arrivals ────────────────────────────────────────────────────
    for tid, att in arrivals.items():
        if tid not in planets:
            continue
        p            = planets[tid]
        planet_owner = p['owner']

        # Split arriving ships: same owner as planet = reinforcement, others = hostile
        reinforce   = p['ships']          # planet's current garrison
        hostile     = {}                  # owner -> ships (attackers)
        for owner, ships in att.items():
            if owner == planet_owner:
                reinforce += ships        # same side — just add
            else:
                hostile[owner] = hostile.get(owner, 0.0) + ships

        if not hostile:
            # Pure reinforcement, no fight
            p['ships'] = reinforce
            continue

        # ── Combat ─────────────────────────────────────────────────────────────
        # When multiple hostile factions arrive at the same time they fight
        # each other first (largest eats the rest), then the winner fights the
        # defender garrison.

        if planet_owner == -1:
            # Neutral planet: no defender loyalty — garrison fights everyone
            # Treat neutral garrison as belonging to a fictional -1 faction
            # alongside the attackers.
            factions = sorted(hostile.items(), key=lambda kv: -kv[1])
        else:
            # Owned planet: garrison reinforced above = reinforce
            # All hostile factions combine against the defender.
            # (Simplification: they cooperate against the defender;
            #  if multiple hostile factions, biggest one captures.)
            factions = sorted(hostile.items(), key=lambda kv: -kv[1])

        # Resolve multi-faction combat: iterate factions sorted by size.
        # Largest eats each subsequent faction; last survivor fights defender.
        current_owner  = planet_owner
        current_ships  = reinforce        # defender ships
        for f_owner, f_ships in factions:
            if f_owner == current_owner:
                current_ships += f_ships  # same side by now
            elif f_ships > current_ships:
                # Attacker wins this engagement
                current_ships = f_ships - current_ships
                current_owner = f_owner
            else:
                current_ships -= f_ships  # defender holds

        p['owner'] = current_owner
        p['ships'] = max(0.0, current_ships)

    # ── 3. Production ──────────────────────────────────────────────────────────
    for p in planets.values():
        if p['owner'] != -1:
            p['ships'] += p['prod']

    # ── 4. Orbit ───────────────────────────────────────────────────────────────
    omega = state['omega']
    for p in planets.values():
        if p['is_orb']:
            p['theta'] += omega
            p['x'] = SUN_X + p['r'] * math.cos(p['theta'])
            p['y'] = SUN_Y + p['r'] * math.sin(p['theta'])
        # Comets: held static — straight-line trajectory unknown without launch data

    state['step'] += 1
    return state


# ── N-step prediction ──────────────────────────────────────────────────────────

def predict(state: dict, n: int) -> dict:
    """
    Return a new state representing the game n steps from now.
    Original state is not modified.
    """
    s = copy_state(state)
    for _ in range(n):
        step_state(s)
    return s


# ── Add planned moves to a state (for our own planner) ────────────────────────

def apply_moves(state: dict, moves: list) -> dict:
    """
    Inject our planned moves into a state as new fleets.
    moves: list of (src_pid, dest_pid, ships_to_send)

    Uses planet ids directly — no angles needed at the physics layer.
    Deducts ships from source planet immediately (as the game does).
    Always leaves at least 1 ship on the source.
    """
    planets = state['planets']
    for src_pid, dest_pid, ships in moves:
        if src_pid not in planets or dest_pid not in planets:
            continue
        src   = planets[src_pid]
        tgt   = planets[dest_pid]
        ships = min(float(ships), src['ships'] - 1.0)
        if ships <= 0:
            continue
        src['ships'] -= ships
        eta = travel_time(src['x'], src['y'], tgt['x'], tgt['y'], int(ships))
        state['fleets'].append({
            'owner':      state['me'],
            'ships':      ships,
            'target_pid': dest_pid,
            'eta':        max(1.0, eta),
        })
    return state


# ── Convenience snapshot ───────────────────────────────────────────────────────

def summarise(state: dict, me: int = None) -> dict:
    """
    Return a compact summary of a state for easy inspection/scoring.
    Returns:
        my_planets, my_ships, my_prod,
        enemy_planets, enemy_ships, enemy_prod,
        neutral_planets,
        fleets_mine, fleets_enemy  (counts)
        in_flight_mine, in_flight_enemy  (ship totals)
    """
    if me is None:
        me = state['me']

    mp = ep = np_ = 0
    ms = es = 0.0
    mpr = epr = 0.0
    fm = fe = 0
    ims = ies = 0.0

    for p in state['planets'].values():
        if p['owner'] == me:
            mp  += 1; ms  += p['ships']; mpr += p['prod']
        elif p['owner'] == -1:
            np_ += 1
        else:
            ep  += 1; es  += p['ships']; epr += p['prod']

    for fl in state['fleets']:
        if fl['owner'] == me:
            fm += 1; ims += fl['ships']
        else:
            fe += 1; ies += fl['ships']

    return {
        'step':           state['step'],
        'my_planets':     mp,  'my_ships':     ms,  'my_prod':     mpr,
        'enemy_planets':  ep,  'enemy_ships':  es,  'enemy_prod':  epr,
        'neutral_planets': np_,
        'fleets_mine':    fm,  'in_flight_mine':  ims,
        'fleets_enemy':   fe,  'in_flight_enemy': ies,
    }
