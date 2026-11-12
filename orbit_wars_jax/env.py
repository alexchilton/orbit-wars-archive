"""
JAX implementation of Orbit Wars per-step game logic.

Map generation stays in Python (reference implementation).
Only the per-step interpreter is in JAX, enabling vmap over batched envs.

State dict keys
───────────────
planets        [MAX_PLANETS, 7]   float32  [id, owner, x, y, radius, ships, prod]
planet_valid   [MAX_PLANETS]      bool
is_orbiting    [MAX_PLANETS]      bool
orb_r          [MAX_PLANETS]      float32  orbital radius (0 if static)
init_angle     [MAX_PLANETS]      float32  initial angle for orbit calc
fleets         [MAX_FLEETS, 7]    float32  [id, owner, x, y, angle, from_id, ships]
fleet_valid    [MAX_FLEETS]       bool
comet_paths    [5, 4, MAX_PL, 2]  float32  pre-computed trajectories
comet_lengths  [5, 4]             int32    actual path lengths per comet
comet_path_idx [5]                int32    last-used index (-1 = not yet moved)
comet_active   [5]                bool     True once spawned
comet_slots    [5, 4]             int32    planet-array slot indices
comet_ships    [5]                int32    initial ship count per group
angular_vel    scalar             float32
step           scalar             int32
next_fleet_id  scalar             int32
done           scalar             bool
"""

import math
import random as py_random

import jax
import jax.numpy as jnp
import numpy as np

# ── Sizes ─────────────────────────────────────────────────────────────────────
MAX_PLANETS        = 64    # 40 regular + 20 comet (5 groups × 4) + 4 buffer
MAX_FLEETS         = 512   # 40 launches/step × ~12 step avg travel = ~480 peak; 512 is safe
MAX_PATH_LEN       = 64   # comet visible-segment steps (ref: 5–40)
N_COMET_GROUPS     = 5
N_COMETS_PER_GROUP = 4
MAX_LAUNCHES       = 20   # max launches per player per step
N_PLAYERS          = 2

# ── Game constants ────────────────────────────────────────────────────────────
BOARD   = 100.0
CENTER  = 50.0
SUN_R   = 10.0
ROT_LIM = 50.0    # orbital_radius + planet_radius < ROT_LIM → orbiting
MAX_SPD = 6.0

COMET_SPAWN_STEPS = np.array([50, 150, 250, 350, 450], dtype=np.int32)

# Planet column indices
PI, PO, PX, PY, PR, PS, PP = 0, 1, 2, 3, 4, 5, 6
# Fleet column indices
FI, FO, FX, FY, FA, FF, FS = 0, 1, 2, 3, 4, 5, 6


# ── Geometry (all pure, vmappable) ────────────────────────────────────────────

def swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
    """True if fleet path A→B comes within r of planet path P0→P1."""
    d0x = ax - p0x;  d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a   = dvx*dvx + dvy*dvy
    b   = 2.0 * (d0x*dvx + d0y*dvy)
    c   = d0x*d0x + d0y*d0y - r*r
    disc = b*b - 4.0*a*c
    sq   = jnp.sqrt(jnp.maximum(disc, 0.0))
    safe_2a = jnp.where(a > 1e-12, 2.0*a, 1.0)
    t1 = (-b - sq) / safe_2a
    t2 = (-b + sq) / safe_2a
    stationary = c <= 0.0
    moving     = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    return jnp.where(a < 1e-12, stationary, moving)


def seg_pt_dist_sq(px, py, vx, vy, wx, wy):
    """Squared distance from point (px,py) to segment (vx,vy)→(wx,wy)."""
    l2 = (wx-vx)**2 + (wy-vy)**2
    t  = jnp.clip(
        ((px-vx)*(wx-vx) + (py-vy)*(wy-vy)) / jnp.where(l2 > 0.0, l2, 1.0),
        0.0, 1.0
    )
    qx = vx + t*(wx-vx);  qy = vy + t*(wy-vy)
    return (px-qx)**2 + (py-qy)**2


# ── Step function ─────────────────────────────────────────────────────────────

def step_env(state, actions_p0, actions_p1):
    """
    Advance game by one turn.

    actions_p{0,1}: [MAX_LAUNCHES, 3] float32
        Each row: [from_planet_id, angle_rad, num_ships]
        No-op rows: from_planet_id = -1

    Returns: new state dict.
    """
    step_num = state['step'] + 1

    planets    = state['planets']
    pv         = state['planet_valid']
    fleets     = state['fleets']
    fv         = state['fleet_valid']
    nfid       = state['next_fleet_id']
    av         = state['angular_vel']

    comet_path_idx = state['comet_path_idx']   # [5] int32
    comet_active   = state['comet_active']     # [5] bool
    comet_slots    = state['comet_slots']      # [5, 4] int32
    comet_lengths  = state['comet_lengths']    # [5, 4] int32
    comet_paths    = state['comet_paths']      # [5, 4, MAX_PATH_LEN, 2]
    comet_ships    = state['comet_ships']      # [5] int32

    # ── Phase 0: expire comets whose path ended last step ────────────────────
    # comet_path_idx[g] = index used in last movement (or -1 if never moved)
    # Expired when active and path_idx >= length for any comet in group
    # (all 4 comets in a group share path_idx, so check min length)
    min_lengths = comet_lengths.min(axis=1)   # [5]
    expired_group = comet_active & (comet_path_idx >= min_lengths)  # [5] bool

    # Scatter expired_group into per-planet-slot mask
    # planet_exp[slot] = OR over all (g,k) where comet_slots[g,k]==slot of expired_group[g]
    slots_flat   = comet_slots.reshape(-1)          # [20]
    exp_flat     = jnp.repeat(expired_group, N_COMETS_PER_GROUP)  # [20]

    def slot_is_expired(slot_idx):
        return jnp.any((slots_flat == slot_idx) & exp_flat)

    planet_expired = jax.vmap(slot_is_expired)(jnp.arange(MAX_PLANETS))  # [MAX_PLANETS]
    pv = pv & ~planet_expired

    # ── Phase 1: spawn comet group if this is a spawn step ───────────────────
    spawning = jnp.array(COMET_SPAWN_STEPS, dtype=jnp.int32) == step_num  # [5]
    new_comet_active = comet_active | spawning

    # For newly spawning groups, reset path_idx to -1 and activate planet slots
    new_comet_path_idx = jnp.where(spawning, -1, comet_path_idx)

    # Activate planet slots for spawning groups (position set to off-board)
    # Use scan over (g, k) pairs
    gk_idx = jnp.arange(N_COMET_GROUPS * N_COMETS_PER_GROUP)

    def spawn_comet(carry, idx):
        planets_c, pv_c = carry
        g = idx // N_COMETS_PER_GROUP
        k = idx  % N_COMETS_PER_GROUP
        slot = comet_slots[g, k]
        is_spawn = spawning[g]
        # Set off-board position; ships already set at init
        planets_c = jnp.where(
            is_spawn,
            planets_c.at[slot, PX].set(-99.0).at[slot, PY].set(-99.0)
                      .at[slot, PS].set(comet_ships[g].astype(jnp.float32))
                      .at[slot, PO].set(-1.0),
            planets_c
        )
        pv_c = jnp.where(is_spawn, pv_c.at[slot].set(True), pv_c)
        return (planets_c, pv_c), None

    (planets, pv), _ = jax.lax.scan(spawn_comet, (planets, pv), gk_idx)

    # ── Phase 2: fleet launch ─────────────────────────────────────────────────
    # Process all player actions sequentially (player 0 then 1)
    p0_tagged = jnp.concatenate([actions_p0, jnp.zeros((MAX_LAUNCHES, 1))], axis=1)
    p1_tagged = jnp.concatenate([actions_p1, jnp.ones((MAX_LAUNCHES, 1))], axis=1)
    all_actions = jnp.concatenate([p0_tagged, p1_tagged], axis=0)  # [2*MAX_LAUNCHES, 4]

    def launch_one(carry, act4):
        planets_c, fleets_c, fv_c, nfid_c = carry
        from_id   = act4[0]
        angle     = act4[1]
        n_ships   = act4[2].astype(jnp.int32)
        player_id = act4[3].astype(jnp.int32)

        # Find planet slot by ID
        pid_match   = (planets_c[:, PI] == from_id) & pv  # pv from outer scope
        planet_slot = jnp.argmax(pid_match)
        found       = jnp.any(pid_match)
        planet      = planets_c[planet_slot]

        valid = (
            found
            & (from_id >= 0.0)
            & (planet[PO].astype(jnp.int32) == player_id)
            & (n_ships > 0)
            & (n_ships <= planet[PS].astype(jnp.int32))
        )

        # First empty fleet slot (guard: fail launch if array is full)
        first_empty = jnp.argmax(~fv_c)
        has_slot    = ~fv_c[first_empty]
        valid       = valid & has_slot

        # Fleet start: just outside planet radius
        sx = planet[PX] + jnp.cos(angle) * (planet[PR] + 0.1)
        sy = planet[PY] + jnp.sin(angle) * (planet[PR] + 0.1)

        new_fleet = jnp.array([
            nfid_c.astype(jnp.float32), player_id.astype(jnp.float32),
            sx, sy, angle, from_id, n_ships.astype(jnp.float32)
        ])

        fleets_c  = jnp.where(valid, fleets_c.at[first_empty].set(new_fleet), fleets_c)
        fv_c      = jnp.where(valid, fv_c.at[first_empty].set(True), fv_c)
        new_ps    = planet[PS] - jnp.where(valid, n_ships.astype(jnp.float32), 0.0)
        planets_c = jnp.where(valid, planets_c.at[planet_slot, PS].set(new_ps), planets_c)
        nfid_c    = nfid_c + jnp.where(valid, 1, 0)

        return (planets_c, fleets_c, fv_c, nfid_c), None

    (planets, fleets, fv, nfid), _ = jax.lax.scan(
        launch_one, (planets, fleets, fv, nfid), all_actions
    )

    # ── Phase 3: production ───────────────────────────────────────────────────
    owned   = (planets[:, PO] >= 0.0) & pv
    planets = planets.at[:, PS].set(
        jnp.where(owned, planets[:, PS] + planets[:, PP], planets[:, PS])
    )

    # ── Phase 4: compute planet new positions ─────────────────────────────────
    # Orbiting planets: reference uses obs.step (before increment) for rotation.
    # obs.step=0 on first call → no rotation on step 1, av*1 on step 2, etc.
    cur_angle  = state['init_angle'] + av * state['step'].astype(jnp.float32)
    orbit_x    = CENTER + state['orb_r'] * jnp.cos(cur_angle)
    orbit_y    = CENTER + state['orb_r'] * jnp.sin(cur_angle)
    p_new_x    = jnp.where(state['is_orbiting'], orbit_x, planets[:, PX])
    p_new_y    = jnp.where(state['is_orbiting'], orbit_y, planets[:, PY])

    # Comet positions: increment path_idx, read from pre-computed paths
    # new_path_idx[g] = new_comet_path_idx[g] + 1  (for active groups)
    moved_path_idx = jnp.where(new_comet_active, new_comet_path_idx + 1, new_comet_path_idx)

    # Scatter comet new positions into p_new_x / p_new_y
    # For each (g, k): if active and idx < length, use paths[g,k,idx]
    #                  if active and idx >= length → expired (handled separately)
    # check_collision[slot] = False if this is the comet's FIRST movement (old_pos off-board)

    check_collision = jnp.ones(MAX_PLANETS, dtype=bool)  # default True

    def update_comet_pos(carry, idx):
        p_new_x_c, p_new_y_c, check_c = carry
        g    = idx // N_COMETS_PER_GROUP
        k    = idx  % N_COMETS_PER_GROUP
        slot = comet_slots[g, k]
        pidx = moved_path_idx[g]
        active = new_comet_active[g]
        in_bounds = pidx < comet_lengths[g, k]
        valid_move = active & in_bounds

        # Clamp index for safe array access
        safe_idx = jnp.clip(pidx, 0, MAX_PATH_LEN - 1)
        cx = comet_paths[g, k, safe_idx, 0]
        cy = comet_paths[g, k, safe_idx, 1]

        p_new_x_c = jnp.where(valid_move, p_new_x_c.at[slot].set(cx), p_new_x_c)
        p_new_y_c = jnp.where(valid_move, p_new_y_c.at[slot].set(cy), p_new_y_c)

        # No collision check on first placement (old pos was off-board)
        first_move = active & (pidx == 0)
        check_c = jnp.where(first_move, check_c.at[slot].set(False), check_c)

        return (p_new_x_c, p_new_y_c, check_c), None

    (p_new_x, p_new_y, check_collision), _ = jax.lax.scan(
        update_comet_pos, (p_new_x, p_new_y, check_collision), gk_idx
    )

    p_old_x = planets[:, PX]
    p_old_y = planets[:, PY]

    # ── Phase 5: fleet movement + collision detection ─────────────────────────
    speeds = 1.0 + (MAX_SPD - 1.0) * (
        jnp.log(jnp.maximum(fleets[:, FS], 1.0)) / jnp.log(1000.0)
    ) ** 1.5
    speeds = jnp.minimum(speeds, MAX_SPD)

    f_old_x = fleets[:, FX]
    f_old_y = fleets[:, FY]
    f_new_x = f_old_x + jnp.cos(fleets[:, FA]) * speeds
    f_new_y = f_old_y + jnp.sin(fleets[:, FA]) * speeds

    # Collision matrix: [MAX_FLEETS, MAX_PLANETS]
    # hit[f, p] = swept_pair_hit(fleet_old, fleet_new, planet_old, planet_new, radius)
    # Vectorise with vmap over fleets, then over planets
    def fleet_vs_all_planets(fox, foy, fnx, fny):
        def vs_planet(pox, poy, pnx, pny, pr, pvalid, check):
            h = swept_pair_hit(fox, foy, fnx, fny, pox, poy, pnx, pny, pr)
            return h & pvalid & check
        return jax.vmap(vs_planet)(p_old_x, p_old_y, p_new_x, p_new_y,
                                   planets[:, PR], pv, check_collision)

    hits = jax.vmap(fleet_vs_all_planets)(f_old_x, f_old_y, f_new_x, f_new_y)
    # hits: [MAX_FLEETS, MAX_PLANETS]

    # For each fleet: first planet hit (lowest planet index)
    valid_hits = hits & pv[None, :]
    first_hit = jnp.argmin(
        jnp.where(valid_hits, jnp.arange(MAX_PLANETS)[None, :], MAX_PLANETS),
        axis=1
    )   # [MAX_FLEETS], MAX_PLANETS = no hit
    any_hit = jnp.any(valid_hits, axis=1)  # [MAX_FLEETS]

    # Fleets that hit nothing: check OOB and sun
    oob = ~((0.0 <= f_new_x) & (f_new_x <= BOARD) & (0.0 <= f_new_y) & (f_new_y <= BOARD))
    sun_hit = seg_pt_dist_sq(CENTER, CENTER, f_old_x, f_old_y, f_new_x, f_new_y) < SUN_R**2

    fleet_removed = fv & (any_hit | oob | sun_hit)
    fleet_alive   = fv & ~fleet_removed

    # Update fleet positions for those that survived
    fleets = fleets.at[:, FX].set(jnp.where(fleet_alive, f_new_x, fleets[:, FX]))
    fleets = fleets.at[:, FY].set(jnp.where(fleet_alive, f_new_y, fleets[:, FY]))
    fv = fleet_alive

    # ── Phase 6: apply planet movement ───────────────────────────────────────
    planets = planets.at[:, PX].set(jnp.where(pv, p_new_x, planets[:, PX]))
    planets = planets.at[:, PY].set(jnp.where(pv, p_new_y, planets[:, PY]))

    # Expire comets that just ran out of path
    exp_this_step = jnp.zeros(MAX_PLANETS, dtype=bool)

    def mark_expired(carry, idx):
        exp_c = carry
        g    = idx // N_COMETS_PER_GROUP
        k    = idx  % N_COMETS_PER_GROUP
        slot = comet_slots[g, k]
        pidx = moved_path_idx[g]
        just_expired = new_comet_active[g] & (pidx >= comet_lengths[g, k])
        exp_c = jnp.where(just_expired, exp_c.at[slot].set(True), exp_c)
        return exp_c, None

    exp_this_step, _ = jax.lax.scan(mark_expired, exp_this_step, gk_idx)
    pv = pv & ~exp_this_step

    # ── Phase 7: combat resolution ────────────────────────────────────────────
    # arriving_ships[p, player] = ships from player that hit planet p this step
    first_hit_mask = (jnp.arange(MAX_PLANETS)[None, :] == first_hit[:, None]) & any_hit[:, None]
    # [MAX_FLEETS, MAX_PLANETS]

    fleet_owners_oh = (
        fleets[:, FO].astype(jnp.int32)[:, None, None]
        == jnp.arange(N_PLAYERS)[None, None, :]
    )  # [MAX_FLEETS, 1, N_PLAYERS]

    hit_contrib = (
        first_hit_mask[:, :, None].astype(jnp.float32)
        * fleet_owners_oh.astype(jnp.float32)
        * fleets[:, FS][:, None, None]
        * fv[:, None, None].astype(jnp.float32)  # only removed fleets count
        # wait — we already set fv = fleet_alive above. We need the REMOVED fleets.
    )
    # Actually we need to use fleet_removed, not fv, because arriving fleets
    # are the ones that were removed by hitting a planet. Let me fix this.
    # fleet_removed = fleets that hit a planet (any_hit & fv_before)
    # We already computed fleet_removed above. Let's use that mask.
    hit_contrib = (
        first_hit_mask[:, :, None].astype(jnp.float32)
        * fleet_owners_oh.astype(jnp.float32)
        * fleets[:, FS][:, None, None]
        * fleet_removed[:, None, None].astype(jnp.float32)
    )
    arriving = hit_contrib.sum(axis=0)  # [MAX_PLANETS, N_PLAYERS]

    def resolve_planet(planet, pvalid, arr):
        """Resolve combat at one planet."""
        any_attack = jnp.any(arr > 0.0)

        top_idx    = jnp.argmax(arr)
        top_ships  = arr[top_idx]
        arr_no_top = arr.at[top_idx].set(0.0)
        second_ships = arr_no_top.max()

        tied          = (top_ships == second_ships) & (top_ships > 0.0)
        survivor_sh   = jnp.where(tied, 0.0, top_ships - second_ships)
        survivor_own  = jnp.where(survivor_sh > 0.0, top_idx.astype(jnp.float32), -1.0)

        garrison      = planet[PS]
        planet_own    = planet[PO]

        reinforce     = (survivor_own == planet_own) & (survivor_sh > 0.0)
        fight         = (survivor_own != planet_own) & (survivor_sh > 0.0)
        conquered     = fight & (survivor_sh > garrison)

        new_garrison  = jnp.where(
            reinforce, garrison + survivor_sh,
            jnp.where(fight, jnp.abs(garrison - survivor_sh), garrison)
        )
        new_own       = jnp.where(conquered, survivor_own, planet_own)

        new_planet    = planet.at[PS].set(new_garrison).at[PO].set(new_own)
        return jnp.where(any_attack & pvalid, new_planet, planet)

    planets = jax.vmap(resolve_planet)(planets, pv, arriving)

    # ── Phase 8: termination ─────────────────────────────────────────────────
    alive_mask = pv & (planets[:, PO] >= 0.0)
    p0_alive   = jnp.any(alive_mask & (planets[:, PO] == 0.0))
    p1_alive   = jnp.any(alive_mask & (planets[:, PO] == 1.0))
    # Also check fleets
    p0_alive   = p0_alive | jnp.any(fv & (fleets[:, FO] == 0.0))
    p1_alive   = p1_alive | jnp.any(fv & (fleets[:, FO] == 1.0))

    step_done  = step_num >= 498  # ref: step >= episodeSteps - 2
    elim_done  = ~p0_alive | ~p1_alive
    done       = state['done'] | step_done | elim_done

    return {
        **state,
        'planets':        planets,
        'planet_valid':   pv,
        'fleets':         fleets,
        'fleet_valid':    fv,
        'comet_path_idx': moved_path_idx,
        'comet_active':   new_comet_active,
        'step':           step_num,
        'next_fleet_id':  nfid,
        'done':           done,
    }


# ── Initialisation ────────────────────────────────────────────────────────────

def init_state(seed: int, n_players: int = 2):
    """
    Build JAX env state from the reference Python env.

    Runs the reference env with no-op agents for 460 steps to extract exact
    comet paths (which are seed-deterministic, independent of agent actions).
    This guarantees full parity with the reference for all 5 comet groups.
    """
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset()
    obs0 = env.state[0].observation

    ref_planets  = obs0.planets[:]
    init_planets = obs0.initial_planets[:]
    n_reg        = len(ref_planets)
    angular_vel  = float(obs0.angular_velocity)

    assert n_reg <= MAX_PLANETS - N_COMET_GROUPS * N_COMETS_PER_GROUP, \
        f"Too many regular planets: {n_reg}"

    # ── Regular planet geometry ───────────────────────────────────────────────
    planets_np    = np.zeros((MAX_PLANETS, 7), dtype=np.float32)
    pv_np         = np.zeros(MAX_PLANETS, dtype=bool)
    is_orb_np     = np.zeros(MAX_PLANETS, dtype=bool)
    orb_r_np      = np.zeros(MAX_PLANETS, dtype=np.float32)
    init_angle_np = np.zeros(MAX_PLANETS, dtype=np.float32)

    for i, (p, ip) in enumerate(zip(ref_planets, init_planets)):
        planets_np[i] = p
        pv_np[i]      = True
        dx = ip[2] - CENTER;  dy = ip[3] - CENTER
        r  = math.sqrt(dx**2 + dy**2)
        orb_r_np[i]      = r
        init_angle_np[i] = math.atan2(dy, dx)
        if r + p[4] < ROT_LIM:
            is_orb_np[i] = True

    # ── Collect exact comet paths from reference run ──────────────────────────
    # Comet paths are fully determined by episode seed — independent of actions.
    # The reference env reuses the same planet IDs for each comet group, so we
    # collect data at the exact spawn step rather than keying by planet_ids.
    spawn_steps_set = set(int(s) for s in COMET_SPAWN_STEPS)
    comet_groups = {}   # spawn_step (int) → {paths, ships}

    for noop_step in range(1, 461):
        env.step([[], []])
        obs = env.state[0].observation
        if obs.comets and noop_step in spawn_steps_set and noop_step not in comet_groups:
            group = obs.comets[0]
            pid   = group['planet_ids'][0]
            shps  = next((int(p[5]) for p in obs.planets if p[0] == pid), 0)
            comet_groups[noop_step] = {'paths': group['paths'], 'ships': shps}

    # ── Build comet arrays ────────────────────────────────────────────────────
    comet_paths_np    = np.full((N_COMET_GROUPS, N_COMETS_PER_GROUP, MAX_PATH_LEN, 2),
                                -99.0, dtype=np.float32)
    comet_lengths_np  = np.zeros((N_COMET_GROUPS, N_COMETS_PER_GROUP), dtype=np.int32)
    comet_slots_np    = np.zeros((N_COMET_GROUPS, N_COMETS_PER_GROUP), dtype=np.int32)
    comet_ships_np    = np.zeros(N_COMET_GROUPS, dtype=np.int32)
    comet_path_idx_np = np.full(N_COMET_GROUPS, -1, dtype=np.int32)
    comet_active_np   = np.zeros(N_COMET_GROUPS, dtype=bool)

    comet_slot_base = MAX_PLANETS - N_COMET_GROUPS * N_COMETS_PER_GROUP  # = 44

    for g, spawn_step in enumerate(COMET_SPAWN_STEPS):
        sp  = int(spawn_step)
        dat = comet_groups.get(sp)
        n_ships = dat['ships'] if dat else 0
        comet_ships_np[g] = n_ships

        if dat is not None:
            for k, path in enumerate(dat['paths']):
                plen = min(len(path), MAX_PATH_LEN)
                comet_lengths_np[g, k] = plen
                for t, pos in enumerate(path[:plen]):
                    comet_paths_np[g, k, t, 0] = float(pos[0])
                    comet_paths_np[g, k, t, 1] = float(pos[1])

        for k in range(N_COMETS_PER_GROUP):
            slot = comet_slot_base + g * N_COMETS_PER_GROUP + k
            comet_slots_np[g, k] = slot
            next_id = n_reg + g * N_COMETS_PER_GROUP + k
            planets_np[slot] = [next_id, -1, -99.0, -99.0, 1.0, float(n_ships), 1.0]

    state = {
        'planets':        jnp.array(planets_np),
        'planet_valid':   jnp.array(pv_np),
        'is_orbiting':    jnp.array(is_orb_np),
        'orb_r':          jnp.array(orb_r_np),
        'init_angle':     jnp.array(init_angle_np),
        'fleets':         jnp.zeros((MAX_FLEETS, 7), dtype=jnp.float32),
        'fleet_valid':    jnp.zeros(MAX_FLEETS, dtype=bool),
        'comet_paths':    jnp.array(comet_paths_np),
        'comet_lengths':  jnp.array(comet_lengths_np),
        'comet_path_idx': jnp.array(comet_path_idx_np),
        'comet_active':   jnp.array(comet_active_np),
        'comet_slots':    jnp.array(comet_slots_np),
        'comet_ships':    jnp.array(comet_ships_np),
        'angular_vel':    jnp.array(angular_vel, dtype=jnp.float32),
        'step':           jnp.array(0, dtype=jnp.int32),
        'next_fleet_id':  jnp.array(0, dtype=jnp.int32),
        'done':           jnp.array(False),
    }
    return state
