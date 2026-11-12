"""
Frame-by-frame parity test: JAX env vs reference Python env.

Tests steps 0-48 (before first comet spawn at step 50) to verify
fleet launch, movement, collision, production, combat, and planet rotation.

Usage:
    python parity_test.py           # steps 0-48, seed 42
    python parity_test.py --steps 100 --seed 7
"""

import argparse
import math
import sys

import numpy as np
import jax.numpy as jnp

from env import init_state, step_env, MAX_LAUNCHES, CENTER, ROT_LIM

ATOL = 1e-3  # float tolerance for position/ship comparisons


# ── Simple deterministic agent ────────────────────────────────────────────────

def nearest_agent(obs, player_id):
    """
    Greedy: for each owned planet, send half ships toward nearest enemy/neutral.
    Returns list of [from_id, angle, ships] moves.
    """
    planets = obs.planets if hasattr(obs, 'planets') else obs.get('planets', [])
    moves = []
    my_planets  = [p for p in planets if p[1] == player_id]
    targets     = [p for p in planets if p[1] != player_id]
    if not targets:
        return moves
    for mp in my_planets:
        ships = int(mp[5]) // 2
        if ships <= 0:
            continue
        best = min(targets, key=lambda t: math.hypot(mp[2]-t[2], mp[3]-t[3]))
        angle = math.atan2(best[3]-mp[3], best[2]-mp[2])
        moves.append([int(mp[0]), angle, ships])
    return moves


def moves_to_jax_actions(moves, player_id=None):
    """Pack a move list into [MAX_LAUNCHES, 3] jnp array (no-ops = -1)."""
    arr = np.full((MAX_LAUNCHES, 3), -1.0, dtype=np.float32)
    for i, m in enumerate(moves[:MAX_LAUNCHES]):
        arr[i] = [float(m[0]), float(m[1]), float(m[2])]
    return jnp.array(arr)


# ── Extract comparable state from reference env ───────────────────────────────

def planet_list(obs_or_state, is_jax=False):
    """Return list of (x, y, owner, ships) for all active planets."""
    if is_jax:
        planets = np.array(obs_or_state['planets'])
        valid   = np.array(obs_or_state['planet_valid'])
        return [(float(planets[i, 2]), float(planets[i, 3]),
                 int(planets[i, 1]), float(planets[i, 5]))
                for i in range(len(valid)) if valid[i]]
    else:
        return [(float(p[2]), float(p[3]), int(p[1]), float(p[5]))
                for p in obs_or_state.planets]


def ref_fleet_state(obs):
    """Return sorted list of (owner, x, y, ships) for active fleets."""
    fleets = [(int(f[1]), float(f[2]), float(f[3]), float(f[6]))
              for f in obs.fleets]
    return sorted(fleets)


def jax_fleet_state(state):
    fleets  = np.array(state['fleets'])
    valid   = np.array(state['fleet_valid'])
    out = [(int(fleets[i, 1]), float(fleets[i, 2]), float(fleets[i, 3]), float(fleets[i, 6]))
           for i in range(len(valid)) if valid[i]]
    return sorted(out)


# ── Comparison helpers ────────────────────────────────────────────────────────

def compare_step(step_n, ref_obs, jax_state, verbose=False):
    ref_p = planet_list(ref_obs, is_jax=False)
    jax_p = planet_list(jax_state, is_jax=True)

    errors = []

    # Match planets by position (positions are unique; IDs can differ for comets)
    if len(ref_p) != len(jax_p):
        errors.append(f"Planet count: ref={len(ref_p)} jax={len(jax_p)}")
    else:
        ref_sorted = sorted(ref_p)
        jax_sorted = sorted(jax_p)
        for i, (rp, jp) in enumerate(zip(ref_sorted, jax_sorted)):
            rx, ry, ro, rs = rp
            jx, jy, jo, js = jp
            if abs(rx-jx) > ATOL or abs(ry-jy) > ATOL:
                errors.append(f"  Planet {i}: pos ref=({rx:.3f},{ry:.3f}) jax=({jx:.3f},{jy:.3f})")
            elif ro != jo:
                errors.append(f"  Planet {i} at ({rx:.1f},{ry:.1f}): owner ref={ro} jax={jo}")
            elif abs(rs-js) > ATOL:
                errors.append(f"  Planet {i} at ({rx:.1f},{ry:.1f}): ships ref={rs:.1f} jax={js:.1f}")

    # Check fleets: match each ref fleet to closest JAX fleet (Hungarian-style greedy)
    ref_f = ref_fleet_state(ref_obs)
    jax_f = jax_fleet_state(jax_state)
    if len(ref_f) != len(jax_f):
        errors.append(f"  Fleet count: ref={len(ref_f)} jax={len(jax_f)}")
    else:
        # Match by (owner, ships) groups, then compare positions
        from collections import defaultdict
        ref_by_key = defaultdict(list)
        jax_by_key = defaultdict(list)
        for f in ref_f:
            ref_by_key[(f[0], int(f[3]))].append((f[1], f[2]))
        for f in jax_f:
            jax_by_key[(f[0], int(f[3]))].append((f[1], f[2]))
        for key in ref_by_key:
            rlist = sorted(ref_by_key[key])
            jlist = sorted(jax_by_key.get(key, []))
            if len(rlist) != len(jlist):
                errors.append(f"  Fleet group owner={key[0]} ships={key[1]}: count ref={len(rlist)} jax={len(jlist)}")
            else:
                for i, (rp, jp) in enumerate(zip(rlist, jlist)):
                    if abs(rp[0]-jp[0]) > ATOL or abs(rp[1]-jp[1]) > ATOL:
                        errors.append(f"  Fleet owner={key[0]} ships={key[1]}: pos ref=({rp[0]:.3f},{rp[1]:.3f}) jax=({jp[0]:.3f},{jp[1]:.3f})")

    if errors:
        print(f"FAIL step {step_n}:")
        for e in errors:
            print(e)
        return False
    if verbose:
        print(f"  step {step_n:3d}: OK  planets={len(ref_p)} fleets={len(ref_fleet_state(ref_obs))}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def run_parity(seed=42, max_steps=49, verbose=True):
    from kaggle_environments import make

    print(f"Parity test: seed={seed}, steps=0..{max_steps}")
    print("Initialising reference env...")
    ref_env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    ref_env.reset()

    print("Initialising JAX env...")
    jax_state = init_state(seed=seed, n_players=2)

    # Check initial state (step 0)
    ref_obs = ref_env.state[0].observation
    ok = compare_step(0, ref_obs, jax_state, verbose=verbose)
    if not ok:
        print("FAIL at step 0 (initial state)")
        return False

    passed = 0
    for s in range(1, max_steps + 1):
        # Get actions from deterministic agent on current obs
        ref_obs = ref_env.state[0].observation
        moves_p0 = nearest_agent(ref_obs, player_id=0)
        moves_p1 = nearest_agent(ref_obs, player_id=1)

        # Step reference env
        ref_env.step([moves_p0, moves_p1])
        ref_obs_after = ref_env.state[0].observation

        # Step JAX env
        act_p0 = moves_to_jax_actions(moves_p0)
        act_p1 = moves_to_jax_actions(moves_p1)
        jax_state = step_env(jax_state, act_p0, act_p1)

        ok = compare_step(s, ref_obs_after, jax_state, verbose=verbose)
        if not ok:
            print(f"FAIL at step {s}")
            return False
        passed += 1

    print(f"\nPASSED {passed}/{max_steps} steps")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--steps", type=int, default=49)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ok = run_parity(seed=args.seed, max_steps=args.steps, verbose=not args.quiet)
    sys.exit(0 if ok else 1)
