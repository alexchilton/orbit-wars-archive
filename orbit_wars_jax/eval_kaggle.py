"""
Evaluate the JAX RL agent against baselines using the real Kaggle environment.

The JAX model is wrapped as a proper agent(obs, config) function, so this
tests exactly what would be submitted to the leaderboard.

Usage:
    python3 -u eval_kaggle.py [--ckpt path] [--games N] [--seed S]
"""

import argparse
import glob
import math
import os
import pickle
import sys

import numpy as np
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from env import (MAX_PLANETS, MAX_FLEETS,
                 PI, PO, PX, PY, PR, PS, PP,
                 FI, FO, FX, FY, FA, FF, FS,
                 CENTER, ROT_LIM)
from obs import encode_obs
from act import greedy_actions
from model import OrbitWarsModel


# ── Build a minimal JAX state from a Kaggle observation ──────────────────────

def obs_to_jax_state(obs):
    """Convert a Kaggle orbit_wars observation to a minimal JAX state dict."""
    planets_np = np.zeros((MAX_PLANETS, 7), dtype=np.float32)
    pv_np      = np.zeros(MAX_PLANETS, dtype=bool)
    is_orb_np  = np.zeros(MAX_PLANETS, dtype=bool)

    for p in obs.planets:
        # p = [id, owner, x, y, radius, ships, production]
        pid = int(p[0])
        if 0 <= pid < MAX_PLANETS:
            planets_np[pid] = [p[0], p[1], p[2], p[3], p[4], p[5], p[6]]
            pv_np[pid] = True
            r = math.sqrt((p[2] - CENTER)**2 + (p[3] - CENTER)**2)
            if r + p[4] < ROT_LIM:
                is_orb_np[pid] = True

    fleets_np = np.zeros((MAX_FLEETS, 7), dtype=np.float32)
    fv_np     = np.zeros(MAX_FLEETS, dtype=bool)

    for i, f in enumerate(obs.fleets):
        if i >= MAX_FLEETS:
            break
        # f = [id, owner, x, y, angle, from_planet_id, ships]
        fleets_np[i] = [f[0], f[1], f[2], f[3], f[4], f[5], f[6]]
        fv_np[i] = True

    return {
        'planets':      jnp.array(planets_np),
        'planet_valid': jnp.array(pv_np),
        'fleets':       jnp.array(fleets_np),
        'fleet_valid':  jnp.array(fv_np),
        'is_orbiting':  jnp.array(is_orb_np),
    }


# ── Convert JAX actions to Kaggle format ─────────────────────────────────────

def jax_actions_to_kaggle(actions_jax):
    """Convert [MAX_LAUNCHES, 3] JAX array → list of [planet_id, angle, ships]."""
    arr = np.array(actions_jax)
    moves = []
    for row in arr:
        if row[0] >= 0:
            moves.append([int(row[0]), float(row[1]), int(row[2])])
    return moves


# ── Build Kaggle-compatible agent closure ─────────────────────────────────────

def make_jax_agent(params, model):
    """Return an agent(obs, config) function backed by the JAX model."""
    def agent(obs, config):
        state = obs_to_jax_state(obs)
        player_id = int(obs.player)
        enc = encode_obs(state, player_id)
        tgt, frac, _ = model.apply(params, enc)
        actions = greedy_actions(tgt, frac, enc, state)
        return jax_actions_to_kaggle(actions)
    return agent


# ── Load v131 ─────────────────────────────────────────────────────────────────

def load_v131():
    candidates = [
        os.path.join(os.path.dirname(HERE), 'orbit_wars_agent.py'),
        os.path.expanduser('~/orbit_wars_agent.py'),
    ]
    for path in candidates:
        if os.path.exists(path):
            sys.path.insert(0, os.path.dirname(path))
            import orbit_wars_agent
            return orbit_wars_agent.agent
    raise FileNotFoundError("orbit_wars_agent.py not found")


# ── Run one game using the Kaggle env ─────────────────────────────────────────

def run_kaggle_game(agent0, agent1, seed):
    """
    Run one game in the real Kaggle env.
    agent0 plays as player 0, agent1 as player 1.
    Returns 0 or 1 (winner index).
    """
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    out = env.run([agent0, agent1])
    # Rewards: +1 for winner, -1 for loser
    r0 = out[-1][0].reward
    r1 = out[-1][1].reward
    return 0 if r0 >= r1 else 1


# ── Matchup ───────────────────────────────────────────────────────────────────

def matchup(name, rl_agent, baseline_agent, seeds):
    n = len(seeds)
    rl_wins = 0
    for i, seed in enumerate(seeds):
        if i % 2 == 0:
            winner = run_kaggle_game(rl_agent, baseline_agent, seed)
            if winner == 0:
                rl_wins += 1
        else:
            winner = run_kaggle_game(baseline_agent, rl_agent, seed)
            if winner == 1:
                rl_wins += 1
    wr = rl_wins / n
    print(f"  vs {name:<10s}: {rl_wins}/{n} wins  ({wr*100:.1f}%)")
    return wr


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',  default=None)
    parser.add_argument('--games', type=int, default=10)
    parser.add_argument('--seed',  type=int, default=100)
    args = parser.parse_args()

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpts = sorted(glob.glob(os.path.join(HERE, 'checkpoints', 'ckpt_*.pkl')))
        if not ckpts:
            ckpts = sorted(glob.glob(os.path.join(HERE, 'checkpoints', 'params_*.pkl')))
        ckpt_path = ckpts[-1]

    print(f"Loading checkpoint: {ckpt_path}")
    with open(ckpt_path, 'rb') as f:
        data = pickle.load(f)
    params = data['params'] if isinstance(data, dict) and 'params' in data else data

    model = OrbitWarsModel()
    rl_agent = make_jax_agent(params, model)

    seeds = list(range(args.seed, args.seed + args.games))

    print(f"\nRunning {args.games} games per matchup (seed base={args.seed})")
    print("-" * 50)

    def random_agent(obs, config):
        """Launches half ships from each owned planet to a random target."""
        import random as _random
        player_id = int(obs.player)
        my     = [p for p in obs.planets if int(p[1]) == player_id]
        others = [p for p in obs.planets if int(p[1]) != player_id]
        if not my or not others:
            return []
        moves = []
        for mp in my:
            ships = int(mp[5]) // 2
            if ships <= 0:
                continue
            tgt = _random.choice(others)
            angle = math.atan2(tgt[3]-mp[3], tgt[2]-mp[2])
            moves.append([int(mp[0]), angle, ships])
        return moves

    matchup('random',  rl_agent, random_agent,  seeds)

    # Pool agents (difficulty ladder: rage < prospector < dual < bully)
    # Pool agents take agent(obs) only — wrap to (obs, config) for Kaggle env
    _pool_dir = os.path.expanduser('~/DataspellProjects/orbit_wars/submission')
    _pool_agents = [
        ('rage',       'pool_rage'),
        ('prospector', 'pool_prospector'),
        ('dual',       'pool_dual'),
        ('bully',      'pool_bully'),
    ]
    if _pool_dir not in sys.path:
        sys.path.insert(0, _pool_dir)
    for label, module_name in _pool_agents:
        try:
            mod = __import__(module_name)
            _fn = mod.agent
            def pool_agent(obs, config, fn=_fn):
                return fn(obs)
            matchup(label, rl_agent, pool_agent, seeds)
        except Exception as e:
            print(f"  {label}: skipped ({e})")

    print("-" * 50)


if __name__ == '__main__':
    main()
