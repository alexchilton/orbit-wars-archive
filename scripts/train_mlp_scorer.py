"""
train_mlp_scorer.py — Train MLP to approximate physics-sim action scores.

Training signal: REGRESSION on sim-improvement score per action.
  For each candidate (src, tgt) action at each game step:
    1. Run the physics sim 25 steps with that action applied → future_score
    2. Run the physics sim 25 steps with NO action → baseline_score
    3. Label = future_score - baseline_score  (positive = good action)

The MLP learns: features(src, tgt) → expected state improvement.
This is "amortised simulation" — MLP approximates the expensive sim at inference.

Advantages over binary win/loss labelling:
  • Fine-grained signal per action (not noisy game-level outcome)
  • Works regardless of opponent (even vs random)
  • Generates data from any agent gameplay; signal comes from the sim, not the winner
  • MLP can generalise beyond the generating agent's behaviour

Usage:
    python train_mlp_scorer.py --games 100 --out submission/mlp_scorer.pkl

Output pickle: {'layers': [{'W': ..., 'b': ...}, ...], 'means': ..., 'stds': ...}
Pure-numpy inference — no torch needed in the submission.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pickle
import random
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, 'submission'))
sys.path.insert(0, _ROOT)

import kaggle_environments  # noqa: E402

from submission.physics_sim import (  # noqa: E402
    parse_obs, predict, copy_state,
)
from submission.main_mlp_2p import (  # noqa: E402
    ships_needed_for_takeover,
    path_crosses_sun,
    multi_leg_path,
    solve_intercept,
    action_features,
)

FEATURE_DIM  = 16
_SIM_HORIZON = 25   # sim steps for scoring
_PROD_HORIZON = 40  # production valuation window


def load_agent(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


# ── State scorer (identical to main_pred_2p._score_state) ─────────────────────

def _score_state(state: dict, me: int, remaining_steps: int) -> float:
    my_prod = ep_prod = my_ships = ep_ships = 0.0
    for p in state['planets'].values():
        if p['owner'] == me:
            my_prod  += p['prod']
            my_ships += p['ships']
        elif p['owner'] >= 0:
            ep_prod  += p['prod']
            ep_ships += p['ships']
    prod_horizon = min(remaining_steps, _PROD_HORIZON)
    score = (my_prod * prod_horizon + my_ships) - (ep_prod * prod_horizon + ep_ships)
    for fl in state['fleets']:
        if fl['owner'] == me:
            score += fl['ships'] * 0.5
        elif fl['owner'] >= 0:
            score -= fl['ships'] * 0.5
    return score


# ── Sim-based data collection ──────────────────────────────────────────────────

def collect_step_samples(state: dict, me: int, step_limit: int = 60) -> list[tuple]:
    """
    At a single game step, score all candidate actions via physics sim.
    Returns [(features_16, improvement_score), ...].
    """
    if state['step'] > step_limit:
        return []   # skip late game — production advantage is mostly locked in

    samples = []
    omega   = state.get('omega', 0.01)
    rem     = max(1, 400 - state['step'])

    my_pids  = {pid for pid, p in state['planets'].items() if p['owner'] == me}
    ep_pids  = {pid for pid, p in state['planets'].items()
                if p['owner'] >= 0 and p['owner'] != me}
    neu_pids = {pid for pid, p in state['planets'].items() if p['owner'] < 0}
    tgt_set  = ep_pids | neu_pids

    # Baseline: simulate doing nothing for _SIM_HORIZON steps
    try:
        baseline_future = predict(state, _SIM_HORIZON)
        baseline_score  = _score_state(baseline_future, me, rem - _SIM_HORIZON)
    except Exception:
        return []

    for src_pid in my_pids:
        src = state['planets'][src_pid]
        avail = src['ships']
        if avail < 10:
            continue

        for tgt_pid in tgt_set:
            tgt = state['planets'][tgt_pid]
            try:
                ix, iy, eta = solve_intercept(
                    src['x'], src['y'], tgt['x'], tgt['y'],
                    tgt['is_orb'], omega, int(avail)
                )
                if path_crosses_sun(src['x'], src['y'], ix, iy):
                    wp, _ = multi_leg_path(src['x'], src['y'], ix, iy)
                    if wp is None:
                        continue

                needed = ships_needed_for_takeover(
                    tgt['ships'], tgt['prod'], eta, tgt['owner']
                )
                if avail < needed:
                    continue

                ships = needed if tgt['owner'] < 0 else min(
                    max(needed, int(avail * 0.65)), int(avail * 0.90)
                )
                if ships < 1:
                    continue

                # Simulate this action
                st = copy_state(state)
                st['planets'][src_pid]['ships'] = max(
                    0, st['planets'][src_pid]['ships'] - ships
                )
                st['fleets'].append({
                    'owner': me, 'ships': ships,
                    'target_pid': tgt_pid, 'eta': eta,
                })
                future       = predict(st, _SIM_HORIZON)
                action_score = _score_state(future, me, rem - _SIM_HORIZON)

                improvement = action_score - baseline_score
                feat        = action_features(
                    state, src_pid, tgt_pid, eta, ships, needed, me
                )
                samples.append((feat, improvement))

            except Exception:
                pass

    return samples


def collect_game_data(agent_a, agent_b, seed: int,
                      sample_every: int = 3,
                      collect_both_sides: bool = True) -> list[tuple]:
    """
    Run one game (agent_a vs agent_b) and collect sim-based samples.
    If collect_both_sides=True, also collects from agent_b's perspective —
    this ensures the training distribution covers losing positions too.
    """
    all_samples: list[tuple] = []
    step_counters = [0, 0]

    def make_wrap(player_slot):
        def wrap(obs, cfg=None):
            state = parse_obs(obs)
            me    = state['me']

            if step_counters[player_slot] % sample_every == 0:
                s = collect_step_samples(state, me)
                all_samples.extend(s)
            step_counters[player_slot] += 1

            agent = agent_a if player_slot == 0 else agent_b
            return agent(obs)
        return wrap

    if collect_both_sides:
        agents = [make_wrap(0), make_wrap(1)]
    else:
        agents = [make_wrap(0), agent_b]

    env = kaggle_environments.make(
        'orbit_wars', debug=False, configuration={'seed': seed}
    )
    env.run(agents)
    return all_samples


# ── 3-layer MLP: train in PyTorch, export to pure-numpy ───────────────────────

class MLP:
    def __init__(self, hidden1: int = 64, hidden2: int = 32):
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 300, lr: float = 1e-3, batch_size: int = 4096):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        opt     = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.MSELoss()

        for ep in range(epochs):
            self.net.train()
            total_loss = 0.0
            for Xb, yb in dl:
                opt.zero_grad()
                loss = loss_fn(self.net(Xb), yb)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(Xb)
            sched.step()
            if ep % 50 == 0:
                avg = total_loss / len(X)
                with torch.no_grad():
                    pred = self.net(Xt).squeeze().numpy()
                corr = float(np.corrcoef(pred, y)[0, 1]) if y.std() > 0 else 0.0
                print(f"  ep {ep:3d}  mse={avg:.2f}  corr={corr:.3f}")

    def export_numpy(self) -> dict:
        layers = []
        for layer in self.net:
            if hasattr(layer, 'weight'):
                layers.append({
                    'W': layer.weight.detach().numpy().copy(),
                    'b': layer.bias.detach().numpy().copy(),
                })
        return {'layers': layers}


def numpy_predict(model_dict: dict, x: np.ndarray) -> float:
    """Pure-numpy forward pass (no torch needed at inference)."""
    h = x.copy()
    layers = model_dict['layers']
    for i, lay in enumerate(layers):
        h = h @ lay['W'].T + lay['b']
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    return float(h[0])


# ── Main ───────────────────────────────────────────────────────────────────────

def collect_replay_data(replay_dir: str, sample_every: int = 3,
                        max_replays: int = 0) -> list[tuple]:
    """Extract sim-labelled samples from bovard replay JSONs.

    For each replay step, reconstruct the game state from the observation,
    then run collect_step_samples() to generate physics-sim improvement labels.
    """
    import json

    all_samples: list[tuple] = []
    replay_files = []
    for root, _dirs, files in os.walk(replay_dir):
        for fname in files:
            if fname.endswith('.json'):
                replay_files.append(os.path.join(root, fname))
    random.shuffle(replay_files)
    if max_replays > 0:
        replay_files = replay_files[:max_replays]

    errors = 0
    for ri, rpath in enumerate(replay_files):
        try:
            with open(rpath) as f:
                data = json.load(f)
            if 'steps' not in data:
                continue
            steps = data['steps']
            num_players = len(steps[0])
            # Only use winning player's perspective for quality signal
            rewards = data.get('rewards', [0] * num_players)
            winner = max(range(num_players), key=lambda i: rewards[i] or -999)

            for step_idx, step in enumerate(steps):
                if step_idx % sample_every != 0:
                    continue
                agent_data = step[winner]
                obs = agent_data.get('observation')
                if obs is None or 'planets' not in obs:
                    continue
                try:
                    state = parse_obs(obs)
                    s = collect_step_samples(state, winner)
                    all_samples.extend(s)
                except Exception:
                    pass
        except Exception:
            errors += 1

        if (ri + 1) % 100 == 0:
            print(f"  {ri+1}/{len(replay_files)} replays, "
                  f"{len(all_samples)} samples ({errors} errors)")

    print(f"Parsed {len(replay_files)} replays ({errors} errors), "
          f"{len(all_samples)} samples")
    return all_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games',       type=int, default=100)
    parser.add_argument('--out',         type=str, default='submission/mlp_scorer.pkl')
    parser.add_argument('--epochs',      type=int, default=300)
    parser.add_argument('--agent',       type=str,
                        default='submission/main_v131_plus_denial.py',
                        help='Agent whose steps we collect sim data from')
    parser.add_argument('--vs',          type=str, default='random',
                        help='Opponent (path or "random")')
    parser.add_argument('--sample-every', type=int, default=3,
                        help='Collect a sample every N steps (3 = ~133 steps/game)')
    parser.add_argument('--replay-dir',  type=str, default='',
                        help='If set, extract data from replay JSONs instead of live games')
    parser.add_argument('--max-replays', type=int, default=0,
                        help='Max replays to parse (0 = all)')
    args = parser.parse_args()

    all_samples: list[tuple] = []

    if args.replay_dir:
        print(f"Extracting data from replays in {args.replay_dir}...")
        all_samples = collect_replay_data(
            args.replay_dir, args.sample_every, args.max_replays)
    else:
        print("Loading agents...")
        agent_a = load_agent(args.agent, 'agent_a')
        agent_b = 'random' if args.vs == 'random' else load_agent(args.vs, 'agent_b')

        print(f"Generating training data ({args.games} games, sample every {args.sample_every} steps)...")
        seeds = [random.randint(0, 2**31) for _ in range(args.games)]

        for i, seed in enumerate(seeds):
            try:
                s = collect_game_data(agent_a, agent_b, seed, args.sample_every,
                                      collect_both_sides=True)
                all_samples.extend(s)
            except Exception as e:
                print(f"  game {i} seed={seed} failed: {e}")
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{args.games} games done, {len(all_samples)} samples")

    if not all_samples:
        print("No samples collected.")
        sys.exit(1)

    X = np.array([s[0] for s in all_samples], dtype=np.float32)
    y = np.array([s[1] for s in all_samples], dtype=np.float32)

    print(f"\nDataset: {len(X)} samples")
    print(f"  improvement: mean={y.mean():.1f}  std={y.std():.1f}  "
          f"min={y.min():.1f}  max={y.max():.1f}")
    print(f"  positive (good actions): {(y > 0).mean():.1%}")

    # Normalise features AND targets
    means  = X.mean(axis=0)
    stds   = X.std(axis=0) + 1e-8
    X_n    = (X - means) / stds
    y_mean = y.mean()
    y_std  = y.std() + 1e-8
    y_n    = (y - y_mean) / y_std

    print(f"\nTraining MLP ({FEATURE_DIM}→64→32→1, regression, {args.epochs} epochs)...")
    mlp = MLP()
    mlp.fit(X_n, y_n, epochs=args.epochs)

    model_dict = mlp.export_numpy()
    model_dict['means']  = means
    model_dict['stds']   = stds
    model_dict['y_mean'] = float(y_mean)
    model_dict['y_std']  = float(y_std)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"\nSaved → {args.out}  ({os.path.getsize(args.out)//1024}KB)")

    # Quick sanity check: correlation on held-out 10%
    n_val = max(100, len(X_n) // 10)
    idx   = np.random.choice(len(X_n), n_val, replace=False)
    preds = np.array([numpy_predict(model_dict, X_n[j]) for j in idx])
    corr  = float(np.corrcoef(preds, y_n[idx])[0, 1])
    print(f"Validation correlation (n={n_val}): r={corr:.3f}  "
          f"(>0.3 = useful, >0.6 = good)")


if __name__ == '__main__':
    main()
