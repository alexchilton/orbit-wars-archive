"""Diagnostic: run SB3 agent against tier 7+ opponents, capture step-by-step data."""

import os
import sys
import math

os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from ppo_gnn.sb3_env import OrbitWarsEnv, NOOP_ACTION, NUM_FRACTIONS, MAX_ACTIONS
from ppo_gnn.train_ppo_edge import load_agent_from_file
import numpy as np


def run_diagnostic_game(opponent_name, opponent_fn, verbose=True):
    """Run one game, return step-by-step diagnostics."""
    pool = [(99, opponent_name, opponent_fn)]  # tier 99 so always eligible
    env = OrbitWarsEnv(opponent_pool=pool, mode="2p", max_tier=99)
    obs, info = env.reset()

    history = []
    step = 0
    while True:
        mask = env.action_masks()
        # Pick best valid actions greedily (highest-index valid = highest-scored candidate)
        # For diagnosis, just use all-noop to see pure heuristic floor
        # Actually let's use a simple policy: pick first valid candidate with fraction 5 (50%)
        action = np.array([NOOP_ACTION] * MAX_ACTIONS)
        valid_count = info.get("num_valid", env._num_valid)

        # Simple greedy: send top candidates
        slots_used = 0
        for c in range(min(valid_count, MAX_ACTIONS)):
            frac_idx = 5  # ~50% of available ships
            act_idx = c * NUM_FRACTIONS + frac_idx
            if mask[act_idx]:
                action[slots_used] = act_idx
                slots_used += 1
                if slots_used >= MAX_ACTIONS:
                    break

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        record = {
            "step": step,
            "my_planets": info.get("my_planets", 0),
            "actions_sent": info.get("actions_sent", 0),
            "reward": reward,
        }
        history.append(record)

        if terminated or truncated:
            record["won"] = info.get("won", False)
            record["final"] = True
            break

    return history


def analyze_game(history, opponent_name):
    """Print analysis of a game."""
    final = history[-1]
    result = "WIN" if final.get("won") else "LOSS"
    total_steps = final["step"]

    # Key milestones
    planets_at = {}
    for h in history:
        s = h["step"]
        if s in (10, 25, 50, 100, 150, 200):
            planets_at[s] = h["my_planets"]

    total_actions = sum(h["actions_sent"] for h in history)
    noop_steps = sum(1 for h in history if h["actions_sent"] == 0)
    noop_pct = 100 * noop_steps / max(total_steps, 1)

    # Planet trajectory
    peak_planets = max(h["my_planets"] for h in history)
    # When did we start losing planets?
    peak_step = next(h["step"] for h in history if h["my_planets"] == peak_planets)

    # Economy phases
    expanding = next((h["step"] for h in history if h["my_planets"] >= 5), None)
    contracting = None
    if peak_planets > 3:
        for h in history[peak_step:]:
            if h["my_planets"] < peak_planets - 2:
                contracting = h["step"]
                break

    print(f"\n{'='*60}")
    print(f"vs {opponent_name}: {result} in {total_steps} steps")
    print(f"{'='*60}")
    print(f"  Peak planets: {peak_planets} at step {peak_step}")
    print(f"  Milestones: {planets_at}")
    print(f"  Total fleets sent: {total_actions}, noop steps: {noop_pct:.0f}%")
    if expanding:
        print(f"  Reached 5 planets at step {expanding}")
    if contracting:
        print(f"  Started losing ground at step {contracting}")

    # Last 50 steps planet trend
    if total_steps > 50:
        last50 = [h["my_planets"] for h in history[-50:]]
        print(f"  Last 50 steps planets: {last50[0]} -> {last50[-1]} (trend: {last50[-1] - last50[0]:+d})")

    # Action density in phases
    if total_steps > 100:
        early_acts = sum(h["actions_sent"] for h in history[:50])
        mid_acts = sum(h["actions_sent"] for h in history[50:150]) if total_steps > 150 else 0
        late_acts = sum(h["actions_sent"] for h in history[150:]) if total_steps > 150 else 0
        print(f"  Fleets by phase: early(0-50)={early_acts}, mid(50-150)={mid_acts}, late(150+)={late_acts}")

    return result


def main():
    root = Path(__file__).parent.parent

    opponents = []
    candidates = [
        ("baseline", "submission/pool_baseline.py"),
        ("v131_2p", "submission/pool_v131_2p.py"),
        ("ykhnkf_dist", "submission/ext/pool_ykhnkf_dist.py"),
        ("shunlite", "submission/ext/pool_shunlite.py"),
        # Also run against one we beat for comparison
        ("bully", "submission/pool_bully.py"),
    ]

    for name, rel_path in candidates:
        full = root / rel_path
        if full.exists():
            try:
                opponents.append((name, load_agent_from_file(str(full))))
            except Exception as e:
                print(f"Skip {name}: {e}")

    if not opponents:
        print("No opponents loaded!")
        return

    print(f"Running diagnostics against {len(opponents)} opponents...")
    results = {}
    for name, fn in opponents:
        games = []
        for game_num in range(3):
            history = run_diagnostic_game(name, fn)
            result = analyze_game(history, f"{name} (game {game_num+1})")
            games.append(result)
        results[name] = games
        wins = sum(1 for g in games if g == "WIN")
        print(f"\n  >>> {name}: {wins}/3 wins")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, games in results.items():
        wins = sum(1 for g in games if g == "WIN")
        print(f"  {name}: {wins}/3")


if __name__ == "__main__":
    main()
