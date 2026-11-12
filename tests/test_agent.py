"""
Local testing harness for Orbit Wars agents.

Usage:
    # 20 games: our agent vs baseline
    python test_agent.py --games 20

    # 20 games: our agent vs random
    python test_agent.py --games 20 --opponent random

    # Compare two saved versions head-to-head
    python test_agent.py --games 20 --agent snapshots/v3.py --opponent snapshots/v5.py

    # Also play both sides (swap player order) to remove position bias
    python test_agent.py --games 20 --swap
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Callable, Optional

# Suppress kaggle_environments startup noise
import os
os.environ.setdefault("KAGGLE_ENVIRONMENTS_QUIET", "1")

from kaggle_environments import make  # noqa: E402


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

def load_agent_from_file(path: str) -> Callable:
    """Load an agent(obs, config) callable from a .py file."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Agent file not found: {p}")
    module_name = f"_agent_{p.stem}_{id(p)}"
    spec = importlib.util.spec_from_file_location(module_name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod  # required for dataclasses to resolve their own module
    spec.loader.exec_module(mod)
    if not hasattr(mod, "agent"):
        raise AttributeError(f"No top-level agent() function found in {p}")
    return mod.agent


def load_baseline_agent() -> Callable:
    """Load the baseline agent, stripping the %%writefile magic line."""
    baseline_path = Path(__file__).parent / "baseline_agent.py"
    if not baseline_path.exists():
        raise FileNotFoundError("baseline_agent.py not found — run the notebook extraction first")
    src = baseline_path.read_text(encoding="utf-8")
    # Strip Jupyter magic if present
    lines = src.splitlines()
    if lines and lines[0].startswith("%%"):
        src = "\n".join(lines[1:])
    import types
    mod = types.ModuleType("_baseline_module")
    exec(compile(src, str(baseline_path), "exec"), mod.__dict__)
    if not hasattr(mod, "agent"):
        raise AttributeError("No agent() function found in baseline_agent.py")
    return mod.agent


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def run_game(agent_a: Callable, agent_b: Callable, seed: Optional[int] = None) -> dict:
    """Run one game and return result dict."""
    env = make("orbit_wars", debug=False)
    env.run([agent_a, agent_b])
    final = env.steps[-1]
    reward_a = final[0].reward or 0
    reward_b = final[1].reward or 0
    # planets are raw lists: [id, owner, x, y, radius, ships, production]
    planets = env.steps[-1][0].observation.get("planets") or []
    ships_a = sum(p[5] for p in planets if p[1] == 0)
    ships_b = sum(p[5] for p in planets if p[1] == 1)
    return {
        "reward_a": reward_a,
        "reward_b": reward_b,
        "ships_a": ships_a,
        "ships_b": ships_b,
        "winner": "A" if reward_a > reward_b else ("B" if reward_b > reward_a else "draw"),
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_results(results: list, agent_a_label: str, agent_b_label: str, elapsed: float) -> None:
    wins_a = sum(1 for r in results if r["winner"] == "A")
    wins_b = sum(1 for r in results if r["winner"] == "B")
    draws   = sum(1 for r in results if r["winner"] == "draw")
    n = len(results)

    avg_ships_a = sum(r["ships_a"] for r in results) / max(1, n)
    avg_ships_b = sum(r["ships_b"] for r in results) / max(1, n)

    print()
    print("=" * 56)
    print(f"  Results after {n} games  ({elapsed:.1f}s total, {elapsed/n:.1f}s/game)")
    print("=" * 56)
    print(f"  {agent_a_label:<24}  wins: {wins_a:>3}  ({100*wins_a/n:.0f}%)")
    print(f"  {agent_b_label:<24}  wins: {wins_b:>3}  ({100*wins_b/n:.0f}%)")
    print(f"  {'draws':<24}       {draws:>3}  ({100*draws/n:.0f}%)")
    print("-" * 56)
    print(f"  Avg ships A: {avg_ships_a:>6.0f}   Avg ships B: {avg_ships_b:>6.0f}")
    print("=" * 56)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Orbit Wars local test harness")
    parser.add_argument("--agent",    default="orbit_wars_agent.py",
                        help="Path to agent A .py file (default: orbit_wars_agent.py)")
    parser.add_argument("--opponent", default="baseline",
                        help="'baseline', 'random', or path to a .py file (default: baseline)")
    parser.add_argument("--games",    type=int, default=20,
                        help="Number of games to play (default: 20)")
    parser.add_argument("--swap",     action="store_true",
                        help="Also play with player order swapped (doubles game count)")
    args = parser.parse_args()

    # Load agents
    cwd = Path(__file__).parent
    agent_a = load_agent_from_file(str(cwd / args.agent))
    agent_a_label = Path(args.agent).stem

    if args.opponent == "baseline":
        agent_b = load_baseline_agent()
        agent_b_label = "baseline"
    elif args.opponent == "random":
        agent_b = "random"
        agent_b_label = "random"
    else:
        agent_b = load_agent_from_file(str(cwd / args.opponent))
        agent_b_label = Path(args.opponent).stem

    print(f"\nRunning {args.games} games: {agent_a_label} vs {agent_b_label}")
    if args.swap:
        print(f"  + {args.games} games swapped (total {args.games * 2})")
    print()

    results = []
    t0 = time.time()

    for i in range(args.games):
        r = run_game(agent_a, agent_b)
        results.append(r)
        marker = "✓" if r["winner"] == "A" else ("✗" if r["winner"] == "B" else "=")
        print(f"  Game {i+1:>3}/{args.games}  {marker}  "
              f"ships {r['ships_a']:>5} vs {r['ships_b']:>5}", flush=True)

    if args.swap:
        print(f"\n  --- Swapped order ---\n")
        swapped = []
        for i in range(args.games):
            r = run_game(agent_b, agent_a)
            # Flip perspective so A is always our agent
            flipped = {
                "reward_a": r["reward_b"],
                "reward_b": r["reward_a"],
                "ships_a":  r["ships_b"],
                "ships_b":  r["ships_a"],
                "winner":   "A" if r["winner"] == "B" else ("B" if r["winner"] == "A" else "draw"),
            }
            swapped.append(flipped)
            marker = "✓" if flipped["winner"] == "A" else ("✗" if flipped["winner"] == "B" else "=")
            print(f"  Game {i+1:>3}/{args.games}  {marker}  "
                  f"ships {flipped['ships_a']:>5} vs {flipped['ships_b']:>5}", flush=True)
        results.extend(swapped)

    print_results(results, agent_a_label, agent_b_label, time.time() - t0)


if __name__ == "__main__":
    main()
