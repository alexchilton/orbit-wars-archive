#!/usr/bin/env python3
"""
4P Pool Tournament — tests all 4p candidate variants against a diverse opponent pool.

Pool bots (weak/diverse opponents):
  pool_bully      — BullyBot (Python port): strongest → weakest, 1 fleet at a time
  pool_prospector — ProspectorBot: best prod-ROI target, 1 fleet at a time
  pool_rage       — RageBot: every planet attacks nearest enemy relentlessly
  pool_dual       — DualBot: mode-switching based on ships/prod advantage
  pool_baseline   — Orbit Wars baseline agent (strong rule-based)

Main competitors (our agents):
  shunlite        — main_fc_rl_shunlite (best 2p, used as 4p opponent for comparison)
  plus4p          — main_v131_plus_4p (current best 4p submission)
  political       — main_v131_plus_4p_political (4p political targeting variant)

Tournament format:
  - Multiple lineups mixing our candidates with diverse pool opponents
  - 16 games per lineup (rotating seats)
  - Primary metric: 1st place rate (winner-take-all scoring)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

os.environ.setdefault("KAGGLE_ENVIRONMENTS_QUIET", "1")

ROOT = Path(__file__).resolve().parent
SUBMISSION = ROOT / "submission"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments import make  # noqa: E402


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_agent(path: Path, label: str) -> Callable[..., Any]:
    spec = importlib.util.spec_from_file_location(f"_pool_{label}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.agent


def run_4p_game(agent_list: List[Callable]) -> Dict[str, Any]:
    env = make("orbit_wars", debug=False)
    env.run(agent_list)
    final = env.steps[-1]
    rewards = [player.reward or 0 for player in final]
    planets = final[-1].observation.get("planets") or []
    ships = [sum(p[5] for p in planets if p[1] == idx) for idx in range(len(agent_list))]
    ranks = [1 + sum(1 for other in rewards if other > reward) for reward in rewards]
    return {"rewards": rewards, "ships": ships, "ranks": ranks}


def main() -> None:
    parser = argparse.ArgumentParser(description="4P Pool Tournament")
    parser.add_argument("--games", type=int, default=16, help="Games per lineup")
    parser.add_argument("--log-jsonl", default="", help="Optional JSONL log path")
    args = parser.parse_args()

    log_dir = ROOT / "benchmark_logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_path = Path(args.log_jsonl) if args.log_jsonl else log_dir / f"4p-pool-{stamp}.jsonl"

    print("Loading agents...", flush=True)

    agent_paths = {
        # Pool opponents (diverse / weaker)
        "bully":      SUBMISSION / "pool_bully.py",
        "prospector": SUBMISSION / "pool_prospector.py",
        "rage":       SUBMISSION / "pool_rage.py",
        "dual":       SUBMISSION / "pool_dual.py",
        "baseline":   SUBMISSION / "pool_baseline.py",
        # Our agents
        "shunlite":   SUBMISSION / "main_fc_rl_shunlite.py",
        "plus4p":     SUBMISSION / "main_v131_plus_4p.py",
        "political":  SUBMISSION / "main_v131_plus_4p_political.py",
    }

    # Also load shunlite-based mechanic variants if they exist
    shunlite_variants = {
        "sh_wave":    SUBMISSION / "main_shunlite_wave.py",
        "sh_intercept": SUBMISSION / "main_shunlite_intercept.py",
        "sh_evac":    SUBMISSION / "main_shunlite_evac.py",
        "sh_denial":  SUBMISSION / "main_shunlite_denial.py",
        "sh_prodexp": SUBMISSION / "main_shunlite_prodexpand.py",
        "v131_wave":  SUBMISSION / "main_v131_plus_wave.py",
        "v131_intercept": SUBMISSION / "main_v131_plus_intercept.py",
        "v131_evac":  SUBMISSION / "main_v131_plus_evac.py",
        "v131_denial": SUBMISSION / "main_v131_plus_denial.py",
        "v131_prodexp": SUBMISSION / "main_v131_plus_prodexpand.py",
    }

    agents = {}
    for label, path in agent_paths.items():
        if path.exists():
            try:
                agents[label] = load_agent(path, label)
                print(f"  ✓ {label}", flush=True)
            except Exception as e:
                print(f"  ✗ {label}: {e}", flush=True)
        else:
            print(f"  - {label}: not found ({path.name})", flush=True)

    for label, path in shunlite_variants.items():
        if path.exists():
            try:
                agents[label] = load_agent(path, label)
                print(f"  ✓ {label}", flush=True)
            except Exception as e:
                print(f"  ✗ {label}: {e}", flush=True)

    # Define lineups — each is (name, [4 agent labels])
    # Primary: our 4p candidates vs pool
    lineups = []

    # Lineup group 1: plus4p vs diverse pool
    if all(k in agents for k in ["plus4p", "bully", "prospector", "rage"]):
        lineups.append(("plus4p_vs_pool_ABC", ["plus4p", "bully", "prospector", "rage"]))
    if all(k in agents for k in ["plus4p", "dual", "baseline", "shunlite"]):
        lineups.append(("plus4p_vs_pool_DEF", ["plus4p", "dual", "baseline", "shunlite"]))

    # Lineup group 2: political vs diverse pool
    if all(k in agents for k in ["political", "bully", "prospector", "rage"]):
        lineups.append(("political_vs_pool_ABC", ["political", "bully", "prospector", "rage"]))
    if all(k in agents for k in ["political", "dual", "baseline", "shunlite"]):
        lineups.append(("political_vs_pool_DEF", ["political", "dual", "baseline", "shunlite"]))

    # Lineup group 3: plus4p vs political head-to-head in pool context
    if all(k in agents for k in ["plus4p", "political", "baseline", "shunlite"]):
        lineups.append(("plus4p_vs_political_strong_pool", ["plus4p", "political", "baseline", "shunlite"]))
    if all(k in agents for k in ["plus4p", "political", "dual", "rage"]):
        lineups.append(("plus4p_vs_political_weak_pool", ["plus4p", "political", "dual", "rage"]))

    # Lineup group 4: shunlite mechanic variants vs pool (if built)
    for variant in ["sh_wave", "sh_denial", "sh_evac", "sh_intercept", "sh_prodexp"]:
        if all(k in agents for k in [variant, "plus4p", "shunlite", "baseline"]):
            lineups.append((f"{variant}_vs_strong", [variant, "plus4p", "shunlite", "baseline"]))
        if all(k in agents for k in [variant, "dual", "prospector", "rage"]):
            lineups.append((f"{variant}_vs_weak", [variant, "dual", "prospector", "rage"]))

    # Lineup group 5: v131_plus mechanic variants vs pool (if built)
    for variant in ["v131_wave", "v131_denial", "v131_evac", "v131_intercept", "v131_prodexp"]:
        if all(k in agents for k in [variant, "plus4p", "shunlite", "baseline"]):
            lineups.append((f"{variant}_vs_strong", [variant, "plus4p", "shunlite", "baseline"]))
        if all(k in agents for k in [variant, "dual", "prospector", "rage"]):
            lineups.append((f"{variant}_vs_weak", [variant, "dual", "prospector", "rage"]))

    if not lineups:
        print("ERROR: no valid lineups could be formed. Check that agents loaded correctly.")
        return

    print(f"\nRunning {len(lineups)} lineups × {args.games} games = {len(lineups) * args.games} total games", flush=True)

    # Track stats per agent across all lineups
    global_stats: Dict[str, Dict] = defaultdict(lambda: {
        "games": 0, "wins": 0, "top2": 0, "reward_sum": 0.0, "rank_sum": 0.0
    })
    lineup_results: Dict[str, Dict] = {}

    with log_path.open("a", encoding="utf-8") as handle:
        for lineup_name, labels in lineups:
            local_stats: Dict[str, Dict] = defaultdict(lambda: {
                "games": 0, "wins": 0, "top2": 0, "reward_sum": 0.0, "rank_sum": 0.0
            })
            print(f"\n[4P] {lineup_name} ({' | '.join(labels)})", flush=True)

            for game in range(args.games):
                # Rotate seats each game
                order = [labels[(i + game) % len(labels)] for i in range(len(labels))]
                result = run_4p_game([agents[lbl] for lbl in order])

                ranked = sorted(
                    zip(order, result["rewards"], result["ranks"]),
                    key=lambda item: item[2]
                )
                winner = ranked[0][0]

                for seat, lbl in enumerate(order):
                    reward = result["rewards"][seat]
                    rank = result["ranks"][seat]
                    for stats in [local_stats[lbl], global_stats[lbl]]:
                        stats["games"] += 1
                        stats["reward_sum"] += reward
                        stats["rank_sum"] += rank
                        if rank <= 2:
                            stats["top2"] += 1
                        if rank == 1:
                            stats["wins"] += 1

                payload = {
                    "ts": now_iso(), "lineup": lineup_name, "game": game + 1,
                    "order": order, "rewards": result["rewards"],
                    "ranks": result["ranks"], "ships": result["ships"],
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
                handle.flush()

                ranked_str = " > ".join(
                    f"{lbl}(r{rank})" for lbl, _, rank in ranked
                )
                print(f"  Game {game+1:02d}: {ranked_str}  winner={winner}", flush=True)

            # Per-lineup summary
            lineup_results[lineup_name] = {}
            for lbl, stats in local_stats.items():
                g = max(stats["games"], 1)
                lineup_results[lineup_name][lbl] = {
                    "games": stats["games"],
                    "1st_rate": round(stats["wins"] / g * 100, 1),
                    "top2_rate": round(stats["top2"] / g * 100, 1),
                    "avg_rank": round(stats["rank_sum"] / g, 2),
                }
            print(f"  Lineup summary:", flush=True)
            for lbl, r in sorted(lineup_results[lineup_name].items(), key=lambda x: -x[1]["1st_rate"]):
                print(f"    {lbl:20s}  1st={r['1st_rate']:5.1f}%  top2={r['top2_rate']:5.1f}%  avg_rank={r['avg_rank']:.2f}", flush=True)

    # Global summary
    print("\n" + "=" * 70, flush=True)
    print("GLOBAL SUMMARY (across all lineups)", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Agent':<22} {'Games':>5} {'1st%':>7} {'Top2%':>7} {'AvgRank':>8}", flush=True)
    print("-" * 55, flush=True)

    sorted_agents = sorted(
        global_stats.items(),
        key=lambda x: x[1]["wins"] / max(x[1]["games"], 1),
        reverse=True
    )
    for lbl, stats in sorted_agents:
        g = max(stats["games"], 1)
        print(
            f"{lbl:<22} {stats['games']:>5} "
            f"{stats['wins']/g*100:>7.1f}% "
            f"{stats['top2']/g*100:>7.1f}% "
            f"{stats['rank_sum']/g:>8.2f}",
            flush=True
        )

    print(f"\nLog: {log_path}", flush=True)


if __name__ == "__main__":
    main()
