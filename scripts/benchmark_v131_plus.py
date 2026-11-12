#!/usr/bin/env python3
"""Benchmark split v131-plus variants in 2-player and 4-player Orbit Wars."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments import make  # noqa: E402


V131_PATH = Path("/Users/alexchilton/Downloads/main_v131.py")
SHUNLITE_PATH = ROOT / "submission" / "main_fc_rl_shunlite.py"
PLUS_2P_PATH = ROOT / "submission" / "main_v131_plus_2p.py"
PLUS_4P_PATH = ROOT / "submission" / "main_v131_plus_4p.py"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_agent(path: Path, label: str) -> Callable[..., Any]:
    spec = importlib.util.spec_from_file_location(f"_v131_plus_{label}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.agent


def run_2p_game(agent_a: Callable[..., Any], agent_b: Callable[..., Any]) -> Dict[str, Any]:
    env = make("orbit_wars", debug=False)
    env.run([agent_a, agent_b])
    final = env.steps[-1]
    reward_a = final[0].reward or 0
    reward_b = final[1].reward or 0
    planets = final[-1].observation.get("planets") or []
    ships_a = sum(p[5] for p in planets if p[1] == 0)
    ships_b = sum(p[5] for p in planets if p[1] == 1)
    return {
        "reward_a": reward_a,
        "reward_b": reward_b,
        "ships_a": ships_a,
        "ships_b": ships_b,
    }


def run_4p_game(agent_list: List[Callable[..., Any]]) -> Dict[str, Any]:
    env = make("orbit_wars", debug=False)
    env.run(agent_list)
    final = env.steps[-1]
    rewards = [player.reward or 0 for player in final]
    planets = final[-1].observation.get("planets") or []
    ships = [sum(p[5] for p in planets if p[1] == idx) for idx in range(len(agent_list))]
    ranks = [1 + sum(1 for other in rewards if other > reward) for reward in rewards]
    return {"rewards": rewards, "ships": ships, "ranks": ranks}


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported type for JSON serialization: {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark split v131-plus variants")
    parser.add_argument("--games-2p", type=int, default=24, help="Games per 2p matchup, across both seats")
    parser.add_argument("--games-4p", type=int, default=12, help="Games per 4p lineup")
    parser.add_argument("--log-jsonl", default="", help="Optional per-game JSONL log path")
    args = parser.parse_args()

    log_dir = ROOT / "benchmark_logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_path = Path(args.log_jsonl) if args.log_jsonl else log_dir / f"v131-plus-{stamp}.jsonl"

    print("Loading agents...", flush=True)
    agents = {
        "v131": load_agent(V131_PATH, "v131"),
        "shunlite": load_agent(SHUNLITE_PATH, "shunlite"),
        "shunlite_b": load_agent(SHUNLITE_PATH, "shunlite_b"),
        "shunlite_c": load_agent(SHUNLITE_PATH, "shunlite_c"),
        "plus2p": load_agent(PLUS_2P_PATH, "plus2p"),
        "plus4p": load_agent(PLUS_4P_PATH, "plus4p"),
    }

    two_player_matchups = [
        ("plus2p_vs_v131", "plus2p", "v131"),
        ("plus2p_vs_shunlite", "plus2p", "shunlite"),
        ("v131_vs_shunlite", "v131", "shunlite"),
    ]
    four_player_lineups = [
        ("plus4p_head_to_head", ["plus4p", "v131", "shunlite", "shunlite_b"]),
        ("plus4p_vs_triple_shunlite", ["plus4p", "shunlite", "shunlite_b", "shunlite_c"]),
        ("v131_vs_triple_shunlite", ["v131", "shunlite", "shunlite_b", "shunlite_c"]),
    ]

    summary: Dict[str, Any] = {
        "started_at": now_iso(),
        "log_path": log_path,
        "two_player": {},
        "four_player": {},
    }

    with log_path.open("a", encoding="utf-8") as handle:
        for matchup_name, left_label, right_label in two_player_matchups:
            stats = {"games": 0, "wins": 0, "losses": 0, "draws": 0, "reward_diff_sum": 0.0, "ship_diff_sum": 0.0}
            print(f"\n[2P] {matchup_name}", flush=True)
            for game in range(args.games_2p):
                seat = game % 2
                if seat == 0:
                    result = run_2p_game(agents[left_label], agents[right_label])
                    reward_diff = result["reward_a"] - result["reward_b"]
                    ship_diff = result["ships_a"] - result["ships_b"]
                else:
                    result = run_2p_game(agents[right_label], agents[left_label])
                    reward_diff = result["reward_b"] - result["reward_a"]
                    ship_diff = result["ships_b"] - result["ships_a"]

                stats["games"] += 1
                stats["reward_diff_sum"] += reward_diff
                stats["ship_diff_sum"] += ship_diff
                if reward_diff > 0:
                    stats["wins"] += 1
                    outcome = "WIN"
                elif reward_diff < 0:
                    stats["losses"] += 1
                    outcome = "LOSS"
                else:
                    stats["draws"] += 1
                    outcome = "DRAW"

                payload = {
                    "ts": now_iso(),
                    "mode": "2p",
                    "matchup": matchup_name,
                    "game": game + 1,
                    "seat": seat,
                    "variant": left_label,
                    "opponent": right_label,
                    "outcome": outcome,
                    "reward_diff": reward_diff,
                    "ship_diff": ship_diff,
                }
                handle.write(json.dumps(payload, default=json_default, sort_keys=True) + "\n")
                handle.flush()
                score_rate = (stats["wins"] + 0.5 * stats["draws"]) / stats["games"]
                print(
                    f"  Game {game + 1:02d}/{args.games_2p}: {outcome:4s}  "
                    f"reward_diff={reward_diff:+.0f}  ship_diff={ship_diff:+.0f}  "
                    f"score_rate={100.0 * score_rate:4.1f}%",
                    flush=True,
                )

            summary["two_player"][matchup_name] = {
                **stats,
                "score_rate": (stats["wins"] + 0.5 * stats["draws"]) / max(1, stats["games"]),
                "avg_reward_diff": stats["reward_diff_sum"] / max(1, stats["games"]),
                "avg_ship_diff": stats["ship_diff_sum"] / max(1, stats["games"]),
            }

        for lineup_name, labels in four_player_lineups:
            stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"games": 0, "wins": 0, "top2": 0, "reward_sum": 0.0, "rank_sum": 0.0})
            print(f"\n[4P] {lineup_name}", flush=True)
            for game in range(args.games_4p):
                order = [labels[(idx + game) % len(labels)] for idx in range(len(labels))]
                result = run_4p_game([agents[label] for label in order])
                ranked = sorted(zip(order, result["rewards"], result["ranks"]), key=lambda item: (item[2], -item[1]))
                winner_label = ranked[0][0]

                for seat, label in enumerate(order):
                    reward = result["rewards"][seat]
                    rank = result["ranks"][seat]
                    stats[label]["games"] += 1
                    stats[label]["reward_sum"] += reward
                    stats[label]["rank_sum"] += rank
                    if rank <= 2:
                        stats[label]["top2"] += 1
                    if rank == 1:
                        stats[label]["wins"] += 1

                payload = {
                    "ts": now_iso(),
                    "mode": "4p",
                    "lineup": lineup_name,
                    "game": game + 1,
                    "order": order,
                    "rewards": result["rewards"],
                    "ranks": result["ranks"],
                    "ships": result["ships"],
                }
                handle.write(json.dumps(payload, default=json_default, sort_keys=True) + "\n")
                handle.flush()
                ranked_str = " > ".join(f"{label}(r={reward:.0f},p={rank})" for label, reward, rank in ranked)
                print(f"  Game {game + 1:02d}/{args.games_4p}: {ranked_str}  winner={winner_label}", flush=True)

            summary["four_player"][lineup_name] = {
                label: {
                    **bucket,
                    "first_rate": bucket["wins"] / max(1, bucket["games"]),
                    "top2_rate": bucket["top2"] / max(1, bucket["games"]),
                    "avg_reward": bucket["reward_sum"] / max(1, bucket["games"]),
                    "avg_rank": bucket["rank_sum"] / max(1, bucket["games"]),
                }
                for label, bucket in stats.items()
            }

    summary["finished_at"] = now_iso()
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, default=json_default, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
