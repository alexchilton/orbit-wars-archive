"""Run an overnight Orbit Wars benchmark sweep and log rolling results."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

os.environ.setdefault("KAGGLE_ENVIRONMENTS_QUIET", "1")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import test_agent as harness  # noqa: E402
from kaggle_environments import make  # noqa: E402


def now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def agent_label(path: str) -> str:
    """Return a short label for an agent path or built-in name."""
    return Path(path).stem if path.endswith(".py") else path


def run_2p(agent_a: Callable[..., Any], agent_b: Callable[..., Any]) -> Dict[str, Any]:
    """Run one 2-player game and return rewards, ships, and winner."""
    env = make("orbit_wars", debug=False)
    env.run([agent_a, agent_b])
    final = env.steps[-1]
    reward_a = final[0].reward or 0
    reward_b = final[1].reward or 0
    planets = final[-1].observation.get("planets") or []
    ships_a = sum(p[5] for p in planets if p[1] == 0)
    ships_b = sum(p[5] for p in planets if p[1] == 1)
    if reward_a > reward_b:
        winner = "A"
    elif reward_b > reward_a:
        winner = "B"
    else:
        winner = "draw"
    return {
        "reward_a": reward_a,
        "reward_b": reward_b,
        "ships_a": ships_a,
        "ships_b": ships_b,
        "winner": winner,
    }


def run_4p(agent_list: List[Callable[..., Any] | str]) -> Dict[str, Any]:
    """Run one 4-player game and return rewards, ships, and ranks."""
    env = make("orbit_wars", debug=False)
    env.run(agent_list)
    final = env.steps[-1]
    rewards = [player.reward or 0 for player in final]
    planets = final[-1].observation.get("planets") or []
    ships = [sum(p[5] for p in planets if p[1] == idx) for idx in range(len(agent_list))]
    ranks = [1 + sum(1 for other in rewards if other > reward) for reward in rewards]
    return {
        "rewards": rewards,
        "ships": ships,
        "ranks": ranks,
    }


def new_2p_stats() -> Dict[str, float]:
    """Create an empty aggregate bucket for 2-player results."""
    return {
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "reward_diff_sum": 0.0,
        "ship_diff_sum": 0.0,
    }


def new_4p_stats() -> Dict[str, float]:
    """Create an empty aggregate bucket for 4-player results."""
    return {
        "games": 0,
        "reward_sum": 0.0,
        "rank_sum": 0.0,
        "top2": 0,
        "ship_sum": 0.0,
    }


def summarize_2p(stats: Dict[str, float]) -> Dict[str, float]:
    """Convert raw 2-player counters into readable metrics."""
    games = max(1, int(stats["games"]))
    summary = deepcopy(stats)
    summary["win_rate"] = stats["wins"] / games
    summary["score_rate"] = (stats["wins"] + 0.5 * stats["draws"]) / games
    summary["avg_reward_diff"] = stats["reward_diff_sum"] / games
    summary["avg_ship_diff"] = stats["ship_diff_sum"] / games
    return summary


def summarize_4p(stats: Dict[str, float]) -> Dict[str, float]:
    """Convert raw 4-player counters into readable metrics."""
    games = max(1, int(stats["games"]))
    summary = deepcopy(stats)
    summary["avg_reward"] = stats["reward_sum"] / games
    summary["avg_rank"] = stats["rank_sum"] / games
    summary["top2_rate"] = stats["top2"] / games
    summary["avg_ships"] = stats["ship_sum"] / games
    return summary


def build_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build the current summary payload from the rolling aggregates."""
    summary = {
        "meta": deepcopy(state["meta"]),
        "two_player": {},
        "four_player": {},
    }

    for variant, opponents in state["two_player"].items():
        summary["two_player"][variant] = {
            opponent: summarize_2p(stats) for opponent, stats in opponents.items()
        }

    for variant, stats in state["four_player"].items():
        summary["four_player"][variant] = summarize_4p(stats)

    ranking = []
    for variant, opponents in summary["two_player"].items():
        if "v5" not in opponents or "baseline" not in opponents:
            continue
        composite = 0.7 * opponents["v5"]["score_rate"] + 0.3 * opponents["baseline"]["score_rate"]
        tiebreak = opponents["v5"]["avg_ship_diff"] + opponents["baseline"]["avg_ship_diff"]
        ranking.append((composite, tiebreak, variant))
    ranking.sort(reverse=True)
    summary["meta"]["two_player_ranking"] = [variant for _, _, variant in ranking]

    four_player_ranking = sorted(
        (
            (stats["avg_rank"], -stats["top2_rate"], -stats["avg_reward"], variant)
            for variant, stats in summary["four_player"].items()
        )
    )
    summary["meta"]["four_player_ranking"] = [variant for _, _, _, variant in four_player_ranking]
    return summary


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """Append a JSON object as a single line."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    """Parse arguments, run benchmark cycles, and persist rolling results."""
    parser = argparse.ArgumentParser(description="Run an overnight Orbit Wars benchmark sweep")
    parser.add_argument("--hours", type=float, default=10.0, help="How long to keep benchmarking")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[f"snapshots/v{i}.py" for i in range(6, 16)],
        help="Variant agent paths to benchmark",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=["snapshots/v5.py", "baseline"],
        help="Two-player benchmark opponents",
    )
    parser.add_argument(
        "--include-four-player",
        action="store_true",
        help="Also run 4-player seat-rotation games versus v5, baseline, and random",
    )
    parser.add_argument(
        "--jsonl-log",
        default="",
        help="Path to append per-game JSONL records; default uses benchmark_logs/",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Path to rewrite rolling summary JSON; default uses benchmark_logs/",
    )
    args = parser.parse_args()

    log_dir = ROOT / "benchmark_logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    jsonl_path = Path(args.jsonl_log) if args.jsonl_log else log_dir / f"overnight-{timestamp}.jsonl"
    summary_path = Path(args.summary_json) if args.summary_json else log_dir / f"overnight-{timestamp}-summary.json"

    loaded: Dict[str, Callable[..., Any] | str] = {}
    for path in args.variants:
        loaded[path] = harness.load_agent_from_file(str(ROOT / path))
    for benchmark in args.benchmarks:
        if benchmark == "baseline":
            loaded[benchmark] = harness.load_baseline_agent()
        else:
            loaded[benchmark] = harness.load_agent_from_file(str(ROOT / benchmark))
    loaded["random"] = "random"

    state: Dict[str, Any] = {
        "meta": {
            "started_at": now_iso(),
            "deadline_at": datetime.fromtimestamp(time.time() + args.hours * 3600, tz=timezone.utc).isoformat(),
            "cycles_completed": 0,
            "jsonl_log": str(jsonl_path),
            "summary_json": str(summary_path),
            "variants": [agent_label(path) for path in args.variants],
            "benchmarks": [agent_label(path) for path in args.benchmarks],
            "include_four_player": args.include_four_player,
        },
        "two_player": {
            agent_label(path): {agent_label(benchmark): new_2p_stats() for benchmark in args.benchmarks}
            for path in args.variants
        },
        "four_player": {agent_label(path): new_4p_stats() for path in args.variants},
    }

    deadline = time.time() + args.hours * 3600
    summary_path.write_text(json.dumps(build_summary(state), indent=2), encoding="utf-8")

    cycle = 0
    while time.time() < deadline:
        cycle += 1
        for variant_path in args.variants:
            variant_name = agent_label(variant_path)
            variant_agent = loaded[variant_path]
            for benchmark_path in args.benchmarks:
                benchmark_name = agent_label(benchmark_path)
                benchmark_agent = loaded[benchmark_path]
                if time.time() >= deadline:
                    break

                for seat in (0, 1):
                    started = time.time()
                    if seat == 0:
                        result = run_2p(variant_agent, benchmark_agent)
                        reward_diff = result["reward_a"] - result["reward_b"]
                        ship_diff = result["ships_a"] - result["ships_b"]
                        winner = result["winner"]
                        variant_won = winner == "A"
                        variant_lost = winner == "B"
                    else:
                        result = run_2p(benchmark_agent, variant_agent)
                        reward_diff = result["reward_b"] - result["reward_a"]
                        ship_diff = result["ships_b"] - result["ships_a"]
                        if result["winner"] == "B":
                            variant_won = True
                            variant_lost = False
                            winner = variant_name
                        elif result["winner"] == "A":
                            variant_won = False
                            variant_lost = True
                            winner = benchmark_name
                        else:
                            variant_won = False
                            variant_lost = False
                            winner = "draw"

                    bucket = state["two_player"][variant_name][benchmark_name]
                    bucket["games"] += 1
                    bucket["wins"] += int(variant_won)
                    bucket["losses"] += int(variant_lost)
                    bucket["draws"] += int(not variant_won and not variant_lost)
                    bucket["reward_diff_sum"] += reward_diff
                    bucket["ship_diff_sum"] += ship_diff

                    event = {
                        "timestamp": now_iso(),
                        "cycle": cycle,
                        "mode": "2p",
                        "variant": variant_name,
                        "benchmark": benchmark_name,
                        "seat": seat,
                        "winner": winner,
                        "reward_diff": reward_diff,
                        "ship_diff": ship_diff,
                        "duration_sec": round(time.time() - started, 3),
                    }
                    append_jsonl(jsonl_path, event)
                    summary_path.write_text(json.dumps(build_summary(state), indent=2), encoding="utf-8")
                    print(
                        f"[2p][cycle {cycle}] {variant_name} vs {benchmark_name} seat={seat} "
                        f"winner={winner} reward_diff={reward_diff:+.3f} ship_diff={ship_diff:+.1f}",
                        flush=True,
                    )

            if args.include_four_player and time.time() < deadline:
                opponents = [loaded["snapshots/v5.py"], loaded["baseline"], loaded["random"]]
                for seat in range(4):
                    started = time.time()
                    lineup = list(opponents)
                    lineup.insert(seat, variant_agent)
                    result = run_4p(lineup)
                    reward = result["rewards"][seat]
                    rank = result["ranks"][seat]
                    ships = result["ships"][seat]
                    bucket = state["four_player"][variant_name]
                    bucket["games"] += 1
                    bucket["reward_sum"] += reward
                    bucket["rank_sum"] += rank
                    bucket["top2"] += int(rank <= 2)
                    bucket["ship_sum"] += ships

                    event = {
                        "timestamp": now_iso(),
                        "cycle": cycle,
                        "mode": "4p",
                        "variant": variant_name,
                        "seat": seat,
                        "reward": reward,
                        "rank": rank,
                        "ships": ships,
                        "duration_sec": round(time.time() - started, 3),
                    }
                    append_jsonl(jsonl_path, event)
                    summary_path.write_text(json.dumps(build_summary(state), indent=2), encoding="utf-8")
                    print(
                        f"[4p][cycle {cycle}] {variant_name} seat={seat} "
                        f"rank={rank} reward={reward:+.3f} ships={ships:.1f}",
                        flush=True,
                    )

        state["meta"]["cycles_completed"] = cycle
        state["meta"]["updated_at"] = now_iso()
        summary_path.write_text(json.dumps(build_summary(state), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
