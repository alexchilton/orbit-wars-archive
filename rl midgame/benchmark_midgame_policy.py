from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("KAGGLE_ENVIRONMENTS_QUIET", "1")

WORKSPACE_DIR = Path(__file__).resolve().parent
ROOT = WORKSPACE_DIR.parent
GENOME_DIR = ROOT / "genome test"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
if str(GENOME_DIR) not in sys.path:
    sys.path.insert(0, str(GENOME_DIR))

import test_agent  # noqa: E402
from midgame_rl_agent import MidgameRLConfig, load_policy_agent  # noqa: E402
from weird_opponents import greedy_agent, turtle_agent  # noqa: E402

DEFAULT_OPPONENTS = ["baseline", "v21", "v23", "v16", "mtmr", "greedy", "turtle", "random"]


def load_reference_agents() -> Dict[str, Any]:
    return {
        "baseline": test_agent.load_baseline_agent(),
        "v21": test_agent.load_agent_from_file(str(ROOT / "snapshots" / "v21.py")),
        "v23": test_agent.load_agent_from_file(str(ROOT / "snapshots" / "v23_state_pivot.py")),
        "v16": test_agent.load_agent_from_file(str(ROOT / "snapshots" / "v16_broken.py")),
        "mtmr": test_agent.load_agent_from_file(str(ROOT / "snapshots" / "mtmr_trial_copy.py")),
        "greedy": greedy_agent,
        "turtle": turtle_agent,
        "random": "random",
    }


def run_series(candidate: Any, opponent: Any, games_per_seat: int) -> Dict[str, float]:
    wins = losses = draws = 0
    reward_diff = 0.0
    ship_diff = 0.0

    for _ in range(games_per_seat):
        result = test_agent.run_game(candidate, opponent)
        reward_diff += result["reward_a"] - result["reward_b"]
        ship_diff += result["ships_a"] - result["ships_b"]
        if result["winner"] == "A":
            wins += 1
        elif result["winner"] == "B":
            losses += 1
        else:
            draws += 1

    for _ in range(games_per_seat):
        result = test_agent.run_game(opponent, candidate)
        reward_diff += result["reward_b"] - result["reward_a"]
        ship_diff += result["ships_b"] - result["ships_a"]
        if result["winner"] == "B":
            wins += 1
        elif result["winner"] == "A":
            losses += 1
        else:
            draws += 1

    games = max(1, games_per_seat * 2)
    return {
        "games": float(games),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "score_rate": (wins + 0.5 * draws) / games,
        "avg_reward_diff": reward_diff / games,
        "avg_ship_diff": ship_diff / games,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a saved RL midgame policy")
    parser.add_argument("--policy", required=True, help="Path to policy JSON produced by train_midgame_policy.py")
    parser.add_argument("--games-per-seat", type=int, default=3, help="Games per seat for each opponent")
    parser.add_argument("--activation-turn", type=int, default=24, help="First turn where RL may activate")
    parser.add_argument("--max-turn", type=int, default=160, help="Last turn where RL may activate")
    parser.add_argument("--top-k", type=int, default=8, help="Top heuristic missions exposed to the policy")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=DEFAULT_OPPONENTS,
        help="Opponent pool to benchmark against",
    )
    parser.add_argument(
        "--summary-out",
        default=str(WORKSPACE_DIR / "results" / "benchmark_summary.json"),
        help="Where to write the benchmark summary JSON",
    )
    args = parser.parse_args()

    references = load_reference_agents()
    rl_config = MidgameRLConfig(
        activation_turn=args.activation_turn,
        max_turn=args.max_turn,
        min_candidates=2,
        top_k=args.top_k,
        contested_only=True,
        explore=False,
    )
    candidate = load_policy_agent(args.policy, rl_config=rl_config, explore=False)
    summary: Dict[str, Any] = {
        "policy": str(Path(args.policy).resolve()),
        "games_per_seat": args.games_per_seat,
        "results": {},
    }

    for name in args.opponents:
        series = run_series(candidate, references[name], args.games_per_seat)
        summary["results"][name] = series
        print(
            f"{name:<8} score={series['score_rate']:.3f} "
            f"reward={series['avg_reward_diff']:+.3f} ships={series['avg_ship_diff']:+.1f}",
            flush=True,
        )

    output = Path(args.summary_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
