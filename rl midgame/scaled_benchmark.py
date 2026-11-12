"""Scaled benchmark with confidence intervals and significance testing.

Run 100+ games per opponent to get statistically meaningful results.
Usable as both a library and CLI script.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

WORKSPACE_DIR = Path(__file__).resolve().parent
ROOT = WORKSPACE_DIR.parent
GENOME_DIR = ROOT / "genome test"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
if str(GENOME_DIR) not in sys.path:
    sys.path.insert(0, str(GENOME_DIR))

os.environ.setdefault("KAGGLE_ENVIRONMENTS_QUIET", "1")

import test_agent  # noqa: E402
from midgame_rl_agent import MidgameRLConfig, load_policy_agent  # noqa: E402
from weird_opponents import greedy_agent, turtle_agent  # noqa: E402

DEFAULT_OPPONENTS = ["baseline", "v21", "v23", "v16", "mtmr", "greedy", "turtle", "random"]

# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------


def wilson_ci(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    p_hat = wins / total
    denom = 1 + z * z / total
    centre = (p_hat + z * z / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / total + z * z / (4 * total * total))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def binomial_p_value(wins: int, total: int, p0: float = 0.5) -> float:
    """One-sided p-value: P(X >= wins) under H0: p = p0, using normal approx."""
    if total == 0:
        return 1.0
    p_hat = wins / total
    se = math.sqrt(p0 * (1 - p0) / total)
    if se == 0:
        return 0.0 if p_hat > p0 else 1.0
    z = (p_hat - p0) / se
    # Standard normal survival function via error function
    return 0.5 * math.erfc(z / math.sqrt(2))


def is_significant(wins: int, total: int, alpha: float = 0.05) -> bool:
    """Return True if win-rate is significantly different from 50%."""
    return binomial_p_value(wins, total) < alpha


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ScaledBenchmarkResult:
    opponent: str
    games: int
    wins: int
    losses: int
    draws: int
    score_rate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool
    avg_reward_diff: float
    avg_ship_diff: float


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------


def load_reference_agents() -> Dict[str, Any]:
    """Mirror of benchmark_midgame_policy.load_reference_agents()."""
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


def run_scaled_series(
    candidate: Any,
    opponent: Any,
    total_games: int = 100,
    opponent_name: str = "opponent",
) -> ScaledBenchmarkResult:
    """Run *total_games* between candidate and opponent, alternating seats."""
    wins = losses = draws = 0
    reward_diff = 0.0
    ship_diff = 0.0

    for i in range(total_games):
        if i % 2 == 0:
            result = test_agent.run_game(candidate, opponent)
            reward_diff += result["reward_a"] - result["reward_b"]
            ship_diff += result["ships_a"] - result["ships_b"]
            if result["winner"] == "A":
                wins += 1
            elif result["winner"] == "B":
                losses += 1
            else:
                draws += 1
        else:
            result = test_agent.run_game(opponent, candidate)
            reward_diff += result["reward_b"] - result["reward_a"]
            ship_diff += result["ships_b"] - result["ships_a"]
            if result["winner"] == "B":
                wins += 1
            elif result["winner"] == "A":
                losses += 1
            else:
                draws += 1

    games = max(total_games, 1)
    score = (wins + 0.5 * draws) / games
    ci_lo, ci_hi = wilson_ci(wins, games)
    pval = binomial_p_value(wins, games)

    return ScaledBenchmarkResult(
        opponent=opponent_name,
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        score_rate=score,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        p_value=pval,
        significant=is_significant(wins, games),
        avg_reward_diff=reward_diff / games,
        avg_ship_diff=ship_diff / games,
    )


# ---------------------------------------------------------------------------
# Full benchmark suite
# ---------------------------------------------------------------------------


def _format_table(results: List[ScaledBenchmarkResult]) -> str:
    """Return a human-readable table string."""
    header = (
        f"{'Opponent':<10} {'Games':>5} {'W':>4} {'L':>4} {'D':>4} "
        f"{'Score':>6} {'95% CI':>13} {'p-val':>7} {'Sig':>3} "
        f"{'ΔRew':>7} {'ΔShip':>7}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        sig_mark = " * " if r.significant else "   "
        lines.append(
            f"{r.opponent:<10} {r.games:>5} {r.wins:>4} {r.losses:>4} {r.draws:>4} "
            f"{r.score_rate:>6.1%} [{r.ci_lower:.3f},{r.ci_upper:.3f}] "
            f"{r.p_value:>7.4f} {sig_mark} "
            f"{r.avg_reward_diff:>+7.1f} {r.avg_ship_diff:>+7.1f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def run_full_benchmark(
    policy_path: str | Path,
    opponents: Optional[List[str]] = None,
    games_per_opponent: int = 100,
    rl_config: Optional[MidgameRLConfig] = None,
) -> List[ScaledBenchmarkResult]:
    """Benchmark a policy against every opponent in the pool."""
    if opponents is None:
        opponents = list(DEFAULT_OPPONENTS)
    if rl_config is None:
        rl_config = MidgameRLConfig()

    candidate = load_policy_agent(policy_path, rl_config=rl_config, explore=False)
    references = load_reference_agents()

    results: List[ScaledBenchmarkResult] = []
    for name in opponents:
        opp = references[name]
        print(f"  ▶ {name} ({games_per_opponent} games) …", flush=True)
        res = run_scaled_series(candidate, opp, total_games=games_per_opponent, opponent_name=name)
        results.append(res)
        print(f"    score {res.score_rate:.1%}  CI [{res.ci_lower:.3f},{res.ci_upper:.3f}]")

    print()
    print(_format_table(results))
    return results


# ---------------------------------------------------------------------------
# Head-to-head policy comparison
# ---------------------------------------------------------------------------


def compare_policies(
    policy_a_path: str | Path,
    policy_b_path: str | Path,
    opponents: Optional[List[str]] = None,
    games: int = 50,
    rl_config: Optional[MidgameRLConfig] = None,
) -> Dict[str, Any]:
    """Compare two policies head-to-head and against a shared opponent pool."""
    if opponents is None:
        opponents = list(DEFAULT_OPPONENTS)
    if rl_config is None:
        rl_config = MidgameRLConfig()

    agent_a = load_policy_agent(policy_a_path, rl_config=rl_config, explore=False)
    agent_b = load_policy_agent(policy_b_path, rl_config=rl_config, explore=False)

    # Head-to-head
    print("=== Head-to-head ===")
    h2h = run_scaled_series(agent_a, agent_b, total_games=games, opponent_name="policy_b")
    print(_format_table([h2h]))

    # Each policy vs opponents
    references = load_reference_agents()
    results_a: List[ScaledBenchmarkResult] = []
    results_b: List[ScaledBenchmarkResult] = []

    for name in opponents:
        opp = references[name]
        print(f"  ▶ vs {name} …", flush=True)
        ra = run_scaled_series(agent_a, opp, total_games=games, opponent_name=name)
        rb = run_scaled_series(agent_b, opp, total_games=games, opponent_name=name)
        results_a.append(ra)
        results_b.append(rb)

    print("\n=== Policy A ===")
    print(_format_table(results_a))
    print("\n=== Policy B ===")
    print(_format_table(results_b))

    return {
        "head_to_head": asdict(h2h),
        "policy_a": [asdict(r) for r in results_a],
        "policy_b": [asdict(r) for r in results_b],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scaled benchmark with confidence intervals and significance testing"
    )
    parser.add_argument("--policy", required=True, help="Path to policy JSON")
    parser.add_argument("--games", type=int, default=100, help="Total games per opponent")
    parser.add_argument("--opponents", nargs="+", default=DEFAULT_OPPONENTS, help="Opponent names")
    parser.add_argument("--compare", default=None, help="Second policy JSON for head-to-head")
    parser.add_argument("--activation-turn", type=int, default=24)
    parser.add_argument("--max-turn", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--summary-out",
        default=str(WORKSPACE_DIR / "results" / "scaled_benchmark_summary.json"),
        help="Where to save JSON results",
    )
    args = parser.parse_args()

    rl_config = MidgameRLConfig(
        activation_turn=args.activation_turn,
        max_turn=args.max_turn,
        min_candidates=2,
        top_k=args.top_k,
        contested_only=True,
        explore=False,
    )

    if args.compare:
        summary = compare_policies(
            args.policy,
            args.compare,
            opponents=args.opponents,
            games=args.games,
            rl_config=rl_config,
        )
    else:
        results = run_full_benchmark(
            args.policy,
            opponents=args.opponents,
            games_per_opponent=args.games,
            rl_config=rl_config,
        )
        summary = {
            "policy": str(Path(args.policy).resolve()),
            "games_per_opponent": args.games,
            "results": [asdict(r) for r in results],
        }

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
