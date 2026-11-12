"""
Fair comparison between linear and MLP reranker policies.

Runs identical pretrain + fine-tune pipelines for both model types,
then benchmarks each with 100-game series against reference opponents.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
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

from midgame_policy import create_policy
from midgame_rl_agent import MidgameRLConfig, build_agent
from pretrain_from_heuristic import (
    collect_heuristic_ranking_samples,
    evaluate_policy_accuracy,
    pretrain_policy,
)
from replay_midgame_experiment import (
    select_replay_candidates,
    train_on_candidates,
    _load_json,
)
from benchmark_midgame_policy import load_reference_agents, run_series


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_OPPONENTS = ["baseline", "v21", "v23", "v16", "mtmr"]
DEFAULT_REPLAY_GLOBS = ["kaggle_replays/*/episode-*-replay.json"]


@dataclass
class ComparisonConfig:
    pretrain_positions: int = 80
    pretrain_epochs: int = 30
    pretrain_lr: float = 0.02
    finetune_episodes: int = 6
    finetune_lr: float = 0.05
    finetune_horizon: int = 50
    benchmark_games_per_seat: int = 50
    mlp_hidden_size: int = 64
    seed: int = 7
    player_name: str = "alex chilton"
    replay_globs: List[str] = field(default_factory=lambda: list(DEFAULT_REPLAY_GLOBS))
    opponents: List[str] = field(default_factory=lambda: list(DEFAULT_OPPONENTS))


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _resolve_replay_paths(config: ComparisonConfig) -> List[Path]:
    paths: List[Path] = []
    for pattern in config.replay_globs:
        paths.extend(sorted(ROOT.glob(pattern)))
    return sorted(set(paths))


def train_and_evaluate(
    kind: str,
    config: ComparisonConfig,
) -> Dict[str, Any]:
    """Run the full pretrain → fine-tune → benchmark pipeline for *kind*."""

    print(f"\n{'='*60}")
    print(f"  Training pipeline: {kind.upper()}")
    print(f"{'='*60}")
    t0 = time.time()

    # -- create policy -------------------------------------------------------
    policy = create_policy(
        kind=kind,
        hidden_size=config.mlp_hidden_size,
        seed=config.seed,
    )

    # -- collect pretrain samples -------------------------------------------
    replay_paths = _resolve_replay_paths(config)
    if not replay_paths:
        raise FileNotFoundError(
            f"No replays matched globs: {config.replay_globs}"
        )
    print(f"  Found {len(replay_paths)} replay files")

    pretrain_samples = collect_heuristic_ranking_samples(
        replay_paths=replay_paths,
        player_name=config.player_name,
        max_positions=config.pretrain_positions,
        seed=config.seed,
    )
    print(f"  Collected {len(pretrain_samples)} pretrain samples")

    # -- pretrain ------------------------------------------------------------
    pretrain_result = pretrain_policy(
        policy=policy,
        samples=pretrain_samples,
        epochs=config.pretrain_epochs,
        learning_rate=config.pretrain_lr,
    )
    pretrain_acc = evaluate_policy_accuracy(policy, pretrain_samples)
    print(f"  Pretrain accuracy: {pretrain_acc['target_accuracy']:.3f}")

    # -- fine-tune with replay RL -------------------------------------------
    candidates = select_replay_candidates(
        replay_paths=replay_paths,
        player_name=config.player_name,
        horizon=config.finetune_horizon,
    )
    print(f"  Selected {len(candidates)} replay candidates for fine-tuning")

    replay_by_path: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        if c.replay_path not in replay_by_path:
            replay_by_path[c.replay_path] = _load_json(c.replay_path)

    rl_config = MidgameRLConfig(explore=True)
    finetune_result = train_on_candidates(
        policy=policy,
        replay_by_path=replay_by_path,
        candidates=candidates,
        episodes=config.finetune_episodes,
        learning_rate=config.finetune_lr,
        horizon=config.finetune_horizon,
        rl_config=rl_config,
    )
    print(f"  Fine-tuning complete ({len(finetune_result.get('logs', []))} log entries)")

    # -- benchmark -----------------------------------------------------------
    ref_agents = load_reference_agents()
    candidate_agent = build_agent(policy=policy, rl_config=MidgameRLConfig())

    benchmark_results: Dict[str, Dict[str, float]] = {}
    for opp_name in config.opponents:
        if opp_name not in ref_agents:
            print(f"  WARNING: opponent '{opp_name}' not found, skipping")
            continue
        print(f"  Benchmarking vs {opp_name} ({config.benchmark_games_per_seat * 2} games) ...", end=" ", flush=True)
        result = run_series(
            candidate=candidate_agent,
            opponent=ref_agents[opp_name],
            games_per_seat=config.benchmark_games_per_seat,
        )
        benchmark_results[opp_name] = result
        print(f"score_rate={result['score_rate']:.3f}")

    elapsed = time.time() - t0
    print(f"  Pipeline complete in {elapsed:.1f}s")

    return {
        "kind": kind,
        "pretrain_accuracy": pretrain_acc,
        "finetune_logs": finetune_result.get("logs", []),
        "benchmark_results": benchmark_results,
        "elapsed_seconds": elapsed,
        "policy": policy,
    }


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------

def _aggregate_score_rate(benchmark_results: Dict[str, Dict[str, float]]) -> float:
    if not benchmark_results:
        return 0.0
    rates = [r["score_rate"] for r in benchmark_results.values()]
    return sum(rates) / len(rates)


def run_comparison(config: ComparisonConfig) -> Dict[str, Any]:
    """Train and benchmark both linear and MLP, then compare."""
    linear_results = train_and_evaluate("linear", config)
    mlp_results = train_and_evaluate("mlp", config)

    linear_agg = _aggregate_score_rate(linear_results["benchmark_results"])
    mlp_agg = _aggregate_score_rate(mlp_results["benchmark_results"])

    if mlp_agg > linear_agg:
        winner = "mlp"
    elif linear_agg > mlp_agg:
        winner = "linear"
    else:
        winner = "tie"

    format_comparison_table(linear_results, mlp_results)

    return {
        "config": asdict(config),
        "linear": _serialisable_results(linear_results),
        "mlp": _serialisable_results(mlp_results),
        "linear_aggregate_score_rate": linear_agg,
        "mlp_aggregate_score_rate": mlp_agg,
        "winner": winner,
    }


def _serialisable_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Strip non-serialisable objects (policy) for JSON output."""
    return {k: v for k, v in results.items() if k != "policy"}


# ---------------------------------------------------------------------------
# Formatted output
# ---------------------------------------------------------------------------

def _wilson_interval(wins: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95 % confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _significance_marker(
    sr1: float, n1: float, sr2: float, n2: float,
) -> str:
    """Rough two-proportion z-test; returns *, **, or '' for p < .05, .01."""
    if n1 == 0 or n2 == 0:
        return ""
    p_pool = (sr1 * n1 + sr2 * n2) / (n1 + n2)
    if p_pool <= 0 or p_pool >= 1:
        return ""
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return ""
    z = abs(sr1 - sr2) / se
    if z >= 2.576:
        return "**"
    if z >= 1.960:
        return "*"
    return ""


def format_comparison_table(
    linear_results: Dict[str, Any],
    mlp_results: Dict[str, Any],
) -> None:
    """Print a formatted comparison table to stdout."""
    lin_bench = linear_results["benchmark_results"]
    mlp_bench = mlp_results["benchmark_results"]
    all_opps = sorted(set(lin_bench) | set(mlp_bench))

    header = f"{'Opponent':<12} {'Linear SR':>10} {'MLP SR':>10} {'Diff':>8} {'Sig':>4}"
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("  COMPARISON: Linear vs MLP Reranker")
    print(sep)
    print(header)
    print(sep)

    total_lin_score, total_mlp_score = 0.0, 0.0
    total_lin_games, total_mlp_games = 0.0, 0.0

    for opp in all_opps:
        lr = lin_bench.get(opp, {})
        mr = mlp_bench.get(opp, {})
        l_sr = lr.get("score_rate", 0.0)
        m_sr = mr.get("score_rate", 0.0)
        l_n = lr.get("games", 0)
        m_n = mr.get("games", 0)
        diff = m_sr - l_sr
        sig = _significance_marker(l_sr, l_n, m_sr, m_n)
        print(f"{opp:<12} {l_sr:>10.3f} {m_sr:>10.3f} {diff:>+8.3f} {sig:>4}")
        total_lin_score += l_sr * l_n
        total_mlp_score += m_sr * m_n
        total_lin_games += l_n
        total_mlp_games += m_n

    agg_lin = total_lin_score / total_lin_games if total_lin_games else 0.0
    agg_mlp = total_mlp_score / total_mlp_games if total_mlp_games else 0.0
    agg_diff = agg_mlp - agg_lin
    agg_sig = _significance_marker(agg_lin, total_lin_games, agg_mlp, total_mlp_games)

    print(sep)
    print(f"{'AGGREGATE':<12} {agg_lin:>10.3f} {agg_mlp:>10.3f} {agg_diff:>+8.3f} {agg_sig:>4}")
    print(sep)

    # Pretrain accuracy
    lin_pa = linear_results.get("pretrain_accuracy", {}).get("target_accuracy", 0.0)
    mlp_pa = mlp_results.get("pretrain_accuracy", {}).get("target_accuracy", 0.0)
    print(f"\n  Pretrain accuracy  — Linear: {lin_pa:.3f}  MLP: {mlp_pa:.3f}")

    # Elapsed time
    lin_t = linear_results.get("elapsed_seconds", 0.0)
    mlp_t = mlp_results.get("elapsed_seconds", 0.0)
    print(f"  Training time      — Linear: {lin_t:.1f}s   MLP: {mlp_t:.1f}s")

    # Winner
    if agg_mlp > agg_lin:
        winner = "MLP"
    elif agg_lin > agg_mlp:
        winner = "Linear"
    else:
        winner = "Tie"

    ci_lin = _wilson_interval(agg_lin * total_lin_games, total_lin_games)
    ci_mlp = _wilson_interval(agg_mlp * total_mlp_games, total_mlp_games)
    print(f"\n  Winner: {winner}  (Δ = {agg_diff:+.3f})")
    print(f"  Linear 95% CI: [{ci_lin[0]:.3f}, {ci_lin[1]:.3f}]")
    print(f"  MLP    95% CI: [{ci_mlp[0]:.3f}, {ci_mlp[1]:.3f}]")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare linear vs MLP midgame reranker policies.",
    )
    parser.add_argument(
        "--replay-glob", nargs="+", default=DEFAULT_REPLAY_GLOBS,
        help="Glob patterns for replay files (relative to ROOT).",
    )
    parser.add_argument("--player-name", default="alex chilton")
    parser.add_argument("--pretrain-positions", type=int, default=80)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--finetune-episodes", type=int, default=6)
    parser.add_argument("--benchmark-games", type=int, default=50,
                        help="Games per seat (total = 2× this).")
    parser.add_argument(
        "--opponents", nargs="+", default=DEFAULT_OPPONENTS,
        help="Reference agents to benchmark against.",
    )
    parser.add_argument("--summary-out", type=str, default=None,
                        help="Path to write JSON summary.")
    parser.add_argument("--save-policies", action="store_true",
                        help="Save both trained policies to disk.")
    args = parser.parse_args()

    config = ComparisonConfig(
        pretrain_positions=args.pretrain_positions,
        pretrain_epochs=args.pretrain_epochs,
        finetune_episodes=args.finetune_episodes,
        benchmark_games_per_seat=args.benchmark_games,
        player_name=args.player_name,
        replay_globs=args.replay_glob,
        opponents=args.opponents,
    )

    # Run the full train+evaluate for both linear and MLP, then compare
    linear_results = train_and_evaluate("linear", config)
    mlp_results = train_and_evaluate("mlp", config)

    linear_agg = _aggregate_score_rate(linear_results["benchmark_results"])
    mlp_agg = _aggregate_score_rate(mlp_results["benchmark_results"])

    if mlp_agg > linear_agg:
        winner = "mlp"
    elif linear_agg > mlp_agg:
        winner = "linear"
    else:
        winner = "tie"

    format_comparison_table(linear_results, mlp_results)

    summary = {
        "config": asdict(config),
        "linear": _serialisable_results(linear_results),
        "mlp": _serialisable_results(mlp_results),
        "linear_aggregate_score_rate": linear_agg,
        "mlp_aggregate_score_rate": mlp_agg,
        "winner": winner,
    }

    # Optionally save policies
    if args.save_policies:
        out_dir = WORKSPACE_DIR / "comparison_policies"
        out_dir.mkdir(parents=True, exist_ok=True)
        for res in (linear_results, mlp_results):
            kind = res["kind"]
            path = out_dir / f"{kind}_policy.json"
            res["policy"].save_json(path)
            print(f"  Saved {kind} policy → {path}")

    # Write summary JSON
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"  Summary written → {out}")

    print(f"\nDone. Winner: {winner}")


if __name__ == "__main__":
    main()
