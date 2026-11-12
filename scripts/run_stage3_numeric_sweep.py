from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
GENOME_DIR = ROOT / "genome test"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GENOME_DIR) not in sys.path:
    sys.path.insert(0, str(GENOME_DIR))

from genetic_search import load_reference_agents, run_four_player_series, run_two_player_pool  # noqa: E402


DEFAULT_BALANCED_AGENT = ROOT / "submission" / "main_stage3_balanced_release_candidate.py"
DEFAULT_TWO_PLAYER_AGENT = ROOT / "submission" / "main_stage3_two_player_release_candidate.py"
DEFAULT_FORCE_CONCENTRATION_AGENT = ROOT / "submission" / "variant_b_force_concentration.py"
DEFAULT_FORCE_PLUS_DEDUP_AGENT = ROOT / "submission" / "variant_e_force_plus_dedup.py"


PRESET_SPACES: Dict[str, "OrderedDict[str, tuple[float | int, ...]]"] = {
    "balanced": OrderedDict(
        [
            ("FINISHING_THRESHOLD", (0.24, 0.28, 0.32, 0.35)),
            ("FINISHING_PROD_RATIO", (1.12, 1.18, 1.25, 1.32)),
            ("PROACTIVE_KEEP_RATIO", (0.14, 0.18, 0.22)),
            ("STACKED_PROACTIVE_KEEP_RATIO", (0.18, 0.22, 0.26)),
            ("FOUR_PLAYER_RUNAWAY_PRIORITY", (1.22, 1.34, 1.46)),
            ("FOUR_PLAYER_RUNAWAY_STRENGTH_RATIO", (1.12, 1.18, 1.24)),
            ("FOUR_PLAYER_RECOVERY_HOSTILE_BONUS", (1.04, 1.10, 1.16)),
            ("FOUR_PLAYER_FRONTLINE_RESERVE_RATIO", (0.14, 0.18, 0.22)),
        ]
    ),
    "concentration": OrderedDict(
        [
            ("FORCE_CONCENTRATION_MIN_RATIO", (0.55, 0.65, 0.75)),
            ("FINISHING_THRESHOLD", (0.28, 0.32, 0.35)),
            ("FINISHING_PROD_RATIO", (1.18, 1.25, 1.32)),
            ("PROACTIVE_KEEP_RATIO", (0.14, 0.18, 0.22)),
            ("STACKED_PROACTIVE_KEEP_RATIO", (0.18, 0.22, 0.26)),
            ("FOUR_PLAYER_FRONTLINE_RESERVE_RATIO", (0.14, 0.18, 0.22)),
            ("FOUR_PLAYER_DOUBLE_FRONT_RESERVE_RATIO", (0.22, 0.28, 0.34)),
            ("FOUR_PLAYER_PIVOT_PROD_RATIO", (1.10, 1.18, 1.26)),
            ("FOUR_PLAYER_PIVOT_HOSTILE_BONUS", (1.10, 1.18, 1.26)),
            ("FOUR_PLAYER_PREP_HOSTILE_DAMP", (0.76, 0.82, 0.88)),
        ]
    ),
    "two_player": OrderedDict(
        [
            ("FINISHING_THRESHOLD", (0.20, 0.24, 0.28, 0.32)),
            ("FINISHING_PROD_RATIO", (1.08, 1.15, 1.22, 1.28)),
            ("PROACTIVE_KEEP_RATIO", (0.12, 0.18, 0.24)),
            ("MTMR_OPENING_LAUNCH_CAP", (1, 2, 3)),
            ("MTMR_DUEL_OPENING_LAUNCH_CAP", (2, 3, 4)),
            ("MTMR_SAFE_NEUTRAL_LIMIT", (0, 1, 2)),
            ("MTMR_DUEL_HOSTILITY_PROD_RATIO", (1.05, 1.12, 1.18)),
            ("MTMR_STAGE_MIN_SHIPS", (12, 14, 16)),
        ]
    ),
    "four_player": OrderedDict(
        [
            ("FOUR_PLAYER_CAUTIOUS_LAUNCH_CAP", (3, 4, 5)),
            ("FOUR_PLAYER_PRESSURED_LAUNCH_CAP", (2, 3, 4)),
            ("FOUR_PLAYER_RECOVERY_LAUNCH_CAP", (4, 5, 6)),
            ("FOUR_PLAYER_FRONTLINE_RESERVE_RATIO", (0.14, 0.18, 0.22)),
            ("FOUR_PLAYER_DOUBLE_FRONT_RESERVE_RATIO", (0.22, 0.28, 0.34)),
            ("FOUR_PLAYER_RUNAWAY_PRIORITY", (1.22, 1.34, 1.46)),
            ("FOUR_PLAYER_RUNAWAY_PROD_RATIO", (1.08, 1.12, 1.18)),
            ("FOUR_PLAYER_RECOVERY_HOSTILE_BONUS", (1.04, 1.10, 1.16)),
        ]
    ),
}

PRESET_TWO_PLAYER_OPPONENTS: Dict[str, List[str]] = {
    "balanced": [
        "baseline",
        "oldbase_balanced",
        "oldbase_two_player",
        "v21",
        "v23",
        "v16",
        "mtmr",
        "greedy",
        "turtle",
        "random",
    ],
    "concentration": [
        "baseline",
        "oldbase_balanced",
        "oldbase_two_player",
        "v21",
        "v23",
        "v16",
        "mtmr",
        "greedy",
        "turtle",
        "random",
    ],
    "two_player": [
        "baseline",
        "oldbase_two_player",
        "oldbase_balanced",
        "v21",
        "v23",
        "mtmr",
        "greedy",
        "turtle",
        "random",
    ],
    "four_player": [
        "baseline",
        "oldbase_balanced",
        "v23",
        "random",
    ],
}

PRESET_FOUR_PLAYER_OPPONENTS: Dict[str, List[str]] = {
    "balanced": ["oldbase_balanced", "baseline", "v23"],
    "concentration": ["oldbase_balanced", "baseline", "v23"],
    "two_player": ["oldbase_balanced", "baseline", "v23"],
    "four_player": ["oldbase_balanced", "baseline", "v23"],
}

PRESET_WEIGHTS: Dict[str, tuple[float, float]] = {
    "balanced": (0.55, 0.45),
    "concentration": (0.55, 0.45),
    "two_player": (0.8, 0.2),
    "four_player": (0.25, 0.75),
}

MODULE_CACHE: Dict[Path, Any] = {}
MODULE_DEFAULTS: Dict[Path, Dict[str, float | int]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a targeted numeric hyperparameter sweep around the stage3 champions")
    parser.add_argument("--preset", choices=sorted(PRESET_SPACES), default="balanced")
    parser.add_argument(
        "--agent-path",
        help="Frozen stage3 submission file to tune. Defaults to the stage3 balanced or two-player release candidate.",
    )
    parser.add_argument("--mode", choices=("random", "grid"), default="random")
    parser.add_argument("--trials", type=int, default=48, help="Target number of evaluated configs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--games-per-seat", type=int, default=4)
    parser.add_argument("--skip-four-player", action="store_true")
    parser.add_argument("--emit-top", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="genome test/results_stage3_numeric_sweep",
        help="Directory for logs and summary JSON",
    )
    parser.add_argument(
        "--generated-dir",
        default="genome test/generated_stage3_numeric_sweep",
        help="Directory where the top frozen candidates are written",
    )
    parser.add_argument(
        "--two-player-opponents",
        nargs="*",
        default=None,
        help="Override the default 2p opponent list for the chosen preset",
    )
    parser.add_argument(
        "--four-player-opponents",
        nargs="*",
        default=None,
        help="Override the default 4p opponent lineup (must contain exactly 3 names)",
    )
    return parser.parse_args()


def default_agent_path(preset: str) -> Path:
    if preset == "concentration":
        if DEFAULT_FORCE_CONCENTRATION_AGENT.exists():
            return DEFAULT_FORCE_CONCENTRATION_AGENT
        if DEFAULT_FORCE_PLUS_DEDUP_AGENT.exists():
            return DEFAULT_FORCE_PLUS_DEDUP_AGENT
        return DEFAULT_BALANCED_AGENT
    return DEFAULT_TWO_PLAYER_AGENT if preset == "two_player" else DEFAULT_BALANCED_AGENT


def load_or_import_module(agent_path: Path, tunable_names: Sequence[str]) -> Any:
    agent_path = agent_path.resolve()
    if agent_path not in MODULE_CACHE:
        module_name = f"_stage3_numeric_sweep_{agent_path.stem}_{abs(hash(str(agent_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, agent_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load candidate agent from {agent_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        MODULE_CACHE[agent_path] = module
        MODULE_DEFAULTS[agent_path] = {name: getattr(module, name) for name in tunable_names}
    return MODULE_CACHE[agent_path]


def reset_agent_memory(module: Any) -> None:
    memory = getattr(module, "AGENT_MEMORY", None)
    if isinstance(memory, MutableMapping):
        memory.clear()
        memory.update(
            {
                "last_owners": {},
                "last_step": None,
                "player": None,
                "last_error": None,
            }
        )


def build_candidate(agent_path: Path, tunable_names: Sequence[str], overrides: Mapping[str, float | int]) -> Any:
    module = load_or_import_module(agent_path, tunable_names)
    for name, value in MODULE_DEFAULTS[agent_path.resolve()].items():
        setattr(module, name, value)
    for name, value in overrides.items():
        setattr(module, name, value)
    reset_agent_memory(module)
    return module.agent


def config_signature(overrides: Mapping[str, float | int]) -> str:
    return json.dumps(dict(sorted(overrides.items())), sort_keys=True)


def weighted_average(parts: Sequence[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in parts)
    if total_weight <= 0:
        return 0.0
    return sum(value * weight for value, weight in parts) / total_weight


def evaluate_candidate(
    agent_path: Path,
    overrides: Mapping[str, float | int],
    references: Dict[str, Any],
    two_player_opponents: Sequence[str],
    four_player_opponents: Sequence[str],
    games_per_seat: int,
    preset: str,
    skip_four_player: bool,
) -> Dict[str, Any]:
    tunable_names = list(PRESET_SPACES[preset].keys())
    candidate = build_candidate(agent_path, tunable_names, overrides)
    two_player = run_two_player_pool(candidate, references, two_player_opponents, games_per_seat)
    if skip_four_player:
        four_player = {
            "games": 0.0,
            "wins": 0.0,
            "top2": 0.0,
            "rank_sum": 0.0,
            "reward_sum": 0.0,
            "ship_sum": 0.0,
            "avg_rank": 4.0,
            "top2_rate": 0.0,
            "avg_reward": 0.0,
            "avg_ships": 0.0,
            "score": 0.0,
        }
    else:
        four_player = run_four_player_series(candidate, references, four_player_opponents)
    two_weight, four_weight = PRESET_WEIGHTS[preset]
    primary_score = weighted_average(
        [
            (two_player["score"], two_weight),
            (four_player["score"], four_weight if not skip_four_player else 0.0),
        ]
    )
    ship_tiebreak = sum(series["avg_ship_diff"] for series in two_player["series"].values()) + four_player["avg_ships"]
    return {
        "overrides": dict(overrides),
        "signature": config_signature(overrides),
        "two_player": two_player,
        "four_player": four_player,
        "primary_score": primary_score,
        "ship_tiebreak": ship_tiebreak,
    }


def iter_grid(space: "OrderedDict[str, tuple[float | int, ...]]") -> Iterable[Dict[str, float | int]]:
    keys = list(space.keys())
    for combo in itertools.product(*(space[key] for key in keys)):
        yield dict(zip(keys, combo))


def sample_random_configs(
    space: "OrderedDict[str, tuple[float | int, ...]]",
    rng: random.Random,
    target_count: int,
    seen: set[str],
) -> List[Dict[str, float | int]]:
    total_space = math.prod(len(values) for values in space.values())
    keys = list(space.keys())
    picked: List[Dict[str, float | int]] = []
    attempts = 0
    max_attempts = max(1000, target_count * 50)
    while len(seen) + len(picked) < total_space and len(picked) < target_count and attempts < max_attempts:
        attempts += 1
        candidate = {key: rng.choice(space[key]) for key in keys}
        signature = config_signature(candidate)
        if signature in seen or any(config_signature(existing) == signature for existing in picked):
            continue
        picked.append(candidate)
    if len(picked) < target_count and len(seen) + len(picked) < total_space:
        for candidate in iter_grid(space):
            signature = config_signature(candidate)
            if signature in seen or any(config_signature(existing) == signature for existing in picked):
                continue
            picked.append(candidate)
            if len(picked) >= target_count:
                break
    return picked


def write_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_existing_results(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.exists():
        return []
    results: List[Dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if payload.get("record_type") == "complete":
            results.append(payload["result"])
    return results


def summarize_results(
    results: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    agent_path: Path,
    space: "OrderedDict[str, tuple[float | int, ...]]",
    output_path: Path,
) -> None:
    ranked = sorted(results, key=lambda item: (item["primary_score"], item["ship_tiebreak"]), reverse=True)
    summary = {
        "meta": {
            "preset": args.preset,
            "agent_path": str(agent_path.resolve()),
            "mode": args.mode,
            "trials_requested": args.trials,
            "trials_completed": len(results),
            "games_per_seat": args.games_per_seat,
            "skip_four_player": args.skip_four_player,
            "seed": args.seed,
        },
        "parameter_space": {name: list(values) for name, values in space.items()},
        "best": ranked[0] if ranked else None,
        "top": ranked[: args.emit_top],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_overridden_agent(base_agent_path: Path, overrides: Mapping[str, float | int], output_path: Path) -> None:
    source = base_agent_path.read_text(encoding="utf-8").rstrip() + "\n\n"
    source += "# --- Numeric sweep overrides ----------------------------------------------\n"
    for name, value in overrides.items():
        source += f"{name} = {value!r}\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8")


def emit_top_agents(
    ranked: Sequence[Dict[str, Any]],
    base_agent_path: Path,
    generated_dir: Path,
    emit_top: int,
) -> None:
    generated_dir.mkdir(parents=True, exist_ok=True)
    for index, result in enumerate(ranked[:emit_top], start=1):
        primary = f"{result['primary_score']:.4f}".replace(".", "p")
        output_path = generated_dir / f"top-{index:02d}-{primary}.py"
        write_overridden_agent(base_agent_path, result["overrides"], output_path)


def main() -> None:
    args = parse_args()
    space = PRESET_SPACES[args.preset]
    agent_path = Path(args.agent_path).resolve() if args.agent_path else default_agent_path(args.preset).resolve()
    if not agent_path.exists():
        raise SystemExit(f"Missing base agent for sweep: {agent_path}")

    output_dir = Path(args.output_dir).resolve()
    generated_dir = Path(args.generated_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "search_log.jsonl"
    summary_path = output_dir / "search_summary.json"

    references = load_reference_agents()
    two_player_opponents = args.two_player_opponents or PRESET_TWO_PLAYER_OPPONENTS[args.preset]
    four_player_opponents = args.four_player_opponents or PRESET_FOUR_PLAYER_OPPONENTS[args.preset]
    if not args.skip_four_player and len(four_player_opponents) != 3:
        raise SystemExit("4-player evaluation requires exactly 3 opponent names")

    results = load_existing_results(log_path) if args.resume else []
    seen = {result["signature"] for result in results}
    rng = random.Random(args.seed)

    if args.mode == "grid":
        all_configs = list(iter_grid(space))
        pending = [config for config in all_configs if config_signature(config) not in seen][: max(0, args.trials - len(results))]
    else:
        pending = sample_random_configs(space, rng, max(0, args.trials - len(results)), seen)

    summarize_results(results, args, agent_path, space, summary_path)
    write_jsonl(
        log_path,
        {
            "record_type": "start",
            "preset": args.preset,
            "agent_path": str(agent_path.resolve()),
            "mode": args.mode,
            "trials_requested": args.trials,
            "trials_completed": len(results),
            "trials_pending": len(pending),
            "games_per_seat": args.games_per_seat,
            "skip_four_player": args.skip_four_player,
            "seed": args.seed,
        },
    )
    print(
        f"Starting sweep preset={args.preset} agent={agent_path.name} "
        f"completed={len(results)} pending={len(pending)}",
        flush=True,
    )

    start_time = time.time()
    for index, overrides in enumerate(pending, start=len(results) + 1):
        result = evaluate_candidate(
            agent_path=agent_path,
            overrides=overrides,
            references=references,
            two_player_opponents=two_player_opponents,
            four_player_opponents=four_player_opponents,
            games_per_seat=args.games_per_seat,
            preset=args.preset,
            skip_four_player=args.skip_four_player,
        )
        results.append(result)
        seen.add(result["signature"])
        write_jsonl(
            log_path,
            {
                "record_type": "complete",
                "evaluated_index": index,
                "elapsed_sec": round(time.time() - start_time, 3),
                "result": result,
            },
        )
        ranked = sorted(results, key=lambda item: (item["primary_score"], item["ship_tiebreak"]), reverse=True)
        summarize_results(results, args, agent_path, space, summary_path)
        emit_top_agents(ranked, agent_path, generated_dir, args.emit_top)
        best = ranked[0]
        print(
            f"[{index}/{args.trials}] primary={result['primary_score']:.4f} "
            f"2p={result['two_player']['score']:.4f} 4p={result['four_player']['score']:.4f} "
            f"best={best['primary_score']:.4f}",
            flush=True,
        )

    summarize_results(results, args, agent_path, space, summary_path)
    ranked = sorted(results, key=lambda item: (item["primary_score"], item["ship_tiebreak"]), reverse=True)
    emit_top_agents(ranked, agent_path, generated_dir, args.emit_top)
    if ranked:
        print(json.dumps({"best_primary_score": ranked[0]["primary_score"], "best_overrides": ranked[0]["overrides"]}, indent=2))


if __name__ == "__main__":
    main()
