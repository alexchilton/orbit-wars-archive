from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("KAGGLE_ENVIRONMENTS_QUIET", "1")

WORKSPACE_DIR = Path(__file__).resolve().parent
ROOT = WORKSPACE_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import test_agent  # noqa: E402
from run_overnight_benchmark import run_4p  # noqa: E402

from genome_agent import (  # noqa: E402
    GENE_SPACE,
    GenomeConfig,
    PRESET_GENOMES,
    build_agent,
    crossover_genomes,
    mutate_genome,
    random_genome,
    write_agent_wrapper,
)
from weird_opponents import greedy_agent, turtle_agent  # noqa: E402

DEFAULT_TWO_PLAYER_OPPONENTS = [
    "baseline",
    "oldbase_balanced",
    "oldbase_two_player",
    "release_candidate_v2",
    "v21",
    "v23",
    "v16",
    "mtmr",
    "greedy",
    "turtle",
    "random",
]
DEFAULT_FOUR_PLAYER_OPPONENTS = [
    "oldbase_balanced",
    "release_candidate_v2",
    "s2_4p_antidogpile",
]
STAGE3_WRAPPER_PATHS = {
    "stage3_balanced": WORKSPACE_DIR / "generated_stage3" / "current-balanced.py",
    "stage3_two_player": WORKSPACE_DIR / "generated_stage3" / "current-two-player.py",
    "stage3_four_player": WORKSPACE_DIR / "generated_stage3" / "current-four-player.py",
}


def load_reference_agents() -> Dict[str, Any]:
    references = {
        "baseline": test_agent.load_baseline_agent(),
        "oldbase_balanced": test_agent.load_agent_from_file(
            str(ROOT / "submission" / "main_stage2_oldbase_current_balanced.py")
        ),
        "oldbase_two_player": test_agent.load_agent_from_file(
            str(ROOT / "submission" / "main_stage2_oldbase_current_two_player.py")
        ),
        "release_candidate_v2": test_agent.load_agent_from_file(
            str(ROOT / "submission" / "main_release_candidate_v2.py")
        ),
        "s2_4p_antidogpile": test_agent.load_agent_from_file(
            str(ROOT / "submission" / "main_s2_4p_antidogpile.py")
        ),
        "v21": test_agent.load_agent_from_file(str(ROOT / "snapshots" / "v21.py")),
        "v23": test_agent.load_agent_from_file(str(ROOT / "snapshots" / "v23_state_pivot.py")),
        "v16": test_agent.load_agent_from_file(str(ROOT / "snapshots" / "v16_broken.py")),
        "mtmr": test_agent.load_agent_from_file(str(ROOT / "snapshots" / "mtmr_trial_copy.py")),
        "random": "random",
        "greedy": greedy_agent,
        "turtle": turtle_agent,
    }
    for name, path in STAGE3_WRAPPER_PATHS.items():
        if path.exists():
            references[name] = test_agent.load_agent_from_file(str(path))
    return references


def weighted_average(parts: Sequence[Tuple[float, float]]) -> float:
    numerator = sum(value * weight for value, weight in parts if weight > 0.0)
    denominator = sum(weight for _value, weight in parts if weight > 0.0)
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_two_player_series(candidate: Any, opponent: Any, games_per_seat: int) -> Dict[str, float]:
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


def run_two_player_pool(
    candidate: Any,
    opponents: Dict[str, Any],
    names: Sequence[str],
    games_per_seat: int,
    existing_series: Optional[Dict[str, Dict[str, float]]] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    series: Dict[str, Dict[str, float]] = dict(existing_series or {})
    total = len(names)
    for index, name in enumerate(names, start=1):
        if name in series:
            continue
        result = run_two_player_series(candidate, opponents[name], games_per_seat)
        series[name] = result
        if progress_callback is not None:
            progress_callback(
                name,
                {
                    "index": index,
                    "total": total,
                    "result": result,
                },
            )
    score = sum(item["score_rate"] for item in series.values()) / max(1, len(series))
    return {
        "series": series,
        "score": score,
    }


def run_four_player_series(
    candidate: Any,
    references: Dict[str, Any],
    opponent_names: Sequence[str],
    existing_seat_results: Optional[Dict[int, Dict[str, Any]]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, float]:
    names = ["candidate", *opponent_names]
    if len(names) != 4:
        raise ValueError("4-player evaluation requires exactly 3 opponent names")
    agent_map = {"candidate": candidate, **references}
    seat_results: Dict[int, Dict[str, Any]] = dict(existing_seat_results or {})
    bucket = {
        "games": 0.0,
        "wins": 0.0,
        "top2": 0.0,
        "rank_sum": 0.0,
        "reward_sum": 0.0,
        "ship_sum": 0.0,
    }

    for seat in range(4):
        if seat not in seat_results:
            order = names[seat:] + names[:seat]
            result = run_4p([agent_map[name] for name in order])
            seat_results[seat] = {
                "seat": seat,
                "order": order,
                "result": result,
            }
            if progress_callback is not None:
                progress_callback(seat_results[seat])
        seat_payload = seat_results[seat]
        order = seat_payload["order"]
        result = seat_payload["result"]
        idx = order.index("candidate")
        rank = float(result["ranks"][idx])
        reward = float(result["rewards"][idx])
        ships = float(result["ships"][idx])
        bucket["games"] += 1.0
        bucket["rank_sum"] += rank
        bucket["reward_sum"] += reward
        bucket["ship_sum"] += ships
        if rank == 1.0:
            bucket["wins"] += 1.0
        if rank <= 2.0:
            bucket["top2"] += 1.0

    games = max(1.0, bucket["games"])
    avg_rank = bucket["rank_sum"] / games
    normalized_rank = 1.0 - ((avg_rank - 1.0) / 3.0)
    top2_rate = bucket["top2"] / games
    return {
        **bucket,
        "avg_rank": avg_rank,
        "top2_rate": top2_rate,
        "avg_reward": bucket["reward_sum"] / games,
        "avg_ships": bucket["ship_sum"] / games,
        "score": 0.7 * normalized_rank + 0.3 * top2_rate,
    }


def evaluate_fixed_pool(
    genome: GenomeConfig,
    candidate: Any,
    references: Dict[str, Any],
    two_player_opponents: Sequence[str],
    four_player_opponents: Sequence[str],
    games_per_seat: int,
    include_four_player: bool,
    existing_partial: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    existing_partial = existing_partial or {}
    two_player = run_two_player_pool(
        candidate,
        references,
        two_player_opponents,
        games_per_seat,
        existing_series=existing_partial.get("fixed_two_player"),
        progress_callback=(
            None
            if progress_callback is None
            else lambda opponent, payload: progress_callback("fixed_two_player", {"opponent": opponent, **payload})
        ),
    )
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
    if include_four_player:
        four_player = run_four_player_series(
            candidate,
            references,
            four_player_opponents,
            existing_seat_results=existing_partial.get("four_player_seats"),
            progress_callback=(
                None
                if progress_callback is None
                else lambda payload: progress_callback("four_player", payload)
            ),
        )
    return {
        "genome": asdict(genome),
        "slug": genome.slug(),
        "fixed_two_player": two_player["series"],
        "fixed_two_player_score": two_player["score"],
        "four_player": four_player,
    }


def run_population_round_robin(
    population: Sequence[GenomeConfig],
    agents: Dict[GenomeConfig, Any],
    games_per_seat: int,
    existing_pair_results: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
    progress_callback: Optional[Callable[[GenomeConfig, GenomeConfig, Dict[str, float], int, int], None]] = None,
) -> Dict[GenomeConfig, Dict[str, float]]:
    stats = {
        genome: {
            "games": 0.0,
            "points": 0.0,
            "reward_diff_sum": 0.0,
            "ship_diff_sum": 0.0,
            "score_rate": 0.0,
            "avg_reward_diff": 0.0,
            "avg_ship_diff": 0.0,
        }
        for genome in population
    }
    if games_per_seat <= 0 or len(population) < 2:
        return stats

    existing_pair_results = existing_pair_results or {}
    total_pairs = (len(population) * (len(population) - 1)) // 2
    pair_index = 0
    for left_index in range(len(population)):
        for right_index in range(left_index + 1, len(population)):
            left = population[left_index]
            right = population[right_index]
            pair_key = tuple(sorted((left.slug(), right.slug())))
            if pair_key in existing_pair_results:
                series = existing_pair_results[pair_key]
            else:
                series = run_two_player_series(agents[left], agents[right], games_per_seat)
            pair_index += 1
            if progress_callback is not None and pair_key not in existing_pair_results:
                progress_callback(left, right, series, pair_index, total_pairs)
            games = series["games"]
            left_points = series["score_rate"] * games
            right_points = games - left_points
            stats[left]["games"] += games
            stats[left]["points"] += left_points
            stats[left]["reward_diff_sum"] += series["avg_reward_diff"] * games
            stats[left]["ship_diff_sum"] += series["avg_ship_diff"] * games
            stats[right]["games"] += games
            stats[right]["points"] += right_points
            stats[right]["reward_diff_sum"] -= series["avg_reward_diff"] * games
            stats[right]["ship_diff_sum"] -= series["avg_ship_diff"] * games

    for genome, bucket in stats.items():
        games = max(1.0, bucket["games"])
        bucket["score_rate"] = bucket["points"] / games
        bucket["avg_reward_diff"] = bucket["reward_diff_sum"] / games
        bucket["avg_ship_diff"] = bucket["ship_diff_sum"] / games
    return stats


def build_champion_mutants(
    champion_history: Sequence[Tuple[str, GenomeConfig]],
    rng: random.Random,
    limit: int,
) -> List[Tuple[str, Any]]:
    mutants: List[Tuple[str, Any]] = []
    seen: set[GenomeConfig] = set()
    for label, genome in reversed(champion_history):
        if len(mutants) >= limit:
            break
        mutant = mutate_genome(genome, rng, mutation_rate=0.35)
        if mutant in seen:
            mutant = mutate_genome(mutant, rng, mutation_rate=0.55)
        if mutant in seen:
            continue
        seen.add(mutant)
        mutants.append((f"mutant-{label}-{mutant.slug()}", build_agent(mutant)))
    return mutants


def run_mutant_pool(
    candidate: Any,
    mutants: Sequence[Tuple[str, Any]],
    games_per_seat: int,
    existing_series: Optional[Dict[str, Dict[str, float]]] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    if games_per_seat <= 0 or not mutants:
        return {
            "series": {},
            "games": 0.0,
            "score_rate": 0.0,
            "avg_reward_diff": 0.0,
            "avg_ship_diff": 0.0,
        }
    series: Dict[str, Dict[str, float]] = dict(existing_series or {})
    total = len(mutants)
    for index, (name, opponent) in enumerate(mutants, start=1):
        if name in series:
            continue
        result = run_two_player_series(candidate, opponent, games_per_seat)
        series[name] = result
        if progress_callback is not None:
            progress_callback(
                name,
                {
                    "index": index,
                    "total": total,
                    "result": result,
                },
            )
    games = sum(result["games"] for result in series.values())
    if games <= 0.0:
        return {
            "series": series,
            "games": 0.0,
            "score_rate": 0.0,
            "avg_reward_diff": 0.0,
            "avg_ship_diff": 0.0,
        }
    points = sum(result["score_rate"] * result["games"] for result in series.values())
    reward_sum = sum(result["avg_reward_diff"] * result["games"] for result in series.values())
    ship_sum = sum(result["avg_ship_diff"] * result["games"] for result in series.values())
    return {
        "series": series,
        "games": games,
        "score_rate": points / games,
        "avg_reward_diff": reward_sum / games,
        "avg_ship_diff": ship_sum / games,
    }


def compose_generation_record(
    generation: int,
    genome: GenomeConfig,
    fixed: Dict[str, Any],
    self_play: Dict[str, float],
    mutant_pool: Dict[str, Any],
) -> Dict[str, Any]:
    objective_2p = weighted_average([
        (fixed["fixed_two_player_score"], 0.40),
        (self_play["score_rate"], 0.40 if self_play["games"] > 0 else 0.0),
        (mutant_pool["score_rate"], 0.20 if mutant_pool["games"] > 0 else 0.0),
    ])
    four_player_score = fixed["four_player"]["score"]
    balanced_score = weighted_average([
        (objective_2p, 0.55),
        (four_player_score, 0.45 if fixed["four_player"]["games"] > 0 else 0.0),
    ])
    return {
        "generation": generation,
        "genome": fixed["genome"],
        "slug": fixed["slug"],
        "fixed_two_player": fixed["fixed_two_player"],
        "fixed_two_player_score": fixed["fixed_two_player_score"],
        "self_play": self_play,
        "mutant_pool": mutant_pool,
        "objective_2p": objective_2p,
        "four_player": fixed["four_player"],
        "balanced_score": balanced_score,
        "combined_score": balanced_score,
    }


def dominates(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_pair = (left["objective_2p"], left["four_player"]["score"])
    right_pair = (right["objective_2p"], right["four_player"]["score"])
    return (
        left_pair[0] >= right_pair[0]
        and left_pair[1] >= right_pair[1]
        and (left_pair[0] > right_pair[0] or left_pair[1] > right_pair[1])
    )


def pareto_front(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    front: List[Dict[str, Any]] = []
    for candidate in results:
        if any(dominates(other, candidate) for other in results if other is not candidate):
            continue
        front.append(candidate)
    return sorted(
        front,
        key=lambda item: (item["balanced_score"], item["objective_2p"], item["four_player"]["score"]),
        reverse=True,
    )


def tournament_select(
    scored_population: Sequence[Tuple[GenomeConfig, Dict[str, Any]]],
    rng: random.Random,
    size: int = 3,
) -> GenomeConfig:
    contenders = rng.sample(list(scored_population), k=min(size, len(scored_population)))
    contenders.sort(
        key=lambda item: (
            item[1]["balanced_score"],
            item[1]["objective_2p"],
            item[1]["four_player"]["score"],
        ),
        reverse=True,
    )
    return contenders[0][0]


def build_covering_population(population_size: int, rng: random.Random) -> List[GenomeConfig]:
    population: List[GenomeConfig] = list(PRESET_GENOMES.values())
    seen = set(population)
    gene_names = list(GENE_SPACE)
    offsets = {name: rng.randrange(len(GENE_SPACE[name])) for name in gene_names}

    idx = 0
    while len(population) < population_size and idx < population_size * 24:
        payload: Dict[str, str] = {}
        for gene_index, name in enumerate(gene_names):
            options = GENE_SPACE[name]
            stride = 1 if len(options) == 1 else 1 + ((2 * gene_index) % len(options))
            while math.gcd(stride, len(options)) != 1:
                stride += 1
            choice_index = (offsets[name] + idx * stride + gene_index) % len(options)
            payload[name] = options[choice_index]
        genome = GenomeConfig(**payload)
        if genome not in seen:
            population.append(genome)
            seen.add(genome)
        idx += 1

    while len(population) < population_size:
        genome = random_genome(rng)
        if genome in seen:
            continue
        population.append(genome)
        seen.add(genome)
    return population[:population_size]


def select_generation_elites(
    scored: Sequence[Tuple[GenomeConfig, Dict[str, Any]]],
    elite_count: int,
) -> List[GenomeConfig]:
    if not scored:
        return []
    sorted_balanced = sorted(
        scored,
        key=lambda item: (item[1]["balanced_score"], item[1]["objective_2p"], item[1]["four_player"]["score"]),
        reverse=True,
    )
    best_2p = max(scored, key=lambda item: item[1]["objective_2p"])
    best_4p = max(scored, key=lambda item: item[1]["four_player"]["score"])
    best_aggressive = max(
        (item for item in scored if item[0].style_profile == "aggressive"),
        key=lambda item: (item[1]["balanced_score"], item[1]["objective_2p"], item[1]["four_player"]["score"]),
        default=None,
    )
    best_conservative = max(
        (item for item in scored if item[0].style_profile == "conservative"),
        key=lambda item: (item[1]["balanced_score"], item[1]["objective_2p"], item[1]["four_player"]["score"]),
        default=None,
    )
    front_records = pareto_front([result for _genome, result in scored])
    front_slugs = {record["slug"] for record in front_records}

    elites: List[GenomeConfig] = []
    seen: set[GenomeConfig] = set()
    seed_candidates: List[Optional[Tuple[GenomeConfig, Dict[str, Any]]]] = [
        sorted_balanced[0],
        best_2p,
        best_4p,
        best_aggressive,
        best_conservative,
    ]
    for candidate in seed_candidates:
        if candidate is None:
            continue
        genome, _result = candidate
        if genome not in seen:
            elites.append(genome)
            seen.add(genome)

    for genome, result in sorted_balanced:
        if len(elites) >= elite_count:
            break
        if result["slug"] not in front_slugs or genome in seen:
            continue
        elites.append(genome)
        seen.add(genome)

    for genome, _result in sorted_balanced:
        if len(elites) >= elite_count:
            break
        if genome in seen:
            continue
        elites.append(genome)
        seen.add(genome)
    return elites


def best_record_by_slug(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for record in records:
        existing = best.get(record["slug"])
        if existing is None or (
            record["balanced_score"],
            record["objective_2p"],
            record["four_player"]["score"],
        ) > (
            existing["balanced_score"],
            existing["objective_2p"],
            existing["four_player"]["score"],
        ):
            best[record["slug"]] = record
    return list(best.values())


def record_to_genome(record: Dict[str, Any]) -> GenomeConfig:
    return GenomeConfig.from_dict(record["genome"])


def fixed_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "genome": record["genome"],
        "slug": record["slug"],
        "fixed_two_player": record["fixed_two_player"],
        "fixed_two_player_score": record["fixed_two_player_score"],
        "four_player": record["four_player"],
    }


def load_resume_records(jsonl_path: Path) -> Dict[str, Any]:
    if not jsonl_path.exists():
        return {
            "complete": {},
            "partial_genomes": {},
            "partial_self_play": {},
            "current_progress": None,
        }
    records: Dict[Tuple[int, str], Dict[str, Any]] = {}
    partial_genomes: Dict[Tuple[int, str], Dict[str, Any]] = {}
    partial_self_play: Dict[int, Dict[Tuple[str, str], Dict[str, float]]] = {}
    current_progress: Optional[Dict[str, Any]] = None
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_type = record.get("record_type", "complete")
            if record_type == "complete":
                key = (int(record["generation"]), str(record["slug"]))
                records[key] = record
                partial_genomes.pop(key, None)
                continue
            if record_type != "partial":
                continue
            current_progress = record.get("progress")
            phase = record.get("phase")
            generation = int(record["generation"])
            if phase == "self_play":
                pair_key = tuple(sorted((str(record["left_slug"]), str(record["right_slug"]))))
                partial_self_play.setdefault(generation, {})[pair_key] = record["result"]
                continue
            slug = str(record["slug"])
            genome_key = (generation, slug)
            state = partial_genomes.setdefault(
                genome_key,
                {
                    "fixed_two_player": {},
                    "mutant_pool": {},
                    "four_player_seats": {},
                },
            )
            if phase == "fixed_two_player":
                state["fixed_two_player"][str(record["opponent"])] = record["result"]
            elif phase == "mutant_pool":
                state["mutant_pool"][str(record["opponent"])] = record["result"]
            elif phase == "four_player":
                state["four_player_seats"][int(record["seat"])] = {
                    "seat": int(record["seat"]),
                    "order": record["order"],
                    "result": record["result"],
                }
    return {
        "complete": records,
        "partial_genomes": partial_genomes,
        "partial_self_play": partial_self_play,
        "current_progress": current_progress,
    }


def load_resume_summary(summary_path: Path) -> Dict[str, Any]:
    if not summary_path.exists():
        return {
            "generation_summaries": [],
            "current_progress": None,
        }
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "generation_summaries": payload.get("generation_summaries", []),
        "current_progress": payload.get("current_progress"),
    }


def build_summary_payload(
    args: argparse.Namespace,
    generation_summaries: Sequence[Dict[str, Any]],
    all_records: Sequence[Dict[str, Any]],
    current_progress: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    unique_records = best_record_by_slug(all_records)
    unique_records.sort(
        key=lambda item: (item["balanced_score"], item["objective_2p"], item["four_player"]["score"]),
        reverse=True,
    )
    if unique_records:
        overall_balanced = max(unique_records, key=lambda item: item["balanced_score"])
        overall_2p = max(unique_records, key=lambda item: item["objective_2p"])
        overall_4p = max(unique_records, key=lambda item: item["four_player"]["score"])
        overall_front = pareto_front(unique_records)
    else:
        overall_balanced = None
        overall_2p = None
        overall_4p = None
        overall_front = []

    return {
        "meta": {
            "population": args.population,
            "generations": args.generations,
            "games_per_seat": args.games_per_seat,
            "self_play_games_per_seat": args.self_play_games_per_seat,
            "mutant_games_per_seat": args.mutant_games_per_seat,
            "champion_mutants": args.champion_mutants,
            "seed": args.seed,
            "skip_four_player": args.skip_four_player,
            "two_player_opponents": args.two_player_opponents,
            "four_player_opponents": args.four_player_opponents,
            "resume": args.resume,
            "records_completed": len(all_records),
        },
        "generation_summaries": list(generation_summaries),
        "champions": {
            "balanced": overall_balanced,
            "two_player": overall_2p,
            "four_player": overall_4p,
        },
        "pareto_front": overall_front,
        "top_balanced": unique_records[: max(args.emit_top, 5)],
        "current_progress": current_progress,
    }


def write_summary_file(
    summary_path: Path,
    args: argparse.Namespace,
    generation_summaries: Sequence[Dict[str, Any]],
    all_records: Sequence[Dict[str, Any]],
    current_progress: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = build_summary_payload(args, generation_summaries, all_records, current_progress=current_progress)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def emit_current_wrappers(
    generated_dir: Path,
    summary_payload: Dict[str, Any],
    generation: Optional[int],
    emit_top: int,
) -> Dict[str, Path]:
    champions = summary_payload.get("champions", {})
    pareto = summary_payload.get("pareto_front", [])
    emit_queue: List[Tuple[str, Dict[str, Any]]] = []
    if champions.get("balanced") is not None:
        emit_queue.append(("current-balanced", champions["balanced"]))
    if champions.get("two_player") is not None:
        emit_queue.append(("current-two-player", champions["two_player"]))
    if champions.get("four_player") is not None:
        emit_queue.append(("current-four-player", champions["four_player"]))
    for index, record in enumerate(pareto, start=1):
        emit_queue.append((f"current-pareto-{index}", record))

    emitted: Dict[str, Path] = {}
    for prefix, record in emit_queue:
        if len(emitted) >= emit_top:
            break
        if record is None:
            continue
        genome = record_to_genome(record)
        slug = record["slug"]
        if slug in emitted:
            continue
        path = write_agent_wrapper(genome, generated_dir / f"{prefix}.py")
        emitted[slug] = path
        if generation is not None:
            snap_name = f"generation-{generation:03d}-{prefix}-{slug}.py"
            write_agent_wrapper(genome, generated_dir / snap_name)
    return emitted


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Orbit Wars genome search")
    parser.add_argument("--population", type=int, default=8, help="Population size per generation")
    parser.add_argument("--generations", type=int, default=3, help="Number of generations to run")
    parser.add_argument("--games-per-seat", type=int, default=2, help="Fixed-pool games from each seat per opponent")
    parser.add_argument("--self-play-games-per-seat", type=int, default=1, help="Self-play games from each seat per pairing")
    parser.add_argument("--mutant-games-per-seat", type=int, default=1, help="Games from each seat versus mutated past champions")
    parser.add_argument("--champion-mutants", type=int, default=3, help="How many mutated past champions to include")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--elitism", type=int, default=4, help="Elite genomes to carry forward")
    parser.add_argument("--mutation-rate", type=float, default=0.25, help="Per-gene mutation rate")
    parser.add_argument("--emit-top", type=int, default=4, help="How many wrappers to emit at the end")
    parser.add_argument("--output-dir", type=str, default=str(WORKSPACE_DIR / "results"), help="Directory for jsonl/summary output")
    parser.add_argument("--generated-dir", type=str, default=str(WORKSPACE_DIR / "generated"), help="Directory for emitted wrapper files")
    parser.add_argument("--skip-four-player", action="store_true", help="Disable 4p evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume from genome test/results/search_log.jsonl")
    parser.add_argument(
        "--two-player-opponents",
        nargs="*",
        default=DEFAULT_TWO_PLAYER_OPPONENTS,
        choices=[
            "baseline",
            "oldbase_balanced",
            "oldbase_two_player",
            "release_candidate_v2",
            "s2_4p_antidogpile",
            "stage3_balanced",
            "stage3_two_player",
            "stage3_four_player",
            "v21",
            "v23",
            "v16",
            "mtmr",
            "random",
            "greedy",
            "turtle",
        ],
        help="Subset of the local hall-of-fame opponents to use during 2p evaluation",
    )
    parser.add_argument(
        "--four-player-opponents",
        nargs="*",
        default=DEFAULT_FOUR_PLAYER_OPPONENTS,
        choices=[
            "baseline",
            "oldbase_balanced",
            "oldbase_two_player",
            "release_candidate_v2",
            "s2_4p_antidogpile",
            "stage3_balanced",
            "stage3_two_player",
            "stage3_four_player",
            "v21",
            "v23",
            "v16",
            "mtmr",
            "random",
            "greedy",
            "turtle",
        ],
        help="Exactly 3 opponents to use during the fixed 4p evaluation",
    )
    args = parser.parse_args()
    if not args.skip_four_player and len(args.four_player_opponents) != 3:
        raise SystemExit("--four-player-opponents requires exactly 3 names")

    rng = random.Random(args.seed)
    references = load_reference_agents()
    missing_opponents = sorted({
        name
        for name in [*args.two_player_opponents, *args.four_player_opponents]
        if name not in references
    })
    if missing_opponents:
        raise SystemExit(
            "Missing opponent wrappers: "
            + ", ".join(missing_opponents)
            + ". Finish the stage3 run or emit its current wrappers first."
        )
    output_dir = Path(args.output_dir).resolve()
    generated_dir = Path(args.generated_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "search_log.jsonl"
    summary_path = output_dir / "search_summary.json"
    resume_state = load_resume_records(jsonl_path) if args.resume else {
        "complete": {},
        "partial_genomes": {},
        "partial_self_play": {},
        "current_progress": None,
    }
    resume_summary = load_resume_summary(summary_path) if args.resume else {
        "generation_summaries": [],
        "current_progress": None,
    }

    population = build_covering_population(args.population, rng)
    fixed_cache: Dict[GenomeConfig, Dict[str, Any]] = {}
    all_records: List[Dict[str, Any]] = sorted(
        resume_state["complete"].values(),
        key=lambda item: (int(item["generation"]), str(item["slug"])),
    )
    existing_complete_keys = {
        (int(record["generation"]), str(record["slug"]))
        for record in all_records
    }
    generation_summaries: List[Dict[str, Any]] = list(resume_summary.get("generation_summaries", []))
    champion_history: List[Tuple[str, GenomeConfig]] = []
    current_progress: Optional[Dict[str, Any]] = resume_summary.get("current_progress") or resume_state.get("current_progress")

    for generation_summary in generation_summaries:
        generation_index = int(generation_summary["generation"])
        champion_history.extend([
            (f"g{generation_index}-balanced", record_to_genome(generation_summary["best_balanced"])),
            (f"g{generation_index}-2p", record_to_genome(generation_summary["best_2p"])),
            (f"g{generation_index}-4p", record_to_genome(generation_summary["best_4p"])),
        ])

    def update_progress(progress: Optional[Dict[str, Any]]) -> None:
        nonlocal current_progress
        current_progress = progress
        write_summary_file(summary_path, args, generation_summaries, all_records, current_progress=current_progress)

    def log_partial(payload: Dict[str, Any]) -> None:
        append_jsonl(jsonl_path, payload)
        update_progress(payload.get("progress"))

    write_summary_file(summary_path, args, generation_summaries, all_records, current_progress=current_progress)

    for generation in range(args.generations):
        candidate_agents = {genome: build_agent(genome) for genome in population}
        total_pairs = (len(population) * (len(population) - 1)) // 2
        if args.self_play_games_per_seat > 0 and total_pairs > 0:
            update_progress({
                "generation": generation,
                "phase": "self_play",
                "pair_index": 0,
                "pair_total": total_pairs,
            })
        self_play_stats = run_population_round_robin(
            population,
            candidate_agents,
            args.self_play_games_per_seat,
            existing_pair_results=resume_state["partial_self_play"].get(generation, {}),
            progress_callback=lambda left, right, series, pair_index, total_pairs: log_partial({
                "record_type": "partial",
                "phase": "self_play",
                "generation": generation,
                "left_slug": left.slug(),
                "right_slug": right.slug(),
                "result": series,
                "progress": {
                    "generation": generation,
                    "phase": "self_play",
                    "left_slug": left.slug(),
                    "right_slug": right.slug(),
                    "pair_index": pair_index,
                    "pair_total": total_pairs,
                },
            }),
        )
        mutant_pool = build_champion_mutants(champion_history, rng, args.champion_mutants)
        scored: List[Tuple[GenomeConfig, Dict[str, Any]]] = []

        for index, genome in enumerate(population, start=1):
            resume_key = (generation, genome.slug())
            was_complete = resume_key in existing_complete_keys
            if resume_key in resume_state["complete"]:
                record = resume_state["complete"][resume_key]
                fixed_cache.setdefault(genome, fixed_from_record(record))
            else:
                partial_state = resume_state["partial_genomes"].get(resume_key, {})
                next_fixed_opponent = next(
                    (
                        opponent
                        for opponent in args.two_player_opponents
                        if opponent not in partial_state.get("fixed_two_player", {})
                    ),
                    None,
                )
                if next_fixed_opponent is not None:
                    update_progress({
                        "generation": generation,
                        "phase": "fixed_two_player",
                        "slug": genome.slug(),
                        "genome_index": index,
                        "genome_total": len(population),
                        "opponent": next_fixed_opponent,
                        "index": len(partial_state.get("fixed_two_player", {})),
                        "total": len(args.two_player_opponents),
                    })
                if genome not in fixed_cache:
                    fixed_cache[genome] = evaluate_fixed_pool(
                        genome,
                        candidate_agents[genome],
                        references,
                        two_player_opponents=args.two_player_opponents,
                        four_player_opponents=args.four_player_opponents,
                        games_per_seat=args.games_per_seat,
                        include_four_player=not args.skip_four_player,
                        existing_partial=resume_state["partial_genomes"].get(resume_key),
                        progress_callback=lambda phase, payload, slug=genome.slug(), genome_payload=genome.to_dict(), ordinal=index: log_partial({
                            "record_type": "partial",
                            "phase": phase,
                            "generation": generation,
                            "slug": slug,
                            "genome": genome_payload,
                            **(
                                {"opponent": payload["opponent"], "result": payload["result"]}
                                if phase == "fixed_two_player"
                                else {"seat": payload["seat"], "order": payload["order"], "result": payload["result"]}
                            ),
                            "progress": {
                                "generation": generation,
                                "phase": phase,
                                "slug": slug,
                                "genome_index": ordinal,
                                "genome_total": len(population),
                                **(
                                    {"opponent": payload["opponent"], "index": payload["index"], "total": payload["total"]}
                                    if phase == "fixed_two_player"
                                    else {"seat": payload["seat"] + 1, "total": 4}
                                ),
                            },
                        }),
                    )
                record = compose_generation_record(
                    generation=generation,
                    genome=genome,
                    fixed=fixed_cache[genome],
                    self_play=self_play_stats.get(genome, {
                        "games": 0.0,
                        "score_rate": 0.0,
                        "avg_reward_diff": 0.0,
                        "avg_ship_diff": 0.0,
                    }),
                    mutant_pool=(
                        update_progress({
                            "generation": generation,
                            "phase": "mutant_pool",
                            "slug": genome.slug(),
                            "genome_index": index,
                            "genome_total": len(population),
                            "index": len(partial_state.get("mutant_pool", {})),
                            "total": len(mutant_pool),
                        }) or run_mutant_pool(
                            candidate_agents[genome],
                            mutant_pool,
                            args.mutant_games_per_seat,
                            existing_series=partial_state.get("mutant_pool"),
                            progress_callback=lambda name, payload, slug=genome.slug(), genome_payload=genome.to_dict(), ordinal=index: log_partial({
                                "record_type": "partial",
                                "phase": "mutant_pool",
                                "generation": generation,
                                "slug": slug,
                                "genome": genome_payload,
                                "opponent": name,
                                "result": payload["result"],
                                "progress": {
                                    "generation": generation,
                                    "phase": "mutant_pool",
                                    "slug": slug,
                                    "genome_index": ordinal,
                                    "genome_total": len(population),
                                    "opponent": name,
                                    "index": payload["index"],
                                    "total": payload["total"],
                                },
                            }),
                        )
                    ) if mutant_pool else {
                        "games": 0.0,
                        "score_rate": 0.0,
                        "avg_reward_diff": 0.0,
                        "avg_ship_diff": 0.0,
                        "series": {},
                    },
                )
                record["record_type"] = "complete"
                append_jsonl(jsonl_path, record)
                existing_complete_keys.add(resume_key)
                update_progress(None)
            scored.append((genome, record))
            if not was_complete:
                all_records.append(record)
            write_summary_file(summary_path, args, generation_summaries, all_records, current_progress=current_progress)
            print(
                f"[generation {generation}][{index}/{len(population)}] {record['slug']} "
                f"fixed2={record['fixed_two_player_score']:.3f} "
                f"self={record['self_play']['score_rate']:.3f} "
                f"mut={record['mutant_pool']['score_rate']:.3f} "
                f"4p={record['four_player']['score']:.3f} "
                f"bal={record['balanced_score']:.3f}",
                flush=True,
            )

        scored.sort(
            key=lambda item: (item[1]["balanced_score"], item[1]["objective_2p"], item[1]["four_player"]["score"]),
            reverse=True,
        )
        records = [result for _genome, result in scored]
        front = pareto_front(records)
        best_balanced = scored[0][1]
        best_2p = max(records, key=lambda item: item["objective_2p"])
        best_4p = max(records, key=lambda item: item["four_player"]["score"])
        generation_summaries.append({
            "generation": generation,
            "best_balanced": best_balanced,
            "best_2p": best_2p,
            "best_4p": best_4p,
            "pareto_front": front,
        })
        champion_history.extend([
            (f"g{generation}-balanced", record_to_genome(best_balanced)),
            (f"g{generation}-2p", record_to_genome(best_2p)),
            (f"g{generation}-4p", record_to_genome(best_4p)),
        ])
        print(
            f"[generation {generation}] champions: "
            f"balanced={best_balanced['slug']} "
            f"2p={best_2p['slug']} "
            f"4p={best_4p['slug']} "
            f"pareto={len(front)}",
            flush=True,
        )
        summary_payload = write_summary_file(summary_path, args, generation_summaries, all_records, current_progress=None)
        emitted = emit_current_wrappers(generated_dir, summary_payload, generation=generation, emit_top=args.emit_top)
        for path in emitted.values():
            print(f"[emit] {path.relative_to(ROOT)}", flush=True)

        elites = select_generation_elites(scored, max(1, min(args.elitism, len(scored))))
        next_population = list(elites)
        seen = set(next_population)
        while len(next_population) < args.population:
            parent_a = tournament_select(scored, rng)
            parent_b = tournament_select(scored, rng)
            child = crossover_genomes(parent_a, parent_b, rng)
            child = mutate_genome(child, rng, mutation_rate=args.mutation_rate)
            if child in seen:
                child = mutate_genome(child, rng, mutation_rate=max(args.mutation_rate, 0.40))
            if child in seen:
                continue
            next_population.append(child)
            seen.add(child)
        population = next_population

    summary_payload = write_summary_file(summary_path, args, generation_summaries, all_records, current_progress=None)
    emitted = emit_current_wrappers(generated_dir, summary_payload, generation=None, emit_top=args.emit_top)
    for path in emitted.values():
        print(f"[emit] {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
