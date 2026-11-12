from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from kaggle.api.kaggle_api_extended import KaggleApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze opening-position patterns in downloaded Kaggle replays")
    parser.add_argument(
        "--replay-dir",
        default="kaggle_replays/own_rl_stage2_bundle",
        help="Directory containing submission-<id>/episode-*-replay.json files",
    )
    parser.add_argument(
        "--submission-ids",
        type=int,
        nargs="*",
        default=None,
        help="Submission IDs to analyze. Defaults to IDs found under replay-dir/submission-*",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for the JSON summary",
    )
    return parser.parse_args()


def _dist(planet_a: Sequence[float], planet_b: Sequence[float]) -> float:
    return math.hypot(float(planet_a[2]) - float(planet_b[2]), float(planet_a[3]) - float(planet_b[3]))


def _bucket_thresholds(values: Iterable[float]) -> List[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return [0.0, 0.0]
    lower_index = max(0, min(len(ordered) - 1, len(ordered) // 3))
    upper_index = max(0, min(len(ordered) - 1, (2 * len(ordered)) // 3))
    return [ordered[lower_index], ordered[upper_index]]


def _bucket_label(value: float, thresholds: Sequence[float], labels: Sequence[str]) -> str:
    if value <= thresholds[0]:
        return labels[0]
    if value <= thresholds[1]:
        return labels[1]
    return labels[2]


def _load_submission_ids(replay_dir: Path, submission_ids: Sequence[int] | None) -> List[int]:
    if submission_ids:
        return sorted(set(int(submission_id) for submission_id in submission_ids))
    found = []
    for child in replay_dir.glob("submission-*"):
        suffix = child.name.removeprefix("submission-")
        if suffix.isdigit():
            found.append(int(suffix))
    return sorted(set(found))


def _build_episode_map(submission_ids: Sequence[int]) -> Dict[int, dict]:
    api = KaggleApi()
    api.authenticate()
    episode_map: Dict[int, dict] = {}
    for submission_id in submission_ids:
        for episode in api.competition_list_episodes(submission_id):
            agent_rows = []
            for agent in episode.agents:
                agent_rows.append(
                    {
                        "submission_id": int(agent.submission_id),
                        "index": int(agent.index),
                        "reward": float(agent.reward),
                        "team_name": str(agent.team_name),
                    }
                )
            episode_map[int(episode.id)] = {
                "submission_id": submission_id,
                "agents": agent_rows,
            }
    return episode_map


def _extract_opening_metrics(replay_data: dict, player_index: int) -> dict:
    observation = replay_data["steps"][0][player_index]["observation"]
    planets = observation["planets"]
    home_planets = [planet for planet in planets if int(planet[1]) >= 0]
    my_home = next(planet for planet in home_planets if int(planet[1]) == player_index)
    enemy_homes = [planet for planet in home_planets if int(planet[1]) != player_index]
    neutrals = [planet for planet in planets if int(planet[1]) == -1]

    scored_targets = []
    for neutral in neutrals:
        my_distance = _dist(my_home, neutral)
        enemy_distance = min(_dist(enemy_home, neutral) for enemy_home in enemy_homes) if enemy_homes else 999.0
        distance_margin = enemy_distance - my_distance
        production = float(neutral[4])
        garrison = max(int(neutral[5]), 1)
        value_score = (production / garrison) / (my_distance + 1.0)
        scored_targets.append(
            {
                "planet_id": int(neutral[0]),
                "distance": my_distance,
                "enemy_distance": enemy_distance,
                "margin": distance_margin,
                "production": production,
                "garrison": garrison,
                "value_score": value_score,
            }
        )

    nearest_neutral = min(scored_targets, key=lambda row: row["distance"])
    rich_targets = [target for target in scored_targets if target["production"] > 1.5]
    best_target = max(scored_targets, key=lambda row: (row["value_score"], row["margin"], -row["distance"]))

    return {
        "planet_count": len(planets),
        "nearest_neutral_dist": nearest_neutral["distance"],
        "nearest_neutral_prod": nearest_neutral["production"],
        "nearest_rich_dist": min((target["distance"] for target in rich_targets), default=None),
        "closest_enemy_home_dist": min(_dist(my_home, enemy_home) for enemy_home in enemy_homes) if enemy_homes else None,
        "best_target_dist": best_target["distance"],
        "best_target_margin": best_target["margin"],
        "best_target_prod": best_target["production"],
        "best_target_garrison": best_target["garrison"],
        "best_target_value_score": best_target["value_score"],
        "safe_close_count": sum(1 for target in scored_targets if target["margin"] > 5.0 and target["distance"] < 40.0),
        "safe_rich_count": sum(1 for target in rich_targets if target["margin"] > 0.0),
        "home_center_dist": math.hypot(float(my_home[2]) - 50.0, float(my_home[3]) - 50.0),
    }


def _row_opening_family(row: dict, thresholds_by_mode: dict) -> str:
    thresholds = thresholds_by_mode[row["players"]]
    if row["players"] == 2:
        dist_bucket = _bucket_label(row["best_target_dist"], thresholds["best_target_dist"], ["short", "mid", "long"])
        margin_bucket = _bucket_label(row["best_target_margin"], thresholds["best_target_margin"], ["tight", "mid", "safe"])
        return f"2p-{dist_bucket}-{margin_bucket}"
    enemy_bucket = _bucket_label(
        row["closest_enemy_home_dist"],
        thresholds["closest_enemy_home_dist"],
        ["crowded", "mid", "spaced"],
    )
    margin_bucket = _bucket_label(row["best_target_margin"], thresholds["best_target_margin"], ["tight", "mid", "safe"])
    return f"4p-{enemy_bucket}-{margin_bucket}"


def _summarize_rows(rows: List[dict]) -> dict:
    thresholds_by_mode = {}
    for players in sorted({row["players"] for row in rows}):
        mode_rows = [row for row in rows if row["players"] == players]
        thresholds_by_mode[players] = {
            "best_target_dist": _bucket_thresholds(row["best_target_dist"] for row in mode_rows),
            "best_target_margin": _bucket_thresholds(row["best_target_margin"] for row in mode_rows),
            "closest_enemy_home_dist": _bucket_thresholds(
                row["closest_enemy_home_dist"] for row in mode_rows if row["closest_enemy_home_dist"] is not None
            ),
        }

    for row in rows:
        row["opening_family"] = _row_opening_family(row, thresholds_by_mode)

    summary = {
        "thresholds_by_mode": thresholds_by_mode,
        "submissions": {},
    }

    for submission_id in sorted({row["submission_id"] for row in rows}):
        submission_rows = [row for row in rows if row["submission_id"] == submission_id]
        submission_summary = {
            "episodes": len(submission_rows),
            "modes": {},
        }
        for players in sorted({row["players"] for row in submission_rows}):
            mode_rows = [row for row in submission_rows if row["players"] == players]
            wins = [row for row in mode_rows if row["is_unique_win"]]
            mode_summary = {
                "episodes": len(mode_rows),
                "win_rate": len(wins) / len(mode_rows) if mode_rows else 0.0,
                "avg_rank": mean(row["rank"] for row in mode_rows) if mode_rows else 0.0,
                "seat_summary": {},
                "opening_families": {},
                "feature_means": {
                    "wins": {},
                    "losses": {},
                },
            }
            for seat in sorted({row["seat"] for row in mode_rows}):
                seat_rows = [row for row in mode_rows if row["seat"] == seat]
                seat_wins = [row for row in seat_rows if row["is_unique_win"]]
                mode_summary["seat_summary"][str(seat)] = {
                    "episodes": len(seat_rows),
                    "wins": len(seat_wins),
                    "win_rate": len(seat_wins) / len(seat_rows) if seat_rows else 0.0,
                    "avg_rank": mean(row["rank"] for row in seat_rows) if seat_rows else 0.0,
                }
            for family in sorted({row["opening_family"] for row in mode_rows}):
                family_rows = [row for row in mode_rows if row["opening_family"] == family]
                family_wins = [row for row in family_rows if row["is_unique_win"]]
                mode_summary["opening_families"][family] = {
                    "episodes": len(family_rows),
                    "wins": len(family_wins),
                    "win_rate": len(family_wins) / len(family_rows) if family_rows else 0.0,
                    "seeds": [row["seed"] for row in family_rows[:10]],
                }

            for feature_name in [
                "nearest_neutral_dist",
                "nearest_rich_dist",
                "closest_enemy_home_dist",
                "best_target_dist",
                "best_target_margin",
                "best_target_value_score",
                "safe_close_count",
                "safe_rich_count",
            ]:
                win_values = [row[feature_name] for row in wins if row[feature_name] is not None]
                loss_values = [row[feature_name] for row in mode_rows if not row["is_unique_win"] and row[feature_name] is not None]
                if win_values:
                    mode_summary["feature_means"]["wins"][feature_name] = mean(win_values)
                if loss_values:
                    mode_summary["feature_means"]["losses"][feature_name] = mean(loss_values)
            submission_summary["modes"][str(players)] = mode_summary
        summary["submissions"][str(submission_id)] = submission_summary
    return summary


def _load_rows(replay_dir: Path, submission_ids: Sequence[int]) -> List[dict]:
    episode_map = _build_episode_map(submission_ids)
    rows: List[dict] = []

    for submission_id in submission_ids:
        submission_dir = replay_dir / f"submission-{submission_id}"
        for replay_path in sorted(submission_dir.glob("episode-*-replay.json")):
            episode_id = int(replay_path.stem.split("-")[1])
            episode_meta = episode_map.get(episode_id)
            if not episode_meta:
                continue
            replay_data = json.loads(replay_path.read_text(encoding="utf-8"))
            player_meta = next(
                agent
                for agent in episode_meta["agents"]
                if int(agent["submission_id"]) == submission_id
            )
            player_index = int(player_meta["index"])
            rewards = [float(value) for value in replay_data["rewards"]]
            reward = rewards[player_index]
            opponents = [agent for agent in episode_meta["agents"] if int(agent["submission_id"]) != submission_id]
            row = {
                "submission_id": submission_id,
                "episode_id": episode_id,
                "players": len(rewards),
                "seat": player_index,
                "seed": int(replay_data["info"]["seed"]),
                "reward": reward,
                "rank": 1 + sum(1 for other_reward in rewards if other_reward > reward),
                "is_unique_win": reward == max(rewards) and rewards.count(reward) == 1,
                "opponent_submission_ids": [int(agent["submission_id"]) for agent in opponents],
                "opponent_team_names": [str(agent["team_name"]) for agent in opponents],
            }
            row.update(_extract_opening_metrics(replay_data, player_index))
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    replay_dir = Path(args.replay_dir).resolve()
    submission_ids = _load_submission_ids(replay_dir, args.submission_ids)
    rows = _load_rows(replay_dir, submission_ids)
    summary = _summarize_rows(rows)

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
