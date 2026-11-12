from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _summarize_eval(payload: dict[str, Any]) -> dict[str, Any]:
    evaluation = payload.get("evaluation") or []
    improved = same = worse = 0
    heuristic_sum = 0.0
    rl_sum = 0.0
    for item in evaluation:
        heuristic = float(item["heuristic"]["reward_signal"])
        rl = float(item["rl"]["reward_signal"])
        heuristic_sum += heuristic
        rl_sum += rl
        diff = rl - heuristic
        if diff > 1e-9:
            improved += 1
        elif diff < -1e-9:
            worse += 1
        else:
            same += 1
    count = len(evaluation)
    avg_heuristic = heuristic_sum / count if count else None
    avg_rl = rl_sum / count if count else None
    avg_diff = (rl_sum - heuristic_sum) / count if count else None
    return {
        "count": count,
        "improved": improved,
        "same": same,
        "worse": worse,
        "avg_heuristic": avg_heuristic,
        "avg_rl": avg_rl,
        "avg_diff": avg_diff,
    }


def _print_stage3() -> None:
    path = ROOT / "genome test" / "results_stage3" / "search_summary.json"
    payload = _load_json(path)
    print("== Stage3 search ==")
    if payload is None:
        print("missing")
        return
    meta = payload.get("meta") or {}
    progress = payload.get("current_progress") or {}
    champions = payload.get("champions") or {}
    print(
        f"population={meta.get('population')} generations={meta.get('generations')} "
        f"records_completed={meta.get('records_completed')}"
    )
    if progress:
        parts = [
            f"phase={progress.get('phase')}",
            f"generation={progress.get('generation')}",
        ]
        if "pair_index" in progress and "pair_total" in progress:
            parts.append(f"pair={progress.get('pair_index')}/{progress.get('pair_total')}")
        if "genome_index" in progress and "genome_total" in progress:
            parts.append(f"genome={progress.get('genome_index')}/{progress.get('genome_total')}")
        if progress.get("opponent"):
            parts.append(f"opponent={progress.get('opponent')}")
        if progress.get("slug"):
            parts.append(f"slug={progress.get('slug')}")
        print("progress:", " ".join(parts))
    balanced = champions.get("balanced")
    two_player = champions.get("two_player")
    four_player = champions.get("four_player")
    if balanced or two_player or four_player:
        print(
            "champions:",
            f"balanced={balanced and balanced.get('slug')}",
            f"two_player={two_player and two_player.get('slug')}",
            f"four_player={four_player and four_player.get('slug')}",
        )
    else:
        print("champions: none yet")


def _print_rl(title: str, relative_path: str) -> None:
    path = ROOT / relative_path
    payload = _load_json(path)
    print(f"== {title} ==")
    if payload is None:
        print("missing")
        return
    summary = _summarize_eval(payload)
    if "candidate_count" in payload:
        print(f"candidate_count={payload.get('candidate_count')}")
    elif "candidates" in payload:
        print(f"candidate_count={len(payload.get('candidates') or [])}")
    if "template_count" in payload:
        print(f"template_count={payload.get('template_count')}")
    config = payload.get("config") or {}
    if "train_episodes" in config:
        print(f"train_episodes={config.get('train_episodes')}")
    print(
        f"evaluation={summary['count']} improved={summary['improved']} "
        f"same={summary['same']} worse={summary['worse']}"
    )
    print(
        f"avg_heuristic={_fmt_float(summary['avg_heuristic'])} "
        f"avg_rl={_fmt_float(summary['avg_rl'])} "
        f"avg_diff={_fmt_float(summary['avg_diff'])}"
    )


def show_status() -> None:
    _print_stage3()
    print()
    _print_rl("Replay RL stage4", "rl midgame/results/replay_midgame_summary_stage4.json")
    print()
    _print_rl("Synthetic opening RL", "rl midgame/results/synthetic_opening_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show concise Orbit Wars search/RL status")
    parser.add_argument("--repeat", type=float, default=0.0, help="Refresh every N seconds")
    args = parser.parse_args()

    if args.repeat and args.repeat > 0:
        while True:
            print("\033[2J\033[H", end="")
            show_status()
            time.sleep(args.repeat)
    else:
        show_status()


if __name__ == "__main__":
    main()
