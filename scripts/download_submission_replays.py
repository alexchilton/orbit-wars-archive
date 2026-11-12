from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from kaggle_episode_tools import build_api, download_replay


def _load_submission_ids(args: argparse.Namespace) -> List[int]:
    submission_ids = list(args.submission_ids or [])
    if args.submission_ids_file:
        for line in Path(args.submission_ids_file).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            submission_ids.append(int(stripped))
    if not submission_ids:
        raise SystemExit("Provide at least one submission ID via --submission-ids or --submission-ids-file")
    return sorted(set(submission_ids))


def _episode_sort_key(episode) -> tuple[str, str]:
    end_time = str(getattr(episode, "end_time", "") or "")
    create_time = str(getattr(episode, "create_time", "") or "")
    return (end_time, create_time)


def _pick_episodes(episodes: Iterable[object], limit: int) -> List[object]:
    filtered = []
    for episode in episodes:
        state = str(getattr(episode, "state", "") or "").lower()
        if state and not any(token in state for token in {"complete", "completed", "finished"}):
            continue
        filtered.append(episode)
    filtered.sort(key=_episode_sort_key, reverse=True)
    return filtered[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download recent Kaggle replays for a bundle of submission IDs")
    parser.add_argument("--submission-ids", type=int, nargs="*", default=None)
    parser.add_argument("--submission-ids-file")
    parser.add_argument("--episodes-per-submission", type=int, default=5)
    parser.add_argument("--output-dir", default="kaggle_replays/submission_bundle")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission_ids = _load_submission_ids(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    api = build_api()
    manifest = {"submission_ids": submission_ids, "episodes_per_submission": args.episodes_per_submission, "downloads": []}

    for submission_id in submission_ids:
        episodes = api.competition_list_episodes(submission_id)
        selected = _pick_episodes(episodes, args.episodes_per_submission)
        submission_dir = output_dir / f"submission-{submission_id}"
        submission_dir.mkdir(parents=True, exist_ok=True)

        download_rows = []
        for episode in selected:
            episode_id = int(getattr(episode, "id"))
            download_replay(episode_id, path=str(submission_dir), quiet=args.quiet)
            download_rows.append(
                {
                    "episode_id": episode_id,
                    "state": str(getattr(episode, "state", "") or ""),
                    "type": str(getattr(episode, "type", "") or ""),
                    "create_time": str(getattr(episode, "create_time", "") or ""),
                    "end_time": str(getattr(episode, "end_time", "") or ""),
                }
            )

        manifest["downloads"].append({"submission_id": submission_id, "episodes": download_rows})
        if not args.quiet:
            print(f"submission {submission_id}: downloaded {len(download_rows)} replays -> {submission_dir}")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if not args.quiet:
        print(f"Saved manifest -> {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
