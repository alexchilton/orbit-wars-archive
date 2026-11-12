from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent.parent
GENOME_DIR = ROOT / "genome test"
if str(GENOME_DIR) not in sys.path:
    sys.path.insert(0, str(GENOME_DIR))

from genetic_search import emit_current_wrappers  # noqa: E402


def ensure_stage3_wrappers(summary_path: Path, generated_dir: Path, emit_top: int) -> List[str]:
    if not summary_path.exists():
        raise SystemExit(f"Missing stage3 summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    generated_dir.mkdir(parents=True, exist_ok=True)
    emit_current_wrappers(generated_dir, payload, generation=None, emit_top=emit_top)
    names = sorted(path.name for path in generated_dir.glob("current-*.py"))
    if not names:
        raise SystemExit("No current stage3 wrappers could be emitted from the stage3 summary")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a stage4 genome search against current stage3 wrappers")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--games-per-seat", type=int, default=6)
    parser.add_argument("--self-play-games-per-seat", type=int, default=4)
    parser.add_argument("--mutant-games-per-seat", type=int, default=1)
    parser.add_argument("--champion-mutants", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--emit-top", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-four-player", action="store_true")
    parser.add_argument(
        "--results-dir",
        default=str(GENOME_DIR / "results_stage4"),
        help="Output directory for stage4 logs and summary",
    )
    parser.add_argument(
        "--generated-dir",
        default=str(GENOME_DIR / "generated_stage4"),
        help="Directory for emitted stage4 wrappers",
    )
    parser.add_argument(
        "--stage3-summary",
        default=str(GENOME_DIR / "results_stage3" / "search_summary.json"),
        help="Stage3 summary used to emit current champion wrappers",
    )
    parser.add_argument(
        "--stage3-generated-dir",
        default=str(GENOME_DIR / "generated_stage3"),
        help="Directory where current stage3 wrappers are emitted",
    )
    args = parser.parse_args()

    summary_path = Path(args.stage3_summary).resolve()
    stage3_generated_dir = Path(args.stage3_generated_dir).resolve()
    emitted = ensure_stage3_wrappers(summary_path, stage3_generated_dir, emit_top=args.emit_top)
    print(f"Using stage3 wrappers: {', '.join(emitted)}", flush=True)

    env = os.environ.copy()
    env["ORBIT_WARS_BASE_AGENT_PATH"] = str((ROOT / "snapshots" / "stage4_search_base.py").resolve())

    cmd = [
        sys.executable,
        str((GENOME_DIR / "genetic_search.py").resolve()),
        "--population",
        str(args.population),
        "--generations",
        str(args.generations),
        "--games-per-seat",
        str(args.games_per_seat),
        "--self-play-games-per-seat",
        str(args.self_play_games_per_seat),
        "--mutant-games-per-seat",
        str(args.mutant_games_per_seat),
        "--champion-mutants",
        str(args.champion_mutants),
        "--seed",
        str(args.seed),
        "--emit-top",
        str(args.emit_top),
        "--output-dir",
        str(Path(args.results_dir).resolve()),
        "--generated-dir",
        str(Path(args.generated_dir).resolve()),
        "--two-player-opponents",
        "baseline",
        "oldbase_balanced",
        "oldbase_two_player",
        "stage3_balanced",
        "stage3_two_player",
    ]
    if args.skip_four_player:
        cmd.append("--skip-four-player")
    else:
        cmd.extend(["--four-player-opponents", "stage3_balanced", "baseline", "v23"])
    if args.resume:
        cmd.append("--resume")

    print("Launching:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.run(cmd, env=env, check=False).returncode)


if __name__ == "__main__":
    main()
