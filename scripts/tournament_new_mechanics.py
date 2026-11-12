#!/usr/bin/env python3
"""Tournament benchmark for 6 new Orbit Wars variant mechanics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

os.environ.setdefault("KAGGLE_ENVIRONMENTS_QUIET", "1")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments import make  # noqa: E402

SHUNLITE_PATH = ROOT / "submission" / "main_fc_rl_shunlite.py"
PLUS_2P_PATH = ROOT / "submission" / "main_v131_plus_2p.py"
PLUS_4P_PATH = ROOT / "submission" / "main_v131_plus_4p.py"

# New mechanic variant paths (v131_plus variants)
WAVE_PATH = ROOT / "submission" / "main_v131_plus_wave.py"
INTERCEPT_PATH = ROOT / "submission" / "main_v131_plus_intercept.py"
EVAC_PATH = ROOT / "submission" / "main_v131_plus_evac.py"
DENIAL_PATH = ROOT / "submission" / "main_v131_plus_denial.py"
POLITICAL_PATH = ROOT / "submission" / "main_v131_plus_4p_political.py"
PRODEXPAND_PATH = ROOT / "submission" / "main_v131_plus_prodexpand.py"

# New shunlite-based variant paths
SHUN_WAVE_PATH = ROOT / "submission" / "main_shunlite_wave.py"
SHUN_INTERCEPT_PATH = ROOT / "submission" / "main_shunlite_intercept.py"
SHUN_EVAC_PATH = ROOT / "submission" / "main_shunlite_evac.py"
SHUN_DENIAL_PATH = ROOT / "submission" / "main_shunlite_denial.py"
SHUN_PRODEXPAND_PATH = ROOT / "submission" / "main_shunlite_prodexpand.py"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_agent(path: Path, label: str) -> Callable[..., Any]:
    spec = importlib.util.spec_from_file_location(f"_new_mechanic_{label}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.agent


def run_2p_game(agent_a: Callable[..., Any], agent_b: Callable[..., Any]) -> Dict[str, Any]:
    env = make("orbit_wars", debug=False)
    env.run([agent_a, agent_b])
    final = env.steps[-1]
    reward_a = final[0].reward or 0
    reward_b = final[1].reward or 0
    planets = final[-1].observation.get("planets") or []
    ships_a = sum(p[5] for p in planets if p[1] == 0)
    ships_b = sum(p[5] for p in planets if p[1] == 1)
    return {
        "reward_a": reward_a,
        "reward_b": reward_b,
        "ships_a": ships_a,
        "ships_b": ships_b,
    }


def run_4p_game(agent_list: List[Callable[..., Any]]) -> Dict[str, Any]:
    env = make("orbit_wars", debug=False)
    env.run(agent_list)
    final = env.steps[-1]
    rewards = [player.reward or 0 for player in final]
    planets = final[-1].observation.get("planets") or []
    ships = [sum(p[5] for p in planets if p[1] == idx) for idx in range(len(agent_list))]
    ranks = [1 + sum(1 for other in rewards if other > reward) for reward in rewards]
    return {"rewards": rewards, "ships": ships, "ranks": ranks}


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported type for JSON serialization: {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tournament for 6 new Orbit Wars mechanics variants")
    parser.add_argument("--games-2p", type=int, default=20, help="Games per 2p matchup (across both seats)")
    parser.add_argument("--games-4p", type=int, default=12, help="Games per 4p lineup")
    parser.add_argument("--log-jsonl", default="", help="Optional per-game JSONL log path")
    args = parser.parse_args()

    log_dir = ROOT / "benchmark_logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_path = Path(args.log_jsonl) if args.log_jsonl else log_dir / f"new-mechanics-{stamp}.jsonl"

    print("Loading agents...", flush=True)

    agents: Dict[str, Callable[..., Any]] = {}
    load_specs = [
        ("shunlite", SHUNLITE_PATH),
        ("shunlite_b", SHUNLITE_PATH),   # second copy for 4p filler
        ("shunlite_c", SHUNLITE_PATH),   # third copy for 4p filler
        ("plus2p", PLUS_2P_PATH),
        ("plus4p", PLUS_4P_PATH),
        ("wave", WAVE_PATH),
        ("intercept", INTERCEPT_PATH),
        ("evac", EVAC_PATH),
        ("denial", DENIAL_PATH),
        ("political", POLITICAL_PATH),
        ("prodexpand", PRODEXPAND_PATH),
        # Shunlite-based variants
        ("shun_wave", SHUN_WAVE_PATH),
        ("shun_intercept", SHUN_INTERCEPT_PATH),
        ("shun_evac", SHUN_EVAC_PATH),
        ("shun_denial", SHUN_DENIAL_PATH),
        ("shun_prodexpand", SHUN_PRODEXPAND_PATH),
    ]
    for label, path in load_specs:
        print(f"  Loading {label} from {path.name}...", end=" ", flush=True)
        agents[label] = load_agent(path, label)
        print("OK", flush=True)

    # ─── 2-Player matchups ────────────────────────────────────────────────────
    # Each 2p variant (wave, intercept, evac, denial, prodexpand) races against:
    #   - plus2p  (strongest existing 2p baseline)
    #   - shunlite (strong public baseline)
    two_player_matchups = [
        # (matchup_name, variant_label, opponent_label)
        # v131_plus variants (evac & prodexpand now fixed in-place)
        ("wave_vs_plus2p",      "wave",       "plus2p"),
        ("wave_vs_shunlite",    "wave",       "shunlite"),
        ("intercept_vs_plus2p", "intercept",  "plus2p"),
        ("intercept_vs_shunlite","intercept", "shunlite"),
        ("evac_vs_plus2p",      "evac",       "plus2p"),
        ("evac_vs_shunlite",    "evac",       "shunlite"),
        ("denial_vs_plus2p",    "denial",     "plus2p"),
        ("denial_vs_shunlite",  "denial",     "shunlite"),
        ("prodexpand_vs_plus2p","prodexpand",  "plus2p"),
        ("prodexpand_vs_shunlite","prodexpand","shunlite"),
        # Shunlite-based variants vs plus2p and vs shunlite
        ("shun_wave_vs_plus2p",       "shun_wave",       "plus2p"),
        ("shun_wave_vs_shunlite",     "shun_wave",       "shunlite"),
        ("shun_intercept_vs_plus2p",  "shun_intercept",  "plus2p"),
        ("shun_intercept_vs_shunlite","shun_intercept",  "shunlite"),
        ("shun_evac_vs_plus2p",       "shun_evac",       "plus2p"),
        ("shun_evac_vs_shunlite",     "shun_evac",       "shunlite"),
        ("shun_denial_vs_plus2p",     "shun_denial",     "plus2p"),
        ("shun_denial_vs_shunlite",   "shun_denial",     "shunlite"),
        ("shun_prodexpand_vs_plus2p", "shun_prodexpand", "plus2p"),
        ("shun_prodexpand_vs_shunlite","shun_prodexpand","shunlite"),
    ]

    # ─── 4-Player lineups ─────────────────────────────────────────────────────
    # political vs plus4p baseline + two shunlite fillers
    four_player_lineups = [
        ("political_vs_plus4p", ["political", "plus4p", "shunlite", "shunlite_b"]),
        ("political_vs_triple_shunlite", ["political", "shunlite", "shunlite_b", "shunlite_c"]),
        ("plus4p_vs_triple_shunlite", ["plus4p", "shunlite", "shunlite_b", "shunlite_c"]),
    ]

    summary: Dict[str, Any] = {
        "started_at": now_iso(),
        "log_path": str(log_path),
        "games_2p": args.games_2p,
        "games_4p": args.games_4p,
        "two_player": {},
        "four_player": {},
    }

    with log_path.open("a", encoding="utf-8") as handle:
        # ── 2-player loop ─────────────────────────────────────────────────────
        for matchup_name, variant_label, opp_label in two_player_matchups:
            stats: Dict[str, Any] = {"games": 0, "wins": 0, "losses": 0, "draws": 0,
                                     "reward_diff_sum": 0.0, "ship_diff_sum": 0.0}
            print(f"\n[2P] {matchup_name}  ({args.games_2p} games)", flush=True)
            t0 = time.monotonic()

            for game in range(args.games_2p):
                seat = game % 2
                if seat == 0:
                    result = run_2p_game(agents[variant_label], agents[opp_label])
                    reward_diff = result["reward_a"] - result["reward_b"]
                    ship_diff = result["ships_a"] - result["ships_b"]
                else:
                    result = run_2p_game(agents[opp_label], agents[variant_label])
                    reward_diff = result["reward_b"] - result["reward_a"]
                    ship_diff = result["ships_b"] - result["ships_a"]

                stats["games"] += 1
                stats["reward_diff_sum"] += reward_diff
                stats["ship_diff_sum"] += ship_diff
                if reward_diff > 0:
                    stats["wins"] += 1
                    outcome = "WIN"
                elif reward_diff < 0:
                    stats["losses"] += 1
                    outcome = "LOSS"
                else:
                    stats["draws"] += 1
                    outcome = "DRAW"

                payload = {
                    "ts": now_iso(),
                    "mode": "2p",
                    "matchup": matchup_name,
                    "game": game + 1,
                    "seat": seat,
                    "variant": variant_label,
                    "opponent": opp_label,
                    "outcome": outcome,
                    "reward_diff": reward_diff,
                    "ship_diff": ship_diff,
                }
                handle.write(json.dumps(payload, default=json_default, sort_keys=True) + "\n")
                handle.flush()

                score_rate = (stats["wins"] + 0.5 * stats["draws"]) / stats["games"]
                elapsed = time.monotonic() - t0
                print(
                    f"  Game {game + 1:02d}/{args.games_2p}: {outcome:4s}  "
                    f"rdiff={reward_diff:+.0f}  sdiff={ship_diff:+.0f}  "
                    f"score%={100.0 * score_rate:5.1f}  elapsed={elapsed:.0f}s",
                    flush=True,
                )

            summary["two_player"][matchup_name] = {
                **stats,
                "variant": variant_label,
                "opponent": opp_label,
                "score_rate": (stats["wins"] + 0.5 * stats["draws"]) / max(1, stats["games"]),
                "avg_reward_diff": stats["reward_diff_sum"] / max(1, stats["games"]),
                "avg_ship_diff": stats["ship_diff_sum"] / max(1, stats["games"]),
            }

        # ── 4-player loop ─────────────────────────────────────────────────────
        for lineup_name, labels in four_player_lineups:
            label_stats: Dict[str, Dict[str, float]] = defaultdict(
                lambda: {"games": 0, "wins": 0, "top2": 0, "reward_sum": 0.0, "rank_sum": 0.0}
            )
            print(f"\n[4P] {lineup_name}  ({args.games_4p} games)", flush=True)
            t0 = time.monotonic()

            for game in range(args.games_4p):
                order = [labels[(idx + game) % len(labels)] for idx in range(len(labels))]
                result = run_4p_game([agents[label] for label in order])
                ranked = sorted(
                    zip(order, result["rewards"], result["ranks"]),
                    key=lambda item: (item[2], -item[1]),
                )
                winner_label = ranked[0][0]

                for seat, label in enumerate(order):
                    reward = result["rewards"][seat]
                    rank = result["ranks"][seat]
                    label_stats[label]["games"] += 1
                    label_stats[label]["reward_sum"] += reward
                    label_stats[label]["rank_sum"] += rank
                    if rank <= 2:
                        label_stats[label]["top2"] += 1
                    if rank == 1:
                        label_stats[label]["wins"] += 1

                payload = {
                    "ts": now_iso(),
                    "mode": "4p",
                    "lineup": lineup_name,
                    "game": game + 1,
                    "order": order,
                    "rewards": result["rewards"],
                    "ranks": result["ranks"],
                    "ships": result["ships"],
                }
                handle.write(json.dumps(payload, default=json_default, sort_keys=True) + "\n")
                handle.flush()

                elapsed = time.monotonic() - t0
                ranked_str = " > ".join(
                    f"{lbl}(r={r:.0f},p={p})" for lbl, r, p in ranked
                )
                print(
                    f"  Game {game + 1:02d}/{args.games_4p}: {ranked_str}  "
                    f"winner={winner_label}  elapsed={elapsed:.0f}s",
                    flush=True,
                )

            summary["four_player"][lineup_name] = {
                label: {
                    **dict(bucket),
                    "first_rate": bucket["wins"] / max(1, bucket["games"]),
                    "top2_rate": bucket["top2"] / max(1, bucket["games"]),
                    "avg_reward": bucket["reward_sum"] / max(1, bucket["games"]),
                    "avg_rank": bucket["rank_sum"] / max(1, bucket["games"]),
                }
                for label, bucket in label_stats.items()
            }

    summary["finished_at"] = now_iso()

    print("\n" + "=" * 70, flush=True)
    print("=== TOURNAMENT SUMMARY ===", flush=True)
    print("=" * 70, flush=True)

    print("\n[2P RESULTS]", flush=True)
    for matchup_name, mstats in summary["two_player"].items():
        variant = mstats["variant"]
        opp = mstats["opponent"]
        sr = mstats["score_rate"]
        w, l, d, g = mstats["wins"], mstats["losses"], mstats["draws"], mstats["games"]
        avg_r = mstats["avg_reward_diff"]
        print(
            f"  {variant:15s} vs {opp:12s}: {w}W/{l}L/{d}D ({g}g)  "
            f"score%={100.0*sr:5.1f}  avg_rdiff={avg_r:+.1f}",
            flush=True,
        )

    print("\n[4P RESULTS]", flush=True)
    for lineup_name, lineup_stats in summary["four_player"].items():
        print(f"  {lineup_name}:", flush=True)
        for label, lstats in sorted(lineup_stats.items(), key=lambda x: -x[1]["first_rate"]):
            print(
                f"    {label:15s}: 1st={100.0*lstats['first_rate']:5.1f}%  "
                f"top2={100.0*lstats['top2_rate']:5.1f}%  avg_rank={lstats['avg_rank']:.2f}  "
                f"avg_reward={lstats['avg_reward']:.1f}",
                flush=True,
            )

    print(f"\nLog written to: {log_path}", flush=True)
    print(json.dumps(summary, default=json_default, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
