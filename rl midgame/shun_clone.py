"""
Shun_PI replay-scripted opponent for opening training.

Extracts Shun_PI's opening actions from Kaggle Orbit Wars replays,
builds a replay-scripted agent that mimics his first N turns, and
provides pattern analysis across all available replays.

Usage:
    # Analyze Shun's opening patterns
    python "rl midgame/shun_clone.py" --replay-dir /Users/alexchilton/Downloads

    # Test our agent against Shun openers
    python "rl midgame/shun_clone.py" --test --agent orbit_wars_agent.py --games 3
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Replay parsing
# ---------------------------------------------------------------------------

SHUN_NAME = "Shun_PI"
DEFAULT_OPENING_TURNS = 30
REPLAY_PATTERN = "episode-*.json"

# Planet tuple indices in observation: [id, owner, x, y, radius, ships, production]
P_ID, P_OWNER, P_X, P_Y, P_RADIUS, P_SHIPS, P_PROD = range(7)


def load_replay(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def find_shun_index(replay: dict) -> Optional[int]:
    """Return Shun_PI's player index in a replay, or None if absent."""
    names = replay.get("info", {}).get("TeamNames", [])
    if SHUN_NAME in names:
        return names.index(SHUN_NAME)
    return None


def extract_opening_actions(
    replay: dict, player_idx: int, max_turns: int = DEFAULT_OPENING_TURNS
) -> List[Tuple[int, list]]:
    """Extract (step, action_list) pairs for the first `max_turns` turns.

    action_list entries are [[source_planet_id, angle, ships], ...]
    """
    steps = replay.get("steps", [])
    opening = []
    for i, step_data in enumerate(steps):
        if i >= max_turns:
            break
        if player_idx >= len(step_data):
            continue
        action = step_data[player_idx].get("action", [])
        if action is None:
            action = []
        opening.append((i, action))
    return opening


def extract_opening_planets(
    replay: dict, player_idx: int, at_step: int = 0
) -> list:
    """Extract planet state from the replay at a given step."""
    steps = replay.get("steps", [])
    if at_step >= len(steps) or player_idx >= len(steps[at_step]):
        return []
    obs = steps[at_step][player_idx].get("observation", {})
    return obs.get("planets", [])


def get_shun_replays(replay_dir: str) -> List[Tuple[str, dict, int]]:
    """Return list of (path, replay_data, shun_player_idx) for all Shun replays."""
    pattern = os.path.join(replay_dir, REPLAY_PATTERN)
    results = []
    for path in sorted(glob.glob(pattern)):
        replay = load_replay(path)
        idx = find_shun_index(replay)
        if idx is not None:
            results.append((path, replay, idx))
    return results


def get_2p_shun_replays(replay_dir: str) -> List[Tuple[str, dict, int]]:
    """Return only 2-player Shun replays (most useful for 1v1 training)."""
    return [
        (p, r, i)
        for p, r, i in get_shun_replays(replay_dir)
        if len(r["info"]["TeamNames"]) == 2
    ]


# ---------------------------------------------------------------------------
# Replay-scripted agent builder
# ---------------------------------------------------------------------------

def _load_baseline_agent() -> Callable:
    """Load the stage3 baseline agent as fallback."""
    baseline_path = Path(__file__).resolve().parent.parent / "snapshots" / "stage3_search_base.py"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline agent not found: {baseline_path}")
    src = baseline_path.read_text(encoding="utf-8")
    # Strip Jupyter magic if present
    lines = src.splitlines()
    if lines and lines[0].startswith("%%"):
        src = "\n".join(lines[1:])
    mod = types.ModuleType("_shun_baseline_fallback")
    exec(compile(src, str(baseline_path), "exec"), mod.__dict__)
    # The baseline may expose agent() or _base_agent_entrypoint()
    if hasattr(mod, "agent"):
        return mod.agent
    if hasattr(mod, "_base_agent_entrypoint"):
        return mod._base_agent_entrypoint
    raise AttributeError(f"No agent() or _base_agent_entrypoint() found in {baseline_path}")


def _planet_ids_match(replay_planets: list, obs_planets: list) -> bool:
    """Check if the replay's map matches the current game's map.

    We compare planet positions at step 0 — if the same map seed was used
    the planet IDs will correspond. We use a loose check: same count and
    similar positions for the first few planets.
    """
    if not replay_planets or not obs_planets:
        return False
    if len(replay_planets) != len(obs_planets):
        return False
    # Compare positions of first few planets (tolerance for floating point)
    for rp, op in zip(replay_planets[:5], obs_planets[:5]):
        dx = rp[P_X] - op[P_X]
        dy = rp[P_Y] - op[P_Y]
        if math.sqrt(dx * dx + dy * dy) > 1.0:
            return False
    return True


def build_shun_replay_agent(
    replay_path: str,
    fallback_turns: int = DEFAULT_OPENING_TURNS,
) -> Callable:
    """Returns an agent(obs, config) that replays Shun's exact moves for the
    first N turns, then falls back to the baseline heuristic.

    If planet IDs don't match (different map), falls back immediately.
    """
    replay = load_replay(replay_path)
    shun_idx = find_shun_index(replay)
    if shun_idx is None:
        raise ValueError(f"Shun_PI not found in {replay_path}")

    opening = extract_opening_actions(replay, shun_idx, max_turns=fallback_turns)
    action_by_step = {step: actions for step, actions in opening}
    replay_planets = extract_opening_planets(replay, shun_idx, at_step=0)

    baseline_agent = _load_baseline_agent()

    # State persisted across calls via closure
    state = {"map_checked": False, "map_ok": False}

    def agent(obs: Any, config: Any) -> list:
        step = obs.get("step", obs.step) if isinstance(obs, dict) else obs.step
        planets = (
            obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        )

        # On first call, check if the map matches
        if not state["map_checked"]:
            state["map_checked"] = True
            state["map_ok"] = _planet_ids_match(replay_planets, planets)

        # Replay Shun's action if map matches and we're in the opening window
        if state["map_ok"] and step in action_by_step:
            return action_by_step[step]

        # Fallback to baseline
        return baseline_agent(obs, config)

    return agent


def build_random_shun_opener(
    replay_dir: str,
    fallback_turns: int = DEFAULT_OPENING_TURNS,
    only_2p: bool = True,
) -> Callable:
    """Build an agent that picks a random Shun replay each game for its opener.

    Since a new agent closure is needed per game (to reset map-match state),
    this returns a factory function. Call it before each game to get a fresh agent.
    """
    if only_2p:
        replays = get_2p_shun_replays(replay_dir)
    else:
        replays = get_shun_replays(replay_dir)

    if not replays:
        raise ValueError(f"No Shun_PI replays found in {replay_dir}")

    paths = [p for p, _, _ in replays]

    def make_agent() -> Callable:
        chosen = random.choice(paths)
        return build_shun_replay_agent(chosen, fallback_turns=fallback_turns)

    return make_agent


# ---------------------------------------------------------------------------
# Opening pattern analysis
# ---------------------------------------------------------------------------

def extract_shun_opening_patterns(replay_dir: str) -> dict:
    """Analyze Shun's opening patterns across all replays.

    Returns dict with:
        - avg_first_launch_step: mean step of Shun's first fleet launch
        - first_launch_steps: list of first-launch steps per replay
        - typical_early_targets: stats about early target selection
        - ship_send_patterns: how many ships per early launch
        - action_density_by_turn: average launches per turn in first 30
        - n_replays: number of replays analyzed
        - n_2p_replays: number that are 2-player
    """
    all_replays = get_shun_replays(replay_dir)
    if not all_replays:
        return {"error": "No Shun_PI replays found", "n_replays": 0}

    first_launch_steps: List[int] = []
    ships_per_launch: List[int] = []
    launches_per_turn: Dict[int, List[int]] = defaultdict(list)
    early_source_planets: List[int] = []
    total_early_launches = 0
    n_2p = 0

    for path, replay, shun_idx in all_replays:
        n_players = len(replay["info"]["TeamNames"])
        if n_players == 2:
            n_2p += 1

        opening = extract_opening_actions(replay, shun_idx, max_turns=DEFAULT_OPENING_TURNS)

        # First launch step
        first_found = False
        for step, actions in opening:
            n_launches = len(actions) if actions else 0
            launches_per_turn[step].append(n_launches)

            if actions:
                if not first_found:
                    first_launch_steps.append(step)
                    first_found = True
                for act in actions:
                    if len(act) >= 3:
                        src_planet, _angle, ships = act[0], act[1], act[2]
                        ships_per_launch.append(int(ships))
                        early_source_planets.append(int(src_planet))
                        total_early_launches += 1

        if not first_found:
            first_launch_steps.append(DEFAULT_OPENING_TURNS)

    # Compute density: avg launches per turn
    density = {}
    for t in range(DEFAULT_OPENING_TURNS):
        counts = launches_per_turn.get(t, [0])
        density[t] = round(statistics.mean(counts), 3) if counts else 0.0

    # Source planet frequency
    source_freq = defaultdict(int)
    for pid in early_source_planets:
        source_freq[pid] += 1
    top_sources = sorted(source_freq.items(), key=lambda x: -x[1])[:10]

    # Ship count stats
    ship_stats = {}
    if ships_per_launch:
        ship_stats = {
            "mean": round(statistics.mean(ships_per_launch), 1),
            "median": round(statistics.median(ships_per_launch), 1),
            "min": min(ships_per_launch),
            "max": max(ships_per_launch),
            "stdev": round(statistics.stdev(ships_per_launch), 1) if len(ships_per_launch) > 1 else 0,
        }

    return {
        "n_replays": len(all_replays),
        "n_2p_replays": n_2p,
        "avg_first_launch_step": round(statistics.mean(first_launch_steps), 2),
        "median_first_launch_step": round(statistics.median(first_launch_steps), 1),
        "first_launch_steps": first_launch_steps,
        "ship_send_patterns": ship_stats,
        "total_early_launches": total_early_launches,
        "top_source_planets": top_sources,
        "action_density_by_turn": density,
    }


# ---------------------------------------------------------------------------
# Testing against Shun openers
# ---------------------------------------------------------------------------

def _load_agent_from_file(path: str) -> Callable:
    """Load an agent(obs, config) callable from a .py file."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Agent file not found: {p}")
    module_name = f"_agent_{p.stem}_{id(p)}"
    spec = importlib.util.spec_from_file_location(module_name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "agent"):
        raise AttributeError(f"No top-level agent() function found in {p}")
    return mod.agent


def test_against_shun_opener(
    our_agent_path: str,
    n_games: int = 5,
    replay_dir: str = "/Users/alexchilton/Downloads",
    fallback_turns: int = DEFAULT_OPENING_TURNS,
    swap: bool = True,
) -> dict:
    """Play our agent against Shun replay openers and report results.

    For each game, picks a random 2p Shun replay and uses its opener.
    Reports: wins, losses, avg ships at end for both sides.
    """
    # Import kaggle env here so module loads fast without it
    os.environ.setdefault("KAGGLE_ENVIRONMENTS_QUIET", "1")
    from kaggle_environments import make  # noqa

    project_root = Path(__file__).resolve().parent.parent
    agent_path = Path(our_agent_path)
    if not agent_path.is_absolute():
        agent_path = project_root / agent_path

    our_agent = _load_agent_from_file(str(agent_path))
    shun_factory = build_random_shun_opener(
        replay_dir, fallback_turns=fallback_turns, only_2p=True
    )

    results = []
    for i in range(n_games):
        shun_agent = shun_factory()

        # Our agent as player 0
        env = make("orbit_wars", debug=False)
        env.run([our_agent, shun_agent])
        final = env.steps[-1]
        reward_us = final[0].reward or 0
        reward_shun = final[1].reward or 0
        planets = final[0].observation.get("planets", [])
        ships_us = sum(p[P_SHIPS] for p in planets if p[P_OWNER] == 0)
        ships_shun = sum(p[P_SHIPS] for p in planets if p[P_OWNER] == 1)

        result = {
            "game": i + 1,
            "side": "us=P0",
            "reward_us": reward_us,
            "reward_shun": reward_shun,
            "ships_us": ships_us,
            "ships_shun": ships_shun,
            "winner": "us" if reward_us > reward_shun else ("shun" if reward_shun > reward_us else "draw"),
        }
        results.append(result)
        marker = "✓" if result["winner"] == "us" else ("✗" if result["winner"] == "shun" else "=")
        print(f"  Game {i+1:>2}/{n_games} (P0) {marker}  ships {ships_us:>5} vs {ships_shun:>5}")

    if swap:
        print("  --- Swapped (our agent as P1) ---")
        for i in range(n_games):
            shun_agent = shun_factory()

            env = make("orbit_wars", debug=False)
            env.run([shun_agent, our_agent])
            final = env.steps[-1]
            reward_shun = final[0].reward or 0
            reward_us = final[1].reward or 0
            planets = final[0].observation.get("planets", [])
            ships_shun = sum(p[P_SHIPS] for p in planets if p[P_OWNER] == 0)
            ships_us = sum(p[P_SHIPS] for p in planets if p[P_OWNER] == 1)

            result = {
                "game": n_games + i + 1,
                "side": "us=P1",
                "reward_us": reward_us,
                "reward_shun": reward_shun,
                "ships_us": ships_us,
                "ships_shun": ships_shun,
                "winner": "us" if reward_us > reward_shun else ("shun" if reward_shun > reward_us else "draw"),
            }
            results.append(result)
            marker = "✓" if result["winner"] == "us" else ("✗" if result["winner"] == "shun" else "=")
            print(f"  Game {n_games+i+1:>2}/{n_games*2} (P1) {marker}  ships {ships_us:>5} vs {ships_shun:>5}")

    # Summary
    total = len(results)
    wins = sum(1 for r in results if r["winner"] == "us")
    losses = sum(1 for r in results if r["winner"] == "shun")
    draws = sum(1 for r in results if r["winner"] == "draw")
    avg_ships_us = statistics.mean(r["ships_us"] for r in results) if results else 0
    avg_ships_shun = statistics.mean(r["ships_shun"] for r in results) if results else 0

    summary = {
        "total_games": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(wins / total * 100, 1) if total else 0,
        "avg_ships_us": round(avg_ships_us, 1),
        "avg_ships_shun": round(avg_ships_shun, 1),
        "details": results,
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_patterns(patterns: dict) -> None:
    print("\n" + "=" * 60)
    print("  Shun_PI Opening Pattern Analysis")
    print("=" * 60)
    print(f"  Replays analyzed:       {patterns['n_replays']} ({patterns['n_2p_replays']} are 2-player)")
    print(f"  Avg first launch step:  {patterns['avg_first_launch_step']}")
    print(f"  Median first launch:    {patterns['median_first_launch_step']}")
    print(f"  First launch spread:    {patterns['first_launch_steps']}")

    sp = patterns.get("ship_send_patterns", {})
    if sp:
        print(f"\n  Ships per launch:")
        print(f"    mean={sp['mean']}  median={sp['median']}  "
              f"min={sp['min']}  max={sp['max']}  stdev={sp['stdev']}")

    print(f"\n  Total early launches (first 30 turns): {patterns['total_early_launches']}")
    print(f"  Top source planets: {patterns['top_source_planets']}")

    density = patterns.get("action_density_by_turn", {})
    if density:
        print(f"\n  Action density (launches/turn):")
        for t in range(0, DEFAULT_OPENING_TURNS, 5):
            chunk = [density.get(t + j, 0) for j in range(5)]
            labels = [f"T{t+j}:{v:.2f}" for j, v in enumerate(chunk)]
            print(f"    {' '.join(labels)}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shun_PI replay-scripted opener")
    parser.add_argument(
        "--replay-dir",
        default="/Users/alexchilton/Downloads",
        help="Directory containing episode-*.json replays",
    )
    parser.add_argument("--test", action="store_true", help="Run test games against Shun openers")
    parser.add_argument("--agent", default="orbit_wars_agent.py", help="Agent .py to test")
    parser.add_argument("--games", type=int, default=3, help="Number of test games per side")
    parser.add_argument("--no-swap", action="store_true", help="Don't swap player order")
    args = parser.parse_args()

    # Always print pattern analysis
    print("Analyzing Shun_PI opening patterns...")
    patterns = extract_shun_opening_patterns(args.replay_dir)
    _print_patterns(patterns)

    if args.test:
        print(f"Testing {args.agent} vs Shun_PI openers ({args.games} games/side)...\n")
        summary = test_against_shun_opener(
            our_agent_path=args.agent,
            n_games=args.games,
            replay_dir=args.replay_dir,
            swap=not args.no_swap,
        )
        print("\n" + "=" * 60)
        print("  Test Results vs Shun_PI Openers")
        print("=" * 60)
        print(f"  Games:    {summary['total_games']}")
        print(f"  Wins:     {summary['wins']}  ({summary['win_rate']}%)")
        print(f"  Losses:   {summary['losses']}")
        print(f"  Draws:    {summary['draws']}")
        print(f"  Avg ships (us):   {summary['avg_ships_us']}")
        print(f"  Avg ships (shun): {summary['avg_ships_shun']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
