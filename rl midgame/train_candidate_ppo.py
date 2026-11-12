from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("KAGGLE_ENVIRONMENTS_QUIET", "1")

WORKSPACE_DIR = Path(__file__).resolve().parent
ROOT = WORKSPACE_DIR.parent
GENOME_DIR = ROOT / "genome test"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
if str(GENOME_DIR) not in sys.path:
    sys.path.insert(0, str(GENOME_DIR))

import test_agent  # noqa: E402
from midgame_policy import PPOMissionPolicy, create_policy, load_policy_json  # noqa: E402
from midgame_rl_agent import BASE_AGENT, EpisodeRecorder, MidgameRLConfig, build_agent  # noqa: E402


def reward_signal(result: Dict[str, Any], as_player_a: bool) -> float:
    reward = float(result["reward_a"] - result["reward_b"]) if as_player_a else float(result["reward_b"] - result["reward_a"])
    ships = float(result["ships_a"] - result["ships_b"]) if as_player_a else float(result["ships_b"] - result["ships_a"])
    return reward + 0.2 * (ships / max(50.0, abs(ships) + 50.0))


def play_episode(
    policy: PPOMissionPolicy,
    rl_config: MidgameRLConfig,
    opponent: Any,
    *,
    swapped: bool,
    explore: bool,
) -> Dict[str, Any]:
    recorder = EpisodeRecorder()
    learner = build_agent(policy=policy, rl_config=rl_config, recorder=recorder, explore=explore)
    if swapped:
        result = test_agent.run_game(opponent, learner)
        samples = recorder.pop(1)
        reward = reward_signal(result, as_player_a=False)
    else:
        result = test_agent.run_game(learner, opponent)
        samples = recorder.pop(0)
        reward = reward_signal(result, as_player_a=True)
    return {
        "result": result,
        "samples": samples,
        "reward": reward,
        "swapped": swapped,
        "decisions": len(samples),
    }


def make_snapshot_agent(snapshot_policy: PPOMissionPolicy, rl_config: MidgameRLConfig) -> Any:
    return build_agent(policy=snapshot_policy, rl_config=rl_config, recorder=None, explore=False)


def evaluate_policy(
    policy: PPOMissionPolicy,
    rl_config: MidgameRLConfig,
    games: int,
    seed: int,
) -> Dict[str, float]:
    rng = random.Random(seed)
    wins = 0
    draws = 0
    rewards: List[float] = []
    decision_counts: List[int] = []
    for game_index in range(games):
        swapped = (game_index % 2) == 1
        outcome = play_episode(
            policy,
            rl_config,
            BASE_AGENT,
            swapped=swapped,
            explore=False,
        )
        rewards.append(float(outcome["reward"]))
        decision_counts.append(int(outcome["decisions"]))
        result = outcome["result"]
        learner_won = (result["winner"] == "A" and not swapped) or (result["winner"] == "B" and swapped)
        if result["winner"] == "draw":
            draws += 1
        elif learner_won:
            wins += 1
    return {
        "games": float(games),
        "win_rate": wins / max(1, games),
        "draw_rate": draws / max(1, games),
        "avg_reward": sum(rewards) / max(1, len(rewards)),
        "avg_decisions": sum(decision_counts) / max(1, len(decision_counts)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a candidate-based PPO mission policy with self-play snapshots")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--games-per-update", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy-in")
    parser.add_argument("--policy-out", default=str(WORKSPACE_DIR / "results" / "candidate_ppo_policy.json"))
    parser.add_argument("--summary-out", default=str(WORKSPACE_DIR / "results" / "candidate_ppo_summary.json"))
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--self-play-update-interval", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=12)
    parser.add_argument("--alternate-player-sides", action="store_true")
    parser.add_argument("--opponent", choices=("self", "heuristic", "mixed"), default="self")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--activation-turn", type=int, default=16)
    parser.add_argument("--max-turn", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--allow-opening", action="store_true")
    parser.add_argument("--no-contested-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    policy = load_policy_json(args.policy_in) if args.policy_in else create_policy("ppo", hidden_size=args.hidden_size, seed=args.seed)
    if not isinstance(policy, PPOMissionPolicy):
        raise SystemExit("Expected a PPO policy for candidate PPO training")

    rl_config = MidgameRLConfig(
        activation_turn=args.activation_turn,
        max_turn=args.max_turn,
        min_candidates=2,
        top_k=args.top_k,
        contested_only=not args.no_contested_only,
        explore=True,
        allow_opening=args.allow_opening,
    )
    eval_config = MidgameRLConfig(
        activation_turn=args.activation_turn,
        max_turn=args.max_turn,
        min_candidates=2,
        top_k=args.top_k,
        contested_only=not args.no_contested_only,
        explore=False,
        allow_opening=args.allow_opening,
    )

    snapshots: List[PPOMissionPolicy] = [policy.clone()]
    log_rows: List[Dict[str, Any]] = []
    policy_out = Path(args.policy_out)
    policy_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    for update_index in range(1, args.updates + 1):
        batch: List[Dict[str, Any]] = []
        for game_index in range(args.games_per_update):
            swapped = bool(args.alternate_player_sides and ((update_index + game_index) % 2 == 0))
            if args.opponent == "heuristic":
                opponent = BASE_AGENT
                opponent_name = "heuristic"
            elif args.opponent == "mixed" and (update_index + game_index) % 2 == 0:
                opponent = BASE_AGENT
                opponent_name = "heuristic"
            else:
                snapshot = snapshots[(update_index + game_index - 1) % len(snapshots)]
                opponent = make_snapshot_agent(snapshot, eval_config)
                opponent_name = "self"
            outcome = play_episode(policy, rl_config, opponent, swapped=swapped, explore=True)
            batch.append({"samples": outcome["samples"], "reward": outcome["reward"]})
            log_rows.append({
                "update": update_index,
                "opponent": opponent_name,
                "swapped": swapped,
                "reward": float(outcome["reward"]),
                "winner": outcome["result"]["winner"],
                "decisions": int(outcome["decisions"]),
            })

        update_metrics = policy.batch_update(batch, learning_rate=args.learning_rate, epochs=args.epochs)
        eval_metrics = evaluate_policy(policy, eval_config, args.eval_games, seed=args.seed + update_index * 1000)
        summary = {
            "update": update_index,
            "update_metrics": update_metrics,
            "evaluation": eval_metrics,
        }
        log_rows.append(summary)
        print(
            f"[update {update_index:03d}] reward={update_metrics['reward']:+.3f} "
            f"loss={update_metrics['loss']:.4f} eval_win={eval_metrics['win_rate']:.3f} "
            f"eval_reward={eval_metrics['avg_reward']:+.3f}",
            flush=True,
        )

        if update_index % max(1, args.self_play_update_interval) == 0:
            snapshots.append(policy.clone())
            snapshots = snapshots[-4:]

        if update_index % max(1, args.checkpoint_every) == 0 or update_index == args.updates:
            checkpoint_path = policy_out.with_name(f"{policy_out.stem}_u{update_index:03d}{policy_out.suffix}")
            policy.save_json(checkpoint_path)

    policy.save_json(policy_out)
    payload = {
        "config": vars(args),
        "final_policy": str(policy_out),
        "history": log_rows,
    }
    summary_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "final_policy": str(policy_out),
        "summary": str(summary_out),
        "updates": args.updates,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
