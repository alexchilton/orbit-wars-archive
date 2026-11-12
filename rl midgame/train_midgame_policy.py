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
from midgame_policy import create_policy, load_policy_json  # noqa: E402
from midgame_rl_agent import BASE_AGENT, EpisodeRecorder, MidgameRLConfig, build_agent  # noqa: E402
from weird_opponents import greedy_agent, turtle_agent  # noqa: E402

DEFAULT_OPPONENTS = ["baseline", "v21", "v23", "v16", "mtmr", "greedy", "turtle", "random"]


def load_reference_agents() -> Dict[str, Any]:
    return {
        "baseline": test_agent.load_baseline_agent(),
        "v21": test_agent.load_agent_from_file(str(ROOT / "snapshots" / "v21.py")),
        "v23": test_agent.load_agent_from_file(str(ROOT / "snapshots" / "v23_state_pivot.py")),
        "v16": test_agent.load_agent_from_file(str(ROOT / "snapshots" / "v16_broken.py")),
        "mtmr": test_agent.load_agent_from_file(str(ROOT / "snapshots" / "mtmr_trial_copy.py")),
        "greedy": greedy_agent,
        "turtle": turtle_agent,
        "random": "random",
    }


def reward_signal(result: Dict[str, Any], as_player_a: bool) -> float:
    reward = (result["reward_a"] - result["reward_b"]) if as_player_a else (result["reward_b"] - result["reward_a"])
    return reward / 2.0


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def play_episode(
    policy: Any,
    rl_config: MidgameRLConfig,
    opponent: Any,
    opponent_name: str,
    learning_rate: float,
    swapped: bool,
    use_heuristic_baseline: bool = True,
) -> Dict[str, Any]:
    recorder = EpisodeRecorder()
    learner = build_agent(policy=policy, rl_config=rl_config, recorder=recorder, explore=True)

    if swapped:
        result = test_agent.run_game(opponent, learner)
        samples = recorder.pop(1)
        reward = reward_signal(result, as_player_a=False)
    else:
        result = test_agent.run_game(learner, opponent)
        samples = recorder.pop(0)
        reward = reward_signal(result, as_player_a=True)

    # Compute advantage relative to heuristic baseline
    heuristic_reward = 0.0
    if use_heuristic_baseline:
        if swapped:
            heuristic_result = test_agent.run_game(opponent, BASE_AGENT)
            heuristic_reward = reward_signal(heuristic_result, as_player_a=False)
        else:
            heuristic_result = test_agent.run_game(BASE_AGENT, opponent)
            heuristic_reward = reward_signal(heuristic_result, as_player_a=True)
        advantage = reward - heuristic_reward
    else:
        advantage = reward

    update = policy.update(samples, reward=advantage, learning_rate=learning_rate)
    return {
        "opponent": opponent_name,
        "swapped": swapped,
        "reward_signal": reward,
        "heuristic_reward": heuristic_reward,
        "advantage": advantage,
        "decisions": len(samples),
        "winner": result["winner"],
        "ships_a": result["ships_a"],
        "ships_b": result["ships_b"],
        "update": update,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny RL policy for Orbit Wars midgame mission choice")
    parser.add_argument("--episodes", type=int, default=20, help="How many training episodes to run")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="REINFORCE learning rate")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--policy-in", help="Optional existing policy JSON to continue training from")
    parser.add_argument("--policy-model", choices=["linear", "mlp"], default="linear", help="Policy scorer to train")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size for MLP scorer")
    parser.add_argument(
        "--policy-out",
        default=str(WORKSPACE_DIR / "results" / "midgame_policy.json"),
        help="Where to save the trained policy JSON",
    )
    parser.add_argument(
        "--log-path",
        default=str(WORKSPACE_DIR / "results" / "training_log.jsonl"),
        help="Where to append training episode logs",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=DEFAULT_OPPONENTS,
        help="Opponent pool to train against",
    )
    parser.add_argument("--swap", action="store_true", help="Alternate seat order during training")
    parser.add_argument("--no-heuristic-baseline", action="store_true", help="Disable heuristic-baselined reward")
    parser.add_argument("--activation-turn", type=int, default=24, help="First turn where RL may activate")
    parser.add_argument("--max-turn", type=int, default=160, help="Last turn where RL may activate")
    parser.add_argument("--top-k", type=int, default=8, help="Top heuristic missions exposed to the policy")
    parser.add_argument("--no-contested-only", action="store_true", help="Allow RL outside contested windows")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    policy = load_policy_json(args.policy_in) if args.policy_in else create_policy(args.policy_model, hidden_size=args.hidden_size, seed=args.seed)
    rl_config = MidgameRLConfig(
        activation_turn=args.activation_turn,
        max_turn=args.max_turn,
        min_candidates=2,
        top_k=args.top_k,
        contested_only=not args.no_contested_only,
        explore=True,
    )
    references = load_reference_agents()
    log_path = Path(args.log_path)

    total_reward = 0.0
    total_decisions = 0
    for episode in range(1, args.episodes + 1):
        opponent_name = args.opponents[(episode - 1) % len(args.opponents)]
        opponent = references[opponent_name]
        swapped = bool(args.swap and (episode % 2 == 0))
        summary = play_episode(
            policy,
            rl_config,
            opponent=opponent,
            opponent_name=opponent_name,
            learning_rate=args.learning_rate,
            swapped=swapped,
            use_heuristic_baseline=not args.no_heuristic_baseline,
        )
        summary["episode"] = episode
        total_reward += summary["advantage"]
        total_decisions += summary["decisions"]
        append_jsonl(log_path, summary)
        print(
            f"[episode {episode:03d}] opp={opponent_name:<8} swapped={int(swapped)} "
            f"reward={summary['reward_signal']:+.2f} heur={summary['heuristic_reward']:+.2f} "
            f"adv={summary['advantage']:+.2f} decisions={summary['decisions']:>2} "
            f"avg|w|={summary['update']['avg_abs_weight']:.4f}",
            flush=True,
        )

    policy_path = policy.save_json(args.policy_out)
    print(
        f"[done] episodes={args.episodes} avg_advantage={total_reward / max(1, args.episodes):+.3f} "
        f"avg_decisions={total_decisions / max(1, args.episodes):.2f} policy={policy_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
