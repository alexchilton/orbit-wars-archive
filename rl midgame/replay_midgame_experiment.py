from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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

from kaggle_environments import make  # noqa: E402

from midgame_policy import create_policy, load_policy_json  # noqa: E402
from midgame_rl_agent import BASE, BASE_AGENT, EpisodeRecorder, MidgameRLConfig, build_agent  # noqa: E402


@dataclass(frozen=True)
class StepMetrics:
    step: int
    target_ships: float
    target_share: float
    target_rank: int
    margin_vs_best_other: float
    total_ships: float
    ships_by_player: List[float]
    production_by_player: List[float]


@dataclass(frozen=True)
class ReplayCandidate:
    replay_path: str
    episode_id: int
    num_agents: int
    player_index: int
    player_name: str
    start_step: int
    horizon_end_step: int
    start_metrics: StepMetrics
    historical_end_metrics: StepMetrics
    historical_margin_drop: float
    final_reward: float


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _player_index(replay: Dict[str, Any], player_name: str) -> Optional[int]:
    needle = player_name.strip().lower()
    team_names = replay.get("info", {}).get("TeamNames") or []
    for index, name in enumerate(team_names):
        if needle in str(name).lower():
            return index
    agents = replay.get("info", {}).get("Agents") or []
    for index, agent in enumerate(agents):
        if needle in str(agent.get("Name", "")).lower():
            return index
    return None


def compute_step_metrics(observation: Dict[str, Any], player_index: int, num_agents: int, step: int) -> StepMetrics:
    ships = [0.0] * num_agents
    production = [0.0] * num_agents
    for planet in observation.get("planets") or []:
        owner = int(planet[1])
        if 0 <= owner < num_agents:
            ships[owner] += float(planet[5])
            production[owner] += float(planet[6])
    for fleet in observation.get("fleets") or []:
        owner = int(fleet[1])
        if 0 <= owner < num_agents:
            ships[owner] += float(fleet[6])

    total_ships = max(1.0, sum(ships))
    shares = [value / total_ships for value in ships]
    target_share = shares[player_index]
    other_shares = [share for idx, share in enumerate(shares) if idx != player_index]
    margin = target_share - (max(other_shares) if other_shares else 0.0)
    target_rank = 1 + sum(1 for value in ships if value > ships[player_index])
    return StepMetrics(
        step=step,
        target_ships=ships[player_index],
        target_share=target_share,
        target_rank=target_rank,
        margin_vs_best_other=margin,
        total_ships=total_ships,
        ships_by_player=ships,
        production_by_player=production,
    )


def select_replay_candidates(
    replay_paths: Iterable[Path],
    player_name: str,
    min_step: int = 24,
    max_step: int = 180,
    horizon: int = 50,
    max_candidates: int = 3,
    min_margin_drop: float = 0.05,
    candidates_per_replay: int = 1,
    start_rank_max: int = 1,
    min_start_share_two_player: float = 0.45,
    min_start_share_multi: float = 0.30,
    include_wins: bool = False,
) -> List[ReplayCandidate]:
    candidates: List[ReplayCandidate] = []
    for replay_path in replay_paths:
        replay = _load_json(replay_path)
        player_index = _player_index(replay, player_name)
        if player_index is None:
            continue
        rewards = replay.get("rewards") or []
        if player_index >= len(rewards):
            continue
        final_reward_raw = rewards[player_index]
        if final_reward_raw is None:
            continue
        final_reward = float(final_reward_raw)
        if not include_wins and final_reward >= 0:
            continue

        steps = replay.get("steps") or []
        if not steps:
            continue
        num_agents = len(steps[0])
        replay_candidates: List[ReplayCandidate] = []
        latest_start = min(max_step, len(steps) - 1 - horizon)
        for start_step in range(min_step, latest_start + 1):
            start_obs = steps[start_step][0]["observation"]
            end_step = min(len(steps) - 1, start_step + horizon)
            end_obs = steps[end_step][0]["observation"]
            start_metrics = compute_step_metrics(start_obs, player_index, num_agents, start_step)
            end_metrics = compute_step_metrics(end_obs, player_index, num_agents, end_step)
            min_start_share = min_start_share_two_player if num_agents <= 2 else min_start_share_multi
            if start_metrics.target_rank > start_rank_max or start_metrics.target_share < min_start_share:
                continue
            margin_drop = start_metrics.margin_vs_best_other - end_metrics.margin_vs_best_other
            if margin_drop < min_margin_drop:
                continue
            replay_candidates.append(ReplayCandidate(
                replay_path=str(replay_path),
                episode_id=int(replay.get("info", {}).get("EpisodeId", -1)),
                num_agents=num_agents,
                player_index=player_index,
                player_name=str((replay.get("info", {}).get("TeamNames") or [player_name])[player_index]),
                start_step=start_step,
                horizon_end_step=end_step,
                start_metrics=start_metrics,
                historical_end_metrics=end_metrics,
                historical_margin_drop=margin_drop,
                final_reward=final_reward,
            ))
        replay_candidates.sort(key=lambda item: item.historical_margin_drop, reverse=True)
        candidates.extend(replay_candidates[: max(1, candidates_per_replay)])

    candidates.sort(key=lambda item: item.historical_margin_drop, reverse=True)
    return candidates[:max_candidates]


def _restore_env(replay: Dict[str, Any], start_step: int):
    env = make("orbit_wars", configuration=replay.get("configuration") or {}, debug=False)
    env._Environment__set_state(copy.deepcopy(replay["steps"][start_step]))
    env.steps = [None] * (start_step + 1)
    env.logs = [[] for _ in range(start_step + 1)]
    return env


def _historical_actions(replay: Dict[str, Any], step_index: int) -> List[Any]:
    return [agent.get("action") or [] for agent in replay["steps"][step_index]]


def rollout_candidate(
    replay: Dict[str, Any],
    candidate: ReplayCandidate,
    horizon: int,
    controller: Optional[Any] = None,
    recorder: Optional[EpisodeRecorder] = None,
    include_samples: bool = False,
) -> Dict[str, Any]:
    env = _restore_env(replay, candidate.start_step)
    current_step = candidate.start_step
    end_step = min(len(replay["steps"]) - 1, candidate.start_step + horizon)
    while current_step < end_step and not env.done:
        next_step = current_step + 1
        actions = _historical_actions(replay, next_step)
        if controller is not None and env.state[candidate.player_index].status == "ACTIVE":
            obs = env.state[candidate.player_index].observation
            actions[candidate.player_index] = controller(obs, env.configuration)
        env.step(actions)
        current_step = next_step

    end_metrics = compute_step_metrics(
        env.steps[-1][0].observation,
        candidate.player_index,
        candidate.num_agents,
        current_step,
    )
    samples = recorder.pop(candidate.player_index) if recorder is not None else []
    result = {
        "step": current_step,
        "reward_signal": end_metrics.margin_vs_best_other - candidate.start_metrics.margin_vs_best_other,
        "decision_count": len(samples),
        "end_metrics": asdict(end_metrics),
    }
    if include_samples:
        result["samples"] = samples
    return result


def train_on_candidates(
    policy: Any,
    replay_by_path: Dict[str, Dict[str, Any]],
    candidates: List[ReplayCandidate],
    episodes: int,
    learning_rate: float,
    horizon: int,
    rl_config: MidgameRLConfig,
    use_heuristic_baseline: bool = True,
) -> Dict[str, Any]:
    logs: List[Dict[str, Any]] = []
    for episode in range(1, episodes + 1):
        candidate = candidates[(episode - 1) % len(candidates)]
        replay = replay_by_path[candidate.replay_path]
        recorder = EpisodeRecorder()
        controller = build_agent(
            policy=policy,
            rl_config=rl_config,
            recorder=recorder,
            explore=True,
        )
        outcome = rollout_candidate(
            replay,
            candidate,
            horizon=horizon,
            controller=controller,
            recorder=recorder,
            include_samples=True,
        )
        # Compute advantage relative to heuristic baseline
        if use_heuristic_baseline:
            heuristic_outcome = rollout_candidate(
                replay, candidate, horizon=horizon, controller=BASE_AGENT,
            )
            advantage = float(outcome["reward_signal"]) - float(heuristic_outcome["reward_signal"])
        else:
            advantage = float(outcome["reward_signal"])
            heuristic_outcome = {"reward_signal": 0.0}

        update = policy.update(
            outcome["samples"],
            reward=advantage,
            learning_rate=learning_rate,
        )
        logs.append({
            "episode": episode,
            "episode_id": candidate.episode_id,
            "start_step": candidate.start_step,
            "reward_signal": outcome["reward_signal"],
            "heuristic_reward": heuristic_outcome["reward_signal"],
            "advantage": advantage,
            "decision_count": outcome["decision_count"],
            "avg_abs_weight": update["avg_abs_weight"],
        })
    return {"policy": policy, "logs": logs}


def evaluate_candidates(
    replay_by_path: Dict[str, Dict[str, Any]],
    candidates: List[ReplayCandidate],
    policy: Any,
    horizon: int,
    rl_config: MidgameRLConfig,
    heuristic_controller: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for candidate in candidates:
        replay = replay_by_path[candidate.replay_path]
        rl_recorder = EpisodeRecorder()
        rl_controller = build_agent(
            policy=policy,
            rl_config=rl_config,
            recorder=rl_recorder,
            explore=False,
        )
        heuristic = rollout_candidate(
            replay,
            candidate,
            horizon=horizon,
            controller=heuristic_controller or BASE_AGENT,
        )
        rl = rollout_candidate(
            replay,
            candidate,
            horizon=horizon,
            controller=rl_controller,
            recorder=rl_recorder,
        )
        results.append({
            "episode_id": candidate.episode_id,
            "player_name": candidate.player_name,
            "replay_path": candidate.replay_path,
            "start_step": candidate.start_step,
            "historical_margin_drop": candidate.historical_margin_drop,
            "start_metrics": asdict(candidate.start_metrics),
            "historical_end_metrics": asdict(candidate.historical_end_metrics),
            "heuristic": heuristic,
            "rl": rl,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Try replay-derived midgame RL starts on real Orbit Wars episodes")
    parser.add_argument(
        "--replay-glob",
        nargs="+",
        default=["kaggle_replays/*/episode-*-replay.json"],
        help="Replay glob(s) to scan",
    )
    parser.add_argument("--player-name", default="alex chilton", help="Player name to match in replay metadata")
    parser.add_argument("--min-step", type=int, default=24, help="Earliest replay step to consider")
    parser.add_argument("--max-step", type=int, default=180, help="Latest replay step to consider")
    parser.add_argument("--horizon", type=int, default=50, help="How many future steps to score from each start")
    parser.add_argument("--max-candidates", type=int, default=3, help="How many replay starts to keep")
    parser.add_argument("--eval-candidates", type=int, default=10, help="How many top replay starts to evaluate after training")
    parser.add_argument("--train-episodes", type=int, default=6, help="How many replay-start episodes to train on")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Replay-start RL learning rate")
    parser.add_argument("--policy-model", choices=["linear", "mlp"], default="linear", help="Policy scorer to train")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size for MLP scorer")
    parser.add_argument("--candidates-per-replay", type=int, default=1, help="How many top collapse starts to keep from each replay")
    parser.add_argument("--start-rank-max", type=int, default=1, help="Maximum allowed rank at the replay start")
    parser.add_argument("--min-start-share-2p", type=float, default=0.45, help="Minimum ship share at candidate start in 2p replays")
    parser.add_argument("--min-start-share-4p", type=float, default=0.30, help="Minimum ship share at candidate start in 4p+ replays")
    parser.add_argument("--activation-turn", type=int, default=0, help="Earliest turn where replay RL may activate")
    parser.add_argument("--rl-max-turn", type=int, default=220, help="Latest turn where replay RL may activate")
    parser.add_argument("--top-k", type=int, default=8, help="Top heuristic missions exposed to replay RL")
    parser.add_argument("--min-candidates", type=int, default=2, help="Minimum candidate count before replay RL may act")
    parser.add_argument("--no-contested-only", action="store_true", help="Allow replay RL outside contested windows")
    parser.add_argument("--allow-opening-rl", action="store_true", help="Allow replay RL to act during opening turns")
    parser.add_argument("--force-rl-window", action="store_true", help="Bypass contested/opening gating so RL acts whenever enough candidates exist")
    parser.add_argument("--no-heuristic-baseline", action="store_true", help="Disable heuristic-baselined advantage (use raw reward instead)")
    parser.add_argument("--include-wins", action="store_true", help="Also include replays where we won (not just lost)")
    parser.add_argument("--policy-in", help="Start from a pre-trained policy instead of random")
    parser.add_argument(
        "--policy-out",
        default=str(WORKSPACE_DIR / "results" / "replay_midgame_policy.json"),
        help="Where to save the replay-trained policy",
    )
    parser.add_argument(
        "--summary-out",
        default=str(WORKSPACE_DIR / "results" / "replay_midgame_summary.json"),
        help="Where to save the experiment summary",
    )
    args = parser.parse_args()

    replay_paths: List[Path] = []
    for pattern in args.replay_glob:
        replay_paths.extend(sorted(ROOT.glob(pattern)))
    replay_paths = sorted(set(replay_paths))
    if not replay_paths:
        raise SystemExit("No replay files matched")

    candidates = select_replay_candidates(
        replay_paths,
        player_name=args.player_name,
        min_step=args.min_step,
        max_step=args.max_step,
        horizon=args.horizon,
        max_candidates=args.max_candidates,
        candidates_per_replay=args.candidates_per_replay,
        start_rank_max=args.start_rank_max,
        min_start_share_two_player=args.min_start_share_2p,
        min_start_share_multi=args.min_start_share_4p,
        include_wins=args.include_wins,
    )
    if not candidates:
        raise SystemExit("No suitable replay starts found")

    replay_by_path = {str(path): _load_json(path) for path in replay_paths}
    rl_config = MidgameRLConfig(
        activation_turn=args.activation_turn,
        max_turn=args.rl_max_turn,
        min_candidates=args.min_candidates,
        top_k=args.top_k,
        contested_only=not args.no_contested_only,
        explore=True,
        allow_opening=args.allow_opening_rl,
        force_rl_window=args.force_rl_window,
    )
    policy = load_policy_json(args.policy_in) if args.policy_in else create_policy(args.policy_model, hidden_size=args.hidden_size)
    trained = train_on_candidates(
        policy,
        replay_by_path,
        candidates,
        episodes=args.train_episodes,
        learning_rate=args.learning_rate,
        horizon=args.horizon,
        rl_config=rl_config,
        use_heuristic_baseline=not args.no_heuristic_baseline,
    )
    policy: Any = trained["policy"]
    policy_path = policy.save_json(args.policy_out)
    eval_candidates = candidates[: max(1, min(args.eval_candidates, len(candidates)))]
    evaluation = evaluate_candidates(
        replay_by_path,
        eval_candidates,
        policy,
        horizon=args.horizon,
        rl_config=MidgameRLConfig(
            activation_turn=args.activation_turn,
            max_turn=args.rl_max_turn,
            min_candidates=args.min_candidates,
            top_k=args.top_k,
            contested_only=not args.no_contested_only,
            explore=False,
            allow_opening=args.allow_opening_rl,
            force_rl_window=args.force_rl_window,
        ),
    )

    summary = {
        "candidates": [asdict(candidate) for candidate in candidates],
        "config": {
            "horizon": args.horizon,
            "max_candidates": args.max_candidates,
            "eval_candidates": args.eval_candidates,
            "train_episodes": args.train_episodes,
            "learning_rate": args.learning_rate,
            "policy_model": args.policy_model,
            "hidden_size": args.hidden_size,
            "candidates_per_replay": args.candidates_per_replay,
            "start_rank_max": args.start_rank_max,
            "min_start_share_2p": args.min_start_share_2p,
            "min_start_share_4p": args.min_start_share_4p,
            "activation_turn": args.activation_turn,
            "rl_max_turn": args.rl_max_turn,
            "top_k": args.top_k,
            "min_candidates": args.min_candidates,
            "contested_only": not args.no_contested_only,
            "allow_opening_rl": args.allow_opening_rl,
            "force_rl_window": args.force_rl_window,
        },
        "training": trained["logs"],
        "evaluation": evaluation,
        "policy": str(policy_path),
    }
    output = Path(args.summary_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    for item in evaluation:
        print(
            f"episode {item['episode_id']} step {item['start_step']}: "
            f"historical_drop={item['historical_margin_drop']:.3f} "
            f"heuristic_reward={item['heuristic']['reward_signal']:+.3f} "
            f"rl_reward={item['rl']['reward_signal']:+.3f}",
            flush=True,
        )
    print(f"[done] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
