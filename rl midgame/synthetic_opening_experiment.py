from __future__ import annotations

import argparse
import copy
import json
import os
import random
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

from midgame_policy import create_policy  # noqa: E402
from midgame_rl_agent import BASE, BASE_AGENT, EpisodeRecorder, MidgameRLConfig, build_agent  # noqa: E402
from replay_midgame_experiment import compute_step_metrics  # noqa: E402


@dataclass(frozen=True)
class SyntheticOpeningCandidate:
    replay_path: str
    episode_id: int
    num_agents: int
    player_index: int
    player_name: str
    start_step: int
    variant_index: int
    start_metrics: Dict[str, Any]
    state_step: List[Dict[str, Any]]


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


def _mutate_ships(value: Any, scale: float) -> int:
    mutated = int(round(float(value) * scale))
    return max(0, mutated)


def mutate_step_state(
    step_state: List[Dict[str, Any]],
    rng: random.Random,
    planet_ship_jitter: float = 0.18,
    fleet_ship_jitter: float = 0.12,
) -> List[Dict[str, Any]]:
    mutated = copy.deepcopy(step_state)
    owner_scales: Dict[int, float] = {}

    def owner_scale(owner: int, spread: float) -> float:
        if owner not in owner_scales:
            low = max(0.75, 1.0 - spread)
            owner_scales[owner] = rng.uniform(low, 1.0 + spread)
        return owner_scales[owner]

    for agent_state in mutated:
        obs = agent_state.get("observation") or {}
        planets = obs.get("planets") or []
        fleets = obs.get("fleets") or []
        for planet in planets:
            owner = int(planet[1])
            spread = planet_ship_jitter * (1.2 if owner == -1 else 1.0)
            scale = owner_scale(owner, spread)
            planet[5] = _mutate_ships(planet[5], scale)
        for fleet in fleets:
            owner = int(fleet[1])
            scale = owner_scale(owner, fleet_ship_jitter)
            fleet[6] = max(1, _mutate_ships(fleet[6], scale))
    return mutated


def select_opening_templates(
    replay_paths: Iterable[Path],
    player_name: str,
    min_step: int = 4,
    max_step: int = 40,
    max_templates: int = 20,
    start_rank_max: int = 2,
    min_start_share_two_player: float = 0.42,
    min_start_share_multi: float = 0.22,
    stride: int = 4,
) -> List[Dict[str, Any]]:
    templates: List[Dict[str, Any]] = []
    for replay_path in replay_paths:
        replay = _load_json(replay_path)
        player_index = _player_index(replay, player_name)
        if player_index is None:
            continue
        rewards = replay.get("rewards") or []
        if player_index >= len(rewards):
            continue
        final_reward_raw = rewards[player_index]
        if final_reward_raw is None or float(final_reward_raw) >= 0:
            continue
        steps = replay.get("steps") or []
        if not steps:
            continue
        num_agents = len(steps[0])
        latest_start = min(max_step, len(steps) - 1)
        for start_step in range(min_step, latest_start + 1, max(1, stride)):
            start_obs = steps[start_step][0]["observation"]
            metrics = compute_step_metrics(start_obs, player_index, num_agents, start_step)
            min_share = min_start_share_two_player if num_agents <= 2 else min_start_share_multi
            if metrics.target_rank > start_rank_max or metrics.target_share < min_share:
                continue
            templates.append({
                "replay_path": str(replay_path),
                "episode_id": int(replay.get("info", {}).get("EpisodeId", -1)),
                "num_agents": num_agents,
                "player_index": player_index,
                "player_name": str((replay.get("info", {}).get("TeamNames") or [player_name])[player_index]),
                "start_step": start_step,
                "step_state": copy.deepcopy(steps[start_step]),
                "start_metrics": asdict(metrics),
            })
    templates.sort(
        key=lambda item: (
            -item["start_metrics"]["target_share"],
            item["start_metrics"]["target_rank"],
            item["start_step"],
        )
    )
    return templates[:max_templates]


def build_synthetic_candidates(
    templates: List[Dict[str, Any]],
    variants_per_template: int,
    seed: int,
) -> List[SyntheticOpeningCandidate]:
    rng = random.Random(seed)
    candidates: List[SyntheticOpeningCandidate] = []
    for template in templates:
        for variant_index in range(variants_per_template):
            state_step = mutate_step_state(template["step_state"], rng)
            start_obs = state_step[0]["observation"]
            metrics = compute_step_metrics(
                start_obs,
                template["player_index"],
                template["num_agents"],
                template["start_step"],
            )
            candidates.append(SyntheticOpeningCandidate(
                replay_path=template["replay_path"],
                episode_id=template["episode_id"],
                num_agents=template["num_agents"],
                player_index=template["player_index"],
                player_name=template["player_name"],
                start_step=template["start_step"],
                variant_index=variant_index,
                start_metrics=asdict(metrics),
                state_step=state_step,
            ))
    return candidates


def _restore_env_from_step(
    replay: Dict[str, Any],
    state_step: List[Dict[str, Any]],
    start_step: int,
):
    env = make("orbit_wars", configuration=replay.get("configuration") or {}, debug=False)
    env._Environment__set_state(copy.deepcopy(state_step))
    env.steps = [None] * (start_step + 1)
    env.logs = [[] for _ in range(start_step + 1)]
    return env


def rollout_synthetic_candidate(
    replay: Dict[str, Any],
    candidate: SyntheticOpeningCandidate,
    horizon: int,
    controller: Optional[Any] = None,
    recorder: Optional[EpisodeRecorder] = None,
    include_samples: bool = False,
) -> Dict[str, Any]:
    env = _restore_env_from_step(replay, candidate.state_step, candidate.start_step)
    current_step = candidate.start_step
    end_step = candidate.start_step + horizon
    while current_step < end_step and not env.done:
        actions: List[Any] = []
        for idx, agent_state in enumerate(env.state):
            if agent_state.status != "ACTIVE":
                actions.append([])
                continue
            obs = agent_state.observation
            if idx == candidate.player_index and controller is not None:
                actions.append(controller(obs, env.configuration))
            else:
                actions.append(BASE_AGENT(obs, env.configuration))
        env.step(actions)
        current_step += 1

    end_metrics = compute_step_metrics(
        env.steps[-1][0].observation,
        candidate.player_index,
        candidate.num_agents,
        current_step,
    )
    samples = recorder.pop(candidate.player_index) if recorder is not None else []
    result = {
        "step": current_step,
        "reward_signal": end_metrics.margin_vs_best_other - candidate.start_metrics["margin_vs_best_other"],
        "decision_count": len(samples),
        "end_metrics": asdict(end_metrics),
    }
    if include_samples:
        result["samples"] = samples
    return result


def train_on_candidates(
    policy: Any,
    replay_by_path: Dict[str, Dict[str, Any]],
    candidates: List[SyntheticOpeningCandidate],
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
        outcome = rollout_synthetic_candidate(
            replay,
            candidate,
            horizon=horizon,
            controller=controller,
            recorder=recorder,
            include_samples=True,
        )
        # Compute advantage relative to heuristic baseline
        if use_heuristic_baseline:
            heuristic_outcome = rollout_synthetic_candidate(
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
            "variant_index": candidate.variant_index,
            "reward_signal": outcome["reward_signal"],
            "heuristic_reward": heuristic_outcome["reward_signal"],
            "advantage": advantage,
            "decision_count": outcome["decision_count"],
            "avg_abs_weight": update["avg_abs_weight"],
        })
    return {"policy": policy, "logs": logs}


def evaluate_candidates(
    replay_by_path: Dict[str, Dict[str, Any]],
    candidates: List[SyntheticOpeningCandidate],
    policy: Any,
    horizon: int,
    rl_config: MidgameRLConfig,
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
        heuristic = rollout_synthetic_candidate(
            replay,
            candidate,
            horizon=horizon,
            controller=BASE_AGENT,
        )
        rl = rollout_synthetic_candidate(
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
            "variant_index": candidate.variant_index,
            "start_metrics": candidate.start_metrics,
            "heuristic": heuristic,
            "rl": rl,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train opening-capable RL from synthetic early-game states")
    parser.add_argument(
        "--replay-glob",
        nargs="+",
        default=["kaggle_replays/*/episode-*-replay.json"],
        help="Replay glob(s) to scan",
    )
    parser.add_argument("--player-name", default="alex chilton")
    parser.add_argument("--min-step", type=int, default=4)
    parser.add_argument("--max-step", type=int, default=36)
    parser.add_argument("--template-stride", type=int, default=4)
    parser.add_argument("--max-templates", type=int, default=24)
    parser.add_argument("--variants-per-template", type=int, default=4)
    parser.add_argument("--eval-candidates", type=int, default=20)
    parser.add_argument("--train-episodes", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--policy-model", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--start-rank-max", type=int, default=2)
    parser.add_argument("--min-start-share-2p", type=float, default=0.42)
    parser.add_argument("--min-start-share-4p", type=float, default=0.22)
    parser.add_argument("--activation-turn", type=int, default=0)
    parser.add_argument("--rl-max-turn", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-candidates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy-out", default=str(WORKSPACE_DIR / "results" / "synthetic_opening_policy.json"))
    parser.add_argument("--summary-out", default=str(WORKSPACE_DIR / "results" / "synthetic_opening_summary.json"))
    args = parser.parse_args()

    replay_paths: List[Path] = []
    for pattern in args.replay_glob:
        replay_paths.extend(sorted(ROOT.glob(pattern)))
    replay_paths = sorted(set(replay_paths))
    if not replay_paths:
        raise SystemExit("No replay files matched")

    templates = select_opening_templates(
        replay_paths,
        player_name=args.player_name,
        min_step=args.min_step,
        max_step=args.max_step,
        max_templates=args.max_templates,
        start_rank_max=args.start_rank_max,
        min_start_share_two_player=args.min_start_share_2p,
        min_start_share_multi=args.min_start_share_4p,
        stride=args.template_stride,
    )
    if not templates:
        raise SystemExit("No suitable opening templates found")

    candidates = build_synthetic_candidates(
        templates,
        variants_per_template=args.variants_per_template,
        seed=args.seed,
    )
    replay_by_path = {str(path): _load_json(path) for path in replay_paths}
    rl_config = MidgameRLConfig(
        activation_turn=args.activation_turn,
        max_turn=args.rl_max_turn,
        min_candidates=args.min_candidates,
        top_k=args.top_k,
        contested_only=False,
        explore=True,
        allow_opening=True,
        force_rl_window=True,
    )
    policy = create_policy(args.policy_model, hidden_size=args.hidden_size, seed=args.seed)
    trained = train_on_candidates(
        policy,
        replay_by_path,
        candidates,
        episodes=args.train_episodes,
        learning_rate=args.learning_rate,
        horizon=args.horizon,
        rl_config=rl_config,
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
            contested_only=False,
            explore=False,
            allow_opening=True,
            force_rl_window=True,
        ),
    )

    summary = {
        "config": {
            "min_step": args.min_step,
            "max_step": args.max_step,
            "template_stride": args.template_stride,
            "max_templates": args.max_templates,
            "variants_per_template": args.variants_per_template,
            "eval_candidates": args.eval_candidates,
            "train_episodes": args.train_episodes,
            "learning_rate": args.learning_rate,
            "policy_model": args.policy_model,
            "hidden_size": args.hidden_size,
            "horizon": args.horizon,
            "start_rank_max": args.start_rank_max,
            "min_start_share_2p": args.min_start_share_2p,
            "min_start_share_4p": args.min_start_share_4p,
            "activation_turn": args.activation_turn,
            "rl_max_turn": args.rl_max_turn,
            "top_k": args.top_k,
            "min_candidates": args.min_candidates,
            "seed": args.seed,
        },
        "template_count": len(templates),
        "candidate_count": len(candidates),
        "templates": [
            {
                "episode_id": item["episode_id"],
                "replay_path": item["replay_path"],
                "start_step": item["start_step"],
                "start_metrics": item["start_metrics"],
            }
            for item in templates
        ],
        "training": trained["logs"],
        "evaluation": evaluation,
        "policy": str(policy_path),
    }
    output = Path(args.summary_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    for item in evaluation:
        print(
            f"episode {item['episode_id']} step {item['start_step']} variant {item['variant_index']}: "
            f"heuristic_reward={item['heuristic']['reward_signal']:+.3f} "
            f"rl_reward={item['rl']['reward_signal']:+.3f}"
        )
    print(f"[done] wrote {output}")


if __name__ == "__main__":
    main()
