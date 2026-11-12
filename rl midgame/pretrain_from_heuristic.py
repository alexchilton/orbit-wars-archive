"""Imitation warm-start: initialize RL policy weights to approximate heuristic ranking.

This pre-trains the policy via supervised cross-entropy so that it starts by
matching the heuristic's preferred candidate ordering. RL fine-tuning can then
move the policy away from heuristic only when it finds *genuine* improvements,
instead of starting from random weights and damaging the heuristic.

Usage
-----
    python "rl midgame/pretrain_from_heuristic.py" \
        --replay-glob "kaggle_replays/*/episode-*-replay.json" \
        --positions 80 --epochs 30

After pre-training, verify with:
    python "rl midgame/pretrain_from_heuristic.py" --eval-only \
        --policy-in "rl midgame/results/pretrained_policy.json"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

from midgame_features import FEATURE_NAMES, build_defer_vector, build_mission_feature_bundle  # noqa: E402
from midgame_policy import DecisionSample, LinearMissionPolicy, create_policy, load_policy_json  # noqa: E402
from midgame_rl_agent import BASE, BASE_AGENT, MidgameRLConfig  # noqa: E402
from replay_midgame_experiment import _load_json, _restore_env, select_replay_candidates  # noqa: E402


MissionOption = BASE.MissionOption


def _softmax(scores: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not scores:
        return []
    safe_t = max(1e-6, temperature)
    scaled = [s / safe_t for s in scores]
    anchor = max(scaled)
    exps = [math.exp(s - anchor) for s in scaled]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [e / total for e in exps]


def build_heuristic_sample(logic: Any, missions: Sequence[Any], top_k: int) -> Optional[Dict[str, Any]]:
    ranked = sorted(missions, key=lambda mission: -mission.score)
    horizon = min(len(ranked), top_k)
    if horizon < 2:
        return None

    candidates = []
    turn_launch_cap = logic._turn_launch_cap()
    for idx in range(horizon):
        mission = ranked[idx]
        if not logic._mission_can_commit(mission, existing_moves=0):
            continue
        pending_sources = set(mission.source_ids)
        pending_targets = {mission.target_id} if logic._mission_blocks_target(mission) else set()
        follower_bonus = 0.0
        follower_horizon = min(len(ranked), 10)
        for follower in ranked[:follower_horizon]:
            if follower is mission:
                continue
            if logic._mission_can_commit(
                follower,
                extra_used_sources=pending_sources,
                extra_target_ids=pending_targets,
                existing_moves=len(mission.source_ids),
            ):
                follower_bonus = max(follower_bonus, follower.score)
        base_value = mission.score + 0.55 * follower_bonus
        bundle = build_mission_feature_bundle(
            logic,
            mission,
            base_value=base_value,
            existing_moves=0,
            turn_launch_cap=turn_launch_cap,
        )
        candidates.append({
            "mission": mission,
            "base_value": base_value,
            "vector": bundle.vector,
        })

    if len(candidates) < 2:
        return None

    heuristic_best_idx = max(range(len(candidates)), key=lambda i: candidates[i]["base_value"])
    defer_vector = build_defer_vector(candidates[heuristic_best_idx]["vector"])
    return {
        "vectors": [defer_vector] + [candidate["vector"] for candidate in candidates],
        "target_index": heuristic_best_idx + 1,
        "heuristic_best_idx": heuristic_best_idx,
        "candidate_count": len(candidates),
    }


def collect_heuristic_ranking_samples(
    replay_paths: Sequence[Path],
    player_name: str,
    max_positions: int = 100,
    min_step: int = 24,
    max_step: int = 180,
    top_k: int = 8,
    seed: int = 7,
    start_rank_max: int = 2,
    min_start_share_two_player: float = 0.42,
    min_start_share_multi: float = 0.22,
    min_margin_drop: float = 0.02,
    candidates_per_replay: int = 3,
    include_wins: bool = True,
) -> List[Dict[str, Any]]:
    """Collect heuristic-ranking samples from replay states that already pass
    the replay-midgame candidate filters, so warm-start uses positions where
    the RL pipeline can actually act."""
    samples: List[Dict[str, Any]] = []
    replay_by_path: Dict[str, Dict[str, Any]] = {}

    replay_candidates = select_replay_candidates(
        replay_paths,
        player_name=player_name,
        min_step=min_step,
        max_step=max_step,
        horizon=50,
        max_candidates=max(max_positions * max(2, candidates_per_replay), max_positions),
        min_margin_drop=min_margin_drop,
        candidates_per_replay=candidates_per_replay,
        start_rank_max=start_rank_max,
        min_start_share_two_player=min_start_share_two_player,
        min_start_share_multi=min_start_share_multi,
        include_wins=include_wins,
    )
    rng = random.Random(seed)
    rng.shuffle(replay_candidates)

    for candidate in replay_candidates:
        if len(samples) >= max_positions:
            break
        replay = replay_by_path.setdefault(candidate.replay_path, _load_json(Path(candidate.replay_path)))
        try:
            env = _restore_env(replay, candidate.start_step)
            if env.state[candidate.player_index].status != "ACTIVE":
                continue
            obs = env.state[candidate.player_index].observation
            logic = BASE.DecisionLogic(obs, env.configuration)
            logic.reaction_map = logic._build_reaction_map()
            logic.modes = logic._build_modes()
            logic.enemy_priority = logic._build_enemy_priority()
            logic.crashes = BASE.detect_enemy_crashes(logic.world.arrivals_by_planet, logic.state.player)
            sample = build_heuristic_sample(logic, logic._build_all_missions(), top_k=top_k)
            if sample is None:
                continue
            sample.update({
                "step": candidate.start_step,
                "episode_id": candidate.episode_id,
                "replay_path": candidate.replay_path,
                "final_reward": candidate.final_reward,
                "historical_margin_drop": candidate.historical_margin_drop,
            })
            samples.append(sample)
        except Exception:
            continue

    return samples


def pretrain_policy(
    policy: Any,
    samples: List[Dict[str, Any]],
    epochs: int = 20,
    learning_rate: float = 0.02,
) -> Dict[str, Any]:
    """Cross-entropy supervised pre-training toward heuristic ranking."""
    logs: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        for sample in samples:
            vectors = sample["vectors"]
            target = sample["target_index"]
            scores = [policy.score(v) for v in vectors]
            probs = _softmax(scores, policy.temperature)

            policy.update(
                [DecisionSample(
                    feature_vectors=[list(vector) for vector in vectors],
                    chosen_index=target,
                    probabilities=probs,
                    metadata={"pretrain": True},
                )],
                reward=1.0,
                learning_rate=learning_rate,
            )

            # Track accuracy
            predicted = max(range(len(scores)), key=lambda i: scores[i])
            if predicted == target:
                correct += 1
            # Cross-entropy loss
            prob_target = max(1e-10, probs[target])
            total_loss -= math.log(prob_target)

        accuracy = correct / max(1, len(samples))
        avg_loss = total_loss / max(1, len(samples))
        logs.append({"epoch": epoch, "accuracy": accuracy, "avg_loss": avg_loss})
        if epoch <= 3 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"[pretrain epoch {epoch:03d}] accuracy={accuracy:.3f} "
                f"loss={avg_loss:.4f} avg|w|={policy.average_abs_weight():.4f}",
                flush=True,
            )
    return {"logs": logs, "final_accuracy": logs[-1]["accuracy"] if logs else 0.0}


def evaluate_policy_accuracy(
    policy: Any,
    samples: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Measure how often the policy matches the pretrain target and heuristic choice."""
    correct_target = 0
    correct_same = 0
    defer_count = 0
    total = len(samples)

    for sample in samples:
        vectors = sample["vectors"]
        target = sample["target_index"]
        heuristic_best = sample["heuristic_best_idx"]

        scores = [policy.score(v) for v in vectors]
        predicted = max(range(len(scores)), key=lambda i: scores[i])

        if predicted == 0:
            defer_count += 1
        if predicted == target:
            correct_target += 1
        # Also check if RL picks the same underlying candidate as heuristic
        if predicted == 0 or predicted == heuristic_best + 1:
            correct_same += 1

    return {
        "target_accuracy": correct_target / max(1, total),
        "heuristic_agreement": correct_same / max(1, total),
        "defer_rate": defer_count / max(1, total),
        "samples": float(total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train RL policy to imitate heuristic ranking")
    parser.add_argument(
        "--replay-glob",
        nargs="+",
        default=["kaggle_replays/*/episode-*-replay.json"],
        help="Replay glob(s) to scan for training positions",
    )
    parser.add_argument("--player-name", default="alex chilton")
    parser.add_argument("--positions", type=int, default=80, help="Max training positions to collect")
    parser.add_argument("--epochs", type=int, default=30, help="Pre-training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.02, help="Pre-training learning rate")
    parser.add_argument("--min-step", type=int, default=24, help="Earliest replay step to sample for pre-training")
    parser.add_argument("--max-step", type=int, default=180, help="Latest replay step to sample for pre-training")
    parser.add_argument("--top-k", type=int, default=8, help="Top heuristic missions exposed per training state")
    parser.add_argument("--start-rank-max", type=int, default=2)
    parser.add_argument("--min-start-share-2p", type=float, default=0.42)
    parser.add_argument("--min-start-share-4p", type=float, default=0.22)
    parser.add_argument("--min-margin-drop", type=float, default=0.02)
    parser.add_argument("--candidates-per-replay", type=int, default=3)
    parser.add_argument("--losses-only", action="store_true", help="Exclude winning replays from heuristic imitation samples")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy-model", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--policy-in", help="Optional existing policy to evaluate instead of training")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument(
        "--policy-out",
        default=str(WORKSPACE_DIR / "results" / "pretrained_policy.json"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(WORKSPACE_DIR / "results" / "pretrain_summary.json"),
    )
    args = parser.parse_args()

    replay_paths: List[Path] = []
    for pattern in args.replay_glob:
        replay_paths.extend(sorted(ROOT.glob(pattern)))
    replay_paths = sorted(set(replay_paths))
    if not replay_paths:
        raise SystemExit("No replay files matched")

    print(f"Collecting heuristic ranking samples from {len(replay_paths)} replays...", flush=True)
    samples = collect_heuristic_ranking_samples(
        replay_paths,
        player_name=args.player_name,
        max_positions=args.positions,
        min_step=args.min_step,
        max_step=args.max_step,
        top_k=args.top_k,
        seed=args.seed,
        start_rank_max=args.start_rank_max,
        min_start_share_two_player=args.min_start_share_2p,
        min_start_share_multi=args.min_start_share_4p,
        min_margin_drop=args.min_margin_drop,
        candidates_per_replay=args.candidates_per_replay,
        include_wins=not args.losses_only,
    )
    print(f"Collected {len(samples)} training positions", flush=True)
    if not samples:
        raise SystemExit("No training positions found")

    if args.eval_only and args.policy_in:
        policy = load_policy_json(args.policy_in)
        metrics = evaluate_policy_accuracy(policy, samples)
        print(f"Evaluation: {metrics}", flush=True)
        return

    policy = (
        load_policy_json(args.policy_in)
        if args.policy_in
        else create_policy(args.policy_model, hidden_size=args.hidden_size, seed=args.seed)
    )

    # Evaluate before pre-training
    before = evaluate_policy_accuracy(policy, samples)
    print(f"Before pre-training: {before}", flush=True)

    result = pretrain_policy(policy, samples, epochs=args.epochs, learning_rate=args.learning_rate)

    # Evaluate after pre-training
    after = evaluate_policy_accuracy(policy, samples)
    print(f"After pre-training:  {after}", flush=True)

    policy_path = policy.save_json(args.policy_out)
    print(f"Saved pre-trained policy to {policy_path}", flush=True)

    summary = {
        "samples": len(samples),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "before": before,
        "after": after,
        "training": result["logs"],
        "policy": str(policy_path),
    }
    output = Path(args.summary_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
