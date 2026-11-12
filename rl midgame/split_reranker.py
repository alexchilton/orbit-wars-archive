"""Split reranker: separate 2-player and 4-player policies with runtime dispatch.

Trains independent policies for 2-player and 4-player games, then wraps
them in a SplitRerankerPolicy that routes candidates to the correct policy
based on the ``is_two_player`` feature (index 2 in FEATURE_NAMES).

Usage
-----
    python "rl midgame/split_reranker.py" \
        --replay-glob "kaggle_replays/*/episode-*-replay.json" \
        --player-name "alex chilton" \
        --policy-out-combined "rl midgame/results/split_policy.json"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

from midgame_features import FEATURE_NAMES  # noqa: E402
from midgame_policy import (  # noqa: E402
    DecisionSample,
    LinearMissionPolicy,
    MLPMissionPolicy,
    PolicyChoice,
    create_policy,
    load_policy_json,
)
from midgame_rl_agent import MidgameRLConfig, build_agent, load_policy_json as _load_policy  # noqa: E402
from pretrain_from_heuristic import (  # noqa: E402
    collect_heuristic_ranking_samples,
    pretrain_policy,
)
from replay_midgame_experiment import (  # noqa: E402
    _load_json,
    evaluate_candidates,
    select_replay_candidates,
    train_on_candidates,
)

# is_two_player is at index 2 in the feature vector
_IS_TWO_PLAYER_INDEX = FEATURE_NAMES.index("is_two_player")


# ---------------------------------------------------------------------------
# Replay filtering
# ---------------------------------------------------------------------------

def filter_replays_by_player_count(
    replay_paths: Sequence[Path],
    target_count: int,
) -> List[Path]:
    """Return replays matching *target_count* agents.

    *target_count* == 2 keeps strictly 2-player replays.
    *target_count* >= 4 keeps replays with 4 or more agents.
    """
    filtered: List[Path] = []
    for path in replay_paths:
        try:
            replay = _load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        steps = replay.get("steps") or []
        if not steps:
            continue
        num_agents = len(steps[0])
        if target_count == 2 and num_agents == 2:
            filtered.append(path)
        elif target_count >= 4 and num_agents >= 4:
            filtered.append(path)
    return filtered


# ---------------------------------------------------------------------------
# SplitRerankerPolicy
# ---------------------------------------------------------------------------

class SplitRerankerPolicy:
    """Dispatch wrapper that selects a 2-player or 4-player policy at runtime."""

    def __init__(
        self,
        policy_2p: Any,
        policy_4p: Any,
        temperature: float = 1.0,
    ) -> None:
        self.policy_2p = policy_2p
        self.policy_4p = policy_4p
        self.temperature = temperature

    # -- dispatch helper ---------------------------------------------------

    def _select(self, vector: Sequence[float]) -> Any:
        """Pick sub-policy based on the is_two_player feature."""
        if vector[_IS_TWO_PLAYER_INDEX] > 0.5:
            return self.policy_2p
        return self.policy_4p

    # -- public interface (compatible with LinearMissionPolicy) -------------

    def score(self, vector: Sequence[float]) -> float:
        return self._select(vector).score(vector)

    def choose(
        self,
        feature_vectors: Sequence[Sequence[float]],
        rng: Optional[random.Random] = None,
        explore: bool = False,
    ) -> PolicyChoice:
        if not feature_vectors:
            return PolicyChoice(index=0, scores=[], probabilities=[])
        policy = self._select(feature_vectors[0])
        return policy.choose(feature_vectors, rng=rng, explore=explore)

    def update(
        self,
        samples: Sequence[DecisionSample],
        reward: float,
        learning_rate: float,
    ) -> Dict[str, float]:
        """Route updates to the appropriate sub-policy per sample."""
        samples_2p: List[DecisionSample] = []
        samples_4p: List[DecisionSample] = []
        for sample in samples:
            if sample.feature_vectors and sample.feature_vectors[0][_IS_TWO_PLAYER_INDEX] > 0.5:
                samples_2p.append(sample)
            else:
                samples_4p.append(sample)
        info: Dict[str, float] = {}
        if samples_2p:
            info.update({f"2p_{k}": v for k, v in self.policy_2p.update(samples_2p, reward, learning_rate).items()})
        if samples_4p:
            info.update({f"4p_{k}": v for k, v in self.policy_4p.update(samples_4p, reward, learning_rate).items()})
        return info

    def average_abs_weight(self) -> float:
        return (self.policy_2p.average_abs_weight() + self.policy_4p.average_abs_weight()) / 2.0

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "split_reranker",
            "temperature": self.temperature,
            "policy_2p": self.policy_2p.to_dict(),
            "policy_4p": self.policy_4p.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SplitRerankerPolicy":
        def _restore(sub: Dict[str, Any]) -> Any:
            kind = sub.get("type", "linear")
            if kind == "mlp":
                return MLPMissionPolicy.from_dict(sub)
            return LinearMissionPolicy.from_dict(sub)

        return cls(
            policy_2p=_restore(payload["policy_2p"]),
            policy_4p=_restore(payload["policy_4p"]),
            temperature=payload.get("temperature", 1.0),
        )

    def save_json(self, path: str | Path) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return dest

    @classmethod
    def load_json(cls, path: str | Path) -> "SplitRerankerPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train_split_policies(
    replay_paths: Sequence[Path],
    player_name: str,
    policy_model: str = "linear",
    hidden_size: int = 64,
    pretrain_epochs: int = 20,
    pretrain_lr: float = 0.02,
    finetune_episodes: int = 8,
    finetune_lr: float = 0.005,
    max_positions: int = 100,
    horizon: int = 50,
    max_candidates: int = 3,
    seed: int = 7,
    rl_config: Optional[MidgameRLConfig] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Train separate 2-player and 4-player policies.

    Returns ``(policy_2p, policy_4p, training_summary)``.
    """
    rl_config = rl_config or MidgameRLConfig(explore=True)
    summary: Dict[str, Any] = {}

    paths_2p = filter_replays_by_player_count(replay_paths, target_count=2)
    paths_4p = filter_replays_by_player_count(replay_paths, target_count=4)
    summary["replay_counts"] = {"2p": len(paths_2p), "4p": len(paths_4p)}
    print(f"Split replays: {len(paths_2p)} two-player, {len(paths_4p)} four-player+", flush=True)

    results: Dict[str, Any] = {}
    policies: Dict[str, Any] = {}

    for label, paths in [("2p", paths_2p), ("4p", paths_4p)]:
        print(f"\n{'='*60}\nTraining {label} policy ({len(paths)} replays)\n{'='*60}", flush=True)
        policy = create_policy(policy_model, hidden_size=hidden_size, seed=seed)
        bucket: Dict[str, Any] = {}

        if not paths:
            print(f"  [skip] no {label} replays available", flush=True)
            policies[label] = policy
            results[label] = {"skipped": True}
            continue

        # --- pretrain from heuristic samples ---
        samples = collect_heuristic_ranking_samples(
            paths,
            player_name=player_name,
            max_positions=max_positions,
            seed=seed,
        )
        print(f"  [{label}] collected {len(samples)} heuristic positions", flush=True)

        if samples:
            pretrain_result = pretrain_policy(
                policy, samples, epochs=pretrain_epochs, learning_rate=pretrain_lr,
            )
            bucket["pretrain"] = pretrain_result
        else:
            bucket["pretrain"] = {"skipped": True}

        # --- RL fine-tune on replay candidates ---
        candidates = select_replay_candidates(
            paths,
            player_name=player_name,
            horizon=horizon,
            max_candidates=max_candidates,
            include_wins=True,
        )
        print(f"  [{label}] selected {len(candidates)} fine-tune candidates", flush=True)

        if candidates:
            replay_by_path: Dict[str, Any] = {}
            for p in paths:
                replay_by_path.setdefault(str(p), _load_json(p))
            trained = train_on_candidates(
                policy,
                replay_by_path,
                candidates,
                episodes=finetune_episodes,
                learning_rate=finetune_lr,
                horizon=horizon,
                rl_config=rl_config,
            )
            policy = trained["policy"]
            bucket["finetune"] = trained.get("logs", [])
        else:
            bucket["finetune"] = {"skipped": True}

        policies[label] = policy
        results[label] = bucket

    summary["training"] = results
    return policies["2p"], policies["4p"], summary


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def build_split_agent(
    policy_2p_path: str | Path,
    policy_4p_path: str | Path,
    rl_config: Optional[MidgameRLConfig] = None,
    explore: Optional[bool] = None,
) -> Callable[[Any, Any], List[List[float | int]]]:
    """Create a game agent that dispatches to the correct split policy."""
    combined = SplitRerankerPolicy(
        policy_2p=load_policy_json(policy_2p_path),
        policy_4p=load_policy_json(policy_4p_path),
    )
    return build_agent(policy=combined, rl_config=rl_config, explore=explore)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train separate 2-player and 4-player reranker policies",
    )
    parser.add_argument(
        "--replay-glob",
        nargs="+",
        default=["kaggle_replays/*/episode-*-replay.json"],
        help="Glob patterns for replay files (relative to project root)",
    )
    parser.add_argument("--player-name", default="alex chilton")
    parser.add_argument("--policy-model", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20, help="Pre-training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.02, help="Pre-training LR")
    parser.add_argument("--finetune-episodes", type=int, default=8)
    parser.add_argument("--finetune-lr", type=float, default=0.005)
    parser.add_argument("--max-positions", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--policy-out-2p",
        default=str(WORKSPACE_DIR / "results" / "split_policy_2p.json"),
    )
    parser.add_argument(
        "--policy-out-4p",
        default=str(WORKSPACE_DIR / "results" / "split_policy_4p.json"),
    )
    parser.add_argument(
        "--policy-out-combined",
        default=str(WORKSPACE_DIR / "results" / "split_policy.json"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(WORKSPACE_DIR / "results" / "split_reranker_summary.json"),
    )
    args = parser.parse_args()

    # Resolve replay paths
    replay_paths: List[Path] = []
    for pattern in args.replay_glob:
        replay_paths.extend(sorted(ROOT.glob(pattern)))
    replay_paths = sorted(set(replay_paths))
    if not replay_paths:
        raise SystemExit("No replay files matched")
    print(f"Found {len(replay_paths)} total replays", flush=True)

    # Train
    policy_2p, policy_4p, training_summary = train_split_policies(
        replay_paths,
        player_name=args.player_name,
        policy_model=args.policy_model,
        hidden_size=args.hidden_size,
        pretrain_epochs=args.epochs,
        pretrain_lr=args.learning_rate,
        finetune_episodes=args.finetune_episodes,
        finetune_lr=args.finetune_lr,
        max_positions=args.max_positions,
        horizon=args.horizon,
        max_candidates=args.max_candidates,
        seed=args.seed,
    )

    # Save individual policies
    p2p = policy_2p.save_json(args.policy_out_2p)
    p4p = policy_4p.save_json(args.policy_out_4p)
    print(f"Saved 2-player policy to {p2p}", flush=True)
    print(f"Saved 4-player policy to {p4p}", flush=True)

    # Save combined split policy
    combined = SplitRerankerPolicy(policy_2p=policy_2p, policy_4p=policy_4p)
    combined_path = combined.save_json(args.policy_out_combined)
    print(f"Saved combined split policy to {combined_path}", flush=True)

    # Write summary
    training_summary["policy_2p"] = str(p2p)
    training_summary["policy_4p"] = str(p4p)
    training_summary["policy_combined"] = str(combined_path)
    output = Path(args.summary_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(training_summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
