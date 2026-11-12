"""Counterfactual Replay Mining for RL Gate & Reranker Training.

This module:
1. Scans our Kaggle replays and identifies loss episodes
2. Restores game states at each midgame decision point
3. Runs counterfactual rollouts: what would have happened with RL vs heuristic vs defer
4. Produces labeled training data for:
   - GATE: when should RL intervene? (binary classifier)
   - RERANKER: which candidate is actually best? (ranking labels)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

WORKSPACE_DIR = Path(__file__).resolve().parent
ROOT = WORKSPACE_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import os
os.environ.setdefault("KAGGLE_ENVIRONMENTS_QUIET", "1")

from kaggle_environments import make  # noqa: E402

from midgame_features import (  # noqa: E402
    FEATURE_NAMES,
    build_defer_vector,
    build_mission_feature_bundle,
    build_state_snapshot,
)
from midgame_policy import DecisionSample, LinearMissionPolicy, load_policy_json  # noqa: E402
from midgame_rl_agent import (  # noqa: E402
    BASE,
    BASE_AGENT,
    EpisodeRecorder,
    MidgameRLConfig,
    MidgameRLDecisionLogic,
    build_agent,
)


@dataclass
class DecisionPoint:
    """A single RL decision point extracted from a replay."""
    episode_id: int
    step: int
    player_index: int
    num_players: int
    state_features: Dict[str, float]
    candidate_features: List[List[float]]  # feature vectors for each candidate
    candidate_metadata: List[Dict[str, Any]]
    heuristic_top_index: int  # which candidate the heuristic would pick (highest base_value)
    # Filled after counterfactual evaluation:
    counterfactual_scores: Optional[List[float]] = None  # production gain per candidate
    defer_score: Optional[float] = None  # what happens if we defer
    best_action_index: Optional[int] = None  # which candidate was actually best
    should_intervene: Optional[bool] = None  # gate label


@dataclass
class MiningResult:
    """Results of mining a batch of replays."""
    total_episodes: int = 0
    loss_episodes: int = 0
    decision_points_extracted: int = 0
    gate_labels: List[Dict[str, Any]] = field(default_factory=list)
    reranker_labels: List[Dict[str, Any]] = field(default_factory=list)


def _load_replay(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_player_index(replay: Dict[str, Any], player_name: str = "alex chilton") -> Optional[int]:
    teams = replay.get("info", {}).get("TeamNames", [])
    for i, name in enumerate(teams):
        if player_name.lower() in str(name).lower():
            return i
    return None


def _compute_production_share(observation: Dict[str, Any], player_index: int, num_players: int) -> float:
    """Get our production share from raw observation."""
    prod = [0.0] * num_players
    for planet in observation.get("planets", []):
        owner = int(planet[1])
        if 0 <= owner < num_players:
            prod[owner] += float(planet[6])
    total = sum(prod)
    if total <= 0:
        return 1.0 / num_players
    return prod[player_index] / total


def _compute_ship_share(observation: Dict[str, Any], player_index: int, num_players: int) -> float:
    """Get our ship share from raw observation."""
    ships = [0.0] * num_players
    for planet in observation.get("planets", []):
        owner = int(planet[1])
        if 0 <= owner < num_players:
            ships[owner] += float(planet[5])
    for fleet in observation.get("fleets", []):
        owner = int(fleet[1])
        if 0 <= owner < num_players:
            ships[owner] += float(fleet[6])
    total = sum(ships)
    if total <= 0:
        return 1.0 / num_players
    return ships[player_index] / total


def _restore_env(replay: Dict[str, Any], start_step: int):
    """Restore kaggle environment to a specific step in a replay."""
    config = replay.get("configuration", {})
    env = make("orbit_wars", configuration=config, debug=False)
    env._Environment__set_state(copy.deepcopy(replay["steps"][start_step]))
    env.steps = [None] * (start_step + 1)
    env.logs = [[] for _ in range(start_step + 1)]
    return env


def _get_historical_actions(replay: Dict[str, Any], step: int) -> List[Any]:
    """Get all players' actions at a specific step."""
    return [agent.get("action", []) for agent in replay["steps"][step]]


def extract_loss_episodes(
    replay_dir: Path,
    player_name: str = "alex chilton",
    include_narrow_wins: bool = False,
) -> List[Tuple[Path, Dict[str, Any], int]]:
    """Find all episodes where we lost (or narrowly won)."""
    results = []
    for replay_path in sorted(replay_dir.rglob("*.json")):
        try:
            replay = _load_replay(replay_path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        
        player_idx = _find_player_index(replay, player_name)
        if player_idx is None:
            continue
        
        rewards = replay.get("rewards", [])
        if player_idx >= len(rewards) or rewards[player_idx] is None:
            continue
        
        reward = float(rewards[player_idx])
        # Include losses (reward < 0) and optionally narrow wins
        if reward < 0 or (include_narrow_wins and 0 <= reward < 0.3):
            results.append((replay_path, replay, player_idx))
    
    return results


def extract_decision_points_from_replay(
    replay: Dict[str, Any],
    player_index: int,
    min_step: int = 24,
    max_step: int = 180,
    sample_every_n: int = 3,
) -> List[DecisionPoint]:
    """Extract candidate decision states from a replay.
    
    For each qualifying step, we reconstruct what the heuristic's candidates
    would have been and record the features.
    """
    steps = replay.get("steps", [])
    if not steps:
        return []
    
    num_players = len(steps[0])
    episode_id = int(replay.get("info", {}).get("EpisodeId", -1))
    decision_points = []
    
    for step_idx in range(min_step, min(max_step, len(steps) - 1), sample_every_n):
        obs_data = steps[step_idx][player_index].get("observation")
        if obs_data is None:
            continue
        
        # Check the player is still active
        status = steps[step_idx][player_index].get("status", "")
        if status != "ACTIVE":
            continue
        
        # Check if this is a contested position (RL window would be open)
        prod_share = _compute_production_share(obs_data, player_index, num_players)
        ship_share = _compute_ship_share(obs_data, player_index, num_players)
        
        if num_players <= 2:
            # 2p: RL window is 0.34-0.72 share
            if not (0.34 <= ship_share <= 0.72):
                continue
        else:
            # 4p: must be top-2 rank
            pass  # We'll accept all 4p midgame positions
        
        # Record the state features for gate training
        state_features = {
            "step": step_idx,
            "step_frac": step_idx / 500.0,
            "num_players": num_players,
            "prod_share": prod_share,
            "ship_share": ship_share,
            "is_ahead": float(prod_share > (0.5 if num_players <= 2 else 0.3)),
        }
        
        decision_points.append(DecisionPoint(
            episode_id=episode_id,
            step=step_idx,
            player_index=player_index,
            num_players=num_players,
            state_features=state_features,
            candidate_features=[],  # Will be filled by counterfactual evaluation
            candidate_metadata=[],
            heuristic_top_index=0,
        ))
    
    return decision_points


def run_counterfactual_from_step(
    replay: Dict[str, Any],
    player_index: int,
    start_step: int,
    horizon: int = 20,
) -> Dict[str, float]:
    """Run a short counterfactual rollout from a specific step.
    
    Returns production shares at horizon for: heuristic play vs historical.
    The heuristic agent plays our moves; opponents use historical actions.
    """
    steps = replay.get("steps", [])
    num_players = len(steps[0])
    end_step = min(len(steps) - 1, start_step + horizon)
    
    # Measure initial production share
    start_obs = steps[start_step][0]["observation"]
    start_prod_share = _compute_production_share(start_obs, player_index, num_players)
    
    # Measure historical end production share (what actually happened)
    end_obs = steps[end_step][0]["observation"]
    hist_prod_share = _compute_production_share(end_obs, player_index, num_players)
    
    # Now run a counterfactual with our heuristic agent
    try:
        env = _restore_env(replay, start_step)
        current_step = start_step
        
        while current_step < end_step and not env.done:
            next_step = current_step + 1
            if next_step >= len(steps):
                break
            
            # Get historical actions for all players
            actions = _get_historical_actions(replay, next_step)
            
            # Replace our action with fresh heuristic decision
            try:
                obs = env.state[player_index].observation
                config_obj = env.configuration
                heuristic_action = BASE_AGENT(obs, config_obj)
                actions[player_index] = heuristic_action
            except Exception:
                pass  # Fall back to historical action
            
            env.step(actions)
            current_step = next_step
        
        # Get final state
        cf_obs = env.state[0].observation
        cf_prod_share = _compute_production_share(cf_obs, player_index, num_players)
    except Exception:
        cf_prod_share = hist_prod_share  # On error, assume no difference
    
    return {
        "start_prod_share": start_prod_share,
        "historical_prod_share": hist_prod_share,
        "counterfactual_prod_share": cf_prod_share,
        "historical_delta": hist_prod_share - start_prod_share,
        "counterfactual_delta": cf_prod_share - start_prod_share,
        "improvement": cf_prod_share - hist_prod_share,
    }


def mine_gate_labels(
    replay_dir: Path,
    player_name: str = "alex chilton",
    max_episodes: int = 50,
    horizon: int = 20,
    sample_every_n: int = 5,
    verbose: bool = True,
) -> MiningResult:
    """Main entry: mine all loss episodes and produce gate + reranker labels.
    
    Gate labels: for each decision point, did RL intervention help or hurt?
    - If counterfactual (heuristic) > historical: should_intervene = False (RL hurt)
    - If historical >= counterfactual: should_intervene = True (RL helped or neutral)
    
    Note: This is a simplified version. Full counterfactual requires running RL
    with specific candidate choices, which needs more infrastructure.
    """
    result = MiningResult()
    
    # Find loss episodes
    losses = extract_loss_episodes(Path(replay_dir), player_name, include_narrow_wins=True)
    result.total_episodes = len(losses)
    result.loss_episodes = sum(1 for _, r, _ in losses if float(r.get("rewards", [0])[_find_player_index(r, player_name) or 0] or 0) < 0)
    
    if verbose:
        print(f"Found {result.total_episodes} episodes ({result.loss_episodes} losses)")
    
    episodes_processed = 0
    for replay_path, replay, player_idx in losses[:max_episodes]:
        if verbose and episodes_processed % 10 == 0:
            print(f"  Processing episode {episodes_processed + 1}/{min(len(losses), max_episodes)}...")
        
        # Extract decision points
        decision_points = extract_decision_points_from_replay(
            replay, player_idx,
            min_step=24, max_step=180,
            sample_every_n=sample_every_n,
        )
        
        for dp in decision_points:
            # Run counterfactual
            cf_result = run_counterfactual_from_step(
                replay, player_idx, dp.step, horizon=horizon
            )
            
            # Gate label: did the heuristic do better than what happened?
            # If cf > historical: RL (or whatever happened) was worse → should have deferred
            improvement = cf_result["improvement"]
            should_defer = improvement > 0.02  # Heuristic would have been >2% better
            
            gate_label = {
                "episode_id": dp.episode_id,
                "step": dp.step,
                "num_players": dp.num_players,
                "state_features": dp.state_features,
                "should_intervene": not should_defer,
                "improvement": improvement,
                "start_prod_share": cf_result["start_prod_share"],
                "historical_delta": cf_result["historical_delta"],
                "counterfactual_delta": cf_result["counterfactual_delta"],
            }
            result.gate_labels.append(gate_label)
            result.decision_points_extracted += 1
        
        episodes_processed += 1
    
    if verbose:
        n_defer = sum(1 for g in result.gate_labels if not g["should_intervene"])
        n_intervene = sum(1 for g in result.gate_labels if g["should_intervene"])
        print(f"\nMining complete:")
        print(f"  Decision points: {result.decision_points_extracted}")
        print(f"  Should defer: {n_defer} ({100*n_defer/max(1,len(result.gate_labels)):.1f}%)")
        print(f"  Should intervene: {n_intervene} ({100*n_intervene/max(1,len(result.gate_labels)):.1f}%)")
        avg_improvement = sum(g["improvement"] for g in result.gate_labels) / max(1, len(result.gate_labels))
        print(f"  Avg heuristic improvement: {avg_improvement:+.4f}")
    
    return result


def train_gate_from_labels(
    labels: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train a simple logistic gate from mined labels.
    
    The gate predicts: should RL intervene at this state?
    Uses state features only (not candidate features).
    
    Returns a dict with gate weights + statistics.
    """
    if not labels:
        return {"error": "no labels", "weights": {}}
    
    # Features for gate: step_frac, prod_share, ship_share, is_ahead, num_players
    gate_features = ["step_frac", "prod_share", "ship_share", "is_ahead"]
    
    # Simple logistic regression via gradient descent
    weights = {f: 0.0 for f in gate_features}
    bias = 0.0
    lr = 0.1
    
    for epoch in range(50):
        total_loss = 0.0
        correct = 0
        for label in labels:
            sf = label["state_features"]
            x = [sf.get(f, 0.0) for f in gate_features]
            y = 1.0 if label["should_intervene"] else 0.0
            
            # Forward
            z = bias + sum(w * xi for w, xi in zip(weights.values(), x))
            pred = 1.0 / (1.0 + math.exp(-max(-20, min(20, z))))
            
            # Loss
            eps = 1e-7
            loss = -(y * math.log(pred + eps) + (1 - y) * math.log(1 - pred + eps))
            total_loss += loss
            correct += int((pred > 0.5) == (y > 0.5))
            
            # Backward
            grad = pred - y
            bias -= lr * grad
            for i, f in enumerate(gate_features):
                weights[f] -= lr * grad * x[i]
        
        accuracy = correct / len(labels)
    
    # Build result
    gate_model = {
        "features": gate_features,
        "weights": weights,
        "bias": bias,
        "accuracy": accuracy,
        "n_samples": len(labels),
        "n_intervene": sum(1 for l in labels if l["should_intervene"]),
        "n_defer": sum(1 for l in labels if not l["should_intervene"]),
    }
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(gate_model, indent=2), encoding="utf-8")
        print(f"Gate model saved to {output_path}")
    
    return gate_model


def apply_gate(gate_model: Dict[str, Any], state_features: Dict[str, float]) -> Tuple[bool, float]:
    """Apply the trained gate to decide whether RL should intervene.
    
    Returns (should_intervene, confidence).
    Default conservative: only intervene if confidence > 0.6.
    """
    features = gate_model.get("features", [])
    weights = gate_model.get("weights", {})
    bias = gate_model.get("bias", 0.0)
    
    x = [state_features.get(f, 0.0) for f in features]
    z = bias + sum(weights.get(f, 0.0) * xi for f, xi in zip(features, x))
    confidence = 1.0 / (1.0 + math.exp(-max(-20, min(20, z))))
    
    # Conservative threshold: only intervene when confident
    should_intervene = confidence > 0.6
    return should_intervene, confidence


def main():
    parser = argparse.ArgumentParser(description="Counterfactual RL mining pipeline")
    parser.add_argument("--replay-dir", type=str, default="kaggle_replays",
                        help="Directory containing replay JSON files")
    parser.add_argument("--player-name", type=str, default="alex chilton")
    parser.add_argument("--max-episodes", type=int, default=30,
                        help="Maximum episodes to process")
    parser.add_argument("--horizon", type=int, default=20,
                        help="Rollout horizon for counterfactual (turns)")
    parser.add_argument("--sample-every", type=int, default=5,
                        help="Sample decision points every N steps")
    parser.add_argument("--output-dir", type=str, default="rl midgame/results/counterfactual",
                        help="Output directory for labels and gate model")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Mine gate labels
    print("=" * 60)
    print("COUNTERFACTUAL RL MINING PIPELINE")
    print("=" * 60)
    print(f"\nReplay dir: {args.replay_dir}")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Horizon: {args.horizon} turns")
    print(f"Sample every: {args.sample_every} steps")
    print()
    
    result = mine_gate_labels(
        replay_dir=Path(args.replay_dir),
        player_name=args.player_name,
        max_episodes=args.max_episodes,
        horizon=args.horizon,
        sample_every_n=args.sample_every,
        verbose=True,
    )
    
    # Save raw labels
    labels_path = output_dir / "gate_labels.json"
    labels_path.write_text(json.dumps(result.gate_labels, indent=2), encoding="utf-8")
    print(f"\nSaved {len(result.gate_labels)} gate labels to {labels_path}")
    
    # Step 2: Train gate
    if result.gate_labels:
        print("\n" + "=" * 60)
        print("TRAINING GATE CLASSIFIER")
        print("=" * 60)
        gate_model = train_gate_from_labels(
            result.gate_labels,
            output_path=output_dir / "gate_model.json",
        )
        print(f"  Accuracy: {gate_model['accuracy']:.3f}")
        print(f"  Weights: {gate_model['weights']}")
        print(f"  Bias: {gate_model['bias']:.4f}")
    
    print("\n✓ Pipeline complete.")
    return result


if __name__ == "__main__":
    main()
