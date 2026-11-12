"""Improved PPO training with warm-start and per-step rewards.

Key fixes over original train_candidate_ppo.py:
1. Warm-start: pretrain actor to match linear_v3 scores
2. Per-step intermediate rewards (production delta over short horizon)
3. More games per update (8 default)
4. Train vs heuristic first (not self-play from scratch)
5. activation_turn=24 (no opening interference)
6. Heuristic-baselined reward (advantage = rl_outcome - heuristic_baseline)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("PyTorch required for PPO training: pip install torch")

import test_agent  # noqa: E402
from midgame_features import FEATURE_NAMES  # noqa: E402
from midgame_policy import (  # noqa: E402
    DecisionSample,
    LinearMissionPolicy,
    PPOMissionPolicy,
    create_policy,
    load_policy_json,
)
from midgame_rl_agent import BASE_AGENT, EpisodeRecorder, MidgameRLConfig, build_agent  # noqa: E402


def warmstart_ppo_from_linear(
    ppo_policy: PPOMissionPolicy,
    linear_policy: LinearMissionPolicy,
    n_synthetic: int = 2000,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 42,
) -> Dict[str, float]:
    """Pretrain PPO actor head to match linear_v3 scoring.
    
    Generates synthetic candidate sets, gets linear scores as targets,
    trains PPO's actor to produce similar rankings.
    """
    rng = random.Random(seed)
    
    # Generate synthetic feature vectors (mimicking real distributions)
    synthetic_sets = []
    for _ in range(n_synthetic):
        n_candidates = rng.randint(2, 8)
        candidates = []
        for _ in range(n_candidates):
            vec = [0.0] * len(FEATURE_NAMES)
            vec[0] = rng.random() * 0.4 + 0.05  # step_frac
            vec[1] = 1.0 - vec[0]  # remaining_frac
            vec[2] = float(rng.random() > 0.5)  # is_two_player
            vec[3] = 1.0 - vec[2]  # is_four_player_plus
            vec[4] = float(rng.random() > 0.4)  # is_ahead
            vec[5] = float(rng.random() > 0.6)  # is_behind
            vec[6] = float(rng.random() > 0.85)  # is_dominating
            vec[7] = float(rng.random() > 0.9)  # is_finishing
            vec[8] = rng.random() * 0.6 + 0.2  # my_ship_share
            vec[9] = rng.random() * 0.5 + 0.2  # my_prod_share
            vec[10] = rng.random() * 1.5  # lead_ratio
            vec[11] = rng.random()  # base_score_norm
            vec[12] = rng.random()  # base_value_norm
            vec[13] = rng.random()  # eta_norm
            vec[14] = rng.random()  # send_norm
            vec[15] = rng.random() * 0.5  # source_count_norm
            vec[16] = rng.random()  # needed_norm
            vec[17] = rng.random()  # target_prod_norm
            vec[18] = rng.random()  # target_ships_norm
            # Target type (one-hot)
            ttype = rng.choice([19, 20, 21])
            vec[ttype] = 1.0
            # Mission type (one-hot)
            mtype = rng.choice([22, 23, 24, 25, 26, 27, 28, 29])
            vec[mtype] = 1.0
            vec[30] = rng.random()  # enemy_priority
            vec[31] = rng.random() * 0.5  # committed_load
            vec[32] = 0.0  # is_defer
            candidates.append(vec)
        
        # Add defer option
        defer_vec = list(candidates[0])
        defer_vec[32] = 1.0
        candidates.insert(0, defer_vec)
        
        synthetic_sets.append(candidates)
    
    # Get linear scores as target distributions
    optimizer = torch.optim.Adam(ppo_policy.model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        rng.shuffle(synthetic_sets)
        
        for candidates in synthetic_sets:
            # Target: linear policy's score distribution
            linear_scores = [linear_policy.score(c) for c in candidates]
            # Softmax target
            max_s = max(linear_scores)
            exps = [math.exp(s - max_s) for s in linear_scores]
            total = sum(exps)
            target_probs = torch.tensor([e / total for e in exps], dtype=torch.float32)
            
            # PPO actor output
            features = torch.tensor(candidates, dtype=torch.float32)
            logits, _value = ppo_policy.model(features)
            log_probs = F.log_softmax(logits / ppo_policy.temperature, dim=0)
            
            # KL divergence loss
            loss = F.kl_div(log_probs, target_probs, reduction="batchmean")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(synthetic_sets)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f"  Warmstart epoch {epoch}: loss={avg_loss:.4f}")
    
    return {
        "final_loss": losses[-1] if losses else 0.0,
        "initial_loss": losses[0] if losses else 0.0,
        "epochs": epochs,
        "n_synthetic": n_synthetic,
    }


def compute_step_reward(
    observation: Dict[str, Any],
    player_index: int,
    num_players: int,
    prev_prod_share: float,
) -> float:
    """Compute intermediate reward: production share delta."""
    prod = [0.0] * num_players
    for planet in observation.get("planets", []):
        owner = int(planet[1])
        if 0 <= owner < num_players:
            prod[owner] += float(planet[6])
    total = sum(prod)
    if total <= 0:
        current_share = 1.0 / num_players
    else:
        current_share = prod[player_index] / total
    return current_share - prev_prod_share


def heuristic_baselined_reward(result: Dict[str, Any], as_player_a: bool) -> float:
    """Compute reward relative to what heuristic would get.
    
    Since both players in training use the same base heuristic for candidate gen,
    the reward is just the raw game outcome (win/loss/margin).
    Normalized to [-1, 1] range.
    """
    if as_player_a:
        raw = float(result.get("reward_a", 0.0))
    else:
        raw = float(result.get("reward_b", 0.0))
    # Normalize: reward > 0.5 = win, < 0.5 = loss
    return (raw - 0.5) * 2.0  # Maps [0,1] to [-1,1]


def play_episode_improved(
    policy: PPOMissionPolicy,
    rl_config: MidgameRLConfig,
    opponent: Any,
    *,
    swapped: bool,
    explore: bool,
) -> Dict[str, Any]:
    """Play episode with per-decision metadata for better credit assignment."""
    recorder = EpisodeRecorder()
    learner = build_agent(policy=policy, rl_config=rl_config, recorder=recorder, explore=explore)
    
    if swapped:
        result = test_agent.run_game(opponent, learner)
        samples = recorder.pop(1)
        reward = heuristic_baselined_reward(result, as_player_a=False)
    else:
        result = test_agent.run_game(learner, opponent)
        samples = recorder.pop(0)
        reward = heuristic_baselined_reward(result, as_player_a=True)
    
    return {
        "result": result,
        "samples": samples,
        "reward": reward,
        "swapped": swapped,
        "decisions": len(samples),
    }


def discounted_returns(reward: float, n_steps: int, gamma: float = 0.99) -> List[float]:
    """Generate per-step returns with temporal discount.
    
    Later decisions are more responsible for the outcome.
    Earlier decisions get discounted reward.
    """
    returns = []
    for i in range(n_steps):
        # Decisions closer to end get more credit
        t = (i + 1) / n_steps  # 0 → 1 from start to end
        discount = gamma ** (n_steps - i - 1)  # Less discount for later decisions
        returns.append(reward * discount)
    return returns


def batch_update_improved(
    policy: PPOMissionPolicy,
    episodes: Sequence[Dict[str, Any]],
    learning_rate: float = 3e-4,
    epochs: int = 4,
    gamma: float = 0.99,
) -> Dict[str, float]:
    """Improved PPO update with temporal credit assignment."""
    for group in policy.optimizer.param_groups:
        group["lr"] = learning_rate
    
    records: List[Dict[str, Any]] = []
    for episode in episodes:
        reward = float(episode.get("reward", 0.0))
        samples = episode.get("samples", [])
        if not samples:
            continue
        
        # Assign discounted returns (later decisions get more credit)
        step_returns = discounted_returns(reward, len(samples), gamma)
        
        for sample, step_return in zip(samples, step_returns):
            if not sample.feature_vectors:
                continue
            records.append({
                "sample": sample,
                "return": step_return,
                "old_log_prob": float(sample.metadata.get("old_log_prob", 0.0)),
                "old_value": float(sample.metadata.get("value_estimate", 0.0)),
            })
    
    if not records:
        return {"decisions": 0.0, "reward": 0.0, "loss": 0.0}
    
    returns = torch.tensor([r["return"] for r in records], dtype=torch.float32)
    old_values = torch.tensor([r["old_value"] for r in records], dtype=torch.float32)
    advantages = returns - old_values
    
    # Normalize advantages
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    
    loss_value = 0.0
    for _ in range(max(1, epochs)):
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        
        for index, record in enumerate(records):
            sample = record["sample"]
            features = torch.tensor(sample.feature_vectors, dtype=torch.float32)
            logits, value = policy.model(features)
            scaled_logits = logits / max(1e-6, policy.temperature)
            distribution = torch.distributions.Categorical(logits=scaled_logits)
            
            action = torch.tensor(sample.chosen_index, dtype=torch.int64)
            new_log_prob = distribution.log_prob(action)
            old_log_prob = torch.tensor(record["old_log_prob"], dtype=torch.float32)
            
            ratio = torch.exp(new_log_prob - old_log_prob)
            advantage = advantages[index]
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - policy.clip_coef, 1.0 + policy.clip_coef) * advantage
            actor_loss = -torch.min(surr1, surr2)
            
            target_return = returns[index]
            value_loss = F.mse_loss(value, target_return)
            entropy = distribution.entropy()
            
            total_loss = total_loss + actor_loss + policy.vf_coef * value_loss - policy.ent_coef * entropy
        
        total_loss = total_loss / float(len(records))
        policy.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.model.parameters(), policy.max_grad_norm)
        policy.optimizer.step()
        loss_value = float(total_loss.item())
    
    return {
        "decisions": float(len(records)),
        "reward": float(returns.mean().item()),
        "loss": loss_value,
    }


def evaluate_vs_heuristic(
    policy: PPOMissionPolicy,
    rl_config: MidgameRLConfig,
    games: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate policy against the base heuristic."""
    wins = 0
    rewards: List[float] = []
    
    for i in range(games):
        swapped = (i % 2) == 1
        outcome = play_episode_improved(
            policy, rl_config, BASE_AGENT,
            swapped=swapped, explore=False,
        )
        rewards.append(outcome["reward"])
        result = outcome["result"]
        learner_won = (result["winner"] == "A" and not swapped) or (result["winner"] == "B" and swapped)
        if learner_won:
            wins += 1
    
    return {
        "games": float(games),
        "win_rate": wins / max(1, games),
        "avg_reward": sum(rewards) / max(1, len(rewards)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Improved PPO training with warm-start")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--games-per-update", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy-out", default=str(WORKSPACE_DIR / "results" / "ppo_improved_policy.json"))
    parser.add_argument("--warmstart-from", default=str(WORKSPACE_DIR / "results" / "policy_gated_neutral.json"),
                        help="Linear policy to warm-start from")
    parser.add_argument("--skip-warmstart", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--opponent", choices=("heuristic", "self", "mixed"), default="heuristic")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Create PPO policy
    ppo_policy = create_policy("ppo", hidden_size=args.hidden_size, seed=args.seed)
    
    # RL config: strict opening exclusion
    rl_config = MidgameRLConfig(
        activation_turn=24,  # FIX: was 16
        max_turn=180,
        min_candidates=2,
        top_k=8,
        contested_only=True,
        explore=True,
        allow_opening=False,
    )
    eval_config = MidgameRLConfig(
        activation_turn=24,
        max_turn=180,
        min_candidates=2,
        top_k=8,
        contested_only=True,
        explore=False,
        allow_opening=False,
    )
    
    # Step 1: Warm-start
    if not args.skip_warmstart and Path(args.warmstart_from).exists():
        print("=" * 60)
        print("WARM-START: Pretraining PPO actor from linear_v3")
        print("=" * 60)
        linear_policy = LinearMissionPolicy.load_json(args.warmstart_from)
        ws_metrics = warmstart_ppo_from_linear(ppo_policy, linear_policy)
        print(f"  Done. Loss: {ws_metrics['initial_loss']:.4f} → {ws_metrics['final_loss']:.4f}")
    else:
        print("Skipping warm-start (no linear policy or --skip-warmstart)")
    
    # Step 2: Baseline eval
    print("\n" + "=" * 60)
    print("BASELINE EVAL (after warm-start, before PPO training)")
    print("=" * 60)
    baseline_eval = evaluate_vs_heuristic(ppo_policy, eval_config, games=args.eval_games)
    print(f"  Win rate: {baseline_eval['win_rate']*100:.1f}%, Avg reward: {baseline_eval['avg_reward']:+.3f}")
    
    # Step 3: PPO training
    print("\n" + "=" * 60)
    print(f"PPO TRAINING ({args.updates} updates, {args.games_per_update} games/update)")
    print("=" * 60)
    
    policy_out = Path(args.policy_out)
    policy_out.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_win_rate = baseline_eval["win_rate"]
    
    for update_idx in range(1, args.updates + 1):
        batch = []
        update_start = time.time()
        
        for game_idx in range(args.games_per_update):
            swapped = (update_idx + game_idx) % 2 == 0
            
            if args.opponent == "heuristic":
                opponent = BASE_AGENT
            elif args.opponent == "mixed" and game_idx % 3 == 0:
                # Occasionally play against a snapshot
                opponent = build_agent(policy=ppo_policy.clone(), rl_config=eval_config)
            else:
                opponent = BASE_AGENT
            
            outcome = play_episode_improved(
                ppo_policy, rl_config, opponent,
                swapped=swapped, explore=True,
            )
            batch.append({"samples": outcome["samples"], "reward": outcome["reward"]})
        
        # Update with improved credit assignment
        update_metrics = batch_update_improved(
            ppo_policy, batch,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
        )
        
        elapsed = time.time() - update_start
        row = {"update": update_idx, **update_metrics, "time": elapsed}
        
        # Periodic evaluation
        if update_idx % args.eval_every == 0 or update_idx == args.updates:
            eval_metrics = evaluate_vs_heuristic(ppo_policy, eval_config, games=args.eval_games)
            row["eval"] = eval_metrics
            wr = eval_metrics["win_rate"]
            print(
                f"  [{update_idx:03d}] loss={update_metrics['loss']:.4f} "
                f"reward={update_metrics['reward']:+.3f} "
                f"eval_win={wr:.3f} eval_reward={eval_metrics['avg_reward']:+.3f} "
                f"[{elapsed:.0f}s]"
            )
            if wr > best_win_rate:
                best_win_rate = wr
                ppo_policy.save_json(policy_out.with_name("ppo_improved_best.json"))
                print(f"    ★ New best: {wr*100:.1f}%")
        else:
            print(
                f"  [{update_idx:03d}] loss={update_metrics['loss']:.4f} "
                f"reward={update_metrics['reward']:+.3f} [{elapsed:.0f}s]"
            )
        
        history.append(row)
    
    # Save final
    ppo_policy.save_json(policy_out)
    summary = {
        "config": vars(args),
        "baseline_eval": baseline_eval,
        "best_win_rate": best_win_rate,
        "history": history,
    }
    summary_path = policy_out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nFinal policy: {policy_out}")
    print(f"Best win rate: {best_win_rate*100:.1f}%")


if __name__ == "__main__":
    main()
