"""Phase 4: Online PPO fine-tuning of EdgePolicy.

Loads a BC-pretrained EdgePolicy checkpoint and fine-tunes with PPO against
a tiered opponent curriculum. Uses compute_candidate_edges() to build features
from raw kaggle observations at rollout time.

Usage:
    python -m ppo_gnn.train_ppo_edge --checkpoint ppo_gnn/cache/checkpoint_bc_edge.pt --mode 2p
    python -m ppo_gnn.train_ppo_edge --checkpoint ppo_gnn/cache/checkpoint_bc_edge.pt --mode mixed
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .edge_policy import (
    EdgePolicy,
    MAX_ACTIONS,
    MAX_CANDIDATES,
    EDGE_INPUT_DIM,
    FRACTION_BUCKETS,
    compute_candidate_edges,
    solve_intercept,
    _travel_time,
)

os.environ.setdefault("KAGGLE_ENVIRONMENTS_QUIET", "1")


# ---------------------------------------------------------------------------
# PID controller (reused from train_ppo.py)
# ---------------------------------------------------------------------------

class KLPIDController:
    def __init__(self, target_kl=0.01, kp=0.5, ki=0.05, kd=0.1,
                 lr_min=1e-6, lr_max=3e-4):
        self.target_kl = target_kl
        self.kp, self.ki, self.kd = kp, ki, kd
        self.lr_min, self.lr_max = lr_min, lr_max
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, current_kl: float, current_lr: float) -> float:
        error = abs(current_kl) - self.target_kl
        self._integral = max(-0.5, min(0.5, self._integral + error))
        correction = max(0.0, self.kp * error + self.ki * self._integral
                         + self.kd * (error - self._prev_error))
        self._prev_error = error
        return float(max(self.lr_min, min(self.lr_max, current_lr * (1 - correction))))

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0


# ---------------------------------------------------------------------------
# Rollout data structure
# ---------------------------------------------------------------------------

@dataclass
class EdgeRolloutStep:
    edge_features: torch.Tensor   # (K, EDGE_INPUT_DIM)
    edge_mask: torch.Tensor       # (K,)
    action_edges: torch.Tensor    # (MAX_ACTIONS,)  -1 = unused
    action_fractions: torch.Tensor  # (MAX_ACTIONS,)  -1 = unused
    action_count: int
    log_prob: float
    value: float
    reward: float
    done: bool


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_planet_count_bonus(my_planet_count: int) -> float:
    """Dense per-step bonus for holding planets.

    Uses sqrt scaling so early captures are highly rewarded but
    overextending gives diminishing returns. Kept small relative
    to win/loss (±10) so it guides without dominating.
    """
    if my_planet_count <= 0:
        return 0.0
    return 0.005 * math.sqrt(my_planet_count)


class RewardNormalizer:
    """Running mean/std normalizer for rewards (like SB3's VecNormalize)."""

    def __init__(self, clip: float = 10.0, epsilon: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.clip = clip
        self.epsilon = epsilon

    def update(self, rewards: List[float]):
        """Update running stats with a batch of rewards."""
        for r in rewards:
            self.count += 1
            if self.count == 1:
                self.mean = r
                self.var = 0.0
            else:
                delta = r - self.mean
                self.mean += delta / self.count
                delta2 = r - self.mean
                self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, reward: float) -> float:
        """Normalize a single reward."""
        std = math.sqrt(self.var + self.epsilon)
        normed = (reward - self.mean) / std
        return max(-self.clip, min(self.clip, normed))


def compute_gae(
    rollout: List[EdgeRolloutStep],
    gamma: float = 0.997,
    lam: float = 0.95,
    reward_normalizer: RewardNormalizer = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = len(rollout)
    advantages = torch.zeros(T)
    returns = torch.zeros(T)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        s = rollout[t]
        mask = 0.0 if s.done else 1.0
        r = s.reward
        if reward_normalizer is not None:
            r = reward_normalizer.normalize(r)
        delta = r + gamma * next_value * mask - s.value
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        returns[t] = gae + s.value
        next_value = s.value
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    model: EdgePolicy,
    optimizer: torch.optim.Optimizer,
    rollout: List[EdgeRolloutStep],
    epochs: int = 4,
    mini_batch_size: int = 64,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.015,
    entropy_frac_coef: float = 0.05,
    reward_normalizer: RewardNormalizer = None,
    kl_coef: float = 0.1,
    gamma: float = 0.997,
    gae_lambda: float = 0.95,
    device: torch.device = torch.device("cpu"),
) -> dict:
    advantages, returns = compute_gae(rollout, gamma, gae_lambda, reward_normalizer)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    T = len(rollout)
    K = MAX_CANDIDATES

    ef_all = torch.zeros(T, K, EDGE_INPUT_DIM)
    em_all = torch.zeros(T, K)
    ae_all = torch.full((T, MAX_ACTIONS), -1, dtype=torch.long)
    af_all = torch.full((T, MAX_ACTIONS), -1, dtype=torch.long)
    ac_all = torch.zeros(T, dtype=torch.long)
    old_lp = torch.zeros(T)

    for i, s in enumerate(rollout):
        ef_all[i] = s.edge_features
        em_all[i] = s.edge_mask
        ae_all[i] = s.action_edges
        af_all[i] = s.action_fractions
        ac_all[i] = s.action_count
        old_lp[i] = s.log_prob

    # Filter out steps with zero valid edges (all-padding causes NaN in attention)
    valid_steps = em_all.sum(dim=1) > 0  # (T,)
    if valid_steps.sum() < T:
        ef_all = ef_all[valid_steps]
        em_all = em_all[valid_steps]
        ae_all = ae_all[valid_steps]
        af_all = af_all[valid_steps]
        ac_all = ac_all[valid_steps]
        old_lp = old_lp[valid_steps]
        advantages = advantages[valid_steps]
        returns = returns[valid_steps]
        T = ef_all.shape[0]
        if T == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0,
                    "clip_frac": 0, "kl": 0, "explained_var": 0,
                    "weight_nan": False}

    # Clamp old log probs to prevent extreme ratios
    old_lp = old_lp.clamp(min=-20.0)

    ef_all = ef_all.to(device)
    em_all = em_all.to(device)
    ae_all = ae_all.to(device)
    af_all = af_all.to(device)
    ac_all = ac_all.to(device)
    old_lp = old_lp.to(device)
    advantages = advantages.to(device)
    returns = returns.to(device)

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "clip_frac": 0.0, "kl": 0.0}
    n_updates = 0

    for _ in range(epochs):
        perm = torch.randperm(T, device=device)
        for start in range(0, T, mini_batch_size):
            idx = perm[start:start + mini_batch_size]

            log_prob, sel_entropy, frac_entropy, value = model.evaluate_action(
                ef_all[idx], em_all[idx], ae_all[idx], af_all[idx], ac_all[idx],
            )

            # Guard: clamp log_prob to prevent -inf → NaN in ratio/kl
            log_prob = log_prob.clamp(min=-20.0)
            sel_entropy = torch.nan_to_num(sel_entropy, nan=0.0)
            frac_entropy = torch.nan_to_num(frac_entropy, nan=0.0)

            ratio = torch.exp(log_prob - old_lp[idx].clamp(min=-20.0))
            adv = advantages[idx]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, returns[idx])
            # Per-head entropy: selection can be sharp, fractions must stay diverse
            sel_entropy_loss = -sel_entropy.mean()
            frac_entropy_loss = -frac_entropy.mean()
            entropy = sel_entropy + frac_entropy  # for logging
            kl_approx = (old_lp[idx].clamp(min=-20.0) - log_prob).mean()
            loss = (policy_loss + value_coef * value_loss
                    + entropy_coef * sel_entropy_loss
                    + entropy_frac_coef * frac_entropy_loss
                    + kl_coef * kl_approx)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()

            # Check for NaN gradients before stepping
            has_nan_grad = False
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
            if has_nan_grad:
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                clip_frac = ((ratio - 1).abs() > clip_epsilon).float().mean().item()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.mean().item()
            stats["clip_frac"] += clip_frac
            stats["kl"] += kl_approx.item()
            n_updates += 1

    for k in stats:
        stats[k] /= max(n_updates, 1)

    # Explained variance: how well the critic predicts returns
    with torch.no_grad():
        _, _, _, v_pred = model.evaluate_action(
            ef_all, em_all, ae_all, af_all, ac_all,
        )
        var_ret = returns.var()
        if var_ret < 1e-8:
            stats["explained_var"] = 0.0
        else:
            stats["explained_var"] = (1 - (returns - v_pred).var() / var_ret).item()

    # Check for NaN in weights after update
    has_nan = any(torch.isnan(p).any() for p in model.parameters())
    stats["weight_nan"] = has_nan
    return stats


# ---------------------------------------------------------------------------
# Opponent loading (mirrors train_ppo.py)
# ---------------------------------------------------------------------------

def load_agent_from_file(path: str) -> Callable:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Agent file not found: {p}")
    module_name = f"_agent_{p.stem}_{id(p)}"
    spec = importlib.util.spec_from_file_location(module_name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "agent"):
        raise AttributeError(f"No top-level agent() in {p}")
    return mod.agent


def make_edge_self_play_agent(
    model: EdgePolicy,
    num_players: int,
    device: torch.device,
    n_heads: int = 4,
    n_layers: int = 3,
) -> Callable:
    """Snapshot the current model as a frozen kaggle agent callable."""
    shadow = EdgePolicy(
        d_model=model.d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_actions=model.max_actions,
        separate_critic=model.separate_critic,
    )
    shadow.load_state_dict(model.state_dict())
    shadow.to(device)
    shadow.eval()

    def _agent(obs, config=None):
        planets = obs.get("planets", [])
        if not planets:
            return []
        player = obs.get("player", 0)
        fleets = obs.get("fleets", [])
        step = obs.get("step", 0)
        max_steps = obs.get("remainingOverageTime", 500)

        has_planets = any(int(p[1]) == player for p in planets)
        if not has_planets:
            return []

        ef, edge_indices, em, num_valid = compute_candidate_edges(
            planets=planets, fleets=fleets, player_id=player,
            num_players=num_players, step=step, max_steps=max_steps,
        )
        if num_valid == 0:
            return []

        ef = torch.nan_to_num(ef, nan=0.0, posinf=1.0, neginf=-1.0)
        ef = ef.to(device)
        em = em.to(device)

        selected, fracs, is_noop, _, _ = shadow.sample_action(ef, em)
        if is_noop:
            return []

        actions = []
        for cand_idx, frac_idx in zip(selected, fracs):
            src_pidx = edge_indices[cand_idx, 0].item()
            tgt_pidx = edge_indices[cand_idx, 1].item()
            src_p = planets[src_pidx]
            tgt_p = planets[tgt_pidx]
            sx, sy = float(src_p[2]), float(src_p[3])
            tx, ty = float(tgt_p[2]), float(tgt_p[3])
            omega = float(tgt_p[4])
            ships = max(1, int(float(src_p[5]) * FRACTION_BUCKETS[frac_idx]))
            if ships < 5:
                continue
            ix, iy = solve_intercept(sx, sy, tx, ty, omega, ships)
            angle = math.atan2(iy - sy, ix - sx)
            actions.append([int(src_p[0]), angle, ships])
        return actions

    return _agent


def build_opponent_pool(mode: str) -> list:
    root = Path(__file__).parent.parent
    num_players = 2 if mode == "2p" else 4
    ext = root / "submission" / "ext"

    from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent, starter_agent
    pool = [
        (1, "random",  random_agent),
        (3, "starter", starter_agent),
    ]

    candidates = [
        (2, "bully",          "submission/pool_bully.py"),
        (2, "rage",           "submission/pool_rage.py"),
        (2, "dual",           "submission/pool_dual.py"),
        (3, "prospector",     "submission/pool_prospector.py"),
        (3, "nearest_sniper", "submission/ext/pool_baseline_nearest_sniper.py"),
        (3, "sig_starter",    "submission/ext/pool_sigmaborov_starter.py"),
        (6, "pascal_v14",     "submission/ext/pool_pascal_orbitwork_v14.py"),
        # Tier 7: easy wins (>75% WR)
        (7, "tamrazov",       "submission/ext/pool_tamrazov_starwars.py"),
        (7, "kashiwaba_rl",   "submission/ext/pool_kashiwaba_rl.py"),
        (7, "yuriy_arch",     "submission/ext/pool_yuriygreben_architect.py"),
        # Tier 8: stubborn (~16% WR — defensive/economic style)
        (8, "baseline",       "submission/pool_baseline.py"),
        (8, "ykhnkf_dist",    "submission/ext/pool_ykhnkf_distance_prioritized.py"),
        # Tier 9: crackable (~5-10% WR)
        (9, "pilkwang",       "submission/ext/pool_pilkwang_structured.py"),
        (9, "v131_denial",    "submission/main_v131_plus_denial.py"),
        # Tier 10: hard (0-8% WR currently)
        (10, "sig_reinforce", "submission/ext/pool_sigmaborov_reinforce.py"),
        (10, "marco_dg",      "submission/ext/pool_marco_dg_v3.py"),
        (10, "ml_shot_hybrid","submission/ext/pool_ml_shot_hybrid.py"),
        # Tier 11: top tier — 0% WR, the real targets
        (11, "shunlite",      "submission/main_fc_rl_shunlite.py"),
        (11, "v131_2p",       "submission/main_v131_plus_2p.py"),
        (11, "v131_wave",     "submission/main_v131_plus_wave.py"),
    ]
    if mode in ("4p", "mixed"):
        candidates += [
            (10, "plus4p",    "submission/main_v131_plus_4p.py"),
            (11, "political", "submission/main_v131_plus_4p_political.py"),
        ]

    for tier, name, rel_path in candidates:
        full_path = root / rel_path
        if full_path.exists():
            try:
                pool.append((tier, name, load_agent_from_file(str(full_path))))
                print(f"  [tier {tier}] {name}")
            except Exception as e:
                print(f"  Warning: skipped {name}: {e}")
        else:
            print(f"  Skipped {name} (not found)")

    return pool


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

def play_episode(
    model: EdgePolicy,
    opponent,
    mode: str = "2p",
    step_penalty: float = 0.002,
    idle_penalty: float = 0.02,
    max_steps: int = 500,
    win_bonus: float = 10.0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[EdgeRolloutStep], float, dict]:
    from kaggle_environments import make

    env = make("orbit_wars", debug=False)
    num_players = 2 if mode == "2p" else 4

    if mode == "2p":
        trainer = env.train([None, opponent])
    else:
        slots = opponent if isinstance(opponent, list) else [opponent] * 3
        trainer = env.train([None] + slots)

    obs = trainer.reset()
    model.eval()
    rollout: List[EdgeRolloutStep] = []
    env_steps = 0
    num_launches = 0
    num_noops = 0
    prev_shaped_score = 0.0
    prev_my_planets = 1  # Start with 1 planet
    steps_since_last_capture = 999  # large initial value

    for step_idx in range(max_steps):
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        player = obs.get("player", 0)

        if not planets:
            obs, reward, done, info = trainer.step([])
            env_steps += 1
            if done:
                break
            continue

        has_planets = any(int(p[1]) == player for p in planets)
        if not has_planets:
            obs, reward, done, info = trainer.step([])
            env_steps += 1
            if done:
                break
            continue

        ef, edge_indices, em, num_valid = compute_candidate_edges(
            planets=planets, fleets=fleets, player_id=player,
            num_players=num_players, step=step_idx, max_steps=max_steps,
        )
        # Guard against NaN/Inf in features before hitting the model
        ef = torch.nan_to_num(ef, nan=0.0, posinf=1.0, neginf=-1.0)
        ef = ef.to(device)
        em = em.to(device)

        selected, fracs, is_noop, log_prob, value = model.sample_action(ef, em)

        # Build kaggle action list
        actions = []
        my_planets_list = [p for p in planets if int(p[1]) == player]
        num_my_planets = len(my_planets_list)

        if not is_noop and num_valid > 0:
            for cand_idx, frac_idx in zip(selected, fracs):
                if cand_idx >= num_valid:
                    continue
                src_pidx = edge_indices[cand_idx, 0].item()
                tgt_pidx = edge_indices[cand_idx, 1].item()
                src_p = planets[src_pidx]
                tgt_p = planets[tgt_pidx]
                sx, sy = float(src_p[2]), float(src_p[3])
                tx, ty = float(tgt_p[2]), float(tgt_p[3])
                omega = float(tgt_p[4])
                src_fleet = int(float(src_p[5]))

                # Ship reserve — keep some ships on planet
                reserve = min(5, int(src_fleet * 0.2))
                available = src_fleet - reserve
                if available < 1:
                    continue

                ships = max(1, int(available * FRACTION_BUCKETS[frac_idx]))

                # Min send — don't waste tiny fleets
                if ships < 5:
                    continue

                # Orbit intercept — aim at predicted position
                ix, iy = solve_intercept(sx, sy, tx, ty, omega, ships)
                angle = math.atan2(iy - sy, ix - sx)

                # Max ETA filter — don't send across the whole map
                travel_time = _travel_time(sx, sy, ix, iy, ships)
                max_eta = 14 if env_steps < 50 else (20 if env_steps < 150 else 30)
                if travel_time > max_eta:
                    continue

                actions.append([int(src_p[0]), angle, ships])

        num_launches += len(actions)
        if not actions:
            num_noops += 1

        obs, reward, done, info = trainer.step(actions)
        env_steps += 1

        # Dense reward shaping
        my_fleet = my_prod = enemy_fleet = enemy_prod = 0.0
        for p in planets:
            owner = int(p[1])
            ships, prod = float(p[5]), float(p[6])
            if owner == player:
                my_fleet += ships
                my_prod += prod
            elif owner >= 0:
                enemy_fleet += ships
                enemy_prod += prod

        # Delta shaping — reward gaining advantage
        PROD_W = 3.0
        score = (my_fleet + PROD_W * my_prod) - (enemy_fleet + PROD_W * enemy_prod)
        # Aggressive idle penalty — hoarding ships must hurt
        # Scales with fleet size: sitting on 100 ships = -0.04/step, 200 = -0.08/step
        # Over 500 steps at 200 ships that's -40, far exceeding win bonus (10)
        idle_penalty = step_penalty + my_fleet * 0.0004
        shaped_reward = (score - prev_shaped_score) * 0.01 - idle_penalty
        prev_shaped_score = score

        # Capture bonus — one-time signal for planet changes
        my_planets_now = sum(1 for p in planets if int(p[1]) == player)
        if my_planets_now > prev_my_planets:
            gained = my_planets_now - prev_my_planets
            shaped_reward += 1.0 * gained
            # Rapid expansion bonus — reward successive captures
            if steps_since_last_capture < 20:
                shaped_reward += 0.3 * gained
            steps_since_last_capture = 0
        elif my_planets_now < prev_my_planets:
            shaped_reward -= 0.8 * (prev_my_planets - my_planets_now)
        steps_since_last_capture += 1
        prev_my_planets = my_planets_now

        # Dense economy reward — continuous incentive to hold planets + production
        total_planets = sum(1 for p in planets if int(p[1]) >= 0)
        planet_share = my_planets_now / max(total_planets, 1)
        economy_bonus = (planet_share - 0.5) * 0.04

        total_prod = sum(float(p[6]) for p in planets if int(p[1]) >= 0)
        prod_share = my_prod / max(total_prod, 1)
        prod_bonus = (prod_share - 0.5) * 0.04

        # Early-game 3x multiplier (first 50 steps) to force expansion
        early_mult = 3.0 if env_steps < 50 else 1.0
        shaped_reward += (economy_bonus + prod_bonus) * early_mult

        # Dense planet count bonus — "holding planets is always good"
        shaped_reward += compute_planet_count_bonus(my_planets_now)

        # Fleet cost — small penalty per fleet launched to discourage spam
        num_fleets_sent = len(selected)
        if num_fleets_sent > 0:
            shaped_reward -= 0.002 * num_fleets_sent

        # Pack edge tensors for PPO (cpu storage)
        action_edges = torch.full((MAX_ACTIONS,), -1, dtype=torch.long)
        action_fractions = torch.full((MAX_ACTIONS,), -1, dtype=torch.long)
        action_count = min(len(selected), MAX_ACTIONS)
        for ai, (ci, fi) in enumerate(zip(selected[:MAX_ACTIONS], fracs[:MAX_ACTIONS])):
            action_edges[ai] = ci
            action_fractions[ai] = fi

        rollout.append(EdgeRolloutStep(
            edge_features=ef.cpu(),
            edge_mask=em.cpu(),
            action_edges=action_edges,
            action_fractions=action_fractions,
            action_count=action_count,
            log_prob=log_prob,
            value=value,
            reward=shaped_reward,
            done=done,
        ))

        if done:
            break

    # Terminal reward
    raw_reward = reward if reward is not None else 0.0
    hit_horizon = (not done) and (env_steps >= max_steps)

    my_fleet_f = my_prod_f = enemy_fleet_f = enemy_prod_f = 0.0
    owned_final = 0
    if obs and "planets" in obs:
        for p in obs["planets"]:
            owner = int(p[1])
            ships, prod = float(p[5]), float(p[6])
            if owner == player:
                owned_final += 1
                my_fleet_f += ships
                my_prod_f += prod
            elif owner >= 0:
                enemy_fleet_f += ships
                enemy_prod_f += prod

    eliminated = owned_final == 0

    if raw_reward > 0:
        final_reward = win_bonus
    elif raw_reward < 0 or (eliminated and enemy_fleet_f + enemy_prod_f > 0):
        survival_frac = env_steps / max_steps
        final_reward = -win_bonus * (1.0 - 0.5 * survival_frac)
    elif hit_horizon:
        remaining = 500 - env_steps
        time_val = remaining / 500
        exp_fleet = max(50 + env_steps * 3, 1.0)
        exp_prod = max(5 + env_steps * 0.2, 1.0)
        fleet_diff = (my_fleet_f - enemy_fleet_f) / exp_fleet
        prod_diff = (my_prod_f - enemy_prod_f) / exp_prod
        advantage = (1 - time_val) * fleet_diff + time_val * 1.5 * prod_diff
        final_reward = math.tanh(advantage) * (win_bonus / 2.0)
    else:
        final_reward = 0.0

    if rollout and final_reward != 0:
        last = rollout[-1]
        rollout[-1] = EdgeRolloutStep(
            **{**last.__dict__, "reward": final_reward, "done": True}
        )

    ep_stats = {
        "my_fleet": my_fleet_f, "my_prod": my_prod_f,
        "enemy_fleet": enemy_fleet_f, "enemy_prod": enemy_prod_f,
        "env_steps": env_steps, "owned_planets": owned_final,
        "eliminated": eliminated, "hit_horizon": hit_horizon,
        "num_launches": num_launches, "num_noops": num_noops,
    }
    return rollout, final_reward, ep_stats


# ---------------------------------------------------------------------------
# Curriculum helpers
# ---------------------------------------------------------------------------

class CurriculumManager:
    """Win-rate gated curriculum — only advance when mastering current tier.
    Also manages progressive 4p mixing: starts at 5%, increases by 5% each
    time the model gates, capped at 50%.
    """

    def __init__(self, promotion_threshold=0.70, demotion_threshold=0.30,
                 window=50, log_fn=None, enable_4p=False):
        self.current_max_tier = 2  # Start with tier 1-2 (random + bully/rage/dual)
        self.min_tier = 2  # Never demote below this
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold
        self.window = window
        self.recent_wins_2p = []   # Used for promotion/demotion
        self.recent_wins_4p = []   # Tracked separately, not used for promotion
        self.promotions = []  # (episode, new_tier) for graphing
        self.log_fn = log_fn or print
        # Progressive 4p mixing
        self.enable_4p = enable_4p
        self.fourp_ratio = 0.05 if enable_4p else 0.0  # start at 5%
        self.fourp_max = 0.50

    def record(self, won: bool, episode: int, mode: str = "2p"):
        if mode == "4p":
            self.recent_wins_4p.append(won)
            if len(self.recent_wins_4p) > self.window:
                self.recent_wins_4p.pop(0)
        else:
            self.recent_wins_2p.append(won)
            if len(self.recent_wins_2p) > self.window:
                self.recent_wins_2p.pop(0)

    def maybe_promote(self, episode: int, max_available_tier: int):
        # Promotion/demotion based on 2p games only
        if len(self.recent_wins_2p) >= self.window:
            wr = sum(self.recent_wins_2p) / len(self.recent_wins_2p)
            # Demotion check — drop back if getting crushed
            if wr <= self.demotion_threshold and self.current_max_tier > self.min_tier:
                old_tier = self.current_max_tier
                self.current_max_tier -= 1
                self.promotions.append((episode, self.current_max_tier))
                self.log_fn(f"*** CURRICULUM: DEMOTED to tier {self.current_max_tier} "
                           f"(wr={wr:.0%} over {self.window} 2p games) ***")
                self.recent_wins_2p.clear()
                return False
            # Promotion check
            if wr >= self.promotion_threshold and self.current_max_tier < max_available_tier:
                self.current_max_tier += 1
                self.promotions.append((episode, self.current_max_tier))
                self.log_fn(f"*** CURRICULUM: promoted to tier {self.current_max_tier} "
                           f"(wr={wr:.0%} over {self.window} 2p games) ***")
                # Increase 4p ratio on each promotion
                if self.enable_4p:
                    old_ratio = self.fourp_ratio
                    self.fourp_ratio = min(self.fourp_ratio + 0.05, self.fourp_max)
                    self.log_fn(f"    4p ratio: {old_ratio:.0%} -> {self.fourp_ratio:.0%}")
                self.recent_wins_2p.clear()
                return True
        return False

    def pick_mode(self):
        """Return '2p' or '4p' based on current progressive ratio."""
        if self.enable_4p and random.random() < self.fourp_ratio:
            return "4p"
        return "2p"

    def tier_weights(self):
        """Spread weight across tiers so the model doesn't forget lower ones."""
        t = self.current_max_tier
        weights = {}
        if t >= 3:
            weights[t - 2] = 0.1  # keep some easy games
        if t >= 2:
            weights[t - 1] = 0.3
        weights[t] = 0.6
        return weights

    @property
    def rolling_wr(self):
        if not self.recent_wins_2p:
            return 0.0
        return sum(self.recent_wins_2p) / len(self.recent_wins_2p)

    @property
    def rolling_wr_4p(self):
        if not self.recent_wins_4p:
            return 0.0
        return sum(self.recent_wins_4p) / len(self.recent_wins_4p)


def pick_opponent(pool, tier_weights_dict, episode_num, self_play_agent):
    """Sample opponent with curriculum weighting."""
    max_tier = max(t for t, _, _ in pool) if pool else 1
    weights = []
    agents = []
    for tier, name, fn in pool:
        w = tier_weights_dict.get(tier, 0.0)
        if w > 0:
            weights.append(w)
            agents.append((tier, name, fn))

    if not agents:
        return 1, "self_play", self_play_agent

    total = sum(weights)
    r = random.random() * total
    acc = 0.0
    for (tier, name, fn), w in zip(agents, weights):
        acc += w
        if r <= acc:
            return tier, name, fn
    return agents[-1]


def eval_vs_pool(model, pool, mode, device, n_games=2):
    """Quick eval: play n_games vs each pool opponent, return win rate dict."""
    num_players = 2 if mode == "2p" else 4
    results = {}
    model.eval()
    for tier, name, opp_fn in pool:
        wins = 0
        for _ in range(n_games):
            _, reward, _ = play_episode(model, opp_fn, mode=mode, device=device,
                                        win_bonus=1.0, max_steps=300)
            if reward > 0:
                wins += 1
        results[name] = wins / n_games
    return results


def save_progress_graph(episodes, win_rates, rewards, pol_loss, val_loss,
                        tiers, promotions, path,
                        cwr_2p=None, cwr_4p=None, ev=None):
    """Save training progress PNG with 6 subplots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if len(episodes) < 2:
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('PPO Edge Training Progress', fontsize=14)

    def _smooth(data, window=10):
        w = min(window, len(data))
        return [sum(data[max(0,i-w):i+1])/len(data[max(0,i-w):i+1])
                for i in range(len(data))]

    # 1. Win rate
    ax = axes[0, 0]
    ax.plot(episodes, win_rates, 'b-', alpha=0.3, label='raw')
    if len(win_rates) >= 5:
        ax.plot(episodes, _smooth(win_rates), 'b-', linewidth=2, label='smoothed')
    for ep_p, tier_p in promotions:
        ax.axvline(x=ep_p, color='green', linestyle='--', alpha=0.7)
    ax.set_ylabel('Win Rate')
    ax.set_title('Win Rate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Curriculum win rates (2p vs 4p)
    ax = axes[0, 1]
    if cwr_2p:
        ax.plot(episodes[:len(cwr_2p)], cwr_2p, 'b-', alpha=0.5, label='2p cwr')
        if len(cwr_2p) >= 5:
            ax.plot(episodes[:len(cwr_2p)], _smooth(cwr_2p), 'b-', linewidth=2)
    if cwr_4p:
        ax.plot(episodes[:len(cwr_4p)], cwr_4p, 'r-', alpha=0.5, label='4p cwr')
        if len(cwr_4p) >= 5:
            ax.plot(episodes[:len(cwr_4p)], _smooth(cwr_4p), 'r-', linewidth=2)
    ax.axhline(y=0.70, color='green', linestyle=':', alpha=0.5, label='promote')
    ax.axhline(y=0.30, color='red', linestyle=':', alpha=0.5, label='demote')
    ax.set_ylabel('CWR')
    ax.set_title('Curriculum Win Rate (2p=promote, 4p=info)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # 3. Reward
    ax = axes[1, 0]
    ax.plot(episodes, rewards, 'r-', alpha=0.3)
    if len(rewards) >= 5:
        ax.plot(episodes, _smooth(rewards), 'r-', linewidth=2)
    ax.set_ylabel('Avg Reward')
    ax.set_title('Reward')
    ax.grid(True, alpha=0.3)

    # 4. Explained variance
    ax = axes[1, 1]
    if ev:
        ax.plot(episodes[:len(ev)], ev, 'c-', alpha=0.3)
        if len(ev) >= 5:
            ax.plot(episodes[:len(ev)], _smooth(ev), 'c-', linewidth=2)
    ax.set_ylabel('Explained Variance')
    ax.set_title('Critic Quality (EV)')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # 5. Losses
    ax = axes[2, 0]
    ax.plot(episodes, pol_loss, 'g-', label='policy', alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(episodes, val_loss, 'orange', label='value', alpha=0.7)
    ax.set_ylabel('Policy Loss', color='green')
    ax2.set_ylabel('Value Loss', color='orange')
    ax.set_title('Losses')
    ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.3)

    # 6. Curriculum tier
    ax = axes[2, 1]
    ax.plot(episodes, tiers, 'purple', linewidth=2)
    ax.set_ylabel('Max Tier')
    ax.set_title('Curriculum Tier')
    ax.set_xlabel('Episode')
    ax.set_ylim(0, 9)
    ax.grid(True, alpha=0.3)
    for ep_p, tier_p in promotions:
        ax.annotate(f'T{tier_p}', (ep_p, tier_p), fontsize=9, color='green')

    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PPO fine-tuning of EdgePolicy")
    parser.add_argument("--checkpoint", default=None,
                        help="BC edge checkpoint to warm-start from (omit for random init)")
    parser.add_argument("--mode", choices=["2p", "4p", "mixed"], default="2p")
    parser.add_argument("--cache-dir", default="ppo_gnn/cache")
    parser.add_argument("--num-episodes", type=int, default=50000)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=3e-5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--step-penalty", type=float, default=0.002)
    parser.add_argument("--idle-penalty", type=float, default=0.02)
    parser.add_argument("--win-bonus", type=float, default=10.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--entropy-frac-coef", type=float, default=0.05)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--reset-value-head", action="store_true",
                        help="Re-initialize value head weights to force relearning")
    parser.add_argument("--start-tier", type=int, default=2,
                        help="Starting curriculum tier (skip easy opponents)")
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--pid-lr", action="store_true")
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--update-device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    update_device = torch.device(args.update_device) if args.update_device else device
    cache_dir = Path(args.cache_dir)
    log_path = cache_dir / "ppo_edge_train.log"

    # Load checkpoint or init from scratch
    d_model = args.d_model
    separate_critic = True

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        d_model = ckpt.get("d_model", args.d_model)
        separate_critic = ckpt.get("separate_critic", True)
    else:
        sd = None

    model = EdgePolicy(
        d_model=d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_actions=MAX_ACTIONS,
        separate_critic=separate_critic,
    )

    if sd is not None:
        # Filter out mismatched weights (shape changes between versions)
        model_sd = model.state_dict()
        filtered_sd = {}
        skipped = []
        for k, v in sd.items():
            if k in model_sd and v.shape != model_sd[k].shape:
                skipped.append(f"  {k}: ckpt={list(v.shape)} vs model={list(model_sd[k].shape)}")
            else:
                filtered_sd[k] = v

        missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
        if skipped:
            print(f"Reinitialised {len(skipped)} layers (shape mismatch):")
            for s in skipped:
                print(s)
        if missing:
            print(f"Warning: {len(missing)} missing keys in checkpoint")
        init_msg = f"loaded from {args.checkpoint}"
    else:
        init_msg = "random init (from scratch)"

    model.to(device)

    params = model.count_parameters()
    print(f"EdgePolicy PPO: d={d_model}, heads={args.n_heads}, layers={args.n_layers}")
    print(f"Parameters: {params['total']:,}  ({init_msg})")

    # Reset value head if requested (forces critic to relearn → larger advantages)
    if args.reset_value_head:
        for m in [model.value_head]:
            for layer in m:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        # Also reset critic backbone if separate
        if model.separate_critic:
            for m in [model.critic_encoder] + list(model.critic_transformer):
                for layer in m.modules():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
        print("Value head + critic backbone re-initialized")

    # Separate optimizers: faster policy LR, slower value LR
    value_params = set()
    for m in [model.value_head]:
        value_params.update(m.parameters())
    if model.separate_critic:
        for m in [model.critic_encoder] + list(model.critic_transformer):
            value_params.update(m.parameters())
    policy_params = [p for p in model.parameters() if p not in value_params]
    optimizer = torch.optim.Adam([
        {"params": policy_params, "lr": args.lr},
        {"params": list(value_params), "lr": args.value_lr},
    ])
    pid = KLPIDController(target_kl=args.target_kl) if args.pid_lr else None

    # Opponent pool
    print("\nBuilding opponent pool...")
    num_players = 2 if args.mode in ("2p", "mixed") else 4
    pool_2p = build_opponent_pool("2p")
    pool_4p = build_opponent_pool("4p") if args.mode in ("4p", "mixed") else []

    # Win-rate gated curriculum

    best_checkpoint = str(cache_dir / "checkpoint_ppo_edge_best.pt")
    latest_checkpoint = str(cache_dir / "checkpoint_ppo_edge_latest.pt")

    t_start = time.time()
    total_rollout_steps = 0
    wins = losses = draws = 0
    ep = 0
    log_file = open(log_path, "w", buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")

    log(f"PPO EdgePolicy training — {args.mode} mode")
    log(f"Episodes: {args.num_episodes}, episodes_per_update: {args.episodes_per_update}")
    log(f"LR: {args.lr} (value: {args.value_lr}), entropy_coef: {args.entropy_coef}, device: {args.device}")

    # Snapshot for self-play
    self_play_agent = make_edge_self_play_agent(
        model, num_players, device, args.n_heads, args.n_layers)
    best_win_rate = 0.0

    max_available_tier = max(t for t, _, _ in pool_2p) if pool_2p else 2
    enable_4p = args.mode == "mixed" and len(pool_4p) > 0
    curriculum = CurriculumManager(promotion_threshold=0.70, window=50, log_fn=log,
                                   enable_4p=enable_4p)
    if args.start_tier > 2:
        curriculum.current_max_tier = args.start_tier
        # Set 4p ratio to match what it would be after promotions
        if enable_4p:
            promotions_done = args.start_tier - 2
            curriculum.fourp_ratio = min(0.05 + 0.05 * promotions_done, curriculum.fourp_max)
        log(f"Starting at tier {args.start_tier} (4p ratio: {curriculum.fourp_ratio:.0%})")

    # Self-play archive — ring buffer of lagged checkpoints
    selfplay_archive = []  # list of (episode, agent_fn)
    SELFPLAY_SAVE_EVERY = 100  # snapshot every 100 episodes
    SELFPLAY_MAX_ARCHIVE = 10  # keep last 10 snapshots
    SELFPLAY_RATIO = 0.5  # 50% self-play, 50% curriculum

    # Metrics tracking for graphs
    metric_episodes = []
    metric_win_rates = []
    metric_rewards = []
    metric_pol_loss = []
    metric_val_loss = []
    metric_tiers = []
    metric_cwr_2p = []
    metric_cwr_4p = []
    metric_ev = []

    # Per-opponent cumulative tracking
    opp_record: dict = {}  # name -> [wins, played]

    # Reward normalizer (SB3-style running mean/std)
    reward_norm = RewardNormalizer()

    while ep < args.num_episodes:
        # Collect episodes_per_update episodes before updating
        batch_rollout: List[EdgeRolloutStep] = []
        ep_rewards = []
        ep_stats_list = []

        for _ in range(args.episodes_per_update):
            mode = curriculum.pick_mode() if args.mode == "mixed" else args.mode

            pool = pool_2p if mode == "2p" else pool_4p

            # 50% self-play against lagged checkpoints, 50% curriculum
            use_selfplay = selfplay_archive and random.random() < SELFPLAY_RATIO
            if use_selfplay:
                sp_ep, opp_fn = random.choice(selfplay_archive)
                opp_tier = curriculum.current_max_tier
                opp_name = f"self_ep{sp_ep}"
            else:
                tw = curriculum.tier_weights()
                opp_tier, opp_name, opp_fn = pick_opponent(pool, tw, ep, self_play_agent)

            # For 4p, fill remaining slots with pool opponents
            if mode == "4p":
                fillers = [fn for (_, _, fn) in random.choices(pool_4p, k=3)]
                opp = fillers
            else:
                opp = opp_fn

            rollout, final_reward, ep_stat = play_episode(
                model, opp, mode=mode,
                step_penalty=args.step_penalty,
                idle_penalty=args.idle_penalty,
                max_steps=args.max_steps,
                win_bonus=args.win_bonus,
                device=device,
            )
            batch_rollout.extend(rollout)
            ep_rewards.append(final_reward)
            ep_stats_list.append((opp_name, ep_stat))
            total_rollout_steps += len(rollout)

            won = final_reward > 0
            if won:
                wins += 1
            elif final_reward < 0:
                losses += 1
            else:
                draws += 1
            ep += 1

            # Per-opponent tracking
            if opp_name not in opp_record:
                opp_record[opp_name] = [0, 0]
            opp_record[opp_name][1] += 1
            if won:
                opp_record[opp_name][0] += 1

            # Curriculum tracking
            curriculum.record(won, ep, mode=mode)
            curriculum.maybe_promote(ep, max_available_tier)

        if not batch_rollout:
            continue

        # PPO update — move model to update_device for the backward pass
        model.to(update_device)
        model.train()
        # Update reward normalizer with this batch's rewards
        batch_rewards = [s.reward for s in batch_rollout]
        reward_norm.update(batch_rewards)

        update_stats = ppo_update(
            model, optimizer, batch_rollout,
            epochs=args.ppo_epochs,
            mini_batch_size=args.mini_batch_size,
            clip_epsilon=args.clip_epsilon,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            entropy_frac_coef=args.entropy_frac_coef,
            reward_normalizer=reward_norm,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            device=update_device,
        )
        model.eval()
        model.to(device)  # back to rollout device

        # If weights went NaN, reload last good checkpoint
        if update_stats.get("weight_nan"):
            log("WARNING: NaN in weights after update — reloading last checkpoint")
            ckpt = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

        if pid is not None:
            new_lr = pid.step(update_stats["kl"],
                              optimizer.param_groups[0]["lr"])
            optimizer.param_groups[0]["lr"] = new_lr

        # Logging
        elapsed = time.time() - t_start
        total_games = wins + losses + draws
        wr = wins / max(total_games, 1)
        avg_reward = sum(ep_rewards) / max(len(ep_rewards), 1)
        lr_now = optimizer.param_groups[0]["lr"]

        log(
            f"Update ep={ep} ({elapsed:.0f}s) | "
            f"wr={wr:.3f} W={wins} L={losses} D={draws} | "
            f"avg_r={avg_reward:.2f} | "
            f"pol={update_stats['policy_loss']:.4f} "
            f"val={update_stats['value_loss']:.4f} "
            f"ent={update_stats['entropy']:.4f} "
            f"kl={update_stats['kl']:.4f} "
            f"clip={update_stats['clip_frac']:.3f} "
            f"ev={update_stats['explained_var']:.3f} | "
            f"lr={lr_now:.2e} | "
            f"steps={total_rollout_steps} | "
            f"tier={curriculum.current_max_tier} cwr={curriculum.rolling_wr:.0%}"
            f"{f' 4p={curriculum.fourp_ratio:.0%} 4pwr={curriculum.rolling_wr_4p:.0%}' if curriculum.enable_4p else ''}"
        )

        # Per-opponent breakdown
        opp_parts = []
        for name in sorted(opp_record.keys()):
            w, p = opp_record[name]
            opp_parts.append(f"{name}:{w}/{p}")
        if opp_parts:
            log(f"  opponents: {' | '.join(opp_parts)}")

        # Track metrics for graphing
        metric_episodes.append(ep)
        metric_win_rates.append(wr)
        metric_rewards.append(avg_reward)
        metric_pol_loss.append(update_stats['policy_loss'])
        metric_val_loss.append(update_stats['value_loss'])
        metric_tiers.append(curriculum.current_max_tier)
        metric_cwr_2p.append(curriculum.rolling_wr)
        metric_cwr_4p.append(curriculum.rolling_wr_4p)
        metric_ev.append(update_stats['explained_var'])

        # Save latest
        torch.save({
            "model_state_dict": model.state_dict(),
            "episode": ep,
            "d_model": d_model,
            "max_actions": MAX_ACTIONS,
            "separate_critic": separate_critic,
        }, latest_checkpoint)

        # Archive a lagged self-play snapshot every N episodes
        if ep % SELFPLAY_SAVE_EVERY < args.episodes_per_update:
            sp_agent = make_edge_self_play_agent(
                model, num_players, device, args.n_heads, args.n_layers)
            selfplay_archive.append((ep, sp_agent))
            if len(selfplay_archive) > SELFPLAY_MAX_ARCHIVE:
                selfplay_archive.pop(0)
            log(f"  [self-play] archived snapshot at ep {ep} "
                f"({len(selfplay_archive)} in pool)")

        # Eval vs pool
        if ep % args.eval_every < args.episodes_per_update:
            log(f"\n--- Eval at ep {ep} ---")
            eval_pool = pool_2p[:8]  # quick: first 8 opponents
            eval_results = eval_vs_pool(model, eval_pool, "2p", device, n_games=2)
            for name, wr_opp in sorted(eval_results.items(), key=lambda x: -x[1]):
                log(f"  {name}: {wr_opp:.0%}")
            overall_eval_wr = sum(eval_results.values()) / max(len(eval_results), 1)
            log(f"  Overall eval win rate: {overall_eval_wr:.3f}")

            # Save progress graph
            save_progress_graph(
                metric_episodes, metric_win_rates, metric_rewards,
                metric_pol_loss, metric_val_loss, metric_tiers,
                curriculum.promotions,
                str(cache_dir / "ppo_edge_progress.png"),
                cwr_2p=metric_cwr_2p, cwr_4p=metric_cwr_4p, ev=metric_ev,
            )

            if overall_eval_wr > best_win_rate:
                best_win_rate = overall_eval_wr
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "episode": ep,
                    "d_model": d_model,
                    "max_actions": MAX_ACTIONS,
                    "separate_critic": separate_critic,
                    "eval_win_rate": best_win_rate,
                }, best_checkpoint)
                log(f"  *** New best checkpoint (eval_wr={best_win_rate:.3f}) ***\n")

    log(f"\nPPO EdgePolicy training complete.")
    log(f"Best checkpoint: {best_checkpoint}")
    log(f"Latest checkpoint: {latest_checkpoint}")
    log(f"Final win rate: {wins}/{wins+losses+draws} = {wins/(wins+losses+draws):.3f}")
    log_file.close()


if __name__ == "__main__":
    main()
