"""Phase 3: Online PPO fine-tuning against heuristic opponents.

Initializes from Phase 2 (AWR) checkpoint and fine-tunes with standard PPO
using the factored action distribution. Uses kaggle_environments for live play
against a rotating pool of opponents.

The factored log-prob is used for the PPO ratio:
    ratio = exp(new_log_prob - old_log_prob)
where log_prob = log p(source) + log p(target|source) + log p(fraction|source,target)

Usage:
    python -m ppo_gnn.train_ppo --mode 2p --checkpoint ppo_gnn/cache/checkpoint_awr_2p.pt
    python -m ppo_gnn.train_ppo --mode 4p --checkpoint ppo_gnn/cache/checkpoint_awr_4p.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

from .gnn_policy import FRACTION_BUCKETS, NODE_DIM, OrbitWarsGNNPolicy
from .replay_parser import _build_node_features
from .sun_geometry import sun_intersects_path

os.environ.setdefault("KAGGLE_ENVIRONMENTS_QUIET", "1")


# ---------------------------------------------------------------------------
# PID controller for learning rate — keeps KL divergence in a healthy range.
#
# After each PPO update, the measured KL is fed in. If KL > target the LR is
# reduced; if KL < target it recovers. The integral term corrects persistent
# drift; the derivative term prevents oscillation.
#
# Typical PPO target KL: 0.005–0.02. Start with target=0.01.
# ---------------------------------------------------------------------------

class KLPIDController:
    """Proportional-Integral-Derivative controller for PPO learning rate."""

    def __init__(
        self,
        target_kl: float = 0.01,
        kp: float = 0.5,
        ki: float = 0.05,
        kd: float = 0.1,
        lr_min: float = 1e-6,
        lr_max: float = 3e-4,
    ) -> None:
        self.target_kl = target_kl
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.lr_min = lr_min
        self.lr_max = lr_max
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, current_kl: float, current_lr: float) -> float:
        """Return adjusted learning rate given measured KL this update.

        Uses |KL| so both positive and negative divergence trigger LR reduction.
        Only reduces LR, never increases beyond the initial value — prevents
        the runaway LR growth that happens when KL goes persistently negative.
        """
        abs_kl = abs(current_kl)
        error = abs_kl - self.target_kl               # positive → |KL| too high → reduce LR
        self._integral += error
        self._integral = max(-0.5, min(0.5, self._integral))
        derivative = error - self._prev_error
        self._prev_error = error

        correction = self.kp * error + self.ki * self._integral + self.kd * derivative
        # Only allow LR reduction (correction > 0), cap increases at 0
        correction = max(0.0, correction)
        new_lr = current_lr * (1.0 - correction)
        return float(max(self.lr_min, min(self.lr_max, new_lr)))

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RolloutStep:
    node_features: torch.Tensor   # (N, 10)
    positions: torch.Tensor       # (N, 2)
    owned_mask: torch.Tensor      # (N,)
    source: int
    target: int
    fraction: int
    is_noop: bool
    log_prob: float
    value: float
    reward: float
    done: bool


# ---------------------------------------------------------------------------
# Opponent loading
# ---------------------------------------------------------------------------

def load_agent_from_file(path: str) -> Callable:
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
        raise AttributeError(f"No top-level agent() function in {p}")
    return mod.agent


def make_checkpoint_agent(
    checkpoint_path: str,
    num_players: int,
    noop_penalty: float = 4.0,
    device: torch.device = torch.device("cpu"),
) -> Callable:
    """Load a frozen GNN policy from a checkpoint file and return a kaggle agent callable."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    # Detect hidden_dim from checkpoint weights rather than assuming 128
    hidden_dim = 128
    for key, val in state.items():
        if "lin.weight" in key or "lin_l.weight" in key:
            hidden_dim = val.shape[0]
            break
    shadow = OrbitWarsGNNPolicy(hidden_dim=hidden_dim, use_gat=True, mask_sun_targets=True)
    shadow.to(device)
    shadow.eval()
    return make_self_play_agent(shadow, num_players, noop_penalty, device)


def build_opponent_pool(mode: str) -> list[tuple[int, str, Callable | str]]:
    """Build the tiered pool of opponents for curriculum training.

    Returns list of (tier, name, agent_fn) tuples. 8 tiers:

    Tier 1 — truly random / Kaggle built-in starters (absolute floor)
    Tier 2 — simple Java-port pool bots (bully, prospector)
    Tier 3 — slightly harder pool bots (rage, dual)
    Tier 4 — strong rule-based baselines + our BC checkpoint
    Tier 5 — mid-tier external agents + our AWR checkpoint
    Tier 6 — our submission-grade heuristics + kashiwaba RL
    Tier 7 — best mechanic variants + strong externals (LB ~1100-1224)
    Tier 8 — elite external agents (LB ~1000+ pilkwang, leaderboard ceiling)
    """
    root = Path(__file__).parent.parent
    num_players = 2 if mode == "2p" else 4
    ext = root / "submission" / "ext"
    cache = Path(__file__).parent / "cache"

    # Built-in engine agents (always available, no file needed)
    # NOTE: random is the ONLY tier-1 agent — empirically it's the only one the
    # model can beat reliably from scratch (88% win rate). starter_agent wins
    # ~80% against an untrained policy, same as nearest_sniper, so both sit at
    # tier 2/3 where the curriculum can actually teach something.
    from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent, starter_agent
    pool: list[tuple[int, str, Callable | str]] = [
        (1, "random",   random_agent),
        (2, "starter",  starter_agent),   # deliberate strategy — NOT tier 1
    ]
    print(f"  [tier 1] random_agent (builtin)")
    print(f"  [tier 2] starter_agent (builtin)")

    # File-based agents: (tier, name, relative_path)
    candidates = [
        # Tier 2 — simple Java-port pool bots (comparable to starter in difficulty)
        (2, "bully",           "submission/pool_bully.py"),
        (2, "prospector",      "submission/pool_prospector.py"),
        # Tier 3 — nearest-sniper + our harder pool bots
        # nearest_sniper uses efficient distance-sorted expansion — empirically
        # wins ~77% vs untrained policy, similar to starter, but it's very
        # positionally sharp so lives at tier 3 with rage/dual.
        (3, "nearest_sniper",  "submission/ext/pool_baseline_nearest_sniper.py"),
        (3, "rage",            "submission/pool_rage.py"),
        (3, "dual",            "submission/pool_dual.py"),
        # Tier 4 — strong rule-based baselines + our BC checkpoint opponent
        (4, "baseline",        "submission/pool_baseline.py"),
        (4, "sig_starter",     "submission/ext/pool_sigmaborov_starter.py"),
        # Tier 5 — mid-tier externals + ykhnkf (LB ~1100)
        (5, "pascal_v14",      "submission/ext/pool_pascal_orbitwork_v14.py"),
        (5, "ykhnkf_dist",     "submission/ext/pool_ykhnkf_distance_prioritized.py"),
        # Tier 6 — our submission-grade heuristics + kashiwaba RL + sigmaborov reinforce
        (6, "shunlite",        "submission/main_fc_rl_shunlite.py"),
        (6, "v131_2p",         "submission/main_v131_plus_2p.py"),
        (6, "sig_reinforce",   "submission/ext/pool_sigmaborov_reinforce.py"),
        (6, "kashiwaba_rl",    "submission/ext/pool_kashiwaba_rl.py"),
        # Tier 7 — best mechanic variants + strong externals
        (7, "v131_denial",     "submission/main_v131_plus_denial.py"),
        (7, "v131_wave",       "submission/main_v131_plus_wave.py"),
        (7, "tamrazov",        "submission/ext/pool_tamrazov_starwars.py"),
        (7, "yuriy_arch",      "submission/ext/pool_yuriygreben_architect.py"),
        # Tier 8 — elite ceiling agents
        (8, "pilkwang",        "submission/ext/pool_pilkwang_structured.py"),
    ]

    if mode == "4p":
        candidates.extend([
            (6, "plus4p",    "submission/main_v131_plus_4p.py"),
            (7, "political", "submission/main_v131_plus_4p_political.py"),
        ])

    for tier, name, rel_path in candidates:
        full_path = root / rel_path
        if full_path.exists():
            try:
                agent_fn = load_agent_from_file(str(full_path))
                pool.append((tier, name, agent_fn))
                print(f"  [tier {tier}] {name}")
            except Exception as e:
                print(f"  Warning: failed to load {name}: {e}")
        else:
            print(f"  Skipped {name} (not found)")

    # Checkpoint-based opponents: our own GNN at earlier training stages
    # These fill the gap between rule-based baselines and our current best.
    checkpoint_opponents = [
        (4, "opp_bc",      str(cache / "checkpoint_bc_2p.pt")),
        (5, "opp_awr",     str(cache / "checkpoint_awr_2p.pt")),
        (6, "opp_ppo_old", str(cache / "checkpoint_ppo_2p_fwdmetric.pt")),
    ]
    for tier, name, ckpt_path in checkpoint_opponents:
        if Path(ckpt_path).exists():
            try:
                agent_fn = make_checkpoint_agent(ckpt_path, num_players)
                pool.append((tier, name, agent_fn))
                print(f"  [tier {tier}] {name} (checkpoint)")
            except Exception as e:
                print(f"  Warning: failed to load checkpoint {name}: {e}")

    return pool


def make_self_play_agent(
    model: OrbitWarsGNNPolicy,
    num_players: int,
    noop_penalty: float = 4.0,
    device: torch.device = torch.device("cpu"),
) -> Callable:
    """Create a frozen copy of the model wrapped as a kaggle agent function.

    The copy's weights are snapshotted at call time — the training model
    can keep updating without affecting the opponent.
    """
    shadow = OrbitWarsGNNPolicy(
        hidden_dim=model.hidden_dim,
        use_gat=model.use_gat,
        mask_sun_targets=model.mask_sun_targets,
        separate_critic=model.separate_critic,
    )
    shadow.load_state_dict(model.state_dict())
    shadow.to(device)
    shadow.eval()

    def _agent(obs, config):
        planets = obs.get("planets", [])
        if not planets:
            return []
        player = obs.get("player", 0)
        fleets = obs.get("fleets", [])

        has_planets = any(int(p[1]) == player for p in planets)
        if not has_planets:
            return []

        nf, pos, owned = _build_node_features(planets, fleets, player, num_players)
        nf = nf.to(device)
        pos = pos.to(device)
        owned = owned.to(device)

        with torch.no_grad():
            nf_b = nf.unsqueeze(0)
            pos_b = pos.unsqueeze(0)
            om_b = owned.unsqueeze(0)
            N = nf_b.shape[1]

            source_logits, all_target_logits, all_fraction_logits = shadow(
                nf_b, pos_b, om_b,
            )
            source_logits[0, N] -= noop_penalty

            # Multi-action loop
            actions = []
            src_logits = source_logits.squeeze(0).clone()
            for _ in range(N):
                src = torch.distributions.Categorical(logits=src_logits).sample().item()
                if src == N:
                    break
                src_logits[src] = -1e9

                tgt = torch.distributions.Categorical(logits=all_target_logits[0, src]).sample().item()
                frac = torch.distributions.Categorical(logits=all_fraction_logits[0, src, tgt]).sample().item()

                src_planet = planets[src]
                tgt_planet = planets[tgt]
                sx, sy = float(src_planet[2]), float(src_planet[3])
                tx, ty = float(tgt_planet[2]), float(tgt_planet[3])
                if sun_intersects_path(sx, sy, tx, ty):
                    continue
                angle = math.atan2(ty - sy, tx - sx)
                ships = int(float(src_planet[5]) * FRACTION_BUCKETS[frac])
                if ships < 1:
                    continue
                actions.append([int(src_planet[0]), angle, ships])

        return actions

    return _agent


# Curriculum: each stage defines which opponent tiers are available,
# the win-rate threshold to advance, and the self-play/heuristic mix.
# Self-play is intentionally absent in early stages — a weak random policy
# teaches the agent nothing; it needs real heuristic opponents first.
#
# advance_wr: heuristic win rate needed to unlock next stage
# min_games:  minimum heuristic games before considering advancement
# max_tier:   only draw from pool bots of this tier or lower
# self_play:  fraction of episodes using frozen self-play copy
# heuristic:  fraction of episodes using tiered heuristics
CURRICULUM_STAGES = [
    {"name": "floor",        "advance_wr": 0.55, "min_games": 30,  "max_tier": 1, "self_play": 0.00, "heuristic": 1.00},
    {"name": "beginner",     "advance_wr": 0.65, "min_games": 150, "max_tier": 2, "self_play": 0.05, "heuristic": 0.95},
    {"name": "novice",       "advance_wr": 0.58, "min_games": 200, "max_tier": 3, "self_play": 0.10, "heuristic": 0.90},
    {"name": "developing",   "advance_wr": 0.52, "min_games": 250, "max_tier": 4, "self_play": 0.15, "heuristic": 0.85},
    {"name": "intermediate", "advance_wr": 0.46, "min_games": 300, "max_tier": 5, "self_play": 0.20, "heuristic": 0.80},
    {"name": "competent",    "advance_wr": 0.40, "min_games": 350, "max_tier": 6, "self_play": 0.25, "heuristic": 0.75},
    {"name": "advanced",     "advance_wr": 0.35, "min_games": 400, "max_tier": 7, "self_play": 0.30, "heuristic": 0.70},
    {"name": "elite",        "advance_wr": 1.01, "min_games": 500, "max_tier": 8, "self_play": 0.35, "heuristic": 0.65},
]

# How often to refresh the self-play shadow weights (episodes)
SELF_PLAY_REFRESH_INTERVAL = 20
# How often to refresh the "2000 episodes ago" lagging reference (episodes)
LAGGING_REFRESH_INTERVAL = 2000

# Performance-gated horizon schedule: (heuristic_wr_threshold, max_steps)
# Horizon only advances when the agent proves it can beat heuristics at
# the current horizon. No time pressure — it stays until it's ready.
# Thresholds are deliberately high to prevent premature advancement.
HORIZON_STAGES = [
    (0.0, 100),    # start: 100 steps
    (0.65, 200),   # need 65% heur wr to unlock 200 steps
    (0.65, 350),   # need 65% heur wr at 200 steps to unlock 350
    (0.65, 500),   # need 65% heur wr at 350 steps to unlock full game
]


def pick_curriculum_opponent(
    pool: list[tuple[int, str, Callable | str]],
    self_play_agent: Callable | None,
    stage: int,
    stages: list | None = None,
) -> tuple[str, Callable | str]:
    """Pick an opponent based on the current curriculum stage and max tier.

    Within the available tier range, sampling is tier-weighted: lower tiers
    are more likely, ensuring the model always has winnable games for a stable
    gradient signal even at advanced stages.  Weight = (max_tier - tier + 1)^2
    so tier-1 agents are heavily favoured over tier-8 in a mixed pool.
    """
    _stages = stages if stages is not None else CURRICULUM_STAGES
    cfg = _stages[min(stage, len(_stages) - 1)]
    max_tier = cfg["max_tier"]

    r = random.random()
    if r < cfg["self_play"] and self_play_agent is not None:
        return ("self_play", self_play_agent)

    available = [(tier, name, fn) for tier, name, fn in pool if tier <= max_tier]
    if available:
        # Weight inversely by tier so easier opponents are more frequent.
        # This keeps the gradient signal healthy while still exposing the model
        # to hard opponents enough to drive improvement.
        weights = [(max_tier - tier + 1) ** 2 for tier, _, _ in available]
        chosen = random.choices(available, weights=weights, k=1)[0]
        return (chosen[1], chosen[2])

    if self_play_agent is not None:
        return ("self_play", self_play_agent)
    return ("random", "random")


# ---------------------------------------------------------------------------
# GAE and PPO update (unchanged from Phase 3 template)
# ---------------------------------------------------------------------------

def compute_gae(
    rollout: List[RolloutStep],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns from a rollout."""
    T = len(rollout)
    rewards = torch.tensor([s.reward for s in rollout])
    values = torch.tensor([s.value for s in rollout])
    dones = torch.tensor([1.0 if s.done else 0.0 for s in rollout])

    advantages = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = 0.0 if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: OrbitWarsGNNPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: List[RolloutStep],
    max_planets: int = 48,
    epochs: int = 4,
    mini_batch_size: int = 64,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.015,
    kl_coef: float = 0.1,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    device: torch.device = torch.device("cpu"),
    noop_penalty: float = 0.0,
) -> dict[str, float]:
    """Run PPO update on a collected rollout."""
    advantages, returns = compute_gae(rollout, gamma, gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    T = len(rollout)
    M = max_planets
    nf_all = torch.zeros(T, M, NODE_DIM)
    pos_all = torch.zeros(T, M, 2)
    owned_all = torch.zeros(T, M)
    sources = torch.zeros(T, dtype=torch.long)
    targets = torch.zeros(T, dtype=torch.long)
    fractions = torch.zeros(T, dtype=torch.long)
    is_noops = torch.zeros(T)
    old_log_probs = torch.zeros(T)

    for i, step in enumerate(rollout):
        n = step.node_features.shape[0]
        n = min(n, M)
        nf_all[i, :n] = step.node_features[:n]
        pos_all[i, :n] = step.positions[:n]
        owned_all[i, :n] = step.owned_mask[:n]
        sources[i] = step.source if not step.is_noop else M
        targets[i] = step.target if not step.is_noop else 0
        fractions[i] = step.fraction if not step.is_noop else 0
        is_noops[i] = 1.0 if step.is_noop else 0.0
        old_log_probs[i] = step.log_prob

    nf_all = nf_all.to(device)
    pos_all = pos_all.to(device)
    owned_all = owned_all.to(device)
    sources = sources.to(device)
    targets = targets.to(device)
    fractions = fractions.to(device)
    is_noops = is_noops.to(device)
    old_log_probs = old_log_probs.to(device)
    advantages = advantages.to(device)
    returns = returns.to(device)

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0, "kl": 0.0}
    num_updates = 0

    for _ in range(epochs):
        perm = torch.randperm(T, device=device)
        for start in range(0, T, mini_batch_size):
            idx = perm[start : start + mini_batch_size]

            log_prob, entropy, value = model.evaluate_action(
                nf_all[idx], pos_all[idx], owned_all[idx],
                sources[idx], targets[idx], fractions[idx], is_noops[idx],
                noop_penalty=noop_penalty,
            )

            ratio = torch.exp(log_prob - old_log_probs[idx])
            adv = advantages[idx]

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(value, returns[idx])
            entropy_loss = -entropy.mean()

            # KL anchor: penalise large deviations from the old policy.
            # Approximates KL(old || new) ≈ (log_prob_old - log_prob_new).mean()
            # This prevents the 4k→6k regression where the policy drifted too far.
            kl_approx = (old_log_probs[idx] - log_prob).mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss + kl_coef * kl_approx

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                clip_frac = ((ratio - 1).abs() > clip_epsilon).float().mean().item()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += -entropy_loss.item()
            stats["clip_frac"] += clip_frac
            stats["kl"] += kl_approx.item()
            num_updates += 1

    for k in stats:
        stats[k] /= max(num_updates, 1)
    return stats


# ---------------------------------------------------------------------------
# Rollout collection using kaggle_environments
# ---------------------------------------------------------------------------

def play_episode(
    model: OrbitWarsGNNPolicy,
    opponent: "Callable | str | list[Callable | str]",
    mode: str = "2p",
    noop_penalty: float = 4.0,
    idle_penalty: float = 0.02,
    step_penalty: float = 0.002,
    max_steps: int = 500,
    win_bonus: float = 10.0,
    device: torch.device = torch.device("cpu"),
) -> tuple[List[RolloutStep], float, dict[str, float | int | bool]]:
    """Play one full episode against an opponent using kaggle_environments.

    Args:
        opponent: Single callable for 2p, or list of 3 callables for 4p slots 1-3.
        max_steps: Horizon cap. If < 500, episode is cut short and a partial
            terminal reward is assigned based on fleet+production advantage.
        win_bonus: Terminal reward for a win. Default 10.0, use higher for
            heuristic opponents to give stronger learning signal.

    Returns:
        (rollout, final_reward, ep_stats).
    """
    from kaggle_environments import make

    env = make("orbit_wars", debug=False)
    num_players = 2 if mode == "2p" else 4

    # Set up agents: our model in slot 0, opponent(s) in remaining slots
    if mode == "2p":
        trainer = env.train([None, opponent])
    else:
        # 4p: accept a list of 3 opponents (one per slot) so they fight each
        # other, giving the model a realistic chance. Fall back to 3x clone
        # if a single callable is passed.
        if isinstance(opponent, list):
            slots = opponent          # expect exactly 3
        else:
            slots = [opponent] * 3
        trainer = env.train([None] + slots)

    obs = trainer.reset()
    model.eval()
    rollout: List[RolloutStep] = []
    env_steps = 0
    num_launches = 0
    num_noops = 0
    # Dense reward shaping: track fleet+production advantage each step.
    # 0.01 is the shaping scale; 10.0 is the terminal bonus (see end of function).
    prev_shaped_score = 0.0
    prev_enemy_total = 0.0  # track opponent fleet+prod for delta-v signal

    for step_idx in range(max_steps):
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        player = obs.get("player", 0)

        # Skip empty observations (step 0)
        if not planets:
            obs, reward, done, info = trainer.step([])
            env_steps += 1
            if done:
                break
            continue

        # Check if we're still alive
        has_planets = any(int(p[1]) == player for p in planets)
        if not has_planets:
            obs, reward, done, info = trainer.step([])
            env_steps += 1
            if done:
                break
            continue

        nf, pos, owned = _build_node_features(planets, fleets, player, num_players,
                                                step=step_idx, max_steps=max_steps)
        nf = nf.to(device)
        pos = pos.to(device)
        owned = owned.to(device)

        with torch.no_grad():
            nf_b = nf.unsqueeze(0)
            pos_b = pos.unsqueeze(0)
            om_b = owned.unsqueeze(0)
            N = nf_b.shape[1]

            source_logits, all_target_logits, all_fraction_logits = model(
                nf_b, pos_b, om_b,
            )
            h, _, g = model._encode(nf_b, pos_b)
            value = model.value_head(g).squeeze(-1).item()

            # Apply noop penalty
            source_logits[0, N] -= noop_penalty

            # Multi-action loop: sample multiple sources per step
            step_actions = []
            step_log_probs = []
            step_sources = []
            step_targets = []
            step_fracs = []
            src_logits_mut = source_logits.squeeze(0).clone()
            # Unmasked source dist — must match evaluate_action() for PPO ratio
            source_dist_clean = torch.distributions.Categorical(
                logits=source_logits.squeeze(0))
            used_sources = set()

            for _sub in range(N):  # max N sub-actions
                source_dist = torch.distributions.Categorical(logits=src_logits_mut)
                src = source_dist.sample().item()

                if src == N:  # noop = stop firing
                    # Log-prob from CLEAN dist (matches evaluate_action)
                    log_p_src_clean = source_dist_clean.log_prob(
                        torch.tensor(src)).item()
                    step_sources.append(src)
                    step_targets.append(0)
                    step_fracs.append(0)
                    step_log_probs.append(log_p_src_clean)
                    break

                # Log-prob from CLEAN dist (matches evaluate_action)
                log_p_src_clean = source_dist_clean.log_prob(
                    torch.tensor(src)).item()

                # Mask this source for sampling next sub-action
                used_sources.add(src)
                src_logits_mut[src] = -1e9

                tgt_logits = all_target_logits[0, src]
                target_dist = torch.distributions.Categorical(logits=tgt_logits)
                tgt = target_dist.sample().item()
                log_p_tgt = target_dist.log_prob(torch.tensor(tgt)).item()

                frac_logits = all_fraction_logits[0, src, tgt]
                frac_dist = torch.distributions.Categorical(logits=frac_logits)
                frac = frac_dist.sample().item()
                log_p_frac = frac_dist.log_prob(torch.tensor(frac)).item()

                log_prob_sub = log_p_src_clean + log_p_tgt + log_p_frac

                # Convert to Kaggle action
                src_planet = planets[src]
                tgt_planet = planets[tgt]
                sx, sy = float(src_planet[2]), float(src_planet[3])
                tx, ty = float(tgt_planet[2]), float(tgt_planet[3])

                if sun_intersects_path(sx, sy, tx, ty):
                    continue  # skip but don't break
                angle = math.atan2(ty - sy, tx - sx)
                ships = int(float(src_planet[5]) * FRACTION_BUCKETS[frac])
                if ships < 1:
                    continue

                step_actions.append([int(src_planet[0]), angle, ships])
                step_sources.append(src)
                step_targets.append(tgt)
                step_fracs.append(frac)
                step_log_probs.append(log_prob_sub)
            else:
                # Exhausted all sources without noop — add implicit noop
                log_p_noop = source_dist_clean.log_prob(
                    torch.tensor(N)).item()
                step_sources.append(N)
                step_targets.append(0)
                step_fracs.append(0)
                step_log_probs.append(log_p_noop)

            action = step_actions
            is_noop = len(step_actions) == 0
            num_launches += len(step_actions)
            if is_noop:
                num_noops += 1

        # Step the environment with all actions for this step
        obs, reward, done, info = trainer.step(action)
        env_steps += 1

        # Dense reward: fleet + production advantage delta with opponent delta-v.
        my_fleet = 0.0
        my_prod = 0.0
        enemy_fleet = 0.0
        enemy_prod = 0.0
        for p in planets:
            owner = int(p[1])
            ships = float(p[5])
            prod = float(p[6])
            if owner == player:
                my_fleet += ships
                my_prod += prod
            elif owner >= 0 and owner != player:
                enemy_fleet += ships
                enemy_prod += prod

        PROD_WEIGHT = 3.0
        score = (my_fleet + PROD_WEIGHT * my_prod) - (enemy_fleet + PROD_WEIGHT * enemy_prod)
        delta = score - prev_shaped_score
        prev_shaped_score = score

        enemy_total = enemy_fleet + PROD_WEIGHT * enemy_prod
        enemy_delta = enemy_total - prev_enemy_total
        prev_enemy_total = enemy_total

        shaped_reward = delta * 0.01 - enemy_delta * 0.005 - step_penalty

        if is_noop and has_planets and my_fleet > 50:
            shaped_reward -= idle_penalty

        # Record one RolloutStep per sub-action. Last sub-action gets the
        # shaped reward; earlier ones get 0 (they share the same env step).
        n_sub = len(step_sources)
        for si in range(n_sub):
            sub_is_noop = (step_sources[si] == N)
            sub_reward = shaped_reward if si == n_sub - 1 else 0.0
            sub_done = done if si == n_sub - 1 else False
            rollout.append(RolloutStep(
                node_features=nf.cpu(),
                positions=pos.cpu(),
                owned_mask=owned.cpu(),
                source=step_sources[si] if not sub_is_noop else 0,
                target=step_targets[si],
                fraction=step_fracs[si],
                is_noop=sub_is_noop,
                log_prob=step_log_probs[si],
                value=value,
                reward=sub_reward,
                done=sub_done,
            ))

        if done:
            break

    # Terminal reward: ±10.0 to dominate over dense shaping (which uses 0.01 scale)
    raw_reward = reward if reward is not None else 0.0
    hit_horizon = (not done) and (env_steps >= max_steps)

    # Compute final fleet/prod snapshot (needed for horizon-cutoff metric and ep_stats)
    my_fleet_final = 0.0
    my_prod_final = 0.0
    enemy_fleet_final = 0.0
    enemy_prod_final = 0.0
    owned_planets_final = 0
    if obs and "planets" in obs:
        for p in obs["planets"]:
            owner = int(p[1])
            ships = float(p[5])
            prod = float(p[6])
            if owner == player:
                owned_planets_final += 1
                my_fleet_final += ships
                my_prod_final += prod
            elif owner >= 0 and owner != player:
                enemy_fleet_final += ships
                enemy_prod_final += prod

    my_assets_final = my_fleet_final + 3.0 * my_prod_final
    enemy_assets_final = enemy_fleet_final + 3.0 * enemy_prod_final
    eliminated = owned_planets_final == 0

    if raw_reward > 0:
        final_reward = win_bonus
    elif raw_reward < 0:
        final_reward = -win_bonus
        # Survival bonus: lasting longer softens the loss penalty
        if env_steps > 0:
            survival_frac = env_steps / max_steps
            final_reward = final_reward * (1.0 - 0.5 * survival_frac)
    elif eliminated and enemy_assets_final > 0:
        final_reward = -win_bonus
        if env_steps > 0:
            survival_frac = env_steps / max_steps
            final_reward = final_reward * (1.0 - 0.5 * survival_frac)
    elif hit_horizon:
        # Differential horizon metric with time-value weighting and tanh smoothing.
        # Avoids the binary 450× problem: instead normalises by expected game-scale
        # values so proportional differences produce proportional signals.
        #
        # At step 50 (horizon cutoff), time_value ≈ 0.9, so:
        #   advantage ≈ 0.1 × fleet_diff + 1.8 × prod_diff
        # Production is weighted heavily because it compounds over remaining steps,
        # but tanh prevents saturation from dominating the gradient.
        FULL_GAME_LENGTH = 500
        step = env_steps
        remaining_steps = FULL_GAME_LENGTH - step
        time_value = remaining_steps / FULL_GAME_LENGTH

        # Expected game-scale at this step (normalising constants, not hard targets)
        expected_fleet = max(50 + step * 3, 1.0)   # ~200 at step 50
        expected_prod  = max(5  + step * 0.2, 1.0)  # ~15 at step 50

        fleet_diff = (my_fleet_final - enemy_fleet_final) / expected_fleet
        prod_diff  = (my_prod_final  - enemy_prod_final)  / expected_prod

        advantage = (1 - time_value) * fleet_diff + time_value * 1.5 * prod_diff
        advantage_signal = math.tanh(advantage)
        final_reward = advantage_signal * (win_bonus / 2.0)
    else:
        final_reward = 0.0

    if rollout and final_reward != 0:
        # Assign final reward to last step, let GAE propagate it back
        rollout[-1] = RolloutStep(
            **{**rollout[-1].__dict__, "reward": final_reward, "done": True},
        )

    ep_stats = {
        "my_fleet": my_fleet_final,
        "my_prod": my_prod_final,
        "enemy_fleet": enemy_fleet_final,
        "enemy_prod": enemy_prod_final,
        "env_steps": env_steps,
        "owned_planets": owned_planets_final,
        "eliminated": eliminated,
        "hit_horizon": hit_horizon,
        "num_launches": num_launches,
        "num_noops": num_noops,
    }

    return rollout, final_reward, ep_stats


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Parallel episode collection (for --parallel-envs > 1)
# ---------------------------------------------------------------------------
# Module-level globals populated by worker initializer (spawn context).
_par_model: Optional[OrbitWarsGNNPolicy] = None
_par_device: Optional[torch.device] = None
_par_opp_lookup_2p: dict = {}   # name -> callable
_par_opp_lookup_4p: dict = {}   # name -> callable


def _par_init(state_dict_bytes, hidden_dim, use_gat, mask_sun, device_str, noop_penalty):
    """Initializer for each spawned worker: load model and build opponents."""
    global _par_model, _par_device, _par_opp_lookup_2p, _par_opp_lookup_4p
    import io
    _par_device = torch.device(device_str)
    _par_model = OrbitWarsGNNPolicy(
        hidden_dim=hidden_dim, use_gat=use_gat, mask_sun_targets=mask_sun,
    )
    sd = torch.load(io.BytesIO(state_dict_bytes), map_location="cpu", weights_only=True)
    _par_model.load_state_dict(sd)
    _par_model.to(_par_device)
    _par_model.eval()

    # Build opponent pools
    h2p = build_opponent_pool("2p")
    h4p = build_opponent_pool("4p")
    _par_opp_lookup_2p = {name: fn for _, name, fn in h2p}
    _par_opp_lookup_4p = {name: fn for _, name, fn in h4p}
    # Self-play / baseline / lagging all start as copies of current model
    for label in ("self_play", "baseline", "lagging"):
        _par_opp_lookup_2p[label] = make_self_play_agent(
            _par_model, 2, noop_penalty, _par_device)
        _par_opp_lookup_4p[label] = make_self_play_agent(
            _par_model, 4, noop_penalty, _par_device)


def _par_play(task):
    """Worker: play one episode. Resolves opponents from worker-local globals."""
    mode = task["mode"]
    opp_name = task["opp_name"]
    lookup = _par_opp_lookup_4p if mode == "4p" else _par_opp_lookup_2p

    if mode == "4p":
        opp = [lookup[n] for n in task["opp_names_4p"]]
    else:
        opp = lookup[opp_name]

    rollout, reward, stats = play_episode(
        _par_model, opp, mode=mode,
        noop_penalty=task["noop_penalty"],
        idle_penalty=task["idle_penalty"],
        step_penalty=task["step_penalty"],
        max_steps=task["max_steps"],
        win_bonus=task["win_bonus"],
        device=_par_device,
    )
    return {
        "rollout": rollout,
        "final_reward": reward,
        "ep_stats": stats,
        "opp_name": opp_name,
        "log_opp_name": task["log_opp_name"],
        "is_heuristic": task["is_heuristic"],
        "ep_num": task["ep_num"],
    }


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Online PPO")
    parser.add_argument("--mode", choices=["2p", "4p", "mixed"], required=True)
    parser.add_argument("--checkpoint", default=None, help="Phase 2 (AWR) checkpoint (omit for fresh start)")
    parser.add_argument("--cache-dir", default="ppo_gnn/cache")
    parser.add_argument("--num-episodes", type=int, default=1000000,
                        help="Total episodes to train on (runs until performance gates are met)")
    parser.add_argument("--episodes-per-update", type=int, default=4,
                        help="Episodes to collect before each PPO update")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--use-sage", action="store_true")
    parser.add_argument("--mask-sun", action="store_true")
    parser.add_argument("--max-planets", type=int, default=48)
    parser.add_argument("--noop-penalty", type=float, default=4.0)
    parser.add_argument("--idle-penalty", type=float, default=0.02)
    parser.add_argument("--step-penalty", type=float, default=0.002,
                        help="Per-step negative reward to encourage fast wins")
    parser.add_argument("--eval-every", type=int, default=500,
                        help="Evaluate against full pool every N episodes")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum (uniform random opponent)")
    parser.add_argument("--progressive-horizon", action="store_true",
                        help="Enable progressive horizon (50→100→200→500 steps)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Fixed max steps per episode (ignored if --progressive-horizon)")
    parser.add_argument("--separate-critic", action="store_true",
                        help="Use separate GNN backbone for value function (prevents "
                             "policy/value gradient interference)")
    parser.add_argument("--device", default="cpu",
                        help="Device for rollout inference (cpu recommended)")
    parser.add_argument("--update-device", default=None,
                        help="Device for PPO gradient updates (default: same as --device, "
                             "use 'mps' to accelerate updates while keeping rollouts on cpu)")
    parser.add_argument("--pid-lr", action="store_true",
                        help="Enable PID controller to auto-adjust LR to keep KL near --target-kl")
    parser.add_argument("--target-kl", type=float, default=0.01,
                        help="Target KL divergence per update for PID controller (default: 0.01)")
    parser.add_argument("--entropy-coef", type=float, default=0.015,
                        help="Entropy regularization coefficient in PPO loss (default: 0.015; "
                             "raise to 0.05+ to prevent policy collapse in 4p)")
    parser.add_argument("--gamma", type=float, default=0.997,
                        help="Discount factor (default: 0.997; higher = longer horizon)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda for advantage estimation (default: 0.95)")
    parser.add_argument("--parallel-envs", type=int, default=1,
                        help="Number of episodes to collect in parallel (default: 1). "
                             "Uses multiprocessing fork to run env simulations concurrently.")
    args = parser.parse_args()

    device = torch.device(args.device)
    update_device = torch.device(args.update_device) if args.update_device else device
    cache_dir = Path(args.cache_dir)

    # Load model
    model = OrbitWarsGNNPolicy(
        hidden_dim=args.hidden_dim,
        use_gat=not args.use_sage,
        mask_sun_targets=args.mask_sun,
        separate_critic=args.separate_critic,
    )
    if args.checkpoint:
        saved_sd = torch.load(args.checkpoint, weights_only=True)
        # If loading a shared-backbone checkpoint into a separate-critic model,
        # duplicate the policy backbone weights into the critic backbone.
        if args.separate_critic:
            missing, unexpected = model.load_state_dict(saved_sd, strict=False)
            # Copy policy encoder weights to critic encoder for warm start
            if any(k.startswith("critic_") for k in missing):
                copy_map = {
                    "node_encoder": "critic_node_encoder",
                    "edge_encoder": "critic_edge_encoder",
                    "gnn1": "critic_gnn1",
                    "gnn2": "critic_gnn2",
                    "global_proj": "critic_global_proj",
                }
                for src_prefix, dst_prefix in copy_map.items():
                    for k, v in saved_sd.items():
                        if k.startswith(src_prefix + "."):
                            dst_key = dst_prefix + k[len(src_prefix):]
                            if dst_key in dict(model.named_parameters()) or dst_key in dict(model.named_buffers()):
                                model.state_dict()[dst_key].copy_(v)
                # Verify
                still_missing, _ = model.load_state_dict(model.state_dict(), strict=True)
                print(f"Loaded checkpoint with critic backbone initialized from policy weights")
            else:
                print(f"Loaded checkpoint (separate critic weights found)")
        else:
            model.load_state_dict(saved_sd)
            print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("Fresh start (no checkpoint) — random initialization")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_path = str(cache_dir / f"checkpoint_ppo_{args.mode}.pt")

    # Mixed mode: train on both 2p and 4p with a schedule
    is_mixed = args.mode == "mixed"
    if is_mixed:
        heuristics_2p = build_opponent_pool("2p")
        heuristics_4p = build_opponent_pool("4p")
        heuristics = heuristics_2p  # default for curriculum tracking
        print(f"2p opponents ({len(heuristics_2p)}): {[n for _, n, _ in heuristics_2p]}")
        print(f"4p opponents ({len(heuristics_4p)}): {[n for _, n, _ in heuristics_4p]}")
    else:
        heuristics = build_opponent_pool(args.mode)
        print(f"Heuristic opponents ({len(heuristics)}): {[name for _, name, _ in heuristics]}")

    use_curriculum = not args.no_curriculum
    effective_mode = "2p" if args.mode != "4p" else "4p"
    num_players = 2 if effective_mode == "2p" else 4

    # In 4p the random-win baseline is 0.25 (1-in-4), not 0.50.
    # Scale advance_wr thresholds proportionally so the curriculum
    # advances at the same *relative* difficulty as in 2p.
    if args.mode == "4p":
        import copy as _copy
        _scale = 0.5   # 0.25 / 0.50
        CURRICULUM_STAGES_EFFECTIVE = _copy.deepcopy(CURRICULUM_STAGES)
        for _s in CURRICULUM_STAGES_EFFECTIVE:
            if _s["advance_wr"] < 1.0:          # skip the terminal sentinel
                _s["advance_wr"] = round(_s["advance_wr"] * _scale, 3)
    else:
        CURRICULUM_STAGES_EFFECTIVE = CURRICULUM_STAGES

    # Create initial self-play agents (frozen copies of model)
    # For mixed mode, create both 2p and 4p variants
    if is_mixed:
        self_play_fn_2p = make_self_play_agent(model, 2, args.noop_penalty, device)
        self_play_fn_4p = make_self_play_agent(model, 4, args.noop_penalty, device)
        baseline_fn_2p = make_self_play_agent(model, 2, args.noop_penalty, device)
        baseline_fn_4p = make_self_play_agent(model, 4, args.noop_penalty, device)
        lagging_fn_2p = make_self_play_agent(model, 2, args.noop_penalty, device)
        lagging_fn_4p = make_self_play_agent(model, 4, args.noop_penalty, device)
        self_play_fn = self_play_fn_2p  # default
    else:
        self_play_fn = make_self_play_agent(model, num_players, args.noop_penalty, device)

    # Reference opponents for tracking absolute progress:
    # 1. "baseline" — frozen copy of the starting checkpoint (never updated)
    if not is_mixed:
        baseline_fn = make_self_play_agent(model, num_players, args.noop_penalty, device)
    # 2. "lagging" — snapshot from ~2000 episodes ago (refreshed periodically)
    if not is_mixed:
        lagging_fn = make_self_play_agent(model, num_players, args.noop_penalty, device)
    baseline_results: list[float] = []   # rolling window vs baseline
    lagging_results: list[float] = []    # rolling window vs lagging

    # Training state
    total_wins = 0
    total_episodes = 0
    best_heuristic_wr = 0.0
    window_results: list[float] = []       # rolling window (all opponents)
    heuristic_results: list[float] = []    # rolling window (heuristic-only)
    curriculum_stage = 0
    horizon_stage = 0  # index into HORIZON_STAGES

    # Mixed mode: 4p fraction schedule (starts at 5%, ramps to 50%)
    mixed_4p_frac_start = 0.05
    mixed_4p_frac_end = 0.50
    mixed_4p_ramp_episodes = 10000  # linear ramp over this many episodes

    # PID controller for learning rate
    use_pid = args.pid_lr
    pid = KLPIDController(
        target_kl=args.target_kl,
        kp=0.5, ki=0.05, kd=0.1,
        lr_min=1e-6, lr_max=3e-4,
    )
    current_lr = args.lr

    use_progressive = args.progressive_horizon

    print(f"\nStarting PPO training (runs until done, max {args.num_episodes} episodes)")
    print(f"  Mode: {args.mode}, LR: {args.lr}, Noop penalty: {args.noop_penalty}")
    print(f"  Device: rollout={device}, update={update_device}")
    if is_mixed:
        print(f"  Mixed mode: 4p fraction {mixed_4p_frac_start:.0%} → {mixed_4p_frac_end:.0%} "
              f"over {mixed_4p_ramp_episodes} episodes")
    if use_pid:
        print(f"  PID LR control: ON (target KL={args.target_kl}, "
              f"kp=0.5 ki=0.05 kd=0.1, range=[1e-6, 3e-4])")
    print(f"  Entropy coef: {args.entropy_coef}")
    if use_curriculum:
        print(f"  Curriculum: ON (stage 0 = {CURRICULUM_STAGES_EFFECTIVE[0]['name']}, "
              f"max_tier={CURRICULUM_STAGES_EFFECTIVE[0]['max_tier']})")
    else:
        print(f"  Curriculum: OFF (uniform random from all opponents)")
    if use_progressive:
        print(f"  Progressive horizon: ON, performance-gated "
              f"({' → '.join(str(s) for _, s in HORIZON_STAGES)} steps, "
              f"advances at 75% heur wr)")
    else:
        print(f"  Max steps: {args.max_steps}")
    if args.parallel_envs > 1:
        print(f"  Parallel envs: {args.parallel_envs} (fork)")
    print()

    # Helper: build opponent selection for one episode
    def _select_opponent(ep_num, ep_mode):
        """Select opponent(s) for one episode.
        
        Returns (opp_name, opp_fn_or_list, log_name, is_heuristic, opp_names_4p).
        opp_names_4p is a list of 3 opponent names for 4p mode (None for 2p).
        """
        if is_mixed:
            ep_heuristics = heuristics_4p if ep_mode == "4p" else heuristics_2p
            ep_self_play = self_play_fn_4p if ep_mode == "4p" else self_play_fn_2p
            ep_baseline = baseline_fn_4p if ep_mode == "4p" else baseline_fn_2p
            ep_lagging = lagging_fn_4p if ep_mode == "4p" else lagging_fn_2p
        else:
            ep_heuristics = heuristics
            ep_self_play = self_play_fn
            ep_baseline = baseline_fn
            ep_lagging = lagging_fn

        ref_roll = random.random()
        if ref_roll < 0.05:
            opp_name, opp_fn = "baseline", ep_baseline
        elif ref_roll < 0.10:
            opp_name, opp_fn = "lagging", ep_lagging
        elif use_curriculum:
            opp_name, opp_fn = pick_curriculum_opponent(
                ep_heuristics, ep_self_play, curriculum_stage,
                stages=CURRICULUM_STAGES_EFFECTIVE,
            )
        else:
            if ep_heuristics:
                _, opp_name, opp_fn = random.choice(ep_heuristics)
            else:
                opp_name, opp_fn = "self_play", ep_self_play

        opp_names_4p = None
        if ep_mode == "4p":
            extra_slots = []
            extra_names = []
            for _ in range(3):
                en, ef = pick_curriculum_opponent(
                    ep_heuristics, ep_self_play, curriculum_stage,
                    stages=CURRICULUM_STAGES_EFFECTIVE,
                )
                extra_slots.append(ef)
                extra_names.append(en)
            final_opp = extra_slots
            log_opp_name = f"[4p]{opp_name}|{'|'.join(extra_names)}"
            opp_names_4p = extra_names
        else:
            final_opp = opp_fn
            log_opp_name = f"[{ep_mode}]{opp_name}" if is_mixed else opp_name

        is_heuristic = opp_name not in ("random", "self_play", "baseline", "lagging")
        return opp_name, final_opp, log_opp_name, is_heuristic, opp_names_4p

    # Helper: process one episode result into tracking structures
    def _process_result(opp_name, rollout, final_reward, ep_stats, log_opp_name,
                        is_heuristic, ep_num, elapsed, horizon):
        nonlocal total_episodes, total_wins

        combined_rollout.extend(rollout)
        ep_rewards.append(final_reward)
        window_results.append(final_reward)
        if len(window_results) > 50:
            window_results.pop(0)

        if is_heuristic:
            heuristic_results.append(final_reward)
            if len(heuristic_results) > 100:
                heuristic_results.pop(0)

        if opp_name == "baseline":
            baseline_results.append(final_reward)
            if len(baseline_results) > 50:
                baseline_results.pop(0)
        elif opp_name == "lagging":
            lagging_results.append(final_reward)
            if len(lagging_results) > 50:
                lagging_results.pop(0)

        total_episodes += 1
        if final_reward > 0:
            total_wins += 1

        result = "WIN" if final_reward > 0 else ("LOSS" if final_reward < 0 else "DRAW")
        noop_count = sum(1 for s in rollout if s.is_noop)
        launch_count = len(rollout) - noop_count
        horizon_str = f"/{horizon}" if use_progressive else ""
        mf = ep_stats["my_fleet"]
        mp_ = ep_stats["my_prod"]
        ef = ep_stats["enemy_fleet"]
        ep_ = ep_stats["enemy_prod"]
        env_steps = int(ep_stats.get("env_steps", len(rollout)))
        print(
            f"  Ep {ep_num+1:>5}/{args.num_episodes} vs {log_opp_name:<30} "
            f"{result:>4}  steps={env_steps:>3}{horizon_str}  "
            f"fleet={mf:.0f}v{ef:.0f}  prod={mp_:.1f}v{ep_:.1f}  "
            f"launch={launch_count:>3}  noop={noop_count:>3}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    use_parallel = args.parallel_envs > 1
    _par_pool = None
    if use_parallel:
        import io as _io
        _spawn_ctx = mp.get_context("spawn")
        # Serialize current model weights for workers
        _buf = _io.BytesIO()
        torch.save(model.state_dict(), _buf)
        _sd_bytes = _buf.getvalue()
        _par_pool = _spawn_ctx.Pool(
            processes=args.parallel_envs,
            initializer=_par_init,
            initargs=(_sd_bytes, args.hidden_dim, not args.use_sage,
                      args.mask_sun, str(device), args.noop_penalty),
        )

    for ep_start in range(0, args.num_episodes, args.episodes_per_update):
        # Collect episodes
        combined_rollout: List[RolloutStep] = []
        ep_rewards = []

        # Refresh self-play / lagging shadows (before any collection)
        first_ep = ep_start
        if first_ep > 0 and first_ep % SELF_PLAY_REFRESH_INTERVAL == 0:
            if is_mixed:
                self_play_fn_2p = make_self_play_agent(model, 2, args.noop_penalty, device)
                self_play_fn_4p = make_self_play_agent(model, 4, args.noop_penalty, device)
            else:
                self_play_fn = make_self_play_agent(
                    model, num_players, args.noop_penalty, device,
                )
        if first_ep > 0 and first_ep % LAGGING_REFRESH_INTERVAL == 0:
            if is_mixed:
                lagging_fn_2p = make_self_play_agent(model, 2, args.noop_penalty, device)
                lagging_fn_4p = make_self_play_agent(model, 4, args.noop_penalty, device)
            else:
                lagging_fn = make_self_play_agent(
                    model, num_players, args.noop_penalty, device,
                )
            print(f"  --- Refreshed lagging reference at ep {first_ep} ---")

        # Determine horizon for this batch
        if use_progressive:
            horizon = HORIZON_STAGES[horizon_stage][1]
        else:
            horizon = args.max_steps

        if use_parallel:
            # --- Parallel rollout collection ---
            # Prepare task descriptions with string-only opponent references
            tasks = []
            for ep_offset in range(args.episodes_per_update):
                ep_num = ep_start + ep_offset
                if ep_num >= args.num_episodes:
                    break
                if is_mixed:
                    frac_4p = min(mixed_4p_frac_end,
                                  mixed_4p_frac_start + (mixed_4p_frac_end - mixed_4p_frac_start)
                                  * ep_num / max(1, mixed_4p_ramp_episodes))
                    ep_mode = "4p" if random.random() < frac_4p else "2p"
                else:
                    ep_mode = args.mode
                opp_name, _, log_opp_name, is_heuristic, opp_names_4p = _select_opponent(ep_num, ep_mode)
                ep_win_bonus = 15.0 if is_heuristic else 10.0
                tasks.append({
                    "opp_name": opp_name,
                    "opp_names_4p": opp_names_4p,  # list of 3 names or None
                    "log_opp_name": log_opp_name,
                    "is_heuristic": is_heuristic,
                    "mode": ep_mode,
                    "noop_penalty": args.noop_penalty,
                    "idle_penalty": args.idle_penalty,
                    "step_penalty": args.step_penalty,
                    "max_steps": horizon,
                    "win_bonus": ep_win_bonus,
                    "ep_num": ep_num,
                })

            # Dispatch to persistent worker pool
            t0 = time.time()
            results = _par_pool.map(_par_play, tasks)

            batch_elapsed = time.time() - t0
            per_ep = batch_elapsed / max(len(results), 1)

            for res in results:
                _process_result(
                    res["opp_name"], res["rollout"], res["final_reward"],
                    res["ep_stats"], res["log_opp_name"], res["is_heuristic"],
                    res["ep_num"], per_ep, horizon,
                )

        else:
            # --- Serial rollout collection (original path) ---
            for ep_offset in range(args.episodes_per_update):
                ep_num = ep_start + ep_offset
                if ep_num >= args.num_episodes:
                    break

                if is_mixed:
                    frac_4p = min(mixed_4p_frac_end,
                                  mixed_4p_frac_start + (mixed_4p_frac_end - mixed_4p_frac_start)
                                  * ep_num / max(1, mixed_4p_ramp_episodes))
                    ep_mode = "4p" if random.random() < frac_4p else "2p"
                else:
                    ep_mode = args.mode

                opp_name, opp_fn, log_opp_name, is_heuristic, _ = _select_opponent(ep_num, ep_mode)
                ep_win_bonus = 15.0 if is_heuristic else 10.0

                t0 = time.time()
                rollout, final_reward, ep_stats = play_episode(
                    model, opp_fn, mode=ep_mode,
                    noop_penalty=args.noop_penalty,
                    idle_penalty=args.idle_penalty,
                    step_penalty=args.step_penalty,
                    max_steps=horizon,
                    win_bonus=ep_win_bonus,
                    device=device,
                )
                elapsed = time.time() - t0

                _process_result(
                    opp_name, rollout, final_reward, ep_stats,
                    log_opp_name, is_heuristic, ep_num, elapsed, horizon,
                )

        if not combined_rollout:
            continue

        # PPO update (optionally on a different device for acceleration)
        if update_device != device:
            model = model.to(update_device)
        model.train()
        stats = ppo_update(
            model, optimizer, combined_rollout,
            max_planets=args.max_planets,
            entropy_coef=args.entropy_coef,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            device=update_device,
            noop_penalty=args.noop_penalty,
        )
        model.eval()
        if update_device != device:
            model = model.to(device)

        # Refresh parallel worker model weights every 5 updates
        if use_parallel and (ep_start // args.episodes_per_update) % 5 == 4:
            _par_pool.terminate()
            _par_pool.join()
            _buf = _io.BytesIO()
            torch.save(model.state_dict(), _buf)
            _sd_bytes = _buf.getvalue()
            _par_pool = _spawn_ctx.Pool(
                processes=args.parallel_envs,
                initializer=_par_init,
                initargs=(_sd_bytes, args.hidden_dim, not args.use_sage,
                          args.mask_sun, str(device), args.noop_penalty),
            )

        # PID learning rate adjustment
        if use_pid:
            current_lr = pid.step(stats['kl'], current_lr)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

        # Stats
        win_rate_window = sum(1 for r in window_results if r > 0) / max(len(window_results), 1)
        heuristic_wr = (
            sum(1 for r in heuristic_results if r > 0) / max(len(heuristic_results), 1)
            if heuristic_results else 0.0
        )
        avg_reward = sum(ep_rewards) / len(ep_rewards)
        baseline_wr = (
            sum(1 for r in baseline_results if r > 0) / max(len(baseline_results), 1)
            if baseline_results else 0.0
        )
        lagging_wr = (
            sum(1 for r in lagging_results if r > 0) / max(len(lagging_results), 1)
            if lagging_results else 0.0
        )

        stage_name = CURRICULUM_STAGES_EFFECTIVE[min(curriculum_stage, len(CURRICULUM_STAGES_EFFECTIVE) - 1)]["name"]
        max_tier_now = CURRICULUM_STAGES_EFFECTIVE[min(curriculum_stage, len(CURRICULUM_STAGES_EFFECTIVE) - 1)]["max_tier"]
        horizon_now = HORIZON_STAGES[horizon_stage][1] if use_progressive else args.max_steps
        lr_str = f"lr={current_lr:.2e} " if use_pid else ""
        print(
            f"  Update: policy={stats['policy_loss']:.4f} "
            f"value={stats['value_loss']:.4f} "
            f"entropy={stats['entropy']:.3f} "
            f"kl={stats['kl']:.4f} "
            f"clip={stats['clip_frac']:.2f} "
            f"| wr(50)={win_rate_window:.1%} "
            f"heur_wr={heuristic_wr:.1%} "
            f"avg_r={avg_reward:+.2f} "
            f"base_wr={baseline_wr:.0%} "
            f"lag_wr={lagging_wr:.0%} "
            f"{lr_str}"
            f"stage={stage_name}(tier≤{max_tier_now}) "
            f"horizon={horizon_now}",
            flush=True,
        )

        # Sanity summary every 1000 episodes
        if total_episodes > 0 and total_episodes % 1000 == 0:
            overall_wr = total_wins / total_episodes
            print(
                f"\n  === Ep {total_episodes} summary ===\n"
                f"    Overall wr: {overall_wr:.1%} ({total_wins}/{total_episodes})\n"
                f"    Heuristic wr (last 100): {heuristic_wr:.1%} ({len(heuristic_results)} games)\n"
                f"    vs Baseline (last 50): {baseline_wr:.0%} ({len(baseline_results)} games)\n"
                f"    vs Lagging (last 50): {lagging_wr:.0%} ({len(lagging_results)} games)\n"
                f"    Curriculum: {stage_name}, Horizon: {horizon_now}\n"
                f"    Entropy: {stats['entropy']:.3f}, Value loss: {stats['value_loss']:.4f}\n",
                flush=True,
            )

        # Curriculum advancement — gate on HEURISTIC win rate, not overall.
        # Floor stage only has "random" which isn't counted as heuristic, so
        # fall back to window_results (overall WR) for that stage only.
        if use_curriculum:
            stage_cfg = CURRICULUM_STAGES_EFFECTIVE[min(curriculum_stage, len(CURRICULUM_STAGES_EFFECTIVE) - 1)]
            min_games = stage_cfg.get("min_games", 50)
            advance_wr = stage_cfg["advance_wr"]
            if stage_cfg["name"] == "floor":
                _adv_results = window_results
            else:
                _adv_results = heuristic_results
            _adv_wr = sum(1 for r in _adv_results if r > 0) / max(1, len(_adv_results))
            if len(_adv_results) >= min_games and _adv_wr >= advance_wr and curriculum_stage < len(CURRICULUM_STAGES_EFFECTIVE) - 1:
                curriculum_stage += 1
                new_stage = CURRICULUM_STAGES_EFFECTIVE[curriculum_stage]
                # Reset PID integral when stage changes to avoid carry-over drift
                pid.reset()
                print(f"  >>> Curriculum advanced to stage {curriculum_stage}: "
                      f"{new_stage['name']} "
                      f"(max_tier={new_stage['max_tier']} "
                      f"self_play={new_stage['self_play']:.0%} "
                      f"heuristic={new_stage['heuristic']:.0%})")

        # Horizon advancement — performance-gated, not time-based.
        # ONLY uses heuristic WR. In floor stage (no heuristics), horizon
        # stays at 100 until curriculum advances to beginner.
        if use_progressive and horizon_stage < len(HORIZON_STAGES) - 1:
            next_wr_threshold = HORIZON_STAGES[horizon_stage + 1][0]
            horizon_min_games = 30
            if len(heuristic_results) >= horizon_min_games and heuristic_wr >= next_wr_threshold:
                horizon_stage += 1
                new_horizon = HORIZON_STAGES[horizon_stage][1]
                print(f"  >>> Horizon advanced to {new_horizon} steps "
                      f"(heur_wr={heuristic_wr:.1%} >= {next_wr_threshold:.0%})")
                heuristic_results.clear()

        # Save best checkpoint based on heuristic win rate
        if heuristic_wr > best_heuristic_wr and len(heuristic_results) >= 30:
            best_heuristic_wr = heuristic_wr
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best checkpoint (heuristic_wr={heuristic_wr:.1%})")

        # Periodic save — every 100 episodes (use total_episodes for reliability)
        if total_episodes % 100 == 0:
            latest_path = str(cache_dir / f"checkpoint_ppo_{args.mode}_latest.pt")
            torch.save(model.state_dict(), latest_path)
            print(f"  Saved periodic checkpoint (ep {total_episodes})")

        # Periodic eval: every eval_every episodes, play a mini-match vs each heuristic
        # Uses the CURRENT horizon so results are comparable to training games
        if total_episodes % args.eval_every == 0 and heuristics:
            eval_horizon = HORIZON_STAGES[horizon_stage][1] if use_progressive else args.max_steps
            print(f"\n  --- Eval checkpoint (ep {total_episodes}, horizon={eval_horizon}) ---")
            eval_wins = 0
            eval_total = 0
            for tier, h_name, h_fn in heuristics:
                h_wins = 0
                for _ in range(4):  # 4 games per heuristic for better signal
                    _, r, es = play_episode(
                        model, h_fn, mode=args.mode,
                        noop_penalty=args.noop_penalty,
                        idle_penalty=args.idle_penalty,
                        max_steps=eval_horizon,
                        device=device,
                    )
                    eval_total += 1
                    if r > 0:
                        eval_wins += 1
                        h_wins += 1
                print(f"    [tier {tier}] vs {h_name}: {h_wins}/4")
            eval_wr = eval_wins / max(eval_total, 1)
            print(f"  Eval total: {eval_wins}/{eval_total} ({eval_wr:.1%})")
            print()

    # Final save
    latest_path = str(cache_dir / f"checkpoint_ppo_{args.mode}_latest.pt")
    torch.save(model.state_dict(), latest_path)
    overall_wr = total_wins / max(total_episodes, 1)
    print(f"\nPPO training complete. {total_episodes} episodes, "
          f"overall win rate: {overall_wr:.1%}")
    print(f"Best heuristic win rate: {best_heuristic_wr:.1%}")
    if use_parallel and _par_pool is not None:
        _par_pool.terminate()
        _par_pool.join()

    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Latest checkpoint: {latest_path}")


if __name__ == "__main__":
    main()
