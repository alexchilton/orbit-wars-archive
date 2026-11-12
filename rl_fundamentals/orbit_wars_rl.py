from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .fundamental_env import FRACTION_BUCKETS, SimplifiedPlanetEnv
except ImportError:
    from fundamental_env import FRACTION_BUCKETS, SimplifiedPlanetEnv


NODE_FEATURE_DIM = 12
GLOBAL_FEATURE_DIM = 6
HIDDEN_DIM = 64


class ActionEncoder:
    @staticmethod
    def total_actions(num_planets: int) -> int:
        return num_planets * num_planets * len(FRACTION_BUCKETS) + 1

    @staticmethod
    def noop_index(num_planets: int) -> int:
        return ActionEncoder.total_actions(num_planets) - 1

    @staticmethod
    def encode(source: int, target: int, fraction_idx: int, num_planets: int) -> int:
        return source * (num_planets * len(FRACTION_BUCKETS)) + target * len(FRACTION_BUCKETS) + fraction_idx

    @staticmethod
    def decode(action_idx: int, num_planets: int) -> tuple[int | None, int | None, float]:
        if action_idx == ActionEncoder.noop_index(num_planets):
            return None, None, 0.0
        source = action_idx // (num_planets * len(FRACTION_BUCKETS))
        rem = action_idx % (num_planets * len(FRACTION_BUCKETS))
        target = rem // len(FRACTION_BUCKETS)
        fraction_idx = rem % len(FRACTION_BUCKETS)
        return source, target, FRACTION_BUCKETS[fraction_idx]


@dataclass(frozen=True)
class RuntimeConfig:
    deployment_mode: str = "hybrid"
    fallback_threshold: float = 0.05
    opening_search_steps: int = 0
    beam_width: int = 5
    beam_depth: int = 2
    beam_candidates: int = 10


def _to_tensor_state(env: SimplifiedPlanetEnv, player: int) -> tuple[torch.Tensor, ...]:
    obs = env.build_gcn_observation(player)
    num_planets = env.max_planets
    action_mask = torch.zeros(ActionEncoder.total_actions(num_planets), dtype=torch.bool)
    mask_cube = obs["action_mask"]
    for source in range(num_planets):
        for target in range(num_planets):
            for fraction_index, allowed in enumerate(mask_cube[source][target]):
                if allowed:
                    action_mask[ActionEncoder.encode(source, target, fraction_index, num_planets)] = True
    action_mask[ActionEncoder.noop_index(num_planets)] = True
    return (
        torch.tensor(obs["node_features"], dtype=torch.float32),
        torch.tensor(obs["global_features"], dtype=torch.float32),
        torch.tensor(obs["positions"], dtype=torch.float32),
        torch.tensor(obs["velocities"], dtype=torch.float32),
        torch.tensor(obs["valid_mask"], dtype=torch.float32),
        action_mask,
    )


def _heuristic_action_index(env: SimplifiedPlanetEnv, player: int) -> int:
    source, target, fraction_index = env.heuristic_fraction_action(player)
    if source is None or target is None or fraction_index is None:
        return ActionEncoder.noop_index(env.max_planets)
    return ActionEncoder.encode(source, target, fraction_index, env.max_planets)


def _action_to_fraction_tuple(action_idx: int, num_planets: int) -> tuple[int | None, int | None, float]:
    source, target, fraction = ActionEncoder.decode(action_idx, num_planets)
    return source, target, fraction


def _model_forward(model: OrbitWarsGCN, state: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    node_features, global_features, positions, velocities, valid_mask, action_mask = state
    with torch.no_grad():
        logits, value = model(
            node_features.unsqueeze(0),
            global_features.unsqueeze(0),
            positions.unsqueeze(0),
            velocities.unsqueeze(0),
            valid_mask.unsqueeze(0),
            action_mask.unsqueeze(0),
        )
    return logits.squeeze(0), value.squeeze(0)


def _select_action_from_model(
    model: OrbitWarsGCN,
    state: tuple[torch.Tensor, ...],
    deterministic: bool = False,
) -> tuple[int, float, float, float, torch.Tensor]:
    logits, value = _model_forward(model, state)
    distribution = torch.distributions.Categorical(logits=logits)
    action = torch.argmax(logits) if deterministic else distribution.sample()
    probs = torch.softmax(logits, dim=0)
    return (
        int(action.item()),
        float(distribution.log_prob(action).item()),
        float(value.item()),
        float(probs[int(action.item())].item()),
        logits,
    )


def _log_prob_for_action(logits: torch.Tensor, action_idx: int) -> float:
    distribution = torch.distributions.Categorical(logits=logits)
    return float(distribution.log_prob(torch.tensor(action_idx, dtype=torch.int64)).item())


class OrbitWarsGCN(nn.Module):
    def __init__(self, node_dim: int = NODE_FEATURE_DIM, global_dim: int = GLOBAL_FEATURE_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.edge_dim = 3
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gcn1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)
        self.edge_net = nn.Sequential(
            nn.Linear(self.edge_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 2 + self.edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(FRACTION_BUCKETS)),
        )
        self.noop_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        final_noop = self.noop_head[-1]
        if isinstance(final_noop, nn.Linear):
            nn.init.constant_(final_noop.bias, -2.0)

    def compute_edge_features(self, positions: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        diff = pos_j - pos_i
        dist = torch.norm(diff, dim=-1, keepdim=True)
        vel_i = velocities.unsqueeze(2)
        vel_j = velocities.unsqueeze(1)
        rel_vel = torch.norm(vel_j - vel_i, dim=-1, keepdim=True)
        time_to_travel = dist / 2.0
        return torch.cat([dist, rel_vel, time_to_travel], dim=-1)

    def forward(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        valid_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_planets, _ = node_features.shape
        h = self.node_encoder(node_features)
        edge_feats = self.compute_edge_features(positions, velocities)
        edge_weights = torch.sigmoid(self.edge_net(edge_feats))

        messages = (h.unsqueeze(1) * edge_weights).sum(dim=2)
        h = F.relu(self.gcn1(h + messages))
        messages = (h.unsqueeze(1) * edge_weights).sum(dim=2)
        h = F.relu(self.gcn2(h + messages))

        g = self.global_encoder(global_features)
        h_i = h.unsqueeze(2).expand(-1, -1, num_planets, -1)
        h_j = h.unsqueeze(1).expand(-1, num_planets, -1, -1)
        g_exp = g.unsqueeze(1).unsqueeze(1).expand(-1, num_planets, num_planets, -1)
        pair_features = torch.cat([h_i, h_j, g_exp, edge_feats], dim=-1)
        pair_logits = self.action_head(pair_features).reshape(batch_size, -1)

        pooled = (h * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        global_context = torch.cat([pooled, g], dim=-1)
        noop_logits = self.noop_head(global_context)
        logits = torch.cat([pair_logits, noop_logits], dim=1)
        logits = logits.masked_fill(~action_mask.bool(), -1e9)
        value = self.value_head(global_context).squeeze(-1)
        return logits, value


@dataclass
class Transition:
    state: tuple[torch.Tensor, ...]
    action: int
    reward: float
    done: float
    log_prob: float
    value: float


class PPOAgent:
    def __init__(
        self,
        model: OrbitWarsGCN,
        lr: float = 3e-4,
        gamma: float = 0.97,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.02,
    ) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def behavior_clone(self, samples: Sequence[tuple[tuple[torch.Tensor, ...], int]], epochs: int = 6, batch_size: int = 64) -> float:
        if not samples:
            return 0.0
        loss_value = 0.0
        for _ in range(epochs):
            indices = list(range(len(samples)))
            random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                batch_states = [samples[index][0] for index in batch_indices]
                batch_actions = torch.tensor([samples[index][1] for index in batch_indices], dtype=torch.int64)
                node_features = torch.stack([state[0] for state in batch_states])
                global_features = torch.stack([state[1] for state in batch_states])
                positions = torch.stack([state[2] for state in batch_states])
                velocities = torch.stack([state[3] for state in batch_states])
                valid_mask = torch.stack([state[4] for state in batch_states])
                action_mask = torch.stack([state[5] for state in batch_states])
                logits, _values = self.model(node_features, global_features, positions, velocities, valid_mask, action_mask)
                loss = F.cross_entropy(logits, batch_actions)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                loss_value = float(loss.item())
        return loss_value

    def select_action(self, state: tuple[torch.Tensor, ...], deterministic: bool = False) -> tuple[int, float, float]:
        action, log_prob, value, _confidence, _logits = _select_action_from_model(self.model, state, deterministic=deterministic)
        return action, log_prob, value

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for index in reversed(range(len(rewards))):
            next_value = 0.0 if index == len(rewards) - 1 else values[index + 1]
            delta = rewards[index] + self.gamma * next_value * (1 - dones[index]) - values[index]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[index]) * gae
            advantages[index] = gae
        return advantages

    def update(self, transitions: Sequence[Transition], epochs: int = 4, batch_size: int = 64) -> float:
        rewards = torch.tensor([item.reward for item in transitions], dtype=torch.float32)
        dones = torch.tensor([item.done for item in transitions], dtype=torch.float32)
        old_log_probs = torch.tensor([item.log_prob for item in transitions], dtype=torch.float32)
        old_values = torch.tensor([item.value for item in transitions], dtype=torch.float32)
        actions = torch.tensor([item.action for item in transitions], dtype=torch.int64)

        advantages = self.compute_gae(rewards, old_values, dones)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        loss_value = 0.0
        indices = list(range(len(transitions)))
        for _ in range(epochs):
            random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                batch_states = [transitions[index].state for index in batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                node_features = torch.stack([state[0] for state in batch_states])
                global_features = torch.stack([state[1] for state in batch_states])
                positions = torch.stack([state[2] for state in batch_states])
                velocities = torch.stack([state[3] for state in batch_states])
                valid_mask = torch.stack([state[4] for state in batch_states])
                action_mask = torch.stack([state[5] for state in batch_states])

                logits, values = self.model(node_features, global_features, positions, velocities, valid_mask, action_mask)
                distribution = torch.distributions.Categorical(logits=logits)
                new_log_probs = distribution.log_prob(batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns)
                entropy = distribution.entropy().mean()
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                loss_value = float(loss.item())
        return loss_value


def compute_shaped_reward(env: SimplifiedPlanetEnv, before: Dict[str, float | int | bool], after: Dict[str, float | int | bool], done: bool, win: bool) -> float:
    reward = 0.0
    reward += 0.12 * (float(after["my_production"]) - float(before["my_production"]))
    reward += 0.03 * (float(after["my_total"]) - float(before["my_total"]))
    reward += 0.02 * (float(after["prod_gap"]) - float(before["prod_gap"]))
    reward -= 0.015 * max(0.0, float(after["best_enemy_total"]) - float(before["best_enemy_total"]))
    reward -= 0.002
    if env.mode == "four_player":
        reward += 0.04 * (float(before["my_rank"]) - float(after["my_rank"]))
        if bool(after["leader_runaway"]) and int(after["leader_owner"]) != 0:
            reward -= 0.05
    if done:
        reward += 8.0 if win else -8.0
    return reward


def _scored_transition(
    env: SimplifiedPlanetEnv,
    player: int,
    action_idx: int,
    result,
    before: Dict[str, float | int | bool],
    after: Dict[str, float | int | bool],
) -> float:
    source, _target, _fraction = ActionEncoder.decode(action_idx, env.max_planets)
    shaped_reward = result.rewards[player] + compute_shaped_reward(env, before, after, result.done, result.winner == player)
    if source is None and float(before["idle_surplus"]) > 8.0:
        shaped_reward -= 0.12 + 0.008 * float(before["idle_surplus"])
    return shaped_reward


def _candidate_action_indices(env: SimplifiedPlanetEnv, player: int, limit: int) -> List[int]:
    action_mask = _to_tensor_state(env, player)[5]
    valid_actions = [int(index) for index in torch.nonzero(action_mask, as_tuple=False).flatten().tolist()]
    heuristic_idx = _heuristic_action_index(env, player)
    if len(valid_actions) <= limit:
        return valid_actions

    scored: List[tuple[float, int]] = []
    for action_idx in valid_actions:
        if action_idx == ActionEncoder.noop_index(env.max_planets):
            continue
        scored.append((_immediate_action_score(env, player, action_idx), action_idx))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [action for _score, action in scored[: max(1, limit - 2)]]
    for extra in (heuristic_idx, ActionEncoder.noop_index(env.max_planets)):
        if extra not in selected:
            selected.append(extra)
    return selected[:limit]


def _immediate_action_score(env: SimplifiedPlanetEnv, player: int, action_idx: int) -> float:
    sim_env = copy.deepcopy(env)
    before = sim_env.summary(player)
    result = sim_env.step_fraction_actions(
        {player: _action_to_fraction_tuple(action_idx, sim_env.max_planets)},
        default_opponent_mode="heuristic",
    )
    after = sim_env.summary(player)
    return _scored_transition(sim_env, player, action_idx, result, before, after)


def _search_opening_action_index(env: SimplifiedPlanetEnv, player: int, runtime: RuntimeConfig) -> int:
    if runtime.opening_search_steps <= 0 or env.step_count >= runtime.opening_search_steps:
        return _heuristic_action_index(env, player)

    frontier: List[tuple[float, SimplifiedPlanetEnv, int | None]] = [(0.0, copy.deepcopy(env), None)]
    discount = 0.92
    for depth in range(runtime.beam_depth):
        expanded: List[tuple[float, SimplifiedPlanetEnv, int]] = []
        for base_score, sim_env, root_action in frontier:
            candidates = _candidate_action_indices(sim_env, player, runtime.beam_candidates)
            for action_idx in candidates:
                next_env = copy.deepcopy(sim_env)
                before = next_env.summary(player)
                result = next_env.step_fraction_actions(
                    {player: _action_to_fraction_tuple(action_idx, next_env.max_planets)},
                    default_opponent_mode="heuristic",
                )
                after = next_env.summary(player)
                shaped_reward = _scored_transition(next_env, player, action_idx, result, before, after)
                first_action = action_idx if root_action is None else root_action
                expanded.append((base_score + (discount**depth) * shaped_reward, next_env, first_action))
        if not expanded:
            break
        expanded.sort(key=lambda item: item[0], reverse=True)
        frontier = expanded[: runtime.beam_width]

    if frontier:
        return max(frontier, key=lambda item: item[0])[2]
    return _heuristic_action_index(env, player)


def _choose_policy_action(
    env: SimplifiedPlanetEnv,
    player: int,
    model: OrbitWarsGCN,
    runtime: RuntimeConfig,
    deterministic: bool,
    allow_search: bool,
) -> tuple[int, float, float, float, str]:
    state = _to_tensor_state(env, player)
    action_idx, log_prob, value, confidence, logits = _select_action_from_model(model, state, deterministic=deterministic)
    heuristic_idx = _heuristic_action_index(env, player)

    if allow_search and env.step_count < runtime.opening_search_steps:
        searched = _search_opening_action_index(env, player, runtime)
        return searched, _log_prob_for_action(logits, searched), value, confidence, "search"

    if runtime.deployment_mode == "heuristic":
        return heuristic_idx, _log_prob_for_action(logits, heuristic_idx), value, confidence, "heuristic"

    if runtime.deployment_mode == "hybrid":
        idle_surplus = float(env.summary(player)["idle_surplus"])
        if action_idx == ActionEncoder.noop_index(env.max_planets) and idle_surplus > 8.0:
            return heuristic_idx, _log_prob_for_action(logits, heuristic_idx), value, confidence, "fallback"
        if action_idx != heuristic_idx:
            chosen_score = _immediate_action_score(env, player, action_idx)
            heuristic_score = _immediate_action_score(env, player, heuristic_idx)
            if chosen_score <= heuristic_score + runtime.fallback_threshold:
                return heuristic_idx, _log_prob_for_action(logits, heuristic_idx), value, confidence, "fallback"

    return action_idx, log_prob, value, confidence, "policy"


def rollout_episode(
    env: SimplifiedPlanetEnv,
    agent: PPOAgent,
    rng: random.Random,
    deterministic: bool = False,
    expert_mix: float = 0.0,
    runtime: RuntimeConfig | None = None,
    opponent_policy: str = "heuristic",
    opponent_model: OrbitWarsGCN | None = None,
    opening_search: bool = False,
) -> tuple[List[Transition], Dict[str, float]]:
    runtime = runtime or RuntimeConfig()
    env.reset(seed=rng.randint(0, 10_000_000))
    transitions: List[Transition] = []
    total_reward = 0.0
    noop_count = 0
    source_counts = {"policy": 0, "fallback": 0, "heuristic": 0, "search": 0, "expert": 0}
    while True:
        before = env.summary(0)
        state = _to_tensor_state(env, 0)
        action_idx, log_prob, value, _confidence, action_source = _choose_policy_action(
            env,
            0,
            agent.model,
            runtime,
            deterministic=deterministic,
            allow_search=opening_search,
        )
        if not deterministic and rng.random() < expert_mix:
            action_idx = _heuristic_action_index(env, 0)
            logits, _ = _model_forward(agent.model, state)
            log_prob = _log_prob_for_action(logits, action_idx)
            action_source = "expert"
        source_counts[action_source] = source_counts.get(action_source, 0) + 1
        source, target, fraction = ActionEncoder.decode(action_idx, env.max_planets)

        chosen_actions = {0: (source, target, fraction)}
        if source is None or target is None:
            noop_count += 1

        if opponent_policy != "heuristic" and opponent_model is not None:
            for owner in range(1, env.num_players):
                if not env._player_alive(owner):
                    continue
                if opponent_policy == "mixed" and rng.random() < 0.5:
                    continue
                opponent_action, _opp_log_prob, _opp_value, _opp_confidence, _opp_source = _choose_policy_action(
                    env,
                    owner,
                    opponent_model,
                    runtime,
                    deterministic=False,
                    allow_search=False,
                )
                chosen_actions[owner] = _action_to_fraction_tuple(opponent_action, env.max_planets)

        result = env.step_fraction_actions(chosen_actions, default_opponent_mode="heuristic")
        after = env.summary(0)
        shaped_reward = _scored_transition(env, 0, action_idx, result, before, after)
        total_reward += shaped_reward
        transitions.append(
            Transition(
                state=state,
                action=action_idx,
                reward=shaped_reward,
                done=1.0 if result.done else 0.0,
                log_prob=log_prob,
                value=value,
            )
        )
        if result.done:
            return transitions, {
                "reward": total_reward,
                "steps": float(len(transitions)),
                "winner": float(result.winner if result.winner is not None else -1),
                "won": 1.0 if result.winner == 0 else 0.0,
                "noop_rate": noop_count / max(1, len(transitions)),
                "policy_rate": source_counts["policy"] / max(1, len(transitions)),
                "fallback_rate": source_counts["fallback"] / max(1, len(transitions)),
                "search_rate": source_counts["search"] / max(1, len(transitions)),
                "expert_rate": source_counts["expert"] / max(1, len(transitions)),
            }


def collect_behavior_samples(
    mode: str,
    episodes: int,
    seed: int,
    runtime: RuntimeConfig,
    label_mode: str = "mixed",
) -> List[tuple[tuple[torch.Tensor, ...], int]]:
    env = SimplifiedPlanetEnv(mode, seed=seed)
    rng = random.Random(seed)
    samples: List[tuple[tuple[torch.Tensor, ...], int]] = []
    noop_stride = 4
    noop_seen = 0
    for _ in range(episodes):
        env.reset(seed=rng.randint(0, 10_000_000))
        while True:
            state = _to_tensor_state(env, 0)
            if label_mode == "beam" or (label_mode == "mixed" and env.step_count < runtime.opening_search_steps):
                action_index = _search_opening_action_index(env, 0, runtime)
            else:
                action_index = _heuristic_action_index(env, 0)
            if action_index == ActionEncoder.noop_index(env.max_planets):
                if noop_seen % noop_stride == 0:
                    samples.append((state, action_index))
                noop_seen += 1
            else:
                samples.append((state, action_index))
            result = env.step_fraction_actions(
                {0: _action_to_fraction_tuple(action_index, env.max_planets)},
                default_opponent_mode="heuristic",
            )
            if result.done:
                break
    return samples


def evaluate_mode(
    mode: str,
    checkpoint: str | Path,
    episodes: int,
    seed: int,
    runtime: RuntimeConfig | None = None,
    opponent_policy: str = "heuristic",
) -> Dict[str, object]:
    runtime = runtime or RuntimeConfig()
    env = SimplifiedPlanetEnv(mode, seed=seed)
    model = OrbitWarsGCN()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    agent = PPOAgent(model)
    rng = random.Random(seed)
    rewards: List[float] = []
    wins = 0
    noop_rates: List[float] = []
    fallback_rates: List[float] = []
    search_rates: List[float] = []
    policy_rates: List[float] = []
    for _ in range(episodes):
        _transitions, outcome = rollout_episode(
            env,
            agent,
            rng,
            deterministic=True,
            runtime=runtime,
            opponent_policy=opponent_policy,
            opponent_model=model if opponent_policy != "heuristic" else None,
            opening_search=runtime.opening_search_steps > 0,
        )
        rewards.append(outcome["reward"])
        wins += int(outcome["won"])
        noop_rates.append(outcome["noop_rate"])
        fallback_rates.append(outcome["fallback_rate"])
        search_rates.append(outcome["search_rate"])
        policy_rates.append(outcome["policy_rate"])
    return {
        "episodes": float(episodes),
        "win_rate": wins / max(1, episodes),
        "avg_reward": sum(rewards) / max(1, len(rewards)),
        "avg_noop_rate": sum(noop_rates) / max(1, len(noop_rates)),
        "avg_fallback_rate": sum(fallback_rates) / max(1, len(fallback_rates)),
        "avg_search_rate": sum(search_rates) / max(1, len(search_rates)),
        "avg_policy_rate": sum(policy_rates) / max(1, len(policy_rates)),
        "deployment_mode": runtime.deployment_mode,
        "opening_search_steps": float(runtime.opening_search_steps),
    }


def _clone_model(model: OrbitWarsGCN) -> OrbitWarsGCN:
    cloned = OrbitWarsGCN()
    cloned.load_state_dict(copy.deepcopy(model.state_dict()))
    return cloned


def train_mode(
    mode: str,
    episodes: int,
    eval_episodes: int,
    seed: int,
    output_dir: Path,
    runtime: RuntimeConfig | None = None,
    self_play_start: float = 0.8,
) -> Dict[str, object]:
    runtime = runtime or RuntimeConfig()
    env = SimplifiedPlanetEnv(mode, seed=seed)
    model = OrbitWarsGCN()
    agent = PPOAgent(model)
    rng = random.Random(seed)

    behavior_samples = collect_behavior_samples(
        mode,
        episodes=max(24, episodes // 4),
        seed=seed + 17,
        runtime=runtime,
        label_mode="mixed",
    )
    bc_loss = agent.behavior_clone(behavior_samples)
    print(f"[{mode}] warm_start_samples={len(behavior_samples)} bc_loss={bc_loss:.3f}", flush=True)

    reward_window: List[float] = []
    won_window: List[float] = []
    progress_rows: List[Dict[str, object]] = []
    every = max(1, episodes // 8)
    snapshot_interval = max(12, episodes // 6)
    opponent_pool: List[OrbitWarsGCN] = [_clone_model(model)]
    for episode in range(1, episodes + 1):
        progress = (episode - 1) / max(1, episodes - 1)
        expert_mix = 0.35 * (1.0 - progress) + 0.05 * progress
        if progress < self_play_start or len(opponent_pool) == 0:
            opponent_policy = "heuristic"
            opponent_model = None
        else:
            opponent_policy = "mixed"
            opponent_model = rng.choice(opponent_pool)

        transitions, outcome = rollout_episode(
            env,
            agent,
            rng,
            deterministic=False,
            expert_mix=expert_mix,
            runtime=runtime,
            opponent_policy=opponent_policy,
            opponent_model=opponent_model,
            opening_search=progress < 0.35,
        )
        loss = agent.update(transitions)
        reward_window.append(outcome["reward"])
        won_window.append(outcome["won"])
        if len(reward_window) > 100:
            reward_window.pop(0)
            won_window.pop(0)
        if episode % snapshot_interval == 0 or episode == 1:
            opponent_pool.append(_clone_model(agent.model))
            opponent_pool = opponent_pool[-3:]
        if episode % every == 0 or episode == episodes:
            row = {
                "episode": float(episode),
                "avg_reward_100": sum(reward_window) / max(1, len(reward_window)),
                "loss": loss,
                "expert_mix": expert_mix,
                "opponent_policy": opponent_policy,
                "win_rate_100": sum(won_window) / max(1, len(won_window)),
                "noop_rate": outcome["noop_rate"],
                "fallback_rate": outcome["fallback_rate"],
                "search_rate": outcome["search_rate"],
            }
            progress_rows.append(row)
            print(
                f"[{mode}] episode={episode}/{episodes} avg_reward_100={row['avg_reward_100']:.3f} "
                f"loss={loss:.3f} noop={row['noop_rate']:.3f}",
                flush=True,
            )

    checkpoint = output_dir / f"{mode}_gcn_policy.pt"
    torch.save(model.state_dict(), checkpoint)
    evaluation = evaluate_mode(mode, checkpoint, eval_episodes, seed + 1000, runtime=runtime)
    payload: Dict[str, object] = {
        "mode": mode,
        "episodes": episodes,
        "eval_episodes": eval_episodes,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "behavior_clone_loss": bc_loss,
        "behavior_clone_samples": len(behavior_samples),
        "runtime": {
            "deployment_mode": runtime.deployment_mode,
            "fallback_threshold": runtime.fallback_threshold,
            "opening_search_steps": runtime.opening_search_steps,
            "beam_width": runtime.beam_width,
            "beam_depth": runtime.beam_depth,
            "beam_candidates": runtime.beam_candidates,
        },
        "evaluation": evaluation,
        "progress": progress_rows,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output_dir / f"{mode}_gcn_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight GCN/PPO policy on the simplified Orbit Wars fundamentals env")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--env-mode", choices=("two_player", "four_player", "both"), default="both")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--eval-episodes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="rl_fundamentals/results/gcn_run")
    parser.add_argument("--checkpoint")
    parser.add_argument("--deployment-mode", choices=("pure", "hybrid", "heuristic"), default="hybrid")
    parser.add_argument("--fallback-threshold", type=float, default=0.05)
    parser.add_argument("--opening-search-steps", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--beam-depth", type=int, default=2)
    parser.add_argument("--beam-candidates", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = ["two_player", "four_player"] if args.env_mode == "both" else [args.env_mode]
    runtime = RuntimeConfig(
        deployment_mode=args.deployment_mode,
        fallback_threshold=args.fallback_threshold,
        opening_search_steps=args.opening_search_steps,
        beam_width=args.beam_width,
        beam_depth=args.beam_depth,
        beam_candidates=args.beam_candidates,
    )

    if args.mode == "train":
        combined = {"modes": {}}
        for offset, mode in enumerate(modes):
            combined["modes"][mode] = train_mode(
                mode,
                args.episodes,
                args.eval_episodes,
                args.seed + offset * 100,
                output_dir,
                runtime=runtime,
            )
        (output_dir / "gcn_training_summary.json").write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(combined, indent=2, sort_keys=True))
        return

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required in eval mode")
    if len(modes) != 1:
        raise SystemExit("eval mode expects a single --env-mode")
    metrics = evaluate_mode(modes[0], args.checkpoint, args.eval_episodes, args.seed, runtime=runtime)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
