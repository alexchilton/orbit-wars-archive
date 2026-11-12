from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

try:
    from .fundamental_env import SimplifiedPlanetEnv
except ImportError:
    from fundamental_env import SimplifiedPlanetEnv


def _state_key(state: Sequence[int]) -> str:
    return "|".join(str(value) for value in state)


@dataclass
class TabularQPolicy:
    actions: List[str]
    q_values: Dict[str, Dict[str, float]] = field(default_factory=dict)
    visit_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def ensure_state(self, state: Sequence[int]) -> Dict[str, float]:
        key = _state_key(state)
        if key not in self.q_values:
            self.q_values[key] = {action: 0.0 for action in self.actions}
            self.visit_counts[key] = {action: 0 for action in self.actions}
        return self.q_values[key]

    def choose(self, state: Sequence[int], rng: random.Random, epsilon: float) -> str:
        action_values = self.ensure_state(state)
        if rng.random() < epsilon:
            return rng.choice(self.actions)
        return max(self.actions, key=lambda action: (action_values[action], -self.actions.index(action)))

    def update(
        self,
        state: Sequence[int],
        action: str,
        reward: float,
        next_state: Sequence[int],
        done: bool,
        learning_rate: float,
        gamma: float,
    ) -> None:
        values = self.ensure_state(state)
        next_values = self.ensure_state(next_state)
        target = reward if done else reward + gamma * max(next_values.values())
        values[action] += learning_rate * (target - values[action])
        self.visit_counts[_state_key(state)][action] += 1

    def greedy_action(self, state: Sequence[int]) -> str:
        values = self.ensure_state(state)
        return max(self.actions, key=lambda action: (values[action], -self.actions.index(action)))

    def to_dict(self) -> Dict[str, object]:
        return {
            "actions": list(self.actions),
            "q_values": self.q_values,
            "visit_counts": self.visit_counts,
        }

    def save_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return output


def rollout_episode(
    env: SimplifiedPlanetEnv,
    policy: TabularQPolicy,
    rng: random.Random,
    epsilon: float,
    learning_rate: float,
    gamma: float,
    train: bool,
) -> Dict[str, float]:
    env.reset(seed=rng.randint(0, 10_000_000))
    total_reward = 0.0
    steps = 0
    while True:
        state = env.encode_state(0)
        action = policy.choose(state, rng, epsilon if train else 0.0)
        actions_by_player = {0: action}
        for player in range(1, env.num_players):
            if env._player_alive(player):  # internal but intentional for trainer loop
                actions_by_player[player] = env.heuristic_action(player)
        result = env.step(actions_by_player)
        next_state = env.encode_state(0)
        reward = result.rewards[0]
        total_reward += reward
        if train:
            policy.update(state, action, reward, next_state, result.done, learning_rate, gamma)
        steps += 1
        if result.done:
            return {
                "reward": total_reward,
                "steps": float(steps),
                "winner": float(result.winner if result.winner is not None else -1),
                "won": 1.0 if result.winner == 0 else 0.0,
            }


def evaluate_policy(mode: str, policy: TabularQPolicy, episodes: int, seed: int) -> Dict[str, float]:
    env = SimplifiedPlanetEnv(mode, seed=seed)
    rng = random.Random(seed)
    rewards = []
    wins = 0
    for _ in range(episodes):
        outcome = rollout_episode(env, policy, rng, epsilon=0.0, learning_rate=0.0, gamma=0.0, train=False)
        rewards.append(outcome["reward"])
        wins += int(outcome["won"])
    return {
        "episodes": float(episodes),
        "win_rate": wins / max(1, episodes),
        "avg_reward": sum(rewards) / max(1, len(rewards)),
    }


def evaluate_scripted_baseline(mode: str, episodes: int, seed: int) -> Dict[str, float]:
    env = SimplifiedPlanetEnv(mode, seed=seed)
    rng = random.Random(seed)
    policy = TabularQPolicy(actions=env.action_names())
    rewards = []
    wins = 0
    for _ in range(episodes):
        env.reset(seed=rng.randint(0, 10_000_000))
        total_reward = 0.0
        while True:
            actions_by_player = {}
            for player in range(env.num_players):
                if env._player_alive(player):
                    actions_by_player[player] = env.heuristic_action(player)
            result = env.step(actions_by_player)
            total_reward += result.rewards[0]
            if result.done:
                rewards.append(total_reward)
                wins += int(result.winner == 0)
                break
    del policy
    return {
        "episodes": float(episodes),
        "win_rate": wins / max(1, episodes),
        "avg_reward": sum(rewards) / max(1, len(rewards)),
    }


def summarize_policy(policy: TabularQPolicy) -> List[Dict[str, object]]:
    visit_totals = Counter()
    for state_key, counts in policy.visit_counts.items():
        visit_totals[state_key] = sum(counts.values())
    ranked_states = visit_totals.most_common(12)
    summary: List[Dict[str, object]] = []
    for state_key, total in ranked_states:
        values = policy.q_values[state_key]
        best_action = max(policy.actions, key=lambda action: (values[action], -policy.actions.index(action)))
        summary.append(
            {
                "state": state_key,
                "visits": total,
                "best_action": best_action,
                "q_values": {action: round(values[action], 4) for action in policy.actions},
            }
        )
    return summary


def train_mode(
    mode: str,
    episodes: int,
    eval_episodes: int,
    seed: int,
    learning_rate: float = 0.18,
    gamma: float = 0.92,
    epsilon_start: float = 0.24,
    epsilon_end: float = 0.03,
) -> Dict[str, object]:
    env = SimplifiedPlanetEnv(mode, seed=seed)
    policy = TabularQPolicy(actions=env.action_names())
    rng = random.Random(seed)

    reward_window: List[float] = []
    progress_rows: List[Dict[str, float]] = []
    every = max(1, episodes // 8)
    for episode in range(1, episodes + 1):
        progress = (episode - 1) / max(1, episodes - 1)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
        outcome = rollout_episode(env, policy, rng, epsilon, learning_rate, gamma, train=True)
        reward_window.append(outcome["reward"])
        if len(reward_window) > 100:
            reward_window.pop(0)
        if episode % every == 0 or episode == episodes:
            row = {
                "episode": float(episode),
                "epsilon": epsilon,
                "avg_reward_100": sum(reward_window) / max(1, len(reward_window)),
                "states_seen": float(len(policy.q_values)),
            }
            progress_rows.append(row)
            print(
                f"[{mode}] episode={episode}/{episodes} epsilon={epsilon:.3f} "
                f"avg_reward_100={row['avg_reward_100']:.3f} states={int(row['states_seen'])}",
                flush=True,
            )

    evaluation = evaluate_policy(mode, policy, eval_episodes, seed + 1000)
    baseline = evaluate_scripted_baseline(mode, eval_episodes, seed + 2000)
    return {
        "mode": mode,
        "episodes": episodes,
        "eval_episodes": eval_episodes,
        "seed": seed,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "progress": progress_rows,
        "evaluation": evaluation,
        "scripted_baseline": baseline,
        "policy_summary": summarize_policy(policy),
        "policy": policy,
    }


def _write_training_artifacts(result: Dict[str, object], output_dir: Path) -> None:
    policy = result["policy"]
    assert isinstance(policy, TabularQPolicy)
    mode = str(result["mode"])
    policy.save_json(output_dir / f"{mode}_policy.json")

    payload = dict(result)
    del payload["policy"]
    (output_dir / f"{mode}_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fast abstract Q-policy over simplified planet-control fundamentals")
    parser.add_argument("--mode", choices=("two_player", "four_player", "both"), default="both")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.18)
    parser.add_argument("--gamma", type=float, default=0.92)
    parser.add_argument("--epsilon-start", type=float, default=0.24)
    parser.add_argument("--epsilon-end", type=float, default=0.03)
    parser.add_argument("--output-dir", default="rl_fundamentals/results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = ["two_player", "four_player"] if args.mode == "both" else [args.mode]
    combined: Dict[str, object] = {"modes": {}}
    for offset, mode in enumerate(modes):
        result = train_mode(
            mode=mode,
            episodes=args.episodes,
            eval_episodes=args.eval_episodes,
            seed=args.seed + offset * 100,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
        )
        _write_training_artifacts(result, output_dir)
        mode_payload = dict(result)
        del mode_payload["policy"]
        combined["modes"][mode] = mode_payload

    (output_dir / "training_summary.json").write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(combined, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
