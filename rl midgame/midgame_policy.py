from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from midgame_features import FEATURE_NAMES


def _softmax(scores: Sequence[float], temperature: float) -> List[float]:
    if not scores:
        return []
    safe_temperature = max(1e-6, float(temperature))
    scaled = [score / safe_temperature for score in scores]
    anchor = max(scaled)
    exps = [math.exp(score - anchor) for score in scaled]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [value / total for value in exps]


def _sample_index(probabilities: Sequence[float], rng: random.Random) -> int:
    ticket = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if ticket <= cumulative:
            return index
    return max(0, len(probabilities) - 1)


@dataclass
class PolicyChoice:
    index: int
    scores: List[float]
    probabilities: List[float]


@dataclass
class DecisionSample:
    feature_vectors: List[List[float]]
    chosen_index: int
    probabilities: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class _MissionActorCritic(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(features)
        logits = self.policy_head(hidden).squeeze(-1)
        pooled = torch.cat([hidden.mean(dim=0), hidden.max(dim=0).values], dim=0)
        value = self.value_head(pooled).squeeze(-1)
        return logits, value


class LinearMissionPolicy:
    """A tiny linear softmax policy for midgame mission ranking."""

    def __init__(
        self,
        weights: Optional[Sequence[float]] = None,
        temperature: float = 1.0,
    ) -> None:
        self.weights = list(weights) if weights is not None else [0.0] * len(FEATURE_NAMES)
        if len(self.weights) != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} weights, got {len(self.weights)}")
        self.temperature = float(temperature)

    def score(self, vector: Sequence[float]) -> float:
        return sum(weight * value for weight, value in zip(self.weights, vector))

    def choose(
        self,
        feature_vectors: Sequence[Sequence[float]],
        rng: Optional[random.Random] = None,
        explore: bool = False,
    ) -> PolicyChoice:
        if not feature_vectors:
            raise ValueError("Cannot choose from an empty candidate list")
        scores = [self.score(vector) for vector in feature_vectors]
        probabilities = _softmax(scores, self.temperature)
        if explore:
            chooser = rng or random.Random()
            index = _sample_index(probabilities, chooser)
        else:
            index = max(range(len(scores)), key=lambda idx: scores[idx])
        return PolicyChoice(index=index, scores=scores, probabilities=probabilities)

    def update(
        self,
        samples: Sequence[DecisionSample],
        reward: float,
        learning_rate: float,
    ) -> Dict[str, float]:
        if not samples:
            return {"decisions": 0.0, "reward": reward, "avg_abs_weight": self.average_abs_weight()}

        lr = float(learning_rate)
        for sample in samples:
            vectors = sample.feature_vectors
            if not vectors:
                continue
            probabilities = sample.probabilities or _softmax(
                [self.score(vector) for vector in vectors],
                self.temperature,
            )
            chosen = vectors[sample.chosen_index]
            expected = [
                sum(probability * vector[idx] for probability, vector in zip(probabilities, vectors))
                for idx in range(len(self.weights))
            ]
            for idx in range(len(self.weights)):
                gradient = chosen[idx] - expected[idx]
                self.weights[idx] += lr * reward * gradient

        return {
            "decisions": float(len(samples)),
            "reward": reward,
            "avg_abs_weight": self.average_abs_weight(),
        }

    def average_abs_weight(self) -> float:
        return sum(abs(weight) for weight in self.weights) / max(1, len(self.weights))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "linear",
            "feature_names": list(FEATURE_NAMES),
            "temperature": self.temperature,
            "weights": list(self.weights),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LinearMissionPolicy":
        weights = payload.get("weights")
        # Migrate old 32-feature policies to 33 (append is_defer weight=0)
        if weights is not None and len(weights) < len(FEATURE_NAMES):
            weights = list(weights) + [0.0] * (len(FEATURE_NAMES) - len(weights))
        return cls(
            weights=weights,
            temperature=float(payload.get("temperature", 1.0)),
        )

    def save_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return output

    @classmethod
    def load_json(cls, path: str | Path) -> "LinearMissionPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


class MLPMissionPolicy:
    """Small tanh MLP scorer for mission ranking."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: int = 64,
        temperature: float = 1.0,
        seed: int = 7,
        w1: Optional[Sequence[Sequence[float]]] = None,
        b1: Optional[Sequence[float]] = None,
        w2: Optional[Sequence[float]] = None,
        b2: float = 0.0,
    ) -> None:
        self.input_size = int(input_size or len(FEATURE_NAMES))
        self.hidden_size = int(hidden_size)
        self.temperature = float(temperature)
        if self.input_size != len(FEATURE_NAMES):
            raise ValueError(f"Expected input size {len(FEATURE_NAMES)}, got {self.input_size}")

        if w1 is None or b1 is None or w2 is None:
            rng = random.Random(seed)
            self.w1 = [
                [(rng.random() - 0.5) * 0.1 for _ in range(self.input_size)]
                for _ in range(self.hidden_size)
            ]
            self.b1 = [0.0] * self.hidden_size
            self.w2 = [(rng.random() - 0.5) * 0.1 for _ in range(self.hidden_size)]
            self.b2 = 0.0
        else:
            self.w1 = [list(row) for row in w1]
            self.b1 = list(b1)
            self.w2 = list(w2)
            self.b2 = float(b2)

    def _forward(self, vector: Sequence[float]) -> tuple[float, List[float]]:
        hidden: List[float] = []
        for row, bias in zip(self.w1, self.b1):
            z = bias + sum(weight * value for weight, value in zip(row, vector))
            hidden.append(math.tanh(z))
        score = self.b2 + sum(weight * value for weight, value in zip(self.w2, hidden))
        return score, hidden

    def score(self, vector: Sequence[float]) -> float:
        score, _hidden = self._forward(vector)
        return score

    def choose(
        self,
        feature_vectors: Sequence[Sequence[float]],
        rng: Optional[random.Random] = None,
        explore: bool = False,
    ) -> PolicyChoice:
        if not feature_vectors:
            raise ValueError("Cannot choose from an empty candidate list")
        scores = [self.score(vector) for vector in feature_vectors]
        probabilities = _softmax(scores, self.temperature)
        if explore:
            chooser = rng or random.Random()
            index = _sample_index(probabilities, chooser)
        else:
            index = max(range(len(scores)), key=lambda idx: scores[idx])
        return PolicyChoice(index=index, scores=scores, probabilities=probabilities)

    def update(
        self,
        samples: Sequence[DecisionSample],
        reward: float,
        learning_rate: float,
    ) -> Dict[str, float]:
        if not samples:
            return {"decisions": 0.0, "reward": reward, "avg_abs_weight": self.average_abs_weight()}

        scale = float(learning_rate) * float(reward)
        for sample in samples:
            vectors = sample.feature_vectors
            if not vectors:
                continue
            forwards = [self._forward(vector) for vector in vectors]
            scores = [item[0] for item in forwards]
            probabilities = sample.probabilities or _softmax(scores, self.temperature)
            grad_scores = [-probability for probability in probabilities]
            grad_scores[sample.chosen_index] += 1.0

            grad_w1 = [[0.0] * self.input_size for _ in range(self.hidden_size)]
            grad_b1 = [0.0] * self.hidden_size
            grad_w2 = [0.0] * self.hidden_size
            grad_b2 = 0.0

            for vector, (_score, hidden), grad_score in zip(vectors, forwards, grad_scores):
                coeff = scale * grad_score
                if coeff == 0.0:
                    continue
                grad_b2 += coeff
                for hidden_index in range(self.hidden_size):
                    hidden_value = hidden[hidden_index]
                    grad_w2[hidden_index] += coeff * hidden_value
                    delta = coeff * self.w2[hidden_index] * (1.0 - hidden_value * hidden_value)
                    grad_b1[hidden_index] += delta
                    row = grad_w1[hidden_index]
                    for feature_index, feature_value in enumerate(vector):
                        row[feature_index] += delta * feature_value

            self.b2 += grad_b2
            for hidden_index in range(self.hidden_size):
                self.b1[hidden_index] += grad_b1[hidden_index]
                self.w2[hidden_index] += grad_w2[hidden_index]
                row = self.w1[hidden_index]
                grad_row = grad_w1[hidden_index]
                for feature_index in range(self.input_size):
                    row[feature_index] += grad_row[feature_index]

        return {
            "decisions": float(len(samples)),
            "reward": reward,
            "avg_abs_weight": self.average_abs_weight(),
        }

    def average_abs_weight(self) -> float:
        total = sum(abs(value) for row in self.w1 for value in row)
        total += sum(abs(value) for value in self.b1)
        total += sum(abs(value) for value in self.w2)
        total += abs(self.b2)
        denom = self.hidden_size * self.input_size + self.hidden_size + self.hidden_size + 1
        return total / max(1, denom)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "mlp",
            "feature_names": list(FEATURE_NAMES),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "temperature": self.temperature,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MLPMissionPolicy":
        w1 = payload.get("w1")
        saved_input = int(payload.get("input_size", len(FEATURE_NAMES)))
        # Migrate old policies: pad w1 rows if input_size grew
        if w1 is not None and saved_input < len(FEATURE_NAMES):
            pad = len(FEATURE_NAMES) - saved_input
            w1 = [list(row) + [0.0] * pad for row in w1]
            saved_input = len(FEATURE_NAMES)
        return cls(
            input_size=saved_input,
            hidden_size=int(payload.get("hidden_size", 64)),
            temperature=float(payload.get("temperature", 1.0)),
            w1=w1,
            b1=payload.get("b1"),
            w2=payload.get("w2"),
            b2=float(payload.get("b2", 0.0)),
        )

    def save_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return output

    @classmethod
    def load_json(cls, path: str | Path) -> "MLPMissionPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


class PPOMissionPolicy:
    """Candidate-ranking actor-critic policy with PPO updates."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: int = 128,
        temperature: float = 1.0,
        learning_rate: float = 3e-4,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        seed: int = 7,
        state_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.input_size = int(input_size or len(FEATURE_NAMES))
        self.hidden_size = int(hidden_size)
        self.temperature = float(temperature)
        self.learning_rate = float(learning_rate)
        self.clip_coef = float(clip_coef)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)
        if self.input_size != len(FEATURE_NAMES):
            raise ValueError(f"Expected input size {len(FEATURE_NAMES)}, got {self.input_size}")

        torch.manual_seed(seed)
        self.model = _MissionActorCritic(self.input_size, self.hidden_size)
        if state_dict is not None:
            tensor_state = {name: torch.tensor(values, dtype=torch.float32) for name, values in state_dict.items()}
            self.model.load_state_dict(tensor_state)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _candidate_tensor(self, feature_vectors: Sequence[Sequence[float]]) -> torch.Tensor:
        if not feature_vectors:
            raise ValueError("Cannot choose from an empty candidate list")
        return torch.tensor(feature_vectors, dtype=torch.float32)

    def _distribution(self, feature_vectors: Sequence[Sequence[float]]) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        features = self._candidate_tensor(feature_vectors)
        with torch.no_grad():
            logits, value = self.model(features)
        scaled_logits = logits / max(1e-6, self.temperature)
        return torch.distributions.Categorical(logits=scaled_logits), value

    def choose(
        self,
        feature_vectors: Sequence[Sequence[float]],
        rng: Optional[random.Random] = None,
        explore: bool = False,
    ) -> PolicyChoice:
        distribution, _value = self._distribution(feature_vectors)
        probabilities = distribution.probs.detach().cpu().tolist()
        if explore:
            chooser = rng or random.Random()
            index = _sample_index(probabilities, chooser)
        else:
            index = int(torch.argmax(distribution.probs).item())
        scores = distribution.logits.detach().cpu().tolist()
        return PolicyChoice(index=index, scores=scores, probabilities=probabilities)

    def decision_metadata(
        self,
        feature_vectors: Sequence[Sequence[float]],
        choice: PolicyChoice,
    ) -> Dict[str, Any]:
        distribution, value = self._distribution(feature_vectors)
        action = torch.tensor(choice.index, dtype=torch.int64)
        return {
            "policy_kind": "ppo",
            "old_log_prob": float(distribution.log_prob(action).item()),
            "value_estimate": float(value.item()),
            "candidate_count": len(feature_vectors),
        }

    def batch_update(
        self,
        episodes: Sequence[Dict[str, Any]],
        learning_rate: Optional[float] = None,
        epochs: int = 4,
    ) -> Dict[str, float]:
        if learning_rate is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = float(learning_rate)

        records: List[Dict[str, Any]] = []
        for episode in episodes:
            reward = float(episode.get("reward", 0.0))
            for sample in episode.get("samples", []):
                if not sample.feature_vectors:
                    continue
                records.append({
                    "sample": sample,
                    "return": reward,
                    "old_log_prob": float(sample.metadata.get("old_log_prob", 0.0)),
                    "old_value": float(sample.metadata.get("value_estimate", 0.0)),
                })

        if not records:
            return {"decisions": 0.0, "reward": 0.0, "avg_abs_weight": self.average_abs_weight(), "loss": 0.0}

        returns = torch.tensor([record["return"] for record in records], dtype=torch.float32)
        old_values = torch.tensor([record["old_value"] for record in records], dtype=torch.float32)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        loss_value = 0.0
        for _ in range(max(1, int(epochs))):
            total_loss = torch.tensor(0.0, dtype=torch.float32)
            for index, record in enumerate(records):
                sample = record["sample"]
                features = self._candidate_tensor(sample.feature_vectors)
                logits, value = self.model(features)
                scaled_logits = logits / max(1e-6, self.temperature)
                distribution = torch.distributions.Categorical(logits=scaled_logits)
                action = torch.tensor(sample.chosen_index, dtype=torch.int64)
                new_log_prob = distribution.log_prob(action)
                old_log_prob = torch.tensor(record["old_log_prob"], dtype=torch.float32)
                ratio = torch.exp(new_log_prob - old_log_prob)
                advantage = advantages[index]
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * advantage
                actor_loss = -torch.min(surr1, surr2)
                target_return = returns[index]
                value_loss = F.mse_loss(value, target_return)
                entropy = distribution.entropy()
                total_loss = total_loss + actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy

            total_loss = total_loss / float(len(records))
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            loss_value = float(total_loss.item())

        return {
            "decisions": float(len(records)),
            "reward": float(returns.mean().item()),
            "avg_abs_weight": self.average_abs_weight(),
            "loss": loss_value,
        }

    def update(
        self,
        samples: Sequence[DecisionSample],
        reward: float,
        learning_rate: float,
    ) -> Dict[str, float]:
        return self.batch_update([{"samples": list(samples), "reward": reward}], learning_rate=learning_rate, epochs=2)

    def clone(self) -> "PPOMissionPolicy":
        return PPOMissionPolicy.from_dict(self.to_dict())

    def average_abs_weight(self) -> float:
        total = 0.0
        count = 0
        for parameter in self.model.parameters():
            total += float(parameter.detach().abs().sum().item())
            count += parameter.numel()
        return total / max(1, count)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "ppo",
            "feature_names": list(FEATURE_NAMES),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "temperature": self.temperature,
            "learning_rate": self.learning_rate,
            "clip_coef": self.clip_coef,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "state_dict": {name: tensor.detach().cpu().tolist() for name, tensor in self.model.state_dict().items()},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PPOMissionPolicy":
        return cls(
            input_size=int(payload.get("input_size", len(FEATURE_NAMES))),
            hidden_size=int(payload.get("hidden_size", 128)),
            temperature=float(payload.get("temperature", 1.0)),
            learning_rate=float(payload.get("learning_rate", 3e-4)),
            clip_coef=float(payload.get("clip_coef", 0.2)),
            ent_coef=float(payload.get("ent_coef", 0.01)),
            vf_coef=float(payload.get("vf_coef", 0.5)),
            max_grad_norm=float(payload.get("max_grad_norm", 0.5)),
            state_dict=payload.get("state_dict"),
        )

    def save_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return output

    @classmethod
    def load_json(cls, path: str | Path) -> "PPOMissionPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def create_policy(kind: str = "linear", hidden_size: int = 64, seed: int = 7) -> Any:
    if kind == "linear":
        return LinearMissionPolicy()
    if kind == "mlp":
        return MLPMissionPolicy(hidden_size=hidden_size, seed=seed)
    if kind == "ppo":
        return PPOMissionPolicy(hidden_size=hidden_size, seed=seed)
    raise ValueError(f"Unsupported policy kind: {kind}")


def load_policy_json(path: str | Path) -> Any:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    kind = str(payload.get("kind", "linear"))
    if kind == "ppo":
        return PPOMissionPolicy.from_dict(payload)
    if kind == "mlp":
        return MLPMissionPolicy.from_dict(payload)
    return LinearMissionPolicy.from_dict(payload)
