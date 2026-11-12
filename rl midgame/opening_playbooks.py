"""Opening playbook extraction from replays + contextual bandit selector.

Instead of RL for openings (which fails due to synthetic noise and lack of
real imitation targets), we:
1. Extract strong openings from real replays (imitation).
2. Cluster them into playbooks by map similarity.
3. Use a contextual bandit (UCB1 + linear model) to pick the right playbook
   at game start based on map features.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

WORKSPACE_DIR = Path(__file__).resolve().parent
ROOT = WORKSPACE_DIR.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from replay_midgame_experiment import (  # noqa: E402
    StepMetrics,
    compute_step_metrics,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MapFeatures:
    num_players: int
    num_planets: int
    avg_planet_distance: float
    cluster_score: float
    starting_production: float
    nearest_neutral_distance: float
    nearest_enemy_distance: float
    has_orbiting_planets: bool

    def to_vector(self) -> List[float]:
        return [
            float(self.num_players),
            float(self.num_planets),
            self.avg_planet_distance,
            self.cluster_score,
            self.starting_production,
            self.nearest_neutral_distance,
            self.nearest_enemy_distance,
            1.0 if self.has_orbiting_planets else 0.0,
        ]

    @staticmethod
    def from_vector(v: List[float]) -> "MapFeatures":
        return MapFeatures(
            num_players=int(round(v[0])),
            num_planets=int(round(v[1])),
            avg_planet_distance=v[2],
            cluster_score=v[3],
            starting_production=v[4],
            nearest_neutral_distance=v[5],
            nearest_enemy_distance=v[6],
            has_orbiting_planets=v[7] >= 0.5,
        )

    NUM_FEATURES = 8


@dataclass
class OpeningSequence:
    actions: List[List[List[float]]]  # turns 0..K, each a list of moves
    map_features: MapFeatures
    final_reward: float
    episode_id: int
    player_index: int
    midgame_share: float


@dataclass
class Playbook:
    id: int
    representative_actions: List[List[List[float]]]
    map_feature_centroid: MapFeatures
    avg_reward: float
    count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _player_index(replay: Dict[str, Any], player_name: str) -> Optional[int]:
    needle = player_name.strip().lower()
    team_names = replay.get("info", {}).get("TeamNames") or []
    for idx, name in enumerate(team_names):
        if needle in str(name).lower():
            return idx
    return None


def _euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _planet_pos(planet: list) -> Tuple[float, float]:
    """Extract (x, y) from a planet record [id, owner, x, y, ...]."""
    return (float(planet[2]), float(planet[3]))


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_map_features(
    observation: Dict[str, Any],
    player_index: int,
    num_players: int,
    configuration: Optional[Dict[str, Any]] = None,
) -> MapFeatures:
    """Derive MapFeatures from the initial observation."""
    planets = observation.get("planets") or []
    num_planets = len(planets)
    positions = [_planet_pos(p) for p in planets]

    # Average pairwise distance
    pair_dists: List[float] = []
    for i in range(num_planets):
        for j in range(i + 1, num_planets):
            pair_dists.append(_distance(positions[i], positions[j]))
    avg_planet_distance = sum(pair_dists) / len(pair_dists) if pair_dists else 0.0

    # Cluster score: std-dev of pairwise distances / mean (CV).
    # Low CV => uniform spread; high CV => some clusters.
    if pair_dists and avg_planet_distance > 0:
        variance = sum((d - avg_planet_distance) ** 2 for d in pair_dists) / len(pair_dists)
        cluster_score = math.sqrt(variance) / avg_planet_distance
    else:
        cluster_score = 0.0

    # Find the player's starting planet (owned, largest ships)
    owned = [p for p in planets if int(p[1]) == player_index]
    if owned:
        starting = max(owned, key=lambda p: float(p[5]))
        starting_production = float(starting[6])
        start_pos = _planet_pos(starting)
    else:
        starting_production = 0.0
        start_pos = (0.0, 0.0)

    # Nearest neutral
    neutrals = [p for p in planets if int(p[1]) < 0 or int(p[1]) >= num_players]
    if neutrals:
        nearest_neutral_distance = min(_distance(start_pos, _planet_pos(p)) for p in neutrals)
    else:
        nearest_neutral_distance = float("inf")

    # Nearest enemy planet
    enemies = [p for p in planets if 0 <= int(p[1]) < num_players and int(p[1]) != player_index]
    if enemies:
        nearest_enemy_distance = min(_distance(start_pos, _planet_pos(p)) for p in enemies)
    else:
        nearest_enemy_distance = float("inf")

    # Orbiting detection: check configuration or heuristic
    has_orbiting = False
    if configuration and configuration.get("orbiting_planets"):
        has_orbiting = True

    return MapFeatures(
        num_players=num_players,
        num_planets=num_planets,
        avg_planet_distance=avg_planet_distance,
        cluster_score=cluster_score,
        starting_production=starting_production,
        nearest_neutral_distance=nearest_neutral_distance,
        nearest_enemy_distance=nearest_enemy_distance,
        has_orbiting_planets=has_orbiting,
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_openings(
    replay_paths: List[Path],
    player_name: str,
    max_turn: int = 40,
    min_reward: float = 0.0,
) -> List[OpeningSequence]:
    """Extract opening sequences from replays where *player_name* did well."""
    openings: List[OpeningSequence] = []

    for rp in replay_paths:
        try:
            replay = _load_json(rp)
        except (json.JSONDecodeError, OSError):
            continue

        pidx = _player_index(replay, player_name)
        if pidx is None:
            continue

        rewards = replay.get("rewards") or []
        if pidx >= len(rewards):
            continue
        final_reward = float(rewards[pidx]) if rewards[pidx] is not None else 0.0
        if final_reward < min_reward:
            continue

        steps = replay.get("steps") or []
        if len(steps) < 2:
            continue

        num_agents = len(steps[0])
        configuration = replay.get("configuration") or {}
        episode_id = replay.get("info", {}).get("EpisodeId", 0)

        # Initial observation for map features
        initial_obs = steps[0][pidx].get("observation") or {}
        mf = compute_map_features(initial_obs, pidx, num_agents, configuration)

        # Collect actions for turns 0..max_turn-1
        actions: List[List[List[float]]] = []
        effective_end = min(max_turn, len(steps))
        for t in range(effective_end):
            step_data = steps[t]
            if pidx < len(step_data):
                raw_action = step_data[pidx].get("action") or []
                actions.append([[float(v) for v in move] for move in raw_action])
            else:
                actions.append([])

        # Compute midgame share at turn max_turn (or last available step)
        share_step = min(max_turn, len(steps) - 1)
        share_obs = steps[share_step][pidx].get("observation") or {}
        metrics = compute_step_metrics(share_obs, pidx, num_agents, share_step)
        midgame_share = metrics.target_share

        openings.append(OpeningSequence(
            actions=actions,
            map_features=mf,
            final_reward=final_reward,
            episode_id=episode_id,
            player_index=pidx,
            midgame_share=midgame_share,
        ))

    return openings


# ---------------------------------------------------------------------------
# Clustering (k-means from scratch)
# ---------------------------------------------------------------------------

def _normalize_features(vectors: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    """Z-score normalise feature columns. Returns (normalised, means, stds)."""
    if not vectors:
        return [], [], []
    dim = len(vectors[0])
    means = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            means[i] += v[i]
    n = len(vectors)
    means = [m / n for m in means]

    stds = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            stds[i] += (v[i] - means[i]) ** 2
    stds = [math.sqrt(s / n) if s > 0 else 1.0 for s in stds]

    normed = [[(v[i] - means[i]) / stds[i] for i in range(dim)] for v in vectors]
    return normed, means, stds


def _kmeans(vectors: List[List[float]], k: int, max_iter: int = 50, seed: int = 42) -> List[int]:
    """Simple k-means returning cluster assignments."""
    n = len(vectors)
    if n == 0:
        return []
    k = min(k, n)
    rng = random.Random(seed)

    # k-means++ initialisation
    centroids: List[List[float]] = [list(vectors[rng.randint(0, n - 1)])]
    for _ in range(1, k):
        dists = [min(_euclidean(v, c) ** 2 for c in centroids) for v in vectors]
        total = sum(dists)
        if total == 0:
            centroids.append(list(vectors[rng.randint(0, n - 1)]))
            continue
        threshold = rng.random() * total
        cumulative = 0.0
        for idx, d in enumerate(dists):
            cumulative += d
            if cumulative >= threshold:
                centroids.append(list(vectors[idx]))
                break
        else:
            centroids.append(list(vectors[-1]))

    dim = len(vectors[0])
    assignments = [0] * n

    for _ in range(max_iter):
        # Assign
        changed = False
        for i, v in enumerate(vectors):
            best_c = min(range(k), key=lambda c: _euclidean(v, centroids[c]))
            if best_c != assignments[i]:
                changed = True
                assignments[i] = best_c
        if not changed:
            break

        # Update centroids
        for c in range(k):
            members = [vectors[i] for i in range(n) if assignments[i] == c]
            if members:
                centroids[c] = [sum(m[d] for m in members) / len(members) for d in range(dim)]

    return assignments


def cluster_openings(
    openings: List[OpeningSequence],
    n_clusters: int = 8,
) -> List[Playbook]:
    """Cluster openings by map features and build playbooks."""
    if not openings:
        return []

    raw_vecs = [o.map_features.to_vector() for o in openings]
    normed, _means, _stds = _normalize_features(raw_vecs)

    n_clusters = min(n_clusters, len(openings))
    assignments = _kmeans(normed, n_clusters)

    # Group by cluster
    clusters: Dict[int, List[int]] = {}
    for idx, cid in enumerate(assignments):
        clusters.setdefault(cid, []).append(idx)

    playbooks: List[Playbook] = []
    for pid, (cid, members) in enumerate(sorted(clusters.items())):
        member_openings = [openings[i] for i in members]

        # Representative = best midgame share (opening quality signal)
        best = max(member_openings, key=lambda o: o.midgame_share)

        # Centroid of raw (unnormalised) features
        dim = MapFeatures.NUM_FEATURES
        centroid_vec = [0.0] * dim
        for o in member_openings:
            v = o.map_features.to_vector()
            for d in range(dim):
                centroid_vec[d] += v[d]
        centroid_vec = [x / len(member_openings) for x in centroid_vec]

        avg_reward = sum(o.final_reward for o in member_openings) / len(member_openings)

        playbooks.append(Playbook(
            id=pid,
            representative_actions=best.actions,
            map_feature_centroid=MapFeatures.from_vector(centroid_vec),
            avg_reward=avg_reward,
            count=len(member_openings),
        ))

    return playbooks


# ---------------------------------------------------------------------------
# Contextual Bandit (UCB1 + linear model)
# ---------------------------------------------------------------------------

class ContextualBandit:
    """Selects among playbooks given map features.

    Each arm (playbook) has a weight vector of length NUM_FEATURES.
    Expected reward = dot(weights, feature_vector).
    Exploration via UCB1 bonus.
    """

    def __init__(self, n_arms: int, n_features: int = MapFeatures.NUM_FEATURES, c: float = 2.0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.c = c  # UCB exploration constant
        # Linear weights per arm (initialised small random)
        rng = random.Random(0)
        self.weights: List[List[float]] = [
            [rng.gauss(0, 0.01) for _ in range(n_features)] for _ in range(n_arms)
        ]
        self.counts: List[int] = [0] * n_arms
        self.total_pulls: int = 0
        # Online learning rate
        self.lr: float = 0.01

    def _predicted_reward(self, arm: int, features: List[float]) -> float:
        return sum(w * f for w, f in zip(self.weights[arm], features))

    def select(self, map_features: MapFeatures) -> int:
        """Select the best playbook index for these map features (UCB1)."""
        fv = map_features.to_vector()
        self.total_pulls += 1
        best_arm = 0
        best_score = -float("inf")
        for arm in range(self.n_arms):
            pred = self._predicted_reward(arm, fv)
            if self.counts[arm] == 0:
                # Untried arm gets max priority
                ucb = float("inf")
            else:
                ucb = pred + self.c * math.sqrt(math.log(self.total_pulls) / self.counts[arm])
            if ucb > best_score:
                best_score = ucb
                best_arm = arm
        return best_arm

    def update(self, playbook_idx: int, reward: float, map_features: Optional[MapFeatures] = None) -> None:
        """Update the model after observing a reward."""
        self.counts[playbook_idx] += 1
        if map_features is not None:
            fv = map_features.to_vector()
            pred = self._predicted_reward(playbook_idx, fv)
            error = reward - pred
            # Gradient step: w += lr * error * feature
            for i in range(self.n_features):
                self.weights[playbook_idx][i] += self.lr * error * fv[i]

    def save_json(self, path: str) -> None:
        data = {
            "n_arms": self.n_arms,
            "n_features": self.n_features,
            "c": self.c,
            "lr": self.lr,
            "weights": self.weights,
            "counts": self.counts,
            "total_pulls": self.total_pulls,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str) -> "ContextualBandit":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        bandit = cls(
            n_arms=data["n_arms"],
            n_features=data.get("n_features", MapFeatures.NUM_FEATURES),
            c=data.get("c", 2.0),
        )
        bandit.weights = data["weights"]
        bandit.counts = data["counts"]
        bandit.total_pulls = data.get("total_pulls", sum(bandit.counts))
        bandit.lr = data.get("lr", 0.01)
        return bandit


# ---------------------------------------------------------------------------
# Agent wrapper
# ---------------------------------------------------------------------------

class OpeningPlaybookAgent:
    """Wraps playbook selection + fallback into the agent(obs, config) interface."""

    def __init__(
        self,
        playbooks: List[Playbook],
        bandit: ContextualBandit,
        fallback_agent: Callable[[Dict[str, Any], Dict[str, Any]], Any],
    ):
        self.playbooks = playbooks
        self.bandit = bandit
        self.fallback_agent = fallback_agent
        self._selected_playbook: Optional[Playbook] = None
        self._opening_length: int = 0  # set from playbook actions length

    def __call__(self, obs: Dict[str, Any], config: Dict[str, Any]) -> Any:
        step = obs.get("step", 0)

        if step == 0:
            num_players = obs.get("num_players", 2)
            mf = compute_map_features(obs, obs.get("index", 0), num_players, config)
            idx = self.bandit.select(mf)
            idx = min(idx, len(self.playbooks) - 1)
            self._selected_playbook = self.playbooks[idx]
            self._opening_length = len(self._selected_playbook.representative_actions)

        if self._selected_playbook is not None and step < self._opening_length:
            actions = self._selected_playbook.representative_actions[step]
            if actions:
                return actions

        return self.fallback_agent(obs, config)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _playbook_to_dict(pb: Playbook) -> Dict[str, Any]:
    return {
        "id": pb.id,
        "representative_actions": pb.representative_actions,
        "map_feature_centroid": asdict(pb.map_feature_centroid),
        "avg_reward": pb.avg_reward,
        "count": pb.count,
    }


def _playbook_from_dict(d: Dict[str, Any]) -> Playbook:
    return Playbook(
        id=d["id"],
        representative_actions=d["representative_actions"],
        map_feature_centroid=MapFeatures(**d["map_feature_centroid"]),
        avg_reward=d["avg_reward"],
        count=d["count"],
    )


def save_playbooks(playbooks: List[Playbook], path: str) -> None:
    data = [_playbook_to_dict(pb) for pb in playbooks]
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_playbooks(path: str) -> List[Playbook]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_playbook_from_dict(d) for d in data]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract opening playbooks from replays")
    parser.add_argument("--replay-glob", type=str, default="kaggle_replays/*/episode-*-replay.json",
                        help="Glob pattern for replay files (relative to repo root)")
    parser.add_argument("--player-name", type=str, required=True,
                        help="Player/team name to extract openings for")
    parser.add_argument("--max-turn", type=int, default=40,
                        help="Opening length in turns (default: 40)")
    parser.add_argument("--n-clusters", type=int, default=8,
                        help="Number of playbooks / clusters (default: 8)")
    parser.add_argument("--min-reward", type=float, default=0.0,
                        help="Minimum final reward to include a replay (default: 0.0)")
    parser.add_argument("--playbooks-out", type=str, default="opening_playbooks.json",
                        help="Output path for playbooks JSON")
    parser.add_argument("--summary-out", type=str, default=None,
                        help="Output path for human-readable summary")
    args = parser.parse_args()

    replay_paths = sorted(ROOT.glob(args.replay_glob))
    print(f"Found {len(replay_paths)} replay files")

    openings = extract_openings(replay_paths, args.player_name, args.max_turn, args.min_reward)
    print(f"Extracted {len(openings)} openings")

    if not openings:
        print("No openings found — check player name and min-reward")
        return

    playbooks = cluster_openings(openings, args.n_clusters)
    print(f"Created {len(playbooks)} playbooks")

    save_playbooks(playbooks, args.playbooks_out)
    print(f"Saved playbooks to {args.playbooks_out}")

    # Summary
    lines: List[str] = [f"Opening Playbooks Summary ({len(playbooks)} clusters from {len(openings)} openings)\n"]
    for pb in playbooks:
        mf = pb.map_feature_centroid
        lines.append(
            f"  Playbook {pb.id}: count={pb.count}, avg_reward={pb.avg_reward:.3f}, "
            f"planets={mf.num_planets}, avg_dist={mf.avg_planet_distance:.1f}, "
            f"cluster={mf.cluster_score:.2f}, production={mf.starting_production:.1f}, "
            f"orbiting={mf.has_orbiting_planets}"
        )
    summary = "\n".join(lines)
    print(summary)

    if args.summary_out:
        Path(args.summary_out).write_text(summary, encoding="utf-8")
        print(f"Saved summary to {args.summary_out}")


if __name__ == "__main__":
    main()
