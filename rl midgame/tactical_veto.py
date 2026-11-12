"""Tactical verification / veto layer for the Orbit Wars RL midgame reranker.

Sits AFTER the reranker picks a candidate action and can veto the choice
if a simple linear value model predicts the action makes things worse.
Only vetoes when protecting a lead — never vetoes defense or catch-up play.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Feature vector for board-position value estimation
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "my_ship_share",
    "my_prod_share",
    "step_frac",
    "is_two_player",
    "my_planet_frac",
    "frontline_strength_ratio",
]

NUM_FEATURES: int = len(FEATURE_NAMES)


@dataclass
class PositionFeatures:
    """Raw feature vector describing the board from one player's viewpoint."""
    my_ship_share: float = 0.0
    my_prod_share: float = 0.0
    step_frac: float = 0.0
    is_two_player: float = 0.0
    my_planet_frac: float = 0.0
    frontline_strength_ratio: float = 0.0

    def to_list(self) -> List[float]:
        return [
            self.my_ship_share,
            self.my_prod_share,
            self.step_frac,
            self.is_two_player,
            self.my_planet_frac,
            self.frontline_strength_ratio,
        ]


# ---------------------------------------------------------------------------
# PositionValue — linear value model V(s)
# ---------------------------------------------------------------------------

# Sensible initial weights (hand-tuned baseline before any training)
_DEFAULT_WEIGHTS: List[float] = [0.4, 0.3, 0.0, 0.0, 0.2, 0.1]
_DEFAULT_BIAS: float = 0.0


@dataclass
class PositionValue:
    """Linear model: V(s) = sigmoid(w · features + b), output in [0, 1]."""
    weights: List[float] = field(default_factory=lambda: list(_DEFAULT_WEIGHTS))
    bias: float = _DEFAULT_BIAS

    def score(self, features: List[float]) -> float:
        """Estimate win-probability from position features."""
        z = self.bias + sum(w * f for w, f in zip(self.weights, features))
        return _sigmoid(z)

    # -- persistence ---------------------------------------------------------

    def save(self, path: Path) -> None:
        data = {"weights": self.weights, "bias": self.bias}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PositionValue":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(weights=data["weights"], bias=data["bias"])

    # -- training ------------------------------------------------------------

    @classmethod
    def train_from_replays(
        cls,
        replay_paths: Sequence[str | Path],
        player_name: str,
        *,
        lr: float = 0.01,
        epochs: int = 50,
        sample_every: int = 10,
        max_step: int = 400,
    ) -> "PositionValue":
        """Supervised regression from replay outcomes.

        Loads Kaggle environment replay JSONs, extracts position features at
        periodic steps, and trains via gradient descent on BCE loss against
        the final reward normalised to [0, 1].
        """
        samples = _extract_training_samples(
            replay_paths, player_name, sample_every=sample_every, max_step=max_step,
        )
        if not samples:
            return cls()

        model = cls()
        for _ in range(epochs):
            _train_epoch(model, samples, lr)
        return model


# ---------------------------------------------------------------------------
# Feature extraction from live game state (DecisionLogic)
# ---------------------------------------------------------------------------

def extract_features_from_logic(logic: Any) -> PositionFeatures:
    """Build a PositionFeatures from a live DecisionLogic instance."""
    state = logic.state
    player = state.player
    planets = list(state.planets_by_id.values())
    total_planets = max(1, len(planets))

    my_planets = [p for p in planets if p.owner == player]
    enemy_planets = [p for p in planets if p.owner not in (-1, player)]

    my_ships_on_planets = sum(p.ships for p in my_planets)
    total_ships_on_planets = sum(p.ships for p in planets)

    # Include fleets in ship counts
    my_fleet_ships = sum(f.ships for f in state.fleets if f.owner == player)
    total_fleet_ships = sum(f.ships for f in state.fleets)
    my_total_ships = my_ships_on_planets + my_fleet_ships
    total_ships = max(1.0, total_ships_on_planets + total_fleet_ships)

    my_prod = sum(p.production for p in my_planets)
    total_prod = max(1.0, sum(p.production for p in planets))

    step_frac = state.step / max(1, state.remaining_steps + state.step)

    # Frontline strength ratio: compare ships on planets adjacent to enemies
    frontline_ratio = _compute_frontline_ratio(my_planets, enemy_planets, player)

    return PositionFeatures(
        my_ship_share=my_total_ships / total_ships,
        my_prod_share=my_prod / total_prod,
        step_frac=step_frac,
        is_two_player=1.0 if state.num_players == 2 else 0.0,
        my_planet_frac=len(my_planets) / total_planets,
        frontline_strength_ratio=frontline_ratio,
    )


def _compute_frontline_ratio(
    my_planets: List[Any], enemy_planets: List[Any], player: int,
) -> float:
    """Ratio of my frontline ships to enemy frontline ships.

    'Frontline' = planets within interaction range of any enemy planet.
    Uses a simple distance threshold based on typical Orbit Wars board size.
    """
    if not my_planets or not enemy_planets:
        return 1.0

    interaction_dist = 30.0  # typical max useful launch range

    my_front_ships = 0.0
    for mp in my_planets:
        for ep in enemy_planets:
            dx = mp.x - ep.x
            dy = mp.y - ep.y
            if math.sqrt(dx * dx + dy * dy) <= interaction_dist:
                my_front_ships += mp.ships
                break

    enemy_front_ships = 0.0
    for ep in enemy_planets:
        for mp in my_planets:
            dx = ep.x - mp.x
            dy = ep.y - mp.y
            if math.sqrt(dx * dx + dy * dy) <= interaction_dist:
                enemy_front_ships += ep.ships
                break

    return my_front_ships / max(1.0, enemy_front_ships)


# ---------------------------------------------------------------------------
# 1-step forward value estimation
# ---------------------------------------------------------------------------

def estimate_post_launch_value(
    logic: Any,
    mission: Any,
    value_model: PositionValue,
) -> float:
    """Estimate V(s') after launching *mission* from the current state.

    Simple model: subtract committed ships from source planets (they are now
    in flight and don't contribute to planet defense).  Don't add them to the
    target yet — they haven't arrived.
    """
    state = logic.state
    player = state.player
    planets = list(state.planets_by_id.values())

    # Build a ship-delta map from the mission
    ships_sent: Dict[int, int] = {}
    for src_id, ship_count in zip(mission.source_ids, mission.ships):
        ships_sent[src_id] = ships_sent.get(src_id, 0) + ship_count

    # Recompute features with modified ship counts on source planets
    total_planets = max(1, len(planets))
    my_planets_mod: List[Any] = []
    all_ships = 0.0
    my_ships = 0.0
    my_prod = 0.0
    total_prod = 0.0
    enemy_planets: List[Any] = []

    for p in planets:
        effective_ships = p.ships
        if p.id in ships_sent and p.owner == player:
            effective_ships = max(0, p.ships - ships_sent[p.id])

        all_ships += effective_ships
        if p.owner == player:
            my_ships += effective_ships
            my_prod += p.production
            my_planets_mod.append(p)
        if p.owner not in (-1, player):
            enemy_planets.append(p)
        total_prod += p.production

    # Fleet ships still count (including the new ones just launched)
    my_fleet_ships = sum(f.ships for f in state.fleets if f.owner == player)
    total_fleet_ships = sum(f.ships for f in state.fleets)
    launched_ships = sum(ships_sent.values())
    my_total = my_ships + my_fleet_ships + launched_ships
    total = max(1.0, all_ships + total_fleet_ships + launched_ships)

    step_frac = state.step / max(1, state.remaining_steps + state.step)
    frontline_ratio = _compute_frontline_ratio(my_planets_mod, enemy_planets, player)

    features = PositionFeatures(
        my_ship_share=my_total / total,
        my_prod_share=my_prod / max(1.0, total_prod),
        step_frac=step_frac,
        is_two_player=1.0 if state.num_players == 2 else 0.0,
        my_planet_frac=len(my_planets_mod) / total_planets,
        frontline_strength_ratio=frontline_ratio,
    )
    return value_model.score(features.to_list())


# ---------------------------------------------------------------------------
# VetoDecision dataclass
# ---------------------------------------------------------------------------

@dataclass
class VetoDecision:
    """Result of the tactical veto evaluation."""
    mission: Any  # MissionOption
    current_value: float
    post_value: float
    vetoed: bool
    reason: str


# ---------------------------------------------------------------------------
# TacticalVetoLayer
# ---------------------------------------------------------------------------

_DEFENSE_MISSIONS = frozenset({"defend", "reinforce", "emergency_defend"})


class TacticalVetoLayer:
    """Post-reranker veto gate that rejects value-destroying actions.

    Parameters
    ----------
    value_model : PositionValue
        The linear value estimator.
    epsilon : float
        Tolerable value drop before vetoing (default 0.02).
    min_step : int
        Don't veto before this game step (early game is too noisy).
    """

    def __init__(
        self,
        value_model: PositionValue,
        epsilon: float = 0.02,
        min_step: int = 24,
    ) -> None:
        self.value_model = value_model
        self.epsilon = epsilon
        self.min_step = min_step

    def evaluate(self, logic: Any, mission: Any) -> VetoDecision:
        """Decide whether *mission* should be vetoed.

        Rules:
        - Never veto defense missions.
        - Never veto when behind (only protect leads).
        - Veto if V(s') < V(s) - epsilon.
        """
        current_features = extract_features_from_logic(logic)
        current_value = self.value_model.score(current_features.to_list())
        post_value = estimate_post_launch_value(logic, mission, self.value_model)

        # Never veto defense
        mission_type = getattr(mission, "mission", "")
        if mission_type in _DEFENSE_MISSIONS:
            return VetoDecision(
                mission=mission,
                current_value=current_value,
                post_value=post_value,
                vetoed=False,
                reason="defense missions are never vetoed",
            )

        # Never veto when behind
        modes = getattr(logic, "modes", {})
        is_behind = modes.get("is_behind", False)
        if is_behind:
            return VetoDecision(
                mission=mission,
                current_value=current_value,
                post_value=post_value,
                vetoed=False,
                reason="not vetoing while behind",
            )

        # Don't veto in very early game
        step = getattr(logic.state, "step", 0)
        if step < self.min_step:
            return VetoDecision(
                mission=mission,
                current_value=current_value,
                post_value=post_value,
                vetoed=False,
                reason=f"too early to veto (step {step} < {self.min_step})",
            )

        # Core veto check: reject if value drops more than epsilon
        value_drop = current_value - post_value
        if value_drop > self.epsilon:
            return VetoDecision(
                mission=mission,
                current_value=current_value,
                post_value=post_value,
                vetoed=True,
                reason=(
                    f"value drop {value_drop:.4f} exceeds epsilon "
                    f"{self.epsilon:.4f} ({current_value:.4f} -> {post_value:.4f})"
                ),
            )

        return VetoDecision(
            mission=mission,
            current_value=current_value,
            post_value=post_value,
            vetoed=False,
            reason="value change within tolerance",
        )


# ---------------------------------------------------------------------------
# Training helpers (pure Python, no external deps)
# ---------------------------------------------------------------------------

def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _bce_grad(pred: float, target: float) -> float:
    """Gradient of binary cross-entropy w.r.t. the pre-sigmoid logit."""
    return pred - target


def _train_epoch(
    model: PositionValue,
    samples: List[Tuple[List[float], float]],
    lr: float,
) -> float:
    """One epoch of online SGD on BCE loss.  Returns mean loss."""
    total_loss = 0.0
    for features, target in samples:
        z = model.bias + sum(w * f for w, f in zip(model.weights, features))
        pred = _sigmoid(z)
        # BCE loss (clamped for numerical safety)
        p_clamped = max(1e-7, min(1.0 - 1e-7, pred))
        loss = -(target * math.log(p_clamped) + (1.0 - target) * math.log(1.0 - p_clamped))
        total_loss += loss

        grad_z = _bce_grad(pred, target)
        model.bias -= lr * grad_z
        for i in range(len(model.weights)):
            model.weights[i] -= lr * grad_z * features[i]

    return total_loss / max(1, len(samples))


# ---------------------------------------------------------------------------
# Replay feature extraction for training
# ---------------------------------------------------------------------------

def _extract_training_samples(
    replay_paths: Sequence[str | Path],
    player_name: str,
    *,
    sample_every: int = 10,
    max_step: int = 400,
) -> List[Tuple[List[float], float]]:
    """Extract (features, target) pairs from replay JSONs.

    Each replay is a Kaggle environment replay with ``steps``, ``rewards``,
    and ``info.TeamNames`` (or ``info.Agents``).

    Target is the final reward normalised to [0, 1]:
    ``target = max(0.0, min(1.0, final_reward))``
    """
    samples: List[Tuple[List[float], float]] = []

    for rp in replay_paths:
        path = Path(rp)
        if not path.is_file():
            continue
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        player_index = _find_player_index(replay, player_name)
        if player_index is None:
            continue

        rewards = replay.get("rewards") or []
        if player_index >= len(rewards) or rewards[player_index] is None:
            continue
        final_reward = float(rewards[player_index])
        target = max(0.0, min(1.0, final_reward))

        steps = replay.get("steps") or []
        num_agents = len(replay.get("info", {}).get("TeamNames", [])) or len(steps[0]) if steps else 2
        total_steps = len(steps)

        for step_idx in range(0, total_steps, sample_every):
            if step_idx > max_step:
                break
            step_data = steps[step_idx]
            if player_index >= len(step_data):
                continue
            obs = step_data[player_index].get("observation") if isinstance(step_data[player_index], dict) else None
            if obs is None:
                continue

            feats = _features_from_observation(obs, player_index, num_agents, step_idx, total_steps)
            if feats is not None:
                samples.append((feats, target))

    return samples


def _find_player_index(replay: Any, player_name: str) -> Optional[int]:
    """Resolve player name to index in a replay.

    Supports Kaggle environment replays (dict with ``info.TeamNames``) and
    plain step-list replays (returns None so caller can skip or default).
    """
    if not isinstance(replay, dict):
        return None
    needle = player_name.strip().lower()
    team_names = replay.get("info", {}).get("TeamNames") or []
    for idx, name in enumerate(team_names):
        if needle in str(name).lower():
            return idx
    agents = replay.get("info", {}).get("Agents") or []
    for idx, agent in enumerate(agents):
        if needle in str(agent.get("Name", "")).lower():
            return idx
    return None


def _features_from_observation(
    obs: Dict[str, Any],
    player_index: int,
    num_agents: int,
    step_idx: int,
    total_steps: int,
) -> Optional[List[float]]:
    """Build a feature vector from a raw observation dict."""
    raw_planets = obs.get("planets") or []
    raw_fleets = obs.get("fleets") or []

    if not raw_planets:
        return None

    total_planets = len(raw_planets)
    my_planet_count = 0
    enemy_planet_list: List[Tuple[float, float, float]] = []
    my_planet_list: List[Tuple[float, float, float]] = []
    my_ships = 0.0
    total_ships = 0.0
    my_prod = 0.0
    total_prod = 0.0

    for p in raw_planets:
        owner = int(p[1])
        ships = float(p[5])
        prod = float(p[6])
        x, y = float(p[2]), float(p[3])
        total_ships += ships
        total_prod += prod
        if owner == player_index:
            my_ships += ships
            my_prod += prod
            my_planet_count += 1
            my_planet_list.append((x, y, ships))
        elif owner != -1:
            enemy_planet_list.append((x, y, ships))

    for f in raw_fleets:
        owner = int(f[1])
        ships = float(f[6])
        total_ships += ships
        if owner == player_index:
            my_ships += ships

    total_ships = max(1.0, total_ships)
    total_prod = max(1.0, total_prod)

    # Frontline ratio from raw coords
    interaction_dist = 30.0
    my_front = 0.0
    for mx, my_, ms in my_planet_list:
        for ex, ey, _ in enemy_planet_list:
            if math.sqrt((mx - ex) ** 2 + (my_ - ey) ** 2) <= interaction_dist:
                my_front += ms
                break
    enemy_front = 0.0
    for ex, ey, es in enemy_planet_list:
        for mx, my_, _ in my_planet_list:
            if math.sqrt((ex - mx) ** 2 + (ey - my_) ** 2) <= interaction_dist:
                enemy_front += es
                break

    return [
        my_ships / total_ships,
        my_prod / total_prod,
        step_idx / max(1, total_steps),
        1.0 if num_agents == 2 else 0.0,
        my_planet_count / max(1, total_planets),
        my_front / max(1.0, enemy_front),
    ]


# ---------------------------------------------------------------------------
# CLI training entry point
# ---------------------------------------------------------------------------

def train_value_model(
    replay_paths: Sequence[str | Path],
    player_name: str,
    output_path: str | Path = "value_model.json",
    *,
    lr: float = 0.01,
    epochs: int = 50,
    sample_every: int = 10,
    max_step: int = 400,
    verbose: bool = False,
) -> PositionValue:
    """Train a PositionValue model from replay files and save to JSON.

    Importable so it can be called from other scripts.
    """
    model = PositionValue.train_from_replays(
        replay_paths,
        player_name,
        lr=lr,
        epochs=epochs,
        sample_every=sample_every,
        max_step=max_step,
    )
    out = Path(output_path)
    model.save(out)
    if verbose:
        print(f"Saved value model to {out}  (weights={model.weights}, bias={model.bias:.4f})")
    return model


def main() -> None:
    """CLI entry point for standalone training."""
    parser = argparse.ArgumentParser(
        description="Train a tactical veto value model from Orbit Wars replays.",
    )
    parser.add_argument(
        "replays",
        nargs="+",
        help="Paths to replay JSON files (or glob patterns).",
    )
    parser.add_argument(
        "--player",
        required=True,
        help="Player name to train from (substring match).",
    )
    parser.add_argument(
        "--output",
        default="value_model.json",
        help="Output path for the trained model JSON (default: value_model.json).",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--sample-every", type=int, default=10, help="Sample every N steps from each replay.")
    parser.add_argument("--max-step", type=int, default=400, help="Max game step to sample from.")
    parser.add_argument("--verbose", action="store_true", help="Print progress.")

    args = parser.parse_args()

    # Expand glob patterns
    import glob as glob_mod
    paths: List[str] = []
    for pattern in args.replays:
        expanded = glob_mod.glob(pattern)
        paths.extend(expanded if expanded else [pattern])

    if not paths:
        parser.error("No replay files found.")

    if args.verbose:
        print(f"Training from {len(paths)} replay file(s) for player '{args.player}'")

    train_value_model(
        paths,
        args.player,
        output_path=args.output,
        lr=args.lr,
        epochs=args.epochs,
        sample_every=args.sample_every,
        max_step=args.max_step,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
