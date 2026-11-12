from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


ACTION_SETS: Dict[str, List[str]] = {
    "two_player": [
        "bank",
        "expand_safe",
        "expand_rich",
        "pressure_front",
        "reinforce_front",
        "finish_weakest",
    ],
    "four_player": [
        "bank",
        "expand_safe",
        "expand_rich",
        "pressure_front",
        "reinforce_front",
        "deny_leader",
        "finish_weakest",
    ],
}

FRACTION_BUCKETS: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0]
MAX_PLANETS = len(FOUR_PLAYER_TEMPLATE.planets) if "FOUR_PLAYER_TEMPLATE" in globals() else 10


@dataclass(frozen=True)
class PlanetTemplate:
    id: int
    x: float
    y: float
    production: int
    ships: int
    owner: int


@dataclass
class PlanetState:
    id: int
    x: float
    y: float
    production: int
    ships: int
    owner: int


@dataclass
class FleetState:
    owner: int
    source_id: int
    target_id: int
    ships: int
    eta: int


@dataclass
class LaunchOrder:
    owner: int
    source_id: int
    target_id: int
    ships: int
    eta: int


@dataclass(frozen=True)
class BoardTemplate:
    name: str
    mode: str
    num_players: int
    max_steps: int
    planets: Sequence[PlanetTemplate]


@dataclass
class StepResult:
    rewards: Dict[int, float]
    done: bool
    winner: Optional[int]
    launched: Dict[int, Optional[LaunchOrder]]


TWO_PLAYER_TEMPLATE = BoardTemplate(
    name="two_player_core",
    mode="two_player",
    num_players=2,
    max_steps=60,
    planets=[
        PlanetTemplate(0, 0.0, 0.0, 4, 40, 0),
        PlanetTemplate(1, 18.0, 12.0, 2, 12, -1),
        PlanetTemplate(2, 36.0, 18.0, 3, 18, -1),
        PlanetTemplate(3, 50.0, 0.0, 5, 24, -1),
        PlanetTemplate(4, 64.0, -18.0, 3, 18, -1),
        PlanetTemplate(5, 82.0, -12.0, 2, 12, -1),
        PlanetTemplate(6, 100.0, 0.0, 4, 40, 1),
    ],
)

FOUR_PLAYER_TEMPLATE = BoardTemplate(
    name="four_player_core",
    mode="four_player",
    num_players=4,
    max_steps=70,
    planets=[
        PlanetTemplate(0, 0.0, 0.0, 4, 38, 0),
        PlanetTemplate(1, 100.0, 0.0, 4, 38, 1),
        PlanetTemplate(2, 100.0, 100.0, 4, 38, 2),
        PlanetTemplate(3, 0.0, 100.0, 4, 38, 3),
        PlanetTemplate(4, 22.0, 16.0, 2, 12, -1),
        PlanetTemplate(5, 84.0, 22.0, 2, 12, -1),
        PlanetTemplate(6, 78.0, 84.0, 2, 12, -1),
        PlanetTemplate(7, 16.0, 78.0, 2, 12, -1),
        PlanetTemplate(8, 50.0, 35.0, 4, 20, -1),
        PlanetTemplate(9, 50.0, 65.0, 4, 20, -1),
    ],
)


def _bin_value(value: float, thresholds: Sequence[float]) -> int:
    for index, threshold in enumerate(thresholds):
        if value < threshold:
            return index
    return len(thresholds)


class SimplifiedPlanetEnv:
    """Abstract planet-control environment: production, timing, pressure, no moving geometry."""

    def __init__(self, mode: str, seed: int = 7) -> None:
        if mode not in ACTION_SETS:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.template = TWO_PLAYER_TEMPLATE if mode == "two_player" else FOUR_PLAYER_TEMPLATE
        self.rng = random.Random(seed)
        self.step_count = 0
        self.planets: List[PlanetState] = []
        self.fleets: List[FleetState] = []
        self.distance_matrix = self._build_distance_matrix(self.template.planets)
        self.reset()

    @property
    def num_players(self) -> int:
        return self.template.num_players

    @property
    def max_steps(self) -> int:
        return self.template.max_steps

    def action_names(self) -> List[str]:
        return list(ACTION_SETS[self.mode])

    @property
    def max_planets(self) -> int:
        return len(self.template.planets)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng.seed(seed)
        self.step_count = 0
        self.fleets = []
        self.planets = []
        for spec in self.template.planets:
            ships = spec.ships
            if spec.owner == -1:
                ships += self.rng.randint(-2, 2)
                ships = max(6, ships)
            else:
                ships += self.rng.randint(-4, 4)
                ships = max(24, ships)
            self.planets.append(
                PlanetState(
                    id=spec.id,
                    x=spec.x,
                    y=spec.y,
                    production=spec.production,
                    ships=ships,
                    owner=spec.owner,
                )
            )

    def encode_state(self, player: int) -> tuple[int, ...]:
        summary = self.summary(player)
        phase = 0 if self.step_count < self.max_steps * 0.25 else (1 if self.step_count < self.max_steps * 0.65 else 2)
        if self.mode == "two_player":
            return (
                phase,
                _bin_value(summary["prod_gap"], (-4, -1, 2, 5)),
                _bin_value(summary["ship_gap"], (-50, -10, 15, 60)),
                min(3, summary["safe_neutral_count"]),
                min(2, summary["rich_neutral_count"]),
                min(2, summary["threatened_count"]),
                1 if summary["finish_window"] else 0,
                _bin_value(summary["idle_surplus"], (10, 24, 40)),
            )
        return (
            phase,
            max(0, min(3, summary["my_rank"] - 1)),
            _bin_value(summary["leader_gap"], (-80, -25, 5, 35)),
            _bin_value(summary["prod_gap"], (-6, -2, 1, 5)),
            min(3, summary["safe_neutral_count"]),
            min(2, summary["rich_neutral_count"]),
            min(2, summary["threatened_count"]),
            1 if summary["leader_runaway"] else 0,
            1 if summary["weakest_vulnerable"] else 0,
        )

    def heuristic_action(self, player: int) -> str:
        summary = self.summary(player)
        if summary["threatened_count"] > 0:
            return "reinforce_front"
        if self.mode == "four_player" and summary["leader_runaway"] and summary["leader_owner"] != player:
            return "deny_leader"
        if summary["finish_window"]:
            return "finish_weakest"
        if self.step_count < self.max_steps * 0.35 and summary["rich_neutral_count"] > 0:
            return "expand_rich"
        if summary["safe_neutral_count"] > 0:
            return "expand_safe"
        if self.mode == "four_player" and summary["weakest_vulnerable"]:
            return "finish_weakest"
        if summary["enemy_planet_count"] > 0:
            return "pressure_front"
        return "bank"

    def step(self, actions_by_player: Dict[int, str]) -> StepResult:
        before = {player: self.summary(player) for player in range(self.num_players)}
        self._apply_production()
        capture_counts = self._advance_and_resolve_fleets()

        launched: Dict[int, Optional[LaunchOrder]] = {}
        for player in range(self.num_players):
            if not self._player_alive(player):
                launched[player] = None
                continue
            action = actions_by_player.get(player, "bank")
            order = self._plan_action(player, action)
            launched[player] = order
            if order is not None:
                self._apply_launch(order)

        self.step_count += 1
        done = self._is_terminal()
        winner = self._winner() if done else None
        after = {player: self.summary(player) for player in range(self.num_players)}

        rewards: Dict[int, float] = {}
        for player in range(self.num_players):
            rewards[player] = self._dense_reward(player, before[player], after[player], capture_counts.get(player, 0))
        if done:
            rewards.update(self._terminal_rewards())
        return StepResult(rewards=rewards, done=done, winner=winner, launched=launched)

    def step_fraction_action(
        self,
        player: int,
        source_id: int,
        target_id: int,
        fraction: float,
        opponent_mode: str = "heuristic",
    ) -> StepResult:
        return self.step_fraction_actions(
            {player: (source_id, target_id, fraction)},
            default_opponent_mode=opponent_mode,
        )

    def step_fraction_actions(
        self,
        actions_by_player: Dict[int, tuple[int | None, int | None, float]],
        default_opponent_mode: str = "heuristic",
    ) -> StepResult:
        before = {owner: self.summary(owner) for owner in range(self.num_players)}
        self._apply_production()
        capture_counts = self._advance_and_resolve_fleets()

        launched: Dict[int, Optional[LaunchOrder]] = {}
        for owner in range(self.num_players):
            if not self._player_alive(owner):
                launched[owner] = None
                continue
            chosen = actions_by_player.get(owner)
            if chosen is not None:
                source_id, target_id, fraction = chosen
                if source_id is None or target_id is None:
                    order = None
                else:
                    order = self._plan_fraction_launch(owner, source_id, target_id, fraction)
            else:
                order = self._plan_opponent_action(owner, default_opponent_mode)
            launched[owner] = order
            if order is not None:
                self._apply_launch(order)

        self.step_count += 1
        done = self._is_terminal()
        winner = self._winner() if done else None
        after = {owner: self.summary(owner) for owner in range(self.num_players)}

        rewards: Dict[int, float] = {}
        for owner in range(self.num_players):
            rewards[owner] = self._dense_reward(owner, before[owner], after[owner], capture_counts.get(owner, 0))
        if done:
            rewards.update(self._terminal_rewards())
        return StepResult(rewards=rewards, done=done, winner=winner, launched=launched)

    def build_gcn_observation(self, player: int) -> Dict[str, object]:
        num_slots = len(self.template.planets)
        node_features: List[List[float]] = []
        positions: List[List[float]] = []
        velocities: List[List[float]] = []
        valid_mask: List[float] = []

        summary = self.summary(player)
        alive_owners = [owner for owner in range(self.num_players) if self._player_alive(owner)]
        totals = self._player_totals()
        prods = self._player_productions()

        for slot in range(num_slots):
            planet = self.planets[slot]
            nearest_enemy_eta = self._nearest_owner_eta(planet.id, excluded={-1, planet.owner}) or 10
            nearest_me_eta = self._nearest_owner_eta(planet.id, excluded={-1, player}) or 10
            inbound_friend = sum(
                fleet.ships for fleet in self.fleets if fleet.target_id == planet.id and fleet.owner == player
            )
            inbound_enemy = sum(
                fleet.ships for fleet in self.fleets if fleet.target_id == planet.id and fleet.owner not in (-1, player)
            )
            surplus = self._planet_surplus(player, planet.id)
            threat = self._planet_threat(player, planet.id) if planet.owner == player else 0
            owner_friend = 1.0 if planet.owner == player else 0.0
            owner_neutral = 1.0 if planet.owner == -1 else 0.0
            owner_enemy = 1.0 if planet.owner not in (-1, player) else 0.0
            owner_leader = 1.0 if planet.owner == int(summary["leader_owner"]) and planet.owner != player else 0.0

            node_features.append(
                [
                    owner_friend,
                    owner_enemy,
                    owner_neutral,
                    owner_leader,
                    float(planet.production) / 5.0,
                    float(planet.ships) / 80.0,
                    float(surplus) / 60.0,
                    float(threat) / 40.0,
                    float(inbound_friend) / 60.0,
                    float(inbound_enemy) / 60.0,
                    float(nearest_enemy_eta) / 10.0,
                    float(nearest_me_eta) / 10.0,
                ]
            )
            positions.append([planet.x / 100.0, planet.y / 100.0])
            velocities.append(
                [
                    math.cos((2.0 * math.pi * slot) / max(1, num_slots)) * (0.04 if planet.owner == -1 else 0.02),
                    math.sin((2.0 * math.pi * slot) / max(1, num_slots)) * (0.04 if planet.owner == -1 else 0.02),
                ]
            )
            valid_mask.append(1.0)

        global_features = [
            float(summary["my_production"]) / 20.0,
            float(summary["best_enemy_production"]) / 20.0,
            float(summary["my_total"]) / 200.0,
            float(summary["best_enemy_total"]) / 200.0,
            float(self.step_count) / float(self.max_steps),
            float(summary["my_rank"]) / max(1.0, float(len(alive_owners))),
        ]

        action_mask = self.valid_fraction_action_mask(player)
        return {
            "node_features": node_features,
            "global_features": global_features,
            "positions": positions,
            "velocities": velocities,
            "valid_mask": valid_mask,
            "action_mask": action_mask,
        }

    def valid_fraction_action_mask(self, player: int) -> List[List[List[bool]]]:
        num_slots = len(self.template.planets)
        mask = [[[False for _ in FRACTION_BUCKETS] for _ in range(num_slots)] for _ in range(num_slots)]
        for source_id in range(num_slots):
            planet = self.planets[source_id]
            if planet.owner != player:
                continue
            surplus = self._planet_surplus(player, source_id)
            if surplus <= 0:
                continue
            for target_id in range(num_slots):
                if source_id == target_id:
                    continue
                for fraction_index, fraction in enumerate(FRACTION_BUCKETS):
                    send = int(max(0, round(surplus * fraction)))
                    if send <= 0:
                        continue
                    mask[source_id][target_id][fraction_index] = True
        return mask

    def summary(self, player: int) -> Dict[str, float | int | bool]:
        totals = self._player_totals()
        prods = self._player_productions()
        alive = [p for p in range(self.num_players) if self._player_alive(p)]
        my_total = totals[player]
        enemy_totals = [totals[p] for p in alive if p != player] or [0]
        enemy_prods = [prods[p] for p in alive if p != player] or [0]
        leader_owner = max(alive, key=lambda owner: (totals[owner], prods[owner])) if alive else player
        weakest_enemy_owner = None
        enemy_owners = [owner for owner in alive if owner != player]
        if enemy_owners:
            weakest_enemy_owner = min(enemy_owners, key=lambda owner: (totals[owner], prods[owner]))

        my_planets = [planet for planet in self.planets if planet.owner == player]
        enemy_planets = [planet for planet in self.planets if planet.owner not in (-1, player)]
        safe_neutrals = [planet for planet in self.planets if self._is_safe_neutral(player, planet)]
        rich_neutrals = [planet for planet in safe_neutrals if planet.production >= (4 if self.mode == "two_player" else 3)]

        ranked = sorted(alive, key=lambda owner: (totals[owner], prods[owner]), reverse=True)
        my_rank = 1 + sum(1 for owner in ranked if totals[owner] > my_total)

        leader_total = totals[leader_owner]
        best_enemy_total = max(enemy_totals) if enemy_totals else 0
        finish_window = (
            bool(enemy_owners)
            and my_total > best_enemy_total * (1.25 if self.mode == "two_player" else 1.15)
            and len(enemy_planets) <= (2 if self.mode == "two_player" else 3)
        )
        weakest_enemy_total = totals[weakest_enemy_owner] if weakest_enemy_owner is not None else 0
        leader_runaway = (
            leader_owner != player
            and leader_total > max(1, my_total) * 1.18
            and prods[leader_owner] > max(1, prods[player]) + 1
        )

        return {
            "my_total": my_total,
            "best_enemy_total": best_enemy_total,
            "leader_total": leader_total,
            "my_production": prods[player],
            "best_enemy_production": max(enemy_prods) if enemy_prods else 0,
            "prod_gap": prods[player] - (max(enemy_prods) if enemy_prods else 0),
            "ship_gap": my_total - best_enemy_total,
            "leader_gap": my_total - leader_total,
            "leader_owner": leader_owner,
            "my_rank": my_rank,
            "planet_count": len(my_planets),
            "enemy_planet_count": len(enemy_planets),
            "safe_neutral_count": len(safe_neutrals),
            "rich_neutral_count": len(rich_neutrals),
            "threatened_count": sum(1 for planet in my_planets if self._planet_threat(player, planet.id) > 0),
            "idle_surplus": sum(self._planet_surplus(player, planet.id) for planet in my_planets),
            "finish_window": finish_window,
            "leader_runaway": leader_runaway,
            "weakest_enemy_owner": weakest_enemy_owner if weakest_enemy_owner is not None else -1,
            "weakest_vulnerable": weakest_enemy_owner is not None and weakest_enemy_total < max(22, leader_total * 0.72),
        }

    def _dense_reward(
        self,
        player: int,
        before: Dict[str, float | int | bool],
        after: Dict[str, float | int | bool],
        captures: int,
    ) -> float:
        reward = 0.0
        reward += 0.05 * (int(after["my_production"]) - int(before["my_production"]))
        reward += 0.012 * (int(after["my_total"]) - int(before["my_total"]))
        reward -= 0.01 * max(0, int(after["best_enemy_total"]) - int(before["best_enemy_total"]))
        reward += 0.08 * captures
        reward -= 0.03 * max(0, int(after["threatened_count"]) - int(before["threatened_count"]))
        reward -= 0.0015 * max(0, int(after["idle_surplus"]) - int(before["idle_surplus"]))
        if self.mode == "four_player":
            reward += 0.03 * (int(before["my_rank"]) - int(after["my_rank"]))
            if bool(after["leader_runaway"]) and int(after["leader_owner"]) != player:
                reward -= 0.04
        return reward

    def _terminal_rewards(self) -> Dict[int, float]:
        winner = self._winner()
        totals = self._player_totals()
        prods = self._player_productions()
        ranking = sorted(range(self.num_players), key=lambda owner: (totals[owner], prods[owner]), reverse=True)
        rewards: Dict[int, float] = {}
        if self.mode == "two_player":
            for player in range(self.num_players):
                rewards[player] = 1.0 if player == winner else -1.0
            return rewards

        rank_reward = {0: 1.0, 1: 0.25, 2: -0.25, 3: -1.0}
        for index, player in enumerate(ranking):
            rewards[player] = rank_reward.get(index, -1.0)
        return rewards

    def _apply_production(self) -> None:
        for planet in self.planets:
            if planet.owner >= 0:
                planet.ships += planet.production

    def _advance_and_resolve_fleets(self) -> Dict[int, int]:
        for fleet in self.fleets:
            fleet.eta -= 1
        due = [fleet for fleet in self.fleets if fleet.eta <= 0]
        self.fleets = [fleet for fleet in self.fleets if fleet.eta > 0]

        grouped: Dict[int, List[FleetState]] = defaultdict(list)
        for fleet in due:
            grouped[fleet.target_id].append(fleet)

        captures: Dict[int, int] = defaultdict(int)
        for target_id, incoming in grouped.items():
            planet = self.planets[target_id]
            force_by_owner: Dict[int, int] = defaultdict(int)
            force_by_owner[planet.owner] += planet.ships
            for fleet in incoming:
                force_by_owner[fleet.owner] += fleet.ships

            ranked = sorted(force_by_owner.items(), key=lambda item: item[1], reverse=True)
            best_owner, best_force = ranked[0]
            second_force = ranked[1][1] if len(ranked) > 1 else 0
            if len(ranked) > 1 and best_force == second_force:
                tied = {owner for owner, ships in ranked if ships == best_force}
                if planet.owner in tied:
                    winner = planet.owner
                else:
                    winner = -1
                remaining = 0
            else:
                winner = best_owner
                remaining = best_force - second_force

            previous_owner = planet.owner
            planet.owner = winner
            planet.ships = max(0, remaining)
            if winner >= 0 and winner != previous_owner:
                captures[winner] += 1
        return captures

    def _apply_launch(self, order: LaunchOrder) -> None:
        source = self.planets[order.source_id]
        source.ships -= order.ships
        self.fleets.append(
            FleetState(
                owner=order.owner,
                source_id=order.source_id,
                target_id=order.target_id,
                ships=order.ships,
                eta=order.eta,
            )
        )

    def _plan_action(self, player: int, action: str) -> Optional[LaunchOrder]:
        if action == "bank":
            return None
        if not self._player_alive(player):
            return None
        if action == "reinforce_front":
            return self._plan_reinforcement(player)
        if action == "expand_safe":
            return self._plan_neutral_capture(player, rich_bias=False)
        if action == "expand_rich":
            order = self._plan_neutral_capture(player, rich_bias=True)
            return order or self._plan_neutral_capture(player, rich_bias=False)
        if action == "pressure_front":
            return self._plan_enemy_attack(player, focus_owner=None, value_bonus=1.0)
        if action == "deny_leader":
            leader_owner = int(self.summary(player)["leader_owner"])
            if leader_owner == player:
                return self._plan_enemy_attack(player, focus_owner=None, value_bonus=1.0)
            return self._plan_enemy_attack(player, focus_owner=leader_owner, value_bonus=1.2)
        if action == "finish_weakest":
            weakest_owner = int(self.summary(player)["weakest_enemy_owner"])
            if weakest_owner < 0:
                return self._plan_enemy_attack(player, focus_owner=None, value_bonus=1.0)
            return self._plan_enemy_attack(player, focus_owner=weakest_owner, value_bonus=1.25)
        return None

    def _plan_opponent_action(self, player: int, opponent_mode: str) -> Optional[LaunchOrder]:
        if opponent_mode == "heuristic":
            return self._plan_action(player, self.heuristic_action(player))
        return self._plan_action(player, self.heuristic_action(player))

    def _plan_fraction_launch(self, player: int, source_id: int, target_id: int, fraction: float) -> Optional[LaunchOrder]:
        if not self._player_alive(player):
            return None
        if source_id < 0 or source_id >= len(self.planets) or target_id < 0 or target_id >= len(self.planets):
            return None
        if source_id == target_id:
            return None
        source = self.planets[source_id]
        if source.owner != player:
            return None
        surplus = self._planet_surplus(player, source_id)
        if surplus <= 0:
            return None
        send = int(max(0, round(surplus * max(0.0, min(1.0, fraction)))))
        if send <= 0:
            return None
        eta = self.distance_matrix[source_id][target_id]
        target = self.planets[target_id]
        if target.owner == -1:
            needed = target.ships + 2
            if send < needed:
                return None
        elif target.owner != player:
            needed = self._estimate_attack_need(target, eta)
            if send < max(needed // 2, 4):
                return None
        return LaunchOrder(player, source_id, target_id, send, eta)

    def heuristic_fraction_action(self, player: int) -> tuple[int | None, int | None, int | None]:
        action = self.heuristic_action(player)
        order = self._plan_action(player, action)
        if order is None:
            return None, None, None
        source = self.planets[order.source_id]
        surplus = self._planet_surplus(player, order.source_id)
        if source.owner != player or surplus <= 0:
            return None, None, None
        ratio = order.ships / max(1, surplus)
        fraction_index = min(range(len(FRACTION_BUCKETS)), key=lambda idx: abs(FRACTION_BUCKETS[idx] - ratio))
        return order.source_id, order.target_id, fraction_index

    def _plan_neutral_capture(self, player: int, rich_bias: bool) -> Optional[LaunchOrder]:
        candidates = [planet for planet in self.planets if planet.owner == -1]
        best: Optional[tuple[float, LaunchOrder]] = None
        for target in candidates:
            source_id, surplus, eta = self._best_source_for_target(player, target.id)
            if source_id is None or surplus <= 0:
                continue
            enemy_eta = self._nearest_owner_eta(target.id, excluded={-1, player})
            if enemy_eta is not None and eta > enemy_eta + (1 if rich_bias else 0):
                continue
            needed = target.ships + 2
            if surplus < needed:
                continue
            send = min(surplus, needed + max(1, target.production // 2))
            score = (target.production * (8.0 if rich_bias else 6.0)) - needed - 1.4 * eta
            if rich_bias:
                score += 1.5 * target.production
            order = LaunchOrder(player, source_id, target.id, send, eta)
            if best is None or score > best[0]:
                best = (score, order)
        return best[1] if best is not None else None

    def _plan_enemy_attack(self, player: int, focus_owner: Optional[int], value_bonus: float) -> Optional[LaunchOrder]:
        candidates = [planet for planet in self.planets if planet.owner not in (-1, player)]
        best: Optional[tuple[float, LaunchOrder]] = None
        for target in candidates:
            if focus_owner is not None and target.owner != focus_owner:
                continue
            source_id, surplus, eta = self._best_source_for_target(player, target.id)
            if source_id is None or surplus <= 0:
                continue
            needed = self._estimate_attack_need(target, eta)
            if surplus < needed:
                continue
            send = min(surplus, needed + 2 + target.production // 2)
            exposed = 1.4 if target.ships <= target.production * 4 else 1.0
            owner_bonus = value_bonus * (1.2 if focus_owner is not None else 1.0)
            score = owner_bonus * exposed * (target.production * 10.0) - needed - 1.6 * eta
            order = LaunchOrder(player, source_id, target.id, send, eta)
            if best is None or score > best[0]:
                best = (score, order)
        return best[1] if best is not None else None

    def _plan_reinforcement(self, player: int) -> Optional[LaunchOrder]:
        owned = [planet for planet in self.planets if planet.owner == player]
        threatened = sorted(
            ((self._planet_threat(player, planet.id), planet) for planet in owned),
            key=lambda item: item[0],
            reverse=True,
        )
        threatened = [item for item in threatened if item[0] > 0]
        if not threatened:
            return None
        target = threatened[0][1]
        source_candidates = []
        for planet in owned:
            if planet.id == target.id:
                continue
            surplus = self._planet_surplus(player, planet.id)
            if surplus <= 3:
                continue
            eta = self.distance_matrix[planet.id][target.id]
            safety = self._nearest_owner_eta(planet.id, excluded={-1, player}) or 10
            source_candidates.append((surplus + safety - eta, planet.id, surplus, eta))
        if not source_candidates:
            return None
        _score, source_id, surplus, eta = max(source_candidates)
        send = min(surplus, max(4, int(self._planet_threat(player, target.id) + 2)))
        return LaunchOrder(player, source_id, target.id, send, eta)

    def _best_source_for_target(self, player: int, target_id: int) -> tuple[Optional[int], int, int]:
        candidates = []
        for planet in self.planets:
            if planet.owner != player:
                continue
            surplus = self._planet_surplus(player, planet.id)
            if surplus <= 0:
                continue
            eta = self.distance_matrix[planet.id][target_id]
            candidates.append((surplus - eta, planet.id, surplus, eta))
        if not candidates:
            return None, 0, 0
        _score, source_id, surplus, eta = max(candidates)
        return source_id, surplus, eta

    def _estimate_attack_need(self, target: PlanetState, eta: int) -> int:
        if target.owner == -1:
            return target.ships + 2
        inbound = sum(
            fleet.ships
            for fleet in self.fleets
            if fleet.target_id == target.id and fleet.owner == target.owner and fleet.eta <= eta
        )
        return target.ships + eta * target.production + inbound + 2

    def _planet_surplus(self, player: int, planet_id: int) -> int:
        planet = self.planets[planet_id]
        if planet.owner != player:
            return 0
        nearest_enemy = self._nearest_owner_eta(planet_id, excluded={-1, player}) or 10
        reserve = 6 + 2 * planet.production
        if nearest_enemy <= 3:
            reserve += 8
        elif nearest_enemy <= 5:
            reserve += 4
        if self.mode == "four_player":
            reserve += 2
        incoming_enemy = sum(
            fleet.ships for fleet in self.fleets if fleet.target_id == planet_id and fleet.owner not in (-1, player)
        )
        incoming_friend = sum(
            fleet.ships for fleet in self.fleets if fleet.target_id == planet_id and fleet.owner == player
        )
        reserve += max(0, incoming_enemy - incoming_friend) // 2
        return max(0, planet.ships - reserve)

    def _planet_threat(self, player: int, planet_id: int) -> int:
        planet = self.planets[planet_id]
        enemy_near = 0
        for other in self.planets:
            if other.owner in (-1, player):
                continue
            eta = self.distance_matrix[other.id][planet_id]
            if eta <= 4:
                enemy_near += max(0, other.ships - 6) // max(1, eta)
        inbound_enemy = sum(
            fleet.ships for fleet in self.fleets if fleet.target_id == planet_id and fleet.owner not in (-1, player)
        )
        inbound_friend = sum(
            fleet.ships for fleet in self.fleets if fleet.target_id == planet_id and fleet.owner == player
        )
        return max(0, enemy_near + inbound_enemy - inbound_friend - planet.ships)

    def _is_safe_neutral(self, player: int, planet: PlanetState) -> bool:
        if planet.owner != -1:
            return False
        my_eta = self._nearest_owner_eta(planet.id, excluded={-1})
        enemy_eta = self._nearest_owner_eta(planet.id, excluded={-1, player})
        if my_eta is None:
            return False
        if enemy_eta is None:
            return True
        return my_eta <= enemy_eta

    def _nearest_owner_eta(self, target_id: int, excluded: Iterable[int]) -> Optional[int]:
        excluded_set = set(excluded)
        etas = [
            self.distance_matrix[planet.id][target_id]
            for planet in self.planets
            if planet.owner not in excluded_set and planet.id != target_id
        ]
        return min(etas) if etas else None

    def _player_totals(self) -> Dict[int, int]:
        totals = {player: 0 for player in range(self.num_players)}
        for planet in self.planets:
            if planet.owner >= 0:
                totals[planet.owner] += planet.ships
        for fleet in self.fleets:
            totals[fleet.owner] += fleet.ships
        return totals

    def _player_productions(self) -> Dict[int, int]:
        prods = {player: 0 for player in range(self.num_players)}
        for planet in self.planets:
            if planet.owner >= 0:
                prods[planet.owner] += planet.production
        return prods

    def _player_alive(self, player: int) -> bool:
        return any(planet.owner == player for planet in self.planets) or any(fleet.owner == player for fleet in self.fleets)

    def _is_terminal(self) -> bool:
        alive = [player for player in range(self.num_players) if self._player_alive(player)]
        return self.step_count >= self.max_steps or len(alive) <= 1

    def _winner(self) -> Optional[int]:
        totals = self._player_totals()
        prods = self._player_productions()
        alive = [player for player in range(self.num_players) if self._player_alive(player)]
        if not alive:
            return None
        return max(alive, key=lambda player: (totals[player], prods[player]))

    @staticmethod
    def _build_distance_matrix(planets: Sequence[PlanetTemplate]) -> List[List[int]]:
        matrix: List[List[int]] = []
        for source in planets:
            row = []
            for target in planets:
                if source.id == target.id:
                    row.append(0)
                    continue
                distance = math.dist((source.x, source.y), (target.x, target.y))
                row.append(max(1, int(round(distance / 18.0))))
            matrix.append(row)
        return matrix
