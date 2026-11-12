from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

WORKSPACE_DIR = Path(__file__).resolve().parent
ROOT = WORKSPACE_DIR.parent


def resolve_base_agent_path() -> Path:
    override = os.environ.get("ORBIT_WARS_BASE_AGENT_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return ROOT / "snapshots" / "stage4_leaderboard_search_base.py"


BASE_AGENT_PATH = resolve_base_agent_path()
_BASE_MODULE_NAME = "_orbit_wars_genome_base"


def _load_base_module() -> Any:
    if _BASE_MODULE_NAME in sys.modules:
        return sys.modules[_BASE_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, BASE_AGENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base agent from {BASE_AGENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_BASE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()

Planet = BASE.Planet
MissionOption = BASE.MissionOption
PlannedMove = BASE.PlannedMove
distance_planets = BASE.distance_planets
is_orbiting_planet = BASE.is_orbiting_planet

GENE_SPACE: Dict[str, Tuple[str, ...]] = {
    "style_profile": ("balanced", "aggressive", "conservative"),
    "duel_opening": ("v23", "mtmr", "shun"),
    "duel_filter": ("v23", "baseline_rotating"),
    "duel_attack_order": ("v23", "local_pressure", "production_first"),
    "duel_launch_cap": ("v23", "mtmr", "relaxed"),
    "value_profile": ("balanced", "economy", "hostile", "finisher"),
    "followup_profile": ("low", "base", "high"),
    "mode_profile": ("static", "dynamic"),
    "transition_profile": ("base", "earlier_attack", "later_attack"),
    "opening_range_profile": ("base", "local_bias", "eta_focus"),
    "threat_profile": ("v23", "leader_focus", "anti_snowball"),
    "crowd_profile": ("base", "antidogpile", "hard"),
    "position_profile": ("base", "safer_neutrals", "local_safe"),
    "vulture_profile": ("off", "windowed", "aggressive"),
    "conversion_profile": ("base", "protect", "closeout"),
    "pressure_profile": ("off", "guarded"),
    "swarm_profile": ("tight", "base", "loose"),
    "concentration_profile": ("base", "guarded", "strict"),
}

FOLLOWUP_WEIGHTS = {
    "low": 0.35,
    "base": 0.55,
    "high": 0.80,
}

SWARM_TOLERANCES = {
    "tight": 1,
    "base": BASE.SWARM_ETA_TOLERANCE,
    "loose": 3,
}

VULTURE_ETA_LIMIT = 14
VULTURE_MIN_INCOMING = 8
VULTURE_HOSTILE_BONUS = {
    "windowed": 1.16,
    "aggressive": 1.24,
}


@dataclass(frozen=True)
class GenomeConfig:
    style_profile: str = "balanced"
    duel_opening: str = "v23"
    duel_filter: str = "v23"
    duel_attack_order: str = "v23"
    duel_launch_cap: str = "v23"
    value_profile: str = "balanced"
    followup_profile: str = "base"
    mode_profile: str = "static"
    transition_profile: str = "base"
    opening_range_profile: str = "base"
    threat_profile: str = "v23"
    crowd_profile: str = "base"
    position_profile: str = "base"
    vulture_profile: str = "off"
    conversion_profile: str = "base"
    pressure_profile: str = "off"
    swarm_profile: str = "base"
    concentration_profile: str = "base"

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, str]) -> "GenomeConfig":
        return cls(**{field.name: payload[field.name] for field in fields(cls) if field.name in payload})

    def validate(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            allowed = GENE_SPACE[field.name]
            if value not in allowed:
                raise ValueError(f"Invalid gene value for {field.name}: {value!r} not in {allowed}")

    def slug(self) -> str:
        parts = [
            self.style_profile,
            self.duel_opening,
            self.duel_filter,
            self.duel_attack_order,
            self.duel_launch_cap,
            self.value_profile,
            self.followup_profile,
            self.mode_profile,
            self.transition_profile,
            self.opening_range_profile,
            self.threat_profile,
            self.crowd_profile,
            self.position_profile,
            self.vulture_profile,
            self.conversion_profile,
            self.pressure_profile,
            self.swarm_profile,
            self.concentration_profile,
        ]
        return "-".join(part.replace("_", "")[:6] for part in parts)


PRESET_GENOMES: Dict[str, GenomeConfig] = {
    "baseline_base": GenomeConfig(),
    "meta_aggressive": GenomeConfig(
        style_profile="aggressive",
        value_profile="hostile",
        followup_profile="high",
        duel_launch_cap="relaxed",
        swarm_profile="loose",
    ),
    "meta_conservative": GenomeConfig(
        style_profile="conservative",
        value_profile="economy",
        conversion_profile="protect",
        pressure_profile="guarded",
        swarm_profile="tight",
        concentration_profile="guarded",
    ),
    "mtmr_duel": GenomeConfig(
        style_profile="conservative",
        duel_opening="mtmr",
        duel_filter="baseline_rotating",
        duel_attack_order="local_pressure",
        duel_launch_cap="mtmr",
        value_profile="economy",
        transition_profile="later_attack",
        opening_range_profile="eta_focus",
        conversion_profile="protect",
        pressure_profile="guarded",
        concentration_profile="guarded",
    ),
    "todo_constant_tuning": GenomeConfig(
        style_profile="aggressive",
        value_profile="hostile",
        followup_profile="high",
    ),
    "todo_dynamic_thresholds": GenomeConfig(
        mode_profile="dynamic",
        pressure_profile="guarded",
    ),
    "todo_transition_pivot": GenomeConfig(
        transition_profile="earlier_attack",
        value_profile="hostile",
    ),
    "todo_local_opening": GenomeConfig(
        opening_range_profile="eta_focus",
        duel_attack_order="local_pressure",
        transition_profile="later_attack",
        pressure_profile="guarded",
    ),
    "todo_vulture_window": GenomeConfig(
        vulture_profile="windowed",
        threat_profile="leader_focus",
    ),
    "todo_lead_protection": GenomeConfig(
        conversion_profile="protect",
        pressure_profile="guarded",
        concentration_profile="guarded",
    ),
    "todo_endgame_closeout": GenomeConfig(
        conversion_profile="closeout",
        value_profile="finisher",
    ),
    "todo_leader_focus": GenomeConfig(
        threat_profile="leader_focus",
        swarm_profile="tight",
    ),
    "todo_anti_snowball": GenomeConfig(
        style_profile="conservative",
        threat_profile="anti_snowball",
        crowd_profile="antidogpile",
        mode_profile="dynamic",
        value_profile="finisher",
    ),
    "todo_force_concentration": GenomeConfig(
        pressure_profile="guarded",
        concentration_profile="strict",
        value_profile="finisher",
    ),
    "todo_antidogpile_position": GenomeConfig(
        style_profile="conservative",
        threat_profile="leader_focus",
        crowd_profile="hard",
        position_profile="local_safe",
        pressure_profile="guarded",
    ),
}


def random_genome(rng: random.Random) -> GenomeConfig:
    payload = {name: rng.choice(options) for name, options in GENE_SPACE.items()}
    genome = GenomeConfig(**payload)
    genome.validate()
    return genome


def mutate_genome(genome: GenomeConfig, rng: random.Random, mutation_rate: float = 0.25) -> GenomeConfig:
    payload = genome.to_dict()
    mutated = False
    for name, options in GENE_SPACE.items():
        if rng.random() >= mutation_rate:
            continue
        choices = [option for option in options if option != payload[name]]
        if choices:
            payload[name] = rng.choice(choices)
            mutated = True
    if not mutated:
        name = rng.choice(list(GENE_SPACE))
        choices = [option for option in GENE_SPACE[name] if option != payload[name]]
        payload[name] = rng.choice(choices)
    child = GenomeConfig(**payload)
    child.validate()
    return child


def crossover_genomes(left: GenomeConfig, right: GenomeConfig, rng: random.Random) -> GenomeConfig:
    payload = {}
    for field in fields(GenomeConfig):
        payload[field.name] = getattr(left, field.name) if rng.random() < 0.5 else getattr(right, field.name)
    child = GenomeConfig(**payload)
    child.validate()
    return child


class GenomeDecisionLogic(BASE.DecisionLogic):
    def __init__(self, obs: Any, config: Any, genome: GenomeConfig) -> None:
        self.genome = genome
        self.genome.validate()
        super().__init__(obs, config)

    def _build_modes(self) -> Dict[str, Any]:
        modes = dict(super()._build_modes())
        if self.state.num_players <= 2 and self.genome.duel_opening != "mtmr":
            modes["mtmr_safe_neutral_count"] = 0
            modes["mtmr_stage_ready"] = False
            modes["mtmr_hostility_ready"] = True

        vulture_owner_ids: List[int] = []
        if self.genome.vulture_profile != "off" and self.state.num_players >= 4:
            owner_strength = modes.get("owner_strength", {})
            for owner in owner_strength:
                if owner in (-1, self.state.player):
                    continue
                if self._owner_outbound_ratio(owner, owner_strength) >= 0.34:
                    vulture_owner_ids.append(owner)
                    continue
                if any(
                    self._planet_under_third_party_pressure(planet)
                    for planet in self.state.enemy_planets
                    if planet.owner == owner
                ):
                    vulture_owner_ids.append(owner)
        modes["vulture_owner_ids"] = vulture_owner_ids
        modes["vulture_active"] = bool(vulture_owner_ids)

        if self.genome.mode_profile != "dynamic":
            return modes

        owner_prod = modes.get("owner_prod", {})
        max_enemy_prod = max(
            (prod for owner, prod in owner_prod.items() if owner != self.state.player),
            default=0,
        )
        my_prod = owner_prod.get(self.state.player, 0)
        my_total = int(modes.get("my_total", 0))
        enemy_total = int(modes.get("enemy_total", 0))

        if self.state.num_players <= 2:
            domination = (my_total - enemy_total) / max(1, my_total + enemy_total)
            prod_domination = (my_prod - max_enemy_prod) / max(1, my_prod + max_enemy_prod)
            modes["is_ahead"] = domination > 0.14 or (domination > 0.09 and prod_domination > 0.08)
            modes["is_behind"] = domination < -0.17 or (domination < -0.10 and prod_domination < -0.06)
            modes["is_dominating"] = domination > 0.22 and prod_domination > 0.10
            modes["is_finishing"] = domination > 0.30 and prod_domination > 0.12 and self.state.step > 90
        else:
            leader_strength = max(1, int(modes.get("leader_strength", my_total)))
            second_strength = max(1, int(modes.get("second_strength", 0)))
            my_rank = int(modes.get("my_rank", 1))
            lead_ratio = my_total / leader_strength
            second_ratio = my_total / second_strength if second_strength > 0 else 2.0
            modes["is_ahead"] = my_rank == 1 and second_ratio > 1.05
            modes["is_dominating"] = my_rank == 1 and second_ratio > 1.18 and my_prod > max_enemy_prod
            modes["is_behind"] = my_rank >= 3 or (my_rank == 2 and lead_ratio < 0.92)
            modes["is_finishing"] = my_rank == 1 and second_ratio > 1.28 and self.state.step > 110

        mult = 1.0
        if modes.get("is_ahead"):
            mult += 0.05
        if modes.get("is_behind"):
            mult -= 0.03
        if modes.get("is_finishing"):
            mult += 0.10
        if modes.get("is_cleanup"):
            mult += 0.06
        if self.genome.style_profile == "aggressive":
            mult += 0.05
        elif self.genome.style_profile == "conservative":
            mult -= 0.04
        modes["attack_margin_mult"] = mult
        return modes

    def _planet_under_third_party_pressure(self, target: Planet) -> bool:
        if self.state.num_players < 4 or target.owner in (-1, self.state.player):
            return False
        arrivals = getattr(getattr(self, "world", None), "arrivals_by_planet", {}).get(target.id, [])
        incoming = sum(
            ships
            for eta, owner, ships in arrivals
            if owner not in (-1, self.state.player, target.owner)
            and eta <= VULTURE_ETA_LIMIT
        )
        return incoming >= VULTURE_MIN_INCOMING

    def _owner_outbound_ratio(self, owner: int, owner_strength: Dict[int, int]) -> float:
        if owner in (-1, self.state.player):
            return 0.0
        outbound = sum(fleet.ships for fleet in getattr(self.state, "enemy_fleets", []) if fleet.owner == owner)
        return outbound / max(1, owner_strength.get(owner, 0))

    def _build_enemy_priority(self) -> Dict[int, float]:
        if self.genome.threat_profile == "v23" or self.state.num_players <= 2:
            return super()._build_enemy_priority()

        totals: Dict[int, float] = {}
        for planet in self.state.enemy_planets:
            totals[planet.owner] = totals.get(planet.owner, 0.0) + planet.ships + 3.0 * planet.production
        for fleet in self.state.enemy_fleets:
            totals[fleet.owner] = totals.get(fleet.owner, 0.0) + fleet.ships
        ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
        priority: Dict[int, float] = {}
        for idx, (owner, score) in enumerate(ranked):
            if self.genome.threat_profile == "leader_focus":
                priority[owner] = 1.28 if idx == 0 else (1.04 if idx == 1 else 0.90)
            else:
                leader_gap = 0.0 if not ranked else score / max(1.0, ranked[0][1])
                priority[owner] = 1.24 if idx == 0 else max(0.88, 1.06 - 0.10 * idx + 0.08 * leader_gap)
        return priority

    def _mission_total_ships(self, mission: MissionOption) -> int:
        return int(sum(int(ships) for ships in mission.ships))

    def _mission_extra_arrivals(self, mission: MissionOption) -> Tuple[Tuple[int, int, int], ...]:
        return tuple(
            (int(eta), self.state.player, int(ships))
            for eta, ships in zip(mission.etas, mission.ships)
        )

    def _friendly_inbound_to_target(self, target_id: int, horizon: int) -> int:
        live = sum(
            int(ships)
            for eta, owner, ships in self.world.arrivals_by_planet.get(target_id, [])
            if owner == self.state.player and eta <= horizon
        )
        planned = sum(
            int(ships)
            for eta, owner, ships in self.planned_commitments.get(target_id, [])
            if owner == self.state.player and eta <= horizon
        )
        return live + planned

    def _force_concentration_multiplier(self, mission: MissionOption) -> float:
        profile = self.genome.concentration_profile
        if profile == "base" or self.state.is_opening or mission.mission not in {"attack", "snipe", "swarm"}:
            return 1.0

        target = self.state.planets_by_id.get(mission.target_id)
        if target is None or target.owner in (-1, self.state.player):
            return 1.0

        max_eta = max((int(eta) for eta in mission.etas), default=0)
        total_send = self._mission_total_ships(mission)
        extra = self._mission_extra_arrivals(mission)
        before_need = self.world.min_ships_to_own_at(
            target.id,
            max_eta,
            self.state.player,
            planned_commitments=self.planned_commitments,
        )
        after_need = self.world.min_ships_to_own_at(
            target.id,
            max_eta,
            self.state.player,
            planned_commitments=self.planned_commitments,
            extra_arrivals=extra,
        )
        existing_inbound = self._friendly_inbound_to_target(target.id, max_eta + 3)
        improvement = max(0, int(before_need) - int(after_need))
        ahead = bool(self.modes.get("is_ahead") or self.modes.get("is_dominating"))
        small_send = total_send <= max(8, int(target.production) * 3 + 2)
        thin_commit = total_send < max(1, int(math.ceil(before_need * (0.70 if profile == "guarded" else 0.90))))
        non_converting = after_need > 0

        if not non_converting:
            return 1.0
        if ahead and (small_send or thin_commit):
            return 0.45 if profile == "guarded" else 0.25
        if existing_inbound > 0 and improvement < int(total_send * 0.75):
            return 0.55 if profile == "guarded" else 0.35
        if thin_commit:
            return 0.78 if profile == "guarded" else 0.55
        return 1.0

    def _force_concentration_blocks(self, mission: MissionOption) -> bool:
        if self.genome.concentration_profile != "strict":
            return False
        multiplier = self._force_concentration_multiplier(mission)
        return multiplier < 0.30

    def _opening_expand_sort_key(self, target: Planet, available: Sequence[Planet]) -> Tuple[Any, ...]:
        min_distance = min(distance_planets(target, source) for source in available) if available else 10**9
        my_t, enemy_t = self.reaction_map.get(target.id, (10**9, 10**9))
        reaction_gap = enemy_t - my_t
        safe_rank = 0 if self._is_safe_neutral(target) else 1
        orbit_rank = 1 if is_orbiting_planet(target) else 0
        ship_cost = target.ships / max(1, target.production)

        if self.genome.opening_range_profile == "local_bias":
            return (
                safe_rank,
                my_t,
                min_distance,
                orbit_rank,
                -target.production,
                ship_cost,
            )
        if self.genome.opening_range_profile == "eta_focus":
            return (
                1 if my_t > 12 else 0,
                1 if is_orbiting_planet(target) and my_t > 9 else 0,
                1 if reaction_gap < BASE.SAFE_NEUTRAL_MARGIN else 0,
                my_t,
                safe_rank,
                -target.production,
                min_distance,
                ship_cost,
            )
        return (
            orbit_rank,
            safe_rank,
            -target.production,
            min_distance,
            ship_cost,
        )

    def _opening_range_value_multiplier(self, target: Planet, eta: int) -> float:
        if self.state.num_players > 2 or not self.state.is_opening or target.owner != -1:
            return 1.0

        my_t, enemy_t = self.reaction_map.get(target.id, (10**9, 10**9))
        reaction_gap = enemy_t - my_t
        orbiting = is_orbiting_planet(target)

        if self.genome.opening_range_profile == "local_bias":
            mult = 1.0
            if eta <= 9:
                mult *= 1.06
            elif eta >= 14:
                mult *= 0.90
            if orbiting and eta > 10:
                mult *= 0.88
            if reaction_gap < 1:
                mult *= 0.94
            return mult

        if self.genome.opening_range_profile == "eta_focus":
            mult = 1.0
            if eta <= 8:
                mult *= 1.10
            elif eta >= 12:
                mult *= 0.82
            if orbiting and eta > 8:
                mult *= 0.78
            if reaction_gap < BASE.SAFE_NEUTRAL_MARGIN:
                mult *= 0.86
            if my_t >= 12:
                mult *= 0.90
            return mult

        return 1.0

    def _opening_range_confidence_adjustment(self, target: Planet, plan: Any) -> float:
        if self.state.num_players > 2 or not self.state.is_opening or target.owner != -1:
            return 0.0

        my_t, enemy_t = self.reaction_map.get(target.id, (10**9, 10**9))
        reaction_gap = enemy_t - my_t

        if self.genome.opening_range_profile == "local_bias":
            delta = 0.0
            if plan.eta > 12:
                delta -= 0.05
            if is_orbiting_planet(target) and plan.eta > 9:
                delta -= 0.06
            if reaction_gap < 1:
                delta -= 0.04
            return delta

        if self.genome.opening_range_profile == "eta_focus":
            delta = 0.0
            if plan.eta > 10:
                delta -= 0.08
            if is_orbiting_planet(target) and plan.eta > 8:
                delta -= 0.10
            if reaction_gap < BASE.SAFE_NEUTRAL_MARGIN:
                delta -= 0.06
            return delta

        return 0.0

    def _shun_opening_active(self) -> bool:
        if self.state.num_players <= 2:
            return self.genome.duel_opening == "shun" and self.state.step < BASE.SHUN_OPENING_TURN_LIMIT
        return super()._shun_opening_active()

    def _mtmr_opening_active(self) -> bool:
        if self.genome.duel_opening != "mtmr":
            return False
        return super()._mtmr_opening_active()

    def _mtmr_neutral_allowed(self, target: Planet, plan: Any) -> bool:
        if self.genome.duel_opening != "mtmr":
            return True
        return super()._mtmr_neutral_allowed(target, plan)

    def _mtmr_hostile_target_allowed(self, target: Planet, plan: Any) -> bool:
        if self.genome.duel_opening != "mtmr":
            return True
        return super()._mtmr_hostile_target_allowed(target, plan)

    def _baseline_opening_filter(self, target: Planet, plan: Any, src_available: int) -> bool:
        if self.genome.duel_filter != "baseline_rotating":
            return False
        return super()._baseline_opening_filter(target, plan, src_available)

    def _frontline_reserve(self, planet: Planet) -> int:
        reserve = super()._frontline_reserve(planet)
        if (
            self.genome.crowd_profile == "base"
            or self.state.num_players < 4
            or not self.modes.get("four_player_crowded")
            or self.state.step >= getattr(BASE, "FOUR_PLAYER_CROWDED_TURN_LIMIT", 60)
        ):
            return reserve
        if self._frontline_enemy_count(planet) < 1:
            return reserve
        ratio = 0.28 if self.genome.crowd_profile == "antidogpile" else 0.32
        return max(reserve, int(planet.ships * ratio))

    def _four_player_neutral_position_multiplier(self, target: Planet, eta: int) -> float:
        mult = super()._four_player_neutral_position_multiplier(target, eta)
        if (
            self.genome.position_profile == "base"
            or self.state.num_players < 4
            or target.owner != -1
            or self.modes.get("is_dominating")
        ):
            return mult

        my_t, enemy_t = self.reaction_map.get(target.id, (10**9, 10**9))
        race_gap = enemy_t - my_t
        orbiting = is_orbiting_planet(target)

        if self.genome.position_profile == "safer_neutrals":
            if eta >= getattr(BASE, "FOUR_PLAYER_LONG_NEUTRAL_ETA", 15):
                mult *= 0.92
            elif orbiting and eta >= getattr(BASE, "FOUR_PLAYER_LONG_ORBITING_ETA", 11):
                mult *= 0.90
            if race_gap <= 1:
                mult *= 0.92
            if not orbiting and eta <= 10 and race_gap >= BASE.SAFE_NEUTRAL_MARGIN + 1:
                mult *= 1.03
            return mult

        if eta >= 13:
            mult *= 0.86
        elif orbiting and eta >= 9:
            mult *= 0.84
        if race_gap <= 1:
            mult *= 0.88
        elif race_gap == 2:
            mult *= 0.94
        if not orbiting and eta <= 9 and race_gap >= BASE.SAFE_NEUTRAL_MARGIN + 1:
            mult *= 1.07
        return mult

    def _turn_launch_cap(self) -> int:
        if self.state.num_players > 2:
            cap = super()._turn_launch_cap()
            if self.genome.conversion_profile == "protect" and (self.modes.get("is_ahead") or self.modes.get("is_dominating")):
                cap = max(2, cap - 1)
            elif self.genome.conversion_profile == "closeout" and (self.modes.get("is_finishing") or self.state.is_late):
                cap = min(BASE.MAX_TURN_LAUNCHES, cap + 1)
            if self.genome.style_profile == "aggressive":
                return min(BASE.MAX_TURN_LAUNCHES, cap + 1)
            if self.genome.style_profile == "conservative" and not self.modes.get("is_behind"):
                return max(2, cap - 1)
            return cap

        if self.genome.duel_launch_cap == "relaxed" and self.state.step < BASE.DUEL_OPENING_CAP_TURN:
            cap = min(BASE.MAX_TURN_LAUNCHES, BASE.DUEL_OPENING_LAUNCH_CAP + 1)
            if self.genome.conversion_profile == "protect" and self.modes.get("is_ahead"):
                cap = max(2, cap - 1)
            if self.genome.style_profile == "aggressive":
                return min(BASE.MAX_TURN_LAUNCHES, cap + 1)
            if self.genome.style_profile == "conservative":
                return max(2, cap - 1)
            return cap
        if self.genome.duel_launch_cap == "mtmr":
            if self._mtmr_opening_active():
                cap = min(BASE.MAX_TURN_LAUNCHES, BASE.MTMR_DUEL_OPENING_LAUNCH_CAP)
                if self.genome.style_profile == "aggressive":
                    return min(BASE.MAX_TURN_LAUNCHES, cap + 1)
                if self.genome.style_profile == "conservative":
                    return max(2, cap - 1)
                return cap
            if self._shun_opening_active():
                cap = min(BASE.MAX_TURN_LAUNCHES, BASE.SHUN_OPENING_LAUNCH_CAP)
                if self.genome.style_profile == "aggressive":
                    return min(BASE.MAX_TURN_LAUNCHES, cap + 1)
                if self.genome.style_profile == "conservative":
                    return max(1, cap - 1)
                return cap
            if self.state.step < BASE.DUEL_OPENING_CAP_TURN:
                cap = min(BASE.MAX_TURN_LAUNCHES, BASE.MTMR_DUEL_OPENING_LAUNCH_CAP)
                if self.genome.style_profile == "aggressive":
                    return min(BASE.MAX_TURN_LAUNCHES, cap + 1)
                if self.genome.style_profile == "conservative":
                    return max(2, cap - 1)
                return cap
        cap = super()._turn_launch_cap()
        if self.genome.conversion_profile == "protect" and (self.modes.get("is_ahead") or self.modes.get("is_dominating")):
            cap = max(2, cap - 1)
        elif self.genome.conversion_profile == "closeout" and (self.modes.get("is_finishing") or self.state.is_late):
            cap = min(BASE.MAX_TURN_LAUNCHES, cap + 1)
        if self.genome.style_profile == "aggressive":
            return min(BASE.MAX_TURN_LAUNCHES, cap + 1)
        if self.genome.style_profile == "conservative" and self.state.step < BASE.DUEL_OPENING_CAP_TURN:
            return max(2, cap - 1)
        return cap

    def _opening_capture_confidence(self, target: Planet, plan: Any, mission: str) -> float:
        confidence = super()._opening_capture_confidence(target, plan, mission)
        if (
            self.genome.position_profile != "base"
            and self.state.num_players >= 4
            and mission == "expand"
            and target.owner == -1
        ):
            my_t, enemy_t = self.reaction_map.get(target.id, (10**9, 10**9))
            race_gap = enemy_t - my_t
            if self.genome.position_profile == "safer_neutrals":
                if plan.eta >= getattr(BASE, "FOUR_PLAYER_LONG_NEUTRAL_ETA", 15):
                    confidence -= 0.05
                if race_gap <= 1:
                    confidence -= 0.04
            else:
                if plan.eta >= 13:
                    confidence -= 0.08
                elif is_orbiting_planet(target) and plan.eta >= 9:
                    confidence -= 0.07
                if race_gap <= 1:
                    confidence -= 0.06
                if not is_orbiting_planet(target) and plan.eta <= 9 and race_gap >= BASE.SAFE_NEUTRAL_MARGIN + 1:
                    confidence += 0.03
        if self.genome.style_profile == "aggressive":
            if mission in {"attack", "snipe", "swarm"} and target.owner not in (-1, self.state.player):
                confidence += 0.05
            if mission == "expand" and target.owner == -1 and self._is_safe_neutral(target):
                confidence -= 0.03
        elif self.genome.style_profile == "conservative":
            if mission in {"attack", "snipe", "swarm"} and target.owner not in (-1, self.state.player):
                confidence -= 0.06
            if mission == "expand" and target.owner == -1 and self._is_safe_neutral(target):
                confidence += 0.03
        if mission == "expand":
            confidence += self._opening_range_confidence_adjustment(target, plan)
        if self.genome.pressure_profile != "guarded" or self.state.num_players > 2:
            return confidence
        if target.owner == -1 and self._is_contested_neutral(target):
            confidence -= 0.08
        if target.owner == -1 and is_orbiting_planet(target) and plan.eta > 10:
            confidence -= 0.05
        if mission == "expand" and target.owner == -1 and target.production <= 2 and not self._is_safe_neutral(target):
            confidence -= 0.08
        return confidence

    def _target_value(self, target: Planet, eta: int, mission: str) -> float:
        value = super()._target_value(target, eta, mission)
        if value <= 0:
            return value

        hostile = target.owner not in (-1, self.state.player)
        safe_neutral = target.owner == -1 and self._is_safe_neutral(target)
        static = not is_orbiting_planet(target)

        if target.owner == -1:
            value *= self._opening_range_value_multiplier(target, eta)

        if self.genome.value_profile == "economy":
            if safe_neutral:
                value *= 1.10
            if static:
                value *= 1.04
            if target.id in self.state.comet_planet_ids:
                value *= 1.05
            if hostile and self.state.is_opening:
                value *= 0.92
        elif self.genome.value_profile == "hostile":
            if hostile:
                value *= 1.12
            if safe_neutral:
                value *= 0.95
        elif self.genome.value_profile == "finisher":
            if hostile and (self.state.is_late or self.modes.get("is_ahead")):
                value *= 1.16
            if target.owner == -1:
                value *= 0.96

        if self.genome.style_profile == "aggressive":
            if hostile:
                value *= 1.08
            elif safe_neutral:
                value *= 0.95
        elif self.genome.style_profile == "conservative":
            if safe_neutral:
                value *= 1.08
            if static and target.owner == -1:
                value *= 1.04
            if hostile:
                value *= 0.94

        if self.genome.pressure_profile == "guarded" and self.state.num_players <= 2:
            if target.owner == -1 and not safe_neutral:
                value *= 0.95
        if (
            self.genome.vulture_profile != "off"
            and self.state.num_players >= 4
            and hostile
            and (
                target.owner in self.modes.get("vulture_owner_ids", [])
                or self._planet_under_third_party_pressure(target)
            )
        ):
            value *= VULTURE_HOSTILE_BONUS[self.genome.vulture_profile]
        if self.genome.conversion_profile == "protect" and (self.modes.get("is_ahead") or self.modes.get("is_dominating")):
            if target.owner == -1:
                value *= 0.88 if self._is_contested_neutral(target) else 0.93
            elif hostile and (self.state.is_late or self.modes.get("is_finishing")):
                value *= 1.06
        elif self.genome.conversion_profile == "closeout":
            if hostile and (self.state.is_late or self.modes.get("is_finishing") or self.modes.get("is_cleanup")):
                value *= 1.14
            if target.owner == -1 and (self.modes.get("is_ahead") or self.modes.get("is_finishing")):
                value *= 0.90
        return value

    def _build_expand_missions(self, missions: List[MissionOption]) -> None:
        if self.genome.opening_range_profile == "base" or self.state.num_players > 2:
            return super()._build_expand_missions(missions)

        candidates = [
            target for target in self.state.neutral_planets
            if target.id not in self.state.comet_planet_ids
        ]
        if not candidates:
            return

        available = self._available_my_planets()
        if self.state.is_opening and available:
            if self._mtmr_opening_active():
                candidates = [target for target in candidates if target.production >= 2]
            elif self._shun_opening_active():
                candidates = [
                    target for target in candidates
                    if target.production >= 3 or not is_orbiting_planet(target)
                ]
            elif self.state.is_early:
                candidates = [target for target in candidates if target.production >= BASE.OPENING_MIN_PRODUCTION]
            candidates.sort(key=lambda target: self._opening_expand_sort_key(target, available))

        max_eta = 14 if self.state.is_early else (18 if self.state.is_opening else 28)
        for target in candidates:
            if self.expired():
                return
            for donor in sorted(self._available_my_planets(), key=lambda source: distance_planets(source, target)):
                surplus = self._planet_surplus(donor)
                if surplus <= 0:
                    continue
                plan = self._settle_plan(donor, target, surplus, mission="expand", max_turn=max_eta)
                if plan is None:
                    continue
                if not self._candidate_time_valid(target, plan.eta):
                    continue
                if self._baseline_opening_filter(target, plan, surplus):
                    continue
                if not self._opening_mission_allowed(target, plan, "expand"):
                    continue
                value = self._target_value(target, plan.eta, "expand")
                if value <= 0:
                    continue
                score = self._score_mission(value, plan.ships, plan.eta, target, "expand")
                score *= self._opening_capture_confidence(target, plan, "expand")
                missions.append(MissionOption(
                    score=score,
                    source_ids=[donor.id],
                    target_id=target.id,
                    angles=[plan.angle],
                    etas=[plan.eta],
                    ships=[plan.ships],
                    needed=plan.required_ships,
                    mission="expand",
                ))
                break

    def _mission_can_commit(
        self,
        mission: MissionOption,
        extra_used_sources: Optional[set[int]] = None,
        extra_target_ids: Optional[set[int]] = None,
        existing_moves: int = 0,
    ) -> bool:
        if not super()._mission_can_commit(
            mission,
            extra_used_sources=extra_used_sources,
            extra_target_ids=extra_target_ids,
            existing_moves=existing_moves,
        ):
            return False
        return not self._force_concentration_blocks(mission)

    def _build_attack_missions(self, missions: List[MissionOption]) -> None:
        if self.state.num_players > 2:
            if self.genome.vulture_profile == "off":
                return super()._build_attack_missions(missions)

            if not self.state.enemy_planets:
                return
            if self._mtmr_opening_active() or self._shun_opening_active():
                return

            vulture_active = bool(self.modes.get("vulture_active"))
            if not vulture_active and not self.modes.get("four_player_pivot_ready"):
                if self.modes.get("four_player_safe_neutral_count", 0) > BASE.FOUR_PLAYER_PIVOT_SAFE_NEUTRAL_LIMIT:
                    return
            if (
                not vulture_active
                and not self.modes.get("is_dominating")
                and self.state.step < 120
            ):
                safe_neutrals = [
                    target for target in self.state.neutral_planets
                    if target.production >= BASE.OPENING_MIN_PRODUCTION and self._is_safe_neutral(target)
                ]
                if len(safe_neutrals) >= 2:
                    return
            if self.state.is_early and not vulture_active:
                good_neutrals = [target for target in self.state.neutral_planets if target.production >= BASE.OPENING_MIN_PRODUCTION]
                if len(good_neutrals) >= 3:
                    return
            elif self.state.is_opening and len(self.state.neutral_planets) > 6 and not vulture_active:
                return

            if vulture_active:
                targets = sorted(
                    self.state.enemy_planets,
                    key=lambda target: (
                        not self._planet_under_third_party_pressure(target),
                        target.owner not in self.modes.get("vulture_owner_ids", []),
                        target.ships / max(1, target.production),
                        target.ships,
                    ),
                )
            else:
                targets = sorted(self.state.enemy_planets, key=lambda target: (target.ships / max(1, target.production), target.ships))

            for target in targets:
                if self.expired():
                    return
                for donor in sorted(self._available_my_planets(), key=lambda source: distance_planets(source, target)):
                    surplus = self._planet_surplus(donor)
                    if surplus <= 0:
                        continue
                    plan = self._settle_plan(donor, target, surplus, mission="attack", max_turn=35)
                    if plan is None:
                        continue
                    if not self._candidate_time_valid(target, plan.eta):
                        continue
                    if not self._opening_mission_allowed(target, plan, "attack"):
                        continue
                    value = self._target_value(target, plan.eta, "attack")
                    if value <= 0:
                        continue
                    score = self._score_mission(value, plan.ships, plan.eta, target, "attack")
                    score *= self._opening_capture_confidence(target, plan, "attack")
                    missions.append(MissionOption(
                        score=score,
                        source_ids=[donor.id],
                        target_id=target.id,
                        angles=[plan.angle],
                        etas=[plan.eta],
                        ships=[plan.ships],
                        needed=plan.required_ships,
                        mission="attack",
                    ))
                    break
            return

        if self.genome.duel_attack_order == "local_pressure":
            return super()._build_attack_missions(missions)

        if not self.state.enemy_planets:
            return
        if self._mtmr_opening_active() or self._shun_opening_active():
            return
        early_neutral_limit = 3
        opening_neutral_limit = 6
        if self.genome.transition_profile == "earlier_attack":
            early_neutral_limit = 2
            opening_neutral_limit = 4
        elif self.genome.transition_profile == "later_attack":
            early_neutral_limit = 4
            opening_neutral_limit = 8

        if self.state.is_early:
            good_neutrals = [p for p in self.state.neutral_planets if p.production >= BASE.OPENING_MIN_PRODUCTION]
            if len(good_neutrals) >= early_neutral_limit:
                return
        elif self.state.is_opening and len(self.state.neutral_planets) > opening_neutral_limit:
            return

        if self.genome.duel_attack_order == "production_first":
            targets = sorted(
                self.state.enemy_planets,
                key=lambda t: (-t.production, t.ships / max(1, t.production), t.ships),
            )
        else:
            targets = sorted(self.state.enemy_planets, key=lambda t: (t.ships / max(1, t.production), t.ships))

        for target in targets:
            if self.expired():
                return
            for donor in sorted(self._available_my_planets(), key=lambda s: distance_planets(s, target)):
                surplus = self._planet_surplus(donor)
                if surplus <= 0:
                    continue
                plan = self._settle_plan(donor, target, surplus, mission="attack", max_turn=35)
                if plan is None:
                    continue
                if not self._candidate_time_valid(target, plan.eta):
                    continue
                if not self._opening_mission_allowed(target, plan, "attack"):
                    continue
                value = self._target_value(target, plan.eta, "attack")
                if value <= 0:
                    continue
                score = self._score_mission(value, plan.ships, plan.eta, target, "attack")
                score *= self._opening_capture_confidence(target, plan, "attack")
                missions.append(MissionOption(
                    score=score,
                    source_ids=[donor.id],
                    target_id=target.id,
                    angles=[plan.angle],
                    etas=[plan.eta],
                    ships=[plan.ships],
                    needed=plan.required_ships,
                    mission="attack",
                ))
                break

    def _build_swarm_missions(self, missions: List[MissionOption]) -> None:
        if self.state.is_opening or not self.state.enemy_planets:
            return
        available = self._available_my_planets()
        if len(available) < 2:
            return

        eta_tolerance = SWARM_TOLERANCES[self.genome.swarm_profile]
        targets = sorted(self.state.enemy_planets, key=lambda t: (t.ships / max(1, t.production), t.ships))
        for target in targets:
            if self.expired():
                return
            nearby = sorted(available, key=lambda s: distance_planets(s, target))[:BASE.MULTI_SOURCE_TOP_K]
            donor_plans: List[Tuple[Planet, int, float, int]] = []
            for donor in nearby:
                if self.expired():
                    return
                surplus = self._planet_surplus(donor)
                if surplus <= 0:
                    continue
                shot = self._plan_shot(donor, target, surplus)
                if shot is None:
                    continue
                donor_plans.append((donor, surplus, shot[0], shot[1]))
            if len(donor_plans) < 2:
                continue
            donor_plans.sort(key=lambda item: item[3])
            for group_size in (2, 3):
                if self.expired():
                    break
                for i in range(len(donor_plans)):
                    anchor_eta = donor_plans[i][3]
                    group = [donor_plans[i]]
                    for j in range(len(donor_plans)):
                        if i == j:
                            continue
                        if abs(donor_plans[j][3] - anchor_eta) <= eta_tolerance:
                            group.append(donor_plans[j])
                        if len(group) == group_size:
                            break
                    if len(group) < group_size:
                        continue
                    extra = tuple((item[3], self.state.player, item[1]) for item in group)
                    max_eta = max(item[3] for item in group)
                    need = self.world.min_ships_to_own_at(
                        target.id,
                        max_eta,
                        self.state.player,
                        planned_commitments=self.planned_commitments,
                        extra_arrivals=extra,
                    )
                    if need == 0 and self._candidate_time_valid(target, max_eta):
                        total_ships = sum(item[1] for item in group)
                        value = self._target_value(target, max_eta, "swarm")
                        if value <= 0:
                            continue
                        score = self._score_mission(value, total_ships, max_eta, target, "swarm")
                        missions.append(MissionOption(
                            score=score,
                            source_ids=[item[0].id for item in group],
                            target_id=target.id,
                            angles=[item[2] for item in group],
                            etas=[item[3] for item in group],
                            ships=[item[1] for item in group],
                            needed=total_ships,
                            mission="swarm",
                        ))
                        break

    def _commit_missions(self, missions: List[MissionOption]) -> List[PlannedMove]:
        remaining = sorted(missions, key=lambda mission: -mission.score)
        moves: List[PlannedMove] = []
        turn_launch_cap = self._turn_launch_cap()
        followup_weight = FOLLOWUP_WEIGHTS[self.genome.followup_profile]

        while remaining and not self.expired() and len(moves) < turn_launch_cap:
            horizon = min(len(remaining), 10)
            best_idx = None
            best_value = -float("inf")

            for idx in range(horizon):
                mission = remaining[idx]
                if not self._mission_can_commit(mission, existing_moves=len(moves)):
                    continue

                pending_sources = set(mission.source_ids)
                pending_targets = {mission.target_id} if self._mission_blocks_target(mission) else set()
                follower_bonus = 0.0
                follower_horizon = min(len(remaining), 10)
                for follower in remaining[:follower_horizon]:
                    if follower is mission:
                        continue
                    if self._mission_can_commit(
                        follower,
                        extra_used_sources=pending_sources,
                        extra_target_ids=pending_targets,
                        existing_moves=len(moves) + len(mission.source_ids),
                    ):
                        follower_bonus = max(follower_bonus, follower.score)
                value = (mission.score + followup_weight * follower_bonus) * self._force_concentration_multiplier(mission)
                if value > best_value:
                    best_value = value
                    best_idx = idx

            if best_idx is None:
                break

            mission = remaining.pop(best_idx)
            if len(mission.source_ids) == 1:
                src_id = mission.source_ids[0]
                src = self.state.planets_by_id.get(src_id)
                if src is None or self._effective_planet(src).ships < mission.ships[0]:
                    continue
                move = PlannedMove(
                    src_id,
                    mission.target_id,
                    mission.angles[0],
                    mission.ships[0],
                    mission.etas[0],
                    mission.mission,
                )
                moves.append(move)
                self._commit_move(move)
            else:
                if len(moves) + len(mission.source_ids) > turn_launch_cap:
                    continue
                can_commit = True
                for i, sid in enumerate(mission.source_ids):
                    src = self.state.planets_by_id.get(sid)
                    if src is None or self._effective_planet(src).ships < mission.ships[i]:
                        can_commit = False
                        break
                if not can_commit:
                    continue
                for i, sid in enumerate(mission.source_ids):
                    move = PlannedMove(
                        sid,
                        mission.target_id,
                        mission.angles[i],
                        mission.ships[i],
                        mission.etas[i],
                        mission.mission,
                    )
                    moves.append(move)
                    self._commit_move(move)
        return moves


def build_agent(genome: GenomeConfig) -> Callable[[Any, Any], List[List[float | int]]]:
    genome.validate()

    def agent(obs: Any, config: Any) -> List[List[float | int]]:
        try:
            logic = GenomeDecisionLogic(obs, config, genome)
            return logic.decide()
        except Exception as exc:  # preserve the parent agent's fail-closed behavior
            BASE.AGENT_MEMORY["last_error"] = str(exc)
            return []

    return agent


def write_agent_wrapper(genome: GenomeConfig, output_path: Path) -> Path:
    genome.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(genome.to_dict(), indent=2, sort_keys=True)
    wrapper = f"""from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENOME_DIR = ROOT / "genome test"
if str(GENOME_DIR) not in sys.path:
    sys.path.insert(0, str(GENOME_DIR))

from genome_agent import GenomeConfig, build_agent

GENOME = GenomeConfig.from_dict({payload})
agent = build_agent(GENOME)
"""
    output_path.write_text(wrapper, encoding="utf-8")
    return output_path


__all__ = [
    "BASE_AGENT_PATH",
    "GENE_SPACE",
    "FOLLOWUP_WEIGHTS",
    "GenomeConfig",
    "PRESET_GENOMES",
    "GenomeDecisionLogic",
    "build_agent",
    "crossover_genomes",
    "mutate_genome",
    "random_genome",
    "write_agent_wrapper",
]
