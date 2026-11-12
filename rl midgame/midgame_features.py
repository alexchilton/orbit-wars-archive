from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

FEATURE_NAMES: List[str] = [
    "step_frac",
    "remaining_frac",
    "is_two_player",
    "is_four_player_plus",
    "is_ahead",
    "is_behind",
    "is_dominating",
    "is_finishing",
    "my_ship_share",
    "my_prod_share",
    "lead_ratio",
    "base_score_norm",
    "base_value_norm",
    "eta_norm",
    "send_norm",
    "source_count_norm",
    "needed_norm",
    "target_prod_norm",
    "target_ships_norm",
    "target_is_neutral",
    "target_is_enemy",
    "target_is_friendly",
    "mission_attack",
    "mission_expand",
    "mission_defend",
    "mission_reinforce",
    "mission_swarm",
    "mission_snipe",
    "mission_comet",
    "mission_other",
    "enemy_priority",
    "committed_load",
    "is_defer",
]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator <= 0:
        return default
    return numerator / denominator


def _tanh_scale(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return math.tanh(value / scale)


@dataclass(frozen=True)
class MissionFeatureBundle:
    vector: List[float]
    metadata: Dict[str, Any]


def build_state_snapshot(logic: Any) -> Dict[str, float]:
    state = logic.state
    modes = getattr(logic, "modes", {})
    owner_prod = modes.get("owner_prod", {})

    my_total = float(modes.get("my_total", 0.0))
    enemy_total = float(modes.get("enemy_total", 0.0))
    my_prod = float(owner_prod.get(state.player, 0.0))
    max_enemy_prod = max(
        (float(prod) for owner, prod in owner_prod.items() if owner != state.player),
        default=0.0,
    )

    if state.num_players <= 2:
        lead_base = max(1.0, enemy_total)
    else:
        lead_base = max(1.0, float(modes.get("leader_strength", my_total)))

    return {
        "step_frac": _clip01(float(state.step) / 200.0),
        "remaining_frac": _clip01(float(state.remaining_steps) / 200.0),
        "is_two_player": 1.0 if state.num_players <= 2 else 0.0,
        "is_four_player_plus": 1.0 if state.num_players >= 4 else 0.0,
        "is_ahead": 1.0 if modes.get("is_ahead") else 0.0,
        "is_behind": 1.0 if modes.get("is_behind") else 0.0,
        "is_dominating": 1.0 if modes.get("is_dominating") else 0.0,
        "is_finishing": 1.0 if modes.get("is_finishing") else 0.0,
        "my_ship_share": _clip01(_safe_ratio(my_total, my_total + enemy_total, default=0.5)),
        "my_prod_share": _clip01(_safe_ratio(my_prod, my_prod + max_enemy_prod, default=0.5)),
        "lead_ratio": _clip01(_safe_ratio(my_total, lead_base, default=1.0)),
    }


def build_mission_feature_bundle(
    logic: Any,
    mission: Any,
    base_value: float,
    existing_moves: int,
    turn_launch_cap: int,
) -> MissionFeatureBundle:
    state = logic.state
    snapshot = build_state_snapshot(logic)
    target = state.planets_by_id.get(mission.target_id)
    target_owner = -1 if target is None else int(target.owner)
    target_prod = 0.0 if target is None else float(target.production)
    target_ships = 0.0 if target is None else float(target.ships)
    total_send = float(sum(mission.ships))
    mission_type = str(mission.mission)

    vector = [
        snapshot["step_frac"],
        snapshot["remaining_frac"],
        snapshot["is_two_player"],
        snapshot["is_four_player_plus"],
        snapshot["is_ahead"],
        snapshot["is_behind"],
        snapshot["is_dominating"],
        snapshot["is_finishing"],
        snapshot["my_ship_share"],
        snapshot["my_prod_share"],
        snapshot["lead_ratio"],
        _tanh_scale(float(mission.score), 80.0),
        _tanh_scale(float(base_value), 90.0),
        _clip01(_safe_ratio(float(max(mission.etas, default=0)), 28.0)),
        _clip01(_safe_ratio(total_send, 80.0)),
        _clip01(_safe_ratio(float(len(mission.source_ids)), 4.0)),
        _clip01(_safe_ratio(float(mission.needed), 80.0)),
        _clip01(_safe_ratio(target_prod, 10.0)),
        _clip01(_safe_ratio(target_ships, 80.0)),
        1.0 if target_owner == -1 else 0.0,
        1.0 if target_owner not in (-1, state.player) else 0.0,
        1.0 if target_owner == state.player else 0.0,
        1.0 if mission_type == "attack" else 0.0,
        1.0 if mission_type == "expand" else 0.0,
        1.0 if mission_type == "defend" else 0.0,
        1.0 if mission_type == "reinforce" else 0.0,
        1.0 if mission_type == "swarm" else 0.0,
        1.0 if mission_type == "snipe" else 0.0,
        1.0 if mission_type == "comet" else 0.0,
        1.0 if mission_type not in {"attack", "expand", "defend", "reinforce", "swarm", "snipe", "comet"} else 0.0,
        float(getattr(logic, "enemy_priority", {}).get(target_owner, 1.0)),
        _clip01(_safe_ratio(float(existing_moves), float(max(1, turn_launch_cap)))),
        0.0,  # is_defer: real candidates are not defer
    ]

    metadata = {
        "mission": mission_type,
        "target_id": int(mission.target_id),
        "target_owner": target_owner,
        "target_prod": target_prod,
        "target_ships": target_ships,
        "base_score": float(mission.score),
        "base_value": float(base_value),
        "eta": int(max(mission.etas, default=0)),
        "send": total_send,
        "sources": list(mission.source_ids),
    }
    return MissionFeatureBundle(vector=vector, metadata=metadata)


def build_defer_vector(heuristic_best_vector: List[float]) -> List[float]:
    """Build a defer-action feature vector from the heuristic's best candidate."""
    defer = list(heuristic_best_vector)
    defer[-1] = 1.0  # is_defer flag
    return defer
