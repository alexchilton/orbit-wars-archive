from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from genome_agent import BASE

Planet = BASE.Planet
DEFAULT_MAX_SHIP_SPEED = BASE.DEFAULT_MAX_SHIP_SPEED


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_planets(obs: Any) -> List[Planet]:
    return list(BASE.as_planets(_get_field(obs, "planets", []) or []))


def _safe_shot(source: Planet, target: Planet, ships: int, max_speed: float) -> Optional[Tuple[float, int]]:
    result = BASE.estimate_arrival(
        source.x,
        source.y,
        source.radius,
        target.x,
        target.y,
        target.radius,
        ships,
        max_speed=max_speed,
    )
    if result is None:
        return None
    angle, eta, _distance = result
    return BASE.normalize_angle(angle), eta


def _choose_expand_target(source: Planet, neutrals: Sequence[Planet], max_speed: float) -> Optional[Tuple[Planet, int, float]]:
    best: Optional[Tuple[float, Planet, int, float]] = None
    for target in neutrals:
        required = max(1, int(target.ships) + 1)
        if required >= source.ships:
            continue
        shot = _safe_shot(source, target, required, max_speed)
        if shot is None:
            continue
        angle, eta = shot
        score = (2.6 * target.production) - required - 0.12 * eta
        if best is None or score > best[0]:
            best = (score, target, required, angle)
    if best is None:
        return None
    return best[1], best[2], best[3]


def greedy_agent(obs: Any, config: Any) -> List[List[float | int]]:
    planets = _as_planets(obs)
    if not planets:
        return []
    player = int(_get_field(obs, "player", 0))
    max_speed = float(_get_field(config, "shipSpeed", DEFAULT_MAX_SHIP_SPEED))
    my_planets = [planet for planet in planets if planet.owner == player]
    enemy_planets = [planet for planet in planets if planet.owner not in (-1, player)]
    neutral_planets = [planet for planet in planets if planet.owner == -1]
    moves: List[List[float | int]] = []
    targeted_ids: set[int] = set()

    for source in sorted(my_planets, key=lambda planet: planet.ships, reverse=True):
        if len(moves) >= 4 or source.ships <= 4:
            continue
        enemy_target = None
        best_enemy = None
        for target in enemy_planets:
            if target.id in targeted_ids:
                continue
            required = max(1, int(target.ships) + 2)
            if required >= source.ships:
                continue
            shot = _safe_shot(source, target, required, max_speed)
            if shot is None:
                continue
            angle, eta = shot
            distance = BASE.distance_planets(source, target)
            score = target.production * 3.0 - required - 0.18 * distance - 0.08 * eta
            if best_enemy is None or score > best_enemy[0]:
                best_enemy = (score, target, required, angle)
        if best_enemy is not None:
            _, target, required, angle = best_enemy
            enemy_target = [source.id, float(angle), required]
            targeted_ids.add(target.id)

        if enemy_target is not None:
            moves.append(enemy_target)
            continue

        expand = _choose_expand_target(source, neutral_planets, max_speed)
        if expand is None:
            continue
        target, required, angle = expand
        targeted_ids.add(target.id)
        moves.append([source.id, float(angle), required])
    return moves


def turtle_agent(obs: Any, config: Any) -> List[List[float | int]]:
    planets = _as_planets(obs)
    if not planets:
        return []
    player = int(_get_field(obs, "player", 0))
    step = int(_get_field(obs, "step", 0) or 0)
    max_speed = float(_get_field(config, "shipSpeed", DEFAULT_MAX_SHIP_SPEED))
    my_planets = [planet for planet in planets if planet.owner == player]
    neutral_planets = [planet for planet in planets if planet.owner == -1]
    moves: List[List[float | int]] = []

    if not my_planets:
        return moves

    if step < 120:
        for source in sorted(my_planets, key=lambda planet: planet.ships, reverse=True):
            if len(moves) >= 2 or source.ships <= 5:
                continue
            safe_neutrals = sorted(
                neutral_planets,
                key=lambda target: (BASE.distance_planets(source, target), -target.production, target.ships),
            )
            for target in safe_neutrals:
                required = max(1, int(target.ships) + 1)
                if required >= source.ships:
                    continue
                shot = _safe_shot(source, target, required, max_speed)
                if shot is None:
                    continue
                angle, _eta = shot
                moves.append([source.id, float(angle), required])
                break
        return moves

    if len(my_planets) < 2:
        return moves

    defended = sorted(my_planets, key=lambda planet: (planet.ships, -planet.production))
    weakest = defended[0]
    for source in sorted(my_planets, key=lambda planet: planet.ships, reverse=True):
        if source.id == weakest.id or source.ships <= 8:
            continue
        send = max(1, int(source.ships * 0.35))
        shot = _safe_shot(source, weakest, send, max_speed)
        if shot is None:
            continue
        angle, _eta = shot
        moves.append([source.id, float(angle), send])
        break
    return moves


__all__ = ["greedy_agent", "turtle_agent"]
