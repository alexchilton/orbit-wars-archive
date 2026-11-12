from __future__ import annotations

import math
import time
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet = namedtuple("Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])

BOARD_SIZE = 100.0
CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
SUN_CENTER = (CENTER, CENTER)
SUN_RADIUS = 10.0

MIN_SHIP_SPEED = 1.0
DEFAULT_MAX_SHIP_SPEED = 6.0
LOG_1000 = math.log(1000.0)
LAUNCH_CLEARANCE = 0.1

TOTAL_STEPS = 500
BASE_TIMELINE_HORIZON = 90
MAX_INTERCEPT_TURNS = 120
SOFT_ACT_DEADLINE = 0.82

SURPLUS_HORIZON = 20
DEFENSE_LOOKAHEAD = 18
OPENING_SEARCH_LOOKAHEAD = 18
MIDGAME_SEARCH_LOOKAHEAD = 12
OPENING_SEARCH_PLANET_CAP = 3
TARGET_CAP = 10
OPENING_TARGET_CAP = 7
SEND_BUCKET_RATIOS = (0.34, 0.67, 1.0)
ETA_SYNC_TOLERANCE = 2
MOVE_STEP_CAP = 3
MOVE_CANDIDATE_CAP = 6
MIN_SURPLUS_TO_SEND = 6
MAX_ATTACK_ETA = 28
MAX_EXPAND_ETA = 24
MAX_DEFEND_ETA = 18
MIN_ATTACK_STEP = 18


def get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize_angle(angle: float) -> float:
    return angle % (2.0 * math.pi)


def point_in_bounds(x: float, y: float) -> bool:
    return 0.0 <= x <= BOARD_SIZE and 0.0 <= y <= BOARD_SIZE


def distance_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def distance_planets(first: Planet, second: Planet) -> float:
    return distance_xy(first.x, first.y, second.x, second.y)


def point_to_segment_distance(
    point: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return distance_xy(px, py, sx, sy)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / denom))
    return distance_xy(px, py, sx + t * dx, sy + t * dy)


def segment_circle_intersects(
    start: Tuple[float, float],
    end: Tuple[float, float],
    center: Tuple[float, float],
    radius: float,
) -> bool:
    return point_to_segment_distance(center, start, end) < radius


def fleet_speed(num_ships: int, max_speed: float = DEFAULT_MAX_SHIP_SPEED) -> float:
    if num_ships <= 0:
        return 0.0
    if num_ships == 1:
        return MIN_SHIP_SPEED
    ratio = max(0.0, min(1.0, math.log(num_ships) / LOG_1000))
    return MIN_SHIP_SPEED + (max_speed - MIN_SHIP_SPEED) * ratio**1.5


def launch_point(source_x: float, source_y: float, source_radius: float, angle: float) -> Tuple[float, float]:
    clearance = source_radius + LAUNCH_CLEARANCE
    return (
        source_x + math.cos(angle) * clearance,
        source_y + math.sin(angle) * clearance,
    )


def actual_path_geometry(
    source_x: float,
    source_y: float,
    source_radius: float,
    target_x: float,
    target_y: float,
    target_radius: float,
) -> Tuple[float, float, float, float, float, float]:
    angle = math.atan2(target_y - source_y, target_x - source_x)
    start_x, start_y = launch_point(source_x, source_y, source_radius, angle)
    hit_distance = max(
        0.0,
        distance_xy(source_x, source_y, target_x, target_y) - (source_radius + LAUNCH_CLEARANCE) - target_radius,
    )
    end_x = start_x + math.cos(angle) * hit_distance
    end_y = start_y + math.sin(angle) * hit_distance
    return angle, start_x, start_y, end_x, end_y, hit_distance


def estimate_arrival(
    source_x: float,
    source_y: float,
    source_radius: float,
    target_x: float,
    target_y: float,
    target_radius: float,
    ships: int,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
) -> Optional[Tuple[float, int, float]]:
    if ships <= 0:
        return None
    angle, start_x, start_y, end_x, end_y, hit_distance = actual_path_geometry(
        source_x,
        source_y,
        source_radius,
        target_x,
        target_y,
        target_radius,
    )
    if not point_in_bounds(start_x, start_y) or not point_in_bounds(end_x, end_y):
        return None
    if segment_circle_intersects((start_x, start_y), (end_x, end_y), SUN_CENTER, SUN_RADIUS):
        return None
    speed = fleet_speed(max(1, ships), max_speed=max_speed)
    if speed <= 0.0:
        return None
    eta = max(1, int(math.ceil(hit_distance / speed)))
    return normalize_angle(angle), eta, hit_distance


def as_planets(raw_planets: Sequence[Sequence[Any]]) -> List[Planet]:
    return [planet if isinstance(planet, Planet) else Planet(*planet) for planet in raw_planets]


def as_fleets(raw_fleets: Sequence[Sequence[Any]]) -> List[Fleet]:
    return [fleet if isinstance(fleet, Fleet) else Fleet(*fleet) for fleet in raw_fleets]


def count_players(planets: Sequence[Planet], fleets: Sequence[Fleet]) -> int:
    owners = {planet.owner for planet in planets if planet.owner != -1}
    owners.update(fleet.owner for fleet in fleets)
    return max(2, len(owners))


def is_orbiting_planet(planet: Planet) -> bool:
    orbital_radius = distance_xy(planet.x, planet.y, CENTER, CENTER)
    return orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT


def resolve_arrival_event(owner: int, garrison: float, arrivals: Sequence[Tuple[int, int, int]]) -> Tuple[int, float]:
    by_owner: Dict[int, int] = defaultdict(int)
    for _turn, attacker_owner, ships in arrivals:
        by_owner[int(attacker_owner)] += int(ships)
    if not by_owner:
        return owner, max(0.0, garrison)

    ordered = sorted(by_owner.items(), key=lambda item: item[1], reverse=True)
    top_owner, top_ships = ordered[0]
    if len(ordered) > 1:
        second_ships = ordered[1][1]
        if top_ships == second_ships:
            survivor_owner = -1
            survivor_ships = 0
        else:
            survivor_owner = top_owner
            survivor_ships = top_ships - second_ships
    else:
        survivor_owner = top_owner
        survivor_ships = top_ships

    if survivor_ships <= 0:
        return owner, max(0.0, garrison)
    if owner == survivor_owner:
        return owner, garrison + survivor_ships

    garrison -= survivor_ships
    if garrison < 0:
        return survivor_owner, -garrison
    return owner, garrison


def normalize_arrivals(arrivals: Sequence[Tuple[int, int, int]], horizon: int) -> List[Tuple[int, int, int]]:
    normalized: List[Tuple[int, int, int]] = []
    for turns, owner, ships in arrivals:
        if ships <= 0:
            continue
        eta = max(1, int(math.ceil(turns)))
        if eta > horizon:
            continue
        normalized.append((eta, int(owner), int(ships)))
    normalized.sort(key=lambda item: item[0])
    return normalized


def simulate_planet_timeline(
    planet: Planet,
    arrivals: Sequence[Tuple[int, int, int]],
    player: int,
    horizon: int,
) -> Dict[str, Any]:
    horizon = max(0, int(math.ceil(horizon)))
    by_turn: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for item in normalize_arrivals(arrivals, horizon):
        by_turn[item[0]].append(item)

    owner = int(planet.owner)
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}
    min_owned = garrison if owner == player else 0.0
    first_enemy = None
    fall_turn = None

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production

        group = by_turn.get(turn, [])
        previous_owner = owner
        if group:
            if previous_owner == player and first_enemy is None:
                if any(item[1] not in (-1, player) for item in group):
                    first_enemy = turn
            owner, garrison = resolve_arrival_event(owner, garrison, group)
            if previous_owner == player and owner != player and fall_turn is None:
                fall_turn = turn

        owner_at[turn] = owner
        ships_at[turn] = max(0.0, garrison)
        if owner == player:
            min_owned = min(min_owned, garrison)

    keep_needed = 0
    holds_full = True
    if planet.owner == player:

        def survives_with_keep(keep: int) -> bool:
            sim_owner = int(planet.owner)
            sim_garrison = float(keep)
            for turn in range(1, horizon + 1):
                if sim_owner != -1:
                    sim_garrison += planet.production
                group = by_turn.get(turn, [])
                if group:
                    sim_owner, sim_garrison = resolve_arrival_event(sim_owner, sim_garrison, group)
                    if sim_owner != player:
                        return False
            return sim_owner == player

        if survives_with_keep(int(planet.ships)):
            low, high = 0, int(planet.ships)
            while low < high:
                middle = (low + high) // 2
                if survives_with_keep(middle):
                    high = middle
                else:
                    low = middle + 1
            keep_needed = low
        else:
            holds_full = False
            keep_needed = int(planet.ships)

    return {
        "owner_at": owner_at,
        "ships_at": ships_at,
        "keep_needed": keep_needed,
        "min_owned": max(0, int(math.floor(min_owned))) if planet.owner == player else 0,
        "first_enemy": first_enemy,
        "fall_turn": fall_turn,
        "holds_full": holds_full,
        "horizon": horizon,
    }


def state_at_timeline(timeline: Dict[str, Any], turn: int) -> Tuple[int, float]:
    turn = max(0, int(math.ceil(turn)))
    turn = min(turn, timeline["horizon"])
    owner = timeline["owner_at"].get(turn, timeline["owner_at"][timeline["horizon"]])
    ships = timeline["ships_at"].get(turn, timeline["ships_at"][timeline["horizon"]])
    return int(owner), max(0.0, float(ships))


def reaction_probe_ships(source: Planet, target: Planet) -> int:
    ships = int(target.ships) + 1
    if target.owner != -1:
        ships += int(target.production)
    if not is_orbiting_planet(target):
        ships += 1
    return max(1, min(int(source.ships), ships))


@dataclass
class GameState:
    player: int
    step: int
    remaining_steps: int
    num_players: int
    angular_velocity: float
    max_speed: float
    planets: List[Planet]
    fleets: List[Fleet]
    initial_planets: List[Planet]
    comets: List[Dict[str, Any]]
    my_planets: List[Planet]
    enemy_planets: List[Planet]
    neutral_planets: List[Planet]
    my_fleets: List[Fleet]
    enemy_fleets: List[Fleet]
    planets_by_id: Dict[int, Planet]


class PositionPredictor:
    def __init__(
        self,
        current_step: int,
        angular_velocity: float,
        initial_planets: Sequence[Planet],
        current_planets: Sequence[Planet],
        comets: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        self.current_step = current_step
        self.angular_velocity = angular_velocity
        self.initial_by_id = {planet.id: planet for planet in initial_planets}
        self.current_by_id = {planet.id: planet for planet in current_planets}
        self.comet_by_planet_id: Dict[int, Dict[str, Any]] = {}
        for comet in comets or []:
            for planet_id in comet.get("planet_ids", []):
                self.comet_by_planet_id[int(planet_id)] = comet

    def predict_planet_pos(self, planet: Planet, step_offset: int) -> Tuple[float, float]:
        if planet.id in self.comet_by_planet_id:
            comet_pos = self.predict_comet_pos(self.comet_by_planet_id[planet.id], planet.id, step_offset)
            if comet_pos is not None:
                return comet_pos

        initial = self.initial_by_id.get(planet.id, planet)
        if not is_orbiting_planet(initial):
            current = self.current_by_id.get(planet.id, planet)
            return current.x, current.y

        dx = initial.x - CENTER
        dy = initial.y - CENTER
        orbital_radius = math.hypot(dx, dy)
        initial_angle = math.atan2(dy, dx)
        absolute_step = max(0, self.current_step + step_offset)
        angle = initial_angle + self.angular_velocity * absolute_step
        return (
            CENTER + orbital_radius * math.cos(angle),
            CENTER + orbital_radius * math.sin(angle),
        )

    def predict_comet_pos(
        self,
        comet_data: Dict[str, Any],
        planet_id: int,
        step_offset: int,
    ) -> Optional[Tuple[float, float]]:
        planet_ids = list(comet_data.get("planet_ids", []))
        if planet_id not in planet_ids:
            return None
        comet_index = planet_ids.index(planet_id)
        paths = comet_data.get("paths", [])
        if comet_index >= len(paths):
            return None
        future_index = int(comet_data.get("path_index", 0)) + step_offset
        if future_index < 0 or future_index >= len(paths[comet_index]):
            return None
        x, y = paths[comet_index][future_index]
        return float(x), float(y)

    def predict_target_pos(self, planet: Planet, step_offset: int) -> Optional[Tuple[float, float]]:
        if planet.id in self.comet_by_planet_id:
            return self.predict_comet_pos(self.comet_by_planet_id[planet.id], planet.id, step_offset)
        return self.predict_planet_pos(planet, step_offset)

    def comet_remaining_life(self, planet_id: int) -> int:
        comet = self.comet_by_planet_id.get(planet_id)
        if comet is None:
            return 0
        planet_ids = list(comet.get("planet_ids", []))
        if planet_id not in planet_ids:
            return 0
        comet_index = planet_ids.index(planet_id)
        paths = comet.get("paths", [])
        if comet_index >= len(paths):
            return 0
        path_index = int(comet.get("path_index", 0))
        return max(0, len(paths[comet_index]) - path_index)

    def target_can_move(self, planet: Planet) -> bool:
        if planet.id in self.comet_by_planet_id:
            return True
        initial = self.initial_by_id.get(planet.id)
        return bool(initial and is_orbiting_planet(initial))


class InterceptSolver:
    def __init__(
        self,
        predictor: PositionPredictor,
        max_speed: float = DEFAULT_MAX_SHIP_SPEED,
        deadline: Optional[float] = None,
    ) -> None:
        self.predictor = predictor
        self.max_speed = max_speed
        self.deadline = deadline

    def expired(self) -> bool:
        return self.deadline is not None and time.time() >= self.deadline

    def _estimate_to_position(
        self,
        from_x: float,
        from_y: float,
        from_radius: float,
        to_x: float,
        to_y: float,
        to_radius: float,
        ships: int,
    ) -> Optional[Tuple[float, int]]:
        result = estimate_arrival(
            from_x,
            from_y,
            from_radius,
            to_x,
            to_y,
            to_radius,
            ships,
            max_speed=self.max_speed,
        )
        if result is None:
            return None
        angle, eta, _distance = result
        return normalize_angle(angle), eta

    def _validate_intercept(
        self,
        from_x: float,
        from_y: float,
        from_radius: float,
        angle: float,
        ships: int,
        target_planet: Planet,
        max_turns: int,
    ) -> Optional[int]:
        speed = fleet_speed(ships, max_speed=self.max_speed)
        if speed <= 0.0:
            return None
        planets = list(self.predictor.current_by_id.values())
        prev_x, prev_y = launch_point(from_x, from_y, from_radius, angle)
        dx = math.cos(angle) * speed
        dy = math.sin(angle) * speed
        for turn in range(1, max(1, int(max_turns)) + 1):
            next_x = prev_x + dx
            next_y = prev_y + dy
            if not point_in_bounds(next_x, next_y):
                return None
            if segment_circle_intersects((prev_x, prev_y), (next_x, next_y), SUN_CENTER, SUN_RADIUS):
                return None
            for planet in planets:
                planet_pos = self.predictor.predict_target_pos(planet, turn - 1)
                if planet_pos is None:
                    continue
                if segment_circle_intersects((prev_x, prev_y), (next_x, next_y), planet_pos, planet.radius):
                    return turn if planet.id == target_planet.id else None
            for planet in planets:
                old_pos = self.predictor.predict_target_pos(planet, turn - 1)
                new_pos = self.predictor.predict_target_pos(planet, turn)
                if old_pos is None or new_pos is None or old_pos == new_pos:
                    continue
                if point_to_segment_distance((next_x, next_y), old_pos, new_pos) < planet.radius:
                    return turn if planet.id == target_planet.id else None
            prev_x, prev_y = next_x, next_y
        return None

    def _search_safe_intercept(
        self,
        from_x: float,
        from_y: float,
        from_radius: float,
        target_planet: Planet,
        ships: int,
    ) -> Optional[Tuple[float, int]]:
        max_turns = MAX_INTERCEPT_TURNS
        if target_planet.id in self.predictor.comet_by_planet_id:
            max_turns = min(max_turns, max(1, self.predictor.comet_remaining_life(target_planet.id) - 1))

        best: Optional[Tuple[float, int, int]] = None
        for candidate_turn in range(1, max_turns + 1):
            if self.expired():
                break
            future_pos = self.predictor.predict_target_pos(target_planet, candidate_turn)
            if future_pos is None:
                continue
            estimate = self._estimate_to_position(
                from_x,
                from_y,
                from_radius,
                future_pos[0],
                future_pos[1],
                target_planet.radius,
                ships,
            )
            if estimate is None:
                continue
            angle, eta = estimate
            if abs(eta - candidate_turn) > 1:
                continue
            hit_turn = self._validate_intercept(
                from_x,
                from_y,
                from_radius,
                angle,
                ships,
                target_planet,
                max(candidate_turn, eta) + 1,
            )
            if hit_turn is None:
                continue
            score = (abs(hit_turn - candidate_turn), hit_turn, candidate_turn)
            if best is None or score < best:
                best = (angle, hit_turn, candidate_turn)
        return None if best is None else (best[0], best[1])

    def solve_intercept(
        self,
        from_x: float,
        from_y: float,
        num_ships: int,
        target_planet: Planet,
        current_step: int,
        angular_velocity: float,
        initial_planets: Sequence[Planet],
        from_radius: float = 0.0,
    ) -> Optional[Tuple[float, int]]:
        del angular_velocity, initial_planets
        if num_ships <= 0:
            return None

        original_step = self.predictor.current_step
        self.predictor.current_step = current_step
        try:
            fixed = self._estimate_to_position(
                from_x,
                from_y,
                from_radius,
                target_planet.x,
                target_planet.y,
                target_planet.radius,
                num_ships,
            )
            if fixed is None and not self.predictor.target_can_move(target_planet):
                return None
            if fixed is None:
                return self._search_safe_intercept(from_x, from_y, from_radius, target_planet, num_ships)

            angle, eta = fixed
            target_x, target_y = target_planet.x, target_planet.y
            for _ in range(5):
                if self.expired():
                    break
                future_pos = self.predictor.predict_target_pos(target_planet, eta)
                if future_pos is None:
                    return None
                next_estimate = self._estimate_to_position(
                    from_x,
                    from_y,
                    from_radius,
                    future_pos[0],
                    future_pos[1],
                    target_planet.radius,
                    num_ships,
                )
                if next_estimate is None:
                    return self._search_safe_intercept(from_x, from_y, from_radius, target_planet, num_ships)
                next_angle, next_eta = next_estimate
                if (
                    abs(future_pos[0] - target_x) < 0.3
                    and abs(future_pos[1] - target_y) < 0.3
                    and abs(next_eta - eta) <= 1
                ):
                    hit_turn = self._validate_intercept(
                        from_x,
                        from_y,
                        from_radius,
                        next_angle,
                        num_ships,
                        target_planet,
                        max(next_eta, eta) + 1,
                    )
                    if hit_turn is not None and abs(hit_turn - next_eta) <= 1:
                        return next_angle, hit_turn
                angle, eta = next_angle, next_eta
                target_x, target_y = future_pos
            hit_turn = self._validate_intercept(
                from_x,
                from_y,
                from_radius,
                angle,
                num_ships,
                target_planet,
                eta + 1,
            )
            if hit_turn is not None and abs(hit_turn - eta) <= 1:
                return angle, hit_turn
            return self._search_safe_intercept(from_x, from_y, from_radius, target_planet, num_ships)
        finally:
            self.predictor.current_step = original_step


def estimate_fleet_arrival_turn(
    fleet: Fleet,
    planet: Planet,
    predictor: PositionPredictor,
    turns_ahead: int,
    max_speed: float,
) -> Optional[int]:
    speed = fleet_speed(fleet.ships, max_speed=max_speed)
    if speed <= 0.0:
        return None

    prev_x = fleet.x
    prev_y = fleet.y
    dx = math.cos(fleet.angle) * speed
    dy = math.sin(fleet.angle) * speed

    for turn in range(1, turns_ahead + 1):
        next_x = prev_x + dx
        next_y = prev_y + dy
        if not point_in_bounds(next_x, next_y):
            return None
        if segment_circle_intersects((prev_x, prev_y), (next_x, next_y), SUN_CENTER, SUN_RADIUS):
            return None
        target_pos = predictor.predict_target_pos(planet, turn)
        if target_pos is None:
            return None
        if segment_circle_intersects((prev_x, prev_y), (next_x, next_y), target_pos, planet.radius):
            return turn
        prev_x, prev_y = next_x, next_y
    return None


class ProjectedWorld:
    def __init__(
        self,
        state: GameState,
        predictor: PositionPredictor,
        deadline: Optional[float] = None,
    ) -> None:
        self.state = state
        self.player = state.player
        self.predictor = predictor
        self.deadline = deadline
        self.arrivals_by_planet: Dict[int, List[Tuple[int, int, int]]] = {planet.id: [] for planet in state.planets}
        self.base_timeline: Dict[int, Dict[str, Any]] = {}
        self.timeline_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for fleet in state.fleets:
            if self.expired():
                break
            for planet in state.planets:
                eta = estimate_fleet_arrival_turn(
                    fleet,
                    planet,
                    predictor,
                    turns_ahead=BASE_TIMELINE_HORIZON,
                    max_speed=state.max_speed,
                )
                if eta is not None:
                    self.arrivals_by_planet[planet.id].append((eta, fleet.owner, int(fleet.ships)))

        for planet in state.planets:
            self.arrivals_by_planet[planet.id].sort(key=lambda item: item[0])
            self.base_timeline[planet.id] = simulate_planet_timeline(
                planet,
                self.arrivals_by_planet[planet.id],
                self.player,
                BASE_TIMELINE_HORIZON,
            )

    def expired(self) -> bool:
        return self.deadline is not None and time.time() >= self.deadline

    def projected_timeline(
        self,
        target_id: int,
        horizon: int,
        extra_arrivals: Sequence[Tuple[int, int, int]] = (),
    ) -> Dict[str, Any]:
        horizon = max(1, int(math.ceil(horizon)))
        if not extra_arrivals and horizon <= BASE_TIMELINE_HORIZON:
            return self.base_timeline[target_id]

        cache_key = (target_id, horizon)
        if not extra_arrivals and cache_key in self.timeline_cache:
            return self.timeline_cache[cache_key]

        arrivals = [item for item in self.arrivals_by_planet.get(target_id, []) if item[0] <= horizon]
        arrivals.extend(item for item in extra_arrivals if item[0] <= horizon)
        timeline = simulate_planet_timeline(self.state.planets_by_id[target_id], arrivals, self.player, horizon)
        if not extra_arrivals:
            self.timeline_cache[cache_key] = timeline
        return timeline

    def projected_state(
        self,
        target_id: int,
        turn: int,
        extra_arrivals: Sequence[Tuple[int, int, int]] = (),
    ) -> Tuple[int, float]:
        timeline = self.projected_timeline(target_id, turn, extra_arrivals=extra_arrivals)
        return state_at_timeline(timeline, turn)

    def hold_status(self, target_id: int, horizon: int) -> Dict[str, Any]:
        timeline = self.projected_timeline(target_id, horizon)
        return {
            "keep_needed": timeline["keep_needed"],
            "min_owned": timeline["min_owned"],
            "first_enemy": timeline["first_enemy"],
            "fall_turn": timeline["fall_turn"],
            "holds_full": timeline["holds_full"],
        }


class ForwardSimulator:
    def __init__(self, player_id: int, max_speed: float = DEFAULT_MAX_SHIP_SPEED) -> None:
        self.player_id = player_id
        self.max_speed = max_speed

    def simulate(
        self,
        planets: Sequence[Planet],
        fleets: Sequence[Fleet],
        angular_velocity: float,
        initial_planets: Sequence[Planet],
        num_turns: int,
    ) -> List[Tuple[int, int]]:
        planet_state = {
            planet.id: [planet.id, planet.owner, planet.x, planet.y, planet.radius, planet.ships, planet.production]
            for planet in planets
        }
        initial_by_id = {planet.id: planet for planet in initial_planets}
        fleet_state = [
            [fleet.id, fleet.owner, fleet.x, fleet.y, fleet.angle, fleet.from_planet_id, fleet.ships]
            for fleet in fleets
        ]
        history: List[Tuple[int, int]] = []

        for sim_turn in range(1, num_turns + 1):
            for planet in planet_state.values():
                if planet[1] != -1:
                    planet[5] += planet[6]

            arrivals: Dict[int, List[List[Any]]] = defaultdict(list)
            survivors: List[List[Any]] = []
            for fleet in fleet_state:
                speed = fleet_speed(int(fleet[6]), max_speed=self.max_speed)
                old_pos = (fleet[2], fleet[3])
                new_pos = (
                    fleet[2] + math.cos(fleet[4]) * speed,
                    fleet[3] + math.sin(fleet[4]) * speed,
                )
                if not point_in_bounds(*new_pos):
                    continue
                if segment_circle_intersects(old_pos, new_pos, SUN_CENTER, SUN_RADIUS):
                    continue

                hit_planet_id: Optional[int] = None
                for planet in planet_state.values():
                    if segment_circle_intersects(old_pos, new_pos, (planet[2], planet[3]), planet[4]):
                        hit_planet_id = int(planet[0])
                        break
                if hit_planet_id is None:
                    fleet[2], fleet[3] = new_pos
                    survivors.append(fleet)
                else:
                    arrivals[hit_planet_id].append(fleet)

            fleet_state = survivors
            for planet_id, incoming in arrivals.items():
                planet = planet_state.get(planet_id)
                if planet is None:
                    continue
                owner, ships = resolve_arrival_event(
                    int(planet[1]),
                    float(planet[5]),
                    [(1, int(fleet[1]), int(fleet[6])) for fleet in incoming],
                )
                planet[1] = owner
                planet[5] = int(round(ships))

            for planet in planet_state.values():
                initial = initial_by_id.get(int(planet[0]))
                if initial is None or not is_orbiting_planet(initial):
                    continue
                dx = initial.x - CENTER
                dy = initial.y - CENTER
                radius = math.hypot(dx, dy)
                start_angle = math.atan2(dy, dx)
                current_angle = start_angle + angular_velocity * sim_turn
                planet[2] = CENTER + radius * math.cos(current_angle)
                planet[3] = CENTER + radius * math.sin(current_angle)

            my_total = 0
            enemy_total = 0
            for planet in planet_state.values():
                if planet[1] == self.player_id:
                    my_total += int(planet[5])
                elif planet[1] != -1:
                    enemy_total += int(planet[5])
            for fleet in fleet_state:
                if fleet[1] == self.player_id:
                    my_total += int(fleet[6])
                else:
                    enemy_total += int(fleet[6])
            history.append((my_total, enemy_total))

        return history


@dataclass(frozen=True)
class LaunchOption:
    source_id: int
    target_id: int
    ships: int
    angle: float
    eta: int
    mission: str


@dataclass
class StepPlan:
    mission: str
    target_id: int
    launches: List[LaunchOption]
    eval_turn: int
    ships_sent: int
    score: float
    blocks_target: bool = True


def build_send_buckets(surplus: int) -> List[int]:
    if surplus <= 0:
        return []
    values = {max(1, int(math.ceil(surplus * ratio))) for ratio in SEND_BUCKET_RATIOS}
    values.add(surplus)
    return sorted(v for v in values if v > 0)


class DecisionLogic:
    def __init__(self, obs: Any, config: Any) -> None:
        self.obs = obs
        self.config = config
        self.start_time = time.time()
        self.deadline = self.start_time + SOFT_ACT_DEADLINE
        self.state = self._build_state(obs, config)
        self.predictor = PositionPredictor(
            current_step=self.state.step,
            angular_velocity=self.state.angular_velocity,
            initial_planets=self.state.initial_planets,
            current_planets=self.state.planets,
            comets=self.state.comets,
        )
        self.intercept_solver = InterceptSolver(
            self.predictor,
            max_speed=self.state.max_speed,
            deadline=self.deadline,
        )
        self.world = ProjectedWorld(self.state, self.predictor, deadline=self.deadline)
        self.forward_simulator = ForwardSimulator(self.state.player, max_speed=self.state.max_speed)
        self.source_commitments: Dict[int, int] = defaultdict(int)
        self.used_source_ids: set[int] = set()
        self.shot_cache: Dict[Tuple[int, int, int], Optional[Tuple[float, int]]] = {}
        self.surplus_cache: Dict[Tuple[int, int], int] = {}
        self.reaction_cache: Dict[int, int] = self._build_enemy_reaction_map()

    def expired(self) -> bool:
        return time.time() >= self.deadline

    def decide(self) -> List[List[float | int]]:
        steps = self._build_all_steps()
        chosen_steps = self._choose_move_set(steps)
        launches = self._flatten_launches(chosen_steps)
        return self._finalize(launches)

    def _build_state(self, obs: Any, config: Any) -> GameState:
        player = int(get_field(obs, "player", 0))
        step = int(get_field(obs, "step", 0) or 0)
        angular_velocity = float(get_field(obs, "angular_velocity", 0.0) or 0.0)
        max_speed = float(get_field(config, "shipSpeed", DEFAULT_MAX_SHIP_SPEED))

        planets = as_planets(get_field(obs, "planets", []) or [])
        fleets = as_fleets(get_field(obs, "fleets", []) or [])
        initial_planets = as_planets(get_field(obs, "initial_planets", get_field(obs, "planets", [])) or [])
        comets = list(get_field(obs, "comets", []) or [])

        my_planets = [planet for planet in planets if planet.owner == player]
        enemy_planets = [planet for planet in planets if planet.owner not in (-1, player)]
        neutral_planets = [planet for planet in planets if planet.owner == -1]
        my_fleets = [fleet for fleet in fleets if fleet.owner == player]
        enemy_fleets = [fleet for fleet in fleets if fleet.owner != player]

        return GameState(
            player=player,
            step=step,
            remaining_steps=max(1, TOTAL_STEPS - step),
            num_players=count_players(planets, fleets),
            angular_velocity=angular_velocity,
            max_speed=max_speed,
            planets=planets,
            fleets=fleets,
            initial_planets=initial_planets,
            comets=comets,
            my_planets=my_planets,
            enemy_planets=enemy_planets,
            neutral_planets=neutral_planets,
            my_fleets=my_fleets,
            enemy_fleets=enemy_fleets,
            planets_by_id={planet.id: planet for planet in planets},
        )

    def _build_enemy_reaction_map(self) -> Dict[int, int]:
        reaction: Dict[int, int] = {}
        for target in self.state.neutral_planets + self.state.my_planets:
            best_eta: Optional[int] = None
            for enemy in self.state.enemy_planets:
                probe = max(1, min(int(enemy.ships), reaction_probe_ships(enemy, target)))
                shot = self._plan_shot(enemy, target, probe)
                if shot is None:
                    continue
                if best_eta is None or shot[1] < best_eta:
                    best_eta = shot[1]
            if best_eta is not None:
                reaction[target.id] = best_eta
        return reaction

    def _effective_planet(self, planet: Planet) -> Planet:
        committed = self.source_commitments.get(planet.id, 0)
        if committed <= 0:
            return planet
        return planet._replace(ships=max(0, planet.ships - committed))

    def _planet_surplus(self, planet: Planet) -> int:
        cache_key = (planet.id, self.source_commitments.get(planet.id, 0))
        if cache_key in self.surplus_cache:
            return self.surplus_cache[cache_key]
        effective = self._effective_planet(planet)
        if effective.owner != self.state.player:
            self.surplus_cache[cache_key] = 0
            return 0
        hold = self.world.hold_status(planet.id, SURPLUS_HORIZON)
        if not hold["holds_full"]:
            self.surplus_cache[cache_key] = 0
            return 0
        reserve = max(int(hold["keep_needed"]), 2 + int(effective.production))
        if self.state.num_players >= 4 and self.state.enemy_planets:
            nearest_enemy = min(distance_planets(effective, enemy) for enemy in self.state.enemy_planets)
            if nearest_enemy < 28.0:
                reserve = max(reserve, 4 + int(effective.production) + int(0.15 * nearest_enemy))
        surplus = max(0, int(effective.ships) - reserve)
        self.surplus_cache[cache_key] = surplus
        return surplus

    def _available_donors(self, excluded_ids: Iterable[int] = ()) -> List[Planet]:
        excluded = set(excluded_ids)
        donors = []
        for planet in self.state.my_planets:
            if planet.id in self.used_source_ids or planet.id in excluded:
                continue
            effective = self._effective_planet(planet)
            if self._planet_surplus(effective) >= MIN_SURPLUS_TO_SEND:
                donors.append(effective)
        return donors

    def _plan_shot(self, source: Planet, target: Planet, ships: int) -> Optional[Tuple[float, int]]:
        key = (source.id, target.id, int(ships))
        if key not in self.shot_cache:
            self.shot_cache[key] = self.intercept_solver.solve_intercept(
                source.x,
                source.y,
                int(ships),
                target,
                self.state.step,
                self.state.angular_velocity,
                self.state.initial_planets,
                from_radius=source.radius,
            )
        return self.shot_cache[key]

    def _candidate_launches(self, target: Planet, mission: str, excluded_ids: Iterable[int] = ()) -> List[LaunchOption]:
        options: List[LaunchOption] = []
        max_eta = MAX_DEFEND_ETA if mission == "defend" else (MAX_EXPAND_ETA if mission == "expand" else MAX_ATTACK_ETA)
        donors = sorted(self._available_donors(excluded_ids=excluded_ids), key=lambda p: distance_planets(p, target))
        for source in donors:
            if self.expired():
                break
            surplus = self._planet_surplus(source)
            for send in build_send_buckets(surplus):
                if send < MIN_SURPLUS_TO_SEND:
                    continue
                shot = self._plan_shot(source, target, send)
                if shot is None or shot[1] > max_eta:
                    continue
                options.append(LaunchOption(source.id, target.id, send, shot[0], shot[1], mission))
        options.sort(key=lambda opt: (opt.eta, -opt.ships, opt.source_id))
        return options

    def _capture_progress(self, target_id: int, eval_turn: int, launches: Sequence[LaunchOption]) -> Tuple[bool, int]:
        arrivals = tuple((launch.eta, self.state.player, launch.ships) for launch in launches)
        owner, _ships = self.world.projected_state(target_id, eval_turn, extra_arrivals=arrivals)
        return owner == self.state.player, sum(launch.ships for launch in launches)

    def _defense_progress(self, target_id: int, eval_turn: int, launches: Sequence[LaunchOption], hold_until: int) -> Tuple[bool, int]:
        arrivals = tuple((launch.eta, self.state.player, launch.ships) for launch in launches)
        timeline = self.world.projected_timeline(target_id, hold_until, extra_arrivals=arrivals)
        success = all(timeline["owner_at"].get(turn) == self.state.player for turn in range(eval_turn, hold_until + 1))
        return success, sum(launch.ships for launch in launches)

    def _reaction_gap(self, target: Planet, eval_turn: int) -> int:
        reaction_eta = self.reaction_cache.get(target.id)
        if reaction_eta is None:
            return 6
        return reaction_eta - eval_turn

    def _score_capture_step(self, target: Planet, mission: str, eval_turn: int, launches: Sequence[LaunchOption]) -> float:
        ships_sent = sum(launch.ships for launch in launches)
        if ships_sent <= 0:
            return -float("inf")
        horizon_value = max(8, self.state.remaining_steps - eval_turn)
        reaction_gap = self._reaction_gap(target, eval_turn)
        value = float(target.production) * float(horizon_value)
        if mission == "attack":
            value += 0.35 * float(target.ships)
            if self.state.step < MIN_ATTACK_STEP:
                value *= 0.65
        else:
            value *= 1.08 if not is_orbiting_planet(target) else 0.96
        if reaction_gap > 0:
            value *= 1.0 + min(0.25, 0.04 * reaction_gap)
        else:
            value *= max(0.55, 1.0 + 0.08 * reaction_gap)
        if self.state.num_players >= 4 and mission == "attack":
            value *= 0.9
        cost = ships_sent * (1.0 + 0.035 * eval_turn)
        return value / max(1.0, cost)

    def _score_defense_step(self, target: Planet, eval_turn: int, launches: Sequence[LaunchOption], hold_until: int) -> float:
        ships_sent = sum(launch.ships for launch in launches)
        if ships_sent <= 0:
            return -float("inf")
        saved_window = max(4, hold_until - eval_turn + 1)
        value = (target.production * saved_window) + 0.5 * target.ships
        if self.state.num_players >= 4:
            value *= 1.1
        cost = ships_sent * (1.0 + 0.025 * eval_turn)
        return value / max(1.0, cost)

    def _choose_capture_step(self, target: Planet, mission: str) -> Optional[StepPlan]:
        options = self._candidate_launches(target, mission)
        if not options:
            return None
        eval_turns: List[int] = []
        for option in options[: min(len(options), 12)]:
            eval_turns.extend([option.eta, option.eta + 1])
        eval_turns = sorted(set(eval_turns))

        best: Optional[StepPlan] = None
        for eval_turn in eval_turns:
            if self.expired():
                break
            eligible = [opt for opt in options if opt.eta <= eval_turn + ETA_SYNC_TOLERANCE]
            eligible.sort(key=lambda opt: (abs(opt.eta - eval_turn), -(opt.ships / max(1, opt.eta)), opt.source_id))
            chosen: List[LaunchOption] = []
            used_sources: set[int] = set()
            for option in eligible:
                if option.source_id in used_sources:
                    continue
                chosen.append(option)
                used_sources.add(option.source_id)
                success, ships_sent = self._capture_progress(target.id, eval_turn, chosen)
                if success:
                    score = self._score_capture_step(target, mission, eval_turn, chosen)
                    candidate = StepPlan(mission, target.id, list(chosen), eval_turn, ships_sent, score, True)
                    if best is None or candidate.score > best.score:
                        best = candidate
                    break
        return best

    def _choose_defense_step(self, target: Planet, hold_until: int) -> Optional[StepPlan]:
        options = self._candidate_launches(target, "defend", excluded_ids=[target.id])
        if not options:
            return None
        eval_turns = sorted(set(opt.eta for opt in options))
        best: Optional[StepPlan] = None
        for eval_turn in eval_turns:
            if self.expired():
                break
            eligible = [opt for opt in options if opt.eta <= eval_turn + ETA_SYNC_TOLERANCE]
            eligible.sort(key=lambda opt: (abs(opt.eta - eval_turn), -(opt.ships / max(1, opt.eta)), opt.source_id))
            chosen: List[LaunchOption] = []
            used_sources: set[int] = set()
            for option in eligible:
                if option.source_id in used_sources:
                    continue
                chosen.append(option)
                used_sources.add(option.source_id)
                success, ships_sent = self._defense_progress(target.id, eval_turn, chosen, hold_until)
                if success:
                    score = self._score_defense_step(target, eval_turn, chosen, hold_until)
                    candidate = StepPlan("defend", target.id, list(chosen), eval_turn, ships_sent, score, False)
                    if best is None or candidate.score > best.score:
                        best = candidate
                    break
        return best

    def _simple_target_value(self, target: Planet, mission: str) -> float:
        my_distance = min(distance_planets(source, target) for source in self.state.my_planets) if self.state.my_planets else 100.0
        value = float(target.production) / max(1.0, float(target.ships))
        value /= max(1.0, my_distance / 12.0)
        if mission == "attack":
            value += 0.25 * float(target.production)
        if not is_orbiting_planet(target):
            value *= 1.1
        return value

    def _build_defense_steps(self) -> List[StepPlan]:
        steps: List[StepPlan] = []
        threatened: List[Tuple[float, Planet, int]] = []
        for planet in self.state.my_planets:
            hold = self.world.hold_status(planet.id, DEFENSE_LOOKAHEAD)
            if hold["fall_turn"] is None:
                continue
            threatened.append((float(hold["fall_turn"]), planet, int(hold["fall_turn"])))
        threatened.sort(key=lambda item: item[0])
        for _fall, planet, hold_until in threatened[:4]:
            if self.expired():
                break
            step = self._choose_defense_step(planet, hold_until)
            if step is not None:
                steps.append(step)
        return steps

    def _build_capture_steps(self, targets: Sequence[Planet], mission: str) -> List[StepPlan]:
        scored_targets = sorted(targets, key=lambda target: self._simple_target_value(target, mission), reverse=True)
        limit = OPENING_TARGET_CAP if self._is_opening_search() else TARGET_CAP
        steps: List[StepPlan] = []
        for target in scored_targets[:limit]:
            if self.expired():
                break
            step = self._choose_capture_step(target, mission)
            if step is not None:
                steps.append(step)
        return steps

    def _build_all_steps(self) -> List[StepPlan]:
        steps = self._build_defense_steps()
        steps.extend(self._build_capture_steps(self.state.neutral_planets, "expand"))
        if self.state.step >= MIN_ATTACK_STEP or len(self.state.my_planets) >= OPENING_SEARCH_PLANET_CAP:
            steps.extend(self._build_capture_steps(self.state.enemy_planets, "attack"))
        steps.sort(key=lambda step: step.score, reverse=True)
        return steps

    def _launch_conflict(self, step: StepPlan, used_sources: set[int], targeted: set[int]) -> bool:
        if any(launch.source_id in used_sources for launch in step.launches):
            return True
        return step.blocks_target and step.target_id in targeted

    def _flatten_launches(self, steps: Sequence[StepPlan]) -> List[LaunchOption]:
        launches: List[LaunchOption] = []
        for step in steps:
            launches.extend(step.launches)
        return launches

    def _simulate_future_delta(self, launches: Sequence[LaunchOption], lookahead: int) -> float:
        planets = [planet._replace(ships=max(0, planet.ships - self.source_commitments.get(planet.id, 0))) for planet in self.state.planets]
        planet_map = {planet.id: planet for planet in planets}
        for launch in launches:
            source = planet_map.get(launch.source_id)
            if source is None:
                continue
            if source.owner != self.state.player or launch.ships > source.ships:
                return -1e9
            planet_map[launch.source_id] = source._replace(ships=source.ships - launch.ships)

        new_planets = list(planet_map.values())
        new_fleets = list(self.state.fleets)
        next_fleet_id = -1
        for launch in launches:
            source = self.state.planets_by_id.get(launch.source_id)
            if source is None:
                continue
            start_x, start_y = launch_point(source.x, source.y, source.radius, launch.angle)
            new_fleets.append(Fleet(next_fleet_id, self.state.player, start_x, start_y, launch.angle, source.id, launch.ships))
            next_fleet_id -= 1

        history = self.forward_simulator.simulate(
            new_planets,
            new_fleets,
            self.state.angular_velocity,
            self.state.initial_planets,
            lookahead,
        )
        if not history:
            return 0.0
        my_total, enemy_total = history[-1]
        return float(my_total - enemy_total)

    def _evaluate_step_set(self, steps: Sequence[StepPlan]) -> float:
        launches = self._flatten_launches(steps)
        immediate = sum(step.score for step in steps)
        lookahead = OPENING_SEARCH_LOOKAHEAD if self._is_opening_search() else MIDGAME_SEARCH_LOOKAHEAD
        future_delta = self._simulate_future_delta(launches, lookahead)
        return immediate + 0.01 * future_delta

    def _choose_move_set(self, steps: Sequence[StepPlan]) -> List[StepPlan]:
        if not steps:
            return []
        candidates: List[List[StepPlan]] = [[]]
        step_cap = min(MOVE_CANDIDATE_CAP, len(steps))
        for start_idx in range(step_cap):
            seed = steps[start_idx]
            used_sources = {launch.source_id for launch in seed.launches}
            targeted = {seed.target_id} if seed.blocks_target else set()
            combo = [seed]
            for follower in steps:
                if follower is seed or len(combo) >= MOVE_STEP_CAP:
                    continue
                if self._launch_conflict(follower, used_sources, targeted):
                    continue
                combo.append(follower)
                used_sources.update(launch.source_id for launch in follower.launches)
                if follower.blocks_target:
                    targeted.add(follower.target_id)
            candidates.append(combo)

        best_steps: List[StepPlan] = []
        best_value = -float("inf")
        for candidate in candidates:
            if self.expired():
                break
            value = self._evaluate_step_set(candidate)
            if value > best_value:
                best_value = value
                best_steps = candidate
        return best_steps

    def _is_opening_search(self) -> bool:
        return self.state.step < 36 or len(self.state.my_planets) < OPENING_SEARCH_PLANET_CAP

    def _finalize(self, launches: Sequence[LaunchOption]) -> List[List[float | int]]:
        moves: List[List[float | int]] = []
        used_sources: set[int] = set()
        current_by_id = {planet.id: planet for planet in self.state.my_planets}
        for launch in launches:
            if launch.source_id in used_sources:
                continue
            source = current_by_id.get(launch.source_id)
            if source is None or launch.ships <= 0 or launch.ships > source.ships:
                continue
            moves.append([launch.source_id, float(launch.angle), int(launch.ships)])
            used_sources.add(launch.source_id)
        return moves


def agent(obs: Any, config: Any) -> List[List[float | int]]:
    try:
        return DecisionLogic(obs, config).decide()
    except Exception:
        return []


__all__ = [
    "Planet",
    "Fleet",
    "GameState",
    "LaunchOption",
    "StepPlan",
    "DecisionLogic",
    "build_send_buckets",
    "agent",
]
