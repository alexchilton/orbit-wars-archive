from __future__ import annotations

import math
import time
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet = namedtuple("Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])

BOARD_SIZE = 100.0
CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
SUN_CENTER = (CENTER, CENTER)
SUN_RADIUS = 10.0
TOTAL_STEPS = 500

MIN_SHIP_SPEED = 1.0
DEFAULT_MAX_SHIP_SPEED = 6.0
LOG_1000 = math.log(1000.0)
LAUNCH_CLEARANCE = 0.1

MAX_INTERCEPT_TURNS = 120
BASE_TIMELINE_HORIZON = 90
DEFENSE_HORIZON = 28
COMET_LOOKAHEAD = 24
PRESSURE_DISTANCE = 20.0

EARLY_TURN_LIMIT = 40
OPENING_TURN_LIMIT = 80
LATE_REMAINING_TURNS = 60
VERY_LATE_REMAINING_TURNS = 25
SOFT_ACT_DEADLINE = 0.82

MAX_TURN_LAUNCHES = 5
SWARM_ETA_TOLERANCE = 1
MULTI_SOURCE_TOP_K = 4
OPENING_MIN_PRODUCTION = 2
EVAC_MIN_SHIPS = 8

REACTION_TOP_K = 3
SAFE_NEUTRAL_MARGIN = 2
CONTESTED_NEUTRAL_MARGIN = 2

BEHIND_THRESHOLD = -0.20
AHEAD_THRESHOLD = 0.18
FINISHING_THRESHOLD = 0.35
FINISHING_PROD_RATIO = 1.25
AHEAD_MARGIN_BONUS = 0.08
BEHIND_MARGIN_PENALTY = 0.05
FINISHING_MARGIN_BONUS = 0.08

CRASH_EXPLOIT_MIN_SHIPS = 10
CRASH_EXPLOIT_ETA_WINDOW = 2
CRASH_EXPLOIT_POST_DELAY = 1

LATE_CAPTURE_BUFFER = 5
VERY_LATE_CAPTURE_BUFFER = 3

ATTACK_COST_WEIGHT = 0.8
SNIPE_COST_WEIGHT = 1.2

STATIC_VALUE_MULT = 1.4
HOSTILE_VALUE_MULT = 1.85
OPENING_HOSTILE_VALUE_MULT = 1.45
SAFE_NEUTRAL_VALUE_MULT = 1.2
CONTESTED_NEUTRAL_VALUE_MULT = 0.7
CRASH_EXPLOIT_VALUE_MULT = 1.18
SNIPE_VALUE_MULT = 1.12
SWARM_VALUE_MULT = 1.05
REINFORCE_VALUE_MULT = 1.35
COMET_VALUE_MULT = 0.65
EARLY_NEUTRAL_VALUE_MULT = 1.2
LATE_SHIP_VALUE = 0.6
ELIMINATION_BONUS = 18.0
WEAK_ENEMY_THRESHOLD = 45

STATIC_SCORE_MULT = 1.18
EARLY_STATIC_SCORE_MULT = 1.25
SNIPE_SCORE_MULT = 1.12
SWARM_SCORE_MULT = 1.06
CRASH_EXPLOIT_SCORE_MULT = 1.05

NEUTRAL_MARGIN_BASE = 2
HOSTILE_MARGIN_BASE = 3
MARGIN_PROD_WEIGHT = 2
NEUTRAL_MARGIN_CAP = 8
HOSTILE_MARGIN_CAP = 12
STATIC_MARGIN = 4
CONTESTED_MARGIN = 5
FINISHING_SEND_BONUS = 3

AGENT_MEMORY: Dict[str, Any] = {
    "last_owners": {},
    "last_step": None,
    "player": None,
    "last_error": None,
}


def get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict-like observation or an attribute object."""

    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the [0, 2π) interval."""

    return angle % (2.0 * math.pi)


def point_in_bounds(x: float, y: float) -> bool:
    """Return whether a point lies inside the board."""

    return 0.0 <= x <= BOARD_SIZE and 0.0 <= y <= BOARD_SIZE


def distance_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return Euclidean distance between two points."""

    return math.hypot(x1 - x2, y1 - y2)


def distance_planets(first: Planet, second: Planet) -> float:
    """Return center-to-center distance between two planets."""

    return distance_xy(first.x, first.y, second.x, second.y)


def point_to_segment_distance(
    point: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    """Compute the minimum distance from a point to a segment."""

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
    """Return whether a segment intersects a circle."""

    return point_to_segment_distance(center, start, end) < radius


def fleet_speed(num_ships: int, max_speed: float = DEFAULT_MAX_SHIP_SPEED) -> float:
    """Return the Orbit Wars fleet speed for a ship count."""

    if num_ships <= 0:
        return 0.0
    if num_ships == 1:
        return MIN_SHIP_SPEED
    ratio = max(0.0, min(1.0, math.log(num_ships) / LOG_1000))
    return MIN_SHIP_SPEED + (max_speed - MIN_SHIP_SPEED) * ratio**1.5


def launch_point(source_x: float, source_y: float, source_radius: float, angle: float) -> Tuple[float, float]:
    """Return the launch point just outside the source planet."""

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
    """Return the path geometry from source edge to target edge."""

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
    """Estimate angle, ETA, and edge-to-edge path distance for a fleet."""

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
    eta = max(1, int(math.ceil(hit_distance / fleet_speed(max(1, ships), max_speed=max_speed))))
    return angle, eta, hit_distance


def as_planets(raw_planets: Sequence[Sequence[Any]]) -> List[Planet]:
    """Convert raw planet arrays into Planet namedtuples."""

    return [planet if isinstance(planet, Planet) else Planet(*planet) for planet in raw_planets]


def as_fleets(raw_fleets: Sequence[Sequence[Any]]) -> List[Fleet]:
    """Convert raw fleet arrays into Fleet namedtuples."""

    return [fleet if isinstance(fleet, Fleet) else Fleet(*fleet) for fleet in raw_fleets]


def count_players(planets: Sequence[Planet], fleets: Sequence[Fleet]) -> int:
    """Count active players from planets and fleets."""

    owners = {planet.owner for planet in planets if planet.owner != -1}
    owners.update(fleet.owner for fleet in fleets)
    return max(2, len(owners))


def is_orbiting_planet(planet: Planet) -> bool:
    """Return whether a planet rotates around the sun."""

    orbital_radius = distance_xy(planet.x, planet.y, CENTER, CENTER)
    return orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT


def resolve_arrival_event(owner: int, garrison: float, arrivals: Sequence[Tuple[int, int, int]]) -> Tuple[int, float]:
    """Resolve the fleet combat that occurs when arrivals hit a planet."""

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


def normalize_arrivals(
    arrivals: Sequence[Tuple[int, int, int]],
    horizon: int,
) -> List[Tuple[int, int, int]]:
    """Normalize arrival records to sorted integer turns."""

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
    """Simulate the timeline for a single planet."""

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
    """Read the simulated owner and ships at a given turn."""

    turn = max(0, int(math.ceil(turn)))
    turn = min(turn, timeline["horizon"])
    owner = timeline["owner_at"].get(turn, timeline["owner_at"][timeline["horizon"]])
    ships = timeline["ships_at"].get(turn, timeline["ships_at"][timeline["horizon"]])
    return int(owner), max(0.0, float(ships))


def reset_memory_if_needed(player: int, step: int) -> None:
    """Reset per-match agent memory when a new game starts."""

    if step <= 0 or AGENT_MEMORY.get("player") != player:
        AGENT_MEMORY["last_owners"] = {}
        AGENT_MEMORY["last_step"] = None
        AGENT_MEMORY["player"] = player
        AGENT_MEMORY["last_error"] = None


@dataclass
class GameStateView:
    """Structured snapshot of the current observation."""

    player: int
    step: int
    remaining_steps: int
    is_early: bool
    is_opening: bool
    is_late: bool
    is_very_late: bool
    num_players: int
    angular_velocity: float
    max_speed: float
    planets: List[Planet]
    fleets: List[Fleet]
    initial_planets: List[Planet]
    comets: List[Dict[str, Any]]
    comet_planet_ids: set[int]
    my_planets: List[Planet]
    enemy_planets: List[Planet]
    neutral_planets: List[Planet]
    my_fleets: List[Fleet]
    enemy_fleets: List[Fleet]
    planets_by_id: Dict[int, Planet]
    initial_by_id: Dict[int, Planet]


@dataclass
class ThreatInfo:
    """Threat summary for a defended planet."""

    planet: Planet
    threat_turn: int
    keep_needed: int


@dataclass
class CapturePlan:
    """A resolved plan for a single launch."""

    target: Planet
    ships: int
    angle: float
    eta: int
    eval_turn: int
    required_ships: int


@dataclass
class PlannedMove:
    """A committed move plus the target metadata it depends on."""

    source_id: int
    target_id: int
    angle: float
    ships: int
    eta: int
    mission: str


@dataclass
class MissionOption:
    """A scored candidate action for the unified mission queue."""

    score: float
    source_ids: List[int]
    target_id: int
    angles: List[float]
    etas: List[int]
    ships: List[int]
    needed: int
    mission: str
    anchor_turn: Optional[int] = None


class PositionPredictor:
    """Predict future positions for planets and comets."""

    def __init__(
        self,
        current_step: int,
        angular_velocity: float,
        initial_planets: Sequence[Planet],
        current_planets: Sequence[Planet],
        comets: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Store the immutable starting positions and active comet paths."""

        self.current_step = current_step
        self.angular_velocity = angular_velocity
        self.initial_by_id = {planet.id: planet for planet in initial_planets}
        self.current_by_id = {planet.id: planet for planet in current_planets}
        self.comet_by_planet_id: Dict[int, Dict[str, Any]] = {}
        for comet in comets or []:
            for planet_id in comet.get("planet_ids", []):
                self.comet_by_planet_id[int(planet_id)] = comet

    def predict_planet_pos(self, planet: Planet, step_offset: int) -> Tuple[float, float]:
        """Predict a planet position relative to the current observation step."""

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
        """Predict a comet position from its explicit path and path index."""

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
        """Predict either a planet or comet position."""

        if planet.id in self.comet_by_planet_id:
            return self.predict_comet_pos(self.comet_by_planet_id[planet.id], planet.id, step_offset)
        return self.predict_planet_pos(planet, step_offset)

    def comet_remaining_life(self, planet_id: int) -> int:
        """Return how many future path points remain for a comet."""

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
        """Return whether a target can move after launch."""

        if planet.id in self.comet_by_planet_id:
            return True
        initial = self.initial_by_id.get(planet.id)
        return bool(initial and is_orbiting_planet(initial))


class InterceptSolver:
    """Solve edge-to-edge intercepts against moving Orbit Wars planets."""

    def __init__(
        self,
        predictor: PositionPredictor,
        max_speed: float = DEFAULT_MAX_SHIP_SPEED,
        deadline: Optional[float] = None,
    ) -> None:
        """Store the shared predictor and fleet speed cap."""

        self.predictor = predictor
        self.max_speed = max_speed
        self.deadline = deadline

    def expired(self) -> bool:
        """Return whether the planning deadline has been reached."""

        return self.deadline is not None and time.time() >= self.deadline

    def check_sun_collision(
        self,
        from_x: float,
        from_y: float,
        angle: float,
        speed: float,
        steps: int,
    ) -> bool:
        """Trace a simple center-based path and flag sun or boundary collisions."""

        x = from_x
        y = from_y
        dx = math.cos(angle) * speed
        dy = math.sin(angle) * speed
        for _ in range(max(1, steps)):
            next_x = x + dx
            next_y = y + dy
            if not point_in_bounds(next_x, next_y):
                return True
            if segment_circle_intersects((x, y), (next_x, next_y), SUN_CENTER, SUN_RADIUS):
                return True
            x, y = next_x, next_y
        return False

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
        """Estimate a radius-aware launch to a fixed future position."""

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

    def _search_safe_intercept(
        self,
        from_x: float,
        from_y: float,
        from_radius: float,
        target_planet: Planet,
        ships: int,
    ) -> Optional[Tuple[float, int]]:
        """Search candidate target turns until a stable intercept is found."""

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
            delta = abs(eta - candidate_turn)
            if delta > 1:
                continue
            score = (delta, eta, candidate_turn)
            if best is None or score < best:
                best = (angle, eta, candidate_turn)
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
        """Return the first feasible edge-aware launch angle and ETA for a target."""

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
                    return next_angle, next_eta
                angle, eta = next_angle, next_eta
                target_x, target_y = future_pos
            return angle, eta
        finally:
            self.predictor.current_step = original_step


def estimate_fleet_arrival_turn(
    fleet: Fleet,
    planet: Planet,
    predictor: PositionPredictor,
    turns_ahead: int,
    max_speed: float,
) -> Optional[int]:
    """Estimate when a visible in-flight fleet will collide with a target planet."""

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
    """Track projected ownership with visible fleets and planned commitments."""

    def __init__(
        self,
        state: GameStateView,
        predictor: PositionPredictor,
        deadline: Optional[float] = None,
    ) -> None:
        """Build cached arrival ledgers and base planet timelines."""

        self.state = state
        self.player = state.player
        self.predictor = predictor
        self.deadline = deadline
        self.arrivals_by_planet: Dict[int, List[Tuple[int, int, int]]] = {planet.id: [] for planet in state.planets}
        self.base_timeline: Dict[int, Dict[str, Any]] = {}
        self.timeline_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.exact_need_cache: Dict[Tuple[int, int, int], int] = {}

        self.total_visible_ships = sum(int(planet.ships) for planet in state.planets) + sum(
            int(fleet.ships) for fleet in state.fleets
        )
        self.total_production = sum(int(planet.production) for planet in state.planets)

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
        """Return whether the planning deadline has been reached."""

        return self.deadline is not None and time.time() >= self.deadline

    def _search_cap(self, eval_turn: int) -> int:
        """Return an upper bound for binary ownership searches."""

        productive_cap = self.total_production * max(2, eval_turn + 2)
        return max(32, int(self.total_visible_ships + productive_cap + 32))

    def projected_timeline(
        self,
        target_id: int,
        horizon: int,
        planned_commitments: Optional[Mapping[int, Sequence[Tuple[int, int, int]]]] = None,
        extra_arrivals: Sequence[Tuple[int, int, int]] = (),
    ) -> Dict[str, Any]:
        """Return the simulated timeline for a planet under planned commitments."""

        horizon = max(1, int(math.ceil(horizon)))
        planned_commitments = planned_commitments or {}
        if not planned_commitments.get(target_id) and not extra_arrivals and horizon <= BASE_TIMELINE_HORIZON:
            return self.base_timeline[target_id]

        cache_key = (target_id, horizon)
        if not planned_commitments.get(target_id) and not extra_arrivals and cache_key in self.timeline_cache:
            return self.timeline_cache[cache_key]

        arrivals = [item for item in self.arrivals_by_planet.get(target_id, []) if item[0] <= horizon]
        arrivals.extend(item for item in planned_commitments.get(target_id, []) if item[0] <= horizon)
        arrivals.extend(item for item in extra_arrivals if item[0] <= horizon)
        planet = self.state.planets_by_id[target_id]
        timeline = simulate_planet_timeline(planet, arrivals, self.player, horizon)

        if not planned_commitments.get(target_id) and not extra_arrivals:
            self.timeline_cache[cache_key] = timeline
        return timeline

    def projected_state(
        self,
        target_id: int,
        turn: int,
        planned_commitments: Optional[Mapping[int, Sequence[Tuple[int, int, int]]]] = None,
        extra_arrivals: Sequence[Tuple[int, int, int]] = (),
    ) -> Tuple[int, float]:
        """Return projected owner and ships for a planet at a turn."""

        timeline = self.projected_timeline(target_id, turn, planned_commitments=planned_commitments, extra_arrivals=extra_arrivals)
        return state_at_timeline(timeline, turn)

    def hold_status(
        self,
        target_id: int,
        horizon: int,
        planned_commitments: Optional[Mapping[int, Sequence[Tuple[int, int, int]]]] = None,
    ) -> Dict[str, Any]:
        """Return projected hold information for a target planet."""

        timeline = self.projected_timeline(target_id, horizon, planned_commitments=planned_commitments)
        return {
            "keep_needed": timeline["keep_needed"],
            "min_owned": timeline["min_owned"],
            "first_enemy": timeline["first_enemy"],
            "fall_turn": timeline["fall_turn"],
            "holds_full": timeline["holds_full"],
        }

    def min_ships_to_own_by(
        self,
        target_id: int,
        eval_turn: int,
        player: int,
        arrival_turn: Optional[int] = None,
        planned_commitments: Optional[Mapping[int, Sequence[Tuple[int, int, int]]]] = None,
        extra_arrivals: Sequence[Tuple[int, int, int]] = (),
        upper_bound: Optional[int] = None,
    ) -> int:
        """Return the exact ships needed to own a target by eval_turn."""

        planned_commitments = planned_commitments or {}
        eval_turn = max(1, int(math.ceil(eval_turn)))
        arrival_turn = eval_turn if arrival_turn is None else max(1, int(math.ceil(arrival_turn)))
        if arrival_turn > eval_turn:
            cap = upper_bound if upper_bound is not None else self._search_cap(eval_turn)
            return max(1, int(cap)) + 1

        normalized_extra = tuple(
            (max(1, int(math.ceil(turns))), int(owner), int(ships))
            for turns, owner, ships in extra_arrivals
            if ships > 0 and max(1, int(math.ceil(turns))) <= eval_turn
        )

        cache_key = None
        if arrival_turn == eval_turn and not planned_commitments.get(target_id) and not normalized_extra:
            cache_key = (target_id, eval_turn, player)
            cached = self.exact_need_cache.get(cache_key)
            if cached is not None:
                return cached

        owner_before, _ships_before = self.projected_state(
            target_id,
            eval_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=normalized_extra,
        )
        if owner_before == player:
            if cache_key is not None:
                self.exact_need_cache[cache_key] = 0
            return 0

        def owns_with(ships: int) -> bool:
            owner_after, _ = self.projected_state(
                target_id,
                eval_turn,
                planned_commitments=planned_commitments,
                extra_arrivals=normalized_extra + ((arrival_turn, player, int(ships)),),
            )
            return owner_after == player

        if upper_bound is not None:
            high = max(1, int(upper_bound))
            if not owns_with(high):
                return high + 1
        else:
            high = 1
            search_cap = self._search_cap(eval_turn)
            while high <= search_cap and not owns_with(high):
                high *= 2
            if high > search_cap:
                high = search_cap
                if not owns_with(high):
                    return high + 1

        low = 1
        while low < high:
            middle = (low + high) // 2
            if owns_with(middle):
                high = middle
            else:
                low = middle + 1

        if cache_key is not None:
            self.exact_need_cache[cache_key] = low
        return low

    def min_ships_to_own_at(
        self,
        target_id: int,
        turn: int,
        player: int,
        planned_commitments: Optional[Mapping[int, Sequence[Tuple[int, int, int]]]] = None,
        extra_arrivals: Sequence[Tuple[int, int, int]] = (),
        upper_bound: Optional[int] = None,
    ) -> int:
        """Return the exact ships needed to own a target exactly at turn."""

        return self.min_ships_to_own_by(
            target_id,
            turn,
            player,
            arrival_turn=turn,
            planned_commitments=planned_commitments,
            extra_arrivals=extra_arrivals,
            upper_bound=upper_bound,
        )

    def reinforcement_needed_to_hold_until(
        self,
        target_id: int,
        arrival_turn: int,
        hold_until: int,
        planned_commitments: Optional[Mapping[int, Sequence[Tuple[int, int, int]]]] = None,
        upper_bound: Optional[int] = None,
    ) -> int:
        """Return the ships needed for a reinforcement to preserve ownership."""

        planned_commitments = planned_commitments or {}
        target = self.state.planets_by_id[target_id]
        arrival_turn = max(1, int(math.ceil(arrival_turn)))
        hold_until = max(arrival_turn, int(math.ceil(hold_until)))

        if target.owner != self.player:
            return self.min_ships_to_own_by(
                target_id,
                hold_until,
                self.player,
                arrival_turn=arrival_turn,
                planned_commitments=planned_commitments,
                upper_bound=upper_bound,
            )

        def holds_with_reinforcement(ships: int) -> bool:
            timeline = self.projected_timeline(
                target_id,
                hold_until,
                planned_commitments=planned_commitments,
                extra_arrivals=((arrival_turn, self.player, int(ships)),),
            )
            for turn in range(arrival_turn, hold_until + 1):
                if timeline["owner_at"].get(turn) != self.player:
                    return False
            return True

        if upper_bound is not None:
            high = max(1, int(upper_bound))
            if not holds_with_reinforcement(high):
                return high + 1
        else:
            high = 1
            search_cap = self._search_cap(hold_until)
            while high <= search_cap and not holds_with_reinforcement(high):
                high *= 2
            if high > search_cap:
                high = search_cap
                if not holds_with_reinforcement(high):
                    return high + 1

        low = 1
        while low < high:
            middle = (low + high) // 2
            if holds_with_reinforcement(middle):
                high = middle
            else:
                low = middle + 1
        return low


class SurplusCalculator:
    """Estimate how many ships a planet can safely launch."""

    def __init__(self, player_id: int, max_speed: float = DEFAULT_MAX_SHIP_SPEED) -> None:
        """Store the player perspective and movement cap."""

        self.player_id = player_id
        self.max_speed = max_speed

    def get_surplus(
        self,
        planet: Planet,
        my_fleets: Sequence[Fleet],
        enemy_fleets: Sequence[Fleet],
        predictor: PositionPredictor,
        turns_ahead: int = DEFENSE_HORIZON,
    ) -> int:
        """Return the safely sendable ships while preserving ownership."""

        if planet.owner != self.player_id:
            return 0

        arrivals: List[Tuple[int, int, int]] = []
        for fleet in my_fleets:
            eta = estimate_fleet_arrival_turn(
                fleet,
                planet,
                predictor,
                turns_ahead=turns_ahead,
                max_speed=self.max_speed,
            )
            if eta is not None:
                arrivals.append((eta, fleet.owner, int(fleet.ships)))
        for fleet in enemy_fleets:
            eta = estimate_fleet_arrival_turn(
                fleet,
                planet,
                predictor,
                turns_ahead=turns_ahead,
                max_speed=self.max_speed,
            )
            if eta is not None:
                arrivals.append((eta, fleet.owner, int(fleet.ships)))

        timeline = simulate_planet_timeline(planet, arrivals, self.player_id, turns_ahead)
        if not timeline["holds_full"]:
            return 0
        return max(0, int(planet.ships) - int(timeline["keep_needed"]))


class ForwardSimulator:
    """Approximate future ship totals for evaluation and phase heuristics."""

    def __init__(self, player_id: int, max_speed: float = DEFAULT_MAX_SHIP_SPEED) -> None:
        """Store the player perspective and ship speed cap."""

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
        """Simulate a few future turns and return ship totals each turn."""

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

        for _turn in range(1, num_turns + 1):
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
                self._resolve_arrivals(planet, incoming)

            for planet in planet_state.values():
                initial = initial_by_id.get(int(planet[0]))
                if initial is None or not is_orbiting_planet(initial):
                    continue
                dx = initial.x - CENTER
                dy = initial.y - CENTER
                radius = math.hypot(dx, dy)
                start_angle = math.atan2(dy, dx)
                current_angle = start_angle + angular_velocity * _turn
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

    def _resolve_arrivals(self, planet: List[Any], arrivals: Sequence[List[Any]]) -> None:
        """Resolve fleet combat against a mutable planet state."""

        owner, ships = resolve_arrival_event(
            int(planet[1]),
            float(planet[5]),
            [(1, int(fleet[1]), int(fleet[6])) for fleet in arrivals],
        )
        planet[1] = owner
        planet[5] = int(round(ships))


class ScoringFunction:
    """Score a position using production, ships, pressure, and expansion."""

    def score_position(
        self,
        my_planets: Sequence[Planet],
        enemy_planets: Sequence[Planet],
        neutral_planets: Sequence[Planet],
        my_fleets: Sequence[Fleet],
        enemy_fleets: Sequence[Fleet],
    ) -> float:
        """Return a heuristic value for the current board state."""

        my_production = sum(planet.production for planet in my_planets)
        enemy_production = sum(planet.production for planet in enemy_planets)
        my_ships = sum(planet.ships for planet in my_planets) + sum(fleet.ships for fleet in my_fleets)
        enemy_ships = sum(planet.ships for planet in enemy_planets) + sum(fleet.ships for fleet in enemy_fleets)

        pressure = 0.0
        for fleet in enemy_fleets:
            for planet in my_planets:
                if distance_xy(fleet.x, fleet.y, planet.x, planet.y) <= PRESSURE_DISTANCE:
                    pressure += fleet.ships
                    break

        expansion = 0.0
        if my_planets:
            for neutral in neutral_planets:
                closest = min(distance_planets(neutral, planet) for planet in my_planets)
                expansion += neutral.production / max(1.0, closest)

        return (
            10.0 * (my_production - enemy_production)
            + 1.0 * (my_ships - enemy_ships)
            - 2.0 * pressure
            + 5.0 * expansion
        )


def detect_enemy_crashes(
    arrivals_by_planet: Dict[int, List[Tuple[int, int, int]]],
    player: int,
) -> List[Dict[str, Any]]:
    """Detect enemy fleet collisions from the arrival ledger."""

    crashes: List[Dict[str, Any]] = []
    for target_id, arrivals in arrivals_by_planet.items():
        enemy_events = sorted(
            [(eta, owner, ships) for eta, owner, ships in arrivals if owner not in (-1, player) and ships > 0]
        )
        for i in range(len(enemy_events)):
            eta_a, owner_a, ships_a = enemy_events[i]
            for j in range(i + 1, len(enemy_events)):
                eta_b, owner_b, ships_b = enemy_events[j]
                if owner_a == owner_b:
                    continue
                if abs(eta_a - eta_b) > CRASH_EXPLOIT_ETA_WINDOW:
                    break
                if ships_a + ships_b < CRASH_EXPLOIT_MIN_SHIPS:
                    continue
                crashes.append({
                    "target_id": target_id,
                    "crash_turn": max(eta_a, eta_b),
                    "total_ships": ships_a + ships_b,
                })
    return crashes


class DecisionLogic:
    """Main Orbit Wars decision engine with unified mission scoring."""

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

        self.used_donor_ids: set[int] = set()
        self.committed_ships: Dict[int, int] = defaultdict(int)
        self.targeted_planet_ids: set[int] = set()
        self.planned_commitments: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        self.shot_cache: Dict[Tuple[int, int, int], Optional[Tuple[float, int]]] = {}
        self.surplus_cache: Dict[Tuple[int, int], int] = {}
        reset_memory_if_needed(self.state.player, self.state.step)

    def expired(self) -> bool:
        return time.time() >= self.deadline

    def decide(self) -> List[List[float | int]]:
        """Build all missions, rank globally, commit best ones."""

        try:
            self.reaction_map = self._build_reaction_map()
            self.modes = self._build_modes()
            self.crashes = detect_enemy_crashes(self.world.arrivals_by_planet, self.state.player)
            missions = self._build_all_missions()
            moves = self._commit_missions(missions)
            return self._finalize(moves)
        finally:
            AGENT_MEMORY["last_owners"] = {planet.id: planet.owner for planet in self.state.planets}
            AGENT_MEMORY["last_step"] = self.state.step

    def _build_state(self, obs: Any, config: Any) -> GameStateView:
        """Convert raw observation data into grouped namedtuples and phase flags."""

        player = int(get_field(obs, "player", 0))
        step = int(get_field(obs, "step", 0) or 0)
        angular_velocity = float(get_field(obs, "angular_velocity", 0.0) or 0.0)
        max_speed = float(get_field(config, "shipSpeed", DEFAULT_MAX_SHIP_SPEED))

        planets = as_planets(get_field(obs, "planets", []) or [])
        fleets = as_fleets(get_field(obs, "fleets", []) or [])
        initial_planets = as_planets(get_field(obs, "initial_planets", get_field(obs, "planets", [])) or [])
        comets = list(get_field(obs, "comets", []) or [])
        comet_planet_ids = {int(pid) for pid in get_field(obs, "comet_planet_ids", []) or []}

        my_planets = [planet for planet in planets if planet.owner == player]
        enemy_planets = [planet for planet in planets if planet.owner not in (-1, player)]
        neutral_planets = [planet for planet in planets if planet.owner == -1]
        my_fleets = [fleet for fleet in fleets if fleet.owner == player]
        enemy_fleets = [fleet for fleet in fleets if fleet.owner != player]

        remaining_steps = max(1, TOTAL_STEPS - step)
        num_players = count_players(planets, fleets)

        return GameStateView(
            player=player,
            step=step,
            remaining_steps=remaining_steps,
            is_early=step < EARLY_TURN_LIMIT,
            is_opening=step < OPENING_TURN_LIMIT,
            is_late=remaining_steps < LATE_REMAINING_TURNS,
            is_very_late=remaining_steps < VERY_LATE_REMAINING_TURNS,
            num_players=num_players,
            angular_velocity=angular_velocity,
            max_speed=max_speed,
            planets=planets,
            fleets=fleets,
            initial_planets=initial_planets,
            comets=comets,
            comet_planet_ids=comet_planet_ids,
            my_planets=my_planets,
            enemy_planets=enemy_planets,
            neutral_planets=neutral_planets,
            my_fleets=my_fleets,
            enemy_fleets=enemy_fleets,
            planets_by_id={planet.id: planet for planet in planets},
            initial_by_id={planet.id: planet for planet in initial_planets},
        )

    def _finalize(self, moves: Sequence[PlannedMove]) -> List[List[float | int]]:
        sanitized: List[List[float | int]] = []
        used_sources: set[int] = set()
        current_by_id = {planet.id: planet for planet in self.state.my_planets}
        for move in moves:
            if move.source_id in used_sources or move.source_id not in current_by_id:
                continue
            ships = int(move.ships)
            if ships <= 0 or ships > current_by_id[move.source_id].ships:
                continue
            sanitized.append([move.source_id, float(move.angle), ships])
            used_sources.add(move.source_id)
        return sanitized

    def _commit_move(self, move: PlannedMove) -> None:
        self.used_donor_ids.add(move.source_id)
        self.committed_ships[move.source_id] += int(move.ships)
        self.targeted_planet_ids.add(move.target_id)
        self.planned_commitments[move.target_id].append((move.eta, self.state.player, int(move.ships)))

    def _effective_planet(self, planet: Planet) -> Planet:
        committed = self.committed_ships.get(planet.id, 0)
        if committed <= 0:
            return planet
        return planet._replace(ships=max(0, planet.ships - committed))

    def _available_my_planets(self, excluded_ids: Sequence[int] = ()) -> List[Planet]:
        excluded = set(excluded_ids)
        return [
            self._effective_planet(planet)
            for planet in self.state.my_planets
            if planet.id not in self.used_donor_ids and planet.id not in excluded
        ]

    def _plan_shot(self, source: Planet, target: Planet, ships: int) -> Optional[Tuple[float, int]]:
        key = (source.id, target.id, int(ships))
        if key not in self.shot_cache:
            self.shot_cache[key] = self.intercept_solver.solve_intercept(
                source.x, source.y, int(ships), target,
                self.state.step, self.state.angular_velocity,
                self.state.initial_planets, from_radius=source.radius,
            )
        return self.shot_cache[key]

    def _planet_surplus(self, planet: Planet, turns_ahead: int = DEFENSE_HORIZON) -> int:
        cache_key = (planet.id, self.committed_ships.get(planet.id, 0))
        if cache_key in self.surplus_cache:
            return self.surplus_cache[cache_key]
        effective = self._effective_planet(planet)
        if effective.owner != self.state.player:
            self.surplus_cache[cache_key] = 0
            return 0
        hold = self.world.hold_status(planet.id, turns_ahead, planned_commitments=self.planned_commitments)
        if not hold["holds_full"]:
            self.surplus_cache[cache_key] = 0
            return 0
        surplus = max(0, int(effective.ships) - int(hold["keep_needed"]))
        self.surplus_cache[cache_key] = surplus
        return surplus

    def _compute_margin(self, target: Planet, mission: str) -> int:
        if mission in {"snipe", "crash_exploit"}:
            return 0 if self.state.is_late else 1
        if mission == "reinforce":
            return 1 + target.production
        if mission == "evacuate":
            return 0
        if target.owner == -1:
            base = min(NEUTRAL_MARGIN_CAP, NEUTRAL_MARGIN_BASE + target.production * MARGIN_PROD_WEIGHT)
        else:
            base = min(HOSTILE_MARGIN_CAP, HOSTILE_MARGIN_BASE + target.production * MARGIN_PROD_WEIGHT)
        if not is_orbiting_planet(target):
            base += STATIC_MARGIN
        if self._is_contested_neutral(target):
            base += CONTESTED_MARGIN
        if self.modes.get("is_finishing") and target.owner not in (-1, self.state.player):
            base += FINISHING_SEND_BONUS
        if self.state.is_very_late:
            base = min(base, 2)
        elif self.state.is_late:
            base = min(base, 4)
        mult = self.modes.get("attack_margin_mult", 1.0)
        return max(0, int(math.ceil(base * mult)))

    def _preferred_send(self, target: Planet, need: int, eta: int, available: int, mission: str) -> int:
        margin = self._compute_margin(target, mission)
        return min(available, max(need, need + margin))

    def _settle_plan(
        self,
        source: Planet,
        target: Planet,
        max_ships: int,
        mission: str,
        seed_send: Optional[int] = None,
        eval_turn_fn: Optional[Callable[[int], int]] = None,
        need_fn: Optional[Callable[[int, int], int]] = None,
        min_turn: Optional[int] = None,
        max_turn: Optional[int] = None,
        anchor_turn: Optional[int] = None,
        anchor_tolerance: Optional[int] = None,
    ) -> Optional[CapturePlan]:
        if self.expired():
            return None
        max_ships = min(int(max_ships), int(source.ships))
        if max_ships <= 0:
            return None

        tested: Dict[int, Optional[CapturePlan]] = {}
        desired_for: Dict[int, int] = {}
        valid_seeds = {
            max(1, min(max_ships, seed_send or max(1, int(target.ships) + 1))),
            max(1, min(max_ships, int(target.ships) + 1)),
            max(1, min(max_ships, int(target.ships) + int(target.production) + 1)),
            max(1, min(max_ships, max_ships // 2 or 1)),
            max_ships,
        }

        def evaluate(send: int) -> Optional[CapturePlan]:
            send = max(1, min(max_ships, int(send)))
            if send in tested:
                return tested[send]
            shot = self._plan_shot(source, target, send)
            if shot is None:
                tested[send] = None
                return None
            angle, eta = shot
            if min_turn is not None and eta < min_turn:
                tested[send] = None
                return None
            if max_turn is not None and eta > max_turn:
                tested[send] = None
                return None
            if anchor_turn is not None and anchor_tolerance is not None and abs(eta - anchor_turn) > anchor_tolerance:
                tested[send] = None
                return None
            eval_turn = max(eta, int(math.ceil(eval_turn_fn(eta))) if eval_turn_fn is not None else eta)
            if need_fn is not None:
                need = need_fn(eta, max_ships)
            else:
                need = self.world.min_ships_to_own_by(
                    target.id, eval_turn, self.state.player,
                    arrival_turn=eta, planned_commitments=self.planned_commitments,
                    upper_bound=max_ships,
                )
            if need <= 0:
                need = 0
            if need > max_ships:
                tested[send] = None
                return None
            desired = self._preferred_send(target, need, eta, max_ships, mission)
            if mission in {"snipe", "crash_exploit"}:
                desired = max(need, min(max_ships, desired))
            desired_for[send] = desired
            tested[send] = CapturePlan(target=target, ships=send, angle=angle, eta=eta, eval_turn=eval_turn, required_ships=need)
            return tested[send]

        current_send = None
        for candidate in sorted(valid_seeds):
            plan = evaluate(candidate)
            if plan is not None:
                current_send = candidate
                break
        if current_send is None:
            return None
        for _ in range(4):
            if self.expired():
                break
            plan = evaluate(current_send)
            if plan is None:
                break
            desired = desired_for.get(current_send, plan.required_ships)
            if plan.ships >= plan.required_ships and desired == plan.ships:
                return plan
            next_send = max(1, min(max_ships, desired))
            if next_send == current_send:
                break
            current_send = next_send
        candidates = [p for p in tested.values() if p is not None and p.ships >= p.required_ships]
        if not candidates:
            return None

        def candidate_key(plan: CapturePlan) -> Tuple[int, int, int]:
            anchor_delta = abs(plan.eta - anchor_turn) if anchor_turn is not None else 0
            return (anchor_delta, plan.ships, plan.eta)

        return min(candidates, key=candidate_key)

    # ── Utility: reaction map, modes, classification ──

    def _build_reaction_map(self) -> Dict[int, Tuple[int, int]]:
        rmap: Dict[int, Tuple[int, int]] = {}
        targets = [p for p in self.state.planets if p.owner != self.state.player]
        for target in targets:
            if self.expired():
                break
            my_best = 10**9
            for src in sorted(self.state.my_planets, key=lambda s: distance_planets(s, target))[:REACTION_TOP_K]:
                shot = self._plan_shot(src, target, max(1, int(src.ships)))
                if shot is not None:
                    my_best = min(my_best, shot[1])
            enemy_best = 10**9
            for src in sorted(self.state.enemy_planets, key=lambda s: distance_planets(s, target))[:REACTION_TOP_K]:
                if self.expired():
                    break
                shot = self.intercept_solver.solve_intercept(
                    src.x, src.y, max(1, int(src.ships)), target,
                    self.state.step, self.state.angular_velocity,
                    self.state.initial_planets, from_radius=src.radius,
                )
                if shot is not None:
                    enemy_best = min(enemy_best, shot[1])
            rmap[target.id] = (my_best, enemy_best)
        return rmap

    def _build_modes(self) -> Dict[str, Any]:
        my_total = sum(p.ships for p in self.state.my_planets) + sum(f.ships for f in self.state.my_fleets)
        enemy_total = sum(p.ships for p in self.state.enemy_planets) + sum(f.ships for f in self.state.enemy_fleets)
        my_prod = sum(p.production for p in self.state.my_planets)
        enemy_prod = sum(p.production for p in self.state.enemy_planets)
        denom = max(1, my_total + enemy_total)
        domination = (my_total - enemy_total) / denom
        is_behind = domination < BEHIND_THRESHOLD
        is_ahead = domination > AHEAD_THRESHOLD
        is_dominating = is_ahead or (enemy_total > 0 and my_total > enemy_total * 1.25)
        is_finishing = domination > FINISHING_THRESHOLD and my_prod > enemy_prod * FINISHING_PROD_RATIO and self.state.step > 100

        mult = 1.0
        if is_ahead:
            mult += AHEAD_MARGIN_BONUS
        if is_behind:
            mult -= BEHIND_MARGIN_PENALTY
        if is_finishing:
            mult += FINISHING_MARGIN_BONUS

        return {
            "domination": domination,
            "is_behind": is_behind,
            "is_ahead": is_ahead,
            "is_dominating": is_dominating,
            "is_finishing": is_finishing,
            "attack_margin_mult": mult,
            "my_total": my_total,
            "enemy_total": enemy_total,
        }

    def _is_safe_neutral(self, target: Planet) -> bool:
        if target.owner != -1:
            return False
        my_t, enemy_t = self.reaction_map.get(target.id, (10**9, 10**9))
        return my_t <= enemy_t - SAFE_NEUTRAL_MARGIN

    def _is_contested_neutral(self, target: Planet) -> bool:
        if target.owner != -1:
            return False
        my_t, enemy_t = self.reaction_map.get(target.id, (10**9, 10**9))
        return abs(my_t - enemy_t) <= CONTESTED_NEUTRAL_MARGIN

    def _candidate_time_valid(self, target: Planet, eta: int) -> bool:
        buffer = VERY_LATE_CAPTURE_BUFFER if self.state.is_very_late else LATE_CAPTURE_BUFFER
        if eta > self.state.remaining_steps - buffer:
            return False
        if target.id in self.state.comet_planet_ids:
            life = self.predictor.comet_remaining_life(target.id)
            if eta >= life:
                return False
        return True

    # ── Unified scoring ──

    def _target_value(self, target: Planet, eta: int, mission: str) -> float:
        turns_profit = max(1, self.state.remaining_steps - eta)
        if target.id in self.state.comet_planet_ids:
            life = self.predictor.comet_remaining_life(target.id)
            turns_profit = max(0, min(turns_profit, life - eta))
            if turns_profit <= 0:
                return -1.0

        value = float(target.production * turns_profit)

        if not is_orbiting_planet(target):
            value *= STATIC_VALUE_MULT
        if target.owner not in (-1, self.state.player):
            value *= OPENING_HOSTILE_VALUE_MULT if self.state.is_opening else HOSTILE_VALUE_MULT
        if target.owner == -1:
            if self._is_safe_neutral(target):
                value *= SAFE_NEUTRAL_VALUE_MULT
            elif self._is_contested_neutral(target):
                value *= CONTESTED_NEUTRAL_VALUE_MULT
            if self.state.is_early:
                value *= EARLY_NEUTRAL_VALUE_MULT
        if target.id in self.state.comet_planet_ids:
            value *= COMET_VALUE_MULT

        if mission == "snipe":
            value *= SNIPE_VALUE_MULT
        elif mission in {"swarm", "attack"}:
            value *= SWARM_VALUE_MULT
        elif mission == "reinforce":
            value *= REINFORCE_VALUE_MULT
        elif mission == "crash_exploit":
            value *= CRASH_EXPLOIT_VALUE_MULT

        if self.state.is_late:
            value += max(0, target.ships) * LATE_SHIP_VALUE
            if target.owner not in (-1, self.state.player):
                owner_ships = sum(
                    p.ships for p in self.state.enemy_planets if p.owner == target.owner
                )
                if owner_ships <= WEAK_ENEMY_THRESHOLD:
                    value += ELIMINATION_BONUS

        modes = self.modes
        if modes["is_finishing"] and target.owner not in (-1, self.state.player):
            value *= 1.15
        if modes["is_behind"] and target.owner == -1 and self._is_safe_neutral(target):
            value *= 1.08
        if modes["is_dominating"] and target.owner == -1 and self._is_contested_neutral(target):
            value *= 0.92

        return value

    def _score_mission(self, value: float, ships: int, eta: int, target: Planet, mission: str) -> float:
        cost_weight = SNIPE_COST_WEIGHT if mission == "snipe" else ATTACK_COST_WEIGHT
        score = value / (ships + eta * cost_weight + 1.0)
        if not is_orbiting_planet(target):
            score *= STATIC_SCORE_MULT
            if self.state.is_early and target.owner == -1:
                score *= EARLY_STATIC_SCORE_MULT
        if mission == "snipe":
            score *= SNIPE_SCORE_MULT
        elif mission in {"swarm", "attack"}:
            score *= SWARM_SCORE_MULT
        elif mission == "crash_exploit":
            score *= CRASH_EXPLOIT_SCORE_MULT
        return score

    # ── Mission builders ──

    def _build_all_missions(self) -> List[MissionOption]:
        missions: List[MissionOption] = []
        self._build_defense_missions(missions)
        self._build_snipe_missions(missions)
        self._build_comet_missions(missions)
        self._build_expand_missions(missions)
        self._build_attack_missions(missions)
        self._build_swarm_missions(missions)
        self._build_crash_exploit_missions(missions)
        self._build_reinforce_missions(missions)
        self._build_evacuate_missions(missions)
        return missions

    def _build_defense_missions(self, missions: List[MissionOption]) -> None:
        for planet in self.state.my_planets:
            if self.expired():
                return
            hold = self.world.hold_status(planet.id, DEFENSE_HORIZON)
            if hold["fall_turn"] is None:
                continue
            fall_turn = int(hold["fall_turn"])
            hold_until = min(BASE_TIMELINE_HORIZON, fall_turn + 6)
            for donor in sorted(self._available_my_planets(excluded_ids=[planet.id]),
                                key=lambda s: distance_planets(s, planet)):
                if self.expired():
                    return
                surplus = self._planet_surplus(donor)
                if surplus <= 0:
                    continue
                plan = self._settle_plan(
                    donor, planet, surplus, mission="reinforce",
                    eval_turn_fn=lambda _eta, hu=hold_until: hu,
                    need_fn=lambda eta, cap, tid=planet.id, hu=hold_until: self.world.reinforcement_needed_to_hold_until(
                        tid, eta, hu, planned_commitments=self.planned_commitments, upper_bound=cap,
                    ),
                    max_turn=fall_turn,
                )
                if plan is not None and plan.eta <= fall_turn:
                    urgency = 100.0 / max(1, fall_turn) + planet.production * 5.0
                    missions.append(MissionOption(
                        score=urgency, source_ids=[donor.id], target_id=planet.id,
                        angles=[plan.angle], etas=[plan.eta], ships=[plan.ships],
                        needed=plan.required_ships, mission="defend",
                    ))
                    break

    def _build_snipe_missions(self, missions: List[MissionOption]) -> None:
        last_owners = AGENT_MEMORY.get("last_owners", {})
        targets = [p for p in self.state.enemy_planets if last_owners.get(p.id) == -1]
        for target in sorted(targets, key=lambda t: (t.ships, -t.production)):
            if self.expired():
                return
            for donor in sorted(self._available_my_planets(), key=lambda s: distance_planets(s, target)):
                if self.expired():
                    return
                surplus = self._planet_surplus(donor)
                if surplus <= 0:
                    continue
                plan = self._settle_plan(donor, target, surplus, mission="snipe", max_turn=24)
                if plan is None:
                    continue
                if not self._candidate_time_valid(target, plan.eta):
                    continue
                value = self._target_value(target, plan.eta, "snipe")
                if value <= 0:
                    continue
                score = self._score_mission(value, plan.ships, plan.eta, target, "snipe")
                missions.append(MissionOption(
                    score=score, source_ids=[donor.id], target_id=target.id,
                    angles=[plan.angle], etas=[plan.eta], ships=[plan.ships],
                    needed=plan.required_ships, mission="snipe",
                ))
                break

    def _build_comet_missions(self, missions: List[MissionOption]) -> None:
        comet_targets = [
            self.state.planets_by_id[pid]
            for pid in self.state.comet_planet_ids
            if pid in self.state.planets_by_id and self.state.planets_by_id[pid].owner == -1
        ]
        for target in sorted(comet_targets, key=lambda t: (-t.production, t.ships)):
            if self.expired():
                return
            best_plan: Optional[Tuple[Planet, CapturePlan]] = None
            for donor in self._available_my_planets():
                if self.expired():
                    return
                surplus = self._planet_surplus(donor)
                if surplus <= 0:
                    continue
                max_t = min(COMET_LOOKAHEAD, self.predictor.comet_remaining_life(target.id) - 1)
                plan = self._settle_plan(donor, target, surplus, mission="snipe", max_turn=max_t)
                if plan is None:
                    continue
                if best_plan is None or plan.eta < best_plan[1].eta:
                    best_plan = (donor, plan)
            if best_plan is None:
                continue
            donor, plan = best_plan
            enemy_t = self.reaction_map.get(target.id, (10**9, 10**9))[1]
            if enemy_t < plan.eta:
                continue
            value = self._target_value(target, plan.eta, "snipe")
            if value <= 0:
                continue
            score = self._score_mission(value, plan.ships, plan.eta, target, "snipe")
            missions.append(MissionOption(
                score=score, source_ids=[donor.id], target_id=target.id,
                angles=[plan.angle], etas=[plan.eta], ships=[plan.ships],
                needed=plan.required_ships, mission="comet",
            ))

    def _build_expand_missions(self, missions: List[MissionOption]) -> None:
        candidates = [
            p for p in self.state.neutral_planets
            if p.id not in self.state.comet_planet_ids
        ]
        if not candidates:
            return

        if self.state.is_early:
            candidates = [p for p in candidates if p.production >= OPENING_MIN_PRODUCTION]
            available = self._available_my_planets()
            if available:
                candidates.sort(key=lambda t: (
                    1 if is_orbiting_planet(t) else 0,
                    min(distance_planets(t, s) for s in available),
                ))

        max_eta = 14 if self.state.is_early else (18 if self.state.is_opening else 28)
        for target in candidates:
            if self.expired():
                return
            for donor in sorted(self._available_my_planets(), key=lambda s: distance_planets(s, target)):
                surplus = self._planet_surplus(donor)
                if surplus <= 0:
                    continue
                plan = self._settle_plan(donor, target, surplus, mission="expand", max_turn=max_eta)
                if plan is None:
                    continue
                if not self._candidate_time_valid(target, plan.eta):
                    continue
                value = self._target_value(target, plan.eta, "expand")
                if value <= 0:
                    continue
                score = self._score_mission(value, plan.ships, plan.eta, target, "expand")
                missions.append(MissionOption(
                    score=score, source_ids=[donor.id], target_id=target.id,
                    angles=[plan.angle], etas=[plan.eta], ships=[plan.ships],
                    needed=plan.required_ships, mission="expand",
                ))
                break

    def _build_attack_missions(self, missions: List[MissionOption]) -> None:
        if not self.state.enemy_planets:
            return
        if self.state.is_opening and len(self.state.neutral_planets) > 6:
            return
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
                value = self._target_value(target, plan.eta, "attack")
                if value <= 0:
                    continue
                score = self._score_mission(value, plan.ships, plan.eta, target, "attack")
                missions.append(MissionOption(
                    score=score, source_ids=[donor.id], target_id=target.id,
                    angles=[plan.angle], etas=[plan.eta], ships=[plan.ships],
                    needed=plan.required_ships, mission="attack",
                ))
                break

    def _build_swarm_missions(self, missions: List[MissionOption]) -> None:
        if self.state.is_opening or not self.state.enemy_planets:
            return
        available = self._available_my_planets()
        if len(available) < 2:
            return
        targets = sorted(self.state.enemy_planets, key=lambda t: (t.ships / max(1, t.production), t.ships))
        for target in targets:
            if self.expired():
                return
            nearby = sorted(available, key=lambda s: distance_planets(s, target))[:MULTI_SOURCE_TOP_K]
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
            donor_plans.sort(key=lambda x: x[3])
            for group_size in (2, 3):
                if self.expired():
                    break
                for i in range(len(donor_plans)):
                    anchor_eta = donor_plans[i][3]
                    group = [donor_plans[i]]
                    for j in range(len(donor_plans)):
                        if i == j:
                            continue
                        if abs(donor_plans[j][3] - anchor_eta) <= SWARM_ETA_TOLERANCE:
                            group.append(donor_plans[j])
                        if len(group) == group_size:
                            break
                    if len(group) < group_size:
                        continue
                    extra = tuple((g[3], self.state.player, g[1]) for g in group)
                    max_eta = max(g[3] for g in group)
                    need = self.world.min_ships_to_own_at(
                        target.id, max_eta, self.state.player,
                        planned_commitments=self.planned_commitments, extra_arrivals=extra,
                    )
                    if need == 0 and self._candidate_time_valid(target, max_eta):
                        total_ships = sum(g[1] for g in group)
                        value = self._target_value(target, max_eta, "swarm")
                        if value <= 0:
                            continue
                        score = self._score_mission(value, total_ships, max_eta, target, "swarm")
                        missions.append(MissionOption(
                            score=score,
                            source_ids=[g[0].id for g in group],
                            target_id=target.id,
                            angles=[g[2] for g in group],
                            etas=[g[3] for g in group],
                            ships=[g[1] for g in group],
                            needed=total_ships,
                            mission="swarm",
                        ))
                        break

    def _build_crash_exploit_missions(self, missions: List[MissionOption]) -> None:
        if self.state.num_players < 3:
            return
        for crash in self.crashes:
            if self.expired():
                return
            target = self.state.planets_by_id.get(crash["target_id"])
            if target is None or target.owner == self.state.player:
                continue
            desired_arrival = crash["crash_turn"] + CRASH_EXPLOIT_POST_DELAY
            for donor in sorted(self._available_my_planets(), key=lambda s: distance_planets(s, target)):
                surplus = self._planet_surplus(donor)
                if surplus <= 0:
                    continue
                plan = self._settle_plan(
                    donor, target, surplus, mission="crash_exploit",
                    anchor_turn=desired_arrival, anchor_tolerance=CRASH_EXPLOIT_ETA_WINDOW,
                    min_turn=crash["crash_turn"],
                )
                if plan is None:
                    continue
                if not self._candidate_time_valid(target, plan.eta):
                    continue
                value = self._target_value(target, plan.eta, "crash_exploit")
                if value <= 0:
                    continue
                score = self._score_mission(value, plan.ships, plan.eta, target, "crash_exploit")
                missions.append(MissionOption(
                    score=score, source_ids=[donor.id], target_id=target.id,
                    angles=[plan.angle], etas=[plan.eta], ships=[plan.ships],
                    needed=plan.required_ships, mission="crash_exploit",
                    anchor_turn=desired_arrival,
                ))
                break

    def _build_reinforce_missions(self, missions: List[MissionOption]) -> None:
        if len(self.state.my_planets) < 2 or not self.state.enemy_planets:
            return
        for planet in self.state.my_planets:
            if self.expired():
                return
            hold = self.world.hold_status(planet.id, DEFENSE_HORIZON)
            nearest_enemy = min(distance_planets(planet, e) for e in self.state.enemy_planets)
            proximity = 30.0 / max(6.0, nearest_enemy)
            if proximity < 1.5 and hold["fall_turn"] is None:
                continue
            hold_until = hold["fall_turn"] if hold["fall_turn"] is not None else min(BASE_TIMELINE_HORIZON, 18)
            for donor in sorted(
                self._available_my_planets(excluded_ids=[planet.id]),
                key=lambda s: distance_planets(s, planet),
            ):
                if self.expired():
                    return
                surplus = self._planet_surplus(donor)
                if surplus <= 0:
                    continue
                plan = self._settle_plan(
                    donor, planet, max(1, int(surplus * 0.75)), mission="reinforce",
                    eval_turn_fn=lambda _eta, hu=hold_until: hu,
                    need_fn=lambda eta, cap, tid=planet.id, hu=hold_until: self.world.reinforcement_needed_to_hold_until(
                        tid, eta, hu, planned_commitments=self.planned_commitments, upper_bound=cap,
                    ),
                    max_turn=24,
                )
                if plan is None:
                    continue
                value = planet.production * max(1, self.state.remaining_steps - plan.eta)
                value *= REINFORCE_VALUE_MULT
                value *= proximity * 0.5
                score = self._score_mission(value, plan.ships, plan.eta, planet, "reinforce")
                missions.append(MissionOption(
                    score=score, source_ids=[donor.id], target_id=planet.id,
                    angles=[plan.angle], etas=[plan.eta], ships=[plan.ships],
                    needed=plan.required_ships, mission="reinforce",
                ))
                break

    def _build_evacuate_missions(self, missions: List[MissionOption]) -> None:
        for planet in self.state.my_planets:
            if self.expired():
                return
            effective = self._effective_planet(planet)
            if effective.ships < EVAC_MIN_SHIPS:
                continue
            hold = self.world.hold_status(planet.id, DEFENSE_HORIZON)
            if hold["holds_full"]:
                continue
            send = effective.ships
            friendly = [p for p in self.state.my_planets if p.id != planet.id]
            friendly.sort(key=lambda p: distance_planets(p, planet))
            for dest in friendly:
                if self.expired():
                    return
                dest_hold = self.world.hold_status(dest.id, DEFENSE_HORIZON)
                if not dest_hold["holds_full"]:
                    continue
                shot = self._plan_shot(planet, dest, send)
                if shot is not None:
                    missions.append(MissionOption(
                        score=0.5 * send / max(1, shot[1]),
                        source_ids=[planet.id], target_id=dest.id,
                        angles=[shot[0]], etas=[shot[1]], ships=[send],
                        needed=0, mission="evacuate",
                    ))
                    break
            else:
                neutrals = sorted(
                    [p for p in self.state.neutral_planets],
                    key=lambda p: distance_planets(p, planet),
                )
                for dest in neutrals[:3]:
                    if self.expired():
                        return
                    shot = self._plan_shot(planet, dest, send)
                    if shot is None:
                        continue
                    need = self.world.min_ships_to_own_at(
                        dest.id, shot[1], self.state.player,
                        planned_commitments=self.planned_commitments, upper_bound=send,
                    )
                    if need <= send:
                        missions.append(MissionOption(
                            score=0.3 * send / max(1, shot[1]),
                            source_ids=[planet.id], target_id=dest.id,
                            angles=[shot[0]], etas=[shot[1]], ships=[send],
                            needed=need, mission="evacuate",
                        ))
                        break

    # ── Commit loop ──

    def _commit_missions(self, missions: List[MissionOption]) -> List[PlannedMove]:
        missions.sort(key=lambda m: -m.score)
        moves: List[PlannedMove] = []

        for mission in missions:
            if self.expired() or len(moves) >= MAX_TURN_LAUNCHES:
                break

            # Check source availability
            if any(sid in self.used_donor_ids for sid in mission.source_ids):
                continue
            if mission.mission != "reinforce" and mission.mission != "defend" and mission.mission != "evacuate":
                if mission.target_id in self.targeted_planet_ids:
                    continue

            if len(mission.source_ids) == 1:
                # Single-source mission: re-verify surplus
                src_id = mission.source_ids[0]
                src = self.state.planets_by_id.get(src_id)
                if src is None:
                    continue
                effective_src = self._effective_planet(src)
                if effective_src.ships < mission.ships[0]:
                    continue
                move = PlannedMove(
                    src_id, mission.target_id,
                    mission.angles[0], mission.ships[0],
                    mission.etas[0], mission.mission,
                )
                moves.append(move)
                self._commit_move(move)
            else:
                # Multi-source swarm: commit all sources
                can_commit = True
                for i, sid in enumerate(mission.source_ids):
                    src = self.state.planets_by_id.get(sid)
                    if src is None:
                        can_commit = False
                        break
                    effective_src = self._effective_planet(src)
                    if effective_src.ships < mission.ships[i]:
                        can_commit = False
                        break
                if not can_commit:
                    continue
                if len(moves) + len(mission.source_ids) > MAX_TURN_LAUNCHES:
                    continue
                for i, sid in enumerate(mission.source_ids):
                    move = PlannedMove(
                        sid, mission.target_id,
                        mission.angles[i], mission.ships[i],
                        mission.etas[i], mission.mission,
                    )
                    moves.append(move)
                    self._commit_move(move)

        return moves


def agent(obs: Any, config: Any) -> List[List[float | int]]:
    """Kaggle Orbit Wars entrypoint."""

    try:
        logic = DecisionLogic(obs, config)
        return logic.decide()
    except Exception as exc:
        AGENT_MEMORY["last_error"] = str(exc)
        return []


__all__ = [
    "Planet",
    "Fleet",
    "CENTER",
    "ROTATION_RADIUS_LIMIT",
    "PositionPredictor",
    "InterceptSolver",
    "ForwardSimulator",
    "SurplusCalculator",
    "ScoringFunction",
    "DecisionLogic",
    "agent",
]


class _OpponentPressureMixin:
    def _build_enemy_pressure_map(self) -> Dict[int, float]:
        pressure: Dict[int, float] = {}
        for target in self.state.planets:
            total = 0.0
            for enemy in self.state.enemy_planets:
                if enemy.id == target.id:
                    continue
                if target.owner not in (-1, self.state.player) and enemy.owner == target.owner:
                    continue
                probe = max(4, int(enemy.ships * 0.55))
                shot = self.intercept_solver.solve_intercept(
                    enemy.x,
                    enemy.y,
                    probe,
                    target,
                    self.state.step,
                    self.state.angular_velocity,
                    self.state.initial_planets,
                    from_radius=enemy.radius,
                )
                if shot is None or shot[1] > 18:
                    continue
                total += (probe + enemy.production * 2.0) / max(1.0, shot[1])
            pressure[target.id] = total
        return pressure

    def _target_value(self, target: Planet, eta: int, mission: str) -> float:
        value = super()._target_value(target, eta, mission)
        pressure = self.enemy_pressure_map.get(target.id, 0.0)
        if target.owner == -1:
            value -= pressure * (1.5 if self._is_safe_neutral(target) else 3.5)
        elif target.owner == self.state.player:
            value += pressure * 2.0
        return value

    def _build_defense_missions(self, missions: List[MissionOption]) -> None:
        for planet in self.state.my_planets:
            if self.expired():
                return
            hold = self.world.hold_status(planet.id, DEFENSE_HORIZON)
            if hold["fall_turn"] is None and self.enemy_pressure_map.get(planet.id, 0.0) < 1.0:
                continue
            fall_turn = int(hold["fall_turn"] or DEFENSE_HORIZON)
            hold_until = min(BASE_TIMELINE_HORIZON, fall_turn + 6)
            for donor in sorted(self._available_my_planets(excluded_ids=[planet.id]), key=lambda s: distance_planets(s, planet)):
                if self.expired():
                    return
                surplus = self._planet_surplus(donor)
                if surplus <= 0:
                    continue
                plan = self._settle_plan(
                    donor, planet, surplus, mission="reinforce",
                    eval_turn_fn=lambda _eta, hu=hold_until: hu,
                    need_fn=lambda eta, cap, tid=planet.id, hu=hold_until: self.world.reinforcement_needed_to_hold_until(
                        tid, eta, hu, planned_commitments=self.planned_commitments, upper_bound=cap,
                    ),
                    max_turn=fall_turn,
                )
                if plan is not None and plan.eta <= fall_turn:
                    urgency = 100.0 / max(1, fall_turn) + planet.production * 5.0 + self.enemy_pressure_map.get(planet.id, 0.0) * 3.0
                    missions.append(MissionOption(
                        score=urgency, source_ids=[donor.id], target_id=planet.id,
                        angles=[plan.angle], etas=[plan.eta], ships=[plan.ships],
                        needed=plan.required_ships, mission="defend",
                    ))
                    break


# Variant patch: opponent-pressure modelling.
class DecisionLogicV10(_OpponentPressureMixin, DecisionLogic):
    def decide(self) -> List[List[float | int]]:
        try:
            self.reaction_map = self._build_reaction_map()
            self.enemy_pressure_map = self._build_enemy_pressure_map()
            self.modes = self._build_modes()
            self.crashes = detect_enemy_crashes(self.world.arrivals_by_planet, self.state.player)
            missions = self._build_all_missions()
            moves = self._commit_missions(missions)
            return self._finalize(moves)
        finally:
            AGENT_MEMORY["last_owners"] = {planet.id: planet.owner for planet in self.state.planets}
            AGENT_MEMORY["last_step"] = self.state.step


DecisionLogic = DecisionLogicV10


def agent(obs: Any, config: Any) -> List[List[float | int]]:
    try:
        logic = DecisionLogic(obs, config)
        return logic.decide()
    except Exception as exc:
        AGENT_MEMORY["last_error"] = str(exc)
        return []
