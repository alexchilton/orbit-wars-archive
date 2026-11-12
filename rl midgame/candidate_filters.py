from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, Sequence, Set

if TYPE_CHECKING:
    from orbit_wars_agent import DecisionLogic, GameStateView, MissionOption, Planet


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FilterConfig:
    """Thresholds and toggles for every filter in the pipeline."""

    # DedupFilter
    dedup_enabled: bool = True
    dedup_exempt_missions: Set[str] = field(
        default_factory=lambda: {"defend", "reinforce"},
    )

    # FeasibilityFilter
    feasibility_enabled: bool = True
    feasibility_ship_ratio: float = 0.85
    feasibility_neutral_margin: int = 2
    feasibility_enemy_margin: int = 4

    # ReserveFilter
    reserve_enabled: bool = True
    reserve_frontline_distance: float = 30.0
    reserve_min_fraction: float = 0.15

    # ConsolidateCandidate
    consolidate_enabled: bool = True
    consolidate_score: float = 0.0


# ---------------------------------------------------------------------------
# Filter protocol
# ---------------------------------------------------------------------------

class CandidateFilter(Protocol):
    """Any callable that winnows a list of candidates."""

    def __call__(
        self,
        candidates: List[MissionOption],
        logic: DecisionLogic,
        config: FilterConfig,
    ) -> List[MissionOption]: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _distance(p1: Planet, p2: Planet) -> float:
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return math.hypot(dx, dy)


def _total_ships_on_planets(planets: Sequence[Planet]) -> int:
    return sum(p.ships for p in planets)


def _in_flight_target_ids(logic: DecisionLogic) -> Set[int]:
    """Return target planet ids that our in-flight fleets are already heading toward.

    We combine two sources:
    * ``logic.targeted_planet_ids`` — targets committed during this turn's
      planning loop (reliable, always available).
    * Existing owned fleets — we don't have an explicit ``target_id`` on
      the Fleet namedtuple, so we conservatively include the already-
      committed set from ``targeted_planet_ids``.
    """
    return set(logic.targeted_planet_ids)


# ---------------------------------------------------------------------------
# 1. DedupFilter
# ---------------------------------------------------------------------------

def dedup_filter(
    candidates: List[MissionOption],
    logic: DecisionLogic,
    config: FilterConfig,
) -> List[MissionOption]:
    """Reject candidates whose target is already under attack by our fleets.

    Missions listed in ``config.dedup_exempt_missions`` (e.g. *defend*,
    *reinforce*) are never rejected because their targets are friendly
    planets that legitimately receive multiple fleets.
    """
    if not config.dedup_enabled:
        return candidates

    already_targeted = _in_flight_target_ids(logic)
    exempt = config.dedup_exempt_missions

    return [
        c for c in candidates
        if c.mission in exempt or c.target_id not in already_targeted
    ]


# ---------------------------------------------------------------------------
# 2. FeasibilityFilter
# ---------------------------------------------------------------------------

def feasibility_filter(
    candidates: List[MissionOption],
    logic: DecisionLogic,
    config: FilterConfig,
) -> List[MissionOption]:
    """Reject candidates whose ship commitment is too low to succeed.

    For *neutral* targets the arriving force must exceed the garrison plus a
    small flat margin.  For *enemy* targets the garrison grows by
    ``production × max_eta`` each turn, so we account for that as well.

    A candidate passes if ``sum(ships) >= required * config.feasibility_ship_ratio``.
    """
    if not config.feasibility_enabled:
        return candidates

    planets_by_id: Dict[int, Planet] = logic.state.planets_by_id
    player = logic.state.player
    kept: List[MissionOption] = []

    for c in candidates:
        # Defend / reinforce always feasible by definition
        if c.mission in ("defend", "reinforce", "consolidate"):
            kept.append(c)
            continue

        target = planets_by_id.get(c.target_id)
        if target is None:
            kept.append(c)
            continue

        ships_sent = sum(c.ships)
        max_eta = max(c.etas) if c.etas else 1

        if target.owner == 0:
            # Neutral planet — static garrison
            required = target.ships + config.feasibility_neutral_margin
        elif target.owner != player:
            # Enemy planet — garrison grows with production
            required = (
                target.ships
                + target.production * max_eta
                + config.feasibility_enemy_margin
            )
        else:
            # Own planet (shouldn't normally be an attack target)
            kept.append(c)
            continue

        if ships_sent >= required * config.feasibility_ship_ratio:
            kept.append(c)

    return kept


# ---------------------------------------------------------------------------
# 3. ReserveFilter
# ---------------------------------------------------------------------------

def reserve_filter(
    candidates: List[MissionOption],
    logic: DecisionLogic,
    config: FilterConfig,
) -> List[MissionOption]:
    """Reject launches that leave the frontline dangerously thin.

    *Frontline* planets are those within ``config.reserve_frontline_distance``
    of the nearest enemy planet.  After subtracting the ships a candidate
    would launch, the total frontline garrison must remain above
    ``config.reserve_min_fraction`` of total owned ships.
    """
    if not config.reserve_enabled:
        return candidates

    my_planets: List[Planet] = logic.state.my_planets
    enemy_planets: List[Planet] = logic.state.enemy_planets

    if not my_planets or not enemy_planets:
        return candidates

    # Identify frontline planets
    frontline_ids: Set[int] = set()
    for mp in my_planets:
        nearest_enemy_dist = min(_distance(mp, ep) for ep in enemy_planets)
        if nearest_enemy_dist <= config.reserve_frontline_distance:
            frontline_ids.add(mp.id)

    if not frontline_ids:
        return candidates

    total_ships = _total_ships_on_planets(my_planets)
    if total_ships == 0:
        return candidates

    min_frontline = total_ships * config.reserve_min_fraction

    # Current frontline garrison (mutable copy)
    frontline_garrison: Dict[int, int] = {
        p.id: p.ships for p in my_planets if p.id in frontline_ids
    }

    kept: List[MissionOption] = []
    for c in candidates:
        # Simulate ships leaving frontline sources
        cost_on_frontline = 0
        for sid, ship_count in zip(c.source_ids, c.ships):
            if sid in frontline_garrison:
                cost_on_frontline += ship_count

        remaining_frontline = sum(frontline_garrison.values()) - cost_on_frontline
        if remaining_frontline >= min_frontline:
            kept.append(c)

    return kept


# ---------------------------------------------------------------------------
# 4. ConsolidateCandidate
# ---------------------------------------------------------------------------

def make_consolidate_candidate(config: FilterConfig) -> MissionOption:
    """Build a synthetic *consolidate* MissionOption.

    This gives the reranker an explicit "do nothing this turn" option,
    distinct from the implicit *defer* that simply runs out of candidates.

    We import MissionOption lazily to avoid circular-import issues when this
    module is loaded before the base agent.
    """
    try:
        from midgame_rl_agent import BASE
        MissionOption = BASE.MissionOption
    except ImportError:
        from orbit_wars_agent import MissionOption  # noqa: F811

    return MissionOption(
        score=config.consolidate_score,
        source_ids=[],
        target_id=-1,
        angles=[],
        etas=[],
        ships=[],
        needed=0,
        mission="consolidate",
        anchor_turn=None,
    )


# ---------------------------------------------------------------------------
# 5. CandidateFilterPipeline
# ---------------------------------------------------------------------------

@dataclass
class CandidateFilterPipeline:
    """Compose filters into a single pass over the candidate list.

    Usage::

        pipeline = CandidateFilterPipeline()          # default filters
        filtered = pipeline.run(candidates, logic)

    Filters execute in insertion order.  The optional *consolidate*
    candidate is appended **after** all filters have run so that filters
    cannot accidentally remove it.
    """

    config: FilterConfig = field(default_factory=FilterConfig)
    filters: List[CandidateFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.filters:
            self.filters = self._default_filters()

    # -- public API ---------------------------------------------------------

    def run(
        self,
        candidates: List[MissionOption],
        logic: DecisionLogic,
    ) -> List[MissionOption]:
        """Apply every filter then optionally append a consolidate option."""
        result = list(candidates)
        for filt in self.filters:
            result = filt(result, logic, self.config)

        if self.config.consolidate_enabled:
            result.append(make_consolidate_candidate(self.config))

        return result

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _default_filters() -> List[CandidateFilter]:
        return [dedup_filter, feasibility_filter, reserve_filter]
