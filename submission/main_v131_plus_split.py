"""Split submission wrapper: use the tuned 2p and 4p v131-plus variants."""

from __future__ import annotations

from typing import Any, Iterable

from main_v131_plus_2p import agent as agent_2p
from main_v131_plus_4p import agent as agent_4p


def _extract_rows(obs: Any, attr: str) -> Iterable[Any]:
    if isinstance(obs, dict):
        return obs.get(attr, []) or []
    return getattr(obs, attr, []) or []


def _is_four_player(obs: Any) -> bool:
    owners = set()
    for planet in _extract_rows(obs, "planets"):
        if len(planet) > 1:
            owner = int(planet[1])
            if owner >= 0:
                owners.add(owner)
    for fleet in _extract_rows(obs, "fleets"):
        if len(fleet) > 1:
            owner = int(fleet[1])
            if owner >= 0:
                owners.add(owner)
    return len(owners) > 2 or any(owner >= 2 for owner in owners)


def agent(obs: Any) -> list[list[float | int]]:
    if _is_four_player(obs):
        return agent_4p(obs)
    return agent_2p(obs)
