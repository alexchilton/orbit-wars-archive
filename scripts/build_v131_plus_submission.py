#!/usr/bin/env python3
"""Build single-file submission from v131-plus split files."""
import tarfile, io, os

CWD = '/Users/alexchilton/DataspellProjects/orbit_wars'

with open(f'{CWD}/submission/main_v131_plus_denial.py', 'r') as f:
    source_2p = f.read()
with open(f'{CWD}/submission/main_v131_plus_4p_political.py', 'r') as f:
    source_4p = f.read()

main_py = '''"""Single-file Kaggle submission for split v131-plus with CURG+CL28."""

from __future__ import annotations

from typing import Any, Iterable
import types


_SOURCE_2P = ''' + repr(source_2p) + '''

_SOURCE_4P = ''' + repr(source_4p) + '''


def _load_embedded_agent(source: str, module_name: str):
    namespace = {"__name__": module_name, "__builtins__": __builtins__}
    exec(compile(source, module_name, "exec"), namespace)
    return namespace["agent"]


_AGENT_2P = _load_embedded_agent(_SOURCE_2P, "main_v131_plus_2p")
_AGENT_4P = _load_embedded_agent(_SOURCE_4P, "main_v131_plus_4p")


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
        return _AGENT_4P(obs)
    return _AGENT_2P(obs)
'''

out_path = f'{CWD}/submission_v131_plus_curg_cl28.tar.gz'
with tarfile.open(out_path, 'w:gz') as tar:
    info = tarfile.TarInfo(name='main.py')
    data = main_py.encode('utf-8')
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))

print(f"Built: {out_path} ({len(data)} bytes)")
