#!/usr/bin/env python3
"""Build a single-file Kaggle submission for the split v131-plus agent."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_2P = ROOT / "submission" / "main_v131_plus_2p.py"
SOURCE_4P = ROOT / "submission" / "main_v131_plus_4p.py"


def build_single_file(two_player_source: str, four_player_source: str) -> str:
    return f'''"""Single-file Kaggle submission for split v131-plus."""

from __future__ import annotations

from typing import Any, Iterable

_SOURCE_2P = {two_player_source!r}
_SOURCE_4P = {four_player_source!r}


def _load_embedded_agent(source: str, module_name: str):
    namespace = {{
        "__name__": module_name,
        "__file__": "main.py",
    }}
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build single-file split v131-plus submission")
    parser.add_argument(
        "--output-main",
        default=str(ROOT / "submission_single_main.py"),
        help="Path for the generated single-file main.py source",
    )
    parser.add_argument(
        "--output-tar",
        default=str(ROOT / "submission_v131_plus_split_single.tar.gz"),
        help="Path for the generated tar.gz archive",
    )
    args = parser.parse_args()

    main_path = Path(args.output_main).resolve()
    tar_path = Path(args.output_tar).resolve()

    main_source = build_single_file(
        SOURCE_2P.read_text(encoding="utf-8"),
        SOURCE_4P.read_text(encoding="utf-8"),
    )
    main_path.write_text(main_source, encoding="utf-8")

    with tarfile.open(tar_path, "w:gz") as handle:
        handle.add(main_path, arcname="main.py")

    print(main_path)
    print(tar_path)


if __name__ == "__main__":
    main()
