from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent
ROOT = WORKSPACE_DIR.parent
BASE_AGENT_PATH = ROOT / "snapshots" / "stage4_leaderboard_search_base.py"
GENOME_AGENT_PATH = WORKSPACE_DIR / "genome_agent.py"


def extract_genome_payload(wrapper_path: Path) -> dict[str, str]:
    text = wrapper_path.read_text(encoding="utf-8")
    match = re.search(r"GenomeConfig\.from_dict\((\{.*?\})\)", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find GenomeConfig.from_dict(...) payload in {wrapper_path}")
    payload = ast.literal_eval(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError(f"Wrapper payload in {wrapper_path} was not a dict")
    return {str(key): str(value) for key, value in payload.items()}


def extract_wrapper_block(source_path: Path = GENOME_AGENT_PATH) -> str:
    text = source_path.read_text(encoding="utf-8")
    frozen_marker = "\n# --- Frozen genome wrapper -------------------------------------------------\n"
    genome_marker = "\nGENOME = GenomeConfig.from_dict("
    if frozen_marker in text and genome_marker in text:
        start = text.index(frozen_marker) + len(frozen_marker)
        end = text.index(genome_marker, start)
        return text[start:end].rstrip()
    start = text.index("GENE_SPACE: Dict[str, Tuple[str, ...]] = {")
    end = text.index("\n\ndef build_agent(")
    block = text[start:end]
    block = block.replace("BASE.", "")
    return block


def extract_base_source(base_path: Path) -> str:
    base = base_path.read_text(encoding="utf-8")
    marker = "\n# --- Frozen genome wrapper -------------------------------------------------\n"
    if marker in base:
        base = base.split(marker, 1)[0].rstrip() + "\n"
    if "import dataclasses as _dataclasses" not in base:
        base = base.replace(
            "from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple\n",
            """from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import dataclasses as _dataclasses
import sys
import types


def dataclass(_cls=None, **kwargs):
    def wrap(cls):
        module = sys.modules.get(cls.__module__)
        if module is None:
            module = types.ModuleType(cls.__module__)
            sys.modules[cls.__module__] = module
        module.__dict__.update(globals())
        return _dataclasses.dataclass(**kwargs)(cls)

    if _cls is None:
        return wrap
    return wrap(_cls)
""",
            1,
        )
    if "def _base_agent_entrypoint(obs: Any, config: Any) -> List[List[float | int]]:" not in base:
        base = base.replace(
            "def agent(obs: Any, config: Any) -> List[List[float | int]]:",
            "def _base_agent_entrypoint(obs: Any, config: Any) -> List[List[float | int]]:",
            1,
        )
    return base


def build_submission_source(
    payload: dict[str, str],
    base_path: Path = BASE_AGENT_PATH,
    wrapper_source_path: Path = GENOME_AGENT_PATH,
) -> str:
    base = extract_base_source(base_path)
    base = base.rstrip() + "\n\n"
    wrapper_block = extract_wrapper_block(wrapper_source_path)
    payload_literal = "{\n" + "\n".join(
        f'    "{key}": "{value}",' for key, value in sorted(payload.items())
    ) + "\n}"
    append = f"""
# --- Frozen genome wrapper -------------------------------------------------
import random
from dataclasses import asdict, dataclass, fields

{wrapper_block}

GENOME = GenomeConfig.from_dict({payload_literal})


def agent(obs: Any, config: Any) -> List[List[float | int]]:
    try:
        logic = GenomeDecisionLogic(obs, config, GENOME)
        return logic.decide()
    except Exception as exc:
        AGENT_MEMORY["last_error"] = str(exc)
        return []
"""
    return base + append.lstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a generated genome wrapper into a Kaggle-ready single file")
    parser.add_argument("--wrapper", required=True, help="Path to a generated genome wrapper .py file")
    parser.add_argument("--output", required=True, help="Path to the standalone output .py file")
    parser.add_argument(
        "--base-source",
        help="Optional path to the base source to freeze against. Can be either a base snapshot or a prior frozen submission.",
    )
    parser.add_argument(
        "--wrapper-source",
        help="Optional path to the wrapper source to freeze against. Can be either genome_agent.py or a prior frozen submission.",
    )
    args = parser.parse_args()

    wrapper_path = Path(args.wrapper).resolve()
    output_path = Path(args.output).resolve()
    base_path = Path(args.base_source).resolve() if args.base_source else BASE_AGENT_PATH
    wrapper_source_path = Path(args.wrapper_source).resolve() if args.wrapper_source else GENOME_AGENT_PATH
    payload = extract_genome_payload(wrapper_path)
    source = build_submission_source(payload, base_path=base_path, wrapper_source_path=wrapper_source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
