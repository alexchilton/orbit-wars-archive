from __future__ import annotations

import importlib.util
import os
from pathlib import Path


_BASE_PATH = Path(__file__).with_name("main_genome_candidate.py")
_managed_keys = (
    "GEN_SKIP_VALIDATE",
    "GEN_PHASE_COMMIT",
    "GEN_4P_WIN_MODE",
    "GEN_4P_SIMPLE_NEED",
    "GEN_4P_FORCE_RATIO",
    "GEN_4P_RESERVE_SCALE",
    "GEN_4P_PHASE_SCORE_BONUS",
)
_saved = {key: os.environ.get(key) for key in _managed_keys}
os.environ["GEN_SKIP_VALIDATE"] = "1"
os.environ["GEN_PHASE_COMMIT"] = "1"
os.environ["GEN_4P_WIN_MODE"] = "1"
os.environ["GEN_4P_SIMPLE_NEED"] = "1"
os.environ["GEN_4P_FORCE_RATIO"] = "0.40"
os.environ["GEN_4P_RESERVE_SCALE"] = "0.20"
os.environ["GEN_4P_PHASE_SCORE_BONUS"] = "1.75"

try:
    _spec = importlib.util.spec_from_file_location(
        f"_genome_candidate_structural_{os.getpid()}_{id(_BASE_PATH)}",
        _BASE_PATH,
    )
    _module = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_module)
finally:
    for _key, _value in _saved.items():
        if _value is None:
            os.environ.pop(_key, None)
        else:
            os.environ[_key] = _value

agent = _module.agent
