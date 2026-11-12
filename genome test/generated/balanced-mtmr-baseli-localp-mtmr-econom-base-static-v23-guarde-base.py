from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENOME_DIR = ROOT / "genome test"
if str(GENOME_DIR) not in sys.path:
    sys.path.insert(0, str(GENOME_DIR))

from genome_agent import GenomeConfig, build_agent

GENOME = GenomeConfig.from_dict({
  "duel_attack_order": "local_pressure",
  "duel_filter": "baseline_rotating",
  "duel_launch_cap": "mtmr",
  "duel_opening": "mtmr",
  "followup_profile": "base",
  "mode_profile": "static",
  "pressure_profile": "guarded",
  "swarm_profile": "base",
  "threat_profile": "v23",
  "value_profile": "economy"
})
agent = build_agent(GENOME)
