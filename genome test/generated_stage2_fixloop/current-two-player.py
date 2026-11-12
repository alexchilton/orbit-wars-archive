from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENOME_DIR = ROOT / "genome test"
if str(GENOME_DIR) not in sys.path:
    sys.path.insert(0, str(GENOME_DIR))

from genome_agent import GenomeConfig, build_agent

GENOME = GenomeConfig.from_dict({
  "concentration_profile": "guarded",
  "conversion_profile": "protect",
  "duel_attack_order": "v23",
  "duel_filter": "v23",
  "duel_launch_cap": "v23",
  "duel_opening": "v23",
  "followup_profile": "base",
  "mode_profile": "static",
  "pressure_profile": "guarded",
  "style_profile": "conservative",
  "swarm_profile": "tight",
  "threat_profile": "v23",
  "transition_profile": "base",
  "value_profile": "economy"
})
agent = build_agent(GENOME)
