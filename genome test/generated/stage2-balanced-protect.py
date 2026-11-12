from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENOME_DIR = ROOT / "genome test"
if str(GENOME_DIR) not in sys.path:
    sys.path.insert(0, str(GENOME_DIR))

from genome_agent import GenomeConfig, build_agent

GENOME = GenomeConfig.from_dict({
  "style_profile": "balanced",
  "duel_opening": "v23",
  "duel_filter": "v23",
  "duel_attack_order": "v23",
  "duel_launch_cap": "v23",
  "value_profile": "balanced",
  "followup_profile": "base",
  "mode_profile": "static",
  "transition_profile": "base",
  "threat_profile": "v23",
  "conversion_profile": "protect",
  "pressure_profile": "guarded",
  "swarm_profile": "base",
  "concentration_profile": "guarded"
})
agent = build_agent(GENOME)
