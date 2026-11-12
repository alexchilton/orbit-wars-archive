from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENOME_DIR = ROOT / "genome test"
if str(GENOME_DIR) not in sys.path:
    sys.path.insert(0, str(GENOME_DIR))

from genome_agent import GenomeConfig, build_agent

GENOME = GenomeConfig.from_dict({
  "duel_attack_order": "v23",
  "duel_filter": "v23",
  "duel_launch_cap": "relaxed",
  "duel_opening": "v23",
  "followup_profile": "high",
  "mode_profile": "static",
  "pressure_profile": "off",
  "style_profile": "aggressive",
  "swarm_profile": "loose",
  "threat_profile": "v23",
  "value_profile": "hostile"
})
agent = build_agent(GENOME)
