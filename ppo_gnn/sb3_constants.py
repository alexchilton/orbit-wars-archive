"""Constants shared between training (sb3_env.py) and submission inference.

No gymnasium or SB3 dependency — safe to import on Kaggle.
"""

from .edge_policy import EDGE_INPUT_DIM, NUM_FRACTIONS, FRACTION_BUCKETS, MAX_ACTIONS

SB3_MAX_CANDIDATES = 96
NUM_CHOICES = SB3_MAX_CANDIDATES * NUM_FRACTIONS + 1  # 961 (960 candidate+fraction combos + 1 noop)
NOOP_ACTION = SB3_MAX_CANDIDATES * NUM_FRACTIONS      # index 960 = noop

# Global game-state features (who's winning)
GLOBAL_DIM = 10
# Temporal context: last T steps of global state
TEMPORAL_STEPS = 8

CANDIDATE_OBS_DIM = SB3_MAX_CANDIDATES * EDGE_INPUT_DIM  # 96 * 74 = 7104
TEMPORAL_OBS_DIM = TEMPORAL_STEPS * GLOBAL_DIM            # 8 * 10 = 80
OBS_DIM = CANDIDATE_OBS_DIM + TEMPORAL_OBS_DIM            # 7184
