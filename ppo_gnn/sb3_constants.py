"""Constants shared between training (sb3_env.py) and submission inference.

No gymnasium or SB3 dependency — safe to import on Kaggle.
"""

from .edge_policy import EDGE_INPUT_DIM, NUM_FRACTIONS, FRACTION_BUCKETS, MAX_ACTIONS

SB3_MAX_CANDIDATES = 48
NUM_CHOICES = SB3_MAX_CANDIDATES * NUM_FRACTIONS + 1  # 481 (480 candidate+fraction combos + 1 noop)
NOOP_ACTION = SB3_MAX_CANDIDATES * NUM_FRACTIONS      # index 480 = noop
OBS_DIM = SB3_MAX_CANDIDATES * EDGE_INPUT_DIM         # 48 * 74 = 3552
