# Orbit Wars - Current Status & Strategy

*Updated: 2026-05-05*

## Kaggle Leaderboard

| Ref | Elo | Episodes | 2p WR | 4p 1st | Notes |
|---|---|---|---|---|---|
| 52351413 map-adaptive | 925.4 | 32 | 59.1% | 40.0% | Higher Elo but replay metrics worse — 2p regression on close maps (0/2 short-safe, 1/4 short-mid) |
| **52349594 CURG+CL28** | **922.6** | **31** | **64.7%** | **42.9%** | **True best** — strongest replay profile, best balance |
| 52348767 base split | 884.6 | 25 | 64.3% | 36.4% | 4p seat-dependent: 75% from seat 0, 0% from seats 1+2 |

Zachary Maronek (original v131 author) at 855.8 after 10hrs. Our v131-plus improvements are ~70 Elo above his version.

### Replay Analysis Findings
- **Best current candidate: 52349594** (CURG+CL28 only, no map-adaptive)
- **Map-adaptive (52351413) confirmed harmful in 2p** — over-adjusts on easy close maps, consistent with local 40% finding
- **#1 weakness: 4p seat robustness** — all submissions show seat-dependent 4p performance. Likely caused by phase transitions tied to turn number rather than game state, or expand logic that assumes planet positions
- No crash/timeout/error problems — all issues are behavioral
- Keep both active submissions running for data; replace map-adaptive slot when new candidate ready

## Active Candidates (ALL worth continuing)

### 1. v131-plus (on Kaggle)
- **Base:** v131 code with CURG+CL28 grafted in
- **Files:** `main_v131_plus_2p.py`, `main_v131_plus_4p.py`, `main_v131_plus_split.py`
- **Status:** 922-925 Elo, best we have online
- **Next:** New param sweep (`genome_runner_v131_2p.py`, `genome_runner_v131_4p.py`) to find optimal constants

### 2. Shunlite genome candidate
- **Base:** Our original agent with SMASH+CL28+CURG hardcoded
- **File:** `main_genome_candidate.py`
- **Defaults now updated to best sweep values:**
  - `HOSTILE_MARGIN_CAP=8` (was 12) — 70% vs shunlite
  - `STATIC_SCORE_MULT=1.0` (was 1.18) — 60% vs shunlite
  - `PROACTIVE_KEEP_RATIO=0.12` (was 0.18) — 67% in Gen 1
  - `4P_FRONTLINE_RESERVE=0.0` (was 0.18) — zero reserves for 4p
  - `4P_DOUBLE_FRONT_RESERVE=0.0` (was 0.28) — zero reserves for 4p
  - Smash threshold raised to 1.25x (was 0.95x)
- **Status:** Sweep running (PID 14547), Gen 1 in progress
- **Why keep:** On leaderboard the opponent pool is diverse (not just v131). These params could be strong against the field.

### 3. Structural candidate (shunlite + v131 ideas)
- **Base:** Shunlite with GEN_SKIP_VALIDATE=1, GEN_PHASE_COMMIT=1, simplified 4p
- **File:** `main_genome_candidate_structural.py`
- **Best 4p:** **40%** first-rate with zero reserves (best 4p result across all candidates)
- **Status:** Sweep running (PID 31256), Gen 1 in progress
- **Why keep:** Best 4p performer. Skip-validate + phase-commit are proven wins.

### 4. v131-plus with param sweep (NEW)
- **Files:** `genome_runner_v131_2p.py`, `genome_runner_v131_4p.py`
- **Tests against:** v131 original + shunlite (2p), v131 + 2x shunlite (4p)
- **Status:** Ready to launch
- **Why:** Direct improvement path for our Kaggle submission

## Running Sweeps

| Sweep | PID | Progress | Agent | Opponent(s) |
|-------|-----|----------|-------|-------------|
| 2p params (shunlite) | 14547 | 18/27 Gen 0 | genome_candidate | shunlite baseline |
| 4p aggression (shunlite+structural) | 31256 | Gen 1 in progress | genome_candidate + structural | v131 + 2x shunlite |
| v131 2p params | TBD | Not started | v131_plus_2p | v131 original + shunlite |
| v131 4p params | TBD | Not started | v131_plus_4p | v131 + 2x shunlite |

## Key Findings

### Bug Fix: Smash phase fires at parity (FIXED)
- v131-plus smash trigger was `nearby_ships > enemy * 0.95` — fires at opening parity (18 > 17.1)
- On 1-planet start, agent enters smash, only considers enemy targets, can't afford any, returns `[]`
- Opponent expands to 7 planets while we sit idle for 50 turns
- **Fix applied to both 2p and 4p:** `my_planet_count >= 2` guard + threshold raised to `1.25x`
- Confirmed reproducible locally from replay state

### v131-plus 2p sweep (COMPLETE — 57 results, 3 generations)
Tested our v131-plus-2p agent vs v131 original + shunlite, 30 games each.

| Config | vs v131 | vs shunlite | Overall | Gen |
|---|---|---|---|---|
| `SEND_CLEANUP=0.72` | **85%** | 53% | 68% | 1 |
| `CURG_MULT=1.12 + TT_PENALTY=3.5 + SEND_AGGRESSIVE=0.55` | **71%** | **73%** | **72%** | 2 |
| `SEND_CLEANUP=0.72` | 64% | 67% | 66% | 0 |
| `SCORE_TT_PENALTY=3.5` | 62% | 67% | 64% | 0 |
| Baseline | 38% | 67% | 54% | 0 |

**Key insights:**
- `SEND_CLEANUP=0.72` (less commit during cleanup) = 85% vs v131 in Gen 1, dropped to 54% in Gen 2 retests
- `SCORE_TT_PENALTY=3.5` (penalise distant targets more) = prefer nearby = 62% vs v131
- `CURG_MULT=1.12` (stronger comet urgency) slightly better than 1.08
- Best balanced combo: TT_PENALTY=3.5 + CURG_MULT=1.12 + SEND_AGGRESSIVE=0.55 → 71% vs v131, 73% vs shunlite

### v131-plus 4p sweep (COMPLETE — 56 results, 3 generations)
Tested our v131-plus-4p agent vs v131 + 2x shunlite, 20 games each.

| Config | 1st Rate | v131 wins | Gen |
|---|---|---|---|
| `4P_SEND_DOMINATE=0.65` | **40%** | 5/20 | 0 |
| `4P_SEND_DOMINATE=0.65` | **40%** | — | 1,2 |
| `4P_NEUTRAL_DAMP=0.9` | **40%** | 8/20 | 1 |
| `4P_NEUTRAL_DAMP=0.9 + CURG_MULT=1.16` | **40%** | 10/20 | 1 |
| `4P_SEND_DOMINATE=0.9` | 35% | 5/20 | 0 |
| `4P_NEUTRAL_DAMP=0.9` | 35% | 10/20 | 0 |
| Baseline | 30% | 9/20 | 0 |
| `4P_AGGRO_STEP=10` | 20% | 12/20 | 0 |

**Key insights:**
- `4P_SEND_DOMINATE=0.65` most consistent — 40% across all 3 generations
- `4P_NEUTRAL_DAMP=0.9` (less neutral penalty late game) = 40% but drops to 35% in later gens
- `4P_AGGRO_STEP=10` = 20%, confirms early aggro hurts (seat-0-only wins)

### CRITICAL: Param interaction effects (stacking individual winners is HARMFUL)

Sweep tests individual params one-at-a-time. **Combining multiple winners often produces worse results than each alone:**

**v131-plus 2p interactions:**
| Combo | vs v131 | Notes |
|---|---|---|
| SEND_CLEANUP=0.72 alone | 64-85% | High variance across gens |
| TT_PENALTY=3.5 alone | 62% | Consistent |
| TT_PENALTY=3.5 + SEND_AGGRESSIVE=0.55 + SEND_CLEANUP=0.72 | **36%** | WORSE than baseline! |
| TT_PENALTY=3.5 + SEND_AGGRESSIVE=0.55 + CURG_MULT=1.12 | **71%** | Best tested combo |
| All 4 stacked (incl SEND_CLEANUP=0.72) | **38%** | Same as baseline |

**v131-plus 4p interactions:**
| Combo | 1st Rate | Notes |
|---|---|---|
| SEND_DOMINATE=0.65 alone | 40% | Consistent |
| NEUTRAL_DAMP=0.9 alone | 35-40% | |
| SEND_DOMINATE=0.65 + NEUTRAL_DAMP=0.9 | **15%** | WORSE than baseline! |
| SEND_DOMINATE=0.65 + CURG_MULT=1.16 | **30%** | Interaction hurts |

**Shunlite 2p interactions:**
| Combo | Win Rate | Notes |
|---|---|---|
| HOSTILE_MARGIN_CAP=8 alone | 70% | Best single param |
| HOSTILE_MARGIN_CAP=8 + STATIC_SCORE_MULT=1.0 | 53% | Worse together |
| HOSTILE_MARGIN_CAP=8 + PROACTIVE_KEEP_RATIO=0.12 | 37% | Much worse! |

**Rule: Only use the best TESTED combo, not all individual winners stacked.**

### Shunlite 2p sweep (40 results, Gen 1)
| Param | Win Rate | Gen | Notes |
|---|---|---|---|
| HOSTILE_MARGIN_CAP=8 | **70%** | 0 | Confirmed in Gen 1 at 67% |
| PROACTIVE_KEEP_RATIO=0.12 | 67% | 1 | Less defensive reserves |
| STATIC_SCORE_MULT=1.0 | 60% | 0 | Confirmed Gen 1 at 60% |
| HOSTILE_MARGIN_BASE=4 | 60% | 0 | Higher base margin |
| Baseline | 57% | 0 | Reference |

### Shunlite+structural 4p sweep (51 results, Gen 2)
| Candidate | Config | 1st Rate | Gen |
|---|---|---|---|
| Structural | Zero reserves | **40%** | 1 |
| Structural | Zero reserves (variant) | 35% | 1 |
| Baseline | Zero reserves | 30% | 0 |
| Baseline | Zero reserves (confirmed) | 30% | 2 |

### Cross-variant lessons
- Shunlite params that beat shunlite don't necessarily beat v131 head-to-head
- But on the Kaggle leaderboard the opponent pool is mixed — both matter
- "Prefer nearby targets" works across both codebases (TT_PENALTY in v131, ATTACK_COST_WEIGHT in shunlite)
- "Less overcommit" works across both (TAKEOVER_MARGIN in v131, HOSTILE_MARGIN_CAP in shunlite)
- Zero reserves helps 4p across both codebases

## Ideas to Continue

### High Priority
1. **v131-plus param sweep** — find optimal takeover margins, send fractions, aggression timing
2. **Lower takeover margins** — HOSTILE_MARGIN_CAP=8 at 70% shows less overcommit = better. In v131 terms: try TAKEOVER_MARGIN=1.01-1.03
3. **Earlier 4p aggression** — v131 AGGRO_STEP=20 default, try 10-14 (winner-take-all rewards aggression)
4. **Higher 4p send fractions** — current aggressive=0.65, dominate=0.72 → try 0.75-0.85
5. **Combine shunlite winners** — HOSTILE_MARGIN_CAP=8 + STATIC_SCORE_MULT=1.0 + HOSTILE_MARGIN_BASE=4 (Gen 1 will test)

### Comet Mechanics (potentially missing edges)
6. **Comet lifespan off-by-one** — engine spawns comets with path_index=-1, immediately incremented. Test if 18-turn evac is actually safe vs assumed 19
7. **Comet spawn path_index correction** — potential +1 offset in intercept timing. Comet at spawn step starts at path[0] but moves same step → by step N it's at path_index N not N-1
8. **CURG multiplier sweep** — current 1.08, test range 1.04-1.16
9. **Comet score rebalancing** — current base=100, dist_penalty=2.0. Maybe comets are undervalued on tight maps or overvalued on spread
10. **Comet capture cost** — hardcoded 8 ships. Some comets may need fewer or more

### Micro-Edges (from game mechanics analysis)
11. **Discrete production rounding** — `math.ceil(prod * tt)` for low-prod targets (prod<=1 grows in discrete steps)
12. **Production-scaled takeover margins** — high-prod targets grow faster during travel, need more buffer
13. **Combat tie exploitation** — 50 vs 50 = neutral planet. Deliberately force ties on contested planets to waste enemy resources
14. **Phase transition hysteresis** — prevent oscillation when on boundary between expand/grow phases
15. **Adaptive intercept convergence** — orbit_radius-based convergence threshold (0.02 for near-sun, 0.08 for outer)

## What's Been Tested & Results

### Confirmed Harmful
- Map-adaptive overrides on shunlite (40% vs 57% baseline locally)
- "Play like 2p" in 4p (0/14 — gets destroyed)
- STATIC_SCORE_MULT=1.0 vs v131 directly (35% — only helps vs shunlite)
- Combo of best shunlite params vs v131 head-to-head (10% 4p, 35% 2p)
- SNIPE_COST_WEIGHT=1.2 (40%), SWARM_SCORE_MULT=0.9 (37%)

### Confirmed Neutral/Unknown
- Map-adaptive on v131-plus on Kaggle: 925.4 vs 922.6 (too close to call yet)
- NN component of v131: dead weights, files don't exist — all performance is pure heuristics

## Next Submission Candidate (v131-plus "v2")

Bake best TESTED COMBO findings into v131-plus. Replace map-adaptive slot (52351413) when ready.

**2p agent changes (best tested combo from sweep):**
- Smash fix: `my_planet_count >= 2` guard + `1.25x` threshold
- `SCORE_TT_PENALTY=3.5` (was 2.5) — prefer nearby targets
- `CURG_MULT=1.12` (was 1.08) — stronger comet urgency
- `SEND_AGGRESSIVE=0.55` (was 0.4) — slightly more commit when aggressive
- `SEND_CLEANUP=0.82` (UNCHANGED — 0.72 interacts badly with TT_PENALTY=3.5)
- Local validation: 50% vs v131 (24 games) — modest improvement over 38% baseline

**4p agent changes (single best param only):**
- Smash fix: same as 2p
- `4P_SEND_DOMINATE=0.65` (was 0.72) — 40% in sweep, consistent across 3 gens
- `NEUTRAL_DAMP=0.78` (UNCHANGED — interacts badly with SEND_DOMINATE)
- `CURG_MULT=1.08` (UNCHANGED for 4p — interacts badly with SEND_DOMINATE)
- Local validation: pending (v2d test running)

**Shunlite agent changes (single best param only):**
- `HOSTILE_MARGIN_CAP=8` (was 12) — 70% winner, kept
- `STATIC_SCORE_MULT=1.18` (REVERTED from 1.0 — interacts badly with HOSTILE_MARGIN_CAP)
- `PROACTIVE_KEEP_RATIO=0.18` (REVERTED from 0.12 — interacts badly with HOSTILE_MARGIN_CAP)
- Smash threshold 1.25x (was 0.95x) — critical bug fix
- Zero 4p reserves — kept (proven in structural sweep)

**Process:**
1. ~~Wait for sweeps~~ DONE — both v131-plus sweeps complete
2. ~~Hardcode best params~~ DONE — but learned stacking is harmful, using tested combos only
3. Test locally: 20+ games vs v131 in both 2p and 4p — 2p done (50%), 4p running
4. Build single-file submission with `build_v131_plus_submission.py`
5. Submit to replace 52351413 (map-adaptive)

## Strategy

1. **Finish 4p local validation** — v2d test running (SEND_DOMINATE=0.65 only + smash fix)
2. **Sweep overestimates single-param effects** — 15-game samples have high variance. True improvements are more modest than raw sweep numbers suggest. The smash fix is likely the biggest real win.
3. **Never stack individual winners** — always test the full combination. Params interact.
4. **Let Kaggle submissions settle** — 52349594 (CURG+CL28) is true best per replay analysis
5. **Investigate comet mechanics** — potential free Elo from fixing timing bugs (lifespan off-by-one, path_index offset)
6. **Test locally before submitting** — always 20+ games vs v131
7. **Shunlite 2p sweep still running** (PID 14547) — useful for opponent diversity insights
