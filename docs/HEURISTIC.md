# Predictive Agent — Heuristic Decision Engine

## Pipeline Overview

```
obs (raw Kaggle JSON)
    │
    ▼
parse_obs()  +  prune_fleet_registry()
    │
    │  state = { planets: {pid: {owner,ships,prod,x,y,is_orb,...}},
    │             fleets: [...], step, omega, me }
    │
    ├─── Tactical analysis (reads state, pure functions) ────────────────────┐
    │    analyze_threats()        → threats   (inbound enemy fleets)         │
    │    analyze_counter_windows() → windows  (low-garrison enemy planets)   │
    │    build_denial_map()       → denial    (neutrals enemy is heading for)│
    │    find_wave_targets()      → wave_tgt  (sync-attack candidates)       │
    │                                                                         │
    │    Computed globals: my_prod, enemy_prod, ship_ratio, prod_ratio        │
    │    tgt_set = neutrals ∪ (enemies if ratio>1.5 or step>30)              │
    └────────────────────────────────────────────────────────────────────────┘
    │
    ▼
1. EVACUATION  (highest priority)
   ┌─────────────────────────────────────────────────────────────────────┐
   │  For each doomed planet (incoming enemy > garrison + incoming own): │
   │    Send all available ships to nearest friendly planet.             │
   │    Registers in launched{} so beam search sees reduced avail.       │
   └─────────────────────────────────────────────────────────────────────┘
    │
    ▼
2. BEAM SEARCH  (main expansion/attack)
   ┌─────────────────────────────────────────────────────────────────────┐
   │  Top 6 source planets by available ships (skip doomed, skip av<10). │
   │                                                                      │
   │  For each source:                                                    │
   │    a. baseline_score = score( predict(state, 25 steps) )            │
   │       ↑ "do nothing" counterfactual                                 │
   │                                                                      │
   │    b. Pre-filter: rank all tgt_set planets by _rough_roi()          │
   │       Keep top 6 candidates only.                                   │
   │                                                                      │
   │    c. For each candidate:                                            │
   │         eta, intercept_pos = solve_intercept(src, tgt)              │
   │         inject fleet into copy of state                             │
   │         sim_score = score( predict(injected_state, 25 steps) )      │
   │         if sim_score > baseline_score → FIRE, register, break       │
   └─────────────────────────────────────────────────────────────────────┘
    │
    ▼
3. REINFORCEMENT  (defender)
   ┌─────────────────────────────────────────────────────────────────────┐
   │  For each threatened-but-not-doomed planet:                         │
   │    Find nearest friendly planet with spare ships.                   │
   │    Send just enough to survive (deficit × 1.2, capped at 60%).     │
   └─────────────────────────────────────────────────────────────────────┘
    │
    ▼
4. WAVE ATTACK  (second-source guarantee)
   ┌─────────────────────────────────────────────────────────────────────┐
   │  If prod_ratio > 1.4 and wave_targets exist:                        │
   │    Use a second source planet to guarantee simultaneous arrival     │
   │    (ETAs within WAVE_SYNC_TOLERANCE=3 steps of the primary fleet).  │
   └─────────────────────────────────────────────────────────────────────┘
    │
    ▼
output: [(src_pid, angle, ships), ...]
```

---

## _rough_roi — Target Pre-filter

**Formula:**
```
roi = prod × max(0, SIM_HORIZON − eta) − needed × 0.5

where:
  prod         = target planet production rate
  SIM_HORIZON  = 25 steps
  eta          = estimated travel time (steps)
  needed       = ships required to capture target
  0.5          = ship cost weight (sending a ship costs half a score point)
```

**What it encodes:**
- **Production window**: `SIM_HORIZON − eta` is how many steps the captured planet
  has to earn production within the forward simulation. A planet captured in step 5
  earns 20 more steps of production; one in step 24 earns only 1 step.
- **Capture cost**: `needed × 0.5` is the opportunity cost of the ships spent.
  Expensive captures (heavily fortified planets) are penalised.

**Key thresholds:**
| Condition | Result |
|-----------|--------|
| eta > SIM_HORIZON | roi ≤ 0 → ranked below reachable targets |
| needed > av | returns −1e9 sentinel → target never enters beam search |
| Orbiting planet | uses `solve_intercept` ETA (planet has moved by arrival time) |
| Static planet | uses `travel_time` (straight-line distance / fleet_speed(n)) |

**Why this matters:** _rough_roi is a cheap O(1) pre-filter run for every planet.
Only the top 6 by ROI go through the expensive simulation loop. Without it, every
planet would require a 25-step forward simulation (N×M calls per turn).

---

## _score_state — Simulation Scoring

**Formula:**
```
prod_horizon = min(remaining_steps, 40)

score = (my_prod × prod_horizon + my_ships)
      − (enemy_prod × prod_horizon + enemy_ships)
      + Σ(my_fleet_ships × 0.5)
      − Σ(enemy_fleet_ships × 0.5)
```

**What it encodes:**
- **Production valued over 40-step horizon**: capturing a prod=3 planet is worth
  120 score. Early-game expansion compounds — one extra planet earned 200 steps ago
  has produced 600 more ships than if you captured it later.
- **In-transit ships at ±0.5**: sending ships is never free. A fleet of 50 ships costs
  25 score until it arrives and captures something. This prevents the agent from
  firing endlessly without benefit.
- **Neutrals are ignored**: planets with owner=-1 contribute 0 to either side's score.
  Capturing them is desirable, but the value comes from flipping them to owner=0
  (which the simulation handles step-by-step).

**Properties deliberately encoded:**
1. High-production planets > high-ship planets in the early game.
2. The agent prefers a split attack over a concentrated one if both pay off.
3. Winning ships alone doesn't win — you need sustained production advantage.

---

## Baseline Guard

Before firing any attack, the agent computes:
```
baseline_score = score( predict(state, SIM_HORIZON=25) )  # do nothing
```

A move only fires if `sim_score > baseline_score`. This prevents:
- Sending ships at a target that's already destined to be yours (enemy retreating).
- Costly attacks that leave home planets under-defended.
- Fleet-exhaustion: sending the last ships and losing all planets before they arrive.

The baseline is computed once per source planet, then reused for all candidates.

---

## tgt_set — Who Is Targetable?

```python
if enemy_pids and (ship_ratio > 1.5 or prod_ratio > 1.5 or step > 30):
    tgt_set = enemy_pids | neutral_pids   # can attack everyone
else:
    tgt_set = neutral_pids                # early game: expand only
```

**Rationale:**
- Early game (step ≤ 30, equal strength): attacking an entrenched enemy is rarely
  worth it. The defender gets production while your fleet is in transit. Expand
  neutrals first, build ship advantage, then convert that to enemy attacks.
- After step 30, the game is past the expansion phase. Enemy planets become
  legitimate targets regardless of ratio.
- Ratio thresholds (>1.5) catch "decisive advantage" — e.g., we just crushed a
  third player and now dominate production or ships.

Note: being in tgt_set is necessary but not sufficient to trigger an attack.
`_rough_roi` and the baseline guard both independently gate the final decision.

---

## Fleet Speed — Why Distance Matters So Much

```
fleet_speed(n) = 1 + 5 × (log(n) / log(1000))^1.5
```

| Ships | Speed | ETA to 100 units |
|-------|-------|-----------------|
| 10    | 1.9   | 51 steps        |
| 50    | 3.1   | 32 steps        |
| 100   | 3.7   | 27 steps        |
| 500   | 5.3   | 19 steps        |
| 1000  | 6.0   | 17 steps        |

SIM_HORIZON = 25 steps. At typical fleet sizes (10–100 ships), the maximum
profitable attack range is **40–80 units** (beyond that, `_rough_roi ≤ 0`).

**Consequence for sun-blocking:** Non-orbiting planets must be ≥43 units from the
sun. Two non-orbiting planets with a sun-blocking path between them are typically
86–100 units apart. At realistic fleet sizes this is **beyond the profitable range**
— the agent correctly skips these targets. `multi_leg_path` computes the route but
`_rough_roi` filters it out before the expensive sim.

---

## Known Weaknesses

| Weakness | Root Cause | Effect |
|----------|-----------|--------|
| **Dead neural_adjust** | MLP runs every turn, `adj` result never used | Wasted compute, no adaptive aggression |
| **No 4p political targeting** | 2p agent only, treats all enemies equally | Sub-optimal in 4p (should target #2 not #1) |
| **SIM_HORIZON=25 hard cutoff** | Long-range attacks look unprofitable in sim | Agent under-invests in long-range expansion |
| **seed_137 loses to Denial** | Unknown geometry — possibly denials cut off all neutrals early | 0% win rate on that specific map |
| **Single-move beam search** | Only fires one fleet per source planet | May miss coordinated multi-target attacks |
| **av < 10 guard** | Small planets skip beam search | Correct for efficiency, may miss edge-case captures |

---

## Geometry Test Cases (test_geometry_cases.py)

| Class | What it proves |
|-------|----------------|
| `TestCompactMap` | Agent fires when everything is in range; near > far ranking |
| `TestSpreadMap` | FAR targets (ETA > 25 steps) give roi ≤ 0 and are never fired at |
| `TestSunBlockedPath` | `path_crosses_sun` detects correctly; `multi_leg_path` returns valid waypoint; agent correctly skips unprofitable sun-blocked targets |
| `TestOrbitingPlanetGeometry` | `_rough_roi` uses `solve_intercept` ETA for orbiting planets, not naive travel_time |
| `TestProductionPriority` | `_rough_roi` ranks high-prod neutrals above low-prod at same distance |
| `TestEnemyTargetingConditions` | Enemy only enters tgt_set when ratio > 1.5 or step > 30; verified at both sides of each threshold |
| `TestCounterWindow` | Low-garrison enemy is flagged; full-garrison is not |
| `TestWaveAttackGeometry` | Equidistant sources form a wave; very different distances do not |
