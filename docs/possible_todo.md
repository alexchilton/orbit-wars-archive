# Orbit Wars — Future Improvement Ideas

*Opus-approved priority list, captured after v5 submission.*

---

## High Impact, Low Risk

**1. Constant tuning (probably the single biggest win)**
All value multipliers and cost weights (`SWARM_VALUE_MULT`, `ATTACK_COST_TURN_WEIGHT`, `HOSTILE_MARGIN_BASE`, etc.) are educated guesses. A parameter sweep — even a simple grid search against self-play or the baseline — would find better values fast. This is the most reliable gain available without architectural changes.

**2. Multi-step lookahead in `_commit_missions`**
Currently greedy: sorts by score, commits top-ranked first. A mission ranked #2 overall might unlock two strong follow-ups while #1 blocks them. Even a shallow 2-deep search (try each mission as first pick, score the remainder) would help. The mission list is small enough to make this tractable within the time budget.

---

## Medium Impact

**3. Opponent modelling**
We react to visible enemy fleets but don't predict next moves. If an enemy has surplus on a planet adjacent to ours, we should preemptively defend or race. The current reaction map is static — it doesn't project what the enemy is *likely* to do next turn.

**4. Coordinated swarm timing**
Current swarm attacks require arrival within `SWARM_ETA_TOLERANCE = 1` turn but don't stagger launches to enforce this. The farther fleet should launch first. Without staggered sends, combined forces can arrive sequentially and get defeated in detail.

**5. Dynamic mode thresholds**
`behind/ahead/finishing` thresholds are fixed ratios. In a 4-player game, being 20% behind the leader but ahead of the other two is very different from a 1v1 deficit. Thresholds should scale with `num_players` and relative standing.

---

## Speculative but Potentially Large

**6. Diplomatic / threat priority awareness**
In multiplayer, attacking the strongest player while ignoring others wastes resources while #3 catches up. A simple "threat priority" heuristic — target the player most likely to win, not just the nearest — could shift outcomes significantly in 4-player games.

---

## Architecture Notes (v5 state)

- `decide()` now uses a unified mission list built by `_build_all_missions()` then globally ranked and committed by `_commit_missions()` — cleaner than the old priority chain
- `ProjectedWorld` handles exact per-planet timelines with full fleet and commitment tracking
- Swarm attacks verified via `extra_arrivals` against `min_ships_to_own_at` before committing
- `reaction_map`, `modes`, `crashes` pre-computed each turn
- All planning loops respect `expired()` against `SOFT_ACT_DEADLINE = 0.82s`
