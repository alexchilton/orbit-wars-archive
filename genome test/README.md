# Genome test

This directory turns the current Orbit Wars agent family into a **feature genome** so we can search combinations systematically instead of hand-merging whole snapshot files.

## Why this exists

The recent tuning work showed the same pattern over and over:

- a branch can look better in **2-player** and worse in **4-player**
- a "clean" merge can regress because several ideas moved at once
- it is hard to tell whether a gain came from:
  - the duel opener
  - target ordering
  - opening filters
  - commit lookahead
  - threat priority
  - swarm timing

So this workspace treats those pieces as **genes** and lets the benchmark harness evolve combinations.

## Base agent

The genome system now wraps:

- `snapshots/stage4_leaderboard_search_base.py`

That snapshot is based on:

1. `submission/main_release_candidate_v2.py` for the stronger 2p / 4p recovery logic
2. the old stage2 balanced control for safety expectations
3. the anti-dogpile branch for crowded-4p frontline reserve ideas
4. the latest replay analysis for long / low-margin 4p opening dampening

In other words, the genome search now starts from a **leaderboard-oriented safe base**, then varies behavior above it.

## Gene families

The current search space is intentionally discrete and human-readable.

| Gene | Options | Source idea |
| --- | --- | --- |
| `style_profile` | `balanced`, `aggressive`, `conservative` | exploration/exploitation niche |
| `duel_opening` | `v23`, `mtmr`, `shun` | replay-driven opener variants |
| `duel_filter` | `v23`, `baseline_rotating` | baseline-style rotating-neutral discipline |
| `duel_attack_order` | `v23`, `local_pressure`, `production_first` | target ranking variants |
| `duel_launch_cap` | `v23`, `mtmr`, `relaxed` | sparse vs faster launch pacing |
| `value_profile` | `balanced`, `economy`, `hostile`, `finisher` | `possible_todo.md` constant tuning |
| `followup_profile` | `low`, `base`, `high` | `possible_todo.md` commit lookahead weighting |
| `mode_profile` | `static`, `dynamic` | `possible_todo.md` dynamic thresholds |
| `transition_profile` | `base`, `earlier_attack`, `later_attack` | stage-2 midgame transition timing |
| `opening_range_profile` | `base`, `local_bias`, `eta_focus` | replay-driven 2p opening locality / rotator timing |
| `threat_profile` | `v23`, `leader_focus`, `anti_snowball` | `possible_todo.md` diplomatic / threat awareness |
| `crowd_profile` | `base`, `antidogpile`, `hard` | crowded 4p dogpile reserve floor |
| `position_profile` | `base`, `safer_neutrals`, `local_safe` | 4p seat / opening geometry safety bias |
| `vulture_profile` | `off`, `windowed`, `aggressive` | punish overstretched 4p attackers |
| `conversion_profile` | `base`, `protect`, `closeout` | stage-2 lead protection / closeout bias |
| `pressure_profile` | `off`, `guarded` | `possible_todo.md` opponent modelling |
| `swarm_profile` | `tight`, `base`, `loose` | `possible_todo.md` coordinated swarm timing |
| `concentration_profile` | `base`, `guarded`, `strict` | stage-2 force concentration / anti-fragmentation |

## What is implemented

The core file is:

- `genome_agent.py`

It provides:

1. `GenomeConfig` — the discrete genome
2. preset seed genomes
3. mutation and crossover helpers
4. `GenomeDecisionLogic` — a wrapper around the base `DecisionLogic`
5. `build_agent(genome)` — returns an `agent(obs, config)` callable
6. `write_agent_wrapper(...)` — emits locally runnable wrappers for top genomes

### Important design choice

The wrapper does **not** duplicate the whole 3k-line agent. It subclasses and overrides the specific behavior surfaces that matter:

- mode building
- opening style
- launch caps
- opening confidence / filters
- attack ordering
- threat priority
- swarm timing
- mission commit follow-up weighting

That keeps the search space auditable.

## Genetic search runner

The runnable search entrypoint is:

- `genetic_search.py`

It does the following:

1. seeds the population with named presets plus a coverage-biased initial sample
2. benchmarks each genome against:
   - `baseline_agent.py`
   - `submission/main_stage2_oldbase_current_balanced.py`
   - `submission/main_stage2_oldbase_current_two_player.py`
   - `submission/main_release_candidate_v2.py`
   - `submission/main_s2_4p_antidogpile.py`
   - `snapshots/v21.py`
   - `snapshots/v23_state_pivot.py`
   - `snapshots/v16_broken.py`
   - `snapshots/mtmr_trial_copy.py`
   - `random`
   - `greedy`
   - `turtle`
3. adds co-evolutionary pressure through current-population self-play
4. adds mutated past champions as extra robustness opponents
5. keeps **2p**, **4p**, and **balanced** champions separately
6. tracks a **Pareto frontier** on `(robust 2p, 4p)` instead of only one scalar winner
7. preserves the best **aggressive** and **conservative** genomes so crossover does not wash out the extremes
8. writes `search_summary.json` incrementally during the run
9. checkpoints partial progress after each fixed-opponent series, self-play pairing, mutant matchup, and 4-player seat
10. emits current-best wrappers after every generation
11. can resume from `search_log.jsonl`, including partially completed genomes
12. applies elitism, crossover, and mutation
13. emits wrapper agents for the top genomes into `generated/`

### Default scoring

- **fixed 2p score** = average score-rate against the local hall of fame
- **self-play score** = current-generation round-robin result
- **mutant score** = result versus mutated past champions
- **robust 2p objective** = `0.4 * fixed + 0.4 * self-play + 0.2 * mutant`
- **4p score** = weighted blend of normalized average rank and top-2 rate
- **balanced score** = `0.55 * robust_2p + 0.45 * 4p`

The runner still uses the balanced score for breeding pressure, but it also preserves:

- the best **2p** genome
- the best **4p** genome
- the best **balanced** genome
- the **Pareto front**

## Quick start

### Recommended budgets

| Phase | Suggested games per seat | Notes |
| --- | --- | --- |
| Smoke test | `3-5` | catch crashes and obviously bad genomes |
| Full generation search | `10-15` | useful local signal |
| Final validation | `50-100` | before spending Kaggle submission slots |

From the repo root:

```bash
python "genome test/genetic_search.py" --population 6 --generations 2 --games-per-seat 10 --self-play-games-per-seat 3 --mutant-games-per-seat 2 --champion-mutants 3 --two-player-opponents baseline v21 v23 v16 mtmr greedy turtle random --four-player-opponents v23 greedy turtle
```

For a quicker smoke run:

```bash
python "genome test/genetic_search.py" --population 4 --generations 1 --games-per-seat 1 --self-play-games-per-seat 1 --two-player-opponents baseline v21 greedy turtle --skip-four-player
```

To resume an interrupted run:

```bash
python "genome test/genetic_search.py" --resume --population 6 --generations 2 --games-per-seat 10 --self-play-games-per-seat 3 --mutant-games-per-seat 2 --champion-mutants 3 --two-player-opponents baseline v21 v23 v16 mtmr greedy turtle random --four-player-opponents v23 greedy turtle
```

## Output

The search writes:

- `genome test/results/search_log.jsonl`
- `genome test/results/search_summary.json`
- `genome test/generated/*.py`

`search_log.jsonl` now contains both:

- `record_type: "partial"` progress checkpoints
- `record_type: "complete"` finished genome evaluations

The generated wrappers are designed for **local benchmarking**. They import the genome workspace and are not yet frozen into a single-file Kaggle submission.

## Preset genomes

The initial population includes named seeds that map to the earlier analysis:

- `baseline_base`
- `meta_aggressive`
- `meta_conservative`
- `mtmr_duel`
- `todo_constant_tuning`
- `todo_dynamic_thresholds`
- `todo_transition_pivot`
- `todo_local_opening`
- `todo_vulture_window`
- `todo_lead_protection`
- `todo_endgame_closeout`
- `todo_leader_focus`
- `todo_anti_snowball`
- `todo_force_concentration`
- `todo_antidogpile_position`

These are not final agents. They are anchors so the search starts from known ideas rather than pure noise.

## Local hall of fame

The default 2-player opponent pool acts as a small local hall of fame:

- `baseline`
- `oldbase_balanced`
- `oldbase_two_player`
- `release_candidate_v2`
- `v21`
- `v23`
- `v16`
- `mtmr`
- `greedy`
- `turtle`
- `random`

You can trim or expand it with:

```bash
python "genome test/genetic_search.py" --two-player-opponents baseline v21 v23 v16 mtmr random
```

For 4-player fixed evaluation, the default lineup is intentionally less family-only:

- `oldbase_balanced`
- `release_candidate_v2`
- `s2_4p_antidogpile`
- `greedy`
- `turtle`

## How `possible_todo.md` is used

The backlog ideas are included directly as genes or scoring hooks:

| Backlog idea | Genome hook |
| --- | --- |
| Constant tuning | `value_profile` |
| Multi-step lookahead | `followup_profile` |
| Opponent modelling | `pressure_profile` |
| Coordinated swarm timing | `swarm_profile` |
| Dynamic mode thresholds | `mode_profile` |
| Midgame attack timing pivot | `transition_profile` |
| Prefer local / low-ETA openings | `opening_range_profile` |
| Diplomatic / threat awareness | `threat_profile` |
| Punish overstretched attacker | `vulture_profile` |
| Lead protection / closeout | `conversion_profile` |
| Stop fragmented harassment | `concentration_profile` |

So the backlog is now part of the search space instead of living only in notes.

## Suggested workflow

1. Run a quick search with cheap settings.
2. Inspect `search_summary.json`.
3. Re-benchmark the top wrappers with a larger sample.
4. If one genome is clearly better in **2p** but not **4p**, keep separate champions.
5. Only freeze a submission after a larger confirmation run.

## Current limitations

This workspace is stronger than the first draft, but it is still a **heuristic search system**, not learning.

Right now:

- `followup_profile` is still **shallow**: it adjusts the weight of the existing one-step follower bonus in `_commit_missions`; it is not deep tree search
- `pressure_profile` is still **reactive**: it adjusts opening discipline and contested-neutral behavior; it is not predictive opponent modeling
- `mode_profile` is still **hand-tuned dynamic**: it changes thresholds by game shape; it is not learned dynamics
- `opening_range_profile` is still **heuristic**: it biases toward nearer and lower-ETA 2-player openings; it does not explicitly solve a learned opening-book problem

So if the genome search finds a strong local maximum but still trails the ladder badly, that is the signal to hybridize rather than just keep tuning the same heuristic family.

## Next extensions

Good follow-on upgrades for this workspace:

1. add a real Pareto search for separate 2p / 4p champions
2. freeze emitted wrappers into standalone Kaggle files
3. expose more numeric genes around reserves and value multipliers
4. add replay-derived scorers such as sun-loss rate or wasted opening launches
5. add deeper lookahead or a lightweight learned component on top of the GA-winning heuristics
6. add a dedicated 2p opening policy layer for the first ~200 steps

## Stage 2 notes

The next stage is **not lost**. The current workspace now includes the first stage-2 gene:

- `concentration_profile`

It now also includes:

- `transition_profile`
- `conversion_profile`
- `opening_range_profile`

It penalizes or blocks hostile missions that still leave the target uncaptured while drip-feeding small amounts of force into it, especially when:

- you are already ahead
- you already have friendly fleets inbound to the same target
- the new launch still does not convert into ownership

Together, the current stage-2 genome additions now cover:

1. stronger midgame transition timing
2. 2-player local / low-ETA opening bias
3. 4-player vulture / overextension punishment
4. lead protection / stop-overextending while ahead
5. endgame closeout preference
6. anti-fragmentation / force concentration

Still pending for a later extension:

1. more explicit anti-grinder telemetry
2. replay-derived "wasted harassment" scorers

### Replay-driven note on far planets

There is **no direct bonus** for taking a farther neutral just because it is farther away.
The only real upside is indirect:

1. higher production
2. safer reaction timing versus the opponent
3. map shape / follow-up geometry

So if replay analysis shows the agent drifting into long-ETA rotating targets too early, that is a real opening-policy bug, not a hidden distance reward.
