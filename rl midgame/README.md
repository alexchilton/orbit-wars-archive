# RL midgame

This workspace shows how to add a **small RL layer** to Orbit Wars **without handing physics or shot geometry over to RL**.

The idea is:

1. let the base heuristic agent handle:
   - opening theory
   - shot legality
   - sun checks
   - intercept timing
   - send amounts
   - angle selection
2. let RL step in only for the **midgame strategic choice**:
   - *which already-valid mission should we commit next?*

That makes it much closer to your **"opening book + Stockfish in the middlegame"** analogy than to end-to-end RL.

## Where RL fits

The base agent already does this:

1. generates many candidate `MissionOption`s
2. scores them heuristically
3. commits a subset inside `_commit_missions(...)`

So the safest RL seam is:

- keep the heuristic generator
- keep the heuristic shot execution
- let RL **re-rank the top heuristic missions** after the opening

This workspace uses:

- base agent: `snapshots/mtmr_trial_copy_v23.py`
- RL hook point: `_commit_missions(...)`

## What the RL policy controls

The RL policy **does not** output raw actions like:

- launch angle
- ship count
- exact intercept geometry

Instead it chooses among a short list of missions the heuristic already proved are legal and sensible, for example:

- `attack`
- `expand`
- `defend`
- `reinforce`
- `swarm`
- `snipe`
- `comet`

So the action space is:

> **pick one mission from the top-K valid heuristic candidates**

## When RL activates

By default the wrapper only activates RL when all of these are true:

1. the opening is over
2. it is not the very late endgame
3. the turn is inside a configurable window
4. there are at least two valid candidates
5. the position is still strategically contested

That keeps RL focused on the **midgame transition**, which is exactly the part you said feels weakest.

## What goes into the state

The tutorial policy uses a small, interpretable feature vector for each candidate mission. It includes:

### Global game-state features

- turn progress
- remaining steps
- 2-player vs 4-player flag
- whether we are ahead / behind / dominating / finishing
- our ship-share and production-share
- our lead ratio

### Candidate-mission features

- heuristic mission score
- heuristic mission-plus-followup value
- ETA
- ships sent
- source count
- target production
- target ships
- target owner type:
  - neutral
  - enemy
  - friendly
- mission type:
  - attack
  - expand
  - defend
  - reinforce
  - swarm
  - snipe
  - comet
- enemy priority multiplier
- how full the turn launch budget already is

So the policy is learning:

> **"given this game state and these already-valid candidate missions, which one tends to improve final outcome?"**

## How training works

This workspace uses a very small **policy-gradient / REINFORCE-style** setup:

1. run a self-play or opponent game in the local Kaggle harness
2. whenever the RL layer makes a midgame mission choice, store:
   - the candidate feature vectors
   - the chosen candidate
   - the action probabilities
3. at the end of the game, use the final reward as the training signal
4. nudge the policy toward the choices made in games that ended well, and away from those in games that ended badly

That is the simplest way to get a real RL loop without replacing the whole agent.

## Files

| File | Purpose |
| --- | --- |
| `midgame_features.py` | turns `(game state, candidate mission)` into a feature vector |
| `midgame_policy.py` | tiny linear softmax policy + REINFORCE update |
| `midgame_rl_agent.py` | wrapper agent that swaps RL into `_commit_missions(...)` after the opening |
| `train_midgame_policy.py` | tutorial training loop against the local opponent pool |
| `benchmark_midgame_policy.py` | evaluate a saved policy against baseline / v21 / v23 / v16 / mtmr / weird opponents |
| `test_rl_midgame.py` | fast unit tests for the workspace |

## Quick start

### 1. Run the unit tests

```bash
python -m unittest discover -s "rl midgame" -p "test_*.py"
```

### 2. Train a tiny smoke policy

```bash
python "rl midgame/train_midgame_policy.py" --episodes 12 --swap --opponents baseline v21 v23 random
```

That writes:

- `rl midgame/results/training_log.jsonl`
- `rl midgame/results/midgame_policy.json`

### 3. Benchmark the saved policy

```bash
python "rl midgame/benchmark_midgame_policy.py" --policy "rl midgame/results/midgame_policy.json" --games-per-seat 3
```

## Why this is a good first RL design

This setup is intentionally conservative:

- the policy is tiny and interpretable
- the action space is constrained by heuristic legality checks
- the opening stays heuristic
- the late game stays heuristic
- RL only affects the **strategic mission choice** in the middle

That means:

1. less data needed than full end-to-end RL
2. far less chance of breaking the reliable shot pipeline
3. much easier debugging
4. easy upgrade path later

## Natural upgrade path

If this tutorial version proves useful, the next steps are:

1. replace the linear policy with a small MLP
2. log richer trajectory rewards at turn buckets, not only final reward
3. add a **lead-conversion** reward so the policy learns not to throw good openings
4. gate RL on more explicit "critical moment" detectors
5. eventually combine:
   - genome search for the heuristic shell
   - RL for midgame policy
6. split the policy into two layers:
   - a **2p opening strategy policy** until roughly **200-300 opening points / state-strength units**
   - a **midgame conversion policy** like this workspace already demonstrates

### Dual-policy concept

Replay analysis suggests a natural future split:

1. **opening RL (2p only)**:
   - choose which neutrals to prioritize
   - bias toward closer / lower-ETA / better-timed rotating planets
   - decide when to stop expanding and pivot to hostility
   - hand off to midgame RL based on **game-state progress**, not raw step count
2. **midgame RL**:
   - keep the current mission-ranking seam
   - focus on lead conversion and anti-throw decisions

That would match the practical failure modes much better than one monolithic end-to-end policy.

## Important limitation

This is a **local tutorial workspace**, not a finished Kaggle submission path yet.

Like the genome workspace, it is designed to:

- explain the architecture
- let you train and test locally
- make the RL seam concrete

Once a policy looks promising, the last step would be freezing the chosen wrapper and policy into a submission-ready form.
