# GNN-PPO Agent for Orbit Wars — Tutorial

## Why a GNN?

Orbit Wars is fundamentally about relationships between planets: who owns what, how far apart they are, which paths cross the sun. A standard MLP flattens all this into a vector and loses the structure. A Graph Neural Network (GNN) treats each planet as a node and each planet-pair as an edge, letting the model reason about these relationships directly.

The key advantage: a GNN is **permutation-invariant** (planet ordering doesn't matter) and handles **variable planet counts** (20-40 in real Kaggle games) without padding to a fixed size.

## Architecture Overview

```
Planet observations       Edge features (distance, sun, angle)
       |                              |
  Node Encoder (MLP)            Edge Encoder (MLP)
       |                              |
       +--------- GAT Layer 1 --------+
       |                              |
       +--------- GAT Layer 2 --------+
       |                              |
  Node Embeddings              Global Context (mean pool)
       |                              |
       +--- Source Head (which planet to launch from, or noop)
       |
       +--- Target Head (where to send fleet, conditioned on source)
       |
       +--- Fraction Head (how many ships: 25/50/75/100%)
       |
       +--- Value Head (how good is this state?)
```

**Model size:** ~52.7K parameters (hidden_dim=64), well under the 500k budget.

## The Factored Action Distribution

Instead of a flat distribution over all possible (source, target, fraction) combinations (which grows as N^2 * 4 — that's 3,136 for 28 planets), we factor the action into three sequential decisions:

### Step 1: Pick source planet (or noop)
- One logit per planet + one noop logit
- Masked to planets you own with ships > 0
- Padded planets also masked (for variable planet counts)
- Softmax -> sample or argmax

### Step 2: Pick target planet (given source)
- For the chosen source, compute a score for every other planet
- Score depends on: source embedding, target embedding, edge features between them, global context
- Self-loops masked (can't send to yourself)
- Padded planets masked
- Optionally: sun-blocked paths masked
- Softmax -> sample or argmax

### Step 3: Pick fraction (given source and target)
- Choose from {25%, 50%, 75%, 100%} of ships on source planet
- Depends on source embedding, target embedding, global context

### Why factor?

**Credit assignment.** If the agent sends 50 ships from planet 3 to planet 7 and it goes badly, the flat model has to figure out which part was wrong — was planet 3 a bad source? Planet 7 a bad target? 50% too many ships? The factored model gets separate gradients for each decision.

**Exploration efficiency.** The flat model has to explore a 3136-dim space. The factored model explores three smaller spaces (28 + 28 + 4) that compose.

### Joint log-probability

For PPO's ratio computation, we need log p(action). The factored decomposition gives:

```
log p(action) = log p(source) + log p(target | source) + log p(fraction | source, target)
```

For noop, only log p(source=noop) applies (target and fraction terms are 0).

## Sun Avoidance

The sun at (50, 50) with radius 10 destroys fleets that cross it. We handle this two ways:

1. **Edge feature** (always on): For every planet pair, we compute:
   - `sun_intersects`: binary — does the straight-line path cross within radius + safety margin?
   - `sun_clearance`: continuous — how close does the path get to the sun? (0 = through center, 1 = far away)

   These are part of the 6-dim edge features the GNN uses during message passing and target scoring.

2. **Hard mask** (optional, `--mask-sun` flag): Targets where the path crosses the sun are excluded from the target distribution entirely. This prevents the agent from ever selecting sun-blocked paths, but removes the ability to learn that sometimes a longer route around the sun is worth taking.

## The Noop Collapse Problem

A major issue we encountered: the model learns to always noop (do nothing). With ~32% noop rate in the 2P training data, the model's noop logit gets pushed high during BC training, and the agent refuses to ever launch.

### Why it happens
- Noop is "safe" — no risk of losing ships
- Early game observations (before planets are colonized) are naturally noops
- BC loss treats all training samples equally, so the model learns the easy noops first

### Mitigation strategies (from Checkpoint 014)

1. **Negative noop prior** (implemented): The noop head bias is initialized to -2.0, but BC training overrides this (~+2.8 after training).

2. **Inference-time noop penalty** (implemented): At inference, subtract a penalty (default 4.0) from the noop logit. This shifts the sampling distribution toward launching without changing the trained weights:
   ```python
   source_logits[0, N] -= NOOP_PENALTY  # N is the noop index
   ```

3. **Idle surplus penalty** (implemented in PPO): During online training, a small negative reward (-0.02) is applied when the agent noops with >50 surplus ships.

4. **Noop label downsampling** (not yet applied): Subsample noop transitions during prebatching so the model sees fewer noops during training.

5. **Residual policy** (implemented in hybrid agent): Don't trust the model's noop decision at all — use the heuristic as baseline and only let the model override with a launch action.

## Three-Phase Training Pipeline

### Phase 1: Behavioral Cloning (BC)

**What:** Imitate winning players from Kaggle replays.

**Data:** 346 2P + 280 4P replay files -> ~120K 2P + ~126K 4P winner transitions (pre-batched, max_planets=40).

**Data cleaning:** The prebatching step filters out:
- Transitions where actions reference planets beyond max_planets (truncated)
- Transitions where the source planet isn't owned by the player (observation/action mismatch)

**Loss:** Sum of three cross-entropies:
```
L = CE(source_logits, actual_source) + CE(target_logits, actual_target) + CE(fraction_logits, actual_fraction)
```
Uses the efficient `bc_forward()` path (O(N) per sample) instead of full `forward()` (O(N^2)).

**Run:**
```bash
# Parse replays and train 2P model
python -m ppo_gnn.train_bc --mode 2p --replay-dir kaggle_replays --epochs 50

# Train 4P model
python -m ppo_gnn.train_bc --mode 4p --replay-dir kaggle_replays --epochs 50
```

First run parses all replays and caches to `ppo_gnn/cache/transitions_{2p,4p}.pt`. Subsequent runs load from cache.

**Expected results:** val_loss ~2.95 (2P) / ~3.08 (4P) after 50 epochs. Random baseline would be ~8.8.

**Output:** `ppo_gnn/cache/checkpoint_bc_{2p,4p}.pt`

### Phase 2: Offline RL (Advantage-Weighted Regression)

**What:** Improve beyond imitation by learning from outcomes.

**Key insight:** We use ALL transitions (winners AND losers). The value head learns to predict game outcomes, and the advantage `A = return - V(s)` tells us which actions were better/worse than expected. Winning players' actions get upweighted; losing players' actions get downweighted.

**This is the counterfactual step.** The agent learns: "in this board state, the loser attacked planet X and lost. The winner defended planet Y and won. I should defend Y."

**Temperature annealing:** Controls how aggressively the policy deviates from replay behavior. Starts at 1.0 (mild), ends at 0.1 (aggressive). Policy loss increases with lower temperature — this is expected (sharper advantage weighting).

**Reward scale matching:** The `--reward-scale` flag scales the offline returns to match PPO's terminal reward range (±10). Without this, the AWR value head graduates calibrated for [-1, 1] returns, but PPO uses ±10 terminal rewards — the value baseline is immediately wrong and wastes early PPO episodes recalibrating.

**Run:**
```bash
# With reward scale matching (recommended before PPO):
python -m ppo_gnn.train_offline_rl --mode 2p --bc-checkpoint ppo_gnn/cache/checkpoint_bc_2p.pt --reward-scale 10.0 --device mps
python -m ppo_gnn.train_offline_rl --mode 4p --bc-checkpoint ppo_gnn/cache/checkpoint_bc_4p.pt --reward-scale 10.0 --device mps

# Without (original scale, fine if not doing PPO after):
python -m ppo_gnn.train_offline_rl --mode 2p --bc-checkpoint ppo_gnn/cache/checkpoint_bc_2p.pt
```

**Output:** `ppo_gnn/cache/checkpoint_awr_{2p,4p}.pt`

### Phase 3: Online PPO Fine-Tuning

**What:** Polish the agent against live opponents using the full Kaggle environment.

**Why needed:** Offline RL can only learn from states in the replay data. Online PPO explores new states and adapts to specific opponents.

**Setup:** Uses `kaggle_environments.make("orbit_wars")` directly. The agent plays episodes against a rotating pool of opponents with performance-gated progression.

#### Opponent pool

- **Heuristics:** shun_combined, release_candidate_v2, release_candidate_v3
- **Self-play:** Frozen snapshot of current model, refreshed every 20 episodes
- **Baseline:** Frozen copy of the starting checkpoint (never updated) — tracks absolute progress
- **Lagging:** Snapshot from ~2000 episodes ago — confirms the agent is still improving

The curriculum controls the mix (see below). Baseline and lagging games are sampled at 5% each regardless of curriculum stage.

#### Curriculum (opponent mix)

Performance-gated stages that control how many heuristic games the agent faces. Each stage requires a sustained heuristic win rate over 50+ games to advance:

| Stage | Heuristic mix | Advance at | Description |
|-------|--------------|------------|-------------|
| warmup | 10% | 25% heur wr | Mostly self-play + random, learn fundamentals |
| developing | 40% | 35% heur wr | More heuristic exposure |
| intermediate | 60% | 45% heur wr | Balanced mix |
| advanced | 75% | (terminal) | Heuristic-dominant |

#### Progressive horizon (performance-gated)

Instead of playing full 500-step games from the start, the agent starts with short games and earns longer horizons by demonstrating it can beat heuristics at the current horizon:

| Horizon | Advance condition |
|---------|------------------|
| 50 steps | Start here |
| 100 steps | 75% heuristic wr at 50 steps |
| 200 steps | 75% heuristic wr at 100 steps |
| 500 steps (full game) | 75% heuristic wr at 200 steps |

When the horizon advances, the heuristic results window resets so the agent must re-prove itself at the new horizon. Partial games that hit the horizon cap receive a terminal reward based on fleet+production advantage (capped at ±5) instead of the usual ±10 win/loss.

#### Dense reward signal

Per-step reward combines three signals:
- **Own advantage delta** (`delta * 0.01`): growth in fleet + production relative to opponents
- **Opponent delta-v** (`-enemy_delta * 0.005`): penalises opponent growth, rewards opponent losses
- **Step penalty** (`-0.002`): encourages decisive play
- **Idle penalty** (`-0.02`): applied when the agent noops with >50 surplus ships

#### Monitoring

The log output shows per-episode and per-update metrics:

```
Ep   500/1000000 vs self_play       WIN  steps= 49/50  fleet=120v60  prod=8.0v4.0  launch= 48  noop=  1  (0.3s)
Update: policy=0.05 value=1.20 entropy=1.25 clip=0.15 | wr(50)=70.0% heur_wr=15.0% avg_r=+2.50 base_wr=80% lag_wr=90% stage=warmup horizon=50
```

Every 1000 episodes a summary block is printed:
```
=== Ep 1000 summary ===
  Overall wr: 65.0% (650/1000)
  Heuristic wr (last 100): 15.0% (80 games)
  vs Baseline (last 50): 80% (40 games)
  vs Lagging (last 50): 90% (35 games)
  Curriculum: warmup, Horizon: 50
  Entropy: 1.250, Value loss: 1.2000
```

**Key metrics to watch:**
- `heur_wr` — the real scoreboard. Must reach 75% to unlock the next horizon
- `base_wr` — should climb and stay high. Confirms absolute improvement over starting point
- `lag_wr` — should stay >80%. If it drops, the agent is regressing
- `fleet=XvY` ��� raw game state advantage. Want consistently higher than opponent
- `entropy` — should decrease slowly. Collapse (<0.3) = too deterministic. High (>2.0) = not learning
- `value_loss` — should decrease. Spikes when horizon extends are normal

#### Run

```bash
# Recommended: MPS for gradient updates, CPU for rollouts
python -m ppo_gnn.train_ppo --mode 2p \
  --checkpoint ppo_gnn/cache/checkpoint_awr_2p.pt \
  --progressive-horizon --update-device mps

# Resume from latest checkpoint
python -m ppo_gnn.train_ppo --mode 2p \
  --checkpoint ppo_gnn/cache/checkpoint_ppo_2p_latest.pt \
  --progressive-horizon --update-device mps

# Watch progress
tail -f ppo_gnn/cache/ppo_retrain.log
```

**Output:** `ppo_gnn/cache/checkpoint_ppo_{2p,4p}.pt` (best) and `*_latest.pt`

## Using the Trained Agent

### Pure GNN agent

```bash
python test_agent.py --agent ppo_gnn_agent.py --opponent submission/main_s2_shun_combined.py --games 20 --swap
```

The pure agent uses the best available checkpoint (PPO > AWR > BC) with noop penalty and sun safety checks.

### Hybrid residual agent (recommended)

```bash
python test_agent.py --agent ppo_gnn_hybrid_agent.py --opponent submission/main_release_candidate_v3_antidogpile_position.py --games 20 --swap
```

Uses shun_combined as the baseline, lets the GNN propose overrides only when confident (value estimate above threshold). This is the safest deployment option — the model can only help, not hurt.

### Kaggle submission

To use in a Kaggle submission, the model weights and code need to be bundled into a single file. The key conversion:

```python
from ppo_gnn.gnn_policy import OrbitWarsGNNPolicy, FRACTION_BUCKETS
from ppo_gnn.replay_parser import _build_node_features
import math, torch

model = OrbitWarsGNNPolicy(hidden_dim=64)
model.load_state_dict(torch.load("ppo_gnn/cache/checkpoint_ppo_2p.pt", weights_only=True))
model.eval()

def agent(obs, config):
    planets = obs["planets"]
    fleets = obs.get("fleets", [])
    player = obs["player"]
    num_players = 2  # or detect from context

    nf, pos, owned = _build_node_features(planets, fleets, player, num_players)
    src, tgt, frac, is_noop, _, _ = model.sample_action(nf, pos, owned, deterministic=True)

    if is_noop:
        return []

    src_planet = planets[src]
    tgt_planet = planets[tgt]
    angle = math.atan2(
        float(tgt_planet[3]) - float(src_planet[3]),
        float(tgt_planet[2]) - float(src_planet[2]),
    )
    ships = int(float(src_planet[5]) * FRACTION_BUCKETS[frac])
    return [[int(src_planet[0]), angle, ships]]
```

## GAT vs GraphSAGE

Both are available behind a flag (`--use-sage`). GAT uses multi-head attention to learn which planet relationships matter most in each state. GraphSAGE uses a simpler mean-aggregation with learned edge gates.

In practice, GAT typically performs better for heterogeneous graphs (mixed owned/enemy/neutral planets with varying importance), but GraphSAGE trains faster and is more stable early on.

**Recommendation:** Start with GAT (default). If training is unstable, try GraphSAGE.

## File Overview

```
ppo_gnn/
  __init__.py
  gnn_policy.py          # GNN architecture (OrbitWarsGNNPolicy)
  sun_geometry.py         # Sun intersection math for edge features
  replay_parser.py        # Parse Kaggle replays -> training data
  train_bc.py             # Phase 1: Behavioral Cloning
  train_offline_rl.py     # Phase 2: AWR offline RL
  train_ppo.py            # Phase 3: Online PPO with Kaggle env
  test_gnn.py             # 19 unit tests
  tutorial.md             # This file
  cache/                  # Training data and checkpoints
    transitions_{2p,4p}.pt       # Raw parsed replay transitions
    fast_bc_{2p,4p}.pt           # Pre-batched winners for BC
    fast_all_{2p,4p}.pt          # Pre-batched all-players for AWR
    checkpoint_bc_{2p,4p}.pt     # Phase 1 checkpoints
    checkpoint_awr_{2p,4p}.pt    # Phase 2 checkpoints
    checkpoint_ppo_{2p,4p}.pt    # Phase 3 checkpoints

ppo_gnn_agent.py          # Pure GNN agent (for test_agent.py)
ppo_gnn_hybrid_agent.py   # Hybrid GNN + heuristic agent
```

## Evaluating

Run the trained agent in the existing tournament framework:

```bash
# Quick test: 5 games each way
python test_agent.py --agent ppo_gnn_agent.py --opponent submission/main_s2_shun_combined.py --games 5 --swap

# Full benchmark: hybrid agent
python test_agent.py --agent ppo_gnn_hybrid_agent.py --opponent submission/main_s2_shun_combined.py --games 20 --swap

# Overnight benchmark
python run_overnight_benchmark.py  # (configure agents in the script)
```
