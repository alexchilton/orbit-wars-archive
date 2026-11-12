# PPO-GNN Agent Design Spec

## Overview

Upgrade the Orbit Wars RL agent from the current flat-action GCN (`rl_fundamentals/orbit_wars_rl.py`) to a GAT-based GNN with factored action distribution and a three-phase training pipeline that leverages 626 Kaggle replay files for offline RL with counterfactual learning.

## Motivation

1. **Factored action distribution** — the current flat 501-dim Categorical has poor credit assignment. Factoring into source -> target|source -> fraction lets the agent learn "planet 3 is a good source" independently from "planet 7 is a good target".
2. **GAT attention** — multi-head attention over planet relationships replaces the current scalar edge-weight GCN, better capturing which relationships matter per state.
3. **Replay-based offline RL** — 346 2P + 280 4P Kaggle replays (20-40 planets each, 500 steps) provide ~600K+ real state-action pairs. This enables counterfactual learning: "this action led to a loss — the policy should learn to avoid it in similar states".
4. **Variable planet count** — real Kaggle games have 20-40 planets (not the simplified env's 7/10). The GNN handles this naturally.

## Constraints

- Parameter budget: under 500k total
- Hidden dim: 64-128
- PyTorch only (no PyG dependency — roll message passing manually)
- Separate 2P and 4P checkpoints (different games, different strategies)
- Must include unit tests and tutorial

## GNN Policy Network (`gnn_policy.py`)

### Node Features (10 dims per planet)

| Feature | Dim | Description |
|---------|-----|-------------|
| x, y | 2 | Position normalized to [0,1] by /100 |
| is_mine | 1 | Binary: owned by current player |
| is_enemy | 1 | Binary: owned by any enemy |
| is_neutral | 1 | Binary: unowned |
| ships | 1 | Log-scaled ship count: log(1 + ships) |
| production | 1 | Raw production value |
| ship_ratio | 1 | my_ships / total_ships on planet (0 if neutral) |
| inbound_friendly | 1 | Log-scaled sum of friendly fleet ships heading here |
| inbound_enemy | 1 | Log-scaled sum of enemy fleet ships heading here |

### Edge Features (6 dims per planet pair)

| Feature | Dim | Description |
|---------|-----|-------------|
| distance | 1 | Euclidean distance, normalized by board diagonal |
| travel_time | 1 | distance / ship_speed |
| angle_sin | 1 | sin(angle from source to target) |
| angle_cos | 1 | cos(angle from source to target) |
| sun_intersects | 1 | Binary: straight-line path crosses sun danger zone |
| sun_clearance | 1 | Closest approach to sun center, normalized. 0 = through center, 1 = far away |

### Sun Geometry (`sun_geometry.py`)

Given two planet positions and sun center (50, 50) with radius 10:
- Compute closest point on line segment to sun center
- `sun_intersects` = 1 if closest distance < sun_radius + safety_margin (configurable, default 2.0)
- `sun_clearance` = clamp(closest_distance / board_diagonal, 0, 1)
- Optional hard mask: exclude sun-blocked targets from the target distribution (configurable flag)

### Architecture

```
Node encoder:  Linear(10, 64) -> ReLU -> Linear(64, 64) -> ReLU
Edge encoder:  Linear(6, 32) -> ReLU -> Linear(32, 32)

GAT Layer (x2):
  - 4 attention heads, head_dim = 16, total = 64
  - Edge-conditioned: attention score = LeakyReLU(a^T [Wh_i || Wh_j || e_ij])
  - Residual connection + LayerNorm after each layer

GraphSAGE variant (behind flag):
  - Mean aggregation of neighbor embeddings
  - Same 2-layer structure, same dims

Global context:
  - Mean-pool all node embeddings -> Linear(64, 64) -> ReLU
```

### Factored Action Head

The action is decomposed as: **noop vs launch**, then if launch: **source -> target|source -> fraction|source,target**.

1. **Noop/Source selection**: Per-node logits via `Linear(64, 1)` on node embeddings. Noop gets a separate logit from the global context. Softmax over `[noop, planet_0, planet_1, ..., planet_N]`. **Masked**: only owned planets with ships > 0 are valid sources.

2. **Target selection**: Given selected source node s, compute `MLP([h_s, h_t, e_{s,t}, global])` for each candidate target t. Architecture: `Linear(64+64+32+64, 64) -> ReLU -> Linear(64, 1)`. Softmax over all planets except source. **Optionally masked** by sun intersection (configurable).

3. **Fraction selection**: Given source s and target t, compute `MLP([h_s, h_t, global]) -> softmax over {25%, 50%, 75%, 100%}`. Architecture: `Linear(64+64+64, 64) -> ReLU -> Linear(64, 4)`.

**Joint log-probability**: `log p(source) + log p(target | source) + log p(fraction | source, target)`

When noop is selected, target and fraction log-probs are 0.

### Value Head

```
mean_pool(node_embeddings) -> concat(global_context) -> Linear(128, 64) -> ReLU -> Linear(64, 1)
```

Permutation-invariant via mean pooling.

### Estimated Parameters

| Component | Params |
|-----------|--------|
| Node encoder | ~8.5k |
| Edge encoder | ~1.2k |
| GAT layers (x2) | ~33k |
| Global context | ~4.2k |
| Source head | ~65 |
| Target head | ~14.5k |
| Fraction head | ~12.5k |
| Noop head | ~65 |
| Value head | ~8.4k |
| **Total** | **~83k** |

Well under 500k budget. Room to increase hidden dim to 128 if needed (~330k).

## Replay Parser (`replay_parser.py`)

### Input

Kaggle replay JSON files from `kaggle_replays/*/episode-*-replay.json`.

### Processing

1. Load replay, detect mode (2P or 4P) from `len(steps[0])`
2. For each step and each player:
   - Extract observation: planets, fleets, angular_velocity
   - Build node features (10-dim per planet) and edge features (6-dim per pair)
   - Convert angle-based actions `[source_id, angle, ships]` to `(source_id, target_planet_id, fraction_bucket)`:
     - Cast ray from source planet at given angle
     - Find planet closest to ray (smallest perpendicular distance within a cone tolerance)
     - Snap `ships / source_ships` to nearest fraction bucket {0.25, 0.5, 0.75, 1.0}
   - If action is empty list (noop), record as noop
3. Compute discounted returns per player: `R_t = sum_{k=t}^{T} gamma^{k-t} * r_k` where final reward is +1 (win) or -1 (loss), intermediate rewards are 0
4. Tag each transition with `(mode, player_rank, discounted_return)`

### Output

`ReplayDataset` class (torch Dataset) containing:
- `graph_obs`: node features tensor, edge features tensor, valid mask
- `action`: (source_idx, target_idx, fraction_idx) or noop flag
- `action_mask`: valid sources (owned planets with ships)
- `discounted_return`: float
- `player_rank`: 1-4 (4P) or 1-2 (2P)
- `mode`: "2p" or "4p"

Separate datasets for 2P and 4P. Saved as `.pt` files for fast reloading.

### Filtering

- All transitions kept (winners and losers) — losers provide negative signal for AWR
- Transitions where player has been eliminated (0 planets) are dropped
- Noop actions on turns where the player has valid attacks available are flagged (potentially low-quality)

## Training Pipeline

### Phase 1: Behavioral Cloning (`train_bc.py`)

**Goal**: Imitate winning players' actions from replay data.

**Data**: Filter to transitions from rank-1 players (winners). ~300K transitions for 4P, ~170K for 2P.

**Loss**: `L_BC = CE(source_logits, source_target) + CE(target_logits, target_target) + CE(fraction_logits, fraction_target)`

Three cross-entropies summed, one per factored component. For noop actions, only the source CE applies (noop is a source option).

**Training**:
- Adam optimizer, lr=3e-4, weight decay=1e-4
- Batch size 128
- Up to 50 epochs, early stopping on validation loss (10% holdout)
- Gradient clipping at 1.0
- Separate runs for 2P and 4P

**Output**: `checkpoint_bc_2p.pt`, `checkpoint_bc_4p.pt`

### Phase 2: Offline RL via Advantage-Weighted Regression (`train_offline_rl.py`)

**Goal**: Improve beyond imitation by learning from outcomes — upweight actions that led to wins, downweight actions that led to losses.

**Method**: AWR (Advantage-Weighted Regression)
- Initialize from Phase 1 checkpoint
- For each transition in the FULL dataset (winners AND losers):
  1. Compute value estimate `V(s)` using the value head
  2. Compute advantage `A = discounted_return - V(s)`
  3. Compute weight `w = exp(A / temperature)`, clamped to [0, 20]
  4. Policy loss = `-w * log_prob(action|state)` (weighted BC)
  5. Value loss = `MSE(V(s), discounted_return)`
  6. Total loss = `policy_loss + 0.5 * value_loss`

**Temperature**: Start at 1.0, anneal down to 0.1 over training. Lower temperature = more aggressive deviation from replay behavior.

**Counterfactual signal**: Losing players' transitions get negative advantages (their actions were bad given the outcome), so the policy learns to avoid those actions in similar states. Winning players' transitions get positive advantages (reinforced).

**Training**:
- Adam, lr=1e-4 (lower than BC)
- Batch size 256
- 20 epochs over full dataset
- Gradient clipping at 0.5

**Output**: `checkpoint_awr_2p.pt`, `checkpoint_awr_4p.pt`

### Phase 3: Online PPO Fine-Tuning (`train_ppo.py`)

**Goal**: Polish the agent against live opponents, fixing distributional shift from offline training.

**Setup**:
- Initialize from Phase 2 checkpoint
- Play against heuristic opponents (rc3_antidogpile and others from `submission/`)
- Uses existing `SimplifiedPlanetEnv` for 2P (7 planets) or an adapter for full Kaggle env

**PPO details**:
- Factored log-prob for ratio computation: `ratio = exp(new_log_prob - old_log_prob)` where log_prob is the joint factored probability
- Clip epsilon: 0.2
- GAE lambda: 0.95, gamma: 0.99
- Entropy bonus: 0.01 (applied to each factored component)
- Value coefficient: 0.5
- Rollout length: 128 steps
- Num envs: 8 (vectorized)
- Update epochs: 4, mini-batch size: 64

**Self-play option**: Every N updates, snapshot the current policy as a new opponent. Maintain a pool of recent snapshots to play against.

**Output**: `checkpoint_ppo_2p.pt`, `checkpoint_ppo_4p.pt`

### Note on Phase 3 Environment

The existing `SimplifiedPlanetEnv` only supports 7-10 planets. For Phase 3 to train on realistic 20-40 planet boards, we would need either:
- A Kaggle env adapter (if available)
- An extended version of SimplifiedPlanetEnv with configurable planet counts

Phase 3 can start with the simplified env and be upgraded later. Phases 1-2 work entirely from replay data and don't need a live env.

## File Structure

```
ppo_gnn/
  __init__.py
  gnn_policy.py          # GNN model: GAT/GraphSAGE backbone, factored action head, value head
  replay_parser.py       # Load Kaggle replays -> ReplayDataset (2P and 4P)
  sun_geometry.py        # Sun intersection computation for edge features and optional target mask
  train_bc.py            # Phase 1: behavioral cloning from winning replay actions
  train_offline_rl.py    # Phase 2: advantage-weighted regression on full replay data
  train_ppo.py           # Phase 3: online PPO fine-tuning against heuristic opponents
  test_gnn.py            # Unit tests: shapes, masking, sun geometry, replay parsing, log-prob
  tutorial.md            # Walkthrough: architecture, action factoring, training phases, usage
```

## Unit Tests (`test_gnn.py`)

1. **Forward pass shapes**: Create fake batch with 7 planets and 28 planets. Verify source logits shape = (batch, N+1), target logits shape = (batch, N), fraction logits shape = (batch, 4), value shape = (batch,).
2. **Action masking**: Set only 2 planets as owned. Verify source logits are -inf for unowned planets.
3. **Sun intersection**: Known cases — two planets with sun directly between them (should intersect), two planets on same side of sun (should not).
4. **Factored log-prob**: Sample an action, compute joint log-prob, verify it equals sum of three component log-probs.
5. **Replay parser**: Parse a single replay file, verify output shapes and that angle-to-target conversion produces valid planet IDs.
6. **GAT vs GraphSAGE**: Both variants produce same output shapes.

## Tutorial (`tutorial.md`)

Sections:
1. Why GNN for Orbit Wars (relational reasoning between planets)
2. The factored action distribution (with diagrams showing source -> target -> fraction)
3. Sun-avoidance geometry
4. Three-phase training pipeline overview
5. How to run each phase
6. How to evaluate: plug the trained agent into the tournament framework
7. Interpreting attention weights (which planet pairs the model focuses on)
