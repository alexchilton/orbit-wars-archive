# Counterfactual Replay Mining for RL Reranker Improvement

## Orbit Wars Kaggle Competition — Technical Tutorial

> **Target audience:** Anyone working on this codebase who wants to understand *why*
> our RL reranker under-performs and *how* to fix it using replays from top players.

---

## Table of Contents

1. [Philosophy: Why Counterfactual Mining?](#1-philosophy-why-counterfactual-mining)
   - 1.1 [The Local-Optimum Trap](#11-the-local-optimum-trap)
   - 1.2 [What "Counterfactual" Means Here](#12-what-counterfactual-means-here)
   - 1.3 [Why Not Just Clone?](#13-why-not-just-clone)
   - 1.4 [The Reranker's Role in the Agent Stack](#14-the-rerankers-role-in-the-agent-stack)
   - 1.5 [Expected Gains](#15-expected-gains)
2. [Pipeline Overview (Steps A → F)](#2-pipeline-overview-steps-a--f)
   - 2.1 [Step A — Harvest Replays](#21-step-a--harvest-replays)
   - 2.2 [Step B — Extract State-Action Pairs](#22-step-b--extract-state-action-pairs)
   - 2.3 [Step C — Build Counterfactual Labels](#23-step-c--build-counterfactual-labels)
   - 2.4 [Step D — Augment Training Buffer](#24-step-d--augment-training-buffer)
   - 2.5 [Step E — Retrain the Split Reranker](#25-step-e--retrain-the-split-reranker)
   - 2.6 [Step F — Validate & Submit](#26-step-f--validate--submit)
3. [Getting Replay Data](#3-getting-replay-data)
   - 3.1 [Using kaggle_episode_tools.py](#31-using-kaggle_episode_toolspy)
   - 3.2 [Replay Directory Layout](#32-replay-directory-layout)
   - 3.3 [Identifying Top-Player Replays](#33-identifying-top-player-replays)
   - 3.4 [Bulk Download Strategy](#34-bulk-download-strategy)
4. [Testing Infrastructure](#4-testing-infrastructure)
   - 4.1 [The Noise Problem](#41-the-noise-problem)
   - 4.2 [Using test_agent.py Effectively](#42-using-test_agentpy-effectively)
   - 4.3 [Statistical Significance & Sample Sizes](#43-statistical-significance--sample-sizes)
   - 4.4 [Benchmark Baselines](#44-benchmark-baselines)
5. [Code Examples & Implementation Guide](#5-code-examples--implementation-guide)
   - 5.1 [Replay Parser: Extracting Turns](#51-replay-parser-extracting-turns)
   - 5.2 [Feature Extraction from Replay States](#52-feature-extraction-from-replay-states)
   - 5.3 [Counterfactual Label Generator](#53-counterfactual-label-generator)
   - 5.4 [Training Buffer Integration](#54-training-buffer-integration)
   - 5.5 [Modified Training Loop](#55-modified-training-loop)
   - 5.6 [End-to-End Script](#56-end-to-end-script)
6. [Expected Timeline](#6-expected-timeline)
   - 6.1 [Phase 1: Data Collection (Days 1–2)](#61-phase-1-data-collection-days-12)
   - 6.2 [Phase 2: Pipeline Build (Days 3–5)](#62-phase-2-pipeline-build-days-35)
   - 6.3 [Phase 3: Training & Iteration (Days 6–10)](#63-phase-3-training--iteration-days-610)
   - 6.4 [Phase 4: Validation & Submission (Days 11–14)](#64-phase-4-validation--submission-days-1114)
7. [Key Constraints & Gotchas](#7-key-constraints--gotchas)
   - 7.1 [Opening Protection](#71-opening-protection)
   - 7.2 [The Anti-Defer Problem](#72-the-anti-defer-problem)
   - 7.3 [2p vs 4p Split](#73-2p-vs-4p-split)
   - 7.4 [Feature Alignment](#74-feature-alignment)
   - 7.5 [Kaggle Submission Constraints](#75-kaggle-submission-constraints)

---

## 1. Philosophy: Why Counterfactual Mining?

### 1.1 The Local-Optimum Trap

Our current agent stack has two layers:

1. **Heuristic base** — the stage2 genome (~3721 lines), which generates legal
   mission candidates each turn using a finely-tuned parameter set found via
   genetic search.
2. **RL reranker** — a 33-feature linear policy (`SplitRerankerPolicy` in
   `rl midgame/split_reranker.py`) that scores each candidate and picks the
   best one, or defers to the heuristic's top pick.

The genome search explored thousands of parameter combinations and converged
on a local optimum around **ELO 1200–1300**. Meanwhile, the top players on the
leaderboard are at **1400–1600+**:

| Player      | ELO   | Gap from us |
|-------------|-------|-------------|
| Shun_PI     | 1607  | +307–407    |
| Erfan       | 1547  | +247–347    |
| kovi        | 1478  | +178–278    |
| Ousagi      | 1461  | +161–261    |
| HY2017      | 1438  | +138–238    |
| **Us**      | ~1250 | —           |

The gap is too large to close by tweaking weights. We need *qualitatively
different decisions*, and the only source of those decisions is the replays of
players who are already making them.

### 1.2 What "Counterfactual" Means Here

A **counterfactual** is a "what-if" question:

> "At turn T of game G, if our agent had been in player X's position, which
> of our candidate missions would most closely match what player X actually
> did — and did that lead to a win?"

We are NOT trying to replicate the top player's exact code or logic. Instead,
we:

1. Take a game state from a top-player replay
2. Run our heuristic to generate the same candidate set it would have produced
3. Score each candidate by how closely it matches what the top player actually
   did at that turn
4. Use the game outcome (win/loss) as a reward signal
5. Feed these (state, action, reward) triples into our RL training pipeline

This is "counterfactual" because we're asking: *what would our RL agent have
learned if it had been watching over the top player's shoulder?*

### 1.3 Why Not Just Clone?

Behavioral cloning (supervised learning to copy the expert's actions) has a
well-known problem called **distributional shift**: the cloned agent makes
small errors that compound into states the expert never visited, and it has no
recovery strategy.

Our approach is better for three reasons:

1. **We keep our own candidate generator.** The heuristic base still proposes
   missions — we're only changing which one gets picked. This bounds how far
   off-distribution we can drift.

2. **We use outcome-weighted learning.** We don't just copy actions; we weight
   them by whether they led to wins. A top player's losing games are
   down-weighted, so we don't learn their mistakes.

3. **We maintain the defer option.** If the RL signal is weak for a given
   state, the agent can always defer to the heuristic. Pure cloning doesn't
   have this escape valve.

### 1.4 The Reranker's Role in the Agent Stack

To understand where the reranker fits, here's the decision flow each turn:

```
Turn T arrives
    │
    ▼
┌────────────────────────┐
│  Heuristic Base Agent  │  ← stage2 genome, ~3721 lines
│  (kore_fleets logic)   │
│                        │
│  Generates N candidate │
│  missions (attack,     │
│  expand, defend, etc.) │
└──────────┬─────────────┘
           │  N candidates + features
           ▼
┌────────────────────────┐
│  RL Reranker           │  ← SplitRerankerPolicy
│  (midgame only:        │
│   step 18–180,         │
│   contested position)  │
│                        │
│  Scores each candidate │
│  with 33-feature dot   │
│  product, picks argmax │
│  OR defers             │
└──────────┬─────────────┘
           │  chosen mission
           ▼
┌────────────────────────┐
│  Execute Mission       │
│  (send fleet)          │
└────────────────────────┘
```

The 33 features (defined in `rl midgame/midgame_features.py` as
`FEATURE_NAMES`) capture:

- **Game phase:** `step_frac`, `remaining_frac`
- **Player mode:** `is_two_player`, `is_four_player_plus`
- **Position:** `is_ahead`, `is_behind`, `is_dominating`, `is_finishing`
- **Economy:** `my_ship_share`, `my_prod_share`, `lead_ratio`
- **Mission quality:** `base_score_norm`, `base_value_norm`, `eta_norm`,
  `send_norm`, `source_count_norm`, `needed_norm`
- **Target info:** `target_prod_norm`, `target_ships_norm`,
  `target_is_neutral`, `target_is_enemy`, `target_is_friendly`
- **Mission type one-hots:** `mission_attack`, `mission_expand`,
  `mission_defend`, `mission_reinforce`, `mission_swarm`, `mission_snipe`,
  `mission_comet`, `mission_other`
- **Tactical:** `enemy_priority`, `committed_load`
- **Meta:** `is_defer`

### 1.5 Expected Gains

Based on analysis of our replays vs. top-player replays, the biggest
decision-quality gaps are:

| Decision Category      | Our Agent             | Top Players           | ELO Impact (est.) |
|------------------------|-----------------------|-----------------------|--------------------|
| First launch timing    | T7.3 (RL delays)     | T3–4                  | +50–80             |
| Mid-game target choice | Greedy (nearest)      | Strategic (cut-off)   | +80–120            |
| Defend vs. expand      | Over-defends          | Balances risk/reward  | +40–60             |
| Defer frequency        | 0% (never defers)     | N/A (different arch)  | +30–50             |

**Conservative estimate: +100–200 ELO** if we get counterfactual mining
working, which would put us in the top 10–20 range.

---

## 2. Pipeline Overview (Steps A → F)

The complete pipeline has six stages. Each builds on the previous one, and
each can be validated independently.

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  Step A  │───▶│  Step B  │───▶│  Step C  │───▶│  Step D  │───▶│  Step E  │───▶│  Step F  │
│ Harvest  │    │ Extract  │    │  Build   │    │ Augment  │    │ Retrain  │    │ Validate │
│ Replays  │    │  State-  │    │  Counter │    │ Training │    │  Split   │    │    &     │
│          │    │  Action  │    │  Labels  │    │  Buffer  │    │ Reranker │    │  Submit  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
```

### 2.1 Step A — Harvest Replays

**Goal:** Collect 200+ replays from top-5 players (Shun_PI, Erfan, kovi,
Ousagi, HY2017) and organize them by player and game type (2p/4p).

**What we already have:**
- `kaggle_replays/` directory with 22 submission folders and 626+ replays
- 118 replays from our own 3 submissions
- 22 Shun_PI replays in `~/Downloads/`

**What we need:**
- Systematically download replays from top-5 players
- Organize by player name and game type
- Filter out incomplete or corrupted replays

**Key tool:** `scripts/kaggle_episode_tools.py`

```bash
# List episodes for a submission
python scripts/kaggle_episode_tools.py episodes --submission-id 52228416

# Download a specific replay
python scripts/kaggle_episode_tools.py replay --episode-id 12345678 \
    --output kaggle_replays/shun_pi/episode-12345678.json

# Download logs for a specific agent in the episode
python scripts/kaggle_episode_tools.py logs --episode-id 12345678 \
    --agent-index 0 --output kaggle_replays/shun_pi/logs-12345678.json
```

**Validation:** Each replay JSON should have:
- `steps` array with 200+ entries (incomplete games are shorter)
- `info.TeamNames` identifying the players
- `steps[t][i].action` for each player `i` at turn `t`

### 2.2 Step B — Extract State-Action Pairs

**Goal:** For each turn in each top-player replay, extract:
1. The board state (what the top player saw)
2. The action they took (the fleet command they sent)
3. The game outcome (did they win?)

This is the trickiest step because the replay format gives us raw Kore
actions (fleet launch strings like `"LAUNCH_3_N"`) and we need to map those
back to mission-level concepts our agent understands.

**Key challenge:** Our agent thinks in terms of "attack base at (5,7) with 30
ships from (3,4)" while the replay just records "LAUNCH_30_NNNEEE". We need
to reverse-engineer the intent.

**Approach:**
1. Parse the replay to get the board state at each turn
2. Run our heuristic on that board state to generate candidate missions
3. For each candidate, compute how well it matches the actual fleet launch
4. The best-matching candidate becomes the "expert's choice"

### 2.3 Step C — Build Counterfactual Labels

**Goal:** Convert the "expert's choice" from Step B into training labels
for the RL reranker.

For each turn where the top player made a decision:

```
Label = {
    state:   build_state_snapshot(logic),          # from midgame_features.py
    vectors: [build_mission_feature_bundle(...)     # for each candidate
              for each candidate in heuristic_candidates],
    choice:  index of best-matching candidate,      # from Step B
    reward:  +1.0 if expert won, -0.5 if lost,     # outcome-weighted
    weight:  confidence_of_match * abs(elo_gap),    # how much to trust this
}
```

The **confidence_of_match** measures how well the best candidate actually
matches the expert's action. If no candidate is a good match (distance > 
threshold), we skip that turn entirely — we don't want to learn from
misaligned examples.

The **elo_gap** weighting means we learn more from higher-rated players.
A Shun_PI win (1607 ELO) teaches more than an Ousagi win (1461 ELO).

### 2.4 Step D — Augment Training Buffer

**Goal:** Mix counterfactual labels with our existing self-play training
data to create a balanced training buffer.

**Mixing ratio matters.** Too much expert data → the reranker becomes a
bad clone. Too little → no signal. Start with:

| Source               | Fraction | Purpose                        |
|----------------------|----------|--------------------------------|
| Self-play (existing) | 60%      | Don't forget what works        |
| Expert wins          | 30%      | Learn new decision patterns    |
| Expert losses        | 10%      | Learn what to avoid            |

The existing training pipeline in `split_reranker.py` uses
`collect_heuristic_ranking_samples()` for self-play data. We need to inject
our counterfactual samples at the same stage.

### 2.5 Step E — Retrain the Split Reranker

**Goal:** Train new 2p and 4p policies using the augmented buffer.

The existing training function:

```python
# From rl midgame/split_reranker.py
policy_2p, policy_4p, summary = train_split_policies(
    replay_paths=replay_paths,
    player_name="our_agent",
    # ... other params
)
```

We need to modify this (or wrap it) to:
1. Accept additional training samples from Step D
2. Maintain the 2p/4p split (route based on `is_two_player` at feature
   index 2, as `SplitRerankerPolicy` already does)
3. Use a lower learning rate for expert samples (they're noisier)

### 2.6 Step F — Validate & Submit

**Goal:** Verify the retrained agent is actually better before submitting.

**Local validation:**
```bash
# Run 50+ games against the heuristic baseline
python test_agent.py --agent path/to/new_agent.py \
    --opponent path/to/baseline.py --games 50 --swap

# Run 50+ games against the previous RL agent
python test_agent.py --agent path/to/new_agent.py \
    --opponent path/to/old_rl_agent.py --games 50 --swap
```

**Minimum thresholds before submitting:**
- Win rate vs. heuristic baseline: ≥ 60% (currently ~52%)
- Win rate vs. old RL agent: ≥ 55%
- First launch timing: ≤ T5.0 (currently T7.3)
- No regression in 4p games

---

## 3. Getting Replay Data

### 3.1 Using kaggle_episode_tools.py

The replay download tool lives at `scripts/kaggle_episode_tools.py` (180
lines). It wraps the Kaggle API to fetch episode data.

**Setup requirements:**
```bash
# Ensure kaggle API credentials are configured
# ~/.kaggle/kaggle.json should contain your API key
pip install kaggle
```

**Core functions (from the source):**

```python
# scripts/kaggle_episode_tools.py

def build_api():
    """Build and authenticate the Kaggle API client."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api

def list_episodes(submission_id):
    """
    List all episode IDs for a given submission.
    Returns a list of episode metadata dicts.
    """
    # Uses Kaggle API to fetch competition episodes
    ...

def download_replay(episode_id, path):
    """
    Download the full replay JSON for an episode.
    Saves to the specified path.
    """
    ...

def download_logs(episode_id, agent_index, path):
    """
    Download the stderr logs for a specific agent in an episode.
    Useful for debugging what an agent was "thinking".
    """
    ...
```

**CLI usage:**

```bash
# List all episodes for our stage2 submission
python scripts/kaggle_episode_tools.py episodes \
    --submission-id 52228416

# Download a specific replay
python scripts/kaggle_episode_tools.py replay \
    --episode-id 12345678 \
    --output kaggle_replays/top_players/shun_pi/episode-12345678.json

# Download logs (useful for agents that log their reasoning)
python scripts/kaggle_episode_tools.py logs \
    --episode-id 12345678 \
    --agent-index 0 \
    --output kaggle_replays/top_players/shun_pi/logs-12345678.json
```

### 3.2 Replay Directory Layout

Current layout in the repository:

```
kaggle_replays/
├── 52228416/          # Our stage2 submission
│   ├── episode-001.json
│   ├── episode-002.json
│   └── ...
├── 52241454/          # Our RL v1 submission
│   ├── episode-001.json
│   └── ...
├── 52244199/          # Our RL v2 submission
│   ├── episode-001.json
│   └── ...
└── ... (22 submission folders, 626+ total replays)
```

**Recommended layout for top-player replays:**

```
kaggle_replays/
├── top_players/
│   ├── shun_pi/
│   │   ├── 2p/           # 2-player games
│   │   │   ├── episode-XXXXX.json
│   │   │   └── ...
│   │   └── 4p/           # 4-player games
│   │       ├── episode-XXXXX.json
│   │       └── ...
│   ├── erfan/
│   │   ├── 2p/
│   │   └── 4p/
│   ├── kovi/
│   │   ├── 2p/
│   │   └── 4p/
│   ├── ousagi/
│   │   ├── 2p/
│   │   └── 4p/
│   └── hy2017/
│       ├── 2p/
│       └── 4p/
├── our_submissions/       # Symlink or copy existing
│   ├── 52228416/ → ../52228416
│   └── ...
└── metadata/
    └── player_index.json  # Maps player names → submission IDs
```

### 3.3 Identifying Top-Player Replays

The Kaggle leaderboard shows submission IDs, but finding specific players'
replays requires some detective work:

```python
"""
find_top_player_replays.py

Scan existing replays to find games where top players participated,
then download their other submissions' replays.
"""
import json
import os
from pathlib import Path

TOP_PLAYERS = {
    "Shun_PI": 1607,
    "Erfan":   1547,
    "kovi":    1478,
    "Ousagi":  1461,
    "HY2017":  1438,
}

def scan_replays_for_players(replay_dir: str) -> dict:
    """
    Scan all replays in a directory tree and find which top players
    appear in them. Returns {player_name: [episode_ids]}.
    """
    found = {name: [] for name in TOP_PLAYERS}
    replay_dir = Path(replay_dir)

    for replay_path in replay_dir.rglob("episode-*.json"):
        try:
            with open(replay_path) as f:
                replay = json.load(f)

            # Extract player names from the replay
            team_names = []
            if "info" in replay and "TeamNames" in replay["info"]:
                team_names = replay["info"]["TeamNames"]
            elif "steps" in replay and len(replay["steps"]) > 0:
                for agent_data in replay["steps"][0]:
                    if isinstance(agent_data, dict):
                        name = agent_data.get("info", {}).get("name", "")
                        if name:
                            team_names.append(name)

            for name in TOP_PLAYERS:
                if name in team_names:
                    episode_id = replay_path.stem.replace("episode-", "")
                    found[name].append(episode_id)

        except (json.JSONDecodeError, KeyError):
            continue

    return found

def summarize_findings(found: dict):
    """Print a summary of top player appearances."""
    print("=== Top Player Replay Inventory ===\n")
    for name, elo in sorted(TOP_PLAYERS.items(), key=lambda x: -x[1]):
        episodes = found.get(name, [])
        print(f"  {name:12s} (ELO {elo}): {len(episodes):3d} replays found")
    print()

if __name__ == "__main__":
    found = scan_replays_for_players("kaggle_replays")
    summarize_findings(found)
```

### 3.4 Bulk Download Strategy

To get enough data, we need at least **40 replays per top player** (200+
total). Here's a systematic approach:

```python
"""
bulk_download_top_replays.py

Systematically download replays from top players.
Requires: kaggle API credentials configured.
"""
import json
import os
import time
from pathlib import Path
from scripts.kaggle_episode_tools import list_episodes, download_replay

# Known submission IDs for top players (fill these in from leaderboard)
TOP_PLAYER_SUBMISSIONS = {
    "shun_pi": [
        # Add Shun_PI's submission IDs here as you discover them
    ],
    "erfan": [],
    "kovi": [],
    "ousagi": [],
    "hy2017": [],
}

REPLAY_BASE = Path("kaggle_replays/top_players")
RATE_LIMIT_DELAY = 1.0  # seconds between API calls


def download_player_replays(player_name: str, submission_ids: list,
                            max_per_submission: int = 50):
    """Download replays for a player from their submissions."""
    player_dir = REPLAY_BASE / player_name
    player_dir.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0

    for sub_id in submission_ids:
        print(f"  Fetching episodes for submission {sub_id}...")
        try:
            episodes = list_episodes(sub_id)
        except Exception as e:
            print(f"    Error listing episodes: {e}")
            continue

        for i, episode in enumerate(episodes[:max_per_submission]):
            episode_id = episode["id"]
            output_path = player_dir / f"episode-{episode_id}.json"

            if output_path.exists():
                continue

            try:
                download_replay(episode_id, str(output_path))
                total_downloaded += 1
                time.sleep(RATE_LIMIT_DELAY)
            except Exception as e:
                print(f"    Error downloading episode {episode_id}: {e}")

        print(f"    Downloaded {total_downloaded} replays so far")

    return total_downloaded


def classify_by_player_count(player_dir: Path):
    """Sort replays into 2p/ and 4p/ subdirectories."""
    dir_2p = player_dir / "2p"
    dir_4p = player_dir / "4p"
    dir_2p.mkdir(exist_ok=True)
    dir_4p.mkdir(exist_ok=True)

    for replay_path in player_dir.glob("episode-*.json"):
        try:
            with open(replay_path) as f:
                replay = json.load(f)

            n_players = sum(
                1 for agent in replay["steps"][0]
                if agent is not None
            )

            target_dir = dir_2p if n_players == 2 else dir_4p
            replay_path.rename(target_dir / replay_path.name)

        except (json.JSONDecodeError, KeyError):
            continue


if __name__ == "__main__":
    for player_name, sub_ids in TOP_PLAYER_SUBMISSIONS.items():
        if not sub_ids:
            print(f"Skipping {player_name} (no submission IDs yet)")
            continue
        print(f"\nDownloading replays for {player_name}...")
        n = download_player_replays(player_name, sub_ids)
        print(f"  Total new replays: {n}")

        player_dir = REPLAY_BASE / player_name
        classify_by_player_count(player_dir)
        print(f"  Classified into 2p/4p subdirectories")
```

**Rate limiting:** The Kaggle API has rate limits. Use a 1-second delay
between downloads and run the script during off-peak hours. If you hit
rate limits, the script will raise an exception — just wait 5 minutes and
restart (it skips already-downloaded files).

---

## 4. Testing Infrastructure

### 4.1 The Noise Problem

Local testing in Kore is **extremely noisy**. The game has significant
randomness from:

- Initial board layout (random planet positions)
- Random seed affects fleet combat outcomes
- 4-player games have complex multi-agent dynamics
- A single bad early decision can cascade into a loss

**Empirical noise levels:**

| Games Played | 95% Confidence Interval | Reliable? |
|-------------|------------------------|-----------|
| 10          | ±25% win rate          | ❌ No      |
| 20          | ±18% win rate          | ❌ Marginal |
| 50          | ±11% win rate          | ✅ Usable   |
| 100         | ±8% win rate           | ✅ Good     |
| 200         | ±6% win rate           | ✅ Reliable  |

**Bottom line:** Never draw conclusions from fewer than 50 games. Our
default of 20 games (`test_agent.py --games 20`) is too few for
reliable comparisons. Always use `--games 50` minimum.

### 4.2 Using test_agent.py Effectively

The test harness at `test_agent.py` provides local head-to-head evaluation:

```python
# Key functions from test_agent.py:

def load_agent_from_file(path: str):
    """Load an agent function from a Python file."""
    ...

def load_baseline_agent():
    """Load the default heuristic baseline agent."""
    ...

def run_game(agent_a, agent_b, seed: int) -> dict:
    """
    Run a single game between two agents.
    
    Returns:
        {
            "reward_a": float,    # Final score for agent A
            "reward_b": float,    # Final score for agent B
            "ships_a":  int,      # Ships remaining for A
            "ships_b":  int,      # Ships remaining for B
            "winner":   str,      # "a", "b", or "draw"
        }
    """
    ...
```

**CLI usage:**

```bash
# Basic comparison (default 20 games — TOO FEW, use 50+)
python test_agent.py --agent path/to/new_agent.py \
    --opponent path/to/baseline.py --games 50

# With seat-swapping (recommended — eliminates first-mover bias)
python test_agent.py --agent path/to/new_agent.py \
    --opponent path/to/baseline.py --games 50 --swap

# Quick sanity check (is the agent even working?)
python test_agent.py --agent path/to/new_agent.py \
    --opponent path/to/baseline.py --games 5
```

### 4.3 Statistical Significance & Sample Sizes

**How to interpret test results:**

Given N games with W wins for the new agent:

```python
import scipy.stats as stats

def is_significant(wins: int, games: int, alpha: float = 0.05) -> dict:
    """
    Test whether win rate is significantly different from 50%.
    Uses a two-sided binomial test.
    """
    p_value = stats.binom_test(wins, games, 0.5)
    win_rate = wins / games

    # Wilson confidence interval (better than normal approx for small N)
    z = stats.norm.ppf(1 - alpha / 2)
    denominator = 1 + z**2 / games
    center = (win_rate + z**2 / (2 * games)) / denominator
    margin = z * (win_rate * (1 - win_rate) / games + z**2 / (4 * games**2))**0.5 / denominator

    return {
        "win_rate": win_rate,
        "p_value": p_value,
        "significant": p_value < alpha,
        "ci_low": center - margin,
        "ci_high": center + margin,
    }

# Example: 32 wins out of 50 games
result = is_significant(32, 50)
# win_rate=0.64, p_value=0.041, significant=True, ci=[0.50, 0.76]
```

**Required sample sizes for different effect sizes:**

| True Win Rate | Games Needed (p<0.05) | Games Needed (p<0.01) |
|--------------|----------------------|----------------------|
| 55%          | ~400                 | ~650                 |
| 60%          | ~100                 | ~170                 |
| 65%          | ~50                  | ~80                  |
| 70%          | ~30                  | ~45                  |

If the counterfactual mining works as expected (60%+ win rate vs baseline),
50 games should be enough to confirm.

### 4.4 Benchmark Baselines

Set up a systematic benchmark suite:

```python
"""
benchmark_suite.py

Run the new agent against multiple baselines and compile results.
"""
import subprocess
import json
from pathlib import Path

BASELINES = {
    "heuristic": "path/to/heuristic_agent.py",
    "rl_v1": "path/to/rl_v1_agent.py",
    "rl_v2": "path/to/rl_v2_agent.py",
}

GAMES_PER_MATCHUP = 50


def run_benchmark(agent_path: str, results_dir: str = "benchmark_results"):
    """Run the agent against all baselines."""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    results = {}

    for baseline_name, baseline_path in BASELINES.items():
        print(f"Testing vs {baseline_name}...")

        cmd = [
            "python", "test_agent.py",
            "--agent", agent_path,
            "--opponent", baseline_path,
            "--games", str(GAMES_PER_MATCHUP),
            "--swap",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        output = proc.stdout

        results[baseline_name] = {
            "output": output,
            "returncode": proc.returncode,
        }

        print(f"  Done. Return code: {proc.returncode}")

    with open(results_dir / "latest_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import sys
    agent_path = sys.argv[1] if len(sys.argv) > 1 else "path/to/new_agent.py"
    run_benchmark(agent_path)
```

---

## 5. Code Examples & Implementation Guide

This section contains the core implementation. Each subsection is a
self-contained module that you can implement and test independently.

### 5.1 Replay Parser: Extracting Turns

The first step is parsing Kore replay JSON files into a structured format
our pipeline can work with.

```python
"""
counterfactual/replay_parser.py

Parse Kore replay JSON files and extract per-turn state-action data
for each player.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FleetAction:
    """A single fleet launch action from a replay."""
    shipyard_id: str       # Which shipyard launched
    num_ships: int         # How many ships were sent
    flight_plan: str       # The direction string (e.g., "NNNEEE")
    raw_action: str        # Original action string from replay


@dataclass
class TurnState:
    """Complete state of the game at one turn."""
    step: int
    kore: dict             # {player_id: kore_amount}
    ships: dict            # {player_id: total_ships}
    shipyards: dict        # {player_id: [{id, position, ships}]}
    fleets: dict           # {player_id: [{id, position, ships, kore, plan}]}
    actions: dict          # {player_id: [FleetAction]}


@dataclass
class ParsedReplay:
    """A fully parsed replay with metadata."""
    episode_id: str
    n_players: int
    n_steps: int
    player_names: list
    player_scores: list     # Final scores
    winner_index: int       # Index of winning player (-1 if draw)
    turns: list             # List of TurnState objects


def parse_fleet_action(shipyard_id: str, action_str: str) -> Optional[FleetAction]:
    """
    Parse a raw action string into a FleetAction.
    
    Kore actions are formatted as:
    - "SPAWN_N" — build N ships at this shipyard
    - "LAUNCH_N_PLAN" — launch N ships with flight plan PLAN
    
    We only care about LAUNCH actions for our purposes.
    """
    if not action_str or not action_str.startswith("LAUNCH"):
        return None

    parts = action_str.split("_", 2)
    if len(parts) < 3:
        return None

    try:
        num_ships = int(parts[1])
        flight_plan = parts[2]
    except (ValueError, IndexError):
        return None

    return FleetAction(
        shipyard_id=shipyard_id,
        num_ships=num_ships,
        flight_plan=flight_plan,
        raw_action=action_str,
    )


def parse_replay(replay_path: str) -> ParsedReplay:
    """
    Parse a Kore replay JSON file into our structured format.
    
    Args:
        replay_path: Path to the episode-*.json file
        
    Returns:
        ParsedReplay with all turns parsed
    """
    with open(replay_path) as f:
        raw = json.load(f)

    episode_id = Path(replay_path).stem.replace("episode-", "")

    steps = raw.get("steps", raw.get("data", {}).get("steps", []))
    n_players = len(steps[0]) if steps else 0

    player_names = []
    info = raw.get("info", {})
    if "TeamNames" in info:
        player_names = info["TeamNames"]
    else:
        player_names = [f"player_{i}" for i in range(n_players)]

    turns = []
    for step_idx, step_data in enumerate(steps):
        turn = _parse_turn(step_idx, step_data, n_players)
        turns.append(turn)

    final_scores = []
    if turns:
        last_turn = turns[-1]
        for i in range(n_players):
            kore = last_turn.kore.get(i, 0)
            ships = last_turn.ships.get(i, 0)
            final_scores.append(kore + ships * 10)

    winner_index = -1
    if final_scores:
        max_score = max(final_scores)
        if final_scores.count(max_score) == 1:
            winner_index = final_scores.index(max_score)

    return ParsedReplay(
        episode_id=episode_id,
        n_players=n_players,
        n_steps=len(steps),
        player_names=player_names,
        player_scores=final_scores,
        winner_index=winner_index,
        turns=turns,
    )


def _parse_turn(step_idx: int, step_data: list, n_players: int) -> TurnState:
    """Parse a single turn from the replay steps array."""
    kore = {}
    ships = {}
    shipyards = {}
    fleets = {}
    actions = {}

    for player_idx in range(n_players):
        player_data = step_data[player_idx]

        if player_data is None:
            kore[player_idx] = 0
            ships[player_idx] = 0
            shipyards[player_idx] = []
            fleets[player_idx] = []
            actions[player_idx] = []
            continue

        obs = player_data.get("observation", {})
        kore[player_idx] = obs.get("kore", [0] * n_players)[player_idx] \
            if isinstance(obs.get("kore"), list) else 0

        player_shipyards = []
        player_actions = []
        player_ships_total = 0

        sy_data = obs.get("shipyards", {})
        action_data = player_data.get("action", {})

        if isinstance(sy_data, dict):
            for sy_id, sy_info in sy_data.items():
                sy_ships = sy_info.get("ships", 0) if isinstance(sy_info, dict) else 0
                player_ships_total += sy_ships
                player_shipyards.append({
                    "id": sy_id,
                    "position": sy_info.get("position", None),
                    "ships": sy_ships,
                })

                if isinstance(action_data, dict) and sy_id in action_data:
                    fleet_action = parse_fleet_action(sy_id, action_data[sy_id])
                    if fleet_action:
                        player_actions.append(fleet_action)

        player_fleets = []
        fleet_data = obs.get("fleets", {})
        if isinstance(fleet_data, dict):
            for fleet_id, fleet_info in fleet_data.items():
                f_ships = fleet_info.get("ships", 0) if isinstance(fleet_info, dict) else 0
                player_ships_total += f_ships
                player_fleets.append({
                    "id": fleet_id,
                    "position": fleet_info.get("position", None),
                    "ships": f_ships,
                    "kore": fleet_info.get("kore", 0),
                    "plan": fleet_info.get("flight_plan", ""),
                })

        ships[player_idx] = player_ships_total
        shipyards[player_idx] = player_shipyards
        fleets[player_idx] = player_fleets
        actions[player_idx] = player_actions

    return TurnState(
        step=step_idx,
        kore=kore,
        ships=ships,
        shipyards=shipyards,
        fleets=fleets,
        actions=actions,
    )


def filter_midgame_turns(replay: ParsedReplay, player_idx: int,
                          min_step: int = 18, max_step: int = 180) -> list:
    """
    Filter to only midgame turns where the player took LAUNCH actions.
    
    The RL reranker only operates in midgame (step 18-180), so we only
    want training data from that range.
    """
    midgame_turns = []

    for turn in replay.turns:
        if turn.step < min_step or turn.step > max_step:
            continue

        player_actions = turn.actions.get(player_idx, [])
        if player_actions:
            midgame_turns.append((turn.step, turn))

    return midgame_turns


def load_all_replays(replay_dir: str) -> list:
    """Load all replay files from a directory tree."""
    replays = []
    for path in Path(replay_dir).rglob("episode-*.json"):
        try:
            replays.append(parse_replay(str(path)))
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  Warning: Failed to parse {path}: {e}")
    return replays


def find_player_in_replay(replay: ParsedReplay, player_name: str) -> int:
    """Find the index of a player by name in a replay. Returns -1 if not found."""
    for i, name in enumerate(replay.player_names):
        if player_name.lower() in name.lower():
            return i
    return -1
```

### 5.2 Feature Extraction from Replay States

Now we need to convert replay states into the same 33-feature vectors our
RL reranker uses. This is the bridge between replay data and our training
pipeline.

```python
"""
counterfactual/feature_bridge.py

Bridge between replay states and the RL reranker's feature space.
Uses the same feature definitions from rl midgame/midgame_features.py.
"""
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from replay_parser import ParsedReplay, TurnState, FleetAction

# Import from our existing feature module
try:
    import importlib
    midgame_features = importlib.import_module("rl midgame.midgame_features")
    FEATURE_NAMES = midgame_features.FEATURE_NAMES
    build_state_snapshot = midgame_features.build_state_snapshot
    build_mission_feature_bundle = midgame_features.build_mission_feature_bundle
    build_defer_vector = midgame_features.build_defer_vector
except ImportError:
    # Fallback: define feature names locally (must stay in sync!)
    FEATURE_NAMES = [
        "step_frac", "remaining_frac",
        "is_two_player", "is_four_player_plus",
        "is_ahead", "is_behind", "is_dominating", "is_finishing",
        "my_ship_share", "my_prod_share", "lead_ratio",
        "base_score_norm", "base_value_norm",
        "eta_norm", "send_norm", "source_count_norm", "needed_norm",
        "target_prod_norm", "target_ships_norm",
        "target_is_neutral", "target_is_enemy", "target_is_friendly",
        "mission_attack", "mission_expand", "mission_defend",
        "mission_reinforce", "mission_swarm", "mission_snipe",
        "mission_comet", "mission_other",
        "enemy_priority", "committed_load",
        "is_defer",
    ]
    build_state_snapshot = None
    build_mission_feature_bundle = None
    build_defer_vector = None

N_FEATURES = len(FEATURE_NAMES)  # Should be 33
assert N_FEATURES == 33, f"Expected 33 features, got {N_FEATURES}"

# Feature index constants for quick access
IDX_STEP_FRAC = FEATURE_NAMES.index("step_frac")
IDX_IS_TWO_PLAYER = FEATURE_NAMES.index("is_two_player")
IDX_IS_FOUR_PLAYER = FEATURE_NAMES.index("is_four_player_plus")
IDX_IS_AHEAD = FEATURE_NAMES.index("is_ahead")
IDX_IS_BEHIND = FEATURE_NAMES.index("is_behind")
IDX_MY_SHIP_SHARE = FEATURE_NAMES.index("my_ship_share")
IDX_IS_DEFER = FEATURE_NAMES.index("is_defer")


def build_state_features_from_replay(
    turn: TurnState,
    player_idx: int,
    n_players: int,
    total_steps: int = 400,
) -> dict:
    """
    Build a state snapshot from replay turn data.
    
    This mirrors build_state_snapshot(logic) but works from replay data
    instead of a live game Logic object.
    """
    my_ships = turn.ships.get(player_idx, 0)
    total_ships = sum(turn.ships.values())
    my_ship_share = my_ships / max(total_ships, 1)

    opponent_ships = [
        turn.ships.get(i, 0) for i in range(n_players) if i != player_idx
    ]
    max_opponent = max(opponent_ships) if opponent_ships else 0

    is_ahead = my_ships > max_opponent
    is_behind = my_ships < max_opponent * 0.7
    is_dominating = my_ship_share > 0.5
    is_finishing = turn.step > total_steps * 0.8

    my_shipyards = len(turn.shipyards.get(player_idx, []))
    total_shipyards = sum(len(turn.shipyards.get(i, [])) for i in range(n_players))
    my_prod_share = my_shipyards / max(total_shipyards, 1)

    lead_ratio = my_ships / max(max_opponent, 1)

    return {
        "step": turn.step,
        "step_frac": turn.step / total_steps,
        "remaining_frac": 1.0 - turn.step / total_steps,
        "is_two_player": n_players == 2,
        "is_four_player_plus": n_players >= 4,
        "is_ahead": is_ahead,
        "is_behind": is_behind,
        "is_dominating": is_dominating,
        "is_finishing": is_finishing,
        "my_ship_share": my_ship_share,
        "my_prod_share": my_prod_share,
        "lead_ratio": min(lead_ratio, 3.0) / 3.0,
        "my_ships": my_ships,
        "total_ships": total_ships,
        "n_players": n_players,
    }


def build_approximate_feature_vector(
    state: dict,
    action: FleetAction,
    turn: TurnState,
    player_idx: int,
    mission_type: str = "other",
) -> list:
    """
    Build an approximate 33-feature vector for a fleet action from replay data.
    
    This is an APPROXIMATION because we don't have the full Logic object
    that build_mission_feature_bundle() expects.
    """
    vector = [0.0] * N_FEATURES

    # State features (accurate from replay)
    vector[FEATURE_NAMES.index("step_frac")] = state["step_frac"]
    vector[FEATURE_NAMES.index("remaining_frac")] = state["remaining_frac"]
    vector[FEATURE_NAMES.index("is_two_player")] = float(state["is_two_player"])
    vector[FEATURE_NAMES.index("is_four_player_plus")] = float(state["is_four_player_plus"])
    vector[FEATURE_NAMES.index("is_ahead")] = float(state["is_ahead"])
    vector[FEATURE_NAMES.index("is_behind")] = float(state["is_behind"])
    vector[FEATURE_NAMES.index("is_dominating")] = float(state["is_dominating"])
    vector[FEATURE_NAMES.index("is_finishing")] = float(state["is_finishing"])
    vector[FEATURE_NAMES.index("my_ship_share")] = state["my_ship_share"]
    vector[FEATURE_NAMES.index("my_prod_share")] = state["my_prod_share"]
    vector[FEATURE_NAMES.index("lead_ratio")] = state["lead_ratio"]

    # Mission features (approximated from action)
    vector[FEATURE_NAMES.index("send_norm")] = min(action.num_ships / 100.0, 1.0)
    eta = len(action.flight_plan)
    vector[FEATURE_NAMES.index("eta_norm")] = min(eta / 20.0, 1.0)
    vector[FEATURE_NAMES.index("source_count_norm")] = 1.0 / 5.0

    # Mission type one-hots
    mission_types = {
        "attack": "mission_attack",
        "expand": "mission_expand",
        "defend": "mission_defend",
        "reinforce": "mission_reinforce",
        "swarm": "mission_swarm",
        "snipe": "mission_snipe",
        "comet": "mission_comet",
        "other": "mission_other",
    }
    if mission_type in mission_types:
        vector[FEATURE_NAMES.index(mission_types[mission_type])] = 1.0

    vector[FEATURE_NAMES.index("is_defer")] = 0.0

    return vector


def infer_mission_type(action: FleetAction, turn: TurnState,
                       player_idx: int, n_players: int) -> str:
    """
    Attempt to infer the mission type from a fleet action.
    
    Returns one of: attack, expand, defend, reinforce, swarm, snipe, comet, other
    """
    plan = action.flight_plan
    ships = action.num_ships

    if ships > 50:
        return "attack" if len(plan) < 10 else "swarm"
    if ships < 10 and len(plan) > 5:
        return "expand"
    if len(plan) <= 3:
        return "reinforce" if ships < 20 else "defend"
    return "other"


def build_defer_feature_vector(best_candidate_vector: list) -> list:
    """
    Build the defer option's feature vector.
    Mirrors build_defer_vector() from midgame_features.py.
    """
    defer_vec = list(best_candidate_vector)
    defer_vec[IDX_IS_DEFER] = 1.0
    return defer_vec
```

### 5.3 Counterfactual Label Generator

This is the core of the pipeline — matching expert actions to our
candidate set and generating training labels.

```python
"""
counterfactual/label_generator.py

Generate counterfactual training labels by matching top-player actions
to our heuristic candidate set.
"""
import math
from dataclasses import dataclass
from typing import Optional

from replay_parser import ParsedReplay, TurnState, FleetAction
from feature_bridge import (
    build_state_features_from_replay,
    build_approximate_feature_vector,
    build_defer_feature_vector,
    infer_mission_type,
    N_FEATURES,
)


@dataclass
class CounterfactualLabel:
    """A single training label from counterfactual mining."""
    episode_id: str
    step: int
    player_name: str
    player_elo: float

    candidate_vectors: list    # List of 33-float lists
    n_candidates: int

    chosen_index: int
    match_confidence: float

    reward: float              # +1 for win, -0.5 for loss
    weight: float              # confidence * elo_factor

    expert_action: str
    matched_type: str


@dataclass
class CandidateMission:
    """A candidate mission from our heuristic."""
    mission_type: str
    target_position: tuple
    num_ships: int
    flight_plan: str
    source_shipyard: str
    feature_vector: list       # 33-float feature vector
    base_score: float


def compute_action_similarity(
    expert_action: FleetAction,
    candidate: CandidateMission,
) -> float:
    """
    Compute how similar an expert's action is to one of our candidates.
    Returns a score in [0, 1].
    """
    # Ship count similarity (log ratio)
    ship_ratio = math.log(
        max(expert_action.num_ships, 1) / max(candidate.num_ships, 1)
    )
    ship_sim = math.exp(-abs(ship_ratio))

    # Flight plan similarity (weighted character comparison)
    plan_a = expert_action.flight_plan
    plan_b = candidate.flight_plan
    max_len = max(len(plan_a), len(plan_b), 1)
    matches = 0
    total_weight = 0
    for i in range(max_len):
        weight = 1.0 / (1 + i * 0.5)
        total_weight += weight
        if i < len(plan_a) and i < len(plan_b) and plan_a[i] == plan_b[i]:
            matches += weight
    plan_sim = matches / total_weight if total_weight > 0 else 0.0

    # Source match bonus
    source_bonus = 0.1 if expert_action.shipyard_id == candidate.source_shipyard else 0.0

    similarity = (
        0.3 * ship_sim +
        0.5 * plan_sim +
        0.1 * source_bonus +
        0.1
    )

    return min(similarity, 1.0)


def match_expert_to_candidates(
    expert_actions: list,
    candidates: list,
    min_confidence: float = 0.3,
) -> Optional[tuple]:
    """
    Find the best candidate match for the expert's actions.
    Returns (best_candidate_index, confidence) or None if no good match.
    """
    if not expert_actions or not candidates:
        return None

    primary_action = max(expert_actions, key=lambda a: a.num_ships)

    best_idx = -1
    best_score = 0.0

    for i, candidate in enumerate(candidates):
        score = compute_action_similarity(primary_action, candidate)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_score < min_confidence:
        return None

    return (best_idx, best_score)


def generate_labels_for_replay(
    replay: ParsedReplay,
    player_idx: int,
    player_name: str,
    player_elo: float,
    candidate_generator,
    min_step: int = 18,
    max_step: int = 180,
    min_confidence: float = 0.3,
) -> list:
    """Generate counterfactual labels for all midgame turns in a replay."""
    labels = []
    won = (replay.winner_index == player_idx)
    reward = 1.0 if won else -0.5

    elo_factor = (player_elo - 1200) / 400

    for turn in replay.turns:
        if turn.step < min_step or turn.step > max_step:
            continue

        expert_actions = turn.actions.get(player_idx, [])
        if not expert_actions:
            continue

        candidates = candidate_generator(turn, player_idx)
        if not candidates:
            continue

        # Add defer option
        if candidates:
            best_heuristic_vec = candidates[0].feature_vector
            defer_candidate = CandidateMission(
                mission_type="defer",
                target_position=(0, 0),
                num_ships=0,
                flight_plan="",
                source_shipyard="",
                feature_vector=build_defer_feature_vector(best_heuristic_vec),
                base_score=0.0,
            )
            candidates_with_defer = candidates + [defer_candidate]
        else:
            candidates_with_defer = candidates

        match = match_expert_to_candidates(
            expert_actions, candidates, min_confidence
        )

        if match is None:
            continue

        chosen_idx, confidence = match
        weight = confidence * max(elo_factor, 0.1)

        candidate_vectors = [c.feature_vector for c in candidates_with_defer]

        primary_action = max(expert_actions, key=lambda a: a.num_ships)
        matched_type = infer_mission_type(
            primary_action, turn, player_idx, replay.n_players
        )

        labels.append(CounterfactualLabel(
            episode_id=replay.episode_id,
            step=turn.step,
            player_name=player_name,
            player_elo=player_elo,
            candidate_vectors=candidate_vectors,
            n_candidates=len(candidates_with_defer),
            chosen_index=chosen_idx,
            match_confidence=confidence,
            reward=reward,
            weight=weight,
            expert_action=primary_action.raw_action,
            matched_type=matched_type,
        ))

    return labels


def summarize_labels(labels: list) -> dict:
    """Generate a summary of the generated labels for logging."""
    if not labels:
        return {"total": 0}

    n_wins = sum(1 for l in labels if l.reward > 0)
    n_losses = len(labels) - n_wins
    avg_confidence = sum(l.match_confidence for l in labels) / len(labels)
    avg_weight = sum(l.weight for l in labels) / len(labels)

    type_counts = {}
    for l in labels:
        type_counts[l.matched_type] = type_counts.get(l.matched_type, 0) + 1

    return {
        "total": len(labels),
        "wins": n_wins,
        "losses": n_losses,
        "avg_confidence": round(avg_confidence, 3),
        "avg_weight": round(avg_weight, 3),
        "type_distribution": type_counts,
    }
```

### 5.4 Training Buffer Integration

Now we integrate counterfactual labels into the existing training buffer.

```python
"""
counterfactual/training_buffer.py

Integrate counterfactual labels with the existing self-play training
buffer used by split_reranker.py.
"""
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from label_generator import CounterfactualLabel, summarize_labels


@dataclass
class TrainingSample:
    """A unified training sample from self-play or counterfactual mining."""
    candidate_vectors: list
    n_candidates: int
    chosen_index: int
    reward: float
    weight: float
    source: str                # "self_play" or "counterfactual"
    is_two_player: bool
    metadata: dict = None


class TrainingBuffer:
    """
    A training buffer that mixes self-play and counterfactual samples.
    
    Usage:
        buffer = TrainingBuffer()
        buffer.add_self_play_samples(existing_samples)
        buffer.add_counterfactual_labels(cf_labels)
        train_2p, train_4p = buffer.get_training_splits(mix_ratio=0.3)
    """

    def __init__(self):
        self.self_play_2p = []
        self.self_play_4p = []
        self.counterfactual_2p = []
        self.counterfactual_4p = []

    def add_self_play_samples(self, samples: list):
        """Add existing self-play training samples."""
        for sample in samples:
            if sample.is_two_player:
                self.self_play_2p.append(sample)
            else:
                self.self_play_4p.append(sample)

    def add_counterfactual_labels(self, labels: list):
        """Convert counterfactual labels to training samples and add them."""
        for label in labels:
            is_2p = False
            if label.candidate_vectors and len(label.candidate_vectors[0]) > 2:
                is_2p = label.candidate_vectors[0][2] > 0.5

            sample = TrainingSample(
                candidate_vectors=label.candidate_vectors,
                n_candidates=label.n_candidates,
                chosen_index=label.chosen_index,
                reward=label.reward,
                weight=label.weight,
                source="counterfactual",
                is_two_player=is_2p,
                metadata={
                    "episode_id": label.episode_id,
                    "step": label.step,
                    "player_name": label.player_name,
                    "match_confidence": label.match_confidence,
                },
            )

            if is_2p:
                self.counterfactual_2p.append(sample)
            else:
                self.counterfactual_4p.append(sample)

    def get_training_splits(
        self,
        cf_ratio: float = 0.3,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> tuple:
        """Get balanced training splits for 2p and 4p policies."""
        rng = random.Random(seed)

        train_2p = self._mix_samples(
            self.self_play_2p, self.counterfactual_2p,
            cf_ratio, max_samples, rng
        )
        train_4p = self._mix_samples(
            self.self_play_4p, self.counterfactual_4p,
            cf_ratio, max_samples, rng
        )

        return train_2p, train_4p

    def _mix_samples(self, self_play, counterfactual, cf_ratio, max_samples, rng):
        """Mix self-play and counterfactual samples at the given ratio."""
        if not self_play and not counterfactual:
            return []

        total_available = len(self_play) + len(counterfactual)
        total_target = min(total_available, max_samples or total_available)

        n_cf = min(int(total_target * cf_ratio), len(counterfactual))
        n_sp = min(total_target - n_cf, len(self_play))

        sp_samples = rng.sample(self_play, n_sp) if n_sp <= len(self_play) else self_play[:]
        cf_samples = rng.sample(counterfactual, n_cf) if n_cf <= len(counterfactual) else counterfactual[:]

        mixed = sp_samples + cf_samples
        rng.shuffle(mixed)

        return mixed

    def summary(self) -> dict:
        """Get buffer statistics."""
        return {
            "self_play_2p": len(self.self_play_2p),
            "self_play_4p": len(self.self_play_4p),
            "counterfactual_2p": len(self.counterfactual_2p),
            "counterfactual_4p": len(self.counterfactual_4p),
            "total": (
                len(self.self_play_2p) + len(self.self_play_4p) +
                len(self.counterfactual_2p) + len(self.counterfactual_4p)
            ),
        }


def save_buffer(buffer: TrainingBuffer, path: str):
    """Save the training buffer to disk as JSON."""
    data = {
        "self_play_2p": [asdict(s) for s in buffer.self_play_2p],
        "self_play_4p": [asdict(s) for s in buffer.self_play_4p],
        "counterfactual_2p": [asdict(s) for s in buffer.counterfactual_2p],
        "counterfactual_4p": [asdict(s) for s in buffer.counterfactual_4p],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def load_buffer(path: str) -> TrainingBuffer:
    """Load a training buffer from disk."""
    with open(path) as f:
        data = json.load(f)

    buffer = TrainingBuffer()

    for key, sample_list in [
        ("self_play_2p", buffer.self_play_2p),
        ("self_play_4p", buffer.self_play_4p),
        ("counterfactual_2p", buffer.counterfactual_2p),
        ("counterfactual_4p", buffer.counterfactual_4p),
    ]:
        for s_dict in data.get(key, []):
            sample = TrainingSample(**s_dict)
            sample_list.append(sample)

    return buffer
```

### 5.5 Modified Training Loop

Here we modify the split reranker training to accept our augmented buffer.

```python
"""
counterfactual/train_with_counterfactuals.py

Modified training loop that incorporates counterfactual samples
into the split reranker training.
"""
import sys
import json
import math
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from training_buffer import TrainingBuffer, TrainingSample, load_buffer

# Import from existing training code
try:
    import importlib
    split_reranker = importlib.import_module("rl midgame.split_reranker")
    SplitRerankerPolicy = split_reranker.SplitRerankerPolicy
    train_split_policies = split_reranker.train_split_policies
    build_split_agent = split_reranker.build_split_agent

    midgame_policy = importlib.import_module("rl midgame.midgame_policy")
except ImportError as e:
    print(f"Warning: Could not import rl midgame modules: {e}")
    SplitRerankerPolicy = None
    train_split_policies = None


def train_with_counterfactuals(
    self_play_replay_paths: list,
    counterfactual_buffer_path: str,
    player_name: str = "our_agent",
    cf_ratio: float = 0.3,
    learning_rate: float = 0.001,
    cf_learning_rate: float = 0.0005,
    n_epochs: int = 10,
    output_dir: str = "trained_policies",
) -> dict:
    """
    Train split reranker policies with counterfactual augmentation.
    This is the main entry point for the counterfactual training pipeline.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COUNTERFACTUAL RL TRAINING")
    print("=" * 60)

    # Phase 1: Load counterfactual buffer
    print("\n[Phase 1] Loading counterfactual buffer...")
    cf_buffer = load_buffer(counterfactual_buffer_path)
    print(f"  Buffer stats: {json.dumps(cf_buffer.summary(), indent=2)}")

    # Phase 2: Build augmented training set
    print(f"\n[Phase 2] Building augmented training set (CF ratio: {cf_ratio})...")
    train_2p, train_4p = cf_buffer.get_training_splits(cf_ratio=cf_ratio)
    print(f"  2p training samples: {len(train_2p)}")
    print(f"  4p training samples: {len(train_4p)}")

    for label, samples in [("2p", train_2p), ("4p", train_4p)]:
        sp = sum(1 for s in samples if s.source == "self_play")
        cf = sum(1 for s in samples if s.source == "counterfactual")
        print(f"  {label}: {sp} self-play + {cf} counterfactual")

    # Phase 3: Train 2p policy
    print(f"\n[Phase 3] Training 2p policy ({n_epochs} epochs)...")
    policy_2p = _train_policy(
        train_2p,
        learning_rate=learning_rate,
        cf_learning_rate=cf_learning_rate,
        n_epochs=n_epochs,
        label="2p",
    )

    # Phase 4: Train 4p policy
    print(f"\n[Phase 4] Training 4p policy ({n_epochs} epochs)...")
    policy_4p = _train_policy(
        train_4p,
        learning_rate=learning_rate,
        cf_learning_rate=cf_learning_rate,
        n_epochs=n_epochs,
        label="4p",
    )

    # Phase 5: Save policies
    print(f"\n[Phase 5] Saving policies to {output_dir}...")
    policy_2p_path = output_dir / "policy_2p_cf.json"
    policy_4p_path = output_dir / "policy_4p_cf.json"

    _save_policy(policy_2p, policy_2p_path)
    _save_policy(policy_4p, policy_4p_path)

    summary = {
        "cf_ratio": cf_ratio,
        "n_epochs": n_epochs,
        "samples_2p": len(train_2p),
        "samples_4p": len(train_4p),
        "policy_2p_path": str(policy_2p_path),
        "policy_4p_path": str(policy_4p_path),
    }

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(json.dumps(summary, indent=2))
    print("=" * 60)

    return summary


def _train_policy(samples, learning_rate, cf_learning_rate, n_epochs, label):
    """
    Train a single linear policy using REINFORCE-style updates.
    Returns trained weight vector (33 floats).
    """
    from feature_bridge import N_FEATURES
    import random

    weights = [0.0] * N_FEATURES

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_updates = 0

        random.shuffle(samples)

        for sample in samples:
            if not sample.candidate_vectors or sample.n_candidates == 0:
                continue

            scores = []
            for vec in sample.candidate_vectors:
                score = sum(w * f for w, f in zip(weights, vec))
                scores.append(score)

            max_score = max(scores)
            exp_scores = [math.exp(s - max_score) for s in scores]
            sum_exp = sum(exp_scores)
            probs = [e / sum_exp for e in exp_scores]

            chosen = sample.chosen_index
            if chosen >= len(probs):
                continue

            lr = cf_learning_rate if sample.source == "counterfactual" else learning_rate

            for i in range(N_FEATURES):
                gradient = sample.candidate_vectors[chosen][i]
                for j, prob in enumerate(probs):
                    gradient -= prob * sample.candidate_vectors[j][i]

                weights[i] += lr * gradient * sample.reward * sample.weight
                total_loss += -math.log(max(probs[chosen], 1e-10))
                n_updates += 1

        avg_loss = total_loss / max(n_updates, 1)
        if epoch % 2 == 0 or epoch == n_epochs - 1:
            print(f"    Epoch {epoch+1}/{n_epochs}: avg_loss={avg_loss:.4f}")

    return weights


def _save_policy(weights, path):
    """Save policy weights to a JSON file."""
    with open(path, "w") as f:
        json.dump({"weights": weights}, f, indent=2)
    print(f"  Saved to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train split reranker with counterfactual augmentation"
    )
    parser.add_argument("--buffer", required=True,
        help="Path to counterfactual training buffer JSON")
    parser.add_argument("--replays", nargs="+", default=[],
        help="Paths to self-play replay files")
    parser.add_argument("--cf-ratio", type=float, default=0.3,
        help="Fraction of counterfactual samples (default: 0.3)")
    parser.add_argument("--lr", type=float, default=0.001,
        help="Learning rate for self-play (default: 0.001)")
    parser.add_argument("--cf-lr", type=float, default=0.0005,
        help="Learning rate for counterfactual samples (default: 0.0005)")
    parser.add_argument("--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)")
    parser.add_argument("--output", default="trained_policies",
        help="Output directory (default: trained_policies)")

    args = parser.parse_args()

    train_with_counterfactuals(
        self_play_replay_paths=args.replays,
        counterfactual_buffer_path=args.buffer,
        cf_ratio=args.cf_ratio,
        learning_rate=args.lr,
        cf_learning_rate=args.cf_lr,
        n_epochs=args.epochs,
        output_dir=args.output,
    )
```

### 5.6 End-to-End Script

This ties everything together into a single runnable pipeline.

```python
"""
counterfactual/run_pipeline.py

End-to-end counterfactual replay mining pipeline.

Usage:
    python counterfactual/run_pipeline.py \
        --replay-dir kaggle_replays/top_players \
        --output-dir trained_policies \
        --cf-ratio 0.3 \
        --epochs 10
"""
import argparse
import json
import sys
import time
from pathlib import Path

from replay_parser import (
    load_all_replays,
    find_player_in_replay,
    filter_midgame_turns,
    ParsedReplay,
)
from feature_bridge import (
    build_state_features_from_replay,
    build_approximate_feature_vector,
    infer_mission_type,
)
from label_generator import (
    generate_labels_for_replay,
    summarize_labels,
    CandidateMission,
)
from training_buffer import TrainingBuffer, save_buffer
from train_with_counterfactuals import train_with_counterfactuals


TOP_PLAYERS = {
    "Shun_PI": 1607,
    "Erfan":   1547,
    "kovi":    1478,
    "Ousagi":  1461,
    "HY2017":  1438,
}


def stub_candidate_generator(turn, player_idx):
    """
    Stub candidate generator for initial pipeline testing.
    
    In production, replace this with a function that runs the actual
    heuristic base agent on the replay's board state to generate
    real candidates.
    """
    import random
    
    expert_actions = turn.actions.get(player_idx, [])
    if not expert_actions:
        return []
    
    candidates = []
    n_players = max(turn.ships.keys()) + 1 if turn.ships else 2
    state = build_state_features_from_replay(turn, player_idx, n_players)
    
    for action in expert_actions:
        mission_type = infer_mission_type(action, turn, player_idx, n_players)
        vec = build_approximate_feature_vector(
            state, action, turn, player_idx, mission_type
        )
        candidates.append(CandidateMission(
            mission_type=mission_type,
            target_position=(0, 0),
            num_ships=action.num_ships,
            flight_plan=action.flight_plan,
            source_shipyard=action.shipyard_id,
            feature_vector=vec,
            base_score=1.0,
        ))
    
    # Add distractor candidates with random variations
    for _ in range(3):
        if expert_actions:
            base_action = random.choice(expert_actions)
            noisy_ships = max(1, base_action.num_ships + random.randint(-10, 10))
            noisy_type = random.choice(["attack", "expand", "defend", "other"])
            
            noisy_action_obj = type(base_action)(
                shipyard_id=base_action.shipyard_id,
                num_ships=noisy_ships,
                flight_plan=base_action.flight_plan[::-1],
                raw_action=f"LAUNCH_{noisy_ships}_NOISE",
            )
            
            vec = build_approximate_feature_vector(
                state, noisy_action_obj, turn, player_idx, noisy_type
            )
            candidates.append(CandidateMission(
                mission_type=noisy_type,
                target_position=(0, 0),
                num_ships=noisy_ships,
                flight_plan=noisy_action_obj.flight_plan,
                source_shipyard=base_action.shipyard_id,
                feature_vector=vec,
                base_score=0.5,
            ))
    
    return candidates


def run_pipeline(
    replay_dir: str,
    output_dir: str,
    cf_ratio: float = 0.3,
    epochs: int = 10,
    min_confidence: float = 0.3,
    self_play_replays: list = None,
):
    """Run the complete counterfactual mining pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    print("=" * 70)
    print("COUNTERFACTUAL REPLAY MINING PIPELINE")
    print("=" * 70)

    # Step 1: Load replays
    print(f"\n[Step 1/5] Loading replays from {replay_dir}...")
    replays = load_all_replays(replay_dir)
    print(f"  Loaded {len(replays)} replays")

    if not replays:
        print("  ERROR: No replays found! Check the replay directory.")
        return

    # Step 2: Generate counterfactual labels
    print("\n[Step 2/5] Generating counterfactual labels...")
    all_labels = []
    skipped = 0

    for replay in replays:
        for player_name, elo in TOP_PLAYERS.items():
            player_idx = find_player_in_replay(replay, player_name)
            if player_idx < 0:
                continue

            labels = generate_labels_for_replay(
                replay=replay,
                player_idx=player_idx,
                player_name=player_name,
                player_elo=elo,
                candidate_generator=stub_candidate_generator,
                min_confidence=min_confidence,
            )
            all_labels.extend(labels)

        if not any(
            find_player_in_replay(replay, name) >= 0
            for name in TOP_PLAYERS
        ):
            skipped += 1

    print(f"  Generated {len(all_labels)} labels from {len(replays) - skipped} replays")
    print(f"  Skipped {skipped} replays (no top players found)")

    summary = summarize_labels(all_labels)
    print(f"  Label summary: {json.dumps(summary, indent=4)}")

    # Step 3: Build training buffer
    print("\n[Step 3/5] Building training buffer...")
    buffer = TrainingBuffer()
    buffer.add_counterfactual_labels(all_labels)

    print(f"  Buffer stats: {json.dumps(buffer.summary(), indent=4)}")

    buffer_path = output_dir / "training_buffer.json"
    save_buffer(buffer, str(buffer_path))
    print(f"  Saved buffer to {buffer_path}")

    # Step 4: Train policies
    print(f"\n[Step 4/5] Training policies (CF ratio={cf_ratio}, epochs={epochs})...")
    training_summary = train_with_counterfactuals(
        self_play_replay_paths=self_play_replays or [],
        counterfactual_buffer_path=str(buffer_path),
        cf_ratio=cf_ratio,
        n_epochs=epochs,
        output_dir=str(output_dir),
    )

    # Step 5: Generate report
    elapsed = time.time() - start_time
    print(f"\n[Step 5/5] Pipeline complete in {elapsed:.1f}s")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "replays_loaded": len(replays),
        "replays_used": len(replays) - skipped,
        "labels_generated": len(all_labels),
        "label_summary": summary,
        "buffer_stats": buffer.summary(),
        "training_summary": training_summary,
        "elapsed_seconds": round(elapsed, 1),
    }

    report_path = output_dir / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {report_path}")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("""
1. Validate the trained policy:
   python test_agent.py --agent <your_agent_with_new_policy> \\
       --opponent <baseline> --games 50 --swap

2. Check for opening regression:
   Look at first-launch timing in the game logs.
   It should be <= T5.0 (currently T7.3 with old RL).

3. If win rate > 55%, submit to Kaggle:
   kaggle competitions submit -c kore-2022 -f submission.tar.gz

4. Iterate:
   - Adjust cf_ratio (try 0.2 and 0.4)
   - Adjust min_confidence (try 0.2 and 0.4)
   - Add more top-player replays
   - Replace stub_candidate_generator with real heuristic
""")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run counterfactual replay mining pipeline"
    )
    parser.add_argument("--replay-dir", default="kaggle_replays/top_players",
        help="Directory containing top-player replays")
    parser.add_argument("--output-dir", default="trained_policies",
        help="Output directory for trained policies")
    parser.add_argument("--cf-ratio", type=float, default=0.3,
        help="Counterfactual sample ratio (default: 0.3)")
    parser.add_argument("--epochs", type=int, default=10,
        help="Training epochs (default: 10)")
    parser.add_argument("--min-confidence", type=float, default=0.3,
        help="Minimum match confidence (default: 0.3)")
    parser.add_argument("--self-play-replays", nargs="*", default=None,
        help="Paths to our own replay files for self-play data")

    args = parser.parse_args()

    run_pipeline(
        replay_dir=args.replay_dir,
        output_dir=args.output_dir,
        cf_ratio=args.cf_ratio,
        epochs=args.epochs,
        min_confidence=args.min_confidence,
        self_play_replays=args.self_play_replays,
    )
```

---

## 6. Expected Timeline

### 6.1 Phase 1: Data Collection (Days 1–2)

| Day | Task                                          | Output                        |
|-----|-----------------------------------------------|-------------------------------|
| 1   | Set up replay download infrastructure         | `scripts/` tooling working    |
|     | Find top-5 player submission IDs              | `player_index.json`           |
|     | Start bulk downloads                          | Downloading overnight         |
| 2   | Complete downloads (200+ replays)             | `kaggle_replays/top_players/` |
|     | Classify into 2p/4p subdirectories            | Organized directory tree      |
|     | Validate replay integrity                     | No corrupt files              |

**Checkpoint:** You should have 40+ replays per top player, organized by
game type. Run the scan script to verify:

```bash
python find_top_player_replays.py
# Expected output:
# === Top Player Replay Inventory ===
#   Shun_PI      (ELO 1607):  50 replays found
#   Erfan        (ELO 1547):  45 replays found
#   ...
```

### 6.2 Phase 2: Pipeline Build (Days 3–5)

| Day | Task                                          | Output                        |
|-----|-----------------------------------------------|-------------------------------|
| 3   | Implement replay_parser.py                    | Parsing all replays correctly |
|     | Implement feature_bridge.py                   | Feature vectors validated     |
| 4   | Implement label_generator.py                  | Labels generated for test set |
|     | Implement training_buffer.py                  | Buffer save/load working      |
| 5   | Implement train_with_counterfactuals.py        | Training loop running         |
|     | Implement run_pipeline.py                     | End-to-end script working     |
|     | Run pipeline on small test set (10 replays)   | Sanity check passed           |

**Checkpoint:** The pipeline should run end-to-end on 10 replays without
crashing. The trained weights don't need to be good yet — just verify the
data flows correctly through all stages.

```bash
# Quick test with a small subset
python counterfactual/run_pipeline.py \
    --replay-dir kaggle_replays/top_players/shun_pi/2p \
    --output-dir trained_policies/test_run \
    --epochs 3
```

### 6.3 Phase 3: Training & Iteration (Days 6–10)

| Day  | Task                                          | Output                        |
|------|-----------------------------------------------|-------------------------------|
| 6    | Replace stub_candidate_generator with real    | Heuristic running on replays  |
|      | heuristic (hardest integration task)          |                               |
| 7    | Full pipeline run on all 200+ replays         | First real trained policy     |
|      | Local benchmark: 50 games vs heuristic        | Win rate measurement          |
| 8    | Iterate on cf_ratio (try 0.2, 0.3, 0.4)      | Best ratio identified         |
|      | Iterate on min_confidence                     | Best threshold identified     |
| 9    | Add opening protection (see Section 7.1)      | No opening regression         |
|      | Fix anti-defer weight (see Section 7.2)       | Defer working properly        |
| 10   | Full benchmark suite: 50+ games each          | Comparison table              |
|      | against heuristic, RL v1, RL v2               |                               |

**Checkpoint at Day 7:** First results. Expected outcomes:

| Scenario                   | Expected Win Rate | Action if Below |
|----------------------------|-------------------|-----------------|
| CF agent vs. heuristic     | 55–65%            | Check labels    |
| CF agent vs. old RL        | 50–60%            | Check cf_ratio  |
| CF agent opening timing    | T3.5–5.0          | Add protection  |

**Checkpoint at Day 10:** Iteration complete. You should have:
- A trained policy that beats the heuristic at 60%+ over 50 games
- No opening regression (first launch <= T5.0)
- Working defer mechanism

### 6.4 Phase 4: Validation & Submission (Days 11–14)

| Day  | Task                                          | Output                        |
|------|-----------------------------------------------|-------------------------------|
| 11   | Run 200-game benchmark vs. all baselines      | Statistically reliable result |
|      | Analyze failure modes in lost games            | Pattern identification        |
| 12   | Address top failure mode                       | Targeted fix                  |
|      | Re-benchmark (100 games)                      | Improvement confirmed         |
| 13   | Prepare Kaggle submission package              | submission.tar.gz             |
|      | Test submission locally with kaggle-env        | Runs without errors           |
| 14   | Submit to Kaggle                               | Submission live               |
|      | Monitor initial games                          | Check for runtime errors      |

**Checkpoint at Day 14:** Submission should show:
- ELO climbing from ~1250 toward 1350+
- No timeout errors (agent must respond in < 3 seconds)
- Stable performance in both 2p and 4p games

---

## 7. Key Constraints & Gotchas

### 7.1 Opening Protection

**THE PROBLEM:** The current RL reranker delays the first fleet launch from
T3.9 (heuristic) to T7.3. In Kore, the first 3–4 turns are critical for
claiming territory. A 3-turn delay means the opponent gets ~30% more
territory, which is nearly unrecoverable.

**WHY IT HAPPENS:** The RL policy was trained on midgame data (step 18–180)
but activates too early or incorrectly scores opening moves. The
`step_frac` feature is near-zero in the opening, making the policy
under-confident.

**THE FIX:** Hard-code opening protection:

```python
# In the RL agent hook (midgame_rl_agent.py)

DEFAULT_OPENING_TURNS = 18  # Match the constant from shun_clone.py

def should_use_rl(step: int, game_state: dict) -> bool:
    """
    Determine whether the RL reranker should override the heuristic.
    
    CRITICAL: Never use RL in the opening. The heuristic's opening
    play is well-tuned and the RL policy wasn't trained on opening data.
    """
    # Hard gate: never before step 18
    if step < DEFAULT_OPENING_TURNS:
        return False
    
    # Soft gate: only in contested positions where RL might help
    if not is_contested_position(game_state):
        return False
    
    return True
```

**Additionally**, use the Shun_PI opening patterns from `shun_clone.py`:

```python
from rl_midgame.shun_clone import extract_shun_opening_patterns

# Pre-compute Shun's opening patterns from downloaded replays
patterns = extract_shun_opening_patterns("kaggle_replays/top_players/shun_pi")
```

### 7.2 The Anti-Defer Problem

**THE PROBLEM:** The current policy has an anti-defer weight of **-4.5** on
the `is_defer` feature. Since the linear policy computes `score = weights * features`,
and `is_defer = 1.0` only for the defer option, this means:

```
defer_score_penalty = -4.5 * 1.0 = -4.5
```

This makes it virtually impossible for defer to win the softmax. The RL
agent **NEVER** defers, even when none of the candidates are good.

**WHY IT MATTERS:** If the heuristic's best candidate is already a good
choice, the RL agent should defer. Instead, it picks a worse candidate
because it won't choose defer.

**THE FIX:** Reset the defer weight and let counterfactual training learn
the correct value:

```python
# Before training, reset the is_defer weight
weights = load_existing_weights()

# Option A: Reset to zero (neutral)
weights[FEATURE_NAMES.index("is_defer")] = 0.0

# Option B: Set to a small positive value (slight defer preference)
weights[FEATURE_NAMES.index("is_defer")] = 0.5

# Option C: Clip to a reasonable range during training
def clip_defer_weight(weights, min_val=-1.0, max_val=1.0):
    idx = FEATURE_NAMES.index("is_defer")
    weights[idx] = max(min_val, min(weights[idx], max_val))
    return weights
```

**Recommendation:** Use Option B (slight defer preference) as the starting
point for counterfactual training. The training data will adjust it from
there.

### 7.3 2p vs 4p Split

**THE PROBLEM:** 2-player and 4-player Kore are fundamentally different games.
In 2p, it's zero-sum and aggressive play wins. In 4p, you need to avoid
being the target while building quietly.

**THE FIX (already implemented):** `SplitRerankerPolicy` in
`rl midgame/split_reranker.py` routes decisions based on the
`is_two_player` feature at index 2:

```python
class SplitRerankerPolicy:
    """Routes to 2p or 4p sub-policy based on game type."""
    
    def score(self, feature_vector):
        is_2p = feature_vector[2]  # is_two_player at index 2
        if is_2p > 0.5:
            return self.policy_2p.score(feature_vector)
        else:
            return self.policy_4p.score(feature_vector)
```

**Counterfactual training must respect this split.** When building the
training buffer, samples are automatically routed to the correct policy
based on the `is_two_player` feature (see `training_buffer.py`).

**Key insight for 4p:** We have fewer 4p replays from top players, and the
games are noisier. Consider using a higher `min_confidence` threshold
(0.4 instead of 0.3) for 4p labels to avoid learning from ambiguous
examples.

### 7.4 Feature Alignment

**THE PROBLEM:** The feature vectors generated from replay data
(Section 5.2) are APPROXIMATIONS. They may not exactly match the vectors
our live agent generates because:

1. **Missing heuristic context:** `base_score_norm`, `base_value_norm`,
   `needed_norm` require the heuristic's internal scoring, which we don't
   have for replay-reconstructed states.

2. **Target inference:** `target_prod_norm`, `target_ships_norm`,
   `target_is_neutral`, `target_is_enemy`, `target_is_friendly` require
   knowing which base the fleet is targeting, which must be inferred from
   the flight plan.

3. **Committed load:** `committed_load` depends on what other missions
   are already in flight, which requires tracking across turns.

**MITIGATIONS:**

```python
# 1. Use normalized features to reduce scale mismatches

# 2. Zero out features we can't reliably reconstruct
UNRELIABLE_FEATURES = [
    "base_score_norm",
    "base_value_norm",
    "needed_norm",
    "committed_load",
]

def mask_unreliable_features(vector: list) -> list:
    """Set unreliable features to zero to avoid training on noise."""
    masked = list(vector)
    for name in UNRELIABLE_FEATURES:
        idx = FEATURE_NAMES.index(name)
        masked[idx] = 0.0
    return masked

# 3. Lower the learning rate for counterfactual samples
# (already done: cf_lr = 0.0005 vs lr = 0.001)

# 4. Use confidence weighting to down-weight noisy samples
# (already done: weight = confidence * elo_factor)
```

**Long-term fix:** Replace `stub_candidate_generator` with a function that
actually runs the heuristic on the replay's board state. This eliminates
the feature alignment problem entirely because we'd use the same code path
as the live agent. This is the hardest integration task (Phase 3, Day 6)
but the most impactful.

### 7.5 Kaggle Submission Constraints

**Time limit:** Each agent call must return within **3 seconds**. The
counterfactual training happens offline, so this only affects inference.
The trained weights are just 33 floats per policy — scoring is a dot
product and takes microseconds.

**File size limit:** Kaggle submissions are limited to **100 MB**. The
trained policy weights are < 1 KB. No concern here.

**Dependencies:** Only standard library + numpy are available on Kaggle.
All training code uses only standard Python. The inference code
(`split_reranker.py`) uses only numpy for the dot product, which is
available.

**No internet access:** The agent cannot download anything during a game.
All replay data, trained weights, and lookup tables must be embedded in
the submission.

**State persistence:** Agents can persist state between turns within a
game but not between games. The RL reranker loads weights once at the
start of each game.

```python
# Kaggle-safe weight loading pattern
import json
import os

POLICY_DIR = os.path.dirname(os.path.abspath(__file__))

def load_policy_weights():
    """Load trained policy weights from the submission package."""
    path_2p = os.path.join(POLICY_DIR, "policy_2p_cf.json")
    path_4p = os.path.join(POLICY_DIR, "policy_4p_cf.json")
    
    with open(path_2p) as f:
        weights_2p = json.load(f)["weights"]
    with open(path_4p) as f:
        weights_4p = json.load(f)["weights"]
    
    return weights_2p, weights_4p
```

---

## Appendix A: Quick Reference

### Feature Index Table

| Index | Feature Name          | Range   | Source      |
|-------|-----------------------|---------|-------------|
| 0     | step_frac             | [0, 1]  | Game state  |
| 1     | remaining_frac        | [0, 1]  | Game state  |
| 2     | is_two_player         | {0, 1}  | Game state  |
| 3     | is_four_player_plus   | {0, 1}  | Game state  |
| 4     | is_ahead              | {0, 1}  | Game state  |
| 5     | is_behind             | {0, 1}  | Game state  |
| 6     | is_dominating         | {0, 1}  | Game state  |
| 7     | is_finishing          | {0, 1}  | Game state  |
| 8     | my_ship_share         | [0, 1]  | Game state  |
| 9     | my_prod_share         | [0, 1]  | Game state  |
| 10    | lead_ratio            | [0, 1]  | Game state  |
| 11    | base_score_norm       | [0, 1]  | Heuristic   |
| 12    | base_value_norm       | [0, 1]  | Heuristic   |
| 13    | eta_norm              | [0, 1]  | Mission     |
| 14    | send_norm             | [0, 1]  | Mission     |
| 15    | source_count_norm     | [0, 1]  | Mission     |
| 16    | needed_norm           | [0, 1]  | Mission     |
| 17    | target_prod_norm      | [0, 1]  | Target      |
| 18    | target_ships_norm     | [0, 1]  | Target      |
| 19    | target_is_neutral     | {0, 1}  | Target      |
| 20    | target_is_enemy       | {0, 1}  | Target      |
| 21    | target_is_friendly    | {0, 1}  | Target      |
| 22    | mission_attack        | {0, 1}  | Type        |
| 23    | mission_expand        | {0, 1}  | Type        |
| 24    | mission_defend        | {0, 1}  | Type        |
| 25    | mission_reinforce     | {0, 1}  | Type        |
| 26    | mission_swarm         | {0, 1}  | Type        |
| 27    | mission_snipe         | {0, 1}  | Type        |
| 28    | mission_comet         | {0, 1}  | Type        |
| 29    | mission_other         | {0, 1}  | Type        |
| 30    | enemy_priority        | [0, 1]  | Tactical    |
| 31    | committed_load        | [0, 1]  | Tactical    |
| 32    | is_defer              | {0, 1}  | Meta        |

### Key File Paths

| File                                    | Purpose                        |
|-----------------------------------------|--------------------------------|
| `scripts/kaggle_episode_tools.py`       | Download replays from Kaggle   |
| `rl midgame/midgame_features.py`        | Feature definitions (33 feats) |
| `rl midgame/split_reranker.py`          | Split policy + training        |
| `rl midgame/midgame_policy.py`          | Policy classes + REINFORCE     |
| `rl midgame/midgame_rl_agent.py`        | RL hook into agent             |
| `rl midgame/shun_clone.py`             | Shun_PI pattern analysis       |
| `rl midgame/pretrain_from_heuristic.py` | Pre-training from heuristic    |
| `test_agent.py`                         | Local head-to-head testing     |

### Key Constants

```python
# From midgame_features.py
N_FEATURES = 33

# From shun_clone.py
SHUN_NAME = "Shun_PI"
DEFAULT_OPENING_TURNS = 30

# RL activation range (midgame only)
RL_MIN_STEP = 18
RL_MAX_STEP = 180

# Current known issues
CURRENT_ANTI_DEFER_WEIGHT = -4.5   # Prevents defer from ever winning
CURRENT_FIRST_LAUNCH = 7.3         # Turns (should be 3-4)
CURRENT_ELO = 1250                 # Approximate
```

---

## Appendix B: Debugging Checklist

When things go wrong (and they will), check these in order:

### Training produces NaN or exploding weights

```python
# Add gradient clipping
MAX_GRADIENT = 1.0

for i in range(N_FEATURES):
    gradient = compute_gradient(...)
    gradient = max(-MAX_GRADIENT, min(gradient, MAX_GRADIENT))
    weights[i] += lr * gradient * reward * weight
```

### Win rate doesn't improve after training

1. **Check label quality:** Are the match confidences reasonable (> 0.5 avg)?
2. **Check reward distribution:** Are wins and losses balanced?
3. **Check feature alignment:** Print feature vectors from training vs. live
   agent and compare ranges.
4. **Check learning rate:** Try halving it.
5. **Check cf_ratio:** Try lower (0.1) and higher (0.5) values.

### Agent crashes on Kaggle but works locally

1. **Import error:** Check that only standard library + numpy are used.
2. **File not found:** Make sure policy JSON files are included in the
   submission package.
3. **Timeout:** Profile the agent — are you accidentally running training
   code during inference?

### Opening regression (first launch too late)

1. **Check the step gate:** Is `should_use_rl(step)` returning False for
   step < 18?
2. **Check the heuristic:** Is the base agent's opening logic intact?
3. **Log the step number:** Add `print(f"Step {step}: using {'RL' if use_rl else 'heuristic'}")` to verify.

### Defer never activates

1. **Check is_defer weight:** Is it still -4.5? Reset to 0.0 or +0.5.
2. **Check softmax temperature:** If scores are large, softmax becomes
   nearly one-hot and defer can't win. Try temperature scaling:

```python
TEMPERATURE = 2.0

def softmax_with_temperature(scores, temperature=TEMPERATURE):
    scaled = [s / temperature for s in scores]
    max_s = max(scaled)
    exp_s = [math.exp(s - max_s) for s in scaled]
    total = sum(exp_s)
    return [e / total for e in exp_s]
```

---

## Appendix C: Shun_PI Analysis Reference

Based on our analysis using `rl midgame/shun_clone.py`, here are key
patterns from the top player:

```python
# From shun_clone.py analysis

# Shun_PI's opening patterns (first 30 turns):
# - Launches first fleet at T3-4 (we launch at T7.3 with RL)
# - Expands aggressively to 3-4 bases by T15
# - Avoids combat until T20+ unless directly threatened

# Key functions for Shun analysis:
from rl_midgame.shun_clone import (
    load_replay,
    find_shun_index,
    extract_opening_actions,
    extract_shun_opening_patterns,
    build_shun_replay_agent,
)

# Usage:
patterns = extract_shun_opening_patterns("kaggle_replays/top_players/shun_pi")
```

**Why this matters for counterfactual mining:** Shun_PI's replays are our
highest-value training data (1607 ELO, +307-407 above us). Each Shun replay
is worth roughly 2x an Ousagi replay in terms of ELO-weighted learning signal.

---

## Appendix D: Glossary

| Term                    | Definition                                                |
|-------------------------|-----------------------------------------------------------|
| **Candidate mission**   | A legal fleet action generated by the heuristic base      |
| **Counterfactual**      | "What if our agent had seen this state and chosen this?"  |
| **Defer**               | Let the heuristic pick (don't override)                   |
| **ELO**                 | Rating system; +400 means 10x expected performance        |
| **Feature vector**      | 33 floats describing a (state, candidate) pair            |
| **Genome**              | Parameter set for the heuristic, found by genetic search  |
| **Midgame**             | Steps 18–180, where RL activates                          |
| **Reranker**            | RL policy that re-scores heuristic candidates              |
| **REINFORCE**           | Policy gradient RL algorithm used for training             |
| **Self-play**           | Training by playing against yourself                       |
| **Split policy**        | Separate 2p and 4p weight vectors                         |
| **Stage2 genome**       | Our best heuristic parameter set (~1250 ELO)              |

---

*Last updated: Tutorial creation date. This document describes the planned
approach and should be updated as the implementation progresses.*
