#!/usr/bin/env python3
"""Quick 4-player test: candidate vs v131 vs shunlite vs shunlite."""
import os, sys, importlib.util
os.environ['KAGGLE_ENVIRONMENTS_QUIET'] = '1'

CWD = '/Users/alexchilton/DataspellProjects/orbit_wars'
sys.path.insert(0, CWD)

CANDIDATE = f'{CWD}/submission/main_genome_candidate.py'
V131      = '/Users/alexchilton/Downloads/main_v131.py'
SHUNLITE  = f'{CWD}/submission/main_fc_rl_shunlite.py'

N_GAMES = 20

def load_agent(path, label=""):
    spec = importlib.util.spec_from_file_location(f"_agent_{label}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent

print("Loading agents...", flush=True)
candidate_agent = load_agent(CANDIDATE, "candidate")
v131_agent      = load_agent(V131, "v131")
shunlite_agent  = load_agent(SHUNLITE, "shunlite")
shunlite_agent2 = load_agent(SHUNLITE, "shunlite2")

import kaggle_environments

# Kaggle scoring: 1st=1, 2nd=score_2, 3rd=score_3, 4th=score_4
# The user says "1 scores 2,3,4 -" meaning the rewards from the env
# are the actual scores — just use raw rewards directly
labels = ['candidate', 'v131', 'shunlite_A', 'shunlite_B']
agents = [candidate_agent, v131_agent, shunlite_agent, shunlite_agent2]
total_reward = {l: 0.0 for l in labels}
wins   = {l: 0 for l in labels}
positions = {l: [] for l in labels}

for game in range(N_GAMES):
    # Rotate seating each game
    order = [(i + game) % 4 for i in range(4)]
    game_agents = [agents[order[j]] for j in range(4)]
    game_labels = [labels[order[j]] for j in range(4)]

    try:
        env = kaggle_environments.make("orbit_wars", debug=False)
        env.run(game_agents)
        rewards = [env.state[j]['reward'] for j in range(4)]

        # Map back to original labels
        label_rewards = {}
        for j in range(4):
            label_rewards[game_labels[j]] = rewards[j]

        # Rank by reward (higher = better)
        ranked = sorted(label_rewards.items(), key=lambda x: -x[1])
        for pos, (lbl, rew) in enumerate(ranked):
            total_reward[lbl] += rew
            positions[lbl].append(pos + 1)
            if pos == 0:
                wins[lbl] += 1

        ranked_str = ' > '.join(f'{l}({r:.0f})' for l, r in ranked)
        print(f"  Game {game+1}/{N_GAMES}: {ranked_str}", flush=True)
    except Exception as e:
        print(f"  Game {game+1} ERROR: {e}", flush=True)

print(f"\n{'='*50}", flush=True)
print(f"4-PLAYER RESULTS ({N_GAMES} games)", flush=True)
print(f"Scoring: 1st=+1, 2nd/3rd/4th=-1 (winner-take-all)", flush=True)
print(f"{'='*50}", flush=True)
for lbl in labels:
    avg_pos = sum(positions[lbl]) / max(1, len(positions[lbl]))
    avg_rew = total_reward[lbl] / max(1, len(positions[lbl]))
    win_pct = 100.0 * wins[lbl] / max(1, len(positions[lbl]))
    print(f"  {lbl:15s}  1st={wins[lbl]:2d} ({win_pct:4.1f}%)  avg_reward={avg_rew:+.2f}  avg_pos={avg_pos:.2f}", flush=True)
