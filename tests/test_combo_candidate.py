#!/usr/bin/env python3
"""
Test the combined best-params candidate:
  2p: STATIC_SCORE_MULT=1.0
  4p: FRONTLINE_RESERVE=0, DOUBLE_FRONT_RESERVE=0

Runs against v131 in both 2p (20 games swapped) and 4p (20 games rotated).
"""
import os, sys, time, importlib.util

os.environ['KAGGLE_ENVIRONMENTS_QUIET'] = '1'
# Set the winning params
os.environ['GEN_STATIC_SCORE_MULT'] = '1.0'
os.environ['GEN_4P_FRONTLINE_RESERVE'] = '0.0'
os.environ['GEN_4P_DOUBLE_FRONT_RESERVE'] = '0.0'

CWD = '/Users/alexchilton/DataspellProjects/orbit_wars'
sys.path.insert(0, CWD)

CANDIDATE = f'{CWD}/submission/main_genome_candidate.py'
V131 = '/Users/alexchilton/Downloads/main_v131.py'
SHUNLITE = f'{CWD}/submission/main_fc_rl_shunlite.py'

N_2P = 20
N_4P = 20


def load_agent(path, label=""):
    spec = importlib.util.spec_from_file_location(f"_agent_{label}_{id(path)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


print("Loading agents...", flush=True)
candidate = load_agent(CANDIDATE, "combo_candidate")
v131 = load_agent(V131, "v131")
shunlite = load_agent(SHUNLITE, "shunlite")

import kaggle_environments

# ── 2P TEST: combo candidate vs v131 (swapped) ──────────────────────────────
print(f"\n{'='*56}")
print(f"  2P TEST: combo_candidate vs v131  ({N_2P} games x2 swapped)")
print(f"  Params: STATIC_SCORE_MULT=1.0")
print(f"{'='*56}\n")

wins_cand = 0
wins_v131 = 0
draws = 0
t0 = time.time()

for i in range(N_2P):
    # Candidate as P0
    env = kaggle_environments.make("orbit_wars", debug=False)
    env.run([candidate, v131])
    r0 = env.state[0]['reward']
    r1 = env.state[1]['reward']
    if r0 > r1:
        wins_cand += 1
        marker = "W"
    elif r1 > r0:
        wins_v131 += 1
        marker = "L"
    else:
        draws += 1
        marker = "D"
    print(f"  Game {i+1:>2}/{N_2P} (P0) {marker}  reward={r0:.0f} vs {r1:.0f}", flush=True)

for i in range(N_2P):
    # Candidate as P1
    env = kaggle_environments.make("orbit_wars", debug=False)
    env.run([v131, candidate])
    r0 = env.state[0]['reward']
    r1 = env.state[1]['reward']
    if r1 > r0:
        wins_cand += 1
        marker = "W"
    elif r0 > r1:
        wins_v131 += 1
        marker = "L"
    else:
        draws += 1
        marker = "D"
    print(f"  Game {i+1:>2}/{N_2P} (P1) {marker}  reward={r1:.0f} vs {r0:.0f}", flush=True)

total_2p = wins_cand + wins_v131 + draws
elapsed_2p = time.time() - t0
print(f"\n{'─'*56}")
print(f"  2P RESULTS ({total_2p} games, {elapsed_2p:.0f}s)")
print(f"  combo_candidate: {wins_cand} wins ({100*wins_cand/total_2p:.0f}%)")
print(f"  v131:            {wins_v131} wins ({100*wins_v131/total_2p:.0f}%)")
print(f"  draws:           {draws}")
print(f"{'─'*56}\n")

# ── 4P TEST: combo candidate vs v131 vs 2x shunlite ─────────────────────────
print(f"\n{'='*56}")
print(f"  4P TEST: combo_candidate vs v131 vs 2x shunlite  ({N_4P} games)")
print(f"  Params: FRONTLINE_RESERVE=0, DOUBLE_FRONT_RESERVE=0")
print(f"{'='*56}\n")

labels = ['candidate', 'v131', 'shunlite_A', 'shunlite_B']
agents_4p = [candidate, v131, shunlite, shunlite]
wins_4p = {l: 0 for l in labels}
total_reward = {l: 0.0 for l in labels}

t1 = time.time()
for game in range(N_4P):
    order = [(i + game) % 4 for i in range(4)]
    game_agents = [agents_4p[order[j]] for j in range(4)]
    game_labels = [labels[order[j]] for j in range(4)]

    try:
        env = kaggle_environments.make("orbit_wars", debug=False)
        env.run(game_agents)
        rewards = [env.state[j]['reward'] for j in range(4)]

        label_rewards = {}
        for j in range(4):
            label_rewards[game_labels[j]] = rewards[j]

        ranked = sorted(label_rewards.items(), key=lambda x: -x[1])
        for pos, (lbl, rew) in enumerate(ranked):
            total_reward[lbl] += rew
            if pos == 0:
                wins_4p[lbl] += 1

        ranked_str = ' > '.join(f'{l}({r:.0f})' for l, r in ranked)
        print(f"  Game {game+1:>2}/{N_4P}: {ranked_str}", flush=True)
    except Exception as e:
        print(f"  Game {game+1} ERROR: {e}", flush=True)

elapsed_4p = time.time() - t1
print(f"\n{'─'*56}")
print(f"  4P RESULTS ({N_4P} games, {elapsed_4p:.0f}s)")
print(f"  Scoring: winner-take-all (+1/-1/-1/-1)")
print(f"{'─'*56}")
for lbl in labels:
    win_pct = 100.0 * wins_4p[lbl] / N_4P
    avg_rew = total_reward[lbl] / N_4P
    print(f"  {lbl:15s}  1st={wins_4p[lbl]:2d} ({win_pct:4.1f}%)  avg_reward={avg_rew:+.2f}")
print(f"{'─'*56}")
print(f"\n  Total elapsed: {time.time()-t0:.0f}s")
