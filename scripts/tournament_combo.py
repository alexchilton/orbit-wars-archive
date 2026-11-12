"""
Quick tournament to verify denial+wave combined agents.
2p: denial_wave vs denial, denial_wave vs wave, denial_wave vs plus2p
4p: combo vs political, combo vs plus4p, combo vs denial_wave (using 4 agents)
"""
import random
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from ppo_gnn.train_ppo import load_agent_from_file

from kaggle_environments import make

SUBMISSION_DIR = os.path.join(os.path.dirname(__file__), "submission")

def load(name):
    path = os.path.join(SUBMISSION_DIR, name)
    return load_agent_from_file(path)

def run_2p(a1_fn, a2_fn, n=20):
    wins = draws = losses = 0
    for i in range(n):
        env = make("orbit_wars", debug=False)
        if i % 2 == 0:
            env.run([a1_fn, a2_fn])
            final = env.steps[-1]
            r1 = final[0].reward or 0
            r2 = final[1].reward or 0
        else:
            env.run([a2_fn, a1_fn])
            final = env.steps[-1]
            r1 = final[1].reward or 0
            r2 = final[0].reward or 0
        if r1 > r2:   wins += 1
        elif r1 < r2: losses += 1
        else:         draws += 1
    return wins, draws, losses, wins / n * 100

def run_4p(agents_fns, n=16):
    wins = 0
    for i in range(n):
        lineup = agents_fns[:]
        random.shuffle(lineup)
        env = make("orbit_wars", debug=False)
        env.run(lineup)
        final = env.steps[-1]
        rewards = [final[j].reward or 0 for j in range(4)]
        best_r = max(rewards)
        for j, fn in enumerate(lineup):
            if fn is agents_fns[0]:
                if rewards[j] == best_r:
                    wins += 1
                break
    return wins, n


print("=" * 60)
print("Loading 2p agents...")
plus2p       = load("main_v131_plus_2p.py")
denial       = load("main_v131_plus_denial.py")
wave         = load("main_v131_plus_wave.py")
denial_wave  = load("main_v131_plus_denial_wave.py")

print("\n=== 2p TOURNAMENT (20 games each) ===")

matchups_2p = [
    ("denial_wave", denial_wave, "plus2p",  plus2p),
    ("denial_wave", denial_wave, "denial",  denial),
    ("denial_wave", denial_wave, "wave",    wave),
    ("denial",      denial,      "plus2p",  plus2p),   # baseline reference
    ("wave",        wave,        "plus2p",  plus2p),   # baseline reference
]

for n1, f1, n2, f2 in matchups_2p:
    w, d, l, wr = run_2p(f1, f2, n=20)
    print(f"  {n1:<18} vs {n2:<18}  W={w:2d} D={d:1d} L={l:2d}  ({wr:5.1f}%)")

print("\n=== 4p TOURNAMENT (16 games) ===")
print("Loading 4p agents...")
plus4p    = load("main_v131_plus_4p.py")
political = load("main_v131_plus_4p_political.py")
combo_4p  = load("main_v131_plus_4p_combo.py")

# Pool: combo_4p, political, plus4p, denial_wave (2p base in 4p context)
pool_4p = [combo_4p, political, plus4p, denial_wave]
labels  = ["combo_4p", "political", "plus4p", "denial_wave"]

for i, (fn, label) in enumerate(zip(pool_4p, labels)):
    w, total = run_4p([fn] + [pool_4p[j] for j in range(len(pool_4p)) if j != i], n=16)
    print(f"  {label:<20} wins {w:2d}/{total} ({w/total*100:.1f}%)")

print("\nDone.")
