#!/usr/bin/env python3
"""Quick 4p retest with corrected params (NEUTRAL_DAMP=0.78, SEND_DOMINATE=0.65)."""
import importlib.util, os, sys, time
CWD = '/Users/alexchilton/DataspellProjects/orbit_wars'
sys.path.insert(0, CWD)
os.environ['KAGGLE_ENVIRONMENTS_QUIET'] = '1'

def load_agent(path):
    spec = importlib.util.spec_from_file_location(f"_agent_{time.time()}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent

import kaggle_environments
candidate = load_agent(f'{CWD}/submission/main_v131_plus_4p.py')
v131 = load_agent('/Users/alexchilton/Downloads/main_v131.py')
shun1 = load_agent(f'{CWD}/submission/main_fc_rl_shunlite.py')
shun2 = load_agent(f'{CWD}/submission/main_fc_rl_shunlite.py')

labels = ['v131+', 'v131', 'shunA', 'shunB']
agents = [candidate, v131, shun1, shun2]
wins = {l: 0 for l in labels}
n_games = 20

print(f"4P TEST: v131-plus-4p (v2c) vs v131 + 2x shunlite")
print(f"Params: NEUTRAL_DAMP=0.78, SEND_DOMINATE=0.65, CURG_MULT=1.12")

for g in range(n_games):
    order = [(i + g) % 4 for i in range(4)]
    game_agents = [agents[order[j]] for j in range(4)]
    game_labels = [labels[order[j]] for j in range(4)]
    try:
        env = kaggle_environments.make("orbit_wars", debug=False)
        env.run(game_agents)
        rewards = [env.state[j]['reward'] for j in range(4)]
        label_rewards = {game_labels[j]: rewards[j] for j in range(4)}
        ranked = sorted(label_rewards.items(), key=lambda x: -x[1])
        winner = ranked[0][0]
        wins[winner] += 1
        ranked_str = ' > '.join(f'{l}({r:.0f})' for l, r in ranked)
        print(f"  Game {g+1}/{n_games}: {ranked_str}", flush=True)
    except Exception as e:
        print(f"  Game {g+1} ERROR: {e}", flush=True)

first_rate = wins['v131+'] / max(1, n_games)
print(f"\n4P RESULT: v131-plus v2c 1st-rate = {wins['v131+']}/{n_games} = {first_rate:.0%}")
print(f"  Breakdown: v131+={wins['v131+']} v131={wins['v131']} shunA={wins['shunA']} shunB={wins['shunB']}")
