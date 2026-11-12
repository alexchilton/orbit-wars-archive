#!/usr/bin/env python3
"""Quick retest of v131-plus v2b (corrected params) vs v131 original — 2p only."""
import importlib.util, os, sys, time
CWD = '/Users/alexchilton/DataspellProjects/orbit_wars'
sys.path.insert(0, CWD)
os.environ['KAGGLE_ENVIRONMENTS_QUIET'] = '1'

def load_agent(path):
    spec = importlib.util.spec_from_file_location(f"_agent_{time.time()}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent

def run_2p(n_games=24):
    import kaggle_environments
    agent_a = load_agent(f'{CWD}/submission/main_v131_plus_2p.py')
    agent_b = load_agent('/Users/alexchilton/Downloads/main_v131.py')
    wins_a = 0
    print(f"2P TEST: v131-plus-2p (v2b) vs v131 original")
    print(f"Params: SEND_CLEANUP=0.82, SEND_AGGRESSIVE=0.55, TT_PENALTY=3.5, CURG_MULT=1.12")
    for g in range(n_games):
        agents = [agent_a, agent_b] if g % 2 == 0 else [agent_b, agent_a]
        labels = ['v131+', 'v131'] if g % 2 == 0 else ['v131', 'v131+']
        try:
            env = kaggle_environments.make("orbit_wars", debug=False)
            env.run(agents)
            r0, r1 = env.state[0]['reward'], env.state[1]['reward']
            if g % 2 == 0:
                if r0 > r1: wins_a += 1
            else:
                if r1 > r0: wins_a += 1
            winner = labels[0] if r0 > r1 else labels[1]
            print(f"  Game {g+1}/{n_games}: {labels[0]}({r0:.0f}) vs {labels[1]}({r1:.0f}) → {winner}", flush=True)
        except Exception as e:
            print(f"  Game {g+1} ERROR: {e}", flush=True)
    wr = wins_a / max(1, n_games)
    print(f"\n2P RESULT: v131-plus v2b wins {wins_a}/{n_games} = {wr:.0%} vs v131 original")

if __name__ == '__main__':
    run_2p(24)
