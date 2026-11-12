#!/usr/bin/env python3
"""
Local validation test for v131-plus v2 (all sweep findings baked in).
Tests against v131 original in both 2p and 4p modes.
Must pass before Kaggle submission.
"""
import importlib.util, os, sys, time

CWD = '/Users/alexchilton/DataspellProjects/orbit_wars'
sys.path.insert(0, CWD)
os.environ['KAGGLE_ENVIRONMENTS_QUIET'] = '1'

V131_PLUS_2P = f'{CWD}/submission/main_v131_plus_2p.py'
V131_PLUS_4P = f'{CWD}/submission/main_v131_plus_4p.py'
V131_ORIG = '/Users/alexchilton/Downloads/main_v131.py'
SHUNLITE = f'{CWD}/submission/main_fc_rl_shunlite.py'

def load_agent(path):
    spec = importlib.util.spec_from_file_location(f"_agent_{time.time()}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent

def run_2p(n_games=24):
    """Run v131-plus-2p vs v131 original, swapping sides."""
    import kaggle_environments
    agent_a = load_agent(V131_PLUS_2P)
    agent_b = load_agent(V131_ORIG)
    wins_a, wins_b = 0, 0

    print(f"\n{'='*50}")
    print(f"2P TEST: v131-plus-2p (v2) vs v131 original")
    print(f"{'='*50}")

    for g in range(n_games):
        if g % 2 == 0:
            agents = [agent_a, agent_b]
            labels = ['v131+', 'v131']
        else:
            agents = [agent_b, agent_a]
            labels = ['v131', 'v131+']
        try:
            env = kaggle_environments.make("orbit_wars", debug=False)
            env.run(agents)
            r0 = env.state[0]['reward']
            r1 = env.state[1]['reward']
            if g % 2 == 0:
                if r0 > r1: wins_a += 1
                else: wins_b += 1
            else:
                if r1 > r0: wins_a += 1
                else: wins_b += 1
            winner = labels[0] if r0 > r1 else labels[1]
            print(f"  Game {g+1}/{n_games}: {labels[0]}({r0:.0f}) vs {labels[1]}({r1:.0f}) → {winner}", flush=True)
        except Exception as e:
            print(f"  Game {g+1} ERROR: {e}", flush=True)

    wr = wins_a / max(1, n_games)
    print(f"\n2P RESULT: v131-plus v2 wins {wins_a}/{n_games} = {wr:.0%} vs v131 original")
    return wr

def run_4p(n_games=20):
    """Run v131-plus-4p vs v131 + 2x shunlite, rotating seats."""
    import kaggle_environments
    candidate = load_agent(V131_PLUS_4P)
    v131 = load_agent(V131_ORIG)
    shun1 = load_agent(SHUNLITE)
    shun2 = load_agent(SHUNLITE)

    labels = ['v131+', 'v131', 'shunA', 'shunB']
    agents = [candidate, v131, shun1, shun2]
    wins = {l: 0 for l in labels}

    print(f"\n{'='*50}")
    print(f"4P TEST: v131-plus-4p (v2) vs v131 + 2x shunlite")
    print(f"{'='*50}")

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
    print(f"\n4P RESULT: v131-plus v2 1st-rate = {wins['v131+']}/{n_games} = {first_rate:.0%}")
    print(f"  Breakdown: v131+={wins['v131+']} v131={wins['v131']} shunA={wins['shunA']} shunB={wins['shunB']}")
    return first_rate

if __name__ == '__main__':
    print("v131-plus v2 local validation test")
    print("Changes baked in:")
    print("  2p: smash fix, SEND_CLEANUP=0.72, SEND_AGGRESSIVE=0.5, TT_PENALTY=3.5, CURG_MULT=1.12")
    print("  4p: smash fix, NEUTRAL_DAMP=0.9, SEND_DOMINATE=0.65, CURG_MULT=1.12")

    wr_2p = run_2p(24)
    first_4p = run_4p(20)

    print(f"\n{'='*50}")
    print(f"FINAL: 2p WR={wr_2p:.0%}, 4p 1st={first_4p:.0%}")
    if wr_2p >= 0.60 and first_4p >= 0.30:
        print("PASS — ready for Kaggle submission")
    else:
        print("NEEDS REVIEW — below threshold (2p>=60%, 4p>=30%)")
