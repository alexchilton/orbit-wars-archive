#!/usr/bin/env python3
"""
Genome evolutionary search for Orbit Wars heuristics.
Tests combinations of: sun_routing, smash_targets, comet_lookahead,
comet_urgency, inner_orbit (amt, max_ships).
"""
import json, os, random, sys, time
from pathlib import Path

CWD = '/Users/alexchilton/DataspellProjects/orbit_wars'
sys.path.insert(0, CWD)
os.environ['KAGGLE_ENVIRONMENTS_QUIET'] = '1'

BASE_AGENT = str(Path('/Users/alexchilton/DataspellProjects/orbit_wars/submission/main_genome_agent.py').absolute())
ORIG_AGENT = str(Path('/Users/alexchilton/DataspellProjects/orbit_wars/submission/main_fc_rl_shunlite.py').absolute())
RESULTS_FILE = str(Path('/Users/alexchilton/DataspellProjects/orbit_wars/genome_results.jsonl').absolute())
N_GAMES = 30  # games per matchup
CWD = '/Users/alexchilton/DataspellProjects/orbit_wars'

# Genome space
GENOME_SPACE = {
    'sun_routing':       [False, True],
    'smash_targets':     [False, True],
    'comet_lookahead':   [24, 26, 28],
    'comet_urgency':     [False, True],
    'inner_orbit':       [False, True],
    'inner_orbit_amt':   [1.06, 1.08, 1.10],
    'inner_orbit_ships': [30, 40, 60],
}

def genome_to_env(g):
    return {
        'GEN_SUN_ROUTING':       '1' if g['sun_routing'] else '0',
        'GEN_SMASH_TARGETS':     '1' if g['smash_targets'] else '0',
        'GEN_COMET_LOOKAHEAD':   str(g['comet_lookahead']),
        'GEN_COMET_URGENCY':     '1' if g['comet_urgency'] else '0',
        'GEN_INNER_ORBIT':       '1' if g['inner_orbit'] else '0',
        'GEN_INNER_ORBIT_AMT':   str(g['inner_orbit_amt']),
        'GEN_INNER_ORBIT_SHIPS': str(g['inner_orbit_ships']),
    }

def genome_label(g):
    parts = []
    if g['sun_routing']:   parts.append('SUN')
    if g['smash_targets']: parts.append('SMASH')
    if g['comet_lookahead'] != 24: parts.append(f'CL{g["comet_lookahead"]}')
    if g['comet_urgency']: parts.append('CURG')
    if g['inner_orbit']:   parts.append(f'IO{g["inner_orbit_amt"]:.2f}_{g["inner_orbit_ships"]}')
    return '+'.join(parts) if parts else 'BASELINE'

def random_genome():
    return {k: random.choice(v) for k, v in GENOME_SPACE.items()}

def mutate(g, n_mutations=1):
    g2 = dict(g)
    keys = list(GENOME_SPACE.keys())
    for k in random.sample(keys, min(n_mutations, len(keys))):
        g2[k] = random.choice(GENOME_SPACE[k])
    return g2

def load_agent_once(path, env_vars=None):
    """Load an agent module once with optional env vars set at import time."""
    import importlib.util
    if env_vars:
        for k, v in env_vars.items():
            os.environ[k] = v
    spec = importlib.util.spec_from_file_location(f"_agent_{id(env_vars)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if env_vars:
        for k in env_vars:
            os.environ.pop(k, None)
    return mod.agent

def run_matchup(genome, n_games=N_GAMES):
    """Run genome vs baseline shunlite, return win rate of genome."""
    import kaggle_environments

    wins_genome = 0
    wins_orig = 0
    draws = 0

    try:
        # Load both agents ONCE before game loop
        genome_agent = load_agent_once(BASE_AGENT, genome_to_env(genome))
        orig_agent   = load_agent_once(ORIG_AGENT)

        for i in range(n_games):
            swapped = (i % 2 == 1)
            agents = [orig_agent, genome_agent] if swapped else [genome_agent, orig_agent]

            try:
                env_kag = kaggle_environments.make("orbit_wars", debug=False)
                env_kag.run(agents)
                if swapped:
                    r_genome = env_kag.state[1]['reward']
                    r_orig   = env_kag.state[0]['reward']
                else:
                    r_genome = env_kag.state[0]['reward']
                    r_orig   = env_kag.state[1]['reward']

                if r_genome > r_orig:    wins_genome += 1
                elif r_orig > r_genome:  wins_orig += 1
                else:                    draws += 1
            except Exception as ex:
                print(f"  game {i} error: {ex}", flush=True)
                draws += 1

            wr = wins_genome / max(1, wins_genome + wins_orig)
            print(f"  [{i+1}/{n_games}] genome_wr={wr:.1%}  wins={wins_genome} orig={wins_orig} draws={draws}", flush=True)

    except Exception as e:
        import traceback
        print(f"ERROR in matchup: {e}", flush=True)
        traceback.print_exc()
        return 0.5

    wr = wins_genome / max(1, wins_genome + wins_orig)
    return wr

def run_generation(genomes, gen_num):
    results = []
    print(f"\n{'='*60}", flush=True)
    print(f"GENERATION {gen_num}  ({len(genomes)} genomes)", flush=True)
    print(f"{'='*60}", flush=True)

    for i, g in enumerate(genomes):
        label = genome_label(g)
        print(f"\n[{i+1}/{len(genomes)}] {label}", flush=True)
        wr = run_matchup(g)
        results.append({'genome': g, 'label': label, 'wr': wr})

        with open(RESULTS_FILE, 'a') as f:
            f.write(json.dumps({'gen': gen_num, 'label': label, 'wr': wr, 'genome': g}) + '\n')

        print(f"  RESULT: {label}  wr={wr:.1%}", flush=True)

    results.sort(key=lambda x: -x['wr'])
    print(f"\n--- Gen {gen_num} ranking ---", flush=True)
    for r in results:
        print(f"  {r['wr']:.1%}  {r['label']}", flush=True)
    return results

def main():
    random.seed(42)
    print("Genome evolutionary search for Orbit Wars", flush=True)
    print(f"Results → {RESULTS_FILE}", flush=True)

    # Generation 0: ablation — most promising first (comet_urgency showed 63.3%)
    gen0_genomes = [
        # 1. comet urgency only (Sonnet measured 63.3% — validate first)
        {'sun_routing': False, 'smash_targets': False, 'comet_lookahead': 24,
         'comet_urgency': True,  'inner_orbit': False, 'inner_orbit_amt': 1.08, 'inner_orbit_ships': 40},
        # 2. comet urgency + sun routing (best single + structural fix)
        {'sun_routing': True,  'smash_targets': False, 'comet_lookahead': 24,
         'comet_urgency': True,  'inner_orbit': False, 'inner_orbit_amt': 1.08, 'inner_orbit_ships': 40},
        # 3. comet urgency + smash (best single + aggression)
        {'sun_routing': False, 'smash_targets': True,  'comet_lookahead': 24,
         'comet_urgency': True,  'inner_orbit': False, 'inner_orbit_amt': 1.08, 'inner_orbit_ships': 40},
        # 4. sun routing alone (re-test with deflection cap fix)
        {'sun_routing': True,  'smash_targets': False, 'comet_lookahead': 24,
         'comet_urgency': False, 'inner_orbit': False, 'inner_orbit_amt': 1.08, 'inner_orbit_ships': 40},
        # 5. smash alone (re-test with toned-down 4x / 60% send)
        {'sun_routing': False, 'smash_targets': True,  'comet_lookahead': 24,
         'comet_urgency': False, 'inner_orbit': False, 'inner_orbit_amt': 1.08, 'inner_orbit_ships': 40},
        # 6. comet urgency + sun + smash (triple combo)
        {'sun_routing': True,  'smash_targets': True,  'comet_lookahead': 24,
         'comet_urgency': True,  'inner_orbit': False, 'inner_orbit_amt': 1.08, 'inner_orbit_ships': 40},
        # 7. baseline (calibration — should be ~50%)
        {'sun_routing': False, 'smash_targets': False, 'comet_lookahead': 24,
         'comet_urgency': False, 'inner_orbit': False, 'inner_orbit_amt': 1.08, 'inner_orbit_ships': 40},
        # 8. kitchen sink — all on with CL28
        {'sun_routing': True,  'smash_targets': True,  'comet_lookahead': 28,
         'comet_urgency': True,  'inner_orbit': True,  'inner_orbit_amt': 1.08, 'inner_orbit_ships': 40},
    ]

    results = run_generation(gen0_genomes, 0)

    # Generation 1: keep top 3, generate 4 random, try all-on
    top3 = [r['genome'] for r in results[:3]]
    gen1_genomes = top3 + [random_genome() for _ in range(4)] + [
        # all features on
        {'sun_routing': True, 'smash_targets': True, 'comet_lookahead': 28,
         'comet_urgency': True, 'inner_orbit': True, 'inner_orbit_amt': 1.08, 'inner_orbit_ships': 40},
    ]
    results1 = run_generation(gen1_genomes, 1)

    # Generation 2: top 2 + mutations
    top2 = [r['genome'] for r in results1[:2]]
    gen2_genomes = top2 + [mutate(g, n_mutations=1) for g in top2 * 3] + [random_genome() for _ in range(2)]
    results2 = run_generation(gen2_genomes, 2)

    print("\n\nFINAL RESULTS:", flush=True)
    all_results = []
    with open(RESULTS_FILE) as f:
        for line in f:
            all_results.append(json.loads(line))
    all_results.sort(key=lambda x: -x['wr'])
    print("Top 5 genomes:", flush=True)
    for r in all_results[:5]:
        print(f"  {r['wr']:.1%}  Gen{r['gen']}  {r['label']}", flush=True)

if __name__ == '__main__':
    main()
