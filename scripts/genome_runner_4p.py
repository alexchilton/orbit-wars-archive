#!/usr/bin/env python3
"""
4-player genome search for Orbit Wars.
Optimizes for 1st-place rate (winner-take-all: +1/-1/-1/-1).
Tests removing/reducing conservative 4p dampening and throttling.
Matchup: candidate_genome vs v131 vs shunlite vs shunlite, rotating seats.
"""
import json, os, random, sys, time, copy
from pathlib import Path

CWD = '/Users/alexchilton/DataspellProjects/orbit_wars'
sys.path.insert(0, CWD)
os.environ['KAGGLE_ENVIRONMENTS_QUIET'] = '1'

BASE_AGENT = str(Path(CWD, 'submission/main_genome_candidate.py').absolute())
STRUCTURAL_AGENT = str(Path(CWD, 'submission/main_genome_candidate_structural.py').absolute())
V131_AGENT = '/Users/alexchilton/Downloads/main_v131.py'
SHUNLITE   = str(Path(CWD, 'submission/main_fc_rl_shunlite.py').absolute())
RESULTS_FILE = str(Path(CWD, 'genome_results_4p.jsonl').absolute())
N_GAMES = 20  # 4p games are slower

CANDIDATES = [
    ('baseline', BASE_AGENT),
    ('structural', STRUCTURAL_AGENT),
]

# ── 4P parameter space ──────────────────────────────────────────────────────
# Core idea: current 4p code is too conservative for winner-take-all scoring.
# Sweep from current (cautious) to aggressive (no dampening).
PARAM_SPACE = {
    # Launch throttling — higher = more aggressive
    '4P_CAUTIOUS_LAUNCH_CAP':    [6, 8, 10],       # current: 6
    '4P_PRESSURED_LAUNCH_CAP':   [3, 5, 6],         # current: 3
    '4P_RECOVERY_LAUNCH_CAP':    [5, 6, 8],          # current: 5
    # Reserve ratios — lower = commit more ships
    '4P_FRONTLINE_RESERVE':      [0.0, 0.08, 0.18],  # current: 0.18
    '4P_DOUBLE_FRONT_RESERVE':   [0.0, 0.12, 0.28],  # current: 0.28
    # Value dampening — higher = more willing to attack
    '4P_HOSTILE_DAMP':           [0.88, 1.0, 1.10],   # current: 0.88 (12% penalty!)
    '4P_PREP_HOSTILE_DAMP':      [0.92, 1.0, 1.08],   # current: 0.92
    '4P_RECOVERY_NEUTRAL_DAMP':  [0.90, 1.0, 1.05],   # current: 0.90
    # Aggression bonuses — higher = more aggressive when ahead/behind
    '4P_NEUTRAL_BONUS':          [1.06, 1.15, 1.25],   # current: 1.06
    '4P_RUNAWAY_PRIORITY':       [1.34, 1.50, 1.70],   # current: 1.34
    '4P_RUNAWAY_HOSTILE_BONUS':  [1.18, 1.35, 1.50],   # current: 1.18
    '4P_PIVOT_HOSTILE_BONUS':    [1.18, 1.30, 1.45],   # current: 1.18
    '4P_RECOVERY_HOSTILE_BONUS': [1.10, 1.25, 1.40],   # current: 1.10
}

DEFAULTS = {
    '4P_CAUTIOUS_LAUNCH_CAP':    6,
    '4P_PRESSURED_LAUNCH_CAP':   3,
    '4P_RECOVERY_LAUNCH_CAP':    5,
    '4P_FRONTLINE_RESERVE':      0.18,
    '4P_DOUBLE_FRONT_RESERVE':   0.28,
    '4P_HOSTILE_DAMP':           0.88,
    '4P_PREP_HOSTILE_DAMP':      0.92,
    '4P_RECOVERY_NEUTRAL_DAMP':  0.90,
    '4P_NEUTRAL_BONUS':          1.06,
    '4P_RUNAWAY_PRIORITY':       1.34,
    '4P_RUNAWAY_HOSTILE_BONUS':  1.18,
    '4P_PIVOT_HOSTILE_BONUS':    1.18,
    '4P_RECOVERY_HOSTILE_BONUS': 1.10,
}


def genome_to_env(g):
    env = {}
    for k, v in g.items():
        env[f'GEN_{k}'] = str(v)
    return env


def genome_label(g):
    diffs = []
    for k, v in sorted(g.items()):
        if v != DEFAULTS.get(k):
            diffs.append(f'{k}={v}')
    return '|'.join(diffs) if diffs else 'BASELINE'


def random_genome():
    return {k: random.choice(v) for k, v in PARAM_SPACE.items()}


def mutate(g, n_mutations=2):
    g2 = dict(g)
    keys = list(PARAM_SPACE.keys())
    for k in random.sample(keys, min(n_mutations, len(keys))):
        g2[k] = random.choice(PARAM_SPACE[k])
    return g2


def crossover(g1, g2):
    child = {}
    for k in PARAM_SPACE:
        child[k] = g1[k] if random.random() < 0.5 else g2[k]
    return child


def load_agent_once(path, env_vars=None):
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


def run_matchup_4p(candidate_name, candidate_path, genome, n_games=N_GAMES):
    """Run 4p: candidate vs v131 vs shunlite vs shunlite."""
    import kaggle_environments

    try:
        genome_agent = load_agent_once(candidate_path, genome_to_env(genome))
        v131_agent   = load_agent_once(V131_AGENT)
        shun_a       = load_agent_once(SHUNLITE)
        shun_b       = load_agent_once(SHUNLITE)
    except Exception as e:
        import traceback
        print(f"ERROR loading agents: {e}", flush=True)
        traceback.print_exc()
        return 0.25

    labels = ['candidate', 'v131', 'shun_a', 'shun_b']
    agents = [genome_agent, v131_agent, shun_a, shun_b]
    wins = 0
    total = 0

    for game in range(n_games):
        # Rotate seating
        order = [(i + game) % 4 for i in range(4)]
        game_agents = [agents[order[j]] for j in range(4)]
        game_labels = [labels[order[j]] for j in range(4)]

        try:
            env = kaggle_environments.make("orbit_wars", debug=False)
            env.run(game_agents)
            rewards = [env.state[j]['reward'] for j in range(4)]

            label_rewards = {}
            for j in range(4):
                label_rewards[game_labels[j]] = rewards[j]

            ranked = sorted(label_rewards.items(), key=lambda x: -x[1])
            winner = ranked[0][0]
            if winner == 'candidate':
                wins += 1
            total += 1

            print(f"  [{game+1}/{n_games}] {candidate_name:10s} winner={winner:10s}  candidate_1st={wins}/{total} ({100*wins/total:.0f}%)  {' > '.join(f'{l}' for l,_ in ranked)}", flush=True)
        except Exception as e:
            print(f"  game {game+1} error: {e}", flush=True)
            total += 1

    return wins / max(1, total)


def best_genomes(results, limit):
    by_genome = {}
    for row in results:
        key = json.dumps(row['genome'], sort_keys=True)
        existing = by_genome.get(key)
        if existing is None or row['first_rate'] > existing['first_rate']:
            by_genome[key] = row
    ranked = sorted(by_genome.values(), key=lambda row: -row['first_rate'])
    return [row['genome'] for row in ranked[:limit]]


def run_generation(genomes, gen_num):
    results = []
    print(f"\n{'='*60}", flush=True)
    print(f"GENERATION {gen_num}  ({len(genomes)} genomes, 4-PLAYER)", flush=True)
    print(f"{'='*60}", flush=True)

    for i, g in enumerate(genomes):
        label = genome_label(g)
        print(f"\n[{i+1}/{len(genomes)}] {label}", flush=True)
        for candidate_name, candidate_path in CANDIDATES:
            print(f"  Candidate: {candidate_name}", flush=True)
            first_rate = run_matchup_4p(candidate_name, candidate_path, g)
            row = {'gen': gen_num, 'candidate': candidate_name, 'genome': g, 'label': label, 'first_rate': first_rate}
            results.append(row)
            with open(RESULTS_FILE, 'a') as f:
                f.write(json.dumps(row) + '\n')
            print(f"  RESULT: {candidate_name} | {label}  1st_rate={first_rate:.1%}", flush=True)

    results.sort(key=lambda x: -x['first_rate'])
    print(f"\n--- Gen {gen_num} ranking (1st-place rate) ---", flush=True)
    for r in results:
        print(f"  {r['first_rate']:.1%}  {r['candidate']}  {r['label']}", flush=True)
    return results


def main():
    random.seed(42)
    print("4-PLAYER genome search — optimizing for 1st-place rate", flush=True)
    print(f"Scoring: +1/-1/-1/-1 (winner-take-all)", flush=True)
    print(f"Matchup: candidate vs v131 vs shunlite x2", flush=True)
    print(f"Results -> {RESULTS_FILE}", flush=True)

    # ── Gen 0: Key hypotheses ───────────────────────────────────────────────
    gen0 = [
        # 0. Baseline (current conservative 4p)
        dict(DEFAULTS),
        # 1. FULL AGGRO: remove all dampening, max launch caps, no reserves
        {**DEFAULTS,
         '4P_CAUTIOUS_LAUNCH_CAP': 10, '4P_PRESSURED_LAUNCH_CAP': 6,
         '4P_RECOVERY_LAUNCH_CAP': 8,
         '4P_FRONTLINE_RESERVE': 0.0, '4P_DOUBLE_FRONT_RESERVE': 0.0,
         '4P_HOSTILE_DAMP': 1.10, '4P_PREP_HOSTILE_DAMP': 1.08,
         '4P_RECOVERY_NEUTRAL_DAMP': 1.05,
         '4P_NEUTRAL_BONUS': 1.25, '4P_RUNAWAY_PRIORITY': 1.70,
         '4P_RUNAWAY_HOSTILE_BONUS': 1.50, '4P_PIVOT_HOSTILE_BONUS': 1.45,
         '4P_RECOVERY_HOSTILE_BONUS': 1.40},
        # 2. Remove dampening only (keep launch caps)
        {**DEFAULTS,
         '4P_HOSTILE_DAMP': 1.0, '4P_PREP_HOSTILE_DAMP': 1.0,
         '4P_RECOVERY_NEUTRAL_DAMP': 1.0},
        # 3. Remove reserves only (keep dampening)
        {**DEFAULTS,
         '4P_FRONTLINE_RESERVE': 0.0, '4P_DOUBLE_FRONT_RESERVE': 0.0},
        # 4. Raise launch caps only
        {**DEFAULTS,
         '4P_CAUTIOUS_LAUNCH_CAP': 10, '4P_PRESSURED_LAUNCH_CAP': 6,
         '4P_RECOVERY_LAUNCH_CAP': 8},
        # 5. Boost aggression bonuses only
        {**DEFAULTS,
         '4P_RUNAWAY_PRIORITY': 1.70, '4P_RUNAWAY_HOSTILE_BONUS': 1.50,
         '4P_PIVOT_HOSTILE_BONUS': 1.45, '4P_RECOVERY_HOSTILE_BONUS': 1.40},
        # 6. "Play like 2p": no damp + no reserves + high caps
        {**DEFAULTS,
         '4P_CAUTIOUS_LAUNCH_CAP': 10, '4P_PRESSURED_LAUNCH_CAP': 6,
         '4P_RECOVERY_LAUNCH_CAP': 8,
         '4P_FRONTLINE_RESERVE': 0.0, '4P_DOUBLE_FRONT_RESERVE': 0.0,
         '4P_HOSTILE_DAMP': 1.0, '4P_PREP_HOSTILE_DAMP': 1.0,
         '4P_RECOVERY_NEUTRAL_DAMP': 1.0},
        # 7. Moderate: halve the conservatism
        {**DEFAULTS,
         '4P_CAUTIOUS_LAUNCH_CAP': 8, '4P_PRESSURED_LAUNCH_CAP': 5,
         '4P_FRONTLINE_RESERVE': 0.08, '4P_DOUBLE_FRONT_RESERVE': 0.12,
         '4P_HOSTILE_DAMP': 0.94, '4P_PREP_HOSTILE_DAMP': 0.96,
         '4P_RECOVERY_NEUTRAL_DAMP': 0.95},
    ]

    results0 = run_generation(gen0, 0)

    # ── Gen 1: Top 3 + mutations + crossovers ───────────────────────────────
    top3 = best_genomes(results0, 3)
    gen1 = list(top3)
    for g in top3:
        gen1.append(mutate(g, 2))
        gen1.append(mutate(g, 3))
    if len(top3) >= 2:
        gen1.append(crossover(top3[0], top3[1]))
    if len(top3) >= 3:
        gen1.append(crossover(top3[0], top3[2]))

    results1 = run_generation(gen1, 1)

    # ── Gen 2: Top 2 + fine mutations ───────────────────────────────────────
    top2 = best_genomes(results1, 2)
    gen2 = list(top2)
    for g in top2:
        for _ in range(3):
            gen2.append(mutate(g, 1))
    for i in range(len(top2)):
        for j in range(i+1, len(top2)):
            gen2.append(crossover(top2[i], top2[j]))

    results2 = run_generation(gen2, 2)

    # ── Final report ────────────────────────────────────────────────────────
    print("\n\nFINAL 4P RESULTS:", flush=True)
    all_results = []
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                all_results.append(json.loads(line))
    all_results.sort(key=lambda x: -x['first_rate'])
    print("Top 5 genomes (1st-place rate):", flush=True)
    for r in all_results[:5]:
        print(f"  {r['first_rate']:.1%}  Gen{r['gen']}  {r['candidate']}  {r['label']}", flush=True)
    print("\nBest genome:", flush=True)
    print(json.dumps(all_results[0]['genome'], indent=2), flush=True)


if __name__ == '__main__':
    main()
