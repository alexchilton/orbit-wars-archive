#!/usr/bin/env python3
"""
v131-plus 4p parameter sweep for Orbit Wars.
Tests v131-plus-4p against v131 original + 2x shunlite.
Winner-take-all scoring: +1/-1/-1/-1.
"""
import json, os, random, sys, time
from pathlib import Path

CWD = '/Users/alexchilton/DataspellProjects/orbit_wars'
sys.path.insert(0, CWD)
os.environ['KAGGLE_ENVIRONMENTS_QUIET'] = '1'

CANDIDATE_AGENT = str(Path(CWD, 'submission/main_v131_plus_4p.py').absolute())
V131_AGENT = '/Users/alexchilton/Downloads/main_v131.py'
SHUNLITE_AGENT = str(Path(CWD, 'submission/main_fc_rl_shunlite.py').absolute())
RESULTS_FILE = str(Path(CWD, 'genome_results_v131_4p.jsonl').absolute())
N_GAMES = 20

# ── Parameter space ──────────────────────────────────────────────────────────
PARAM_SPACE = {
    # Tier 1 — Aggression timing & send fractions
    '4P_AGGRO_STEP':          [10, 14, 20, 26],
    '4P_SEND_AGGRESSIVE':     [0.55, 0.65, 0.75, 0.85],
    '4P_SEND_DOMINATE':       [0.65, 0.72, 0.82, 0.90],
    '4P_SEND_DOGPILE':        [0.72, 0.82, 0.90],
    # Tier 2 — Runaway detection & response
    '4P_RUNAWAY_STRENGTH':    [1.10, 1.18, 1.25],
    '4P_RUNAWAY_BONUS':       [1.20, 1.35, 1.50],
    '4P_HOSTILE_BONUS':       [12, 18, 24],
    # Tier 3 — Neutral handling & domination
    '4P_NEUTRAL_DAMP':        [0.65, 0.78, 0.90],
    '4P_DOMINATE_RATIO':      [1.10, 1.20, 1.30],
    # Tier 4 — Shared params (also affect 4p via env vars)
    'TAKEOVER_MARGIN':        [1.01, 1.03, 1.05, 1.08],
    'CURG_MULT':              [1.04, 1.08, 1.12, 1.16],
}

DEFAULTS = {
    '4P_AGGRO_STEP':          20,
    '4P_SEND_AGGRESSIVE':     0.65,
    '4P_SEND_DOMINATE':       0.72,
    '4P_SEND_DOGPILE':        0.82,
    '4P_RUNAWAY_STRENGTH':    1.18,
    '4P_RUNAWAY_BONUS':       1.35,
    '4P_HOSTILE_BONUS':       18,
    '4P_NEUTRAL_DAMP':        0.78,
    '4P_DOMINATE_RATIO':      1.20,
    'TAKEOVER_MARGIN':        1.05,
    'CURG_MULT':              1.08,
}


def genome_to_env(g):
    env = {}
    for k, v in g.items():
        env[f'V131_{k}'] = str(v)
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
    spec = importlib.util.spec_from_file_location(f"_agent_{id(env_vars)}_{time.time()}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if env_vars:
        for k in env_vars:
            os.environ.pop(k, None)
    return mod.agent


def run_matchup(genome, gen_num, label, n_games=N_GAMES):
    """Run candidate vs v131 + 2x shunlite in 4p, rotating seats."""
    import kaggle_environments

    candidate = load_agent_once(CANDIDATE_AGENT, genome_to_env(genome))
    v131 = load_agent_once(V131_AGENT)
    shunlite1 = load_agent_once(SHUNLITE_AGENT)
    shunlite2 = load_agent_once(SHUNLITE_AGENT)

    labels = ['candidate', 'v131', 'shunlite_A', 'shunlite_B']
    agents = [candidate, v131, shunlite1, shunlite2]
    wins = {l: 0 for l in labels}

    for game in range(n_games):
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
            wins[winner] += 1

            ranked_str = ' > '.join(f'{l}({r:.0f})' for l, r in ranked)
            print(f"  Game {game+1}/{n_games}: {ranked_str}", flush=True)
        except Exception as e:
            print(f"  Game {game+1} ERROR: {e}", flush=True)

    first_rate = wins['candidate'] / max(1, n_games)
    return first_rate, wins


def run_generation(genomes, gen_num):
    results = []
    print(f"\n{'='*60}", flush=True)
    print(f"GENERATION {gen_num}  ({len(genomes)} genomes)", flush=True)
    print(f"{'='*60}", flush=True)

    for i, g in enumerate(genomes):
        label = genome_label(g)
        print(f"\n[{i+1}/{len(genomes)}] {label}", flush=True)
        first_rate, wins = run_matchup(g, gen_num, label)
        results.append({'genome': g, 'label': label, 'first_rate': first_rate})

        with open(RESULTS_FILE, 'a') as f:
            f.write(json.dumps({
                'gen': gen_num, 'label': label,
                'first_rate': first_rate,
                'wins': wins,
                'genome': g,
            }) + '\n')

        print(f"  RESULT: {label}  1st={first_rate:.0%} (cand={wins['candidate']} v131={wins['v131']} "
              f"shunA={wins['shunlite_A']} shunB={wins['shunlite_B']})", flush=True)

    results.sort(key=lambda x: -x['first_rate'])
    print(f"\n--- Gen {gen_num} ranking ---", flush=True)
    for r in results:
        print(f"  1st={r['first_rate']:.0%}  {r['label']}", flush=True)
    return results


def main():
    random.seed(42)
    print("v131-plus 4p parameter sweep", flush=True)
    print(f"Results -> {RESULTS_FILE}", flush=True)
    print(f"Params: {len(PARAM_SPACE)} dimensions", flush=True)
    print(f"Matchup: candidate vs v131 vs 2x shunlite (rotating seats)", flush=True)

    # ── Gen 0: Baseline + single-param ablations ────────────────────────────
    gen0 = [dict(DEFAULTS)]  # baseline

    for param, values in PARAM_SPACE.items():
        lo, hi = min(values), max(values)
        if lo != DEFAULTS[param]:
            g = dict(DEFAULTS)
            g[param] = lo
            gen0.append(g)
        if hi != DEFAULTS[param]:
            g = dict(DEFAULTS)
            g[param] = hi
            gen0.append(g)

    # Combo: early aggro + high send + low margin
    gen0.append({**DEFAULTS,
        '4P_AGGRO_STEP': 10,
        '4P_SEND_AGGRESSIVE': 0.85,
        'TAKEOVER_MARGIN': 1.01,
    })

    results0 = run_generation(gen0, 0)

    # ── Gen 1: Top 5 + mutations + crossovers ───────────────────────────────
    top5 = [r['genome'] for r in results0[:5]]
    gen1 = list(top5)
    for g in top5:
        gen1.append(mutate(g, 2))
        gen1.append(mutate(g, 3))
    if len(top5) >= 2:
        gen1.append(crossover(top5[0], top5[1]))
        gen1.append(crossover(top5[0], top5[2] if len(top5) > 2 else top5[1]))

    results1 = run_generation(gen1, 1)

    # ── Gen 2: Top 3 + fine mutations ───────────────────────────────────────
    top3 = [r['genome'] for r in results1[:3]]
    gen2 = list(top3)
    for g in top3:
        for _ in range(3):
            gen2.append(mutate(g, 1))
    for i in range(len(top3)):
        for j in range(i + 1, len(top3)):
            gen2.append(crossover(top3[i], top3[j]))

    results2 = run_generation(gen2, 2)

    # ── Final report ────────────────────────────────────────────────────────
    print("\n\nFINAL RESULTS:", flush=True)
    all_results = []
    with open(RESULTS_FILE) as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))
    all_results.sort(key=lambda x: -x['first_rate'])
    print("Top 10 genomes:", flush=True)
    for r in all_results[:10]:
        print(f"  1st={r['first_rate']:.0%}  Gen{r['gen']}  {r['label']}", flush=True)
    print("\nBest genome:", flush=True)
    print(json.dumps(all_results[0]['genome'], indent=2), flush=True)


if __name__ == '__main__':
    main()
