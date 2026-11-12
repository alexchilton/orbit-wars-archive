#!/usr/bin/env python3
"""
Continuous-parameter genome search for Orbit Wars.
Tunes scoring weights (Tier 1) and margin/commitment params (Tier 2)
on top of the winning SMASH+CL28+CURG feature set while logging per-game map geometry.
"""
import hashlib
import json, os, random, sys, time, copy
from itertools import combinations
from pathlib import Path

CWD = '/Users/alexchilton/DataspellProjects/orbit_wars'
sys.path.insert(0, CWD)
os.environ['KAGGLE_ENVIRONMENTS_QUIET'] = '1'

BASE_AGENT = str(Path(CWD, 'submission/main_genome_candidate.py').absolute())
ORIG_AGENT = str(Path(CWD, 'submission/main_fc_rl_shunlite.py').absolute())
RESULTS_FILE = str(Path(CWD, 'genome_results_params.jsonl').absolute())
GAME_RESULTS_FILE = str(Path(CWD, 'genome_results_params_games.jsonl').absolute())
N_GAMES = 30

# ── Parameter space ──────────────────────────────────────────────────────────
# Tier 1: Scoring weights
# Tier 2: Margins & commitment
PARAM_SPACE = {
    # Tier 1 — scoring
    'ATTACK_COST_WEIGHT':       [0.50, 0.65, 0.75, 0.90],
    'SNIPE_COST_WEIGHT':        [0.85, 1.00, 1.08, 1.20],
    'STATIC_SCORE_MULT':        [1.00, 1.10, 1.18, 1.30],
    'EARLY_STATIC_SCORE_MULT':  [1.10, 1.18, 1.25, 1.35],
    'SNIPE_SCORE_MULT':         [0.95, 1.05, 1.12, 1.25],
    'SWARM_SCORE_MULT':         [0.90, 1.00, 1.06, 1.15],
    # Tier 2 — margins & commitment
    'NEUTRAL_MARGIN_BASE':      [1, 2, 3, 4],
    'HOSTILE_MARGIN_BASE':      [1, 2, 3, 4],
    'HOSTILE_MARGIN_CAP':       [8, 10, 12, 16],
    'MARGIN_PROD_WEIGHT':       [1, 2, 3],
    'DEFENSE_HORIZON':          [24, 28, 32],
    'PROACTIVE_KEEP_RATIO':     [0.12, 0.15, 0.18, 0.22],
}

# Defaults (current shunlite values)
DEFAULTS = {
    'ATTACK_COST_WEIGHT':       0.75,
    'SNIPE_COST_WEIGHT':        1.08,
    'STATIC_SCORE_MULT':        1.18,
    'EARLY_STATIC_SCORE_MULT':  1.25,
    'SNIPE_SCORE_MULT':         1.12,
    'SWARM_SCORE_MULT':         1.06,
    'NEUTRAL_MARGIN_BASE':      2,
    'HOSTILE_MARGIN_BASE':      2,
    'HOSTILE_MARGIN_CAP':       12,
    'MARGIN_PROD_WEIGHT':       2,
    'DEFENSE_HORIZON':          28,
    'PROACTIVE_KEEP_RATIO':     0.18,
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


def extract_initial_planets(env_kag):
    obs = None
    if getattr(env_kag, "steps", None) and len(env_kag.steps) > 1:
        obs = env_kag.steps[1][0].get('observation', {})
    if not obs:
        obs = env_kag.state[0].get('observation', {})
    return obs.get('initial_planets') or obs.get('planets') or []


def compute_map_features(raw_planets):
    if not raw_planets:
        return {}

    planets = [
        {
            'id': int(p[0]),
            'owner': int(p[1]),
            'x': float(p[2]),
            'y': float(p[3]),
            'radius': float(p[4]),
            'ships': int(p[5]),
            'prod': int(p[6]),
        }
        for p in raw_planets
    ]
    xs = [p['x'] for p in planets]
    ys = [p['y'] for p in planets]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    short_span = max(1.0, min(x_span, y_span))
    aspect_ratio = max(x_span, y_span) / short_span
    radii = [((p['x'] - 50.0) ** 2 + (p['y'] - 50.0) ** 2) ** 0.5 for p in planets]
    pair_dists = [
        ((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2) ** 0.5
        for a, b in combinations(planets, 2)
    ]
    owned = [p for p in planets if p['owner'] != -1]
    home_distance = None
    if len(owned) >= 2:
        first, second = owned[0], owned[1]
        home_distance = ((first['x'] - second['x']) ** 2 + (first['y'] - second['y']) ** 2) ** 0.5
    rounded = [
        [p['id'], p['owner'], round(p['x'], 3), round(p['y'], 3), round(p['radius'], 3), p['ships'], p['prod']]
        for p in planets
    ]
    raw_hash = json.dumps(sorted(rounded), separators=(",", ":"))
    layout = "balanced"
    if aspect_ratio >= 1.35:
        layout = "wide-x" if x_span >= y_span else "wide-y"
    return {
        'map_hash': hashlib.sha1(raw_hash.encode('utf-8')).hexdigest()[:12],
        'planet_count': len(planets),
        'x_span': round(x_span, 3),
        'y_span': round(y_span, 3),
        'aspect_ratio': round(aspect_ratio, 3),
        'avg_radius': round(sum(radii) / max(1, len(radii)), 3),
        'avg_pair_distance': round(sum(pair_dists) / max(1, len(pair_dists)), 3),
        'max_pair_distance': round(max(pair_dists) if pair_dists else 0.0, 3),
        'inner_orbit_count': sum(1 for p, r in zip(planets, radii) if (r + p['radius']) < 46.0),
        'static_count': sum(1 for r in radii if r < 35.0),
        'home_distance': round(home_distance, 3) if home_distance is not None else None,
        'layout': layout,
    }


def run_matchup(genome, gen_num, label, n_games=N_GAMES):
    import kaggle_environments
    wins_genome = 0
    wins_orig = 0
    draws = 0

    try:
        genome_agent = load_agent_once(BASE_AGENT, genome_to_env(genome))
        orig_agent = load_agent_once(ORIG_AGENT)

        for i in range(n_games):
            swapped = (i % 2 == 1)
            agents = [orig_agent, genome_agent] if swapped else [genome_agent, orig_agent]
            try:
                env_kag = kaggle_environments.make("orbit_wars", debug=False)
                env_kag.run(agents)
                if swapped:
                    r_genome = env_kag.state[1]['reward']
                    r_orig = env_kag.state[0]['reward']
                else:
                    r_genome = env_kag.state[0]['reward']
                    r_orig = env_kag.state[1]['reward']

                if r_genome > r_orig:     wins_genome += 1
                elif r_orig > r_genome:   wins_orig += 1
                else:                     draws += 1

                game_record = {
                    'gen': gen_num,
                    'label': label,
                    'game': i + 1,
                    'swapped': swapped,
                    'r_genome': r_genome,
                    'r_orig': r_orig,
                    'winner': 'genome' if r_genome > r_orig else ('orig' if r_orig > r_genome else 'draw'),
                    'genome': genome,
                }
                game_record.update(compute_map_features(extract_initial_planets(env_kag)))
                with open(GAME_RESULTS_FILE, 'a') as f:
                    f.write(json.dumps(game_record) + '\n')
            except Exception as ex:
                print(f"  game {i} error: {ex}", flush=True)
                draws += 1

            wr = wins_genome / max(1, wins_genome + wins_orig)
            print(f"  [{i+1}/{n_games}] wr={wr:.1%} w={wins_genome} l={wins_orig} d={draws}", flush=True)

    except Exception as e:
        import traceback
        print(f"ERROR in matchup: {e}", flush=True)
        traceback.print_exc()
        return 0.5

    return wins_genome / max(1, wins_genome + wins_orig)


def run_generation(genomes, gen_num):
    results = []
    print(f"\n{'='*60}", flush=True)
    print(f"GENERATION {gen_num}  ({len(genomes)} genomes)", flush=True)
    print(f"{'='*60}", flush=True)

    for i, g in enumerate(genomes):
        label = genome_label(g)
        print(f"\n[{i+1}/{len(genomes)}] {label}", flush=True)
        wr = run_matchup(g, gen_num, label)
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
    random.seed(1337)
    print("Continuous-parameter genome search for Orbit Wars", flush=True)
    print(f"Results -> {RESULTS_FILE}", flush=True)
    print(f"Params: {len(PARAM_SPACE)} dimensions", flush=True)

    # ── Gen 0: Baseline + single-param ablations ────────────────────────────
    # Test each param at its extremes vs baseline to find which matter
    gen0 = [dict(DEFAULTS)]  # baseline

    # Ablate each param to its min and max
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

    # Squeeze: aggressive attack (lower cost weights, tighter margins)
    gen0.append({**DEFAULTS,
        'ATTACK_COST_WEIGHT': 0.50,
        'NEUTRAL_MARGIN_BASE': 1,
        'HOSTILE_MARGIN_BASE': 1,
        'PROACTIVE_KEEP_RATIO': 0.12,
    })
    # Burn: all-in scoring boosts
    gen0.append({**DEFAULTS,
        'STATIC_SCORE_MULT': 1.30,
        'EARLY_STATIC_SCORE_MULT': 1.35,
        'SNIPE_SCORE_MULT': 1.25,
        'SWARM_SCORE_MULT': 1.15,
    })

    results0 = run_generation(gen0, 0)

    # ── Gen 1: Top 4 + mutations + crossovers ───────────────────────────────
    top4 = [r['genome'] for r in results0[:4]]
    gen1 = list(top4)
    # 2 mutations of each top genome
    for g in top4:
        gen1.append(mutate(g, 2))
        gen1.append(mutate(g, 3))
    # 2 crossovers
    if len(top4) >= 2:
        gen1.append(crossover(top4[0], top4[1]))
        gen1.append(crossover(top4[0], top4[2] if len(top4) > 2 else top4[1]))

    results1 = run_generation(gen1, 1)

    # ── Gen 2: Top 3 + fine mutations + crossovers ──────────────────────────
    top3 = [r['genome'] for r in results1[:3]]
    gen2 = list(top3)
    # 3 fine mutations of each (1 param change)
    for g in top3:
        for _ in range(3):
            gen2.append(mutate(g, 1))
    # crossovers of top 3
    for i in range(len(top3)):
        for j in range(i+1, len(top3)):
            gen2.append(crossover(top3[i], top3[j]))

    results2 = run_generation(gen2, 2)

    # ── Gen 3: Top 2 ultra-fine + random exploration ────────────────────────
    top2 = [r['genome'] for r in results2[:2]]
    gen3 = list(top2)
    for g in top2:
        for _ in range(4):
            gen3.append(mutate(g, 1))
    gen3.extend([random_genome() for _ in range(2)])

    results3 = run_generation(gen3, 3)

    # ── Final report ────────────────────────────────────────────────────────
    print("\n\nFINAL RESULTS:", flush=True)
    all_results = []
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                all_results.append(json.loads(line))
    all_results.sort(key=lambda x: -x['wr'])
    print("Top 10 genomes:", flush=True)
    for r in all_results[:10]:
        print(f"  {r['wr']:.1%}  Gen{r['gen']}  {r['label']}", flush=True)
    print("\nBest genome:", flush=True)
    print(json.dumps(all_results[0]['genome'], indent=2), flush=True)


if __name__ == '__main__':
    main()
