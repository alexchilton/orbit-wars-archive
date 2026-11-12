#!/usr/bin/env python3
"""
v131-plus 2p parameter sweep for Orbit Wars.
Tests v131-plus-2p against the original v131 (the real opponent to beat).
Also tests against best shunlite variant for diversity.
"""
import json, os, random, sys, time, copy
from pathlib import Path

CWD = '/Users/alexchilton/DataspellProjects/orbit_wars'
sys.path.insert(0, CWD)
os.environ['KAGGLE_ENVIRONMENTS_QUIET'] = '1'

CANDIDATE_AGENT = str(Path(CWD, 'submission/main_v131_plus_2p.py').absolute())
V131_AGENT = '/Users/alexchilton/Downloads/main_v131.py'
SHUNLITE_AGENT = str(Path(CWD, 'submission/main_fc_rl_shunlite.py').absolute())
RESULTS_FILE = str(Path(CWD, 'genome_results_v131_2p.jsonl').absolute())
N_GAMES = 30  # 15 vs v131 + 15 vs shunlite, swapped sides

# ── Parameter space ──────────────────────────────────────────────────────────
PARAM_SPACE = {
    # Tier 1 — Phase aggression (biggest levers)
    'TAKEOVER_MARGIN':       [1.01, 1.03, 1.05, 1.08],
    'PRESSURE_STEP':         [18, 22, 28, 34],
    'SEND_PRESSURE':         [0.45, 0.55, 0.65, 0.75],
    'SEND_AGGRESSIVE':       [0.35, 0.45, 0.55, 0.65],
    # Tier 2 — Scoring weights
    'SCORE_PROD_WEIGHT':     [14, 18, 22],
    'SCORE_TT_PENALTY':      [1.5, 2.5, 3.5],
    'NEUTRAL_BONUS':         [15, 25, 35],
    'AGGRO_HOSTILE_BONUS':   [25, 35, 45],
    # Tier 3 — Comet fine-tuning
    'CURG_MULT':             [1.04, 1.08, 1.12, 1.16],
    'COMET_LOOKAHEAD':       [24, 28, 32],
    # Tier 4 — Send fractions
    'SEND_DOMINATE':         [0.4, 0.5, 0.6, 0.7],
    'SEND_CLEANUP':          [0.72, 0.82, 0.90],
}

DEFAULTS = {
    'TAKEOVER_MARGIN':       1.05,
    'PRESSURE_STEP':         28,
    'SEND_PRESSURE':         0.55,
    'SEND_AGGRESSIVE':       0.4,
    'SCORE_PROD_WEIGHT':     18,
    'SCORE_TT_PENALTY':      2.5,
    'NEUTRAL_BONUS':         25,
    'AGGRO_HOSTILE_BONUS':   35,
    'CURG_MULT':             1.08,
    'COMET_LOOKAHEAD':       28,
    'SEND_DOMINATE':         0.5,
    'SEND_CLEANUP':          0.82,
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
    """Run candidate vs v131 (half games) and vs shunlite (half games), swapped sides."""
    import kaggle_environments

    games_per_opponent = n_games // 2
    wins = 0
    losses = 0
    draws = 0
    wins_v131 = 0
    wins_shun = 0
    losses_v131 = 0
    losses_shun = 0

    try:
        candidate = load_agent_once(CANDIDATE_AGENT, genome_to_env(genome))
        v131 = load_agent_once(V131_AGENT)
        shunlite = load_agent_once(SHUNLITE_AGENT)

        opponents = [
            ('v131', v131, games_per_opponent),
            ('shunlite', shunlite, games_per_opponent),
        ]

        game_idx = 0
        for opp_name, opp_agent, opp_games in opponents:
            opp_w = 0
            opp_l = 0
            for i in range(opp_games):
                swapped = (i % 2 == 1)
                agents = [opp_agent, candidate] if swapped else [candidate, opp_agent]
                try:
                    env_kag = kaggle_environments.make("orbit_wars", debug=False)
                    env_kag.run(agents)
                    if swapped:
                        r_cand = env_kag.state[1]['reward']
                        r_opp = env_kag.state[0]['reward']
                    else:
                        r_cand = env_kag.state[0]['reward']
                        r_opp = env_kag.state[1]['reward']

                    if r_cand > r_opp:
                        wins += 1
                        opp_w += 1
                    elif r_opp > r_cand:
                        losses += 1
                        opp_l += 1
                    else:
                        draws += 1
                except Exception as ex:
                    print(f"  game {game_idx} error: {ex}", flush=True)
                    draws += 1

                game_idx += 1
                total_played = wins + losses + draws
                wr = wins / max(1, wins + losses)
                print(f"  [{game_idx}/{n_games}] vs {opp_name} wr={wr:.1%} w={wins} l={losses} d={draws}", flush=True)

            if opp_name == 'v131':
                wins_v131 = opp_w
                losses_v131 = opp_l
            else:
                wins_shun = opp_w
                losses_shun = opp_l

    except Exception as e:
        import traceback
        print(f"ERROR in matchup: {e}", flush=True)
        traceback.print_exc()
        return 0.5, 0.5, 0.5

    total = wins + losses
    wr_overall = wins / max(1, total)
    wr_v131 = wins_v131 / max(1, wins_v131 + losses_v131)
    wr_shun = wins_shun / max(1, wins_shun + losses_shun)
    return wr_overall, wr_v131, wr_shun


def run_generation(genomes, gen_num):
    results = []
    print(f"\n{'='*60}", flush=True)
    print(f"GENERATION {gen_num}  ({len(genomes)} genomes)", flush=True)
    print(f"{'='*60}", flush=True)

    for i, g in enumerate(genomes):
        label = genome_label(g)
        print(f"\n[{i+1}/{len(genomes)}] {label}", flush=True)
        wr, wr_v131, wr_shun = run_matchup(g, gen_num, label)
        results.append({'genome': g, 'label': label, 'wr': wr, 'wr_v131': wr_v131, 'wr_shun': wr_shun})

        with open(RESULTS_FILE, 'a') as f:
            f.write(json.dumps({
                'gen': gen_num, 'label': label,
                'wr': wr, 'wr_v131': wr_v131, 'wr_shun': wr_shun,
                'genome': g,
            }) + '\n')

        print(f"  RESULT: {label}  wr={wr:.1%} (v131={wr_v131:.1%} shun={wr_shun:.1%})", flush=True)

    results.sort(key=lambda x: -x['wr'])
    print(f"\n--- Gen {gen_num} ranking ---", flush=True)
    for r in results:
        print(f"  {r['wr']:.1%} (v131={r['wr_v131']:.1%} shun={r['wr_shun']:.1%})  {r['label']}", flush=True)
    return results


def main():
    random.seed(42)
    print("v131-plus 2p parameter sweep", flush=True)
    print(f"Results -> {RESULTS_FILE}", flush=True)
    print(f"Params: {len(PARAM_SPACE)} dimensions", flush=True)
    print(f"Opponents: v131 original + shunlite baseline", flush=True)

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
    all_results.sort(key=lambda x: -x['wr'])
    print("Top 10 genomes:", flush=True)
    for r in all_results[:10]:
        print(f"  {r['wr']:.1%} (v131={r['wr_v131']:.1%} shun={r['wr_shun']:.1%})  Gen{r['gen']}  {r['label']}", flush=True)
    print("\nBest genome:", flush=True)
    print(json.dumps(all_results[0]['genome'], indent=2), flush=True)


if __name__ == '__main__':
    main()
