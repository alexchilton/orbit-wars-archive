#!/usr/bin/env python3
"""
test_pred_harness.py — Fixed-seed tournament harness for the predictive agent.

Runs the new main_pred_2p.py against:
  1. random        — sanity floor (should win ~95%+)
  2. denial agent  — our current production 2p (should be competitive)

Seeds:
  - Historically hard seeds from test_seeds.py
  - Random variety seeds (different map geometries)

Usage:
    cd /Users/alexchilton/DataspellProjects/orbit_wars
    python test_pred_harness.py [--games N] [--seeds all|hard|quick]
"""

import sys, os, math, importlib.util, time, argparse, collections
sys.path.insert(0, '.')

os.environ['KAGGLE_ENVIRONMENTS_QUIET'] = '1'

import kaggle_environments

# ── Load agents ───────────────────────────────────────────────────────────────

def load_agent(path, module_name=None):
    name = module_name or f'_agent_{path.replace("/", "_")}_{time.time()}'
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def make_random_agent():
    """Greedy random agent — expand to nearest cheapest neutral."""
    def random_agent(obs):
        if isinstance(obs, dict):
            player = obs.get('player', 0)
            planets_raw = obs.get('planets', [])
            fleets_raw  = obs.get('fleets', [])
        else:
            player = getattr(obs, 'player', 0)
            planets_raw = getattr(obs, 'planets', [])
            fleets_raw  = getattr(obs, 'fleets', [])
        if not planets_raw:
            return []
        my     = [p for p in planets_raw if p[1] == player]
        others = [p for p in planets_raw if p[1] != player]
        if not my or not others:
            return []
        moves = []
        in_flight = {f[5] for f in fleets_raw if f[1] == player}
        for mp in my:
            msrc_id, _, mx, my_y, _, mships, _ = mp[:7]
            if mships < 8:
                continue
            best, best_cost = None, float('inf')
            for tp in others:
                tid, towner, tx, ty, _, tships, _ = tp[:7]
                if tid in in_flight:
                    continue
                d = math.hypot(tx - mx, ty - my_y)
                if d < best_cost:
                    best_cost, best = d, (tid, tx, ty)
            if best:
                tid, tx, ty = best
                angle = math.atan2(ty - my_y, tx - mx)
                send  = max(5, int(mships * 0.6))
                moves.append([msrc_id, angle, send])
        return moves
    return random_agent


# ── Seed sets ─────────────────────────────────────────────────────────────────

# Seeds from test_seeds.py — historically used for validation
HARD_SEEDS = {
    '75940524': 1984508306,
    '75939703': 839397798,
    '75939481': 143877781,
    '75939243': 1175431795,
    '75938658': 219091541,
    '75939022': 2117651738,
}

# Diverse seeds for broad coverage
VARIETY_SEEDS = {
    'seed_42':   42,
    'seed_137':  137,
    'seed_271':  271,
    'seed_1000': 1000,
    'seed_2025': 2025,
    'seed_9999': 9999,
}

ALL_SEEDS    = {**HARD_SEEDS, **VARIETY_SEEDS}
QUICK_SEEDS  = {'seed_42': 42, 'seed_137': 137, '75939481': 143877781}


# ── Tournament runner ─────────────────────────────────────────────────────────

def run_game(agent_a, agent_b, seed, mode='2p'):
    """Run one game and return (result_a, result_b) where each is the final reward."""
    n_players = 2 if mode == '2p' else 4
    players   = [agent_a, agent_b]
    if n_players == 4:
        players.extend([agent_b, agent_b])

    env = kaggle_environments.make('orbit_wars', debug=False,
                                   configuration={'seed': seed})
    env.run(players)
    rewards = [env.state[j]['reward'] for j in range(n_players)]
    return rewards[0], rewards[1]


def run_matchup(agent_a, agent_b, seeds, games_per_seed=3,
                label_a='A', label_b='B', swap=True):
    """
    Run agent_a vs agent_b on each seed (and optionally both sides).
    Returns win/draw/loss counts for agent_a and per-seed breakdown.
    """
    results = {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0}
    per_seed = {}

    for seed_name, seed_val in seeds.items():
        seed_wins = 0
        seed_games = 0
        for g in range(games_per_seed):
            try:
                ra, rb = run_game(agent_a, agent_b, seed_val)
                results['total'] += 1
                seed_games += 1
                if ra > rb:
                    results['wins'] += 1
                    seed_wins += 1
                elif ra == rb:
                    results['draws'] += 1
                else:
                    results['losses'] += 1
            except Exception as e:
                print(f"  [ERROR] {seed_name} game {g}: {e}")

            if swap:
                try:
                    ra, rb = run_game(agent_b, agent_a, seed_val)
                    results['total'] += 1
                    seed_games += 1
                    if rb > ra:   # agent_a was player 1
                        results['wins'] += 1
                        seed_wins += 1
                    elif ra == rb:
                        results['draws'] += 1
                    else:
                        results['losses'] += 1
                except Exception as e:
                    print(f"  [ERROR] {seed_name} swap game {g}: {e}")

        wr = seed_wins / max(seed_games, 1) * 100
        per_seed[seed_name] = (seed_wins, seed_games, wr)

    total = results['total']
    wr_overall = 100.0 * results['wins'] / max(total, 1)
    return results, per_seed, wr_overall


def print_results(label_a, label_b, results, per_seed, wr):
    W, D, L, T = results['wins'], results['draws'], results['losses'], results['total']
    print(f"\n{'='*60}")
    print(f"  {label_a:30s} vs  {label_b}")
    print(f"  Win rate: {wr:.1f}%   ({W}W / {D}D / {L}L of {T} games)")
    print(f"{'='*60}")
    for seed_name, (sw, sg, swr) in per_seed.items():
        bar = '█' * int(swr / 10) + '░' * (10 - int(swr / 10))
        print(f"  {seed_name:20s}: {bar} {swr:5.1f}%  ({sw}/{sg})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Predictive agent tournament harness')
    parser.add_argument('--games', type=int, default=2,
                        help='Games per seed per matchup (default: 2)')
    parser.add_argument('--seeds', choices=['all', 'hard', 'quick'], default='quick',
                        help='Seed set to use (default: quick)')
    parser.add_argument('--no-swap', action='store_true',
                        help='Do not swap player positions')
    args = parser.parse_args()

    seeds = {'all': ALL_SEEDS, 'hard': HARD_SEEDS, 'quick': QUICK_SEEDS}[args.seeds]
    swap  = not args.no_swap
    N     = args.games

    print(f"\nPredictive agent tournament harness")
    print(f"Seeds: {args.seeds} ({len(seeds)} seeds × {N} games{'×2 (swapped)' if swap else ''})")

    # Load agents
    base_path = '/Users/alexchilton/DataspellProjects/orbit_wars/submission'
    print("\nLoading agents...", flush=True)
    try:
        pred_agent   = load_agent(f'{base_path}/main_pred_2p.py',   'pred_2p')
        denial_agent = load_agent(f'{base_path}/main_v131_plus_denial.py', 'denial')
        random_agent = make_random_agent()
        print("  ✓ pred_agent  (main_pred_2p.py)")
        print("  ✓ denial_agent (main_v131_plus_denial.py)")
        print("  ✓ random_agent")
    except Exception as e:
        print(f"  ✗ Failed to load agent: {e}")
        sys.exit(1)

    # ── Matchup 1: Pred vs Random ────────────────────────────────────────────
    print(f"\n[1/2] Predictive vs Random  ({len(seeds)} seeds × {N} games)...", flush=True)
    t0 = time.time()
    res1, per1, wr1 = run_matchup(pred_agent, random_agent, seeds,
                                   games_per_seed=N, swap=swap,
                                   label_a='Predictive', label_b='Random')
    print_results('Predictive', 'Random', res1, per1, wr1)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ── Matchup 2: Pred vs Denial ─────────────────────────────────────────────
    print(f"\n[2/2] Predictive vs Denial  ({len(seeds)} seeds × {N} games)...", flush=True)
    t0 = time.time()
    res2, per2, wr2 = run_matchup(pred_agent, denial_agent, seeds,
                                   games_per_seed=N, swap=swap,
                                   label_a='Predictive', label_b='Denial')
    print_results('Predictive', 'Denial', res2, per2, wr2)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  vs Random:  {wr1:.1f}%  win rate  (target: >90%)")
    print(f"  vs Denial:  {wr2:.1f}%  win rate  (target: >45% to be competitive)")

    status_vs_random = "✓ PASS" if wr1 >= 90 else "✗ FAIL"
    status_vs_denial = "✓ PASS" if wr2 >= 40 else "? INFO"
    print(f"\n  vs Random:  {status_vs_random}")
    print(f"  vs Denial:  {status_vs_denial}")
    print()

    # Exit non-zero if sanity floor fails
    if wr1 < 70:
        print("ERROR: Win rate vs random is below 70% — something is wrong!")
        sys.exit(1)


if __name__ == '__main__':
    main()
