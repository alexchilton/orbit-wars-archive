"""
Grid-search the analytic ROI weights by running a tournament.
Replaces the beam-search horizon guessing with a clean formula:

    V = w_prod * prod * (remaining - eta)
      + w_deny * prod * (remaining - eta)   # enemy only
      - w_send * ships_sent
      + w_cap  * ships_captured

Optimise (w_prod, w_send, w_cap); w_deny is fixed at w_prod (gain+deny symmetry).
"""
import sys, itertools, importlib, types, copy
sys.path.insert(0, 'submission')

import main_pred_2p as _agent_mod

GAMES_PER_TRIAL = 20   # quick — increase later

# ── Parameter grid ──────────────────────────────────────────────────────────
W_PROD_VALS = [0.5, 1.0, 2.0]
W_SEND_VALS = [0.5, 1.0, 2.0]
W_CAP_VALS  = [0.0, 0.5, 1.0]

# ── Monkey-patch _rough_roi in the agent module ──────────────────────────────
def make_analytic_roi(w_prod, w_send, w_cap):
    """Return a _rough_roi replacement using the analytic formula."""
    import math
    from physics_sim import travel_time
    def _rough_roi(src, tgt, av, omega, remaining_steps=400):
        dist = math.hypot(tgt['x'] - src['x'], tgt['y'] - src['y'])
        if dist < 1:
            return -1e9
        eta = dist / max(av, 1) * 10   # rough ETA (same scaling as original)
        needed = max(1, tgt['ships'] + tgt['prod'] * eta + 1)
        if av < needed:
            return -1e9
        is_enemy      = tgt['owner'] >= 0
        steps_held    = max(0.0, remaining_steps - eta)
        gain          = w_prod * tgt['prod'] * steps_held
        deny          = (w_prod * tgt['prod'] * steps_held) if is_enemy else 0.0
        send_cost     = w_send * needed
        cap_bonus     = w_cap  * max(0.0, tgt['ships'])
        return gain + deny + cap_bonus - send_cost
    return _rough_roi


def run_tournament(w_prod, w_send, w_cap, games=GAMES_PER_TRIAL):
    """Patch rough_roi, run games, return win rate 0..1."""
    import subprocess, json, pathlib, tempfile
    # Write a temp agent file that wraps main_pred_2p with patched weights
    agent_src = pathlib.Path('submission/main_pred_2p.py').read_text()
    # We just run the tournament via test_agent.py — but we need to patch inline.
    # Simpler: write a wrapper that sets module-level constants then delegates.
    wrapper = f"""
import sys; sys.path.insert(0, 'submission')
import main_pred_2p as _M
import math
from physics_sim import travel_time

def _rough_roi(src, tgt, av, omega, remaining_steps=400):
    dist = math.hypot(tgt['x'] - src['x'], tgt['y'] - src['y'])
    if dist < 1:
        return -1e9
    spd  = max(dist / 300, 0.05)
    eta  = dist / spd
    needed = max(1, tgt['ships'] + tgt['prod'] * eta + 1)
    if av < needed:
        return -1e9
    is_enemy   = tgt['owner'] >= 0
    steps_held = max(0.0, remaining_steps - eta)
    gain       = {w_prod} * tgt['prod'] * steps_held
    deny       = ({w_prod} * tgt['prod'] * steps_held) if is_enemy else 0.0
    send_cost  = {w_send} * needed
    cap_bonus  = {w_cap}  * max(0.0, tgt['ships'])
    return gain + deny + cap_bonus - send_cost

_M._rough_roi = _rough_roi

def agent(obs):
    return _M.agent(obs)
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir='submission',
                                     delete=False, prefix='_opt_') as f:
        f.write(wrapper)
        tmp = f.name

    try:
        result = subprocess.run(
            [sys.executable, 'test_agent.py',
             '--agent', tmp,
             '--opponent', 'submission/main_v131_plus_denial.py',
             '--games', str(games), '--swap', '--quiet'],
            capture_output=True, text=True, timeout=120
        )
        out = result.stdout + result.stderr
        # parse "wins:  N" line for our agent
        import re
        m = re.search(r'_opt_\w+\s+wins:\s+(\d+)', out)
        if not m:
            # try generic format
            lines = [l for l in out.splitlines() if 'wins:' in l]
            wins_a = int(re.search(r'wins:\s+(\d+)', lines[0]).group(1)) if lines else 0
        else:
            wins_a = int(m.group(1))
        total = games * 2
        return wins_a / total
    except Exception as e:
        print(f"  ERROR: {e}\n{out[-500:] if 'out' in dir() else ''}")
        return 0.0
    finally:
        import os; os.unlink(tmp)


if __name__ == '__main__':
    print(f"Grid search: {len(W_PROD_VALS)*len(W_SEND_VALS)*len(W_CAP_VALS)} trials × {GAMES_PER_TRIAL} games each\n")
    best_wr = -1.0
    best_params = None
    results = []
    for w_prod, w_send, w_cap in itertools.product(W_PROD_VALS, W_SEND_VALS, W_CAP_VALS):
        wr = run_tournament(w_prod, w_send, w_cap)
        results.append((wr, w_prod, w_send, w_cap))
        print(f"  w_prod={w_prod}  w_send={w_send}  w_cap={w_cap}  →  {wr*100:.0f}%")
        if wr > best_wr:
            best_wr = wr
            best_params = (w_prod, w_send, w_cap)

    results.sort(reverse=True)
    print(f"\n── Top 5 ──────────────────────────────────────────")
    for wr, wp, ws, wc in results[:5]:
        print(f"  {wr*100:.0f}%  w_prod={wp}  w_send={ws}  w_cap={wc}")
    print(f"\n★ Best: w_prod={best_params[0]}  w_send={best_params[1]}  w_cap={best_params[2]}  →  {best_wr*100:.0f}%")
