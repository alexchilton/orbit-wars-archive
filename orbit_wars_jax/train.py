"""
PPO self-play training for Orbit Wars.

WHAT THIS FILE DOES, STEP BY STEP
──────────────────────────────────
1. Pre-initialise N_ENVS game states (different seeds)
2. Loop over iterations:
   a. collect_rollout  – run N_ENVS envs for T_STEPS, both players use
                         current params; store (obs, actions, log_probs,
                         values, rewards, dones)
   b. compute_gae      – turn raw rewards into advantage estimates
   c. ppo_update       – K epochs of gradient descent on the rollout data
   d. reset envs that finished their episode
3. Print metrics; save checkpoints

SELF-PLAY NOTE
──────────────
Both players share the same model weights.  Player 0 and player 1 each
contribute their own trajectories to the training batch — the model sees
itself from both sides, which is the simplest form of self-play.

REWARD
──────
0 every step.  +1 / -1 only when the episode ends (someone is eliminated
or max steps reached).  Sparse reward — intentionally, per the 1st-place
team's findings.
"""

import os
import glob
import math
import time
import pickle
import functools

import jax
import jax.numpy as jnp
import optax
import numpy as np

from env import init_state, step_env, PO, PS, FO, FS
from obs import encode_obs, N_PLANET_TOKENS, N_FLEET_TOKENS, PLANET_FEAT, FLEET_FEAT
from model import OrbitWarsModel, count_params, N_FRAC_BINS
from act import sample_actions, log_prob_and_entropy, greedy_actions

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_ENVS        = 16      # parallel environments
T_STEPS       = 128     # rollout length (steps before each PPO update)
N_PPO_EPOCHS  = 1       # 1 epoch is enough; more epochs dominate wall time (98%) and cause clip_frac explosion
MINIBATCH     = 512     # keep at 512 — larger batches OOM the 16GB GPU during XLA autotuning
TARGET_KL     = 0.015   # early-stop epoch loop if approx_kl exceeds this (prevents clip_frac explosion)

GAMMA         = 0.99    # discount factor
GAE_LAMBDA    = 0.95    # GAE smoothing (higher = lower variance, less bias)
CLIP_EPS      = 0.1     # PPO clip range (keep policy change small)
# Per-head entropy coefs with cosine annealing (start → end over N_ITERS)
# Fraction head: 5 bins, simpler — anneal faster
ENT_FRAC_START = 0.02;  ENT_FRAC_END = 0.002
# Target head: 40 choices, harder — keep higher longer
ENT_TGT_START  = 0.01;  ENT_TGT_END  = 0.001
VF_COEF       = 0.5     # weight of value loss vs policy loss
MAX_GRAD_NORM = 0.5     # gradient clipping threshold

LR            = 2e-4    # peak learning rate
LR_WARMUP     = 200     # linearly ramp LR from 0 to LR over this many updates

N_ITERS       = 50_000  # total training iterations
EVAL_EVERY    = 10      # print metrics every N iterations
SAVE_EVERY    = 1000    # save checkpoint every N iterations
RESET_EVERY   = 4       # reset all envs every N rollouts (~512 steps, > 498 episode)

POOL_SIZE      = 5      # max past-param snapshots kept in opponent pool
SNAPSHOT_EVERY = 500    # freeze a snapshot into the pool every N iterations

SEED          = 42
CKPT_DIR      = "checkpoints"


# ── Learning rate schedule: linear warmup then cosine decay ──────────────────
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LR,
    warmup_steps=LR_WARMUP,
    decay_steps=N_ITERS,
    end_value=LR * 0.1,   # decay to 10% of peak, not zero
)

# ── Module-level singletons (no params needed at construction time) ────────────
model = OrbitWarsModel()

optimizer = optax.chain(
    optax.clip_by_global_norm(MAX_GRAD_NORM),
    optax.adam(lr_schedule),
)


# ── Stack / unstack env states ─────────────────────────────────────────────────

def stack_states(states: list) -> dict:
    """Stack a list of env state dicts into one batched dict (leading dim = N_ENVS)."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


def unstack_states(batched: dict, n: int) -> list:
    """Unstack a batched state dict into a list of N single-env dicts."""
    leaves, treedef = jax.tree_util.tree_flatten(batched)
    return [
        jax.tree_util.tree_unflatten(treedef, [leaf[i] for leaf in leaves])
        for i in range(n)
    ]


# ── Outcome / reward helpers ───────────────────────────────────────────────────

def _player_strength(state, pid: float) -> jnp.ndarray:
    """Total ships owned by player pid (planets + fleets in flight)."""
    owned_p = (state['planets'][:, PO] == pid) & state['planet_valid']
    owned_f = (state['fleets' ][:, FO] == pid) & state['fleet_valid']
    return (
        (state['planets'][:, PS] * owned_p).sum()
      + (state['fleets' ][:, FS] * owned_f).sum()
    )


def compute_rewards(old_state: dict, new_state: dict):
    """
    Returns (reward_p0, reward_p1) scalars.
    Reward is non-zero only at the terminal step (done just flipped True).
    Winner is whoever has more total ships at that moment.
    """
    terminal = new_state['done'] & ~old_state['done']
    s0 = _player_strength(new_state, 0.0)
    s1 = _player_strength(new_state, 1.0)
    p0_wins = s0 > s1
    r0 = jnp.where(terminal, jnp.where(p0_wins,  1.0, -1.0), 0.0)
    r1 = jnp.where(terminal, jnp.where(p0_wins, -1.0,  1.0), 0.0)
    return r0, r1


# ── Vmapped building blocks ────────────────────────────────────────────────────
# Each of these takes a *batched* argument (leading dim = N_ENVS) and
# processes all envs in parallel.

def vmap_encode(batched_state: dict, player_id: int) -> dict:
    return jax.vmap(encode_obs, in_axes=(0, None))(batched_state, player_id)


def vmap_forward(params, batched_obs: dict):
    """Run model on a batch of obs dicts. Returns (tgt [N,N_P,N_P], frac [N,N_P,N_FRAC], val [N])."""
    return jax.vmap(lambda obs: model.apply(params, obs))(batched_obs)


def vmap_step(batched_state, acts_p0, acts_p1):
    return jax.vmap(step_env)(batched_state, acts_p0, acts_p1)


def vmap_sample(tgt, frac, obs, state, rngs):
    """Returns (actions [N,MAX_L,3], lp [N], fb [N,N_P], ti [N,N_P], wl [N,N_P])."""
    return jax.vmap(sample_actions)(tgt, frac, obs, state, rngs)


def vmap_rewards(old_states, new_states):
    r0, r1 = jax.vmap(compute_rewards)(old_states, new_states)
    return r0, r1


# ── Rollout collection ────────────────────────────────────────────────────────

@jax.jit
def collect_rollout(params, opp_params, batched_states: dict, rng):
    """
    Run N_ENVS envs for T_STEPS steps using jax.lax.scan.

    p0 acts with current `params` (the learner).
    p1 acts with `opp_params` (a frozen snapshot from the pool).

    Only p0 trajectories are used for the PPO update — p1's log_probs
    were computed under opp_params, so importance-sampling ratios for p1
    would be meaningless under current params.

    Returns buf dict with arrays [T_STEPS, N_ENVS, ...],
    final states, updated rng, and bootstrap value last_val0.
    """
    n_envs = jax.tree_util.tree_leaves(batched_states)[0].shape[0]

    def scan_step(carry, _):
        states, rng = carry

        rng, rng_p0, rng_p1 = jax.random.split(rng, 3)
        rngs_p0 = jax.random.split(rng_p0, n_envs)
        rngs_p1 = jax.random.split(rng_p1, n_envs)

        obs_p0 = vmap_encode(states, 0)
        obs_p1 = vmap_encode(states, 1)

        tgt0, frac0, val0 = vmap_forward(params,     obs_p0)
        tgt1, frac1, _    = vmap_forward(opp_params, obs_p1)   # val1 not needed

        acts0, lp0, fb0, ti0, wl0 = vmap_sample(tgt0, frac0, obs_p0, states, rngs_p0)
        acts1, _, _, _, _ = vmap_sample(tgt1, frac1, obs_p1, states, rngs_p1)

        old_states = states
        new_states = vmap_step(states, acts0, acts1)
        r0, _      = vmap_rewards(old_states, new_states)

        step_data = {
            'obs_p0':          obs_p0,
            'frac_bins_p0':    fb0,
            'target_idxs_p0':  ti0,
            'will_launch_p0':  wl0,
            'log_probs_p0':    lp0,
            'values_p0':       val0,
            'rewards_p0':      r0,
            'dones':           new_states['done'],
        }
        return (new_states, rng), step_data

    # lax.scan stacks outputs automatically → buf has shape [T_STEPS, N_ENVS, ...]
    (final_states, rng), buf = jax.lax.scan(
        scan_step, (batched_states, rng), None, length=T_STEPS
    )

    # Bootstrap value for GAE (p0 only)
    obs_p0_last = vmap_encode(final_states, 0)
    _, _, last_val0 = vmap_forward(params, obs_p0_last)
    last_val0 = jnp.where(final_states['done'], 0.0, last_val0)

    return buf, final_states, rng, last_val0


# ── GAE ───────────────────────────────────────────────────────────────────────

@jax.jit
def compute_gae(values, rewards, dones, last_value):
    """
    Generalised Advantage Estimation — pure JAX, JIT-compiled.

    values    : [T, N]
    rewards   : [T, N]
    dones     : [T, N] bool
    last_value: [N]    bootstrap value after the last step

    Returns advantages [T, N] and returns [T, N].

    WHY GAE?
      Raw rewards are very sparse here (+/-1 only at end).  GAE uses the
      value function as a baseline to reduce variance in the advantage
      estimate — crucial for stable learning with sparse rewards.

    WHY lax.scan?
      Replaces the Python backward loop, so this runs on GPU and can be
      fused with the rest of the training graph.
    """
    dones_f   = dones.astype(jnp.float32)
    # next_vals[t] = V(s_{t+1}), with last_value as the bootstrap at the end
    next_vals = jnp.concatenate([values[1:], last_value[None, :]], axis=0)

    def gae_step(last_adv, xs):
        val, rew, done, nxt_v = xs
        delta = rew + GAMMA * nxt_v * (1.0 - done) - val
        adv   = delta + GAMMA * GAE_LAMBDA * (1.0 - done) * last_adv
        return adv, adv

    # Scan backward: reverse inputs, scan forward, reverse outputs
    _, advs_rev = jax.lax.scan(
        gae_step,
        jnp.zeros_like(last_value),
        (values[::-1], rewards[::-1], dones_f[::-1], next_vals[::-1]),
    )
    advs    = advs_rev[::-1]
    returns = advs + values
    return advs, returns


# ── PPO loss (JIT-able) ───────────────────────────────────────────────────────

@jax.jit
def _ppo_loss(params, obs, frac_bins, target_idxs, will_launch,
              old_log_probs, advantages, returns, ent_coef_frac, ent_coef_tgt):
    """
    PPO loss for one minibatch.

    WHY THREE LOSS TERMS?
      policy_loss  – make actions that led to good outcomes more likely
                     (clipped to prevent too-large updates)
      value_loss   – make the value estimate accurate
                     (needed so advantages are meaningful)
      entropy      – keep the policy from collapsing to one action
                     (critical for self-play: both players must keep exploring)
    """
    # Re-run model on stored obs with CURRENT params
    tgt_logits, frac_logits, values = jax.vmap(
        lambda o: model.apply(params, o)
    )(obs)

    # Recompute per-head log_prob and entropy under new params
    log_probs, h_frac, h_tgt = jax.vmap(log_prob_and_entropy)(
        tgt_logits, frac_logits, obs, frac_bins, target_idxs, will_launch
    )

    # Importance sampling ratio: how much did the policy change?
    ratios = jnp.exp(log_probs - old_log_probs)

    # Normalise advantages within minibatch (reduces gradient variance)
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Clipped surrogate objective
    pg1 = -adv_norm * ratios
    pg2 = -adv_norm * jnp.clip(ratios, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
    policy_loss = jnp.maximum(pg1, pg2).mean()

    # Value loss — train value head to predict raw returns directly.
    # Normalising returns per-minibatch causes a scale mismatch: the value
    # head is trained on normalised targets but used un-normalised during
    # GAE bootstrap, so the value estimates drift and ev stays near 0.
    value_loss = ((values - returns) ** 2).mean()

    # Per-head entropy bonus with annealed coefficients
    entropy = h_frac.mean() + h_tgt.mean()
    total = (policy_loss + VF_COEF * value_loss
             - ent_coef_frac * h_frac.mean()
             - ent_coef_tgt  * h_tgt.mean())

    aux = {
        'policy_loss': policy_loss,
        'value_loss':  value_loss,
        'entropy':     entropy,
        'clip_frac':   (jnp.abs(ratios - 1.0) > CLIP_EPS).astype(jnp.float32).mean(),
        'approx_kl':   ((ratios - 1.0) - jnp.log(ratios)).mean(),
    }
    return total, aux


# ── PPO update step ────────────────────────────────────────────────────────────

def ppo_update(params, opt_state, buf, adv_p0, ret_p0, rng, update_idx,
               ent_coef_frac, ent_coef_tgt):
    """
    Run N_PPO_EPOCHS of gradient descent on p0 trajectories only.

    p1 acted with frozen opp_params, so its old_log_probs are wrong under
    current params — using them would produce garbage IS ratios.

    Returns updated params, opt_state, and a dict of metrics.
    """
    # Flatten [T, N] → [T*N] for all arrays
    def flat(x):
        if isinstance(x, dict):
            return jax.tree_util.tree_map(lambda a: a.reshape(-1, *a.shape[2:]), x)
        return x.reshape(-1, *x.shape[2:])

    # p0 trajectories only
    obs        = flat(buf['obs_p0'])
    fb         = flat(buf['frac_bins_p0'])
    ti         = flat(buf['target_idxs_p0'])
    wl         = flat(buf['will_launch_p0'])
    old_lp     = flat(buf['log_probs_p0'])
    advantages = adv_p0.reshape(-1)
    returns_   = ret_p0.reshape(-1)

    n_total = advantages.shape[0]

    all_metrics = []
    stop_early = False

    for epoch in range(N_PPO_EPOCHS):
        if stop_early:
            break
        rng, perm_key = jax.random.split(rng)
        perm = jax.random.permutation(perm_key, n_total)

        for start in range(0, n_total, MINIBATCH):
            idx = perm[start: start + MINIBATCH]
            if idx.shape[0] < MINIBATCH // 2:
                continue   # skip tiny last batch

            mb_obs  = jax.tree_util.tree_map(lambda a: a[idx], obs)
            mb_fb   = fb[idx];   mb_ti  = ti[idx];  mb_wl = wl[idx]
            mb_olp  = old_lp[idx]
            mb_adv  = advantages[idx]
            mb_ret  = returns_[idx]

            (loss, aux), grads = jax.value_and_grad(_ppo_loss, has_aux=True)(
                params, mb_obs, mb_fb, mb_ti, mb_wl, mb_olp, mb_adv, mb_ret,
                ent_coef_frac, ent_coef_tgt
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            all_metrics.append(aux)

            # KL early stopping: if the policy moved too far, abort remaining epochs
            if float(aux['approx_kl']) > TARGET_KL:
                stop_early = True
                break

    # Average metrics across all gradient steps
    metrics = jax.tree_util.tree_map(
        lambda *xs: float(jnp.stack(jnp.array(list(xs))).mean()),
        *all_metrics
    )
    # Explained variance: how well does V(s) predict actual returns?
    all_ret = returns_.reshape(-1)
    all_val = flat(buf['values_p0'])
    var_ret = float(jnp.var(all_ret))
    var_res = float(jnp.var(all_ret - all_val))
    metrics['explained_var'] = max(0.0, 1.0 - var_res / (var_ret + 1e-8))

    return params, opt_state, rng, metrics


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("Initialising model...")
    rng   = jax.random.PRNGKey(SEED)

    # Need a sample obs to init model params
    rng, init_key = jax.random.split(rng)
    dummy_state   = init_state(seed=0)
    dummy_obs     = encode_obs(dummy_state, 0)
    params        = model.init(init_key, dummy_obs)
    print(f"  {count_params(params):,} parameters")

    # optimizer is defined at module level; init it here once params are known
    opt_state = optimizer.init(params)

    # ── Resume from checkpoint if one exists ──────────────────────────────────
    resume_iter   = 0
    update_idx    = 0
    total_steps   = 0
    snap_pool_ckpt = []
    existing = sorted(glob.glob(os.path.join(CKPT_DIR, 'ckpt_*.pkl')))
    if existing:
        latest = existing[-1]
        print(f"  Resuming from {latest}")
        with open(latest, 'rb') as f:
            ckpt = pickle.load(f)
        params      = ckpt['params']
        opt_state   = ckpt['opt_state']
        resume_iter = ckpt['iteration']
        update_idx  = ckpt['update_idx']
        total_steps = ckpt['total_steps']
        rng         = ckpt['rng']
        if 'snap_pool' in ckpt:
            snap_pool_ckpt = ckpt['snap_pool']
        else:
            snap_pool_ckpt = []   # old checkpoint, pool starts empty

    # Pre-initialise environments
    print(f"Initialising {N_ENVS} environments (this takes ~{N_ENVS*0.5:.0f}s)...")
    t0 = time.time()
    env_states_list = [init_state(seed=SEED + i) for i in range(N_ENVS)]
    print(f"  done in {time.time()-t0:.1f}s")

    batched_states = stack_states(env_states_list)
    reset_counter  = N_ENVS   # next seed to use for resets

    print(f"\nStarting training: {N_ENVS} envs × {T_STEPS} steps per rollout")
    print(f"  ~{N_ENVS * T_STEPS} transitions per update ({N_PPO_EPOCHS} epochs, p0 only)")
    print("-" * 60)

    t_start      = time.time()
    if resume_iter > 0 and snap_pool_ckpt:
        snap_pool = snap_pool_ckpt
        print(f"  Restored snap_pool ({len(snap_pool)} snapshots)")
    else:
        # Seed pool with random-init (or just-loaded) params so p1 has an
        # active opponent from iter 1 rather than 500 iters of pure self-play.
        snap_pool = [jax.tree_util.tree_map(lambda x: x.copy(), params)]

    for iteration in range(resume_iter + 1, N_ITERS + 1):
        rng, rollout_key = jax.random.split(rng)

        # ── Opponent: sample from pool, or use current params if pool empty ───
        if snap_pool:
            pool_idx  = int(np.random.randint(len(snap_pool)))
            opp_params = snap_pool[pool_idx]
        else:
            opp_params = params   # pure self-play until first snapshot

        # ── Collect rollout ───────────────────────────────────────────────────
        buf, batched_states, rng, last_val0 = collect_rollout(
            params, opp_params, batched_states, rollout_key
        )

        # ── GAE (p0 only) ─────────────────────────────────────────────────────
        adv_p0, ret_p0 = compute_gae(
            buf['values_p0'], buf['rewards_p0'], buf['dones'], last_val0
        )

        # ── Entropy annealing (cosine decay per head) ─────────────────────────
        t = min(iteration / N_ITERS, 1.0)
        cosine_t = 0.5 * (1.0 + math.cos(math.pi * t))
        ent_coef_frac = ENT_FRAC_END + (ENT_FRAC_START - ENT_FRAC_END) * cosine_t
        ent_coef_tgt  = ENT_TGT_END  + (ENT_TGT_START  - ENT_TGT_END)  * cosine_t

        # ── PPO update ────────────────────────────────────────────────────────
        params, opt_state, rng, metrics = ppo_update(
            params, opt_state, buf, adv_p0, ret_p0, rng, update_idx,
            ent_coef_frac, ent_coef_tgt
        )
        update_idx  += 1
        total_steps += N_ENVS * T_STEPS

        # ── Snapshot pool management ──────────────────────────────────────────
        if iteration % SNAPSHOT_EVERY == 0:
            snap_pool.append(jax.tree_util.tree_map(lambda x: x.copy(), params))
            if len(snap_pool) > POOL_SIZE:
                snap_pool.pop(0)   # drop oldest
            print(f"  [pool] snapshot added at iter {iteration} (pool size {len(snap_pool)})")

        # ── Reset envs that finished their episode ────────────────────────────
        if iteration % RESET_EVERY == 0:
            done_np = np.array(batched_states['done'])
            if done_np.any():
                states_list = unstack_states(batched_states, N_ENVS)
                for i in range(N_ENVS):
                    if done_np[i]:
                        states_list[i] = init_state(seed=reset_counter)
                        reset_counter += 1
                batched_states = stack_states(states_list)

        # ── Episode stats (from rollout buffer) ──────────────────────────────
        # Use |reward| to count actual episode endings (done flag stays True
        # across multiple steps until reset, so dones.sum() overcounts).
        rews_p0_np = np.array(buf['rewards_p0'])     # [T, N] float
        n_episodes = int((np.abs(rews_p0_np) > 0).sum())
        n_p0_wins  = int((rews_p0_np > 0).sum())
        win_rate   = n_p0_wins / n_episodes if n_episodes > 0 else float('nan')

        # ── Logging ───────────────────────────────────────────────────────────
        if iteration % EVAL_EVERY == 0:
            elapsed  = time.time() - t_start
            sps      = total_steps / elapsed
            lr_now   = float(lr_schedule(update_idx))
            print(
                f"iter {iteration:5d} | "
                f"steps {total_steps/1e6:.2f}M | "
                f"sps {sps:6.0f} | "
                f"lr {lr_now:.2e} | "
                f"pol {metrics['policy_loss']:+.4f} | "
                f"val {metrics['value_loss']:.4f} | "
                f"ent {metrics['entropy']:.3f} | "
                f"clip {metrics['clip_frac']:.3f} | "
                f"kl {metrics['approx_kl']:.4f} | "
                f"ev {metrics['explained_var']:.3f} | "
                f"eps {n_episodes:3d} | "
                f"wr {win_rate:.2f}"
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if iteration % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CKPT_DIR, f"ckpt_{iteration:05d}.pkl")
            with open(ckpt_path, 'wb') as f:
                pickle.dump({'params': params, 'opt_state': opt_state,
                             'iteration': iteration, 'update_idx': update_idx,
                             'total_steps': total_steps, 'rng': rng,
                             'snap_pool': snap_pool}, f)
            print(f"  saved {ckpt_path}")
