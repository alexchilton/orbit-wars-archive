"""
Action decoder for Orbit Wars.

Converts raw model outputs (logits) into the [MAX_LAUNCHES, 3] action array
that step_env() expects:  each row = [from_planet_id, angle_rad, n_ships]
                          no-op rows have from_planet_id = -1

There are two modes:
  greedy  – argmax everywhere    (for evaluation / submission)
  sample  – categorical sample   (for PPO training, enables exploration)

PPO also needs:
  log_prob   – sum of log P(action) for launched planets (scalar)
  entropy    – policy entropy for the entropy bonus term
  frac_bins, target_idxs, will_launch  – stored so PPO can recompute log_prob
                                         with new params during the update
"""

import jax
import jax.numpy as jnp

from env import MAX_LAUNCHES, PI, PS
from obs import N_PLANET_TOKENS, CENTER
from model import FRAC_VALUES, NO_LAUNCH_BIN, N_FRAC_BINS

_FRAC_JAX = jnp.array(FRAC_VALUES)   # [N_FRAC_BINS]


# ── Greedy (eval / submission) ─────────────────────────────────────────────────

def greedy_actions(target_logits, frac_logits, obs, state):
    """Argmax action selection. No randomness. Returns [MAX_LAUNCHES, 3]."""
    planet_mask  = obs['planet_mask']
    is_mine      = obs['planet_tokens'][:, 7].astype(bool)
    active       = is_mine & planet_mask

    masked_tgt  = jnp.where(planet_mask[None, :], target_logits, -1e9)
    frac_bins   = jnp.argmax(frac_logits, axis=1)
    target_idxs = jnp.argmax(masked_tgt,  axis=1)
    will_launch  = active & (frac_bins != NO_LAUNCH_BIN)

    return _assemble(frac_bins, target_idxs, will_launch, obs, state)


# ── Stochastic (training) ──────────────────────────────────────────────────────

def sample_actions(target_logits, frac_logits, obs, state, rng):
    """
    Categorical-sample actions for training.

    Returns
    -------
    actions      [MAX_LAUNCHES, 3]
    log_prob     scalar  — sum of log P(action) over launched planets
    frac_bins    [N_PLANET_TOKENS] int32  — stored for PPO recompute
    target_idxs  [N_PLANET_TOKENS] int32
    will_launch  [N_PLANET_TOKENS] bool
    """
    planet_mask  = obs['planet_mask']
    is_mine      = obs['planet_tokens'][:, 7].astype(bool)
    active       = is_mine & planet_mask

    rng, rng_f, rng_t = jax.random.split(rng, 3)

    # Sample fraction bin for each planet
    frac_bins   = jax.random.categorical(rng_f, frac_logits)        # [N_P]
    will_launch = active & (frac_bins != NO_LAUNCH_BIN)

    # Sample target (mask padding before sampling)
    masked_tgt  = jnp.where(planet_mask[None, :], target_logits, -1e9)
    target_idxs = jax.random.categorical(rng_t, masked_tgt)         # [N_P]

    # Log-prob for PPO
    log_prob = _log_prob(frac_logits, masked_tgt, frac_bins, target_idxs, will_launch)

    actions = _assemble(frac_bins, target_idxs, will_launch, obs, state)

    return actions, log_prob, frac_bins, target_idxs, will_launch


# ── Log-prob + entropy (called during PPO update with new params) ─────────────

def log_prob_and_entropy(target_logits, frac_logits, obs, frac_bins, target_idxs, will_launch):
    """
    Recompute log_prob and per-head policy entropy for stored (frac_bins, target_idxs)
    under *current* logits.  Called in the PPO gradient computation.

    Returns
    -------
    log_prob      scalar
    h_frac_total  scalar — fraction-head entropy summed over active planets
    h_tgt_total   scalar — target-head entropy summed over launching planets
    """
    planet_mask = obs['planet_mask']
    is_mine     = obs['planet_tokens'][:, 7].astype(bool)
    active      = is_mine & planet_mask

    masked_tgt = jnp.where(planet_mask[None, :], target_logits, -1e9)

    lp = _log_prob(frac_logits, masked_tgt, frac_bins, target_idxs, will_launch)

    # Per-head entropy
    frac_log_p   = jax.nn.log_softmax(frac_logits, axis=-1)
    target_log_p = jax.nn.log_softmax(masked_tgt,  axis=-1)

    h_frac   = -(jax.nn.softmax(frac_logits, axis=-1) * frac_log_p).sum(-1)   # [N_P]
    h_target = -(jax.nn.softmax(masked_tgt,  axis=-1) * target_log_p).sum(-1) # [N_P]

    h_frac_total = jnp.where(active,                     h_frac,   0.0).sum()
    h_tgt_total  = jnp.where(active & will_launch,       h_target, 0.0).sum()

    return lp, h_frac_total, h_tgt_total


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _log_prob(frac_logits, masked_tgt_logits, frac_bins, target_idxs, will_launch):
    """Sum log P(frac_bin) + log P(target_idx) over launching planets."""
    frac_lp   = jax.nn.log_softmax(frac_logits,        axis=-1)  # [N_P, N_FRAC]
    target_lp = jax.nn.log_softmax(masked_tgt_logits,  axis=-1)  # [N_P, N_P]

    idx = jnp.arange(N_PLANET_TOKENS)
    chosen_f_lp = frac_lp  [idx, frac_bins]    # [N_P]
    chosen_t_lp = target_lp[idx, target_idxs]  # [N_P]

    return jnp.where(will_launch, chosen_f_lp + chosen_t_lp, 0.0).sum()


def _assemble(frac_bins, target_idxs, will_launch, obs, state):
    """Pack discrete choices into [MAX_LAUNCHES, 3] for step_env."""
    planet_tokens = obs['planet_tokens']
    planet_slots  = obs['planet_slots']

    # Angle toward target planet (positions are normalised, direction is preserved)
    from_x = planet_tokens[:, 0];  from_y = planet_tokens[:, 1]
    tgt_x  = planet_tokens[target_idxs, 0]
    tgt_y  = planet_tokens[target_idxs, 1]
    angles = jnp.arctan2(tgt_y - from_y, tgt_x - from_x)

    safe_slots = jnp.where(obs['planet_mask'], planet_slots, 0)
    curr_ships  = state['planets'][safe_slots, PS]
    frac_vals   = _FRAC_JAX[frac_bins]
    n_ships     = jnp.maximum(jnp.floor(curr_ships * frac_vals).astype(jnp.int32), 1)

    from_ids = state['planets'][safe_slots, PI].astype(jnp.float32)

    rows = jnp.stack([
        jnp.where(will_launch, from_ids,                    -1.0),
        jnp.where(will_launch, angles,                       0.0),
        jnp.where(will_launch, n_ships.astype(jnp.float32), 0.0),
    ], axis=1)   # [N_PLANET_TOKENS, 3]

    n_take   = min(N_PLANET_TOKENS, MAX_LAUNCHES)
    pad      = jnp.full((MAX_LAUNCHES, 3), -1.0, dtype=jnp.float32)
    return pad.at[:n_take].set(rows[:n_take])
