"""
Forecast ledger for Orbit Wars observations.

WHY THIS EXISTS
───────────────
Our reward is sparse (+/-1 at the end of the game) and the horizon is long
(~500 steps).  Asking a transformer to learn, purely from that signal, that a
fleet launched now will arrive at a given planet in 4 turns is brutally hard:
the credit-assignment path is long and the gradient is tiny.

The 110th-place "Distributed Orbit Wars" silver solution side-stepped this by
feeding the policy a *forecast ledger* — a short look-ahead, computed only from
currently-visible state (no cheating with hidden future comets), that tells each
planet token: "this many friendly / enemy ships are already on their way to you,
and the first enemy arrives in N turns."

This converts a hard long-horizon problem into a near-shallow one.  The network
no longer has to imagine fleet trajectories; it reads them.

WHAT IT COMPUTES
────────────────
Given the current state, we march every *currently-launched* fleet forward
`horizon` steps using the exact same dynamics as `env.step_env`:

  * fleet speed   = 1 + (MAX_SPD-1) * (log(ships)/log(1000))**1.5, capped
  * fleet travel  = straight line along its fixed launch angle
  * planet motion = orbiting planets rotate (init_angle + av * global_step),
                    static planets stay put
  * arrival       = swept segment-vs-moving-circle collision (env's own helper)

For each planet slot and each future step we accumulate incoming ships split by
owner.  We do NOT simulate combat / ownership flips forward — the ledger reports
*incoming pressure*, which is exactly the channel that did the heavy lifting in
the reference solution.  Forward combat resolution is a possible extension.

OUTPUT
──────
  forecast_ledger(state, player_id, horizon) -> [MAX_PLANETS, horizon, LEDGER_FEAT]

  channel 0: friendly incoming ships this step      (log-normalised)
  channel 1: enemy incoming ships this step          (log-normalised)
  channel 2: cumulative friendly incoming <= step    (log-normalised)
  channel 3: cumulative enemy incoming <= step        (log-normalised)

Everything is player-relative: "friendly" == player_id, "enemy" == the other.
The result is aligned to env planet slots, so obs.py can gather it with the same
`planet_slots` indices it already uses for `planet_tokens`.
"""

import jax
import jax.numpy as jnp

from env import (
    MAX_PLANETS, BOARD, CENTER, SUN_R, MAX_SPD,
    PX, PY, PR, FX, FY, FA, FO, FS,
    swept_pair_hit, seg_pt_dist_sq,
)

HORIZON     = 16     # look-ahead depth (matches the reference ledger)
LEDGER_FEAT = 4      # channels per (planet, horizon-step) — see module docstring
MAX_SHIPS   = 200.0  # same normalisation constant obs.py uses


def _fleet_speed(ships: jnp.ndarray) -> jnp.ndarray:
    """Per-step travel distance of a fleet — identical to env.step_env Phase 5."""
    s = 1.0 + (MAX_SPD - 1.0) * (
        jnp.log(jnp.maximum(ships, 1.0)) / jnp.log(1000.0)
    ) ** 1.5
    return jnp.minimum(s, MAX_SPD)


def _planet_xy(state: dict, global_step: jnp.ndarray):
    """Planet positions at an absolute env step (orbiting planets rotate)."""
    planets = state['planets']
    ang = state['init_angle'] + state['angular_vel'] * global_step.astype(jnp.float32)
    ox  = CENTER + state['orb_r'] * jnp.cos(ang)
    oy  = CENTER + state['orb_r'] * jnp.sin(ang)
    x = jnp.where(state['is_orbiting'], ox, planets[:, PX])
    y = jnp.where(state['is_orbiting'], oy, planets[:, PY])
    return x, y


def _lognorm(x: jnp.ndarray) -> jnp.ndarray:
    """Squash a ship count into [0, 1] the same way obs.py squashes ship counts."""
    return jnp.clip(jnp.log1p(x) / jnp.log1p(MAX_SHIPS), 0.0, 1.0)


def forecast_incoming(state: dict, horizon: int = HORIZON) -> jnp.ndarray:
    """
    Per-owner incoming ships per planet per future step.

    Returns [MAX_PLANETS, horizon, N_PLAYERS] where the last axis is owner 0 / 1.
    Owner-relative framing is applied later by `forecast_ledger`.
    """
    planets = state['planets']
    pv      = state['planet_valid']
    pr      = planets[:, PR]
    step0   = state['step']

    fleets = state['fleets']
    fx, fy = fleets[:, FX], fleets[:, FY]
    fa     = fleets[:, FA]
    fo     = fleets[:, FO]
    fs     = fleets[:, FS]
    speed  = _fleet_speed(fs)
    alive  = state['fleet_valid']

    incoming = jnp.zeros((MAX_PLANETS, horizon, 2), dtype=jnp.float32)

    # Vectorised swept collision of one fleet against every planet (env's pattern).
    def fleet_vs_all(fox, foy, fnx, fny, pox, poy, pnx, pny):
        def vs(po_x, po_y, pn_x, pn_y, prad, pvalid):
            return swept_pair_hit(fox, foy, fnx, fny,
                                  po_x, po_y, pn_x, pn_y, prad) & pvalid
        return jax.vmap(vs)(pox, poy, pnx, pny, pr, pv)   # [MAX_PLANETS]

    def step(carry, t):
        fx, fy, alive, incoming = carry

        # Absolute env steps for the swept interval [old -> new].
        g_old = step0 + t          # planet position the fleet starts this step from
        g_new = step0 + t + 1
        pox, poy = _planet_xy(state, g_old)
        pnx, pny = _planet_xy(state, g_new)

        fnx = fx + jnp.cos(fa) * speed
        fny = fy + jnp.sin(fa) * speed

        hits = jax.vmap(fleet_vs_all,
                        in_axes=(0, 0, 0, 0, None, None, None, None))(
            fx, fy, fnx, fny, pox, poy, pnx, pny)        # [F, P]
        hits = hits & alive[:, None]

        first_hit = jnp.argmin(
            jnp.where(hits, jnp.arange(MAX_PLANETS)[None, :], MAX_PLANETS), axis=1)
        any_hit = jnp.any(hits, axis=1)                  # [F]
        first_hit_mask = (jnp.arange(MAX_PLANETS)[None, :] == first_hit[:, None]) \
            & any_hit[:, None]                           # [F, P]

        # Scatter arriving ships into this step, split by owner.
        for pl in (0, 1):
            owner_ships = jnp.where(fo == float(pl), fs, 0.0)            # [F]
            contrib = (first_hit_mask * owner_ships[:, None]).sum(axis=0)  # [P]
            incoming = incoming.at[:, t, pl].add(contrib)

        # Fleets that hit a planet, leave the board, or hit the sun are spent.
        oob = ~((0.0 <= fnx) & (fnx <= BOARD) & (0.0 <= fny) & (fny <= BOARD))
        sun = seg_pt_dist_sq(CENTER, CENTER, fx, fy, fnx, fny) < SUN_R ** 2
        alive = alive & ~(any_hit | oob | sun)

        fx = jnp.where(alive, fnx, fx)
        fy = jnp.where(alive, fny, fy)
        return (fx, fy, alive, incoming), None

    (_, _, _, incoming), _ = jax.lax.scan(
        step, (fx, fy, alive, incoming), jnp.arange(horizon))
    return incoming


def forecast_ledger(state: dict, player_id: int, horizon: int = HORIZON) -> jnp.ndarray:
    """
    Player-relative forecast ledger.  [MAX_PLANETS, horizon, LEDGER_FEAT].

    obs.py gathers this with the same `planet_slots` it uses for planet_tokens.
    """
    incoming = forecast_incoming(state, horizon)   # [P, H, 2] by absolute owner
    me, opp = int(player_id), 1 - int(player_id)

    friendly = incoming[:, :, me]
    enemy    = incoming[:, :, opp]
    cum_friendly = jnp.cumsum(friendly, axis=1)
    cum_enemy    = jnp.cumsum(enemy, axis=1)

    return jnp.stack([
        _lognorm(friendly),      # 0  incoming friendly this step
        _lognorm(enemy),         # 1  incoming enemy this step
        _lognorm(cum_friendly),  # 2  friendly arrived by this step
        _lognorm(cum_enemy),     # 3  enemy arrived by this step
    ], axis=-1)                  # [MAX_PLANETS, horizon, LEDGER_FEAT]