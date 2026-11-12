"""
Observation encoder for Orbit Wars.

Converts the JAX env state dict into per-entity feature arrays that an
entity transformer can read.

WHY per-entity tokens?
  Instead of flattening everything into one big vector, we give the model
  one "row" (token) per planet and per fleet.  The transformer can then
  attend across all entities to decide what matters.  This is much better
  than a flat vector because the number of planets/fleets varies and their
  order is arbitrary.

Output shapes
─────────────
  planet_tokens  [N_PLANET_TOKENS, PLANET_FEAT]  – one row per planet slot
  fleet_tokens   [N_FLEET_TOKENS,  FLEET_FEAT]   – top-K fleets by ship count
  planet_mask    [N_PLANET_TOKENS]  bool  – True = real planet, False = padding
  fleet_mask     [N_FLEET_TOKENS]   bool  – True = real fleet,  False = padding
  planet_slots   [N_PLANET_TOKENS]  int32 – which env slot this token came from
                                            (-1 for padded rows)

All features are normalised to roughly [0, 1] or [-1, 1].
"""

import jax.numpy as jnp

from env import (
    MAX_PLANETS, MAX_FLEETS, BOARD, CENTER,
    PI, PO, PX, PY, PR, PS, PP,   # planet column indices
    FI, FO, FX, FY, FA, FF, FS,   # fleet column indices
)

# ── Sequence lengths (transformer input size) ─────────────────────────────────
N_PLANET_TOKENS = 40   # 32 regular planets + up to 8 comet buffer
N_FLEET_TOKENS  = 64   # keep the 64 largest fleets; ignore the tail

# ── Feature dimensions ────────────────────────────────────────────────────────
PLANET_FEAT = 12   # features per planet token (see stack below)
FLEET_FEAT  = 9    # features per fleet token

# ── Normalisation constants ───────────────────────────────────────────────────
MAX_SHIPS   = 200.0   # ships beyond this get clipped to 1.0
MAX_PROD    = 5.0
COMET_SLOT  = 44      # planet array slots 44-63 are comet slots


def encode_obs(state: dict, player_id: int) -> dict:
    """
    Encode a single game state for one player.

    Parameters
    ----------
    state     : dict returned by init_state() or step_env()
    player_id : 0 or 1

    Returns
    -------
    dict with keys:
      planet_tokens, fleet_tokens, planet_mask, fleet_mask, planet_slots
    """
    planets = state['planets']       # [MAX_PLANETS, 7]
    pv      = state['planet_valid']  # [MAX_PLANETS] bool
    fleets  = state['fleets']        # [MAX_FLEETS,  7]
    fv      = state['fleet_valid']   # [MAX_FLEETS]  bool

    me  = float(player_id)
    opp = 1.0 - me

    # ── Planet features ───────────────────────────────────────────────────────
    # Position: shift origin to board centre, scale to [-1, 1]
    px = (planets[:, PX] - CENTER) / CENTER   # [-1, 1]
    py = (planets[:, PY] - CENTER) / CENTER

    # Distance and direction from board centre (helps the model understand orbits)
    dist  = jnp.sqrt(px**2 + py**2)                              # [0, ~1.4]
    sin_a = jnp.where(dist > 0, py / jnp.maximum(dist, 1e-8), 0.0)
    cos_a = jnp.where(dist > 0, px / jnp.maximum(dist, 1e-8), 0.0)

    # Ownership — one-hot from this player's perspective
    # (the model needs to know "is this mine?" not just "who owns it?")
    is_mine    = (planets[:, PO] == me ).astype(jnp.float32)
    is_enemy   = (planets[:, PO] == opp).astype(jnp.float32)
    is_neutral = (planets[:, PO] <  0.0).astype(jnp.float32)

    # Comet flag: comet planets live in slots 44-63
    slot_idx = jnp.arange(MAX_PLANETS, dtype=jnp.float32)
    is_comet = (slot_idx >= COMET_SLOT).astype(jnp.float32)

    # Orbiting flag (orbiting planets move, static ones don't)
    is_orbit = state['is_orbiting'].astype(jnp.float32)

    ships_norm = jnp.clip(planets[:, PS] / MAX_SHIPS, 0.0, 1.0)
    prod_norm  = jnp.clip(planets[:, PP] / MAX_PROD,  0.0, 1.0)

    # Stack into feature matrix: [MAX_PLANETS, PLANET_FEAT]
    p_feats = jnp.stack([
        px,         # 0   x position (board-centred, normalised)
        py,         # 1   y position
        dist,       # 2   distance from board centre
        sin_a,      # 3   angle from centre (sin)
        cos_a,      # 4   angle from centre (cos)
        ships_norm, # 5   ship count (normalised)
        prod_norm,  # 6   production rate
        is_mine,    # 7   owned by this player
        is_enemy,   # 8   owned by opponent
        is_neutral, # 9   unowned
        is_comet,   # 10  is a comet planet
        is_orbit,   # 11  is orbiting
    ], axis=1)      # → [MAX_PLANETS, 12]

    # Select up to N_PLANET_TOKENS valid planets, in slot order
    # Trick: give invalid slots index MAX_PLANETS (sentinel), then sort ascending.
    # After sorting, valid slots (< MAX_PLANETS) come first.
    valid_indices  = jnp.where(pv, jnp.arange(MAX_PLANETS), MAX_PLANETS)
    sorted_indices = jnp.sort(valid_indices)[:N_PLANET_TOKENS]
    planet_mask    = sorted_indices < MAX_PLANETS            # True = real planet

    safe_idx      = jnp.where(planet_mask, sorted_indices, 0)   # avoid OOB reads
    planet_tokens = p_feats[safe_idx]                            # [N_PLANET_TOKENS, PLANET_FEAT]
    planet_tokens = jnp.where(planet_mask[:, None], planet_tokens, 0.0)  # zero-pad

    planet_slots  = jnp.where(planet_mask, safe_idx, -1).astype(jnp.int32)

    # ── Fleet features ────────────────────────────────────────────────────────
    fx = (fleets[:, FX] - CENTER) / CENTER
    fy = (fleets[:, FY] - CENTER) / CENTER

    f_dist  = jnp.sqrt(fx**2 + fy**2)
    f_sin_d = jnp.sin(fleets[:, FA])    # movement direction (sin)
    f_cos_d = jnp.cos(fleets[:, FA])    # movement direction (cos)

    f_is_mine  = (fleets[:, FO] == me ).astype(jnp.float32)
    f_is_enemy = (fleets[:, FO] == opp).astype(jnp.float32)

    f_ships_norm = jnp.clip(fleets[:, FS] / MAX_SHIPS, 0.0, 1.0)

    # Angle of fleet position from board centre (where on the board is it?)
    f_sin_a = jnp.where(f_dist > 0, fy / jnp.maximum(f_dist, 1e-8), 0.0)
    f_cos_a = jnp.where(f_dist > 0, fx / jnp.maximum(f_dist, 1e-8), 0.0)

    f_feats = jnp.stack([
        fx,           # 0   x position
        fy,           # 1   y position
        f_dist,       # 2   distance from board centre
        f_sin_d,      # 3   movement direction (sin)
        f_cos_d,      # 4   movement direction (cos)
        f_ships_norm, # 5   ship count
        f_is_mine,    # 6   mine
        f_is_enemy,   # 7   enemy
        f_sin_a,      # 8   position angle from centre (sin)
    ], axis=1)        # → [MAX_FLEETS, 9]

    # Select top N_FLEET_TOKENS by ship count (invalid fleets get -1 ships → excluded)
    masked_ships     = jnp.where(fv, fleets[:, FS], -1.0)
    top_fleet_slots  = jnp.argsort(-masked_ships)[:N_FLEET_TOKENS]
    fleet_mask       = masked_ships[top_fleet_slots] >= 0.0   # True = real fleet

    safe_fidx    = jnp.where(fleet_mask, top_fleet_slots, 0)
    fleet_tokens = f_feats[safe_fidx]                          # [N_FLEET_TOKENS, FLEET_FEAT]
    fleet_tokens = jnp.where(fleet_mask[:, None], fleet_tokens, 0.0)

    return {
        'planet_tokens': planet_tokens,   # [N_PLANET_TOKENS, PLANET_FEAT]
        'fleet_tokens':  fleet_tokens,    # [N_FLEET_TOKENS,  FLEET_FEAT]
        'planet_mask':   planet_mask,     # [N_PLANET_TOKENS] bool
        'fleet_mask':    fleet_mask,      # [N_FLEET_TOKENS]  bool
        'planet_slots':  planet_slots,    # [N_PLANET_TOKENS] int32
    }
