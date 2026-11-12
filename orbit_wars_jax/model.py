"""
Entity Transformer policy for Orbit Wars.

ARCHITECTURE OVERVIEW
─────────────────────
Input
  - 24 planet tokens  [N_PLANET_TOKENS, PLANET_FEAT]
  - 64 fleet tokens   [N_FLEET_TOKENS,  FLEET_FEAT]

Processing
  1. Linear projection: each token type → embedding dim D (128)
  2. Add a learned "type embedding" so the model knows planet vs fleet
  3. Concatenate → sequence of 88 tokens, each of size D
  4. 3× Transformer block (self-attention + MLP with residual + LayerNorm)

Output heads (applied to planet hidden states only)
  - target_logits  [N_PLANET_TOKENS, N_PLANET_TOKENS]  which planet to attack/reinforce
  - frac_logits    [N_PLANET_TOKENS, N_FRAC_BINS]       how many ships to send
  - value          scalar                               PPO value estimate

Total params: ~600 K  (3 layers × D=128, 4 heads, 4× MLP expansion)

WHY AN ENTITY TRANSFORMER?
  Planets and fleets are "entities" — their meaning doesn't depend on position
  in a list, and their count varies.  A transformer's self-attention naturally
  handles variable-length unordered sets, which is exactly what we have.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from obs import N_PLANET_TOKENS, N_FLEET_TOKENS, PLANET_FEAT, FLEET_FEAT

# ── Hyperparameters ───────────────────────────────────────────────────────────
D_MODEL    = 128   # embedding dimension throughout
N_HEADS    = 4     # attention heads (each has dimension D_MODEL // N_HEADS = 32)
N_LAYERS   = 3     # transformer depth
MLP_SCALE  = 4     # MLP hidden dim = D_MODEL * MLP_SCALE = 512

# Action space bins: 4 send fractions + 1 noop (hold ships).
# Noop is intentionally last so we can bias against it at init time.
N_FRAC_BINS  = 5
FRAC_VALUES  = np.array([0.25, 0.5, 0.75, 1.0, 0.0], dtype=np.float32)
NO_LAUNCH_BIN = 4

SEQ_LEN = N_PLANET_TOKENS + N_FLEET_TOKENS  # 88 total tokens


# ── Building blocks ───────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One transformer layer:
      x → LayerNorm → self-attention → residual
        → LayerNorm → MLP            → residual

    Using pre-norm (norm before the sub-layer) which trains more stably
    than post-norm, especially at the start of training.
    """
    d_model: int
    n_heads: int
    mlp_scale: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, attn_bias: jnp.ndarray) -> jnp.ndarray:
        """
        x         : [seq_len, d_model]
        attn_bias  : [1, 1, seq_len, seq_len]  additive bias (−1e9 for padded positions)
        returns    : [seq_len, d_model]
        """
        # ── Self-attention branch ─────────────────────────────────────────────
        h = nn.LayerNorm()(x)
        # MultiHeadDotProductAttention expects [batch, seq, features] but we
        # work without an explicit batch dim here — add/remove it around the call.
        h = h[None]                      # [1, seq_len, d]
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,   # total QKV projection size
            out_features=self.d_model,
        )(h, mask=attn_bias)             # mask shape [1, 1, seq_len, seq_len]
        h = h[0]                         # back to [seq_len, d]
        x = x + h

        # ── MLP branch ────────────────────────────────────────────────────────
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.d_model * self.mlp_scale)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model)(h)
        x = x + h

        return x


# ── Main model ────────────────────────────────────────────────────────────────

class OrbitWarsModel(nn.Module):
    """
    Full policy + value network.

    Call it with an obs dict (from encode_obs) and get back:
      target_logits : [N_PLANET_TOKENS, N_PLANET_TOKENS]
      frac_logits   : [N_PLANET_TOKENS, N_FRAC_BINS]
      value         : scalar
    """
    d_model:   int = D_MODEL
    n_heads:   int = N_HEADS
    n_layers:  int = N_LAYERS
    mlp_scale: int = MLP_SCALE

    @nn.compact
    def __call__(self, obs: dict) -> tuple:
        planet_tokens = obs['planet_tokens']   # [N_P, PLANET_FEAT]
        fleet_tokens  = obs['fleet_tokens']    # [N_F, FLEET_FEAT]
        planet_mask   = obs['planet_mask']     # [N_P] bool
        fleet_mask    = obs['fleet_mask']      # [N_F] bool

        # ── 1. Project both token types to the same embedding dim ─────────────
        p = nn.Dense(self.d_model, name='planet_proj')(planet_tokens)  # [N_P, D]
        f = nn.Dense(self.d_model, name='fleet_proj') (fleet_tokens)   # [N_F, D]

        # ── 2. Add learned type embeddings ────────────────────────────────────
        # Without this, the model can't tell planets from fleets once they're
        # projected to the same dimension.
        planet_type = self.param(
            'planet_type_emb', nn.initializers.normal(0.02), (self.d_model,)
        )
        fleet_type = self.param(
            'fleet_type_emb', nn.initializers.normal(0.02), (self.d_model,)
        )
        p = p + planet_type[None, :]   # broadcast over N_P
        f = f + fleet_type[None, :]    # broadcast over N_F

        # ── 3. Concatenate into one sequence ──────────────────────────────────
        tokens = jnp.concatenate([p, f], axis=0)   # [SEQ_LEN, D]

        # ── 4. Attention bias: large negative for padded positions ────────────
        # When attention softmax sees −1e9 for a key, it effectively ignores it.
        valid     = jnp.concatenate([planet_mask, fleet_mask], axis=0)  # [SEQ_LEN]
        attn_bias = jnp.where(
            valid[None, None, None, :],   # broadcast: [1, 1, 1, SEQ_LEN]
            0.0, -1e9
        )                                 # [1, 1, 1, SEQ_LEN]

        # ── 5. Transformer layers ─────────────────────────────────────────────
        for i in range(self.n_layers):
            tokens = TransformerBlock(
                self.d_model, self.n_heads, self.mlp_scale,
                name=f'block_{i}'
            )(tokens, attn_bias)

        tokens = nn.LayerNorm(name='final_ln')(tokens)

        # ── 6. Split out planet hidden states ─────────────────────────────────
        p_hidden = tokens[:N_PLANET_TOKENS]   # [N_P, D]

        # ── 7. Action heads ───────────────────────────────────────────────────
        # target_logits[i, j] = "how good is it to send ships from planet i to planet j"
        target_logits = nn.Dense(N_PLANET_TOKENS, name='target_head')(p_hidden)
        # [N_PLANET_TOKENS, N_PLANET_TOKENS]

        # frac_logits[i, b] = "how likely is bin b (fraction of ships) for planet i"
        # Bias noop bin (index NO_LAUNCH_BIN=4) to -2.0 at init so the policy
        # starts preferring launches; it can still learn to hold back when smart.
        frac_bias_init = np.zeros(N_FRAC_BINS, dtype=np.float32)
        frac_bias_init[NO_LAUNCH_BIN] = -2.0
        frac_logits = nn.Dense(
            N_FRAC_BINS, name='frac_head',
            bias_init=nn.initializers.constant(frac_bias_init),
        )(p_hidden)
        # [N_PLANET_TOKENS, N_FRAC_BINS]

        # ── 8. Value head (mean-pool all valid tokens → scalar) ───────────────
        valid_f     = valid.astype(jnp.float32)
        pooled      = (tokens * valid_f[:, None]).sum(axis=0) / jnp.maximum(valid_f.sum(), 1.0)
        value       = nn.Dense(1, name='value_head')(pooled).squeeze(-1)   # scalar

        return target_logits, frac_logits, value


# ── Utility: count parameters ─────────────────────────────────────────────────

def count_params(params) -> int:
    """Return total number of scalar parameters."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
