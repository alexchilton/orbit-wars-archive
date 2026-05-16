"""Transformer feature extractor for SB3 MaskablePPO.

Ports the EdgeTransformerBlock architecture from edge_policy.py into
SB3's BaseFeaturesExtractor interface. Self-attention over candidate
edges allows the model to compare candidates against each other.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .sb3_constants import SB3_MAX_CANDIDATES, EDGE_INPUT_DIM


class EdgeTransformerBlock(nn.Module):
    """Single transformer block with self-attention over candidate edges.

    Copied from edge_policy.py:629-670 — proven architecture.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, K, d_model) — K candidate edges
            mask: (B, K) — True for PADDED positions (to be ignored)
        """
        if mask is not None:
            B, K_seq = mask.shape
            n_heads = self.attn.num_heads
            float_mask = torch.zeros(B, K_seq, K_seq, device=x.device, dtype=x.dtype)
            pad_cols = mask.unsqueeze(1).expand(-1, K_seq, -1)
            float_mask = float_mask.masked_fill(pad_cols, -1e4)
            float_mask = float_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
            float_mask = float_mask.reshape(B * n_heads, K_seq, K_seq)
            attn_out, _ = self.attn(x, x, x, attn_mask=float_mask, need_weights=False)
        else:
            attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class CandidateTransformerExtractor(BaseFeaturesExtractor):
    """Transformer feature extractor over candidate edges for SB3.

    Takes flat (B, 3552) obs, reshapes to (B, 48, 74), applies:
    1. Edge encoder: Linear(74→128) + ReLU + Linear(128→128) + LayerNorm
    2. 3x EdgeTransformerBlock with self-attention
    3. Returns flat per-candidate features (B, 48*128 = 6144)

    The per-candidate features preserve identity so the action head can
    score candidates individually (pointer-net style).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        features_dim: int = 6144,
        num_candidates: int = SB3_MAX_CANDIDATES,
        edge_dim: int = EDGE_INPUT_DIM,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)
        self.num_candidates = num_candidates
        self.edge_dim = edge_dim
        self.d_model = d_model

        # Edge encoder: project raw features to d_model
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            EdgeTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]

        # Reshape flat obs to (B, num_candidates, edge_dim)
        x = observations.view(B, self.num_candidates, self.edge_dim)

        # Detect padded candidates (all-zero rows)
        pad_mask = (x.abs().sum(dim=-1) < 1e-6)  # (B, K) True for padded

        # Clamp features
        x = x.clamp(-50.0, 50.0)

        # Edge encoder
        x = self.edge_encoder(x)  # (B, K, d_model)
        x = x.clamp(-100.0, 100.0)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask=pad_mask)

        # Return flat per-candidate features (B, K * d_model)
        return x.reshape(B, -1)
