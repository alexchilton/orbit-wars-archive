"""Transformer feature extractor for SB3 MaskablePPO.

Processes candidate edges with spatial self-attention and game history
with temporal self-attention, then combines both for rich representations.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .sb3_constants import (
    SB3_MAX_CANDIDATES, EDGE_INPUT_DIM,
    CANDIDATE_OBS_DIM, GLOBAL_DIM, TEMPORAL_STEPS, TEMPORAL_OBS_DIM,
)


class EdgeTransformerBlock(nn.Module):
    """Single transformer block with self-attention."""

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
    """Transformer feature extractor with spatial + temporal processing.

    Input obs layout: [candidate_features (96*74) | temporal_history (8*10)]

    1. Spatial: Edge encoder → N transformer blocks over 96 candidates
    2. Temporal: Small transformer over last 8 global state snapshots
    3. Combine: Temporal context concatenated to each candidate embedding
       via a projection layer, then added back to candidate embeddings.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        features_dim: int = 128,
        num_candidates: int = SB3_MAX_CANDIDATES,
        edge_dim: int = EDGE_INPUT_DIM,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        pool: bool = True,
        temporal_d_model: int = 64,
        temporal_n_heads: int = 2,
        temporal_n_layers: int = 2,
    ):
        actual_features_dim = features_dim if pool else num_candidates * d_model
        super().__init__(observation_space, actual_features_dim)
        self.num_candidates = num_candidates
        self.edge_dim = edge_dim
        self.d_model = d_model
        self.pool = pool

        # --- Spatial: candidate edge processing ---
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList([
            EdgeTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # --- Temporal: game history processing ---
        self.temporal_encoder = nn.Sequential(
            nn.Linear(GLOBAL_DIM, temporal_d_model),
            nn.ReLU(),
            nn.Linear(temporal_d_model, temporal_d_model),
            nn.LayerNorm(temporal_d_model),
        )
        self.temporal_blocks = nn.ModuleList([
            EdgeTransformerBlock(temporal_d_model, temporal_n_heads, dropout)
            for _ in range(temporal_n_layers)
        ])

        # Project temporal context → d_model to inject into candidate embeddings
        self.temporal_proj = nn.Linear(temporal_d_model, d_model)

        # Output projection if features_dim != d_model
        if features_dim != d_model:
            self.output_proj = nn.Linear(d_model, features_dim)
        else:
            self.output_proj = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]

        # Split obs into candidates and temporal history
        cand_flat = observations[:, :CANDIDATE_OBS_DIM]
        temporal_flat = observations[:, CANDIDATE_OBS_DIM:]

        # --- Spatial processing ---
        x = cand_flat.view(B, self.num_candidates, self.edge_dim)
        pad_mask = (x.abs().sum(dim=-1) < 1e-6)
        x = x.clamp(-50.0, 50.0)
        x = self.edge_encoder(x)
        x = x.clamp(-100.0, 100.0)

        for block in self.blocks:
            x = block(x, mask=pad_mask)

        # --- Temporal processing ---
        t = temporal_flat.view(B, TEMPORAL_STEPS, GLOBAL_DIM)
        t = self.temporal_encoder(t)
        for block in self.temporal_blocks:
            t = block(t)  # No masking — all timesteps are valid
        # Pool temporal: mean over timesteps → (B, temporal_d_model)
        t_pooled = t.mean(dim=1)
        # Project to d_model and add to each candidate
        t_ctx = self.temporal_proj(t_pooled)  # (B, d_model)
        x = x + t_ctx.unsqueeze(1)  # broadcast add to all candidates

        if not self.pool:
            return x.reshape(B, -1)

        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

        if self.output_proj is not None:
            pooled = self.output_proj(pooled)

        return pooled
