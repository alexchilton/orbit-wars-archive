"""Custom MaskableActorCriticPolicy with per-candidate pointer-net scoring.

Instead of mean-pooling candidates then using a giant Linear(128, 2405) action head,
this policy scores each candidate independently via a small MLP applied to its
embedding. Noop competes as a learned bias parameter.

Supports asymmetric feature extractors: large transformer for policy (pi),
smaller one for value (vf) — the critic doesn't need as much capacity.

Architecture:
    Pi:  Transformer(d=256,L=6) → (B, 96, 256) → Scorer(256→128→10) → (B, 961) × 5 slots
    Vf:  Transformer(d=128,L=2) → (B, 96, 128) → masked mean pool → MLP(128→256→1)
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import numpy as np
import torch
import torch as th
import torch.nn as nn
from gymnasium import spaces

from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from .sb3_constants import SB3_MAX_CANDIDATES, NUM_FRACTIONS, NUM_CHOICES, MAX_ACTIONS


class PointerNetPolicy(MaskableActorCriticPolicy):
    """Per-candidate scoring policy for MaskablePPO.

    Replaces the default Linear(latent_dim, num_actions) action head with:
    - A small MLP scorer applied to each candidate's embedding
    - A learned noop bias that competes with candidate scores
    - Shared scores tiled across all 5 action slots

    Supports separate pi/vf feature extractors via vf_features_extractor_kwargs.
    """

    def __init__(self, *args, vf_features_extractor_kwargs: dict | None = None, **kwargs):
        self._vf_features_extractor_kwargs = vf_features_extractor_kwargs
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        # Build the MLP extractor (identity — we handle everything custom)
        self._build_mlp_extractor()

        # Get pi d_model from the feature extractor
        pi_d_model = self.features_extractor.d_model

        # Per-slot independent scorer heads — each slot learns different behavior
        self.slot_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(pi_d_model, 256),
                nn.ReLU(),
                nn.Linear(256, NUM_FRACTIONS),
            )
            for _ in range(MAX_ACTIONS)
        ])
        # Per-slot noop logits — each slot has its own noop preference
        self.noop_logits = nn.Parameter(torch.zeros(MAX_ACTIONS))

        # Build vf feature extractor if asymmetric
        if self._vf_features_extractor_kwargs is not None:
            # Create a separate smaller extractor for the critic
            vf_kwargs = dict(self._vf_features_extractor_kwargs)
            vf_kwargs["pool"] = True  # Critic pools to a single vector
            self.vf_features_extractor = self.features_extractor_class(
                self.observation_space, **vf_kwargs
            )
            vf_d_model = vf_kwargs.get("d_model", 128)
        else:
            self.vf_features_extractor = None
            vf_d_model = pi_d_model

        # Value head
        self.value_mlp = nn.Sequential(
            nn.Linear(vf_d_model, 256),
            nn.Tanh(),
        )
        self.value_net = nn.Linear(256, 1)

        # Dummy action_net for SB3 internals
        self.action_net = nn.Identity()

        # Orthogonal init
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.slot_scorers: 0.01,
                self.value_mlp: np.sqrt(2),
                self.value_net: 1,
            }
            if self.vf_features_extractor is not None:
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Optimizer over all parameters
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        # Print parameter counts
        pi_params = sum(p.numel() for p in self.features_extractor.parameters())
        scorer_params = sum(p.numel() for p in self.slot_scorers.parameters()) + MAX_ACTIONS
        vf_ext_params = sum(p.numel() for p in self.vf_features_extractor.parameters()) if self.vf_features_extractor else 0
        vf_head_params = sum(p.numel() for p in self.value_mlp.parameters()) + sum(p.numel() for p in self.value_net.parameters())
        total = sum(p.numel() for p in self.parameters())
        print(f"PointerNetPolicy params: pi_extractor={pi_params:,} scorer={scorer_params:,} "
              f"vf_extractor={vf_ext_params:,} vf_head={vf_head_params:,} total={total:,} "
              f"({total * 4 / 1e6:.1f}MB)")

    def _pool_for_value(self, features: th.Tensor, d_model: int) -> th.Tensor:
        """Masked mean pool over candidates for value estimation."""
        B = features.shape[0]
        x = features.view(B, SB3_MAX_CANDIDATES, d_model)
        pad_mask = (x.abs().sum(dim=-1) < 1e-6)  # (B, K)
        valid = (~pad_mask).unsqueeze(-1).float()  # (B, K, 1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # (B, D)
        return pooled

    def _score_candidates(self, features: th.Tensor) -> th.Tensor:
        """Score each candidate independently per slot.

        Each slot has its own scorer head — can learn different roles
        (e.g., slot 1 attacks, slot 2 reinforces, slot 3 scouts).

        Returns (B, NUM_CHOICES * MAX_ACTIONS) logits for MultiDiscrete.
        """
        B = features.shape[0]
        pi_d_model = self.features_extractor.d_model
        x = features.view(B, SB3_MAX_CANDIDATES, pi_d_model)

        all_slot_logits = []
        for i, scorer in enumerate(self.slot_scorers):
            scores = scorer(x)  # (B, 96, 10)
            flat = scores.reshape(B, SB3_MAX_CANDIDATES * NUM_FRACTIONS)  # (B, 960)
            noop = self.noop_logits[i].expand(B, 1)  # (B, 1)
            slot_logits = torch.cat([flat, noop], dim=-1)  # (B, 961)
            all_slot_logits.append(slot_logits)

        return torch.cat(all_slot_logits, dim=-1)  # (B, 4805)

    def _extract_vf_features(self, obs: th.Tensor) -> th.Tensor:
        """Extract value features — uses separate vf extractor if available."""
        if self.vf_features_extractor is not None:
            return self.vf_features_extractor(obs)
        # Fallback: pool from pi features
        pi_features = self.extract_features(obs)
        return self._pool_for_value(pi_features, self.features_extractor.d_model)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        """Create action distribution from per-candidate scores."""
        logits = self._score_candidates(latent_pi)
        return self.action_dist.proba_distribution(action_logits=logits)

    def _get_pi_features(self, obs: th.Tensor) -> th.Tensor:
        """Extract pi features only (handles shared vs separate extractors)."""
        features = self.extract_features(obs)
        if isinstance(features, tuple):
            return features[0]  # pi_features from share_features_extractor=False
        return features

    def _get_vf_pooled(self, obs: th.Tensor) -> th.Tensor:
        """Extract value features (pooled)."""
        if self.vf_features_extractor is not None:
            return self.vf_features_extractor(obs)
        pi_features = self._get_pi_features(obs)
        return self._pool_for_value(pi_features, self.features_extractor.d_model)

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass — actor and critic."""
        pi_features = self._get_pi_features(obs)
        vf_pooled = self._get_vf_pooled(obs)
        values = self.value_net(self.value_mlp(vf_pooled))

        distribution = self._get_action_dist_from_latent(pi_features)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """Evaluate actions — used during PPO training updates."""
        pi_features = self._get_pi_features(obs)
        vf_pooled = self._get_vf_pooled(obs)

        distribution = self._get_action_dist_from_latent(pi_features)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(self.value_mlp(vf_pooled))

        return values, log_prob, distribution.entropy()

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """Get value estimate for observations."""
        vf_pooled = self._get_vf_pooled(obs)
        return self.value_net(self.value_mlp(vf_pooled))
