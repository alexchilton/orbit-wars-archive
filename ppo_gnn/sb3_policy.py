"""Custom MaskableActorCriticPolicy with per-candidate pointer-net scoring.

Instead of mean-pooling candidates then using a giant Linear(128, 2405) action head,
this policy scores each candidate independently via a small MLP applied to its
128-dim embedding. Noop competes as a learned bias parameter.

Architecture:
    Transformer → (B, 48, 128) → Scorer(128→64→10) per candidate → (B, 480) + noop → (B, 481)
    Value: same features → masked mean pool → MLP(128→128) → Linear(128→1)
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Union

import numpy as np
import torch
import torch as th
import torch.nn as nn
from gymnasium import spaces

from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from .sb3_constants import SB3_MAX_CANDIDATES, NUM_FRACTIONS, NUM_CHOICES, MAX_ACTIONS


class PointerNetPolicy(MaskableActorCriticPolicy):
    """Per-candidate scoring policy for MaskablePPO.

    Replaces the default Linear(latent_dim, num_actions) action head with:
    - A small MLP scorer applied to each candidate's 128-dim embedding
    - A learned noop bias that competes with candidate scores
    - Shared scores tiled across all 5 action slots
    """

    D_MODEL = 128  # Must match feature extractor d_model

    def _build(self, lr_schedule: Schedule) -> None:
        # Build the MLP extractor (identity with pi=[], vf=[])
        self._build_mlp_extractor()

        # Per-candidate scorer: scores each candidate for each fraction bucket
        self.candidate_scorer = nn.Sequential(
            nn.Linear(self.D_MODEL, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_FRACTIONS),
        )
        # Learned noop logit — competes with best candidate score
        self.noop_logit = nn.Parameter(torch.zeros(1))

        # Value head: pool then MLP
        self.value_mlp = nn.Sequential(
            nn.Linear(self.D_MODEL, 128),
            nn.Tanh(),
        )
        self.value_net = nn.Linear(128, 1)

        # We still need action_dist for SB3 internals but action_net is unused
        # Create a dummy action_net that won't be called
        self.action_net = nn.Identity()

        # Orthogonal init for our custom modules
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.candidate_scorer: 0.01,
                self.value_mlp: np.sqrt(2),
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Optimizer over all parameters
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _pool_for_value(self, features: th.Tensor) -> th.Tensor:
        """Masked mean pool over candidates for value estimation."""
        B = features.shape[0]
        x = features.view(B, SB3_MAX_CANDIDATES, self.D_MODEL)
        pad_mask = (x.abs().sum(dim=-1) < 1e-6)  # (B, K)
        valid = (~pad_mask).unsqueeze(-1).float()  # (B, K, 1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # (B, D)
        return pooled

    def _score_candidates(self, features: th.Tensor) -> th.Tensor:
        """Score each candidate with the pointer-net scorer.

        Returns (B, NUM_CHOICES * MAX_ACTIONS) logits for MultiDiscrete.
        """
        B = features.shape[0]
        x = features.view(B, SB3_MAX_CANDIDATES, self.D_MODEL)
        # Score each candidate: (B, 48, 10)
        scores = self.candidate_scorer(x)
        # Flatten to (B, 480)
        flat = scores.reshape(B, SB3_MAX_CANDIDATES * NUM_FRACTIONS)
        # Append noop: (B, 481)
        noop = self.noop_logit.expand(B, 1)
        slot_logits = torch.cat([flat, noop], dim=-1)
        # Tile for all 5 slots: (B, 2405)
        return slot_logits.repeat(1, MAX_ACTIONS)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        """Create action distribution from per-candidate scores."""
        logits = self._score_candidates(latent_pi)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass — actor and critic."""
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Value from pooled features
        pooled = self._pool_for_value(latent_vf)
        values = self.value_net(self.value_mlp(pooled))

        # Action distribution from per-candidate scores
        distribution = self._get_action_dist_from_latent(latent_pi)
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
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)

        # Value from pooled features
        pooled = self._pool_for_value(latent_vf)
        values = self.value_net(self.value_mlp(pooled))

        return values, log_prob, distribution.entropy()

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """Get value estimate for observations."""
        features = self.extract_features(obs)
        if self.share_features_extractor:
            _, latent_vf = self.mlp_extractor(features)
        else:
            _, vf_features = features
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        pooled = self._pool_for_value(latent_vf)
        return self.value_net(self.value_mlp(pooled))
