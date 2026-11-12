"""Behavioral Cloning for the SB3 MaskablePPO CandidateTransformer policy.

Parses top-10 Kaggle replays into the SB3 observation format (48 candidates x 74 features)
and trains the policy to predict flat MultiDiscrete([481]*5) actions via cross-entropy.

Action encoding: each of 5 slots picks from 480 candidate*fraction combos + 1 noop (index 480).
  action_index = cand_idx * NUM_FRACTIONS + frac_idx  (0..479)
  action_index = 480  (noop)

Usage:
    python -m ppo_gnn.train_bc_sb3 --replay-dir kaggle_replays/episodes --epochs 30
    python -m ppo_gnn.train_bc_sb3 --replay-dir kaggle_replays/episodes --max-step 100 --epochs 50
    python -m ppo_gnn.train_bc_sb3 --skip-parse --device mps
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from .edge_policy import (
    EDGE_INPUT_DIM,
    FRACTION_BUCKETS,
    NUM_FRACTIONS,
    compute_candidate_edges,
)
from .replay_parser import _angle_to_target
from .sb3_constants import (
    SB3_MAX_CANDIDATES,
    NUM_CHOICES,
    NOOP_ACTION,
    MAX_ACTIONS,
    OBS_DIM,
)


# ---------------------------------------------------------------------------
# Replay parsing — maps expert actions to SB3 flat action indices
# ---------------------------------------------------------------------------


def _fraction_to_bucket(ships_sent: float, ships_available: float) -> int:
    """Snap ship fraction to nearest bucket in FRACTION_BUCKETS (10 bins)."""
    if ships_available < 1:
        return 0
    frac = ships_sent / ships_available
    best_idx = 0
    best_diff = abs(frac - FRACTION_BUCKETS[0])
    for i, b in enumerate(FRACTION_BUCKETS[1:], 1):
        d = abs(frac - b)
        if d < best_diff:
            best_diff = d
            best_idx = i
    return best_idx


def parse_replay_sb3(
    filepath: str,
    winners_only: bool = True,
    max_step: int = 0,
) -> List[dict]:
    """Parse a single replay into SB3-format training samples.

    Each sample is one game step for one player:
      - obs: flat (3552,) float32 — 48 candidates x 74 features
      - actions: (5,) int64 — one action index per slot (0..480)
      - mask: (2405,) bool — valid action mask (tiled single_mask)
      - action_count: int — how many real actions (non-noop slots)

    Returns list of sample dicts.
    """
    with open(filepath) as f:
        data = json.load(f)

    if "steps" not in data:
        return []

    steps = data["steps"]
    num_players = len(steps[0])
    rewards = data.get("rewards", [0] * num_players)

    # Determine winners
    if winners_only:
        max_reward = max(r for r in rewards if r is not None)
        winner_set = {i for i, r in enumerate(rewards) if r == max_reward}
    else:
        winner_set = set(range(num_players))

    samples = []
    max_game_step = max_step if max_step > 0 else len(steps)

    for step_idx, step in enumerate(steps):
        if step_idx >= max_game_step:
            break

        for player_id in winner_set:
            agent_data = step[player_id]
            obs = agent_data.get("observation")
            action = agent_data.get("action", [])

            if obs is None or "planets" not in obs:
                continue

            planets = obs["planets"]
            fleets = obs.get("fleets", [])
            angular_velocity = obs.get("angular_velocity", 0.0)

            if not planets:
                continue

            # Check alive
            if not any(int(p[1]) == player_id for p in planets):
                continue

            # Compute candidate edges (SB3 format: 48 candidates, no heuristic filtering)
            edge_features, edge_indices, edge_mask, num_valid = compute_candidate_edges(
                planets=planets,
                fleets=fleets,
                player_id=player_id,
                num_players=num_players,
                step=step_idx,
                max_steps=len(steps),
                max_candidates=SB3_MAX_CANDIDATES,
                angular_velocity=angular_velocity,
                no_filter=True,
            )

            # Flatten observation (matches SB3 env)
            obs_flat = edge_features.numpy().flatten().astype(np.float32)

            # Parse expert actions and map to SB3 action indices
            slot_actions = [NOOP_ACTION] * MAX_ACTIONS  # default all noop
            action_count = 0

            if action and num_valid > 0:
                pid_to_idx = {int(p[0]): i for i, p in enumerate(planets)}

                for act in action:
                    if action_count >= MAX_ACTIONS:
                        break
                    if not isinstance(act, (list, tuple)) or len(act) < 3:
                        continue

                    src_planet_id = int(act[0])
                    angle = float(act[1])
                    ships_sent = float(act[2])

                    if src_planet_id not in pid_to_idx:
                        continue

                    src_idx = pid_to_idx[src_planet_id]
                    src_planet = planets[src_idx]

                    # Must own the source
                    if int(src_planet[1]) != player_id:
                        continue

                    src_x, src_y = float(src_planet[2]), float(src_planet[3])
                    src_ships = float(src_planet[5])

                    # Find target planet by angle
                    tgt_idx = _angle_to_target(src_x, src_y, angle, planets, src_planet_id)
                    if tgt_idx is None:
                        continue

                    # Find this (src, tgt) in candidate edges
                    matched_candidate = -1
                    for k in range(min(num_valid, SB3_MAX_CANDIDATES)):
                        if (edge_indices[k, 0].item() == src_idx and
                                edge_indices[k, 1].item() == tgt_idx):
                            matched_candidate = k
                            break

                    if matched_candidate < 0:
                        # Expert chose an edge not in top-K candidates — skip
                        continue

                    frac_idx = _fraction_to_bucket(ships_sent, src_ships)
                    action_idx = matched_candidate * NUM_FRACTIONS + frac_idx
                    slot_actions[action_count] = action_idx
                    action_count += 1

            samples.append({
                "obs": obs_flat,
                "actions": np.array(slot_actions, dtype=np.int64),
                "action_count": action_count,
            })

    return samples


def parse_and_prebatch_sb3(
    replay_dir: str,
    output_path: str,
    winners_only: bool = True,
    max_step: int = 0,
    max_files: int = 0,
    chunk_size: int = 200,
) -> None:
    """Parse all replays in chunks and save pre-batched tensors."""
    files = []
    for root, _dirs, fnames in os.walk(replay_dir):
        for fname in fnames:
            if fname.endswith(".json"):
                files.append(os.path.join(root, fname))

    if max_files > 0:
        files = files[:max_files]

    print(f"Found {len(files)} replay files, processing in chunks of {chunk_size}")
    if max_step > 0:
        print(f"  Filtering to first {max_step} steps only")

    chunk_paths = []
    total_samples = 0
    total_errors = 0
    total_matched_actions = 0
    total_missed_actions = 0

    for chunk_start in range(0, len(files), chunk_size):
        chunk_files = files[chunk_start:chunk_start + chunk_size]
        chunk_idx = chunk_start // chunk_size
        all_samples = []
        errors = 0

        for filepath in chunk_files:
            try:
                samples = parse_replay_sb3(
                    filepath,
                    winners_only=winners_only,
                    max_step=max_step,
                )
                all_samples.extend(samples)
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  Error: {e}")

        total_errors += errors

        if not all_samples:
            print(f"  Chunk {chunk_idx}: 0 samples (skipped)")
            continue

        N = len(all_samples)
        obs_t = torch.zeros(N, OBS_DIM, dtype=torch.float16)
        actions_t = torch.zeros(N, MAX_ACTIONS, dtype=torch.long)
        action_counts_t = torch.zeros(N, dtype=torch.long)

        for i, s in enumerate(all_samples):
            obs_t[i] = torch.from_numpy(s["obs"]).half()
            actions_t[i] = torch.from_numpy(s["actions"])
            action_counts_t[i] = s["action_count"]
            total_matched_actions += s["action_count"]

        chunk_path = output_path + f".chunk{chunk_idx}.tmp"
        torch.save({
            "obs": obs_t,
            "actions": actions_t,
            "action_counts": action_counts_t,
        }, chunk_path)

        total_samples += N
        chunk_paths.append(chunk_path)
        print(f"  Chunk {chunk_idx}: {len(chunk_files)} files -> {N} samples "
              f"(total: {total_samples})", flush=True)

        del all_samples, obs_t, actions_t, action_counts_t
        gc.collect()

    # Concatenate all chunks
    print(f"\nConcatenating {len(chunk_paths)} chunks ({total_samples} total)...")
    all_data = {"obs": [], "actions": [], "action_counts": []}
    for cp in chunk_paths:
        chunk = torch.load(cp, weights_only=True)
        for k in all_data:
            all_data[k].append(chunk[k])
        del chunk
        os.remove(cp)
        gc.collect()

    merged = {k: torch.cat(v, dim=0) for k, v in all_data.items()}
    del all_data
    gc.collect()

    torch.save(merged, output_path)
    file_size = os.path.getsize(output_path) / 1e6
    print(f"Saved {total_samples} samples to {output_path} ({file_size:.1f} MB)")
    print(f"  {total_errors} parse errors")
    print(f"  {total_matched_actions} expert actions matched to top-48 candidates")

    # Action count distribution
    counts = merged["action_counts"]
    for a in range(MAX_ACTIONS + 1):
        c = (counts == a).sum().item()
        print(f"  {a} actions/step: {c} ({100*c/total_samples:.1f}%)")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SB3BCDataset(Dataset):
    """Pre-batched dataset for SB3 behavioral cloning."""

    def __init__(self, path: str, max_samples: int = 0, filter_noops: bool = False):
        self.data = torch.load(path, weights_only=True, mmap=True)
        self._len = self.data["action_counts"].shape[0]

        # Optionally filter to only samples with at least 1 action
        if filter_noops:
            has_action = self.data["action_counts"] > 0
            indices = has_action.nonzero(as_tuple=True)[0]
            self.data = {k: v[indices].clone() for k, v in self.data.items()}
            self._len = indices.shape[0]
        elif max_samples > 0 and max_samples < self._len:
            indices = torch.randperm(self._len)[:max_samples].sort().values
            self.data = {k: v[indices].clone() for k, v in self.data.items()}
            self._len = max_samples

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        obs = self.data["obs"][idx]
        if obs.dtype == torch.float16:
            obs = obs.float()
        return {
            "obs": obs,
            "actions": self.data["actions"][idx],
            "action_counts": self.data["action_counts"][idx],
        }


# ---------------------------------------------------------------------------
# Policy network (matches SB3 MaskableActorCriticPolicy architecture)
# ---------------------------------------------------------------------------


class SB3PolicyForBC(nn.Module):
    """SB3-compatible policy for behavioral cloning.

    Architecture mirrors what MaskablePPO creates:
      features_extractor (CandidateTransformerExtractor) -> mlp_extractor.policy_net -> action_net

    After BC training, weights are exported with SB3-compatible keys so they
    can be loaded directly into a MaskablePPO model.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_candidates = SB3_MAX_CANDIDATES
        self.edge_dim = EDGE_INPUT_DIM

        # --- Features extractor (CandidateTransformerExtractor) ---
        self.edge_encoder = nn.Sequential(
            nn.Linear(EDGE_INPUT_DIM, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # --- MLP extractor (policy net: pi=[256, 256]) ---
        self.policy_net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # --- Action net: (256) -> (NUM_CHOICES) per slot ---
        # SB3 MaskablePPO with MultiDiscrete creates action_net as Linear(256, sum(nvec))
        # sum(nvec) = 481 * 5 = 2405
        self.action_net = nn.Linear(256, NUM_CHOICES * MAX_ACTIONS)

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for all slots.

        Args:
            obs_flat: (B, OBS_DIM) flat observation

        Returns:
            logits: (B, MAX_ACTIONS, NUM_CHOICES) action logits per slot
        """
        B = obs_flat.shape[0]

        # Reshape to candidate edges
        x = obs_flat.view(B, self.num_candidates, self.edge_dim)
        pad_mask = (x.abs().sum(dim=-1) < 1e-6)  # (B, K)

        x = x.clamp(-50.0, 50.0)
        x = self.edge_encoder(x)
        x = x.clamp(-100.0, 100.0)

        for block in self.blocks:
            x = block(x, mask=pad_mask)

        # Masked mean-pool
        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # (B, d_model)

        # MLP + action logits
        h = self.policy_net(pooled)
        logits_flat = self.action_net(h)  # (B, NUM_CHOICES * MAX_ACTIONS)
        logits = logits_flat.view(B, MAX_ACTIONS, NUM_CHOICES)

        return logits

    def export_sb3_state_dict(self) -> dict:
        """Export weights with SB3-compatible key prefixes.

        MaskablePPO expects keys like:
          features_extractor.edge_encoder.0.weight
          features_extractor.blocks.0.attn.in_proj_weight
          mlp_extractor.policy_net.0.weight
          action_net.weight
        """
        state = {}

        # Features extractor
        for k, v in self.edge_encoder.state_dict().items():
            state[f"features_extractor.edge_encoder.{k}"] = v
        for k, v in self.blocks.state_dict().items():
            state[f"features_extractor.blocks.{k}"] = v

        # MLP extractor (policy net)
        for k, v in self.policy_net.state_dict().items():
            state[f"mlp_extractor.policy_net.{k}"] = v

        # Action net
        for k, v in self.action_net.state_dict().items():
            state[f"action_net.{k}"] = v

        return state


class _TransformerBlock(nn.Module):
    """Transformer block matching CandidateTransformerExtractor.EdgeTransformerBlock."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        if mask is not None:
            B, K = mask.shape
            n_heads = self.attn.num_heads
            float_mask = torch.zeros(B, K, K, device=x.device, dtype=x.dtype)
            pad_cols = mask.unsqueeze(1).expand(-1, K, -1)
            float_mask = float_mask.masked_fill(pad_cols, -1e4)
            float_mask = float_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
            float_mask = float_mask.reshape(B * n_heads, K, K)
            attn_out, _ = self.attn(x, x, x, attn_mask=float_mask, need_weights=False)
        else:
            attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# ---------------------------------------------------------------------------
# BC Loss
# ---------------------------------------------------------------------------


def sb3_bc_loss(
    model: SB3PolicyForBC,
    batch: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, dict]:
    """Factored BC loss decomposed into candidate selection + fraction selection.

    For each slot with an expert action (cand_idx * NUM_FRACTIONS + frac_idx):
      1. Candidate CE: sum logits across fractions for each candidate → 49 classes (48 cands + noop)
      2. Fraction CE: within the chosen candidate's 10 fraction logits → 10 classes

    For noop slots: only candidate CE (target = noop class = 48).

    This factoring makes the problem much easier than flat 481-class CE:
    picking 1-of-49 + 1-of-10 vs 1-of-481.
    """
    obs = batch["obs"].to(device)                    # (B, OBS_DIM)
    actions = batch["actions"].to(device)             # (B, MAX_ACTIONS)
    action_counts = batch["action_counts"].to(device) # (B,)

    logits = model(obs)  # (B, MAX_ACTIONS, NUM_CHOICES)
    B = obs.shape[0]

    total_cand_loss = torch.tensor(0.0, device=device)
    total_frac_loss = torch.tensor(0.0, device=device)
    cand_correct = 0
    cand_total = 0
    frac_correct = 0
    frac_total = 0

    for slot in range(MAX_ACTIONS):
        slot_logits = logits[:, slot, :]  # (B, NUM_CHOICES=481)

        # Reshape to (B, 48, 10) for candidate×fraction + (B, 1) for noop
        cand_frac_logits = slot_logits[:, :SB3_MAX_CANDIDATES * NUM_FRACTIONS].view(B, SB3_MAX_CANDIDATES, NUM_FRACTIONS)
        noop_logit = slot_logits[:, NOOP_ACTION:NOOP_ACTION+1]  # (B, 1)

        # Candidate logits: logsumexp over fractions for each candidate (marginalize fraction)
        cand_scores = cand_frac_logits.logsumexp(dim=-1)  # (B, 48)
        cand_logits_49 = torch.cat([cand_scores, noop_logit], dim=-1)  # (B, 49)

        # Candidate targets
        slot_targets = actions[:, slot]  # (B,) values 0..480
        has_action = (action_counts > slot)  # (B,) bool
        # For action slots: target = cand_idx; for noop slots: target = 48
        cand_targets = torch.where(
            has_action,
            slot_targets // NUM_FRACTIONS,  # cand_idx
            torch.full_like(slot_targets, SB3_MAX_CANDIDATES),  # noop = class 48
        )
        # Clamp to valid range
        cand_targets = cand_targets.clamp(0, SB3_MAX_CANDIDATES)

        # Candidate CE with act upweighting
        act_weight = torch.where(has_action, torch.full((B,), 5.0, device=device),
                                 torch.ones(B, device=device))
        cand_loss = F.cross_entropy(cand_logits_49, cand_targets, reduction='none')
        total_cand_loss = total_cand_loss + (cand_loss * act_weight).mean()

        # Candidate accuracy
        cand_preds = cand_logits_49.argmax(dim=-1)
        cand_correct += (cand_preds == cand_targets).sum().item()
        cand_total += B

        # Fraction CE: only for acting slots
        if has_action.any():
            act_idx = has_action.nonzero(as_tuple=True)[0]
            act_targets = slot_targets[act_idx]
            act_cand_idx = (act_targets // NUM_FRACTIONS).clamp(0, SB3_MAX_CANDIDATES - 1)
            act_frac_target = (act_targets % NUM_FRACTIONS)

            # Get fraction logits for the chosen candidate
            act_cand_frac = cand_frac_logits[act_idx]  # (n_act, 48, 10)
            # Gather the chosen candidate's fraction logits
            chosen_frac_logits = act_cand_frac[
                torch.arange(act_idx.shape[0], device=device), act_cand_idx
            ]  # (n_act, 10)

            frac_loss = F.cross_entropy(chosen_frac_logits, act_frac_target, reduction='mean')
            total_frac_loss = total_frac_loss + frac_loss

            frac_preds = chosen_frac_logits.argmax(dim=-1)
            frac_correct += (frac_preds == act_frac_target).sum().item()
            frac_total += act_idx.shape[0]

    total_cand_loss = total_cand_loss / MAX_ACTIONS
    total_frac_loss = total_frac_loss / MAX_ACTIONS
    total_loss = total_cand_loss + total_frac_loss

    metrics = {
        "loss": total_loss.item(),
        "cand_acc": cand_correct / max(cand_total, 1),
        "frac_acc": frac_correct / max(frac_total, 1),
    }
    return total_loss, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_bc_sb3(
    model: SB3PolicyForBC,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 8,
    device: torch.device = torch.device("cpu"),
    checkpoint_path: str = "ppo_gnn/cache/bc_sb3_checkpoint.pt",
) -> SB3PolicyForBC:
    """Train SB3 policy with behavioral cloning."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        t0 = time.time()

        # Train
        model.train()
        train_loss_sum = 0.0
        train_cand_acc_sum = 0.0
        train_frac_acc_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            loss, metrics = sb3_bc_loss(model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += metrics["loss"]
            train_cand_acc_sum += metrics["cand_acc"]
            train_frac_acc_sum += metrics["frac_acc"]
            train_batches += 1

            if train_batches % 200 == 0:
                avg_loss = train_loss_sum / train_batches
                avg_cand = train_cand_acc_sum / train_batches
                avg_frac = train_frac_acc_sum / train_batches
                print(f"  batch {train_batches}, loss={avg_loss:.4f}, "
                      f"cand_acc={avg_cand:.3f}, frac_acc={avg_frac:.3f}",
                      flush=True)

        scheduler.step()

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_cand_acc_sum = 0.0
        val_frac_acc_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                loss, metrics = sb3_bc_loss(model, batch, device)
                val_loss_sum += metrics["loss"]
                val_cand_acc_sum += metrics["cand_acc"]
                val_frac_acc_sum += metrics["frac_acc"]
                val_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)
        train_cand_acc = train_cand_acc_sum / max(train_batches, 1)
        train_frac_acc = train_frac_acc_sum / max(train_batches, 1)
        val_loss = val_loss_sum / max(val_batches, 1)
        val_cand_acc = val_cand_acc_sum / max(val_batches, 1)
        val_frac_acc = val_frac_acc_sum / max(val_batches, 1)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1}/{epochs} ({elapsed:.0f}s) — "
            f"train: loss={train_loss:.4f} cand={train_cand_acc:.3f} frac={train_frac_acc:.3f} | "
            f"val: loss={val_loss:.4f} cand={val_cand_acc:.3f} frac={val_frac_acc:.3f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save both BC checkpoint and SB3-compatible weights
            torch.save({
                "model_state_dict": model.state_dict(),
                "sb3_state_dict": model.export_sb3_state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_cand_acc": val_cand_acc,
                "val_frac_acc": val_frac_acc,
            }, checkpoint_path)
            print(f"  Saved checkpoint (val_loss={val_loss:.4f}, "
                  f"cand={val_cand_acc:.3f}, frac={val_frac_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best
    ckpt = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


# ---------------------------------------------------------------------------
# Export for SB3 MaskablePPO
# ---------------------------------------------------------------------------


def export_for_sb3(checkpoint_path: str, output_path: str) -> None:
    """Extract SB3-compatible state dict from BC checkpoint.

    The exported .pt file can be loaded into MaskablePPO via:
        model = MaskablePPO.load(...)
        sb3_weights = torch.load(output_path)
        model.policy.load_state_dict(sb3_weights, strict=False)
    """
    ckpt = torch.load(checkpoint_path, weights_only=True)
    sb3_state = ckpt["sb3_state_dict"]
    torch.save(sb3_state, output_path)
    print(f"Exported SB3 weights to {output_path}")
    print(f"  Keys: {len(sb3_state)}")
    total_params = sum(v.numel() for v in sb3_state.values())
    print(f"  Parameters: {total_params:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="SB3 Behavioral Cloning from Kaggle replays")
    parser.add_argument("--replay-dir", default="kaggle_replays/episodes")
    parser.add_argument("--cache-dir", default="ppo_gnn/cache")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-step", type=int, default=0,
                        help="Only use first N steps of each game (0=all)")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Max replay files to parse (0=all)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max training samples (0=all)")
    parser.add_argument("--filter-noops", action="store_true",
                        help="Remove samples where expert did nothing")
    parser.add_argument("--winners-only", action="store_true", default=True)
    parser.add_argument("--all-players", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-parse", action="store_true",
                        help="Skip parsing, load from cache only")
    parser.add_argument("--export-only", action="store_true",
                        help="Only export SB3 weights from existing checkpoint")
    args = parser.parse_args()

    if args.all_players:
        args.winners_only = False

    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = str(cache_dir / "bc_sb3_checkpoint.pt")
    sb3_weights_path = str(cache_dir / "sb3_bc_weights.pt")

    # Export-only mode
    if args.export_only:
        export_for_sb3(checkpoint_path, sb3_weights_path)
        return

    # Determine cache filename (include max_step in name for different configs)
    step_suffix = f"_step{args.max_step}" if args.max_step > 0 else ""
    cache_file = cache_dir / f"fast_bc_sb3{step_suffix}.pt"

    # Parse replays
    if not args.skip_parse and not cache_file.exists():
        print(f"Parsing replays from {args.replay_dir}...")
        parse_and_prebatch_sb3(
            args.replay_dir,
            str(cache_file),
            winners_only=args.winners_only,
            max_step=args.max_step,
            max_files=args.max_files,
            chunk_size=200,
        )

    if not cache_file.exists():
        print(f"ERROR: No cache file at {cache_file}. Run without --skip-parse first.")
        return

    # Load dataset
    print(f"Loading from {cache_file}...")
    dataset = SB3BCDataset(
        str(cache_file),
        max_samples=args.max_samples,
        filter_noops=args.filter_noops,
    )
    print(f"Dataset: {len(dataset)} samples")

    # Action count distribution
    counts = dataset.data["action_counts"]
    for a in range(MAX_ACTIONS + 1):
        c = (counts == a).sum().item()
        print(f"  {a} actions/step: {c} ({100*c/len(dataset):.1f}%)")

    # Split 90/10
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = SB3PolicyForBC(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: d={args.d_model}, heads={args.n_heads}, layers={args.n_layers}")
    print(f"Parameters: {total_params:,}")

    # Train
    model = train_bc_sb3(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        device=device, checkpoint_path=checkpoint_path,
    )

    # Export SB3-compatible weights
    export_for_sb3(checkpoint_path, sb3_weights_path)
    print(f"\nBC training complete!")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  SB3 weights: {sb3_weights_path}")
    print(f"\nTo load into MaskablePPO for fine-tuning:")
    print(f"  python -m ppo_gnn.train_sb3 --checkpoint ppo_gnn/cache/sb3_bc_weights.pt")


if __name__ == "__main__":
    main()
