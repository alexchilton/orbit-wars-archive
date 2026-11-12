"""Behavioral Cloning for the edge-based transformer policy.

Instead of per-sub-action transitions, this groups all actions within a game
step into a single training sample with up to MAX_ACTIONS edge selections.

Training target: given the top-192 candidate edges (from heuristics),
which edges did the winning player actually choose, and with what fraction?

Usage:
    python -m ppo_gnn.train_bc_edge --replay-dir kaggle_replays --epochs 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from .edge_policy import (
    EDGE_INPUT_DIM,
    FRACTION_BUCKETS,
    MAX_ACTIONS,
    MAX_CANDIDATES,
    NUM_FRACTIONS,
    EdgePolicy,
    compute_candidate_edges,
)
from .replay_parser import _build_node_features, _angle_to_target

BOARD_SIZE = 100.0
SUN_X, SUN_Y = 50.0, 50.0


# ---------------------------------------------------------------------------
# Replay parsing for edge-based training
# ---------------------------------------------------------------------------

@dataclass
class EdgeTransition:
    """One game step for one player, with grouped multi-action."""
    edge_features: torch.Tensor     # (MAX_CANDIDATES, EDGE_INPUT_DIM)
    edge_indices: torch.Tensor      # (MAX_CANDIDATES, 2) — [src_idx, tgt_idx]
    edge_mask: torch.Tensor         # (MAX_CANDIDATES,) — 1.0 valid, 0.0 pad
    action_edges: torch.Tensor      # (MAX_ACTIONS,) — candidate indices chosen (-1 = unused)
    action_fractions: torch.Tensor  # (MAX_ACTIONS,) — fraction bucket indices (-1 = unused)
    action_count: int               # how many edges selected (0 = noop)
    discounted_return: float
    player_rank: int
    mode: str


def _fraction_to_bucket_10(ships_sent: float, ships_available: float) -> int:
    """Snap ship fraction to nearest 10% bucket (10 bins)."""
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


def parse_replay_edges(filepath: str, gamma: float = 0.999) -> List[EdgeTransition]:
    """Parse a replay into edge-based transitions (one per step per player).

    Key difference from original parser: all sub-actions within a step are
    grouped into a single EdgeTransition with up to MAX_ACTIONS edge selections.
    """
    with open(filepath) as f:
        data = json.load(f)

    if "steps" not in data:
        return []

    steps = data["steps"]
    num_players = len(steps[0])
    mode = "2p" if num_players == 2 else "4p"
    rewards = data.get("rewards", [0] * num_players)

    # Compute player ranks
    sorted_rewards = sorted(enumerate(rewards), key=lambda x: -(x[1] or -999))
    rank_map = {pid: rank + 1 for rank, (pid, _) in enumerate(sorted_rewards)}

    all_transitions = []

    for step_idx, step in enumerate(steps):
        for player_id in range(num_players):
            agent_data = step[player_id]
            obs = agent_data.get("observation")
            action = agent_data.get("action", [])

            if obs is None or "planets" not in obs:
                continue

            planets = obs["planets"]
            fleets = obs.get("fleets", [])

            if not planets:
                continue

            # Check alive
            has_planets = any(int(p[1]) == player_id for p in planets)
            if not has_planets:
                continue

            # Generate candidate edges using heuristic scorer
            edge_features, edge_indices, edge_mask, num_valid = compute_candidate_edges(
                planets=planets,
                fleets=fleets,
                player_id=player_id,
                num_players=num_players,
                step=step_idx,
                max_steps=len(steps),
            )

            if num_valid == 0:
                continue

            # Parse actions and match to candidate edges
            action_edges = torch.full((MAX_ACTIONS,), -1, dtype=torch.long)
            action_fractions = torch.full((MAX_ACTIONS,), -1, dtype=torch.long)
            action_count = 0

            pid_to_idx = {int(p[0]): i for i, p in enumerate(planets)}

            if action:
                used_sources = set()
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
                    if src_idx in used_sources:
                        continue
                    used_sources.add(src_idx)

                    src_planet = planets[src_idx]
                    src_x, src_y = float(src_planet[2]), float(src_planet[3])
                    src_ships = float(src_planet[5])

                    tgt_idx = _angle_to_target(src_x, src_y, angle, planets, src_planet_id)
                    if tgt_idx is None:
                        continue

                    # Find this (src, tgt) pair in candidate edges
                    matched_candidate = -1
                    for k in range(num_valid):
                        if (edge_indices[k, 0].item() == src_idx and
                                edge_indices[k, 1].item() == tgt_idx):
                            matched_candidate = k
                            break

                    if matched_candidate < 0:
                        # Expert chose an edge not in our top-192 candidates
                        # Skip this action (heuristic didn't rank it high enough)
                        continue

                    frac_idx = _fraction_to_bucket_10(ships_sent, src_ships)
                    action_edges[action_count] = matched_candidate
                    action_fractions[action_count] = frac_idx
                    action_count += 1

            # Compute discounted return
            T = len(steps) - 1
            final_reward = rewards[player_id] if rewards[player_id] is not None else 0.0
            steps_remaining = T - step_idx
            disc_return = final_reward * (gamma ** steps_remaining)

            all_transitions.append(EdgeTransition(
                edge_features=edge_features,
                edge_indices=edge_indices,
                edge_mask=edge_mask,
                action_edges=action_edges,
                action_fractions=action_fractions,
                action_count=action_count,
                discounted_return=disc_return,
                player_rank=rank_map.get(player_id, num_players),
                mode=mode,
            ))

    return all_transitions


def parse_all_replays_edges(
    replay_dir: str,
    gamma: float = 0.999,
    winners_only: bool = True,
    verbose: bool = True,
    max_files: int = 0,
) -> List[EdgeTransition]:
    """Parse all replay files into edge-based transitions."""
    transitions = []
    total_files = 0
    errors = 0
    matched = 0
    missed = 0

    files = []
    for root, _dirs, fnames in os.walk(replay_dir):
        for fname in fnames:
            if fname.endswith(".json"):
                files.append(os.path.join(root, fname))

    if max_files > 0:
        files = files[:max_files]

    for filepath in files:
        total_files += 1
        try:
            trans = parse_replay_edges(filepath, gamma)
            for t in trans:
                if winners_only and t.player_rank != 1:
                    continue
                transitions.append(t)
                if t.action_count > 0:
                    matched += t.action_count
        except Exception as e:
            errors += 1
            if verbose and errors <= 5:
                print(f"  Error parsing {filepath}: {e}")

        if verbose and total_files % 200 == 0:
            print(f"  Parsed {total_files} files, {len(transitions)} transitions so far")

    if verbose:
        print(f"Parsed {total_files} files ({errors} errors)")
        print(f"  Total transitions: {len(transitions)}")
        noop_count = sum(1 for t in transitions if t.action_count == 0)
        act_count = sum(t.action_count for t in transitions)
        print(f"  Noop steps: {noop_count} ({100*noop_count/max(len(transitions),1):.1f}%)")
        print(f"  Total actions matched to candidates: {act_count}")
        for a in range(MAX_ACTIONS + 1):
            c = sum(1 for t in transitions if t.action_count == a)
            print(f"    {a} actions: {c} ({100*c/max(len(transitions),1):.1f}%)")

    return transitions


# ---------------------------------------------------------------------------
# Dataset and pre-batching
# ---------------------------------------------------------------------------

class EdgeBCDataset(Dataset):
    """Dataset of edge-based transitions."""

    def __init__(self, transitions: List[EdgeTransition]):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        t = self.transitions[idx]
        return {
            "edge_features": t.edge_features,         # (K, 74)
            "edge_mask": t.edge_mask,                  # (K,)
            "action_edges": t.action_edges,            # (3,)
            "action_fractions": t.action_fractions,    # (3,)
            "action_count": torch.tensor(t.action_count, dtype=torch.long),
        }


def prebatch_edge_data(
    transitions: List[EdgeTransition], path: str,
) -> None:
    """Pre-batch edge transitions into tensors for fast loading."""
    N = len(transitions)
    K = MAX_CANDIDATES

    # Allocate directly as float16 to halve peak RAM (avoids float32→half copy)
    edge_features = torch.zeros(N, K, EDGE_INPUT_DIM, dtype=torch.float16)
    edge_mask = torch.zeros(N, K, dtype=torch.float16)
    action_edges = torch.zeros(N, MAX_ACTIONS, dtype=torch.long)
    action_fractions = torch.zeros(N, MAX_ACTIONS, dtype=torch.long)
    action_counts = torch.zeros(N, dtype=torch.long)

    # Free each transition after copying to reclaim RAM progressively
    for i in range(N):
        t = transitions[i]
        edge_features[i] = t.edge_features.half()
        edge_mask[i] = t.edge_mask.half()
        action_edges[i] = t.action_edges
        action_fractions[i] = t.action_fractions
        action_counts[i] = t.action_count
        transitions[i] = None  # free source tensor memory as we go

    data = {
        "edge_features": edge_features,
        "edge_mask": edge_mask,
        "action_edges": action_edges,
        "action_fractions": action_fractions,
        "action_counts": action_counts,
    }
    torch.save(data, path)
    print(f"Saved {N} pre-batched edge transitions to {path} "
          f"({os.path.getsize(path) / 1e9:.2f} GB)")


def parse_and_prebatch_chunked(
    replay_dir: str,
    path: str,
    winners_only: bool = True,
    max_files: int = 0,
    chunk_size: int = 500,
) -> None:
    """Parse replays in chunks to avoid OOM, then concatenate into one cache file.

    Processes `chunk_size` replay files at a time, saves each chunk as a temp
    .pt file, then concatenates all chunks into the final output.
    """
    import gc
    import tempfile

    files = []
    for root, _dirs, fnames in os.walk(replay_dir):
        for fname in fnames:
            if fname.endswith(".json"):
                files.append(os.path.join(root, fname))

    if max_files > 0:
        files = files[:max_files]

    print(f"Found {len(files)} replay files, processing in chunks of {chunk_size}")

    chunk_paths = []
    total_transitions = 0
    total_errors = 0

    for chunk_start in range(0, len(files), chunk_size):
        chunk_files = files[chunk_start:chunk_start + chunk_size]
        chunk_idx = chunk_start // chunk_size
        transitions = []
        errors = 0

        for filepath in chunk_files:
            try:
                trans = parse_replay_edges(filepath, gamma=0.999)
                for t in trans:
                    if winners_only and t.player_rank != 1:
                        continue
                    transitions.append(t)
                del trans
            except Exception:
                errors += 1

        total_errors += errors

        if not transitions:
            print(f"  Chunk {chunk_idx}: 0 transitions (skipped)")
            continue

        # Save chunk to temp file
        chunk_path = path + f".chunk{chunk_idx}.tmp"
        N = len(transitions)
        K = MAX_CANDIDATES
        ef = torch.zeros(N, K, EDGE_INPUT_DIM, dtype=torch.float16)
        em = torch.zeros(N, K, dtype=torch.float16)
        ae = torch.zeros(N, MAX_ACTIONS, dtype=torch.long)
        af = torch.zeros(N, MAX_ACTIONS, dtype=torch.long)
        ac = torch.zeros(N, dtype=torch.long)

        for i in range(N):
            t = transitions[i]
            ef[i] = t.edge_features.half()
            em[i] = t.edge_mask.half()
            ae[i] = t.action_edges
            af[i] = t.action_fractions
            ac[i] = t.action_count
            transitions[i] = None

        torch.save({"edge_features": ef, "edge_mask": em,
                     "action_edges": ae, "action_fractions": af,
                     "action_counts": ac}, chunk_path)

        total_transitions += N
        chunk_paths.append(chunk_path)
        print(f"  Chunk {chunk_idx}: {len(chunk_files)} files → {N} transitions "
              f"(total: {total_transitions})", flush=True)

        del transitions, ef, em, ae, af, ac
        gc.collect()

    # Concatenate all chunks
    print(f"\nConcatenating {len(chunk_paths)} chunks ({total_transitions} total)...")
    all_data = {k: [] for k in ["edge_features", "edge_mask", "action_edges",
                                  "action_fractions", "action_counts"]}
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

    torch.save(merged, path)
    print(f"Saved {total_transitions} transitions to {path} "
          f"({os.path.getsize(path) / 1e9:.2f} GB) "
          f"({total_errors} parse errors)")


class PreBatchedEdgeDataset(Dataset):
    """Fast dataset from pre-batched edge tensors.

    Uses memory-mapped loading to avoid loading entire dataset into RAM.
    """

    def __init__(self, path: str, max_samples: int = 0):
        # mmap_mode keeps tensors on disk, loads pages on demand
        self.data = torch.load(path, weights_only=True, mmap=True)
        self._len = self.data["action_counts"].shape[0]
        if max_samples > 0 and max_samples < self._len:
            indices = torch.randperm(self._len)[:max_samples].sort().values
            self.data = {k: v[indices].clone() for k, v in self.data.items()}
            self._len = max_samples

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        out = {}
        for k, v in self.data.items():
            t = v[idx]
            if t.dtype == torch.float16:
                t = t.float()
            out[k] = t
        return out


# ---------------------------------------------------------------------------
# BC Loss
# ---------------------------------------------------------------------------

def edge_bc_loss(
    model: EdgePolicy,
    batch: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, dict]:
    """Compute BC loss with split noop/edge/fraction heads.

    Three independent losses:
    1. Noop BCE: binary act/noop on ALL samples (balanced ~55/45)
    2. Edge CE: which edge to pick, on NON-NOOP samples only (K classes)
    3. Fraction CE: which fraction bucket, on NON-NOOP samples only
    """
    ef = batch["edge_features"].to(device)      # (B, K, 74)
    em = batch["edge_mask"].to(device)           # (B, K)
    ae = batch["action_edges"].to(device)        # (B, MAX_ACTIONS)
    af = batch["action_fractions"].to(device)    # (B, MAX_ACTIONS)
    ac = batch["action_counts"].to(device)       # (B,)

    B, K, _ = ef.shape

    noop_logit, edge_logits, frac_logits, _ = model(ef, em)
    # noop_logit: (B, 1), edge_logits: (B, K), frac_logits: (B, K, NUM_FRACTIONS)

    # --- Loss 1: Binary noop BCE on ALL samples ---
    # Upweight "act" samples heavily to prevent always-noop collapse.
    # Even with ~55/45 split, the model learns confident noop because it's
    # the easier prediction. Use 3x weight on act samples.
    is_noop = (ac == 0).float()  # (B,)
    act_upweight = 3.0
    sample_weight = torch.where(is_noop > 0.5, torch.ones_like(is_noop), torch.full_like(is_noop, act_upweight))
    noop_loss = F.binary_cross_entropy_with_logits(
        noop_logit.squeeze(-1), is_noop, weight=sample_weight, reduction='mean')

    # --- Loss 2 & 3: Edge CE + Fraction CE on NON-NOOP samples only ---
    has_any_action = ac > 0  # (B,) bool
    n_acting = has_any_action.sum().item()

    if n_acting == 0:
        edge_loss = torch.tensor(0.0, device=device)
        frac_loss = torch.tensor(0.0, device=device)
    else:
        total_edge_loss = torch.tensor(0.0, device=device)
        total_frac_loss = torch.tensor(0.0, device=device)
        total_actions = 0

        remaining_mask = em.clone()

        for action_idx in range(MAX_ACTIONS):
            has_action = (ac > action_idx)  # (B,) bool
            if not has_action.any():
                break

            # Subset to samples that have this action (avoids NaN from all-inf rows)
            idx_mask = has_action.nonzero(as_tuple=True)[0]
            n_active = idx_mask.shape[0]

            # Edge CE: softmax over K edges only (no noop)
            sub_edge = edge_logits[idx_mask].clone()
            sub_rmask = remaining_mask[idx_mask]
            sub_edge = sub_edge.masked_fill(sub_rmask < 0.5, float('-inf'))
            edge_target = ae[idx_mask, action_idx].clamp(0, K - 1)
            e_loss = F.cross_entropy(sub_edge, edge_target, reduction='sum')
            total_edge_loss = total_edge_loss + e_loss

            # Fraction CE
            edge_idx = ae[idx_mask, action_idx].clamp(0, K - 1)
            chosen_frac_logits = frac_logits[
                idx_mask, edge_idx
            ]  # (n_active, NUM_FRACTIONS)
            frac_target = af[idx_mask, action_idx].clamp(0, NUM_FRACTIONS - 1)
            f_loss = F.cross_entropy(chosen_frac_logits, frac_target, reduction='sum')
            total_frac_loss = total_frac_loss + f_loss

            total_actions += n_active

            # Update remaining mask
            for b_idx in range(n_active):
                b = idx_mask[b_idx].item()
                edge_val = ae[b, action_idx].item()
                if edge_val >= 0:
                    remaining_mask[b, min(edge_val, K - 1)] = 0.0

        edge_loss = total_edge_loss / max(total_actions, 1)
        frac_loss = total_frac_loss / max(total_actions, 1)

    loss = noop_loss + edge_loss + frac_loss

    metrics = {
        "noop_loss": noop_loss.item(),
        "sel_loss": edge_loss.item() if isinstance(edge_loss, torch.Tensor) else edge_loss,
        "frac_loss": frac_loss.item() if isinstance(frac_loss, torch.Tensor) else frac_loss,
        "total_loss": loss.item(),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_bc_edge(
    model: EdgePolicy,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 8,
    device: torch.device = torch.device("cpu"),
    checkpoint_path: str = "checkpoint_bc_edge.pt",
) -> EdgePolicy:
    """Train edge-based policy with behavioral cloning."""
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
        train_noop_sum = 0.0
        train_sel_sum = 0.0
        train_frac_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            loss, metrics = edge_bc_loss(model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += metrics["total_loss"]
            train_noop_sum += metrics["noop_loss"]
            train_sel_sum += metrics["sel_loss"]
            train_frac_sum += metrics["frac_loss"]
            train_batches += 1

            if train_batches % 500 == 0:
                avg = train_loss_sum / train_batches
                print(f"  batch {train_batches}, loss={avg:.4f}", flush=True)

        scheduler.step()

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_noop_sum = 0.0
        val_sel_sum = 0.0
        val_frac_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                loss, metrics = edge_bc_loss(model, batch, device)
                val_loss_sum += metrics["total_loss"]
                val_noop_sum += metrics["noop_loss"]
                val_sel_sum += metrics["sel_loss"]
                val_frac_sum += metrics["frac_loss"]
                val_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)
        val_loss = val_loss_sum / max(val_batches, 1)
        train_noop = train_noop_sum / max(train_batches, 1)
        train_sel = train_sel_sum / max(train_batches, 1)
        train_frac = train_frac_sum / max(train_batches, 1)
        val_noop = val_noop_sum / max(val_batches, 1)
        val_sel = val_sel_sum / max(val_batches, 1)
        val_frac = val_frac_sum / max(val_batches, 1)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1}/{epochs} ({elapsed:.0f}s) — "
            f"train: {train_loss:.4f} (noop={train_noop:.4f} sel={train_sel:.4f} frac={train_frac:.4f}) | "
            f"val: {val_loss:.4f} (noop={val_noop:.4f} sel={val_sel:.4f} frac={val_frac:.4f}) | "
            f"lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "d_model": model.d_model,
                "max_actions": model.max_actions,
                "separate_critic": model.separate_critic,
            }, checkpoint_path)
            print(f"  Saved checkpoint (val_loss={val_loss:.4f})")
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Edge-based Behavioral Cloning")
    parser.add_argument("--replay-dir", default="kaggle_replays")
    parser.add_argument("--cache-dir", default="ppo_gnn/cache")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=0,
                        help="Max replay files to parse (0=all)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max training samples (0=all)")
    parser.add_argument("--winners-only", action="store_true", default=True)
    parser.add_argument("--all-players", action="store_true",
                        help="Train on all players, not just winners")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-parse", action="store_true",
                        help="Skip parsing, load from cache only")
    args = parser.parse_args()

    if args.all_players:
        args.winners_only = False

    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / "fast_bc_edge.pt"

    if not args.skip_parse and not cache_file.exists():
        print(f"Parsing replays from {args.replay_dir}...")
        parse_and_prebatch_chunked(
            args.replay_dir,
            str(cache_file),
            winners_only=args.winners_only,
            max_files=args.max_files,
            chunk_size=100,
        )

    if not cache_file.exists():
        print(f"ERROR: No cache file at {cache_file}. Run without --skip-parse first.")
        return

    print(f"Loading from {cache_file}...")
    dataset = PreBatchedEdgeDataset(str(cache_file), max_samples=args.max_samples)
    print(f"Dataset: {len(dataset)} samples")

    # Action count distribution
    counts = dataset.data["action_counts"]
    for a in range(MAX_ACTIONS + 1):
        c = (counts == a).sum().item()
        print(f"  {a} actions: {c} ({100*c/len(dataset):.1f}%)")

    # Split 90/10
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = EdgePolicy(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        separate_critic=True,
    )
    params = model.count_parameters()
    print(f"Model: d={args.d_model}, heads={args.n_heads}, layers={args.n_layers}")
    print(f"Parameters: {params['total']:,}")

    checkpoint_path = str(cache_dir / "checkpoint_bc_edge.pt")
    model = train_bc_edge(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        device=device, checkpoint_path=checkpoint_path,
    )
    print(f"\nBC training complete. Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
