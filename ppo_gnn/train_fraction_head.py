"""Train only the fraction head from replay data.

Loads a pretrained EdgePolicy checkpoint, freezes everything except
fraction_head, then trains on expert fraction choices from replays.

The idea: edge selection (where to send) is learned via PPO/BC, but
fraction sizing (how many ships to send) can be refined independently
from expert demonstrations.

Usage:
    python -m ppo_gnn.train_fraction_head \
        --checkpoint ppo_gnn/cache/checkpoint_ppo_edge_best.pt \
        --replay-dir kaggle_replays \
        --epochs 20 --device mps
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .edge_policy import (
    EDGE_INPUT_DIM,
    MAX_ACTIONS,
    MAX_CANDIDATES,
    NUM_FRACTIONS,
    EdgePolicy,
)
from .train_bc_edge import (
    PreBatchedEdgeDataset,
    parse_all_replays_edges,
    prebatch_edge_data,
)


# ---------------------------------------------------------------------------
# Fraction-only loss
# ---------------------------------------------------------------------------

def fraction_only_loss(
    model: EdgePolicy,
    batch: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, dict]:
    """Cross-entropy loss on fraction buckets for expert-chosen edges only.

    Only computes loss on steps where the expert actually sent fleets
    (skips noops). Uses the expert's edge choice to index into
    frac_logits — we're not training edge selection here.
    """
    ef = batch["edge_features"].to(device)      # (B, K, 74)
    em = batch["edge_mask"].to(device)           # (B, K)
    ae = batch["action_edges"].to(device)        # (B, MAX_ACTIONS)
    af = batch["action_fractions"].to(device)    # (B, MAX_ACTIONS)
    ac = batch["action_counts"].to(device)       # (B,)

    B, K, _ = ef.shape

    # Forward pass — only fraction_head is unfrozen, but we need
    # the full forward to get frac_logits from the transformer output
    _, _, frac_logits, _ = model(ef, em)
    # frac_logits: (B, K, NUM_FRACTIONS)

    total_loss = torch.tensor(0.0, device=device)
    total_actions = 0
    correct = 0

    for action_idx in range(MAX_ACTIONS):
        has_action = (ac > action_idx)  # (B,) bool
        if not has_action.any():
            break

        # Get expert's chosen edge and fraction for this slot
        edge_idx = ae[:, action_idx].clamp(0, K - 1)    # (B,)
        frac_target = af[:, action_idx].clamp(0, NUM_FRACTIONS - 1)  # (B,)

        # Index into frac_logits at the expert's chosen edge
        chosen_frac_logits = frac_logits[
            torch.arange(B, device=device), edge_idx
        ]  # (B, NUM_FRACTIONS)

        # Cross-entropy, masked to samples that have this action
        loss = F.cross_entropy(chosen_frac_logits, frac_target, reduction='none')
        total_loss = total_loss + (loss * has_action.float()).sum()
        n_active = has_action.sum().item()
        total_actions += n_active

        # Accuracy
        with torch.no_grad():
            preds = chosen_frac_logits.argmax(dim=-1)
            correct += ((preds == frac_target) & has_action).sum().item()

    if total_actions == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            "frac_loss": 0.0, "frac_acc": 0.0, "n_actions": 0,
        }

    avg_loss = total_loss / total_actions
    accuracy = correct / total_actions

    return avg_loss, {
        "frac_loss": avg_loss.item(),
        "frac_acc": accuracy,
        "n_actions": total_actions,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_fraction_head(
    model: EdgePolicy,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 6,
    device: torch.device = torch.device("cpu"),
    checkpoint_path: str = "checkpoint_frac_head.pt",
    full_checkpoint_path: str = "checkpoint_frac_merged.pt",
) -> EdgePolicy:
    """Train only fraction_head, everything else frozen."""
    model = model.to(device)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze fraction head only
    for param in model.fraction_head.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = torch.optim.Adam(model.fraction_head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        t0 = time.time()

        # Train
        model.train()
        # Keep everything in eval mode except fraction_head
        model.edge_encoder.eval()
        for block in model.transformer:
            block.eval()
        model.selection_head.eval()
        model.noop_head.eval()
        model.phase_head.eval()
        model.value_head.eval()
        if model.separate_critic:
            model.critic_encoder.eval()
            for block in model.critic_transformer:
                block.eval()

        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            loss, metrics = fraction_only_loss(model, batch, device)
            if metrics["n_actions"] == 0:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.fraction_head.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += metrics["frac_loss"]
            train_acc_sum += metrics["frac_acc"]
            train_batches += 1

        scheduler.step()

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                _, metrics = fraction_only_loss(model, batch, device)
                if metrics["n_actions"] == 0:
                    continue
                val_loss_sum += metrics["frac_loss"]
                val_acc_sum += metrics["frac_acc"]
                val_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)
        train_acc = train_acc_sum / max(train_batches, 1)
        val_loss = val_loss_sum / max(val_batches, 1)
        val_acc = val_acc_sum / max(val_batches, 1)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1}/{epochs} ({elapsed:.0f}s) — "
            f"train: loss={train_loss:.4f} acc={train_acc:.1%} | "
            f"val: loss={val_loss:.4f} acc={val_acc:.1%} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save just the fraction head weights
            torch.save({
                "fraction_head_state_dict": model.fraction_head.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, checkpoint_path)
            # Save full merged model for direct use
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "d_model": model.d_model,
                "max_actions": model.max_actions,
                "separate_critic": model.separate_critic,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "fraction_head_only": True,
            }, full_checkpoint_path)
            print(f"  Saved (val_loss={val_loss:.4f}, val_acc={val_acc:.1%})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best
    ckpt = torch.load(checkpoint_path, weights_only=True)
    model.fraction_head.load_state_dict(ckpt["fraction_head_state_dict"])
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train fraction head only from replay data")
    parser.add_argument("--checkpoint", required=True,
                        help="Pretrained EdgePolicy checkpoint to load")
    parser.add_argument("--replay-dir", default="kaggle_replays")
    parser.add_argument("--cache-dir", default="ppo_gnn/cache")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--winners-only", action="store_true", default=True)
    parser.add_argument("--all-players", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-parse", action="store_true",
                        help="Skip parsing, load from cache only")
    args = parser.parse_args()

    if args.all_players:
        args.winners_only = False

    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the same cache as BC edge training
    cache_file = cache_dir / "fast_bc_edge.pt"

    if not args.skip_parse and not cache_file.exists():
        print(f"Parsing replays from {args.replay_dir}...")
        transitions = parse_all_replays_edges(
            args.replay_dir,
            winners_only=args.winners_only,
            max_files=args.max_files,
        )
        print(f"\nPre-batching {len(transitions)} transitions...")
        prebatch_edge_data(transitions, str(cache_file))
        del transitions

    if not cache_file.exists():
        print(f"ERROR: No cache file at {cache_file}. Run without --skip-parse first.")
        return

    print(f"Loading from {cache_file}...")
    dataset = PreBatchedEdgeDataset(str(cache_file), max_samples=args.max_samples)

    # Filter to non-noop samples only (we only care about fraction labels)
    counts = dataset.data["action_counts"]
    has_actions = counts > 0
    n_with_actions = has_actions.sum().item()
    print(f"Dataset: {len(dataset)} total, {n_with_actions} with actions "
          f"({100*n_with_actions/len(dataset):.1f}%)")

    # Fraction distribution
    fracs = dataset.data["action_fractions"]
    valid_fracs = fracs[has_actions.unsqueeze(1).expand_as(fracs)]
    valid_fracs = valid_fracs[valid_fracs >= 0]
    print("Fraction distribution in expert data:")
    for i in range(NUM_FRACTIONS):
        c = (valid_fracs == i).sum().item()
        print(f"  bucket {i} ({int(100*(i+1)/NUM_FRACTIONS)}%): "
              f"{c} ({100*c/max(len(valid_fracs),1):.1f}%)")

    # Split 90/10
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load pretrained model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt)
    d_model = ckpt.get("d_model", args.d_model)
    separate_critic = ckpt.get("separate_critic", True)

    model = EdgePolicy(
        d_model=d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_actions=MAX_ACTIONS,
        separate_critic=separate_critic,
    )

    # Handle shape mismatches (e.g., old fraction head with different bucket count)
    model_sd = model.state_dict()
    filtered_sd = {}
    for k, v in sd.items():
        if k in model_sd and v.shape != model_sd[k].shape:
            print(f"  Skipping {k}: ckpt={list(v.shape)} vs model={list(model_sd[k].shape)}")
        else:
            filtered_sd[k] = v

    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
    if missing:
        print(f"  {len(missing)} missing keys (will use random init)")

    print(f"\nEdgePolicy loaded: d={d_model}, "
          f"fraction_head will be trained from expert data")

    frac_ckpt = str(cache_dir / "checkpoint_frac_head.pt")
    merged_ckpt = str(cache_dir / "checkpoint_frac_merged.pt")

    model = train_fraction_head(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        device=device,
        checkpoint_path=frac_ckpt,
        full_checkpoint_path=merged_ckpt,
    )

    print(f"\nFraction head training complete.")
    print(f"  Fraction head only: {frac_ckpt}")
    print(f"  Full merged model:  {merged_ckpt}")
    print(f"\nTo continue PPO training with the refined fraction head:")
    print(f"  python -m ppo_gnn.train_ppo_edge --checkpoint {merged_ckpt} --mode 2p")


if __name__ == "__main__":
    main()
