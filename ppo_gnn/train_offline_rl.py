"""Phase 2: Offline RL via Advantage-Weighted Regression (AWR).

Improves beyond behavioral cloning by learning from outcomes: upweights actions
that led to wins, downweights actions from losses. Uses the full replay dataset
(winners AND losers) unlike Phase 1 which only uses winners.

The counterfactual signal: losing players' transitions get negative advantages,
so the policy learns to avoid those actions in similar states.

Usage:
    python -m ppo_gnn.train_offline_rl --mode 2p --bc-checkpoint ppo_gnn/cache/checkpoint_bc_2p.pt
    python -m ppo_gnn.train_offline_rl --mode 4p --bc-checkpoint ppo_gnn/cache/checkpoint_bc_4p.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .gnn_policy import OrbitWarsGNNPolicy
from .replay_parser import PreBatchedDataset, ReplayDataset, load_dataset


def awr_loss(
    model: OrbitWarsGNNPolicy,
    batch: dict[str, torch.Tensor],
    temperature: float,
    device: torch.device,
    reward_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute AWR loss: advantage-weighted policy loss + value loss.

    Returns:
        (policy_loss, value_loss)
    """
    nf = batch["node_features"].to(device)
    pos = batch["positions"].to(device)
    owned = batch["owned_mask"].to(device)
    source = batch["source"].to(device)
    target = batch["target"].to(device)
    fraction = batch["fraction"].to(device)
    is_noop = batch["is_noop"].to(device)
    returns = batch["discounted_return"].to(device) * reward_scale

    B = nf.shape[0]
    M = nf.shape[1]

    # Value estimate
    value = model.get_value(nf, pos)
    value_loss = F.mse_loss(value, returns)

    # Advantage
    with torch.no_grad():
        advantage = returns - value.detach()
        weights = torch.exp(advantage / temperature).clamp(0.0, 20.0)

    # Policy log-probs using efficient bc_forward
    num_planets = batch["num_planets"].to(device) if "num_planets" in batch else None
    source_logits, tgt_logits, frac_logits = model.bc_forward(
        nf, pos, owned, source, target, num_planets=num_planets,
    )

    # Source
    source_target = torch.where(is_noop.bool(), torch.full_like(source, M), source)
    log_p_source = -F.cross_entropy(source_logits, source_target, reduction="none")

    # Target and fraction (for non-noop)
    launch_mask = (1.0 - is_noop)
    log_p_target = -F.cross_entropy(tgt_logits, target.clamp(0, M - 1), reduction="none")
    log_p_fraction = -F.cross_entropy(frac_logits, fraction.clamp(0, 3), reduction="none")

    # Joint log-prob
    log_prob = log_p_source + launch_mask * (log_p_target + log_p_fraction)

    # Weighted policy loss (negative because we maximize log-prob)
    policy_loss = -(weights * log_prob).mean()

    return policy_loss, value_loss


def train_awr(
    model: OrbitWarsGNNPolicy,
    train_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-4,
    temp_start: float = 1.0,
    temp_end: float = 0.1,
    value_coef: float = 0.5,
    device: torch.device = torch.device("cpu"),
    checkpoint_path: str = "checkpoint_awr.pt",
    reward_scale: float = 1.0,
) -> OrbitWarsGNNPolicy:
    """Train with advantage-weighted regression."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Anneal temperature
        progress = epoch / max(epochs - 1, 1)
        temperature = temp_start + (temp_end - temp_start) * progress

        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            policy_loss, value_loss = awr_loss(model, batch, temperature, device, reward_scale)
            loss = policy_loss + value_coef * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        avg_pl = total_policy_loss / max(num_batches, 1)
        avg_vl = total_value_loss / max(num_batches, 1)
        print(
            f"Epoch {epoch+1}/{epochs} — "
            f"policy_loss: {avg_pl:.4f}, value_loss: {avg_vl:.4f}, "
            f"temp: {temperature:.3f}"
        )

        torch.save(model.state_dict(), checkpoint_path)

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Offline RL (AWR)")
    parser.add_argument("--mode", choices=["2p", "4p"], required=True)
    parser.add_argument("--bc-checkpoint", required=True, help="Phase 1 checkpoint to initialize from")
    parser.add_argument("--cache-dir", default="ppo_gnn/cache")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temp-start", type=float, default=1.0)
    parser.add_argument("--temp-end", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--use-sage", action="store_true")
    parser.add_argument("--mask-sun", action="store_true")
    parser.add_argument("--max-planets", type=int, default=40)
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="Scale discounted returns to match PPO terminal rewards (use 10.0)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir)

    # Load pre-batched all-player dataset (created by train_bc.py)
    fast_file = cache_dir / f"fast_all_{args.mode}.pt"
    if not fast_file.exists():
        print(f"No pre-batched dataset at {fast_file}. Run train_bc.py first.")
        return
    dataset = PreBatchedDataset(str(fast_file))
    print(f"Loaded {len(dataset)} transitions (all players)")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Load BC-pretrained model
    model = OrbitWarsGNNPolicy(
        hidden_dim=args.hidden_dim,
        use_gat=not args.use_sage,
        mask_sun_targets=args.mask_sun,
    )
    model.load_state_dict(torch.load(args.bc_checkpoint, weights_only=True))
    print(f"Loaded BC checkpoint from {args.bc_checkpoint}")

    checkpoint_path = str(cache_dir / f"checkpoint_awr_{args.mode}.pt")
    if args.reward_scale != 1.0:
        print(f"Reward scale: {args.reward_scale}x (returns will be in [{-args.reward_scale:.0f}, {args.reward_scale:.0f}])")

    model = train_awr(
        model, loader,
        epochs=args.epochs, lr=args.lr,
        temp_start=args.temp_start, temp_end=args.temp_end,
        device=device, checkpoint_path=checkpoint_path,
        reward_scale=args.reward_scale,
    )
    print(f"AWR training complete. Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
