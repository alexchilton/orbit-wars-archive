"""Parse ppo_retrain.log and plot training metrics.

Run once for a snapshot:
    python -m ppo_gnn.plot_training

Run with --watch to auto-refresh every 60s:
    python -m ppo_gnn.plot_training --watch --interval 60
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

LOG_PATH = Path(__file__).parent / "cache" / "ppo_retrain.log"
OUT_PATH  = Path(__file__).parent / "cache" / "training_progress.png"

# Matches lines like:
# Update: policy=... | wr(50)=54.0% heur_wr=0.0% avg_r=+1.88 base_wr=75% lag_wr=0% stage=warmup horizon=50
UPDATE_RE = re.compile(
    r"Update:.*?"
    r"entropy=(?P<entropy>[\d.]+).*?"
    r"kl=(?P<kl>[\d.]+).*?"
    r"wr\(50\)=(?P<wr50>[\d.]+)%.*?"
    r"heur_wr=(?P<heur_wr>[\d.]+)%.*?"
    r"avg_r=(?P<avg_r>[+-]?[\d.]+).*?"
    r"base_wr=(?P<base_wr>[\d.]+)%.*?"
    r"lag_wr=(?P<lag_wr>[\d.]+)%.*?"
    r"stage=(?P<stage>\w+).*?"
    r"horizon=(?P<horizon>\d+)"
)


def parse_log(path: Path) -> dict[str, list]:
    data: dict[str, list] = {
        "update": [], "wr50": [], "heur_wr": [], "avg_r": [],
        "base_wr": [], "lag_wr": [], "entropy": [], "kl": [], "horizon": [],
    }
    for line in path.read_text().splitlines():
        m = UPDATE_RE.search(line)
        if not m:
            continue
        data["update"].append(len(data["update"]) + 1)
        data["wr50"].append(float(m.group("wr50")))
        data["heur_wr"].append(float(m.group("heur_wr")))
        data["avg_r"].append(float(m.group("avg_r")))
        data["base_wr"].append(float(m.group("base_wr")))
        data["lag_wr"].append(float(m.group("lag_wr")))
        data["entropy"].append(float(m.group("entropy")))
        data["kl"].append(float(m.group("kl")))
        data["horizon"].append(int(m.group("horizon")))
    return data


def smooth(vals: list[float], window: int = 20) -> list[float]:
    if len(vals) < window:
        return vals
    return list(np.convolve(vals, np.ones(window) / window, mode="valid"))


def plot(data: dict[str, list], out: Path) -> None:
    n = len(data["update"])
    if n == 0:
        print("No update lines found yet.")
        return

    eps = [u * 4 for u in data["update"]]  # 4 episodes per update
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(
        f"PPO-GNN Training  —  {n} updates / ~{eps[-1]} episodes  "
        f"(horizon={data['horizon'][-1]})",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Win rates ────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_ylabel("Win rate", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    for key, label, color, lw in [
        ("wr50",     "Overall wr(50)",  "#4c8cff", 1.2),
        ("heur_wr",  "Heuristic wr",    "#ff4c4c", 2.0),
        ("base_wr",  "vs Baseline",     "#888888", 1.0),
        ("lag_wr",   "vs Lagging",      "#aaaaaa", 1.0),
    ]:
        raw = [v / 100 for v in data[key]]
        ax.plot(eps, raw, alpha=0.25, color=color, lw=0.6)
        s = smooth(raw)
        ax.plot(eps[len(eps) - len(s):], s, label=label, color=color, lw=lw)

    ax.axhline(0.25, color="#ff4c4c", ls="--", lw=0.8, alpha=0.4, label="25% target (stage advance)")
    ax.legend(fontsize=8, loc="upper left")

    # Shade horizon changes
    _shade_horizons(ax, eps, data["horizon"])

    # ── Panel 2: Avg reward ───────────────────────────────────────────────────
    ax = axes[1]
    ax.set_ylabel("Avg reward (4-ep batch)", fontsize=10)
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.plot(eps, data["avg_r"], alpha=0.25, color="#22bb66", lw=0.6)
    s = smooth(data["avg_r"])
    ax.plot(eps[len(eps) - len(s):], s, color="#22bb66", lw=1.5, label="avg_r (smoothed)")
    ax.legend(fontsize=8, loc="upper left")
    _shade_horizons(ax, eps, data["horizon"])

    # ── Panel 3: Entropy & KL ─────────────────────────────────────────────────
    ax = axes[2]
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Entropy / KL", fontsize=10)
    ax.plot(eps, data["entropy"], alpha=0.25, color="#aa44ff", lw=0.6)
    s = smooth(data["entropy"])
    ax.plot(eps[len(eps) - len(s):], s, color="#aa44ff", lw=1.5, label="Entropy (smoothed)")
    ax.plot(eps, data["kl"], alpha=0.2, color="#ff9900", lw=0.6)
    s = smooth(data["kl"])
    ax.plot(eps[len(eps) - len(s):], s, color="#ff9900", lw=1.2, label="KL (smoothed)")
    ax.legend(fontsize=8, loc="upper right")
    _shade_horizons(ax, eps, data["horizon"])

    plt.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}  ({n} updates, ~{eps[-1]} episodes)")


def _shade_horizons(ax, eps, horizons):
    """Add a subtle background shade when horizon changes."""
    if not horizons:
        return
    prev_h = horizons[0]
    shade = False
    start_ep = eps[0]
    for ep, h in zip(eps, horizons):
        if h != prev_h:
            if shade:
                ax.axvspan(start_ep, ep, color="#ffffaa", alpha=0.3)
            start_ep = ep
            shade = not shade
            prev_h = h
    if shade:
        ax.axvspan(start_ep, eps[-1], color="#ffffaa", alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",      default=str(LOG_PATH))
    parser.add_argument("--out",      default=str(OUT_PATH))
    parser.add_argument("--watch",    action="store_true", help="Auto-refresh")
    parser.add_argument("--interval", type=int, default=60, help="Refresh interval (s)")
    args = parser.parse_args()

    log = Path(args.log)
    out = Path(args.out)

    if args.watch:
        print(f"Watching {log} — refreshing every {args.interval}s. Ctrl-C to stop.")
        while True:
            try:
                data = parse_log(log)
                plot(data, out)
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        data = parse_log(log)
        plot(data, out)


if __name__ == "__main__":
    main()
