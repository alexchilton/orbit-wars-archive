#!/usr/bin/env python3
"""Live progress plotter for PPO multi-action training runs.

Usage:
    python3 ppo_gnn/plot_progress.py              # one-shot plot
    python3 ppo_gnn/plot_progress.py --live 30    # refresh every 30s
"""

import argparse
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


LOG_DIR = Path(__file__).parent / "cache"
LOG_FILES = {
    "fresh": LOG_DIR / "ppo_multi_action_fresh.log",
    "bc": LOG_DIR / "ppo_multi_action_bc.log",
}

UPDATE_RE = re.compile(
    r"Update:.*?"
    r"entropy=([\d.]+)\s+"
    r"kl=([\d.]+)\s+"
    r"clip=([\d.]+)\s+\|\s+"
    r"wr\(50\)=([\d.]+)%\s+"
    r"heur_wr=([\d.]+)%\s+"
    r"avg_r=([\+\-\d.]+)\s+"
    r".*?lr=([\d.e\+\-]+)\s+"
    r"stage=(\w+)"
)

EP_RE = re.compile(
    r"Ep\s+(\d+)/\d+\s+vs\s+\[(\dp)\](\S+)\s+(WIN|LOSS)\s+"
    r"steps=(\d+).*?launch=(\d+)\s+noop=(\d+)"
)


def parse_log(path: Path):
    updates = []
    episodes = []
    if not path.exists():
        return updates, episodes
    for line in path.read_text().splitlines():
        m = UPDATE_RE.search(line)
        if m:
            updates.append({
                "entropy": float(m.group(1)),
                "kl": float(m.group(2)),
                "clip": float(m.group(3)),
                "wr": float(m.group(4)),
                "heur_wr": float(m.group(5)),
                "avg_r": float(m.group(6)),
                "lr": float(m.group(7)),
                "stage": m.group(8),
            })
        m = EP_RE.search(line)
        if m:
            episodes.append({
                "ep": int(m.group(1)),
                "mode": m.group(2),
                "opp": m.group(3),
                "win": m.group(4) == "WIN",
                "steps": int(m.group(5)),
                "launch": int(m.group(6)),
                "noop": int(m.group(7)),
            })
    return updates, episodes


def rolling_wr(episodes, window=20):
    """Compute rolling win rate over a sliding window."""
    wins = []
    wr = []
    for ep in episodes:
        wins.append(1.0 if ep["win"] else 0.0)
        start = max(0, len(wins) - window)
        wr.append(sum(wins[start:]) / len(wins[start:]) * 100)
    return wr


def plot(fig, axes):
    for ax in axes.flat:
        ax.clear()

    colors = {"fresh": "#2196F3", "bc": "#FF5722"}
    has_data = False

    for name, path in LOG_FILES.items():
        updates, episodes = parse_log(path)
        if not updates:
            continue
        has_data = True
        c = colors[name]
        x = list(range(len(updates)))

        # Win rate
        axes[0, 0].plot(x, [u["wr"] for u in updates], color=c, label=name, linewidth=1.5)
        axes[0, 0].set_title("Win Rate (50-game)")
        axes[0, 0].set_ylabel("%")
        axes[0, 0].axhline(85, color="gray", linestyle="--", alpha=0.5, label="floor gate" if name == "fresh" else "")

        # Avg reward
        axes[0, 1].plot(x, [u["avg_r"] for u in updates], color=c, label=name, linewidth=1.5)
        axes[0, 1].set_title("Avg Reward")

        # Entropy
        axes[1, 0].plot(x, [u["entropy"] for u in updates], color=c, label=name, linewidth=1.5)
        axes[1, 0].set_title("Entropy")

        # KL divergence
        axes[1, 1].plot(x, [u["kl"] for u in updates], color=c, label=name, linewidth=1.5)
        axes[1, 1].set_title("KL Divergence")
        axes[1, 1].axhline(0.05, color="gray", linestyle="--", alpha=0.5)

        # LR
        axes[2, 0].plot(x, [u["lr"] for u in updates], color=c, label=name, linewidth=1.5)
        axes[2, 0].set_title("Learning Rate")
        axes[2, 0].set_yscale("log")

        # Launches per episode (from episode data)
        if episodes:
            ep_x = [e["ep"] for e in episodes]
            axes[2, 1].plot(ep_x, [e["launch"] for e in episodes], color=c, alpha=0.3, linewidth=0.5)
            # Rolling average
            window = 20
            launches = [e["launch"] for e in episodes]
            rolling = []
            for i in range(len(launches)):
                s = max(0, i - window)
                rolling.append(sum(launches[s:i+1]) / (i - s + 1))
            axes[2, 1].plot(ep_x, rolling, color=c, label=name, linewidth=1.5)
            axes[2, 1].set_title("Launches/Episode")

    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[2, 0].set_xlabel("Update #")
    axes[2, 1].set_xlabel("Episode")

    if not has_data:
        axes[0, 0].text(0.5, 0.5, "No data yet", ha="center", va="center", transform=axes[0, 0].transAxes)

    fig.suptitle("PPO Multi-Action Training Progress", fontsize=14, fontweight="bold")
    fig.tight_layout()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", type=int, default=0, help="Refresh interval in seconds (0 = one-shot)")
    args = parser.parse_args()

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    if args.live > 0:
        plt.ion()
        while True:
            plot(fig, axes)
            plt.draw()
            plt.pause(args.live)
    else:
        plot(fig, axes)
        plt.show()


if __name__ == "__main__":
    main()
