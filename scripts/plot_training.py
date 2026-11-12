#!/usr/bin/env python3
"""
Plot PPO training progression from a log file.

Usage:
    python plot_training.py [log_file]

Defaults to: ppo_gnn/cache/ppo_mixed_bc_v3.log
"""

import re
import sys
import pathlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Parse ──────────────────────────────────────────────────────────────────────

LOG_DEFAULT = pathlib.Path("ppo_gnn/cache/ppo_mixed_bc_v3.log")

UPDATE_RE = re.compile(
    r"Update:.*?"
    r"policy=(?P<policy>[+-]?\d+\.\d+).*?"
    r"value=(?P<value>\d+\.\d+).*?"
    r"entropy=(?P<entropy>\d+\.\d+).*?"
    r"kl=(?P<kl>\d+\.\d+).*?"
    r"clip=(?P<clip>\d+\.\d+).*?"
    r"wr\(50\)=(?P<wr50>\d+\.\d+)%.*?"
    r"heur_wr=(?P<heur_wr>\d+\.\d+)%.*?"
    r"avg_r=(?P<avg_r>[+-]\d+\.\d+).*?"
    r"base_wr=(?P<base_wr>\d+)%.*?"
    r"lag_wr=(?P<lag_wr>\d+)%.*?"
    r"lr=(?P<lr>[\d.e+-]+)"
)

CHECKPOINT_RE = re.compile(r"Saved best checkpoint \(heuristic_wr=(\d+\.\d+)%\)")
STAGE_RE      = re.compile(r"stage=(\S+)")


def parse_log(path: pathlib.Path):
    rows, checkpoints, stages = [], [], []
    with path.open() as f:
        for line in f:
            m = UPDATE_RE.search(line)
            if m:
                d = {k: float(v) for k, v in m.groupdict().items()}
                sm = STAGE_RE.search(line)
                stages.append(sm.group(1) if sm else "")
                rows.append(d)
            cm = CHECKPOINT_RE.search(line)
            if cm:
                checkpoints.append((len(rows) - 1, float(cm.group(1))))
    return rows, checkpoints, stages


def smooth(values, w=5):
    """Simple moving average."""
    out = []
    for i, v in enumerate(values):
        sl = values[max(0, i - w + 1): i + 1]
        out.append(sum(sl) / len(sl))
    return out


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot(log_path: pathlib.Path):
    rows, checkpoints, stages = parse_log(log_path)
    if not rows:
        print("No Update lines found — check the log path.")
        return

    n      = len(rows)
    steps  = list(range(1, n + 1))
    wr50   = [r["wr50"]   for r in rows]
    heur   = [r["heur_wr"] for r in rows]
    base   = [r["base_wr"] for r in rows]
    lag    = [r["lag_wr"]  for r in rows]
    avg_r  = [r["avg_r"]   for r in rows]
    policy = [r["policy"]  for r in rows]
    value  = [r["value"]   for r in rows]
    entropy= [r["entropy"] for r in rows]
    kl     = [r["kl"]      for r in rows]
    clip   = [r["clip"]    for r in rows]
    lr     = [r["lr"]      for r in rows]

    fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True)
    fig.suptitle(f"PPO Training — {log_path.name}  ({n} updates)", fontsize=13, fontweight="bold")

    def ax_plot(ax, ys, label, color, ylabel, ylim=None, smooth_w=0):
        vals = smooth(ys, smooth_w) if smooth_w else ys
        ax.plot(steps, vals, color=color, lw=1.4, label=label)
        if smooth_w:
            ax.plot(steps, ys, color=color, lw=0.4, alpha=0.3)
        for idx, wr_val in checkpoints:
            ax.axvline(idx + 1, color="gold", lw=1, alpha=0.7, linestyle="--")
        ax.set_ylabel(ylabel, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.25)

    # Row 0: win rates
    ax = axes[0, 0]
    ax.plot(steps, wr50,  color="steelblue",  lw=1.5, label="wr(50) — rolling vs curriculum")
    ax.plot(steps, heur,  color="darkorange",  lw=1.5, label="heur_wr — vs heuristic agents")
    ax.plot(steps, base,  color="green",       lw=1.0, label="base_wr — vs baseline", alpha=0.7)
    ax.plot(steps, lag,   color="red",         lw=1.0, label="lag_wr  — vs lagging",  alpha=0.7)
    for idx, _ in checkpoints:
        ax.axvline(idx + 1, color="gold", lw=1, alpha=0.7, linestyle="--", label="_")
    ax.set_ylabel("Win rate (%)", fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.25)
    ax.set_title("Win Rates", fontsize=10)

    # Row 0: avg reward
    ax_plot(axes[0, 1], avg_r, "avg_reward", "purple", "Avg reward", smooth_w=5)
    axes[0, 1].axhline(0, color="black", lw=0.5, linestyle=":")
    axes[0, 1].set_title("Average Reward (smoothed)", fontsize=10)

    # Row 1: policy + value loss
    ax_plot(axes[1, 0], policy, "policy loss", "steelblue", "Policy loss", smooth_w=5)
    axes[1, 0].axhline(0, color="black", lw=0.5, linestyle=":")
    axes[1, 0].set_title("Policy Loss (smoothed)", fontsize=10)

    ax_plot(axes[1, 1], value, "value loss", "tomato", "Value loss", smooth_w=5)
    axes[1, 1].set_title("Value Loss (smoothed)", fontsize=10)

    # Row 2: entropy + KL
    ax_plot(axes[2, 0], entropy, "entropy", "teal", "Entropy", smooth_w=3)
    axes[2, 0].set_title("Entropy", fontsize=10)

    ax_plot(axes[2, 1], kl, "KL divergence", "darkorange", "KL div", smooth_w=3)
    axes[2, 1].set_title("KL Divergence", fontsize=10)

    # Row 3: clip fraction + LR
    ax_plot(axes[3, 0], clip, "clip fraction", "slategray", "Clip frac", smooth_w=3)
    axes[3, 0].set_title("Clip Fraction", fontsize=10)
    axes[3, 0].set_xlabel("Update step", fontsize=9)

    ax = axes[3, 1]
    ax.plot(steps, lr, color="black", lw=1.2, label="learning rate")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    for idx, _ in checkpoints:
        ax.axvline(idx + 1, color="gold", lw=1, alpha=0.7, linestyle="--")
    ax.set_ylabel("LR (log scale)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_title("Learning Rate", fontsize=10)
    ax.set_xlabel("Update step", fontsize=9)

    # Stage change annotations on top plot
    prev_stage = ""
    for i, stage in enumerate(stages):
        if stage != prev_stage:
            axes[0, 0].axvline(i + 1, color="gray", lw=0.8, linestyle=":", alpha=0.6)
            axes[0, 0].text(i + 1, 2, stage.split("(")[0], fontsize=6,
                            rotation=90, va="bottom", color="gray")
            prev_stage = stage

    plt.tight_layout()
    out_path = log_path.with_suffix(".png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    log_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else LOG_DEFAULT
    if not log_path.exists():
        print(f"Log not found: {log_path}")
        sys.exit(1)
    plot(log_path)
