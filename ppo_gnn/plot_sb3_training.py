"""Parse sb3_train.log and plot training metrics.

Run once for a snapshot:
    python -m ppo_gnn.plot_sb3_training

Run with --watch to auto-refresh every 60s:
    python -m ppo_gnn.plot_sb3_training --watch --interval 60
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

LOG_PATH = Path(__file__).parent / "cache" / "sb3_train.log"
OUT_PATH = Path(__file__).parent / "cache" / "sb3_training_progress.png"

# Matches SB3 log lines like:
# Update ep=380 (1892s) | wr=0.857 W=319 L=61 avg_r=+3.21 | pol=-0.09 val=0.04 ent=22.89 kl=0.11 clip=0.61 ev=0.84 | lr=1.00e-04 | steps=59883 | tier=9 | acts=88 noop=61% planets=19.9 | early: fleet@11 p25=2.6 p50=6.5
UPDATE_RE = re.compile(
    r"Update ep=(?P<ep>\d+)\s+\((?P<elapsed>\d+)s\)\s+\|\s+"
    r"wr=(?P<wr>[\d.]+)\s+W=(?P<wins>\d+)\s+L=(?P<losses>\d+)"
    r"(?:\s+avg_r=(?P<avg_r>[+-]?[\d.]+))?"  # optional — added mid-run
    r"\s+\|\s+"
    r"pol=(?P<pol>[+-]?[\d.]+)\s+val=(?P<val>[\d.]+)\s+ent=(?P<ent>[\d.]+)\s+"
    r"kl=(?P<kl>[\d.]+)\s+clip=(?P<clip>[\d.]+)\s+ev=(?P<ev>[+-]?[\d.]+)\s+\|\s+"
    r"lr=(?P<lr>[\d.e+-]+)\s+\|\s+steps=(?P<steps>\d+)\s+\|\s+tier=(?P<tier>[\d-]+)\s+\|\s+"
    r"acts=(?P<acts>\d+)\s+noop=(?P<noop>\d+)%\s+planets=(?P<planets>[\d.]+)"
)

PROMO_RE = re.compile(r"\*\*\* PROMOTED to tier (\d+)")
DEMO_RE = re.compile(r"\*\*\* DEMOTED to tier (\d+)")


def parse_log(path: Path) -> dict[str, list]:
    data: dict[str, list] = {
        "ep": [], "elapsed": [], "wr": [], "wins": [], "losses": [],
        "avg_r": [], "pol": [], "val": [], "ent": [], "kl": [], "clip": [],
        "ev": [], "lr": [], "steps": [], "tier": [],
        "acts": [], "noop": [], "planets": [],
        "promotions": [], "demotions": [],
    }
    for line in path.read_text().splitlines():
        m = UPDATE_RE.search(line)
        if m:
            data["ep"].append(int(m.group("ep")))
            data["elapsed"].append(int(m.group("elapsed")))
            data["wr"].append(float(m.group("wr")))
            data["wins"].append(int(m.group("wins")))
            data["losses"].append(int(m.group("losses")))
            data["avg_r"].append(float(m.group("avg_r")) if m.group("avg_r") else 0.0)
            data["pol"].append(float(m.group("pol")))
            data["val"].append(float(m.group("val")))
            data["ent"].append(float(m.group("ent")))
            data["kl"].append(float(m.group("kl")))
            data["clip"].append(float(m.group("clip")))
            data["ev"].append(float(m.group("ev")))
            data["lr"].append(float(m.group("lr")))
            data["steps"].append(int(m.group("steps")))
            data["tier"].append(int(m.group("tier")))
            data["acts"].append(int(m.group("acts")))
            data["noop"].append(int(m.group("noop")))
            data["planets"].append(float(m.group("planets")))
            continue

        pm = PROMO_RE.search(line)
        if pm:
            data["promotions"].append((data["ep"][-1] if data["ep"] else 0, int(pm.group(1))))
        dm = DEMO_RE.search(line)
        if dm:
            data["demotions"].append((data["ep"][-1] if data["ep"] else 0, int(dm.group(1))))

    return data


def smooth(vals: list[float], window: int = 10) -> list[float]:
    if len(vals) < window:
        return vals
    return list(np.convolve(vals, np.ones(window) / window, mode="valid"))


def plot(data: dict[str, list], out: Path) -> None:
    n = len(data["ep"])
    if n == 0:
        print("No update lines found yet.")
        return

    eps = data["ep"]
    steps = data["steps"]
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(
        f"SB3 MaskablePPO Training  |  {n} updates / ep={eps[-1]} / "
        f"{steps[-1]:,} steps / tier={data['tier'][-1]}  |  "
        f"{data['elapsed'][-1] // 60}min elapsed",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Win rate + Tier ─────────────────────────────────────────────
    ax = axes[0]
    ax.set_ylabel("Win rate", fontsize=10, color="#4c8cff")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    ax.plot(eps, data["wr"], alpha=0.3, color="#4c8cff", lw=0.6)
    s = smooth(data["wr"])
    ax.plot(eps[len(eps) - len(s):], s, label="Win rate (smoothed)", color="#4c8cff", lw=2.0)
    ax.axhline(0.70, color="#22bb66", ls="--", lw=0.8, alpha=0.5, label="Promote (70%)")
    ax.axhline(0.30, color="#ff4c4c", ls="--", lw=0.8, alpha=0.5, label="Demote (30%)")

    # Tier on right axis
    ax2 = ax.twinx()
    ax2.set_ylabel("Tier", fontsize=10, color="#ff9900")
    ax2.plot(eps, data["tier"], color="#ff9900", lw=1.5, alpha=0.7, label="Tier")
    ax2.set_ylim(0, max(data["tier"]) + 2)

    # Promotions/demotions markers
    for ep_val, tier_val in data["promotions"]:
        ax.axvline(ep_val, color="#22bb66", ls=":", lw=0.8, alpha=0.5)
    for ep_val, tier_val in data["demotions"]:
        ax.axvline(ep_val, color="#ff4c4c", ls=":", lw=0.8, alpha=0.5)

    ax.legend(fontsize=8, loc="lower left")
    ax2.legend(fontsize=8, loc="lower right")

    # ── Panel 2: Avg reward + Planets ────────────────────────────────────────
    ax = axes[1]
    ax.set_ylabel("Avg reward", fontsize=10, color="#22bb66")
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.plot(eps, data["avg_r"], alpha=0.3, color="#22bb66", lw=0.6)
    s = smooth(data["avg_r"])
    ax.plot(eps[len(eps) - len(s):], s, color="#22bb66", lw=1.5, label="avg_r (smoothed)")

    ax2 = ax.twinx()
    ax2.set_ylabel("Planets", fontsize=10, color="#888888")
    ax2.plot(eps, data["planets"], alpha=0.3, color="#888888", lw=0.6)
    s = smooth(data["planets"])
    ax2.plot(eps[len(eps) - len(s):], s, color="#888888", lw=1.5, label="Planets (smoothed)")

    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="upper right")

    # ── Panel 3: KL + Clip + Entropy ─────────────────────────────────────────
    ax = axes[2]
    ax.set_ylabel("KL / Clip", fontsize=10)

    ax.plot(eps, data["kl"], alpha=0.25, color="#ff9900", lw=0.6)
    s = smooth(data["kl"])
    ax.plot(eps[len(eps) - len(s):], s, color="#ff9900", lw=1.5, label="KL (smoothed)")

    ax.plot(eps, data["clip"], alpha=0.25, color="#ff4c4c", lw=0.6)
    s = smooth(data["clip"])
    ax.plot(eps[len(eps) - len(s):], s, color="#ff4c4c", lw=1.2, label="Clip frac (smoothed)")

    ax2 = ax.twinx()
    ax2.set_ylabel("Entropy", fontsize=10, color="#aa44ff")
    ax2.plot(eps, data["ent"], alpha=0.25, color="#aa44ff", lw=0.6)
    s = smooth(data["ent"])
    ax2.plot(eps[len(eps) - len(s):], s, color="#aa44ff", lw=1.5, label="Entropy (smoothed)")

    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="upper right")

    # ── Panel 4: Policy loss + Value loss + EV ───────────────────────────────
    ax = axes[3]
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)

    ax.plot(eps, data["pol"], alpha=0.25, color="#4c8cff", lw=0.6)
    s = smooth(data["pol"])
    ax.plot(eps[len(eps) - len(s):], s, color="#4c8cff", lw=1.5, label="Policy loss (smoothed)")

    ax.plot(eps, data["val"], alpha=0.25, color="#ff4c4c", lw=0.6)
    s = smooth(data["val"])
    ax.plot(eps[len(eps) - len(s):], s, color="#ff4c4c", lw=1.2, label="Value loss (smoothed)")

    ax2 = ax.twinx()
    ax2.set_ylabel("Explained Var", fontsize=10, color="#22bb66")
    ax2.set_ylim(-0.1, 1.1)
    ax2.plot(eps, data["ev"], alpha=0.25, color="#22bb66", lw=0.6)
    s = smooth(data["ev"])
    ax2.plot(eps[len(eps) - len(s):], s, color="#22bb66", lw=1.5, label="EV (smoothed)")

    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="upper right")

    # Noop % + actions as text annotation in bottom
    ax.text(
        0.01, -0.15,
        f"Latest: acts={data['acts'][-1]}, noop={data['noop'][-1]}%, "
        f"lr={data['lr'][-1]:.1e}",
        transform=ax.transAxes, fontsize=8, color="gray",
    )

    plt.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}  ({n} updates, ep={eps[-1]}, {steps[-1]:,} steps)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=str(LOG_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--watch", action="store_true", help="Auto-refresh")
    parser.add_argument("--interval", type=int, default=60, help="Refresh interval (s)")
    args = parser.parse_args()

    log = Path(args.log)
    out = Path(args.out)

    if args.watch:
        print(f"Watching {log} -- refreshing every {args.interval}s. Ctrl-C to stop.")
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
