"""Build SB3 MaskablePPO submission tarball.

Includes ppo_gnn/ modules directly — single source of truth, no copy-paste drift.

Usage:
    python scripts/build_sb3_submission.py
    python scripts/build_sb3_submission.py --weights ppo_gnn/cache/sb3_best_model.zip
    python scripts/build_sb3_submission.py --output ~/Desktop/submission_sb3_ppo.tar.gz
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def export_weights(sb3_zip_path: str, output_dir: Path):
    """Export SB3 MaskablePPO weights to standalone .pt file + VecNormalize stats."""
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import VecNormalize

    print(f"Loading SB3 model from: {sb3_zip_path}")
    model = MaskablePPO.load(sb3_zip_path, device="cpu")

    # Export policy state_dict
    weights_path = output_dir / "sb3_policy_weights.pt"
    state = model.policy.state_dict()
    # Only keep keys needed for inference (features_extractor + mlp + action_net)
    inference_keys = {k: v for k, v in state.items()
                     if k.startswith(("features_extractor.", "mlp_extractor.policy_net.", "action_net."))}
    torch.save(inference_keys, weights_path)
    print(f"  Saved {len(inference_keys)} weight tensors to {weights_path}")

    # Export VecNormalize stats
    vec_norm_path = Path(sb3_zip_path).parent / "sb3_vec_normalize.pkl"
    if vec_norm_path.exists():
        import pickle
        with open(vec_norm_path, "rb") as f:
            vec_norm = pickle.load(f)
        obs_rms = vec_norm.obs_rms
        stats_path = output_dir / "vec_normalize_stats.npz"
        np.savez(stats_path,
                 mean=obs_rms.mean,
                 var=obs_rms.var,
                 clip_obs=np.array([vec_norm.clip_obs]))
        print(f"  Saved VecNormalize stats to {stats_path}")
    else:
        print(f"  WARNING: No VecNormalize found at {vec_norm_path}")
        # Try cache dir
        alt_path = output_dir / "vec_normalize_stats.npz"
        if alt_path.exists():
            print(f"  Using existing {alt_path}")
        else:
            print(f"  No VecNormalize stats available!")


def build_tarball(root: Path, output: Path, weights_dir: Path):
    """Assemble the submission tarball."""
    with tarfile.open(output, "w:gz") as tar:
        # main.py entry point
        tar.add(root / "submission/main_sb3_ppo.py", arcname="main.py")

        # ppo_gnn package (inference-needed files only)
        tar.add(root / "ppo_gnn/__init__.py", arcname="ppo_gnn/__init__.py")
        tar.add(root / "ppo_gnn/sun_geometry.py", arcname="ppo_gnn/sun_geometry.py")
        tar.add(root / "ppo_gnn/edge_policy.py", arcname="ppo_gnn/edge_policy.py")
        tar.add(root / "ppo_gnn/sb3_constants.py", arcname="ppo_gnn/sb3_constants.py")

        # Model weights and normalization stats
        weights_pt = weights_dir / "sb3_policy_weights.pt"
        norm_npz = weights_dir / "vec_normalize_stats.npz"
        if weights_pt.exists():
            tar.add(weights_pt, arcname="sb3_policy_weights.pt")
        else:
            raise FileNotFoundError(f"Missing weights: {weights_pt}")
        if norm_npz.exists():
            tar.add(norm_npz, arcname="vec_normalize_stats.npz")
        else:
            print(f"  WARNING: {norm_npz} not found, submission will run without normalization")

    size_kb = output.stat().st_size / 1024
    print(f"\nBuilt: {output} ({size_kb:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Build SB3 submission tarball")
    parser.add_argument("--weights", type=str, default="ppo_gnn/cache/sb3_best_model.zip",
                        help="Path to SB3 model .zip (will export weights from it)")
    parser.add_argument("--output", type=str, default="submission_sb3_ppo.tar.gz",
                        help="Output tarball path")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip weight export, use existing .pt and .npz in cache/")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    output = Path(args.output) if "/" in args.output else root / args.output
    weights_dir = root / "ppo_gnn" / "cache"

    if not args.skip_export:
        export_weights(args.weights, weights_dir)

    build_tarball(root, output, weights_dir)

    # Verification hint
    print(f"\nTo test locally:")
    print(f"  mkdir -p /tmp/test_sub && tar xzf {output} -C /tmp/test_sub")
    print(f"  cd /tmp/test_sub && python -c \"from main import agent; print('OK')\"")


if __name__ == "__main__":
    main()
