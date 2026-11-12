"""SB3 MaskablePPO training for Orbit Wars.

Uses heuristic-filtered 192 candidates × 10 fractions = Discrete(1920) action space.

Usage:
    python -m ppo_gnn.train_sb3
    python -m ppo_gnn.train_sb3 --total-timesteps 2000000 --device mps
    python -m ppo_gnn.train_sb3 --checkpoint ppo_gnn/cache/sb3_best_model.zip
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KAGGLE_ENVIRONMENTS_QUIET", "1")


def main():
    # Disable PyTorch distribution validation — 481-dim uniform triggers false
    # Simplex constraint failures due to float32 precision in PyTorch 2.9+
    import torch
    torch.distributions.Distribution.set_default_validate_args(False)

    parser = argparse.ArgumentParser(description="SB3 MaskablePPO for Orbit Wars")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--start-tier", type=int, default=99, help="Max tier (99=all opponents)")
    parser.add_argument("--min-tier", type=int, default=1, help="Floor tier — skip easy opponents")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint .zip")
    parser.add_argument("--bc-weights", type=str, default=None, help="Load BC pre-trained weights (.pt) into fresh model")
    parser.add_argument("--n-steps", type=int, default=8192, help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-freq", type=int, default=80000, help="Eval every N steps (~200 episodes)")
    parser.add_argument("--max-steps", type=int, default=500, help="Max game steps (shorter = faster credit assignment)")
    parser.add_argument("--mode", type=str, default="2p", choices=["2p", "mixed"])
    parser.add_argument("--log-path", type=str, default="ppo_gnn/cache/sb3_train.log")
    args = parser.parse_args()

    # Lazy imports so argparse --help is fast
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from .sb3_env import OrbitWarsEnv
    from .sb3_callbacks import WinRateCallback, EvalCallback, CheckpointCallback, SelfPlayCallback
    from .sb3_feature_extractor import CandidateTransformerExtractor
    from .sb3_policy import PointerNetPolicy
    from .train_ppo_edge import build_opponent_pool, load_agent_from_file

    # --- Build opponent pool ---
    print("Loading opponent pool...")
    pool = build_opponent_pool(args.mode)
    print(f"  {len(pool)} opponents loaded")

    # --- Build eval opponents ---
    eval_opponents = []
    root = Path(__file__).parent.parent
    from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent
    eval_opponents.append(("random", random_agent))

    eval_candidates = [
        ("bully", "submission/pool_bully.py"),
        ("rage", "submission/pool_rage.py"),
        ("prospector", "submission/pool_prospector.py"),
        ("dual", "submission/pool_dual.py"),
        ("nearest_sniper", "submission/ext/pool_baseline_nearest_sniper.py"),
        ("baseline", "submission/pool_baseline.py"),
        ("starter", None),
    ]
    for name, rel_path in eval_candidates:
        if rel_path is None:
            from kaggle_environments.envs.orbit_wars.orbit_wars import starter_agent
            eval_opponents.append((name, starter_agent))
        else:
            full_path = root / rel_path
            if full_path.exists():
                try:
                    eval_opponents.append((name, load_agent_from_file(str(full_path))))
                except Exception as e:
                    print(f"  Skipped eval opponent {name}: {e}")

    print(f"  {len(eval_opponents)} eval opponents")

    # --- Create environment ---
    def make_env():
        return OrbitWarsEnv(
            opponent_pool=pool,
            mode=args.mode,
            max_tier=args.start_tier,
            min_tier=args.min_tier,
            max_steps=args.max_steps,
        )

    env = DummyVecEnv([make_env])

    # --- Create or load model ---
    vec_norm_path = Path("ppo_gnn/cache/sb3_vec_normalize.pkl")
    if args.checkpoint and vec_norm_path.exists():
        print(f"Loading VecNormalize stats from {vec_norm_path}")
        env = VecNormalize.load(str(vec_norm_path), env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model = MaskablePPO.load(
            args.checkpoint,
            env=env,
            device=args.device,
        )
        model.learning_rate = args.lr
        model.ent_coef = args.ent_coef
    else:
        model = MaskablePPO(
            PointerNetPolicy,
            env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.997,
            gae_lambda=0.98,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                features_extractor_class=CandidateTransformerExtractor,
                features_extractor_kwargs=dict(
                    features_dim=256,
                    d_model=256,
                    n_heads=8,
                    n_layers=6,
                    pool=False,  # Return per-candidate embeddings
                ),
                vf_features_extractor_kwargs=dict(
                    features_dim=128,
                    d_model=128,
                    n_heads=4,
                    n_layers=2,
                ),
                share_features_extractor=False,
                net_arch=[],  # No MLP extractor — handled by custom policy
            ),
            verbose=1,
            device=args.device,
        )

    # --- Load BC pre-trained weights ---
    if args.bc_weights:
        bc_state = torch.load(args.bc_weights, map_location=args.device, weights_only=True)
        policy_state = model.policy.state_dict()

        loaded = 0
        for k, v in bc_state.items():
            if k in policy_state:
                policy_state[k] = v
                loaded += 1
            # Copy features_extractor weights to pi_features_extractor
            if k.startswith("features_extractor."):
                pi_key = "pi_" + k
                if pi_key in policy_state:
                    policy_state[pi_key] = v
                    loaded += 1

        model.policy.load_state_dict(policy_state)
        print(f"Loaded {loaded} BC weight tensors into policy")

    log_msg = (
        f"SB3 MaskablePPO training — {args.mode} mode\n"
        f"Total timesteps: {args.total_timesteps}, n_steps: {args.n_steps}\n"
        f"LR: {args.lr}, ent_coef: {args.ent_coef}, device: {args.device}\n"
    )
    print(log_msg)
    if args.log_path:
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        with open(args.log_path, "w") as f:
            f.write(log_msg)

    # --- Callbacks ---
    callbacks = [
        WinRateCallback(
            window=50,
            promotion_threshold=0.70,
            demotion_threshold=0.30,
            log_path=args.log_path,
        ),
        EvalCallback(
            eval_opponents=eval_opponents,
            eval_freq=args.eval_freq,
            n_eval_episodes=2,
            save_path="ppo_gnn/cache",
            log_path=args.log_path,
        ),
        SelfPlayCallback(
            snapshot_freq=50,
            max_archive=5,
            self_play_tier=7,
            log_path=args.log_path,
        ),
        CheckpointCallback(
            save_freq=50000,
            save_path="ppo_gnn/cache",
        ),
    ]

    # --- Train ---
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")

    # Save final model
    model.save("ppo_gnn/cache/sb3_final_model")
    env.save("ppo_gnn/cache/sb3_vec_normalize.pkl")
    print("Saved final model and VecNormalize stats.")


if __name__ == "__main__":
    main()
