"""Custom SB3 callbacks for Orbit Wars training."""

from __future__ import annotations

import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class WinRateCallback(BaseCallback):
    """Track win rate from episode info dicts and manage curriculum tier.

    Logs in a format matching the old train_ppo_edge.py style:
    Update ep=8 (133s) | wr=0.125 W=1 L=7 | avg_r=-3.98 | ent=1.90 kl=0.05 clip=0.23 ev=0.97 | lr=3e-04 | steps=3262 | tier=2
      opponents: bully:0/2 | random:1/1
    """

    def __init__(
        self,
        window: int = 50,
        promotion_threshold: float = 0.70,
        demotion_threshold: float = 0.30,
        log_path: Optional[str] = None,
        log_every: int = 10,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.window = window
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold
        self.log_path = log_path
        self.log_every = log_every

        self.recent_wins: deque = deque(maxlen=window)
        self.total_wins = 0
        self.total_losses = 0
        self.total_episodes = 0
        self._start_time = None
        self._log_file = None
        self._opponent_stats: dict = {}  # name -> [wins, total]
        self._recent_actions: deque = deque(maxlen=window)
        self._recent_noops: deque = deque(maxlen=window)
        self._recent_planets: deque = deque(maxlen=window)
        self._recent_first_fleet: deque = deque(maxlen=window)
        self._recent_planets_25: deque = deque(maxlen=window)
        self._recent_planets_50: deque = deque(maxlen=window)
        self._recent_rewards: deque = deque(maxlen=window)
        self._recent_ships_per_fleet: deque = deque(maxlen=window)
        self._recent_captures: deque = deque(maxlen=window)
        self._recent_losses: deque = deque(maxlen=window)
        self._recent_model_noop_pct: deque = deque(maxlen=window)
        self._recent_vetoed_pct: deque = deque(maxlen=window)
        self._4p_wins = 0
        self._4p_total = 0
        self._last_logged_ep = 0

    def _on_training_start(self):
        self._start_time = time.time()
        if self.log_path:
            self._log_file = open(self.log_path, "a")

    def _log(self, msg: str):
        print(msg)
        if self._log_file:
            self._log_file.write(msg + "\n")
            self._log_file.flush()

    def _get_tier(self) -> int:
        try:
            venv = self.training_env
            while hasattr(venv, "venv"):
                venv = venv.venv
            return venv.envs[0].max_tier
        except (AttributeError, IndexError):
            return -1

    def _get_env(self):
        try:
            venv = self.training_env
            while hasattr(venv, "venv"):
                venv = venv.venv
            return venv.envs[0]
        except (AttributeError, IndexError):
            return None

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", np.array([]))
        infos = self.locals.get("infos", [])

        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            won = info.get("won", False)
            opp_name = info.get("opponent", "unknown")
            self.recent_wins.append(won)
            self.total_episodes += 1
            if won:
                self.total_wins += 1
            else:
                self.total_losses += 1

            # Track per-opponent stats
            if opp_name not in self._opponent_stats:
                self._opponent_stats[opp_name] = [0, 0]
            self._opponent_stats[opp_name][1] += 1
            if won:
                self._opponent_stats[opp_name][0] += 1

            # Track 4p games
            if info.get("is_4p", False):
                self._4p_total += 1
                if won:
                    self._4p_wins += 1

            # Track action stats
            self._recent_actions.append(info.get("episode_actions", 0))
            self._recent_noops.append(info.get("noop_pct", 0))
            self._recent_planets.append(info.get("my_planets", 0))
            self._recent_first_fleet.append(info.get("first_fleet_step", -1))
            self._recent_planets_25.append(info.get("planets_at_25", 0))
            self._recent_planets_50.append(info.get("planets_at_50", 0))
            self._recent_rewards.append(info.get("episode_reward", 0))
            self._recent_ships_per_fleet.append(info.get("ships_per_fleet", 0))
            self._recent_captures.append(info.get("episode_captures", 0))
            self._recent_losses.append(info.get("episode_losses", 0))
            self._recent_model_noop_pct.append(info.get("model_noop_pct", 0))
            self._recent_vetoed_pct.append(info.get("vetoed_pct", 0))

            # Log update line every N episodes
            if self.total_episodes - self._last_logged_ep >= self.log_every:
                self._last_logged_ep = self.total_episodes
                wr = sum(self.recent_wins) / len(self.recent_wins) if self.recent_wins else 0
                elapsed = time.time() - self._start_time
                tier = self._get_tier()

                # Pull SB3 training stats from logger if available
                sb3_log = {}
                try:
                    logger = self.model.logger
                    if hasattr(logger, "name_to_value"):
                        sb3_log = dict(logger.name_to_value)
                except Exception:
                    pass

                ent = sb3_log.get("train/entropy_loss", 0)
                kl = sb3_log.get("train/approx_kl", 0)
                clip = sb3_log.get("train/clip_fraction", 0)
                ev = sb3_log.get("train/explained_variance", 0)
                pol_loss = sb3_log.get("train/policy_gradient_loss", 0)
                val_loss = sb3_log.get("train/value_loss", 0)
                lr = sb3_log.get("train/learning_rate", 0)

                # Action stats
                avg_actions = sum(self._recent_actions) / max(len(self._recent_actions), 1)
                avg_noop_pct = sum(self._recent_noops) / max(len(self._recent_noops), 1)
                avg_planets = sum(self._recent_planets) / max(len(self._recent_planets), 1)

                # Early-game stats
                valid_ff = [x for x in self._recent_first_fleet if x >= 0]
                avg_first_fleet = sum(valid_ff) / max(len(valid_ff), 1) if valid_ff else -1
                avg_p25 = sum(self._recent_planets_25) / max(len(self._recent_planets_25), 1)
                avg_p50 = sum(self._recent_planets_50) / max(len(self._recent_planets_50), 1)
                avg_reward = sum(self._recent_rewards) / max(len(self._recent_rewards), 1)

                fourp_wr = self._4p_wins / max(self._4p_total, 1)
                avg_spf = sum(self._recent_ships_per_fleet) / max(len(self._recent_ships_per_fleet), 1)
                avg_captures = sum(self._recent_captures) / max(len(self._recent_captures), 1)
                avg_losses = sum(self._recent_losses) / max(len(self._recent_losses), 1)
                avg_model_noop = sum(self._recent_model_noop_pct) / max(len(self._recent_model_noop_pct), 1)
                avg_vetoed = sum(self._recent_vetoed_pct) / max(len(self._recent_vetoed_pct), 1)
                self._log(
                    f"Update ep={self.total_episodes} ({elapsed:.0f}s) | "
                    f"wr={wr:.3f} W={self.total_wins} L={self.total_losses} avg_r={avg_reward:+.2f} | "
                    f"4p={self._4p_total} 4pwr={fourp_wr:.0%} | "
                    f"pol={pol_loss:.4f} val={val_loss:.4f} ent={abs(ent):.4f} "
                    f"kl={kl:.4f} clip={clip:.3f} ev={ev:.3f} | "
                    f"lr={lr:.2e} | steps={self.num_timesteps} | tier={tier} | "
                    f"acts={avg_actions:.0f} noop={avg_noop_pct:.0f}% "
                    f"[model={avg_model_noop:.0f}% veto={avg_vetoed:.0f}%] "
                    f"planets={avg_planets:.1f} "
                    f"ships/fleet={avg_spf:.0f} cap={avg_captures:.1f} lost={avg_losses:.1f} | "
                    f"early: fleet@{avg_first_fleet:.0f} p25={avg_p25:.1f} p50={avg_p50:.1f}"
                )
                # Per-opponent breakdown
                opp_parts = []
                for oname in sorted(self._opponent_stats.keys()):
                    w, t = self._opponent_stats[oname]
                    opp_parts.append(f"{oname}:{w}/{t}")
                if opp_parts:
                    self._log(f"  opponents: {' | '.join(opp_parts)}")

            # Curriculum promotion/demotion
            if len(self.recent_wins) >= self.window:
                wr = sum(self.recent_wins) / len(self.recent_wins)
                env = self._get_env()
                if env is None:
                    continue

                if wr >= self.promotion_threshold:
                    env.set_tier(env.max_tier + 1)
                    self._log(
                        f"*** PROMOTED to tier {env.max_tier} "
                        f"(wr={wr:.2f} over {self.window} games)"
                    )
                    self.recent_wins.clear()

                elif wr <= self.demotion_threshold and env.max_tier > 2:
                    env.set_tier(env.max_tier - 1)
                    self._log(
                        f"*** DEMOTED to tier {env.max_tier} "
                        f"(wr={wr:.2f} over {self.window} games)"
                    )
                    self.recent_wins.clear()

        return True

    def _on_training_end(self):
        if self._log_file:
            self._log_file.close()


class EvalCallback(BaseCallback):
    """Periodic evaluation against fixed opponent set."""

    def __init__(
        self,
        eval_opponents: List[Tuple[str, Callable]],
        eval_freq: int = 10000,
        n_eval_episodes: int = 2,
        save_path: str = "ppo_gnn/cache",
        log_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_opponents = eval_opponents
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = Path(save_path)
        self.log_path = log_path
        self.best_eval_wr = -1.0
        self._last_eval_step = 0

    def _log(self, msg: str):
        print(msg)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(msg + "\n")

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps

        from .sb3_env import OrbitWarsEnv

        self._log(f"\n--- Eval at step {self.num_timesteps} ---")
        total_wins = 0
        total_games = 0
        opp_results = []

        for opp_name, opp_fn in self.eval_opponents:
            wins = 0
            for _ in range(self.n_eval_episodes):
                eval_env = OrbitWarsEnv(
                    opponent_pool=[(1, opp_name, opp_fn)],
                    mode="2p",
                    max_tier=99,
                )
                obs, _ = eval_env.reset()
                done = False
                while not done:
                    mask = eval_env.action_masks()
                    action, _ = self.model.predict(
                        obs, deterministic=True,
                        action_masks=mask,
                    )
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                if info.get("won", False):
                    wins += 1
            total_wins += wins
            total_games += self.n_eval_episodes
            opp_results.append(f"{opp_name}:{wins}/{self.n_eval_episodes}")

        eval_wr = total_wins / max(total_games, 1)
        self._log(f"  eval_wr={eval_wr:.3f} ({total_wins}/{total_games}) | " + " | ".join(opp_results))

        if eval_wr > self.best_eval_wr:
            self.best_eval_wr = eval_wr
            save_file = self.save_path / "sb3_best_model"
            self.model.save(str(save_file))
            self._log(f"  *** New best checkpoint (eval_wr={eval_wr:.3f}) ***")

        return True


class SelfPlayCallback(BaseCallback):
    """Periodically snapshot the current model and add it as a self-play opponent.

    Every `snapshot_freq` episodes, saves a frozen copy of the policy and
    injects it into the env's opponent pool. This prevents overfitting to
    the static opponent pool and creates a curriculum of past selves.
    """

    def __init__(
        self,
        snapshot_freq: int = 100,
        max_archive: int = 5,
        self_play_tier: int = 1,
        log_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.snapshot_freq = snapshot_freq
        self.max_archive = max_archive
        self.self_play_tier = self_play_tier
        self.log_path = log_path
        self._total_episodes = 0
        self._last_snapshot_ep = 0
        self._archive: list = []

    def _log(self, msg: str):
        print(msg)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(msg + "\n")

    def _get_env(self):
        try:
            venv = self.training_env
            while hasattr(venv, "venv"):
                venv = venv.venv
            return venv.envs[0]
        except (AttributeError, IndexError):
            return None

    def _make_self_play_agent(self, snapshot_path: str, ep_label: int):
        """Create an agent function from a saved SB3 model checkpoint."""
        from sb3_contrib import MaskablePPO
        from .sb3_env import (
            OrbitWarsEnv, SB3_MAX_CANDIDATES, NUM_FRACTIONS,
            NOOP_ACTION, EDGE_INPUT_DIM, MAX_ACTIONS, NUM_CHOICES,
        )
        from .edge_policy import compute_candidate_edges
        import torch

        # Load a frozen copy of the model
        frozen_model = MaskablePPO.load(snapshot_path, device="cpu")

        # Capture VecNormalize stats for proper observation normalization
        obs_mean = None
        obs_var = None
        clip_obs = 10.0
        try:
            # training_env IS the VecNormalize wrapper
            venv = self.training_env
            if hasattr(venv, "obs_rms"):
                obs_mean = venv.obs_rms.mean.copy()
                obs_var = venv.obs_rms.var.copy()
                clip_obs = venv.clip_obs
        except Exception:
            pass

        def agent_fn(obs, config):
            """Self-play agent wrapping a frozen SB3 model."""
            planets = obs.get("planets", [])
            fleets = obs.get("fleets", [])
            player = obs.get("player", 0)
            step = obs.get("step", 0)
            angular_velocity = obs.get("angular_velocity", 0.0)

            if not planets or not any(int(p[1]) == player for p in planets):
                return []

            ef, edge_indices, em, num_valid = compute_candidate_edges(
                planets=planets,
                fleets=fleets,
                player_id=player,
                num_players=2,
                step=step,
                max_steps=500,
                max_candidates=SB3_MAX_CANDIDATES,
                angular_velocity=angular_velocity,
            )
            ef = torch.nan_to_num(ef, nan=0.0, posinf=1.0, neginf=-1.0)
            obs_flat = ef.numpy().flatten().astype(np.float32)

            # Apply VecNormalize (same transform as training)
            if obs_mean is not None:
                obs_flat = (obs_flat - obs_mean) / np.sqrt(obs_var + 1e-8)
                obs_flat = np.clip(obs_flat, -clip_obs, clip_obs)

            # Build action mask
            single_mask = np.zeros(NUM_CHOICES, dtype=bool)
            nv = min(num_valid, SB3_MAX_CANDIDATES)
            for i in range(nv):
                single_mask[i * NUM_FRACTIONS : (i + 1) * NUM_FRACTIONS] = True
            single_mask[NOOP_ACTION] = True
            mask = np.tile(single_mask, MAX_ACTIONS)

            action, _ = frozen_model.predict(
                obs_flat, deterministic=True, action_masks=mask,
            )

            # Decode actions (simplified — reuse env logic)
            import math
            from .edge_policy import solve_intercept, _travel_time
            from .sun_geometry import sun_intersects_path, SUN_X, SUN_Y

            actions = []
            committed = {}
            for slot_action in action:
                slot_action = int(slot_action)
                if slot_action == NOOP_ACTION:
                    continue
                cand_idx = slot_action // NUM_FRACTIONS
                frac_idx = slot_action % NUM_FRACTIONS
                if cand_idx >= nv:
                    continue

                src_pidx = edge_indices[cand_idx, 0].item()
                tgt_pidx = edge_indices[cand_idx, 1].item()
                src_p = planets[src_pidx]
                tgt_p = planets[tgt_pidx]
                if int(src_p[1]) != player:
                    continue

                sx, sy = float(src_p[2]), float(src_p[3])
                tx, ty = float(tgt_p[2]), float(tgt_p[3])
                tgt_planet_r = float(tgt_p[4])
                tgt_orbital_r = math.hypot(tx - 50.0, ty - 50.0)
                omega = angular_velocity if (tgt_orbital_r + tgt_planet_r < 48.0) else 0.0

                src_fleet = int(float(src_p[5]))
                already = committed.get(src_pidx, 0)
                available = src_fleet - max(5, int(src_fleet * 0.15)) - already
                if available < 5:
                    continue

                from .edge_policy import FRACTION_BUCKETS
                ships = max(1, int(available * FRACTION_BUCKETS[frac_idx]))
                if ships < 5:
                    continue

                ix, iy = solve_intercept(sx, sy, tx, ty, omega, ships)
                angle = math.atan2(iy - sy, ix - sx)
                actions.append([int(src_p[0]), angle, ships])
                committed[src_pidx] = already + ships

            return actions

        return agent_fn

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", np.array([]))
        for done in dones:
            if done:
                self._total_episodes += 1

        if self._total_episodes - self._last_snapshot_ep >= self.snapshot_freq:
            self._last_snapshot_ep = self._total_episodes
            env = self._get_env()
            if env is None:
                return True

            # Save current model to temp file
            snapshot_path = str(Path(os.path.dirname(__file__)) / "cache" / f"sb3_selfplay_ep{self._total_episodes}")
            self.model.save(snapshot_path)

            # Create agent function from snapshot
            try:
                agent_fn = self._make_self_play_agent(snapshot_path, self._total_episodes)
                name = f"self_ep{self._total_episodes}"
                env.opponent_pool.append((self.self_play_tier, name, agent_fn))
                self._archive.append(name)
                self._log(f"  [self-play] Added {name} to pool (archive: {len(self._archive)})")

                # Remove oldest if over limit
                if len(self._archive) > self.max_archive:
                    old_name = self._archive.pop(0)
                    env.opponent_pool = [
                        (t, n, fn) for t, n, fn in env.opponent_pool if n != old_name
                    ]
                    self._log(f"  [self-play] Removed {old_name} (archive capped at {self.max_archive})")
            except Exception as e:
                self._log(f"  [self-play] Failed to create snapshot: {e}")

        return True


class CheckpointCallback(BaseCallback):
    """Save model periodically."""

    def __init__(
        self,
        save_freq: int = 50000,
        save_path: str = "ppo_gnn/cache",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self._last_save_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save_step >= self.save_freq:
            self._last_save_step = self.num_timesteps
            save_file = self.save_path / f"sb3_checkpoint_{self.num_timesteps}"
            self.model.save(str(save_file))
            # Also save VecNormalize stats
            try:
                venv = self.training_env
                if hasattr(venv, "save"):
                    venv.save(str(self.save_path / "sb3_vec_normalize.pkl"))
            except Exception:
                pass
            if self.verbose:
                print(f"  [checkpoint] saved at step {self.num_timesteps}")
        return True
