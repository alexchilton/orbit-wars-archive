"""Gymnasium wrapper for Orbit Wars with heuristic-filtered multi-action space.

Wraps the kaggle orbit_wars environment as a gymnasium.Env with:
- Observation: flattened (48*74,) = (3552,) candidate edge features
- Action: MultiDiscrete([481] * 5) — 5 slots, each 480 candidates + 1 noop
- Action masking via action_masks() for MaskablePPO
- Source deduplication: tracks committed ships to avoid draining a planet
"""

from __future__ import annotations

import math
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from .edge_policy import (
    MAX_ACTIONS,
    EDGE_INPUT_DIM,
    FRACTION_BUCKETS,
    NUM_FRACTIONS,
    compute_candidate_edges,
    solve_intercept,
    _travel_time,
)
from .sb3_constants import (
    SB3_MAX_CANDIDATES, NUM_CHOICES, NOOP_ACTION, OBS_DIM,
    CANDIDATE_OBS_DIM, GLOBAL_DIM, TEMPORAL_STEPS, TEMPORAL_OBS_DIM,
)
from .sun_geometry import sun_intersects_path, SUN_X, SUN_Y

os.environ.setdefault("KAGGLE_ENVIRONMENTS_QUIET", "1")


def compute_planet_count_bonus(my_planet_count: int) -> float:
    if my_planet_count <= 0:
        return 0.0
    return 0.005 * math.sqrt(my_planet_count)


def build_action_mask(planets, player, env_steps, edge_indices, num_valid, angular_velocity):
    """Build heuristic action mask — standalone function usable by env and self-play.

    Returns: (single_mask, intercept_cache)
        single_mask: np.ndarray of shape (NUM_CHOICES,), dtype=bool
        intercept_cache: dict mapping cand_idx -> (ix, iy) or None
    """
    single_mask = np.zeros(NUM_CHOICES, dtype=bool)
    single_mask[NOOP_ACTION] = True
    intercept_cache = {}

    if planets is None or num_valid == 0:
        return single_mask, intercept_cache

    max_eta = 14 if env_steps < 50 else (20 if env_steps < 150 else 30)

    for cand_idx in range(num_valid):
        src_pidx = edge_indices[cand_idx, 0].item()
        tgt_pidx = edge_indices[cand_idx, 1].item()
        src_p = planets[src_pidx]
        tgt_p = planets[tgt_pidx]

        if int(src_p[1]) != player:
            intercept_cache[cand_idx] = None
            continue

        sx, sy = float(src_p[2]), float(src_p[3])
        tx, ty = float(tgt_p[2]), float(tgt_p[3])
        tgt_planet_r = float(tgt_p[4])
        tgt_orbital_r = math.hypot(tx - 50.0, ty - 50.0)
        omega = angular_velocity if (tgt_orbital_r + tgt_planet_r < 48.0) else 0.0
        src_fleet = int(float(src_p[5]))
        tgt_owner = int(tgt_p[1])
        tgt_ships = int(float(tgt_p[5]))
        tgt_prod = float(tgt_p[6])

        base_reserve = max(5, int(src_fleet * 0.15))
        available = src_fleet - base_reserve
        if available < 5:
            intercept_cache[cand_idx] = None
            continue

        # Sun check — once per candidate using mid-range ships
        mid_ships = max(5, int(available * 0.5))
        ix, iy = solve_intercept(sx, sy, tx, ty, omega, mid_ships)

        sun_blocked = False
        if sun_intersects_path(sx, sy, ix, iy):
            dx, dy = ix - sx, iy - sy
            dist_val = math.sqrt(dx * dx + dy * dy)
            if dist_val > 1e-6:
                mid_x, mid_y = (sx + ix) / 2, (sy + iy) / 2
                sun_dist = math.sqrt((mid_x - SUN_X) ** 2 + (mid_y - SUN_Y) ** 2)
                if sun_dist < 1e-6:
                    sun_blocked = True
                else:
                    perp_x = -(SUN_Y - mid_y) / sun_dist
                    perp_y = (SUN_X - mid_x) / sun_dist
                    offset = 14.0 / dist_val
                    ix2 = ix + perp_x * offset * dist_val * 0.3
                    iy2 = iy + perp_y * offset * dist_val * 0.3
                    if sun_intersects_path(sx, sy, ix2, iy2):
                        sun_blocked = True
                    else:
                        ix, iy = ix2, iy2
            else:
                sun_blocked = True

        if sun_blocked:
            intercept_cache[cand_idx] = None
            continue

        intercept_cache[cand_idx] = (ix, iy)

        # Per-fraction viability checks
        for frac_idx in range(NUM_FRACTIONS):
            ships = max(1, int(available * FRACTION_BUCKETS[frac_idx]))
            if ships < 5:
                continue

            travel_time = _travel_time(sx, sy, ix, iy, ships)
            if travel_time > max_eta:
                continue

            # Capture viability
            if tgt_owner != player:
                if tgt_owner >= 0:
                    needed = tgt_ships + tgt_prod * travel_time * 1.05 + 1
                    if ships < needed * 0.7:
                        continue
                else:
                    if ships < tgt_ships + 1:
                        continue

            single_mask[cand_idx * NUM_FRACTIONS + frac_idx] = True

    return single_mask, intercept_cache


def compute_global_features(planets, fleets, player: int, step: int, max_steps: int) -> np.ndarray:
    """Compute global game-state features (who's winning).

    Standalone version for use by self-play agents and other callers
    that don't have access to the OrbitWarsEnv instance.
    """
    my_planets = my_prod = my_ships = 0.0
    enemy_planets = enemy_prod = enemy_ships = 0.0
    for p in planets:
        owner = int(p[1])
        ships_val, prod_val = float(p[5]), float(p[6])
        if owner == player:
            my_planets += 1
            my_prod += prod_val
            my_ships += ships_val
        elif owner >= 0:
            enemy_planets += 1
            enemy_prod += prod_val
            enemy_ships += ships_val

    for f in fleets:
        f_owner = int(f[1])
        f_ships = float(f[6])
        if f_owner == player:
            my_ships += f_ships
        elif f_owner >= 0:
            enemy_ships += f_ships

    total_planets = my_planets + enemy_planets
    total_prod = my_prod + enemy_prod
    total_ships = my_ships + enemy_ships

    return np.array([
        my_planets / 48.0,
        enemy_planets / 48.0,
        my_prod / max(total_prod, 1.0),
        enemy_prod / max(total_prod, 1.0),
        my_ships / max(total_ships, 1.0),
        enemy_ships / max(total_ships, 1.0),
        step / max(max_steps, 1.0),
        (my_planets - enemy_planets) / max(total_planets, 1.0),
        (my_prod - enemy_prod) / max(total_prod, 1.0),
        (my_ships - enemy_ships) / max(total_ships, 1.0),
    ], dtype=np.float32)


class OrbitWarsEnv(gym.Env):
    """Gymnasium environment for Orbit Wars with heuristic candidate filtering.

    Action space: MultiDiscrete([1921, 1921, 1921, 1921, 1921])
    Each slot picks a candidate*fraction (0-1919) or noop (1920).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_pool: Optional[List[Tuple[int, str, Callable]]] = None,
        mode: str = "2p",
        max_tier: int = 2,
        min_tier: int = 1,
        max_steps: int = 500,
        step_penalty: float = 0.002,
        win_bonus: float = 1.0,
    ):
        super().__init__()

        self.opponent_pool = opponent_pool or []
        self.mode = mode
        self.max_tier = max_tier
        self.min_tier = min_tier
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.win_bonus = win_bonus

        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32,
        )
        # 5 action slots, each can pick from 1921 options (1920 candidate*fracs + noop)
        self.action_space = gym.spaces.MultiDiscrete(
            [NUM_CHOICES] * MAX_ACTIONS
        )

        # State kept across steps
        self._trainer = None
        self._obs = None
        self._edge_indices = None
        self._edge_mask = None
        self._num_valid = 0
        self._planets = None
        self._intercept_cache = {}
        self._player = 0
        self._angular_velocity = 0.0
        self._temporal_buffer = np.zeros((TEMPORAL_STEPS, GLOBAL_DIM), dtype=np.float32)
        self._env_steps = 0
        self._prev_shaped_score = 0.0
        self._prev_my_planets = 1
        self._steps_since_capture = 999
        self._num_players = 2 if mode == "2p" else 4
        self._mixed_mode = (mode == "mixed")
        self._game_count = 0
        self._current_opponent_name = "unknown"
        self._episode_actions_sent = 0
        self._episode_noops = 0
        self._episode_reward = 0.0
        self._episode_ships_sent = 0
        self._episode_captures = 0
        self._episode_losses = 0
        self._episode_model_noops = 0
        self._episode_vetoed = 0
        self._episode_unique_targets = 0
        self._episode_multi_target_steps = 0
        self._episode_action_steps = 0
        self._first_fleet_step = -1
        self._planets_at_25 = 0
        self._planets_at_50 = 0

    def set_tier(self, tier: int):
        self.max_tier = tier

    def _pick_opponent(self):
        eligible = [
            (t, n, fn) for t, n, fn in self.opponent_pool
            if self.min_tier <= t <= self.max_tier
        ]
        if not eligible:
            # Fall back to any opponent in the pool rather than random
            eligible = [(t, n, fn) for t, n, fn in self.opponent_pool if t <= self.max_tier]
        if not eligible:
            eligible = self.opponent_pool
        t, name, fn = random.choice(eligible)
        self._current_opponent_name = name
        return fn

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        from kaggle_environments import make
        env = make("orbit_wars", debug=False)

        opponent = self._pick_opponent()

        # Mixed mode: alternate 2p and 4p games (70% 2p, 30% 4p)
        if self._mixed_mode:
            self._game_count += 1
            use_4p = (self._game_count % 10) < 3  # 30% 4p games
            self._is_4p_game = use_4p
        else:
            use_4p = (self._num_players == 4)
            self._is_4p_game = use_4p

        if not use_4p:
            self._trainer = env.train([None, opponent])
        else:
            slots = [opponent] * 3
            self._trainer = env.train([None] + slots)

        self._obs = self._trainer.reset()
        self._env_steps = 0
        self._prev_shaped_score = 0.0
        self._prev_my_planets = 1
        self._steps_since_capture = 999
        self._episode_actions_sent = 0
        self._episode_noops = 0
        self._episode_reward = 0.0
        self._episode_ships_sent = 0
        self._episode_captures = 0
        self._episode_losses = 0
        self._episode_model_noops = 0
        self._episode_vetoed = 0
        self._episode_unique_targets = 0
        self._episode_multi_target_steps = 0
        self._episode_action_steps = 0
        self._first_fleet_step = -1
        self._planets_at_25 = 0
        self._planets_at_50 = 0
        self._temporal_buffer = np.zeros((TEMPORAL_STEPS, GLOBAL_DIM), dtype=np.float32)

        obs_flat = self._compute_obs()
        return obs_flat, {"num_valid": self._num_valid}

    def _compute_global_features(self) -> np.ndarray:
        """Compute global game-state features (who's winning)."""
        planets = self._obs.get("planets", []) if self._obs else []
        fleets = self._obs.get("fleets", []) if self._obs else []
        return compute_global_features(
            planets, fleets, self._player, self._env_steps, self.max_steps
        )

    def _compute_obs(self) -> np.ndarray:
        """Extract candidate features + global state + temporal history."""
        planets = self._obs.get("planets", [])
        fleets = self._obs.get("fleets", [])
        self._player = self._obs.get("player", 0)
        self._planets = planets
        self._angular_velocity = self._obs.get("angular_velocity", 0.0)

        if not planets or not any(int(p[1]) == self._player for p in planets):
            self._edge_indices = torch.zeros((SB3_MAX_CANDIDATES, 2), dtype=torch.long)
            self._edge_mask = torch.zeros(SB3_MAX_CANDIDATES)
            self._num_valid = 0
            # Still update temporal buffer with global features
            global_feats = self._compute_global_features()
            self._temporal_buffer[:-1] = self._temporal_buffer[1:]
            self._temporal_buffer[-1] = global_feats
            obs = np.zeros(OBS_DIM, dtype=np.float32)
            obs[CANDIDATE_OBS_DIM:] = self._temporal_buffer.flatten()
            return obs

        # In mixed mode, detect actual num_players from game state
        num_players = 4 if getattr(self, "_is_4p_game", False) else self._num_players

        ef, edge_indices, em, num_valid = compute_candidate_edges(
            planets=planets,
            fleets=fleets,
            player_id=self._player,
            num_players=num_players,
            step=self._env_steps,
            max_steps=self.max_steps,
            max_candidates=SB3_MAX_CANDIDATES,
            angular_velocity=self._obs.get("angular_velocity", 0.0),
        )
        ef = torch.nan_to_num(ef, nan=0.0, posinf=1.0, neginf=-1.0)

        self._edge_indices = edge_indices
        self._edge_mask = em
        self._num_valid = min(num_valid, SB3_MAX_CANDIDATES)

        # Compute global features and update temporal buffer
        global_feats = self._compute_global_features()
        self._temporal_buffer[:-1] = self._temporal_buffer[1:]
        self._temporal_buffer[-1] = global_feats

        # Concat: [candidate_features | temporal_history]
        candidate_flat = ef.numpy().flatten().astype(np.float32)
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[:CANDIDATE_OBS_DIM] = candidate_flat
        obs[CANDIDATE_OBS_DIM:] = self._temporal_buffer.flatten()
        return obs

    def action_masks(self) -> np.ndarray:
        """Return boolean mask for MultiDiscrete([481]*5).

        Delegates to build_action_mask() — shared with self-play agent.
        """
        single_mask, self._intercept_cache = build_action_mask(
            planets=self._planets,
            player=self._player,
            env_steps=self._env_steps,
            edge_indices=self._edge_indices,
            num_valid=self._num_valid,
            angular_velocity=self._angular_velocity,
        )
        return np.tile(single_mask, MAX_ACTIONS)

    def _decode_actions(self, multi_action: np.ndarray) -> Tuple[List[List], int, int]:
        """Convert MultiDiscrete action array to kaggle action list.

        Most heuristic checks are now in action_masks(). This only handles:
        - Source overdraw (sequential slot dependency, can't be pre-masked)
        - Angle computation using cached intercept points

        Returns: (actions, model_noops, vetoed) — counts for diagnostics.
        """
        if self._planets is None:
            return [], 0, 0

        planets = self._planets
        player = self._player
        actions = []
        model_noops = 0
        vetoed = 0
        committed = {}  # src_pidx -> ships already committed this turn

        for slot_action in multi_action:
            slot_action = int(slot_action)
            if slot_action == NOOP_ACTION:
                model_noops += 1
                continue

            cand_idx = slot_action // NUM_FRACTIONS
            frac_idx = slot_action % NUM_FRACTIONS

            # Skip only physically impossible actions (no veto on valid candidates)
            if cand_idx >= self._num_valid:
                model_noops += 1  # treat as wasted slot, not veto
                continue

            # Use cached intercept — if None, skip silently (no penalty)
            cached = self._intercept_cache.get(cand_idx)
            if cached is None:
                model_noops += 1
                continue

            ix, iy = cached
            src_pidx = self._edge_indices[cand_idx, 0].item()
            tgt_pidx = self._edge_indices[cand_idx, 1].item()
            src_p = planets[src_pidx]

            sx, sy = float(src_p[2]), float(src_p[3])
            src_fleet = int(float(src_p[5]))

            # Overdraw check — physical constraint, can't send ships that don't exist
            base_reserve = max(5, int(src_fleet * 0.15))
            already_committed = committed.get(src_pidx, 0)
            available = src_fleet - base_reserve - already_committed
            if available < 5:
                model_noops += 1
                continue

            ships = max(1, int(available * FRACTION_BUCKETS[frac_idx]))
            if ships < 5:
                model_noops += 1
                continue

            angle = math.atan2(iy - sy, ix - sx)
            actions.append([int(src_p[0]), angle, ships])
            committed[src_pidx] = already_committed + ships

        return actions, model_noops, vetoed

    def _truncation_reward_short(self, my_planets_now, enemy_planets, my_fleet, enemy_fleet):
        """Truncation reward for short games (max_steps < 500).

        Directly compares planet ownership and fleet strength.
        """
        if my_planets_now + enemy_planets > 0:
            planet_share = (my_planets_now - enemy_planets) / (my_planets_now + enemy_planets)
        else:
            planet_share = 0.0
        total_fleet = my_fleet + enemy_fleet
        fleet_share = (my_fleet - enemy_fleet) / max(total_fleet, 1.0)
        advantage = 0.8 * planet_share + 0.2 * fleet_share
        return math.tanh(advantage * 2.0) * self.win_bonus

    def _truncation_reward_full(self, my_planets_now, my_fleet, enemy_fleet, my_prod, enemy_prod):
        """Truncation reward for full-length games (max_steps=500).

        Uses expected resource curves for normalization.
        """
        exp_fleet = max(50 + self._env_steps * 3, 1.0)
        exp_prod = max(5 + self._env_steps * 0.2, 1.0)
        fleet_diff = (my_fleet - enemy_fleet) / exp_fleet
        prod_diff = (my_prod - enemy_prod) / exp_prod
        time_val = (self.max_steps - self._env_steps) / self.max_steps
        advantage = (1 - time_val) * fleet_diff + time_val * 1.5 * prod_diff
        return math.tanh(advantage) * self.win_bonus

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        actions, model_noops, vetoed = self._decode_actions(action)

        total_ships_sent = sum(a[2] for a in actions) if actions else 0
        self._obs, raw_reward, done, info = self._trainer.step(actions)
        self._env_steps += 1
        num_sent = len(actions)
        self._episode_actions_sent += num_sent
        self._episode_ships_sent += total_ships_sent
        self._episode_model_noops += model_noops
        self._episode_vetoed += vetoed
        if num_sent == 0:
            self._episode_noops += 1

        # Track slot diversity — how many unique target planets per step
        if num_sent > 0:
            self._episode_action_steps += 1
            # actions are [planet_id, angle, ships] — count unique planet_ids as sources
            unique_sources = len(set(a[0] for a in actions))
            self._episode_unique_targets += unique_sources
            if unique_sources > 1:
                self._episode_multi_target_steps += 1

        # --- Pure win/loss reward (no delta shaping) ---
        planets = self._obs.get("planets", []) if self._obs else []
        fleets = self._obs.get("fleets", []) if self._obs else []
        player = self._player

        my_fleet = my_prod = enemy_fleet = enemy_prod = 0.0
        my_planets_now = 0
        for p in planets:
            owner = int(p[1])
            ships_val, prod_val = float(p[5]), float(p[6])
            if owner == player:
                my_fleet += ships_val
                my_prod += prod_val
                my_planets_now += 1
            elif owner >= 0:
                enemy_fleet += ships_val
                enemy_prod += prod_val

        # Include in-flight ships (for terminal tiebreak calculation only)
        for f in fleets:
            f_owner = int(f[1])
            f_ships = float(f[6])
            if f_owner == player:
                my_fleet += f_ships
            elif f_owner >= 0:
                enemy_fleet += f_ships

        shaped_reward = 0.0

        # Conditional noop penalty — only if valid actions existed
        if num_sent == 0 and my_planets_now >= 1:
            had_valid = any(v is not None for v in self._intercept_cache.values())
            if had_valid:
                shaped_reward -= 0.005

        # --- Dense reward shaping: planet captures only (no loss penalty) ---
        planet_delta = my_planets_now - self._prev_my_planets
        if planet_delta > 0:
            shaped_reward += 0.05 * planet_delta   # capture bonus
            self._episode_captures += planet_delta
        elif planet_delta < 0:
            self._episode_losses += -planet_delta  # track but don't penalize

        # --- Dense reward shaping: production advantage ---
        prod_advantage = (my_prod - enemy_prod) / max(my_prod + enemy_prod, 1.0)
        shaped_reward += 0.002 * prod_advantage  # small nudge toward production lead

        # Track milestones (captures/losses already tracked above)
        if num_sent > 0 and self._first_fleet_step < 0:
            self._first_fleet_step = self._env_steps
        if self._env_steps == 25:
            self._planets_at_25 = my_planets_now
        elif self._env_steps == 50:
            self._planets_at_50 = my_planets_now
        self._prev_my_planets = my_planets_now

        # Terminal reward
        terminated = bool(done)
        truncated = (not done) and (self._env_steps >= self.max_steps)

        if terminated or truncated:
            owned_final = my_planets_now
            eliminated = owned_final == 0

            if raw_reward is not None and raw_reward > 0:
                shaped_reward += self.win_bonus
            elif (raw_reward is not None and raw_reward < 0) or (
                eliminated and enemy_fleet + enemy_prod > 0
            ):
                survival_frac = self._env_steps / self.max_steps
                shaped_reward += -self.win_bonus * (1.0 - 0.5 * survival_frac)
            elif truncated:
                enemy_planets = sum(1 for p in planets if int(p[1]) >= 0 and int(p[1]) != player)
                if self.max_steps < 500:
                    shaped_reward += self._truncation_reward_short(
                        my_planets_now, enemy_planets, my_fleet, enemy_fleet)
                else:
                    shaped_reward += self._truncation_reward_full(
                        my_planets_now, my_fleet, enemy_fleet, my_prod, enemy_prod)

        # Recompute obs for next step
        if not (terminated or truncated):
            obs_flat = self._compute_obs()
        else:
            obs_flat = np.zeros(OBS_DIM, dtype=np.float32)

        self._episode_reward += shaped_reward
        won = raw_reward is not None and raw_reward > 0
        info_out = {
            "won": won,
            "env_steps": self._env_steps,
            "my_planets": my_planets_now,
            "actions_sent": num_sent,
            "opponent": self._current_opponent_name,
            "is_4p": getattr(self, "_is_4p_game", False),
        }
        if terminated or truncated:
            info_out["episode_actions"] = self._episode_actions_sent
            info_out["episode_noops"] = self._episode_noops
            noop_pct = 100 * self._episode_noops / max(self._env_steps, 1)
            info_out["noop_pct"] = noop_pct
            info_out["first_fleet_step"] = self._first_fleet_step
            info_out["planets_at_25"] = self._planets_at_25
            info_out["planets_at_50"] = self._planets_at_50
            info_out["episode_reward"] = self._episode_reward
            info_out["ships_per_fleet"] = (
                self._episode_ships_sent / max(self._episode_actions_sent, 1)
            )
            info_out["episode_captures"] = self._episode_captures
            info_out["episode_losses"] = self._episode_losses
            # Noop breakdown: model chose noop vs heuristic vetoed
            total_slots = self._env_steps * MAX_ACTIONS
            info_out["model_noop_pct"] = 100 * self._episode_model_noops / max(total_slots, 1)
            info_out["vetoed_pct"] = 100 * self._episode_vetoed / max(total_slots, 1)
            # Slot diversity: avg unique sources per action step, and % of steps with multi-source
            info_out["avg_unique_sources"] = (
                self._episode_unique_targets / max(self._episode_action_steps, 1)
            )
            info_out["multi_source_pct"] = (
                100 * self._episode_multi_target_steps / max(self._episode_action_steps, 1)
            )

        return obs_flat, shaped_reward, terminated, truncated, info_out
