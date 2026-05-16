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
from .sb3_constants import SB3_MAX_CANDIDATES, NUM_CHOICES, NOOP_ACTION, OBS_DIM
from .sun_geometry import sun_intersects_path, closest_approach_to_sun, SUN_X, SUN_Y

os.environ.setdefault("KAGGLE_ENVIRONMENTS_QUIET", "1")


def compute_planet_count_bonus(my_planet_count: int) -> float:
    if my_planet_count <= 0:
        return 0.0
    return 0.005 * math.sqrt(my_planet_count)


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
        win_bonus: float = 10.0,
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
        self._player = 0
        self._angular_velocity = 0.0
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
        self._first_fleet_step = -1
        self._planets_at_25 = 0
        self._planets_at_50 = 0

        obs_flat = self._compute_obs()
        return obs_flat, {"num_valid": self._num_valid}

    def _compute_obs(self) -> np.ndarray:
        """Extract candidate features from current observation."""
        planets = self._obs.get("planets", [])
        fleets = self._obs.get("fleets", [])
        self._player = self._obs.get("player", 0)
        self._planets = planets
        self._angular_velocity = self._obs.get("angular_velocity", 0.0)

        if not planets or not any(int(p[1]) == self._player for p in planets):
            self._edge_indices = torch.zeros((SB3_MAX_CANDIDATES, 2), dtype=torch.long)
            self._edge_mask = torch.zeros(SB3_MAX_CANDIDATES)
            self._num_valid = 0
            return np.zeros(OBS_DIM, dtype=np.float32)

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

        return ef.numpy().flatten().astype(np.float32)

    def action_masks(self) -> np.ndarray:
        """Return boolean mask for MultiDiscrete([1921]*5).

        Shape: (5 * 1921,) = (9605,) — concatenated masks for each slot.
        Each slot: 1920 candidate*fraction bools + 1 noop (always valid).
        """
        single_mask = np.zeros(NUM_CHOICES, dtype=bool)
        for i in range(self._num_valid):
            single_mask[i * NUM_FRACTIONS : (i + 1) * NUM_FRACTIONS] = True
        # Noop is always valid
        single_mask[NOOP_ACTION] = True
        # Same mask for all 5 slots
        return np.tile(single_mask, MAX_ACTIONS)

    def _decode_actions(self, multi_action: np.ndarray) -> Tuple[List[List], int, int]:
        """Convert MultiDiscrete action array to kaggle action list.

        Handles source deduplication: tracks committed ships per source planet
        so multiple actions from the same source don't overdraw.

        Returns: (actions, model_noops, vetoed) — counts for diagnostics.
        """
        if self._planets is None:
            return [], 0, 0

        planets = self._planets
        player = self._player
        actions = []
        model_noops = 0  # model deliberately chose noop
        vetoed = 0  # model chose action but heuristic rejected it
        # Track committed ships per source planet index to avoid overdraw
        committed = {}  # src_pidx -> ships already committed this turn

        # Early-game heuristic: count good neutrals to decide if we should skip enemy attacks
        is_early = self._env_steps < 40
        good_neutrals_available = 0
        if is_early:
            for p in planets:
                if int(p[1]) < 0 and float(p[6]) >= 2.0:  # neutral with prod >= 2
                    good_neutrals_available += 1

        for slot_action in multi_action:
            slot_action = int(slot_action)
            if slot_action == NOOP_ACTION:
                model_noops += 1
                continue

            cand_idx = slot_action // NUM_FRACTIONS
            frac_idx = slot_action % NUM_FRACTIONS

            if cand_idx >= self._num_valid:
                vetoed += 1
                continue

            src_pidx = self._edge_indices[cand_idx, 0].item()
            tgt_pidx = self._edge_indices[cand_idx, 1].item()
            src_p = planets[src_pidx]
            tgt_p = planets[tgt_pidx]

            if int(src_p[1]) != player:
                vetoed += 1
                continue

            sx, sy = float(src_p[2]), float(src_p[3])
            tx, ty = float(tgt_p[2]), float(tgt_p[3])
            # p[4] is planet RADIUS, not omega. Use global angular_velocity
            # for orbiting planets (orbital_r + planet_r < 48)
            tgt_planet_r = float(tgt_p[4])
            tgt_orbital_r = math.hypot(tx - 50.0, ty - 50.0)
            omega = self._angular_velocity if (tgt_orbital_r + tgt_planet_r < 48.0) else 0.0
            src_fleet = int(float(src_p[5]))
            tgt_owner = int(tgt_p[1])
            tgt_ships = int(float(tgt_p[5]))
            tgt_prod = float(tgt_p[6])

            # Early-game heuristic filters (shunlite/v131 style)
            if is_early:
                # Skip enemy attacks if 3+ good neutrals still available
                if tgt_owner >= 0 and tgt_owner != player and good_neutrals_available >= 3:
                    vetoed += 1
                    continue
                # Skip low-prod neutrals early (only grab prod >= 2)
                if tgt_owner < 0 and tgt_prod < 2.0:
                    vetoed += 1
                    continue
                # Skip low-prod orbiting neutrals (hard to intercept, low value)
                # But allow in first 20 steps — need to expand with whatever's nearby
                if self._env_steps >= 20 and tgt_owner < 0 and abs(omega) > 0.001 and tgt_prod < 3.0:
                    vetoed += 1
                    continue

            # v131-style threat-aware reserve: hold back more if inbound enemies
            base_reserve = max(5, int(src_fleet * 0.15))
            already_committed = committed.get(src_pidx, 0)
            available = src_fleet - base_reserve - already_committed
            if available < 1:
                vetoed += 1
                continue

            ships = max(1, int(available * FRACTION_BUCKETS[frac_idx]))
            # Adaptive min ships: very low early to allow opening expansion
            if self._env_steps < 20:
                min_ships = 5  # opening: send what you can
            elif self._env_steps < 100:
                min_ships = 10
            else:
                min_ships = 15 + self._env_steps // 50
            if ships < min_ships:
                vetoed += 1
                continue

            # Orbit intercept
            ix, iy = solve_intercept(sx, sy, tx, ty, omega, ships)

            # v131: safe_angle — post-intercept sun check
            if sun_intersects_path(sx, sy, ix, iy):
                # Try tangent offset around sun
                dx, dy = ix - sx, iy - sy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 1e-6:
                    # Nudge angle to clear the sun
                    mid_x, mid_y = (sx + ix) / 2, (sy + iy) / 2
                    sun_dist = math.sqrt((mid_x - SUN_X)**2 + (mid_y - SUN_Y)**2)
                    if sun_dist < 1e-6:
                        vetoed += 1
                        continue  # path goes through sun center, skip
                    # Rotate angle away from sun
                    base_angle = math.atan2(dy, dx)
                    perp_x = -(SUN_Y - mid_y) / sun_dist
                    perp_y = (SUN_X - mid_x) / sun_dist
                    offset = 14.0 / dist  # ~14 units clearance
                    ix2 = ix + perp_x * offset * dist * 0.3
                    iy2 = iy + perp_y * offset * dist * 0.3
                    if sun_intersects_path(sx, sy, ix2, iy2):
                        vetoed += 1
                        continue  # still blocked, skip this action
                    ix, iy = ix2, iy2


            angle = math.atan2(iy - sy, ix - sx)

            # Max ETA filter (v131 phase-aware)
            travel_time = _travel_time(sx, sy, ix, iy, ships)
            max_eta = 14 if self._env_steps < 50 else (20 if self._env_steps < 150 else 30)
            if travel_time > max_eta:
                vetoed += 1
                continue

            # v131: capture viability — ensure enough ships to take target
            if tgt_owner != self._player:
                if tgt_owner >= 0:
                    # Enemy planet: garrison + production during travel
                    needed = tgt_ships + tgt_prod * travel_time * 1.05 + 1
                    if ships < needed * 0.7:
                        vetoed += 1
                        continue
                else:
                    # Neutral planet: fixed garrison, no production growth
                    if ships < tgt_ships + 1:
                        vetoed += 1
                        continue

            actions.append([int(src_p[0]), angle, ships])
            committed[src_pidx] = already_committed + ships

        return actions, model_noops, vetoed

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

        # --- Minimal reward shaping (Option A: 3 signals only) ---
        planets = self._obs.get("planets", []) if self._obs else []
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

        # Signal 1: Delta score — captures, production, fleet advantage all implicit
        PROD_W = 3.0
        score = (my_fleet + PROD_W * my_prod) - (enemy_fleet + PROD_W * enemy_prod)
        shaped_reward = (score - self._prev_shaped_score) * 0.01
        self._prev_shaped_score = score

        # Signal 2: Flat noop penalty — one simple push toward action
        if num_sent == 0 and my_planets_now >= 1:
            shaped_reward -= 0.2

        # Track milestones and turnover (for logging only, not reward)
        if my_planets_now > self._prev_my_planets:
            self._episode_captures += my_planets_now - self._prev_my_planets
        elif my_planets_now < self._prev_my_planets:
            self._episode_losses += self._prev_my_planets - my_planets_now
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
                exp_fleet = max(50 + self._env_steps * 3, 1.0)
                exp_prod = max(5 + self._env_steps * 0.2, 1.0)
                fleet_diff = (my_fleet - enemy_fleet) / exp_fleet
                prod_diff = (my_prod - enemy_prod) / exp_prod
                time_val = (500 - self._env_steps) / 500
                advantage = (1 - time_val) * fleet_diff + time_val * 1.5 * prod_diff
                shaped_reward += math.tanh(advantage) * (self.win_bonus / 2.0)

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

        return obs_flat, shaped_reward, terminated, truncated, info_out
