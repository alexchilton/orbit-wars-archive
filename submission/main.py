"""SB3 MaskablePPO Transformer agent for Orbit Wars.

Standalone Kaggle submission — all dependencies inlined.
Loads policy weights from sb3_policy_weights.pt.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUN_X = 50.0
SUN_Y = 50.0
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0
BOARD_DIAGONAL = math.sqrt(BOARD_SIZE ** 2 + BOARD_SIZE ** 2)
DEFAULT_SAFETY_MARGIN = 2.0
INNER_ORBIT_THRESHOLD = 48.0

MAX_CANDIDATES = 48
MAX_ACTIONS = 5
FRACTION_BUCKETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
NUM_FRACTIONS = len(FRACTION_BUCKETS)
NUM_CHOICES = MAX_CANDIDATES * NUM_FRACTIONS + 1  # 481
NOOP_ACTION = MAX_CANDIDATES * NUM_FRACTIONS  # 480

EDGE_INPUT_DIM = 74
TAKEOVER_MARGIN = 1.05
SCORE_PROD_WEIGHT = 18.0
SCORE_TT_PENALTY = 3.5
NEUTRAL_BONUS = 25.0
ENEMY_BONUS_EARLY = 15.0
ENEMY_BONUS_PRESSURE = 26.0
ENEMY_BONUS_AGGRO = 35.0
DENIAL_BASE_SCORE = 75.0
DENIAL_PROD_WEIGHT = 8.0
DENIAL_URGENCY_WEIGHT = 20.0
PROACTIVE_BONUS = 15.0
SNIPER_BONUS = 60.0
SNIPER_RANGE = 35.0
SNIPER_MIN_SHIPS = 40
HIGH_PROD_THRESHOLD = 4.0

_LOG1000 = math.log(1000)
_MAX_SPEED_MINUS_1 = 5.0

# ---------------------------------------------------------------------------
# Sun geometry
# ---------------------------------------------------------------------------


def closest_approach_to_sun(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return math.sqrt((x1 - SUN_X) ** 2 + (y1 - SUN_Y) ** 2)
    t = max(0.0, min(1.0, ((SUN_X - x1) * dx + (SUN_Y - y1) * dy) / seg_len_sq))
    px, py = x1 + t * dx, y1 + t * dy
    return math.sqrt((px - SUN_X) ** 2 + (py - SUN_Y) ** 2)


def sun_intersects_path(x1, y1, x2, y2):
    return closest_approach_to_sun(x1, y1, x2, y2) < SUN_RADIUS + DEFAULT_SAFETY_MARGIN


# ---------------------------------------------------------------------------
# Fleet physics
# ---------------------------------------------------------------------------


def _fleet_speed(ships):
    if ships <= 0:
        return 1.0
    return 1.0 + _MAX_SPEED_MINUS_1 * (math.log(max(ships, 1)) / _LOG1000) ** 1.5


def _travel_time(x1, y1, x2, y2, ships):
    dx, dy = x2 - x1, y2 - y1
    if ships <= 0:
        return 999.0
    return math.sqrt(dx * dx + dy * dy) / _fleet_speed(ships)


def predict_orbit(x, y, omega, dt):
    theta = math.atan2(y - SUN_Y, x - SUN_X)
    r = math.hypot(x - SUN_X, y - SUN_Y)
    return SUN_X + r * math.cos(theta + omega * dt), SUN_Y + r * math.sin(theta + omega * dt)


def solve_intercept(fx, fy, tx, ty, omega, ships, iterations=25):
    if omega == 0:
        return tx, ty
    t = _travel_time(fx, fy, tx, ty, ships)
    ix, iy = tx, ty
    for _ in range(iterations):
        ix, iy = predict_orbit(tx, ty, omega, t)
        t2 = _travel_time(fx, fy, ix, iy, ships)
        if abs(t2 - t) < 0.05:
            break
        t = t2
    return ix, iy


def _ships_needed_for_takeover(tgt_ships, tgt_prod, tt, owner, margin=TAKEOVER_MARGIN):
    if owner == -1:
        return int(tgt_ships * margin) + 1
    return int((tgt_ships + tgt_prod * tt) * margin) + 1


# ---------------------------------------------------------------------------
# Candidate edge computation (from edge_policy.py)
# ---------------------------------------------------------------------------


def _build_node_features_compact(
    planets, fleets, player_id, num_players, step, max_steps,
    planet_ships, planet_prod, planet_owner, planet_x, planet_y,
    planet_orbit_vel, inbound_friendly, inbound_enemy,
    my_total_ships, my_total_prod, total_prod,
):
    N = len(planets)
    feats = torch.zeros(N, 28)
    step_progress = step / max(max_steps, 1)

    enemy_total_prod = 0.0
    my_planet_count = 0
    total_owned = 0
    total_ships_global = sum(planet_ships)
    strongest_enemy = 0.0
    per_player_ships = {}
    my_fleets_intransit = 0.0

    for i in range(N):
        o = planet_owner[i]
        if o >= 0:
            total_owned += 1
            per_player_ships[o] = per_player_ships.get(o, 0) + planet_ships[i]
            if o == player_id:
                my_planet_count += 1
            else:
                enemy_total_prod += planet_prod[i]

    for f in fleets:
        fo, fs = int(f[1]), float(f[6])
        per_player_ships[fo] = per_player_ships.get(fo, 0) + fs
        total_ships_global += fs
        if fo == player_id:
            my_fleets_intransit += fs

    for pid, ships in per_player_ships.items():
        if pid != player_id:
            strongest_enemy = max(strongest_enemy, ships)

    my_total_all = per_player_ships.get(player_id, 0)

    for i in range(N):
        x_n = planet_x[i] / 100.0
        y_n = planet_y[i] / 100.0
        is_mine = float(planet_owner[i] == player_id)
        is_enemy = float(planet_owner[i] >= 0 and planet_owner[i] != player_id)
        is_neutral = float(planet_owner[i] < 0)
        log_ships = math.log1p(planet_ships[i]) / 6.5
        prod_n = planet_prod[i] / 10.0
        ships_i = planet_ships[i]

        threat_ratio = min(inbound_enemy[i] / max(ships_i, 1), 3.0) / 3.0
        friendly_log = math.log1p(inbound_friendly[i]) / 6.5
        enemy_log = math.log1p(inbound_enemy[i]) / 6.5
        future_prod = planet_prod[i] * (max_steps - step) / max(max_steps, 1)
        prod_adv = min(max((my_total_prod - enemy_total_prod) / 20.0, -1.0), 1.0)

        nearest_enemy_dist = 200.0
        nearest_enemy_eta = 50.0
        for j in range(N):
            if planet_owner[j] >= 0 and planet_owner[j] != player_id:
                d = math.sqrt((planet_x[j] - planet_x[i]) ** 2 + (planet_y[j] - planet_y[i]) ** 2)
                if d < nearest_enemy_dist:
                    nearest_enemy_dist = d

        surplus = min(max((ships_i - inbound_enemy[i]) / 6.5, -1.0), 1.0)

        friendly_near = 0
        enemy_near = 0
        for j in range(N):
            if j == i:
                continue
            d = math.sqrt((planet_x[j] - planet_x[i]) ** 2 + (planet_y[j] - planet_y[i]) ** 2)
            if d <= 30.0:
                if planet_owner[j] == player_id:
                    friendly_near += 1
                elif planet_owner[j] >= 0:
                    enemy_near += 1

        nearest_friendly = 200.0
        for j in range(N):
            if j != i and planet_owner[j] == player_id:
                d = math.sqrt((planet_x[j] - planet_x[i]) ** 2 + (planet_y[j] - planet_y[i]) ** 2)
                if d < nearest_friendly:
                    nearest_friendly = d

        is_frontline = float(nearest_enemy_dist < nearest_friendly) if is_mine else 0.0

        inbound_f_count = 0
        inbound_e_count = 0
        for f in fleets:
            f_x, f_y = float(f[2]), float(f[3])
            d = math.sqrt((f_x - planet_x[i]) ** 2 + (f_y - planet_y[i]) ** 2)
            if d < 40.0:
                if int(f[1]) == player_id:
                    inbound_f_count += 1
                else:
                    inbound_e_count += 1

        local_balance = 0.0
        total_near = friendly_near + enemy_near
        if total_near > 0:
            local_balance = (friendly_near - enemy_near) / total_near

        feats[i] = torch.tensor([
            x_n, y_n,
            is_mine, is_enemy, is_neutral,
            log_ships, prod_n,
            threat_ratio, friendly_log, enemy_log,
            step_progress,
            future_prod / 10.0,
            prod_adv,
            min(nearest_enemy_dist / 100.0, 1.0),
            min(nearest_enemy_eta / 50.0, 1.0),
            surplus,
            planet_orbit_vel[i] / 0.05,
            num_players / 4.0,
            my_planet_count / max(total_owned, 1),
            my_total_all / max(total_ships_global, 1),
            strongest_enemy / max(total_ships_global, 1),
            min(friendly_near / 5.0, 1.0),
            min(enemy_near / 5.0, 1.0),
            local_balance,
            is_frontline,
            min(inbound_f_count / 5.0, 1.0),
            min(inbound_e_count / 5.0, 1.0),
            my_fleets_intransit / max(my_total_all, 1),
        ])

    return feats


def compute_candidate_edges(planets, fleets, player_id, num_players, step, max_steps,
                            max_candidates=MAX_CANDIDATES, min_ships_to_send=10,
                            angular_velocity=0.0):
    N = len(planets)
    planet_ships = [float(p[5]) for p in planets]
    planet_prod = [float(p[6]) for p in planets]
    planet_owner = [int(p[1]) for p in planets]
    planet_x = [float(p[2]) for p in planets]
    planet_y = [float(p[3]) for p in planets]

    planet_radius = [float(p[4]) for p in planets]
    planet_orbit_vel = []
    for i in range(N):
        orbital_r = math.hypot(planet_x[i] - SUN_X, planet_y[i] - SUN_Y)
        if orbital_r + planet_radius[i] < INNER_ORBIT_THRESHOLD:
            planet_orbit_vel.append(angular_velocity)
        else:
            planet_orbit_vel.append(0.0)

    my_planets = [i for i in range(N) if planet_owner[i] == player_id and planet_ships[i] >= min_ships_to_send]

    if not my_planets:
        return (torch.zeros(max_candidates, EDGE_INPUT_DIM),
                torch.zeros(max_candidates, 2, dtype=torch.long),
                torch.zeros(max_candidates), 0)

    # Inbound ships (with fleet-to-nearest mapping)
    inbound_friendly = [0.0] * N
    inbound_enemy = [0.0] * N
    fleet_nearest_planet = {}
    in_flight_to = {}
    for fi, f in enumerate(fleets):
        f_owner, f_ships = int(f[1]), float(f[6])
        f_x, f_y = float(f[2]), float(f[3])
        best_dist, best_idx = float('inf'), 0
        for i in range(N):
            d = math.sqrt((f_x - planet_x[i]) ** 2 + (f_y - planet_y[i]) ** 2)
            if d < best_dist:
                best_dist, best_idx = d, i
        fleet_nearest_planet[fi] = best_idx
        if f_owner == player_id:
            inbound_friendly[best_idx] += f_ships
            in_flight_to[best_idx] = in_flight_to.get(best_idx, 0) + f_ships
        else:
            inbound_enemy[best_idx] += f_ships

    # Threat assessment (v131-style)
    threats = [0.0] * N
    for fi, f in enumerate(fleets):
        f_owner = int(f[1])
        if f_owner == player_id or f_owner < 0:
            continue
        nearest = fleet_nearest_planet.get(fi, -1)
        if nearest >= 0 and planet_owner[nearest] == player_id:
            f_x, f_y = float(f[2]), float(f[3])
            d = math.sqrt((f_x - planet_x[nearest]) ** 2 + (f_y - planet_y[nearest]) ** 2)
            if d < 400:
                threats[nearest] += float(f[6])

    # Denial scoring (v131-style)
    denial_bonus = {}

    # Reactive denial: enemy fleets heading for neutrals
    for fi, f in enumerate(fleets):
        f_owner = int(f[1])
        if f_owner == player_id or f_owner < 0:
            continue
        nearest = fleet_nearest_planet.get(fi, -1)
        if nearest < 0 or planet_owner[nearest] != -1:
            continue
        f_x, f_y = float(f[2]), float(f[3])
        f_ships = max(int(f[6]), 1)
        eta = _travel_time(f_x, f_y, planet_x[nearest], planet_y[nearest], f_ships)
        dbonus = (DENIAL_BASE_SCORE
                  + planet_prod[nearest] * DENIAL_PROD_WEIGHT
                  + (1.0 / max(eta, 0.5)) * DENIAL_URGENCY_WEIGHT)
        if nearest not in denial_bonus or dbonus > denial_bonus[nearest][0]:
            denial_bonus[nearest] = (dbonus, eta)

    # Proactive denial: high-prod neutrals enemy can reach
    enemy_planets = [i for i in range(N)
                     if planet_owner[i] >= 0 and planet_owner[i] != player_id]
    neutral_planets = [i for i in range(N) if planet_owner[i] < 0]

    for ni in neutral_planets:
        if planet_prod[ni] < HIGH_PROD_THRESHOLD:
            continue
        if ni in denial_bonus:
            continue
        if not enemy_planets:
            continue
        min_enemy_tt = min(_travel_time(planet_x[ei], planet_y[ei],
                                         planet_x[ni], planet_y[ni], 20)
                          for ei in enemy_planets)
        if min_enemy_tt > 20:
            continue
        min_my_tt = min(_travel_time(planet_x[mi], planet_y[mi],
                                      planet_x[ni], planet_y[ni], 20)
                        for mi in my_planets)
        if min_my_tt < min_enemy_tt:
            denial_bonus[ni] = (PROACTIVE_BONUS, min_enemy_tt)

    # Sniper nest denial: neutrals close to our big planets
    big_planets = [i for i in my_planets if planet_ships[i] >= SNIPER_MIN_SHIPS]
    for ni in neutral_planets:
        for bi in big_planets:
            sd = math.sqrt((planet_x[ni] - planet_x[bi]) ** 2 +
                           (planet_y[ni] - planet_y[bi]) ** 2)
            if sd <= SNIPER_RANGE:
                sb = SNIPER_BONUS + planet_prod[ni] * DENIAL_PROD_WEIGHT
                if ni not in denial_bonus or sb > denial_bonus[ni][0]:
                    my_tt = _travel_time(planet_x[bi], planet_y[bi],
                                          planet_x[ni], planet_y[ni], 20)
                    denial_bonus[ni] = (sb, my_tt)
                break

    # Global stats
    my_total_ships = sum(planet_ships[i] for i in range(N) if planet_owner[i] == player_id)
    my_total_prod = sum(planet_prod[i] for i in range(N) if planet_owner[i] == player_id)
    enemy_total_prod = sum(planet_prod[i] for i in enemy_planets)
    total_prod = sum(planet_prod[i] for i in range(N) if planet_owner[i] >= 0)
    step_progress = step / max(max_steps, 1)

    # Phase detection (simplified v131)
    my_planet_count = len(my_planets)
    enemy_planet_count = len(enemy_planets)
    prod_ratio = my_total_prod / max(enemy_total_prod, 0.1)
    ship_ratio = my_total_ships / max(
        sum(planet_ships[i] for i in enemy_planets), 0.1)

    is_early = step_progress < 0.1
    is_pressure = (step_progress >= 0.07 and prod_ratio >= 0.95 and ship_ratio >= 0.9)
    is_aggressive = (prod_ratio >= 2.0 or ship_ratio >= 2.5)
    is_cleanup = (step_progress >= 0.28 and enemy_planet_count <= 4
                  and prod_ratio >= 1.9 and ship_ratio >= 1.8)

    # Score all edges
    scored_edges = []
    for src_idx in my_planets:
        sx, sy = planet_x[src_idx], planet_y[src_idx]
        src_ships = planet_ships[src_idx]

        threat = threats[src_idx]
        under_threat = threat > src_ships * 0.4
        if under_threat and my_planet_count > 1:
            threat_penalty = -20.0
        else:
            threat_penalty = 0.0

        for tgt_idx in range(N):
            if tgt_idx == src_idx:
                continue

            tx, ty = planet_x[tgt_idx], planet_y[tgt_idx]
            tgt_owner = planet_owner[tgt_idx]
            tgt_ships = planet_ships[tgt_idx]
            tgt_prod = planet_prod[tgt_idx]

            # Skip own planets (v131: hold ships, don't shuffle)
            if tgt_owner == player_id:
                continue

            dist = math.sqrt((tx - sx) ** 2 + (ty - sy) ** 2)
            if dist < 1e-6:
                continue

            sun_blocked = sun_intersects_path(sx, sy, tx, ty)

            # v131-style travel time (speed depends on fleet size)
            tt = _travel_time(sx, sy, tx, ty, int(src_ships * 0.5))

            # Max ETA filter (v131-style, phase-aware)
            max_eta = 14 if is_early else (20 if step_progress < 0.4 else 30)
            if tt > max_eta:
                continue

            # v131 scoring formula
            score = tgt_prod * SCORE_PROD_WEIGHT - tt * SCORE_TT_PENALTY

            needed = _ships_needed_for_takeover(tgt_ships, tgt_prod, tt, tgt_owner)
            can_take = src_ships >= needed

            if tgt_owner < 0:  # neutral
                score += NEUTRAL_BONUS
                if can_take:
                    score += 10.0
            else:  # enemy
                if is_cleanup:
                    score += 55.0 - tgt_ships * 0.04
                elif is_aggressive:
                    score += ENEMY_BONUS_AGGRO - tgt_ships * 0.12
                elif is_pressure:
                    score += ENEMY_BONUS_PRESSURE - tgt_ships * 0.07
                else:
                    score += ENEMY_BONUS_EARLY - tgt_ships * 0.1
                if can_take:
                    score += 15.0

            # Denial bonus (v131)
            if tgt_owner < 0 and tgt_idx in denial_bonus:
                dbonus, enemy_eta = denial_bonus[tgt_idx]
                if tt <= enemy_eta:
                    score += dbonus

            # Sun penalty
            if sun_blocked:
                score -= 15.0

            # Orbit penalty
            orbit_vel = abs(planet_orbit_vel[tgt_idx])
            if orbit_vel < 0.001:
                score += 2.0
            else:
                score -= 6.0

            # Already-targeted penalty
            already_sent = in_flight_to.get(tgt_idx, 0)
            if already_sent > 0:
                score -= min(already_sent / 50.0, 10.0) * 3.0

            # ROI
            if needed > 0:
                roi = tgt_prod / needed
                score += roi * 5.0

            score += threat_penalty
            scored_edges.append((score, src_idx, tgt_idx))

    # Top-K with exploration wild cards
    scored_edges.sort(key=lambda x: x[0], reverse=True)
    n_heuristic = int(max_candidates * 0.8)
    heuristic_edges = scored_edges[:n_heuristic]
    remaining = scored_edges[n_heuristic:]
    n_explore = min(max_candidates - len(heuristic_edges), len(remaining))
    if n_explore > 0 and remaining:
        stride = max(1, len(remaining) // n_explore)
        explore_edges = remaining[::stride][:n_explore]
    else:
        explore_edges = []

    top_edges = heuristic_edges + explore_edges
    num_valid = len(top_edges)

    edge_features = torch.zeros(max_candidates, EDGE_INPUT_DIM)
    edge_indices = torch.zeros(max_candidates, 2, dtype=torch.long)
    edge_mask = torch.zeros(max_candidates)

    node_feats_raw = _build_node_features_compact(
        planets, fleets, player_id, num_players, step, max_steps,
        planet_ships, planet_prod, planet_owner, planet_x, planet_y,
        planet_orbit_vel, inbound_friendly, inbound_enemy,
        my_total_ships, my_total_prod, total_prod,
    )

    for k, (heur_score, src_idx, tgt_idx) in enumerate(top_edges):
        src_feats = node_feats_raw[src_idx]
        tgt_feats = node_feats_raw[tgt_idx]

        sx, sy = planet_x[src_idx], planet_y[src_idx]
        tx, ty = planet_x[tgt_idx], planet_y[tgt_idx]
        dist = math.sqrt((tx - sx) ** 2 + (ty - sy) ** 2)
        dist_norm = dist / BOARD_DIAGONAL
        travel_time_val = dist / 6.0
        angle = math.atan2(ty - sy, tx - sx)
        sun_blocked = sun_intersects_path(sx, sy, tx, ty)
        sun_clear = closest_approach_to_sun(sx, sy, tx, ty) / BOARD_DIAGONAL

        spatial = torch.tensor([
            dist_norm, travel_time_val / 25.0,
            math.sin(angle), math.cos(angle),
            float(sun_blocked), min(sun_clear, 1.0),
        ])

        tgt_owner = planet_owner[tgt_idx]
        tgt_ships_val = planet_ships[tgt_idx]
        tgt_prod_val = planet_prod[tgt_idx]
        src_ships_val = planet_ships[src_idx]
        defense = tgt_ships_val + (inbound_enemy[tgt_idx] if tgt_owner != player_id else 0)

        heuristic_feats = torch.tensor([
            min(heur_score / 10.0, 1.0),
            min(src_ships_val / max(defense + 1, 1), 3.0) / 3.0,
            float(tgt_owner < 0),
            float(tgt_owner >= 0 and tgt_owner != player_id),
            float(tgt_owner == player_id),
            tgt_prod_val / 5.0,
            min(inbound_friendly[tgt_idx] / 100.0, 1.0),
            min(inbound_enemy[tgt_idx] / 100.0, 1.0),
            min(defense / 100.0, 1.0),
            min(src_ships_val / 100.0, 1.0),
            step_progress,
            float(k) / max(num_valid - 1, 1),
        ])

        edge_features[k] = torch.cat([src_feats, tgt_feats, spatial, heuristic_feats])
        edge_indices[k, 0] = src_idx
        edge_indices[k, 1] = tgt_idx
        edge_mask[k] = 1.0

    return edge_features, edge_indices, edge_mask, num_valid


# ---------------------------------------------------------------------------
# Transformer model (matches sb3_feature_extractor.py)
# ---------------------------------------------------------------------------


class EdgeTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        if mask is not None:
            B, K = mask.shape
            n_heads = self.attn.num_heads
            float_mask = torch.zeros(B, K, K, device=x.device, dtype=x.dtype)
            pad_cols = mask.unsqueeze(1).expand(-1, K, -1)
            float_mask = float_mask.masked_fill(pad_cols, -1e4)
            float_mask = float_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
            float_mask = float_mask.reshape(B * n_heads, K, K)
            attn_out, _ = self.attn(x, x, x, attn_mask=float_mask, need_weights=False)
        else:
            attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class PolicyNetwork(nn.Module):
    """Standalone policy network matching SB3 CandidateTransformerExtractor + MLP heads."""

    def __init__(self, d_model=128, n_heads=4, n_layers=3, n_candidates=48, edge_dim=74):
        super().__init__()
        self.n_candidates = n_candidates
        self.edge_dim = edge_dim

        # Feature extractor
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList([
            EdgeTransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        # Policy head
        self.pi_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.action_net = nn.Linear(d_model, NUM_CHOICES * MAX_ACTIONS)

    def forward(self, obs_flat, action_mask):
        B = obs_flat.shape[0]
        x = obs_flat.view(B, self.n_candidates, self.edge_dim)
        pad_mask = (x.abs().sum(dim=-1) < 1e-6)
        x = x.clamp(-50.0, 50.0)
        x = self.edge_encoder(x)
        x = x.clamp(-100.0, 100.0)
        for block in self.blocks:
            x = block(x, mask=pad_mask)

        valid_mask = (~pad_mask).unsqueeze(-1).float()
        valid_count = valid_mask.sum(dim=1).clamp(min=1)
        pooled = (x * valid_mask).sum(dim=1) / valid_count

        pi_features = self.pi_mlp(pooled)
        logits = self.action_net(pi_features)  # (B, NUM_CHOICES * MAX_ACTIONS)

        # Reshape to (B, MAX_ACTIONS, NUM_CHOICES) and apply mask
        logits = logits.view(B, MAX_ACTIONS, NUM_CHOICES)
        mask_reshaped = action_mask.view(B, MAX_ACTIONS, NUM_CHOICES)
        logits = logits.masked_fill(~mask_reshaped, -1e8)

        return logits


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_MODEL = None
_DEVICE = None
_OBS_MEAN = None
_OBS_VAR = None
_CLIP_OBS = 10.0


def _load_model():
    global _MODEL, _DEVICE, _OBS_MEAN, _OBS_VAR, _CLIP_OBS
    if _MODEL is not None:
        return

    _DEVICE = torch.device("cpu")
    _MODEL = PolicyNetwork()

    # __file__ is not defined in Kaggle's exec() sandbox
    try:
        _dir = os.path.dirname(__file__)
    except NameError:
        _dir = "/kaggle_simulations/agent"

    # Load VecNormalize stats
    norm_path = os.path.join(_dir, "vec_normalize_stats.npz")
    if os.path.exists(norm_path):
        stats = np.load(norm_path)
        _OBS_MEAN = stats["mean"]
        _OBS_VAR = stats["var"]
        _CLIP_OBS = float(stats["clip_obs"][0])

    weights_path = os.path.join(_dir, "sb3_policy_weights.pt")
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")

    # SB3 policy.state_dict() is a flat OrderedDict with dotted keys like:
    #   features_extractor.edge_encoder.0.weight
    #   mlp_extractor.policy_net.0.weight
    #   action_net.0.bias
    # Extract sub-dicts by prefix
    def extract_prefix(prefix):
        p = prefix + "."
        return {k[len(p):]: v for k, v in state.items() if k.startswith(p)}

    fe_state = extract_prefix("features_extractor")

    _MODEL.edge_encoder.load_state_dict(
        {k.replace("edge_encoder.", ""): v for k, v in fe_state.items()
         if k.startswith("edge_encoder.")})

    # Load transformer blocks
    block_state = {k: v for k, v in fe_state.items() if k.startswith("blocks.")}
    blocks_sd = _MODEL.blocks.state_dict()
    for k in blocks_sd:
        full_key = f"blocks.{k}"
        if full_key in block_state:
            blocks_sd[k] = block_state[full_key]
    _MODEL.blocks.load_state_dict(blocks_sd)

    # Policy MLP + action net
    pi_state = extract_prefix("mlp_extractor.policy_net")
    _MODEL.pi_mlp.load_state_dict(pi_state)
    _MODEL.action_net.load_state_dict(extract_prefix("action_net"))

    _MODEL.eval()


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------


def agent(obs, config=None):
    _load_model()

    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])
    player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
    step = obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)
    angular_velocity = obs.get("angular_velocity", 0.0) if isinstance(obs, dict) else getattr(obs, "angular_velocity", 0.0)

    if not planets or not any(int(p[1]) == player for p in planets):
        return []

    ef, edge_indices, em, num_valid = compute_candidate_edges(
        planets=planets, fleets=fleets, player_id=player,
        num_players=2, step=step, max_steps=500,
        max_candidates=MAX_CANDIDATES, angular_velocity=angular_velocity,
    )
    ef = torch.nan_to_num(ef, nan=0.0, posinf=1.0, neginf=-1.0)
    obs_np = ef.numpy().flatten().astype(np.float32)

    # Apply VecNormalize (must match training normalization)
    if _OBS_MEAN is not None:
        obs_np = (obs_np - _OBS_MEAN) / np.sqrt(_OBS_VAR + 1e-8)
        obs_np = np.clip(obs_np, -_CLIP_OBS, _CLIP_OBS)

    obs_flat = torch.from_numpy(obs_np).unsqueeze(0)  # (1, 3552)

    # Build action mask
    nv = min(num_valid, MAX_CANDIDATES)
    single_mask = np.zeros(NUM_CHOICES, dtype=bool)
    for i in range(nv):
        single_mask[i * NUM_FRACTIONS: (i + 1) * NUM_FRACTIONS] = True
    single_mask[NOOP_ACTION] = True
    mask = np.tile(single_mask, MAX_ACTIONS)
    mask_t = torch.from_numpy(mask).unsqueeze(0)  # (1, MAX_ACTIONS * NUM_CHOICES)

    with torch.inference_mode():
        logits = _MODEL(obs_flat.view(1, -1), mask_t)  # (1, MAX_ACTIONS, NUM_CHOICES)
        actions_idx = logits.argmax(dim=-1).squeeze(0).numpy()  # (MAX_ACTIONS,)

    # Decode actions
    actions = []
    committed = {}

    for slot_action in actions_idx:
        slot_action = int(slot_action)
        if slot_action == NOOP_ACTION:
            continue

        cand_idx = slot_action // NUM_FRACTIONS
        frac_idx = slot_action % NUM_FRACTIONS
        if cand_idx >= nv:
            continue

        src_pidx = edge_indices[cand_idx, 0].item()
        tgt_pidx = edge_indices[cand_idx, 1].item()
        src_p, tgt_p = planets[src_pidx], planets[tgt_pidx]

        if int(src_p[1]) != player:
            continue

        sx, sy = float(src_p[2]), float(src_p[3])
        tx, ty = float(tgt_p[2]), float(tgt_p[3])
        tgt_planet_r = float(tgt_p[4])
        tgt_orbital_r = math.hypot(tx - 50.0, ty - 50.0)
        omega = angular_velocity if (tgt_orbital_r + tgt_planet_r < INNER_ORBIT_THRESHOLD) else 0.0

        src_fleet = int(float(src_p[5]))
        already = committed.get(src_pidx, 0)
        base_reserve = max(5, int(src_fleet * 0.15))
        available = src_fleet - base_reserve - already
        if available < 5:
            continue

        ships = max(1, int(available * FRACTION_BUCKETS[frac_idx]))
        if ships < 5:
            continue

        ix, iy = solve_intercept(sx, sy, tx, ty, omega, ships)

        # Sun avoidance
        if sun_intersects_path(sx, sy, ix, iy):
            dx, dy = ix - sx, iy - sy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 1e-6:
                mid_x, mid_y = (sx + ix) / 2, (sy + iy) / 2
                sun_dist = math.sqrt((mid_x - SUN_X) ** 2 + (mid_y - SUN_Y) ** 2)
                if sun_dist < 1e-6:
                    continue
                perp_x = -(SUN_Y - mid_y) / sun_dist
                perp_y = (SUN_X - mid_x) / sun_dist
                offset = 14.0 / dist
                ix2 = ix + perp_x * offset * dist * 0.3
                iy2 = iy + perp_y * offset * dist * 0.3
                if sun_intersects_path(sx, sy, ix2, iy2):
                    continue
                ix, iy = ix2, iy2

        angle = math.atan2(iy - sy, ix - sx)
        actions.append([int(src_p[0]), angle, ships])
        committed[src_pidx] = already + ships

    return actions
