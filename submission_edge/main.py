"""Standalone edge-based transformer agent for Orbit Wars.

Bundles: sun geometry, candidate edge generation, EdgePolicy, agent function.
Loads weights from weights.pt. Auto-detects 2p/4p.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Constants ──────────────────────────────────────────────────────────
BOARD_SIZE = 100.0
SUN_X, SUN_Y, SUN_RADIUS = 50.0, 50.0, 10.0
BOARD_DIAGONAL = math.sqrt(BOARD_SIZE**2 + BOARD_SIZE**2)
DEFAULT_SAFETY_MARGIN = 2.0

MAX_CANDIDATES = 192
MAX_ACTIONS = 5
FRACTION_BUCKETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
NUM_FRACTIONS = len(FRACTION_BUCKETS)

NUM_PHASES = 4
FEAT_IS_NEUTRAL = 64
FEAT_IS_ENEMY = 65
FEAT_IS_FRIENDLY = 66
EDGE_INPUT_DIM = 74


# ─── Sun geometry ───────────────────────────────────────────────────────
def closest_approach_to_sun(x1, y1, x2, y2, sun_x=SUN_X, sun_y=SUN_Y):
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return math.sqrt((x1 - sun_x)**2 + (y1 - sun_y)**2)
    t = ((sun_x - x1) * dx + (sun_y - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.sqrt((proj_x - sun_x)**2 + (proj_y - sun_y)**2)


def sun_intersects_path(x1, y1, x2, y2, sun_radius=SUN_RADIUS,
                        safety_margin=DEFAULT_SAFETY_MARGIN):
    return closest_approach_to_sun(x1, y1, x2, y2) < sun_radius + safety_margin


# ─── Fleet speed & orbit intercept ────────────────────────────────────
_LOG1000 = math.log(1000)
_MAX_SPEED_MINUS_1 = 5.0


def _fleet_speed(ships: int) -> float:
    if ships <= 0:
        return 1.0
    return 1.0 + _MAX_SPEED_MINUS_1 * (math.log(max(ships, 1)) / _LOG1000) ** 1.5


def _travel_time(x1, y1, x2, y2, ships) -> float:
    dx = x2 - x1
    dy = y2 - y1
    if ships <= 0:
        return 999.0
    return math.sqrt(dx * dx + dy * dy) / _fleet_speed(ships)


def predict_orbit(x: float, y: float, omega: float, dt: float):
    theta = math.atan2(y - SUN_Y, x - SUN_X)
    r = math.hypot(x - SUN_X, y - SUN_Y)
    return SUN_X + r * math.cos(theta + omega * dt), SUN_Y + r * math.sin(theta + omega * dt)


def solve_intercept(fx, fy, tx, ty, omega, ships, iterations=25):
    """Iteratively predict where an orbiting planet will be when our fleet arrives."""
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


# ─── Node features (28-dim) ────────────────────────────────────────────
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
    my_fleets_intransit = 0.0
    total_ships_global = sum(planet_ships)
    strongest_enemy = 0.0

    per_player_ships = {}
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
        fo = int(f[1])
        fs = float(f[6])
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
        future_prod = (planet_prod[i] * (max_steps - step) / max(max_steps, 1))
        prod_adv = min(max((my_total_prod - enemy_total_prod) / 20.0, -1.0), 1.0)

        nearest_enemy_dist = 200.0
        for j in range(N):
            if planet_owner[j] >= 0 and planet_owner[j] != player_id:
                d = math.sqrt((planet_x[j] - planet_x[i])**2 +
                              (planet_y[j] - planet_y[i])**2)
                if d < nearest_enemy_dist:
                    nearest_enemy_dist = d

        surplus = min(max((ships_i - inbound_enemy[i]) / 6.5, -1.0), 1.0)

        friendly_near = 0
        enemy_near = 0
        for j in range(N):
            if j == i:
                continue
            d = math.sqrt((planet_x[j] - planet_x[i])**2 +
                          (planet_y[j] - planet_y[i])**2)
            if d <= 30.0:
                if planet_owner[j] == player_id:
                    friendly_near += 1
                elif planet_owner[j] >= 0:
                    enemy_near += 1

        nearest_friendly = 200.0
        for j in range(N):
            if j != i and planet_owner[j] == player_id:
                d = math.sqrt((planet_x[j] - planet_x[i])**2 +
                              (planet_y[j] - planet_y[i])**2)
                if d < nearest_friendly:
                    nearest_friendly = d

        is_frontline = float(nearest_enemy_dist < nearest_friendly) if is_mine else 0.0

        inbound_f_count = 0
        inbound_e_count = 0
        for f in fleets:
            f_x, f_y = float(f[2]), float(f[3])
            d = math.sqrt((f_x - planet_x[i])**2 + (f_y - planet_y[i])**2)
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
            min(50.0 / 50.0, 1.0),  # nearest_enemy_eta placeholder
            surplus,
            planet_orbit_vel[i] / 4.0,
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


# ─── Candidate edge generation ─────────────────────────────────────────
def compute_candidate_edges(planets, fleets, player_id, num_players, step,
                            max_steps, max_candidates=MAX_CANDIDATES,
                            min_ships_to_send=2):
    N = len(planets)
    planet_ships = [float(p[5]) for p in planets]
    planet_prod = [float(p[6]) for p in planets]
    planet_owner = [int(p[1]) for p in planets]
    planet_x = [float(p[2]) for p in planets]
    planet_y = [float(p[3]) for p in planets]
    planet_orbit_vel = [float(p[4]) for p in planets]

    my_planets = [i for i in range(N)
                  if planet_owner[i] == player_id and planet_ships[i] >= min_ships_to_send]

    if not my_planets:
        return (
            torch.zeros(max_candidates, EDGE_INPUT_DIM),
            torch.zeros(max_candidates, 2, dtype=torch.long),
            torch.zeros(max_candidates),
            0,
        )

    inbound_friendly = [0.0] * N
    inbound_enemy = [0.0] * N
    for f in fleets:
        f_owner = int(f[1])
        f_ships = float(f[6])
        f_x, f_y = float(f[2]), float(f[3])
        best_dist = float('inf')
        best_idx = 0
        for i in range(N):
            d = math.sqrt((f_x - planet_x[i])**2 + (f_y - planet_y[i])**2)
            if d < best_dist:
                best_dist = d
                best_idx = i
        if f_owner == player_id:
            inbound_friendly[best_idx] += f_ships
        else:
            inbound_enemy[best_idx] += f_ships

    my_total_ships = sum(planet_ships[i] for i in range(N) if planet_owner[i] == player_id)
    my_total_prod = sum(planet_prod[i] for i in range(N) if planet_owner[i] == player_id)
    total_prod = sum(planet_prod[i] for i in range(N) if planet_owner[i] >= 0)
    step_progress = step / max(max_steps, 1)

    scored_edges = []
    for src_idx in my_planets:
        sx, sy = planet_x[src_idx], planet_y[src_idx]
        src_ships = planet_ships[src_idx]

        for tgt_idx in range(N):
            if tgt_idx == src_idx:
                continue

            tx, ty = planet_x[tgt_idx], planet_y[tgt_idx]
            tgt_owner = planet_owner[tgt_idx]
            tgt_ships = planet_ships[tgt_idx]
            tgt_prod = planet_prod[tgt_idx]

            dist = math.sqrt((tx - sx)**2 + (ty - sy)**2)
            if dist < 1e-6:
                continue

            sun_blocked = sun_intersects_path(sx, sy, tx, ty)

            score = 0.0
            dist_score = max(0, 1.0 - dist / BOARD_DIAGONAL) * 2.0
            score += dist_score

            prod_score = tgt_prod / 5.0
            score += prod_score

            defense = tgt_ships + inbound_enemy[tgt_idx] if tgt_owner != player_id else tgt_ships
            takeover_ratio = src_ships / max(defense + 1, 1)
            if tgt_owner < 0:
                score += min(takeover_ratio, 2.0) * 1.5
            elif tgt_owner != player_id:
                score += min(takeover_ratio, 2.0) * 2.0
            else:
                continue

            if sun_blocked:
                score -= 3.0

            orbit_vel = abs(planet_orbit_vel[tgt_idx])
            if orbit_vel < 0.1:
                score += 1.0
            elif orbit_vel > 1.0:
                score -= 0.5

            travel_time = dist / 6.0
            step_progress_local = step / max(max_steps, 1)
            max_eta = 14 if step_progress_local < 0.1 else (20 if step_progress_local < 0.4 else 30)
            if travel_time > max_eta:
                continue

            if defense > 0:
                roi = tgt_prod / (defense + 1)
                score += roi * 2.0

            scored_edges.append((score, src_idx, tgt_idx))

    # Top-K candidate selection by score
    scored_edges.sort(key=lambda x: x[0], reverse=True)
    top_edges = scored_edges[:max_candidates]
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
        dist = math.sqrt((tx - sx)**2 + (ty - sy)**2)
        dist_norm = dist / BOARD_DIAGONAL
        travel_time = dist / 6.0
        angle = math.atan2(ty - sy, tx - sx)
        sun_blocked = sun_intersects_path(sx, sy, tx, ty)
        sun_clear = closest_approach_to_sun(sx, sy, tx, ty) / BOARD_DIAGONAL

        spatial = torch.tensor([
            dist_norm, travel_time / 25.0,
            math.sin(angle), math.cos(angle),
            float(sun_blocked), min(sun_clear, 1.0),
        ])

        tgt_owner = planet_owner[tgt_idx]
        tgt_ships = planet_ships[tgt_idx]
        tgt_prod = planet_prod[tgt_idx]
        src_ships = planet_ships[src_idx]
        defense = tgt_ships + (inbound_enemy[tgt_idx] if tgt_owner != player_id else 0)

        heuristic = torch.tensor([
            min(heur_score / 10.0, 1.0),
            min(src_ships / max(defense + 1, 1), 3.0) / 3.0,
            float(tgt_owner < 0),
            float(tgt_owner >= 0 and tgt_owner != player_id),
            float(tgt_owner == player_id),
            tgt_prod / 5.0,
            min(inbound_friendly[tgt_idx] / 100.0, 1.0),
            min(inbound_enemy[tgt_idx] / 100.0, 1.0),
            min(defense / 100.0, 1.0),
            min(src_ships / 100.0, 1.0),
            step_progress,
            float(k) / max(num_valid - 1, 1),
        ])

        edge_features[k] = torch.cat([src_feats, tgt_feats, spatial, heuristic])
        edge_indices[k] = torch.tensor([src_idx, tgt_idx])
        edge_mask[k] = 1.0

    return edge_features, edge_indices, edge_mask, num_valid


# ─── Transformer block ─────────────────────────────────────────────────
class EdgeTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        if mask is not None:
            B, K_seq = mask.shape
            n_heads = self.attn.num_heads
            float_mask = torch.zeros(B, K_seq, K_seq, device=x.device, dtype=x.dtype)
            pad_cols = mask.unsqueeze(1).expand(-1, K_seq, -1)
            float_mask = float_mask.masked_fill(pad_cols, -1e4)
            float_mask = float_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
            float_mask = float_mask.reshape(B * n_heads, K_seq, K_seq)
            attn_out, _ = self.attn(x, x, x, attn_mask=float_mask, need_weights=False)
        else:
            attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# ─── EdgePolicy (inference only) ───────────────────────────────────────
class EdgePolicy(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=3, dropout=0.1,
                 max_actions=MAX_ACTIONS, separate_critic=True):
        super().__init__()
        self.d_model = d_model
        self.max_actions = max_actions
        self.separate_critic = separate_critic

        self.edge_encoder = nn.Sequential(
            nn.Linear(EDGE_INPUT_DIM, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.transformer = nn.ModuleList([
            EdgeTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.selection_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        self.noop_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        self.fraction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, NUM_FRACTIONS),
        )

        self.phase_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, NUM_PHASES),
        )

        if separate_critic:
            self.critic_encoder = nn.Sequential(
                nn.Linear(EDGE_INPUT_DIM, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            self.critic_transformer = nn.ModuleList([
                EdgeTransformerBlock(d_model, n_heads, dropout)
                for _ in range(n_layers)
            ])

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def _encode(self, edge_features, edge_mask):
        edge_features = edge_features.clamp(-50.0, 50.0)
        h = self.edge_encoder(edge_features)
        h = h.clamp(-100.0, 100.0)
        pad_mask = (edge_mask < 0.5)
        for block in self.transformer:
            h = block(h, mask=pad_mask)
            h = h.clamp(-100.0, 100.0)
        return h

    def forward(self, edge_features, edge_mask):
        h = self._encode(edge_features, edge_mask)

        edge_logits = self.selection_head(h).squeeze(-1)
        edge_logits = edge_logits.masked_fill(edge_mask < 0.5, float('-inf'))

        valid_count = edge_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        g = (h * edge_mask.unsqueeze(-1)).sum(dim=1) / valid_count
        noop_logit = self.noop_head(g)

        frac_logits = self.fraction_head(h)

        return noop_logit, edge_logits, frac_logits

    def sample_action(self, edge_features, edge_mask, deterministic=False):
        ef = edge_features.unsqueeze(0)
        em = edge_mask.unsqueeze(0)

        with torch.no_grad():
            noop_logit, edge_logits, frac_logits = self.forward(ef, em)

        noop_logit = noop_logit[0, 0]
        edge_logits = edge_logits[0]
        frac_logits = frac_logits[0]

        if torch.isnan(noop_logit):
            noop_logit = torch.tensor(0.0)
        if torch.isnan(edge_logits).any():
            edge_logits = torch.nan_to_num(edge_logits, nan=0.0)
        if torch.isnan(frac_logits).any():
            frac_logits = torch.nan_to_num(frac_logits, nan=0.0)

        # Binary noop decision
        noop_prob = torch.sigmoid(noop_logit)
        if deterministic:
            do_noop = noop_prob > 0.5
        else:
            do_noop = torch.bernoulli(noop_prob).bool()

        if do_noop:
            return [], [], True

        # Sample edges
        selected = []
        fractions = []
        remaining_mask = edge_mask.clone()

        for _ in range(self.max_actions):
            masked_logits = edge_logits.clone()
            masked_logits = masked_logits.masked_fill(remaining_mask < 0.5, float('-inf'))

            if (remaining_mask > 0.5).sum() == 0:
                break

            if deterministic:
                chosen = masked_logits.argmax().item()
            else:
                dist = torch.distributions.Categorical(logits=masked_logits)
                chosen = dist.sample().item()

            selected.append(chosen)
            remaining_mask[chosen] = 0.0

            f_logits = frac_logits[chosen]
            if deterministic:
                frac_idx = f_logits.argmax().item()
            else:
                f_dist = torch.distributions.Categorical(logits=f_logits)
                frac_idx = f_dist.sample().item()

            fractions.append(frac_idx)

        return selected, fractions, len(selected) == 0


# ─── Load model ─────────────────────────────────────────────────────────
import inspect as _inspect
_THIS_DIR = os.path.dirname(os.path.abspath(
    globals().get("__file__", _inspect.getfile(lambda: None))
))
_WEIGHTS_PATH = os.path.join(_THIS_DIR, "weights.pt")
_model = EdgePolicy(d_model=128, n_heads=4, n_layers=3, separate_critic=True)
_state = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=False)
if isinstance(_state, dict) and "model_state_dict" in _state:
    _state = _state["model_state_dict"]
_model.load_state_dict(_state, strict=False)  # strict=False: critic loaded but unused
_model.eval()

_step_counter = 0


def agent(obs, config):
    global _step_counter

    planets = obs.get("planets", [])
    if not planets:
        return []

    player = obs.get("player", 0)
    fleets = obs.get("fleets", [])
    step = obs.get("step", _step_counter)
    max_steps = config.get("episodeSteps", 400) if config else 400

    has_planets = any(int(p[1]) == player for p in planets)
    if not has_planets:
        return []

    owners = set(int(p[1]) for p in planets if int(p[1]) >= 0)
    for f in fleets:
        fo = int(f[1])
        if fo >= 0:
            owners.add(fo)
    num_players = max(len(owners), 2)

    edge_features, edge_indices, edge_mask, num_valid = compute_candidate_edges(
        planets, fleets, player, num_players, step, max_steps,
    )

    if num_valid == 0:
        _step_counter += 1
        return []

    selected, fractions, is_noop = _model.sample_action(
        edge_features, edge_mask, deterministic=True,
    )

    if is_noop:
        _step_counter += 1
        return []

    actions = []
    for edge_idx, frac_idx in zip(selected, fractions):
        src_idx, tgt_idx = edge_indices[edge_idx].tolist()
        src_planet = planets[src_idx]
        tgt_planet = planets[tgt_idx]
        sx, sy = float(src_planet[2]), float(src_planet[3])
        tx, ty = float(tgt_planet[2]), float(tgt_planet[3])

        ships = int(float(src_planet[5]) * FRACTION_BUCKETS[frac_idx])
        if ships < 1:
            continue

        # Predict where the orbiting target will be when our fleet arrives
        omega = float(tgt_planet[4])  # orbit_vel
        ix, iy = solve_intercept(sx, sy, tx, ty, omega, ships)
        angle = math.atan2(iy - sy, ix - sx)
        actions.append([int(src_planet[0]), angle, ships])

    _step_counter += 1
    return actions
