"""Standalone GNN agent for Orbit Wars — self-contained submission.

Bundles: sun geometry, node feature builder, GNN policy (GAT), agent function.
Loads weights from weights.pt. Auto-detects 2p/4p.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Constants ──────────────────────────────────────────────────────────
BOARD_SIZE = 100.0
SUN_X, SUN_Y, SUN_RADIUS = 50.0, 50.0, 10.0
SAFETY_MARGIN = 2.0
DEFAULT_SHIP_SPEED = 1.0
NODE_DIM = 16
EDGE_DIM = 6
FRACTION_BUCKETS = [0.25, 0.50, 0.75, 1.00]
NOOP_PENALTY = 8.0
HIDDEN_DIM = 64

# ─── Sun geometry ───────────────────────────────────────────────────────
def closest_approach_to_sun(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return math.sqrt((x1 - SUN_X)**2 + (y1 - SUN_Y)**2)
    t = max(0.0, min(1.0, ((SUN_X - x1) * dx + (SUN_Y - y1) * dy) / seg_len_sq))
    proj_x, proj_y = x1 + t * dx, y1 + t * dy
    return math.sqrt((proj_x - SUN_X)**2 + (proj_y - SUN_Y)**2)

def sun_intersects_path(x1, y1, x2, y2):
    return closest_approach_to_sun(x1, y1, x2, y2) < SUN_RADIUS + SAFETY_MARGIN

def compute_sun_edge_features_batch(positions, sun_x=SUN_X, sun_y=SUN_Y,
                                     sun_radius=SUN_RADIUS, safety_margin=SAFETY_MARGIN):
    B, N, _ = positions.shape
    board_diag = math.sqrt(BOARD_SIZE**2 + BOARD_SIZE**2)
    sun = torch.tensor([sun_x, sun_y], dtype=positions.dtype, device=positions.device)
    p1 = positions.unsqueeze(2)
    p2 = positions.unsqueeze(1)
    d = p2 - p1
    seg_len_sq = (d * d).sum(dim=-1, keepdim=True).clamp(min=1e-12)
    sun_offset = sun.view(1, 1, 1, 2) - p1
    t = (sun_offset * d).sum(dim=-1, keepdim=True) / seg_len_sq
    t = t.clamp(0.0, 1.0)
    proj = p1 + t * d
    dist = torch.norm(proj - sun.view(1, 1, 1, 2), dim=-1)
    sun_intersects = (dist < sun_radius + safety_margin).float()
    sun_clearance = (dist / board_diag).clamp(0.0, 1.0)
    return sun_intersects, sun_clearance


# ─── Feature builder ───────────────────────────────────────────────────
def _build_node_features(planets, fleets, player_id, num_players, step=0, max_steps=400):
    N = len(planets)
    nf = torch.zeros(N, NODE_DIM)
    pos = torch.zeros(N, 2)
    owned = torch.zeros(N)

    inbound_friendly = [0.0] * N
    inbound_enemy = [0.0] * N
    pid_to_idx = {}
    for i, p in enumerate(planets):
        pid_to_idx[int(p[0])] = i

    for fleet in fleets:
        f_owner = int(fleet[1])
        f_ships = float(fleet[6])
        fx, fy = float(fleet[2]), float(fleet[3])
        f_angle = float(fleet[4])
        cos_a, sin_a = math.cos(f_angle), math.sin(f_angle)
        best_perp, best_idx = float("inf"), None
        for i, p in enumerate(planets):
            px, py = float(p[2]), float(p[3])
            dx, dy = px - fx, py - fy
            dot = dx * cos_a + dy * sin_a
            if dot <= 0:
                continue
            perp = abs(dx * sin_a - dy * cos_a)
            if perp < best_perp:
                best_perp = perp
                best_idx = i
        if best_idx is not None and best_perp < 12.0:
            if f_owner == player_id:
                inbound_friendly[best_idx] += f_ships
            else:
                inbound_enemy[best_idx] += f_ships

    for i, p in enumerate(planets):
        px, py = float(p[2]), float(p[3])
        p_owner = int(p[1])
        p_ships = float(p[5])
        p_prod = float(p[6])
        pos[i, 0] = px / BOARD_SIZE
        pos[i, 1] = py / BOARD_SIZE
        is_mine = 1.0 if p_owner == player_id else 0.0
        is_enemy = 1.0 if p_owner >= 0 and p_owner != player_id else 0.0
        is_neutral = 1.0 if p_owner < 0 else 0.0
        nf[i, 0] = px / BOARD_SIZE
        nf[i, 1] = py / BOARD_SIZE
        nf[i, 2] = is_mine
        nf[i, 3] = is_enemy
        nf[i, 4] = is_neutral
        nf[i, 5] = math.log1p(p_ships) / 6.5
        nf[i, 6] = p_prod / 10.0
        nf[i, 7] = min(inbound_enemy[i] / max(1.0, p_ships), 3.0) / 3.0
        nf[i, 8] = math.log1p(inbound_friendly[i]) / 6.5
        nf[i, 9] = math.log1p(inbound_enemy[i]) / 6.5
        if is_mine > 0 and p_ships > 0:
            owned[i] = 1.0

    my_total_prod = sum(float(p[6]) for p in planets if int(p[1]) == player_id)
    enemy_total_prod = sum(float(p[6]) for p in planets
                          if int(p[1]) >= 0 and int(p[1]) != player_id)
    remaining_frac = max(0.0, (max_steps - step)) / max_steps
    step_frac = step / max_steps

    for i, p in enumerate(planets):
        px, py = float(p[2]), float(p[3])
        p_owner = int(p[1])
        p_prod = float(p[6])
        is_mine = (p_owner == player_id)
        is_enemy_flag = (p_owner >= 0 and p_owner != player_id)

        nf[i, 10] = step_frac
        nf[i, 11] = p_prod * remaining_frac * 2.0
        nf[i, 12] = min(max((my_total_prod - enemy_total_prod) / 20.0, -1.0), 1.0)

        min_enemy_dist = 100.0
        for j, p2 in enumerate(planets):
            if j == i:
                continue
            if int(p2[1]) >= 0 and int(p2[1]) != player_id:
                dx = px - float(p2[2])
                dy = py - float(p2[3])
                d = math.sqrt(dx * dx + dy * dy)
                if d < min_enemy_dist:
                    min_enemy_dist = d
        nf[i, 13] = min_enemy_dist / 100.0

        min_enemy_eta = 1.0
        for fleet in fleets:
            f_owner = int(fleet[1])
            if f_owner == player_id or f_owner < 0:
                continue
            fx, fy = float(fleet[2]), float(fleet[3])
            f_angle = float(fleet[4])
            dx, dy = px - fx, py - fy
            dot = dx * math.cos(f_angle) + dy * math.sin(f_angle)
            if dot <= 0:
                continue
            dist = math.sqrt(dx * dx + dy * dy)
            eta = dist / DEFAULT_SHIP_SPEED
            norm_eta = min(eta / 50.0, 1.0)
            if norm_eta < min_enemy_eta:
                min_enemy_eta = norm_eta
        nf[i, 14] = min_enemy_eta

        if is_mine:
            surplus = float(nf[i, 5]) * 6.5
            threat = float(nf[i, 9]) * 6.5
            nf[i, 15] = min(max((surplus - threat) / 6.5, -1.0), 1.0)
        elif is_enemy_flag:
            nf[i, 15] = -0.5
        else:
            nf[i, 15] = 0.0

    return nf, pos, owned


# ─── GNN Policy ────────────────────────────────────────────────────────
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, num_heads=4):
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.a_tgt = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.a_edge = nn.Linear(edge_dim, num_heads, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, h, edge_feats):
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.num_heads, self.head_dim)
        score_src = (Wh * self.a_src).sum(-1)
        score_tgt = (Wh * self.a_tgt).sum(-1)
        score_edge = self.a_edge(edge_feats)
        attn = score_src.unsqueeze(2) + score_tgt.unsqueeze(1) + score_edge
        attn = F.leaky_relu(attn, 0.2)
        attn = F.softmax(attn, dim=2)
        out = (attn.unsqueeze(-1) * Wh.unsqueeze(1)).sum(dim=2)
        out = out.reshape(B, N, -1)
        return self.norm(out + self.residual(h))


class OrbitWarsGNNPolicy(nn.Module):
    def __init__(self, hidden_dim=64, use_gat=True, mask_sun_targets=True,
                 sun_safety_margin=2.0, ship_speed=6.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_gat = use_gat
        self.mask_sun_targets = mask_sun_targets
        self.sun_safety_margin = sun_safety_margin
        self.ship_speed = ship_speed

        self.node_encoder = nn.Sequential(
            nn.Linear(NODE_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        edge_enc_dim = hidden_dim // 2
        self.edge_encoder = nn.Sequential(
            nn.Linear(EDGE_DIM, edge_enc_dim), nn.ReLU(),
            nn.Linear(edge_enc_dim, edge_enc_dim),
        )
        self.gnn1 = GATLayer(hidden_dim, hidden_dim, edge_enc_dim, 4)
        self.gnn2 = GATLayer(hidden_dim, hidden_dim, edge_enc_dim, 4)
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)

        self.source_head = nn.Linear(hidden_dim, 1)
        self.noop_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        target_in = hidden_dim + hidden_dim + edge_enc_dim + hidden_dim
        self.target_head = nn.Sequential(
            nn.Linear(target_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )
        frac_in = hidden_dim + hidden_dim + hidden_dim
        self.fraction_head = nn.Sequential(
            nn.Linear(frac_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, len(FRACTION_BUCKETS)),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )

    def compute_edge_features(self, positions):
        B, N, _ = positions.shape
        board_diag = math.sqrt(100**2 + 100**2)
        p1 = positions.unsqueeze(2)
        p2 = positions.unsqueeze(1)
        diff = p2 - p1
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)
        dist_norm = dist / board_diag
        travel_time = dist / self.ship_speed
        angle = torch.atan2(diff[..., 1:2], diff[..., 0:1])
        angle_sin = torch.sin(angle)
        angle_cos = torch.cos(angle)
        sun_inter, sun_clear = compute_sun_edge_features_batch(
            positions, safety_margin=self.sun_safety_margin)
        return torch.cat([dist_norm, travel_time, angle_sin, angle_cos,
                          sun_inter.unsqueeze(-1), sun_clear.unsqueeze(-1)], dim=-1)

    def forward(self, node_features, positions, owned_mask):
        B, N, _ = node_features.shape
        h = self.node_encoder(node_features)
        raw_edges = self.compute_edge_features(positions)
        edge_enc = self.edge_encoder(raw_edges)
        h = self.gnn1(h, edge_enc)
        h = self.gnn2(h, edge_enc)
        g = F.relu(self.global_proj(h.mean(dim=1)))

        src_logits = self.source_head(h).squeeze(-1)
        noop_logit = self.noop_head(g)
        src_logits = src_logits.masked_fill(owned_mask == 0, -1e4)
        source_logits = torch.cat([src_logits, noop_logit], dim=-1)

        h_s = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_t = h.unsqueeze(1).expand(-1, N, -1, -1)
        g_exp = g.unsqueeze(1).unsqueeze(1).expand(-1, N, N, -1)
        target_input = torch.cat([h_s, h_t, edge_enc, g_exp], dim=-1)
        all_target_logits = self.target_head(target_input).squeeze(-1)

        self_mask = torch.eye(N, device=h.device, dtype=torch.bool).unsqueeze(0)
        all_target_logits = all_target_logits.masked_fill(self_mask, -1e4)

        if self.mask_sun_targets:
            sun_inter, _ = compute_sun_edge_features_batch(
                positions, safety_margin=self.sun_safety_margin)
            all_target_logits = all_target_logits.masked_fill(sun_inter.bool(), -1e4)

        frac_input = torch.cat([h_s, h_t, g_exp], dim=-1)
        all_fraction_logits = self.fraction_head(frac_input)

        return source_logits, all_target_logits, all_fraction_logits


# ─── Load model ─────────────────────────────────────────────────────────
# Kaggle exec() doesn't set __file__, so resolve weights path robustly
import inspect as _inspect
_THIS_DIR = os.path.dirname(os.path.abspath(
    globals().get("__file__", _inspect.getfile(lambda: None))
))
_WEIGHTS_PATH = os.path.join(_THIS_DIR, "weights.pt")
_model = OrbitWarsGNNPolicy(hidden_dim=HIDDEN_DIM, use_gat=True, mask_sun_targets=True)
_state = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=False)
if isinstance(_state, dict) and "model_state_dict" in _state:
    _state = _state["model_state_dict"]
_model.load_state_dict(_state)
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

    nf, pos, owned = _build_node_features(planets, fleets, player, num_players, step, max_steps)

    with torch.no_grad():
        nf_b = nf.unsqueeze(0)
        pos_b = pos.unsqueeze(0)
        om_b = owned.unsqueeze(0)
        N = nf_b.shape[1]

        source_logits, target_logits, fraction_logits = _model(nf_b, pos_b, om_b)
        source_logits[0, N] -= NOOP_PENALTY

        # Multi-action: fire from every owned planet (skip noop sampling)
        actions = []
        for src_idx in range(N):
            if owned[src_idx] < 0.5:
                continue

            tgt = torch.distributions.Categorical(logits=target_logits[0, src_idx]).sample().item()
            frac = torch.distributions.Categorical(logits=fraction_logits[0, src_idx, tgt]).sample().item()

            src_planet = planets[src_idx]
            tgt_planet = planets[tgt]
            sx, sy = float(src_planet[2]), float(src_planet[3])
            tx, ty = float(tgt_planet[2]), float(tgt_planet[3])
            if sun_intersects_path(sx, sy, tx, ty):
                continue

            ships = int(float(src_planet[5]) * FRACTION_BUCKETS[frac])
            if ships < 1:
                continue

            actions.append([int(src_planet[0]), math.atan2(ty - sy, tx - sx), ships])

    _step_counter += 1
    return actions
