"""Parse Kaggle Orbit Wars replay JSON files into training datasets.

Each replay contains 500 steps x N players, with full observations (planets,
fleets) and actions [[source_id, angle_radians, num_ships], ...].

This module converts those into (graph_obs, factored_action, discounted_return)
tuples suitable for behavioral cloning and offline RL.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .gnn_policy import FRACTION_BUCKETS, NODE_DIM

BOARD_SIZE = 100.0
SUN_X, SUN_Y = 50.0, 50.0
DEFAULT_SHIP_SPEED = 6.0


@dataclass
class Transition:
    node_features: torch.Tensor   # (N, 10)
    positions: torch.Tensor       # (N, 2)
    owned_mask: torch.Tensor      # (N,)
    source_idx: int               # planet index or -1 for noop
    target_idx: int               # planet index or -1 for noop
    fraction_idx: int             # 0-3 or -1 for noop
    is_noop: bool
    discounted_return: float
    player_rank: int              # 1-based rank (1 = winner)
    mode: str                     # "2p" or "4p"


def _angle_to_target(
    source_x: float, source_y: float, angle: float,
    planets: List[List], source_id: int,
    cone_tolerance: float = 0.3,
) -> Optional[int]:
    """Convert an angle-based action to the target planet index.

    Finds the planet closest to the ray from the source at the given angle.
    Uses perpendicular distance with a forward-only constraint.
    """
    best_idx = None
    best_score = float("inf")

    for i, p in enumerate(planets):
        pid = int(p[0])
        if pid == source_id:
            continue
        px, py = float(p[2]), float(p[3])
        dx = px - source_x
        dy = py - source_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            continue

        # Angle from source to this planet
        planet_angle = math.atan2(dy, dx)
        # Angular difference (wrapped to [-pi, pi])
        diff = (planet_angle - angle + math.pi) % (2 * math.pi) - math.pi
        if abs(diff) > cone_tolerance:
            continue

        # Score: distance * angular deviation (prefer close + on-target)
        score = dist * (1.0 + abs(diff))
        if score < best_score:
            best_score = score
            best_idx = i

    return best_idx


def _fraction_to_bucket(ships_sent: float, ships_available: float) -> int:
    """Snap ship fraction to nearest bucket index."""
    if ships_available < 1:
        return 0
    frac = ships_sent / ships_available
    best_idx = 0
    best_diff = abs(frac - FRACTION_BUCKETS[0])
    for i, b in enumerate(FRACTION_BUCKETS[1:], 1):
        d = abs(frac - b)
        if d < best_diff:
            best_diff = d
            best_idx = i
    return best_idx


def _build_node_features(
    planets: List[List], fleets: List[List], player_id: int, num_players: int,
    step: int = 0, max_steps: int = 400,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build node features, positions, and owned_mask from observation data.

    Returns:
        node_features: (N, NODE_DIM)
        positions: (N, 2)
        owned_mask: (N,) — 1.0 for owned planets with ships > 0
    """
    N = len(planets)
    nf = torch.zeros(N, NODE_DIM)
    pos = torch.zeros(N, 2)
    owned = torch.zeros(N)

    # Pre-compute inbound fleets per planet (ships + count)
    inbound_friendly = [0.0] * N
    inbound_enemy = [0.0] * N
    inbound_friendly_count = [0] * N
    inbound_enemy_count = [0] * N
    # Build planet_id -> index mapping
    pid_to_idx = {}
    for i, p in enumerate(planets):
        pid_to_idx[int(p[0])] = i

    # Global fleet totals for in-transit features
    my_fleet_ships = 0.0
    enemy_fleet_ships = 0.0

    for fleet in fleets:
        # fleet: [id, owner, x, y, angle, from_planet_id, ships]
        f_owner = int(fleet[1])
        f_ships = float(fleet[6])
        fx, fy  = float(fleet[2]), float(fleet[3])
        f_angle = float(fleet[4])
        cos_a   = math.cos(f_angle)
        sin_a   = math.sin(f_angle)

        if f_owner == player_id:
            my_fleet_ships += f_ships
        else:
            enemy_fleet_ships += f_ships

        best_perp = float("inf")
        best_planet_idx = None
        for i, p in enumerate(planets):
            px, py = float(p[2]), float(p[3])
            dx, dy = px - fx, py - fy
            dot  = dx * cos_a + dy * sin_a
            if dot <= 0:
                continue
            perp = abs(dx * sin_a - dy * cos_a)
            if perp < best_perp:
                best_perp = perp
                best_planet_idx = i
        if best_planet_idx is not None and best_perp < 12.0:
            if f_owner == player_id:
                inbound_friendly[best_planet_idx] += f_ships
                inbound_friendly_count[best_planet_idx] += 1
            else:
                inbound_enemy[best_planet_idx] += f_ships
                inbound_enemy_count[best_planet_idx] += 1

    # ── Pre-compute global aggregates ─────────────────────────────────────
    my_total_ships = 0.0
    enemy_total_ships = 0.0
    my_total_prod = 0.0
    enemy_total_prod = 0.0
    my_planet_count = 0
    enemy_planet_count = 0
    per_enemy_ships: Dict[int, float] = {}  # owner -> total ships

    for p in planets:
        p_owner = int(p[1])
        p_ships = float(p[5])
        p_prod = float(p[6])
        if p_owner == player_id:
            my_total_ships += p_ships
            my_total_prod += p_prod
            my_planet_count += 1
        elif p_owner >= 0:
            enemy_total_ships += p_ships
            enemy_total_prod += p_prod
            enemy_planet_count += 1
            per_enemy_ships[p_owner] = per_enemy_ships.get(p_owner, 0) + p_ships

    # Include in-transit ships
    my_total_ships += my_fleet_ships
    enemy_total_ships += enemy_fleet_ships
    total_ships = my_total_ships + enemy_total_ships + 1e-6
    strongest_enemy_ships = max(per_enemy_ships.values()) if per_enemy_ships else 0.0
    total_planets_owned = my_planet_count + enemy_planet_count
    remaining_frac = max(0.0, (max_steps - step)) / max_steps
    step_frac = step / max_steps

    # Pre-compute planet positions for neighbor lookups
    planet_positions = [(float(p[2]), float(p[3])) for p in planets]
    planet_owners = [int(p[1]) for p in planets]
    planet_ships = [float(p[5]) for p in planets]

    NEIGHBOR_RADIUS = 30.0

    for i, p in enumerate(planets):
        px, py = planet_positions[i]
        p_owner = planet_owners[i]
        p_ships = planet_ships[i]
        p_prod = float(p[6])
        p_orbit_vel = float(p[4])

        pos[i, 0] = px / BOARD_SIZE
        pos[i, 1] = py / BOARD_SIZE

        is_mine = 1.0 if p_owner == player_id else 0.0
        is_enemy = 1.0 if p_owner >= 0 and p_owner != player_id else 0.0
        is_neutral = 1.0 if p_owner < 0 else 0.0

        if is_mine > 0 and p_ships > 0:
            owned[i] = 1.0

        # ── Original features [0-15] ──────────────────────────────────────
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
        nf[i, 10] = step_frac
        nf[i, 11] = p_prod * remaining_frac * 2.0
        nf[i, 12] = min(max((my_total_prod - enemy_total_prod) / 20.0, -1.0), 1.0)

        # [13] Distance to nearest enemy planet
        min_enemy_dist = 100.0
        for j in range(N):
            if j == i:
                continue
            if planet_owners[j] >= 0 and planet_owners[j] != player_id:
                dx = px - planet_positions[j][0]
                dy = py - planet_positions[j][1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < min_enemy_dist:
                    min_enemy_dist = d
        nf[i, 13] = min_enemy_dist / 100.0

        # [14] Nearest enemy fleet ETA
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

        # [15] Ship surplus ratio
        if is_mine > 0:
            surplus = float(nf[i, 5]) * 6.5
            threat = float(nf[i, 9]) * 6.5
            nf[i, 15] = min(max((surplus - threat) / 6.5, -1.0), 1.0)
        elif is_enemy > 0:
            nf[i, 15] = -0.5
        else:
            nf[i, 15] = 0.0

        # ── New features [16-27] ──────────────────────────────────────────

        # [16] Planet orbit angular velocity (planets move around the sun!)
        nf[i, 16] = p_orbit_vel / 4.0  # typical range ~0-4 rad

        # [17] Num players indicator (2p=0.5, 3p=0.75, 4p=1.0)
        nf[i, 17] = num_players / 4.0

        # [18] My planet fraction (what share of owned planets are mine)
        nf[i, 18] = my_planet_count / max(1, total_planets_owned)

        # [19] My total military share (garrison + in-transit ships)
        nf[i, 19] = my_total_ships / total_ships

        # [20] Strongest enemy's military share
        nf[i, 20] = strongest_enemy_ships / total_ships

        # [21] Friendly neighbors (my planets within radius)
        friendly_neighbors = 0
        enemy_neighbors = 0
        friendly_ships_nearby = 0.0
        enemy_ships_nearby = 0.0
        for j in range(N):
            if j == i:
                continue
            dx = px - planet_positions[j][0]
            dy = py - planet_positions[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            if d <= NEIGHBOR_RADIUS:
                if planet_owners[j] == player_id:
                    friendly_neighbors += 1
                    friendly_ships_nearby += planet_ships[j]
                elif planet_owners[j] >= 0:
                    enemy_neighbors += 1
                    enemy_ships_nearby += planet_ships[j]
        nf[i, 21] = min(friendly_neighbors / 5.0, 1.0)

        # [22] Enemy neighbors
        nf[i, 22] = min(enemy_neighbors / 5.0, 1.0)

        # [23] Local force balance (my ships vs enemy ships nearby)
        local_total = friendly_ships_nearby + enemy_ships_nearby + 1e-6
        nf[i, 23] = (friendly_ships_nearby - enemy_ships_nearby) / max(local_total, 1.0)
        nf[i, 23] = min(max(nf[i, 23].item(), -1.0), 1.0)

        # [24] Is frontline (nearest enemy closer than nearest friendly)
        min_friendly_dist = 100.0
        for j in range(N):
            if j == i or planet_owners[j] != player_id:
                continue
            dx = px - planet_positions[j][0]
            dy = py - planet_positions[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < min_friendly_dist:
                min_friendly_dist = d
        nf[i, 24] = 1.0 if min_enemy_dist < min_friendly_dist else 0.0

        # [25] Inbound fleet count (friendly) — number of separate fleets, not ships
        nf[i, 25] = min(inbound_friendly_count[i] / 5.0, 1.0)

        # [26] Inbound fleet count (enemy)
        nf[i, 26] = min(inbound_enemy_count[i] / 5.0, 1.0)

        # [27] My in-transit ship ratio (how much of my force is airborne)
        nf[i, 27] = my_fleet_ships / max(my_total_ships, 1.0)

    return nf, pos, owned


def parse_replay(filepath: str, gamma: float = 0.999) -> List[Transition]:
    """Parse a single replay file into transitions.

    Args:
        filepath: Path to the replay JSON.
        gamma: Discount factor for computing returns.

    Returns:
        List of Transition objects, one per (step, player) with valid observation.
    """
    with open(filepath) as f:
        data = json.load(f)

    if "steps" not in data:
        return []

    steps = data["steps"]
    num_players = len(steps[0])
    mode = "2p" if num_players == 2 else "4p"
    rewards = data.get("rewards", [0] * num_players)

    # Compute player ranks from rewards
    sorted_rewards = sorted(enumerate(rewards), key=lambda x: -(x[1] or -999))
    rank_map = {}
    for rank_idx, (pid, _) in enumerate(sorted_rewards):
        rank_map[pid] = rank_idx + 1

    # Collect raw transitions per player
    player_transitions: Dict[int, List[dict]] = {p: [] for p in range(num_players)}

    for step_idx, step in enumerate(steps):
        for player_id in range(num_players):
            agent_data = step[player_id]
            obs = agent_data.get("observation")
            action = agent_data.get("action", [])

            if obs is None or "planets" not in obs:
                continue

            planets = obs["planets"]
            fleets = obs.get("fleets", [])

            if not planets:
                continue

            # Check if player is still alive (owns at least one planet)
            has_planets = any(int(p[1]) == player_id for p in planets)
            if not has_planets:
                continue

            node_features, positions, owned_mask = _build_node_features(
                planets, fleets, player_id, num_players,
                step=step_idx, max_steps=len(steps),
            )

            # Build planet_id -> index mapping for this step
            pid_to_idx = {int(p[0]): i for i, p in enumerate(planets)}

            # Convert actions — emit one transition per sub-action
            if not action:
                player_transitions[player_id].append({
                    "nf": node_features, "pos": positions, "om": owned_mask,
                    "source": -1, "target": -1, "fraction": -1, "is_noop": True,
                    "step": step_idx,
                })
            else:
                used_sources = set()
                for act in action:
                    if not isinstance(act, (list, tuple)) or len(act) < 3:
                        continue
                    src_planet_id = int(act[0])
                    angle = float(act[1])
                    ships_sent = float(act[2])

                    if src_planet_id not in pid_to_idx:
                        continue

                    src_idx = pid_to_idx[src_planet_id]
                    if src_idx in used_sources:
                        continue  # skip duplicate sources within same step
                    used_sources.add(src_idx)

                    src_planet = planets[src_idx]
                    src_x, src_y = float(src_planet[2]), float(src_planet[3])
                    src_ships = float(src_planet[5])

                    tgt_idx = _angle_to_target(src_x, src_y, angle, planets, src_planet_id)
                    if tgt_idx is None:
                        continue

                    frac_idx = _fraction_to_bucket(ships_sent, src_ships)

                    player_transitions[player_id].append({
                        "nf": node_features, "pos": positions, "om": owned_mask,
                        "source": src_idx, "target": tgt_idx, "fraction": frac_idx,
                        "is_noop": False, "step": step_idx,
                    })

                # After all sub-actions, add a noop transition (teaches model when to stop)
                if used_sources:
                    player_transitions[player_id].append({
                        "nf": node_features, "pos": positions, "om": owned_mask,
                        "source": -1, "target": -1, "fraction": -1, "is_noop": True,
                        "step": step_idx,
                    })

    # Compute discounted returns
    all_transitions = []
    for player_id, trans_list in player_transitions.items():
        if not trans_list:
            continue
        final_reward = rewards[player_id] if rewards[player_id] is not None else 0.0
        rank = rank_map.get(player_id, num_players)

        # Assign discounted return: final_reward * gamma^(T - t)
        T = len(steps) - 1
        for t_data in trans_list:
            steps_remaining = T - t_data["step"]
            disc_return = final_reward * (gamma ** steps_remaining)

            all_transitions.append(Transition(
                node_features=t_data["nf"],
                positions=t_data["pos"],
                owned_mask=t_data["om"],
                source_idx=t_data["source"],
                target_idx=t_data["target"],
                fraction_idx=t_data["fraction"],
                is_noop=t_data["is_noop"],
                discounted_return=disc_return,
                player_rank=rank,
                mode=mode,
            ))

    return all_transitions


def parse_all_replays(
    replay_dir: str,
    gamma: float = 0.999,
    verbose: bool = True,
) -> Tuple[List[Transition], List[Transition]]:
    """Parse all replay files, split into 2P and 4P datasets.

    Returns:
        (transitions_2p, transitions_4p)
    """
    trans_2p: List[Transition] = []
    trans_4p: List[Transition] = []
    total_files = 0
    errors = 0

    for root, _dirs, files in os.walk(replay_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            total_files += 1
            try:
                transitions = parse_replay(os.path.join(root, fname), gamma)
                for t in transitions:
                    if t.mode == "2p":
                        trans_2p.append(t)
                    else:
                        trans_4p.append(t)
            except Exception as e:
                errors += 1
                if verbose:
                    print(f"  Error parsing {fname}: {e}")

    if verbose:
        print(f"Parsed {total_files} files ({errors} errors)")
        print(f"  2P transitions: {len(trans_2p)}")
        print(f"  4P transitions: {len(trans_4p)}")

    return trans_2p, trans_4p


class ReplayDataset(Dataset):
    """PyTorch Dataset wrapping a list of Transitions.

    Pads/truncates to max_planets for batching. Planets beyond the actual
    count are zero-padded with owned_mask = 0.
    """

    def __init__(
        self,
        transitions: List[Transition],
        max_planets: int = 48,
        winners_only: bool = False,
    ):
        if winners_only:
            transitions = [t for t in transitions if t.player_rank == 1]
        self.transitions = transitions
        self.max_planets = max_planets

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.transitions[idx]
        N = t.node_features.shape[0]
        M = self.max_planets

        # Pad to max_planets
        nf = torch.zeros(M, NODE_DIM)
        pos = torch.zeros(M, 2)
        owned = torch.zeros(M)
        valid = torch.zeros(M)

        n = min(N, M)
        nf[:n] = t.node_features[:n]
        pos[:n] = t.positions[:n]
        owned[:n] = t.owned_mask[:n]
        valid[:n] = 1.0

        source = t.source_idx if not t.is_noop else M  # noop = M (the N+1 slot after padding)
        target = t.target_idx if not t.is_noop else 0
        fraction = t.fraction_idx if not t.is_noop else 0

        return {
            "node_features": nf,
            "positions": pos,
            "owned_mask": owned,
            "valid_mask": valid,
            "source": torch.tensor(source, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
            "fraction": torch.tensor(fraction, dtype=torch.long),
            "is_noop": torch.tensor(1.0 if t.is_noop else 0.0),
            "discounted_return": torch.tensor(t.discounted_return, dtype=torch.float32),
            "player_rank": torch.tensor(t.player_rank, dtype=torch.long),
            "num_planets": torch.tensor(n, dtype=torch.long),
        }


def save_dataset(transitions: List[Transition], path: str) -> None:
    """Save transitions to disk as a .pt file."""
    torch.save(transitions, path)
    print(f"Saved {len(transitions)} transitions to {path}")


def load_dataset(path: str) -> List[Transition]:
    """Load transitions from a .pt file."""
    return torch.load(path, weights_only=False)


def prebatch_and_save(
    transitions: List[Transition],
    path: str,
    max_planets: int = 40,
    winners_only: bool = False,
) -> None:
    """Convert transitions to pre-padded tensors and save as a single dict.

    Much faster to load than a list of Transition objects.
    """
    if winners_only:
        transitions = [t for t in transitions if t.player_rank == 1]

    N = len(transitions)
    M = max_planets

    nf = torch.zeros(N, M, NODE_DIM)
    pos = torch.zeros(N, M, 2)
    owned = torch.zeros(N, M)
    sources = torch.zeros(N, dtype=torch.long)
    targets = torch.zeros(N, dtype=torch.long)
    fractions = torch.zeros(N, dtype=torch.long)
    is_noops = torch.zeros(N)
    returns = torch.zeros(N)
    ranks = torch.zeros(N, dtype=torch.long)
    num_planets = torch.zeros(N, dtype=torch.long)

    # Filter out samples where action references invalid planets
    valid_transitions = []
    skipped_oob = 0
    skipped_unowned = 0
    for t in transitions:
        if not t.is_noop:
            if t.source_idx >= M or t.target_idx >= M:
                skipped_oob += 1
                continue  # action references a planet we'd truncate
            # Skip if source planet isn't owned by this player in the observation
            if t.owned_mask[t.source_idx].item() == 0:
                skipped_unowned += 1
                continue
        valid_transitions.append(t)

    if skipped_oob > 0:
        print(f"  Skipped {skipped_oob} transitions with actions beyond max_planets={M}")
    if skipped_unowned > 0:
        print(f"  Skipped {skipped_unowned} transitions with unowned source planet")
    transitions = valid_transitions
    N = len(transitions)

    # Re-allocate with correct size
    nf = torch.zeros(N, M, NODE_DIM)
    pos = torch.zeros(N, M, 2)
    owned = torch.zeros(N, M)
    sources = torch.zeros(N, dtype=torch.long)
    targets = torch.zeros(N, dtype=torch.long)
    fractions = torch.zeros(N, dtype=torch.long)
    is_noops = torch.zeros(N)
    returns = torch.zeros(N)
    ranks = torch.zeros(N, dtype=torch.long)
    num_planets = torch.zeros(N, dtype=torch.long)

    for i, t in enumerate(transitions):
        n = min(t.node_features.shape[0], M)
        nf[i, :n] = t.node_features[:n]
        pos[i, :n] = t.positions[:n]
        owned[i, :n] = t.owned_mask[:n]
        sources[i] = t.source_idx if not t.is_noop else M
        targets[i] = t.target_idx if not t.is_noop else 0
        fractions[i] = t.fraction_idx if not t.is_noop else 0
        is_noops[i] = 1.0 if t.is_noop else 0.0
        returns[i] = t.discounted_return
        ranks[i] = t.player_rank
        num_planets[i] = n

    data = {
        "node_features": nf, "positions": pos, "owned_mask": owned,
        "source": sources, "target": targets, "fraction": fractions,
        "is_noop": is_noops, "discounted_return": returns,
        "player_rank": ranks, "num_planets": num_planets,
    }
    torch.save(data, path)
    print(f"Saved {N} pre-batched transitions to {path}")


class PreBatchedDataset(Dataset):
    """Fast dataset from pre-tensorized data."""

    def __init__(self, path: str, max_samples: int = 0):
        self.data = torch.load(path, weights_only=True)
        self._len = self.data["source"].shape[0]
        if max_samples > 0 and max_samples < self._len:
            indices = torch.randperm(self._len)[:max_samples]
            self.data = {k: v[indices] for k, v in self.data.items()}
            self._len = max_samples

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in self.data.items():
            t = v[idx]
            if t.dtype == torch.float16:
                t = t.float()
            out[k] = t
        return out
