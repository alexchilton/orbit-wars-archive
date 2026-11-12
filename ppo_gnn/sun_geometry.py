"""Sun intersection geometry for Orbit Wars.

Computes whether the straight-line path between two planets intersects the
sun's danger zone, and how close the path gets. Used as both an edge feature
in the GNN and optionally as a hard mask on target selection.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

SUN_X = 50.0
SUN_Y = 50.0
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0
BOARD_DIAGONAL = math.sqrt(BOARD_SIZE**2 + BOARD_SIZE**2)
DEFAULT_SAFETY_MARGIN = 2.0


def closest_approach_to_sun(
    x1: float, y1: float, x2: float, y2: float,
    sun_x: float = SUN_X, sun_y: float = SUN_Y,
) -> float:
    """Closest distance from the line segment (x1,y1)-(x2,y2) to the sun center."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return math.sqrt((x1 - sun_x) ** 2 + (y1 - sun_y) ** 2)
    t = ((sun_x - x1) * dx + (sun_y - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.sqrt((proj_x - sun_x) ** 2 + (proj_y - sun_y) ** 2)


def sun_intersects_path(
    x1: float, y1: float, x2: float, y2: float,
    sun_radius: float = SUN_RADIUS,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> bool:
    """Check if the straight-line path between two points crosses the sun danger zone."""
    dist = closest_approach_to_sun(x1, y1, x2, y2)
    return dist < sun_radius + safety_margin


def sun_clearance_normalized(
    x1: float, y1: float, x2: float, y2: float,
) -> float:
    """Normalized clearance: 0 = through sun center, 1 = far away."""
    dist = closest_approach_to_sun(x1, y1, x2, y2)
    return min(dist / BOARD_DIAGONAL, 1.0)


def compute_sun_edge_features_batch(
    positions: torch.Tensor,
    sun_x: float = SUN_X,
    sun_y: float = SUN_Y,
    sun_radius: float = SUN_RADIUS,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute sun intersection and clearance for all planet pairs.

    Args:
        positions: (batch, num_planets, 2) — planet (x, y) coordinates.

    Returns:
        sun_intersects: (batch, num_planets, num_planets) — binary, 1 if path crosses danger zone.
        sun_clearance: (batch, num_planets, num_planets) — normalized clearance [0, 1].
    """
    B, N, _ = positions.shape
    sun = torch.tensor([sun_x, sun_y], dtype=positions.dtype, device=positions.device)

    # p1: (B, N, 1, 2), p2: (B, 1, N, 2)
    p1 = positions.unsqueeze(2)
    p2 = positions.unsqueeze(1)

    d = p2 - p1  # (B, N, N, 2)
    seg_len_sq = (d * d).sum(dim=-1, keepdim=True).clamp(min=1e-12)  # (B, N, N, 1)

    sun_offset = sun.view(1, 1, 1, 2) - p1  # (B, N, N, 2)
    t = (sun_offset * d).sum(dim=-1, keepdim=True) / seg_len_sq  # (B, N, N, 1)
    t = t.clamp(0.0, 1.0)

    proj = p1 + t * d  # (B, N, N, 2)
    dist = torch.norm(proj - sun.view(1, 1, 1, 2), dim=-1)  # (B, N, N)

    sun_intersects = (dist < sun_radius + safety_margin).float()
    sun_clearance = (dist / BOARD_DIAGONAL).clamp(0.0, 1.0)

    return sun_intersects, sun_clearance
