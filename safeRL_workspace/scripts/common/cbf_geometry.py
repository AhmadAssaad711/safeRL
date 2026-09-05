"""Vectorized finite-difference HOCBF geometry.

The canonical notebook historically evaluated the centerline clearance one
neighbor at a time.  This module keeps the same nine-point finite-difference
stencil and ellipse-radius equations, but evaluates a batch of neighbors in
NumPy.  It intentionally has no dependency on notebook state so spawned
workers and direct notebook execution can share the implementation.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _wrapped_signed_dx(raw_dx: float, road_length: Optional[float]) -> float:
    """Return the shortest signed longitudinal distance on a ring road."""

    dx = float(raw_dx)
    if road_length is None:
        return dx
    length = float(road_length)
    if not np.isfinite(length) or length <= 0.0:
        return dx
    return float(((dx + 0.5 * length) % length) - 0.5 * length)


def _relative_state(
    ego: dict[str, float],
    other: dict[str, float],
) -> tuple[float, float, float, float]:
    if "signed_dx" in other:
        dx = float(other["signed_dx"])
    else:
        dx = _wrapped_signed_dx(
            float(other["x"]) - float(ego["x"]),
            other.get("road_length", ego.get("road_length")),
        )
    return (
        dx,
        float(other["y"]) - float(ego["y"]),
        float(other["vx"]) - float(ego["vx"]),
        float(other["vy"]) - float(ego["vy"]),
    )


def _inflated_axes(
    length: np.ndarray | float,
    width: np.ndarray | float,
    eps_side: float,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.maximum(
        np.asarray(length, dtype=float) / np.sqrt(2.0) + 2.0 * float(eps_side),
        1e-6,
    )
    b = np.maximum(
        np.asarray(width, dtype=float) / np.sqrt(2.0) + 2.0 * float(eps_side),
        1e-6,
    )
    return a, b


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _clearance_batch(
    points: np.ndarray,
    *,
    ego_length: float,
    ego_width: float,
    ego_heading: float,
    other_lengths: np.ndarray,
    other_widths: np.ndarray,
    other_headings: np.ndarray,
    eps_side: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate clearance for ``(neighbor, stencil_point, xy)`` points."""

    points = np.asarray(points, dtype=float)
    radius = np.linalg.norm(points, axis=-1)
    phi = np.where(
        radius < 1e-9,
        0.0,
        np.arctan2(points[..., 1], points[..., 0]),
    )
    ego_a, ego_b = _inflated_axes(ego_length, ego_width, eps_side)
    other_a, other_b = _inflated_axes(
        other_lengths[:, None], other_widths[:, None], eps_side
    )

    ego_delta = _wrap_angle(phi - float(ego_heading))
    other_delta = _wrap_angle(phi - other_headings[:, None])
    ego_cos = np.cos(ego_delta)
    ego_sin = np.sin(ego_delta)
    other_cos = np.cos(other_delta)
    other_sin = np.sin(other_delta)
    ego_denom = np.sqrt((ego_b * ego_cos) ** 2 + (ego_a * ego_sin) ** 2)
    other_denom = np.sqrt(
        (other_b * other_cos) ** 2 + (other_a * other_sin) ** 2
    )
    ego_radius = ego_a * ego_b / np.maximum(ego_denom, 1e-9)
    other_radius = other_a * other_b / np.maximum(other_denom, 1e-9)
    required_distance = ego_radius + other_radius
    return (
        radius - required_distance,
        radius,
        ego_radius,
        other_radius,
    )


def batch_centerline_barrier_derivatives(
    points: np.ndarray,
    *,
    ego: dict[str, float],
    neighbors: list[dict[str, float]],
    eps_side: float,
    fd_step: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Return the notebook's finite-difference geometry for many neighbors.

    ``points`` contains each neighbor's relative ``[dx, dy]`` position and
    must have shape ``(N, 2)``.  The returned arrays retain one row per input
    neighbor and use the same central differences as
    ``centerline_barrier_derivatives``.
    """

    p = np.asarray(points, dtype=float).reshape(-1, 2)
    count = int(p.shape[0])
    if count != len(neighbors):
        raise ValueError(
            "points and neighbors must contain the same number of rows"
        )
    if count == 0:
        return {
            "h": np.empty(0, dtype=float),
            "grad": np.empty((0, 2), dtype=float),
            "hessian": np.empty((0, 2, 2), dtype=float),
            "center_distance": np.empty(0, dtype=float),
            "l_ego": np.empty(0, dtype=float),
            "l_other": np.empty(0, dtype=float),
        }

    other_lengths = np.asarray(
        [float(item["length"]) for item in neighbors], dtype=float
    )
    other_widths = np.asarray(
        [float(item["width"]) for item in neighbors], dtype=float
    )
    other_headings = np.asarray(
        [float(item.get("heading", 0.0)) for item in neighbors], dtype=float
    )
    step = float(fd_step)
    offsets = np.asarray(
        [
            [0.0, 0.0],
            [step, 0.0],
            [-step, 0.0],
            [0.0, step],
            [0.0, -step],
            [step, step],
            [step, -step],
            [-step, step],
            [-step, -step],
        ],
        dtype=float,
    )
    stencil_points = p[:, None, :] + offsets[None, :, :]
    h_values, distances, ego_radii, other_radii = _clearance_batch(
        stencil_points,
        ego_length=float(ego["length"]),
        ego_width=float(ego["width"]),
        ego_heading=float(ego.get("heading", 0.0)),
        other_lengths=other_lengths,
        other_widths=other_widths,
        other_headings=other_headings,
        eps_side=float(eps_side),
    )
    h0 = h_values[:, 0]
    h_px = h_values[:, 1]
    h_mx = h_values[:, 2]
    h_py = h_values[:, 3]
    h_my = h_values[:, 4]
    h_pp = h_values[:, 5]
    h_pm = h_values[:, 6]
    h_mp = h_values[:, 7]
    h_mm = h_values[:, 8]
    grad = np.column_stack(
        [
            (h_px - h_mx) / (2.0 * step),
            (h_py - h_my) / (2.0 * step),
        ]
    )
    mixed = (h_pp - h_pm - h_mp + h_mm) / (4.0 * step**2)
    hessian = np.stack(
        [
            np.column_stack(
                [
                    (h_px - 2.0 * h0 + h_mx) / (step**2),
                    mixed,
                ]
            ),
            np.column_stack(
                [
                    mixed,
                    (h_py - 2.0 * h0 + h_my) / (step**2),
                ]
            ),
        ],
        axis=1,
    )
    return {
        "h": h0,
        "grad": grad,
        "hessian": hessian,
        "center_distance": distances[:, 0],
        "l_ego": ego_radii[:, 0],
        "l_other": other_radii[:, 0],
    }


def batch_pairwise_hocbf_constraints(
    ego: dict[str, float],
    neighbors: list[dict[str, float]],
    *,
    eps_side: float,
    k0: float,
    k1: float,
) -> dict[str, np.ndarray]:
    """Build HOCBF rows and diagnostics for a neighbor batch."""

    count = len(neighbors)
    if count == 0:
        return {
            "A": np.empty((0, 2), dtype=float),
            "b": np.empty(0, dtype=float),
            "h": np.empty(0, dtype=float),
            "center_distance": np.empty(0, dtype=float),
            "required_distance": np.empty(0, dtype=float),
            "h_dot": np.empty(0, dtype=float),
            "hddot_without_ego": np.empty(0, dtype=float),
        }

    relative = np.asarray(
        [_relative_state(ego, item) for item in neighbors], dtype=float
    )
    geometry = batch_centerline_barrier_derivatives(
        relative[:, :2],
        ego=ego,
        neighbors=neighbors,
        eps_side=float(eps_side),
    )
    velocities = relative[:, 2:4]
    other_accelerations = np.asarray(
        [
            [float(item.get("ax", 0.0)), float(item.get("ay", 0.0))]
            for item in neighbors
        ],
        dtype=float,
    )
    h_dot = np.einsum("ni,ni->n", geometry["grad"], velocities)
    hddot_without_ego = (
        np.einsum(
            "ni,nij,nj->n",
            velocities,
            geometry["hessian"],
            velocities,
        )
        + np.einsum(
            "ni,ni->n", geometry["grad"], other_accelerations
        )
    )
    b = (
        hddot_without_ego
        + float(k1) * h_dot
        + float(k0) * geometry["h"]
    )
    return {
        "A": np.asarray(geometry["grad"], dtype=float),
        "b": np.asarray(b, dtype=float),
        "h": np.asarray(geometry["h"], dtype=float),
        "center_distance": np.asarray(geometry["center_distance"], dtype=float),
        "required_distance": np.asarray(
            geometry["l_ego"] + geometry["l_other"], dtype=float
        ),
        "h_dot": np.asarray(h_dot, dtype=float),
        "hddot_without_ego": np.asarray(hddot_without_ego, dtype=float),
    }

