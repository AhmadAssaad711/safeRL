"""Vectorized relative-position HOCBF geometry.

The pairwise safety set is a fixed, axis-aligned ellipse using the
minimum-area enclosing ellipse for the 3.6 m by 1.8 m reference vehicle.
Its semi-axes are ``3.6 / sqrt(2)`` and ``1.8 / sqrt(2)``.  This module keeps
the shared batch implementation independent of notebook state so spawned
workers and direct notebook execution use the same barrier and derivatives.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


# These are CBF semi-axes, not the physical dimensions of the simulator
# bodies.  They are the minimum-area enclosing ellipse for the configured
# 3.6 m by 1.8 m reference vehicle.  The corresponding full axes are
# approximately 5.0912 m and 2.5456 m.
CBF_REFERENCE_VEHICLE_LENGTH_M = 3.6
CBF_REFERENCE_VEHICLE_WIDTH_M = 1.8
CBF_RELATIVE_ELLIPSE_A = CBF_REFERENCE_VEHICLE_LENGTH_M / np.sqrt(2.0)
CBF_RELATIVE_ELLIPSE_B = CBF_REFERENCE_VEHICLE_WIDTH_M / np.sqrt(2.0)


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


def _fixed_ellipse_values(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``h``, center distance, and the two directional radii.

    ``points`` may have any leading shape ending in ``(2,)``.  Splitting the
    directional boundary equally between the two vehicles preserves the
    existing ``l_ego + l_other`` diagnostic while making the barrier itself
    the requested relative-position ellipse.
    """

    points = np.asarray(points, dtype=float)
    dx = points[..., 0]
    dy = points[..., 1]
    h = (
        (dx / float(CBF_RELATIVE_ELLIPSE_A)) ** 2
        + (dy / float(CBF_RELATIVE_ELLIPSE_B)) ** 2
        - 1.0
    )
    radius = np.hypot(dx, dy)
    phi = np.where(radius < 1e-9, 0.0, np.arctan2(dy, dx))
    direction_denom = np.sqrt(
        (np.cos(phi) / float(CBF_RELATIVE_ELLIPSE_A)) ** 2
        + (np.sin(phi) / float(CBF_RELATIVE_ELLIPSE_B)) ** 2
    )
    boundary_radius = 1.0 / np.maximum(direction_denom, 1e-12)
    half_boundary_radius = 0.5 * boundary_radius
    return h, radius, half_boundary_radius, half_boundary_radius


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
    """Evaluate the fixed ellipse for ``(neighbor, stencil_point, xy)`` points.

    The vehicle dimensions, headings, and ``eps_side`` arguments remain in
    the signature for compatibility with existing callers and manifests, but
    they do not alter this fixed relative-position geometry.
    """

    del (
        ego_length,
        ego_width,
        ego_heading,
        other_lengths,
        other_widths,
        other_headings,
        eps_side,
    )
    return _fixed_ellipse_values(points)


def batch_centerline_barrier_derivatives(
    points: np.ndarray,
    *,
    ego: dict[str, float],
    neighbors: list[dict[str, float]],
    eps_side: float,
    fd_step: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Return the analytic fixed-ellipse geometry for many neighbors.

    ``points`` contains each neighbor's relative ``[dx, dy]`` position and
    must have shape ``(N, 2)``.  The returned arrays retain one row per input
    neighbor.  ``fd_step`` is retained for API compatibility but is not used
    because the requested quadratic barrier has exact derivatives.
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

    del ego, neighbors, eps_side, fd_step
    h0, distances, l_ego, l_other = _fixed_ellipse_values(p)
    grad = np.column_stack(
        [
            2.0 * p[:, 0] / float(CBF_RELATIVE_ELLIPSE_A) ** 2,
            2.0 * p[:, 1] / float(CBF_RELATIVE_ELLIPSE_B) ** 2,
        ]
    )
    hessian = np.zeros((count, 2, 2), dtype=float)
    hessian[:, 0, 0] = 2.0 / float(CBF_RELATIVE_ELLIPSE_A) ** 2
    hessian[:, 1, 1] = 2.0 / float(CBF_RELATIVE_ELLIPSE_B) ** 2
    return {
        "h": h0,
        "grad": grad,
        "hessian": hessian,
        "center_distance": distances,
        "l_ego": l_ego,
        "l_other": l_other,
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
