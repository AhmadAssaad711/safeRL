from __future__ import annotations

import numpy as np

from scripts.common.cbf_geometry import (
    CBF_RELATIVE_ELLIPSE_A,
    CBF_RELATIVE_ELLIPSE_B,
    batch_centerline_barrier_derivatives,
    batch_pairwise_hocbf_constraints,
)


def test_fixed_geometry_is_heading_and_footprint_invariant_with_analytic_derivatives():
    point = np.asarray([[1.2, -0.4]], dtype=float)
    ego = {"heading": 1.1, "length": 8.0, "width": 4.0}
    neighbor = {
        "heading": -0.9,
        "length": 1.0,
        "width": 0.5,
        "x": 1.2,
        "y": -0.4,
        "vx": 0.0,
        "vy": 0.0,
    }

    geometry = batch_centerline_barrier_derivatives(
        point, ego=ego, neighbors=[neighbor], eps_side=10.0
    )
    expected_h = (
        (point[0, 0] / CBF_RELATIVE_ELLIPSE_A) ** 2
        + (point[0, 1] / CBF_RELATIVE_ELLIPSE_B) ** 2
        - 1.0
    )
    expected_gradient = np.asarray(
        [
            2.0 * point[0, 0] / CBF_RELATIVE_ELLIPSE_A**2,
            2.0 * point[0, 1] / CBF_RELATIVE_ELLIPSE_B**2,
        ]
    )
    expected_hessian = np.diag(
        [2.0 / CBF_RELATIVE_ELLIPSE_A**2, 2.0 / CBF_RELATIVE_ELLIPSE_B**2]
    )
    np.testing.assert_allclose(geometry["h"], [expected_h])
    np.testing.assert_allclose(geometry["grad"], [expected_gradient])
    np.testing.assert_allclose(geometry["hessian"], [expected_hessian])

    changed = batch_centerline_barrier_derivatives(
        point,
        ego={"heading": -2.4, "length": 2.0, "width": 1.0},
        neighbors=[{**neighbor, "heading": 2.7, "length": 12.0, "width": 6.0}],
        eps_side=0.0,
    )
    for field in ("h", "grad", "hessian"):
        np.testing.assert_allclose(changed[field], geometry[field])


def test_pairwise_constraints_wrap_longitudinal_position_on_periodic_road():
    ego = {"x": 379.5, "y": 5.0, "vx": 16.0, "vy": 0.0}
    neighbor = {
        "x": 0.5,
        "y": 5.0,
        "vx": 16.0,
        "vy": 0.0,
        "road_length": 380.0,
    }

    batch = batch_pairwise_hocbf_constraints(
        ego, [neighbor], eps_side=0.1, k0=2.5, k1=1.5
    )

    np.testing.assert_allclose(
        batch["h"], [(1.0 / CBF_RELATIVE_ELLIPSE_A) ** 2 - 1.0]
    )
