from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.training.run_cbf_filter_ablation import (
    bootstrap_notebook_namespace,
    exec_required_notebook_cells,
)


def _notebook_namespace() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    namespace = bootstrap_notebook_namespace(root)
    exec_required_notebook_cells(
        root / "notebooks" / "lanelessKaralakou.ipynb",
        namespace,
    )
    return namespace


def test_batched_hocbf_geometry_matches_scalar_notebook_reference():
    namespace = _notebook_namespace()
    rng = np.random.default_rng(173)
    ego = {
        "x": 120.0,
        "y": 5.1,
        "vx": 16.0,
        "vy": 0.2,
        "heading": 0.01,
        "length": 3.5,
        "width": 1.8,
        "road_length": 380.0,
    }
    neighbors = []
    for _ in range(12):
        signed_dx = float(rng.uniform(-90.0, 90.0))
        neighbors.append(
            {
                "x": ego["x"] + signed_dx,
                "y": ego["y"] + float(rng.uniform(-4.5, 4.5)),
                "vx": float(rng.uniform(10.0, 25.0)),
                "vy": float(rng.uniform(-2.0, 2.0)),
                "heading": float(rng.uniform(-0.2, 0.2)),
                "ax": float(rng.uniform(-3.0, 3.0)),
                "ay": float(rng.uniform(-3.0, 3.0)),
                "length": 3.5,
                "width": 1.8,
                "signed_dx": signed_dx,
            }
        )

    batch = namespace["batch_pairwise_hocbf_constraints"](
        ego,
        neighbors,
        eps_side=0.1,
        k0=5.29,
        k1=3.68,
    )
    for index, neighbor in enumerate(neighbors):
        other_acc = np.asarray([neighbor["ax"], neighbor["ay"]], dtype=float)
        scalar = namespace["pairwise_hocbf_constraint"](
            ego,
            neighbor,
            eps_side=0.1,
            k0=5.29,
            k1=3.68,
            other_acc=other_acc,
        )
        dx, dy, dvx, dvy = namespace["pairwise_relative_state"](
            ego, neighbor
        )
        h_value, grad, hessian, _, _, _ = namespace[
            "centerline_barrier_derivatives"
        ](
            np.asarray([dx, dy], dtype=float),
            ego,
            neighbor,
            0.1,
        )
        velocity = np.asarray([dvx, dvy], dtype=float)
        h_dot = float(np.asarray(grad, dtype=float) @ velocity)
        hddot_without_ego = float(
            velocity.T @ hessian @ velocity
            + np.asarray(grad, dtype=float) @ other_acc
        )
        expected = {
            "A": scalar[0],
            "b": scalar[1],
            "h": scalar[2],
            "center_distance": scalar[3],
            "required_distance": scalar[4],
            "h_dot": h_dot,
            "hddot_without_ego": hddot_without_ego,
        }
        for name, value in expected.items():
            np.testing.assert_allclose(
                np.asarray(batch[name][index]),
                np.asarray(value),
                rtol=0.0,
                atol=1e-12,
                err_msg=f"batch field {name} differs for neighbor {index}",
            )


def test_fixed_relative_position_ellipse_axes_and_derivatives():
    namespace = _notebook_namespace()
    # The relative barrier's semi-axes are the Minkowski-sum (pairwise) extent
    # of both vehicles' individual minimum-area-enclosing ellipses, i.e. twice
    # the single-vehicle 3.6 x 1.8 m reference values.
    ellipse_a = 2.0 * 3.6 / np.sqrt(2.0)
    ellipse_b = 2.0 * 1.8 / np.sqrt(2.0)
    ego = {
        "x": 0.0,
        "y": 0.0,
        "vx": 16.0,
        "vy": 0.0,
        "heading": 0.4,
        "length": 3.5,
        "width": 1.8,
    }
    other = {
        "x": float(ellipse_a / 2.0),
        "y": float(ellipse_b / 2.0),
        "vx": 15.0,
        "vy": 0.2,
        "heading": -0.3,
        "ax": 0.1,
        "ay": -0.2,
        "length": 8.0,
        "width": 4.0,
        "signed_dx": float(ellipse_a / 2.0),
    }

    clearance = namespace["pairwise_centerline_clearance"]
    h_longitudinal, _, l_ego, l_other = clearance(
        np.asarray([ellipse_a, 0.0]), ego, other, eps_side=0.0
    )
    h_lateral, _, _, _ = clearance(
        np.asarray([0.0, ellipse_b]), ego, other, eps_side=100.0
    )
    h_center, _, _, _ = clearance(
        np.asarray([0.0, 0.0]), ego, other, eps_side=0.1
    )
    np.testing.assert_allclose(h_longitudinal, 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(h_lateral, 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(h_center, -1.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        l_ego + l_other, ellipse_a, rtol=0.0, atol=1e-12
    )

    point = np.asarray([ellipse_a / 2.0, ellipse_b / 2.0], dtype=float)
    h, gradient, hessian, _, _, _ = namespace[
        "centerline_barrier_derivatives"
    ](point, ego, other, eps_side=0.1, fd_step=0.25)
    expected_h = (point[0] / ellipse_a) ** 2 + (point[1] / ellipse_b) ** 2 - 1.0
    expected_gradient = np.asarray(
        [2.0 * point[0] / ellipse_a**2, 2.0 * point[1] / ellipse_b**2]
    )
    expected_hessian = np.diag([2.0 / ellipse_a**2, 2.0 / ellipse_b**2])
    np.testing.assert_allclose(h, expected_h, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=0.0, atol=1e-12)

    batch = namespace["batch_pairwise_hocbf_constraints"](
        ego,
        [other],
        eps_side=0.1,
        k0=5.29,
        k1=4.6,
    )
    np.testing.assert_allclose(batch["h"][0], expected_h, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        batch["A"][0], expected_gradient, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        batch["required_distance"][0],
        namespace["ellipse_radius_along_line"](
            ellipse_a, ellipse_b, np.arctan2(ellipse_b / 2.0, ellipse_a / 2.0)
        ),
        rtol=0.0,
        atol=1e-12,
    )


def test_notebook_kpi_geometry_matches_the_fixed_pairwise_barrier():
    namespace = _notebook_namespace()
    geometry_metadata = namespace["ENV_CONFIG"]["cbf_geometry"]
    # The relative barrier's semi-axes are the pairwise (Minkowski-sum)
    # extent, i.e. twice the single-vehicle 3.6 x 1.8 m reference values --
    # equal to full_major_axis_m/full_minor_axis_m, not the single-vehicle
    # ellipse alone.
    ellipse_a = 2.0 * 3.6 / np.sqrt(2.0)
    ellipse_b = 2.0 * 1.8 / np.sqrt(2.0)
    assert geometry_metadata == {
        "model": "minimum_area_enclosing_vehicle_ellipse",
        "reference_vehicle_length_m": 3.6,
        "reference_vehicle_width_m": 1.8,
        "relative_ellipse_a_m": ellipse_a,
        "relative_ellipse_b_m": ellipse_b,
        "full_major_axis_m": ellipse_a,
        "full_minor_axis_m": ellipse_b,
    }

    ego = SimpleNamespace(
        position=np.asarray([379.5, 5.0]), length=9.0, width=4.0, heading=1.2
    )
    neighbor = SimpleNamespace(
        position=np.asarray([0.5, 5.5]), length=1.0, width=0.5, heading=-2.3
    )
    base = SimpleNamespace(
        vehicle=ego,
        road=SimpleNamespace(vehicles=[ego, neighbor]),
        config={"sensing_range": 90.0, "road_width": 10.2},
        _signed_distance=lambda start, end: ((end - start + 190.0) % 380.0) - 190.0,
    )
    metrics = namespace["kpi_neighbor_and_h_metrics"](
        SimpleNamespace(unwrapped=base), eps_side=100.0
    )

    dx, dy = 1.0, 0.5
    expected_h = (dx / ellipse_a) ** 2 + (dy / ellipse_b) ** 2 - 1.0
    expected_radius = 1.0 / np.sqrt(
        (np.cos(np.arctan2(dy, dx)) / ellipse_a) ** 2
        + (np.sin(np.arctan2(dy, dx)) / ellipse_b) ** 2
    )
    np.testing.assert_allclose(metrics["kpi_pairwise_h_min"], expected_h)
    np.testing.assert_allclose(metrics["kpi_h_min"], expected_h)
    np.testing.assert_allclose(metrics["kpi_min_required_distance_m"], expected_radius)
