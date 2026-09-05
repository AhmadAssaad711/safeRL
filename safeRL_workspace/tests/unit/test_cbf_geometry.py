from __future__ import annotations

from pathlib import Path

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

