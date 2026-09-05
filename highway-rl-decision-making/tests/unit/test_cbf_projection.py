from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch as th


from scripts.common.cbf_projection import (
    CBFContextLayout,
    append_cbf_context,
    project_polytope_2d_numpy,
    project_polytope_2d_torch,
    split_cbf_context_numpy,
)


BOX_ROWS = np.asarray(
    [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
    dtype=np.float64,
)
BOX_BOUNDS = np.ones(4, dtype=np.float64)


@pytest.mark.parametrize(
    ("target", "expected", "source_code", "expected_jacobian"),
    [
        (
            [0.2, -0.3],
            [0.2, -0.3],
            0,
            [[1.0, 0.0], [0.0, 1.0]],
        ),
        (
            [2.0, 0.4],
            [1.0, 0.4],
            1,
            [[0.0, 0.0], [0.0, 1.0]],
        ),
        (
            [2.0, 2.0],
            [1.0, 1.0],
            2,
            [[0.0, 0.0], [0.0, 0.0]],
        ),
    ],
)
def test_exact_projection_and_kkt_jacobian(
    target, expected, source_code, expected_jacobian
):
    numpy_result = project_polytope_2d_numpy(
        target, BOX_ROWS, BOX_BOUNDS
    )
    assert numpy_result.feasible
    np.testing.assert_allclose(numpy_result.action, expected, atol=1e-7)

    target_tensor = th.tensor([target], dtype=th.double, requires_grad=True)
    rows_tensor = th.tensor(BOX_ROWS[None], dtype=th.double)
    bounds_tensor = th.tensor(BOX_BOUNDS[None], dtype=th.double)

    def projected(value: th.Tensor) -> th.Tensor:
        return project_polytope_2d_torch(
            value, rows_tensor, bounds_tensor
        ).action

    torch_result = project_polytope_2d_torch(
        target_tensor, rows_tensor, bounds_tensor
    )
    np.testing.assert_allclose(
        torch_result.action.detach().numpy()[0], expected, atol=1e-7
    )
    assert int(torch_result.source_code.item()) == source_code
    jacobian = th.autograd.functional.jacobian(projected, target_tensor).reshape(2, 2)
    np.testing.assert_allclose(
        jacobian.detach().numpy(), expected_jacobian, atol=1e-7
    )


def test_torch_and_numpy_projectors_match_random_feasible_polytopes():
    rng = np.random.default_rng(41)
    targets = rng.normal(0.0, 2.0, size=(64, 2))
    extra_rows = rng.normal(size=(64, 3, 2))
    # Every extra halfspace contains zero, so the set stays feasible.
    extra_bounds = rng.uniform(0.1, 1.5, size=(64, 3))
    rows = np.concatenate(
        [extra_rows, np.broadcast_to(BOX_ROWS, (64, 4, 2))], axis=1
    )
    bounds = np.concatenate(
        [extra_bounds, np.broadcast_to(BOX_BOUNDS, (64, 4))], axis=1
    )
    torch_result = project_polytope_2d_torch(
        th.tensor(targets, dtype=th.double),
        th.tensor(rows, dtype=th.double),
        th.tensor(bounds, dtype=th.double),
    )
    assert bool(torch_result.feasible.all())
    for index in range(len(targets)):
        numpy_result = project_polytope_2d_numpy(
            targets[index], rows[index], bounds[index]
        )
        assert numpy_result.feasible
        np.testing.assert_allclose(
            torch_result.action[index].detach().numpy(),
            numpy_result.action,
            atol=2e-6,
        )


def test_infeasible_set_is_explicitly_labelled():
    rows = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=float)
    bounds = np.asarray([0.0, -1.0], dtype=float)  # x <= 0 and x >= 1
    numpy_result = project_polytope_2d_numpy([0.2, 0.0], rows, bounds)
    assert not numpy_result.feasible
    assert numpy_result.fallback_used
    assert numpy_result.source.startswith("fallback:")
    assert np.all(np.isfinite(numpy_result.action))

    torch_result = project_polytope_2d_torch(
        th.tensor([[0.2, 0.0]]),
        th.tensor(rows[None], dtype=th.float32),
        th.tensor(bounds[None], dtype=th.float32),
    )
    assert not bool(torch_result.feasible.item())
    assert bool(torch_result.fallback_used.item())


def test_infeasible_numpy_and_torch_fallbacks_are_identical():
    rows = np.asarray(
        [
            [1.0, 0.0],   # x <= -0.2
            [-1.0, 0.0],  # x >= 0.8
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )
    bounds = np.asarray([-0.2, -0.8, 3.0, 3.0, 3.0, 3.0], dtype=float)
    target = np.asarray([2.0, 0.0], dtype=float)
    numpy_result = project_polytope_2d_numpy(target, rows, bounds)
    torch_result = project_polytope_2d_torch(
        th.tensor(target[None], dtype=th.double),
        th.tensor(rows[None], dtype=th.double),
        th.tensor(bounds[None], dtype=th.double),
    )
    assert numpy_result.fallback_used
    assert bool(torch_result.fallback_used.item())
    assert int(torch_result.source_code.item()) == 3
    np.testing.assert_allclose(
        torch_result.action.detach().numpy()[0],
        numpy_result.action,
        atol=1e-7,
    )
    assert float(torch_result.max_violation.item()) == pytest.approx(
        numpy_result.max_violation, abs=1e-7
    )


def test_infeasible_axis_cbf_fallback_never_escapes_the_physical_box():
    # This is the shield-only training failure: an impossible right-boundary
    # CBF face required ay <= -3.1749 while the actuator box required ay >= -3.
    # The labelled fallback must retain the physical box, not return their
    # midpoint outside it.
    rows = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )
    bounds = np.asarray([-3.17488777, 3.0, 3.0, 3.0, 3.0], dtype=float)
    low = np.asarray([-3.0, -3.0], dtype=float)
    high = np.asarray([3.0, 3.0], dtype=float)
    target = np.asarray([0.0, 0.0], dtype=float)

    numpy_result = project_polytope_2d_numpy(
        target,
        rows,
        bounds,
        action_low=low,
        action_high=high,
    )
    torch_result = project_polytope_2d_torch(
        th.tensor(target[None], dtype=th.double),
        th.tensor(rows[None], dtype=th.double),
        th.tensor(bounds[None], dtype=th.double),
        action_low=th.tensor(low, dtype=th.double),
        action_high=th.tensor(high, dtype=th.double),
    )

    assert not numpy_result.feasible
    assert numpy_result.fallback_used
    assert not bool(torch_result.feasible.item())
    assert bool(torch_result.fallback_used.item())
    assert np.all(numpy_result.action >= low)
    assert np.all(numpy_result.action <= high)
    np.testing.assert_allclose(numpy_result.action, [0.0, -3.0], atol=1e-7)
    np.testing.assert_allclose(
        torch_result.action.detach().numpy()[0], numpy_result.action, atol=1e-7
    )
    assert float(torch_result.max_violation.item()) == pytest.approx(
        numpy_result.max_violation, abs=1e-7
    )


def test_cbf_context_round_trip_preserves_state_rows_bounds_and_mask():
    layout = CBFContextLayout(base_observation_dim=42, max_constraints=18)
    state = np.arange(42, dtype=np.float32)
    augmented = append_cbf_context(
        state, BOX_ROWS, BOX_BOUNDS, layout=layout
    )
    base, rows, bounds, mask = split_cbf_context_numpy(
        augmented, layout=layout
    )
    np.testing.assert_array_equal(base, state)
    np.testing.assert_array_equal(rows[:4], BOX_ROWS.astype(np.float32))
    np.testing.assert_array_equal(bounds[:4], BOX_BOUNDS.astype(np.float32))
    np.testing.assert_array_equal(mask[:4], np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(mask[4:], np.zeros(14, dtype=np.float32))
