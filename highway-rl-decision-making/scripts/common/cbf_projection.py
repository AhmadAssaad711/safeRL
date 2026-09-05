"""Exact two-dimensional CBF projection utilities.

This module is deliberately algorithm independent.  It provides one compact
representation of the state-dependent linear action set

    A(s) a <= b(s)

and matching NumPy (hard execution) and PyTorch (differentiable policy mean)
projectors.  In two dimensions the Euclidean projection is either the target,
a projection onto one face, or an intersection of two faces, so an exact
active-set enumeration is both small and dependency free.

The no-slack CBF set can be empty.  Both implementations therefore return an
explicit feasibility flag.  The NumPy path uses a labelled least-violating
fallback; callers must not describe that fallback as a safe QP solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch as th


ACTION_DIM = 2
DEFAULT_MAX_CONSTRAINTS = 18
DEFAULT_FEASIBILITY_TOL = 1e-6
DEFAULT_PARALLEL_TOL = 1e-10


@dataclass(frozen=True)
class NumpyProjection2D:
    """Result of a hard projection or an explicit infeasible-set fallback."""

    action: np.ndarray
    feasible: bool
    fallback_used: bool
    source: str
    active_indices: np.ndarray
    max_violation: float


@dataclass(frozen=True)
class TorchProjection2D:
    """Batched differentiable projection result."""

    action: th.Tensor
    feasible: th.Tensor
    fallback_used: th.Tensor
    source_code: th.Tensor
    selected_index: th.Tensor
    max_violation: th.Tensor


@dataclass(frozen=True)
class CBFContextLayout:
    """Flat observation layout for the fixed-size CBF constraint context."""

    base_observation_dim: int = 42
    max_constraints: int = DEFAULT_MAX_CONSTRAINTS

    @property
    def rows_start(self) -> int:
        return int(self.base_observation_dim)

    @property
    def rows_stop(self) -> int:
        return self.rows_start + int(self.max_constraints) * ACTION_DIM

    @property
    def bounds_start(self) -> int:
        return self.rows_stop

    @property
    def bounds_stop(self) -> int:
        return self.bounds_start + int(self.max_constraints)

    @property
    def mask_start(self) -> int:
        return self.bounds_stop

    @property
    def mask_stop(self) -> int:
        return self.mask_start + int(self.max_constraints)

    @property
    def observation_dim(self) -> int:
        return self.mask_stop


def _numpy_inputs(
    target: Any,
    rows: Any,
    bounds: Any,
    mask: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_array = np.asarray(target, dtype=float).reshape(-1)
    if target_array.size != ACTION_DIM or not np.all(np.isfinite(target_array)):
        raise ValueError("target must contain two finite action components")
    row_array = np.asarray(rows, dtype=float).reshape((-1, ACTION_DIM))
    bound_array = np.asarray(bounds, dtype=float).reshape(-1)
    if row_array.shape[0] != bound_array.size:
        raise ValueError("constraint row and bound counts disagree")
    if mask is None:
        mask_array = np.ones(bound_array.size, dtype=bool)
    else:
        mask_array = np.asarray(mask, dtype=bool).reshape(-1)
        if mask_array.size != bound_array.size:
            raise ValueError("constraint mask and bound counts disagree")
    finite = np.all(np.isfinite(row_array), axis=1) & np.isfinite(bound_array)
    if np.any(mask_array & ~finite):
        raise ValueError("active constraints must be finite")
    row_norm_sq = np.sum(row_array * row_array, axis=1)
    contradictory_zero = mask_array & (row_norm_sq <= 1e-20) & (bound_array < 0.0)
    if np.any(contradictory_zero):
        # Retain the row so the projection is correctly labelled infeasible.
        pass
    mask_array = mask_array & finite
    return target_array, row_array, bound_array, mask_array


def _numpy_action_bounds(
    action_low: Any | None,
    action_high: Any | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Validate optional hard actuator bounds for the fallback path."""

    if action_low is None and action_high is None:
        return None, None
    if action_low is None or action_high is None:
        raise ValueError("action_low and action_high must be provided together")
    low = np.asarray(action_low, dtype=float).reshape(-1)
    high = np.asarray(action_high, dtype=float).reshape(-1)
    if (
        low.size != ACTION_DIM
        or high.size != ACTION_DIM
        or not np.all(np.isfinite(low))
        or not np.all(np.isfinite(high))
        or np.any(low > high)
    ):
        raise ValueError("action bounds must be finite two-dimensional intervals")
    return low, high


def max_constraint_violation_numpy(
    action: Any,
    rows: Any,
    bounds: Any,
    mask: Any | None = None,
) -> float:
    action_array, row_array, bound_array, mask_array = _numpy_inputs(
        action, rows, bounds, mask
    )
    if not np.any(mask_array):
        return 0.0
    return float(np.max(row_array[mask_array] @ action_array - bound_array[mask_array]))


def _enumerate_numpy_candidates(
    target: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    mask: np.ndarray,
    *,
    parallel_tol: float,
) -> tuple[np.ndarray, list[str]]:
    candidates: list[np.ndarray] = [target]
    sources = ["raw"]
    active_indices = np.flatnonzero(mask)
    for index in active_indices:
        row = rows[index]
        norm_sq = float(row @ row)
        if norm_sq <= 1e-20:
            continue
        residual = float(row @ target - bounds[index])
        candidates.append(target - residual * row / norm_sq)
        sources.append(f"face:{int(index)}")

    for first_position, first in enumerate(active_indices):
        first_row = rows[first]
        for second in active_indices[first_position + 1 :]:
            second_row = rows[second]
            determinant = float(
                first_row[0] * second_row[1] - first_row[1] * second_row[0]
            )
            scale = max(float(np.linalg.norm(first_row) * np.linalg.norm(second_row)), 1.0)
            if abs(determinant) <= float(parallel_tol) * scale:
                continue
            candidate = np.asarray(
                [
                    (bounds[first] * second_row[1] - first_row[1] * bounds[second])
                    / determinant,
                    (first_row[0] * bounds[second] - bounds[first] * second_row[0])
                    / determinant,
                ],
                dtype=float,
            )
            candidates.append(candidate)
            sources.append(f"vertex:{int(first)},{int(second)}")
    return np.asarray(candidates, dtype=float), sources


def _least_violating_grid_candidate(
    target: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    mask: np.ndarray,
    candidates: np.ndarray,
    *,
    grid_size: int,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Choose a deterministic minimax fallback when the polytope is empty."""

    active_rows = rows[mask]
    active_bounds = bounds[mask]
    # The CBF rows can themselves be axis-aligned.  Do not infer actuator
    # limits from those rows when the caller has supplied the physical box:
    # an infeasible CBF boundary constraint must never expand the action box.
    if action_low is None:
        # Retain row inference for standalone projector use in unit tests.
        lower = np.full(ACTION_DIM, -np.inf, dtype=float)
        upper = np.full(ACTION_DIM, np.inf, dtype=float)
        for row, bound in zip(active_rows, active_bounds):
            for axis in range(ACTION_DIM):
                unit = np.zeros(ACTION_DIM, dtype=float)
                unit[axis] = 1.0
                if np.allclose(row, unit, atol=1e-10):
                    upper[axis] = min(upper[axis], float(bound))
                elif np.allclose(row, -unit, atol=1e-10):
                    lower[axis] = max(lower[axis], float(-bound))
    else:
        assert action_high is not None
        lower = np.asarray(action_low, dtype=float).copy()
        upper = np.asarray(action_high, dtype=float).copy()
    finite_pool = candidates[np.all(np.isfinite(candidates), axis=1)]
    for axis in range(ACTION_DIM):
        if not np.isfinite(lower[axis]):
            lower[axis] = (
                float(np.min(finite_pool[:, axis]))
                if finite_pool.size
                else float(target[axis] - 1.0)
            )
        if not np.isfinite(upper[axis]):
            upper[axis] = (
                float(np.max(finite_pool[:, axis]))
                if finite_pool.size
                else float(target[axis] + 1.0)
            )
        if abs(float(upper[axis] - lower[axis])) <= 1e-12:
            lower[axis] -= 1.0
            upper[axis] += 1.0
    if action_low is None and np.any(lower > upper):
        # Contradictory box rows: still return a finite, labelled fallback.
        midpoint = 0.5 * (lower + upper)
        midpoint[~np.isfinite(midpoint)] = 0.0
        return midpoint.astype(float), "fallback:contradictory_box"

    axes = [
        np.unique(
            np.r_[
                np.linspace(lower[axis], upper[axis], max(int(grid_size), 3)),
                np.clip(target[axis], lower[axis], upper[axis]),
                lower[axis],
                upper[axis],
            ]
        )
        for axis in range(ACTION_DIM)
    ]
    grid_x, grid_y = np.meshgrid(axes[0], axes[1], indexing="ij")
    grid = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])
    # Use this same bounded grid in the Torch fallback.  Analytic face/vertex
    # candidates are deliberately excluded here: for an empty polytope they
    # can miss the minimax compromise between contradictory faces, causing
    # the differentiable mean and hard executor to choose different actions.
    pool = grid
    # Action bounds remain hard even when the no-slack CBF faces conflict.
    # Analytic face/vertex candidates may lie outside those bounds, so do not
    # let the least-violating CBF fallback select them.
    in_box = np.all(
        (pool >= lower.reshape(1, -1) - 1e-9)
        & (pool <= upper.reshape(1, -1) + 1e-9),
        axis=1,
    )
    pool = pool[in_box]
    if pool.size == 0:
        pool = np.asarray([0.5 * (lower + upper)], dtype=float)
    if active_rows.size:
        positive = np.maximum(pool @ active_rows.T - active_bounds.reshape(1, -1), 0.0)
        max_violation = np.max(positive, axis=1)
        total_violation = np.sum(positive * positive, axis=1)
    else:
        max_violation = np.zeros(pool.shape[0], dtype=float)
        total_violation = np.zeros(pool.shape[0], dtype=float)
    distance = np.sum((pool - target.reshape(1, ACTION_DIM)) ** 2, axis=1)
    tie_tol = 1e-7
    min_max = float(np.min(max_violation))
    max_tie = max_violation <= min_max + tie_tol
    min_total = float(np.min(total_violation[max_tie]))
    total_tie = max_tie & (total_violation <= min_total + tie_tol)
    tied_indices = np.flatnonzero(total_tie)
    best = int(tied_indices[int(np.argmin(distance[tied_indices]))])
    return pool[best].astype(float), "fallback:least_violating"


def project_polytope_2d_numpy(
    target: Any,
    rows: Any,
    bounds: Any,
    mask: Any | None = None,
    *,
    feasibility_tol: float = DEFAULT_FEASIBILITY_TOL,
    active_tol: float = 1e-5,
    parallel_tol: float = DEFAULT_PARALLEL_TOL,
    fallback_grid_size: int = 41,
    action_low: Any | None = None,
    action_high: Any | None = None,
) -> NumpyProjection2D:
    """Project ``target`` onto a 2D action set with an optional hard box.

    When the CBF set is empty, ``action_low``/``action_high`` constrain the
    labelled least-violating fallback.  They are deliberately separate from
    the CBF rows because CBF boundary rows can conflict with the actuator box.
    """

    target_array, row_array, bound_array, mask_array = _numpy_inputs(
        target, rows, bounds, mask
    )
    physical_low, physical_high = _numpy_action_bounds(action_low, action_high)
    candidates, sources = _enumerate_numpy_candidates(
        target_array,
        row_array,
        bound_array,
        mask_array,
        parallel_tol=float(parallel_tol),
    )
    finite = np.all(np.isfinite(candidates), axis=1)
    if np.any(mask_array):
        violations = (
            candidates @ row_array[mask_array].T
            - bound_array[mask_array].reshape(1, -1)
        )
        feasible_candidates = finite & np.all(
            violations <= float(feasibility_tol), axis=1
        )
    else:
        feasible_candidates = finite
    if physical_low is not None:
        feasible_candidates &= np.all(
            (candidates >= physical_low.reshape(1, -1) - float(feasibility_tol))
            & (candidates <= physical_high.reshape(1, -1) + float(feasibility_tol)),
            axis=1,
        )

    if np.any(feasible_candidates):
        candidate_indices = np.flatnonzero(feasible_candidates)
        squared_distance = np.sum(
            (candidates[candidate_indices] - target_array.reshape(1, ACTION_DIM)) ** 2,
            axis=1,
        )
        selected = int(candidate_indices[int(np.argmin(squared_distance))])
        action = candidates[selected]
        feasible = True
        fallback_used = False
        source = sources[selected]
    else:
        action, source = _least_violating_grid_candidate(
            target_array,
            row_array,
            bound_array,
            mask_array,
            candidates,
            grid_size=int(fallback_grid_size),
            action_low=physical_low,
            action_high=physical_high,
        )
        feasible = False
        fallback_used = True

    if physical_low is not None:
        action = np.clip(action, physical_low, physical_high)

    if np.any(mask_array):
        residual = row_array @ action - bound_array
        active_indices = np.flatnonzero(mask_array & (np.abs(residual) <= float(active_tol)))
        max_violation = float(np.max(residual[mask_array]))
    else:
        active_indices = np.zeros(0, dtype=np.int64)
        max_violation = 0.0
    return NumpyProjection2D(
        action=np.asarray(action, dtype=np.float32),
        feasible=bool(feasible),
        fallback_used=bool(fallback_used),
        source=str(source),
        active_indices=active_indices.astype(np.int64),
        max_violation=max_violation,
    )


def _batched_torch_inputs(
    target: th.Tensor,
    rows: th.Tensor,
    bounds: th.Tensor,
    mask: th.Tensor | None,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if rows.ndim == 2:
        rows = rows.unsqueeze(0)
    if bounds.ndim == 1:
        bounds = bounds.unsqueeze(0)
    if target.ndim != 2 or target.shape[-1] != ACTION_DIM:
        raise ValueError("target must have shape [batch, 2]")
    if rows.ndim != 3 or rows.shape[-1] != ACTION_DIM:
        raise ValueError("rows must have shape [batch, constraints, 2]")
    if bounds.ndim != 2 or bounds.shape != rows.shape[:2]:
        raise ValueError("bounds must have shape [batch, constraints]")
    if rows.shape[0] != target.shape[0]:
        raise ValueError("target and constraint batch sizes disagree")
    if mask is None:
        mask = th.ones_like(bounds, dtype=th.bool)
    else:
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != bounds.shape:
            raise ValueError("mask must have shape [batch, constraints]")
        mask = mask > 0.5
    # The safety geometry is fixed during each PPO minibatch update.
    return target, rows.detach(), bounds.detach(), mask.detach()


def _torch_action_bounds(
    action_low: Any | None,
    action_high: Any | None,
    *,
    batch_size: int,
    dtype: th.dtype,
    device: th.device,
) -> tuple[th.Tensor | None, th.Tensor | None]:
    """Return validated [batch, 2] hard bounds for a batched projector."""

    if action_low is None and action_high is None:
        return None, None
    if action_low is None or action_high is None:
        raise ValueError("action_low and action_high must be provided together")
    low = th.as_tensor(action_low, dtype=dtype, device=device)
    high = th.as_tensor(action_high, dtype=dtype, device=device)
    if low.ndim == 1:
        low = low.reshape(1, ACTION_DIM).expand(batch_size, -1)
    if high.ndim == 1:
        high = high.reshape(1, ACTION_DIM).expand(batch_size, -1)
    if low.shape != (batch_size, ACTION_DIM) or high.shape != (batch_size, ACTION_DIM):
        raise ValueError("action bounds must have shape [2] or [batch, 2]")
    if (
        not bool(th.isfinite(low).all().item())
        or not bool(th.isfinite(high).all().item())
        or bool((low > high).any().item())
    ):
        raise ValueError("action bounds must be finite two-dimensional intervals")
    return low, high


def _torch_least_violating_grid_candidate(
    target: th.Tensor,
    rows: th.Tensor,
    bounds: th.Tensor,
    mask: th.Tensor,
    analytic_candidates: th.Tensor,
    *,
    grid_size: int = 41,
    tie_tol: float = 1e-7,
    action_low: th.Tensor | None = None,
    action_high: th.Tensor | None = None,
) -> th.Tensor:
    """Torch equivalent of the labelled NumPy infeasible-set fallback."""

    # An empty set has no projection Jacobian.  The fallback is behavioral and
    # explicitly excluded from projection-gradient claims/losses.
    target = target.detach()
    analytic_candidates = analytic_candidates.detach()
    dtype, device = target.dtype, target.device
    if action_low is None:
        lower = th.full((ACTION_DIM,), -th.inf, dtype=dtype, device=device)
        upper = th.full((ACTION_DIM,), th.inf, dtype=dtype, device=device)
        row_tol = 1e-9 if dtype == th.float64 else 1e-6
        for axis in range(ACTION_DIM):
            unit = th.zeros(ACTION_DIM, dtype=dtype, device=device)
            unit[axis] = 1.0
            positive_unit = mask & th.isclose(
                rows, unit.reshape(1, -1), atol=row_tol, rtol=0.0
            ).all(dim=1)
            negative_unit = mask & th.isclose(
                rows, -unit.reshape(1, -1), atol=row_tol, rtol=0.0
            ).all(dim=1)
            if bool(positive_unit.any().item()):
                upper[axis] = bounds[positive_unit].min()
            if bool(negative_unit.any().item()):
                lower[axis] = (-bounds[negative_unit]).max()
    else:
        assert action_high is not None
        lower = action_low.detach().to(dtype=dtype, device=device).reshape(ACTION_DIM)
        upper = action_high.detach().to(dtype=dtype, device=device).reshape(ACTION_DIM)

    finite_pool = analytic_candidates[th.isfinite(analytic_candidates).all(dim=1)]
    for axis in range(ACTION_DIM):
        if not bool(th.isfinite(lower[axis]).item()):
            lower[axis] = (
                finite_pool[:, axis].min()
                if finite_pool.numel()
                else target[axis] - 1.0
            )
        if not bool(th.isfinite(upper[axis]).item()):
            upper[axis] = (
                finite_pool[:, axis].max()
                if finite_pool.numel()
                else target[axis] + 1.0
            )
        if abs(float((upper[axis] - lower[axis]).detach().cpu().item())) <= 1e-12:
            lower[axis] = lower[axis] - 1.0
            upper[axis] = upper[axis] + 1.0
    if action_low is None and bool((lower > upper).any().item()):
        return 0.5 * (lower + upper)

    axes: list[th.Tensor] = []
    for axis in range(ACTION_DIM):
        clipped_target = th.minimum(
            th.maximum(target[axis], lower[axis]), upper[axis]
        ).reshape(1)
        axes.append(
            th.unique(
                th.cat(
                    [
                        th.linspace(
                            lower[axis],
                            upper[axis],
                            max(int(grid_size), 3),
                            dtype=dtype,
                            device=device,
                        ),
                        clipped_target,
                        lower[axis].reshape(1),
                        upper[axis].reshape(1),
                    ]
                ),
                sorted=True,
            )
        )
    grid_x, grid_y = th.meshgrid(axes[0], axes[1], indexing="ij")
    pool = th.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    active_rows = rows[mask]
    active_bounds = bounds[mask]
    if active_rows.numel():
        violations = th.relu(pool @ active_rows.T - active_bounds.reshape(1, -1))
        max_violation = violations.max(dim=1).values
        total_violation = violations.square().sum(dim=1)
    else:
        max_violation = th.zeros(pool.shape[0], dtype=dtype, device=device)
        total_violation = th.zeros_like(max_violation)
    distance = (pool - target.reshape(1, ACTION_DIM)).square().sum(dim=1)
    min_max = max_violation.min()
    max_tie = max_violation <= min_max + float(tie_tol)
    total_score = th.where(max_tie, total_violation, th.full_like(total_violation, th.inf))
    min_total = total_score.min()
    total_tie = max_tie & (total_violation <= min_total + float(tie_tol))
    distance_score = th.where(total_tie, distance, th.full_like(distance, th.inf))
    return pool[distance_score.argmin()]


def project_polytope_2d_torch(
    target: th.Tensor,
    rows: th.Tensor,
    bounds: th.Tensor,
    mask: th.Tensor | None = None,
    *,
    feasibility_tol: float = DEFAULT_FEASIBILITY_TOL,
    parallel_tol: float = DEFAULT_PARALLEL_TOL,
    action_low: Any | None = None,
    action_high: Any | None = None,
) -> TorchProjection2D:
    """Exact batched 2D projection with an almost-everywhere KKT gradient.

    Candidate selection is discrete, but gradients flow through the selected
    analytic candidate.  Consequently the Jacobian with respect to the target
    is identity in the interior, the tangent projector on one face, and zero
    at an independent two-face vertex.
    """

    target, rows, bounds, mask = _batched_torch_inputs(target, rows, bounds, mask)
    batch_size, constraint_count, _ = rows.shape
    device, dtype = target.device, target.dtype
    physical_low, physical_high = _torch_action_bounds(
        action_low,
        action_high,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
    )
    row_norm_sq = rows.square().sum(dim=2)
    valid_row = mask & (row_norm_sq > 1e-20)

    residual = (rows * target.unsqueeze(1)).sum(dim=2) - bounds
    safe_norm_sq = row_norm_sq.clamp_min(1e-20)
    single_candidates = target.unsqueeze(1) - (
        residual / safe_norm_sq
    ).unsqueeze(2) * rows

    first, second = th.triu_indices(
        constraint_count, constraint_count, offset=1, device=device
    )
    first_rows = rows[:, first, :]
    second_rows = rows[:, second, :]
    determinant = (
        first_rows[:, :, 0] * second_rows[:, :, 1]
        - first_rows[:, :, 1] * second_rows[:, :, 0]
    )
    row_scale = (
        th.linalg.vector_norm(first_rows, dim=2)
        * th.linalg.vector_norm(second_rows, dim=2)
    ).clamp_min(1.0)
    pair_valid = (
        mask[:, first]
        & mask[:, second]
        & (determinant.abs() > float(parallel_tol) * row_scale)
    )
    safe_determinant = th.where(
        pair_valid,
        determinant,
        th.ones_like(determinant),
    )
    first_bounds = bounds[:, first]
    second_bounds = bounds[:, second]
    pair_x = (
        first_bounds * second_rows[:, :, 1]
        - first_rows[:, :, 1] * second_bounds
    ) / safe_determinant
    pair_y = (
        first_rows[:, :, 0] * second_bounds
        - first_bounds * second_rows[:, :, 0]
    ) / safe_determinant
    pair_candidates = th.stack([pair_x, pair_y], dim=2)

    candidates = th.cat(
        [target.unsqueeze(1), single_candidates, pair_candidates], dim=1
    )
    raw_valid = th.ones((batch_size, 1), dtype=th.bool, device=device)
    candidate_valid = th.cat([raw_valid, valid_row, pair_valid], dim=1)
    finite_candidate = th.isfinite(candidates).all(dim=2)
    candidate_valid = candidate_valid & finite_candidate

    all_residuals = th.einsum("bkd,bmd->bkm", candidates, rows) - bounds.unsqueeze(1)
    constrained_residuals = th.where(
        mask.unsqueeze(1),
        all_residuals,
        th.full_like(all_residuals, -th.inf),
    )
    max_violation = constrained_residuals.amax(dim=2)
    no_constraints = ~mask.any(dim=1)
    max_violation = th.where(
        no_constraints.unsqueeze(1), th.zeros_like(max_violation), max_violation
    )
    feasible_candidate = candidate_valid & (
        max_violation <= float(feasibility_tol)
    )
    if physical_low is not None:
        feasible_candidate &= (
            (candidates >= physical_low.unsqueeze(1) - float(feasibility_tol))
            & (candidates <= physical_high.unsqueeze(1) + float(feasibility_tol))
        ).all(dim=2)
    squared_distance = (candidates - target.unsqueeze(1)).square().sum(dim=2)
    infinity = th.full_like(squared_distance, th.inf)
    feasible_score = th.where(feasible_candidate, squared_distance, infinity)
    feasible_distance, feasible_index = feasible_score.min(dim=1)
    has_feasible = th.isfinite(feasible_distance)

    # Empty no-slack sets use the exact same bounded minimax grid fallback as
    # the NumPy hard executor.  It is explicitly source code 3, never a
    # successful safe projection.
    selected_actions: list[th.Tensor] = []
    selected_indices: list[th.Tensor] = []
    for batch_index in range(batch_size):
        if bool(has_feasible[batch_index].item()):
            index = feasible_index[batch_index]
            selected_actions.append(candidates[batch_index, index])
            selected_indices.append(index)
        else:
            selected_actions.append(
                _torch_least_violating_grid_candidate(
                    target[batch_index],
                    rows[batch_index],
                    bounds[batch_index],
                    mask[batch_index],
                    candidates[batch_index],
                    action_low=(
                        None
                        if physical_low is None
                        else physical_low[batch_index]
                    ),
                    action_high=(
                        None
                        if physical_high is None
                        else physical_high[batch_index]
                    ),
                )
            )
            selected_indices.append(
                th.full((), -1, dtype=th.long, device=device)
            )
    selected_action = th.stack(selected_actions, dim=0)
    if physical_low is not None:
        selected_action = th.maximum(
            th.minimum(selected_action, physical_high), physical_low
        )
    selected_index = th.stack(selected_indices, dim=0)
    selected_residuals = (
        th.einsum("bd,bmd->bm", selected_action, rows) - bounds
    )
    selected_masked_residuals = th.where(
        mask,
        selected_residuals,
        th.full_like(selected_residuals, -th.inf),
    )
    selected_max_violation = selected_masked_residuals.amax(dim=1)
    selected_max_violation = th.where(
        no_constraints, th.zeros_like(selected_max_violation), selected_max_violation
    )
    # 0=raw, 1=single face, 2=two-face vertex, 3=infeasible fallback.
    source_code = th.where(
        ~has_feasible,
        th.full_like(selected_index, 3),
        th.where(
            selected_index == 0,
            th.zeros_like(selected_index),
            th.where(
                selected_index <= constraint_count,
                th.ones_like(selected_index),
                th.full_like(selected_index, 2),
            ),
        ),
    )
    return TorchProjection2D(
        action=selected_action,
        feasible=has_feasible,
        fallback_used=~has_feasible,
        source_code=source_code,
        selected_index=selected_index,
        max_violation=selected_max_violation,
    )


def projection_jacobian_from_active_rows(
    active_rows: Any,
    *,
    action_dim: int = ACTION_DIM,
    rcond: float = 1e-7,
) -> np.ndarray:
    """Return ``I - A.T (A A.T)^+ A`` for an active constraint set."""

    rows = np.asarray(active_rows, dtype=float).reshape((-1, int(action_dim)))
    identity = np.eye(int(action_dim), dtype=float)
    if rows.size == 0:
        return identity.astype(np.float32)
    jacobian = identity - rows.T @ np.linalg.pinv(rows @ rows.T, rcond=rcond) @ rows
    jacobian = 0.5 * (jacobian + jacobian.T)
    return jacobian.astype(np.float32)


def pad_cbf_context(
    rows: Any,
    bounds: Any,
    *,
    max_constraints: int = DEFAULT_MAX_CONSTRAINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad a variable-size constraint system without changing its row order."""

    row_array = np.asarray(rows, dtype=np.float32).reshape((-1, ACTION_DIM))
    bound_array = np.asarray(bounds, dtype=np.float32).reshape(-1)
    if row_array.shape[0] != bound_array.size:
        raise ValueError("constraint row and bound counts disagree")
    if row_array.shape[0] > int(max_constraints):
        raise ValueError(
            f"constraint count {row_array.shape[0]} exceeds padded capacity {max_constraints}"
        )
    padded_rows = np.zeros((int(max_constraints), ACTION_DIM), dtype=np.float32)
    padded_bounds = np.zeros(int(max_constraints), dtype=np.float32)
    mask = np.zeros(int(max_constraints), dtype=np.float32)
    count = row_array.shape[0]
    padded_rows[:count] = row_array
    padded_bounds[:count] = bound_array
    mask[:count] = 1.0
    return padded_rows, padded_bounds, mask


def append_cbf_context(
    observation: Any,
    rows: Any,
    bounds: Any,
    *,
    layout: CBFContextLayout = CBFContextLayout(),
) -> np.ndarray:
    base = np.asarray(observation, dtype=np.float32).reshape(-1)
    if base.size != int(layout.base_observation_dim):
        raise ValueError(
            f"expected {layout.base_observation_dim} base observation values, got {base.size}"
        )
    padded_rows, padded_bounds, mask = pad_cbf_context(
        rows, bounds, max_constraints=layout.max_constraints
    )
    return np.concatenate(
        [base, padded_rows.reshape(-1), padded_bounds, mask]
    ).astype(np.float32)


def split_cbf_context_numpy(
    observation: Any,
    *,
    layout: CBFContextLayout = CBFContextLayout(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(observation, dtype=np.float32)
    if array.shape[-1] != int(layout.observation_dim):
        raise ValueError(
            f"expected augmented observation width {layout.observation_dim}, got {array.shape[-1]}"
        )
    base = array[..., : layout.base_observation_dim]
    rows = array[..., layout.rows_start : layout.rows_stop].reshape(
        (*array.shape[:-1], layout.max_constraints, ACTION_DIM)
    )
    bounds = array[..., layout.bounds_start : layout.bounds_stop]
    mask = array[..., layout.mask_start : layout.mask_stop]
    return base, rows, bounds, mask


def split_cbf_context_torch(
    observation: th.Tensor,
    *,
    layout: CBFContextLayout = CBFContextLayout(),
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    if observation.shape[-1] != int(layout.observation_dim):
        raise ValueError(
            f"expected augmented observation width {layout.observation_dim}, got {observation.shape[-1]}"
        )
    base = observation[..., : layout.base_observation_dim]
    rows = observation[..., layout.rows_start : layout.rows_stop].reshape(
        (*observation.shape[:-1], layout.max_constraints, ACTION_DIM)
    )
    bounds = observation[..., layout.bounds_start : layout.bounds_stop]
    mask = observation[..., layout.mask_start : layout.mask_stop]
    return base, rows, bounds, mask
