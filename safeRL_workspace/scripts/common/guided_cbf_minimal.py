"""Minimal guided DDPG-CBF replay buffer and actor loss.

The notebook still defines the environment, reward, and CBF shield. This module
overrides only the guided reward-plus-loss pieces used by Python scripts.
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple, Optional

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DDPG
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import polyak_update


def _actor_gradient_diagnostics(
    actor: th.nn.Module,
    q_loss: th.Tensor,
    weighted_cbf_loss: th.Tensor,
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Measure the two actor-gradient components without touching ``.grad``.

    ``weighted_cbf_loss`` is the complete CBF contribution to the optimized
    objective (including the current lambda), so the reported ratio and cosine
    describe the gradients that are actually combined by the actor update.
    ``torch.autograd.grad`` leaves optimizer gradient buffers unchanged; both
    calls retain the graph for the subsequent, ordinary ``actor_loss.backward``.
    The norm ratio uses a stabilized denominator and is therefore zero when the
    weighted CBF gradient is zero.  Cosine alignment is undefined in that case
    (or when the Q gradient is zero), so it is returned as NaN with validity 0.
    """

    parameters = tuple(parameter for parameter in actor.parameters() if parameter.requires_grad)
    if not parameters:
        return {
            "g_q_norm": 0.0,
            "g_cbf_norm": 0.0,
            "g_cbf_to_g_q_ratio": 0.0,
            "g_q_g_cbf_cosine": float("nan"),
            "g_q_g_cbf_cosine_valid": 0.0,
        }

    def component_gradients(loss: th.Tensor) -> tuple[Optional[th.Tensor], ...]:
        if not loss.requires_grad:
            return tuple(None for _ in parameters)
        return th.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

    q_gradients = component_gradients(q_loss)
    cbf_gradients = component_gradients(weighted_cbf_loss)
    q_squared_norm = q_loss.new_zeros(())
    cbf_squared_norm = q_loss.new_zeros(())
    dot_product = q_loss.new_zeros(())
    for q_gradient, cbf_gradient in zip(q_gradients, cbf_gradients):
        if q_gradient is not None:
            q_squared_norm = q_squared_norm + q_gradient.detach().square().sum()
        if cbf_gradient is not None:
            cbf_squared_norm = cbf_squared_norm + cbf_gradient.detach().square().sum()
        if q_gradient is not None and cbf_gradient is not None:
            dot_product = dot_product + (q_gradient.detach() * cbf_gradient.detach()).sum()

    q_norm = th.sqrt(q_squared_norm)
    cbf_norm = th.sqrt(cbf_squared_norm)
    safe_epsilon = max(float(epsilon), th.finfo(q_norm.dtype).tiny)
    norm_ratio = cbf_norm / q_norm.clamp_min(safe_epsilon)
    q_norm_value = float(q_norm.cpu().item())
    cbf_norm_value = float(cbf_norm.cpu().item())
    cosine_valid = bool(q_norm_value > 0.0 and cbf_norm_value > 0.0)
    if cosine_valid:
        cosine = float(dot_product.cpu().item()) / (q_norm_value * cbf_norm_value)
        cosine = float(np.clip(cosine, -1.0, 1.0))
    else:
        cosine = float("nan")
    return {
        "g_q_norm": q_norm_value,
        "g_cbf_norm": cbf_norm_value,
        "g_cbf_to_g_q_ratio": float(norm_ratio.cpu().item()),
        "g_q_g_cbf_cosine": cosine,
        "g_q_g_cbf_cosine_valid": float(cosine_valid),
    }


def _local_projection_target(
    current_actions: th.Tensor,
    behavior_actions: th.Tensor,
    safe_behavior_actions: th.Tensor,
    projection_jacobians: th.Tensor,
) -> th.Tensor:
    """Return a detached local CBF projection target for the current actor.

    Around the replayed behavior action ``a_b``, the recorded CBF projection is
    approximated as ``F(s, a) = a_safe_b + J_b (a - a_b)``.  Evaluating that
    affine map at the current actor action cancels historical behavior noise in
    locally tangential directions (where ``J_b`` acts like the identity), while
    retaining supervision in constrained normal directions.  A feasible-side
    gate restores the projection's identity branch once the current actor has
    moved inward from the recorded boundary.  The result is detached so the
    actor loss differentiates the prediction, not its target.
    """

    local_delta = (current_actions - behavior_actions.detach()).unsqueeze(-1)
    safe_behavior = safe_behavior_actions.detach()
    behavior = behavior_actions.detach()
    affine_target = safe_behavior + th.bmm(
        projection_jacobians.detach(),
        local_delta,
    ).squeeze(-1)
    # The affine active-set map projects onto the recorded boundary even when
    # the current actor has already moved into the feasible half-space.  Use
    # the historical outward correction to gate that map; feasible-side
    # current actions should remain fixed points of F(s, a).
    outward = behavior - safe_behavior
    outside_score = ((current_actions - safe_behavior) * outward).sum(dim=1, keepdim=True)
    has_recorded_correction = outward.norm(dim=1, keepdim=True) > 1e-8
    use_projection = has_recorded_correction & (outside_score > 0.0)
    target = th.where(use_projection, affine_target, current_actions)
    return target.clamp(-1.0, 1.0).detach()


class CBFGuidedReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    safe_actions: th.Tensor
    interventions: th.Tensor
    projection_jacobians: th.Tensor


class CBFGuidedReplayBuffer(ReplayBuffer):
    """Replay buffer with actor-scale CBF targets and local projection Jacobians."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        action_shape = (self.buffer_size, self.n_envs, self.action_dim)
        scalar_shape = (self.buffer_size, self.n_envs, 1)
        jacobian_shape = (self.buffer_size, self.n_envs, self.action_dim, self.action_dim)
        self.safe_actions = np.zeros(action_shape, dtype=np.float32)
        self.interventions = np.zeros(scalar_shape, dtype=np.float32)
        self.projection_jacobians = np.broadcast_to(
            np.eye(self.action_dim, dtype=np.float32),
            jacobian_shape,
        ).copy()

    @staticmethod
    def _read_safe_action(info: dict[str, Any]) -> Optional[np.ndarray]:
        if "safe_action_phys" in info:
            action = np.asarray(info["safe_action_phys"], dtype=np.float32).reshape(-1)
        elif "cbf_a_safe_x" in info and "cbf_a_safe_y" in info:
            action = np.asarray([info["cbf_a_safe_x"], info["cbf_a_safe_y"]], dtype=np.float32)
        else:
            return None
        if action.size < 2 or not np.all(np.isfinite(action[:2])):
            return None
        return action[:2].astype(np.float32)

    def _to_actor_scale(self, action_phys: np.ndarray) -> np.ndarray:
        low = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)[: self.action_dim]
        high = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)[: self.action_dim]
        action_phys = np.asarray(action_phys, dtype=np.float32).reshape(-1)[: self.action_dim]
        scaled = 2.0 * ((np.clip(action_phys, low, high) - low) / np.maximum(high - low, 1e-6)) - 1.0
        return np.clip(scaled, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _read_event_intervention(info: dict[str, Any]) -> bool:
        if "cbf_event_intervened" in info:
            return bool(info["cbf_event_intervened"])
        if "intervention" in info:
            return bool(info["intervention"])
        correction = info.get("cbf_correction_norm", info.get("correction_norm"))
        if correction is None:
            return False
        threshold = float(info.get("cbf_event_intervention_threshold", 0.03))
        return bool(float(correction) > threshold)

    def _read_projection_jacobian(
        self,
        info: dict[str, Any],
        raw_action_scaled: np.ndarray,
        safe_action_scaled: np.ndarray,
        intervened: bool,
    ) -> np.ndarray:
        identity = np.eye(self.action_dim, dtype=np.float32)

        for key in ["cbf_projection_jacobian_scaled", "projection_jacobian_scaled"]:
            if key not in info:
                continue
            candidate = np.asarray(info[key], dtype=np.float32)
            if candidate.shape == identity.shape and np.all(np.isfinite(candidate)):
                return candidate

        for key in ["cbf_active_constraint_rows_scaled", "active_constraint_rows_scaled"]:
            if key not in info:
                continue
            rows = np.asarray(info[key], dtype=np.float32).reshape(-1, self.action_dim)
            if rows.size == 0 or not np.all(np.isfinite(rows)):
                continue
            gram = rows @ rows.T
            try:
                projection = identity - rows.T @ np.linalg.pinv(gram, rcond=1e-5) @ rows
            except np.linalg.LinAlgError:
                continue
            projection = 0.5 * (projection + projection.T)
            if projection.shape == identity.shape and np.all(np.isfinite(projection)):
                return projection.astype(np.float32)

        if not intervened:
            return identity

        correction = np.asarray(raw_action_scaled - safe_action_scaled, dtype=np.float32).reshape(-1)[: self.action_dim]
        correction_norm = float(np.linalg.norm(correction))
        if not np.isfinite(correction_norm) or correction_norm <= 1e-6:
            return identity
        normal = correction / correction_norm
        projection = identity - np.outer(normal, normal)
        return projection.astype(np.float32)

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        slot = self.pos
        raw_actions_scaled = np.asarray(action, dtype=np.float32).reshape((self.n_envs, self.action_dim))
        self.safe_actions[slot] = raw_actions_scaled

        for env_idx, info in enumerate(infos):
            safe_action_scaled = raw_actions_scaled[env_idx]
            safe_phys = self._read_safe_action(info)
            if safe_phys is not None:
                safe_action_scaled = self._to_actor_scale(safe_phys)
                self.safe_actions[slot, env_idx] = safe_action_scaled
            intervened = self._read_event_intervention(info)
            self.interventions[slot, env_idx, 0] = float(intervened)
            self.projection_jacobians[slot, env_idx] = self._read_projection_jacobian(
                info,
                raw_actions_scaled[env_idx],
                safe_action_scaled,
                intervened,
            )

        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> CBFGuidedReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.safe_actions[batch_inds, env_indices, :],
            self.interventions[batch_inds, env_indices, :],
            self.projection_jacobians[batch_inds, env_indices, :, :],
        )
        return CBFGuidedReplayBufferSamples(*tuple(map(self.to_torch, data)))


def _projection_from_active_rows(rows: np.ndarray, action_dim: int) -> np.ndarray:
    identity = np.eye(action_dim, dtype=np.float32)
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, action_dim)
    if rows.size == 0 or not np.all(np.isfinite(rows)):
        return identity
    try:
        projection = identity - rows.T @ np.linalg.pinv(rows @ rows.T, rcond=1e-5) @ rows
    except np.linalg.LinAlgError:
        return identity
    projection = 0.5 * (projection + projection.T)
    if not np.all(np.isfinite(projection)):
        return identity
    return projection.astype(np.float32)


def _positive_kkt_support(
    rows: np.ndarray,
    outward_correction: np.ndarray,
    *,
    relative_contribution_tolerance: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Identify tight constraints with a positive contribution to a projection.

    For the Euclidean projection ``safe = argmin ||a - raw||²`` with
    ``A a <= b``, stationarity gives ``raw - safe = A_active.T @ lambda`` with
    non-negative multipliers.  Solving that small NNLS system removes merely
    coincident tight rows (zero multiplier) from the normal-space report.
    Contribution thresholding uses ``||lambda_i A_i||``, which is invariant to
    rescaling an individual constraint row.
    """

    rows_array = np.asarray(rows, dtype=float)
    if rows_array.size == 0:
        rows_array = np.zeros((0, 2), dtype=float)
    rows_array = rows_array.reshape(-1, 2)
    correction = np.asarray(outward_correction, dtype=float).reshape(-1)[:2]
    correction_norm = float(np.linalg.norm(correction))
    if (
        rows_array.shape[0] == 0
        or not np.all(np.isfinite(rows_array))
        or not np.all(np.isfinite(correction))
        or correction_norm <= 1e-10
    ):
        return np.zeros(0, dtype=int), np.zeros(rows_array.shape[0], dtype=float), 0.0

    try:
        from scipy.optimize import nnls

        multipliers, residual_norm = nnls(rows_array.T, correction)
    except Exception:
        # SciPy is already a runtime dependency, but retain a deterministic
        # numerical fallback for unusual installations.
        unconstrained, *_ = np.linalg.lstsq(rows_array.T, correction, rcond=1e-10)
        multipliers = np.maximum(np.asarray(unconstrained, dtype=float), 0.0)
        residual_norm = float(np.linalg.norm(rows_array.T @ multipliers - correction))

    contribution_norms = np.abs(multipliers) * np.linalg.norm(rows_array, axis=1)
    threshold = max(1e-8, float(relative_contribution_tolerance) * correction_norm)
    support = np.flatnonzero(contribution_norms > threshold).astype(int)
    return support, np.asarray(multipliers, dtype=float), float(residual_norm)


def _score_constraint_violation(rows: np.ndarray, bounds: np.ndarray, action: np.ndarray) -> dict[str, float]:
    rows = np.asarray(rows, dtype=float).reshape(-1, 2)
    bounds = np.asarray(bounds, dtype=float).reshape(-1)
    action = np.asarray(action, dtype=float).reshape(-1)[:2]
    if rows.size == 0:
        return {
            "max_constraint_violation": 0.0,
            "positive_violation_l2": 0.0,
            "positive_violation_sum": 0.0,
        }
    positive = np.maximum(rows @ action - bounds, 0.0)
    return {
        "max_constraint_violation": float(np.max(positive)) if positive.size else 0.0,
        "positive_violation_l2": float(np.sqrt(np.sum(positive**2))) if positive.size else 0.0,
        "positive_violation_sum": float(np.sum(positive)) if positive.size else 0.0,
    }


def _grid_least_violating_bounded_action(
    a_rl: np.ndarray,
    constraint_rows: list[np.ndarray],
    constraint_bounds: list[float],
    ax_bounds: tuple[float, float],
    ay_bounds: tuple[float, float],
) -> np.ndarray:
    lb = np.asarray([float(ax_bounds[0]), float(ay_bounds[0])], dtype=float)
    ub = np.asarray([float(ax_bounds[1]), float(ay_bounds[1])], dtype=float)
    target = np.clip(np.asarray(a_rl, dtype=float).reshape(-1)[:2], lb, ub)
    rows = np.asarray(constraint_rows, dtype=float).reshape(-1, 2)
    bounds = np.asarray(constraint_bounds, dtype=float).reshape(-1)
    ax_grid = np.unique(np.r_[np.linspace(lb[0], ub[0], 49), target[0], lb[0], ub[0]])
    ay_grid = np.unique(np.r_[np.linspace(lb[1], ub[1], 49), target[1], lb[1], ub[1]])
    best_action = target.copy()
    best_score: tuple[float, float, float] | None = None
    for ax in ax_grid:
        for ay in ay_grid:
            candidate = np.asarray([ax, ay], dtype=float)
            scores = _score_constraint_violation(rows, bounds, candidate)
            score = (
                scores["max_constraint_violation"],
                scores["positive_violation_l2"],
                float(np.sum((candidate - target) ** 2)),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_action = candidate
    return np.clip(best_action, lb, ub).astype(np.float32)


def _continuous_least_violating_bounded_action(
    namespace: dict[str, Any],
    target: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    lb_action: np.ndarray,
    ub_action: np.ndarray,
    grid_action: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    target = np.clip(np.asarray(target, dtype=float).reshape(-1)[:2], lb_action, ub_action)
    rows = np.asarray(rows, dtype=float).reshape(-1, 2)
    bounds = np.asarray(bounds, dtype=float).reshape(-1)
    action_weight = float(namespace.get("CBF_SOFT_QP_ACTION_WEIGHT", 1.0))
    violation_weight = float(namespace.get("CBF_CONTINUOUS_FALLBACK_VIOLATION_WEIGHT", 10_000.0))
    best_action = np.clip(np.asarray(grid_action, dtype=float).reshape(-1)[:2], lb_action, ub_action)
    best_scores = _score_constraint_violation(rows, bounds, best_action)
    best_score: tuple[float, float, float] = (
        best_scores["max_constraint_violation"],
        best_scores["positive_violation_l2"],
        float(np.sum((best_action - target) ** 2)),
    )
    error = ""

    def objective(action: np.ndarray) -> float:
        action = np.asarray(action, dtype=float).reshape(-1)[:2]
        positive = np.maximum(rows @ action - bounds, 0.0)
        return float(violation_weight * np.sum(positive**2) + action_weight * np.sum((action - target) ** 2))

    starts = [
        target,
        best_action,
        np.clip(np.zeros(2, dtype=float), lb_action, ub_action),
        np.asarray([lb_action[0], lb_action[1]], dtype=float),
        np.asarray([lb_action[0], ub_action[1]], dtype=float),
        np.asarray([ub_action[0], lb_action[1]], dtype=float),
        np.asarray([ub_action[0], ub_action[1]], dtype=float),
    ]
    try:
        from scipy.optimize import minimize

        for start in starts:
            result = minimize(
                objective,
                np.clip(np.asarray(start, dtype=float), lb_action, ub_action),
                method="L-BFGS-B",
                bounds=[(float(lb_action[0]), float(ub_action[0])), (float(lb_action[1]), float(ub_action[1]))],
                options={"maxiter": int(namespace.get("CBF_CONTINUOUS_FALLBACK_MAXITER", 80)), "ftol": 1e-10},
            )
            if not bool(result.success) and not np.all(np.isfinite(result.x)):
                continue
            candidate = np.clip(np.asarray(result.x, dtype=float).reshape(-1)[:2], lb_action, ub_action)
            scores = _score_constraint_violation(rows, bounds, candidate)
            score = (
                scores["max_constraint_violation"],
                scores["positive_violation_l2"],
                float(np.sum((candidate - target) ** 2)),
            )
            if score < best_score:
                best_score = score
                best_action = candidate
                best_scores = scores
    except Exception as exc:
        error = repr(exc)

    info = {
        "continuous_success": error == "",
        "continuous_error": error,
        "continuous_objective": objective(best_action),
        **best_scores,
    }
    return best_action.astype(np.float32), info


def _linprog_minimax_bounded_action(
    namespace: dict[str, Any],
    target: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    lb_action: np.ndarray,
    ub_action: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    target = np.clip(np.asarray(target, dtype=float).reshape(-1)[:2], lb_action, ub_action)
    rows = np.asarray(rows, dtype=float).reshape(-1, 2)
    bounds = np.asarray(bounds, dtype=float).reshape(-1)
    action = target.copy()
    error = ""
    success = False
    try:
        from scipy.optimize import linprog

        m_constraints = int(rows.shape[0])
        a_ub = np.zeros((m_constraints + 1, 3), dtype=float)
        b_ub = np.zeros(m_constraints + 1, dtype=float)
        a_ub[:m_constraints, :2] = rows
        a_ub[:m_constraints, 2] = -1.0
        b_ub[:m_constraints] = bounds
        a_ub[m_constraints, 2] = -1.0
        b_ub[m_constraints] = 0.0
        t_upper = float(namespace.get("CBF_MINIMAX_FALLBACK_T_UPPER", 10_000.0))
        result = linprog(
            c=np.asarray([0.0, 0.0, 1.0], dtype=float),
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=[
                (float(lb_action[0]), float(ub_action[0])),
                (float(lb_action[1]), float(ub_action[1])),
                (0.0, t_upper),
            ],
            method="highs",
        )
        success = bool(result.success and np.all(np.isfinite(result.x)))
        if success:
            action = np.clip(np.asarray(result.x[:2], dtype=float), lb_action, ub_action)
    except Exception as exc:
        error = repr(exc)

    scores = _score_constraint_violation(rows, bounds, action)
    info = {
        "linprog_success": success,
        "linprog_error": error,
        **scores,
    }
    return action.astype(np.float32), info


def _soft_least_violating_bounded_action(
    namespace: dict[str, Any],
    a_rl: np.ndarray,
    constraint_rows: list[np.ndarray],
    constraint_bounds: list[float],
    ax_bounds: tuple[float, float],
    ay_bounds: tuple[float, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    lb_action = np.asarray([float(ax_bounds[0]), float(ay_bounds[0])], dtype=float)
    ub_action = np.asarray([float(ax_bounds[1]), float(ay_bounds[1])], dtype=float)
    target = np.clip(np.asarray(a_rl, dtype=float).reshape(-1)[:2], lb_action, ub_action)
    rows = np.asarray(constraint_rows, dtype=float).reshape(-1, 2)
    bounds = np.asarray(constraint_bounds, dtype=float).reshape(-1)
    grid_action = _grid_least_violating_bounded_action(
        target,
        constraint_rows,
        constraint_bounds,
        ax_bounds,
        ay_bounds,
    )
    grid_scores = _score_constraint_violation(rows, bounds, grid_action)
    linprog_action, linprog_info = _linprog_minimax_bounded_action(
        namespace,
        target,
        rows,
        bounds,
        lb_action,
        ub_action,
    )
    linprog_scores = _score_constraint_violation(rows, bounds, linprog_action)
    continuous_action, continuous_info = _continuous_least_violating_bounded_action(
        namespace,
        target,
        rows,
        bounds,
        lb_action,
        ub_action,
        linprog_action,
    )
    continuous_scores = _score_constraint_violation(rows, bounds, continuous_action)
    if rows.size == 0:
        info = {
            "soft_qp_success": True,
            "soft_qp_used": False,
            "soft_qp_error": "",
            "soft_qp_slack_l2": 0.0,
            "soft_qp_slack_max": 0.0,
            "fallback_source": "no_constraints",
            "linprog_success": bool(linprog_info.get("linprog_success", False)),
            "linprog_error": str(linprog_info.get("linprog_error", "")),
            "continuous_success": bool(continuous_info.get("continuous_success", False)),
            "continuous_error": str(continuous_info.get("continuous_error", "")),
            **grid_scores,
        }
        return target.astype(np.float32), info

    solution = None
    soft_error = ""
    m_constraints = int(rows.shape[0])
    try:
        sparse = namespace["sparse"]
        solve_qp = namespace["solve_qp"]
        action_weight = float(namespace.get("CBF_SOFT_QP_ACTION_WEIGHT", 1.0))
        slack_weight = float(namespace.get("CBF_SOFT_QP_SLACK_WEIGHT", 5_000.0))
        slack_upper = float(namespace.get("CBF_SOFT_QP_SLACK_UPPER", 1_000.0))
        dim = 2 + m_constraints
        p_diag = np.r_[
            np.full(2, 2.0 * action_weight, dtype=float),
            np.full(m_constraints, 2.0 * slack_weight, dtype=float),
        ]
        p_matrix = sparse.diags(p_diag, format="csc")
        q_vector = np.r_[-2.0 * action_weight * target, np.zeros(m_constraints, dtype=float)]
        g_matrix = np.zeros((m_constraints, dim), dtype=float)
        g_matrix[:, :2] = rows
        g_matrix[np.arange(m_constraints), 2 + np.arange(m_constraints)] = -1.0
        lower = np.r_[lb_action, np.zeros(m_constraints, dtype=float)]
        upper = np.r_[ub_action, np.full(m_constraints, slack_upper, dtype=float)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"OSQP exited.*")
            solution = solve_qp(
                p_matrix,
                q_vector,
                G=sparse.csc_matrix(g_matrix),
                h=bounds.astype(float),
                lb=lower,
                ub=upper,
                solver=namespace.get("CBF_QP_SOLVER", "osqp"),
                verbose=False,
            )
    except Exception as exc:
        soft_error = repr(exc)

    soft_success = solution is not None and bool(np.all(np.isfinite(solution)))
    if soft_success:
        solution = np.asarray(solution, dtype=float).reshape(-1)
        soft_action = np.clip(solution[:2], lb_action, ub_action)
        soft_slack = np.maximum(solution[2:], 0.0)
        soft_scores = _score_constraint_violation(rows, bounds, soft_action)
        soft_score = (
            soft_scores["max_constraint_violation"],
            soft_scores["positive_violation_l2"],
            float(np.sum((soft_action - target) ** 2)),
        )
        grid_score = (
            grid_scores["max_constraint_violation"],
            grid_scores["positive_violation_l2"],
            float(np.sum((grid_action.astype(float) - target) ** 2)),
        )
        if grid_score < soft_score:
            best_action = grid_action.astype(float)
            best_scores = grid_scores
            fallback_source = "grid_refinement"
        else:
            best_action = soft_action
            best_scores = soft_scores
            fallback_source = "soft_qp"
        linprog_score = (
            linprog_scores["max_constraint_violation"],
            linprog_scores["positive_violation_l2"],
            float(np.sum((linprog_action.astype(float) - target) ** 2)),
        )
        current_score = (
            best_scores["max_constraint_violation"],
            best_scores["positive_violation_l2"],
            float(np.sum((np.asarray(best_action, dtype=float) - target) ** 2)),
        )
        if linprog_score < current_score:
            best_action = linprog_action.astype(float)
            best_scores = linprog_scores
            fallback_source = "linprog_minimax"
        continuous_score = (
            continuous_scores["max_constraint_violation"],
            continuous_scores["positive_violation_l2"],
            float(np.sum((continuous_action.astype(float) - target) ** 2)),
        )
        current_score = (
            best_scores["max_constraint_violation"],
            best_scores["positive_violation_l2"],
            float(np.sum((np.asarray(best_action, dtype=float) - target) ** 2)),
        )
        if continuous_score < current_score:
            best_action = continuous_action.astype(float)
            best_scores = continuous_scores
            fallback_source = "continuous_refinement"
        info = {
            "soft_qp_success": True,
            "soft_qp_used": True,
            "soft_qp_error": "",
            "soft_qp_slack_l2": float(np.linalg.norm(soft_slack)),
            "soft_qp_slack_max": float(np.max(soft_slack)) if soft_slack.size else 0.0,
            "fallback_source": fallback_source,
            "linprog_success": bool(linprog_info.get("linprog_success", False)),
            "linprog_error": str(linprog_info.get("linprog_error", "")),
            "continuous_success": bool(continuous_info.get("continuous_success", False)),
            "continuous_error": str(continuous_info.get("continuous_error", "")),
            **best_scores,
        }
        return np.asarray(best_action, dtype=np.float32), info

    continuous_score = (
        continuous_scores["max_constraint_violation"],
        continuous_scores["positive_violation_l2"],
        float(np.sum((continuous_action.astype(float) - target) ** 2)),
    )
    grid_score = (
        grid_scores["max_constraint_violation"],
        grid_scores["positive_violation_l2"],
        float(np.sum((grid_action.astype(float) - target) ** 2)),
    )
    if continuous_score < grid_score:
        fallback_action = continuous_action.astype(np.float32)
        fallback_source = "continuous_after_soft_qp_failure"
        fallback_scores = continuous_scores
    else:
        fallback_action = grid_action.astype(np.float32)
        fallback_source = "grid_after_soft_qp_failure"
        fallback_scores = grid_scores
    linprog_score = (
        linprog_scores["max_constraint_violation"],
        linprog_scores["positive_violation_l2"],
        float(np.sum((linprog_action.astype(float) - target) ** 2)),
    )
    current_score = (
        fallback_scores["max_constraint_violation"],
        fallback_scores["positive_violation_l2"],
        float(np.sum((fallback_action.astype(float) - target) ** 2)),
    )
    if linprog_score < current_score:
        fallback_action = linprog_action.astype(np.float32)
        fallback_source = "linprog_after_soft_qp_failure"
        fallback_scores = linprog_scores
    info = {
        "soft_qp_success": False,
        "soft_qp_used": True,
        "soft_qp_error": soft_error,
        "soft_qp_slack_l2": np.nan,
        "soft_qp_slack_max": np.nan,
        "fallback_source": fallback_source,
        "linprog_success": bool(linprog_info.get("linprog_success", False)),
        "linprog_error": str(linprog_info.get("linprog_error", "")),
        "continuous_success": bool(continuous_info.get("continuous_success", False)),
        "continuous_error": str(continuous_info.get("continuous_error", "")),
        **fallback_scores,
    }
    return fallback_action.astype(np.float32), info


def _install_robust_cbf_fallback(namespace: dict[str, Any]) -> None:
    current = namespace.get("_least_violating_bounded_action")
    if getattr(current, "_robust_cbf_fallback", False):
        return
    if current is not None and "_grid_least_violating_bounded_action_original" not in namespace:
        namespace["_grid_least_violating_bounded_action_original"] = current

    def robust_least_violating_bounded_action(
        a_rl: np.ndarray,
        constraint_rows: list[np.ndarray],
        constraint_bounds: list[float],
        ax_bounds: tuple[float, float],
        ay_bounds: tuple[float, float],
    ) -> np.ndarray:
        action, _ = _soft_least_violating_bounded_action(
            namespace,
            a_rl,
            constraint_rows,
            constraint_bounds,
            ax_bounds,
            ay_bounds,
        )
        return action.astype(np.float32)

    robust_least_violating_bounded_action._robust_cbf_fallback = True  # type: ignore[attr-defined]
    namespace["_least_violating_bounded_action"] = robust_least_violating_bounded_action
    namespace["cbf_soft_fallback_projection"] = lambda *args, **kwargs: _soft_least_violating_bounded_action(
        namespace,
        *args,
        **kwargs,
    )


def _install_cbf_projection_reporting(namespace: dict[str, Any]) -> None:
    original_filter = namespace.get("cbf_filter_2d")
    pairwise_constraint = namespace.get("pairwise_hocbf_constraint")
    if original_filter is None or pairwise_constraint is None:
        return
    if getattr(original_filter, "_guided_projection_reporting", False):
        return

    def cbf_filter_2d_with_projection(
        a_rl,
        ego: dict[str, float],
        neighbors: list[dict[str, float]],
        road_width: float,
        ax_bounds: tuple[float, float] | None = None,
        ay_bounds: tuple[float, float] | None = None,
        eps_side: float | None = None,
        k0: float | None = None,
        k1: float | None = None,
        max_neighbor_constraints: Optional[int] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        ax_bounds = namespace["CBF_AX_BOUNDS"] if ax_bounds is None else ax_bounds
        ay_bounds = namespace["CBF_AY_BOUNDS"] if ay_bounds is None else ay_bounds
        eps_side = float(namespace["CBF_EPS_SIDE"] if eps_side is None else eps_side)
        k0 = float(namespace["CBF_K0"] if k0 is None else k0)
        k1 = float(namespace["CBF_K1"] if k1 is None else k1)
        if max_neighbor_constraints is None:
            max_neighbor_constraints = namespace.get("CBF_MAX_NEIGHBOR_CONSTRAINTS")

        a_rl = np.asarray(a_rl, dtype=float).reshape(-1)
        if a_rl.size < 2:
            raise ValueError("a_rl must contain [ax, ay].")
        a_rl = a_rl[:2]

        active_neighbors = list(neighbors)
        if max_neighbor_constraints is not None:
            active_neighbors = active_neighbors[: int(max_neighbor_constraints)]

        rows: list[np.ndarray] = []
        bounds: list[float] = []
        constraint_types: list[str] = []
        constraint_neighbor_ranks: list[int] = []
        min_h = np.inf
        min_center_distance = np.inf
        min_required_distance = np.inf
        neighbor_constraints = 0
        for neighbor_rank, neighbor in enumerate(active_neighbors):
            finite_values = [
                ego.get("x", 0.0),
                ego.get("y", 0.0),
                ego.get("vx", 0.0),
                ego.get("vy", 0.0),
                ego.get("length", 0.0),
                ego.get("width", 0.0),
                neighbor.get("x", 0.0),
                neighbor.get("y", 0.0),
                neighbor.get("vx", 0.0),
                neighbor.get("vy", 0.0),
                neighbor.get("length", 0.0),
                neighbor.get("width", 0.0),
                neighbor.get("ax", 0.0),
                neighbor.get("ay", 0.0),
            ]
            if not np.all(np.isfinite(np.asarray(finite_values, dtype=float))):
                continue
            other_acc = np.asarray(
                [float(neighbor.get("ax", 0.0)), float(neighbor.get("ay", 0.0))],
                dtype=float,
            )
            try:
                row, bound, h_ij, center_distance, required_distance = pairwise_constraint(
                    ego,
                    neighbor,
                    eps_side=eps_side,
                    k0=k0,
                    k1=k1,
                    other_acc=other_acc,
                )
            except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
                continue
            if not np.all(np.isfinite(np.asarray(row, dtype=float))) or not np.isfinite(float(bound)):
                continue
            rows.append(np.asarray(row, dtype=float).reshape(-1)[:2])
            bounds.append(float(bound))
            signed_dx = float(neighbor.get("signed_dx", neighbor.get("x", 0.0) - ego.get("x", 0.0)))
            lateral_offset = abs(float(neighbor.get("y", 0.0) - ego.get("y", 0.0)))
            alongside_distance = 0.5 * (
                float(ego.get("length", 0.0)) + float(neighbor.get("length", 0.0))
            )
            if abs(signed_dx) <= max(alongside_distance, 1e-6) and lateral_offset > 0.5:
                neighbor_type = "neighbor_side"
            elif signed_dx >= 0.0:
                neighbor_type = "neighbor_front"
            else:
                neighbor_type = "neighbor_rear"
            constraint_types.append(neighbor_type)
            constraint_neighbor_ranks.append(int(neighbor_rank))
            min_h = min(min_h, float(h_ij))
            min_center_distance = min(min_center_distance, float(center_distance))
            min_required_distance = min(min_required_distance, float(required_distance))
            neighbor_constraints += 1

        ego_y = float(ego["y"])
        ego_vy = float(ego["vy"])
        ego_half_width = 0.5 * float(ego["width"])
        h_left = ego_y - ego_half_width
        h_right = float(road_width) - ego_half_width - ego_y
        rows.extend([np.asarray([0.0, -1.0], dtype=float), np.asarray([0.0, 1.0], dtype=float)])
        bounds.extend([float(k1 * ego_vy + k0 * h_left), float(-k1 * ego_vy + k0 * h_right)])
        constraint_types.extend(["road_left", "road_right"])
        constraint_neighbor_ranks.extend([-1, -1])

        low = np.asarray([float(ax_bounds[0]), float(ay_bounds[0])], dtype=np.float32)
        high = np.asarray([float(ax_bounds[1]), float(ay_bounds[1])], dtype=np.float32)
        half_range = 0.5 * np.maximum(high - low, 1e-6)
        rows_arr = np.asarray(rows, dtype=np.float32).reshape(-1, 2)
        bounds_arr = np.asarray(bounds, dtype=np.float32).reshape(-1)
        rl_constraint_values = rows_arr @ a_rl.astype(np.float32) - bounds_arr
        raw_is_feasible = (
            bool(np.all(rl_constraint_values <= float(namespace.get("CBF_QP_FEASIBILITY_TOL", 1e-3))))
            and bool(np.all(a_rl >= low - float(namespace.get("CBF_QP_FEASIBILITY_TOL", 1e-3))))
            and bool(np.all(a_rl <= high + float(namespace.get("CBF_QP_FEASIBILITY_TOL", 1e-3))))
        )
        qp_error = ""
        projection_solver = "raw"
        if raw_is_feasible:
            qp_success = True
            a_safe = a_rl.copy()
            fallback_info: dict[str, Any] = {
                "soft_qp_success": False,
                "soft_qp_used": False,
                "soft_qp_error": "",
                "soft_qp_slack_l2": 0.0,
                "soft_qp_slack_max": 0.0,
                "fallback_source": "none",
            }
        else:
            solution = None
            direct_error = ""
            fallback_info = {
                "soft_qp_success": False,
                "soft_qp_used": False,
                "soft_qp_error": "",
                "soft_qp_slack_l2": np.nan,
                "soft_qp_slack_max": np.nan,
                "fallback_source": "none",
            }
            direct_projector = namespace.get("project_action_to_linear_constraints_2d")
            if callable(direct_projector):
                try:
                    solution = direct_projector(
                        a_rl,
                        rows,
                        bounds,
                        ax_bounds=ax_bounds,
                        ay_bounds=ay_bounds,
                        feasibility_tol=float(namespace.get("CBF_QP_FEASIBILITY_TOL", 1e-3)),
                    )
                except Exception as exc:
                    direct_error = repr(exc)

            qp_success = solution is not None and bool(np.all(np.isfinite(solution)))
            if qp_success:
                a_safe = np.clip(np.asarray(solution, dtype=float).reshape(-1)[:2], low, high)
                projection_solver = str(namespace.get("CBF_PROJECTION_SOLVER", "active_set_2d"))
            else:
                # The active-set projection is exact in 2D. Retain OSQP only
                # as a numerical fallback for degenerate or infeasible sets.
                solution = None
                try:
                    sparse = namespace["sparse"]
                    solve_qp = namespace["solve_qp"]
                    p_matrix = sparse.csc_matrix(2.0 * np.eye(2, dtype=float))
                    q_vector = -2.0 * a_rl
                    g_matrix = sparse.csc_matrix(rows_arr.astype(float))
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=r"OSQP exited.*")
                        solution = solve_qp(
                            p_matrix,
                            q_vector,
                            G=g_matrix,
                            h=bounds_arr.astype(float),
                            lb=low.astype(float),
                            ub=high.astype(float),
                            solver=namespace.get("CBF_QP_SOLVER", "osqp"),
                            verbose=False,
                        )
                except Exception as exc:
                    qp_error = repr(exc)
                qp_success = solution is not None and bool(np.all(np.isfinite(solution)))
                if qp_success:
                    a_safe = np.clip(np.asarray(solution, dtype=float).reshape(-1)[:2], low, high)
                    projection_solver = str(namespace.get("CBF_QP_SOLVER", "osqp"))
                else:
                    if not qp_error:
                        qp_error = direct_error
                    projection_solver = "fallback"
                    if "cbf_soft_fallback_projection" in namespace:
                        a_safe, fallback_info = namespace["cbf_soft_fallback_projection"](
                            a_rl,
                            rows,
                            bounds,
                            ax_bounds,
                            ay_bounds,
                        )
                    elif "_least_violating_bounded_action" in namespace:
                        a_safe = namespace["_least_violating_bounded_action"](
                            a_rl,
                            rows,
                            bounds,
                            ax_bounds,
                            ay_bounds,
                        )
                        fallback_info["fallback_source"] = "legacy_grid"
                    else:
                        a_safe = np.asarray([float(ax_bounds[0]), 0.0], dtype=float)
                        fallback_info["fallback_source"] = "emergency_brake"

        safe = np.asarray(a_safe, dtype=np.float32).reshape(-1)[:2]
        safe_constraint_values = rows_arr @ safe - bounds_arr
        correction_norm = float(np.linalg.norm(safe - a_rl.astype(np.float32)))
        if not np.isfinite(min_h):
            min_h = np.nan
        if not np.isfinite(min_center_distance):
            min_center_distance = np.nan
        if not np.isfinite(min_required_distance):
            min_required_distance = np.nan
        # Include action-box faces in the active-set audit.  Tightness is
        # normalized by each row's action-scale magnitude, then a positive-KKT
        # support test removes coincident zero-multiplier rows.
        box_rows = np.asarray(
            [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        box_bounds = np.asarray([-low[0], high[0], -low[1], high[1]], dtype=np.float32)
        candidate_rows = np.vstack([rows_arr, box_rows]).astype(np.float32)
        candidate_bounds = np.concatenate([bounds_arr, box_bounds]).astype(np.float32)
        candidate_types = tuple(
            list(constraint_types)
            + ["action_ax_min", "action_ax_max", "action_ay_min", "action_ay_max"]
        )
        candidate_indices = np.concatenate(
            [np.arange(rows_arr.shape[0], dtype=np.int32), np.asarray([-1, -2, -3, -4], dtype=np.int32)]
        )
        candidate_residuals = candidate_rows @ safe - candidate_bounds
        action_magnitude_scale = np.sum(
            np.abs(candidate_rows) * half_range.reshape(1, -1),
            axis=1,
        )
        residual_scales = np.maximum.reduce(
            [action_magnitude_scale, np.abs(candidate_bounds), np.ones_like(candidate_bounds)]
        )
        active_tol = max(float(namespace.get("CBF_QP_ACTIVE_TOL", 1e-3)), 0.0)
        tight_mask = np.abs(candidate_residuals) <= active_tol * residual_scales
        tight_rows = candidate_rows[tight_mask]
        tight_types = tuple(
            constraint_type
            for constraint_type, is_tight in zip(candidate_types, tight_mask)
            if bool(is_tight)
        )
        tight_indices = candidate_indices[tight_mask]

        fallback_used = bool(not qp_success)
        kkt_residual_norm = np.nan
        reported_multipliers = np.zeros(0, dtype=np.float32)
        if fallback_used or tight_rows.shape[0] == 0:
            active_scaled = np.zeros((0, 2), dtype=np.float32)
            active_physical = np.zeros((0, 2), dtype=np.float32)
            reported_active_types: list[str] = []
            reported_active_indices = np.zeros(0, dtype=np.int32)
        else:
            support, tight_multipliers, kkt_residual_norm = _positive_kkt_support(
                tight_rows,
                a_rl.astype(np.float32) - safe,
            )
            active_physical = tight_rows[support].astype(np.float32)
            active_scaled = active_physical * half_range.reshape(1, -1)
            reported_active_types = [tight_types[index] for index in support]
            reported_active_indices = tight_indices[support].astype(np.int32)
            reported_multipliers = tight_multipliers[support].astype(np.float32)
        projection = _projection_from_active_rows(active_scaled, 2)

        info = {
            "a_safe": safe.astype(np.float32),
            "correction_norm": correction_norm,
            "raw_feasible": bool(raw_is_feasible),
            "max_constraint_violation_rl": float(np.max(rl_constraint_values)) if len(rl_constraint_values) else 0.0,
            "max_constraint_violation_safe": float(np.max(safe_constraint_values)) if len(safe_constraint_values) else 0.0,
            "min_h": float(min_h),
            "min_center_distance": float(min_center_distance),
            "min_required_distance": float(min_required_distance),
            "eps_side": float(eps_side),
            "k0": float(k0),
            "k1": float(k1),
            "projection_solver": str(projection_solver),
            "qp_success": bool(qp_success),
            "fallback_used": fallback_used,
            "qp_error": qp_error,
            "soft_qp_success": bool(fallback_info.get("soft_qp_success", False)),
            "soft_qp_used": bool(fallback_info.get("soft_qp_used", False)),
            "soft_qp_error": str(fallback_info.get("soft_qp_error", "")),
            "soft_qp_slack_l2": float(fallback_info.get("soft_qp_slack_l2", 0.0)),
            "soft_qp_slack_max": float(fallback_info.get("soft_qp_slack_max", 0.0)),
            "fallback_source": str(fallback_info.get("fallback_source", "none")),
            "linprog_fallback_success": bool(fallback_info.get("linprog_success", False)),
            "linprog_fallback_error": str(fallback_info.get("linprog_error", "")),
            "continuous_fallback_success": bool(fallback_info.get("continuous_success", False)),
            "continuous_fallback_error": str(fallback_info.get("continuous_error", "")),
            "fallback_max_constraint_violation": float(
                fallback_info.get("max_constraint_violation", np.max(safe_constraint_values) if len(safe_constraint_values) else 0.0)
            ),
            "fallback_positive_violation_l2": float(fallback_info.get("positive_violation_l2", 0.0)),
            "num_neighbor_constraints": int(neighbor_constraints),
            "left_boundary_h": float(h_left),
            "right_boundary_h": float(h_right),
            "min_boundary_h": float(min(h_left, h_right)),
            "constraint_rows_physical": rows_arr.astype(np.float32),
            "constraint_bounds_physical": bounds_arr.astype(np.float32),
            "constraint_types": tuple(constraint_types),
            "constraint_neighbor_ranks": np.asarray(constraint_neighbor_ranks, dtype=np.int32),
            "tight_constraint_rows_physical": tight_rows.astype(np.float32),
            "tight_constraint_types": tuple(tight_types),
            "tight_constraint_indices": tight_indices.astype(np.int32),
            "active_constraint_rows_physical": active_physical.astype(np.float32),
            "active_constraint_types": tuple(reported_active_types),
            "active_constraint_indices": np.asarray(reported_active_indices, dtype=np.int32),
            "active_constraint_multipliers": reported_multipliers,
            "active_constraint_kkt_residual_norm": float(kkt_residual_norm),
            "active_constraint_selection": "positive_kkt_contribution",
            "active_constraint_rows_scaled": active_scaled.astype(np.float32),
            "cbf_active_constraint_rows_scaled": active_scaled.astype(np.float32),
            "projection_jacobian_scaled": projection,
            "cbf_projection_jacobian_scaled": projection,
            "cbf_active_constraint_count": int(active_scaled.shape[0]),
        }
        return np.asarray(a_safe, dtype=np.float32), info

    cbf_filter_2d_with_projection._guided_projection_reporting = True  # type: ignore[attr-defined]
    namespace["cbf_filter_2d"] = cbf_filter_2d_with_projection

    wrapper_cls = namespace.get("SafetyFilteredAccelerationWrapper")
    if wrapper_cls is None or getattr(wrapper_cls.step, "_guided_projection_reporting", False):
        return

    def step_with_projection_info(self, action):
        a_rl = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        ego = namespace["get_ego_state"](self)
        neighbors = namespace["get_neighbor_states"](self, neighbor_range=self.neighbor_range)
        road_width = float(namespace["_lane_free_base"](self).config["road_width"])
        filter_kwargs = {
            "ax_bounds": self.ax_bounds,
            "ay_bounds": self.ay_bounds,
            "eps_side": self.eps_side,
            "k0": self.k0,
            "k1": self.k1,
        }
        if hasattr(self, "max_neighbor_constraints"):
            filter_kwargs["max_neighbor_constraints"] = self.max_neighbor_constraints
        a_safe, filter_info = namespace["cbf_filter_2d"](a_rl, ego, neighbors, road_width, **filter_kwargs)
        normalized_action = namespace["_physical_to_normalized_action"](self, a_safe)
        obs, reward, terminated, truncated, info = self.env.step(normalized_action)
        correction_norm = float(filter_info["correction_norm"])
        correction_penalty = float(self.lambda_filter * correction_norm**2)
        reward = float(reward) - correction_penalty

        info = dict(info)
        info.update(
            {
                "cbf_a_rl_x": float(a_rl[0]),
                "cbf_a_rl_y": float(a_rl[1]),
                "cbf_a_safe_x": float(a_safe[0]),
                "cbf_a_safe_y": float(a_safe[1]),
                "cbf_correction_norm": correction_norm,
                "cbf_intervened": bool(correction_norm > 1e-6),
                "cbf_raw_feasible": bool(filter_info.get("raw_feasible", False)),
                "cbf_max_constraint_violation_rl": float(filter_info.get("max_constraint_violation_rl", 0.0)),
                "cbf_max_constraint_violation_safe": float(filter_info.get("max_constraint_violation_safe", 0.0)),
                "cbf_min_h": float(filter_info["min_h"]),
                "cbf_min_center_distance": float(filter_info["min_center_distance"]),
                "cbf_min_required_distance": float(filter_info["min_required_distance"]),
                "cbf_eps_side": float(filter_info["eps_side"]),
                "cbf_k0": float(filter_info.get("k0", self.k0)),
                "cbf_k1": float(filter_info.get("k1", self.k1)),
                "cbf_projection_solver": str(filter_info.get("projection_solver", "osqp")),
                "cbf_qp_success": bool(filter_info["qp_success"]),
                "cbf_fallback_used": bool(filter_info.get("fallback_used", not filter_info["qp_success"])),
                "cbf_qp_error": str(filter_info["qp_error"]),
                "cbf_soft_qp_success": bool(filter_info.get("soft_qp_success", False)),
                "cbf_soft_qp_used": bool(filter_info.get("soft_qp_used", False)),
                "cbf_soft_qp_error": str(filter_info.get("soft_qp_error", "")),
                "cbf_soft_qp_slack_l2": float(filter_info.get("soft_qp_slack_l2", 0.0)),
                "cbf_soft_qp_slack_max": float(filter_info.get("soft_qp_slack_max", 0.0)),
                "cbf_fallback_source": str(filter_info.get("fallback_source", "none")),
                "cbf_linprog_fallback_success": bool(filter_info.get("linprog_fallback_success", False)),
                "cbf_linprog_fallback_error": str(filter_info.get("linprog_fallback_error", "")),
                "cbf_continuous_fallback_success": bool(filter_info.get("continuous_fallback_success", False)),
                "cbf_continuous_fallback_error": str(filter_info.get("continuous_fallback_error", "")),
                "cbf_fallback_max_constraint_violation": float(
                    filter_info.get("fallback_max_constraint_violation", filter_info.get("max_constraint_violation_safe", 0.0))
                ),
                "cbf_fallback_positive_violation_l2": float(filter_info.get("fallback_positive_violation_l2", 0.0)),
                "cbf_num_neighbor_constraints": int(filter_info["num_neighbor_constraints"]),
                "cbf_min_boundary_h": float(filter_info["min_boundary_h"]),
                "cbf_left_boundary_h": float(filter_info["left_boundary_h"]),
                "cbf_right_boundary_h": float(filter_info["right_boundary_h"]),
                "cbf_filter_reward_penalty": correction_penalty,
                "cbf_active_constraint_count": int(filter_info.get("cbf_active_constraint_count", 0)),
            }
        )
        for key in (
            "constraint_rows_physical",
            "constraint_bounds_physical",
            "constraint_types",
            "constraint_neighbor_ranks",
            "tight_constraint_rows_physical",
            "tight_constraint_types",
            "tight_constraint_indices",
            "active_constraint_rows_physical",
            "active_constraint_types",
            "active_constraint_indices",
            "active_constraint_multipliers",
            "active_constraint_kkt_residual_norm",
            "active_constraint_selection",
            "active_constraint_rows_scaled",
            "cbf_active_constraint_rows_scaled",
            "projection_jacobian_scaled",
            "cbf_projection_jacobian_scaled",
        ):
            if key in filter_info:
                info[key] = filter_info[key]
        return obs, reward, terminated, truncated, info

    step_with_projection_info._guided_projection_reporting = True  # type: ignore[attr-defined]
    wrapper_cls.step = step_with_projection_info


class GuidedCBFDDPG(DDPG):
    """DDPG with standard critic loss and a minimal CBF safe-action actor term."""

    def __init__(
        self,
        *args,
        lambda_bc: float = 0.10,
        lambda_bc_final: float | None = None,
        lambda_bc_decay_steps: int = 0,
        bc_delta: float = 0.03,
        bc_action_scale: float = 1.0,
        bc_weight_max: float = 5.0,
        bc_loss_mode: str = "weighted_replay",
        use_projected_q: bool = True,
        projected_q_weight: float = 0.0,
        critic_action_mode: str = "raw",
        actor_action_mode: str = "raw",
        cbf_projection_steps: int = 2,
        cbf_projection_fd_step: float = 1e-3,
        cbf_road_width: float = 10.2,
        cbf_sensing_range: float = 90.0,
        cbf_obs_vmax: float = 24.0,
        cbf_obs_vymax: float = 7.2,
        cbf_eps_side: float = 0.10,
        cbf_k0: float = 5.29,
        cbf_k1: float = 3.68,
        **kwargs,
    ) -> None:
        if kwargs.get("replay_buffer_class") is None:
            kwargs["replay_buffer_class"] = CBFGuidedReplayBuffer
        self.lambda_bc = float(lambda_bc)
        self.lambda_bc_initial = float(lambda_bc)
        self.lambda_bc_final = None if lambda_bc_final is None else float(lambda_bc_final)
        self.lambda_bc_decay_steps = int(max(lambda_bc_decay_steps, 0))
        self.bc_delta = float(bc_delta)
        self.bc_action_scale = float(max(bc_action_scale, 1e-6))
        self.bc_weight_max = float(bc_weight_max)
        self.bc_loss_mode = str(bc_loss_mode).strip().lower()
        self.use_projected_q = bool(use_projected_q)
        self.projected_q_weight = float(np.clip(projected_q_weight, 0.0, 1.0))
        self.critic_action_mode = str(critic_action_mode).strip().lower()
        self.actor_action_mode = str(actor_action_mode).strip().lower()
        if self.critic_action_mode not in {"raw", "safe"}:
            raise ValueError(f"critic_action_mode must be 'raw' or 'safe', got {critic_action_mode!r}")
        if self.actor_action_mode not in {"raw", "diff_cbf"}:
            raise ValueError(f"actor_action_mode must be 'raw' or 'diff_cbf', got {actor_action_mode!r}")
        if self.bc_loss_mode not in {"weighted_replay", "masked_replay_mse", "local_projection_mse"}:
            raise ValueError(
                "bc_loss_mode must be 'weighted_replay', 'masked_replay_mse', "
                "or 'local_projection_mse', "
                f"got {bc_loss_mode!r}"
            )
        self.cbf_projection_steps = int(max(cbf_projection_steps, 1))
        self.cbf_projection_fd_step = float(max(cbf_projection_fd_step, 1e-5))
        self.cbf_road_width = float(cbf_road_width)
        self.cbf_sensing_range = float(cbf_sensing_range)
        self.cbf_obs_vmax = float(cbf_obs_vmax)
        self.cbf_obs_vymax = float(cbf_obs_vymax)
        self.cbf_eps_side = float(cbf_eps_side)
        self.cbf_k0 = float(cbf_k0)
        self.cbf_k1 = float(cbf_k1)
        super().__init__(*args, **kwargs)

    def _action_bounds_tensors(self, action: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        low = th.as_tensor(self.action_space.low, device=action.device, dtype=action.dtype).reshape(1, -1)[:, : action.shape[1]]
        high = th.as_tensor(self.action_space.high, device=action.device, dtype=action.dtype).reshape(1, -1)[:, : action.shape[1]]
        return low, high

    def _actor_to_physical_action(self, action_scaled: th.Tensor) -> th.Tensor:
        low, high = self._action_bounds_tensors(action_scaled)
        return low + 0.5 * (action_scaled + 1.0) * (high - low)

    def _physical_to_actor_action(self, action_phys: th.Tensor) -> th.Tensor:
        low, high = self._action_bounds_tensors(action_phys)
        return (2.0 * (action_phys - low) / th.clamp(high - low, min=1e-6) - 1.0).clamp(-1.0, 1.0)

    def _ellipse_clearance_from_obs(
        self,
        px: th.Tensor,
        py: th.Tensor,
        ego_length: th.Tensor,
        ego_width: th.Tensor,
        other_length: th.Tensor,
        other_width: th.Tensor,
    ) -> th.Tensor:
        eps = th.as_tensor(1e-6, device=px.device, dtype=px.dtype)
        radius = th.sqrt(px.square() + py.square() + eps)
        phi = th.atan2(py, px)

        def inflated_radius(length: th.Tensor, width: th.Tensor) -> th.Tensor:
            a = length / np.sqrt(2.0) + 2.0 * self.cbf_eps_side
            b = width / np.sqrt(2.0) + 2.0 * self.cbf_eps_side
            cos_phi = th.cos(phi)
            sin_phi = th.sin(phi)
            denom = th.sqrt((b * cos_phi).square() + (a * sin_phi).square() + eps)
            return a * b / th.clamp(denom, min=1e-6)

        required = inflated_radius(ego_length, ego_width) + inflated_radius(other_length, other_width)
        return radius - required

    def _neighbor_constraint_from_obs(
        self,
        obs_rows: th.Tensor,
        neighbor_index: int,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        ego = obs_rows[:, 0, :]
        other = obs_rows[:, neighbor_index, :]
        dtype = obs_rows.dtype
        device = obs_rows.device
        fd = th.as_tensor(self.cbf_projection_fd_step, device=device, dtype=dtype)

        ego_y = 0.5 * self.cbf_road_width * (ego[:, 0] + 1.0)
        ego_vx = ego[:, 2] * self.cbf_obs_vmax
        ego_vy = ego[:, 3] * self.cbf_obs_vymax
        ego_length = th.clamp(ego[:, 4] * 5.15, min=1e-3)
        ego_width = th.clamp(ego[:, 5] * 1.84, min=1e-3)

        dx = other[:, 0] * self.cbf_sensing_range
        dy = other[:, 1] * self.cbf_road_width
        other_vx = other[:, 2] * self.cbf_obs_vmax
        other_vy = other[:, 3] * self.cbf_obs_vymax
        other_length = th.clamp(other[:, 4] * 5.15, min=1e-3)
        other_width = th.clamp(other[:, 5] * 1.84, min=1e-3)
        valid = (other[:, 4] > 1e-4) & (other[:, 5] > 1e-4)

        def h_at(x: th.Tensor, y: th.Tensor) -> th.Tensor:
            return self._ellipse_clearance_from_obs(x, y, ego_length, ego_width, other_length, other_width)

        h0 = h_at(dx, dy)
        h_px = h_at(dx + fd, dy)
        h_mx = h_at(dx - fd, dy)
        h_py = h_at(dx, dy + fd)
        h_my = h_at(dx, dy - fd)
        h_pp = h_at(dx + fd, dy + fd)
        h_pm = h_at(dx + fd, dy - fd)
        h_mp = h_at(dx - fd, dy + fd)
        h_mm = h_at(dx - fd, dy - fd)

        grad_x = (h_px - h_mx) / (2.0 * fd)
        grad_y = (h_py - h_my) / (2.0 * fd)
        h_xx = (h_px - 2.0 * h0 + h_mx) / fd.square()
        h_yy = (h_py - 2.0 * h0 + h_my) / fd.square()
        h_xy = (h_pp - h_pm - h_mp + h_mm) / (4.0 * fd.square())

        dvx = other_vx - ego_vx
        dvy = other_vy - ego_vy
        h_dot = grad_x * dvx + grad_y * dvy
        v_h_v = h_xx * dvx.square() + 2.0 * h_xy * dvx * dvy + h_yy * dvy.square()
        bound = v_h_v + self.cbf_k1 * h_dot + self.cbf_k0 * h0
        row = th.stack([grad_x, grad_y], dim=1)
        row = th.where(valid.unsqueeze(1), row, th.zeros_like(row))
        bound = th.where(valid, bound, th.full_like(bound, 1e6))
        return row.detach(), bound.detach().unsqueeze(1), valid.detach().unsqueeze(1)

    def _constraints_from_observation(self, observations: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        obs = observations[:, :42]
        rows = obs.reshape(obs.shape[0], 6, 7)
        constraints: list[th.Tensor] = []
        bounds: list[th.Tensor] = []
        masks: list[th.Tensor] = []

        for neighbor_index in range(1, rows.shape[1]):
            row, bound, mask = self._neighbor_constraint_from_obs(rows, neighbor_index)
            constraints.append(row)
            bounds.append(bound)
            masks.append(mask)

        ego = rows[:, 0, :]
        ego_y = 0.5 * self.cbf_road_width * (ego[:, 0] + 1.0)
        ego_vy = ego[:, 3] * self.cbf_obs_vymax
        ego_width = th.clamp(ego[:, 5] * 1.84, min=1e-3)
        h_left = ego_y - 0.5 * ego_width
        h_right = self.cbf_road_width - 0.5 * ego_width - ego_y
        left_row = th.zeros((observations.shape[0], 2), device=observations.device, dtype=observations.dtype)
        right_row = th.zeros_like(left_row)
        left_row[:, 1] = -1.0
        right_row[:, 1] = 1.0
        constraints.extend([left_row.detach(), right_row.detach()])
        bounds.extend(
            [
                (self.cbf_k1 * ego_vy + self.cbf_k0 * h_left).detach().unsqueeze(1),
                (-self.cbf_k1 * ego_vy + self.cbf_k0 * h_right).detach().unsqueeze(1),
            ]
        )
        valid_boundary = th.ones((observations.shape[0], 1), device=observations.device, dtype=observations.dtype)
        masks.extend([valid_boundary, valid_boundary])

        return th.stack(constraints, dim=1), th.cat(bounds, dim=1), th.cat(masks, dim=1)

    def _diff_cbf_project_from_obs(self, observations: th.Tensor, action_scaled: th.Tensor) -> tuple[th.Tensor, dict[str, float]]:
        action_phys = self._actor_to_physical_action(action_scaled)
        low, high = self._action_bounds_tensors(action_phys)
        safe_phys = action_phys.clamp(low, high)
        constraint_rows, constraint_bounds, constraint_masks = self._constraints_from_observation(observations)

        for _ in range(self.cbf_projection_steps):
            for constraint_index in range(constraint_rows.shape[1]):
                row = constraint_rows[:, constraint_index, :]
                bound = constraint_bounds[:, constraint_index].unsqueeze(1)
                mask = constraint_masks[:, constraint_index].unsqueeze(1)
                violation = th.relu((row * safe_phys).sum(dim=1, keepdim=True) - bound) * mask
                denom = row.square().sum(dim=1, keepdim=True).clamp(min=1e-6)
                safe_phys = (safe_phys - violation * row / denom).clamp(low, high)

        safe_scaled = self._physical_to_actor_action(safe_phys)
        correction = th.norm(safe_scaled - action_scaled, dim=1)
        with th.no_grad():
            constraint_values = (constraint_rows * safe_phys.unsqueeze(1)).sum(dim=2) - constraint_bounds
            masked_values = th.where(constraint_masks > 0.5, constraint_values, th.full_like(constraint_values, -1e6))
            max_violation = th.relu(masked_values.max(dim=1).values)
        stats = {
            "diff_cbf_correction": float(correction.detach().mean().cpu().item()),
            "diff_cbf_active_rate": float((correction.detach() > 1e-6).float().mean().cpu().item()),
            "diff_cbf_max_violation": float(max_violation.detach().mean().cpu().item()),
        }
        return safe_scaled, stats

    def _current_lambda_bc(self) -> float:
        lambda_final = getattr(self, "lambda_bc_final", None)
        decay_steps = int(max(getattr(self, "lambda_bc_decay_steps", 0), 0))
        lambda_initial = float(getattr(self, "lambda_bc_initial", getattr(self, "lambda_bc", 0.0)))
        if lambda_final is None or decay_steps <= 0:
            return float(getattr(self, "lambda_bc", lambda_initial))
        progress = float(np.clip(float(getattr(self, "num_timesteps", 0)) / max(float(decay_steps), 1.0), 0.0, 1.0))
        return float(lambda_initial + progress * (float(lambda_final) - lambda_initial))

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, actor_rl_losses, bc_losses, critic_losses = [], [], [], []
        actor_raw_q_losses, actor_projected_q_losses = [], []
        bc_mask_rates, bc_weight_means = [], []
        actor_g_q_norms, actor_g_cbf_norms = [], []
        actor_g_cbf_to_g_q_ratios, actor_g_q_g_cbf_cosines = [], []
        actor_g_q_g_cbf_cosine_validities = []
        projection_trace_means, projection_active_rates, projected_action_gaps = [], [], []
        diff_cbf_corrections, diff_cbf_active_rates, diff_cbf_max_violations = [], [], []
        critic_raw_q_means, critic_safe_q_means = [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = getattr(replay_data, "discounts", None)
            if discounts is None:
                discounts = self.gamma

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            critic_actions = replay_data.safe_actions if self.critic_action_mode == "safe" else replay_data.actions
            current_q_values = self.critic(replay_data.observations, critic_actions)
            with th.no_grad():
                critic_raw_q_means.append(self.critic.q1_forward(replay_data.observations, replay_data.actions).mean().item())
                critic_safe_q_means.append(self.critic.q1_forward(replay_data.observations, replay_data.safe_actions).mean().item())
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                a_pred = self.actor(replay_data.observations)
                actor_q_action = a_pred
                diff_cbf_actor_update = self.actor_action_mode == "diff_cbf"
                use_projected_update = (
                    self.use_projected_q
                    and self.projected_q_weight > 0.0
                    and hasattr(replay_data, "projection_jacobians")
                    and not diff_cbf_actor_update
                )
                if diff_cbf_actor_update:
                    actor_q_action, diff_stats = self._diff_cbf_project_from_obs(replay_data.observations, a_pred)
                    diff_cbf_corrections.append(diff_stats["diff_cbf_correction"])
                    diff_cbf_active_rates.append(diff_stats["diff_cbf_active_rate"])
                    diff_cbf_max_violations.append(diff_stats["diff_cbf_max_violation"])
                if use_projected_update:
                    projection_jacobians = replay_data.projection_jacobians.detach()
                    local_delta = (a_pred - replay_data.actions).unsqueeze(-1)
                    actor_q_action = replay_data.safe_actions.detach() + th.bmm(projection_jacobians, local_delta).squeeze(-1)
                    actor_q_action = actor_q_action.clamp(-1.0, 1.0)
                    projection_identity = th.eye(
                        actor_q_action.shape[1],
                        device=actor_q_action.device,
                        dtype=actor_q_action.dtype,
                    ).unsqueeze(0)
                    projection_delta = projection_jacobians - projection_identity
                    projection_active = th.norm(projection_delta, dim=(1, 2)) > 1e-6
                    projection_active_rates.append(projection_active.float().mean().item())
                    projection_trace_means.append(th.diagonal(projection_jacobians, dim1=1, dim2=2).sum(dim=1).mean().item())
                    projected_action_gaps.append(th.norm(actor_q_action - a_pred, dim=1).mean().item())

                raw_q_actor_loss = -self.critic.q1_forward(replay_data.observations, a_pred).mean()
                projected_q_actor_loss = raw_q_actor_loss
                if diff_cbf_actor_update:
                    projected_q_actor_loss = -self.critic.q1_forward(replay_data.observations, actor_q_action).mean()
                    rl_actor_loss = projected_q_actor_loss
                elif use_projected_update:
                    projected_q_actor_loss = -self.critic.q1_forward(replay_data.observations, actor_q_action).mean()
                    rl_actor_loss = (
                        (1.0 - self.projected_q_weight) * raw_q_actor_loss
                        + self.projected_q_weight * projected_q_actor_loss
                    )
                else:
                    rl_actor_loss = raw_q_actor_loss

                if diff_cbf_actor_update:
                    correction = th.norm(actor_q_action - a_pred, dim=1, keepdim=True)
                    mask_float = (correction > 1e-6).float()
                    weights = 1.0 + th.clamp(
                        correction / self.bc_action_scale,
                        min=0.0,
                        max=self.bc_weight_max,
                    )
                    bc_per_sample = ((a_pred - actor_q_action) ** 2).sum(dim=1, keepdim=True)
                    bc_loss = bc_per_sample.mean()
                else:
                    correction = th.norm(replay_data.safe_actions - replay_data.actions, dim=1, keepdim=True)
                    if self.bc_loss_mode == "local_projection_mse":
                        if not hasattr(replay_data, "projection_jacobians"):
                            raise RuntimeError(
                                "local_projection_mse requires projection_jacobians in the replay buffer"
                            )
                        local_target = _local_projection_target(
                            a_pred,
                            replay_data.actions,
                            replay_data.safe_actions,
                            replay_data.projection_jacobians,
                        )
                        bc_per_sample = ((a_pred - local_target) ** 2).sum(dim=1, keepdim=True)
                        # Preserve the intervention decision recorded at this
                        # state; do not relabel it using the current actor.
                        mask_float = (replay_data.interventions > 0.5).float()
                        weights = th.ones_like(correction)
                    elif self.bc_loss_mode == "masked_replay_mse":
                        bc_per_sample = ((a_pred - replay_data.safe_actions) ** 2).sum(dim=1, keepdim=True)
                        # Exact replay-target form of L_corr: only meaningful CBF
                        # corrections supervise the actor, with no extra magnitude weighting.
                        mask_float = (correction > self.bc_delta).float()
                        weights = th.ones_like(correction)
                    else:
                        bc_per_sample = ((a_pred - replay_data.safe_actions) ** 2).sum(dim=1, keepdim=True)
                        mask_float = (replay_data.interventions > 0.5).float()
                        weights = 1.0 + th.clamp(
                            correction / self.bc_action_scale,
                            min=0.0,
                            max=self.bc_weight_max,
                        )
                    bc_loss = (mask_float * weights * bc_per_sample).sum() / (mask_float.sum() + 1e-6)
                lambda_bc_current = self._current_lambda_bc()
                weighted_cbf_loss = lambda_bc_current * bc_loss
                actor_loss = rl_actor_loss + weighted_cbf_loss
                gradient_diagnostics = _actor_gradient_diagnostics(
                    self.actor,
                    rl_actor_loss,
                    weighted_cbf_loss,
                )

                actor_losses.append(actor_loss.item())
                actor_rl_losses.append(rl_actor_loss.item())
                actor_raw_q_losses.append(raw_q_actor_loss.item())
                actor_projected_q_losses.append(projected_q_actor_loss.item())
                bc_losses.append(bc_loss.item())
                bc_mask_rates.append(mask_float.mean().item())
                actor_g_q_norms.append(gradient_diagnostics["g_q_norm"])
                actor_g_cbf_norms.append(gradient_diagnostics["g_cbf_norm"])
                actor_g_cbf_to_g_q_ratios.append(gradient_diagnostics["g_cbf_to_g_q_ratio"])
                actor_g_q_g_cbf_cosines.append(gradient_diagnostics["g_q_g_cbf_cosine"])
                actor_g_q_g_cbf_cosine_validities.append(
                    gradient_diagnostics["g_q_g_cbf_cosine_valid"]
                )
                if mask_float.sum().item() > 0.0:
                    bc_weight_means.append((mask_float * weights).sum().item() / (mask_float.sum().item() + 1e-6))
                else:
                    bc_weight_means.append(0.0)

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/actor_rl_loss", np.mean(actor_rl_losses))
            self.logger.record("train/actor_raw_q_loss", np.mean(actor_raw_q_losses))
            self.logger.record("train/actor_projected_q_loss", np.mean(actor_projected_q_losses))
            self.logger.record("train/cbf_critic_action_mode_safe", float(self.critic_action_mode == "safe"))
            self.logger.record("train/cbf_actor_action_mode_diff", float(self.actor_action_mode == "diff_cbf"))
            self.logger.record("train/cbf_critic_q_raw_mean", np.mean(critic_raw_q_means))
            self.logger.record("train/cbf_critic_q_safe_mean", np.mean(critic_safe_q_means))
            self.logger.record("train/cbf_projected_q_weight", self.projected_q_weight)
            self.logger.record("train/cbf_lambda_bc_current", self._current_lambda_bc())
            self.logger.record("train/cbf_bc_loss_mode_masked_replay", float(self.bc_loss_mode == "masked_replay_mse"))
            self.logger.record(
                "train/cbf_bc_loss_mode_local_projection",
                float(self.bc_loss_mode == "local_projection_mse"),
            )
            self.logger.record("train/cbf_bc_loss", np.mean(bc_losses))
            self.logger.record("train/cbf_bc_mask_rate", np.mean(bc_mask_rates))
            self.logger.record("train/cbf_bc_weight", np.mean(bc_weight_means))
            self.logger.record("train/actor_g_q_norm", np.mean(actor_g_q_norms))
            self.logger.record("train/actor_g_cbf_norm", np.mean(actor_g_cbf_norms))
            self.logger.record("train/actor_g_cbf_to_g_q_ratio", np.mean(actor_g_cbf_to_g_q_ratios))
            finite_cosines = [value for value in actor_g_q_g_cbf_cosines if np.isfinite(value)]
            self.logger.record(
                "train/actor_g_q_g_cbf_cosine",
                np.mean(finite_cosines) if finite_cosines else np.nan,
            )
            self.logger.record(
                "train/actor_g_q_g_cbf_cosine_valid_rate",
                np.mean(actor_g_q_g_cbf_cosine_validities),
            )
            if diff_cbf_corrections:
                self.logger.record("train/diff_cbf_correction", np.mean(diff_cbf_corrections))
                self.logger.record("train/diff_cbf_active_rate", np.mean(diff_cbf_active_rates))
                self.logger.record("train/diff_cbf_max_violation", np.mean(diff_cbf_max_violations))
            if projection_trace_means:
                self.logger.record("train/cbf_projection_trace", np.mean(projection_trace_means))
                self.logger.record("train/cbf_projection_active_rate", np.mean(projection_active_rates))
                self.logger.record("train/cbf_projected_action_gap", np.mean(projected_action_gaps))
        self.logger.record("train/critic_loss", np.mean(critic_losses))


def install_minimal_guided_cbf(namespace: dict[str, Any]) -> None:
    """Install minimal guided-CBF definitions into a notebook-derived namespace."""

    namespace.setdefault("CBF_SOFT_QP_ACTION_WEIGHT", 1.0)
    namespace.setdefault("CBF_SOFT_QP_SLACK_WEIGHT", 5_000.0)
    namespace.setdefault("CBF_SOFT_QP_SLACK_UPPER", 1_000.0)
    namespace.setdefault("GUIDED_CBF_LAMBDA_BC", 0.10)
    namespace.setdefault("GUIDED_CBF_BC_DELTA", 0.03)
    namespace.setdefault("GUIDED_CBF_ACTION_SCALE", 1.0)
    namespace.setdefault("GUIDED_CBF_WEIGHT_MAX", 5.0)
    namespace.setdefault("GUIDED_CBF_USE_PROJECTED_Q", True)
    namespace.setdefault("GUIDED_CBF_PROJECTED_Q_WEIGHT", 0.0)
    namespace.setdefault("GUIDED_CBF_CRITIC_ACTION_MODE", "raw")
    namespace.setdefault("GUIDED_CBF_ACTOR_ACTION_MODE", "raw")
    namespace.setdefault("GUIDED_CBF_DIFF_PROJECTION_STEPS", 2)
    namespace.setdefault("GUIDED_CBF_DIFF_PROJECTION_FD_STEP", 1e-3)
    _install_robust_cbf_fallback(namespace)
    namespace["install_cbf_projection_reporting"] = lambda: _install_cbf_projection_reporting(namespace)
    if bool(namespace.get("GUIDED_CBF_ENABLE_PROJECTION_REPORTING", False)):
        _install_cbf_projection_reporting(namespace)
    namespace.setdefault("GUIDED_DDPG_CBF_TOTAL_TIMESTEPS", namespace.get("DDPG_CBF_TOTAL_TIMESTEPS"))
    namespace.setdefault(
        "GUIDED_DDPG_CBF_MODEL_PATH",
        namespace["ARTIFACT_DIR"] / "guided_ddpg_cbf_flat42_vmax24_noslack_tuned_laneless_karalakou.zip",
    )
    namespace.setdefault(
        "GUIDED_DDPG_CBF_HISTORY_PATH",
        namespace["ARTIFACT_DIR"] / "guided_ddpg_cbf_flat42_vmax24_noslack_tuned_laneless_karalakou_eval_history.csv",
    )

    def make_guided_cbf_single_env(
        seed: int | None = None,
        render_mode: Optional[str] = None,
        lambda_filter: float | None = None,
        eps_side: float | None = None,
        env_config: Optional[dict[str, Any]] = None,
        reward_config: Optional[dict[str, float]] = None,
        normalize_observation: Optional[bool] = None,
    ) -> gym.Env:
        env = gym.make(
            "lane-free-v0",
            render_mode=render_mode,
            config=env_config or namespace["ENV_CONFIG"],
        )
        env = namespace["KaralakouRewardWrapper"](env, reward_config=reward_config or namespace["REWARD_CONFIG"])
        env = namespace["SafetyFilteredAccelerationWrapper"](
            env,
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"] if lambda_filter is None else lambda_filter,
            eps_side=namespace["CBF_EPS_SIDE"] if eps_side is None else eps_side,
            k0=namespace["CBF_K0"],
            k1=namespace["CBF_K1"],
        )
        normalize = namespace["NORMALIZE_RL_OBSERVATIONS"] if normalize_observation is None else normalize_observation
        if normalize:
            env = namespace["LaneFreeObservationNormalizationWrapper"](env, clip=namespace["OBSERVATION_CLIP"])
        env = Monitor(env)
        env.reset(seed=namespace["SEED"] if seed is None else seed)
        return env

    def make_guided_cbf_training_env(
        seed: int | None = None,
        lambda_filter: float | None = None,
        eps_side: float | None = None,
        env_config: Optional[dict[str, Any]] = None,
        reward_config: Optional[dict[str, float]] = None,
        normalize_observation: Optional[bool] = None,
        n_envs: int = 1,
        use_subproc: bool = False,
    ):
        def _single_env(env_seed: int) -> gym.Env:
            return make_guided_cbf_single_env(
                seed=env_seed,
                render_mode=None,
                lambda_filter=lambda_filter,
                eps_side=eps_side,
                env_config=env_config,
                reward_config=reward_config,
                normalize_observation=normalize_observation,
            )

        return namespace["_make_vectorized_env"](
            _single_env,
            seed=namespace["SEED"] if seed is None else seed,
            n_envs=n_envs,
            use_subproc=use_subproc,
            start_method=namespace["DDPG_SUBPROC_START_METHOD"],
        )

    namespace.update(
        {
            "CBFGuidedReplayBufferSamples": CBFGuidedReplayBufferSamples,
            "CBFGuidedReplayBuffer": CBFGuidedReplayBuffer,
            "GuidedCBFDDPG": GuidedCBFDDPG,
            "make_guided_cbf_single_env": make_guided_cbf_single_env,
            "make_guided_cbf_training_env": make_guided_cbf_training_env,
        }
    )
