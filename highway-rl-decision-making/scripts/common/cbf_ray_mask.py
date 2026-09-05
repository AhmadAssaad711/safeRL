"""CBF-derived continuous action ray mask for the laneless DDPG experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor


RAY_MASK_DIRECTION_EPS = 1e-8
RAY_MASK_CONSTRAINT_EPS = 1e-8
RAY_MASK_FEASIBILITY_TOL = 1e-5
RAY_MASK_BOUNDARY_SHRINK = 1e-6


def _as_action(action: Any) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32).reshape(-1)
    if array.size < 2:
        raise ValueError("action must contain two components")
    return array[:2].astype(np.float32)


def _global_action_from_latent(
    z: np.ndarray,
    ax_bounds: tuple[float, float],
    ay_bounds: tuple[float, float],
) -> np.ndarray:
    z = np.clip(_as_action(z), -1.0, 1.0)
    lows = np.asarray([ax_bounds[0], ay_bounds[0]], dtype=np.float32)
    highs = np.asarray([ax_bounds[1], ay_bounds[1]], dtype=np.float32)
    return (lows + 0.5 * (z + 1.0) * (highs - lows)).astype(np.float32)


def _constraint_arrays(rows: list[np.ndarray], bounds: list[float]) -> tuple[np.ndarray, np.ndarray]:
    if rows:
        return np.asarray(rows, dtype=float), np.asarray(bounds, dtype=float)
    return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float)


def _append_box_constraints(
    rows: list[np.ndarray],
    bounds: list[float],
    lb: np.ndarray,
    ub: np.ndarray,
) -> None:
    rows.extend(
        [
            np.asarray([1.0, 0.0], dtype=float),
            np.asarray([-1.0, 0.0], dtype=float),
            np.asarray([0.0, 1.0], dtype=float),
            np.asarray([0.0, -1.0], dtype=float),
        ]
    )
    bounds.extend([float(ub[0]), float(-lb[0]), float(ub[1]), float(-lb[1])])


def _max_violation(rows: np.ndarray, bounds: np.ndarray, action: np.ndarray) -> float:
    if rows.size == 0:
        return 0.0
    return float(np.max(rows @ np.asarray(action, dtype=float).reshape(2) - bounds))


def _is_feasible(rows: np.ndarray, bounds: np.ndarray, action: np.ndarray, tol: float = RAY_MASK_FEASIBILITY_TOL) -> bool:
    return _max_violation(rows, bounds, action) <= tol


def build_cbf_action_constraints(
    namespace: dict[str, Any],
    ego: dict[str, float],
    neighbors: list[dict[str, float]],
    road_width: float,
    ax_bounds: tuple[float, float],
    ay_bounds: tuple[float, float],
    eps_side: float,
    k0: float,
    k1: float,
    max_neighbor_constraints: Optional[int],
) -> dict[str, Any]:
    """Build the linear CBF action set rows @ a <= bounds, including box limits."""

    if max_neighbor_constraints is not None:
        neighbors = list(neighbors)[: int(max_neighbor_constraints)]

    constraint_rows: list[np.ndarray] = []
    constraint_bounds: list[float] = []
    min_h = np.inf
    min_center_distance = np.inf
    min_required_distance = np.inf
    neighbor_constraints = 0

    for neighbor in neighbors:
        other_acc = np.asarray([float(neighbor.get("ax", 0.0)), float(neighbor.get("ay", 0.0))], dtype=float)
        A, b, h_ij, center_distance, required_distance = namespace["pairwise_hocbf_constraint"](
            ego,
            neighbor,
            eps_side=eps_side,
            k0=k0,
            k1=k1,
            other_acc=other_acc,
        )
        constraint_rows.append(np.asarray(A, dtype=float))
        constraint_bounds.append(float(b))
        min_h = min(min_h, float(h_ij))
        min_center_distance = min(min_center_distance, float(center_distance))
        min_required_distance = min(min_required_distance, float(required_distance))
        neighbor_constraints += 1

    ego_y = float(ego["y"])
    ego_vy = float(ego["vy"])
    ego_half_width = 0.5 * float(ego["width"])
    h_left = ego_y - ego_half_width
    h_right = float(road_width) - ego_half_width - ego_y

    constraint_rows.append(np.asarray([0.0, -1.0], dtype=float))
    constraint_bounds.append(float(k1 * ego_vy + k0 * h_left))
    constraint_rows.append(np.asarray([0.0, 1.0], dtype=float))
    constraint_bounds.append(float(-k1 * ego_vy + k0 * h_right))

    lb = np.asarray([float(ax_bounds[0]), float(ay_bounds[0])], dtype=float)
    ub = np.asarray([float(ax_bounds[1]), float(ay_bounds[1])], dtype=float)

    all_rows = list(constraint_rows)
    all_bounds = list(constraint_bounds)
    _append_box_constraints(all_rows, all_bounds, lb, ub)
    rows, bounds = _constraint_arrays(all_rows, all_bounds)

    return {
        "rows": rows,
        "bounds": bounds,
        "cbf_rows": constraint_rows,
        "cbf_bounds": constraint_bounds,
        "lb": lb,
        "ub": ub,
        "min_h": np.nan if not np.isfinite(min_h) else float(min_h),
        "min_center_distance": np.nan if not np.isfinite(min_center_distance) else float(min_center_distance),
        "min_required_distance": np.nan if not np.isfinite(min_required_distance) else float(min_required_distance),
        "num_neighbor_constraints": int(neighbor_constraints),
        "left_boundary_h": float(h_left),
        "right_boundary_h": float(h_right),
        "min_boundary_h": float(min(h_left, h_right)),
    }


def _least_violating_center(
    namespace: dict[str, Any],
    target: np.ndarray,
    system: dict[str, Any],
    ax_bounds: tuple[float, float],
    ay_bounds: tuple[float, float],
) -> np.ndarray:
    if "_least_violating_bounded_action" in namespace:
        return namespace["_least_violating_bounded_action"](
            target,
            system["cbf_rows"],
            system["cbf_bounds"],
            ax_bounds,
            ay_bounds,
        ).astype(np.float32)

    rows, bounds = system["rows"], system["bounds"]
    lb, ub = system["lb"], system["ub"]
    ax_grid = np.unique(np.r_[np.linspace(lb[0], ub[0], 25), target[0], lb[0], ub[0]])
    ay_grid = np.unique(np.r_[np.linspace(lb[1], ub[1], 25), target[1], lb[1], ub[1]])
    best_action = np.clip(target, lb, ub)
    best_score: tuple[float, float, float] | None = None
    for ax in ax_grid:
        for ay in ay_grid:
            candidate = np.asarray([ax, ay], dtype=float)
            violations = np.maximum(rows @ candidate - bounds, 0.0)
            score = (
                float(np.max(violations)) if violations.size else 0.0,
                float(np.sum(violations**2)),
                float(np.sum((candidate - target) ** 2)),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_action = candidate
    return np.clip(best_action, lb, ub).astype(np.float32)


def choose_ray_center(
    namespace: dict[str, Any],
    system: dict[str, Any],
    previous_safe_action: Optional[np.ndarray],
    global_action: np.ndarray,
    ax_bounds: tuple[float, float],
    ay_bounds: tuple[float, float],
) -> tuple[np.ndarray, str, bool]:
    rows, bounds = system["rows"], system["bounds"]
    lb, ub = system["lb"], system["ub"]

    candidates: list[tuple[str, np.ndarray]] = []
    if previous_safe_action is not None:
        candidates.append(("previous_safe", np.clip(_as_action(previous_safe_action), lb, ub)))
    candidates.append(("zero", np.clip(np.zeros(2, dtype=np.float32), lb, ub)))
    candidates.append(("global_clipped", np.clip(_as_action(global_action), lb, ub)))

    for name, candidate in candidates:
        if _is_feasible(rows, bounds, candidate):
            return candidate.astype(np.float32), name, True

    least_violating = _least_violating_center(namespace, global_action, system, ax_bounds, ay_bounds)
    return least_violating.astype(np.float32), "least_violating", _is_feasible(rows, bounds, least_violating)


def ray_map_action(
    z: np.ndarray,
    center: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    z = np.clip(_as_action(z), -1.0, 1.0).astype(float)
    center = _as_action(center).astype(float)
    z_norm = float(np.linalg.norm(z))
    if z_norm <= RAY_MASK_DIRECTION_EPS:
        return center.astype(np.float32), {"rho": 0.0, "rho_max": 0.0, "ray_norm": z_norm}

    direction = z / z_norm
    rho = float(min(z_norm, 1.0))
    slack = bounds - rows @ center
    denom = rows @ direction
    active = denom > RAY_MASK_CONSTRAINT_EPS
    if not np.any(active):
        rho_max = 0.0
    else:
        steps = slack[active] / denom[active]
        rho_max = float(max(0.0, np.min(steps)))
    step = max(0.0, rho_max - RAY_MASK_BOUNDARY_SHRINK) * rho
    return (center + step * direction).astype(np.float32), {
        "rho": rho,
        "rho_max": rho_max,
        "ray_norm": z_norm,
    }


def cbf_ray_mask_filter_2d(
    namespace: dict[str, Any],
    z,
    ego: dict[str, float],
    neighbors: list[dict[str, float]],
    road_width: float,
    previous_safe_action: Optional[np.ndarray] = None,
    ax_bounds: tuple[float, float] | None = None,
    ay_bounds: tuple[float, float] | None = None,
    eps_side: float | None = None,
    k0: float | None = None,
    k1: float | None = None,
    max_neighbor_constraints: Optional[int] = None,
    backup_qp: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    ax_bounds = namespace["CBF_AX_BOUNDS"] if ax_bounds is None else ax_bounds
    ay_bounds = namespace["CBF_AY_BOUNDS"] if ay_bounds is None else ay_bounds
    eps_side = float(namespace["CBF_EPS_SIDE"] if eps_side is None else eps_side)
    k0 = float(namespace["CBF_K0"] if k0 is None else k0)
    k1 = float(namespace["CBF_K1"] if k1 is None else k1)
    if max_neighbor_constraints is None:
        max_neighbor_constraints = namespace.get("CBF_MAX_NEIGHBOR_CONSTRAINTS")

    z = np.clip(_as_action(z), -1.0, 1.0)
    global_action = _global_action_from_latent(z, ax_bounds, ay_bounds)
    system = build_cbf_action_constraints(
        namespace,
        ego,
        neighbors,
        road_width,
        ax_bounds,
        ay_bounds,
        eps_side,
        k0,
        k1,
        max_neighbor_constraints,
    )
    rows, bounds = system["rows"], system["bounds"]
    center, center_source, center_feasible = choose_ray_center(
        namespace,
        system,
        previous_safe_action,
        global_action,
        ax_bounds,
        ay_bounds,
    )

    qp_info: dict[str, Any] = {}
    backup_used = False
    if center_feasible:
        a_safe, ray_info = ray_map_action(z, center, rows, bounds)
        safe_violation = _max_violation(rows, bounds, a_safe)
    else:
        a_safe = center
        ray_info = {"rho": 0.0, "rho_max": 0.0, "ray_norm": float(np.linalg.norm(z))}
        safe_violation = _max_violation(rows, bounds, a_safe)

    if backup_qp and (not center_feasible or safe_violation > RAY_MASK_FEASIBILITY_TOL):
        backup_used = True
        a_safe, qp_info = namespace["cbf_filter_2d"](
            a_safe,
            ego,
            neighbors,
            road_width,
            ax_bounds=ax_bounds,
            ay_bounds=ay_bounds,
            eps_side=eps_side,
            k0=k0,
            k1=k1,
            max_neighbor_constraints=max_neighbor_constraints,
        )
        safe_violation = _max_violation(rows, bounds, a_safe)

    correction_norm = float(np.linalg.norm(a_safe - global_action))
    info = {
        "a_global": global_action.astype(np.float32),
        "a_safe": np.asarray(a_safe, dtype=np.float32),
        "correction_norm": correction_norm,
        "ray_center": center.astype(np.float32),
        "ray_center_source": center_source,
        "ray_center_feasible": bool(center_feasible),
        "ray_rho": float(ray_info["rho"]),
        "ray_rho_max": float(ray_info["rho_max"]),
        "ray_norm": float(ray_info["ray_norm"]),
        "ray_backup_qp_used": bool(backup_used),
        "ray_max_constraint_violation_global": _max_violation(rows, bounds, global_action),
        "max_constraint_violation_rl": _max_violation(rows, bounds, global_action),
        "max_constraint_violation_safe": float(safe_violation),
        "min_h": system["min_h"],
        "min_center_distance": system["min_center_distance"],
        "min_required_distance": system["min_required_distance"],
        "eps_side": float(eps_side),
        "k0": float(k0),
        "k1": float(k1),
        "qp_success": bool(qp_info.get("qp_success", True)),
        "fallback_used": bool(qp_info.get("fallback_used", False)),
        "qp_error": str(qp_info.get("qp_error", "")),
        "num_neighbor_constraints": int(system["num_neighbor_constraints"]),
        "left_boundary_h": float(system["left_boundary_h"]),
        "right_boundary_h": float(system["right_boundary_h"]),
        "min_boundary_h": float(system["min_boundary_h"]),
    }
    return np.asarray(a_safe, dtype=np.float32), info


class RayMaskedSafetyFilteredAccelerationWrapper(gym.Wrapper):
    """State-dependent CBF ray mask from latent actions to safe physical acceleration."""

    def __init__(
        self,
        env: gym.Env,
        namespace: dict[str, Any],
        lambda_filter: float,
        neighbor_range: float,
        ax_bounds: tuple[float, float],
        ay_bounds: tuple[float, float],
        eps_side: float,
        k0: float,
        k1: float,
        max_neighbor_constraints: Optional[int],
        backup_qp: bool = True,
    ) -> None:
        super().__init__(env)
        self.namespace = namespace
        self.lambda_filter = float(lambda_filter)
        self.neighbor_range = float(neighbor_range)
        self.ax_bounds = ax_bounds
        self.ay_bounds = ay_bounds
        self.eps_side = float(eps_side)
        self.k0 = float(k0)
        self.k1 = float(k1)
        self.max_neighbor_constraints = max_neighbor_constraints
        self.backup_qp = bool(backup_qp)
        self._last_ray_safe_action: Optional[np.ndarray] = None
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        self._last_ray_safe_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        z = np.clip(_as_action(action), -1.0, 1.0)
        ego = self.namespace["get_ego_state"](self)
        neighbors = self.namespace["get_neighbor_states"](self, neighbor_range=self.neighbor_range)
        road_width = float(self.namespace["_lane_free_base"](self).config["road_width"])
        a_safe, filter_info = cbf_ray_mask_filter_2d(
            self.namespace,
            z,
            ego,
            neighbors,
            road_width,
            previous_safe_action=self._last_ray_safe_action,
            ax_bounds=self.ax_bounds,
            ay_bounds=self.ay_bounds,
            eps_side=self.eps_side,
            k0=self.k0,
            k1=self.k1,
            max_neighbor_constraints=self.max_neighbor_constraints,
            backup_qp=self.backup_qp,
        )
        self._last_ray_safe_action = np.asarray(a_safe, dtype=np.float32)
        normalized_action = self.namespace["_physical_to_normalized_action"](self, a_safe)
        obs, reward, terminated, truncated, info = self.env.step(normalized_action)
        correction_penalty = self.lambda_filter * float(filter_info["correction_norm"]) ** 2
        reward = float(reward) - correction_penalty

        global_action = np.asarray(filter_info["a_global"], dtype=np.float32)
        ray_center = np.asarray(filter_info["ray_center"], dtype=np.float32)
        info = dict(info)
        info.update(
            {
                "cbf_a_rl_x": float(global_action[0]),
                "cbf_a_rl_y": float(global_action[1]),
                "cbf_a_safe_x": float(a_safe[0]),
                "cbf_a_safe_y": float(a_safe[1]),
                "cbf_ray_z_x": float(z[0]),
                "cbf_ray_z_y": float(z[1]),
                "cbf_ray_center_x": float(ray_center[0]),
                "cbf_ray_center_y": float(ray_center[1]),
                "cbf_ray_center_source": str(filter_info["ray_center_source"]),
                "cbf_ray_center_feasible": bool(filter_info["ray_center_feasible"]),
                "cbf_ray_rho": float(filter_info["ray_rho"]),
                "cbf_ray_rho_max": float(filter_info["ray_rho_max"]),
                "cbf_ray_norm": float(filter_info["ray_norm"]),
                "cbf_ray_backup_qp_used": bool(filter_info["ray_backup_qp_used"]),
                "cbf_correction_norm": float(filter_info["correction_norm"]),
                "cbf_intervened": bool(filter_info["correction_norm"] > 1e-6 or filter_info["ray_backup_qp_used"]),
                "cbf_max_constraint_violation_rl": float(filter_info["max_constraint_violation_rl"]),
                "cbf_max_constraint_violation_safe": float(filter_info["max_constraint_violation_safe"]),
                "cbf_ray_max_constraint_violation_global": float(filter_info["ray_max_constraint_violation_global"]),
                "cbf_min_h": float(filter_info["min_h"]),
                "cbf_min_center_distance": float(filter_info["min_center_distance"]),
                "cbf_min_required_distance": float(filter_info["min_required_distance"]),
                "cbf_eps_side": float(filter_info["eps_side"]),
                "cbf_k0": float(filter_info["k0"]),
                "cbf_k1": float(filter_info["k1"]),
                "cbf_qp_success": bool(filter_info["qp_success"]),
                "cbf_fallback_used": bool(filter_info["fallback_used"]),
                "cbf_qp_error": str(filter_info["qp_error"]),
                "cbf_num_neighbor_constraints": int(filter_info["num_neighbor_constraints"]),
                "cbf_min_boundary_h": float(filter_info["min_boundary_h"]),
                "cbf_left_boundary_h": float(filter_info["left_boundary_h"]),
                "cbf_right_boundary_h": float(filter_info["right_boundary_h"]),
                "cbf_filter_reward_penalty": float(correction_penalty),
            }
        )
        return obs, reward, terminated, truncated, info


def install_ray_mask_cbf(namespace: dict[str, Any]) -> None:
    artifact_dir = Path(namespace["ARTIFACT_DIR"])
    namespace.setdefault("DDPG_CBF_RAY_MASK_TOTAL_TIMESTEPS", namespace.get("DDPG_CBF_TOTAL_TIMESTEPS", 50_000))
    namespace.setdefault(
        "DDPG_CBF_RAY_MASK_MODEL_PATH",
        artifact_dir / "ddpg_cbf_ray_mask_flat42_vmax24_noslack_tuned_laneless_karalakou.zip",
    )
    namespace.setdefault(
        "DDPG_CBF_RAY_MASK_HISTORY_PATH",
        artifact_dir / "ddpg_cbf_ray_mask_flat42_vmax24_noslack_tuned_laneless_karalakou_eval_history.csv",
    )

    def make_ray_mask_single_env(
        seed: int = None,
        render_mode: Optional[str] = None,
        lambda_filter: Optional[float] = None,
        eps_side: Optional[float] = None,
        env_config: Optional[dict[str, Any]] = None,
        reward_config: Optional[dict[str, float]] = None,
        normalize_observation: Optional[bool] = None,
        backup_qp: bool = True,
    ) -> gym.Env:
        env = gym.make("lane-free-v0", render_mode=render_mode, config=env_config or namespace["ENV_CONFIG"])
        env = namespace["KaralakouRewardWrapper"](env, reward_config=reward_config or namespace["REWARD_CONFIG"])
        env = RayMaskedSafetyFilteredAccelerationWrapper(
            env,
            namespace=namespace,
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"] if lambda_filter is None else lambda_filter,
            neighbor_range=namespace["CBF_NEIGHBOR_RANGE"],
            ax_bounds=namespace["CBF_AX_BOUNDS"],
            ay_bounds=namespace["CBF_AY_BOUNDS"],
            eps_side=namespace["CBF_EPS_SIDE"] if eps_side is None else eps_side,
            k0=namespace["CBF_K0"],
            k1=namespace["CBF_K1"],
            max_neighbor_constraints=namespace.get("CBF_MAX_NEIGHBOR_CONSTRAINTS"),
            backup_qp=backup_qp,
        )
        normalize = namespace["NORMALIZE_RL_OBSERVATIONS"] if normalize_observation is None else normalize_observation
        if normalize:
            env = namespace["LaneFreeObservationNormalizationWrapper"](env, clip=namespace["OBSERVATION_CLIP"])
        env = Monitor(env)
        env.reset(seed=namespace["SEED"] if seed is None else seed)
        return env

    def make_ray_mask_training_env(
        seed: int = None,
        lambda_filter: Optional[float] = None,
        eps_side: Optional[float] = None,
        env_config: Optional[dict[str, Any]] = None,
        reward_config: Optional[dict[str, float]] = None,
        normalize_observation: Optional[bool] = None,
        n_envs: int = 1,
        use_subproc: bool = False,
        backup_qp: bool = True,
    ):
        def _single_env(env_seed: int) -> gym.Env:
            return make_ray_mask_single_env(
                seed=env_seed,
                render_mode=None,
                lambda_filter=lambda_filter,
                eps_side=eps_side,
                env_config=env_config,
                reward_config=reward_config,
                normalize_observation=normalize_observation,
                backup_qp=backup_qp,
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
            "build_cbf_action_constraints": lambda *args, **kwargs: build_cbf_action_constraints(namespace, *args, **kwargs),
            "cbf_ray_mask_filter_2d": lambda *args, **kwargs: cbf_ray_mask_filter_2d(namespace, *args, **kwargs),
            "RayMaskedSafetyFilteredAccelerationWrapper": RayMaskedSafetyFilteredAccelerationWrapper,
            "make_ray_mask_single_env": make_ray_mask_single_env,
            "make_ray_mask_training_env": make_ray_mask_training_env,
        }
    )
