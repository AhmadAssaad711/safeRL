"""PPO environment adapters for the staged CBF experiments.

All new PPO variants use one common *physical* action interface.  This avoids
the legacy confound where nominal PPO used normalized actions while a CBF
trained policy used a different physical Box.  The wrapper also appends the
exact simulator-state CBF constraint context to each observation.  Actor and
value networks are configured to consume only the original 42 state features;
the appended context is retained for the optimization/action layer and PPO
minibatch recomputation.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from scripts.common.cbf_projection import (
    CBFContextLayout,
    NumpyProjection2D,
    append_cbf_context,
    project_polytope_2d_numpy,
)
from scripts.common.cbf_ray_mask import build_cbf_action_constraints


def constraint_system_hash(rows: np.ndarray, bounds: np.ndarray) -> str:
    """Stable short hash used to verify pre-state/execution context parity."""

    digest = hashlib.sha256()
    # Context is persisted in the float32 observation buffer.  Hash that exact
    # representation so a pre-state round trip does not create a false drift.
    digest.update(np.asarray(rows, dtype=np.float32).tobytes(order="C"))
    digest.update(np.asarray(bounds, dtype=np.float32).tobytes(order="C"))
    return digest.hexdigest()[:16]


class CBFContextPhysicalActionWrapper(gym.Wrapper):
    """Expose common physical actions and exact padded CBF context.

    During projected-PPO collection, the algorithm computes ``P_s(z)`` from
    the stored observation context and calls :meth:`set_projection_record`
    before stepping this wrapper with the feasible action.  This preserves the
    Gym action-space contract: the simulator receives only an in-Box action,
    while the rollout buffer independently stores the unbounded Gaussian
    latent ``z``.

    For ordinary deterministic evaluation, ``project_inputs=True`` lets the
    wrapper apply the same hard projection directly.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        namespace: dict[str, Any],
        ax_bounds: tuple[float, float],
        ay_bounds: tuple[float, float],
        neighbor_range: float,
        eps_side: float,
        k0: float,
        k1: float,
        max_neighbor_constraints: Optional[int],
        base_observation_dim: Optional[int] = None,
        max_constraints: int = 18,
        project_inputs: bool = False,
        lambda_delta: float = 0.0,
        lambda_intervention: float = 0.0,
        correction_epsilon: float = 0.03,
        action_rate_penalty_lambda: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.namespace = namespace
        self.ax_bounds = (float(ax_bounds[0]), float(ax_bounds[1]))
        self.ay_bounds = (float(ay_bounds[0]), float(ay_bounds[1]))
        self.neighbor_range = float(neighbor_range)
        self.eps_side = float(eps_side)
        self.k0 = float(k0)
        self.k1 = float(k1)
        self.max_neighbor_constraints = (
            None
            if max_neighbor_constraints is None
            else int(max_neighbor_constraints)
        )
        if base_observation_dim is None:
            if not isinstance(env.observation_space, gym.spaces.Box):
                raise TypeError("Projected PPO currently requires a flat Box observation")
            base_observation_dim = int(np.prod(env.observation_space.shape))
        self.layout = CBFContextLayout(
            base_observation_dim=int(base_observation_dim),
            max_constraints=int(max_constraints),
        )
        required_capacity = (
            (0 if self.max_neighbor_constraints is None else self.max_neighbor_constraints)
            + 2
            + 4
        )
        if self.max_neighbor_constraints is None:
            raise ValueError(
                "Projected PPO requires a finite max_neighbor_constraints for padded context"
            )
        if required_capacity > self.layout.max_constraints:
            raise ValueError(
                f"CBF context capacity {self.layout.max_constraints} is below required {required_capacity}"
            )
        self.project_inputs = bool(project_inputs)
        self.lambda_delta = float(lambda_delta)
        self.lambda_intervention = float(lambda_intervention)
        self.correction_epsilon = float(correction_epsilon)
        self.action_rate_penalty_lambda = float(action_rate_penalty_lambda)
        if not np.isfinite(self.action_rate_penalty_lambda) or self.action_rate_penalty_lambda < 0.0:
            raise ValueError("action_rate_penalty_lambda must be finite and non-negative")

        self.action_space = gym.spaces.Box(
            low=np.asarray([self.ax_bounds[0], self.ay_bounds[0]], dtype=np.float32),
            high=np.asarray([self.ax_bounds[1], self.ay_bounds[1]], dtype=np.float32),
            dtype=np.float32,
        )
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("Projected PPO currently requires a flat Box observation")
        if int(np.prod(env.observation_space.shape)) != self.layout.base_observation_dim:
            raise ValueError(
                "Base observation width disagrees with the CBF context layout: "
                f"{env.observation_space.shape} vs {self.layout.base_observation_dim}"
            )
        base_low = np.asarray(env.observation_space.low, dtype=np.float32).reshape(-1)
        base_high = np.asarray(env.observation_space.high, dtype=np.float32).reshape(-1)
        context_width = self.layout.observation_dim - self.layout.base_observation_dim
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                [base_low, np.full(context_width, -np.inf, dtype=np.float32)]
            ),
            high=np.concatenate(
                [base_high, np.full(context_width, np.inf, dtype=np.float32)]
            ),
            dtype=np.float32,
        )
        self._last_system: Optional[dict[str, Any]] = None
        self._pending_projection: Optional[dict[str, Any]] = None
        self._previous_executed_action_normalized: Optional[np.ndarray] = None

    @property
    def physical_low(self) -> np.ndarray:
        return np.asarray(self.action_space.low, dtype=np.float32)

    @property
    def physical_high(self) -> np.ndarray:
        return np.asarray(self.action_space.high, dtype=np.float32)

    def _constraint_system(self) -> dict[str, Any]:
        ego = self.namespace["get_ego_state"](self)
        neighbors = self.namespace["get_neighbor_states"](
            self, neighbor_range=self.neighbor_range
        )
        road_width = float(self.namespace["_lane_free_base"](self).config["road_width"])
        system = build_cbf_action_constraints(
            self.namespace,
            ego,
            neighbors,
            road_width,
            self.ax_bounds,
            self.ay_bounds,
            self.eps_side,
            self.k0,
            self.k1,
            self.max_neighbor_constraints,
        )
        if int(system["rows"].shape[0]) > self.layout.max_constraints:
            raise RuntimeError(
                f"CBF produced {system['rows'].shape[0]} rows for a "
                f"{self.layout.max_constraints}-row context"
            )
        system["hash"] = constraint_system_hash(system["rows"], system["bounds"])
        return system

    def _augment_observation(
        self, observation: np.ndarray, system: Optional[dict[str, Any]] = None
    ) -> np.ndarray:
        system = self._constraint_system() if system is None else system
        self._last_system = copy.deepcopy(system)
        return append_cbf_context(
            observation,
            system["rows"],
            system["bounds"],
            layout=self.layout,
        )

    def current_constraint_system(self) -> dict[str, Any]:
        """Return a copy of the exact context represented in the last observation."""

        if self._last_system is None:
            self._last_system = self._constraint_system()
        return copy.deepcopy(self._last_system)

    def project_current_action(self, raw_action: Any) -> tuple[np.ndarray, dict[str, Any]]:
        system = self.current_constraint_system()
        result = project_polytope_2d_numpy(
            raw_action,
            system["rows"],
            system["bounds"],
            action_low=self.physical_low,
            action_high=self.physical_high,
        )
        return result.action.copy(), self._projection_record(
            raw_action=raw_action,
            safe_action=result.action,
            result=result,
            system=system,
            cbf_applied=True,
        )

    def _projection_record(
        self,
        *,
        raw_action: Any,
        safe_action: Any,
        result: Optional[NumpyProjection2D],
        system: dict[str, Any],
        cbf_applied: bool,
    ) -> dict[str, Any]:
        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)[:2]
        safe = np.asarray(safe_action, dtype=np.float32).reshape(-1)[:2]
        rows = np.asarray(system["rows"], dtype=np.float32).reshape(-1, 2)
        bounds = np.asarray(system["bounds"], dtype=np.float32).reshape(-1)
        raw_max_violation = (
            float(np.max(rows @ raw - bounds)) if rows.shape[0] else 0.0
        )
        half_range = np.maximum(0.5 * (self.physical_high - self.physical_low), 1e-6)
        correction_normalized = float(np.linalg.norm((safe - raw) / half_range))
        intervened = bool(cbf_applied and correction_normalized > self.correction_epsilon)
        return {
            "raw_action": raw.copy(),
            "safe_action": safe.copy(),
            "cbf_applied": bool(cbf_applied),
            "correction_norm_physical": float(np.linalg.norm(safe - raw)),
            "correction_norm_normalized": correction_normalized,
            "intervened": intervened,
            "feasible": bool(True if result is None else result.feasible),
            "fallback_used": bool(False if result is None else result.fallback_used),
            "projection_source": "box" if result is None else str(result.source),
            "max_constraint_violation_safe": (
                0.0 if result is None else float(result.max_violation)
            ),
            "max_constraint_violation_raw": raw_max_violation,
            "raw_feasible": bool(raw_max_violation <= 1e-6),
            "active_indices": (
                np.zeros(0, dtype=np.int64)
                if result is None
                else result.active_indices.copy()
            ),
            "constraint_hash": str(system["hash"]),
            "system": copy.deepcopy(system),
        }

    @staticmethod
    def _hocbf_diagnostics(
        system: dict[str, Any], safe_action: np.ndarray
    ) -> dict[str, Any]:
        """Evaluate the non-box HOCBF rows for one executed substep.

        ``build_cbf_action_constraints`` represents the desired condition
        ``h_ddot + k1 h_dot + k0 h >= 0`` as ``row @ a <= bound``.  Box
        constraints are intentionally excluded below: saturation is useful to
        log separately, whereas this margin answers the CBF stability
        question directly.
        """

        rows = np.asarray(system.get("cbf_rows", ()), dtype=float)
        bounds = np.asarray(system.get("cbf_bounds", ()), dtype=float).reshape(-1)
        action = np.asarray(safe_action, dtype=float).reshape(-1)[:2]
        if rows.size == 0 or bounds.size == 0:
            return {
                "hocbf_margin": float("inf"),
                "max_hocbf_violation_safe": 0.0,
                "hocbf_condition_satisfied": True,
            }
        rows = rows.reshape(-1, 2)
        slack = bounds - rows @ action
        min_margin = float(np.min(slack))
        max_violation = float(np.max(-slack))
        return {
            "hocbf_margin": min_margin,
            "max_hocbf_violation_safe": max(0.0, max_violation),
            "hocbf_condition_satisfied": bool(max_violation <= 1e-5),
        }

    def _project_substep_action(
        self, raw_action: Any
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Project a fresh physical action against the current physics state."""

        # Do not use ``current_constraint_system`` here: it intentionally
        # caches the policy-rate observation context.  At 100 Hz the traffic
        # state changes during the ten simulator frames, so each substep needs
        # a new HOCBF polytope.
        system = self._constraint_system()
        result = project_polytope_2d_numpy(
            raw_action,
            system["rows"],
            system["bounds"],
            action_low=self.physical_low,
            action_high=self.physical_high,
        )
        record = self._projection_record(
            raw_action=raw_action,
            safe_action=result.action,
            result=result,
            system=system,
            cbf_applied=True,
        )
        record.update(self._hocbf_diagnostics(system, result.action))
        return result.action.copy(), record

    def _substep_filter_enabled(self, record: dict[str, Any]) -> bool:
        if not bool(record.get("cbf_applied", False)):
            return False
        base = self.namespace["_lane_free_base"](self)
        return bool(base.config.get("cbf_substep_filtering", False)) and hasattr(
            base, "set_ego_substep_action_filter"
        )

    def _initial_safety_diagnostics(self) -> dict[str, Any]:
        """Check h >= 0 and psi_1 = h_dot + k1 h >= 0 at reset."""

        ego = self.namespace["get_ego_state"](self)
        neighbors = self.namespace["get_neighbor_states"](
            self, neighbor_range=self.neighbor_range
        )
        h_values: list[float] = []
        psi_values: list[float] = []
        geometry = self.namespace.get("pairwise_cbf_geometry")
        relative_state = self.namespace.get("pairwise_relative_state")
        derivatives = self.namespace.get("centerline_barrier_derivatives")
        if geometry is not None and relative_state is not None and derivatives is not None:
            for neighbor in neighbors:
                h_value = float(geometry(ego, neighbor, eps_side=self.eps_side)[0])
                dx, dy, dvx, dvy = relative_state(ego, neighbor)
                h_derivative, gradient, _hessian, *_ = derivatives(
                    np.asarray([dx, dy], dtype=float),
                    ego,
                    neighbor,
                    self.eps_side,
                )
                del h_derivative
                h_dot = float(
                    np.asarray(gradient, dtype=float)
                    @ np.asarray([dvx, dvy], dtype=float)
                )
                h_values.append(h_value)
                psi_values.append(h_dot + self.k1 * h_value)

        base = self.namespace["_lane_free_base"](self)
        road_width = float(base.config["road_width"])
        ego_half_width = 0.5 * float(ego["width"])
        left_h = float(ego["y"] - ego_half_width)
        right_h = float(road_width - ego_half_width - ego["y"])
        h_values.extend([left_h, right_h])
        psi_values.extend(
            [
                float(ego["vy"] + self.k1 * left_h),
                float(-ego["vy"] + self.k1 * right_h),
            ]
        )
        min_h = float(np.min(h_values)) if h_values else np.nan
        min_psi = float(np.min(psi_values)) if psi_values else np.nan
        tolerance = float(self.namespace.get("CBF_QP_FEASIBILITY_TOL", 1e-5))
        safe = bool(
            np.isfinite(min_h)
            and min_h >= -tolerance
            and np.isfinite(min_psi)
            and min_psi >= -tolerance
        )
        return {
            "cbf_initial_min_h": min_h,
            "cbf_initial_min_psi": min_psi,
            "cbf_initial_safe_set": safe,
        }

    @staticmethod
    def _aggregate_substep_record(
        record: dict[str, Any], substeps: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Turn physics-rate projections into one policy-rate execution record."""

        if not substeps:
            return record
        aggregate = copy.deepcopy(record)
        normalized = np.asarray(
            [step.get("correction_norm_normalized", 0.0) for step in substeps],
            dtype=float,
        )
        physical = np.asarray(
            [step.get("correction_norm_physical", 0.0) for step in substeps],
            dtype=float,
        )
        normalized = normalized[np.isfinite(normalized)]
        physical = physical[np.isfinite(physical)]
        # The square of the reported policy-rate norm is exactly the mean
        # per-substep squared correction.  That preserves the reward and
        # safety-critic cost contract without hiding a 10x scale change.
        aggregate["correction_norm_normalized"] = float(
            np.sqrt(np.mean(normalized**2)) if normalized.size else 0.0
        )
        aggregate["correction_norm_physical"] = float(
            np.sqrt(np.mean(physical**2)) if physical.size else 0.0
        )
        aggregate["intervened"] = bool(
            any(bool(step.get("intervened", False)) for step in substeps)
        )
        aggregate["feasible"] = bool(
            all(bool(step.get("feasible", True)) for step in substeps)
        )
        aggregate["fallback_used"] = bool(
            any(bool(step.get("fallback_used", False)) for step in substeps)
        )
        aggregate["substep_count"] = int(len(substeps))
        aggregate["substep_intervention_steps"] = int(
            sum(bool(step.get("intervened", False)) for step in substeps)
        )
        aggregate["substep_fallback_steps"] = int(
            sum(bool(step.get("fallback_used", False)) for step in substeps)
        )
        margins = np.asarray(
            [step.get("hocbf_margin", np.nan) for step in substeps], dtype=float
        )
        margins = margins[np.isfinite(margins)]
        violations = np.asarray(
            [step.get("max_hocbf_violation_safe", np.nan) for step in substeps],
            dtype=float,
        )
        violations = violations[np.isfinite(violations)]
        aggregate["hocbf_margin"] = float(np.min(margins)) if margins.size else np.nan
        aggregate["max_hocbf_violation_safe"] = (
            float(np.max(violations)) if violations.size else np.nan
        )
        aggregate["hocbf_condition_satisfied"] = bool(
            all(
                bool(step.get("hocbf_condition_satisfied", True))
                for step in substeps
            )
        )
        # Policy diagnostics retain the latent/raw action from the PPO sample
        # but expose the last actually executed substep action and state.
        last = substeps[-1]
        aggregate["safe_action"] = np.asarray(last["safe_action"], dtype=np.float32)
        aggregate["active_indices"] = np.asarray(
            last.get("active_indices", ()), dtype=np.int64
        )
        aggregate["constraint_hash"] = str(last.get("constraint_hash", ""))
        aggregate["system"] = copy.deepcopy(last["system"])
        aggregate["projection_source"] = "substep_active_set_2d"
        safe_violations = np.asarray(
            [step.get("max_constraint_violation_safe", np.nan) for step in substeps],
            dtype=float,
        )
        safe_violations = safe_violations[np.isfinite(safe_violations)]
        if safe_violations.size:
            aggregate["max_constraint_violation_safe"] = float(
                np.max(safe_violations)
            )
        return aggregate

    def set_projection_record(
        self,
        raw_action: Any,
        safe_action: Any,
        *,
        feasible: bool,
        fallback_used: bool,
        projection_source: str,
        max_constraint_violation_safe: float,
        active_indices: Any = (),
        constraint_hash: Optional[str] = None,
        cbf_applied: bool = True,
    ) -> None:
        """Stage algorithm-computed ``z``/``P(z)`` data for the next step."""

        system = self.current_constraint_system()
        if constraint_hash is not None and str(constraint_hash) != str(system["hash"]):
            raise RuntimeError(
                "Projected action constraint context does not match the current simulator state"
            )
        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)[:2]
        safe = np.asarray(safe_action, dtype=np.float32).reshape(-1)[:2]
        if not self.action_space.contains(safe):
            raise ValueError(f"Executed projected action is outside the physical Box: {safe}")
        result = NumpyProjection2D(
            action=safe,
            feasible=bool(feasible),
            fallback_used=bool(fallback_used),
            source=str(projection_source),
            active_indices=np.asarray(active_indices, dtype=np.int64).reshape(-1),
            max_violation=float(max_constraint_violation_safe),
        )
        self._pending_projection = self._projection_record(
            raw_action=raw,
            safe_action=safe,
            result=result,
            system=system,
            cbf_applied=bool(cbf_applied),
        )

    def reset(self, **kwargs):
        self._pending_projection = None
        self._previous_executed_action_normalized = None
        observation, info = self.env.reset(**kwargs)
        system = self._constraint_system()
        info = dict(info)
        initial_safety = self._initial_safety_diagnostics()
        info.update(initial_safety)
        base = self.namespace["_lane_free_base"](self)
        traffic_safety = base.config.get("traffic_safety", {})
        # An explicit top-level setting is authoritative.  This lets an
        # evaluation protocol retain the source run's CBF-safe spawn sampler
        # (and therefore paired initial states) while allowing a deployment
        # gain sweep to inspect candidates whose psi_1 condition is not
        # satisfied at reset.  When the top-level key is absent, preserve the
        # historical inference from the traffic safe-spawn configuration.
        configured_initial_safe = base.config.get("cbf_require_initial_safe_set")
        if configured_initial_safe is None:
            require_initial_safe = bool(
                isinstance(traffic_safety, dict)
                and traffic_safety.get("spawn_cbf_safe_set", False)
            )
        else:
            require_initial_safe = bool(configured_initial_safe)
        if require_initial_safe and not bool(initial_safety["cbf_initial_safe_set"]):
            raise RuntimeError(
                "CBF reset violated h >= 0 or psi_1 >= 0: "
                f"min_h={initial_safety['cbf_initial_min_h']:.6f}, "
                f"min_psi={initial_safety['cbf_initial_min_psi']:.6f}"
            )
        info["cbf_constraint_hash"] = str(system["hash"])
        info["cbf_constraint_count"] = int(system["rows"].shape[0])
        return self._augment_observation(observation, system), info

    def _box_record(self, action: Any, system: dict[str, Any]) -> dict[str, Any]:
        raw = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        safe = np.clip(raw, self.physical_low, self.physical_high).astype(np.float32)
        return self._projection_record(
            raw_action=raw,
            safe_action=safe,
            result=None,
            system=system,
            cbf_applied=False,
        )

    def step(self, action):
        system = self.current_constraint_system()
        if self._pending_projection is not None:
            record = self._pending_projection
            self._pending_projection = None
            safe_action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
            if not np.allclose(safe_action, record["safe_action"], atol=1e-6):
                raise RuntimeError("Staged projected action differs from the executed action")
        elif self.project_inputs:
            safe_action, record = self.project_current_action(action)
        else:
            record = self._box_record(action, system)
            safe_action = record["safe_action"]

        use_substep_filter = self._substep_filter_enabled(record)
        simulator_action = (
            np.asarray(record["raw_action"], dtype=np.float32)
            if use_substep_filter
            else safe_action
        )
        normalized_action = np.asarray(
            self.namespace["_physical_to_normalized_action"](
                self, simulator_action
            ),
            dtype=np.float32,
        ).reshape(-1)[:2]
        substep_records: list[dict[str, Any]] = []
        base = self.namespace["_lane_free_base"](self)
        previous_filter = None
        if use_substep_filter:
            def _filter_substep(
                proposed_physical_action: np.ndarray, _frame_index: int
            ) -> tuple[np.ndarray, dict[str, Any]]:
                safe_substep, substep_record = self._project_substep_action(
                    proposed_physical_action
                )
                substep_records.append(substep_record)
                return safe_substep, substep_record

            previous_filter = base.set_ego_substep_action_filter(_filter_substep)
        try:
            observation, reward, terminated, truncated, info = self.env.step(
                normalized_action
            )
        finally:
            if use_substep_filter:
                # The environment owns no policy state; leaving a callback
                # installed would accidentally filter a later raw rollout.
                base.set_ego_substep_action_filter(previous_filter)

        record = self._aggregate_substep_record(record, substep_records)
        executed_action = np.asarray(record["safe_action"], dtype=np.float32)
        executed_normalized_action = np.asarray(
            self.namespace["_physical_to_normalized_action"](
                self, executed_action
            ),
            dtype=np.float32,
        ).reshape(-1)[:2]
        if self._previous_executed_action_normalized is None:
            action_delta_norm_sq = 0.0
        else:
            action_delta = (
                executed_normalized_action - self._previous_executed_action_normalized
            )
            action_delta_norm_sq = float(np.dot(action_delta, action_delta))
        action_rate_penalty = self.action_rate_penalty_lambda * action_delta_norm_sq

        correction_penalty = (
            self.lambda_delta * float(record["correction_norm_normalized"]) ** 2
            + self.lambda_intervention * float(record["intervened"])
        )
        reward = float(reward) - float(correction_penalty) - float(action_rate_penalty)
        self._previous_executed_action_normalized = executed_normalized_action.copy()
        info = dict(info)
        raw = np.asarray(record["raw_action"], dtype=np.float32)
        safe = np.asarray(record["safe_action"], dtype=np.float32)
        record_system = record["system"]
        info.update(
            {
                "latent_action_z_phys": raw.copy(),
                "raw_action_phys": raw.copy(),
                "safe_action_phys": safe.copy(),
                "intervention": bool(record["intervened"]),
                "cbf_event_intervened": bool(record["intervened"]),
                "cbf_event_intervention_threshold": float(self.correction_epsilon),
                "cbf_a_rl_x": float(raw[0]),
                "cbf_a_rl_y": float(raw[1]),
                "cbf_a_safe_x": float(safe[0]),
                "cbf_a_safe_y": float(safe[1]),
                "cbf_correction_norm": float(record["correction_norm_physical"]),
                "cbf_correction_norm_normalized": float(
                    record["correction_norm_normalized"]
                ),
                "cbf_intervened": bool(record["intervened"]),
                "cbf_raw_feasible": bool(record["raw_feasible"]),
                "cbf_qp_success": bool(record["feasible"]),
                "cbf_fallback_used": bool(record["fallback_used"]),
                "cbf_projection_solver": "active_set_2d_shared",
                "cbf_projection_source": str(record["projection_source"]),
                "cbf_substep_filter_enabled": bool(use_substep_filter),
                "cbf_substep_count": int(record.get("substep_count", 0)),
                "cbf_substep_intervention_steps": int(
                    record.get("substep_intervention_steps", 0)
                ),
                "cbf_substep_fallback_steps": int(
                    record.get("substep_fallback_steps", 0)
                ),
                "cbf_hocbf_min_margin": float(
                    record.get("hocbf_margin", np.nan)
                ),
                "cbf_hocbf_max_violation_safe": float(
                    record.get("max_hocbf_violation_safe", np.nan)
                ),
                "cbf_hocbf_condition_satisfied": bool(
                    record.get("hocbf_condition_satisfied", True)
                ),
                "cbf_max_constraint_violation_safe": float(
                    record["max_constraint_violation_safe"]
                ),
                "cbf_max_constraint_violation_raw": float(
                    record["max_constraint_violation_raw"]
                ),
                "cbf_constraint_hash": str(record["constraint_hash"]),
                "cbf_constraint_count": int(record_system["rows"].shape[0]),
                "cbf_active_constraint_indices": np.asarray(
                    record["active_indices"], dtype=np.int64
                ),
                "cbf_min_h": float(record_system["min_h"]),
                "cbf_min_center_distance": float(
                    record_system["min_center_distance"]
                ),
                "cbf_min_required_distance": float(
                    record_system["min_required_distance"]
                ),
                "cbf_num_neighbor_constraints": int(
                    record_system["num_neighbor_constraints"]
                ),
                "cbf_left_boundary_h": float(record_system["left_boundary_h"]),
                "cbf_right_boundary_h": float(record_system["right_boundary_h"]),
                "cbf_min_boundary_h": float(record_system["min_boundary_h"]),
                "cbf_filter_norm_reward_penalty": float(
                    self.lambda_delta
                    * float(record["correction_norm_normalized"]) ** 2
                ),
                "cbf_filter_event_reward_penalty": float(
                    self.lambda_intervention * float(record["intervened"])
                ),
                "cbf_filter_reward_penalty": float(correction_penalty),
                "cbf_correction_reward": -float(correction_penalty),
                "action_delta_norm_sq": float(action_delta_norm_sq),
                "action_rate_penalty": float(action_rate_penalty),
                "action_rate_penalty_lambda": float(
                    self.action_rate_penalty_lambda
                ),
            }
        )
        next_system = self._constraint_system()
        return (
            self._augment_observation(observation, next_system),
            reward,
            terminated,
            truncated,
            info,
        )
