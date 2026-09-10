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

import hashlib
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from scripts.common.action_units import (
    normalized_action_delta_norm,
    physical_to_normalized_action,
    unbounded_action_clip_norm,
    validate_matching_physical_action_bounds,
)
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
        psi1_gain: Optional[float] = None,
        base_observation_dim: Optional[int] = None,
        max_constraints: int = 18,
        project_inputs: bool = False,
        lambda_delta: float = 0.0,
        lambda_intervention: float = 0.0,
        correction_epsilon: float = 0.03,
        action_rate_penalty_lambda: float = 0.0,
        hocbf_reward_lambda: float = 0.0,
        hocbf_reward_scale: float = 1.0,
        hocbf_reward_margin: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.namespace = namespace
        self.ax_bounds = (float(ax_bounds[0]), float(ax_bounds[1]))
        self.ay_bounds = (float(ay_bounds[0]), float(ay_bounds[1]))
        base = namespace.get("_lane_free_base", lambda wrapper: wrapper.unwrapped)(self)
        env_low, env_high = validate_matching_physical_action_bounds(
            [self.ax_bounds[0], self.ay_bounds[0]],
            [self.ax_bounds[1], self.ay_bounds[1]],
            base.config,
        )
        # Keep the wrapper's public physical Box tied to the simulator config
        # after the construction-time cross-check above.
        self._physical_low = env_low.astype(np.float32)
        self._physical_high = env_high.astype(np.float32)
        self.neighbor_range = float(neighbor_range)
        self.eps_side = float(eps_side)
        self.k0 = float(k0)
        self.k1 = float(k1)
        self.psi1_gain = float(
            namespace.get("CBF_PSI1_GAIN", 2.3)
            if psi1_gain is None
            else psi1_gain
        )
        if not np.isfinite(self.psi1_gain) or self.psi1_gain <= 0.0:
            raise ValueError("psi1_gain must be finite and positive")
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
        self.hocbf_reward_lambda = float(hocbf_reward_lambda)
        self.hocbf_reward_scale = float(hocbf_reward_scale)
        self.hocbf_reward_margin = float(hocbf_reward_margin)
        if not np.isfinite(self.action_rate_penalty_lambda) or self.action_rate_penalty_lambda < 0.0:
            raise ValueError("action_rate_penalty_lambda must be finite and non-negative")
        if not np.isfinite(self.hocbf_reward_lambda) or self.hocbf_reward_lambda < 0.0:
            raise ValueError("hocbf_reward_lambda must be finite and non-negative")
        if not np.isfinite(self.hocbf_reward_scale) or self.hocbf_reward_scale <= 0.0:
            raise ValueError("hocbf_reward_scale must be finite and positive")
        if not np.isfinite(self.hocbf_reward_margin) or self.hocbf_reward_margin < 0.0:
            raise ValueError("hocbf_reward_margin must be finite and non-negative")

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
        self._last_reset_info: dict[str, Any] = {}

    @property
    def physical_low(self) -> np.ndarray:
        return self._physical_low.copy()

    @property
    def physical_high(self) -> np.ndarray:
        return self._physical_high.copy()

    @property
    def last_reset_info(self) -> dict[str, Any]:
        """Return the accepted reset diagnostics for provenance-aware callers."""

        return dict(self._last_reset_info)

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
        self._last_system = system
        return append_cbf_context(
            observation,
            system["rows"],
            system["bounds"],
            layout=self.layout,
        )

    def current_constraint_system(self) -> dict[str, Any]:
        """Return the exact context represented in the last observation.

        ``system`` dicts are freshly built by ``_constraint_system`` for each
        observation and are never mutated afterward, so callers receive the
        same object rather than a deep copy of it.
        """

        if self._last_system is None:
            self._last_system = self._constraint_system()
        return self._last_system

    def project_current_action(self, raw_action: Any) -> tuple[np.ndarray, dict[str, Any]]:
        system = self.current_constraint_system()
        actor_latent = np.asarray(raw_action, dtype=np.float32).reshape(-1)[:2]
        box_clipped = np.clip(
            actor_latent, self.physical_low, self.physical_high
        ).astype(np.float32)
        result = project_polytope_2d_numpy(
            box_clipped,
            system["rows"],
            system["bounds"],
            action_low=self.physical_low,
            action_high=self.physical_high,
        )
        return result.action.copy(), self._projection_record(
            actor_latent_phys=actor_latent,
            box_clipped_phys=box_clipped,
            safe_action=result.action,
            result=result,
            system=system,
            cbf_applied=True,
        )

    def _projection_record(
        self,
        *,
        raw_action: Any | None = None,
        actor_latent_phys: Any | None = None,
        box_clipped_phys: Any | None = None,
        safe_action: Any,
        result: Optional[NumpyProjection2D],
        system: dict[str, Any],
        cbf_applied: bool,
    ) -> dict[str, Any]:
        latent_source = actor_latent_phys if actor_latent_phys is not None else raw_action
        if latent_source is None:
            raise ValueError("projection records require actor_latent_phys")
        latent = np.asarray(latent_source, dtype=np.float32).reshape(-1)[:2]
        safe = np.asarray(safe_action, dtype=np.float32).reshape(-1)[:2]
        box = (
            np.clip(latent, self.physical_low, self.physical_high)
            if box_clipped_phys is None
            else np.asarray(box_clipped_phys, dtype=np.float32).reshape(-1)[:2]
        )
        box = np.clip(box, self.physical_low, self.physical_high).astype(np.float32)
        rows = np.asarray(system["rows"], dtype=np.float32).reshape(-1, 2)
        bounds = np.asarray(system["bounds"], dtype=np.float32).reshape(-1)
        cbf_rows = np.asarray(system.get("cbf_rows", ()), dtype=np.float32).reshape(-1, 2)
        cbf_bounds = np.asarray(system.get("cbf_bounds", ()), dtype=np.float32).reshape(-1)

        def _max_violation(action: np.ndarray, constraint_rows: np.ndarray, constraint_bounds: np.ndarray) -> float:
            return (
                float(np.max(constraint_rows @ action - constraint_bounds))
                if constraint_rows.shape[0]
                else 0.0
            )

        raw_max_violation = _max_violation(latent, rows, bounds)
        box_max_violation = _max_violation(box, rows, bounds)
        raw_cbf_violation = _max_violation(latent, cbf_rows, cbf_bounds)
        box_cbf_violation = _max_violation(box, cbf_rows, cbf_bounds)
        correction_normalized = normalized_action_delta_norm(
            safe, box, self.physical_low, self.physical_high
        )
        actuator_clip_norm = unbounded_action_clip_norm(
            latent, box, self.physical_low, self.physical_high
        )
        intervened = bool(cbf_applied and correction_normalized > self.correction_epsilon)
        return {
            # Canonical action-stage names.  The historical raw/safe aliases
            # below remain for older result readers.
            "actor_latent_phys": latent.copy(),
            "box_clipped_phys": box.copy(),
            "cbf_safe_phys": safe.copy(),
            "executed_phys": safe.copy(),
            "simulator_action_normalized": np.zeros(2, dtype=np.float32),
            "raw_action": latent.copy(),
            "safe_action": safe.copy(),
            "cbf_applied": bool(cbf_applied),
            "actuator_clip_norm": actuator_clip_norm,
            "correction_norm_physical": float(np.linalg.norm(safe - box)),
            "correction_norm_normalized": correction_normalized,
            "cbf_correction_norm_normalized": correction_normalized,
            "intervened": intervened,
            "feasible": bool(True if result is None else result.feasible),
            "fallback_used": bool(False if result is None else result.fallback_used),
            "projection_source": "box" if result is None else str(result.source),
            "max_constraint_violation_safe": (
                0.0 if result is None else float(result.max_violation)
            ),
            "max_constraint_violation_raw": raw_max_violation,
            "max_constraint_violation_box": box_max_violation,
            "max_cbf_constraint_violation_raw": raw_cbf_violation,
            "max_cbf_constraint_violation_box": box_cbf_violation,
            "raw_feasible": bool(raw_max_violation <= 1e-6),
            "box_clipped_feasible": bool(box_max_violation <= 1e-6),
            "raw_cbf_feasible": bool(raw_cbf_violation <= 1e-6),
            "box_clipped_cbf_feasible": bool(box_cbf_violation <= 1e-6),
            "active_indices": (
                np.zeros(0, dtype=np.int64)
                if result is None
                else result.active_indices.copy()
            ),
            "constraint_hash": str(system["hash"]),
            # Constraint systems are freshly constructed for each observation
            # or physics substep and are never mutated after this record is
            # created.  Keep a short-lived reference here; deep-copying all
            # row/bound/geometry arrays at 100 Hz dominated diagnostic work
            # without changing the public info payload.
            "system": system,
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
                "hocbf_mean_violation": 0.0,
                "hocbf_max_abs_residual": 0.0,
            }
        rows = rows.reshape(-1, 2)
        slack = bounds - rows @ action
        min_margin = float(np.min(slack))
        violations = np.maximum(-slack, 0.0)
        max_violation = float(np.max(-slack))
        return {
            "hocbf_margin": min_margin,
            "max_hocbf_violation_safe": max(0.0, max_violation),
            "hocbf_condition_satisfied": bool(max_violation <= 1e-5),
            "hocbf_mean_violation": float(np.mean(violations)),
            "hocbf_max_abs_residual": float(np.max(np.abs(slack))),
        }

    def _project_substep_action(
        self, raw_action: Any
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Project a fresh physical action against the current physics state."""

        # Do not use ``current_constraint_system`` here: it intentionally
        # caches the policy-rate observation context.  This method is used
        # only by the explicit physics-rate callback mode.  The requested
        # policy-rate experiment leaves that callback disabled, so its hard
        # CBF projection is evaluated once per policy action instead.
        system = self._constraint_system()
        actor_latent = np.asarray(raw_action, dtype=np.float32).reshape(-1)[:2]
        box_clipped = np.clip(
            actor_latent, self.physical_low, self.physical_high
        ).astype(np.float32)
        result = project_polytope_2d_numpy(
            box_clipped,
            system["rows"],
            system["bounds"],
            action_low=self.physical_low,
            action_high=self.physical_high,
        )
        record = self._projection_record(
            actor_latent_phys=actor_latent,
            box_clipped_phys=box_clipped,
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
        """Check h >= 0 and psi_1 = h_dot + 2.3 h >= 0 at reset.

        ``self.k1`` is the coefficient of ``h_dot`` in the second-order
        HOCBF condition.  It is intentionally not reused here: under the
        critical alternative ``(k1, k0) = (4.6, 5.29)``, the first-level
        factor is ``sqrt(k0) = 2.3``.
        """

        ego = self.namespace["get_ego_state"](self)
        neighbors = list(self.namespace["get_neighbor_states"](
            self, neighbor_range=self.neighbor_range
        ))
        max_neighbor_constraints = getattr(self, "max_neighbor_constraints", None)
        if max_neighbor_constraints is not None:
            neighbors = neighbors[: max_neighbor_constraints]
        h_values: list[float] = []
        psi_values: list[float] = []
        batch_builder = self.namespace.get("batch_pairwise_hocbf_constraints")
        if callable(batch_builder) and neighbors:
            batch = batch_builder(
                ego,
                neighbors,
                eps_side=self.eps_side,
                k0=self.k0,
                k1=self.k1,
            )
            h_values.extend(np.asarray(batch["h"], dtype=float).tolist())
            psi_values.extend(
                (
                    np.asarray(batch["h_dot"], dtype=float)
                    + self.psi1_gain * np.asarray(batch["h"], dtype=float)
                ).tolist()
            )
        else:
            geometry = self.namespace.get("pairwise_cbf_geometry")
            relative_state = self.namespace.get("pairwise_relative_state")
            derivatives = self.namespace.get("centerline_barrier_derivatives")
            if geometry is None or relative_state is None or derivatives is None:
                raise RuntimeError("CBF reset diagnostics require pairwise geometry")
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
                psi_values.append(h_dot + self.psi1_gain * h_value)

        base = self.namespace["_lane_free_base"](self)
        road_width = float(base.config["road_width"])
        ego_half_width = 0.5 * float(ego["width"])
        left_h = float(ego["y"] - ego_half_width)
        right_h = float(road_width - ego_half_width - ego["y"])
        h_values.extend([left_h, right_h])
        psi_values.extend(
            [
                float(ego["vy"] + self.psi1_gain * left_h),
                float(-ego["vy"] + self.psi1_gain * right_h),
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
            "cbf_initial_min_psi1": min_psi,
            # Keep the historical alias for existing result readers.
            "cbf_initial_min_psi": min_psi,
            "cbf_psi1_gain": float(self.psi1_gain),
            "cbf_initial_safe_set": safe,
        }

    @staticmethod
    def _aggregate_substep_record(
        record: dict[str, Any], substeps: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Turn physics-rate projections into one policy-rate execution record."""

        if not substeps:
            return record
        # Records are internal and the system reference is immutable for the
        # lifetime of this policy step.  A shallow copy avoids cloning every
        # substep's full constraint matrix while preserving all scalar/array
        # values that are replaced below.
        aggregate = dict(record)
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
        aggregate["cbf_correction_norm_normalized"] = aggregate[
            "correction_norm_normalized"
        ]
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
        aggregate["cbf_safe_phys"] = aggregate["safe_action"].copy()
        aggregate["executed_phys"] = aggregate["safe_action"].copy()
        aggregate["active_indices"] = np.asarray(
            last.get("active_indices", ()), dtype=np.int64
        )
        aggregate["constraint_hash"] = str(last.get("constraint_hash", ""))
        aggregate["system"] = last["system"]
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
        box_clipped_phys: Any | None = None,
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
            actor_latent_phys=raw,
            box_clipped_phys=box_clipped_phys,
            safe_action=safe,
            result=result,
            system=system,
            cbf_applied=bool(cbf_applied),
        )

    def reset(self, **kwargs):
        self._pending_projection = None
        self._previous_executed_action_normalized = None
        base = self.namespace["_lane_free_base"](self)
        reset_feasibility = base.config.get("cbf_reset_feasibility", {})
        if reset_feasibility is None:
            reset_feasibility = {}
        if not isinstance(reset_feasibility, dict):
            raise TypeError("cbf_reset_feasibility must be a mapping when provided")
        feasibility_enabled = bool(reset_feasibility.get("enabled", False))
        if feasibility_enabled:
            if "max_attempts" not in reset_feasibility:
                raise ValueError(
                    "cbf_reset_feasibility.enabled requires an explicit max_attempts"
                )
            max_attempts = int(reset_feasibility["max_attempts"])
            if max_attempts <= 0:
                raise ValueError("cbf_reset_feasibility.max_attempts must be positive")
            if kwargs.get("seed") is None:
                raise ValueError(
                    "CBF reset-feasibility sampling requires an explicit reset seed"
                )
            initial_seed = int(kwargs["seed"])
        else:
            max_attempts = 1
            initial_seed = None

        for retry_count in range(max_attempts):
            candidate_kwargs = dict(kwargs)
            candidate_seed = None
            if feasibility_enabled:
                candidate_seed = int(initial_seed + retry_count)
                candidate_kwargs["seed"] = candidate_seed
            observation, info = self.env.reset(**candidate_kwargs)
            initial_safety = self._initial_safety_diagnostics()
            if not feasibility_enabled or bool(initial_safety["cbf_initial_safe_set"]):
                break
        else:
            raise RuntimeError(
                "CBF reset-feasibility sampler exhausted candidate seeds: "
                f"initial_seed={initial_seed}, max_attempts={max_attempts}, "
                f"min_h={initial_safety['cbf_initial_min_h']:.6f}, "
                f"min_psi1={initial_safety['cbf_initial_min_psi1']:.6f}"
            )

        system = self._constraint_system()
        info = dict(info)
        info.update(initial_safety)
        reset_metadata = {
            "cbf_reset_feasibility_enabled": feasibility_enabled,
            "cbf_reset_max_attempts": int(max_attempts),
            "cbf_reset_retry_count": int(retry_count),
            "cbf_reset_candidate_seed": candidate_seed,
        }
        info.update(reset_metadata)
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
                f"min_psi1={initial_safety['cbf_initial_min_psi1']:.6f} "
                f"(gain={self.psi1_gain:.6f})"
            )
        info["cbf_constraint_hash"] = str(system["hash"])
        info["cbf_constraint_count"] = int(system["rows"].shape[0])
        self._last_reset_info = {
            **initial_safety,
            **reset_metadata,
        }
        return self._augment_observation(observation, system), info

    def _box_record(self, action: Any, system: dict[str, Any]) -> dict[str, Any]:
        raw = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        safe = np.clip(raw, self.physical_low, self.physical_high).astype(np.float32)
        return self._projection_record(
            actor_latent_phys=raw,
            box_clipped_phys=safe,
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

        # The direct HOCBF reward is intentionally evaluated once at the
        # policy/CBF rate.  It uses the box-clipped physical actor action after only
        # actuator-box clipping; no CBF projection is included in this term.
        # Thus a treatment run can be compared with raw nominal PPO without
        # silently changing the action that reaches the simulator.
        hocbf_reward_action = np.clip(
            np.asarray(record["box_clipped_phys"], dtype=np.float32).reshape(-1)[:2],
            self.physical_low,
            self.physical_high,
        )
        raw_hocbf = self._hocbf_diagnostics(system, hocbf_reward_action)
        hocbf_reward_violation = max(
            0.0,
            self.hocbf_reward_margin - float(raw_hocbf["hocbf_margin"]),
        )
        hocbf_reward_penalty = self.hocbf_reward_lambda * (
            hocbf_reward_violation / self.hocbf_reward_scale
        ) ** 2

        use_substep_filter = self._substep_filter_enabled(record)
        simulator_action = (
            np.asarray(record["raw_action"], dtype=np.float32)
            if use_substep_filter
            else safe_action
        )
        normalized_action = physical_to_normalized_action(
            simulator_action, self.physical_low, self.physical_high
        )
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
        configured_execution = np.asarray(record["safe_action"], dtype=np.float32)
        accelerations = np.asarray(
            getattr(base, "_last_accelerations", np.empty((0, 2))), dtype=np.float32
        )
        if accelerations.ndim == 2 and accelerations.shape[0] > 0:
            executed_action = accelerations[0, :2].copy()
            execution_source = "simulator_last_acceleration"
        else:
            executed_action = configured_execution.copy()
            execution_source = "cbf_safe_fallback"
        record["executed_phys"] = executed_action.copy()
        executed_normalized_action = physical_to_normalized_action(
            executed_action, self.physical_low, self.physical_high
        )
        record["executed_normalized"] = executed_normalized_action.copy()
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
        reward = (
            float(reward)
            - float(correction_penalty)
            - float(action_rate_penalty)
            - float(hocbf_reward_penalty)
        )
        previous_executed_normalized = (
            np.zeros(2, dtype=np.float32)
            if self._previous_executed_action_normalized is None
            else self._previous_executed_action_normalized.copy()
        )
        self._previous_executed_action_normalized = executed_normalized_action.copy()
        info = dict(info)
        latent = np.asarray(record["actor_latent_phys"], dtype=np.float32)
        box = np.asarray(record["box_clipped_phys"], dtype=np.float32)
        safe = np.asarray(record["cbf_safe_phys"], dtype=np.float32)
        record_system = record["system"]
        info.update(
            {
                "simulator_action_normalized": normalized_action.copy(),
                "actor_latent_phys": latent.copy(),
                # The actor mean is available from the policy action-stages
                # API. Do not publish a misleading None value when this
                # lower-level wrapper receives only a sampled action.
                "box_clipped_phys": box.copy(),
                "cbf_safe_phys": safe.copy(),
                "executed_phys": executed_action.copy(),
                "previous_executed_normalized": previous_executed_normalized.copy(),
                "simulator_execution_source": execution_source,
                "simulator_action_matches_cbf_safe": bool(
                    np.allclose(executed_action, safe, atol=1e-5)
                ),
                "latent_action_z_phys": latent.copy(),
                "raw_action_phys": latent.copy(),
                "safe_action_phys": safe.copy(),
                "actuator_clip_norm": float(record["actuator_clip_norm"]),
                "intervention": bool(record["intervened"]),
                "cbf_event_intervened": bool(record["intervened"]),
                "cbf_event_intervention_threshold": float(self.correction_epsilon),
                "cbf_a_rl_x": float(latent[0]),
                "cbf_a_rl_y": float(latent[1]),
                "cbf_a_safe_x": float(safe[0]),
                "cbf_a_safe_y": float(safe[1]),
                "cbf_correction_norm": float(record["correction_norm_physical"]),
                "cbf_correction_norm_normalized": float(
                    record["correction_norm_normalized"]
                ),
                "cbf_intervened": bool(record["intervened"]),
                "cbf_raw_feasible": bool(record["raw_feasible"]),
                "cbf_box_clipped_feasible": bool(
                    record["box_clipped_feasible"]
                ),
                "cbf_raw_cbf_feasible": bool(record["raw_cbf_feasible"]),
                "cbf_box_clipped_cbf_feasible": bool(
                    record["box_clipped_cbf_feasible"]
                ),
                "cbf_qp_success": bool(record["feasible"]),
                "cbf_fallback_used": bool(record["fallback_used"]),
                "cbf_projection_solver": "active_set_2d_shared",
                "cbf_projection_source": str(record["projection_source"]),
                "cbf_substep_filter_enabled": bool(use_substep_filter),
                "cbf_substep_count": int(record.get("substep_count", 0)),
                "cbf_callback_evaluation_count": int(
                    record.get("substep_count", 0)
                ),
                "cbf_update_count": int(
                    record.get("substep_count", 0)
                    if use_substep_filter
                    else bool(record.get("cbf_applied", False))
                ),
                "cbf_execution_schedule": (
                    "physics_substep"
                    if use_substep_filter
                    else ("policy" if bool(record.get("cbf_applied", False)) else "none")
                ),
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
                "cbf_hocbf_raw_min_margin": float(
                    raw_hocbf["hocbf_margin"]
                ),
                "cbf_hocbf_raw_max_violation": float(
                    raw_hocbf["max_hocbf_violation_safe"]
                ),
                "cbf_hocbf_raw_mean_violation": float(
                    raw_hocbf["hocbf_mean_violation"]
                ),
                "cbf_hocbf_raw_max_abs_residual": float(
                    raw_hocbf["hocbf_max_abs_residual"]
                ),
                "cbf_hocbf_reward_violation": float(hocbf_reward_violation),
                "cbf_hocbf_reward_penalty": float(hocbf_reward_penalty),
                "cbf_hocbf_reward_lambda": float(self.hocbf_reward_lambda),
                "cbf_hocbf_reward_scale": float(self.hocbf_reward_scale),
                "cbf_hocbf_reward_margin": float(self.hocbf_reward_margin),
                "cbf_psi1_gain": float(self.psi1_gain),
                "cbf_k0": float(self.k0),
                "cbf_k1": float(self.k1),
                "cbf_max_constraint_violation_safe": float(
                    record["max_constraint_violation_safe"]
                ),
                "cbf_max_constraint_violation_raw": float(
                    record["max_constraint_violation_raw"]
                ),
                "cbf_max_constraint_violation_box": float(
                    record["max_constraint_violation_box"]
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
