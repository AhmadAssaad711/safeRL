"""Margin-hinged state cost over the HOCBF barriers.

The cost is evaluated on a state, not on an action::

    C(s) = max_i { lambda_h,c * [delta_h,c - h_i]_+^2
                 + lambda_psi,c * [delta_psi,c - psi_1,i]_+^2 }

with ``psi_1,i = h_dot_i + psi1_gain * h_i`` and the class ``c`` of barrier
``i`` being either ``vehicle`` or ``boundary``.

Two design points, both deliberate:

* **Positive margins.** ``delta_h`` and ``delta_psi`` start the feedback before
  the corresponding condition is violated, so the policy is pushed away from
  poor clearance-and-closing-speed combinations rather than only punished once
  ``h`` or ``psi_1`` has already gone negative. They are also what keeps a
  ``max`` aggregation usable: ``C`` is exactly zero whenever every barrier sits
  outside both margins, so the term is not a constant per-step tax in safe
  states.
* **Separate normalization per class.** Vehicle ``h`` is the dimensionless
  relative-ellipse value and boundary ``h`` is a clearance in metres; measured
  on a trained nominal policy their per-step minima have medians of 1.553 and
  0.000 respectively. One shared coefficient cannot make those commensurate,
  so each class carries its own margin and coefficient.

``lambda_*`` are multiplicative coefficients rather than divisors: a previous
formulation divided by a scale ``sigma`` inside the square, and
``lambda = 1 / sigma**2`` is the same function with one fewer inversion to
reason about when tuning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

VEHICLE_CLASS = "vehicle"
BOUNDARY_CLASS = "boundary"

# Measured starting point; see docs/hocbf_reward_investigation.md. Vehicle
# values are in relative-ellipse units, boundary values in metres (and m/s).
DEFAULT_DELTA_H = (1.0, 0.5)
DEFAULT_LAMBDA_H = (1.0, 4.0)
DEFAULT_DELTA_PSI1 = (1.0, 0.5)
DEFAULT_LAMBDA_PSI1 = (0.25, 1.0)
DEFAULT_PSI1_GAIN = 2.3
DEFAULT_BOUNDARY_COUNT = 2


def _pair(value: Any, name: str) -> tuple[float, float]:
    values = tuple(float(item) for item in np.asarray(value, dtype=float).reshape(-1))
    if len(values) != 2:
        raise ValueError(f"{name} must hold exactly two values (vehicle, boundary)")
    return values[0], values[1]


@dataclass(frozen=True)
class StateCostParameters:
    """Resolved coefficients of the state cost.

    Every pair is ordered ``(vehicle, boundary)``.
    """

    lambda_s: float = 0.0
    delta_h: tuple[float, float] = DEFAULT_DELTA_H
    lambda_h: tuple[float, float] = DEFAULT_LAMBDA_H
    delta_psi1: tuple[float, float] = DEFAULT_DELTA_PSI1
    lambda_psi1: tuple[float, float] = DEFAULT_LAMBDA_PSI1
    psi1_gain: float = DEFAULT_PSI1_GAIN
    boundary_count: int = DEFAULT_BOUNDARY_COUNT

    def __post_init__(self) -> None:
        object.__setattr__(self, "lambda_s", float(self.lambda_s))
        object.__setattr__(self, "psi1_gain", float(self.psi1_gain))
        object.__setattr__(self, "boundary_count", int(self.boundary_count))
        for name in ("delta_h", "lambda_h", "delta_psi1", "lambda_psi1"):
            object.__setattr__(self, name, _pair(getattr(self, name), name))

        if not math.isfinite(self.lambda_s) or self.lambda_s < 0.0:
            raise ValueError("lambda_s must be finite and non-negative")
        if not math.isfinite(self.psi1_gain) or self.psi1_gain <= 0.0:
            raise ValueError("psi1_gain must be finite and positive")
        if self.boundary_count < 0:
            raise ValueError("boundary_count must be non-negative")
        for name in ("delta_h", "delta_psi1"):
            for value in getattr(self, name):
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} entries must be finite and >= 0")
        for name in ("lambda_h", "lambda_psi1"):
            for value in getattr(self, name):
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} entries must be finite and >= 0")

    @property
    def enabled(self) -> bool:
        return self.lambda_s > 0.0

    def as_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe record for signatures and manifests."""

        return {
            "lambda_s": self.lambda_s,
            "psi1_gain": self.psi1_gain,
            "boundary_count": self.boundary_count,
            "delta_h_vehicle": self.delta_h[0],
            "delta_h_boundary": self.delta_h[1],
            "lambda_h_vehicle": self.lambda_h[0],
            "lambda_h_boundary": self.lambda_h[1],
            "delta_psi1_vehicle": self.delta_psi1[0],
            "delta_psi1_boundary": self.delta_psi1[1],
            "lambda_psi1_vehicle": self.lambda_psi1[0],
            "lambda_psi1_boundary": self.lambda_psi1[1],
        }


def _class_vector(
    total: int, boundary_count: int, vehicle_value: float, boundary_value: float
) -> np.ndarray:
    values = np.full(int(total), float(vehicle_value), dtype=float)
    if boundary_count > 0:
        values[-int(boundary_count):] = float(boundary_value)
    return values


def barrier_state_cost(
    h: Sequence[float] | np.ndarray,
    h_dot: Sequence[float] | np.ndarray,
    params: StateCostParameters,
) -> tuple[float, dict[str, Any]]:
    """Return ``C(s)`` and its diagnostics for one constraint system.

    ``h`` and ``h_dot`` are the per-barrier arrays produced by
    ``build_cbf_action_constraints``: vehicle rows first, then the road
    boundary rows, which is what ``params.boundary_count`` splits on.
    """

    h_values = np.asarray(h, dtype=float).reshape(-1)
    h_dot_values = np.asarray(h_dot, dtype=float).reshape(-1)
    if h_values.size == 0:
        raise ValueError("state cost requires at least one barrier")
    if h_dot_values.size != h_values.size:
        # A silently zeroed safety term would be worse than a hard failure;
        # h_dot is missing only on the non-batch geometry path.
        raise ValueError(
            "state cost requires one h_dot per barrier; got "
            f"{h_dot_values.size} for {h_values.size} barriers"
        )
    boundary_count = min(int(params.boundary_count), int(h_values.size))

    psi1 = h_dot_values + float(params.psi1_gain) * h_values
    delta_h = _class_vector(h_values.size, boundary_count, *params.delta_h)
    lambda_h = _class_vector(h_values.size, boundary_count, *params.lambda_h)
    delta_psi1 = _class_vector(h_values.size, boundary_count, *params.delta_psi1)
    lambda_psi1 = _class_vector(h_values.size, boundary_count, *params.lambda_psi1)

    h_hinge = np.maximum(delta_h - h_values, 0.0)
    psi1_hinge = np.maximum(delta_psi1 - psi1, 0.0)
    h_term = lambda_h * h_hinge**2
    psi1_term = lambda_psi1 * psi1_hinge**2
    per_barrier = h_term + psi1_term

    index = int(np.argmax(per_barrier))
    is_boundary = boundary_count > 0 and index >= h_values.size - boundary_count
    diagnostics = {
        "cost": float(per_barrier[index]),
        "h_term": float(h_term[index]),
        "psi1_term": float(psi1_term[index]),
        "argmax_index": index,
        "argmax_class": BOUNDARY_CLASS if is_boundary else VEHICLE_CLASS,
        "argmax_h": float(h_values[index]),
        "argmax_psi1": float(psi1[index]),
        "barrier_count": int(h_values.size),
        "boundary_count": int(boundary_count),
    }
    return float(per_barrier[index]), diagnostics


def state_cost_parameters_from_mapping(
    config: Mapping[str, Any] | None,
) -> StateCostParameters:
    """Build parameters from a flat mapping such as a stored run config."""

    if not config:
        return StateCostParameters()
    return StateCostParameters(
        lambda_s=float(config.get("lambda_s", 0.0)),
        delta_h=(
            float(config.get("delta_h_vehicle", DEFAULT_DELTA_H[0])),
            float(config.get("delta_h_boundary", DEFAULT_DELTA_H[1])),
        ),
        lambda_h=(
            float(config.get("lambda_h_vehicle", DEFAULT_LAMBDA_H[0])),
            float(config.get("lambda_h_boundary", DEFAULT_LAMBDA_H[1])),
        ),
        delta_psi1=(
            float(config.get("delta_psi1_vehicle", DEFAULT_DELTA_PSI1[0])),
            float(config.get("delta_psi1_boundary", DEFAULT_DELTA_PSI1[1])),
        ),
        lambda_psi1=(
            float(config.get("lambda_psi1_vehicle", DEFAULT_LAMBDA_PSI1[0])),
            float(config.get("lambda_psi1_boundary", DEFAULT_LAMBDA_PSI1[1])),
        ),
        psi1_gain=float(config.get("psi1_gain", DEFAULT_PSI1_GAIN)),
        boundary_count=int(config.get("boundary_count", DEFAULT_BOUNDARY_COUNT)),
    )
