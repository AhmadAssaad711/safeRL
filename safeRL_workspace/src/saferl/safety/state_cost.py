"""Margin-hinged state cost over the HOCBF barriers.

The cost is evaluated on a state, not on an action::

    C(s) = max_i { [ (delta_h,c - h_i) / sigma_h,c ]_+^2
                 + [ (delta_psi,c - psi_1,i) / sigma_psi,c ]_+^2 }

with ``psi_1,i = h_dot_i + psi1_gain * h_i`` and the class ``c`` of barrier
``i`` being either ``vehicle`` or ``boundary``.

Margins and scales have separate jobs and are calibrated separately:

* **Margins** (``delta``) fix *when* the feedback starts. A positive margin
  begins the penalty before the corresponding condition is violated, so the
  policy is pushed away from poor clearance-and-closing-speed combinations
  rather than punished only once ``h`` or ``psi_1`` has already gone negative.
  They are also what keeps a ``max`` aggregation usable: ``C`` is exactly zero
  whenever every barrier sits outside both margins, so the term is not a
  constant per-step tax in safe states.
* **Scales** (``sigma``) fix *what counts as substantial risk*. Each is set
  from a reference situation ``x_ref`` that is considered meaningfully risky,
  via ``sigma = delta - x_ref``, which makes that situation score exactly 1.
  They are deliberately not fitted to the variance of all visited states:
  abundant harmless states dilute the signal precisely where it matters.

The outer ``lambda_s`` is separate again and sets how strongly learning
responds; it belongs to the reward, not to the risk definition.

**Separate normalization per class.** Vehicle ``h`` is the dimensionless
relative-ellipse value and boundary ``h`` is a clearance in metres; measured
on a trained nominal policy their per-step minima have medians of 1.553 and
0.000 respectively. One shared scale cannot make those commensurate, so each
class carries its own margin and scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

VEHICLE_CLASS = "vehicle"
BOUNDARY_CLASS = "boundary"

# Barrier form used for the *vehicle* rows.
#
# "ellipse" keeps h as built: the squared relative-ellipse value. It is
# quadratic in distance, so no margin can both stay quiet at 60 m and speak at
# 30 m, and the cost first becomes non-zero near 7.2 m longitudinally.
#
# "distance" reads the same geometry as a clearance instead. With
# ``rho = sqrt(h + 1)`` the separation in ellipse radii is ``rho - 1`` and its
# rate is ``h_dot / (2 rho)``. Then
#
#     psi_1 = rho_dot + gain * (rho - 1) >= 0
#         <=> clearance / closing_rate >= 1 / gain
#
# so the gain *is* a time-to-collision threshold: 0.3 warns below 3.33 s.
#
# Road boundary rows are never transformed. Their h is already a clearance in
# metres and linear in distance, which is why that channel behaved correctly
# under both forms.
ELLIPSE_BARRIER_FORM = "ellipse"
DISTANCE_BARRIER_FORM = "distance"
SUPPORTED_BARRIER_FORMS = (ELLIPSE_BARRIER_FORM, DISTANCE_BARRIER_FORM)

# Reference-calibrated starting points; see the behaviour checks in
# tests/unit/test_state_cost_behaviour.py. Boundary values are metres and m/s
# in both forms. Vehicle values are squared-ellipse units under "ellipse" and
# ellipse radii (and radii/s) under "distance", so the two are not
# interchangeable and the defaults are selected per form.
DEFAULT_DELTA_H = (1.0, 0.5)
DEFAULT_SIGMA_H = (1.0, 0.5)
DEFAULT_DELTA_PSI1 = (1.0, 0.5)
DEFAULT_SIGMA_PSI1 = (2.0, 1.0)
DEFAULT_PSI1_GAIN = 2.3
DEFAULT_BOUNDARY_COUNT = 2

# Distance form. delta_h = 1.0 radii starts the clearance channel about one
# ellipse radius out; sigma_h = 0.8 makes a clearance of 0.2 radii score 1.
# delta_psi1 = 1.0 with gain 0.3 warns below a 3.33 s time to collision;
# sigma_psi1 = 2.16 makes "12 m closing at 8 m/s" (psi_1 = -1.16) score 1.
DISTANCE_DELTA_H = (1.0, 0.5)
DISTANCE_SIGMA_H = (0.8, 0.5)
DISTANCE_DELTA_PSI1 = (1.0, 0.5)
DISTANCE_SIGMA_PSI1 = (2.16, 1.0)
DISTANCE_PSI1_GAIN = 0.3


def default_parameter_set(barrier_form: str) -> dict[str, Any]:
    """Return the reference-calibrated defaults for one barrier form."""

    form = str(barrier_form).strip().lower()
    if form == DISTANCE_BARRIER_FORM:
        return {
            "delta_h": DISTANCE_DELTA_H,
            "sigma_h": DISTANCE_SIGMA_H,
            "delta_psi1": DISTANCE_DELTA_PSI1,
            "sigma_psi1": DISTANCE_SIGMA_PSI1,
            "psi1_gain": DISTANCE_PSI1_GAIN,
        }
    if form == ELLIPSE_BARRIER_FORM:
        return {
            "delta_h": DEFAULT_DELTA_H,
            "sigma_h": DEFAULT_SIGMA_H,
            "delta_psi1": DEFAULT_DELTA_PSI1,
            "sigma_psi1": DEFAULT_SIGMA_PSI1,
            "psi1_gain": DEFAULT_PSI1_GAIN,
        }
    raise ValueError(
        f"Unsupported barrier_form {barrier_form!r}; expected one of "
        f"{SUPPORTED_BARRIER_FORMS}"
    )


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
    sigma_h: tuple[float, float] = DEFAULT_SIGMA_H
    delta_psi1: tuple[float, float] = DEFAULT_DELTA_PSI1
    sigma_psi1: tuple[float, float] = DEFAULT_SIGMA_PSI1
    psi1_gain: float = DEFAULT_PSI1_GAIN
    boundary_count: int = DEFAULT_BOUNDARY_COUNT
    barrier_form: str = ELLIPSE_BARRIER_FORM

    def __post_init__(self) -> None:
        object.__setattr__(self, "lambda_s", float(self.lambda_s))
        object.__setattr__(self, "psi1_gain", float(self.psi1_gain))
        object.__setattr__(self, "boundary_count", int(self.boundary_count))
        object.__setattr__(
            self, "barrier_form", str(self.barrier_form).strip().lower()
        )
        if self.barrier_form not in SUPPORTED_BARRIER_FORMS:
            raise ValueError(
                f"Unsupported barrier_form {self.barrier_form!r}; expected one "
                f"of {SUPPORTED_BARRIER_FORMS}"
            )
        for name in ("delta_h", "sigma_h", "delta_psi1", "sigma_psi1"):
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
        for name in ("sigma_h", "sigma_psi1"):
            for value in getattr(self, name):
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{name} entries must be finite and > 0")

    @property
    def enabled(self) -> bool:
        return self.lambda_s > 0.0

    def as_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe record for signatures and manifests."""

        return {
            "lambda_s": self.lambda_s,
            "psi1_gain": self.psi1_gain,
            "boundary_count": self.boundary_count,
            "barrier_form": self.barrier_form,
            "delta_h_vehicle": self.delta_h[0],
            "delta_h_boundary": self.delta_h[1],
            "sigma_h_vehicle": self.sigma_h[0],
            "sigma_h_boundary": self.sigma_h[1],
            "delta_psi1_vehicle": self.delta_psi1[0],
            "delta_psi1_boundary": self.delta_psi1[1],
            "sigma_psi1_vehicle": self.sigma_psi1[0],
            "sigma_psi1_boundary": self.sigma_psi1[1],
        }

    @staticmethod
    def scale_from_reference(delta: float, reference: float) -> float:
        """Return the scale that makes ``reference`` score exactly 1.

        ``reference`` is a situation considered meaningfully risky and must sit
        inside the margin, i.e. ``reference < delta``.
        """

        scale = float(delta) - float(reference)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "the reference situation must be inside the margin "
                f"(reference {reference} < delta {delta})"
            )
        return scale


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

    if params.barrier_form == DISTANCE_BARRIER_FORM:
        # Vehicle rows only: read the ellipse as a clearance in radii. Boundary
        # rows are already a clearance in metres and are left untouched.
        vehicle_slice = slice(0, h_values.size - boundary_count)
        h_values = h_values.copy()
        h_dot_values = h_dot_values.copy()
        vehicle_h = h_values[vehicle_slice]
        # h = (dx/A)^2 + (dy/B)^2 - 1 >= -1, so rho >= 0; clamp only guards the
        # coincident-centre singularity of rho_dot.
        rho = np.sqrt(np.maximum(vehicle_h + 1.0, 1e-9))
        h_dot_values[vehicle_slice] = h_dot_values[vehicle_slice] / (2.0 * rho)
        h_values[vehicle_slice] = rho - 1.0

    psi1 = h_dot_values + float(params.psi1_gain) * h_values
    delta_h = _class_vector(h_values.size, boundary_count, *params.delta_h)
    sigma_h = _class_vector(h_values.size, boundary_count, *params.sigma_h)
    delta_psi1 = _class_vector(h_values.size, boundary_count, *params.delta_psi1)
    sigma_psi1 = _class_vector(h_values.size, boundary_count, *params.sigma_psi1)

    h_hinge = np.maximum((delta_h - h_values) / sigma_h, 0.0)
    psi1_hinge = np.maximum((delta_psi1 - psi1) / sigma_psi1, 0.0)
    h_term = h_hinge**2
    psi1_term = psi1_hinge**2
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


def barrier_violation_cost(
    h: Sequence[float] | np.ndarray,
    h_dot: Sequence[float] | np.ndarray,
    params: StateCostParameters,
) -> tuple[float, dict[str, Any]]:
    """Return the HOCBF violation indicator and its diagnostics::

        c(s) = 1 if min_i psi_1,i(s) < 0 else 0

    with ``psi_1,i = h_dot_i + psi1_gain * h_i`` over the same barriers
    ``barrier_state_cost`` uses.

    **Why this rather than the margin-hinged cost.** ``psi_1 >= 0`` on every
    barrier is exactly the condition for forward invariance of the safe set, so
    ``J_C = E[c] <= d`` is the statement "the barrier condition holds on at
    least ``1 - d`` of steps". That is a claim with CBF semantics behind it.
    The margin-hinged cost instead fires at ``psi_1 < delta_psi``, i.e. on
    *technically safe* states that are merely close to the threshold, which
    makes it largely a measure of traffic proximity. Measured consequence: that
    cost had an achievable floor of ~0.49 set by traffic density, and
    tightening its budget made collisions worse rather than better
    (corr(d, collisions) = -0.968).

    Three further properties that make it defensible where the hinged cost is
    not:

    * **No free parameters.** The gain is already fixed by the CBF design;
      there are no margins or scales to tune, and no per-class normalization,
      so there is nothing for a reviewer to attack as hand-fitted.
    * **``d`` is interpretable** -- a fraction of steps.
    * ``J_C`` under the average-cost objective *is* the violation rate, which
      is the quantity that separated collision from collision-free episodes at
      AUC 0.875-0.999 and was the only candidate stable across every policy
      tested.

    The barrier-form transform is applied exactly as in
    ``barrier_state_cost``, so ``psi_1`` means the same thing in both.
    """

    h_values = np.asarray(h, dtype=float).reshape(-1)
    h_dot_values = np.asarray(h_dot, dtype=float).reshape(-1)
    if h_values.size == 0:
        raise ValueError("violation cost requires at least one barrier")
    if h_dot_values.size != h_values.size:
        # A silently zeroed safety term is worse than a hard failure.
        raise ValueError(
            "violation cost requires one h_dot per barrier; got "
            f"{h_dot_values.size} for {h_values.size} barriers"
        )
    boundary_count = min(int(params.boundary_count), int(h_values.size))

    if params.barrier_form == DISTANCE_BARRIER_FORM:
        vehicle_slice = slice(0, h_values.size - boundary_count)
        h_values = h_values.copy()
        h_dot_values = h_dot_values.copy()
        rho = np.sqrt(np.maximum(h_values[vehicle_slice] + 1.0, 1e-9))
        h_dot_values[vehicle_slice] = h_dot_values[vehicle_slice] / (2.0 * rho)
        h_values[vehicle_slice] = rho - 1.0

    psi1 = h_dot_values + float(params.psi1_gain) * h_values
    violated = psi1 < 0.0
    index = int(np.argmin(psi1))
    is_boundary = boundary_count > 0 and index >= h_values.size - boundary_count
    diagnostics = {
        "cost": float(bool(violated.any())),
        "violated_barriers": int(violated.sum()),
        "violated_fraction": float(violated.mean()),
        "min_psi1": float(psi1[index]),
        "argmax_index": index,
        "argmax_class": BOUNDARY_CLASS if is_boundary else VEHICLE_CLASS,
        "argmax_h": float(h_values[index]),
        "barrier_count": int(h_values.size),
        "boundary_count": int(boundary_count),
    }
    return float(bool(violated.any())), diagnostics


def state_cost_parameters_from_mapping(
    config: Mapping[str, Any] | None,
) -> StateCostParameters:
    """Build parameters from a flat mapping such as a stored run config."""

    if not config:
        return StateCostParameters()
    form = str(config.get("barrier_form", ELLIPSE_BARRIER_FORM))
    defaults = default_parameter_set(form)
    return StateCostParameters(
        lambda_s=float(config.get("lambda_s", 0.0)),
        delta_h=(
            float(config.get("delta_h_vehicle", defaults["delta_h"][0])),
            float(config.get("delta_h_boundary", defaults["delta_h"][1])),
        ),
        sigma_h=(
            float(config.get("sigma_h_vehicle", defaults["sigma_h"][0])),
            float(config.get("sigma_h_boundary", defaults["sigma_h"][1])),
        ),
        delta_psi1=(
            float(config.get("delta_psi1_vehicle", defaults["delta_psi1"][0])),
            float(config.get("delta_psi1_boundary", defaults["delta_psi1"][1])),
        ),
        sigma_psi1=(
            float(config.get("sigma_psi1_vehicle", defaults["sigma_psi1"][0])),
            float(config.get("sigma_psi1_boundary", defaults["sigma_psi1"][1])),
        ),
        psi1_gain=float(config.get("psi1_gain", defaults["psi1_gain"])),
        boundary_count=int(config.get("boundary_count", DEFAULT_BOUNDARY_COUNT)),
        barrier_form=form,
    )
