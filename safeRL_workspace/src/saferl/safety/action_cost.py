"""Action-distance safety cost over the HOCBF-feasible action set.

The cost is evaluated on a state-action pair, not on a state::

    c_a(s, a) = 1/2 * || a - Pi_{U_safe(s)}(a) ||^2
              = 1/2 * d^2( a, U_safe(s) )

where ``U_safe(s) = {u : A(s) u <= b(s), u_min <= u <= u_max}`` is the
HOCBF-feasible action polytope and ``Pi`` its Euclidean projection.

This module owns only the *cost definition*. The geometry and the projection
itself stay in their existing compatibility location
(``scripts.common.cbf_projection``), exactly as ``state_cost`` takes the
per-barrier ``h``/``h_dot`` arrays rather than rebuilding the constraint
system. Callers pass the raw action and the already-projected safe action.

**The projection is a measurement here, not a filter.** In the CMDP-Action
variant the executed action is the raw policy action; ``Pi`` is evaluated only
to score how incompatible that action was with the safe set. Nothing about the
transition changes. That is what separates this cost from the detached actor
loss, which uses the same quantity as a regression target instead.

Two properties make ``c_a`` cheaper to learn than a state-only cost:

* it is a deterministic, known function of ``(s, a)``, so the immediate term
  of the cost TD residual carries no transition noise; and
* it is zero exactly on the feasible set, so it is not a per-step tax in safe
  states -- the same property the state cost gets from its margins.

**Infeasible states.** When ``U_safe(s)`` is empty there is no mathematically
meaningful projection distance, so::

    c_a(s, a) = 1/2 ||a - Pi_{U_safe(s)}(a)||^2   if U_safe(s) != {}
              = c_infeas                          if U_safe(s) == {}

and ``c_infeas`` is a fixed conservative constant. The distance to the solver's
fallback action is *not* used for this: the fallback is a solver artifact, and
``1/2 ||a - a_fallback||^2`` can come out small precisely on the states that
are maximally problematic. It remains available as a diagnostic
(``fallback_cost_mean``) but never scores the cost.

The default ``c_infeas`` is computed from the physical action bounds, so it is
an upper bound on any attainable in-box distance and cannot be undercut by a
feasible sample. Scoring infeasible states as *zero* is actively hazardous --
it makes the cheapest way to satisfy the constraint be to drive into states
where no safe action exists -- and is retained only for parity against the
masked actor losses.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

# How ``c_a`` is defined on states whose no-slack safe set is empty.
#
# "box_bound" score a fixed conservative constant computed from the physical
#             action bounds: the largest squared distance attainable inside the
#             box. The default, and the only rule that cannot be undercut on a
#             maximally problematic state.
# "fallback"  score the distance to the solver's labelled fallback action.
#             Diagnostic only: the fallback is a solver artifact, so this can
#             be small exactly where the state is worst.
# "zero"      score zero. Matches how the existing actor losses *mask* those
#             samples, and is retained only for parity experiments against
#             them. HAZARDOUS as a CMDP cost; see the module docstring.
BOX_BOUND_RULE = "box_bound"
FALLBACK_RULE = "fallback"
ZERO_RULE = "zero"
SUPPORTED_INFEASIBLE_RULES = (BOX_BOUND_RULE, FALLBACK_RULE, ZERO_RULE)
HAZARDOUS_INFEASIBLE_RULES = (ZERO_RULE,)
# Rules that score an infeasible state from a solver-dependent quantity rather
# than from a fixed conservative constant.
DIAGNOSTIC_ONLY_INFEASIBLE_RULES = (FALLBACK_RULE,)


def _as_2d(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a batch of action vectors")
    return array


def box_bound_cost(action_low: Any, action_high: Any) -> float:
    """Return the largest ``c_a`` attainable inside the action box.

    Used by the ``"box_bound"`` infeasible rule. The extreme separation inside
    an axis-aligned box is corner to opposite corner, so the bound is
    ``1/2 * ||high - low||^2``.
    """

    low = np.asarray(action_low, dtype=float).reshape(-1)
    high = np.asarray(action_high, dtype=float).reshape(-1)
    if low.size != high.size:
        raise ValueError("action_low and action_high must have equal width")
    if np.any(high < low):
        raise ValueError("action_high must be >= action_low componentwise")
    return 0.5 * float(np.square(high - low).sum())


def action_distance_cost(
    raw_actions: Any,
    safe_actions: Any,
    *,
    feasible: Optional[Any] = None,
    infeasible_rule: str = BOX_BOUND_RULE,
    action_low: Optional[Any] = None,
    action_high: Optional[Any] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return per-sample ``c_a`` and its diagnostics.

    ``raw_actions`` is what the policy proposed and ``safe_actions`` is its
    projection onto ``U_safe(s)``; on infeasible samples the latter is whatever
    labelled fallback the solver produced. Both are physical actions, already
    box-clipped by the caller if that is the intended reference -- this
    function does not clip, because whether actuator saturation counts as a
    safety cost is the caller's decision, not the cost's.

    The cost is returned *unclipped and unnormalized*: the ``(1 - gamma_c)``
    objective scaling and any diagnostic clipping belong to the CMDP
    configuration, not to the risk definition.
    """

    rule = str(infeasible_rule).strip().lower()
    if rule not in SUPPORTED_INFEASIBLE_RULES:
        raise ValueError(
            f"Unsupported infeasible_rule {infeasible_rule!r}; expected one of "
            f"{SUPPORTED_INFEASIBLE_RULES}"
        )

    raw = _as_2d(raw_actions, name="raw_actions")
    safe = _as_2d(safe_actions, name="safe_actions")
    if raw.shape != safe.shape:
        raise ValueError(
            "raw_actions and safe_actions must have the same shape "
            f"({raw.shape} != {safe.shape})"
        )

    delta = raw - safe
    measured = 0.5 * np.square(delta).sum(axis=1)
    cost = measured.copy()

    if feasible is None:
        feasible_mask = np.ones(raw.shape[0], dtype=bool)
    else:
        feasible_mask = np.asarray(feasible, dtype=bool).reshape(-1)
        if feasible_mask.size != raw.shape[0]:
            raise ValueError(
                "feasible must supply one flag per sample "
                f"({feasible_mask.size} != {raw.shape[0]})"
            )

    infeasible_count = int((~feasible_mask).sum())
    bound: Optional[float] = None
    if rule == BOX_BOUND_RULE:
        # Required unconditionally rather than only when an infeasible sample
        # appears: a latent failure on the first infeasible batch mid-training
        # is strictly worse than failing on the first call.
        if action_low is None or action_high is None:
            raise ValueError(
                "the 'box_bound' infeasible rule requires action_low and "
                "action_high"
            )
        bound = box_bound_cost(action_low, action_high)
        cost = np.where(feasible_mask, cost, bound)
    elif rule == ZERO_RULE:
        cost = np.where(feasible_mask, cost, 0.0)
    # FALLBACK_RULE keeps the measured distance to the fallback action as-is.

    if not np.all(np.isfinite(cost)):
        # A silently NaN cost would poison the cost critic and the dual
        # variable without ever failing loudly.
        raise ValueError("action-distance cost produced a non-finite value")

    feasible_costs = cost[feasible_mask]
    fallback_costs = measured[~feasible_mask]
    diagnostics = {
        # The solver-fallback distance on infeasible samples. Reported so the
        # discarded rule stays observable, never used to score the cost.
        "fallback_cost_mean": (
            float(fallback_costs.mean()) if fallback_costs.size else 0.0
        ),
        "cost_mean": float(cost.mean()) if cost.size else 0.0,
        "cost_max": float(cost.max()) if cost.size else 0.0,
        "correction_norm_mean": (
            float(np.linalg.norm(delta, axis=1).mean()) if delta.size else 0.0
        ),
        "active_fraction": float((cost > 0.0).mean()) if cost.size else 0.0,
        "feasible_cost_mean": (
            float(feasible_costs.mean()) if feasible_costs.size else 0.0
        ),
        "infeasible_count": infeasible_count,
        "infeasible_fraction": (
            float(infeasible_count) / float(raw.shape[0]) if raw.shape[0] else 0.0
        ),
        "infeasible_rule": rule,
        "sample_count": int(raw.shape[0]),
    }
    if bound is not None:
        diagnostics["infeasible_cost_value"] = float(bound)
    return cost.astype(np.float64, copy=False), diagnostics


def action_cost_scale_from_reference(reference_correction_norm: float) -> float:
    """Return the ``c_a`` a given correction norm produces.

    Mirrors ``StateCostParameters.scale_from_reference``: it turns a physically
    meaningful reference situation into the cost scale it scores, so a cost
    limit can be quoted as "a sustained correction of this many m/s^2" rather
    than as a bare number.
    """

    norm = float(reference_correction_norm)
    if not math.isfinite(norm) or norm < 0.0:
        raise ValueError("reference_correction_norm must be finite and >= 0")
    return 0.5 * norm * norm
