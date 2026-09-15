"""Contract tests for the margin-hinged HOCBF state cost."""

from __future__ import annotations

import math

import numpy as np
import pytest

from saferl.safety.state_cost import (
    BOUNDARY_CLASS,
    VEHICLE_CLASS,
    StateCostParameters,
    barrier_state_cost,
    state_cost_parameters_from_mapping,
)

# One vehicle row plus the two boundary rows, with deliberately different
# margins and coefficients per class so a mix-up cannot pass.
PARAMS = StateCostParameters(
    lambda_s=1.0,
    delta_h=(1.0, 0.5),
    sigma_h=(1.0, 0.5),
    delta_psi1=(1.0, 0.5),
    sigma_psi1=(2.0, 1.0),
    psi1_gain=2.0,
    boundary_count=2,
)


def _cost(h, h_dot, params=PARAMS):
    return barrier_state_cost(np.asarray(h), np.asarray(h_dot), params)


def test_safe_state_costs_exactly_zero():
    """Every barrier outside both margins must give C = 0, not a constant tax."""

    # psi1 = h_dot + 2*h, so these are 20.0, 12.0 and 12.0: all above delta.
    cost, diagnostics = _cost([10.0, 6.0, 6.0], [0.0, 0.0, 0.0])
    assert cost == 0.0
    assert diagnostics["h_term"] == 0.0
    assert diagnostics["psi1_term"] == 0.0


def test_hinge_activates_only_inside_the_margin():
    """The h hinge is zero at the margin and positive just inside it."""

    at_margin, _ = _cost([1.0, 6.0, 6.0], [10.0, 0.0, 0.0])
    inside, _ = _cost([0.5, 6.0, 6.0], [10.0, 0.0, 0.0])
    assert at_margin == 0.0
    assert inside > 0.0


def test_cost_is_monotone_in_h_and_psi1():
    """Reducing clearance or first-order margin may never reduce the cost."""

    previous = -math.inf
    for h in (1.0, 0.8, 0.4, 0.0, -0.4):
        cost, _ = _cost([h, 6.0, 6.0], [10.0, 0.0, 0.0])
        assert cost >= previous
        previous = cost

    previous = -math.inf
    for h_dot in (2.0, 0.0, -1.0, -3.0):
        # h fixed above its margin, so only the psi1 hinge can move.
        cost, _ = _cost([2.0, 6.0, 6.0], [h_dot, 0.0, 0.0])
        assert cost >= previous
        previous = cost


def test_aggregation_is_max_not_sum():
    """Two equally bad barriers cost the same as one; a sum would double it."""

    one, _ = _cost([0.0, 6.0, 6.0], [0.0, 0.0, 0.0])
    two, _ = _cost([0.0, 0.0, 6.0, 6.0], [0.0, 0.0, 0.0, 0.0])
    assert two == pytest.approx(one)


def test_vehicle_and_boundary_are_normalized_separately():
    """The same numeric h scores differently for a vehicle and a boundary."""

    # Vehicle h = 0.5 -> hinge 0.5, lambda 1.0   -> 0.25
    # Boundary h = 0.5 -> at its margin          -> 0.0
    vehicle_cost, vehicle_diag = _cost([0.5, 6.0, 6.0], [10.0, 0.0, 0.0])
    boundary_cost, boundary_diag = _cost([10.0, 0.5, 6.0], [0.0, 10.0, 0.0])
    assert vehicle_cost == pytest.approx(0.25)
    assert vehicle_diag["argmax_class"] == VEHICLE_CLASS
    assert boundary_cost == 0.0

    # Boundary h = 0.0 -> hinge 0.5, lambda 4.0 -> 1.0, and it wins the max.
    cost, diagnostics = _cost([0.5, 0.0, 6.0], [10.0, 10.0, 0.0])
    assert cost == pytest.approx(1.0)
    assert diagnostics["argmax_class"] == BOUNDARY_CLASS
    assert diagnostics["argmax_index"] == 1


def test_psi1_uses_the_configured_gain():
    """psi1 = h_dot + gain*h, so the gain changes which states are penalized."""

    low_gain = StateCostParameters(
        lambda_s=1.0,
        delta_h=(0.0, 0.0),
        sigma_h=(1.0, 1.0),
        delta_psi1=(1.0, 1.0),
        sigma_psi1=(1.0, 1.0),
        psi1_gain=0.5,
        boundary_count=0,
    )
    high_gain = StateCostParameters(
        lambda_s=1.0,
        delta_h=(0.0, 0.0),
        sigma_h=(1.0, 1.0),
        delta_psi1=(1.0, 1.0),
        sigma_psi1=(1.0, 1.0),
        psi1_gain=4.0,
        boundary_count=0,
    )
    # h = 0.5, h_dot = 0 -> psi1 is 0.25 under the low gain and 2.0 under the
    # high one, so only the low gain is inside the margin.
    assert _cost([0.5], [0.0], low_gain)[0] > 0.0
    assert _cost([0.5], [0.0], high_gain)[0] == 0.0


def test_terms_add_within_a_barrier():
    """Both hinges contribute to the same barrier's cost."""

    cost, diagnostics = _cost([0.0, 6.0, 6.0], [0.0, 0.0, 0.0])
    # h hinge: (1-0)^2 * 1.0 = 1.0; psi1 = 0 -> (1-0)^2 * 0.25 = 0.25
    assert diagnostics["h_term"] == pytest.approx(1.0)
    assert diagnostics["psi1_term"] == pytest.approx(0.25)
    assert cost == pytest.approx(1.25)


def test_lambda_s_is_an_exact_off_switch():
    off = StateCostParameters(lambda_s=0.0)
    assert off.enabled is False
    assert StateCostParameters(lambda_s=0.1).enabled is True


def test_missing_h_dot_raises_rather_than_scoring_zero():
    with pytest.raises(ValueError):
        barrier_state_cost(np.asarray([1.0, 2.0, 3.0]), np.asarray([]), PARAMS)
    with pytest.raises(ValueError):
        barrier_state_cost(np.asarray([]), np.asarray([]), PARAMS)


@pytest.mark.parametrize(
    "field, value",
    [
        ("lambda_s", -1.0),
        ("psi1_gain", 0.0),
        ("delta_h", (-1.0, 0.5)),
        ("sigma_h", (1.0, -2.0)),
        ("delta_psi1", (float("nan"), 0.5)),
        ("sigma_psi1", (1.0, float("inf"))),
    ],
)
def test_invalid_parameters_are_rejected(field, value):
    with pytest.raises(ValueError):
        StateCostParameters(**{field: value})


def test_pairs_must_hold_two_values():
    with pytest.raises(ValueError):
        StateCostParameters(delta_h=(1.0, 2.0, 3.0))


def test_round_trip_through_a_flat_mapping():
    restored = state_cost_parameters_from_mapping(PARAMS.as_dict())
    assert restored == PARAMS

