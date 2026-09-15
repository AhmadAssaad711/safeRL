"""Behaviour gate for the state cost, on real barrier geometry.

A safety signal is only worth amplifying if it warns early, separates
worsening situations from improving ones, and leaves ordinary driving alone.
These checks encode exactly that, against the real relative-ellipse barrier
rather than synthetic numbers, so a change to the geometry or to the cost has
to keep satisfying them:

    Far away, not approaching             -> almost zero
    Approaching fast, room to react       -> noticeable and increasing
    Close and still approaching           -> large
    Braking successfully, danger falling  -> decreasing, even while the gap shrinks

**Two of these do not hold for the current risk definition, by measurement.**
``h`` is the squared relative-ellipse value, so it grows quadratically with
distance and ``psi_1 = h_dot + gain*h`` inherits that position term: at 30 m
closing at 6 m/s, ``h = 33.7`` and ``psi_1 = 63.7``, both far outside any margin
that still means something near contact. The cost cannot become non-zero until
``h < delta_h``, about 7.2 m longitudinally at ``delta_h = 1``, so it is a
last-moment signal rather than an early warning, and it stays silent through an
entire braking recovery.

Those two checks are marked ``xfail(strict=True)`` rather than deleted: they are
the acceptance criteria, the failure is a property of the barrier's units and
not of the margins, and a future change to a distance-like barrier should flip
them to passing and fail this file loudly if it does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.common.cbf_geometry import batch_pairwise_hocbf_constraints
from saferl.safety.state_cost import (
    VEHICLE_CLASS,
    StateCostParameters,
    barrier_state_cost,
)

K0 = 4.0
K1 = 8.5
EPS_SIDE = 0.10
PSI1_GAIN = 2.3
ROAD_WIDTH = 10.2
EGO_HALF_WIDTH = 0.9
EGO_SPEED = 14.0

PARAMS = StateCostParameters(
    lambda_s=1.0,
    delta_h=(1.0, 0.5),
    sigma_h=(1.0, 0.5),
    delta_psi1=(1.0, 0.5),
    sigma_psi1=(2.0, 1.0),
    psi1_gain=PSI1_GAIN,
    boundary_count=2,
)


def _vehicle_barrier(dx: float, dy: float, closing: float, lateral: float = 0.0):
    """Return (h, h_dot) for one neighbour at a relative offset and rate.

    ``closing`` is positive when the gap is shrinking.
    """

    ego = {
        "x": 0.0,
        "y": 0.5 * ROAD_WIDTH,
        "vx": EGO_SPEED,
        "vy": 0.0,
        "ax": 0.0,
        "ay": 0.0,
        "width": 1.8,
        "length": 3.5,
    }
    neighbor = {
        "x": float(dx),
        "y": 0.5 * ROAD_WIDTH + float(dy),
        # A leader slower than the ego by `closing` closes the gap.
        "vx": EGO_SPEED - float(closing),
        "vy": float(lateral),
        "ax": 0.0,
        "ay": 0.0,
        "width": 1.8,
        "length": 3.5,
    }
    batch = batch_pairwise_hocbf_constraints(
        ego, [neighbor], eps_side=EPS_SIDE, k0=K0, k1=K1
    )
    return float(batch["h"][0]), float(batch["h_dot"][0])


def _cost_of_vehicle(dx: float, dy: float, closing: float, lateral: float = 0.0) -> float:
    """State cost for a single neighbour plus two comfortable boundaries."""

    h, h_dot = _vehicle_barrier(dx, dy, closing, lateral)
    # Ego centred, not drifting: both boundaries are far outside their margin.
    boundary_h = [0.5 * ROAD_WIDTH - EGO_HALF_WIDTH] * 2
    boundary_h_dot = [0.0, 0.0]
    cost, _diagnostics = barrier_state_cost(
        np.asarray([h, *boundary_h]),
        np.asarray([h_dot, *boundary_h_dot]),
        PARAMS,
    )
    return cost


def test_far_away_and_not_approaching_is_almost_zero():
    """Ordinary driving must not be taxed."""

    assert _cost_of_vehicle(dx=60.0, dy=0.0, closing=0.0) == pytest.approx(0.0)
    assert _cost_of_vehicle(dx=40.0, dy=0.0, closing=2.0) == pytest.approx(0.0)
    # Receding is never risk, however close.
    assert _cost_of_vehicle(dx=12.0, dy=0.0, closing=-6.0) == pytest.approx(0.0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "squared-ellipse h is 33.7 at 30 m, so no margin that is meaningful "
        "near contact can warn this early; the term first bites near 7.2 m"
    ),
)
def test_approaching_fast_with_room_is_noticeable_and_increasing():
    """The signal must arrive while there is still time to react."""

    approach = [
        _cost_of_vehicle(dx=distance, dy=0.0, closing=6.0)
        for distance in (30.0, 25.0, 20.0, 15.0)
    ]
    assert approach[0] > 0.0, "no warning at 30 m while closing at 6 m/s"
    assert all(
        later > earlier for earlier, later in zip(approach, approach[1:])
    ), f"not increasing as the gap closes: {approach}"


def test_close_and_still_approaching_is_large():
    """The worst case must dominate the ordinary one by a wide margin."""

    close = _cost_of_vehicle(dx=6.0, dy=0.0, closing=6.0)
    moderate = _cost_of_vehicle(dx=25.0, dy=0.0, closing=6.0)
    assert close > 1.0
    assert close > 5.0 * moderate


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the term is identically zero through this recovery: psi_1 rises "
        "3.1 -> 6.4 as braking takes effect, which is the right direction, but "
        "stays about 5x outside delta_psi1 the whole time"
    ),
)
def test_braking_successfully_decreases_the_cost_even_as_the_gap_shrinks():
    """Recovery must be rewarded while it is happening, not after it completes.

    The ego brakes over successive steps: the gap keeps shrinking, but the
    closing speed falls. A signal that only tracks clearance would keep rising
    here and would teach the policy that braking does not help.
    """

    recovery = [
        _cost_of_vehicle(dx=12.0, dy=0.0, closing=8.0),
        _cost_of_vehicle(dx=11.0, dy=0.0, closing=5.0),
        _cost_of_vehicle(dx=10.4, dy=0.0, closing=2.5),
        _cost_of_vehicle(dx=10.1, dy=0.0, closing=0.5),
    ]
    assert all(
        later < earlier for earlier, later in zip(recovery, recovery[1:])
    ), f"cost did not fall while braking succeeded: {recovery}"


def test_a_wall_approached_costs_more_than_a_wall_driven_parallel_to():
    """The boundary channel must react to drift, not only to position."""

    def wall_cost(distance_m: float, lateral_speed: float) -> float:
        # Left boundary at `distance_m`, ego drifting into it at
        # `lateral_speed`; the far neighbour and right boundary stay benign.
        h = [40.0, float(distance_m), ROAD_WIDTH - 2 * EGO_HALF_WIDTH - distance_m]
        h_dot = [0.0, -float(lateral_speed), float(lateral_speed)]
        cost, _ = barrier_state_cost(np.asarray(h), np.asarray(h_dot), PARAMS)
        return cost

    parallel = wall_cost(0.4, 0.0)
    drifting_in = wall_cost(0.4, 1.0)
    assert drifting_in > parallel
    # Leaving is not *better* than riding parallel, only no worse: the h
    # channel is rate-blind, and psi_1 is already outside its margin in both
    # cases, so the two score identically. Recorded rather than asserted away.
    assert wall_cost(0.4, -1.0) <= parallel


# ---------------------------------------------------------------------------
# The same four checks against the distance barrier form, where the gain is a
# time-to-collision threshold. These must pass outright: the whole point of
# that form is to satisfy the acceptance table the ellipse form cannot.
# ---------------------------------------------------------------------------

DISTANCE_PARAMS = StateCostParameters(
    lambda_s=1.0,
    delta_h=(1.0, 0.5),
    sigma_h=(0.8, 0.5),
    delta_psi1=(1.0, 0.5),
    sigma_psi1=(2.16, 1.0),
    psi1_gain=0.3,
    boundary_count=2,
    barrier_form="distance",
)


def _distance_cost(dx: float, closing: float, dy: float = 0.0) -> float:
    h, h_dot = _vehicle_barrier(dx, dy, closing)
    boundary_h = [0.5 * ROAD_WIDTH - EGO_HALF_WIDTH] * 2
    cost, _ = barrier_state_cost(
        np.asarray([h, *boundary_h]),
        np.asarray([h_dot, 0.0, 0.0]),
        DISTANCE_PARAMS,
    )
    return cost


def test_distance_form_leaves_ordinary_driving_alone():
    assert _distance_cost(dx=60.0, closing=0.0) == pytest.approx(0.0)
    assert _distance_cost(dx=40.0, closing=2.0) == pytest.approx(0.0)
    assert _distance_cost(dx=12.0, closing=-6.0) == pytest.approx(0.0)


def test_distance_form_warns_early_and_keeps_rising():
    approach = [
        _distance_cost(dx=distance, closing=6.0)
        for distance in (30.0, 25.0, 20.0, 15.0)
    ]
    assert approach[0] > 0.0, "no warning at 30 m while closing at 6 m/s"
    assert all(later > earlier for earlier, later in zip(approach, approach[1:]))


def test_distance_form_is_large_when_close_and_closing():
    close = _distance_cost(dx=6.0, closing=6.0)
    moderate = _distance_cost(dx=25.0, closing=6.0)
    assert close > 1.0
    assert close > 5.0 * moderate


def test_distance_form_falls_through_a_braking_recovery():
    """The check the ellipse form cannot satisfy at any margin or scale."""

    recovery = [
        _distance_cost(dx=12.0, closing=8.0),
        _distance_cost(dx=11.0, closing=5.0),
        _distance_cost(dx=10.4, closing=2.5),
        _distance_cost(dx=10.1, closing=0.5),
    ]
    assert all(
        later < earlier for earlier, later in zip(recovery, recovery[1:])
    ), f"cost did not fall while braking succeeded: {recovery}"


def test_the_gain_is_a_time_to_collision_threshold():
    """psi_1 >= 0 exactly when clearance/closing_rate >= 1/gain."""

    import math

    from saferl.safety.state_cost import barrier_state_cost as cost_fn

    gain = float(DISTANCE_PARAMS.psi1_gain)
    for dx, closing in ((30.0, 6.0), (20.0, 6.0), (12.0, 8.0)):
        h, h_dot = _vehicle_barrier(dx, 0.0, closing)
        rho = math.sqrt(h + 1.0)
        clearance = rho - 1.0
        rate = -h_dot / (2.0 * rho)
        time_to_contact = clearance / rate
        psi1 = -rate + gain * clearance
        assert (psi1 >= 0.0) == (time_to_contact >= 1.0 / gain)
        # And the cost engine agrees with the hand computation.
        boundary_h = [0.5 * ROAD_WIDTH - EGO_HALF_WIDTH] * 2
        _cost_value, diagnostics = cost_fn(
            np.asarray([h, *boundary_h]),
            np.asarray([h_dot, 0.0, 0.0]),
            DISTANCE_PARAMS,
        )
        if diagnostics["argmax_class"] == VEHICLE_CLASS:
            assert diagnostics["argmax_psi1"] == pytest.approx(psi1, rel=1e-6)
