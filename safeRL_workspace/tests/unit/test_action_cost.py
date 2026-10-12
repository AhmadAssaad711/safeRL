"""Contract tests for the HOCBF action-distance safety cost."""

from __future__ import annotations

import numpy as np
import pytest

from saferl.safety.action_cost import (
    ACTION_COST_FORMS,
    BOX_BOUND_RULE,
    DIAGNOSTIC_ONLY_INFEASIBLE_RULES,
    FALLBACK_RULE,
    HAZARDOUS_INFEASIBLE_RULES,
    PROJECTION_COST_FORM,
    SUPPORTED_INFEASIBLE_RULES,
    VIOLATION_COST_FORM,
    ZERO_RULE,
    action_cost_scale_from_reference,
    action_distance_cost,
    box_bound_cost,
    constraint_violation_cost,
)

# The canonical physical acceleration box, [-3, 3] on both components.
LOW = (-3.0, -3.0)
HIGH = (3.0, 3.0)
BOX = {"action_low": LOW, "action_high": HIGH}


def test_feasible_action_costs_exactly_zero():
    """An action already inside U_safe must cost 0, not a small residual.

    The cost must not be a per-step tax in safe states, otherwise the
    constraint binds everywhere and the dual variable never switches off.
    """

    actions = np.array([[1.0, -2.0], [0.0, 0.0]])
    cost, diagnostics = action_distance_cost(actions, actions, **BOX)
    assert np.array_equal(cost, np.zeros(2))
    assert diagnostics["active_fraction"] == 0.0


def test_cost_is_half_squared_euclidean_distance():
    """c_a = 1/2 ||a - a_safe||^2, not the norm and not the unhalved square."""

    raw = np.array([[3.0, 4.0]])
    safe = np.array([[0.0, 0.0]])
    cost, diagnostics = action_distance_cost(raw, safe, **BOX)
    # ||(3,4)|| = 5, so the cost is 0.5 * 25.
    assert cost[0] == pytest.approx(12.5)
    assert diagnostics["correction_norm_mean"] == pytest.approx(5.0)


def test_cost_sums_over_both_action_components():
    """Both acceleration components must enter; one-axis scoring is a bug."""

    cost, _ = action_distance_cost([[1.0, 1.0]], [[0.0, 0.0]], **BOX)
    assert cost[0] == pytest.approx(1.0)


def test_single_action_is_accepted_as_one_sample():
    cost, diagnostics = action_distance_cost([2.0, 0.0], [0.0, 0.0], **BOX)
    assert cost.shape == (1,)
    assert diagnostics["sample_count"] == 1


def test_fallback_rule_is_diagnostic_only():
    """Scoring by the solver fallback is available but not a scoring rule.

    The fallback can sit close to the raw action on a maximally problematic
    state -- here it gives 0.5 against the box bound's 36.0 -- which is exactly
    why it must not define the cost.
    """

    cost, diagnostics = action_distance_cost(
        [[2.0, 0.0], [2.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0]],
        feasible=[True, False],
        infeasible_rule=FALLBACK_RULE,
    )
    assert cost[0] == pytest.approx(2.0)
    assert cost[1] == pytest.approx(0.5)
    assert diagnostics["infeasible_count"] == 1
    assert diagnostics["infeasible_fraction"] == pytest.approx(0.5)
    assert FALLBACK_RULE in DIAGNOSTIC_ONLY_INFEASIBLE_RULES


def test_box_bound_rule_scores_the_action_box_diameter():
    """An infeasible state is scored at the worst cost the box allows."""

    cost, diagnostics = action_distance_cost(
        [[2.0, 0.0], [2.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0]],
        feasible=[True, False],
        infeasible_rule=BOX_BOUND_RULE,
        **BOX,
    )
    # 0.5 * (6^2 + 6^2) = 36.0 for the [-3, 3]^2 box.
    assert cost[1] == pytest.approx(36.0)
    assert diagnostics["infeasible_cost_value"] == pytest.approx(36.0)
    # The feasible sample is untouched by the rule.
    assert cost[0] == pytest.approx(2.0)


def test_box_bound_is_the_locked_default():
    """The default must be the fixed conservative constant, not the fallback."""

    import inspect

    default = inspect.signature(action_distance_cost).parameters[
        "infeasible_rule"
    ].default
    assert default == BOX_BOUND_RULE

    cost, _ = action_distance_cost(
        [[2.0, 0.0]], [[1.0, 0.0]], feasible=[False], **BOX
    )
    assert cost[0] == pytest.approx(36.0)


def test_box_bound_cannot_be_undercut_by_any_feasible_sample():
    """No in-box feasible pair may score above the infeasibility constant."""

    bound = box_bound_cost(LOW, HIGH)
    corners = np.array([[3.0, 3.0], [-3.0, -3.0], [3.0, -3.0]])
    opposite = -corners
    cost, _ = action_distance_cost(corners, opposite, **BOX)
    assert np.all(cost <= bound + 1e-9)


def test_fallback_distance_stays_observable_under_box_bound():
    """The discarded rule is reported as a diagnostic, never scored."""

    _, diagnostics = action_distance_cost(
        [[2.0, 0.0]], [[1.0, 0.0]], feasible=[False],
        infeasible_rule=BOX_BOUND_RULE, **BOX,
    )
    assert diagnostics["fallback_cost_mean"] == pytest.approx(0.5)


def test_box_bound_rule_requires_the_box():
    with pytest.raises(ValueError, match="action_low and action_high"):
        action_distance_cost(
            [[1.0, 0.0]], [[0.0, 0.0]], feasible=[False],
            infeasible_rule=BOX_BOUND_RULE,
        )


def test_zero_rule_is_supported_but_declared_hazardous():
    """Scoring infeasible states as free is reachable only on purpose.

    It makes the cheapest way to satisfy the constraint be to reach states
    where no safe action exists, so it exists for parity with the masked actor
    losses and is flagged rather than offered as a default.
    """

    cost, _ = action_distance_cost(
        [[2.0, 0.0]], [[0.0, 0.0]], feasible=[False], infeasible_rule=ZERO_RULE
    )
    assert cost[0] == 0.0
    assert ZERO_RULE in HAZARDOUS_INFEASIBLE_RULES
    assert FALLBACK_RULE not in HAZARDOUS_INFEASIBLE_RULES


def test_default_rule_is_neither_hazardous_nor_diagnostic_only():
    import inspect

    default = inspect.signature(action_distance_cost).parameters[
        "infeasible_rule"
    ].default
    assert default in SUPPORTED_INFEASIBLE_RULES
    assert default not in HAZARDOUS_INFEASIBLE_RULES
    assert default not in DIAGNOSTIC_ONLY_INFEASIBLE_RULES


def test_unknown_infeasible_rule_is_rejected():
    with pytest.raises(ValueError, match="Unsupported infeasible_rule"):
        action_distance_cost([[0.0, 0.0]], [[0.0, 0.0]], infeasible_rule="mask")


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same shape"):
        action_distance_cost([[0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], **BOX)


def test_feasible_flag_count_is_validated():
    with pytest.raises(ValueError, match="one flag per sample"):
        action_distance_cost(
            [[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]], feasible=[True],
            **BOX,
        )


def test_non_finite_cost_fails_loudly():
    """A NaN cost would poison the critic and the dual variable silently."""

    with pytest.raises(ValueError, match="non-finite"):
        action_distance_cost([[np.inf, 0.0]], [[0.0, 0.0]], **BOX)


def test_box_bound_cost_rejects_inverted_box():
    with pytest.raises(ValueError, match=">= action_low"):
        box_bound_cost((1.0, 1.0), (0.0, 0.0))


# --------------------------------------------------------------------------
# Constraint-violation form: the reformulation that removes the actionless
# branch of the projection form.
# --------------------------------------------------------------------------


def _rows(*constraints, max_constraints=4):
    """Build (rows, bounds, mask) for one sample from (a_vec, b) pairs."""

    rows = np.zeros((1, max_constraints, 2))
    bounds = np.zeros((1, max_constraints))
    mask = np.zeros((1, max_constraints))
    for i, (vec, b) in enumerate(constraints):
        rows[0, i] = vec
        bounds[0, i] = b
        mask[0, i] = 1.0
    return rows, bounds, mask


def test_violation_cost_is_zero_on_the_safe_set():
    """Still zero exactly where the action satisfies every constraint."""

    rows, bounds, mask = _rows(((1.0, 0.0), -1.0))  # a_x <= -1
    cost, diagnostics = constraint_violation_cost([[-2.0, 0.0]], rows, bounds, mask)
    assert cost[0] == 0.0
    assert diagnostics["active_fraction"] == 0.0


def test_violation_cost_is_half_squared_hinge_summed_over_rows():
    rows, bounds, mask = _rows(((1.0, 0.0), -1.0), ((0.0, 1.0), 0.0))
    # a = (2, 3): row 0 violated by 3, row 1 by 3 -> 0.5*(9+9) = 9
    cost, diagnostics = constraint_violation_cost([[2.0, 3.0]], rows, bounds, mask)
    assert cost[0] == pytest.approx(9.0)
    assert diagnostics["violated_rows_mean"] == pytest.approx(2.0)


def test_violation_cost_ignores_masked_rows():
    """An inactive row must not contribute, however bad it looks."""

    rows = np.zeros((1, 4, 2))
    bounds = np.zeros((1, 4))
    mask = np.zeros((1, 4))
    rows[0, 0] = (1.0, 0.0); bounds[0, 0] = -1.0; mask[0, 0] = 1.0
    rows[0, 1] = (1.0, 0.0); bounds[0, 1] = -99.0; mask[0, 1] = 0.0  # inactive
    cost, _ = constraint_violation_cost([[0.0, 0.0]], rows, bounds, mask)
    assert cost[0] == pytest.approx(0.5)  # only row 0: violation 1


def test_violation_cost_stays_graded_where_no_safe_action_exists():
    """The property the projection form loses.

    These two rows are jointly unsatisfiable (a_x <= -1 and -a_x <= -2, i.e.
    a_x >= 2), so the safe set is empty. The projection form would score every
    action with one identical constant; the violation form must still order
    them, and the gradient must be non-zero.
    """

    rows, bounds, mask = _rows(((1.0, 0.0), -1.0), ((-1.0, 0.0), -2.0))
    near, _ = constraint_violation_cost([[0.5, 0.0]], rows, bounds, mask)
    far, _ = constraint_violation_cost([[3.0, 0.0]], rows, bounds, mask)
    best, _ = constraint_violation_cost([[0.5, 0.0]], rows, bounds, mask)
    # a_x = 0.5 violates row 0 by 1.5 and row 1 by 1.5 -> 0.5*(2.25+2.25)
    assert near[0] == pytest.approx(2.25)
    # a_x = 3.0 violates row 0 by 4, row 1 not at all -> 0.5*16
    assert far[0] == pytest.approx(8.0)
    assert near[0] != far[0]
    assert best[0] == near[0]


def test_trapped_states_have_a_positive_floor_no_action_can_escape():
    """An empty safe set shows up as a floor, not as a flat constant."""

    rows, bounds, mask = _rows(((1.0, 0.0), -1.0), ((-1.0, 0.0), -2.0))
    grid = np.linspace(-3.0, 3.0, 61).reshape(-1, 1)
    actions = np.hstack([grid, np.zeros_like(grid)])
    batch_rows = np.repeat(rows, actions.shape[0], axis=0)
    batch_bounds = np.repeat(bounds, actions.shape[0], axis=0)
    batch_mask = np.repeat(mask, actions.shape[0], axis=0)
    cost, _ = constraint_violation_cost(actions, batch_rows, batch_bounds, batch_mask)
    assert cost.min() > 0.0        # genuinely trapped
    assert cost.max() > cost.min()  # but the action still matters


def test_violation_floor_vanishes_as_the_safe_set_stops_being_empty():
    """Continuity at the feasibility boundary, which the 0 -> 36 jump lacks.

    Sweeping the second bound takes the safe set from empty to non-empty; the
    achievable floor must fall smoothly to zero rather than step.
    """

    # Row 0 is a_x <= -1. Row 1 is -a_x <= b1, i.e. a_x >= -b1, so the pair is
    # satisfiable exactly when -b1 <= -1, i.e. b1 >= 1. b1 = 1 leaves the
    # single point a_x = -1.
    floors = []
    for b1 in (-2.0, -1.0, 0.0, 1.0):
        rows, bounds, mask = _rows(((1.0, 0.0), -1.0), ((-1.0, 0.0), b1))
        grid = np.linspace(-3.0, 3.0, 241).reshape(-1, 1)
        actions = np.hstack([grid, np.zeros_like(grid)])
        cost, _ = constraint_violation_cost(
            actions,
            np.repeat(rows, actions.shape[0], axis=0),
            np.repeat(bounds, actions.shape[0], axis=0),
            np.repeat(mask, actions.shape[0], axis=0),
        )
        floors.append(cost.min())
    assert floors[0] > 0.0                      # deeply trapped
    assert floors[-1] == pytest.approx(0.0)     # feasible again, floor gone
    assert all(floors[i] >= floors[i + 1] for i in range(len(floors) - 1))


def test_violation_cost_agrees_with_projection_up_to_scale_on_one_constraint():
    """Both forms vanish exactly on the safe set and order actions the same.

    With a single unit-normal constraint active they are the same number, which
    is what keeps the quantity comparable to the detached actor loss where the
    projection is defined.
    """

    rows, bounds, mask = _rows(((1.0, 0.0), -1.0))
    action = np.array([[2.0, 0.0]])
    violation, _ = constraint_violation_cost(action, rows, bounds, mask)
    # The projection of a_x = 2 onto a_x <= -1 is -1, a gap of 3.
    projection, _ = action_distance_cost(action, [[-1.0, 0.0]], **BOX)
    assert violation[0] == pytest.approx(projection[0])


def test_violation_cost_is_invariant_to_rescaling_a_constraint_row():
    """(A_i, b_i) and (k A_i, k b_i) are the SAME halfspace.

    Without normalizing by ||A_i|| the cost changes by k^2, which is how the
    first version of this reformulation ended up with a cost of ~354 in the
    rows' arbitrary units and a cost-critic explained variance of 0.0000.
    """

    action = [[2.0, 1.0]]
    base, _ = constraint_violation_cost(action, *_rows(((1.0, 0.0), -1.0)))
    for k in (0.1, 2.0, 37.0, 1000.0):
        scaled, _ = constraint_violation_cost(
            action, *_rows(((k, 0.0), -k))
        )
        assert scaled[0] == pytest.approx(base[0]), f"not invariant at k={k}"


def test_violation_cost_is_a_distance_in_action_units():
    """A non-unit row must be measured by Euclidean distance, not raw residual."""

    # Row (3, 4) has norm 5, bound 0. For a = (3, 4): residual = 25,
    # distance = 25/5 = 5, so the cost is 0.5 * 25 = 12.5.
    cost, _ = constraint_violation_cost([[3.0, 4.0]], *_rows(((3.0, 4.0), 0.0)))
    assert cost[0] == pytest.approx(12.5)


def test_violation_cost_equals_the_projection_distance_for_one_constraint():
    """With a single active row the two forms are the same number.

    This is what keeps the quantity comparable to the detached actor loss,
    which regresses against the projection, and it now holds for any row scale
    rather than only for unit normals.
    """

    action = np.array([[2.0, 0.0]])
    # Row (5, 0), bound -5 is the halfspace a_x <= -1; projection is a_x = -1.
    violation, _ = constraint_violation_cost(action, *_rows(((5.0, 0.0), -5.0)))
    projection, _ = action_distance_cost(action, [[-1.0, 0.0]], **BOX)
    assert violation[0] == pytest.approx(projection[0])


def test_violation_cost_ignores_a_degenerate_zero_row():
    """A zero row constrains nothing and must not divide by zero."""

    rows, bounds, mask = _rows(((0.0, 0.0), -1.0))
    cost, _ = constraint_violation_cost([[5.0, 5.0]], rows, bounds, mask)
    assert cost[0] == 0.0
    assert np.isfinite(cost[0])


def test_violation_cost_validates_its_shapes():
    rows, bounds, mask = _rows(((1.0, 0.0), -1.0))
    with pytest.raises(ValueError, match="rows must be"):
        constraint_violation_cost([[0.0, 0.0]], rows[0], bounds, mask)
    with pytest.raises(ValueError, match="bounds must be"):
        constraint_violation_cost([[0.0, 0.0]], rows, bounds[0], mask)
    with pytest.raises(ValueError, match="mask must be"):
        constraint_violation_cost([[0.0, 0.0]], rows, bounds, mask[0])


def test_action_cost_forms_are_declared():
    assert PROJECTION_COST_FORM in ACTION_COST_FORMS
    assert VIOLATION_COST_FORM in ACTION_COST_FORMS


def test_reference_scale_matches_a_sustained_correction():
    """A quoted correction norm maps to the cost it scores."""

    assert action_cost_scale_from_reference(2.0) == pytest.approx(2.0)
    assert action_cost_scale_from_reference(0.0) == 0.0
    with pytest.raises(ValueError):
        action_cost_scale_from_reference(-1.0)


def test_cost_is_returned_unclipped_and_unnormalized():
    """c_a stays the raw geometric quantity B2 and C2 must share.

    Clipping or normalizing here would silently make CMDP-Action optimize a
    different quantity from the detached actor loss, and the comparison between
    the two mechanisms would no longer be about the mechanism.
    """

    cost, _ = action_distance_cost([[6.0, 0.0]], [[0.0, 0.0]], **BOX)
    assert cost[0] == pytest.approx(18.0)
