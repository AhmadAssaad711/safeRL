"""Contract tests for the HOCBF action-distance safety cost."""

from __future__ import annotations

import numpy as np
import pytest

from saferl.safety.action_cost import (
    BOX_BOUND_RULE,
    DIAGNOSTIC_ONLY_INFEASIBLE_RULES,
    FALLBACK_RULE,
    HAZARDOUS_INFEASIBLE_RULES,
    SUPPORTED_INFEASIBLE_RULES,
    ZERO_RULE,
    action_cost_scale_from_reference,
    action_distance_cost,
    box_bound_cost,
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
