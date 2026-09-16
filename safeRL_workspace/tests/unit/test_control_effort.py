"""Contract tests for the quadratic control-effort reward term."""

from __future__ import annotations

import argparse

import pytest

from saferl.rewards import (
    DEFAULT_CONTROL_EFFORT_WEIGHT,
    add_reward_variant_arguments,
    apply_reward_variant_arguments,
    control_effort_weight,
    make_control_effort_wrapper,
    wrap_reward_variants,
)

AX_MAX = 3.0
AY_MAX = 3.0


class _FakeBase:
    def __init__(self) -> None:
        self.config = {
            "bounds": {
                "ax_min": -AX_MAX, "ax_max": AX_MAX,
                "ay_min": -AY_MAX, "ay_max": AY_MAX,
            }
        }


class _StubWrapper:
    """Minimal stand-in exposing what the control-effort subclass consumes."""

    def __init__(self, applied, base_reward=1.0) -> None:
        self.reward_config: dict = {}
        self._applied = applied
        self._base_reward = base_reward
        self.base_env = _FakeBase()

    def _current_applied_acceleration(self):
        return self._applied

    def _karalakou_reward(self, previous_dx, previous_ego_x=None):
        return self._base_reward, {"reward": self._base_reward, "cx": 0.0}


def _reward(applied, weight, base_reward=1.0):
    wrapper_type = make_control_effort_wrapper(_StubWrapper)
    wrapper = wrapper_type.__new__(wrapper_type)
    _StubWrapper.__init__(wrapper, applied, base_reward)
    wrapper.reward_config = {"control_effort_weight": weight}
    return wrapper._karalakou_reward({})


def test_weight_defaults_to_zero_and_validates():
    assert control_effort_weight({}) == DEFAULT_CONTROL_EFFORT_WEIGHT == 0.0
    assert control_effort_weight({"control_effort_weight": 0.25}) == 0.25
    with pytest.raises(ValueError):
        control_effort_weight({"control_effort_weight": -1.0})
    with pytest.raises(ValueError):
        control_effort_weight({"control_effort_weight": float("nan")})


def test_zero_action_costs_nothing():
    total, components = _reward([0.0, 0.0], 0.5)
    assert components["control_effort"] == pytest.approx(0.0)
    assert total == pytest.approx(1.0)


def test_full_saturation_costs_the_whole_weight():
    """Both axes at the bound give a normalized effort of exactly 1."""

    total, components = _reward([AX_MAX, -AY_MAX], 0.5)
    assert components["control_effort"] == pytest.approx(1.0)
    assert components["control_effort_penalty"] == pytest.approx(0.5)
    assert total == pytest.approx(0.5)


def test_penalty_is_quadratic_and_never_saturates():
    """Doubling the action quadruples the cost, unlike the tanh jerk term."""

    _t1, half = _reward([1.5, 0.0], 1.0)
    _t2, full = _reward([3.0, 0.0], 1.0)
    assert full["control_effort"] == pytest.approx(4.0 * half["control_effort"])


def test_axes_are_normalized_separately():
    _t1, longitudinal = _reward([1.5, 0.0], 1.0)
    _t2, lateral = _reward([0.0, 1.5], 1.0)
    assert longitudinal["control_effort"] == pytest.approx(lateral["control_effort"])


def test_components_still_sum_to_the_returned_reward():
    total, components = _reward([2.0, 1.0], 0.4, base_reward=0.9)
    assert components["reward"] == pytest.approx(total)
    assert components["reward_before_control_effort"] - components[
        "control_effort_penalty"
    ] == pytest.approx(total)


def test_wrapper_is_only_applied_when_the_weight_is_positive():
    class _Base:
        pass

    assert wrap_reward_variants(_Base, {}) is _Base
    assert wrap_reward_variants(_Base, {"control_effort_weight": 0.0}) is _Base
    assert wrap_reward_variants(_Base, {"control_effort_weight": 0.3}) is not _Base


def test_cli_flag_round_trips_and_validates():
    parser = argparse.ArgumentParser()
    add_reward_variant_arguments(parser)
    reward_config: dict = {}
    apply_reward_variant_arguments(
        parser.parse_args(["--control-effort-weight", "0.3"]), reward_config
    )
    assert reward_config["control_effort_weight"] == pytest.approx(0.3)

    # Omitted flag must leave the notebook reward untouched.
    untouched: dict = {}
    apply_reward_variant_arguments(parser.parse_args([]), untouched)
    assert "control_effort_weight" not in untouched

    with pytest.raises(ValueError):
        apply_reward_variant_arguments(
            parser.parse_args(["--control-effort-weight", "-0.1"]), {}
        )
