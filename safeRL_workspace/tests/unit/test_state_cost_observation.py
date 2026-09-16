"""Contract tests for exposing the state cost's hinge terms to the policy.

Three things here would fail silently rather than loudly, so they are pinned:

* the features must be built from the *current* state, because the reward's
  copy is evaluated on the post-step system and reusing it would leak s(t+1)
  into the policy input;
* the observing arm and the charged arm must use identical margins, or the
  control stops being a control while keeping the same observation width;
* the features must stay inside the declared clip, because the policy is a
  Tanh MLP and the hinge terms are squared and unbounded above.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from saferl.observations import (
    DEFAULT_STATE_COST_FEATURE_CLIP,
    STATE_COST_FEATURE_CLIP_KEY,
    STATE_COST_FEATURE_COUNT,
    STATE_COST_FEATURE_PARAMS_KEY,
    STATE_COST_OBSERVE_KEY,
    resolve_state_cost_feature_clip,
    resolve_state_cost_feature_params,
    resolve_state_cost_observe,
    state_cost_observation_features,
)
from saferl.safety.state_cost import StateCostParameters, barrier_state_cost
from scripts.training import run_ppo_cbf_progression as progression

# One vehicle barrier and the two road-boundary rows the constraint builder
# always appends, which is what boundary_count splits on.
PARAMS = StateCostParameters(
    lambda_s=0.0563,
    delta_h=(6.0, 0.5),
    sigma_h=(3.0, 0.25),
    delta_psi1=(8.0, 1.0),
    sigma_psi1=(4.0, 0.5),
    psi1_gain=2.3,
    boundary_count=2,
)


def _system(h: list[float], h_dot: list[float]) -> dict[str, object]:
    return {
        "barrier_h": np.asarray(h, dtype=float),
        "barrier_h_dot": np.asarray(h_dot, dtype=float),
    }


class TestFeatureValues:
    def test_features_are_exactly_the_two_hinge_terms(self) -> None:
        h = [2.0, 4.0, 3.0]
        h_dot = [-1.0, 0.5, 0.2]
        features = state_cost_observation_features(h, h_dot, PARAMS, clip=1e9)
        _cost, diagnostics = barrier_state_cost(h, h_dot, PARAMS)
        assert features.shape == (STATE_COST_FEATURE_COUNT,)
        assert features[0] == pytest.approx(diagnostics["h_term"], rel=1e-6)
        assert features[1] == pytest.approx(diagnostics["psi1_term"], rel=1e-6)

    def test_features_sum_to_the_cost(self) -> None:
        """Their sum is C(s), which is what the reward is charged on."""

        h = [1.5, 5.0, 2.0]
        h_dot = [-2.0, 0.0, 0.1]
        features = state_cost_observation_features(h, h_dot, PARAMS, clip=1e9)
        cost, _diagnostics = barrier_state_cost(h, h_dot, PARAMS)
        assert float(features.sum()) == pytest.approx(cost, rel=1e-6)

    def test_safe_state_gives_zero(self) -> None:
        """Far and receding must cost nothing, so the feature must be silent."""

        features = state_cost_observation_features(
            [500.0, 50.0, 50.0], [10.0, 1.0, 1.0], PARAMS
        )
        np.testing.assert_allclose(features, 0.0)

    def test_features_rise_as_the_state_gets_worse(self) -> None:
        near = state_cost_observation_features([1.0, 5.0, 5.0], [-3.0, 0.0, 0.0], PARAMS, clip=1e9)
        far = state_cost_observation_features([5.0, 5.0, 5.0], [-0.5, 0.0, 0.0], PARAMS, clip=1e9)
        assert near.sum() > far.sum()

    def test_features_are_float32_for_the_observation_space(self) -> None:
        features = state_cost_observation_features([1.0, 5.0, 5.0], [-3.0, 0.0, 0.0], PARAMS)
        assert features.dtype == np.float32


class TestClip:
    def test_clip_bounds_a_near_collision(self) -> None:
        """Unbounded squared hinges would saturate a Tanh first layer."""

        features = state_cost_observation_features(
            [-0.99, 5.0, 5.0], [-50.0, 0.0, 0.0], PARAMS, clip=4.0
        )
        assert float(features.max()) <= 4.0
        # It really did need clipping, otherwise this asserts nothing.
        unclipped = state_cost_observation_features(
            [-0.99, 5.0, 5.0], [-50.0, 0.0, 0.0], PARAMS, clip=1e9
        )
        assert float(unclipped.max()) > 4.0

    def test_features_are_never_negative(self) -> None:
        features = state_cost_observation_features(
            [100.0, 100.0, 100.0], [50.0, 50.0, 50.0], PARAMS
        )
        assert float(features.min()) >= 0.0

    def test_invalid_clip_is_rejected(self) -> None:
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                state_cost_observation_features([1.0, 5.0, 5.0], [0.0, 0.0, 0.0], PARAMS, clip=bad)


class TestEnvConfigResolution:
    def test_observing_defaults_to_off(self) -> None:
        assert resolve_state_cost_observe({}) is False
        assert resolve_state_cost_observe({STATE_COST_OBSERVE_KEY: True}) is True

    def test_clip_defaults_and_validates(self) -> None:
        assert resolve_state_cost_feature_clip({}) == DEFAULT_STATE_COST_FEATURE_CLIP
        assert resolve_state_cost_feature_clip({STATE_COST_FEATURE_CLIP_KEY: 2.5}) == 2.5
        with pytest.raises(ValueError):
            resolve_state_cost_feature_clip({STATE_COST_FEATURE_CLIP_KEY: -1.0})

    def test_params_round_trip_through_the_env_config(self) -> None:
        """A stored run config must rebuild the exact margins training used."""

        stored = {STATE_COST_FEATURE_PARAMS_KEY: PARAMS.as_dict()}
        restored = resolve_state_cost_feature_params(stored)
        assert restored.delta_h == PARAMS.delta_h
        assert restored.sigma_h == PARAMS.sigma_h
        assert restored.delta_psi1 == PARAMS.delta_psi1
        assert restored.sigma_psi1 == PARAMS.sigma_psi1
        assert restored.psi1_gain == PARAMS.psi1_gain
        assert restored.barrier_form == PARAMS.barrier_form

    def test_absent_params_refuse_to_guess(self) -> None:
        """Defaulting the margins would change the feature silently."""

        with pytest.raises(ValueError):
            resolve_state_cost_feature_params({})

    def test_fallback_is_used_only_when_supplied(self) -> None:
        assert resolve_state_cost_feature_params({}, PARAMS) is PARAMS


class TestObservationWidth:
    @pytest.mark.parametrize(
        "neighbors_count, without, with_features", [(5, 31, 33), (7, 41, 43)]
    )
    def test_width_gains_exactly_two_features(
        self, neighbors_count: int, without: int, with_features: int
    ) -> None:
        """41D at neighbors_count 7 becomes 43D; the width stays derived."""

        env_config = {
            "neighbors_count": neighbors_count,
            "observation_include_vehicle_dimensions": False,
            "ppo_append_previous_action": True,
            "drop_ego_speed_slot": True,
        }
        assert progression._base_observation_dim(env_config) == without
        assert (
            progression._base_observation_dim(
                {**env_config, STATE_COST_OBSERVE_KEY: True}
            )
            == with_features
        )

    def test_width_is_additive_with_the_dropped_slot(self) -> None:
        base = {
            "neighbors_count": 7,
            "observation_include_vehicle_dimensions": False,
            "ppo_append_previous_action": True,
        }
        assert progression._base_observation_dim(base) == 42
        assert progression._base_observation_dim({**base, STATE_COST_OBSERVE_KEY: True}) == 44
        assert (
            progression._base_observation_dim(
                {**base, "drop_ego_speed_slot": True, STATE_COST_OBSERVE_KEY: True}
            )
            == 43
        )


class TestArmParameterSharing:
    """The observing control and the charged arm must describe the same risk."""

    @staticmethod
    def _args(**overrides):
        # Mirrors tests/protocol/test_ppo_cbf_progression.py: resolved_state_cost
        # reads its inputs with getattr defaults, so a namespace is enough and
        # keeps the test independent of the parser's full surface.
        values = {
            "state_cost_observe": True,
            "state_cost_lambda": 0.0563,
            "state_cost_barrier_form": "ellipse",
            "state_cost_delta_h": None,
            "state_cost_sigma_h": None,
            "state_cost_delta_psi1": None,
            "state_cost_sigma_psi1": None,
            "state_cost_psi1_gain": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_only_lambda_separates_the_two_arms(self) -> None:
        args = self._args()
        charged = progression.resolved_state_cost(args, {"state_cost_reward": True})
        observing = progression.resolved_state_cost(args, {})
        assert charged.lambda_s == pytest.approx(0.0563)
        # The control arm is never charged, whatever lambda was passed.
        assert observing.lambda_s == 0.0
        for field in ("delta_h", "sigma_h", "delta_psi1", "sigma_psi1", "psi1_gain", "barrier_form"):
            assert getattr(charged, field) == getattr(observing, field)

    def test_features_are_identical_across_the_arms(self) -> None:
        args = self._args()
        charged = progression.resolved_state_cost(args, {"state_cost_reward": True})
        observing = progression.resolved_state_cost(args, {})
        h, h_dot = [2.0, 4.0, 3.0], [-1.0, 0.5, 0.2]
        np.testing.assert_allclose(
            state_cost_observation_features(h, h_dot, charged),
            state_cost_observation_features(h, h_dot, observing),
            rtol=0,
            atol=0,
        )

    def test_observing_is_off_without_the_flag(self) -> None:
        """An omitted flag must leave every historical layout untouched."""

        args = self._args(state_cost_observe=False)
        assert progression.resolved_state_cost(args, {}) == StateCostParameters()

    def test_a_charged_arm_is_unaffected_by_the_observe_flag(self) -> None:
        """Adding the flag must not perturb the reward the treatment arm pays."""

        charged_with = progression.resolved_state_cost(
            self._args(), {"state_cost_reward": True}
        )
        charged_without = progression.resolved_state_cost(
            self._args(state_cost_observe=False), {"state_cost_reward": True}
        )
        assert charged_with == charged_without
