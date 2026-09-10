from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.evaluation import evaluate_cbf_random_actions as evaluator


def test_uniform_physical_actions_are_seeded_and_stay_inside_canonical_box():
    low = np.asarray([-3.0, -3.0], dtype=np.float32)
    high = np.asarray([3.0, 3.0], dtype=np.float32)
    rng = np.random.default_rng(71)
    first = [evaluator.uniform_physical_action(rng, low, high) for _ in range(4)]
    repeat_rng = np.random.default_rng(71)
    repeat = [evaluator.uniform_physical_action(repeat_rng, low, high) for _ in range(4)]

    assert all(np.array_equal(left, right) for left, right in zip(first, repeat))
    assert all(np.all(action >= low) and np.all(action <= high) for action in first)


def test_current_environment_keeps_canonical_timing_and_mtm_defaults(monkeypatch):
    namespace = {
        "ENV_CONFIG": {
            "traffic_model": "mtm",
            "simulation_frequency": 100,
            "policy_frequency": 20,
            "cbf_frequency": 20,
            "dt": 0.01,
        }
    }
    defaults = {"mtm": {"leader_range": 90.0}}
    monkeypatch.setattr(evaluator.progression, "MTM_CONGESTED_UNCERTAIN_UPDATES", defaults)

    env_config, frequencies = evaluator._resolved_environment(copy.deepcopy(namespace))

    assert frequencies["physics_hz"] == 100.0
    assert frequencies["policy_hz"] == 20.0
    assert frequencies["cbf_hz"] == 20.0
    assert env_config["mtm"]["leader_range"] == 90.0


def test_current_environment_rejects_noncanonical_cbf_rate():
    namespace = {
        "ENV_CONFIG": {
            "traffic_model": "force",
            "simulation_frequency": 100,
            "policy_frequency": 20,
            "cbf_frequency": 100,
            "cbf_substep_filtering": True,
            "dt": 0.01,
        }
    }

    with pytest.raises(ValueError, match="canonical 100/20/20"):
        evaluator._resolved_environment(namespace)


def test_random_evaluator_explicitly_enables_bounded_feasible_resets():
    source = {"road_length": 380.0}

    resolved = evaluator._enable_cbf_feasible_resets(source, max_attempts=17)

    assert source == {"road_length": 380.0}
    assert resolved["cbf_reset_feasibility"] == {
        "enabled": True,
        "max_attempts": 17,
    }
    with pytest.raises(ValueError, match="positive"):
        evaluator._enable_cbf_feasible_resets(source, max_attempts=0)


def test_effective_cbf_applies_the_requested_gains():
    namespace = {
        "CBF_AX_BOUNDS": [-3.0, 3.0],
        "CBF_AY_BOUNDS": [-3.0, 3.0],
        "CBF_EPS_SIDE": 0.10,
        "CBF_K0": 5.29,
        "CBF_K1": 3.68,
        "CBF_PSI1_GAIN": 2.3,
    }

    effective = evaluator._effective_cbf(namespace, k0=2.5, k1=1.5)

    assert effective["CBF_K0"] == 2.5
    assert effective["CBF_K1"] == 1.5
    assert namespace["CBF_K0"] == 2.5
    assert namespace["CBF_K1"] == 1.5


def test_random_episode_uses_cbf_mode_and_the_episode_seeded_predictor(monkeypatch):
    namespace = {
        "CBF_AX_BOUNDS": [-3.0, 3.0],
        "CBF_AY_BOUNDS": [-3.0, 3.0],
    }
    captured: dict[str, object] = {}

    def fake_evaluate(_namespace, **kwargs):
        captured.update(kwargs)
        predictor = evaluator.progression._predict_evaluation_action
        captured["actions"] = [
            predictor(None, np.zeros(2), action_source=kwargs["action_source"])
            for _ in range(2)
        ]
        return {"ok": True}

    original_predictor = evaluator.progression._predict_evaluation_action
    monkeypatch.setattr(evaluator.progression, "evaluate_completed_episode", fake_evaluate)

    result = evaluator._evaluate_episode_with_random_actions(
        namespace,
        episode_index=3,
        episode_seed=71,
        env_config={},
        reward_config={},
        args=object(),
    )

    expected_rng = np.random.default_rng(71)
    expected = [
        evaluator.uniform_physical_action(expected_rng, [-3.0, -3.0], [3.0, 3.0])
        for _ in range(2)
    ]
    assert result == {"ok": True}
    assert captured["model"] is None
    assert captured["mode"] == "cbf"
    assert captured["action_source"] == evaluator.RANDOM_ACTION_SOURCE
    assert all(np.array_equal(actual, wanted) for actual, wanted in zip(captured["actions"], expected))
    assert evaluator.progression._predict_evaluation_action is original_predictor


def test_random_episode_records_accepted_feasible_reset_metadata(monkeypatch):
    namespace = {
        "CBF_AX_BOUNDS": [-3.0, 3.0],
        "CBF_AY_BOUNDS": [-3.0, 3.0],
    }

    class FakeEnvironment:
        def get_wrapper_attr(self, name):
            assert name == "last_reset_info"
            return {
                "cbf_reset_feasibility_enabled": True,
                "cbf_reset_max_attempts": 17,
                "cbf_reset_retry_count": 2,
                "cbf_reset_candidate_seed": 73,
            }

    monkeypatch.setattr(
        evaluator.progression,
        "make_evaluation_env",
        lambda *args, **kwargs: FakeEnvironment(),
    )

    def fake_evaluate(_namespace, **kwargs):
        evaluator.progression.make_evaluation_env()
        return {"episode_index": kwargs["episode_index"]}

    monkeypatch.setattr(evaluator.progression, "evaluate_completed_episode", fake_evaluate)

    row = evaluator._evaluate_episode_with_random_actions(
        namespace,
        episode_index=3,
        episode_seed=71,
        env_config={},
        reward_config={},
        args=object(),
    )

    assert row["cbf_reset_feasibility_enabled"] is True
    assert row["cbf_reset_max_attempts"] == 17
    assert row["cbf_reset_retry_count"] == 2
    assert row["cbf_reset_candidate_seed"] == 73
