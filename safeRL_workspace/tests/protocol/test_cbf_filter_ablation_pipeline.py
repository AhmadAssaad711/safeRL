from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest


import scripts.training.run_cbf_filter_ablation as pipeline


class ScriptedCollisionEnv(gym.Env):
    """Five-step deterministic protocol fixture with two collision incidents."""

    metadata = {}

    def __init__(self) -> None:
        super().__init__()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.config = {
            "terminate_on_collision": True,
            "dt": 1.0,
            "simulation_frequency": 1,
            "policy_frequency": 1,
            "bounds": {"ax_min": -2.0, "ax_max": 2.0, "ay_min": -1.0, "ay_max": 1.0},
        }
        self.global_step = 0
        self.vehicle = SimpleNamespace(
            position=np.zeros(2, dtype=float),
            vx=10.0,
            vy=0.0,
            desired_speed=10.0,
            length=5.0,
            width=2.0,
            driver_profile="fixture",
        )
        self.road = SimpleNamespace(vehicles=[self.vehicle])
        self._last_accelerations = np.zeros((1, 2), dtype=float)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.global_step = 0
        self.vehicle.position[:] = 0.0
        return np.asarray([self.global_step], dtype=np.float32), {}

    def step(self, action):
        self.global_step += 1
        self.vehicle.position[0] += 2.0
        rewards = {1: 1.0, 2: -5.0, 3: 2.0, 4: -3.0, 5: 4.0}
        event_counts = {2: 2, 4: 1}
        events = event_counts.get(self.global_step, 0)
        active = events > 0
        terminated = active
        info = {
            "ego_collision_events": events,
            "ego_collision": active,
            "collisions": events,
            "active_collisions": events,
        }
        obs = np.asarray([self.global_step], dtype=np.float32)
        return obs, rewards[self.global_step], terminated, False, info


def _scenario_args() -> argparse.Namespace:
    return argparse.Namespace(
        eval_scenarios=1,
        eval_episodes=1,
        eval_timesteps=5,
        eval_horizon=5,
        correction_epsilon=0.03,
        eps_side=0.1,
        k0=5.29,
        k1=3.68,
    )


def test_fixed_timestep_evaluation_resets_and_counts_exact_events(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "make_evaluation_env",
        lambda *args, **kwargs: pipeline.ProtocolMetricsWrapper(ScriptedCollisionEnv()),
    )
    namespace = {
        "kpi_neighbor_and_h_metrics": lambda env, eps_side: {
            "kpi_h_min": np.nan,
            "kpi_boundary_h_min": np.nan,
        }
    }
    env_config = ScriptedCollisionEnv().config

    row, segments = pipeline.evaluate_scenario(
        namespace,
        model=None,
        variant="fixture",
        mode="raw",
        scenario_seed=11,
        training_seed=7,
        env_config=env_config,
        reward_config={},
        args=_scenario_args(),
    )

    assert row["timesteps"] == 5
    assert row["distinct_ego_collision_events"] == 3
    assert row["ego_collision_incidents"] == 2
    assert row["ego_collision_active_timesteps"] == 2
    assert row["total_distance_m"] == pytest.approx(10.0)
    assert row["distance_per_collision_m"] == pytest.approx(10.0 / 3.0)
    assert row["distance_per_collision_right_censored"] == 0
    assert row["distance_per_collision_exposure_bound_m"] == pytest.approx(10.0 / 3.0)
    assert row["collision_events_per_m"] == pytest.approx(0.3)
    assert row["ego_collisions_per_km"] == pytest.approx(300.0)
    assert row["return_per_timestep"] == pytest.approx(-0.2)
    assert row["first_collision_step"] == 2
    assert row["time_to_first_collision_s"] == pytest.approx(2.0)
    assert row["distance_to_first_collision_m"] == pytest.approx(4.0)
    assert row["collision_transition_timesteps"] == 2
    assert row["collision_transition_return"] == pytest.approx(-8.0)
    assert row["post_collision_timesteps"] == 0
    assert row["post_collision_return"] == pytest.approx(0.0)
    assert row["reset_calls_total"] == 3
    assert row["resets_after_collision"] == 2
    assert row["episode_length_mean"] == pytest.approx(5.0 / 3.0)
    assert [segment["steps"] for segment in segments] == [2, 2, 1]
    assert [segment["right_censored"] for segment in segments] == [0, 0, 1]


def _summary_row(seed: int, distance: float, collisions: int, timesteps: int, total_return: float) -> dict:
    row = {
        "training_seed": seed,
        "variant": "a_nominal",
        "mode": "raw",
        "timesteps": timesteps,
        "total_time_s": float(timesteps),
        "total_return": total_return,
        "task_return": total_return,
        "correction_return": 0.0,
        "distinct_ego_collision_events": collisions,
        "ego_collision_incidents": int(collisions > 0),
        "ego_collision_active_timesteps": int(collisions > 0),
        "distinct_all_pair_collision_events": collisions,
        "active_collision_pair_timesteps": collisions,
        "total_distance_m": distance,
        "collision_transition_timesteps": int(collisions > 0),
        "collision_transition_return": 0.0,
        "post_collision_timesteps": 0,
        "post_collision_return": 0.0,
        "reset_calls_total": 1 + int(collisions > 0),
        "resets_after_collision": int(collisions > 0),
        "resets_after_truncation_only": 0,
        "resets_after_other_terminal": 0,
        "episode_segments": 1,
        "completed_segments": int(collisions > 0),
        "right_censored_segments": int(collisions == 0),
        "episode_length_sum": timesteps,
        "collision_survived_without_reset": 0,
        "active_collision_without_event": 0,
        "event_without_active_collision": 0,
        "h_min": 1.0,
        "first_collision_observed": int(collisions > 0),
        "time_to_first_collision_s": 1.0 if collisions else np.nan,
        "distance_to_first_collision_m": distance if collisions else np.nan,
        "first_collision_censor_time_s": float(timesteps),
        "first_collision_censor_distance_m": distance,
    }
    for metric in (
        "h_violation_rate",
        "mean_abs_speed_error",
        "mean_jerk_norm",
        "IR",
        "mean_delta_a",
        "p95_delta_a",
        "qp_failure_rate",
        "qp_fallback_rate",
        "nominal_action_saturation_rate",
        "safe_action_saturation_rate",
        "executed_action_saturation_rate",
    ):
        row[metric] = 0.0
    return row


def test_hierarchical_summary_uses_ratio_of_sums_then_equal_seed_weights():
    scenarios = pd.DataFrame(
        [
            _summary_row(1, distance=10.0, collisions=1, timesteps=10, total_return=10.0),
            _summary_row(1, distance=90.0, collisions=0, timesteps=90, total_return=90.0),
            _summary_row(2, distance=10.0, collisions=2, timesteps=10, total_return=0.0),
        ]
    )

    within_seed = pipeline.summarize_within_training_seed(scenarios)
    seed_one = within_seed.loc[within_seed["training_seed"] == 1].iloc[0]
    seed_two = within_seed.loc[within_seed["training_seed"] == 2].iloc[0]
    assert seed_one["distance_per_collision_m"] == pytest.approx(100.0)
    assert seed_one["ego_collisions_per_km"] == pytest.approx(10.0)
    assert seed_one["return_per_timestep"] == pytest.approx(1.0)
    assert seed_two["distance_per_collision_m"] == pytest.approx(5.0)
    assert seed_two["ego_collisions_per_km"] == pytest.approx(200.0)

    across_seed = pipeline.summarize_across_training_seeds(within_seed).iloc[0]
    assert across_seed["distance_per_collision_m_seed_mean"] == pytest.approx(52.5)
    assert across_seed["ego_collisions_per_km_seed_mean"] == pytest.approx(105.0)
    assert across_seed["ego_collisions_per_km_seed_variance"] == pytest.approx(18_050.0)
    assert across_seed["distance_per_collision_m_seed_mean"] != pytest.approx(110.0 / 3.0)


def test_zero_collision_ratio_semantics_are_not_epsilon_clamped():
    assert np.isinf(pipeline._distance_per_collision(123.0, 0.0))
    assert pipeline._distance_per_collision_exposure_bound(123.0, 0.0) == 123.0
    assert pipeline._collisions_per_km(0.0, 123.0) == 0.0
    assert np.isnan(pipeline._distance_per_collision(0.0, 0.0))
    assert np.isinf(pipeline._collisions_per_km(1.0, 0.0))


def test_linearized_ttc_is_zero_when_unsafe_and_finite_when_not_closing():
    assert pipeline.linearized_ttc_from_barriers(
        [(-0.1, 2.0), (5.0, -1.0)], cap_s=30.0
    ) == pytest.approx(0.0)
    assert pipeline.linearized_ttc_from_barriers(
        [(3.0, 0.0), (4.0, 1.0)], cap_s=30.0
    ) == pytest.approx(30.0)
    assert pipeline.linearized_ttc_from_barriers(
        [(3.0, -0.5), (4.0, -2.0)], cap_s=30.0
    ) == pytest.approx(2.0)


def test_within_seed_p95_is_pooled_from_step_rows_not_mean_scenario_p95():
    scenarios = pd.DataFrame(
        [
            _summary_row(1, distance=10.0, collisions=0, timesteps=50, total_return=0.0),
            _summary_row(1, distance=10.0, collisions=0, timesteps=50, total_return=0.0),
        ]
    )
    scenarios.loc[:, "p95_delta_a"] = [1.0, 9.0]
    scenarios.loc[:, "shadow_p95_delta_a"] = [2.0, 8.0]
    step_values = np.arange(100, dtype=float)
    step_metrics = pd.DataFrame(
        {
            "training_seed": 1,
            "variant": "a_nominal",
            "mode": "raw",
            "applied_delta_norm_scaled": step_values,
            "shadow_delta_norm_scaled": step_values[::-1],
        }
    )

    row = pipeline.summarize_within_training_seed(
        scenarios,
        step_metrics=step_metrics,
    ).iloc[0]
    assert row["mean_scenario_p95_delta_a"] == pytest.approx(5.0)
    assert row["p95_delta_a"] == pytest.approx(np.quantile(step_values, 0.95))
    assert row["shadow_p95_delta_a"] == pytest.approx(np.quantile(step_values, 0.95))


def test_total_distance_is_full_path_length_not_only_longitudinal_progress():
    env = ScriptedCollisionEnv()
    assert pipeline._step_path_distance(
        env,
        np.asarray([1.0, 2.0]),
        np.asarray([4.0, 6.0]),
    ) == pytest.approx(5.0)


def test_strict_checkpoint_rejects_hash_mismatch_and_corruption(tmp_path):
    bundle = tmp_path / "000000003"
    bundle.mkdir()
    config_hash = "training-hash"
    model_class = "fixture.Model"
    state = {
        "schema_version": pipeline.PIPELINE_SCHEMA_VERSION,
        "timestep": 3,
        "n_updates": 2,
        "training_config_hash": config_hash,
    }
    payload_data = {
        pipeline.CHECKPOINT_PAYLOADS["model"]: b"model",
        pipeline.CHECKPOINT_PAYLOADS["replay_buffer"]: b"replay",
        pipeline.CHECKPOINT_PAYLOADS["base_environment"]: b"environment",
    }
    for name, data in payload_data.items():
        (bundle / name).write_bytes(data)
    with (bundle / pipeline.CHECKPOINT_PAYLOADS["pipeline_state"]).open("wb") as handle:
        pickle.dump(state, handle)
    payload_names = list(payload_data) + [pipeline.CHECKPOINT_PAYLOADS["pipeline_state"]]
    checksums = {
        name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        for name in payload_names
    }
    manifest = {
        "schema_version": pipeline.PIPELINE_SCHEMA_VERSION,
        "timestep": 3,
        "n_updates": 2,
        "training_config_hash": config_hash,
        "model_class": model_class,
        "checksums": checksums,
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded_manifest, loaded_state = pipeline.validate_checkpoint_bundle(
        bundle,
        config_hash,
        expected_model_class=model_class,
    )
    assert loaded_manifest["timestep"] == loaded_state["timestep"] == 3
    with pytest.raises(RuntimeError, match="configuration hash mismatch"):
        pipeline.validate_checkpoint_bundle(
            bundle,
            "different-hash",
            expected_model_class=model_class,
        )

    (bundle / pipeline.CHECKPOINT_PAYLOADS["replay_buffer"]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="missing or corrupt"):
        pipeline.validate_checkpoint_bundle(
            bundle,
            config_hash,
            expected_model_class=model_class,
        )


def test_configuration_hash_is_order_independent_but_value_sensitive():
    assert pipeline.canonical_config_hash({"a": 1, "b": [2, 3]}) == pipeline.canonical_config_hash(
        {"b": [2, 3], "a": 1}
    )
    assert pipeline.canonical_config_hash({"a": 1}) != pipeline.canonical_config_hash({"a": 2})


def test_lane_free_action_conversion_handles_asymmetric_zero_crossing_bounds():
    config = {
        "bounds": {
            "ax_min": -6.0,
            "ax_max": 3.0,
            "ay_min": -4.0,
            "ay_max": 4.0,
        }
    }
    commands = [
        ([-1.0, -1.0], [-6.0, -4.0]),
        ([-0.5, -0.5], [-3.0, -2.0]),
        ([0.0, 0.0], [0.0, 0.0]),
        ([0.5, 0.5], [1.5, 2.0]),
        ([1.0, 1.0], [3.0, 4.0]),
    ]
    for normalized, expected_physical in commands:
        physical = pipeline.normalized_to_physical(np.asarray(normalized), config)
        assert physical == pytest.approx(expected_physical)
        assert pipeline.physical_to_normalized(physical, config) == pytest.approx(normalized)


def test_registered_variants_form_complete_filtered_two_by_two():
    args = argparse.Namespace(
        lambda_delta=0.025,
        lambda_intervention=0.02,
    )
    observed = {
        cell: pipeline.variant_spec(variant, args)
        for cell, variant in pipeline.FACTORIAL_VARIANTS.items()
    }
    assert set(observed) == {(False, False), (True, False), (False, True), (True, True)}
    for (reward_on, loss_on), spec in observed.items():
        assert spec["filtered"] is True
        assert spec["actor_loss"] is loss_on
        assert (spec["lambda_delta"] > 0.0) is reward_on
        assert (spec["lambda_intervention"] > 0.0) is reward_on
    assert all(
        pipeline.model_class_for_variant(variant) is pipeline.GuidedCBFDDPG
        for variant in pipeline.FACTORIAL_VARIANTS.values()
    )


def test_factorial_effects_compute_main_effects_and_interaction():
    cell_values = {
        (False, False): 1.0,
        (True, False): 3.0,
        (False, True): 4.0,
        (True, True): 10.0,
    }
    rows = [
        {
            "training_seed": 7,
            "mode": "raw",
            "variant": pipeline.FACTORIAL_VARIANTS[cell],
            "return_per_timestep": value,
        }
        for cell, value in cell_values.items()
    ]
    effects = pipeline.factorial_effects(pd.DataFrame(rows)).set_index("effect")
    assert effects.loc["reward_main_effect", "effect_return_per_timestep"] == pytest.approx(4.0)
    assert effects.loc["actor_loss_main_effect", "effect_return_per_timestep"] == pytest.approx(5.0)
    assert effects.loc["reward_actor_interaction", "effect_return_per_timestep"] == pytest.approx(4.0)
