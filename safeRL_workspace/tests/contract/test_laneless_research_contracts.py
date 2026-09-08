from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from lane_free_env import LaneFreeTrafficEnv
from saferl.artifacts import RunDirectory, manifest_digest
from saferl.config import DEFAULT_LANELESS_RESEARCH_CONFIG


def test_new_research_configuration_centralizes_canonical_contract():
    config = DEFAULT_LANELESS_RESEARCH_CONFIG

    assert config.frequencies.physics_hz == 100
    assert config.frequencies.policy_hz == 20
    assert config.frequencies.cbf_hz == 20
    assert config.frequencies.dt_s == pytest.approx(0.01)
    assert config.frequencies.physics_frames_per_policy_action == 5
    assert config.frequencies.physics_frames_per_cbf_update == 5

    assert config.expected_observation_shape == (32,)
    assert config.observation.base_dimension == 30
    assert config.action.normalized_low == -1.0
    assert config.action.normalized_high == 1.0
    assert config.action.physical_low == (-3.0, -3.0)
    assert config.action.physical_high == (3.0, 3.0)
    assert config.safety.eps_side == pytest.approx(0.10)
    assert config.safety.k0 == pytest.approx(5.29)
    assert config.safety.k1 == pytest.approx(3.68)
    assert config.safety.relative_ellipse_a_m == pytest.approx(3.6 / np.sqrt(2.0))
    assert config.safety.relative_ellipse_b_m == pytest.approx(1.8 / np.sqrt(2.0))

    resolved = config.resolved_environment_config()
    assert resolved["simulation_frequency"] == 100
    assert resolved["policy_frequency"] == 20
    assert resolved["cbf_frequency"] == 20
    assert resolved["traffic_safety"]["dynamics_guard"] is True
    assert resolved["traffic_safety"]["guard_ego_interactions"] is False
    assert resolved["bounds"] == {
        "ax_min": -3.0,
        "ax_max": 3.0,
        "ay_min": -3.0,
        "ay_max": 3.0,
    }
    assert resolved["cbf_geometry"] == {
        "model": "minimum_area_enclosing_vehicle_ellipse",
        "reference_vehicle_length_m": 3.6,
        "reference_vehicle_width_m": 1.8,
        "relative_ellipse_a_m": pytest.approx(3.6 / np.sqrt(2.0)),
        "relative_ellipse_b_m": pytest.approx(1.8 / np.sqrt(2.0)),
        "full_major_axis_m": pytest.approx(2.0 * 3.6 / np.sqrt(2.0)),
        "full_minor_axis_m": pytest.approx(2.0 * 1.8 / np.sqrt(2.0)),
    }


def test_environment_contract_preserves_normalized_spaces_and_observation_rows():
    config = DEFAULT_LANELESS_RESEARCH_CONFIG.resolved_environment_config()
    config.update(
        {
            "vehicles_count": 5,
            "neighbors_count": 2,
            "episode_steps": 100,
            "duration": 100,
        }
    )
    env = LaneFreeTrafficEnv(config=copy.deepcopy(config))
    try:
        assert env.action_space.shape == (2,)
        np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0])
        np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0])
        assert env.observation_space.shape == (15,)
    finally:
        env.close()


def _initial_scenario_signature(env: LaneFreeTrafficEnv) -> np.ndarray:
    values: list[float] = []
    for vehicle in env.road.vehicles:
        values.extend(np.asarray(vehicle.position, dtype=float).reshape(-1)[:2])
        values.extend(np.asarray(vehicle.velocity, dtype=float).reshape(-1)[:2])
        values.append(float(vehicle.desired_speed))
    return np.asarray(values, dtype=float)


def test_same_seed_produces_the_same_initial_scenario():
    config = DEFAULT_LANELESS_RESEARCH_CONFIG.resolved_environment_config()
    config.update(
        {
            "vehicles_count": 5,
            "neighbors_count": 2,
            "episode_steps": 100,
            "duration": 100,
        }
    )
    first = LaneFreeTrafficEnv(config=copy.deepcopy(config))
    second = LaneFreeTrafficEnv(config=copy.deepcopy(config))
    try:
        first_obs, _ = first.reset(seed=307)
        second_obs, _ = second.reset(seed=307)
        np.testing.assert_array_equal(first_obs, second_obs)
        np.testing.assert_array_equal(
            _initial_scenario_signature(first),
            _initial_scenario_signature(second),
        )
    finally:
        first.close()
        second.close()


def test_new_run_layout_is_self_contained_and_manifest_is_required(tmp_path):
    run = RunDirectory.create(tmp_path, "run_2026_09_07_contract")
    manifest = {
        "schema_version": 1,
        "algorithm": "PPO",
        "environment_config": DEFAULT_LANELESS_RESEARCH_CONFIG.resolved_environment_config(),
        "reward_config": {"name": "Karalakou"},
        "cbf_config": {"enabled": False},
        "observation_definition": DEFAULT_LANELESS_RESEARCH_CONFIG.observation.definition,
        "seed": 307,
        "frequencies": {"physics_hz": 100, "policy_hz": 20, "cbf_hz": 20},
        "training_steps": 50_000,
        "git_commit": "deadbeef",
        "model_path": str(run.model_path),
        "evaluation_protocol": {"distance_m": 1_000, "max_policy_steps": 3_000},
    }
    run.initialize(
        config=DEFAULT_LANELESS_RESEARCH_CONFIG.as_dict(),
        manifest=manifest,
        notes="Contract test run.",
    )

    assert run.config_path.is_file()
    assert run.manifest_path.is_file()
    assert run.notes_path.is_file()
    assert run.evaluation_dir.is_dir()
    assert json.loads(run.manifest_path.read_text(encoding="utf-8"))["algorithm"] == "PPO"
    assert len(manifest_digest(manifest)) == 64

    with pytest.raises(FileExistsError):
        RunDirectory.create(tmp_path, "run_2026_09_07_contract")
