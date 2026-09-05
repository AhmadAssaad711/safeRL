from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
import scripts.training.run_cbf_filter_ablation as pipeline
import scripts.training.run_ppo_formulation_screen as screen
from scripts.training.train_safety_potential_variants import MTM_CONGESTED_UNCERTAIN_UPDATES, deep_update


@pytest.fixture(scope="module")
def formulation_setup():
    namespace = pipeline.bootstrap_notebook_namespace(ROOT)
    pipeline.exec_required_notebook_cells(
        ROOT / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    env_config = copy.deepcopy(namespace["ENV_CONFIG"])
    env_config["traffic_model"] = "mtm"
    deep_update(env_config, copy.deepcopy(MTM_CONGESTED_UNCERTAIN_UPDATES))
    env_config["ego_boundary_force"] = True
    reward_config = pipeline.make_base_reward_config(namespace)
    return namespace, env_config, reward_config


def make_env(formulation_setup, formulation: str):
    namespace, env_config, reward_config = formulation_setup
    return pipeline.make_raw_env(
        screen.make_formulation_namespace(namespace, formulation),
        seed=307,
        env_config=env_config,
        reward_config=reward_config,
    )


def find_formulation_wrapper(env):
    current = env
    while hasattr(current, "env"):
        if hasattr(current, "formulation_id"):
            return current
        current = current.env
    raise AssertionError("formulation wrapper not found")


def test_registry_freezes_q0_and_records_exact_variant_spaces():
    assert tuple(screen.FORMULATIONS) == (
        "P0_current",
        "P1_reward",
        "P2_observed",
        "P3_jerk",
        "P4_reference",
    )
    for parameters in screen.PPO_CONFIGS.values():
        assert parameters == screen.Q0_PPO_PARAMETERS
    assert screen.FORMULATIONS["P2_observed"]["observation_dim"] == 49
    assert screen.FORMULATIONS["P3_jerk"]["action_low"] == [-8.0, -8.0]
    assert screen.REFERENCE_GAINS == {"k_v": 1.0, "k_p": 1.0, "k_d": 2.0}


def test_all_formulations_share_initial_base_state_and_declared_spaces(
    formulation_setup,
):
    state_hashes = []
    for formulation in screen.FORMULATION_ORDER:
        env = make_env(formulation_setup, formulation)
        try:
            observation, _ = env.reset(seed=900_000)
            state_hashes.append(pipeline.initial_state_hash(env))
            assert observation.dtype == np.float32
            assert observation.shape == (
                screen.FORMULATIONS[formulation]["observation_dim"],
            )
            assert env.observation_space.contains(observation)
            np.testing.assert_allclose(
                env.action_space.low,
                screen.FORMULATIONS[formulation]["action_low"],
            )
            np.testing.assert_allclose(
                env.action_space.high,
                screen.FORMULATIONS[formulation]["action_high"],
            )
        finally:
            env.close()
    assert len(set(state_hashes)) == 1


def test_p2_explicit_features_match_pre_action_state(formulation_setup):
    env = make_env(formulation_setup, "P2_observed")
    try:
        observation, _ = env.reset(seed=307)
        wrapper = find_formulation_wrapper(env)
        base = env.unwrapped
        target_y, target_speed, _ = wrapper._lateral_target_and_speed()
        left, right, _ = screen.boundary_state(base)
        expected = np.asarray(
            [
                target_speed / 24.0,
                np.clip(2.0 * target_y / base.config["road_width"] - 1.0, -1.0, 1.0),
                left / base.config["road_width"],
                right / base.config["road_width"],
                0.0,
                0.0,
                wrapper._potential_field_cost(),
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(observation[-7:], expected, rtol=1e-6, atol=1e-6)
        assert np.all(np.isfinite(observation))
    finally:
        env.close()


def test_p3_uses_exact_tanh_jerk_integration_and_no_hidden_boundary_force(
    formulation_setup,
):
    env = make_env(formulation_setup, "P3_jerk")
    try:
        env.reset(seed=307)
        latent = np.asarray([1.0, -1.0], dtype=np.float32)
        _, reward, _, _, info = env.step(latent)
        expected_jerk = 6.0 * np.tanh(latent)
        # Integrate at the configured policy interval (currently 0.10 s), not
        # a stale hard-coded interval from an older environment configuration.
        expected_acceleration = screen._policy_dt(env.unwrapped) * expected_jerk
        np.testing.assert_allclose(
            env.unwrapped._last_accelerations[0], expected_acceleration, rtol=1e-6
        )
        assert env.unwrapped.config["ego_boundary_force"] is False
        assert info["formulation_normalized_jerk_command_sq"] == pytest.approx(
            float(np.sum(np.square(np.tanh(latent))))
        )
        assert reward == pytest.approx(info["formulation_native_reward"])
    finally:
        env.close()


def test_p4_reference_controller_uses_pre_action_state(formulation_setup):
    env = make_env(formulation_setup, "P4_reference")
    try:
        env.reset(seed=307)
        base = env.unwrapped
        vx = float(base.vehicle.vx)
        y = float(base.vehicle.position[1])
        vy = float(base.vehicle.vy)
        center_y = 0.5 * float(base.config["road_width"])
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        expected = np.asarray(
            [
                np.clip(12.0 - vx, -3.0, 3.0),
                np.clip(center_y - y - 2.0 * vy, -3.0, 3.0),
            ]
        )
        np.testing.assert_allclose(base._last_accelerations[0], expected, rtol=1e-6)
        assert info["formulation_reference_speed"] == pytest.approx(12.0)
        assert info["formulation_reference_y"] == pytest.approx(center_y)
        assert base.config["ego_boundary_force"] is False
    finally:
        env.close()


def test_formulation_one_reward_decomposition_and_no_overtake_bonus(
    formulation_setup,
):
    env = make_env(formulation_setup, "P1_reward")
    try:
        env.reset(seed=307)
        _, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        expected = (
            info["formulation_speed_tracking_reward"]
            + info["formulation_lateral_tracking_reward"]
            + info["formulation_speed_progress_reward"]
            - 1.5 * info["formulation_cf"]
            - 0.5 * info["formulation_boundary_cost"]
            - 0.03 * info["formulation_normalized_acceleration_sq"]
            - 0.08 * info["formulation_normalized_acceleration_delta_sq"]
            - 20.0 * info["formulation_collision_event"]
        )
        assert reward == pytest.approx(expected)
        assert reward == pytest.approx(info["formulation_common_form1_reward"])
        assert "formulation_overtake_bonus" not in info
    finally:
        env.close()


def test_ranking_uses_common_reward_not_mixed_native_return():
    frame = pd.DataFrame(
        {
            "pilot_config": ["P0_current", "P1_reward"],
            "ego_collisions_per_km_seed_mean": [5.0, 4.0],
            "distance_per_collision_exposure_bound_m_seed_mean": [200.0, 250.0],
            "common_form1_return_per_timestep_seed_mean": [-1.0, -0.5],
            "return_per_timestep_seed_mean": [10.0, -0.5],
            "executed_action_saturation_rate_seed_mean": [0.1, 0.1],
            "mean_jerk_norm_seed_mean": [1.0, 1.0],
            "mean_abs_target_speed_error_seed_mean": [2.0, 2.0],
        }
    )
    ranked = screen.rank_formulations(frame)
    assert ranked.iloc[0]["pilot_config"] == "P1_reward"
    assert "rank_return_per_timestep" not in ranked.columns
