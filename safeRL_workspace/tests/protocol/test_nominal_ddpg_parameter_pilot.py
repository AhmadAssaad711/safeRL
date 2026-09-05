from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch as th
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise


import scripts.training.run_nominal_ddpg_parameter_pilot as pilot


def test_pilot_table_matches_the_prescribed_four_configurations():
    assert pilot.PILOT_CONFIGS == {
        "P0_current": {
            "actor_lr": 1e-3,
            "critic_lr": 1e-3,
            "batch_size": 64,
            "gamma": 0.98,
            "tau": 0.001,
            "learning_starts": 1_000,
            "buffer_size": 100_000,
            "ou_sigma": 0.1,
        },
        "P1_stable": {
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "learning_starts": 5_000,
            "buffer_size": 500_000,
            "ou_sigma": 0.1,
        },
        "P2_more_exploration": {
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "learning_starts": 5_000,
            "buffer_size": 500_000,
            "ou_sigma": 0.2,
        },
        "P3_slower_critic": {
            "actor_lr": 1e-4,
            "critic_lr": 3e-4,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "learning_starts": 5_000,
            "buffer_size": 500_000,
            "ou_sigma": 0.1,
        },
    }


def test_split_learning_rates_survive_model_save_and_load(tmp_path):
    env = gym.make("Pendulum-v1")
    try:
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(1, dtype=np.float32),
            sigma=0.1 * np.ones(1, dtype=np.float32),
        )
        model = pilot.SplitLearningRateDDPG(
            "MlpPolicy",
            env,
            actor_learning_rate=1e-4,
            critic_learning_rate=3e-4,
            learning_starts=10,
            buffer_size=100,
            batch_size=8,
            action_noise=action_noise,
            policy_kwargs={"net_arch": [8]},
            verbose=0,
            device="cpu",
        )
        assert model.actor.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
        assert model.critic.optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
        model.action_noise()
        model.action_noise()
        expected_ou_state = model.action_noise.noise_prev.copy()

        model_path = tmp_path / "split_lr_model"
        model.save(str(model_path))
        restored = pilot.SplitLearningRateDDPG.load(str(model_path), env=env, device="cpu")
        assert restored.actor_learning_rate == pytest.approx(1e-4)
        assert restored.critic_learning_rate == pytest.approx(3e-4)
        assert restored.actor.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
        assert restored.critic.optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
        np.testing.assert_array_equal(restored.action_noise.noise_prev, expected_ou_state)
    finally:
        env.close()


def _checkpoint_row(
    *,
    config: str,
    seed: int,
    step: int,
    distance_m: float,
    collisions: int,
    metric_value: float,
) -> dict:
    row = {
        "pilot_config": config,
        "training_seed": seed,
        "model_timestep": step,
        "total_distance_m": distance_m,
        "distinct_ego_collision_events": collisions,
        "scenarios": 1,
        "timesteps": 100,
        "total_time_s": 25.0,
        "total_return": 100.0 * metric_value,
        "task_return": 100.0 * metric_value,
        "correction_return": 0.0,
        "collision_free_scenarios": int(collisions == 0),
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
        "episode_length_sum": 100,
    }
    for metric in (
        "return_per_timestep",
        "mean_abs_speed_error",
        "episode_length_mean",
        "nominal_action_saturation_rate",
        "first_collision_observed_rate",
        "time_to_first_collision_observed_mean_s",
        "distance_to_first_collision_observed_mean_m",
        "time_to_first_collision_restricted_mean_s",
        "distance_to_first_collision_restricted_mean_m",
        "critic_mse",
        "td_abs_mean",
        "td_abs_p95",
        "q_mean",
        "q_abs_mean",
        "q_abs_p95",
        "q_abs_max",
        "target_q_abs_mean",
        "reward_abs_max",
        "q_scale_reference",
        "q_scale_ratio",
        "q_scale_excess_log10",
        "q_target_magnitude_log_gap",
        "q_nonfinite_rate",
        "latest_train_actor_loss",
        "latest_train_critic_loss",
    ):
        row[metric] = metric_value
    return row


def test_final_three_selection_discards_earlier_checkpoints_and_uses_ratio_of_sums():
    checkpoints = pd.DataFrame(
        [
            _checkpoint_row(
                config="P1_stable",
                seed=307,
                step=10_000,
                distance_m=10_000.0,
                collisions=100,
                metric_value=100.0,
            ),
            _checkpoint_row(
                config="P1_stable",
                seed=307,
                step=20_000,
                distance_m=100.0,
                collisions=1,
                metric_value=1.0,
            ),
            _checkpoint_row(
                config="P1_stable",
                seed=307,
                step=30_000,
                distance_m=200.0,
                collisions=0,
                metric_value=2.0,
            ),
            _checkpoint_row(
                config="P1_stable",
                seed=307,
                step=40_000,
                distance_m=300.0,
                collisions=2,
                metric_value=3.0,
            ),
        ]
    )

    result = pilot.final_three_seed_averages(checkpoints).iloc[0]

    assert result["checkpoint_steps"] == "20000,30000,40000"
    assert result["total_distance_m"] == pytest.approx(600.0)
    assert result["distinct_ego_collision_events"] == pytest.approx(3.0)
    assert result["distance_per_collision_m"] == pytest.approx(200.0)
    assert result["return_per_timestep"] == pytest.approx(2.0)


def test_across_seed_statistics_weight_training_seeds_equally():
    seed_rows = pd.DataFrame(
        [
            {
                **_checkpoint_row(
                    config="P1_stable",
                    seed=307,
                    step=50_000,
                    distance_m=100.0,
                    collisions=1,
                    metric_value=1.0,
                ),
                "checkpoint_count": 3,
                "checkpoint_steps": "30000,40000,50000",
                "distance_per_collision_m": 100.0,
                "distance_per_collision_right_censored": 0,
                "distance_per_collision_exposure_bound_m": 100.0,
                "ego_collisions_per_km": 10.0,
            },
            {
                **_checkpoint_row(
                    config="P1_stable",
                    seed=1307,
                    step=50_000,
                    distance_m=900.0,
                    collisions=1,
                    metric_value=3.0,
                ),
                "checkpoint_count": 3,
                "checkpoint_steps": "30000,40000,50000",
                "distance_per_collision_m": 900.0,
                "distance_per_collision_right_censored": 0,
                "distance_per_collision_exposure_bound_m": 900.0,
                "ego_collisions_per_km": 10.0 / 9.0,
            },
        ]
    ).drop(columns=["model_timestep"])

    across = pilot.across_seed_final_three(seed_rows).iloc[0]

    assert across["training_seeds"] == 2
    assert across["distance_per_collision_m_seed_mean"] == pytest.approx(500.0)
    assert across["distance_per_collision_m_seed_variance"] == pytest.approx(320_000.0)
    assert across["return_per_timestep_seed_mean"] == pytest.approx(2.0)


def test_zero_collision_final_window_remains_right_censored_not_epsilon_clamped():
    checkpoints = pd.DataFrame(
        [
            _checkpoint_row(
                config="P2_more_exploration",
                seed=307,
                step=step,
                distance_m=50.0,
                collisions=0,
                metric_value=1.0,
            )
            for step in (30_000, 40_000, 50_000)
        ]
    )

    result = pilot.final_three_seed_averages(checkpoints).iloc[0]

    assert np.isinf(result["distance_per_collision_m"])
    assert result["distance_per_collision_right_censored"] == 1
    assert result["distance_per_collision_exposure_bound_m"] == pytest.approx(150.0)


def test_resume_uses_remaining_sb3_target_timesteps():
    remaining, learn_total_timesteps = pilot.sb3_resume_learn_target_timesteps(
        target_timesteps=50_000,
        current_timesteps=40_000,
    )

    assert remaining == 10_000
    assert learn_total_timesteps == 10_000


def test_resume_rejects_checkpoint_beyond_target():
    with pytest.raises(RuntimeError, match="exceeds target"):
        pilot.sb3_resume_learn_target_timesteps(
            target_timesteps=50_000,
            current_timesteps=60_000,
        )


def test_output_directory_lock_rejects_only_a_second_invocation(tmp_path):
    output_dir = tmp_path / "pilot"
    first = pilot.OutputDirectoryRunLock(output_dir)
    second = pilot.OutputDirectoryRunLock(output_dir)

    first.acquire()
    try:
        with pytest.raises(RuntimeError, match=r"already holds the output lock.*pid=.*host="):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()


def test_confirmation_defaults_are_p0_p2_three_seeds_and_150k():
    args = Namespace(
        stage="confirm",
        selected_configs=None,
        screen_ranking=None,
    )

    assert pilot.resolve_selected_configs(args, Path.cwd()) == [
        "P0_current",
        "P2_more_exploration",
    ]
    assert pilot.DEFAULT_CONFIRM_SEEDS == (307, 1307, 2307)
    assert pilot.expected_checkpoint_steps(150_000, 10_000) == list(
        range(10_000, 150_001, 10_000)
    )
    with pytest.raises(ValueError, match="locked to the screened finalists"):
        pilot.resolve_selected_configs(
            Namespace(
                stage="confirm",
                selected_configs=["P1_stable", "P3_slower_critic"],
                screen_ranking=None,
            ),
            Path.cwd(),
        )


def test_discounted_return_and_terminal_vs_censored_calibration_targets():
    np.testing.assert_allclose(
        pilot.pipeline.discounted_return_to_go([1.0, 2.0, 100.0], gamma=0.5),
        [27.0, 52.0, 100.0],
    )
    anchors = [
        {"anchor_segment_step": 0, "anchor_global_step": 10, "q_value": 30.0},
        {"anchor_segment_step": 1, "anchor_global_step": 11, "q_value": 50.0},
    ]
    terminal_samples: list[dict] = []
    pilot.pipeline.append_critic_calibration_segment(
        terminal_samples,
        anchors=anchors,
        rewards=[1.0, 2.0, 100.0],
        gamma=0.5,
        terminal_observed=True,
        truncated=False,
        collision_terminal=True,
        tail_q=np.nan,
        reward_scale="environment_raw",
        variant="P0_current",
        mode="raw",
        training_seed=307,
        scenario_seed=900000,
        segment_index=0,
        censor_reason="",
    )
    assert [row["empirical_discounted_return"] for row in terminal_samples] == [27.0, 52.0]
    assert all(row["target_kind"] == "terminal_mc" for row in terminal_samples)

    censored_samples: list[dict] = []
    pilot.pipeline.append_critic_calibration_segment(
        censored_samples,
        anchors=anchors[:1],
        rewards=[1.0, 2.0],
        gamma=0.5,
        terminal_observed=False,
        truncated=True,
        collision_terminal=False,
        tail_q=8.0,
        reward_scale="environment_raw",
        variant="P0_current",
        mode="raw",
        training_seed=307,
        scenario_seed=900000,
        segment_index=1,
        censor_reason="environment_truncation",
    )
    sample = censored_samples[0]
    assert np.isnan(sample["empirical_discounted_return"])
    assert sample["partial_discounted_return"] == pytest.approx(2.0)
    assert sample["gamma_tail"] == pytest.approx(0.25)
    assert sample["bootstrapped_discounted_return"] == pytest.approx(4.0)
    assert sample["right_censored"] == 1


def _calibration_sample(q: float, empirical: float, *, exact: bool = True) -> dict:
    return {
        "q_value": q,
        "empirical_discounted_return": empirical if exact else np.nan,
        "bootstrapped_discounted_return": empirical,
        "terminal_mc_included": int(exact),
        "gamma_tail": 0.0 if exact else 0.5,
    }


def test_calibration_summary_measures_overestimation_and_excludes_censored():
    summary = pilot.summarize_critic_calibration_samples(
        [
            _calibration_sample(2.0, 1.0),
            _calibration_sample(5.0, 3.0),
            _calibration_sample(100.0, 90.0, exact=False),
        ]
    )

    assert summary["critic_calibration_anchor_count"] == 3
    assert summary["critic_calibration_exact_anchor_count"] == 2
    assert summary["critic_calibration_censored_anchor_count"] == 1
    assert summary["critic_calibration_exact_coverage"] == pytest.approx(2 / 3)
    assert summary["critic_calibration_bias_mean"] == pytest.approx(1.5)
    assert summary["critic_calibration_mae"] == pytest.approx(1.5)
    assert summary["critic_calibration_rmse"] == pytest.approx(np.sqrt(2.5))
    assert summary["critic_calibration_overestimation_rate"] == pytest.approx(1.0)
    assert summary["critic_calibration_pearson_r"] == pytest.approx(1.0)
    assert summary["critic_calibration_empirical_on_q_slope"] == pytest.approx(2 / 3)
    assert summary["critic_calibration_empirical_on_q_intercept"] == pytest.approx(-1 / 3)


def test_calibration_checkpoint_aggregation_weights_scenarios_not_anchor_counts():
    large_scenario = pilot.summarize_critic_calibration_samples(
        [_calibration_sample(11.0, 1.0) for _ in range(100)]
    )
    small_scenario = pilot.summarize_critic_calibration_samples(
        [_calibration_sample(1.0, 1.0)]
    )

    checkpoint = pilot.aggregate_calibration_scenario_rows([large_scenario, small_scenario])

    assert checkpoint["critic_calibration_finite_exact_anchor_count"] == 101
    assert checkpoint["critic_calibration_bias_mean"] == pytest.approx(5.0)
    assert checkpoint["critic_calibration_mae"] == pytest.approx(5.0)


class _FakePolicy:
    def __init__(self):
        self.scaled_action = None

    def scale_action(self, action):
        self.scaled_action = np.asarray(action).copy()
        return np.asarray(action) * 0.5

    def obs_to_tensor(self, obs):
        return th.as_tensor(np.asarray(obs), dtype=th.float32).reshape(1, -1), False


class _FakeModel:
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.policy = _FakePolicy()
        self.device = "cpu"
        self.critic_action = None

    def get_vec_normalize_env(self):
        return None

    def predict(self, obs, deterministic=True):
        assert deterministic
        return np.asarray([0.5, -0.5], dtype=np.float32), None

    def critic(self, obs_tensor, action_tensor):
        self.critic_action = action_tensor.detach().cpu().numpy().copy()
        return (action_tensor.sum(dim=1, keepdim=True),)


def test_calibration_critic_receives_the_exact_actor_action_in_replay_scale():
    model = _FakeModel()
    physical, q_value = pilot.pipeline.policy_action_and_q_physical(
        model=model,
        obs=np.asarray([1.0, 2.0], dtype=np.float32),
        env_config={
            "bounds": {"ax_min": -4.0, "ax_max": 2.0, "ay_min": -2.0, "ay_max": 2.0}
        },
        rng=np.random.default_rng(1),
        compute_q=True,
    )

    np.testing.assert_allclose(model.policy.scaled_action, [[0.5, -0.5]])
    np.testing.assert_allclose(model.critic_action, [[0.25, -0.25]])
    # The lane-free environment uses a zero-preserving piecewise map when an
    # action bound straddles zero: positive normalized longitudinal actions
    # scale by ax_max, while negative ones scale by |ax_min|.
    np.testing.assert_allclose(physical, [1.0, -1.0])
    assert q_value == pytest.approx(0.0)


def test_evaluation_coverage_rejects_missing_and_duplicate_rows():
    scenario_rows = []
    for step in (10_000, 20_000):
        for scenario_seed in (900000, 900001):
            scenario_rows.append(
                {
                    "pilot_config": "P0_current",
                    "training_seed": 307,
                    "model_timestep": step,
                    "scenario_seed": scenario_seed,
                    "timesteps": 800,
                }
            )
    scenarios = pd.DataFrame(scenario_rows)
    diagnostics = pd.DataFrame(
        [
            {"pilot_config": "P0_current", "training_seed": 307, "model_timestep": step}
            for step in (10_000, 20_000)
        ]
    )
    pilot.validate_run_evaluation_coverage(
        scenarios,
        diagnostics,
        pd.DataFrame(),
        pilot_config="P0_current",
        training_seed=307,
        checkpoint_steps=[10_000, 20_000],
        eval_seeds=[900000, 900001],
        eval_timesteps=800,
        calibration_enabled=False,
    )

    with pytest.raises(RuntimeError, match="coverage mismatch"):
        pilot.validate_run_evaluation_coverage(
            scenarios.iloc[:-1],
            diagnostics,
            pd.DataFrame(),
            pilot_config="P0_current",
            training_seed=307,
            checkpoint_steps=[10_000, 20_000],
            eval_seeds=[900000, 900001],
            eval_timesteps=800,
            calibration_enabled=False,
        )
    with pytest.raises(RuntimeError, match="Duplicate evaluation scenario keys"):
        pilot.validate_run_evaluation_coverage(
            pd.concat([scenarios, scenarios.iloc[[0]]], ignore_index=True),
            diagnostics,
            pd.DataFrame(),
            pilot_config="P0_current",
            training_seed=307,
            checkpoint_steps=[10_000, 20_000],
            eval_seeds=[900000, 900001],
            eval_timesteps=800,
            calibration_enabled=False,
        )


def test_final_three_requires_aligned_checkpoint_steps_across_runs():
    rows = []
    for config, steps in (
        ("P0_current", (10_000, 20_000, 30_000)),
        ("P2_more_exploration", (10_000, 20_000, 40_000)),
    ):
        rows.extend(
            _checkpoint_row(
                config=config,
                seed=307,
                step=step,
                distance_m=100.0,
                collisions=1,
                metric_value=1.0,
            )
            for step in steps
        )

    with pytest.raises(RuntimeError, match="not aligned"):
        pilot.final_three_seed_averages(pd.DataFrame(rows))


def _confirmation_rank_row(config: str, values: tuple[float, float, float, float, float]) -> dict:
    distance, ret, speed_error, episode_length, saturation = values
    row = {
        "pilot_config": config,
        "distance_per_collision_exposure_bound_m_seed_mean": distance,
        "return_per_timestep_seed_mean": ret,
        "mean_abs_speed_error_seed_mean": speed_error,
        "episode_length_mean_seed_mean": episode_length,
        "nominal_action_saturation_rate_seed_mean": saturation,
        "distance_per_collision_exposure_bound_m_seed_min": distance,
        "return_per_timestep_seed_min": ret,
        "mean_abs_speed_error_seed_max": speed_error,
        "episode_length_mean_seed_min": episode_length,
        "nominal_action_saturation_rate_seed_max": saturation,
        "critic_calibration_q_nonfinite_rate_seed_mean": 0.0,
        "critic_calibration_finite_exact_anchor_count_seed_mean": 100.0,
    }
    for metric in (
        "distance_per_collision_exposure_bound_m",
        "return_per_timestep",
        "mean_abs_speed_error",
        "episode_length_mean",
        "nominal_action_saturation_rate",
    ):
        row[f"{metric}_seed_variance"] = 1.0
    return row


def test_confirmation_rank_uses_rollouts_not_scale_dependent_critic_diagnostics():
    p0 = _confirmation_rank_row("P0_current", (400.0, 0.29, 3.5, 85.0, 0.17))
    p2 = _confirmation_rank_row("P2_more_exploration", (460.0, 0.30, 3.4, 100.0, 0.08))
    p0.update({"critic_mse_seed_mean": 0.1, "q_abs_mean_seed_mean": 1.0})
    p2.update({"critic_mse_seed_mean": 10_000.0, "q_abs_mean_seed_mean": 1_000.0})

    ranked = pilot.rank_confirmation_rollout(pd.DataFrame([p0, p2]))

    assert ranked.iloc[0]["pilot_config"] == "P2_more_exploration"
    assert "critic_mse_seed_mean" in ranked.columns


def test_paired_seed_differences_preserve_seed_pairing():
    rows = []
    for seed, p0, p2 in ((307, 1.0, 3.0), (1307, 2.0, 6.0), (2307, 5.0, 4.0)):
        rows.extend(
            [
                {
                    "pilot_config": "P0_current",
                    "training_seed": seed,
                    "checkpoint_steps": "130000,140000,150000",
                    "return_per_timestep": p0,
                },
                {
                    "pilot_config": "P2_more_exploration",
                    "training_seed": seed,
                    "checkpoint_steps": "130000,140000,150000",
                    "return_per_timestep": p2,
                },
            ]
        )

    paired, summary = pilot.paired_seed_differences(pd.DataFrame(rows))

    assert paired["delta_return_per_timestep"].tolist() == pytest.approx([2.0, 4.0, -1.0])
    assert summary.iloc[0]["delta_return_per_timestep_mean"] == pytest.approx(5 / 3)
    assert summary.iloc[0]["delta_return_per_timestep_variance"] == pytest.approx(19 / 3)
