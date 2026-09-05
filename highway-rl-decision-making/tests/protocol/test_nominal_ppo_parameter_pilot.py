from __future__ import annotations

import json
import pickle
from argparse import Namespace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer


import scripts.training.run_nominal_ppo_parameter_pilot as pilot


def test_ppo_pilot_table_and_exact_boundaries():
    assert list(pilot.PPO_CONFIGS) == [
        "Q0_current_aligned",
        "Q1_stable",
        "Q2_exploratory",
        "Q3_conservative_update",
    ]
    assert pilot.PPO_CONFIGS["Q0_current_aligned"]["learning_rate"] == pytest.approx(3e-4)
    assert pilot.PPO_CONFIGS["Q1_stable"]["log_std_init"] == pytest.approx(-0.5)
    assert pilot.PPO_CONFIGS["Q2_exploratory"]["ent_coef"] == pytest.approx(0.02)
    assert pilot.PPO_CONFIGS["Q3_conservative_update"]["clip_range"] == pytest.approx(0.15)
    for config in pilot.PPO_CONFIGS.values():
        assert pilot.validate_rollout_alignment(
            config,
            n_envs=1,
            target_timesteps=50_000,
            checkpoint_interval=10_000,
        ) == 1_000
    assert pilot.expected_checkpoint_steps(50_000, 10_000) == [
        10_000,
        20_000,
        30_000,
        40_000,
        50_000,
    ]


def test_old_2048_rollout_is_rejected_for_exact_50k():
    config = dict(pilot.PPO_CONFIGS["Q0_current_aligned"])
    config.update(n_steps=2_048, batch_size=128)
    with pytest.raises(ValueError, match="target_timesteps.*rollout_size"):
        pilot.validate_rollout_alignment(
            config,
            n_envs=1,
            target_timesteps=50_000,
            checkpoint_interval=10_000,
        )


def test_parallel_config_keeps_the_same_global_ppo_rollout_size():
    args = Namespace(n_envs=8, global_rollout_size=1_000)
    config = pilot.effective_ppo_config("Q1_stable", args)

    assert config["n_steps"] == 125
    assert pilot.validate_rollout_alignment(
        config,
        n_envs=8,
        target_timesteps=50_000,
        checkpoint_interval=10_000,
    ) == 1_000


def test_progress_reward_override_is_explicit_and_non_mutating():
    base = {"progress_reward_weight": 0.0, "progress_clip": 1.25}

    unchanged = pilot.apply_reward_overrides(base, Namespace())
    enabled = pilot.apply_reward_overrides(
        base, Namespace(progress_reward_weight=0.5)
    )

    assert unchanged == base
    assert enabled["progress_reward_weight"] == pytest.approx(0.5)
    assert enabled["progress_clip"] == pytest.approx(1.25)
    assert base["progress_reward_weight"] == pytest.approx(0.0)


def test_progress_reward_override_rejects_nonfinite_weight():
    with pytest.raises(ValueError, match="progress-reward-weight must be finite"):
        pilot.apply_reward_overrides(
            {"progress_reward_weight": 0.0},
            Namespace(progress_reward_weight=np.inf),
        )


def test_jerk_reward_overrides_are_explicit_and_non_mutating():
    base = {"jerk_penalty_weight": 0.02, "jerk_scale": 10.0}

    enabled = pilot.apply_reward_overrides(
        base, Namespace(jerk_penalty_weight=0.01, jerk_scale=8.0)
    )

    assert enabled["jerk_penalty_weight"] == pytest.approx(0.01)
    assert enabled["jerk_scale"] == pytest.approx(8.0)
    assert base["jerk_penalty_weight"] == pytest.approx(0.02)
    assert base["jerk_scale"] == pytest.approx(10.0)


def test_jerk_reward_overrides_reject_invalid_values():
    with pytest.raises(ValueError, match="jerk-penalty-weight must be finite and non-negative"):
        pilot.apply_reward_overrides(
            {"jerk_penalty_weight": 0.02},
            Namespace(jerk_penalty_weight=-0.1),
        )
    with pytest.raises(ValueError, match="jerk-scale must be finite and positive"):
        pilot.apply_reward_overrides(
            {"jerk_scale": 10.0},
            Namespace(jerk_scale=0.0),
        )


def test_nominal_pilot_evaluates_only_the_final_checkpoint_by_default():
    assert pilot.evaluation_steps(
        50_000, 10_000, evaluate_checkpoints=False
    ) == [50_000]
    assert pilot.evaluation_steps(
        50_000, 10_000, evaluate_checkpoints=True
    ) == [10_000, 20_000, 30_000, 40_000, 50_000]
    assert not pilot.checkpoint_evaluation_enabled(
        Namespace(evaluate_checkpoints=False)
    )
    # Other runners which reuse the shared PPO machinery retain their legacy
    # per-checkpoint behavior until they explicitly opt into final-only mode.
    assert pilot.checkpoint_evaluation_enabled(Namespace())


class _FakeLogger:
    def record(self, *_args, **_kwargs):
        return None


class _FakeModel:
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    logger = _FakeLogger()


def test_raw_ppo_action_clipping_is_counted_and_resumable():
    callback = pilot.PPOActionClipCallback()
    callback.model = _FakeModel()
    callback.locals = {
        "actions": np.asarray([[1.5, -0.25]], dtype=np.float32),
        "clipped_actions": np.asarray([[1.0, -0.25]], dtype=np.float32),
    }
    assert callback._on_step()
    state = callback.state_dict()

    restored = pilot.PPOActionClipCallback()
    restored.load_state_dict(state)
    metrics = restored.consume_checkpoint_metrics()
    assert metrics["actor_raw_action_components"] == 2
    assert metrics["actor_raw_action_clip_rate"] == pytest.approx(0.5)
    assert metrics["actor_raw_action_saturation_rate"] == pytest.approx(0.5)
    assert metrics["actor_raw_action_abs_max"] == pytest.approx(1.5)


class _FakeRolloutBuffer:
    def __init__(self):
        self.observations = np.asarray(
            [[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32
        )
        self.returns = np.asarray([[3.0], [7.0]], dtype=np.float32)
        self.values = np.asarray([[0.0], [0.0]], dtype=np.float32)


class _FakeValuePolicy:
    training = True

    def set_training_mode(self, mode):
        self.training = bool(mode)

    def obs_to_tensor(self, observations):
        return th.as_tensor(observations, dtype=th.float32), False

    def predict_values(self, observations):
        return observations.sum(dim=1, keepdim=True)


class _FakeValueModel:
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
    rollout_buffer = _FakeRolloutBuffer()
    policy = _FakeValuePolicy()


def test_rollout_diagnostics_survive_sb3_buffer_reset_and_flattening():
    model = _FakeValueModel()
    cache = pilot.PPORolloutDiagnosticsCache()
    cache.model = model
    cache._on_rollout_end()

    # SB3 resets the buffer before the next on_rollout_start and its generator
    # may flatten arrays during PPO.train(); the copied snapshot must survive.
    model.rollout_buffer.observations = np.zeros((4,), dtype=np.float32)
    model.rollout_buffer.returns = np.zeros((2,), dtype=np.float32)
    diagnostics = pilot.ppo_value_diagnostics(model, cache.consume())

    assert diagnostics["rollout_value_samples"] == 2
    assert diagnostics["rollout_value_target_mse"] == pytest.approx(0.0)
    assert diagnostics["rollout_preupdate_value_target_mse"] == pytest.approx(29.0)


def test_real_sb3_rollout_buffer_lifecycle_does_not_destroy_cached_diagnostics():
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    buffer = RolloutBuffer(
        2,
        observation_space,
        action_space,
        device="cpu",
        gamma=0.99,
        gae_lambda=0.95,
        n_envs=1,
    )
    for observation, reward in (([1.0, 2.0], 1.0), ([3.0, 4.0], 2.0)):
        buffer.add(
            np.asarray([observation], dtype=np.float32),
            np.asarray([[0.0]], dtype=np.float32),
            np.asarray([reward], dtype=np.float32),
            np.asarray([False]),
            th.tensor([0.0]),
            th.tensor([0.0]),
        )
    buffer.compute_returns_and_advantage(
        last_values=th.tensor([0.0]), dones=np.asarray([True])
    )
    model = _FakeValueModel()
    model.rollout_buffer = buffer
    cache = pilot.PPORolloutDiagnosticsCache()
    cache.model = model
    cache._on_rollout_end()
    expected_returns = cache.snapshot["returns"].copy()

    list(buffer.get(batch_size=2))  # SB3 flattens in-place for PPO.train().
    buffer.reset()  # The next collect_rollouts() does this before on_rollout_start.
    snapshot = cache.consume()

    np.testing.assert_array_equal(snapshot["returns"], expected_returns)
    diagnostics = pilot.ppo_value_diagnostics(model, snapshot)
    assert diagnostics["rollout_value_samples"] == 2
    assert diagnostics["rollout_value_nonfinite_rate"] == pytest.approx(0.0)


def _checkpoint_row(step: int, distance: float, collisions: int, value: float) -> dict:
    return {
        "pilot_config": "Q1_stable",
        "variant": "Q1_stable",
        "mode": "raw",
        "training_seed": 307,
        "model_timestep": step,
        "scenarios": 10,
        "timesteps": 8_000,
        "total_time_s": 2_000.0,
        "total_return": 8_000.0 * value,
        "task_return": 8_000.0 * value,
        "correction_return": 0.0,
        "collision_free_scenarios": int(collisions == 0) * 10,
        "collision_transition_timesteps": collisions,
        "collision_transition_return": -2.5 * collisions,
        "post_collision_timesteps": 0,
        "post_collision_return": 0.0,
        "reset_calls_total": 10 + collisions,
        "resets_after_collision": collisions,
        "resets_after_truncation_only": 0,
        "resets_after_other_terminal": 0,
        "episode_segments": 10 + collisions,
        "completed_segments": collisions,
        "right_censored_segments": 10,
        "episode_length_sum": 8_000,
        "total_distance_m": distance,
        "distinct_ego_collision_events": collisions,
        "distance_per_collision_m": np.inf if collisions == 0 else distance / collisions,
        "distance_per_collision_right_censored": int(collisions == 0),
        "distance_per_collision_exposure_bound_m": distance if collisions == 0 else distance / collisions,
        "ego_collisions_per_km": 1_000 * collisions / distance,
        "return_per_timestep": value,
        "episode_length_mean": 8_000 / (10 + collisions),
        "mean_abs_speed_error": value,
        "nominal_action_saturation_rate": value / 10,
        "latest_train_value_loss": value,
        "rollout_value_nonfinite_rate": 0.0,
        "actor_raw_action_clip_rate": value / 10,
        "actor_raw_action_components": 20_000,
        "actor_raw_action_clipped_components": 2_000 * value,
        "actor_raw_action_saturated_components": 2_000 * value,
        "actor_raw_action_abs_sum": 10_000 * value,
        "actor_raw_action_saturation_rate": value / 10,
        "actor_raw_action_abs_mean": value / 2,
        "actor_raw_action_abs_max": value,
        "actor_raw_action_clip_rate_cumulative": value / 10,
        "n_updates": step // 100,
        "completed_rollouts": step // 1_000,
    }


def test_final_three_uses_ratio_of_sums_and_discards_early_checkpoint():
    checkpoint = pd.DataFrame(
        [
            _checkpoint_row(10_000, 100_000.0, 100, 100.0),
            _checkpoint_row(20_000, 100.0, 1, 1.0),
            _checkpoint_row(30_000, 200.0, 0, 2.0),
            _checkpoint_row(40_000, 300.0, 2, 3.0),
        ]
    )
    result = pilot.final_three_seed_averages(checkpoint).iloc[0]
    assert result["checkpoint_steps"] == "20000,30000,40000"
    assert result["total_distance_m"] == pytest.approx(600.0)
    assert result["distinct_ego_collision_events"] == pytest.approx(3.0)
    assert result["distance_per_collision_m"] == pytest.approx(200.0)
    assert result["return_per_timestep"] == pytest.approx(2.0)
    assert result["latest_train_value_loss"] == pytest.approx(2.0)
    assert result["actor_raw_action_components"] == 60_000
    assert result["actor_raw_action_clip_rate"] == pytest.approx(0.2)
    assert result["actor_raw_action_abs_max"] == pytest.approx(3.0)
    assert result["completed_rollouts"] == 40


def test_final_window_supports_a_single_final_only_evaluation():
    checkpoint = pd.DataFrame([_checkpoint_row(50_000, 500.0, 2, 4.0)])
    result = pilot.final_three_seed_averages(checkpoint).iloc[0]

    assert result["checkpoint_count"] == 1
    assert result["checkpoint_steps"] == "50000"
    assert result["total_distance_m"] == pytest.approx(500.0)
    assert result["distinct_ego_collision_events"] == pytest.approx(2.0)
    assert result["distance_per_collision_m"] == pytest.approx(250.0)
    assert result["return_per_timestep"] == pytest.approx(4.0)


def test_ppo_checkpoint_validator_requires_no_replay_payload(tmp_path):
    bundle = tmp_path / "000010000"
    bundle.mkdir()
    for name, data in {
        "model.zip": b"model",
        "env.pkl": b"environment",
    }.items():
        (bundle / name).write_bytes(data)
    state = {
        "schema_version": pilot.PPO_CHECKPOINT_SCHEMA_VERSION,
        "phase": "post_update_boundary",
        "timestep": 10_000,
        "n_updates": 100,
        "completed_rollouts": 10,
        "rollout_size": 1_000,
        "n_steps": 1_000,
        "n_envs": 1,
        "training_config_hash": "abc",
    }
    with (bundle / "state.pkl").open("wb") as handle:
        pickle.dump(state, handle)
    payloads = ["model.zip", "env.pkl", "state.pkl"]
    manifest = {
        "schema_version": pilot.PPO_CHECKPOINT_SCHEMA_VERSION,
        "phase": "post_update_boundary",
        "timestep": 10_000,
        "n_updates": 100,
        "rollout_size": 1_000,
        "training_config_hash": "abc",
        "model_class": "stable_baselines3.ppo.ppo.PPO",
        "checksums": {
            name: pilot.pipeline.file_sha256(bundle / name) for name in payloads
        },
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded_manifest, loaded_state = pilot.validate_ppo_checkpoint_bundle(
        bundle,
        "abc",
        expected_model_class="stable_baselines3.ppo.ppo.PPO",
        expected_rollout_size=1_000,
    )
    assert loaded_manifest["timestep"] == 10_000
    assert loaded_state["phase"] == "post_update_boundary"
    assert not (bundle / "replay.pkl").exists()


def test_resume_budget_uses_only_remaining_steps():
    assert pilot.sb3_resume_learn_target_timesteps(50_000, 30_000) == (20_000, 20_000)
    with pytest.raises(RuntimeError, match="exceeds target"):
        pilot.sb3_resume_learn_target_timesteps(50_000, 60_000)


def test_notebook_ppo_cell_delegates_to_canonical_cbf_progression():
    notebook_path = Path(__file__).resolve().parents[2] / "notebooks" / "lanelessKaralakou.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("id") == "eb9eade5"
    )
    compile(source, f"{notebook_path}:ppo-cbf-progression", "exec")
    assert "run_ppo_cbf_progression.py" in source
    assert "PPO_1M_TIMESTEPS_PER_POLICY" in source
    assert "PPO_1M_RUN_TRAINING" in source
    assert "PPO_1M_FORCE_RETRAIN" in source
    assert "PPO_1M_REQUIRE_CUDA = True" in source
    assert '"--traffic-model", "mtm"' in source
    assert '"ppo_nominal"' in source
    assert '"ppo_cbf_reward"' in source
    assert '"ppo_cbf_nd_reward_actor"' in source
    assert '"ppo_cbf_nd_actor_only"' in source
    assert '"ppo_cbf_diff_reward_only"' in source
    assert '"ppo_cbf_projected_reward_off"' in source
    assert '"ppo_cbf_integrated_actor_only"' in source
    assert '"ppo_cbf_integrated_actor_critic"' not in source
    assert '"--lambda-critic"' not in source
    assert '"--lambda-detached-actor", "0.10"' in source
    assert "model = PPO(" not in source
    assert "subprocess.Popen" in source
    assert "stdout=subprocess.PIPE" in source
    assert "stderr=subprocess.STDOUT" in source
    assert "PPO_1M_POST_TRAIN_EVAL_EPISODES = 200" in source
    assert "--task-distance-m" in source
