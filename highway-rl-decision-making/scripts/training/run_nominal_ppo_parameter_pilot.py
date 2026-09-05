"""Exact 50k nominal-PPO parameter pilot for the lane-free environment.

This is the authoritative PPO pilot entry point.  It keeps the environment,
reward, traffic setup, and tuned CBF snapshot fixed, while training without a
CBF.  Every evaluation uses the corrected fixed-timestep collision protocol.

The default screen trains Q0--Q3 with one common seed for exactly 50,000
timesteps.  It retains lightweight 10,000-step model snapshots but evaluates
only the final post-update policy by default; per-checkpoint evaluation is an
explicit opt-in because it serializes a large simulator workload into training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import scripts.training.run_cbf_filter_ablation as pipeline
import scripts.training.run_nominal_ddpg_parameter_pilot as pilot_common
from scripts.common.laneless_script_config import (
    active_traffic_model,
    add_env_config_args,
    env_config_from_args,
)
from scripts.training.train_safety_potential_variants import MTM_CONGESTED_UNCERTAIN_UPDATES, deep_update
from scripts.common.ppo_observation_variants import install_previous_action_observation
from scripts.common.ppo_parallel_worker import make_parallel_subproc_training_env


PPO_PILOT_SCHEMA_VERSION = 3
PPO_CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_TRAINING_SEED = 307
DEFAULT_EVAL_SEED_START = 900_000
DEFAULT_EVAL_SCENARIOS = 10
DEFAULT_TIMESTEPS = 50_000
DEFAULT_CHECKPOINT_INTERVAL = 10_000
DEFAULT_EVAL_TIMESTEPS = 800
DEFAULT_GLOBAL_ROLLOUT_SIZE = 1_000


def validate_training_device(device: str) -> str:
    """Reject a silent CPU fallback when a CUDA run was requested."""

    requested = str(device).strip()
    if requested.lower().startswith("cuda") and not th.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested ({requested!r}) but torch.cuda.is_available() is False"
        )
    return requested


def configure_parallel_runtime() -> None:
    """Avoid CPU-thread oversubscription beside SubprocVecEnv workers."""

    th.set_num_threads(1)
    try:
        th.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits this only before its first parallel operation.
        pass


def default_tensorboard_root() -> Path:
    """Keep TensorBoard event files off long OneDrive workspace paths."""

    configured = os.environ.get("NOMINAL_PPO_PILOT_TENSORBOARD_ROOT")
    if configured:
        return Path(configured)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "agv_ppo_tensorboard"
    return Path(tempfile.gettempdir()) / "agv_ppo_tensorboard"


def effective_ppo_config(pilot_config: str, args: argparse.Namespace) -> dict[str, Any]:
    """Keep each PPO rollout at a fixed global transition count.

    Eight workers with ``n_steps=125`` still collect 1,000 transitions per
    PPO update, matching the original one-worker ``n_steps=1000`` protocol.
    """

    config = copy.deepcopy(PPO_CONFIGS[pilot_config])
    n_envs = max(1, int(args.n_envs))
    global_rollout_size = int(
        getattr(args, "global_rollout_size", DEFAULT_GLOBAL_ROLLOUT_SIZE)
    )
    if global_rollout_size <= 0 or global_rollout_size % n_envs != 0:
        raise ValueError(
            "global_rollout_size must be positive and divisible by n_envs: "
            f"{global_rollout_size=} {n_envs=}"
        )
    config["n_steps"] = global_rollout_size // n_envs
    return config


def apply_reward_overrides(
    reward_config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Return the fixed task reward with explicit pilot-only overrides applied."""

    config = copy.deepcopy(reward_config)
    progress_reward_weight = getattr(args, "progress_reward_weight", None)
    if progress_reward_weight is not None:
        if not np.isfinite(float(progress_reward_weight)):
            raise ValueError("--progress-reward-weight must be finite")
        config["progress_reward_weight"] = float(progress_reward_weight)
    jerk_penalty_weight = getattr(args, "jerk_penalty_weight", None)
    if jerk_penalty_weight is not None:
        if not np.isfinite(float(jerk_penalty_weight)) or float(jerk_penalty_weight) < 0.0:
            raise ValueError("--jerk-penalty-weight must be finite and non-negative")
        config["jerk_penalty_weight"] = float(jerk_penalty_weight)
    jerk_scale = getattr(args, "jerk_scale", None)
    if jerk_scale is not None:
        if not np.isfinite(float(jerk_scale)) or float(jerk_scale) <= 0.0:
            raise ValueError("--jerk-scale must be finite and positive")
        config["jerk_scale"] = float(jerk_scale)
    return config


PPO_CHECKPOINT_PAYLOADS = {
    "model": "model.zip",
    "base_environment": "env.pkl",
    "pipeline_state": "state.pkl",
    "vecnormalize": "vec.pkl",
}

PPO_CONFIGS: dict[str, dict[str, float | int]] = {
    "Q0_current_aligned": {
        "learning_rate": 3e-4,
        "n_steps": 1_000,
        "batch_size": 100,
        "n_epochs": 10,
        "gamma": 0.98,
        "gae_lambda": 0.95,
        "clip_range": 0.20,
        "ent_coef": 0.010,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "log_std_init": 0.0,
    },
    "Q1_stable": {
        "learning_rate": 1e-4,
        "n_steps": 1_000,
        "batch_size": 100,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.20,
        "ent_coef": 0.005,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "log_std_init": -0.5,
    },
    "Q2_exploratory": {
        "learning_rate": 1e-4,
        "n_steps": 1_000,
        "batch_size": 100,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.20,
        "ent_coef": 0.020,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "log_std_init": -0.25,
    },
    "Q3_conservative_update": {
        "learning_rate": 2e-4,
        "n_steps": 1_000,
        "batch_size": 100,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.15,
        "ent_coef": 0.010,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "log_std_init": -0.5,
    },
}


def _append_frame(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        mode="a",
        header=not path.exists() or path.stat().st_size == 0,
        index=False,
    )


def _finite_mean(values: Any, default: float = np.nan) -> float:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float(default)


def _finite_p95(values: np.ndarray, default: float = np.nan) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, 95)) if array.size else float(default)


def _run_dir(output_dir: Path, training_seed: int, pilot_config: str) -> Path:
    return output_dir / f"seed_{int(training_seed)}" / str(pilot_config)


def validate_rollout_alignment(
    config_values: dict[str, Any],
    *,
    n_envs: int,
    target_timesteps: int,
    checkpoint_interval: int,
) -> int:
    """Reject PPO settings that would overshoot or create pre-update labels."""

    n_steps = int(config_values["n_steps"])
    batch_size = int(config_values["batch_size"])
    rollout_size = n_steps * int(n_envs)
    if rollout_size <= 0 or batch_size <= 1:
        raise ValueError("PPO rollout and batch sizes must be positive, with batch_size > 1")
    if batch_size > rollout_size or rollout_size % batch_size != 0:
        raise ValueError(
            f"batch_size={batch_size} must divide rollout_size={rollout_size} exactly"
        )
    if int(target_timesteps) % rollout_size != 0:
        raise ValueError(
            f"target_timesteps={target_timesteps} must be divisible by rollout_size={rollout_size}"
        )
    if int(checkpoint_interval) % rollout_size != 0:
        raise ValueError(
            f"checkpoint_interval={checkpoint_interval} must be divisible by rollout_size={rollout_size}"
        )
    if int(target_timesteps) % int(checkpoint_interval) != 0:
        raise ValueError("Target timesteps must be divisible by checkpoint interval")
    return rollout_size


def expected_checkpoint_steps(target_timesteps: int, checkpoint_interval: int) -> list[int]:
    return pilot_common.expected_checkpoint_steps(target_timesteps, checkpoint_interval)


def evaluation_steps(
    target_timesteps: int,
    checkpoint_interval: int,
    *,
    evaluate_checkpoints: bool,
) -> list[int]:
    """Return the model timesteps that receive the full 10-seed evaluation."""

    if bool(evaluate_checkpoints):
        return expected_checkpoint_steps(target_timesteps, checkpoint_interval)
    return [int(target_timesteps)]


def checkpoint_evaluation_enabled(args: argparse.Namespace) -> bool:
    """Keep legacy callers opt-in while the nominal pilot defaults final-only."""

    return bool(getattr(args, "evaluate_checkpoints", True))


def sb3_resume_learn_target_timesteps(
    target_timesteps: int, current_timesteps: int
) -> tuple[int, int]:
    return pilot_common.sb3_resume_learn_target_timesteps(
        target_timesteps, current_timesteps
    )


class PPOActionClipCallback(BaseCallback):
    """Track the Gaussian actor's raw actions before Box clipping."""

    def __init__(self, saturation_tolerance: float = 1e-3) -> None:
        super().__init__(verbose=0)
        self.saturation_tolerance = float(saturation_tolerance)
        self.total_components = 0
        self.clipped_components = 0
        self.saturated_components = 0
        self.raw_abs_sum = 0.0
        self.raw_abs_max = 0.0
        self.window_total_components = 0
        self.window_clipped_components = 0
        self.window_saturated_components = 0
        self.window_raw_abs_sum = 0.0
        self.window_raw_abs_max = 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            key: copy.deepcopy(value)
            for key, value in self.__dict__.items()
            if key
            in {
                "saturation_tolerance",
                "total_components",
                "clipped_components",
                "saturated_components",
                "raw_abs_sum",
                "raw_abs_max",
                "window_total_components",
                "window_clipped_components",
                "window_saturated_components",
                "window_raw_abs_sum",
                "window_raw_abs_max",
            }
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key, value in state.items():
            if hasattr(self, key):
                setattr(self, key, copy.deepcopy(value))

    def _on_step(self) -> bool:
        raw = np.asarray(self.locals.get("actions", []), dtype=float)
        clipped = np.asarray(self.locals.get("clipped_actions", raw), dtype=float)
        if raw.size == 0:
            return True
        action_dim = int(np.asarray(self.model.action_space.low, dtype=float).size)
        if action_dim <= 0 or raw.size % action_dim != 0:
            raise RuntimeError("PPO raw action shape disagrees with the action space")
        raw = raw.reshape(-1, action_dim)
        clipped = clipped.reshape(-1, action_dim)
        if raw.shape != clipped.shape:
            raise RuntimeError("PPO raw and clipped action shapes disagree")
        low = np.broadcast_to(np.asarray(self.model.action_space.low, dtype=float), raw.shape)
        high = np.broadcast_to(np.asarray(self.model.action_space.high, dtype=float), raw.shape)
        tolerance = self.saturation_tolerance * np.maximum(high - low, 1.0)
        clipped_mask = np.abs(raw - clipped) > 1e-7
        saturated_mask = (
            clipped_mask
            | (np.abs(raw - low) <= tolerance)
            | (np.abs(raw - high) <= tolerance)
        )
        count = int(raw.size)
        clipped_count = int(np.count_nonzero(clipped_mask))
        saturated_count = int(np.count_nonzero(saturated_mask))
        raw_abs_sum = float(np.sum(np.abs(raw)))
        raw_abs_max = float(np.max(np.abs(raw)))
        self.total_components += count
        self.clipped_components += clipped_count
        self.saturated_components += saturated_count
        self.raw_abs_sum += raw_abs_sum
        self.raw_abs_max = max(self.raw_abs_max, raw_abs_max)
        self.window_total_components += count
        self.window_clipped_components += clipped_count
        self.window_saturated_components += saturated_count
        self.window_raw_abs_sum += raw_abs_sum
        self.window_raw_abs_max = max(self.window_raw_abs_max, raw_abs_max)
        self.logger.record(
            "rollout/actor_raw_action_clip_rate_cumulative",
            self.clipped_components / max(self.total_components, 1),
        )
        return True

    def consume_checkpoint_metrics(self) -> dict[str, float]:
        denominator = max(self.window_total_components, 1)
        result = {
            "actor_raw_action_components": float(self.window_total_components),
            "actor_raw_action_clipped_components": float(
                self.window_clipped_components
            ),
            "actor_raw_action_saturated_components": float(
                self.window_saturated_components
            ),
            "actor_raw_action_abs_sum": float(self.window_raw_abs_sum),
            "actor_raw_action_clip_rate": float(
                self.window_clipped_components / denominator
            ),
            "actor_raw_action_saturation_rate": float(
                self.window_saturated_components / denominator
            ),
            "actor_raw_action_abs_mean": float(self.window_raw_abs_sum / denominator),
            "actor_raw_action_abs_max": float(self.window_raw_abs_max),
            "actor_raw_action_clip_rate_cumulative": float(
                self.clipped_components / max(self.total_components, 1)
            ),
        }
        self.window_total_components = 0
        self.window_clipped_components = 0
        self.window_saturated_components = 0
        self.window_raw_abs_sum = 0.0
        self.window_raw_abs_max = 0.0
        return result


class PPORolloutDiagnosticsCache(BaseCallback):
    """Copy rollout targets before SB3 resets/flattens its on-policy buffer."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.snapshot: Optional[dict[str, np.ndarray]] = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buffer = self.model.rollout_buffer
        if not isinstance(buffer.observations, np.ndarray):
            raise RuntimeError("PPO pilot diagnostics require a flat Box observation space")
        self.snapshot = {
            "observations": np.asarray(buffer.observations).copy(),
            "returns": np.asarray(buffer.returns, dtype=float).copy(),
            "old_values": np.asarray(buffer.values, dtype=float).copy(),
        }

    def consume(self) -> dict[str, np.ndarray]:
        if self.snapshot is None:
            raise RuntimeError("No completed PPO rollout is available for post-update diagnostics")
        snapshot, self.snapshot = self.snapshot, None
        return snapshot


def ppo_value_diagnostics(
    model: PPO, rollout_snapshot: dict[str, np.ndarray]
) -> dict[str, float]:
    """Measure post-update value fit against a pre-reset rollout snapshot."""

    observations = np.asarray(rollout_snapshot["observations"])
    observation_shape = tuple(int(value) for value in model.observation_space.shape)
    flat_observations = observations.reshape((-1, *observation_shape))
    returns = np.asarray(rollout_snapshot["returns"], dtype=float).reshape(-1)
    old_values = np.asarray(rollout_snapshot["old_values"], dtype=float).reshape(-1)
    was_training = bool(model.policy.training)
    try:
        model.policy.set_training_mode(False)
        obs_tensor, _ = model.policy.obs_to_tensor(flat_observations)
        with th.no_grad():
            current_values = (
                model.policy.predict_values(obs_tensor).detach().cpu().numpy().reshape(-1)
            )
    finally:
        model.policy.set_training_mode(was_training)
    if current_values.shape != returns.shape:
        raise RuntimeError(
            f"PPO value/return shape mismatch: {current_values.shape} vs {returns.shape}"
        )
    error = current_values - returns
    finite = np.isfinite(current_values) & np.isfinite(returns)
    finite_error = error[finite]
    value_abs = np.abs(current_values[finite])
    return_abs = np.abs(returns[finite])
    old_error = old_values - returns
    return {
        "rollout_value_samples": float(returns.size),
        "rollout_value_nonfinite_rate": float(1.0 - np.mean(finite)),
        "rollout_value_target_mse": (
            float(np.mean(np.square(finite_error))) if finite_error.size else np.nan
        ),
        "rollout_value_target_abs_error_mean": (
            float(np.mean(np.abs(finite_error))) if finite_error.size else np.nan
        ),
        "rollout_value_target_abs_error_p95": _finite_p95(np.abs(finite_error)),
        "rollout_value_abs_mean": (
            float(np.mean(value_abs)) if value_abs.size else np.nan
        ),
        "rollout_value_abs_p95": _finite_p95(value_abs),
        "rollout_value_abs_max": (
            float(np.max(value_abs)) if value_abs.size else np.nan
        ),
        "rollout_return_abs_mean": (
            float(np.mean(return_abs)) if return_abs.size else np.nan
        ),
        "rollout_return_abs_p95": _finite_p95(return_abs),
        "rollout_preupdate_value_target_mse": (
            float(np.mean(np.square(old_error[np.isfinite(old_error)])))
            if np.isfinite(old_error).any()
            else np.nan
        ),
    }


PPO_LOGGER_METRICS = {
    "latest_train_policy_gradient_loss": "train/policy_gradient_loss",
    "latest_train_value_loss": "train/value_loss",
    "latest_train_entropy_loss": "train/entropy_loss",
    "latest_train_approx_kl": "train/approx_kl",
    "latest_train_clip_fraction": "train/clip_fraction",
    "latest_train_explained_variance": "train/explained_variance",
    "latest_train_loss": "train/loss",
    "latest_train_std": "train/std",
    "latest_train_learning_rate": "train/learning_rate",
    "latest_train_clip_range": "train/clip_range",
}


def ppo_checkpoint_diagnostics(
    model: PPO, rollout_snapshot: dict[str, np.ndarray]
) -> dict[str, float]:
    result = ppo_value_diagnostics(model, rollout_snapshot)
    for output_name, logger_name in PPO_LOGGER_METRICS.items():
        result[output_name] = pipeline._as_float(
            model.logger.name_to_value.get(logger_name)
        )
    result.update(
        {
            "n_updates": float(model._n_updates),
            "completed_rollouts": float(
                int(model.num_timesteps) // (int(model.n_steps) * int(model.n_envs))
            ),
        }
    )
    return result


def aggregate_checkpoint_scenarios(rows: list[dict[str, Any]]) -> dict[str, float]:
    return pilot_common.aggregate_checkpoint_scenarios(rows)


class PPOEvaluationCallback(BaseCallback):
    """Evaluate fixed scenarios only after the requested post-update rollout."""

    def __init__(
        self,
        *,
        namespace: dict[str, Any],
        pilot_config: str,
        training_seed: int,
        config_values: dict[str, Any],
        args: argparse.Namespace,
        env_config: dict[str, Any],
        reward_config: dict[str, float],
        scenario_path: Path,
        diagnostics_path: Path,
        action_callback: PPOActionClipCallback,
        rollout_diagnostics_cache: PPORolloutDiagnosticsCache,
        interval: int,
    ) -> None:
        super().__init__(verbose=0)
        self.namespace = namespace
        self.pilot_config = str(pilot_config)
        self.training_seed = int(training_seed)
        self.config_values = copy.deepcopy(config_values)
        self.args = args
        self.env_config = copy.deepcopy(env_config)
        self.reward_config = copy.deepcopy(reward_config)
        self.scenario_path = Path(scenario_path)
        self.diagnostics_path = Path(diagnostics_path)
        self.action_callback = action_callback
        self.rollout_diagnostics_cache = rollout_diagnostics_cache
        self.interval = int(interval)
        self.pending = False
        self.next_eval_step = self.interval
        self.evaluated_steps: set[int] = set()
        if self.scenario_path.exists() and self.scenario_path.stat().st_size:
            existing = pd.read_csv(self.scenario_path, usecols=["model_timestep"])
            self.evaluated_steps = set(
                pd.to_numeric(existing["model_timestep"], errors="raise")
                .astype(int)
                .tolist()
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "pending": bool(self.pending),
            "next_eval_step": int(self.next_eval_step),
            "evaluated_steps": sorted(int(step) for step in self.evaluated_steps),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        saved_steps = {int(step) for step in state.get("evaluated_steps", [])}
        if self.evaluated_steps and saved_steps != self.evaluated_steps:
            raise RuntimeError(
                "Strict-resume evaluation state disagrees with truncated evaluation log"
            )
        self.evaluated_steps = saved_steps
        self.pending = bool(state.get("pending", False))
        self.next_eval_step = int(state.get("next_eval_step", self.interval))

    def _on_training_start(self) -> None:
        current = int(self.model.num_timesteps)
        if not self.evaluated_steps:
            self.next_eval_step = ((current // self.interval) + 1) * self.interval

    def _on_step(self) -> bool:
        if int(self.num_timesteps) >= int(self.next_eval_step):
            self.pending = True
        return True

    def _on_rollout_start(self) -> None:
        # SB3 calls this after PPO.train() for the preceding rollout.
        if self.pending:
            self._evaluate_current_checkpoint()

    def _on_training_end(self) -> None:
        step = int(self.model.num_timesteps)
        if step % self.interval == 0 and step not in self.evaluated_steps:
            self._evaluate_current_checkpoint()

    def _evaluate_current_checkpoint(self) -> None:
        step = int(self.model.num_timesteps)
        if step in self.evaluated_steps:
            self.pending = False
            self.next_eval_step = ((step // self.interval) + 1) * self.interval
            return
        if step <= 0 or step % self.interval != 0:
            raise RuntimeError(f"PPO evaluation is not on an aligned post-update boundary: {step}")

        rng_state = pipeline.capture_rng_state(self.model)
        policy_training = bool(self.model.policy.training)
        rows: list[dict[str, Any]] = []
        try:
            for scenario_seed in self.args.eval_seeds:
                row, _ = pipeline.evaluate_scenario(
                    self.namespace,
                    model=self.model,
                    variant=self.pilot_config,
                    mode="raw",
                    scenario_seed=int(scenario_seed),
                    training_seed=self.training_seed,
                    env_config=self.env_config,
                    reward_config=self.reward_config,
                    args=self.args,
                    critic_calibration_samples=None,
                )
                row.update(
                    {
                        "pilot_config": self.pilot_config,
                        "model_timestep": step,
                        **self.config_values,
                    }
                )
                rows.append(row)

            diagnostics = {
                **ppo_checkpoint_diagnostics(
                    self.model, self.rollout_diagnostics_cache.consume()
                ),
                **self.action_callback.consume_checkpoint_metrics(),
                "pilot_config": self.pilot_config,
                "variant": self.pilot_config,
                "training_seed": self.training_seed,
                "model_timestep": step,
                **self.config_values,
            }
            _append_frame(self.scenario_path, pd.DataFrame(rows))
            _append_frame(self.diagnostics_path, pd.DataFrame([diagnostics]))
            for name, value in aggregate_checkpoint_scenarios(rows).items():
                self.logger.record(f"eval/{name}", value)
            for name, value in diagnostics.items():
                if name.startswith(("latest_train_", "rollout_value_", "actor_raw_")):
                    scalar = pipeline._as_float(value)
                    if np.isfinite(scalar):
                        self.logger.record(f"diagnostics/{name}", scalar)
            self.logger.dump(step=step)
        finally:
            self.model.policy.set_training_mode(policy_training)
            pipeline.restore_rng_state(self.model, rng_state)

        self.evaluated_steps.add(step)
        self.pending = False
        self.next_eval_step = ((step // self.interval) + 1) * self.interval
        print(
            f"[ppo-pilot] evaluated {self.pilot_config} seed={self.training_seed} step={step:,}",
            flush=True,
        )


def _latest_checkpoint_bundle(run_dir: Path) -> Path:
    pointer = run_dir / "latest_checkpoint.json"
    data = json.loads(pointer.read_text(encoding="utf-8"))
    relative = Path(str(data["checkpoint"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe checkpoint pointer in {pointer}")
    bundle = (run_dir / relative).resolve()
    if run_dir.resolve() not in bundle.parents:
        raise RuntimeError(f"Checkpoint pointer escapes run directory: {pointer}")
    if not bundle.is_dir() or not bundle.name.isdigit():
        raise RuntimeError(f"Checkpoint pointer does not target a numeric bundle: {pointer}")
    pointer_step = int(data.get("timestep", -1))
    if pointer_step < 0 or int(bundle.name) != pointer_step:
        raise RuntimeError(
            f"Checkpoint pointer timestep disagrees with its target: {pointer}"
        )
    return bundle


def validate_ppo_checkpoint_bundle(
    bundle: Path,
    expected_config_hash: str,
    *,
    expected_model_class: str,
    expected_rollout_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = bundle / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Incomplete PPO strict checkpoint: {bundle}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != PPO_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported PPO checkpoint schema in {bundle}")
    if str(manifest.get("training_config_hash")) != str(expected_config_hash):
        raise RuntimeError(
            "Strict-resume configuration hash mismatch: "
            f"saved={manifest.get('training_config_hash')} current={expected_config_hash}"
        )
    if str(manifest.get("model_class")) != str(expected_model_class):
        raise RuntimeError("Strict-resume PPO model-class mismatch")
    if str(manifest.get("phase")) != "post_update_boundary":
        raise RuntimeError("PPO checkpoint was not saved at a post-update boundary")
    if int(manifest.get("rollout_size", -1)) != int(expected_rollout_size):
        raise RuntimeError("Strict-resume PPO rollout-size mismatch")
    checksums = manifest.get("checksums")
    required = {
        PPO_CHECKPOINT_PAYLOADS["model"],
        PPO_CHECKPOINT_PAYLOADS["base_environment"],
        PPO_CHECKPOINT_PAYLOADS["pipeline_state"],
    }
    allowed = required | {PPO_CHECKPOINT_PAYLOADS["vecnormalize"]}
    if not isinstance(checksums, dict) or not required.issubset(checksums):
        raise RuntimeError("PPO strict checkpoint has an incomplete checksum table")
    if not set(checksums).issubset(allowed):
        raise RuntimeError("PPO strict checkpoint contains unexpected payloads")
    for name, expected_checksum in checksums.items():
        path = bundle / name
        if not path.exists() or pipeline.file_sha256(path) != str(expected_checksum):
            raise RuntimeError(f"PPO checkpoint payload is missing or corrupt: {path}")
    with (bundle / PPO_CHECKPOINT_PAYLOADS["pipeline_state"]).open("rb") as handle:
        state = pickle.load(handle)
    step = int(manifest.get("timestep", -1))
    if int(state.get("schema_version", -1)) != PPO_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("Unsupported PPO checkpoint state schema")
    if int(state.get("timestep", -2)) != step:
        raise RuntimeError("PPO checkpoint timestep disagreement")
    if int(state.get("n_updates", -2)) != int(manifest.get("n_updates", -1)):
        raise RuntimeError("PPO checkpoint update-count disagreement")
    if str(state.get("training_config_hash")) != str(expected_config_hash):
        raise RuntimeError("PPO checkpoint state configuration hash mismatch")
    if str(state.get("phase")) != "post_update_boundary":
        raise RuntimeError("PPO checkpoint state has an invalid phase")
    if int(state.get("rollout_size", -1)) != int(expected_rollout_size):
        raise RuntimeError("PPO checkpoint state rollout-size mismatch")
    if int(state.get("n_steps", -1)) * int(state.get("n_envs", -1)) != int(
        expected_rollout_size
    ):
        raise RuntimeError("PPO checkpoint n_steps/n_envs do not match rollout size")
    if int(state.get("completed_rollouts", -1)) != step // int(expected_rollout_size):
        raise RuntimeError("PPO checkpoint completed-rollout count is inconsistent")
    if step <= 0 or step % int(expected_rollout_size) != 0:
        raise RuntimeError("PPO checkpoint timestep is not rollout-aligned")
    return manifest, state


class PPOStrictCheckpointCallback(BaseCallback):
    """Save model/environment/RNG state at post-update PPO boundaries."""

    def __init__(
        self,
        *,
        run_dir: Path,
        checkpoint_interval: int,
        rollout_size: int,
        training_config_hash: str,
        metrics_callback: pipeline.TrainingMetricsCallback,
        action_callback: PPOActionClipCallback,
        evaluation_callback: PPOEvaluationCallback,
        tracked_log_paths: list[Path],
        resume_state: Optional[dict[str, Any]],
        strict_retention: int,
    ) -> None:
        super().__init__(verbose=0)
        self.run_dir = Path(run_dir)
        self.checkpoint_interval = int(checkpoint_interval)
        self.rollout_size = int(rollout_size)
        self.training_config_hash = str(training_config_hash)
        self.metrics_callback = metrics_callback
        self.action_callback = action_callback
        self.evaluation_callback = evaluation_callback
        self.tracked_log_paths = [Path(path) for path in tracked_log_paths]
        self.resume_state = resume_state
        self.strict_retention = max(int(strict_retention), 1)
        self.next_checkpoint_step = self.checkpoint_interval
        self.pending = False
        self.last_saved_step = -1

    def _on_training_start(self) -> None:
        current = int(self.model.num_timesteps)
        self.next_checkpoint_step = ((current // self.checkpoint_interval) + 1) * self.checkpoint_interval
        if self.resume_state is not None:
            self.last_saved_step = current
            self.next_checkpoint_step = int(
                self.resume_state.get("next_checkpoint_step", self.next_checkpoint_step)
            )

    def _on_step(self) -> bool:
        if int(self.num_timesteps) >= int(self.next_checkpoint_step):
            self.pending = True
        return True

    def _on_rollout_start(self) -> None:
        if self.pending and int(self.model.num_timesteps) > self.last_saved_step:
            self._save_bundle()

    def _on_training_end(self) -> None:
        step = int(self.model.num_timesteps)
        if step > self.last_saved_step:
            self._save_bundle()

    def _save_bundle(self) -> None:
        step = int(self.model.num_timesteps)
        if step % self.rollout_size != 0 or step % self.checkpoint_interval != 0:
            raise RuntimeError(f"Refusing to save misaligned PPO checkpoint at {step}")

        root = self.run_dir / "ckpt"
        root.mkdir(parents=True, exist_ok=True)
        final_dir = root / f"{step:09d}"
        temp_dir = root / f".tmp_{step:09d}_{os.getpid()}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if final_dir.exists():
            raise RuntimeError(f"Refusing to overwrite PPO strict checkpoint: {final_dir}")
        temp_dir.mkdir(parents=True)

        base_blob, environment_state = pipeline.capture_environment_state(
            self.model.get_env()
        )
        log_offsets = {
            str(path.resolve().relative_to(self.run_dir.resolve())): (
                int(path.stat().st_size) if path.exists() else 0
            )
            for path in self.tracked_log_paths
        }
        state = {
            "schema_version": PPO_CHECKPOINT_SCHEMA_VERSION,
            "phase": "post_update_boundary",
            "timestep": step,
            "n_updates": int(self.model._n_updates),
            "completed_rollouts": int(step // self.rollout_size),
            "rollout_size": self.rollout_size,
            "n_steps": int(self.model.n_steps),
            "n_envs": int(self.model.n_envs),
            "batch_size": int(self.model.batch_size),
            "training_config_hash": self.training_config_hash,
            "environment_state": environment_state,
            "rng_state": pipeline.capture_rng_state(self.model),
            "metrics_callback_state": self.metrics_callback.state_dict(),
            "action_callback_state": self.action_callback.state_dict(),
            "evaluation_callback_state": self.evaluation_callback.state_dict(),
            "next_checkpoint_step": (
                (step // self.checkpoint_interval) + 1
            )
            * self.checkpoint_interval,
            "log_offsets": log_offsets,
        }
        self.model.save(str(temp_dir / PPO_CHECKPOINT_PAYLOADS["model"]))
        (temp_dir / PPO_CHECKPOINT_PAYLOADS["base_environment"]).write_bytes(base_blob)
        with (temp_dir / PPO_CHECKPOINT_PAYLOADS["pipeline_state"]).open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        payload_names = [
            PPO_CHECKPOINT_PAYLOADS["model"],
            PPO_CHECKPOINT_PAYLOADS["base_environment"],
            PPO_CHECKPOINT_PAYLOADS["pipeline_state"],
        ]
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None:
            vec_normalize.save(str(temp_dir / PPO_CHECKPOINT_PAYLOADS["vecnormalize"]))
            payload_names.append(PPO_CHECKPOINT_PAYLOADS["vecnormalize"])
        manifest = {
            "schema_version": PPO_CHECKPOINT_SCHEMA_VERSION,
            "phase": "post_update_boundary",
            "timestep": step,
            "n_updates": int(self.model._n_updates),
            "rollout_size": self.rollout_size,
            "training_config_hash": self.training_config_hash,
            "model_class": pipeline._qualified_name(self.model),
            "checksums": {
                name: pipeline.file_sha256(temp_dir / name) for name in payload_names
            },
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        temp_dir.replace(final_dir)

        snapshot_dir = self.run_dir / "model_checkpoints"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_dir / f"{step:09d}.zip"
        shutil.copy2(final_dir / PPO_CHECKPOINT_PAYLOADS["model"], snapshot)
        (snapshot_dir / f"{step:09d}.json").write_text(
            json.dumps(
                {
                    "schema_version": PPO_PILOT_SCHEMA_VERSION,
                    "phase": "post_update_boundary",
                    "timestep": step,
                    "model_sha256": pipeline.file_sha256(snapshot),
                    "training_config_hash": self.training_config_hash,
                    "strict_bundle_may_be_pruned": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        latest_tmp = self.run_dir / ".latest_checkpoint.json.tmp"
        latest = self.run_dir / "latest_checkpoint.json"
        latest_tmp.write_text(
            json.dumps(
                {
                    "checkpoint": str(final_dir.relative_to(self.run_dir)),
                    "timestep": step,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(latest_tmp, latest)
        bundles = sorted(
            path for path in root.iterdir() if path.is_dir() and path.name.isdigit()
        )
        for old in bundles[: max(len(bundles) - self.strict_retention, 0)]:
            shutil.rmtree(old)
        self.last_saved_step = step
        self.next_checkpoint_step = (
            (step // self.checkpoint_interval) + 1
        ) * self.checkpoint_interval
        self.pending = False
        print(f"[ppo-pilot] strict checkpoint step={step:,}: {final_dir}", flush=True)


class PPOModelSnapshotCallback(BaseCallback):
    """Post-update model snapshots for parallel SubprocVecEnv training.

    A SubprocVecEnv cannot be serialized into the single-environment strict
    resume bundle.  We still retain a verified model snapshot at every
    post-update checkpoint boundary; parallel pilots are intentionally
    fresh/non-resumable.
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        checkpoint_interval: int,
        rollout_size: int,
        training_config_hash: str,
    ) -> None:
        super().__init__(verbose=0)
        self.run_dir = Path(run_dir)
        self.checkpoint_interval = int(checkpoint_interval)
        self.rollout_size = int(rollout_size)
        self.training_config_hash = str(training_config_hash)
        self.next_checkpoint_step = self.checkpoint_interval
        self.pending = False
        self.last_saved_step = -1

    def _on_training_start(self) -> None:
        current = int(self.model.num_timesteps)
        self.next_checkpoint_step = (
            (current // self.checkpoint_interval) + 1
        ) * self.checkpoint_interval

    def _on_step(self) -> bool:
        if int(self.num_timesteps) >= int(self.next_checkpoint_step):
            self.pending = True
        return True

    def _on_rollout_start(self) -> None:
        if self.pending and int(self.model.num_timesteps) > self.last_saved_step:
            self._save_snapshot()

    def _on_training_end(self) -> None:
        if int(self.model.num_timesteps) > self.last_saved_step:
            self._save_snapshot()

    def _save_snapshot(self) -> None:
        step = int(self.model.num_timesteps)
        if step % self.rollout_size != 0 or step % self.checkpoint_interval != 0:
            raise RuntimeError(f"Refusing to save an unaligned PPO snapshot at {step}")
        snapshot_dir = self.run_dir / "model_checkpoints"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_dir / f"{step:09d}.zip"
        if snapshot.exists():
            raise RuntimeError(f"Refusing to overwrite PPO model snapshot: {snapshot}")
        self.model.save(str(snapshot))
        (snapshot_dir / f"{step:09d}.json").write_text(
            json.dumps(
                {
                    "schema_version": PPO_PILOT_SCHEMA_VERSION,
                    "phase": "post_update_boundary",
                    "timestep": step,
                    "model_sha256": pipeline.file_sha256(snapshot),
                    "training_config_hash": self.training_config_hash,
                    "resume_supported": False,
                    "reason": "parallel_subproc_environment_state_not_serialized",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.last_saved_step = step
        self.next_checkpoint_step = (
            (step // self.checkpoint_interval) + 1
        ) * self.checkpoint_interval
        self.pending = False
        print(f"[ppo-pilot] parallel model snapshot step={step:,}: {snapshot}", flush=True)


def build_ppo_model(
    *,
    train_env: Any,
    config_values: dict[str, Any],
    training_seed: int,
    device: str,
    tensorboard_log: Path,
) -> PPO:
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=float(config_values["learning_rate"]),
        n_steps=int(config_values["n_steps"]),
        batch_size=int(config_values["batch_size"]),
        n_epochs=int(config_values["n_epochs"]),
        gamma=float(config_values["gamma"]),
        gae_lambda=float(config_values["gae_lambda"]),
        clip_range=float(config_values["clip_range"]),
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=float(config_values["ent_coef"]),
        vf_coef=float(config_values["vf_coef"]),
        max_grad_norm=float(config_values["max_grad_norm"]),
        use_sde=False,
        policy_kwargs={
            "net_arch": {"pi": [256, 128], "vf": [256, 128]},
            "activation_fn": th.nn.Tanh,
            "ortho_init": True,
            "log_std_init": float(config_values["log_std_init"]),
        },
        tensorboard_log=str(tensorboard_log),
        verbose=0,
        seed=int(training_seed),
        device=device,
    )
    if str(device).lower().startswith("cuda") and model.device.type != "cuda":
        raise RuntimeError(
            f"PPO silently selected {model.device!s} although CUDA was requested"
        )
    return model


def validate_normalized_action_space(model_or_env: Any) -> None:
    action_space = model_or_env.action_space
    if tuple(action_space.shape) != (2,):
        raise RuntimeError(f"Expected two continuous actions, got {action_space}")
    if not np.allclose(action_space.low, -1.0) or not np.allclose(
        action_space.high, 1.0
    ):
        raise RuntimeError(
            f"PPO pilot requires normalized [-1,1]^2 actions, got {action_space}"
        )


def ppo_config_payload(
    *,
    project_root: Path,
    pilot_config: str,
    config_values: dict[str, Any],
    training_seed: int,
    target_timesteps: int,
    args: argparse.Namespace,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    rollout_size: int,
) -> dict[str, Any]:
    source_paths = {
        "ppo_pilot_runner": Path(__file__).resolve(),
        "shared_protocol_pipeline": project_root / "scripts" / "training" / "run_cbf_filter_ablation.py",
        "shared_nominal_pilot": project_root
        / "scripts"
        / "run_nominal_ddpg_parameter_pilot.py",
        "script_config": project_root / "scripts" / "common" / "laneless_script_config.py",
        "mtm_training_config": project_root
        / "scripts"
        / "train_safety_potential_variants.py",
        "notebook": project_root / "notebooks" / "lanelessKaralakou.ipynb",
        "environment": project_root / "laneless highway env" / "lane_free_env.py",
    }
    return {
        "schema_version": PPO_PILOT_SCHEMA_VERSION,
        "study": "nominal_ppo_50k_parameter_pilot",
        "pilot_config": pilot_config,
        "observation_variant": (
            "target_y_plus_previous_action"
            if bool(getattr(args, "observation_at1", False))
            else "target_y_only"
        ),
        "training_seed": int(training_seed),
        "target_timesteps": int(target_timesteps),
        "parameters": copy.deepcopy(config_values),
        "fixed_training": {
            "algorithm": "stable_baselines3.PPO",
            "filtered_training": False,
            "n_envs": int(args.n_envs),
            "vectorized_backend": (
                "SubprocVecEnv"
                if int(args.n_envs) > 1 and bool(getattr(args, "use_subproc", False))
                else "DummyVecEnv"
            ),
            "rollout_size": int(rollout_size),
            "policy_net_arch": {"pi": [256, 128], "vf": [256, 128]},
            "activation": "torch.nn.Tanh",
            "ortho_init": True,
            "clip_range_vf": None,
            "normalize_advantage": True,
            "use_sde": False,
            "terminate_on_collision": True,
            "episode_reset_reseed": False,
            "one_continuous_learn_call": True,
            "strict_resume_supported": bool(int(args.n_envs) == 1),
        },
        "evaluation": {
            "mode": "raw",
            "scenario_seeds": [int(seed) for seed in args.eval_seeds],
            "timestep_budget": int(args.eval_timesteps),
            "cadence": (
                "every_checkpoint"
                if checkpoint_evaluation_enabled(args)
                else "final_only"
            ),
            "model_timesteps": evaluation_steps(
                target_timesteps,
                int(args.checkpoint_interval),
                evaluate_checkpoints=checkpoint_evaluation_enabled(args),
            ),
            "checkpoint_phase": "post_update_boundary",
            "deterministic": True,
            "terminate_on_collision": True,
            "reset_immediately_after_collision": True,
            "distinct_collision_events": True,
            "primary_safety_metric": {
                "name": "distance_per_collision_m",
                "formula": "total_distance_m / distinct_ego_collision_events",
                "zero_collision_value": "infinity (right-censored)",
                "finite_companion": "distance_per_collision_exposure_bound_m",
            },
            "scenario_weighting": "aggregate within training seed before across-seed statistics",
        },
        "model_snapshots": {
            "checkpoint_interval": int(args.checkpoint_interval),
            "post_update_boundary": True,
        },
        "env_config": copy.deepcopy(env_config),
        "reward_config": copy.deepcopy(reward_config),
        "fixed_cbf_snapshot": copy.deepcopy(args.cbf_snapshot),
        "device": str(args.device),
        "strict_checkpoint_retention": int(args.strict_checkpoint_retention),
        "runtime_versions": pipeline._package_versions(),
        "source_hashes": {
            name: pipeline.file_sha256(path) for name, path in source_paths.items()
        },
    }


def _expected_model_class() -> str:
    return pipeline._class_qualified_name(PPO)


def preflight_runs(
    *,
    output_dir: Path,
    run_specs: list[tuple[int, str]],
    args: argparse.Namespace,
    project_root: Path,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    target_timesteps: int,
) -> None:
    if int(args.n_envs) > 1 and bool(args.resume):
        raise ValueError("Parallel PPO pilots do not support strict resume; start a fresh output directory")
    if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to mix a fresh PPO pilot with existing artifacts: {output_dir}")
    for training_seed, pilot_config in run_specs:
        config_values = effective_ppo_config(pilot_config, args)
        rollout_size = validate_rollout_alignment(
            config_values,
            n_envs=int(args.n_envs),
            target_timesteps=target_timesteps,
            checkpoint_interval=int(args.checkpoint_interval),
        )
        run_dir = _run_dir(output_dir, training_seed, pilot_config)
        if not args.resume:
            continue
        pointer = run_dir / "latest_checkpoint.json"
        if not pointer.exists():
            if run_dir.exists() and any(run_dir.iterdir()):
                raise RuntimeError(
                    f"Incomplete PPO run has no strict checkpoint; model_final.zip is never a resume source: {run_dir}"
                )
            continue
        payload = ppo_config_payload(
            project_root=project_root,
            pilot_config=pilot_config,
            config_values=config_values,
            training_seed=training_seed,
            target_timesteps=target_timesteps,
            args=args,
            env_config=env_config,
            reward_config=reward_config,
            rollout_size=rollout_size,
        )
        validate_ppo_checkpoint_bundle(
            _latest_checkpoint_bundle(run_dir),
            pipeline.canonical_config_hash(payload),
            expected_model_class=_expected_model_class(),
            expected_rollout_size=rollout_size,
        )


def train_one_run(
    namespace: dict[str, Any],
    *,
    pilot_config: str,
    training_seed: int,
    target_timesteps: int,
    args: argparse.Namespace,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    output_dir: Path,
) -> dict[str, Any]:
    config_values = effective_ppo_config(pilot_config, args)
    rollout_size = validate_rollout_alignment(
        config_values,
        n_envs=int(args.n_envs),
        target_timesteps=target_timesteps,
        checkpoint_interval=int(args.checkpoint_interval),
    )
    run_dir = _run_dir(output_dir, training_seed, pilot_config)
    run_dir.mkdir(parents=True, exist_ok=True)
    monitor_path = run_dir / "training.monitor.csv"
    training_metrics_path = run_dir / "training_episodes.csv"
    scenario_path = run_dir / "evaluation_scenarios.csv"
    diagnostics_path = run_dir / "checkpoint_diagnostics.csv"
    tracked_logs = [
        monitor_path,
        training_metrics_path,
        scenario_path,
        diagnostics_path,
    ]
    project_root = Path(namespace["PROJECT_ROOT"])
    payload = ppo_config_payload(
        project_root=project_root,
        pilot_config=pilot_config,
        config_values=config_values,
        training_seed=training_seed,
        target_timesteps=target_timesteps,
        args=args,
        env_config=env_config,
        reward_config=reward_config,
        rollout_size=rollout_size,
    )
    config_hash = pipeline.canonical_config_hash(payload)
    pointer = run_dir / "latest_checkpoint.json"
    resume_this_run = bool(args.resume and pointer.exists())
    resume_bundle: Optional[Path] = None
    resume_manifest: Optional[dict[str, Any]] = None
    resume_state: Optional[dict[str, Any]] = None
    if resume_this_run:
        resume_bundle = _latest_checkpoint_bundle(run_dir)
        resume_manifest, resume_state = validate_ppo_checkpoint_bundle(
            resume_bundle,
            config_hash,
            expected_model_class=_expected_model_class(),
            expected_rollout_size=rollout_size,
        )
        expected_logs = {str(path.relative_to(run_dir)): path for path in tracked_logs}
        saved_offsets = resume_state.get("log_offsets", {})
        if set(saved_offsets) != set(expected_logs):
            raise RuntimeError(
                f"Strict-resume PPO log set mismatch: saved={sorted(saved_offsets)} current={sorted(expected_logs)}"
            )
        for relative, size in saved_offsets.items():
            pipeline._truncate_to_checkpoint(expected_logs[relative], int(size))
    else:
        pipeline.seed_everything(training_seed)

    output_identity = hashlib.sha256(str(run_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    tensorboard_root = default_tensorboard_root()
    tensorboard_log = tensorboard_root / f"{config_hash[:12]}_{output_identity}"
    parent_step = 0 if resume_manifest is None else int(resume_manifest["timestep"])
    tensorboard_session = (
        f"fresh_{time.time_ns()}"
        if not resume_this_run
        else f"resume_{parent_step:09d}_{time.time_ns()}"
    )

    train_args = copy.copy(args)
    if int(args.n_envs) > 1:
        train_env = make_parallel_subproc_training_env(
            project_root=project_root,
            seed=training_seed,
            n_envs=int(args.n_envs),
            env_config=env_config,
            reward_config=reward_config,
            observation_at1=bool(args.observation_at1),
            monitor_path=monitor_path,
        )
    else:
        train_env = pipeline.make_training_env(
            namespace,
            filtered=False,
            seed=training_seed,
            env_config=env_config,
            reward_config=reward_config,
            args=train_args,
            monitor_path=monitor_path,
            append_monitor=resume_this_run,
        )
    if int(args.n_envs) > 1 and not isinstance(train_env, SubprocVecEnv):
        train_env.close()
        raise RuntimeError(
            "Parallel PPO requested but training did not create a SubprocVecEnv"
        )
    validate_normalized_action_space(train_env)

    if resume_bundle is not None:
        vec_path = resume_bundle / PPO_CHECKPOINT_PAYLOADS["vecnormalize"]
        if vec_path.exists():
            train_env = VecNormalize.load(str(vec_path), train_env)
        model = PPO.load(
            str(resume_bundle / PPO_CHECKPOINT_PAYLOADS["model"]),
            env=train_env,
            device=args.device,
            force_reset=False,
        )
        model.tensorboard_log = str(tensorboard_log)
        pipeline.restore_environment_state(
            train_env,
            (resume_bundle / PPO_CHECKPOINT_PAYLOADS["base_environment"]).read_bytes(),
            resume_state["environment_state"],
        )
        saved_step = int(resume_manifest["timestep"])
        if int(model.num_timesteps) != saved_step or int(resume_state["timestep"]) != saved_step:
            raise RuntimeError("PPO checkpoint timestep disagreement after load")
        if int(model._n_updates) != int(resume_state["n_updates"]):
            raise RuntimeError("PPO checkpoint update-count disagreement after load")
        restored_obs = pipeline._base_vec_env(train_env)._obs_from_buf()
        if not pipeline._observations_equal(model._last_obs, restored_obs):
            raise RuntimeError("Saved PPO observation does not match restored environment")
        restored_dones = np.asarray(pipeline._base_vec_env(train_env).buf_dones, dtype=bool)
        if not np.array_equal(
            np.asarray(model._last_episode_starts, dtype=bool), restored_dones
        ):
            raise RuntimeError(
                "Saved PPO episode-start flags do not match restored environment dones"
            )
        if (
            int(model.n_steps) != int(resume_state["n_steps"])
            or int(model.n_envs) != int(resume_state["n_envs"])
            or int(model.batch_size) != int(resume_state["batch_size"])
        ):
            raise RuntimeError("Restored PPO rollout configuration disagrees with checkpoint")
        pipeline.restore_rng_state(model, resume_state["rng_state"])
    else:
        model = build_ppo_model(
            train_env=train_env,
            config_values=config_values,
            training_seed=training_seed,
            device=args.device,
            tensorboard_log=tensorboard_log,
        )
    validate_normalized_action_space(model)

    metadata = {
        "schema_version": PPO_PILOT_SCHEMA_VERSION,
        "training_config_hash": config_hash,
        "training_config": payload,
        "pilot_config": pilot_config,
        "training_seed": int(training_seed),
        "target_timesteps": int(target_timesteps),
        "parameters": config_values,
        "resumed_from": None if resume_bundle is None else str(resume_bundle),
        "tensorboard_log": str(tensorboard_log),
        "tensorboard_session": tensorboard_session,
        "n_envs": int(args.n_envs),
        "vectorized_backend": type(train_env).__name__,
        "model_device": str(getattr(model, "device", args.device)),
        "strict_resume_supported": bool(int(args.n_envs) == 1),
        "evaluation_cadence": (
            "every_checkpoint"
            if checkpoint_evaluation_enabled(args)
            else "final_only"
        ),
        "evaluation_model_timesteps": evaluation_steps(
            target_timesteps,
            int(args.checkpoint_interval),
            evaluate_checkpoints=checkpoint_evaluation_enabled(args),
        ),
    }
    config_path = run_dir / "run_config.json"
    config_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    metrics_callback = pipeline.TrainingMetricsCallback(
        path=training_metrics_path,
        training_seed=training_seed,
        variant=pilot_config,
    )
    action_callback = PPOActionClipCallback()
    rollout_diagnostics_cache = PPORolloutDiagnosticsCache()
    evaluation_callback = PPOEvaluationCallback(
        namespace=namespace,
        pilot_config=pilot_config,
        training_seed=training_seed,
        config_values=config_values,
        args=args,
        env_config=env_config,
        reward_config=reward_config,
        scenario_path=scenario_path,
        diagnostics_path=diagnostics_path,
        action_callback=action_callback,
        rollout_diagnostics_cache=rollout_diagnostics_cache,
        interval=(
            int(args.checkpoint_interval)
            if checkpoint_evaluation_enabled(args)
            else int(target_timesteps)
        ),
    )
    if resume_state is not None:
        metrics_callback.load_state_dict(resume_state.get("metrics_callback_state", {}))
        action_callback.load_state_dict(resume_state.get("action_callback_state", {}))
        evaluation_callback.load_state_dict(
            resume_state.get("evaluation_callback_state", {})
        )
    if int(args.n_envs) == 1:
        checkpoint_callback: BaseCallback = PPOStrictCheckpointCallback(
            run_dir=run_dir,
            checkpoint_interval=int(args.checkpoint_interval),
            rollout_size=rollout_size,
            training_config_hash=config_hash,
            metrics_callback=metrics_callback,
            action_callback=action_callback,
            evaluation_callback=evaluation_callback,
            tracked_log_paths=tracked_logs,
            resume_state=resume_state,
            strict_retention=int(args.strict_checkpoint_retention),
        )
    else:
        checkpoint_callback = PPOModelSnapshotCallback(
            run_dir=run_dir,
            checkpoint_interval=int(args.checkpoint_interval),
            rollout_size=rollout_size,
            training_config_hash=config_hash,
        )
    callbacks = CallbackList(
        [
            metrics_callback,
            action_callback,
            rollout_diagnostics_cache,
            evaluation_callback,
            checkpoint_callback,
        ]
    )

    started = time.perf_counter()
    try:
        remaining, learn_timesteps = sb3_resume_learn_target_timesteps(
            target_timesteps, model.num_timesteps
        )
        if remaining > 0:
            # One call per process/run. PPO continuation uses only the remaining
            # budget because reset_num_timesteps=False adds the current counter.
            model.learn(
                total_timesteps=learn_timesteps,
                callback=callbacks,
                reset_num_timesteps=False,
                tb_log_name=tensorboard_session,
                # SB3 dumps the preceding update immediately before the next
                # PPO.train(), which otherwise creates two differently-aged
                # train scalars at each checkpoint step. The post-update
                # evaluation callback performs the coherent TensorBoard dump.
                log_interval=None,
                progress_bar=False,
            )
        if int(model.num_timesteps) != int(target_timesteps):
            raise RuntimeError(
                f"PPO ended at {model.num_timesteps}, expected exactly {target_timesteps}"
            )
    finally:
        train_env.close()

    final_path = run_dir / "model_final.zip"
    model.save(str(final_path))
    metadata.update(
        {
            "elapsed_sec_this_process": float(time.perf_counter() - started),
            "completed_timesteps": int(model.num_timesteps),
            "n_updates": int(model._n_updates),
            "model_final_is_resume_source": False,
        }
    )
    config_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {
        "pilot_config": pilot_config,
        "training_seed": int(training_seed),
        "model_path": str(final_path),
        "run_dir": str(run_dir),
        "training_config_hash": config_hash,
    }


def validate_run_evaluation_coverage(
    scenarios: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    pilot_config: str,
    training_seed: int,
    checkpoint_steps: list[int],
    eval_seeds: list[int],
    eval_timesteps: int,
) -> None:
    pilot_common.validate_run_evaluation_coverage(
        scenarios,
        diagnostics,
        pd.DataFrame(),
        pilot_config=pilot_config,
        training_seed=training_seed,
        checkpoint_steps=checkpoint_steps,
        eval_seeds=eval_seeds,
        eval_timesteps=eval_timesteps,
        calibration_enabled=False,
    )


def collect_run_outputs(
    output_dir: Path,
    run_specs: list[tuple[int, str]],
    *,
    target_timesteps: int,
    checkpoint_interval: int,
    eval_seeds: list[int],
    eval_timesteps: int,
    evaluation_model_timesteps: Optional[list[int]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    steps = (
        [int(step) for step in evaluation_model_timesteps]
        if evaluation_model_timesteps is not None
        else expected_checkpoint_steps(target_timesteps, checkpoint_interval)
    )
    if not steps or len(set(steps)) != len(steps):
        raise ValueError("evaluation_model_timesteps must be non-empty and unique")
    for training_seed, pilot_config in run_specs:
        run_dir = _run_dir(output_dir, training_seed, pilot_config)
        scenario_path = run_dir / "evaluation_scenarios.csv"
        diagnostics_path = run_dir / "checkpoint_diagnostics.csv"
        if not scenario_path.exists() or not diagnostics_path.exists():
            raise RuntimeError(f"PPO run is missing evaluation outputs: {run_dir}")
        scenarios = pd.read_csv(scenario_path)
        diagnostics = pd.read_csv(diagnostics_path)
        validate_run_evaluation_coverage(
            scenarios,
            diagnostics,
            pilot_config=pilot_config,
            training_seed=training_seed,
            checkpoint_steps=steps,
            eval_seeds=eval_seeds,
            eval_timesteps=eval_timesteps,
        )
        scenario_frames.append(scenarios)
        diagnostic_frames.append(diagnostics)
    all_scenarios = pd.concat(scenario_frames, ignore_index=True)
    all_diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    if "initial_state_hash" in all_scenarios:
        counts = all_scenarios.groupby("scenario_seed")["initial_state_hash"].nunique(
            dropna=False
        )
        if counts.ne(1).any():
            raise RuntimeError(
                "Fixed PPO evaluation seeds did not reproduce paired initial states"
            )
    all_scenarios.to_csv(output_dir / "evaluation_scenarios.csv", index=False)
    all_diagnostics.to_csv(output_dir / "checkpoint_diagnostics.csv", index=False)
    return all_scenarios, all_diagnostics


def build_checkpoint_seed_summary(
    scenarios: pd.DataFrame, diagnostics: pd.DataFrame
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for timestep, group in scenarios.groupby("model_timestep", sort=True):
        seed_summary = pipeline.summarize_within_training_seed(
            group.drop(columns=["model_timestep", "pilot_config"], errors="ignore")
        )
        seed_summary.insert(0, "model_timestep", int(timestep))
        seed_summary["pilot_config"] = seed_summary["variant"]
        pieces.append(seed_summary)
    summary = pd.concat(pieces, ignore_index=True)
    parameter_names = set(next(iter(PPO_CONFIGS.values())).keys())
    diagnostic_columns = [
        column
        for column in diagnostics.columns
        if column not in {"variant", *parameter_names}
    ]
    diagnostic_merge = diagnostics[diagnostic_columns].copy()
    merged = summary.merge(
        diagnostic_merge,
        on=["pilot_config", "training_seed", "model_timestep"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(summary) or len(merged) != len(diagnostic_merge):
        raise RuntimeError("PPO checkpoint summary and diagnostics keys disagree")
    return merged


FINAL_WINDOW_SUM_METRICS = (
    "scenarios",
    "timesteps",
    "total_time_s",
    "total_return",
    "task_return",
    "correction_return",
    "collision_free_scenarios",
    "ego_collision_incidents",
    "ego_collision_active_timesteps",
    "distinct_all_pair_collision_events",
    "active_collision_pair_timesteps",
    "collision_transition_timesteps",
    "collision_transition_return",
    "post_collision_timesteps",
    "post_collision_return",
    "reset_calls_total",
    "resets_after_collision",
    "resets_after_truncation_only",
    "resets_after_other_terminal",
    "episode_segments",
    "completed_segments",
    "right_censored_segments",
    "episode_length_sum",
    "collision_survived_without_reset",
    "active_collision_without_event",
    "event_without_active_collision",
    "actor_raw_action_components",
    "actor_raw_action_clipped_components",
    "actor_raw_action_saturated_components",
    "actor_raw_action_abs_sum",
)


def final_three_seed_averages(checkpoint_summary: pd.DataFrame) -> pd.DataFrame:
    parameter_names = set(next(iter(PPO_CONFIGS.values())).keys())
    derived_or_special = {
        "model_timestep",
        "pilot_config",
        "variant",
        "training_seed",
        "total_distance_m",
        "distinct_ego_collision_events",
        "distance_per_collision_m",
        "distance_per_collision_right_censored",
        "distance_per_collision_exposure_bound_m",
        "ego_collisions_per_km",
        "return_per_timestep",
        "episode_length_mean",
        "actor_raw_action_clip_rate",
        "actor_raw_action_saturation_rate",
        "actor_raw_action_abs_mean",
        "actor_raw_action_abs_max",
        "actor_raw_action_clip_rate_cumulative",
        "n_updates",
        "completed_rollouts",
        *FINAL_WINDOW_SUM_METRICS,
        *parameter_names,
    }
    rows: list[dict[str, Any]] = []
    step_sets: set[tuple[int, ...]] = set()
    for (pilot_config, training_seed), group in checkpoint_summary.groupby(
        ["pilot_config", "training_seed"], sort=True
    ):
        steps = sorted(pd.to_numeric(group["model_timestep"], errors="raise").astype(int).unique())
        step_sets.add(tuple(steps))
        if not steps:
            raise RuntimeError(
                f"{pilot_config} seed={training_seed} has no evaluated PPO checkpoint"
            )
        # Legacy studies use the final three evaluated checkpoints.  The
        # default nominal pilot evaluates only the final checkpoint, which
        # intentionally produces a one-checkpoint trailing window.
        selected = steps[-min(3, len(steps)):]
        final = group[group["model_timestep"].isin(selected)]
        totals = {
            metric: float(pd.to_numeric(final[metric], errors="coerce").fillna(0).sum())
            for metric in FINAL_WINDOW_SUM_METRICS
            if metric in final
        }
        distance = float(pd.to_numeric(final["total_distance_m"], errors="coerce").sum())
        collisions = float(
            pd.to_numeric(final["distinct_ego_collision_events"], errors="coerce").sum()
        )
        row: dict[str, Any] = {
            "pilot_config": str(pilot_config),
            "training_seed": int(training_seed),
            "checkpoint_count": len(selected),
            "checkpoint_steps": ",".join(str(step) for step in selected),
            **totals,
            "total_distance_m": distance,
            "distinct_ego_collision_events": collisions,
            "distance_per_collision_m": pipeline._distance_per_collision(
                distance, collisions
            ),
            "distance_per_collision_right_censored": int(collisions == 0),
            "distance_per_collision_exposure_bound_m": pipeline._distance_per_collision_exposure_bound(
                distance, collisions
            ),
            "ego_collisions_per_km": pipeline._collisions_per_km(
                collisions, distance
            ),
            "return_per_timestep": pipeline._ratio(
                totals["total_return"], totals["timesteps"]
            ),
            "episode_length_mean": pipeline._ratio(
                totals["episode_length_sum"], totals["episode_segments"]
            ),
            "actor_raw_action_clip_rate": pipeline._ratio(
                totals.get("actor_raw_action_clipped_components", 0.0),
                totals.get("actor_raw_action_components", 0.0),
            ),
            "actor_raw_action_saturation_rate": pipeline._ratio(
                totals.get("actor_raw_action_saturated_components", 0.0),
                totals.get("actor_raw_action_components", 0.0),
            ),
            "actor_raw_action_abs_mean": pipeline._ratio(
                totals.get("actor_raw_action_abs_sum", 0.0),
                totals.get("actor_raw_action_components", 0.0),
            ),
            "actor_raw_action_abs_max": float(
                pd.to_numeric(final["actor_raw_action_abs_max"], errors="coerce").max()
            ),
        }
        latest = final.sort_values("model_timestep").iloc[-1]
        for latest_metric in (
            "n_updates",
            "completed_rollouts",
            "actor_raw_action_clip_rate_cumulative",
        ):
            if latest_metric in final:
                row[latest_metric] = pipeline._as_float(latest[latest_metric])
        for column in final.columns:
            if column in derived_or_special:
                continue
            numeric = pd.to_numeric(final[column], errors="coerce")
            if numeric.notna().any():
                row[column] = float(numeric.mean())
        rows.append(row)
    if len(step_sets) != 1:
        raise RuntimeError(f"PPO checkpoint grids are not paired: {sorted(step_sets)}")
    return pd.DataFrame(rows).sort_values(
        ["pilot_config", "training_seed"]
    ).reset_index(drop=True)


def across_seed_final_three(seed_averages: pd.DataFrame) -> pd.DataFrame:
    return pilot_common.across_seed_final_three(seed_averages)


def rank_final_three(across_seed: pd.DataFrame) -> pd.DataFrame:
    ranked = across_seed.copy()
    criteria = {
        "distance_per_collision_exposure_bound_m_seed_mean": False,
        "return_per_timestep_seed_mean": False,
        "mean_abs_speed_error_seed_mean": True,
        "episode_length_mean_seed_mean": False,
        "nominal_action_saturation_rate_seed_mean": True,
    }
    rank_columns: list[str] = []
    for metric, ascending in criteria.items():
        if metric not in ranked:
            raise RuntimeError(f"PPO ranking metric is missing: {metric}")
        rank_name = f"rank_{metric.removesuffix('_seed_mean')}"
        ranked[rank_name] = pd.to_numeric(ranked[metric], errors="coerce").rank(
            method="min", ascending=ascending, na_option="bottom"
        )
        rank_columns.append(rank_name)
    ranked["rollout_mean_rank"] = ranked[rank_columns].mean(axis=1)
    nonfinite = pd.to_numeric(
        ranked.get("rollout_value_nonfinite_rate_seed_mean", 0.0), errors="coerce"
    )
    ranked["diagnostic_nonfinite"] = nonfinite.fillna(1.0).gt(0).astype(int)
    ranked = ranked.sort_values(
        [
            "diagnostic_nonfinite",
            "rollout_mean_rank",
            "rank_distance_per_collision_exposure_bound_m",
        ]
    ).reset_index(drop=True)
    ranked.insert(0, "overall_rank", np.arange(1, len(ranked) + 1))
    return ranked


def write_summaries(
    *, output_dir: Path, scenarios: pd.DataFrame, diagnostics: pd.DataFrame
) -> pd.DataFrame:
    checkpoint = build_checkpoint_seed_summary(scenarios, diagnostics)
    checkpoint.to_csv(output_dir / "checkpoint_seed_summary.csv", index=False)
    seed_averages = final_three_seed_averages(checkpoint)
    across_seed = across_seed_final_three(seed_averages)
    ranking = rank_final_three(across_seed)
    final_only = bool(
        pd.to_numeric(seed_averages["checkpoint_count"], errors="raise").eq(1).all()
    )
    if final_only:
        seed_output = output_dir / "final_evaluation_seed_averages.csv"
        across_output = output_dir / "final_evaluation_across_seeds.csv"
        ranking_output = output_dir / "ranking_final_evaluation.csv"
        selection_rule = (
            "Equal-weight rollout rank over the final post-update checkpoint only. "
            "Scenarios are aggregated within each training seed before across-seed statistics."
        )
    else:
        seed_output = output_dir / "final_three_seed_averages.csv"
        across_output = output_dir / "final_three_across_seeds.csv"
        ranking_output = output_dir / "ranking_final_three.csv"
        selection_rule = (
            "Equal-weight rollout rank over the final three post-update checkpoints. "
            "Scenarios are aggregated within each training seed before across-seed statistics."
        )
    seed_averages.to_csv(seed_output, index=False)
    across_seed.to_csv(across_output, index=False)
    ranking.to_csv(ranking_output, index=False)
    (output_dir / "selected_best_two.json").write_text(
        json.dumps(
            {
                "selection_rule": selection_rule,
                "single_seed_screen_warning": (
                    "This 50k screen has one training seed; across-seed variance is undefined. "
                    "Confirm finalists with independent training seeds."
                ),
                "distance_per_collision_note": (
                    "Collision-free distance/collision is right-censored at infinity; the finite "
                    "driven-distance exposure lower bound is used for ordering."
                ),
                "ppo_diagnostics_note": (
                    "Value-fit, KL, entropy, clipping, and policy-standard-deviation diagnostics "
                    "are reported but do not outweigh fixed-seed rollout performance."
                ),
                "selected_configs": ranking["pilot_config"].head(2).tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the exact 50k nominal-PPO Q0--Q3 lane-free parameter pilot."
    )
    parser.add_argument("--stage", choices=("screen", "summarize"), default="screen")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument(
        "--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_TRAINING_SEED])
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--eval-seed-start", type=int, default=DEFAULT_EVAL_SEED_START)
    parser.add_argument("--eval-scenarios", type=int, default=DEFAULT_EVAL_SCENARIOS)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--eval-timesteps", type=int, default=DEFAULT_EVAL_TIMESTEPS)
    parser.add_argument(
        "--evaluate-checkpoints",
        action="store_true",
        help=(
            "run the full fixed-seed evaluation at every model snapshot; disabled by "
            "default because it serializes a large simulator workload into training"
        ),
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="number of training environments; use --use-subproc for true parallel workers",
    )
    parser.add_argument(
        "--use-subproc",
        action="store_true",
        help="require SubprocVecEnv when --n-envs is greater than one",
    )
    parser.add_argument(
        "--global-rollout-size",
        type=int,
        default=DEFAULT_GLOBAL_ROLLOUT_SIZE,
        help="global transitions collected per PPO update (must divide checkpoint interval)",
    )
    parser.add_argument(
        "--observation-at1",
        action="store_true",
        help="append the previous normalized executed action a[t-1] to the y-target observation",
    )
    parser.add_argument(
        "--progress-reward-weight",
        type=float,
        default=None,
        help=(
            "override the forward-progress reward weight; normalized progress is "
            "clipped by the task's fixed progress_clip setting"
        ),
    )
    parser.add_argument(
        "--jerk-penalty-weight",
        type=float,
        default=None,
        help="override the bounded applied-physical-jerk reward penalty weight",
    )
    parser.add_argument(
        "--jerk-scale",
        type=float,
        default=None,
        help="reference applied jerk magnitude in m/s^3 for the bounded penalty",
    )
    parser.add_argument("--strict-checkpoint-retention", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--eval-horizon", type=int, default=DEFAULT_EVAL_TIMESTEPS, help=argparse.SUPPRESS
    )
    parser.add_argument("--k0", type=float, default=5.29, help=argparse.SUPPRESS)
    parser.add_argument("--k1", type=float, default=3.68, help=argparse.SUPPRESS)
    parser.add_argument("--eps-side", type=float, default=0.10, help=argparse.SUPPRESS)
    parser.add_argument(
        "--correction-epsilon", type=float, default=0.03, help=argparse.SUPPRESS
    )
    add_env_config_args(parser)
    parser.set_defaults(traffic_model="mtm", env_config_json=None, env_config_file=None)
    return parser.parse_args()


def _main_resolved(
    args: argparse.Namespace, project_root: Path, output_dir: Path
) -> int:
    if args.stage == "summarize":
        scenarios = pd.read_csv(output_dir / "evaluation_scenarios.csv")
        diagnostics = pd.read_csv(output_dir / "checkpoint_diagnostics.csv")
        ranking = write_summaries(
            output_dir=output_dir, scenarios=scenarios, diagnostics=diagnostics
        )
        print(ranking.to_string(index=False), flush=True)
        return 0

    target_timesteps = int(args.timesteps)
    checkpoint_interval = int(args.checkpoint_interval)
    if target_timesteps <= 0 or checkpoint_interval <= 0:
        raise ValueError("PPO timestep budgets must be positive")
    args.device = validate_training_device(args.device)
    if int(args.n_envs) <= 0:
        raise ValueError("--n-envs must be positive")
    if int(args.n_envs) > 1 and not bool(args.use_subproc):
        raise ValueError(
            "Parallel PPO requires --use-subproc; DummyVecEnv would not parallelize simulation"
        )
    if int(args.n_envs) > 1 and bool(args.resume):
        raise ValueError(
            "Parallel PPO does not support strict resume; use a fresh output directory"
        )
    selected_configs = list(args.configs or PPO_CONFIGS.keys())
    unknown = [name for name in selected_configs if name not in PPO_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown PPO pilot configurations: {unknown}")
    training_seeds = [int(seed) for seed in args.seeds]
    if len(training_seeds) != 1 or len(set(training_seeds)) != 1:
        raise ValueError("The quick PPO screening pilot requires one common training seed")
    if args.eval_seeds is None:
        args.eval_seeds = [
            int(args.eval_seed_start) + index for index in range(int(args.eval_scenarios))
        ]
    else:
        args.eval_seeds = [int(seed) for seed in args.eval_seeds]
    if not args.eval_seeds or len(set(args.eval_seeds)) != len(args.eval_seeds):
        raise ValueError("PPO evaluation seeds must be non-empty and unique")
    if int(args.eval_timesteps) <= 0:
        raise ValueError("PPO evaluation timestep budget must be positive")
    args.eval_scenarios = len(args.eval_seeds)
    args.eval_episodes = len(args.eval_seeds)
    args.eval_horizon = int(args.eval_timesteps)

    namespace = pipeline.bootstrap_notebook_namespace(project_root)
    pipeline.exec_required_notebook_cells(
        project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    namespace["DEVICE"] = args.device
    args.cbf_snapshot = pilot_common.fixed_cbf_snapshot(namespace)
    args.k0 = float(args.cbf_snapshot["k0"])
    args.k1 = float(args.cbf_snapshot["k1"])
    args.eps_side = float(args.cbf_snapshot["eps_side"])
    env_config = env_config_from_args(args, namespace["ENV_CONFIG"])
    if active_traffic_model(env_config) == "mtm":
        deep_update(env_config, copy.deepcopy(MTM_CONGESTED_UNCERTAIN_UPDATES))
    if not bool(env_config.get("terminate_on_collision", False)):
        raise RuntimeError("PPO pilot requires terminate_on_collision=True")
    reward_config = apply_reward_overrides(
        pipeline.make_base_reward_config(namespace), args
    )
    # Both pilots in this study expose the desired lateral target in the
    # existing second observation slot.  The optional at-1 variant appends the
    # previous normalized command for jerk-aware policy conditioning.
    reward_config["expose_target_y"] = True
    if bool(args.observation_at1):
        install_previous_action_observation(namespace)
    run_specs = [
        (training_seed, pilot_config)
        for training_seed in training_seeds
        for pilot_config in selected_configs
    ]
    for _, pilot_config in run_specs:
        validate_rollout_alignment(
            effective_ppo_config(pilot_config, args),
            n_envs=int(args.n_envs),
            target_timesteps=target_timesteps,
            checkpoint_interval=checkpoint_interval,
        )
    preflight_runs(
        output_dir=output_dir,
        run_specs=run_specs,
        args=args,
        project_root=project_root,
        env_config=env_config,
        reward_config=reward_config,
        target_timesteps=target_timesteps,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    root_config = {
        "schema_version": PPO_PILOT_SCHEMA_VERSION,
        "study": "nominal_ppo_50k_parameter_pilot",
        "stage": "screen",
        "selected_configs": selected_configs,
        "training_seeds": training_seeds,
        "target_timesteps": target_timesteps,
        "checkpoint_interval": checkpoint_interval,
        "evaluation_cadence": (
            "every_checkpoint" if bool(args.evaluate_checkpoints) else "final_only"
        ),
        "evaluation_model_timesteps": evaluation_steps(
            target_timesteps,
            checkpoint_interval,
            evaluate_checkpoints=bool(args.evaluate_checkpoints),
        ),
        "eval_seeds": args.eval_seeds,
        "eval_timesteps": int(args.eval_timesteps),
        "environment_and_cbf_tuning_changed": False,
        "filtered_training": False,
        "n_envs": int(args.n_envs),
        "vectorized_backend": (
            "SubprocVecEnv" if int(args.n_envs) > 1 else "DummyVecEnv"
        ),
        "global_rollout_size": int(args.global_rollout_size),
        "device_requested": str(args.device),
        "cuda_available": bool(th.cuda.is_available()),
        "cuda_device_name": (
            th.cuda.get_device_name(0) if th.cuda.is_available() else None
        ),
        "torch_num_threads": int(th.get_num_threads()),
        "strict_resume_supported": bool(int(args.n_envs) == 1),
        "fixed_cbf_snapshot": args.cbf_snapshot,
        "episode_reset_reseed": False,
        "post_update_checkpoints": True,
        "primary_safety_metric": {
            "name": "distance_per_collision_m",
            "formula": "total_distance_m / distinct_ego_collision_events",
            "zero_collision_value": "infinity (right-censored)",
        },
        "configurations": {
            name: effective_ppo_config(name, args) for name in selected_configs
        },
        "env_config": env_config,
        "reward_config": reward_config,
        "observation_variant": (
            "target_y_plus_previous_action"
            if bool(args.observation_at1)
            else "target_y_only"
        ),
        "observation_dimensions": 32 if bool(args.observation_at1) else 30,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(root_config, indent=2, default=str), encoding="utf-8"
    )
    print(
        "[ppo-pilot] starting"
        f" configs={selected_configs} seed={training_seeds[0]}"
        f" timesteps={target_timesteps:,} snapshots_every={checkpoint_interval:,}"
        f" evaluation={'every_snapshot' if bool(args.evaluate_checkpoints) else 'final_only'}"
        f" eval={len(args.eval_seeds)}x{args.eval_timesteps}"
        f" n_envs={int(args.n_envs)} backend={'subproc' if int(args.n_envs) > 1 else 'dummy'}"
        f" device={args.device}",
        flush=True,
    )
    model_rows: list[dict[str, Any]] = []
    for training_seed, pilot_config in run_specs:
        print(
            f"[ppo-pilot] train {pilot_config} seed={training_seed} parameters={effective_ppo_config(pilot_config, args)}",
            flush=True,
        )
        model_rows.append(
            train_one_run(
                namespace,
                pilot_config=pilot_config,
                training_seed=training_seed,
                target_timesteps=target_timesteps,
                args=args,
                env_config=env_config,
                reward_config=reward_config,
                output_dir=output_dir,
            )
        )
    pd.DataFrame(model_rows).to_csv(output_dir / "model_manifest.csv", index=False)
    scenarios, diagnostics = collect_run_outputs(
        output_dir,
        run_specs,
        target_timesteps=target_timesteps,
        checkpoint_interval=checkpoint_interval,
        eval_seeds=[int(seed) for seed in args.eval_seeds],
        eval_timesteps=int(args.eval_timesteps),
        evaluation_model_timesteps=evaluation_steps(
            target_timesteps,
            checkpoint_interval,
            evaluate_checkpoints=bool(args.evaluate_checkpoints),
        ),
    )
    ranking = write_summaries(
        output_dir=output_dir, scenarios=scenarios, diagnostics=diagnostics
    )
    report_columns = [
        "overall_rank",
        "pilot_config",
        "distance_per_collision_m_seed_mean",
        "ego_collisions_per_km_seed_mean",
        "return_per_timestep_seed_mean",
        "mean_abs_speed_error_seed_mean",
        "episode_length_mean_seed_mean",
        "nominal_action_saturation_rate_seed_mean",
        "actor_raw_action_clip_rate_seed_mean",
        "latest_train_value_loss_seed_mean",
        "latest_train_approx_kl_seed_mean",
    ]
    print(ranking[report_columns].to_string(index=False), flush=True)
    print(f"[ppo-pilot] complete: {output_dir}", flush=True)
    return 0


def main() -> int:
    pipeline.set_stable_native_defaults()
    configure_parallel_runtime()
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()
    project_root = pipeline.find_project_root(args.project_root or Path.cwd())
    default_output = project_root / "artifacts" / "nominal_ppo_parameter_pilot" / "screen"
    output_dir = (args.output_dir or default_output).resolve()
    with pilot_common.OutputDirectoryRunLock(output_dir):
        return _main_resolved(args, project_root, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
