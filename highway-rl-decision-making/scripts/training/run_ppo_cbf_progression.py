"""Canonical PPO progression: shield, reward feedback, and integrated CBF PPO.

The primary study is:

1. train nominal PPO once and deploy the same checkpoint raw and with CBF;
2. train non-differentiable hard-CBF policies with reward-only,
   reward-plus-detached-actor, and detached-actor-only feedback;
3. retain the historical shield-only control and filtered factorial ladder;
4. compare differentiable CBF reward-only, reward-plus-actor, and actor-only
   feedback while retaining a final hard projection for each sampled action;
5. train the ordinary PPO value critic from the resulting reward-shaped
   returns, without a separate auxiliary CBF safety-critic objective.

Every newly trained model is immediately checked over 200 complete episodes
with the external CBF OFF and another 200 paired episodes with it ON; both
10-KPI tables are printed inline and saved with the model.  The longer final
evaluation still runs after the full ladder.  MTM is the default traffic model.
The default invocation ensures that every requested variant has a checkpoint:
an exactly matching completed checkpoint is reused, while a missing one is
trained.  ``--skip-training`` is evaluation-only and accepts only verified
exact checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable

# Eight vectorized simulator workers already provide the intended CPU
# parallelism.  Set these before NumPy/PyTorch import so one BLAS/OpenMP pool
# per worker cannot oversubscribe the machine and freeze the desktop during a
# long CUDA-backed PPO run.
for _native_thread_key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ.setdefault(_native_thread_key, "1")

import gymnasium as gym
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

import scripts.training.run_cbf_filter_ablation as protocol
from scripts.evaluation.evaluate_laneless_karalakou import TEN_KPI_SPECS
from scripts.evaluation.evaluate_ppo_cbf_counterfactuals import run_counterfactual_analysis
from scripts.common.laneless_script_config import (
    active_traffic_model,
    add_env_config_args,
    env_config_from_args,
)
from scripts.common.ppo_cbf_env import CBFContextPhysicalActionWrapper
from scripts.common.ppo_observation_variants import install_previous_action_observation
from scripts.common.projected_ppo_cbf import (
    DetachedCBFActorCriticPolicy,
    DetachedCBFActorPPO,
    LatentActionPPO,
    ProjectedCBFActorCriticPolicy,
    ProjectedCBFPPO,
    context_ignoring_policy_kwargs,
)
from scripts.training.run_nominal_ppo_parameter_pilot import PPOActionClipCallback, PPO_CONFIGS
from scripts.training.train_safety_potential_variants import MTM_CONGESTED_UNCERTAIN_UPDATES


PROGRESSION_SCHEMA_VERSION = 8
PPO_TRAINING_IMPLEMENTATION_VERSION = 12
TRAINING_SIGNATURE_FILE = "training_signature.json"
TRAINING_PENDING_SIGNATURE_FILE = "training_signature.pending.json"
TRAINING_COMPLETION_FILE = "training_complete.json"
DEFAULT_SEED = 307
DEFAULT_TIMESTEPS = 50_000
DEFAULT_EVAL_SEED_START = 900_000
DEFAULT_EVAL_SCENARIOS = 10
DEFAULT_EVAL_TIMESTEPS = 800
DEFAULT_POST_TRAIN_EVAL_EPISODES = 200
DEFAULT_POST_TRAIN_EVAL_SEED_START = 1_100_000
DEFAULT_POST_TRAIN_EVAL_WORKERS = 20
DEFAULT_TASK_DISTANCE_M = 1_000.0
DEFAULT_TASK_MAX_POLICY_STEPS = 3_000
POST_TRAIN_EVAL_SUMMARY_BLOCKS = 10
DEFAULT_PPO_CONFIG = "Q0_current_aligned"
# The notebook and direct CLI runs use a fixed 20-worker rollout pool.
DEFAULT_NUM_ENVS = 20

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "ppo_nominal": {
        "label": "PPO without CBF",
        "level": 1,
        "execution_mode": "box",
        "reward_penalty": False,
        "projected_mean": False,
        "differentiable_actor_loss": False,
        "detached_actor_loss": False,
        "safety_critic": False,
    },
    "ppo_cbf_shield_only": {
        "label": "PPO trained with CBF execution (reward-off control)",
        "level": 2,
        "execution_mode": "cbf",
        "reward_penalty": False,
        "projected_mean": False,
        "differentiable_actor_loss": False,
        "detached_actor_loss": False,
        "safety_critic": False,
    },
    "ppo_cbf_reward": {
        "label": "PPO trained with CBF execution + intervention reward",
        "level": 2,
        "execution_mode": "cbf",
        "reward_penalty": True,
        "projected_mean": False,
        "differentiable_actor_loss": False,
        "detached_actor_loss": False,
        "safety_critic": False,
    },
    "ppo_cbf_nd_reward_actor": {
        "label": "PPO with non-differentiable CBF reward + detached actor loss",
        "level": 2,
        "execution_mode": "cbf",
        "reward_penalty": True,
        "projected_mean": False,
        "differentiable_actor_loss": False,
        "detached_actor_loss": True,
        "safety_critic": False,
    },
    "ppo_cbf_nd_actor_only": {
        "label": "PPO with non-differentiable detached CBF actor loss only",
        "level": 2,
        "execution_mode": "cbf",
        "reward_penalty": False,
        "projected_mean": False,
        "differentiable_actor_loss": False,
        "detached_actor_loss": True,
        "safety_critic": False,
    },
    "ppo_cbf_diff_reward_only": {
        "label": "PPO with differentiable CBF projection + reward only",
        "level": 3,
        "execution_mode": "cbf",
        "reward_penalty": True,
        "projected_mean": True,
        "differentiable_actor_loss": False,
        "detached_actor_loss": False,
        "safety_critic": False,
    },
    "ppo_cbf_projected_reward_off": {
        "label": "PPO projected CBF (reward-off architecture control)",
        "level": 3,
        "execution_mode": "cbf",
        "reward_penalty": False,
        "projected_mean": True,
        "differentiable_actor_loss": True,
        "detached_actor_loss": False,
        "safety_critic": False,
    },
    "ppo_cbf_projected": {
        "label": "PPO with differentiable CBF final optimization layer",
        "level": 3,
        "execution_mode": "cbf",
        "reward_penalty": True,
        "projected_mean": True,
        "differentiable_actor_loss": True,
        "detached_actor_loss": False,
        "safety_critic": False,
    },
    "ppo_cbf_integrated_actor_only": {
        "label": "PPO with differentiable CBF reward + actor loss",
        "level": 3,
        "execution_mode": "cbf",
        "reward_penalty": True,
        "projected_mean": True,
        "differentiable_actor_loss": True,
        "detached_actor_loss": False,
        "safety_critic": False,
    },
}
# Preserve the historical default ladder. The canonical 1M notebook passes
# its requested variants explicitly, so older calls do not silently add
# new multi-million-step jobs.
DEFAULT_VARIANTS = (
    "ppo_nominal",
    "ppo_cbf_shield_only",
    "ppo_cbf_reward",
    "ppo_cbf_projected_reward_off",
    "ppo_cbf_projected",
)
EVALUATION_MODES = ("raw", "cbf")
COLLISION_EVENT_ATTRIBUTION_SCHEMA_VERSION = 1


def _base_observation_features(env_config: dict[str, Any]) -> list[str]:
    features = ["dx", "dy", "vx", "vy"]
    if bool(env_config.get("observation_include_vehicle_dimensions", True)):
        features.extend(["vehicle_length", "vehicle_width"])
    features.append("desired_speed")
    return features


def _base_observation_dim(env_config: dict[str, Any]) -> int:
    rows = 1 + int(env_config.get("neighbors_count", 5))
    base = rows * len(_base_observation_features(env_config))
    return base + (2 if bool(env_config.get("ppo_append_previous_action", False)) else 0)


def _ensure_ppo_observation_variant(
    namespace: dict[str, Any], env_config: dict[str, Any]
) -> None:
    """Install the canonical 32D target-y + previous-action observation once."""

    if not bool(env_config.get("ppo_append_previous_action", False)):
        return
    if namespace.get("PPO_OBSERVATION_VARIANT") == "target_y_plus_previous_action":
        return
    install_previous_action_observation(namespace)


POST_TRAIN_EPISODE_MEAN_COLUMNS = {
    "episode_return",
    "episode_length_steps",
    "distance_completion_rate",
}
POST_TRAIN_POOLED_COLUMNS = {
    "ego_collisions_per_km",
    "h_min",
}
POST_TRAIN_KPI_SPECS: tuple[tuple[str, str], ...] = (
    *TEN_KPI_SPECS,
    ("Distance-based completion rate", "distance_completion_rate"),
)
POST_TRAIN_STEP_WEIGHTED_COLUMNS = tuple(
    column
    for _label, column in POST_TRAIN_KPI_SPECS
    if column not in POST_TRAIN_EPISODE_MEAN_COLUMNS | POST_TRAIN_POOLED_COLUMNS
)
FILTERED_FACTORIAL_VARIANTS = {
    (False, False): "ppo_cbf_shield_only",
    (True, False): "ppo_cbf_reward",
    (False, True): "ppo_cbf_projected_reward_off",
    (True, True): "ppo_cbf_projected",
}

# TensorBoard appends a generated event filename and Stable-Baselines3 adds a
# run-name directory below the supplied root.  Stay below the project's proven
# 248-character guard rather than relying on the system TEMP fallback.
WINDOWS_TENSORBOARD_PATH_LIMIT = 248
TENSORBOARD_VARIANT_IDS = {
    "ppo_nominal": "nom",
    "ppo_cbf_shield_only": "shld",
    "ppo_cbf_reward": "rwd",
    "ppo_cbf_nd_reward_actor": "ndra",
    "ppo_cbf_nd_actor_only": "ndao",
    "ppo_cbf_diff_reward_only": "dfro",
    "ppo_cbf_projected_reward_off": "pro0",
    "ppo_cbf_projected": "pro",
    "ppo_cbf_integrated_actor_only": "iao",
}


@dataclass(frozen=True)
class VariantTrainingResult:
    """Result of ensuring one exact PPO checkpoint."""

    model_path: Path
    trained: bool
    tensorboard_log_dir: Path | None


def _finite_mean(values: Iterable[float], default: float = np.nan) -> float:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float(default)


def _finite_min(values: Iterable[float], default: float = np.nan) -> float:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    return float(np.min(array)) if array.size else float(default)


def _deep_set_defaults(base: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Fill MTM defaults without silently overwriting an explicit study setup."""

    for key, value in defaults.items():
        if key not in base:
            base[key] = copy.deepcopy(value)
        elif isinstance(base[key], dict) and isinstance(value, dict):
            _deep_set_defaults(base[key], value)
    return base


def _evaluation_horizon_steps(env_config: dict[str, Any]) -> int:
    """Return the episode horizon in policy steps, not simulator frames."""

    configured_steps = float(
        env_config.get("episode_steps", env_config.get("duration", 2000))
    )
    simulation_frequency = float(env_config.get("simulation_frequency", 20.0))
    policy_frequency = float(env_config.get("policy_frequency", 10.0))
    frames_per_policy_step = max(
        1, int(round(simulation_frequency / max(policy_frequency, 1e-9)))
    )
    return max(1, int(np.ceil(configured_steps / frames_per_policy_step)))


def _full_horizon_survival_flag(
    *, episode_length_steps: float, collision_events: int, env_config: dict[str, Any]
) -> float:
    """Legacy time-based diagnostic retained in raw episode files."""

    return float(
        float(episode_length_steps) >= _evaluation_horizon_steps(env_config)
        and int(collision_events) == 0
    )


def _ensure_full_horizon_survival_metric(
    episode_metrics: pd.DataFrame, *, env_config: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Add the metric to legacy saved episode rows when it is not already present."""

    if "full_horizon_survival_rate" in episode_metrics.columns:
        return episode_metrics
    env_config = env_config or {}
    enriched = episode_metrics.copy()
    lengths = pd.to_numeric(enriched["episode_length_steps"], errors="coerce")
    collisions = pd.to_numeric(
        enriched["distinct_ego_collision_events"], errors="coerce"
    ).fillna(0.0)
    enriched["full_horizon_survival_rate"] = [
        _full_horizon_survival_flag(
            episode_length_steps=length,
            collision_events=int(collision),
            env_config=env_config,
        )
        if np.isfinite(length)
        else 0.0
        for length, collision in zip(lengths, collisions)
    ]
    return enriched


def _distance_completion_target_m(
    env_config: dict[str, Any], *, task_distance_m: float | None = None
) -> float:
    """Return the strict collision-free distance-task completion target."""

    target_distance_m = float(
        task_distance_m
        if task_distance_m is not None
        else env_config.get("evaluation_task_distance_m", DEFAULT_TASK_DISTANCE_M)
    )
    if not np.isfinite(target_distance_m) or target_distance_m <= 0.0:
        raise ValueError(
            "Distance-based completion requires a positive finite task distance."
        )
    return target_distance_m


def _distance_completion_flag(
    *,
    total_distance_m: float,
    collision_events: int,
    env_config: dict[str, Any],
    task_distance_m: float | None = None,
) -> float:
    """Mark a collision-free episode that reaches the strict task distance."""

    distance_m = float(total_distance_m)
    return float(
        np.isfinite(distance_m)
        and distance_m >= _distance_completion_target_m(
            env_config, task_distance_m=task_distance_m
        )
        and int(collision_events) == 0
    )


def _ensure_distance_completion_metric(
    episode_metrics: pd.DataFrame, *, env_config: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Add distance completion to legacy episode rows using simulator distance."""

    if "distance_completion_rate" in episode_metrics.columns:
        return episode_metrics
    env_config = env_config or {}
    enriched = episode_metrics.copy()
    distances = pd.to_numeric(enriched["total_distance_m"], errors="coerce")
    collisions = pd.to_numeric(
        enriched["distinct_ego_collision_events"], errors="coerce"
    ).fillna(0.0)
    enriched["distance_completion_rate"] = [
        _distance_completion_flag(
            total_distance_m=distance,
            collision_events=int(collision),
            env_config=env_config,
        )
        for distance, collision in zip(distances, collisions)
    ]
    return enriched


def _task_distance_from_args(args: argparse.Namespace, env_config: dict[str, Any]) -> float:
    """Resolve a positive explicit task distance, defaulting to 1 km."""

    return _distance_completion_target_m(
        env_config,
        task_distance_m=getattr(args, "task_distance_m", DEFAULT_TASK_DISTANCE_M),
    )


def _task_max_policy_steps_from_args(args: argparse.Namespace) -> int:
    """Resolve the policy-step timeout for a complete distance-task episode."""

    value = int(
        getattr(args, "task_max_policy_steps", DEFAULT_TASK_MAX_POLICY_STEPS)
    )
    if value <= 0:
        raise ValueError("task_max_policy_steps must be positive")
    return value


class DistanceTaskEvaluationWrapper(gym.Wrapper):
    """Cap an evaluation episode at an exact collision-free task distance.

    The simulator may wrap around its short physical road.  This wrapper keeps
    a separate path-distance task, terminates on the first collision, and
    reports a capped task distance so an evaluation record can never exceed
    the requested completion distance.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        task_distance_m: float,
        max_policy_steps: int,
    ) -> None:
        super().__init__(env)
        self.task_distance_m = _distance_completion_target_m(
            {}, task_distance_m=float(task_distance_m)
        )
        self.max_policy_steps = int(max_policy_steps)
        if self.max_policy_steps <= 0:
            raise ValueError("max_policy_steps must be positive")
        self._task_distance_traveled_m = 0.0
        self._task_steps = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._task_distance_traveled_m = 0.0
        self._task_steps = 0
        info = dict(info)
        info.update(
            {
                "task_distance_m": float(self.task_distance_m),
                "task_distance_traveled_m": 0.0,
                "task_distance_step_m": 0.0,
                "task_completed": False,
                "task_timeout": False,
            }
        )
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        raw_step_distance = float(info.get("pipeline_distance_step_m", 0.0))
        if not np.isfinite(raw_step_distance):
            raw_step_distance = 0.0
        remaining = max(self.task_distance_m - self._task_distance_traveled_m, 0.0)
        counted_step_distance = min(max(raw_step_distance, 0.0), remaining)
        self._task_distance_traveled_m = min(
            self.task_distance_m,
            self._task_distance_traveled_m + counted_step_distance,
        )
        self._task_steps += 1
        collision = bool(
            info.get("ego_collision", False)
            or int(info.get("ego_collision_events", 0)) > 0
        )
        completed = bool(
            not collision
            and self._task_distance_traveled_m >= self.task_distance_m - 1e-9
        )
        timeout = bool(
            not completed
            and not collision
            and (
                self._task_steps >= self.max_policy_steps
                or bool(truncated)
            )
        )
        # Treat a reported ego collision as terminal even if a caller supplied
        # a simulator config with collision termination disabled.  Completion
        # is collision-free by definition, so the collision has precedence on
        # a transition that reaches the distance cap as well.
        terminated = bool(terminated or collision or completed)
        truncated = bool(truncated or timeout)
        info.update(
            {
                "task_distance_m": float(self.task_distance_m),
                "task_distance_traveled_m": float(self._task_distance_traveled_m),
                "task_distance_step_m": float(counted_step_distance),
                "task_completed": completed,
                "task_timeout": timeout,
                "task_collision_terminated": bool(collision),
                "evaluation_distance_cap_m": float(self.task_distance_m),
            }
        )
        return observation, reward, terminated, truncated, info


def _predict_evaluation_action(
    model: LatentActionPPO,
    observation: np.ndarray,
    *,
    action_source: str = "policy",
) -> np.ndarray:
    """Return a deterministic action for the requested evaluation ablation.

    ``policy`` preserves the normal deployment path.  ``raw_actor_mean`` is
    an inference-only ablation for projected policies: it bypasses the
    architectural ``mu_raw -> mu_safe`` CBF projection and retains only the
    physical action-box bounds.
    """

    if action_source == "policy":
        action, _ = model.predict(observation, deterministic=True)
    elif action_source == "raw_actor_mean":
        if not hasattr(model, "predict_action_stages"):
            raise TypeError(
                "raw_actor_mean evaluation requires the projected-policy "
                "diagnostic API predict_action_stages()"
            )
        stages = model.predict_action_stages(observation, deterministic=True)
        if "mu_raw" not in stages:
            raise KeyError("predict_action_stages() did not return mu_raw")
        action = stages["mu_raw"]
        low = np.asarray(model.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(model.action_space.high, dtype=np.float32).reshape(-1)
        action = np.clip(action, low, high)
    else:
        raise ValueError(
            f"Unknown evaluation action source {action_source!r}; expected "
            "'policy' or 'raw_actor_mean'."
        )
    return np.asarray(action, dtype=np.float32).reshape(-1)[:2]


def _variant_dir(output_dir: Path, variant: str, training_seed: int) -> Path:
    return output_dir / variant / f"seed_{int(training_seed)}"


def _canonical_payload(payload: Any) -> Any:
    """Return a JSON-stable payload for checkpoint identity comparisons."""

    return json.loads(json.dumps(payload, sort_keys=True, default=str))


def training_topology(args: argparse.Namespace) -> dict[str, Any]:
    """Describe the rollout workers that determine PPO collection semantics."""

    n_envs = int(getattr(args, "n_envs", 1))
    if n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    if n_envs == 1:
        return {
            "n_envs": 1,
            "backend": "dummy",
            "start_method": None,
        }
    # ``spawn`` is safe on Windows and avoids unsafe fork interactions with
    # PyTorch and notebook-defined environment classes.
    return {
        "n_envs": n_envs,
        "backend": "subproc",
        "start_method": "spawn",
    }


def _effective_training_settings(
    variant: str, args: argparse.Namespace
) -> dict[str, float]:
    """Return only loss/reward settings that affect the selected variant."""

    spec = VARIANT_SPECS[variant]
    return {
        "lambda_delta": (
            float(args.lambda_delta) if bool(spec["reward_penalty"]) else 0.0
        ),
        "lambda_intervention": (
            float(args.lambda_intervention)
            if bool(spec["reward_penalty"])
            else 0.0
        ),
        "lambda_mean": (
            float(args.lambda_mean)
            if bool(spec.get("differentiable_actor_loss", False))
            else 0.0
        ),
        "lambda_detached_actor": (
            float(args.lambda_detached_actor)
            if bool(spec.get("detached_actor_loss", False))
            else 0.0
        ),
        "lambda_sample": (
            float(args.lambda_sample)
            if bool(spec.get("differentiable_actor_loss", False))
            else 0.0
        ),
        "lambda_critic": (
            float(getattr(args, "lambda_critic", 0.10))
            if bool(spec.get("safety_critic", False))
            else 0.0
        ),
        "safety_critic_gamma": (
            float(getattr(args, "safety_critic_gamma", 0.99))
            if bool(spec.get("safety_critic", False))
            else 0.0
        ),
        "safety_critic_cost_clip": (
            float(getattr(args, "safety_critic_cost_clip", 1.0))
            if bool(spec.get("safety_critic", False))
            else 0.0
        ),
        "action_rate_penalty": (
            float(getattr(args, "action_rate_penalty", 0.0))
            if str(variant) == "ppo_nominal"
            else 0.0
        ),
    }


def _cbf_training_snapshot(namespace: dict[str, Any]) -> dict[str, Any]:
    """Capture every notebook-level CBF setting used by the PPO wrappers."""

    keys = (
        "CBF_AX_BOUNDS",
        "CBF_AY_BOUNDS",
        "CBF_EPS_SIDE",
        "CBF_K0",
        "CBF_K1",
        "CBF_NEIGHBOR_RANGE",
        "CBF_MAX_NEIGHBOR_CONSTRAINTS",
        "CBF_QP_FEASIBILITY_TOL",
        "CBF_TARGET_PAIR_DY",
    )
    return {
        key: namespace[key]
        for key in keys
        if key in namespace
    }


def training_signature(
    namespace: dict[str, Any],
    *,
    variant: str,
    training_seed: int,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build the complete identity of a trainable PPO variant.

    A model is reusable only when this payload matches byte-for-byte after
    canonical JSON serialization.  This deliberately includes the effective
    reward/CBF settings and environment, rather than relying on a filename.
    """

    signature = {
        "schema_version": PROGRESSION_SCHEMA_VERSION,
        "training_implementation_version": PPO_TRAINING_IMPLEMENTATION_VERSION,
        "variant": str(variant),
        "variant_spec": VARIANT_SPECS[variant],
        "training_seed": int(training_seed),
        "timesteps": int(args.timesteps),
        "ppo_config_name": str(args.ppo_config),
        "tensorboard_run_label": str(
            getattr(args, "tensorboard_run_label", None) or ""
        ),
        "ppo_config": resolved_ppo_config(args),
        "collection_topology": training_topology(args),
        "model_contract": {
            "algorithm_class": (
                "ProjectedCBFPPO"
                if bool(VARIANT_SPECS[variant]["projected_mean"])
                else (
                    "DetachedCBFActorPPO"
                    if bool(
                        VARIANT_SPECS[variant].get(
                            "detached_actor_loss", False
                        )
                    )
                    else "LatentActionPPO"
                )
            ),
            "policy_class": (
                "ProjectedCBFActorCriticPolicy"
                if bool(VARIANT_SPECS[variant]["projected_mean"])
                else (
                    "DetachedCBFActorCriticPolicy"
                    if bool(
                        VARIANT_SPECS[variant].get(
                            "detached_actor_loss", False
                        )
                    )
                    else "MlpPolicy"
                )
            ),
            "net_arch": {"pi": [256, 128], "vf": [256, 128]},
            "activation": "torch.nn.Tanh",
            "base_observation_dim": _base_observation_dim(env_config),
            "base_observation_features": _base_observation_features(env_config),
            "observation_variant": (
                "target_y_plus_previous_executed_action"
                if bool(env_config.get("ppo_append_previous_action", False))
                else "base_vehicle_table"
            ),
            "previous_action_semantics": (
                "last_normalized_executed_physics_action"
                if bool(env_config.get("ppo_append_previous_action", False))
                else None
            ),
            "max_cbf_constraints": 18,
            "ordinary_value_critic": True,
            "ordinary_value_target": "reward_shaped_return",
            "safety_critic_head": bool(
                VARIANT_SPECS[variant].get("safety_critic", False)
            ),
            "safety_critic_loss_enabled": bool(
                VARIANT_SPECS[variant].get("safety_critic", False)
            ),
            "detached_actor_loss_enabled": bool(
                VARIANT_SPECS[variant].get("detached_actor_loss", False)
            ),
            "differentiable_actor_loss_enabled": bool(
                VARIANT_SPECS[variant].get(
                    "differentiable_actor_loss", False
                )
            ),
            "actor_cbf_gradient_path": (
                (
                    "differentiable_projection_plus_auxiliary_mean_loss"
                    if bool(
                        VARIANT_SPECS[variant].get(
                            "differentiable_actor_loss", False
                        )
                    )
                    else "differentiable_projection_only"
                )
                if bool(VARIANT_SPECS[variant]["projected_mean"])
                else (
                    "stop_gradient_hard_projection_target"
                    if bool(
                        VARIANT_SPECS[variant].get(
                            "detached_actor_loss", False
                        )
                    )
                    else "none"
                )
            ),
        },
        "action_contract": {
            "buffer_action": "latent Gaussian sample z",
            "environment_action": (
                "hard CBF projection P_s(z)"
                if VARIANT_SPECS[variant]["execution_mode"] == "cbf"
                else "physical action-box projection"
            ),
            "project_inputs_during_collection": False,
            "cbf_substep_filtering": bool(
                env_config.get("cbf_substep_filtering", False)
                and VARIANT_SPECS[variant]["execution_mode"] == "cbf"
            ),
            "physics_substeps_per_policy_action": max(
                1,
                int(
                    round(
                        float(env_config.get("simulation_frequency", 1.0))
                        / max(
                            float(env_config.get("policy_frequency", 1.0)),
                            1e-9,
                        )
                    )
                ),
            ),
        },
        "effective_training_settings": _effective_training_settings(
            variant, args
        ),
        "correction_epsilon": float(args.correction_epsilon),
        "cbf": _cbf_training_snapshot(namespace),
        "env_config": env_config,
        "reward_config": reward_config,
    }
    return _canonical_payload(signature)


def _signature_path(run_dir: Path) -> Path:
    return run_dir / TRAINING_SIGNATURE_FILE


def _pending_signature_path(run_dir: Path) -> Path:
    return run_dir / TRAINING_PENDING_SIGNATURE_FILE


def _completion_path(run_dir: Path) -> Path:
    return run_dir / TRAINING_COMPLETION_FILE


def _signature_mismatch_message(
    run_dir: Path,
    *,
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> str:
    expected_hash = protocol.canonical_config_hash(expected)[:12]
    observed_hash = protocol.canonical_config_hash(observed)[:12]
    return (
        f"Existing checkpoint is not an exact match for the requested run: "
        f"{run_dir} (requested={expected_hash}, saved={observed_hash}). "
        "Use a new output directory or --force-retrain; evaluation-only mode "
        "cannot use a mismatched checkpoint."
    )


def resolve_existing_variant_checkpoint(
    output_dir: Path,
    *,
    variant: str,
    training_seed: int,
    expected_signature: dict[str, Any],
) -> Path:
    """Return an exactly matching completed checkpoint or raise clearly."""

    run_dir = _variant_dir(output_dir, variant, training_seed)
    model_path = run_dir / "model_final.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No completed checkpoint for {variant} seed={training_seed}: "
            f"{model_path}. Enable training to create it."
        )
    signature_path = _signature_path(run_dir)
    if not signature_path.exists():
        raise RuntimeError(
            f"Cannot verify existing checkpoint because its training signature is "
            f"missing: {signature_path}. Use --force-retrain to create a verified run."
        )
    observed_signature = _canonical_payload(
        json.loads(signature_path.read_text(encoding="utf-8"))
    )
    if observed_signature != expected_signature:
        raise RuntimeError(
            _signature_mismatch_message(
                run_dir,
                expected=expected_signature,
                observed=observed_signature,
            )
        )
    completion_path = _completion_path(run_dir)
    if not completion_path.exists():
        raise RuntimeError(
            f"Cannot verify existing checkpoint because its completion record is "
            f"missing: {completion_path}. Use --force-retrain to create a verified run."
        )
    completion = _canonical_payload(
        json.loads(completion_path.read_text(encoding="utf-8"))
    )
    expected_hash = protocol.canonical_config_hash(expected_signature)
    if completion.get("training_signature_hash") != expected_hash:
        raise RuntimeError(
            f"Checkpoint completion record does not match its requested training "
            f"signature: {completion_path}. Use --force-retrain to replace it."
        )
    if completion.get("model_file") != model_path.name:
        raise RuntimeError(
            f"Checkpoint completion record names a different model file: "
            f"{completion_path}. Use --force-retrain to replace it."
        )
    try:
        completion_timesteps = int(completion.get("num_timesteps"))
    except (TypeError, ValueError):
        raise RuntimeError(
            f"Checkpoint completion record has no valid training budget: "
            f"{completion_path}. Use --force-retrain to replace it."
        ) from None
    if completion_timesteps != int(expected_signature["timesteps"]):
        raise RuntimeError(
            f"Checkpoint completion record has the wrong training budget: "
            f"{completion_path}. Use --force-retrain to replace it."
        )
    observed_hash = protocol.file_sha256(model_path)
    if completion.get("model_sha256") != observed_hash:
        raise RuntimeError(
            f"Checkpoint file checksum does not match its completion record: "
            f"{model_path}. Use --force-retrain to replace it."
        )
    return model_path


def _latest_rollout_checkpoint(run_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in (run_dir / "checkpoints").glob("rollout_*_steps.zip"):
        match = re.search(r"rollout_(\d+)_steps\.zip$", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _is_retryable_pending_run(run_dir: Path, pending_signature_path: Path) -> bool:
    """Whether a failed run can safely restart from its pending signature.

    A worker can fail while constructing the vector environment, before it has
    emitted a monitor row or an aligned checkpoint.  In that case the pending
    signature is the only file; empty worker-monitor directories are harmless.
    Any additional file is treated as meaningful partial output and still
    requires checkpoint-based resumption or an explicit force retrain.
    """

    if not pending_signature_path.exists():
        return False
    return all(
        path == pending_signature_path
        for path in run_dir.rglob("*")
        if path.is_file()
    )


def _tensorboard_event_representative(log_dir: Path) -> Path:
    """A deliberately long event path produced by Stable-Baselines3."""

    return log_dir / "train_9999" / ("events.out.tfevents." + "x" * 72)


def _tensorboard_path_is_safe(log_dir: Path) -> bool:
    return (
        os.name != "nt"
        or len(str(_tensorboard_event_representative(log_dir)))
        < WINDOWS_TENSORBOARD_PATH_LIMIT
    )


def _tensorboard_event_files(log_dir: Path) -> list[str]:
    if not log_dir.exists():
        return []
    return [
        str(path.resolve())
        for path in sorted(log_dir.rglob("events.out.tfevents.*"))
        if path.is_file()
    ]


def _tensorboard_log_dir(
    namespace: dict[str, Any],
    run_dir: Path,
    variant: str,
    training_seed: int,
    tensorboard_run_label: str | None = None,
) -> Path:
    """Return a durable artifact directory that is safe on long Windows paths.

    Short projects retain the intuitive ``<model>/tensorboard`` location.
    The current OneDrive workspace is too deep for TensorBoard's event names,
    so it first uses the enclosing workspace's ``artifacts/tb/ppo`` tree.
    If that path is still too long, use a short persistent per-user directory
    rather than the non-persistent Windows TEMP directory.  A user can set
    ``HIGHWAY_RL_TENSORBOARD_ROOT`` to provide an explicit persistent root.
    """

    preferred = run_dir / "tensorboard"
    if _tensorboard_path_is_safe(preferred):
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred

    project_root = Path(namespace["PROJECT_ROOT"]).resolve()
    compact_variant = TENSORBOARD_VARIANT_IDS[variant]
    run_label = (
        f"{str(tensorboard_run_label).strip()}_{compact_variant}_{int(training_seed)}"
        if tensorboard_run_label and str(tensorboard_run_label).strip()
        else f"{compact_variant}_{int(training_seed)}"
    )
    durable_candidates = [project_root.parent / "artifacts" / "tb" / "ppo" / run_label]
    project_id = hashlib.sha1(str(project_root).encode("utf-8")).hexdigest()[:10]
    fallback_run_id = "{}_{}_{}".format(
        compact_variant,
        int(training_seed),
        hashlib.sha1(run_label.encode("utf-8")).hexdigest()[:10],
    )
    fallback_roots: list[Path] = []
    configured_root = os.environ.get("HIGHWAY_RL_TENSORBOARD_ROOT")
    if configured_root:
        fallback_roots.append(Path(configured_root))
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        fallback_roots.append(Path(local_app_data) / "highway_rl_tb")
    fallback_roots.append(Path.home() / ".highway_rl_tb")
    for root in fallback_roots:
        durable = root / project_id / fallback_run_id
        if durable not in durable_candidates:
            durable_candidates.append(durable)
    creation_errors: list[str] = []
    for durable in durable_candidates:
        if not _tensorboard_path_is_safe(durable):
            continue
        try:
            durable.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            creation_errors.append(f"{durable} ({exc})")
        else:
            return durable
    tried = "; ".join(str(path) for path in durable_candidates)
    if creation_errors:
        tried = f"{tried}; mkdir errors: {'; '.join(creation_errors)}"
    raise RuntimeError(
        "No Windows-safe persistent TensorBoard path is available. "
        f"Tried {tried}; set HIGHWAY_RL_TENSORBOARD_ROOT to a short writable path."
    )


def _existing_tensorboard_log_dir(run_dir: Path) -> Path | None:
    """Recover the durable TensorBoard directory recorded for a reused model."""

    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    tensorboard = config.get("tensorboard", {})
    value = tensorboard.get("log_dir") if isinstance(tensorboard, dict) else None
    return Path(str(value)) if value else None


def _legacy_tensorboard_log_dir(
    run_dir: Path, variant: str, training_seed: int
) -> Path:
    """Locate the old TEMP fallback used before TensorBoard became an artifact."""

    digest = hashlib.sha1(str(run_dir.resolve()).encode("utf-8")).hexdigest()[:10]
    return (
        Path(tempfile.gettempdir())
        / "ppo_cbf_tb"
        / f"{variant[:8]}_{int(training_seed)}_{digest}"
    )


def _write_tensorboard_metadata(run_dir: Path, manifest: dict[str, Any]) -> None:
    """Store the TensorBoard location both beside and inside the model metadata."""

    (run_dir / "tb.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        return
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    config["tensorboard"] = manifest
    config_path.write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )


def restore_legacy_tensorboard_artifacts(
    namespace: dict[str, Any],
    *,
    run_dir: Path,
    variant: str,
    training_seed: int,
    tensorboard_run_label: str | None = None,
) -> Path | None:
    """Copy legacy TEMP event files into the durable model-associated artifact.

    This is intentionally a copy rather than a move: an interrupted migration
    cannot make the only remaining history disappear.  The copied events live
    under ``legacy/`` so they cannot collide with a later force-retrain's
    Stable-Baselines3 ``train_N`` run directories.
    """

    source_dir = _legacy_tensorboard_log_dir(run_dir, variant, training_seed)
    source_events = _tensorboard_event_files(source_dir)
    if not source_events:
        return _existing_tensorboard_log_dir(run_dir)
    log_dir = _tensorboard_log_dir(
        namespace,
        run_dir,
        variant,
        training_seed,
        tensorboard_run_label=tensorboard_run_label,
    )
    copied_event_files: list[str] = []
    for source_name in source_events:
        source = Path(source_name)
        destination = log_dir / "legacy" / source.relative_to(source_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or destination.stat().st_size != source.stat().st_size:
            shutil.copy2(source, destination)
        copied_event_files.append(str(destination.resolve()))
    all_event_files = _tensorboard_event_files(log_dir)
    manifest = {
        "schema_version": PROGRESSION_SCHEMA_VERSION,
        "variant": variant,
        "training_seed": int(training_seed),
        "log_dir": str(log_dir.resolve()),
        "command": f'tensorboard --logdir "{log_dir.resolve()}"',
        "legacy_source_dir": str(source_dir.resolve()),
        "legacy_event_files": copied_event_files,
        "new_event_files": [],
        "all_event_files": all_event_files,
        "retrain_note": (
            "Legacy event files were copied from the former TEMP fallback. "
            "Fresh forced retrains use separate train_N directories."
        ),
    }
    _write_tensorboard_metadata(run_dir, manifest)
    print(
        "[ppo-progression] migrated legacy TensorBoard events "
        f"variant={variant} logdir={log_dir.resolve()}",
        flush=True,
    )
    return log_dir


def _base_environment(
    namespace: dict[str, Any],
    *,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
) -> gym.Env:
    _ensure_ppo_observation_variant(namespace, env_config)
    env = gym.make(
        "lane-free-v0", render_mode=None, config=copy.deepcopy(env_config)
    )
    env = namespace["KaralakouRewardWrapper"](
        env, reward_config=copy.deepcopy(reward_config)
    )
    if namespace.get("NORMALIZE_RL_OBSERVATIONS", False):
        env = namespace["LaneFreeObservationNormalizationWrapper"](
            env, clip=namespace["OBSERVATION_CLIP"]
        )
    return env


def make_ppo_cbf_env(
    namespace: dict[str, Any],
    *,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    project_inputs: bool,
    lambda_delta: float,
    lambda_intervention: float,
    correction_epsilon: float,
    action_rate_penalty_lambda: float = 0.0,
    monitor_path: Path | None = None,
) -> gym.Env:
    """Build the shared physical-action/context environment for every level."""

    env = _base_environment(
        namespace, env_config=env_config, reward_config=reward_config
    )
    env = CBFContextPhysicalActionWrapper(
        env,
        namespace=namespace,
        ax_bounds=namespace["CBF_AX_BOUNDS"],
        ay_bounds=namespace["CBF_AY_BOUNDS"],
        neighbor_range=float(namespace["CBF_NEIGHBOR_RANGE"]),
        eps_side=float(namespace["CBF_EPS_SIDE"]),
        k0=float(namespace["CBF_K0"]),
        k1=float(namespace["CBF_K1"]),
        max_neighbor_constraints=int(namespace["CBF_MAX_NEIGHBOR_CONSTRAINTS"]),
        base_observation_dim=int(np.prod(env.observation_space.shape)),
        project_inputs=bool(project_inputs),
        lambda_delta=float(lambda_delta),
        lambda_intervention=float(lambda_intervention),
        correction_epsilon=float(correction_epsilon),
        action_rate_penalty_lambda=float(action_rate_penalty_lambda),
    )
    if "KPIInfoWrapper" in namespace:
        env = namespace["KPIInfoWrapper"](
            env, intervention_threshold=float(correction_epsilon)
        )
    env = protocol.ProtocolMetricsWrapper(env)
    if monitor_path is not None:
        monitor_path.parent.mkdir(parents=True, exist_ok=True)
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=protocol.TRAINING_MONITOR_INFO_KEYS,
        )
    return env


def _make_ppo_worker_env(
    *,
    project_root: str,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    cbf_snapshot: dict[str, Any],
    lambda_delta: float,
    lambda_intervention: float,
    correction_epsilon: float,
    action_rate_penalty_lambda: float,
    monitor_path: str,
) -> gym.Env:
    """Create a PPO environment inside a spawned rollout worker.

    Notebook classes live in an execution namespace that cannot be reliably
    cloudpickled on Windows.  Each worker therefore reconstructs only the
    required notebook definitions locally and receives plain configuration
    data from the learner process.
    """

    worker_project_root = Path(project_root)
    protocol.set_stable_native_defaults()
    worker_namespace = protocol.bootstrap_notebook_namespace(worker_project_root)
    protocol.exec_required_notebook_cells(
        worker_project_root / "notebooks" / "lanelessKaralakou.ipynb",
        worker_namespace,
    )
    worker_namespace.update(copy.deepcopy(cbf_snapshot))
    return make_ppo_cbf_env(
        worker_namespace,
        env_config=copy.deepcopy(env_config),
        reward_config=copy.deepcopy(reward_config),
        # Collection stages P(z) explicitly and steps only the safe action.
        project_inputs=False,
        lambda_delta=float(lambda_delta),
        lambda_intervention=float(lambda_intervention),
        correction_epsilon=float(correction_epsilon),
        action_rate_penalty_lambda=float(action_rate_penalty_lambda),
        monitor_path=Path(monitor_path),
    )


def compact_worker_monitor_path(monitor_path: Path, rank: int) -> Path:
    """Return a short, unique monitor path for a spawned rollout worker.

    The experiment run path is intentionally descriptive.  On Windows, adding
    both ``training_monitors`` and a long per-worker filename can exceed the
    legacy path limit before a subprocess has even initialized.  Monitor files
    are internal rollout diagnostics, so keep their names compact while still
    retaining one distinct file per worker in the variant run directory.
    """

    if int(rank) < 0:
        raise ValueError("worker rank must be non-negative")
    # Stable-Baselines3 appends ``.monitor.csv`` unless the supplied filename
    # already ends with that suffix.  Keep the short name final as written.
    return monitor_path.parent / f"m{int(rank)}.monitor.csv"


def make_training_vec_env(
    namespace: dict[str, Any],
    *,
    variant: str,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    spec: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
    monitor_path: Path,
) -> VecEnv:
    """Create one local env or independent spawned rollout workers."""

    topology = training_topology(args)
    n_envs = int(topology["n_envs"])
    reward_on = bool(spec["reward_penalty"])
    lambda_delta = float(args.lambda_delta) if reward_on else 0.0
    lambda_intervention = float(args.lambda_intervention) if reward_on else 0.0
    action_rate_penalty_lambda = (
        float(args.action_rate_penalty)
        if str(variant) == "ppo_nominal"
        else 0.0
    )

    if n_envs == 1:
        def factory() -> gym.Env:
            return make_ppo_cbf_env(
                namespace,
                env_config=env_config,
                reward_config=reward_config,
                project_inputs=False,
                lambda_delta=lambda_delta,
                lambda_intervention=lambda_intervention,
                correction_epsilon=float(args.correction_epsilon),
                action_rate_penalty_lambda=action_rate_penalty_lambda,
                monitor_path=monitor_path,
            )

        vec_env: VecEnv = DummyVecEnv([factory])
    else:
        project_root = Path(namespace["PROJECT_ROOT"]).resolve()
        worker_common = {
            "project_root": str(project_root),
            "env_config": copy.deepcopy(env_config),
            "reward_config": copy.deepcopy(reward_config),
            "cbf_snapshot": _cbf_training_snapshot(namespace),
            "lambda_delta": lambda_delta,
            "lambda_intervention": lambda_intervention,
            "correction_epsilon": float(args.correction_epsilon),
            "action_rate_penalty_lambda": action_rate_penalty_lambda,
        }
        env_fns = [
            partial(
                _make_ppo_worker_env,
                **worker_common,
                monitor_path=str(compact_worker_monitor_path(monitor_path, rank)),
            )
            for rank in range(n_envs)
        ]
        vec_env = SubprocVecEnv(
            env_fns,
            start_method=str(topology["start_method"]),
        )
        if int(vec_env.num_envs) != n_envs:
            vec_env.close()
            raise RuntimeError(
                "Subprocess PPO environment count differs from the requested "
                f"count ({vec_env.num_envs} != {n_envs})"
            )
    vec_env.seed(int(seed))
    return vec_env


def resolved_ppo_config(args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(PPO_CONFIGS[str(args.ppo_config)])
    if args.n_steps is not None:
        config["n_steps"] = int(args.n_steps)
    if args.batch_size is not None:
        config["batch_size"] = int(args.batch_size)
    if args.n_epochs is not None:
        config["n_epochs"] = int(args.n_epochs)
    topology = training_topology(args)
    n_envs = int(topology["n_envs"])
    rollout_size = int(config["n_steps"])
    if rollout_size <= 0 or rollout_size % n_envs != 0:
        raise ValueError(
            "--n-steps is the global PPO rollout size and must be positive "
            f"and divisible by n_envs ({rollout_size} vs {n_envs})"
        )
    if int(config["batch_size"]) > rollout_size or rollout_size % int(
        config["batch_size"]
    ) != 0:
        raise ValueError("PPO batch_size must divide the global rollout size")
    if int(args.timesteps) % rollout_size != 0:
        raise ValueError(
            f"timesteps={args.timesteps} must be divisible by global rollout "
            f"size={rollout_size}"
        )
    config["n_steps"] = rollout_size // n_envs
    config["global_rollout_steps"] = rollout_size
    config["n_envs"] = n_envs
    return config


def build_model(
    *,
    variant: str,
    train_env: VecEnv,
    config: dict[str, Any],
    training_seed: int,
    args: argparse.Namespace,
    tensorboard_log: Path,
    base_observation_dim: int,
) -> LatentActionPPO:
    spec = VARIANT_SPECS[variant]
    common_policy_kwargs: dict[str, Any] = {
        "net_arch": {"pi": [256, 128], "vf": [256, 128]},
        "activation_fn": th.nn.Tanh,
        "ortho_init": True,
        "log_std_init": float(config["log_std_init"]),
    }
    common_model_kwargs: dict[str, Any] = {
        "learning_rate": float(config["learning_rate"]),
        "n_steps": int(config["n_steps"]),
        "batch_size": int(config["batch_size"]),
        "n_epochs": int(config["n_epochs"]),
        "gamma": float(config["gamma"]),
        "gae_lambda": float(config["gae_lambda"]),
        "clip_range": float(config["clip_range"]),
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": float(config["ent_coef"]),
        "vf_coef": float(config["vf_coef"]),
        "max_grad_norm": float(config["max_grad_norm"]),
        "use_sde": False,
        "tensorboard_log": str(tensorboard_log),
        "verbose": 0,
        "seed": int(training_seed),
        "device": str(args.device),
        "execution_mode": str(spec["execution_mode"]),
        "cbf_base_observation_dim": int(base_observation_dim),
        "cbf_max_constraints": 18,
    }
    if bool(spec["projected_mean"]):
        projected_policy_kwargs = {
            **common_policy_kwargs,
            "cbf_base_observation_dim": int(base_observation_dim),
            "cbf_max_constraints": 18,
            "use_safety_critic": bool(spec.get("safety_critic", False)),
        }
        return ProjectedCBFPPO(
            ProjectedCBFActorCriticPolicy,
            train_env,
            lambda_mean=(
                float(args.lambda_mean)
                if bool(spec.get("differentiable_actor_loss", False))
                else 0.0
            ),
            lambda_sample=(
                float(args.lambda_sample)
                if bool(spec.get("differentiable_actor_loss", False))
                else 0.0
            ),
            lambda_critic=(
                float(args.lambda_critic)
                if bool(spec.get("safety_critic", False))
                else 0.0
            ),
            safety_gamma=float(args.safety_critic_gamma),
            safety_cost_clip=float(args.safety_critic_cost_clip),
            policy_kwargs=projected_policy_kwargs,
            **common_model_kwargs,
        )
    if bool(spec.get("detached_actor_loss", False)):
        detached_policy_kwargs = {
            **common_policy_kwargs,
            "cbf_base_observation_dim": int(base_observation_dim),
        }
        return DetachedCBFActorPPO(
            DetachedCBFActorCriticPolicy,
            train_env,
            lambda_actor=float(args.lambda_detached_actor),
            policy_kwargs=detached_policy_kwargs,
            **common_model_kwargs,
        )
    return LatentActionPPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=context_ignoring_policy_kwargs(
            base_observation_dim=int(base_observation_dim),
            policy_kwargs=common_policy_kwargs
        ),
        **common_model_kwargs,
    )


def load_model(
    variant: str,
    path: Path,
    device: str,
    env: VecEnv | None = None,
) -> LatentActionPPO:
    spec = VARIANT_SPECS[variant]
    if bool(spec["projected_mean"]):
        model_class = ProjectedCBFPPO
    elif bool(spec.get("detached_actor_loss", False)):
        model_class = DetachedCBFActorPPO
    else:
        model_class = LatentActionPPO
    return model_class.load(str(path), device=device, env=env)


def train_variant(
    namespace: dict[str, Any],
    *,
    variant: str,
    training_seed: int,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    args: argparse.Namespace,
    output_dir: Path,
) -> VariantTrainingResult:
    run_dir = _variant_dir(output_dir, variant, training_seed)
    model_path = run_dir / "model_final.zip"
    expected_signature = training_signature(
        namespace,
        variant=variant,
        training_seed=training_seed,
        env_config=env_config,
        reward_config=reward_config,
        args=args,
    )
    signature_path = _signature_path(run_dir)
    pending_signature_path = _pending_signature_path(run_dir)
    retryable_pending_run = False
    if model_path.exists() and not args.force_retrain:
        model_path = resolve_existing_variant_checkpoint(
            output_dir,
            variant=variant,
            training_seed=training_seed,
            expected_signature=expected_signature,
        )
        print(
            f"[ppo-progression] reuse exact {variant}: {model_path}",
            flush=True,
        )
        tensorboard_log_dir = restore_legacy_tensorboard_artifacts(
            namespace,
            run_dir=run_dir,
            variant=variant,
            training_seed=training_seed,
            tensorboard_run_label=getattr(args, "tensorboard_run_label", None),
        )
        return VariantTrainingResult(
            model_path=model_path,
            trained=False,
            tensorboard_log_dir=tensorboard_log_dir,
        )
    if pending_signature_path.exists() and not args.force_retrain:
        observed_signature = _canonical_payload(
            json.loads(pending_signature_path.read_text(encoding="utf-8"))
        )
        if observed_signature != expected_signature:
            raise RuntimeError(
                _signature_mismatch_message(
                    run_dir,
                    expected=expected_signature,
                    observed=observed_signature,
                )
            )
        retryable_pending_run = _is_retryable_pending_run(
            run_dir, pending_signature_path
        )
    elif run_dir.exists() and any(run_dir.iterdir()) and not args.force_retrain:
        raise RuntimeError(
            "Partial PPO artifacts have no pending training signature and cannot "
            f"be safely resumed: {run_dir}. Use --force-retrain to replace them."
        )
    resume_checkpoint = (
        None if args.force_retrain else _latest_rollout_checkpoint(run_dir)
    )
    if (
        run_dir.exists()
        and any(run_dir.iterdir())
        and not args.force_retrain
        and resume_checkpoint is None
        and not retryable_pending_run
    ):
        raise RuntimeError(
            "Partial artifacts exist but no rollout-aligned checkpoint can be "
            f"resumed: {run_dir}"
        )
    if retryable_pending_run:
        print(
            f"[ppo-progression] retrying pre-rollout failure for {variant}: "
            f"{run_dir}",
            flush=True,
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol.seed_everything(training_seed)
    config = resolved_ppo_config(args)
    pending_signature_path.write_text(
        json.dumps(expected_signature, indent=2, sort_keys=True), encoding="utf-8"
    )
    tensorboard_log_dir = _tensorboard_log_dir(
        namespace,
        run_dir,
        variant,
        training_seed,
        tensorboard_run_label=getattr(args, "tensorboard_run_label", None),
    )
    tensorboard_before = set(_tensorboard_event_files(tensorboard_log_dir))
    train_env = make_training_vec_env(
        namespace,
        variant=variant,
        env_config=env_config,
        reward_config=reward_config,
        spec=VARIANT_SPECS[variant],
        args=args,
        seed=training_seed,
        monitor_path=run_dir / "training.monitor.csv",
    )
    if resume_checkpoint is None:
        model = build_model(
            variant=variant,
            train_env=train_env,
            config=config,
            training_seed=training_seed,
            args=args,
            tensorboard_log=tensorboard_log_dir,
            base_observation_dim=_base_observation_dim(env_config),
        )
    else:
        # Supply the vectorized environment while loading so SB3 constructs a
        # rollout buffer with the saved worker count rather than its one-env
        # default.  ``set_env`` rejects a changed worker count too late.
        model = load_model(variant, resume_checkpoint, args.device, env=train_env)
        model.tensorboard_log = str(tensorboard_log_dir)
        if (
            int(model.n_steps) != int(config["n_steps"])
            or int(model.batch_size) != int(config["batch_size"])
            or int(model.n_epochs) != int(config["n_epochs"])
            or int(model.n_envs) != int(config["n_envs"])
            or int(model.rollout_buffer.n_envs) != int(config["n_envs"])
            or str(model.execution_mode) != str(VARIANT_SPECS[variant]["execution_mode"])
        ):
            raise RuntimeError(
                f"Checkpoint PPO configuration mismatch: {resume_checkpoint}"
            )
        print(
            f"[ppo-progression] resume {variant} from "
            f"step={model.num_timesteps:,}: {resume_checkpoint}",
            flush=True,
        )
    training_metrics_path = run_dir / "training_episodes.csv"
    metrics_callback = protocol.TrainingMetricsCallback(
        path=training_metrics_path,
        training_seed=training_seed,
        variant=variant,
    )
    if resume_checkpoint is not None and training_metrics_path.exists():
        existing_metrics = pd.read_csv(training_metrics_path)
        if not existing_metrics.empty:
            metrics_callback.episode_index = int(
                pd.to_numeric(
                    existing_metrics["episode_index"], errors="coerce"
                ).max()
            )
            metrics_callback.resets_after_collision = int(
                pd.to_numeric(
                    existing_metrics["resets_after_collision"], errors="coerce"
                ).iloc[-1]
            )
    action_callback = PPOActionClipCallback()
    checkpoint_frequency_effective = max(
        int(config["global_rollout_steps"]),
        (
            (
                int(args.checkpoint_freq)
                + int(config["global_rollout_steps"])
                - 1
            )
            // int(config["global_rollout_steps"])
        )
        * int(config["global_rollout_steps"]),
    )
    # CheckpointCallback counts parent callback invocations, not individual
    # transitions.  One SubprocVecEnv step advances all workers at once.
    checkpoint_save_freq = checkpoint_frequency_effective // int(
        config["n_envs"]
    )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_save_freq,
        save_path=str(checkpoint_dir),
        name_prefix="rollout",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=0,
    )
    started = time.perf_counter()
    try:
        remaining_timesteps = int(args.timesteps) - int(model.num_timesteps)
        if remaining_timesteps < 0:
            raise RuntimeError(
                f"Checkpoint step {model.num_timesteps} exceeds target {args.timesteps}"
            )
        if remaining_timesteps:
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=CallbackList(
                    [metrics_callback, action_callback, checkpoint_callback]
                ),
                reset_num_timesteps=resume_checkpoint is None,
                tb_log_name="train",
                log_interval=1,
                progress_bar=False,
            )
        model.save(str(model_path))
        signature_path.write_text(
            json.dumps(expected_signature, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        completion = {
            "schema_version": PROGRESSION_SCHEMA_VERSION,
            "training_signature_hash": protocol.canonical_config_hash(
                expected_signature
            ),
            "model_file": model_path.name,
            "model_sha256": protocol.file_sha256(model_path),
            "num_timesteps": int(model.num_timesteps),
        }
        _completion_path(run_dir).write_text(
            json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8"
        )
        pending_signature_path.unlink(missing_ok=True)
        diagnostics = getattr(model, "cbf_training_diagnostics", None)
        if diagnostics:
            diagnostic_frame = pd.DataFrame(diagnostics)
            diagnostic_frame.to_csv(
                run_dir / "actor_gradients.csv",
                index=False,
            )
            latest = diagnostic_frame.iloc[-1]
            print(
                "[ppo-progression] CBF actor-feedback gradients",
                {
                    "g_cbf/g_ppo": float(
                        latest.get("g_cbf_to_g_ppo_ratio", np.nan)
                    ),
                    "cos(g_ppo,g_cbf)": float(
                        latest.get("g_ppo_g_cbf_cosine", np.nan)
                    ),
                    "mean_internalization_loss": float(
                        latest.get("mean_loss", np.nan)
                    ),
                },
                flush=True,
            )
    finally:
        train_env.close()
    tensorboard_event_files = _tensorboard_event_files(tensorboard_log_dir)
    tensorboard_new_event_files = [
        path for path in tensorboard_event_files if path not in tensorboard_before
    ]
    tensorboard_manifest = {
        "schema_version": PROGRESSION_SCHEMA_VERSION,
        "variant": variant,
        "training_seed": int(training_seed),
        "log_dir": str(tensorboard_log_dir.resolve()),
        "command": f'tensorboard --logdir "{tensorboard_log_dir.resolve()}"',
        "new_event_files": tensorboard_new_event_files,
        "all_event_files": tensorboard_event_files,
        "retrain_note": (
            "Stable-Baselines3 creates a new train_N subdirectory for a fresh "
            "forced retrain, so TensorBoard histories remain distinguishable."
        ),
    }
    effective_settings = _effective_training_settings(variant, args)
    run_config = {
        "schema_version": PROGRESSION_SCHEMA_VERSION,
        "variant": variant,
        "variant_spec": VARIANT_SPECS[variant],
        "training_seed": int(training_seed),
        "timesteps": int(args.timesteps),
        "ppo_config_name": str(args.ppo_config),
        "tensorboard_run_label": str(
            getattr(args, "tensorboard_run_label", None) or ""
        ),
        "ppo_config": config,
        **effective_settings,
        "training_signature": expected_signature,
        "training_signature_hash": protocol.canonical_config_hash(
            expected_signature
        ),
        "buffer_action": "latent Gaussian sample z",
        "environment_action": (
            "hard CBF projection P_s(z)"
            if VARIANT_SPECS[variant]["execution_mode"] == "cbf"
            else "physical action-box projection"
        ),
        "common_physical_action_bounds": {
            "ax": list(namespace["CBF_AX_BOUNDS"]),
            "ay": list(namespace["CBF_AY_BOUNDS"]),
        },
        "traffic_model": active_traffic_model(env_config),
        "collection_topology": training_topology(args),
        "global_rollout_steps": int(config["global_rollout_steps"]),
        "per_env_rollout_steps": int(config["n_steps"]),
        "checkpoint_frequency_requested": int(args.checkpoint_freq),
        "checkpoint_frequency_effective": int(checkpoint_frequency_effective),
        "checkpoint_callback_frequency": int(checkpoint_save_freq),
        "tensorboard": tensorboard_manifest,
        "resumed_from": (
            None if resume_checkpoint is None else str(resume_checkpoint)
        ),
        "resume_semantics": (
            "rollout-aligned PPO model/optimizer continuation; simulator and "
            "process RNG state are re-seeded, so this is crash recovery rather "
            "than bit-exact trajectory replay"
        ),
        "env_config": env_config,
        "reward_config": reward_config,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str), encoding="utf-8"
    )
    _write_tensorboard_metadata(run_dir, tensorboard_manifest)
    print(
        f"[ppo-progression] trained {variant} steps={model.num_timesteps:,} "
        f"model={model_path}",
        flush=True,
    )
    print(
        "[ppo-progression] TensorBoard "
        f"variant={variant} logdir={tensorboard_log_dir.resolve()}",
        flush=True,
    )
    return VariantTrainingResult(
        model_path=model_path,
        trained=True,
        tensorboard_log_dir=tensorboard_log_dir,
    )


def make_evaluation_env(
    namespace: dict[str, Any],
    *,
    mode: str,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    correction_epsilon: float,
    task_distance_m: float = DEFAULT_TASK_DISTANCE_M,
    task_max_policy_steps: int = DEFAULT_TASK_MAX_POLICY_STEPS,
) -> gym.Env:
    env = make_ppo_cbf_env(
        namespace,
        env_config=env_config,
        reward_config=reward_config,
        project_inputs=mode == "cbf",
        lambda_delta=0.0,
        lambda_intervention=0.0,
        correction_epsilon=correction_epsilon,
        monitor_path=None,
    )
    return DistanceTaskEvaluationWrapper(
        env,
        task_distance_m=float(task_distance_m),
        max_policy_steps=int(task_max_policy_steps),
    )


def evaluate_scenario(
    namespace: dict[str, Any],
    *,
    model: LatentActionPPO,
    variant: str,
    mode: str,
    training_seed: int,
    scenario_seed: int,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    args: argparse.Namespace,
    action_source: str = "policy",
) -> dict[str, Any]:
    env = make_evaluation_env(
        namespace,
        mode=mode,
        env_config=env_config,
        reward_config=reward_config,
        correction_epsilon=float(args.correction_epsilon),
        task_distance_m=_task_distance_from_args(args, env_config),
        task_max_policy_steps=_task_max_policy_steps_from_args(args),
    )
    try:
        observation, _ = env.reset(seed=int(scenario_seed))
        policy_dt = protocol._policy_dt(env)
        rewards: list[float] = []
        speed_errors: list[float] = []
        lateral_errors: list[float] = []
        h_values: list[float] = []
        jerk_norms: list[float] = []
        interventions: list[float] = []
        corrections: list[float] = []
        qp_failures: list[float] = []
        shadow_interventions: list[float] = []
        shadow_corrections: list[float] = []
        episode_returns: list[float] = []
        episode_lengths: list[float] = []
        segment_return = 0.0
        segment_steps = 0
        total_distance_m = 0.0
        collision_events = 0
        previous_acceleration = np.zeros(2, dtype=float)

        for step in range(int(args.eval_timesteps)):
            pre_state = protocol.cbf_state_occupancy_metrics(
                namespace,
                env,
                eps_side=float(namespace["CBF_EPS_SIDE"]),
                ttc_cap_s=float(args.ttc_cap),
            )
            h_values.append(float(pre_state.get("h_min", np.nan)))
            action = _predict_evaluation_action(
                model,
                observation,
                action_source=action_source,
            )
            context_wrapper = env.get_wrapper_attr("project_current_action")
            shadow_safe, shadow_record = context_wrapper(action)
            shadow_correction = float(shadow_record["correction_norm_normalized"])
            shadow_corrections.append(shadow_correction)
            shadow_interventions.append(
                float(shadow_correction > float(args.correction_epsilon))
            )

            observation, reward, terminated, truncated, info = env.step(action)
            info = dict(info)
            rewards.append(float(reward))
            segment_return += float(reward)
            segment_steps += 1
            total_distance_m += float(
                info.get(
                    "task_distance_step_m",
                    info.get("pipeline_distance_step_m", 0.0),
                )
            )
            step_collision_events = max(
                int(info.get("ego_collision_events", 0)), 0
            )
            if (
                step_collision_events == 0
                and bool(
                    info.get("ego_collision", False)
                    or info.get("task_collision_terminated", False)
                )
            ):
                step_collision_events = 1
            collision_events += step_collision_events
            base = env.unwrapped
            speed_errors.append(
                float(
                    info.get(
                        "karalakou_abs_speed_deviation",
                        abs(float(base.vehicle.vx) - float(base.vehicle.desired_speed)),
                    )
                )
            )
            lateral_error = float(info.get("karalakou_lat_y_error_m", np.nan))
            if np.isfinite(lateral_error):
                lateral_errors.append(lateral_error)
            accelerations = np.asarray(
                getattr(base, "_last_accelerations", np.empty((0, 2))),
                dtype=float,
            )
            if accelerations.ndim == 2 and accelerations.shape[0] > 0:
                acceleration = accelerations[0, :2]
                jerk_norms.append(
                    float(
                        np.linalg.norm(acceleration - previous_acceleration)
                        / max(policy_dt, 1e-6)
                    )
                )
                previous_acceleration = acceleration.copy()
            if mode == "cbf":
                corrections.append(
                    float(info.get("cbf_correction_norm_normalized", 0.0))
                )
                interventions.append(float(info.get("cbf_event_intervened", False)))
                qp_failures.append(float(not bool(info.get("cbf_qp_success", True))))
            else:
                corrections.append(0.0)
                interventions.append(0.0)
                qp_failures.append(0.0)

            if terminated or truncated:
                episode_returns.append(float(segment_return))
                episode_lengths.append(float(segment_steps))
                # A scenario is one strict distance task, not a concatenation
                # of reset episodes.  Continuing here would make a reported
                # "scenario distance" exceed the 1 km completion cap.
                segment_steps = 0
                break

        if segment_steps > 0:
            episode_returns.append(float(segment_return))
            episode_lengths.append(float(segment_steps))
        collisions_per_km = (
            1000.0 * float(collision_events) / float(total_distance_m)
            if total_distance_m > 1e-9
            else np.nan
        )
        return {
            "schema_version": PROGRESSION_SCHEMA_VERSION,
            "variant": variant,
            "variant_label": VARIANT_SPECS[variant]["label"],
            "mode": mode,
            "action_source": action_source,
            "training_seed": int(training_seed),
            "scenario_seed": int(scenario_seed),
            "timesteps": int(len(rewards)),
            "episode_return": _finite_mean(episode_returns),
            "episode_length_steps": _finite_mean(episode_lengths),
            "ego_collisions_per_km": float(collisions_per_km),
            "h_min": _finite_min(h_values),
            "qp_failure_rate": _finite_mean(qp_failures, default=0.0),
            "mean_abs_speed_deviation": _finite_mean(speed_errors, default=0.0),
            "mean_lat_y_error_m": _finite_mean(lateral_errors),
            "event_intervention_rate": _finite_mean(interventions, default=0.0),
            "mean_correction_norm": _finite_mean(corrections, default=0.0),
            "mean_jerk_norm": _finite_mean(jerk_norms, default=0.0),
            "shadow_event_intervention_rate": _finite_mean(
                shadow_interventions, default=0.0
            ),
            "shadow_mean_correction_norm": _finite_mean(
                shadow_corrections, default=0.0
            ),
            "total_return": float(np.sum(rewards)),
            "total_distance_m": float(total_distance_m),
            "distinct_ego_collision_events": int(collision_events),
        }
    finally:
        env.close()


def evaluate_completed_episode(
    namespace: dict[str, Any],
    *,
    model: LatentActionPPO,
    variant: str,
    mode: str,
    training_seed: int,
    episode_index: int,
    episode_seed: int,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    args: argparse.Namespace,
    action_source: str = "policy",
) -> dict[str, Any]:
    """Evaluate exactly one complete episode for the immediate post-train table."""

    env = make_evaluation_env(
        namespace,
        mode=mode,
        env_config=env_config,
        reward_config=reward_config,
        correction_epsilon=float(args.correction_epsilon),
        task_distance_m=_task_distance_from_args(args, env_config),
        task_max_policy_steps=_task_max_policy_steps_from_args(args),
    )
    try:
        observation, _ = env.reset(seed=int(episode_seed))
        policy_dt = protocol._policy_dt(env)
        rewards: list[float] = []
        speed_errors: list[float] = []
        lateral_errors: list[float] = []
        h_values: list[float] = []
        jerk_norms: list[float] = []
        interventions: list[float] = []
        corrections: list[float] = []
        qp_failures: list[float] = []
        shadow_interventions: list[float] = []
        shadow_corrections: list[float] = []
        total_distance_m = 0.0
        collision_events = 0
        collision_events_direct_qp_failure = 0
        collision_events_direct_qp_failure_or_fallback = 0
        collision_event_records: list[dict[str, Any]] = []
        first_collision_step: int | None = None
        first_collision_qp_failure_same_step = False
        first_collision_qp_failure_or_fallback_same_step = False
        previous_step_qp_failure = False
        any_prior_qp_failure = False
        previous_acceleration = np.zeros(2, dtype=float)

        while True:
            pre_state = protocol.cbf_state_occupancy_metrics(
                namespace,
                env,
                eps_side=float(namespace["CBF_EPS_SIDE"]),
                ttc_cap_s=float(args.ttc_cap),
            )
            h_values.append(float(pre_state.get("h_min", np.nan)))
            action = _predict_evaluation_action(
                model,
                observation,
                action_source=action_source,
            )
            context_wrapper = env.get_wrapper_attr("project_current_action")
            _shadow_safe, shadow_record = context_wrapper(action)
            shadow_correction = float(shadow_record["correction_norm_normalized"])
            shadow_corrections.append(shadow_correction)
            shadow_interventions.append(
                float(shadow_correction > float(args.correction_epsilon))
            )

            observation, reward, terminated, truncated, info = env.step(action)
            info = dict(info)
            rewards.append(float(reward))
            # The event-level QP attribution is populated after collision
            # count normalization below, on this same policy transition.
            step_distance_m = float(
                info.get(
                    "task_distance_step_m",
                    info.get("pipeline_distance_step_m", 0.0),
                )
            )
            total_distance_m += step_distance_m
            step_collision_events = max(
                int(info.get("ego_collision_events", 0)), 0
            )
            if (
                step_collision_events == 0
                and bool(
                    info.get("ego_collision", False)
                    or info.get("task_collision_terminated", False)
                )
            ):
                step_collision_events = 1
            collision_events += step_collision_events
            policy_step = int(len(rewards))
            if mode == "cbf":
                cbf_qp_success = bool(info.get("cbf_qp_success", True))
                cbf_fallback_used = bool(info.get("cbf_fallback_used", False))
                cbf_substep_fallback_steps = int(
                    info.get("cbf_substep_fallback_steps", 0)
                )
                qp_failure_same_step = bool(not cbf_qp_success)
                qp_failure_or_fallback_same_step = bool(
                    qp_failure_same_step
                    or cbf_fallback_used
                    or cbf_substep_fallback_steps > 0
                )
            else:
                cbf_qp_success = True
                cbf_fallback_used = False
                cbf_substep_fallback_steps = 0
                qp_failure_same_step = False
                qp_failure_or_fallback_same_step = False
            if step_collision_events > 0:
                event_start_index = int(collision_events - step_collision_events + 1)
                if first_collision_step is None:
                    first_collision_step = policy_step
                    first_collision_qp_failure_same_step = bool(
                        qp_failure_same_step
                    )
                    first_collision_qp_failure_or_fallback_same_step = bool(
                        qp_failure_or_fallback_same_step
                    )
                if qp_failure_same_step:
                    collision_events_direct_qp_failure += step_collision_events
                if qp_failure_or_fallback_same_step:
                    collision_events_direct_qp_failure_or_fallback += (
                        step_collision_events
                    )
                for event_offset in range(step_collision_events):
                    collision_event_records.append(
                        {
                            "schema_version": PROGRESSION_SCHEMA_VERSION,
                            "collision_event_attribution_schema_version": (
                                COLLISION_EVENT_ATTRIBUTION_SCHEMA_VERSION
                            ),
                            "evaluation_kind": "collision_event_qp_attribution",
                            "variant": variant,
                            "variant_label": VARIANT_SPECS[variant]["label"],
                            "mode": mode,
                            "action_source": action_source,
                            "external_cbf": "ON" if mode == "cbf" else "OFF",
                            "training_seed": int(training_seed),
                            "episode_index": int(episode_index),
                            "scenario_seed": int(episode_seed),
                            "episode_seed": int(episode_seed),
                            "collision_event_index": int(
                                event_start_index + event_offset
                            ),
                            "policy_step": policy_step,
                            "time_s": float(policy_step * policy_dt),
                            "step_distance_m": step_distance_m,
                            "cumulative_distance_m": float(total_distance_m),
                            "collision_terminal_step": bool(
                                info.get("task_collision_terminated", False)
                            ),
                            "qp_failure_same_step": bool(qp_failure_same_step),
                            "qp_failure_previous_step": bool(
                                previous_step_qp_failure
                            ),
                            "qp_failure_seen_before_step": bool(any_prior_qp_failure),
                            "qp_failure_or_fallback_same_step": bool(
                                qp_failure_or_fallback_same_step
                            ),
                            "cbf_qp_success": bool(cbf_qp_success),
                            "cbf_fallback_used": bool(cbf_fallback_used),
                            "cbf_substep_fallback_steps": int(
                                cbf_substep_fallback_steps
                            ),
                            "cbf_substep_count": int(
                                info.get("cbf_substep_count", 0)
                            ),
                            "cbf_raw_feasible": bool(
                                info.get("cbf_raw_feasible", True)
                            ),
                            "cbf_event_intervened": bool(
                                info.get("cbf_event_intervened", False)
                            ),
                            "cbf_correction_norm_normalized": float(
                                info.get("cbf_correction_norm_normalized", 0.0)
                            ),
                            "cbf_hocbf_condition_satisfied": bool(
                                info.get("cbf_hocbf_condition_satisfied", True)
                            ),
                            "cbf_hocbf_min_margin": float(
                                info.get("cbf_hocbf_min_margin", np.nan)
                            ),
                            "cbf_max_constraint_violation_raw": float(
                                info.get(
                                    "cbf_max_constraint_violation_raw", np.nan
                                )
                            ),
                            "cbf_max_constraint_violation_safe": float(
                                info.get(
                                    "cbf_max_constraint_violation_safe", np.nan
                                )
                            ),
                            "h_min_before_step": float(
                                pre_state.get("h_min", np.nan)
                            ),
                            "psi1_min_before_step": float(
                                pre_state.get("psi1_min", np.nan)
                            ),
                        }
                    )
            previous_step_qp_failure = bool(qp_failure_same_step)
            any_prior_qp_failure = bool(
                any_prior_qp_failure or qp_failure_same_step
            )
            base = env.unwrapped
            speed_errors.append(
                float(
                    info.get(
                        "karalakou_abs_speed_deviation",
                        abs(float(base.vehicle.vx) - float(base.vehicle.desired_speed)),
                    )
                )
            )
            lateral_error = float(info.get("karalakou_lat_y_error_m", np.nan))
            if np.isfinite(lateral_error):
                lateral_errors.append(lateral_error)
            accelerations = np.asarray(
                getattr(base, "_last_accelerations", np.empty((0, 2))),
                dtype=float,
            )
            if accelerations.ndim == 2 and accelerations.shape[0] > 0:
                acceleration = accelerations[0, :2]
                jerk_norms.append(
                    float(
                        np.linalg.norm(acceleration - previous_acceleration)
                        / max(policy_dt, 1e-6)
                    )
                )
                previous_acceleration = acceleration.copy()
            if mode == "cbf":
                corrections.append(
                    float(info.get("cbf_correction_norm_normalized", 0.0))
                )
                interventions.append(float(info.get("cbf_event_intervened", False)))
                qp_failures.append(float(not bool(info.get("cbf_qp_success", True))))
            else:
                corrections.append(0.0)
                interventions.append(0.0)
                qp_failures.append(0.0)

            if terminated or truncated:
                break

        collisions_per_km = (
            1000.0 * float(collision_events) / float(total_distance_m)
            if total_distance_m > 1e-9
            else np.nan
        )
        return {
            "schema_version": PROGRESSION_SCHEMA_VERSION,
            "evaluation_kind": "post_training_complete_episodes",
            "variant": variant,
            "variant_label": VARIANT_SPECS[variant]["label"],
            "mode": mode,
            "action_source": action_source,
            "external_cbf": "ON" if mode == "cbf" else "OFF",
            "training_seed": int(training_seed),
            "episode_index": int(episode_index),
            # Keep the common name too, so generic scenario-oriented tools can
            # use this exact-episode file without a special conversion.
            "scenario_seed": int(episode_seed),
            "episode_seed": int(episode_seed),
            "timesteps": int(len(rewards)),
            "episode_return": float(np.sum(rewards)),
            "episode_length_steps": float(len(rewards)),
            "ego_collisions_per_km": float(collisions_per_km),
            "h_min": _finite_min(h_values),
            "qp_failure_rate": _finite_mean(qp_failures, default=0.0),
            "mean_abs_speed_deviation": _finite_mean(speed_errors, default=0.0),
            "mean_lat_y_error_m": _finite_mean(lateral_errors),
            "event_intervention_rate": _finite_mean(interventions, default=0.0),
            "mean_correction_norm": _finite_mean(corrections, default=0.0),
            "mean_jerk_norm": _finite_mean(jerk_norms, default=0.0),
            # Retain the former time-based value for backwards-compatible raw
            # episode files; public completion KPI aggregation is distance-based.
            "full_horizon_survival_rate": _full_horizon_survival_flag(
                episode_length_steps=len(rewards),
                collision_events=collision_events,
                env_config=env_config,
            ),
            "distance_completion_rate": _distance_completion_flag(
                total_distance_m=total_distance_m,
                collision_events=collision_events,
                env_config=env_config,
                task_distance_m=_task_distance_from_args(args, env_config),
            ),
            "task_distance_m": float(
                info.get(
                    "task_distance_m",
                    _task_distance_from_args(args, env_config),
                )
            ),
            "task_completed": bool(info.get("task_completed", False)),
            "task_timeout": bool(info.get("task_timeout", False)),
            "task_collision_terminated": bool(
                info.get("task_collision_terminated", False)
            ),
            "shadow_event_intervention_rate": _finite_mean(
                shadow_interventions, default=0.0
            ),
            "shadow_mean_correction_norm": _finite_mean(
                shadow_corrections, default=0.0
            ),
            "total_distance_m": float(total_distance_m),
            "distinct_ego_collision_events": int(collision_events),
            "collision_events_direct_qp_failure_same_step": int(
                collision_events_direct_qp_failure
            ),
            "collision_events_direct_qp_failure_or_fallback_same_step": int(
                collision_events_direct_qp_failure_or_fallback
            ),
            "collision_events_without_direct_qp_failure_same_step": int(
                collision_events - collision_events_direct_qp_failure
            ),
            "first_collision_policy_step": int(first_collision_step or 0),
            "first_collision_qp_failure_same_step": bool(
                first_collision_qp_failure_same_step
            ),
            "first_collision_qp_failure_or_fallback_same_step": bool(
                first_collision_qp_failure_or_fallback_same_step
            ),
            "collision_event_log_count": int(len(collision_event_records)),
            # The parallel writer removes this private field before writing
            # the episode-level CSV and stores it in collision_events.csv.
            "_collision_event_records": collision_event_records,
        }
    finally:
        env.close()


_POST_TRAIN_EVAL_WORKER_STATE: dict[str, Any] | None = None


def _initialize_post_train_eval_worker(
    project_root: str,
    model_path: str,
    variant: str,
    device: str,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    eval_args: argparse.Namespace,
    cbf_settings: dict[str, Any],
) -> None:
    """Create one isolated CPU evaluator for complete-episode evaluation."""

    protocol.set_stable_native_defaults()
    try:
        th.set_num_threads(1)
        th.set_num_interop_threads(1)
    except RuntimeError:
        pass

    root = Path(project_root).resolve()
    namespace = protocol.bootstrap_notebook_namespace(root)
    protocol.exec_required_notebook_cells(
        root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    namespace["DEVICE"] = str(device)
    namespace.update(copy.deepcopy(cbf_settings))
    model = load_model(variant, Path(model_path), str(device))

    global _POST_TRAIN_EVAL_WORKER_STATE
    _POST_TRAIN_EVAL_WORKER_STATE = {
        "namespace": namespace,
        "model": model,
        "variant": variant,
        "training_seed": int(eval_args.training_seed)
        if hasattr(eval_args, "training_seed")
        else 0,
        "env_config": env_config,
        "reward_config": reward_config,
        "eval_args": eval_args,
    }


def _evaluate_post_train_episode_worker(
    task: tuple[str, int, int, str, int]
) -> tuple[str, int, dict[str, Any]]:
    """Evaluate one complete episode in a worker-owned environment/model."""

    state = _POST_TRAIN_EVAL_WORKER_STATE
    if state is None:
        raise RuntimeError("post-training evaluation worker was not initialized")
    mode, episode_index, episode_seed, action_source, training_seed = task
    row = evaluate_completed_episode(
        state["namespace"],
        model=state["model"],
        variant=state["variant"],
        mode=mode,
        training_seed=int(training_seed),
        episode_index=int(episode_index),
        episode_seed=int(episode_seed),
        env_config=state["env_config"],
        reward_config=state["reward_config"],
        args=state["eval_args"],
        action_source=action_source,
    )
    return mode, int(episode_index), row


def _ordered_episode_rows(
    rows: list[dict[str, Any]], modes: tuple[str, ...]
) -> list[dict[str, Any]]:
    mode_order = {mode: index for index, mode in enumerate(modes)}
    return sorted(
        rows,
        key=lambda row: (
            mode_order.get(str(row.get("mode", "")), len(modes)),
            int(row.get("episode_index", 0)),
        ),
    )


def _evaluate_complete_episode_rows(
    namespace: dict[str, Any],
    *,
    model_path: Path,
    variant: str,
    training_seed: int,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    args: argparse.Namespace,
    modes: tuple[str, ...],
    episode_count: int,
    seed_start: int,
    action_source: str,
    progress_path: Path,
    status_path: Path,
    progress_started: float,
    progress_variant: str,
) -> list[dict[str, Any]]:
    """Evaluate complete episodes serially or with the global worker count."""

    expected_rows = int(episode_count) * len(modes)
    rows: list[dict[str, Any]] = []
    _write_episode_progress_snapshot(
        progress_path=progress_path,
        status_path=status_path,
        rows=rows,
        variant=progress_variant,
        expected_episodes=expected_rows,
        started_at=progress_started,
    )

    def record(row: dict[str, Any]) -> None:
        rows.append(row)
        _write_episode_progress_snapshot(
            progress_path=progress_path,
            status_path=status_path,
            rows=rows,
            variant=progress_variant,
            expected_episodes=expected_rows,
            started_at=progress_started,
        )
        _print_episode_progress(
            row=row,
            completed=len(rows),
            expected=expected_rows,
            variant=progress_variant,
        )

    # Keep direct/unit callers that predate the worker option on the safe
    # serial path; the CLI and notebook both provide the global default of 20.
    workers = int(getattr(args, "post_train_eval_workers", 1))
    if workers <= 1:
        model = load_model(variant, model_path, args.device)
        for mode in modes:
            for episode_index in range(int(episode_count)):
                record(
                    evaluate_completed_episode(
                        namespace,
                        model=model,
                        variant=variant,
                        mode=mode,
                        training_seed=int(training_seed),
                        episode_index=episode_index + 1,
                        episode_seed=int(seed_start) + episode_index,
                        env_config=env_config,
                        reward_config=reward_config,
                        args=args,
                        action_source=action_source,
                    )
                )
        return _ordered_episode_rows(rows, modes)

    cbf_keys = (
        "CBF_AX_BOUNDS",
        "CBF_AY_BOUNDS",
        "CBF_EPS_SIDE",
        "CBF_K0",
        "CBF_K1",
        "CBF_MAX_NEIGHBOR_CONSTRAINTS",
        "CBF_NEIGHBOR_RANGE",
        "CBF_QP_FEASIBILITY_TOL",
        "CBF_TARGET_PAIR_DY",
    )
    cbf_settings = {
        key: copy.deepcopy(namespace[key])
        for key in cbf_keys
        if key in namespace
    }
    worker_args = copy.copy(args)
    worker_args.training_seed = int(training_seed)
    tasks = [
        (mode, episode_index + 1, int(seed_start) + episode_index, action_source, int(training_seed))
        for mode in modes
        for episode_index in range(int(episode_count))
    ]
    print(
        f"[ppo-progression] complete-episode workers={workers} "
        f"evaluation_device=cpu pending={len(tasks)}",
        flush=True,
    )
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_initialize_post_train_eval_worker,
        initargs=(
            str(Path(namespace["PROJECT_ROOT"]).resolve()),
            str(model_path),
            variant,
            "cpu",
            env_config,
            reward_config,
            worker_args,
            cbf_settings,
        ),
    )
    try:
        futures = [
            executor.submit(_evaluate_post_train_episode_worker, task)
            for task in tasks
        ]
        for future in as_completed(futures):
            _mode, _episode_index, row = future.result()
            record(row)
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return _ordered_episode_rows(rows, modes)


def _post_training_eval_paths(run_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Use concise names because the current model directories are already deep."""

    root = run_dir / "pe"
    root.mkdir(parents=True, exist_ok=True)
    return root / "e.csv", root / "b.csv", root / "kpi.csv", root / "m.json"


def _write_episode_progress_snapshot(
    *,
    progress_path: Path,
    status_path: Path,
    rows: list[dict[str, Any]],
    variant: str,
    expected_episodes: int,
    state: str = "running",
    started_at: float | None = None,
) -> None:
    """Persist an evaluation row/status after every completed episode.

    The canonical 200+200 evaluation used to keep all rows in memory and
    write them only after the last episode.  That made a long, healthy run
    look frozen and discarded useful progress if the notebook output reader
    was interrupted.  Keep the same final schema, but make a small CSV
    snapshot and JSON status record visible after each episode.
    """

    progress_path.parent.mkdir(parents=True, exist_ok=True)
    public_rows = [
        {key: value for key, value in row.items() if not str(key).startswith("_")}
        for row in rows
    ]
    pd.DataFrame(public_rows).to_csv(progress_path, index=False)
    by_mode: dict[str, int] = {}
    for row in rows:
        mode = str(row.get("mode", "unknown"))
        by_mode[mode] = by_mode.get(mode, 0) + 1
    last = rows[-1] if rows else {}
    status = {
        "schema_version": PROGRESSION_SCHEMA_VERSION,
        "state": str(state),
        "variant": str(variant),
        "expected_episodes": int(expected_episodes),
        "completed_episodes": int(len(rows)),
        "completed_by_mode": by_mode,
        "last_mode": last.get("mode"),
        "last_episode_index": last.get("episode_index"),
        "last_episode_seed": last.get("episode_seed"),
        "last_episode_steps": last.get("timesteps"),
        "last_distance_m": last.get("total_distance_m"),
        "last_collision_events": last.get("distinct_ego_collision_events"),
        "last_distance_completion": last.get("distance_completion_rate"),
        "elapsed_s": (
            float(max(time.perf_counter() - started_at, 0.0))
            if started_at is not None
            else None
        ),
        "episode_metrics_path": str(progress_path.resolve()),
        "updated_at_epoch_s": float(time.time()),
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = status_path.with_suffix(status_path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    temporary.replace(status_path)


def _print_episode_progress(
    *, row: dict[str, Any], completed: int, expected: int, variant: str
) -> None:
    """Emit one flushed, human-readable line for notebook/terminal monitors."""

    deployment = "ON" if str(row.get("mode")) == "cbf" else "OFF"
    print(
        "[ppo-progression] eval episode "
        f"{completed}/{expected} variant={variant} CBF={deployment} "
        f"seed={int(row.get('episode_seed', 0))} "
        f"steps={int(row.get('timesteps', 0))} "
        f"distance={float(row.get('total_distance_m', 0.0)):.1f}m "
        f"collisions={int(row.get('distinct_ego_collision_events', 0))} "
        f"complete={bool(row.get('distance_completion_rate', False))}",
        flush=True,
    )


def _post_training_summary_geometry(episode_count: int) -> tuple[int, int]:
    """Split complete episodes into as many as ten equal pooled blocks."""

    if episode_count <= 0:
        raise ValueError("Post-training evaluation must contain at least one episode")
    block_count = min(POST_TRAIN_EVAL_SUMMARY_BLOCKS, int(episode_count))
    while int(episode_count) % block_count:
        block_count -= 1
    return block_count, int(episode_count) // block_count


def _weighted_episode_metric(group: pd.DataFrame, column: str) -> float:
    """Pool a per-episode timestep mean using its completed-step exposure."""

    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(group["timesteps"], errors="coerce").to_numpy(
        dtype=float
    )
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def summarize_post_training_episodes(
    episode_metrics: pd.DataFrame,
    *,
    env_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Pool complete-episode data before calculating the public KPI table.

    ``ten_kpi_table`` is intentionally a scenario-level summary.  Passing it
    individual episodes made it average rates such as ``collisions / km`` as
    ratios, which is invalid whenever episode exposure differs.  Here, each
    block pools the ego events, distance, and timesteps first; the table then
    reports mean +/- sample SD across the equal blocks, like the main evaluator.
    """

    episode_metrics = _ensure_distance_completion_metric(
        episode_metrics, env_config=env_config
    )
    required = {
        "variant",
        "variant_label",
        "mode",
        "training_seed",
        "episode_index",
        "episode_seed",
        "timesteps",
        "total_distance_m",
        "distinct_ego_collision_events",
        *(column for _label, column in POST_TRAIN_KPI_SPECS),
    }
    missing = sorted(required - set(episode_metrics.columns))
    if missing:
        raise KeyError(
            "Post-training episode metrics are missing required columns: "
            + ", ".join(missing)
        )
    counts = episode_metrics.groupby(["variant", "mode"], sort=False).size()
    if counts.empty or counts.nunique() != 1:
        raise ValueError(
            "Every post-training variant/mode must have the same complete "
            "episode count before KPI aggregation."
        )
    episodes_per_mode = int(counts.iloc[0])
    summary_blocks, episodes_per_block = _post_training_summary_geometry(
        episodes_per_mode
    )
    block_rows: list[dict[str, Any]] = []
    for (variant, mode), group in episode_metrics.groupby(
        ["variant", "mode"], sort=False
    ):
        group = group.sort_values("episode_index", kind="stable").reset_index(
            drop=True
        )
        if group["episode_index"].duplicated().any():
            raise ValueError(
                f"Duplicate post-training episode index for {variant} mode={mode}"
            )
        for block_index in range(summary_blocks):
            start = block_index * episodes_per_block
            block = group.iloc[start : start + episodes_per_block]
            if len(block) != episodes_per_block:
                raise RuntimeError("Incomplete post-training summary block")
            total_distance_m = float(
                pd.to_numeric(block["total_distance_m"], errors="coerce").sum()
            )
            collision_events = int(
                pd.to_numeric(
                    block["distinct_ego_collision_events"], errors="coerce"
                ).fillna(0.0).sum()
            )
            total_steps = int(
                pd.to_numeric(block["timesteps"], errors="coerce").fillna(0).sum()
            )
            row: dict[str, Any] = {
                "schema_version": PROGRESSION_SCHEMA_VERSION,
                "evaluation_kind": "post_training_pooled_episode_block",
                "variant": str(variant),
                "variant_label": VARIANT_SPECS[str(variant)]["label"],
                "mode": str(mode),
                "external_cbf": "ON" if str(mode) == "cbf" else "OFF",
                "training_seed": int(block["training_seed"].iloc[0]),
                "summary_block": int(block_index + 1),
                "scenario_seed": int(block["episode_seed"].iloc[0]),
                "episode_seed_start": int(block["episode_seed"].iloc[0]),
                "episode_seed_end": int(block["episode_seed"].iloc[-1]),
                "episodes_in_block": int(len(block)),
                "timesteps": total_steps,
                "episode_return": _finite_mean(
                    pd.to_numeric(block["episode_return"], errors="coerce")
                ),
                "episode_length_steps": _finite_mean(
                    pd.to_numeric(
                        block["episode_length_steps"], errors="coerce"
                    )
                ),
                "distance_completion_rate": _finite_mean(
                    pd.to_numeric(
                        block["distance_completion_rate"], errors="coerce"
                    )
                ),
                "ego_collisions_per_km": (
                    1000.0 * float(collision_events) / total_distance_m
                    if total_distance_m > 1e-9
                    else np.nan
                ),
                "h_min": _finite_min(
                    pd.to_numeric(block["h_min"], errors="coerce")
                ),
                "total_return": float(
                    pd.to_numeric(block["episode_return"], errors="coerce").sum()
                ),
                "total_distance_m": total_distance_m,
                "distinct_ego_collision_events": collision_events,
            }
            for column in POST_TRAIN_STEP_WEIGHTED_COLUMNS:
                row[column] = _weighted_episode_metric(block, column)
            for column in (
                "shadow_event_intervention_rate",
                "shadow_mean_correction_norm",
            ):
                if column in block:
                    row[column] = _weighted_episode_metric(block, column)
            block_rows.append(row)
    block_metrics = pd.DataFrame(block_rows)
    table = ten_kpi_table(block_metrics, kpi_specs=POST_TRAIN_KPI_SPECS)
    table.insert(0, "training_seed", int(block_metrics["training_seed"].iloc[0]))
    table.insert(
        3, "external_cbf", table["mode"].map({"raw": "OFF", "cbf": "ON"})
    )
    table["episodes_per_mode"] = episodes_per_mode
    table["summary_blocks"] = summary_blocks
    table["episodes_per_summary_block"] = episodes_per_block
    table["aggregation"] = (
        "pooled per block; Mean/SD across equal complete-episode blocks"
    )
    return block_metrics, table, {
        "episodes_per_mode": episodes_per_mode,
        "summary_blocks": summary_blocks,
        "episodes_per_summary_block": episodes_per_block,
    }


def _upsert_post_training_kpi_summary(
    output_dir: Path, table: pd.DataFrame
) -> Path:
    """Maintain one easy-to-open table while keeping per-model copies nearby."""

    summary_path = output_dir / "post_train_200ep_kpis.csv"
    if summary_path.is_file():
        existing = pd.read_csv(summary_path)
        keys = ["training_seed", "variant", "mode", "KPI"]
        if set(keys).issubset(existing.columns):
            wanted = table[keys].drop_duplicates()
            existing = existing.merge(wanted, on=keys, how="left", indicator=True)
            existing = existing.loc[existing["_merge"] == "left_only"].drop(
                columns="_merge"
            )
        else:
            existing = pd.DataFrame()
        table = pd.concat([existing, table], ignore_index=True)
    table.to_csv(summary_path, index=False)
    return summary_path


def _record_post_training_evaluation(
    run_dir: Path, manifest: dict[str, Any]
) -> None:
    """Link the exact 400-episode result to the model's run configuration."""

    config_path = run_dir / "run_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        config = {}
    config["post_training_evaluation"] = manifest
    config_path.write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )


def print_post_training_results(table: pd.DataFrame, *, variant: str) -> None:
    """Print the requested pooled 200 OFF / 200 ON KPI tables inline."""

    label = VARIANT_SPECS[variant]["label"]
    for mode in EVALUATION_MODES:
        group = table.loc[table["mode"] == mode]
        deployment = "ON" if mode == "cbf" else "OFF"
        episodes = int(group["episodes_per_mode"].iloc[0])
        blocks = int(group["summary_blocks"].iloc[0])
        per_block = int(group["episodes_per_summary_block"].iloc[0])
        print(
            "\n[ppo-progression] post-training pooled KPI table: "
            f"{label} | external CBF {deployment} | {episodes} episodes "
            f"({blocks} blocks x {per_block})",
            flush=True,
        )
        print(
            group[["KPI", "Mean", "SD", "N"]].to_string(
                index=False, float_format=lambda value: f"{value:.3f}"
            ),
            flush=True,
        )


def evaluate_post_training_model(
    namespace: dict[str, Any],
    *,
    model_path: Path,
    variant: str,
    training_seed: int,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    """Run the required paired 200-episode OFF/ON check after one model trains."""

    episode_count = int(args.post_train_eval_episodes)
    run_dir = _variant_dir(output_dir, variant, training_seed)
    print(
        "[ppo-progression] starting post-training evaluation "
        f"variant={variant} external_cbf=OFF/ON episodes_per_mode={episode_count}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    episodes_path, blocks_path, kpi_path, manifest_path = _post_training_eval_paths(
        run_dir
    )
    progress_status_path = episodes_path.with_name("progress.json")
    expected_rows = episode_count * len(EVALUATION_MODES)
    progress_started = time.perf_counter()
    rows = _evaluate_complete_episode_rows(
        namespace,
        model_path=model_path,
        variant=variant,
        training_seed=int(training_seed),
        env_config=env_config,
        reward_config=reward_config,
        args=args,
        modes=tuple(EVALUATION_MODES),
        episode_count=episode_count,
        seed_start=int(args.post_train_eval_seed_start),
        action_source="policy",
        progress_path=episodes_path,
        status_path=progress_status_path,
        progress_started=progress_started,
        progress_variant=variant,
    )
    metrics = pd.DataFrame(rows)
    if len(metrics) != expected_rows:
        raise RuntimeError(
            f"Post-training evaluation produced {len(metrics)} episodes; "
            f"expected {expected_rows}."
        )
    counts = metrics.groupby("mode", sort=False).size()
    if any(int(counts.get(mode, 0)) != episode_count for mode in EVALUATION_MODES):
        raise RuntimeError(
            "Post-training evaluation did not produce the requested paired "
            "episode count for both external CBF modes."
        )

    _write_episode_progress_snapshot(
        progress_path=episodes_path,
        status_path=progress_status_path,
        rows=rows,
        variant=variant,
        expected_episodes=expected_rows,
        state="complete",
        started_at=progress_started,
    )
    metrics.to_csv(episodes_path, index=False)
    block_metrics, table, summary_geometry = summarize_post_training_episodes(
        metrics, env_config=env_config
    )
    block_metrics.to_csv(blocks_path, index=False)
    table.to_csv(kpi_path, index=False)
    root_summary_path = _upsert_post_training_kpi_summary(output_dir, table)
    manifest = {
        "schema_version": PROGRESSION_SCHEMA_VERSION,
        "evaluation_kind": "post_training_complete_episodes",
        "model_path": str(model_path.resolve()),
        "model_sha256": protocol.file_sha256(model_path),
        **summary_geometry,
        "modes": list(EVALUATION_MODES),
        "external_cbf": {"raw": "OFF", "cbf": "ON"},
        "episode_seed_start": int(args.post_train_eval_seed_start),
        "evaluation_workers": int(getattr(args, "post_train_eval_workers", 1)),
        "episode_metrics_path": str(episodes_path.resolve()),
        "episode_progress_status_path": str(progress_status_path.resolve()),
        "pooled_block_metrics_path": str(blocks_path.resolve()),
        "kpi_table_path": str(kpi_path.resolve()),
        "study_kpi_table_path": str(root_summary_path.resolve()),
        "kpi_aggregation": (
            "Collision and timestep rates are pooled within equal blocks "
            "before Mean/SD is calculated; return and episode length are "
            "episode means; h_min is the block minimum."
        ),
        "completion_definition": (
            "distance_completion_rate=1 when capped task distance reaches "
            f"{_task_distance_from_args(args, env_config):.6g} m and "
            "distinct_ego_collision_events is zero."
        ),
        "complete": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _record_post_training_evaluation(run_dir, manifest)
    print_post_training_results(table, variant=variant)
    return table


def evaluate_raw_actor_ablation(
    namespace: dict[str, Any],
    *,
    model_path: Path,
    training_seed: int,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    """Evaluate the projected variant with its raw actor mean at deployment.

    This is intentionally separate from the ordinary OFF/ON evaluation.  The
    ordinary ``raw`` deployment still uses ``mu_safe`` for a projected policy;
    this ablation executes ``mu_raw`` with only physical action-box clipping.
    """

    variant = "ppo_cbf_projected"
    episode_count = int(args.raw_actor_eval_episodes)
    ablation_dir = (
        output_dir / "raw_actor_ablation" / variant / f"seed_{int(training_seed)}"
    )
    ablation_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[ppo-progression] starting raw-actor-mean ablation "
        f"variant={variant} external_cbf=OFF episodes={episode_count}",
        flush=True,
    )
    episodes_path = ablation_dir / "episodes.csv"
    progress_status_path = ablation_dir / "progress.json"
    progress_started = time.perf_counter()
    rows = _evaluate_complete_episode_rows(
        namespace,
        model_path=model_path,
        variant=variant,
        training_seed=int(training_seed),
        env_config=env_config,
        reward_config=reward_config,
        args=args,
        modes=("raw",),
        episode_count=episode_count,
        seed_start=int(args.raw_actor_eval_seed_start),
        action_source="raw_actor_mean",
        progress_path=episodes_path,
        status_path=progress_status_path,
        progress_started=progress_started,
        progress_variant=f"{variant}:raw_actor_mean",
    )
    metrics = pd.DataFrame(rows)
    if len(metrics) != episode_count:
        raise RuntimeError(
            "Raw-actor ablation produced "
            f"{len(metrics)} episodes; expected {episode_count}."
        )
    blocks_path = ablation_dir / "blocks.csv"
    kpi_path = ablation_dir / "kpi.csv"
    manifest_path = ablation_dir / "manifest.json"
    _write_episode_progress_snapshot(
        progress_path=episodes_path,
        status_path=progress_status_path,
        rows=rows,
        variant=variant,
        expected_episodes=episode_count,
        state="complete",
        started_at=progress_started,
    )
    metrics.to_csv(episodes_path, index=False)
    block_metrics, table, summary_geometry = summarize_post_training_episodes(
        metrics, env_config=env_config
    )
    block_metrics.to_csv(blocks_path, index=False)
    table.to_csv(kpi_path, index=False)
    manifest = {
        "schema_version": PROGRESSION_SCHEMA_VERSION,
        "evaluation_kind": "raw_actor_mean_ablation",
        "action_source": "raw_actor_mean",
        "action_semantics": (
            "Execute the projected policy's neural-network mean mu_raw; "
            "bypass mu_raw-to-mu_safe CBF projection and retain only physical "
            "action-box clipping."
        ),
        "model_path": str(model_path.resolve()),
        "model_sha256": protocol.file_sha256(model_path),
        "variant": variant,
        "training_seed": int(training_seed),
        "external_cbf": "OFF",
        "evaluation_workers": int(getattr(args, "post_train_eval_workers", 1)),
        "episode_seed_start": int(args.raw_actor_eval_seed_start),
        **summary_geometry,
        "episode_metrics_path": str(episodes_path.resolve()),
        "episode_progress_status_path": str(progress_status_path.resolve()),
        "pooled_block_metrics_path": str(blocks_path.resolve()),
        "kpi_table_path": str(kpi_path.resolve()),
        "complete": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        "\n[ppo-progression] raw-actor-mean ablation KPI table "
        f"| {episode_count} episodes | external CBF OFF",
        flush=True,
    )
    print(
        table[["KPI", "Mean", "SD", "N"]].to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        ),
        flush=True,
    )
    return table


def _saved_env_config(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Load the environment config associated with a saved post-train run."""

    for config_path in (run_dir / "run_config.json", output_dir / "study_config.json"):
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        candidate = payload.get("env_config")
        if isinstance(candidate, dict):
            return candidate
    return {}


def repair_post_training_summaries(output_dir: Path) -> pd.DataFrame:
    """Rebuild valid post-training KPI tables from already saved episode rows.

    Interrupted runs may already contain complete ``pe/e.csv`` files.  Those
    raw rows are valid; only their former generic summaries were wrong.  This
    repair never re-runs a simulator episode or changes a checkpoint.
    """

    output_dir = Path(output_dir)
    repaired_tables: list[pd.DataFrame] = []
    for episodes_path in sorted(output_dir.glob("*/seed_*/pe/e.csv")):
        run_dir = episodes_path.parents[1]
        try:
            episode_metrics = pd.read_csv(episodes_path)
            env_config = _saved_env_config(run_dir, output_dir)
            episode_metrics = _ensure_distance_completion_metric(
                episode_metrics, env_config=env_config
            )
            episode_metrics.to_csv(episodes_path, index=False)
            block_metrics, table, summary_geometry = summarize_post_training_episodes(
                episode_metrics, env_config=env_config
            )
        except (OSError, ValueError, KeyError) as error:
            print(
                "[ppo-progression] skipped incomplete post-training episode "
                f"file {episodes_path}: {error}",
                flush=True,
            )
            continue
        _episodes_path, blocks_path, kpi_path, manifest_path = _post_training_eval_paths(
            run_dir
        )
        block_metrics.to_csv(blocks_path, index=False)
        table.to_csv(kpi_path, index=False)
        model_path = run_dir / "model_final.zip"
        variant = str(episode_metrics["variant"].iloc[0])
        training_seed = int(episode_metrics["training_seed"].iloc[0])
        episode_seed_start = int(
            pd.to_numeric(episode_metrics["episode_seed"], errors="coerce").min()
        )
        manifest = {
            "schema_version": PROGRESSION_SCHEMA_VERSION,
            "evaluation_kind": "post_training_complete_episodes",
            "model_path": str(model_path.resolve()),
            "model_sha256": (
                protocol.file_sha256(model_path) if model_path.is_file() else None
            ),
            **summary_geometry,
            "modes": list(EVALUATION_MODES),
            "external_cbf": {"raw": "OFF", "cbf": "ON"},
            "episode_seed_start": episode_seed_start,
            "episode_metrics_path": str(episodes_path.resolve()),
            "pooled_block_metrics_path": str(blocks_path.resolve()),
            "kpi_table_path": str(kpi_path.resolve()),
            "study_kpi_table_path": str(
                (output_dir / "post_train_200ep_kpis.csv").resolve()
            ),
            "kpi_aggregation": (
                "Collision and timestep rates are pooled within equal blocks "
                "before Mean/SD is calculated; return and episode length are "
                "episode means; h_min is the block minimum."
            ),
        "completion_definition": (
            "distance_completion_rate=1 when capped task distance reaches "
            f"{_task_distance_from_args(args, env_config):.6g} m and "
            "distinct_ego_collision_events is zero."
        ),
            "complete": True,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _record_post_training_evaluation(run_dir, manifest)
        repaired_tables.append(table)
        print(
            "[ppo-progression] repaired post-training summary "
            f"variant={variant} seed={training_seed}",
            flush=True,
        )
    summary_path = output_dir / "post_train_200ep_kpis.csv"
    if repaired_tables:
        combined = pd.concat(repaired_tables, ignore_index=True)
        combined.to_csv(summary_path, index=False)
        return combined
    return pd.DataFrame()


def ten_kpi_table(
    metrics: pd.DataFrame,
    *,
    kpi_specs: tuple[tuple[str, str], ...] = TEN_KPI_SPECS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, mode), group in metrics.groupby(["variant", "mode"], sort=False):
        for label, column in kpi_specs:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            rows.append(
                {
                    "variant": variant,
                    "variant_label": VARIANT_SPECS[str(variant)]["label"],
                    "mode": mode,
                    "KPI": label,
                    "Mean": float(values.mean()) if len(values) else np.nan,
                    "SD": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "N": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def factorial_effects(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired 2x2 reward/projection effects within shielded PPO training."""

    required_variants = set(FILTERED_FACTORIAL_VARIANTS.values())
    if not required_variants.issubset(set(metrics["variant"].astype(str))):
        return pd.DataFrame(), pd.DataFrame()
    effect_rows: list[dict[str, Any]] = []
    index_columns = ["training_seed", "scenario_seed", "mode"]
    for keys, group in metrics.groupby(index_columns, sort=True):
        by_variant = group.set_index("variant")
        if not required_variants.issubset(set(by_variant.index.astype(str))):
            continue
        for label, column in TEN_KPI_SPECS:
            y00 = float(by_variant.loc[FILTERED_FACTORIAL_VARIANTS[(False, False)], column])
            y10 = float(by_variant.loc[FILTERED_FACTORIAL_VARIANTS[(True, False)], column])
            y01 = float(by_variant.loc[FILTERED_FACTORIAL_VARIANTS[(False, True)], column])
            y11 = float(by_variant.loc[FILTERED_FACTORIAL_VARIANTS[(True, True)], column])
            effects = {
                "reward_main": 0.5 * ((y10 - y00) + (y11 - y01)),
                "projected_actor_main": 0.5 * ((y01 - y00) + (y11 - y10)),
                "reward_x_projected_actor": y11 - y10 - y01 + y00,
            }
            for effect, value in effects.items():
                effect_rows.append(
                    {
                        **dict(zip(index_columns, keys)),
                        "KPI": label,
                        "column": column,
                        "effect": effect,
                        "value": float(value),
                    }
                )
    scenario_effects = pd.DataFrame(effect_rows)
    if scenario_effects.empty:
        return scenario_effects, pd.DataFrame()
    summary = (
        scenario_effects.groupby(["mode", "KPI", "column", "effect"], sort=False)[
            "value"
        ]
        .agg(Mean="mean", SD="std", N="count")
        .reset_index()
    )
    summary["SD"] = summary["SD"].fillna(0.0)
    return scenario_effects, summary


def print_inline_results(metrics: pd.DataFrame, table: pd.DataFrame) -> None:
    for (variant, mode), group in table.groupby(["variant", "mode"], sort=False):
        label = VARIANT_SPECS[str(variant)]["label"]
        deployment = "ON" if str(mode) == "cbf" else "OFF"
        print(
            f"\n[ppo-progression] 10-KPI final evaluation: {label} | CBF {deployment}",
            flush=True,
        )
        print(
            group[["KPI", "Mean", "SD", "N"]].to_string(
                index=False, float_format=lambda value: f"{value:.3f}"
            ),
            flush=True,
        )

    # The Level-1 result is deliberately one checkpoint under two deployments.
    nominal = metrics[metrics["variant"] == "ppo_nominal"]
    if set(nominal["mode"]) == set(EVALUATION_MODES):
        paired = nominal.pivot_table(
            index=["training_seed", "scenario_seed"],
            columns="mode",
            values=[column for _, column in TEN_KPI_SPECS],
            aggfunc="first",
        )
        print(
            "\n[ppo-progression] Level 1 uses the SAME ppo_nominal checkpoint "
            "for CBF OFF and CBF ON.",
            flush=True,
        )
        print(
            f"[ppo-progression] paired scenarios={len(paired)}",
            flush=True,
        )


def evaluate_all(
    namespace: dict[str, Any],
    *,
    model_paths: dict[tuple[int, str], Path],
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    progress_path = output_dir / "evaluation_scenarios_progress.csv"
    progress_status_path = output_dir / "evaluation_scenarios_progress.json"
    expected_rows = len(model_paths) * len(EVALUATION_MODES) * len(args.eval_seeds)
    progress_started = time.perf_counter()
    _write_episode_progress_snapshot(
        progress_path=progress_path,
        status_path=progress_status_path,
        rows=rows,
        variant="all_variants",
        expected_episodes=expected_rows,
        started_at=progress_started,
    )
    for (training_seed, variant), model_path in model_paths.items():
        model = load_model(variant, model_path, args.device)
        for mode in EVALUATION_MODES:
            for scenario_seed in args.eval_seeds:
                row = evaluate_scenario(
                    namespace,
                    model=model,
                    variant=variant,
                    mode=mode,
                    training_seed=training_seed,
                    scenario_seed=int(scenario_seed),
                    env_config=env_config,
                    reward_config=reward_config,
                    args=args,
                )
                rows.append(row)
                _write_episode_progress_snapshot(
                    progress_path=progress_path,
                    status_path=progress_status_path,
                    rows=rows,
                    variant="all_variants",
                    expected_episodes=expected_rows,
                    started_at=progress_started,
                )
                print(
                    "[ppo-progression] scenario evaluation "
                    f"{len(rows)}/{expected_rows} variant={variant} "
                    f"CBF={'ON' if mode == 'cbf' else 'OFF'} "
                    f"seed={int(scenario_seed)}",
                    flush=True,
                )
    metrics = pd.DataFrame(rows)
    _write_episode_progress_snapshot(
        progress_path=progress_path,
        status_path=progress_status_path,
        rows=rows,
        variant="all_variants",
        expected_episodes=expected_rows,
        state="complete",
        started_at=progress_started,
    )
    metrics.to_csv(output_dir / "evaluation_scenarios.csv", index=False)
    table = ten_kpi_table(metrics)
    table.to_csv(output_dir / "ten_kpi_summary.csv", index=False)
    print_inline_results(metrics, table)
    scenario_effects, effect_summary = factorial_effects(metrics)
    if not scenario_effects.empty:
        scenario_effects.to_csv(
            output_dir / "factorial_effects_scenarios.csv", index=False
        )
        effect_summary.to_csv(
            output_dir / "factorial_effects_summary.csv", index=False
        )
        print(
            "\n[ppo-progression] paired filtered-training 2x2 effects "
            "(reward x projected actor)",
            flush=True,
        )
        print(
            effect_summary.to_string(
                index=False, float_format=lambda value: f"{value:.3f}"
            ),
            flush=True,
        )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the canonical PPO-to-CBF progression."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=DEFAULT_NUM_ENVS,
        help=(
            "Parallel rollout workers. Values above one use Windows-safe "
            "spawned subprocesses; default=%(default)s."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_SEED])
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--ppo-config", choices=sorted(PPO_CONFIGS), default=DEFAULT_PPO_CONFIG)
    parser.add_argument(
        "--tensorboard-run-label",
        default=None,
        help=(
            "Optional namespace added to the durable TensorBoard directory. "
            "Use a unique label when running the same variant/seed at a new budget."
        ),
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Global transitions per PPO rollout; it must divide evenly across --n-envs.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument(
        "--collision-penalty",
        type=float,
        default=None,
        help=(
            "Override the nominal terminal ego-collision penalty in the base "
            "reward; omitted means use the notebook value."
        ),
    )
    parser.add_argument(
        "--reward-mode",
        choices=("reciprocal", "additive"),
        default=None,
        help="Select the bounded additive reward or the legacy reciprocal reward.",
    )
    parser.add_argument(
        "--speed-reward-weight",
        type=float,
        default=None,
        help="Weight for the bounded additive speed-tracking term.",
    )
    parser.add_argument(
        "--lateral-reward-weight",
        type=float,
        default=None,
        help="Weight for the bounded additive lateral-tracking term.",
    )
    parser.add_argument(
        "--lateral-y-weight",
        type=float,
        default=None,
        help=(
            "Override `wy`, the active lateral target-error cost weight in "
            "the reciprocal Karalakou reward."
        ),
    )
    parser.add_argument(
        "--expose-target-y",
        action="store_true",
        help=(
            "Expose normalized reward-derived target_y in the unused second ego "
            "observation slot without changing the base actor input shape."
        ),
    )
    parser.add_argument(
        "--remove-vehicle-dimensions",
        action="store_true",
        help=(
            "Omit vehicle length and width from every observed vehicle row; "
            "reduces the learned base observation from 42 to 30 features."
        ),
    )
    parser.add_argument(
        "--risk-penalty-weight",
        type=float,
        default=None,
        help="Weight for the bounded additive potential-field risk penalty.",
    )
    parser.add_argument(
        "--risk-potential-shaping-weight",
        type=float,
        default=None,
        help=(
            "Weight for potential-based shaping from the existing Karalakou "
            "potential field; zero disables it."
        ),
    )
    parser.add_argument(
        "--risk-potential-shaping-gamma",
        type=float,
        default=None,
        help="Discount used inside the potential-based risk-shaping transition.",
    )
    parser.add_argument(
        "--safety-potential-formulation",
        choices=("none", "compact_quadratic", "cbf_violation", "predicted_cpa", "ttc", "ttc_ellipse"),
        default=None,
        help="Direct safety formulation; ttc_ellipse logs and penalizes ellipse and TTC costs independently.",
    )
    parser.add_argument(
        "--safety-potential-weight",
        type=float,
        default=None,
        help="Weight for the direct safety-potential penalty in additive reward mode.",
    )
    parser.add_argument(
        "--safety-ellipse-weight",
        type=float,
        default=None,
        help="Independent weight for the compact elliptical safety penalty.",
    )
    parser.add_argument(
        "--safety-ttc-weight",
        type=float,
        default=None,
        help="Independent weight for the TTC safety penalty.",
    )
    parser.add_argument(
        "--safety-potential-warning-h",
        type=float,
        default=None,
        help="Positive warning-boundary h_w for the compact safety potential.",
    )
    parser.add_argument(
        "--safety-potential-eps-side",
        type=float,
        default=None,
        help="Footprint inflation buffer used to construct the pairwise ellipse axes.",
    )
    parser.add_argument(
        "--safety-cbf-alpha",
        type=float,
        default=None,
        help="Alpha gain in psi = h_dot + alpha*h for cbf_violation.",
    )
    parser.add_argument(
        "--safety-cbf-psi-scale",
        type=float,
        default=None,
        help="Normalization scale for the CBF-condition violation potential.",
    )
    parser.add_argument(
        "--safety-prediction-horizon",
        type=float,
        default=None,
        help="Constant-relative-velocity lookahead horizon in seconds for predicted_cpa.",
    )
    parser.add_argument(
        "--safety-prediction-epsilon",
        type=float,
        default=None,
        help="Positive denominator regularization for predicted_cpa.",
    )
    parser.add_argument(
        "--safety-ttc-warning-horizon",
        type=float,
        default=None,
        help="TTC warning horizon in seconds for ttc and ttc_ellipse.",
    )
    parser.add_argument(
        "--speed-reward-sigma",
        type=float,
        default=None,
        help="Speed-error scale in m/s for the additive Gaussian tracking term.",
    )
    parser.add_argument(
        "--lateral-reward-sigma",
        type=float,
        default=None,
        help="Lateral-error scale in m for the additive Gaussian tracking term.",
    )
    parser.add_argument(
        "--progress-reward-weight",
        type=float,
        default=None,
        help=(
            "Override the forward-progress reward weight in the base task "
            "reward; omitted means use the notebook value."
        ),
    )
    parser.add_argument(
        "--overtake-bonus",
        type=float,
        default=None,
        help=(
            "Override the overtaking bonus in the base task reward; omitted "
            "means use the notebook value."
        ),
    )
    parser.add_argument(
        "--collision-reward-override",
        action="store_true",
        help=(
            "On an active collision step, replace the entire reward with "
            "collision_penalty instead of adding it to the positive reward."
        ),
    )
    parser.add_argument(
        "--action-rate-penalty",
        type=float,
        default=0.0,
        help=(
            "Nominal-only penalty coefficient for the squared difference "
            "between consecutive normalized executed actions."
        ),
    )
    parser.add_argument("--lambda-delta", type=float, default=0.05)
    parser.add_argument("--lambda-intervention", type=float, default=0.10)
    parser.add_argument(
        "--lambda-mean",
        type=float,
        default=0.10,
        help="Coefficient for the differentiable projected-mean actor loss.",
    )
    parser.add_argument(
        "--lambda-detached-actor",
        type=float,
        default=0.10,
        help=(
            "Coefficient for the non-differentiable stopped-gradient hard-CBF "
            "mean-target loss; active only for detached actor-feedback variants."
        ),
    )
    parser.add_argument(
        "--lambda-sample",
        type=float,
        default=0.0,
        help="Optional fresh-sample internalization loss; default off for the primary run.",
    )
    parser.add_argument(
        "--lambda-critic",
        type=float,
        default=0.0,
        help=(
            "Legacy/custom auxiliary CBF safety-critic coefficient. All "
            "canonical study variants keep it at zero and train only the "
            "ordinary PPO reward critic."
        ),
    )
    parser.add_argument(
        "--safety-critic-gamma",
        type=float,
        default=0.99,
        help="Discount for the safety critic's future CBF-correction cost target.",
    )
    parser.add_argument(
        "--safety-critic-cost-clip",
        type=float,
        default=1.0,
        help="Positive clip for the per-step normalized squared CBF correction cost.",
    )
    parser.add_argument("--correction-epsilon", type=float, default=0.03)
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
        help="Periodic checkpoint interval; rounded up to a complete PPO rollout.",
    )
    parser.add_argument("--eval-seed-start", type=int, default=DEFAULT_EVAL_SEED_START)
    parser.add_argument("--eval-scenarios", type=int, default=DEFAULT_EVAL_SCENARIOS)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--eval-timesteps", type=int, default=DEFAULT_EVAL_TIMESTEPS)
    parser.add_argument(
        "--task-distance-m",
        type=float,
        default=DEFAULT_TASK_DISTANCE_M,
        help=(
            "Strict collision-free completion distance for every evaluation "
            "episode; default=1,000 m."
        ),
    )
    parser.add_argument(
        "--task-max-policy-steps",
        type=int,
        default=DEFAULT_TASK_MAX_POLICY_STEPS,
        help=(
            "Maximum policy decisions for one complete distance-task evaluation "
            "episode."
        ),
    )
    parser.add_argument("--ttc-cap", type=float, default=30.0)
    parser.add_argument(
        "--post-train-eval-episodes",
        type=int,
        default=DEFAULT_POST_TRAIN_EVAL_EPISODES,
        help=(
            "Completed episodes immediately after each newly trained model, "
            "for each external-CBF mode; default=%(default)s."
        ),
    )
    parser.add_argument(
        "--post-train-eval-seed-start",
        type=int,
        default=DEFAULT_POST_TRAIN_EVAL_SEED_START,
        help=(
            "First paired episode seed for the immediate OFF/ON evaluation; "
            "default=%(default)s."
        ),
    )
    parser.add_argument(
        "--post-train-eval-workers",
        type=int,
        default=DEFAULT_POST_TRAIN_EVAL_WORKERS,
        help=(
            "Independent single-threaded CPU workers for all complete-episode "
            "post-training evaluations; default=%(default)s."
        ),
    )
    parser.add_argument(
        "--skip-post-train-evaluation",
        action="store_true",
        help="Skip the immediate paired complete-episode KPI evaluation.",
    )
    parser.add_argument(
        "--post-train-evaluate-reused",
        action="store_true",
        help=(
            "Also run the immediate KPI evaluation for an exact checkpoint "
            "that was reused rather than newly trained."
        ),
    )
    parser.add_argument(
        "--raw-actor-eval",
        action="store_true",
        help=(
            "Run an additional evaluation of ppo_cbf_projected using its "
            "raw neural-network mean mu_raw, bypassing the architectural "
            "CBF mean projection while retaining physical action bounds."
        ),
    )
    parser.add_argument(
        "--raw-actor-eval-episodes",
        type=int,
        default=DEFAULT_POST_TRAIN_EVAL_EPISODES,
        help=(
            "Complete episodes for the optional raw-actor-mean ablation; "
            "default=%(default)s."
        ),
    )
    parser.add_argument(
        "--raw-actor-eval-seed-start",
        type=int,
        default=DEFAULT_POST_TRAIN_EVAL_SEED_START,
        help=(
            "First episode seed for the optional raw-actor-mean ablation; "
            "default=%(default)s."
        ),
    )
    parser.add_argument(
        "--repair-post-train-summaries",
        action="store_true",
        help=(
            "Rebuild post-training KPI tables from saved complete-episode "
            "rows without training or re-evaluating models."
        ),
    )
    parser.add_argument(
        "--skip-training",
        "--use-existing-results",
        dest="skip_training",
        action="store_true",
        help=(
            "Evaluation-only mode: require an exactly matching completed "
            "checkpoint for every requested variant."
        ),
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-counterfactual", action="store_true")
    parser.add_argument("--counterfactual-scenarios", type=int, default=2)
    parser.add_argument("--counterfactual-steps", type=int, default=400)
    parser.add_argument("--states-per-stratum", type=int, default=20)
    parser.add_argument("--stochastic-samples", type=int, default=32)
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Deliberately replace an existing checkpoint instead of reusing it.",
    )
    add_env_config_args(parser)
    parser.set_defaults(traffic_model="mtm")
    return parser.parse_args()


def main() -> int:
    protocol.set_stable_native_defaults()
    # The learner is CUDA-bound in the canonical notebook; limiting its host
    # thread pools prevents competition with the eight CPU simulator workers.
    # (``set_num_interop_threads`` can only be called once in a process.)
    try:
        th.set_num_threads(1)
        th.set_num_interop_threads(1)
    except RuntimeError:
        pass
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()
    if args.skip_training and args.force_retrain:
        raise ValueError("--skip-training cannot be combined with --force-retrain")
    if int(args.n_envs) <= 0:
        raise ValueError("--n-envs must be positive")
    unknown = [variant for variant in args.variants if variant not in VARIANT_SPECS]
    if unknown:
        raise ValueError(f"Unknown PPO progression variants: {unknown}")
    if args.raw_actor_eval and "ppo_cbf_projected" not in args.variants:
        raise ValueError(
            "--raw-actor-eval requires ppo_cbf_projected in --variants"
        )
    if args.eval_seeds is None:
        args.eval_seeds = [
            int(args.eval_seed_start) + index
            for index in range(int(args.eval_scenarios))
        ]
    if not args.eval_seeds:
        raise ValueError("At least one evaluation seed is required")
    if int(args.eval_timesteps) <= 0:
        raise ValueError("Evaluation timestep budget must be positive")
    _task_distance_from_args(args, {})
    _task_max_policy_steps_from_args(args)
    if int(args.post_train_eval_episodes) <= 0:
        raise ValueError("post-train-eval-episodes must be positive")
    if int(getattr(args, "post_train_eval_workers", 1)) <= 0:
        raise ValueError("post-train-eval-workers must be positive")
    if int(args.checkpoint_freq) <= 0:
        raise ValueError("checkpoint-freq must be positive")
    if not np.isfinite(float(args.action_rate_penalty)) or float(
        args.action_rate_penalty
    ) < 0.0:
        raise ValueError("--action-rate-penalty must be finite and non-negative")
    for name in (
        "lambda_mean",
        "lambda_detached_actor",
        "lambda_sample",
        "lambda_critic",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    if not np.isfinite(float(args.safety_critic_gamma)) or not 0.0 <= float(
        args.safety_critic_gamma
    ) <= 1.0:
        raise ValueError("--safety-critic-gamma must be finite and lie in [0, 1]")
    if not np.isfinite(float(args.safety_critic_cost_clip)) or float(
        args.safety_critic_cost_clip
    ) <= 0.0:
        raise ValueError("--safety-critic-cost-clip must be finite and positive")
    for name in (
        "speed_reward_weight",
        "lateral_reward_weight",
        "lateral_y_weight",
        "risk_penalty_weight",
        "risk_potential_shaping_weight",
        "safety_potential_weight",
        "safety_ellipse_weight",
        "safety_ttc_weight",
        "speed_reward_sigma",
        "lateral_reward_sigma",
    ):
        value = getattr(args, name)
        if value is not None and (
            not np.isfinite(float(value)) or float(value) < 0.0
        ):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    for name in (
        "counterfactual_scenarios",
        "counterfactual_steps",
        "states_per_stratum",
        "stochastic_samples",
        "raw_actor_eval_episodes",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    project_root = protocol.find_project_root(args.project_root or Path.cwd())
    output_dir = (
        args.output_dir
        or project_root / "artifacts" / "ppo_cbf_progression_parallel_v3"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.repair_post_train_summaries:
        repaired = repair_post_training_summaries(output_dir)
        print(
            "[ppo-progression] repaired post-training KPI rows="
            f"{len(repaired)} output={output_dir}",
            flush=True,
        )
        return 0

    namespace = protocol.bootstrap_notebook_namespace(project_root)
    protocol.exec_required_notebook_cells(
        project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    namespace["DEVICE"] = args.device
    env_config = env_config_from_args(args, namespace["ENV_CONFIG"])
    if active_traffic_model(env_config) == "mtm":
        _deep_set_defaults(
            env_config, copy.deepcopy(MTM_CONGESTED_UNCERTAIN_UPDATES)
        )
    if args.remove_vehicle_dimensions:
        env_config["observation_include_vehicle_dimensions"] = False
    if not bool(env_config.get("terminate_on_collision", False)):
        raise RuntimeError("PPO progression requires terminate_on_collision=True")
    reward_config = protocol.make_base_reward_config(namespace)
    if args.collision_penalty is not None:
        if not np.isfinite(float(args.collision_penalty)):
            raise ValueError("--collision-penalty must be finite")
        reward_config["collision_penalty"] = float(args.collision_penalty)
    if args.reward_mode is not None:
        reward_config["reward_mode"] = str(args.reward_mode)
    if args.progress_reward_weight is not None:
        if not np.isfinite(float(args.progress_reward_weight)):
            raise ValueError("--progress-reward-weight must be finite")
        reward_config["progress_reward_weight"] = float(args.progress_reward_weight)
    if args.overtake_bonus is not None:
        if not np.isfinite(float(args.overtake_bonus)):
            raise ValueError("--overtake-bonus must be finite")
        reward_config["overtake_bonus"] = float(args.overtake_bonus)
    if args.lateral_y_weight is not None:
        reward_config["wy"] = float(args.lateral_y_weight)
    if args.expose_target_y:
        reward_config["expose_target_y"] = True
    if args.collision_reward_override:
        reward_config["collision_reward_override"] = True
    for name in (
        "speed_reward_weight",
        "lateral_reward_weight",
        "risk_penalty_weight",
        "risk_potential_shaping_weight",
        "safety_potential_weight",
        "safety_ellipse_weight",
        "safety_ttc_weight",
        "speed_reward_sigma",
        "lateral_reward_sigma",
    ):
        value = getattr(args, name)
        if value is not None:
            reward_config[name] = float(value)
    if args.safety_potential_formulation is not None:
        reward_config["safety_potential_formulation"] = str(args.safety_potential_formulation)
    if args.safety_potential_warning_h is not None:
        if not np.isfinite(float(args.safety_potential_warning_h)) or float(args.safety_potential_warning_h) <= 0.0:
            raise ValueError("--safety-potential-warning-h must be finite and positive")
        reward_config["safety_potential_warning_h"] = float(args.safety_potential_warning_h)
    if args.safety_potential_eps_side is not None:
        if not np.isfinite(float(args.safety_potential_eps_side)) or float(args.safety_potential_eps_side) < 0.0:
            raise ValueError("--safety-potential-eps-side must be finite and non-negative")
        reward_config["safety_potential_eps_side"] = float(args.safety_potential_eps_side)
    if args.safety_cbf_alpha is not None:
        if not np.isfinite(float(args.safety_cbf_alpha)) or float(args.safety_cbf_alpha) < 0.0:
            raise ValueError("--safety-cbf-alpha must be finite and non-negative")
        reward_config["safety_cbf_alpha"] = float(args.safety_cbf_alpha)
    if args.safety_cbf_psi_scale is not None:
        if not np.isfinite(float(args.safety_cbf_psi_scale)) or float(args.safety_cbf_psi_scale) <= 0.0:
            raise ValueError("--safety-cbf-psi-scale must be finite and positive")
        reward_config["safety_cbf_psi_scale"] = float(args.safety_cbf_psi_scale)
    if args.safety_prediction_horizon is not None:
        if not np.isfinite(float(args.safety_prediction_horizon)) or float(args.safety_prediction_horizon) < 0.0:
            raise ValueError("--safety-prediction-horizon must be finite and non-negative")
        reward_config["safety_prediction_horizon"] = float(args.safety_prediction_horizon)
    if args.safety_prediction_epsilon is not None:
        if not np.isfinite(float(args.safety_prediction_epsilon)) or float(args.safety_prediction_epsilon) <= 0.0:
            raise ValueError("--safety-prediction-epsilon must be finite and positive")
        reward_config["safety_prediction_epsilon"] = float(args.safety_prediction_epsilon)
    if args.safety_ttc_warning_horizon is not None:
        if not np.isfinite(float(args.safety_ttc_warning_horizon)) or float(args.safety_ttc_warning_horizon) <= 0.0:
            raise ValueError("--safety-ttc-warning-horizon must be finite and positive")
        reward_config["safety_ttc_warning_horizon"] = float(args.safety_ttc_warning_horizon)
    if args.risk_potential_shaping_gamma is not None:
        if not np.isfinite(float(args.risk_potential_shaping_gamma)) or not 0.0 <= float(args.risk_potential_shaping_gamma) <= 1.0:
            raise ValueError("--risk-potential-shaping-gamma must lie in [0, 1]")
        reward_config["risk_potential_shaping_gamma"] = float(args.risk_potential_shaping_gamma)
    config = resolved_ppo_config(args)
    print(
        "[ppo-progression] starting",
        {
            "variants": args.variants,
            "seeds": args.seeds,
            "timesteps": int(args.timesteps),
            "collision_penalty": float(reward_config["collision_penalty"]),
            "reward_mode": str(reward_config.get("reward_mode", "reciprocal")),
            "progress_reward_weight": float(
                reward_config["progress_reward_weight"]
            ),
            "overtake_bonus": float(reward_config["overtake_bonus"]),
            "collision_reward_override": bool(
                reward_config.get("collision_reward_override", False)
            ),
            "lateral_y_weight": float(reward_config["wy"]),
            "expose_target_y": bool(reward_config.get("expose_target_y", False)),
            "remove_vehicle_dimensions": bool(args.remove_vehicle_dimensions),
            "base_observation_dim": _base_observation_dim(env_config),
            "observation_variant": (
                "target_y_plus_previous_executed_action"
                if bool(env_config.get("ppo_append_previous_action", False))
                else "base_vehicle_table"
            ),
            "physics_hz": float(env_config.get("simulation_frequency", np.nan)),
            "policy_hz": float(env_config.get("policy_frequency", np.nan)),
            "cbf_substeps_per_policy_action": max(
                1,
                int(
                    round(
                        float(env_config.get("simulation_frequency", 1.0))
                        / max(
                            float(env_config.get("policy_frequency", 1.0)),
                            1e-9,
                        )
                    )
                ),
            ),
            "cbf_substep_filtering": bool(
                env_config.get("cbf_substep_filtering", False)
            ),
            "speed_reward_weight": float(
                reward_config.get("speed_reward_weight", 0.25)
            ),
            "lateral_reward_weight": float(
                reward_config.get("lateral_reward_weight", 0.25)
            ),
            "risk_penalty_weight": float(
                reward_config.get("risk_penalty_weight", 0.5)
            ),
            "risk_potential_shaping_weight": float(
                reward_config.get("risk_potential_shaping_weight", 0.0)
            ),
            "risk_potential_shaping_gamma": float(
                reward_config.get("risk_potential_shaping_gamma", 0.99)
            ),
            "safety_potential_formulation": str(
                reward_config.get("safety_potential_formulation", "none")
            ),
            "safety_potential_weight": float(
                reward_config.get("safety_potential_weight", 0.0)
            ),
            "safety_ellipse_weight": float(
                reward_config.get("safety_ellipse_weight", 0.0)
            ),
            "safety_ttc_weight": float(
                reward_config.get("safety_ttc_weight", 0.0)
            ),
            "safety_potential_warning_h": float(
                reward_config.get("safety_potential_warning_h", 4.0)
            ),
            "speed_reward_sigma": float(
                reward_config.get("speed_reward_sigma", 4.0)
            ),
            "lateral_reward_sigma": float(
                reward_config.get("lateral_reward_sigma", 1.0)
            ),
            "action_rate_penalty": float(args.action_rate_penalty),
            "ppo_config": config,
            "collection_topology": training_topology(args),
            "traffic_model": active_traffic_model(env_config),
            "post_training_evaluation": {
                "enabled": not bool(args.skip_post_train_evaluation),
                "episodes_per_external_cbf_mode": int(
                    args.post_train_eval_episodes
                ),
                "workers": int(getattr(args, "post_train_eval_workers", 1)),
                "modes": list(EVALUATION_MODES),
                "evaluate_reused": bool(args.post_train_evaluate_reused),
            },
            "checkpoint_policy": (
                "evaluation-only exact reuse"
                if args.skip_training
                else (
                    "force retrain"
                    if args.force_retrain
                    else "ensure checkpoint (reuse exact, train missing)"
                )
            ),
            "action_space": {
                "ax": namespace["CBF_AX_BOUNDS"],
                "ay": namespace["CBF_AY_BOUNDS"],
            },
        },
        flush=True,
    )

    model_paths: dict[tuple[int, str], Path] = {}
    training_results: dict[tuple[int, str], VariantTrainingResult] = {}
    post_training_evaluated: dict[tuple[int, str], bool] = {}
    for training_seed in [int(seed) for seed in args.seeds]:
        for variant in args.variants:
            expected_signature = training_signature(
                namespace,
                variant=variant,
                training_seed=training_seed,
                env_config=env_config,
                reward_config=reward_config,
                args=args,
            )
            if args.skip_training:
                path = resolve_existing_variant_checkpoint(
                    output_dir,
                    variant=variant,
                    training_seed=training_seed,
                    expected_signature=expected_signature,
                )
                print(
                    f"[ppo-progression] evaluation-only exact reuse {variant}: "
                    f"{path}",
                    flush=True,
                )
                tensorboard_log_dir = restore_legacy_tensorboard_artifacts(
                    namespace,
                    run_dir=_variant_dir(output_dir, variant, training_seed),
                    variant=variant,
                    training_seed=training_seed,
                    tensorboard_run_label=getattr(
                        args, "tensorboard_run_label", None
                    ),
                )
                result = VariantTrainingResult(
                    model_path=path,
                    trained=False,
                    tensorboard_log_dir=tensorboard_log_dir,
                )
            else:
                result = train_variant(
                    namespace,
                    variant=variant,
                    training_seed=training_seed,
                    env_config=env_config,
                    reward_config=reward_config,
                    args=args,
                    output_dir=output_dir,
                )
            path = result.model_path
            model_paths[(training_seed, variant)] = path
            training_results[(training_seed, variant)] = result
            should_post_evaluate = (
                not args.skip_post_train_evaluation
                and (result.trained or args.post_train_evaluate_reused)
            )
            if should_post_evaluate:
                evaluate_post_training_model(
                    namespace,
                    model_path=path,
                    variant=variant,
                    training_seed=training_seed,
                    env_config=env_config,
                    reward_config=reward_config,
                    args=args,
                    output_dir=output_dir,
                )
            post_training_evaluated[(training_seed, variant)] = bool(
                should_post_evaluate
            )

    manifest = pd.DataFrame(
        [
            {
                "training_seed": seed,
                "variant": variant,
                "variant_label": VARIANT_SPECS[variant]["label"],
                "model_path": str(path),
                "tensorboard_log_dir": (
                    str(training_results[(seed, variant)].tensorboard_log_dir)
                    if training_results[(seed, variant)].tensorboard_log_dir
                    else None
                ),
                "trained_this_invocation": bool(
                    training_results[(seed, variant)].trained
                ),
                "post_train_200ep_evaluated": bool(
                    post_training_evaluated[(seed, variant)]
                ),
                "checkpoint_policy": (
                    "evaluation_only_reuse"
                    if args.skip_training
                    else (
                        "force_retrain"
                        if args.force_retrain
                        else "ensure_checkpoint"
                    )
                ),
            }
            for (seed, variant), path in model_paths.items()
        ]
    )
    manifest.to_csv(output_dir / "model_manifest.csv", index=False)
    study_config = {
        "schema_version": PROGRESSION_SCHEMA_VERSION,
        "variants": list(args.variants),
        "variant_specs": {
            variant: VARIANT_SPECS[variant] for variant in args.variants
        },
        "filtered_factorial_variants": {
            f"reward_{int(reward)}_projected_{int(projected)}": variant
            for (reward, projected), variant in FILTERED_FACTORIAL_VARIANTS.items()
            if variant in args.variants
        },
        "training_seeds": list(map(int, args.seeds)),
        "timesteps": int(args.timesteps),
        "training_mode": (
            "evaluation_only" if args.skip_training else "ensure_checkpoint"
        ),
        "force_retrain": bool(args.force_retrain),
        "ppo_config_name": str(args.ppo_config),
        "tensorboard_run_label": str(
            getattr(args, "tensorboard_run_label", None) or ""
        ),
        "ppo_config": resolved_ppo_config(args),
        "collection_topology": training_topology(args),
        "traffic_model": active_traffic_model(env_config),
        "common_physical_action_bounds": {
            "ax": list(namespace["CBF_AX_BOUNDS"]),
            "ay": list(namespace["CBF_AY_BOUNDS"]),
        },
        "lambda_delta": float(args.lambda_delta),
        "lambda_intervention": float(args.lambda_intervention),
        "lambda_mean": float(args.lambda_mean),
        "lambda_detached_actor": float(args.lambda_detached_actor),
        "lambda_sample": float(args.lambda_sample),
        "lambda_critic": float(args.lambda_critic),
        "safety_critic_gamma": float(args.safety_critic_gamma),
        "safety_critic_cost_clip": float(args.safety_critic_cost_clip),
        "action_rate_penalty": float(args.action_rate_penalty),
        "correction_epsilon": float(args.correction_epsilon),
        "collision_penalty": float(reward_config["collision_penalty"]),
        "reward_mode": str(reward_config.get("reward_mode", "reciprocal")),
        "progress_reward_weight": float(
            reward_config["progress_reward_weight"]
        ),
        "overtake_bonus": float(reward_config["overtake_bonus"]),
        "collision_reward_override": bool(
            reward_config.get("collision_reward_override", False)
        ),
            "lateral_y_weight": float(reward_config["wy"]),
            "remove_vehicle_dimensions": bool(args.remove_vehicle_dimensions),
            "base_observation_dim": _base_observation_dim(env_config),
        "expose_target_y": bool(reward_config.get("expose_target_y", False)),
        "speed_reward_weight": float(
            reward_config.get("speed_reward_weight", 0.25)
        ),
        "lateral_reward_weight": float(
            reward_config.get("lateral_reward_weight", 0.25)
        ),
        "risk_penalty_weight": float(
            reward_config.get("risk_penalty_weight", 0.5)
        ),
        "risk_potential_shaping_weight": float(
            reward_config.get("risk_potential_shaping_weight", 0.0)
        ),
        "risk_potential_shaping_gamma": float(
            reward_config.get("risk_potential_shaping_gamma", 0.99)
        ),
        "safety_potential_formulation": str(
            reward_config.get("safety_potential_formulation", "none")
        ),
        "safety_potential_weight": float(
            reward_config.get("safety_potential_weight", 0.0)
        ),
        "safety_ellipse_weight": float(
            reward_config.get("safety_ellipse_weight", 0.0)
        ),
        "safety_ttc_weight": float(
            reward_config.get("safety_ttc_weight", 0.0)
        ),
        "safety_potential_warning_h": float(
            reward_config.get("safety_potential_warning_h", 4.0)
        ),
        "safety_potential_eps_side": float(
            reward_config.get("safety_potential_eps_side", 0.10)
        ),
        "safety_cbf_alpha": float(
            reward_config.get("safety_cbf_alpha", 1.0)
        ),
        "safety_cbf_psi_scale": float(
            reward_config.get("safety_cbf_psi_scale", 1.0)
        ),
        "safety_prediction_horizon": float(
            reward_config.get("safety_prediction_horizon", 2.0)
        ),
        "safety_prediction_epsilon": float(
            reward_config.get("safety_prediction_epsilon", 1e-6)
        ),
        "safety_ttc_warning_horizon": float(
            reward_config.get("safety_ttc_warning_horizon", 3.0)
        ),
        "speed_reward_sigma": float(
            reward_config.get("speed_reward_sigma", 4.0)
        ),
        "lateral_reward_sigma": float(
            reward_config.get("lateral_reward_sigma", 1.0)
        ),
        "evaluation_seeds": list(map(int, args.eval_seeds)),
        "evaluation_timesteps": int(args.eval_timesteps),
        "evaluation_task_distance_m": float(_task_distance_from_args(args, env_config)),
        "evaluation_task_max_policy_steps": int(
            _task_max_policy_steps_from_args(args)
        ),
        "post_training_evaluation": {
            "enabled": not bool(args.skip_post_train_evaluation),
            "episodes_per_external_cbf_mode": int(args.post_train_eval_episodes),
            "workers": int(getattr(args, "post_train_eval_workers", 1)),
            "modes": list(EVALUATION_MODES),
            "external_cbf": {"raw": "OFF", "cbf": "ON"},
            "episode_seed_start": int(args.post_train_eval_seed_start),
            "evaluate_reused": bool(args.post_train_evaluate_reused),
        },
        "raw_actor_mean_ablation": {
            "enabled": bool(args.raw_actor_eval),
            "variant": "ppo_cbf_projected",
            "action_source": "raw_actor_mean",
            "external_cbf": "OFF",
            "workers": int(getattr(args, "post_train_eval_workers", 1)),
            "episodes": int(args.raw_actor_eval_episodes),
            "episode_seed_start": int(args.raw_actor_eval_seed_start),
        },
        "counterfactual_enabled": bool(
            not args.skip_evaluation and not args.skip_counterfactual
        ),
        "env_config": env_config,
        "reward_config": reward_config,
    }
    (output_dir / "study_config.json").write_text(
        json.dumps(study_config, indent=2, default=str), encoding="utf-8"
    )
    if not args.skip_evaluation:
        evaluate_all(
            namespace,
            model_paths=model_paths,
            env_config=env_config,
            reward_config=reward_config,
            args=args,
            output_dir=output_dir,
        )
        if not args.skip_counterfactual:
            counterfactual_seeds = list(map(int, args.eval_seeds))[
                : int(args.counterfactual_scenarios)
            ]
            while len(counterfactual_seeds) < int(args.counterfactual_scenarios):
                counterfactual_seeds.append(
                    int(args.eval_seed_start) + len(counterfactual_seeds)
                )
            run_counterfactual_analysis(
                namespace,
                model_paths,
                load_model=load_model,
                make_env=make_evaluation_env,
                device=str(args.device),
                env_config=env_config,
                reward_config=reward_config,
                correction_epsilon=float(args.correction_epsilon),
                output_dir=output_dir / "counterfactuals",
                scenario_seeds=counterfactual_seeds,
                steps_per_scenario=int(args.counterfactual_steps),
                states_per_stratum=int(args.states_per_stratum),
                stochastic_samples=int(args.stochastic_samples),
                neighbor_range=float(namespace["CBF_NEIGHBOR_RANGE"]),
                eps_side=float(namespace["CBF_EPS_SIDE"]),
                k0=float(namespace["CBF_K0"]),
                k1=float(namespace["CBF_K1"]),
                ttc_cap=float(args.ttc_cap),
                seed=int(args.eval_seed_start) + 777,
            )
    if args.raw_actor_eval:
        for training_seed in [int(seed) for seed in args.seeds]:
            model_path = model_paths[(training_seed, "ppo_cbf_projected")]
            evaluate_raw_actor_ablation(
                namespace,
                model_path=model_path,
                training_seed=training_seed,
                env_config=env_config,
                reward_config=reward_config,
                args=args,
                output_dir=output_dir,
            )
    print(f"[ppo-progression] complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
