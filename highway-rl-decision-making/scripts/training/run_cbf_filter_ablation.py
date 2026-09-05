"""Controlled CBF-filter internalization ablation.

This runner uses the lane-free notebook only as the source of the simulator,
reward wrapper, and CBF-QP implementation.  It deliberately keeps the four
learned-policy variants identical except for the treatment named in the study:

    A: nominal DDPG (no CBF while training; contextual control)
    B: CBF-filtered training, reward off, actor loss off
    C: CBF-filtered training, reward on, actor loss off
    D: CBF-filtered training, reward off, actor loss on
    E: CBF-filtered training, reward on, actor loss on

B--E are the pre-registered 2x2 factorial cells.  A is not part of the
factorial contrast; it isolates the effect of collecting filtered experience.

Evaluation is paired by reset seed and uses a fixed timestep budget.  The simulator's
traffic reacts to the ego vehicle, so the reset state is shared, not the future
traffic trajectory.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import importlib.metadata
import json
import os
import pickle
import random
import shutil
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Union

import cloudpickle
import gymnasium as gym
import numpy as np
import pandas as pd
import torch as th
import stable_baselines3 as sb3
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, VectorizedActionNoise
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from scripts.training.cbf_lambda_event_bc_pilot_sweep import (
    find_project_root,
    set_stable_native_defaults,
)
from scripts.common.guided_cbf_minimal import GuidedCBFDDPG, install_minimal_guided_cbf
from scripts.common.laneless_script_config import active_traffic_model, add_env_config_args, env_config_from_args
from scripts.training.train_safety_potential_variants import MTM_CONGESTED_UNCERTAIN_UPDATES, deep_update


warnings.filterwarnings("ignore", message="OSQP exited.*")

PIPELINE_SCHEMA_VERSION = 4
CHECKPOINT_PAYLOADS = {
    "model": "model.zip",
    "replay_buffer": "replay.pkl",
    "base_environment": "env.pkl",
    "pipeline_state": "state.pkl",
    "vecnormalize": "vec.pkl",
}
VARIANTS = ("a_nominal", "b_filtered", "c_reward", "d_loss", "e_reward_actor")
FACTORIAL_VARIANTS = {
    (False, False): "b_filtered",
    (True, False): "c_reward",
    (False, True): "d_loss",
    (True, True): "e_reward_actor",
}
RANDOM_VARIANT = "f_random"
MODES = ("raw", "cbf")
DEFAULT_TRAINING_SEEDS = (307, 1307, 2307)
COMPARISONS = {
    "runtime_filter_a": ("a_nominal", "cbf", "a_nominal", "raw"),
    "filtered_experience": ("b_filtered", "raw", "a_nominal", "raw"),
    "reward_effect_loss_off": ("c_reward", "raw", "b_filtered", "raw"),
    "loss_effect_reward_off": ("d_loss", "raw", "b_filtered", "raw"),
    "reward_effect_loss_on": ("e_reward_actor", "raw", "d_loss", "raw"),
    "loss_effect_reward_on": ("e_reward_actor", "raw", "c_reward", "raw"),
    "runtime_filter_b": ("b_filtered", "cbf", "b_filtered", "raw"),
    "runtime_filter_c": ("c_reward", "cbf", "c_reward", "raw"),
    "runtime_filter_d": ("d_loss", "cbf", "d_loss", "raw"),
    "runtime_filter_e": ("e_reward_actor", "cbf", "e_reward_actor", "raw"),
    "actor_vs_random_with_cbf": ("e_reward_actor", "cbf", RANDOM_VARIANT, "cbf"),
}

NOTEBOOK_SETUP_MARKERS = (
    "class KaralakouRewardWrapper",
    "ENV_CONFIG = {",
    "class LaneFreeObservationNormalizationWrapper",
    "from qpsolvers import solve_qp",
    "CBF_AX_BOUNDS =",
    "def _lane_free_base",
    "class SafetyFilteredAccelerationWrapper",
    "# Tuned DDPG-CBF shield overrides",
)

TRAINING_MONITOR_INFO_KEYS = (
    "pipeline_episode_distance_m",
    "pipeline_episode_distinct_ego_collision_events",
    "pipeline_episode_ego_collision_active_timesteps",
    "pipeline_episode_collision_transition_return",
    "pipeline_episode_collision_transition_timesteps",
    "pipeline_episode_post_collision_return",
    "pipeline_episode_post_collision_timesteps",
    "pipeline_episode_time_to_first_collision_s",
    "pipeline_episode_distance_to_first_collision_m",
    "pipeline_episode_action_saturation_mean",
)


def _jsonable(value: Any) -> Any:
    """Convert configuration values to a stable, finite JSON representation."""

    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"Non-finite configuration value: {value!r}")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def canonical_config_hash(payload: dict[str, Any]) -> str:
    normalized = _jsonable(payload)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qualified_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__name__}"


def _class_qualified_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


def _space_rng_state(space: gym.Space) -> Any:
    generator = getattr(space, "_np_random", None)
    if generator is None:
        return None
    return copy.deepcopy(generator.bit_generator.state)


def _restore_space_rng_state(space: gym.Space, state: Any) -> None:
    if state is None:
        return
    generator = getattr(space, "_np_random", None)
    if generator is None:
        space.seed(0)
        generator = getattr(space, "_np_random", None)
    if generator is None:
        raise RuntimeError(f"Cannot restore RNG for space {_qualified_name(space)}")
    generator.bit_generator.state = copy.deepcopy(state)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator > 0.0:
        return float(numerator / denominator)
    if numerator > 0.0:
        return float(np.inf)
    return float(np.nan)


def _distance_per_collision(distance_m: float, collisions: float) -> float:
    return _ratio(float(distance_m), float(collisions))


def _distance_per_collision_exposure_bound(distance_m: float, collisions: float) -> float:
    """Finite observed value, or the driven-distance lower bound when censored."""

    if float(collisions) > 0.0:
        return float(distance_m) / float(collisions)
    return float(max(float(distance_m), 0.0))


def _collisions_per_km(collisions: float, distance_m: float) -> float:
    return _ratio(1000.0 * float(collisions), float(distance_m))


def _step_path_distance(env: gym.Env, previous_position: np.ndarray, current_position: np.ndarray) -> float:
    """Euclidean path length for one transition, with longitudinal wraparound."""

    base = env.unwrapped
    previous = np.asarray(previous_position, dtype=float).reshape(-1)
    current = np.asarray(current_position, dtype=float).reshape(-1)
    if previous.size < 2 or current.size < 2 or not np.all(np.isfinite(previous[:2] + current[:2])):
        return 0.0
    if hasattr(base, "_signed_distance"):
        dx = float(base._signed_distance(float(previous[0]), float(current[0])))
    else:
        dx = float(current[0] - previous[0])
    dy = float(current[1] - previous[1])
    return float(np.hypot(dx, dy))


def _policy_dt(env: gym.Env) -> float:
    config = env.unwrapped.config
    dt = float(config.get("dt", 1.0 / max(float(config.get("simulation_frequency", 1.0)), 1e-6)))
    frames = max(
        1,
        int(round(float(config.get("simulation_frequency", 1.0)) / max(float(config.get("policy_frequency", 1.0)), 1e-6))),
    )
    return float(dt * frames)


class ProtocolMetricsWrapper(gym.Wrapper):
    """Attach exact event, exposure, reward-boundary, and saturation metrics.

    The base simulator's ``ego_collision_events`` is authoritative.  Active
    contact is recorded separately and is never promoted to a new event.
    """

    def __init__(self, env: gym.Env, saturation_tolerance: float = 1e-3) -> None:
        super().__init__(env)
        self.saturation_tolerance = float(saturation_tolerance)
        self.reset_calls_total = 0
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self._episode_steps = 0
        self._episode_return = 0.0
        self._episode_distance_m = 0.0
        self._episode_distinct_ego_collision_events = 0
        self._episode_ego_collision_active_timesteps = 0
        self._episode_collision_transition_return = 0.0
        self._episode_collision_transition_timesteps = 0
        self._episode_post_collision_return = 0.0
        self._episode_post_collision_timesteps = 0
        self._episode_action_saturation_sum = 0.0
        self._episode_first_collision_step: Optional[int] = None
        self._episode_time_to_first_collision_s = np.nan
        self._episode_distance_to_first_collision_m = np.nan
        self._collision_seen_without_reset = False
        self._previous_position = np.full(2, np.nan, dtype=float)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reset_calls_total += 1
        self._reset_episode_state()
        self._previous_position = np.asarray(self.unwrapped.vehicle.position[:2], dtype=float).copy()
        info = dict(info)
        info["pipeline_reset_calls_total"] = int(self.reset_calls_total)
        return obs, info

    def _action_saturation(self, action: Any) -> float:
        array = np.asarray(action, dtype=float).reshape(-1)
        low = np.asarray(self.action_space.low, dtype=float).reshape(-1)[: array.size]
        high = np.asarray(self.action_space.high, dtype=float).reshape(-1)[: array.size]
        tolerance = self.saturation_tolerance * np.maximum(high - low, 1.0)
        saturated = (np.abs(array - low) <= tolerance) | (np.abs(array - high) <= tolerance)
        return float(np.mean(saturated)) if saturated.size else 0.0

    def step(self, action):
        post_collision_before_step = bool(self._collision_seen_without_reset)
        action_saturation = self._action_saturation(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        current_position = np.asarray(self.unwrapped.vehicle.position[:2], dtype=float).copy()
        distance_step_m = _step_path_distance(self, self._previous_position, current_position)
        self._previous_position = current_position
        distinct_events = max(int(info.get("ego_collision_events", 0)), 0)
        active_collision = bool(info.get("ego_collision", False))
        collision_transition = bool(distinct_events > 0 or (active_collision and not post_collision_before_step))

        self._episode_steps += 1
        self._episode_return += float(reward)
        self._episode_distance_m += distance_step_m
        self._episode_distinct_ego_collision_events += distinct_events
        self._episode_ego_collision_active_timesteps += int(active_collision)
        self._episode_action_saturation_sum += action_saturation
        if collision_transition:
            self._episode_collision_transition_return += float(reward)
            self._episode_collision_transition_timesteps += 1
            if self._episode_first_collision_step is None:
                self._episode_first_collision_step = int(self._episode_steps)
                self._episode_time_to_first_collision_s = float(self._episode_steps * _policy_dt(self))
                self._episode_distance_to_first_collision_m = float(self._episode_distance_m)
        elif post_collision_before_step:
            self._episode_post_collision_return += float(reward)
            self._episode_post_collision_timesteps += 1

        self._collision_seen_without_reset = bool(post_collision_before_step or active_collision or distinct_events > 0)
        collision_survived = bool(self._collision_seen_without_reset and not (terminated or truncated))
        info.update(
            {
                "pipeline_distance_step_m": float(distance_step_m),
                "pipeline_distinct_ego_collision_events": int(distinct_events),
                "pipeline_ego_collision_active_timestep": int(active_collision),
                "pipeline_distinct_all_pair_collision_events": max(int(info.get("collisions", 0)), 0),
                "pipeline_active_collision_pairs": max(int(info.get("active_collisions", 0)), 0),
                "pipeline_collision_transition": int(collision_transition),
                "pipeline_post_collision_timestep": int(post_collision_before_step and not collision_transition),
                "pipeline_collision_survived_without_reset": int(collision_survived),
                "pipeline_action_saturation": float(action_saturation),
            }
        )

        if terminated or truncated:
            info.update(
                {
                    "pipeline_episode_distance_m": float(self._episode_distance_m),
                    "pipeline_episode_distinct_ego_collision_events": int(
                        self._episode_distinct_ego_collision_events
                    ),
                    "pipeline_episode_ego_collision_active_timesteps": int(
                        self._episode_ego_collision_active_timesteps
                    ),
                    "pipeline_episode_collision_transition_return": float(
                        self._episode_collision_transition_return
                    ),
                    "pipeline_episode_collision_transition_timesteps": int(
                        self._episode_collision_transition_timesteps
                    ),
                    "pipeline_episode_post_collision_return": float(self._episode_post_collision_return),
                    "pipeline_episode_post_collision_timesteps": int(self._episode_post_collision_timesteps),
                    "pipeline_episode_time_to_first_collision_s": float(
                        self._episode_time_to_first_collision_s
                    ),
                    "pipeline_episode_distance_to_first_collision_m": float(
                        self._episode_distance_to_first_collision_m
                    ),
                    "pipeline_episode_action_saturation_mean": float(
                        self._episode_action_saturation_sum / max(self._episode_steps, 1)
                    ),
                    "pipeline_episode_return_per_timestep": float(
                        self._episode_return / max(self._episode_steps, 1)
                    ),
                }
            )
        return obs, reward, terminated, truncated, info


class TrainingMetricsCallback(BaseCallback):
    """Persist episode metrics and expose action/protocol metrics to TensorBoard."""

    FIELDNAMES = (
        "training_seed",
        "variant",
        "global_timestep",
        "episode_index",
        "episode_return",
        "episode_length",
        "return_per_timestep",
        "total_distance_m",
        "distinct_ego_collision_events",
        "ego_collision_active_timesteps",
        "distance_per_collision_m",
        "distance_per_collision_right_censored",
        "distance_per_collision_exposure_bound_m",
        "ego_collisions_per_km",
        "time_to_first_collision_s",
        "distance_to_first_collision_m",
        "collision_transition_timesteps",
        "collision_transition_return",
        "post_collision_timesteps",
        "post_collision_return",
        "reset_calls_total",
        "resets_after_collision",
        "action_saturation_mean",
    )

    def __init__(self, *, path: Path, training_seed: int, variant: str) -> None:
        super().__init__(verbose=0)
        self.path = Path(path)
        self.training_seed = int(training_seed)
        self.variant = str(variant)
        self.episode_index = 0
        self.action_saturation_sum = 0.0
        self.action_saturation_count = 0
        self.resets_after_collision = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "episode_index": int(self.episode_index),
            "action_saturation_sum": float(self.action_saturation_sum),
            "action_saturation_count": int(self.action_saturation_count),
            "resets_after_collision": int(self.resets_after_collision),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.episode_index = int(state.get("episode_index", 0))
        self.action_saturation_sum = float(state.get("action_saturation_sum", 0.0))
        self.action_saturation_count = int(state.get("action_saturation_count", 0))
        self.resets_after_collision = int(state.get("resets_after_collision", 0))

    def _append_row(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow({key: row.get(key, np.nan) for key in self.FIELDNAMES})

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
        for env_index, info in enumerate(infos):
            saturation = _as_float(info.get("pipeline_action_saturation"), default=0.0)
            self.action_saturation_sum += saturation
            self.action_saturation_count += 1
            self.logger.record(
                "rollout/action_saturation_mean",
                self.action_saturation_sum / max(self.action_saturation_count, 1),
            )
            self.logger.record(
                "rollout/collision_active_timestep",
                float(info.get("pipeline_ego_collision_active_timestep", 0)),
            )
            self.logger.record(
                "rollout/distinct_collision_events",
                float(info.get("pipeline_distinct_ego_collision_events", 0)),
            )
            if env_index >= len(dones) or not bool(dones[env_index]):
                continue

            episode = dict(info.get("episode", {}))
            episode_return = _as_float(episode.get("r"), default=np.nan)
            episode_length = int(_as_float(episode.get("l"), default=0.0))
            distance_m = _as_float(info.get("pipeline_episode_distance_m"), default=0.0)
            collision_events = _as_float(
                info.get("pipeline_episode_distinct_ego_collision_events"), default=0.0
            )
            return_per_timestep = _as_float(
                info.get("pipeline_episode_return_per_timestep"),
                default=episode_return / max(episode_length, 1),
            )
            self.episode_index += 1
            self.resets_after_collision += int(collision_events > 0)
            row = {
                "training_seed": self.training_seed,
                "variant": self.variant,
                "global_timestep": int(self.num_timesteps),
                "episode_index": int(self.episode_index),
                "episode_return": episode_return,
                "episode_length": episode_length,
                "return_per_timestep": return_per_timestep,
                "total_distance_m": distance_m,
                "distinct_ego_collision_events": collision_events,
                "ego_collision_active_timesteps": _as_float(
                    info.get("pipeline_episode_ego_collision_active_timesteps"), default=0.0
                ),
                "distance_per_collision_m": _distance_per_collision(distance_m, collision_events),
                "distance_per_collision_right_censored": int(collision_events == 0.0),
                "distance_per_collision_exposure_bound_m": _distance_per_collision_exposure_bound(
                    distance_m, collision_events
                ),
                "ego_collisions_per_km": _collisions_per_km(collision_events, distance_m),
                "time_to_first_collision_s": _as_float(
                    info.get("pipeline_episode_time_to_first_collision_s")
                ),
                "distance_to_first_collision_m": _as_float(
                    info.get("pipeline_episode_distance_to_first_collision_m")
                ),
                "collision_transition_return": _as_float(
                    info.get("pipeline_episode_collision_transition_return"), default=0.0
                ),
                "collision_transition_timesteps": _as_float(
                    info.get("pipeline_episode_collision_transition_timesteps"), default=0.0
                ),
                "post_collision_timesteps": _as_float(
                    info.get("pipeline_episode_post_collision_timesteps"), default=0.0
                ),
                "post_collision_return": _as_float(
                    info.get("pipeline_episode_post_collision_return"), default=0.0
                ),
                "reset_calls_total": int(self.episode_index + 1),
                "resets_after_collision": int(self.resets_after_collision),
                "action_saturation_mean": _as_float(
                    info.get("pipeline_episode_action_saturation_mean"), default=0.0
                ),
            }
            self._append_row(row)
            self.logger.record("rollout/episode_return", episode_return)
            self.logger.record("rollout/episode_length", episode_length)
            self.logger.record("rollout/return_per_timestep", return_per_timestep)
            self.logger.record("rollout/distance_m", distance_m)
            self.logger.record("rollout/collisions_per_km", row["ego_collisions_per_km"])
            self.logger.record("rollout/reset_calls_total", row["reset_calls_total"])
            self.action_saturation_sum = 0.0
            self.action_saturation_count = 0
        return True


def _wrapper_chain(single_env: gym.Env) -> list[gym.Env]:
    wrappers: list[gym.Env] = []
    current = single_env
    while isinstance(current, gym.Wrapper):
        wrappers.append(current)
        current = current.env
    return wrappers


def _capture_kpi_state(wrapper: gym.Env, base: gym.Env) -> dict[str, Any]:
    keys = (
        "_episode_steps",
        "_episode_time_s",
        "_episode_distance_m",
        "_episode_lateral_shift_m",
        "_episode_ego_collisions",
        "_episode_total_collisions",
        "_previous_y",
        "_previous_acceleration",
    )
    state = {key: copy.deepcopy(getattr(wrapper, key)) for key in keys}
    id_to_index = {id(vehicle): index for index, vehicle in enumerate(base.road.vehicles)}
    previous = getattr(wrapper, "_previous_dx_by_vehicle", {})
    state["_previous_dx_by_vehicle_indices"] = {
        int(id_to_index[vehicle_id]): copy.deepcopy(value)
        for vehicle_id, value in previous.items()
        if vehicle_id in id_to_index
    }
    return state


def _restore_kpi_state(wrapper: gym.Env, base: gym.Env, state: dict[str, Any]) -> None:
    for key, value in state.items():
        if key == "_previous_dx_by_vehicle_indices":
            continue
        setattr(wrapper, key, copy.deepcopy(value))
    previous = state.get("_previous_dx_by_vehicle_indices", {})
    wrapper._previous_dx_by_vehicle = {
        id(base.road.vehicles[int(index)]): copy.deepcopy(value)
        for index, value in previous.items()
    }


def _capture_monitor_state(wrapper: Monitor) -> dict[str, Any]:
    keys = (
        "rewards",
        "needs_reset",
        "episode_returns",
        "episode_lengths",
        "episode_times",
        "total_steps",
        "current_reset_info",
    )
    state = {key: copy.deepcopy(getattr(wrapper, key)) for key in keys if hasattr(wrapper, key)}
    state["elapsed_since_start_s"] = float(time.time() - float(wrapper.t_start))
    return state


def _restore_monitor_state(wrapper: Monitor, state: dict[str, Any]) -> None:
    for key, value in state.items():
        if key == "elapsed_since_start_s":
            continue
        setattr(wrapper, key, copy.deepcopy(value))
    wrapper.t_start = float(time.time() - float(state.get("elapsed_since_start_s", 0.0)))


def _capture_protocol_state(wrapper: ProtocolMetricsWrapper) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in wrapper.__dict__.items()
        if key != "env"
    }


def _restore_protocol_state(wrapper: ProtocolMetricsWrapper, state: dict[str, Any]) -> None:
    for key, value in state.items():
        setattr(wrapper, key, copy.deepcopy(value))


def _capture_wrapper_states(single_env: gym.Env) -> list[dict[str, Any]]:
    base = single_env.unwrapped
    states: list[dict[str, Any]] = []
    stateless_names = {
        "KaralakouRewardWrapper",
        "PreviousActionObservationWrapper",
        "CorrectionRewardSafetyFilteredAccelerationWrapper",
        "SafetyFilteredAccelerationWrapper",
        "LaneFreeObservationNormalizationWrapper",
    }
    for wrapper in _wrapper_chain(single_env):
        name = type(wrapper).__name__
        if isinstance(wrapper, Monitor):
            state = _capture_monitor_state(wrapper)
            kind = "monitor"
        elif isinstance(wrapper, ProtocolMetricsWrapper):
            state = _capture_protocol_state(wrapper)
            kind = "protocol"
        elif name == "KPIInfoWrapper":
            state = _capture_kpi_state(wrapper, base)
            kind = "kpi"
        elif name == "OrderEnforcing":
            state = {"_has_reset": bool(getattr(wrapper, "_has_reset", False))}
            kind = "order_enforcing"
        elif name == "PassiveEnvChecker":
            state = {
                key: copy.deepcopy(value)
                for key, value in wrapper.__dict__.items()
                if key.startswith("checked_") or key.startswith("_checked_")
            }
            kind = "passive_checker"
        elif name in stateless_names:
            state = {}
            kind = "stateless"
        else:
            raise RuntimeError(f"Strict checkpoint does not recognize wrapper {_qualified_name(wrapper)}")
        states.append({"class": _qualified_name(wrapper), "kind": kind, "state": state})
    return states


def _restore_wrapper_states(single_env: gym.Env, states: list[dict[str, Any]]) -> None:
    wrappers = _wrapper_chain(single_env)
    actual = [_qualified_name(wrapper) for wrapper in wrappers]
    expected = [str(item["class"]) for item in states]
    if actual != expected:
        raise RuntimeError(f"Wrapper chain mismatch on resume: expected {expected}, got {actual}")
    base = single_env.unwrapped
    for wrapper, item in zip(wrappers, states):
        kind = item["kind"]
        state = item["state"]
        if kind == "monitor":
            _restore_monitor_state(wrapper, state)
        elif kind == "protocol":
            _restore_protocol_state(wrapper, state)
        elif kind == "kpi":
            _restore_kpi_state(wrapper, base, state)
        elif kind == "order_enforcing":
            wrapper._has_reset = bool(state["_has_reset"])
        elif kind == "passive_checker":
            for key, value in state.items():
                setattr(wrapper, key, copy.deepcopy(value))
        elif kind != "stateless":
            raise RuntimeError(f"Unknown wrapper checkpoint kind: {kind!r}")


def _base_vec_env(env: Any) -> DummyVecEnv:
    current = env
    while isinstance(current, VecNormalize):
        current = current.venv
    if not isinstance(current, DummyVecEnv) or int(current.num_envs) != 1:
        raise RuntimeError("Strict resume currently requires one DummyVecEnv environment")
    return current


def capture_environment_state(env: Any) -> tuple[bytes, dict[str, Any]]:
    vec = _base_vec_env(env)
    single = vec.envs[0]
    base_blob = cloudpickle.dumps(single.unwrapped)
    vec_keys = ("reset_infos", "_seeds", "_options", "buf_obs", "buf_dones", "buf_rews", "buf_infos", "actions")
    vec_state = {key: copy.deepcopy(getattr(vec, key)) for key in vec_keys if hasattr(vec, key)}
    normalization_state = None
    if isinstance(env, VecNormalize):
        normalization_state = {
            "returns": copy.deepcopy(env.returns),
            "old_obs": copy.deepcopy(env.old_obs),
            "old_reward": copy.deepcopy(env.old_reward),
            "training": bool(env.training),
            "norm_obs": bool(env.norm_obs),
            "norm_reward": bool(env.norm_reward),
        }
    state = {
        "wrapper_states": _capture_wrapper_states(single),
        "vec_state": vec_state,
        "normalization_state": normalization_state,
    }
    return base_blob, state


def restore_environment_state(env: Any, base_blob: bytes, state: dict[str, Any]) -> None:
    vec = _base_vec_env(env)
    single = vec.envs[0]
    restored_base = cloudpickle.loads(base_blob)
    target_base = single.unwrapped
    target_base.__dict__.clear()
    target_base.__dict__.update(restored_base.__dict__)
    if hasattr(target_base, "road") and target_base.road.np_random is not target_base.np_random:
        raise RuntimeError("Restored road/environment RNG objects are inconsistent")
    _restore_wrapper_states(single, state["wrapper_states"])
    for key, value in state["vec_state"].items():
        setattr(vec, key, copy.deepcopy(value))
    normalization_state = state.get("normalization_state")
    if normalization_state is not None:
        if not isinstance(env, VecNormalize):
            raise RuntimeError("Checkpoint contains VecNormalize state but resumed environment does not")
        for key, value in normalization_state.items():
            setattr(env, key, copy.deepcopy(value))


def capture_rng_state(model: Any) -> dict[str, Any]:
    action_noise_present = hasattr(model, "action_noise")
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": th.get_rng_state(),
        "cuda": th.cuda.get_rng_state_all() if th.cuda.is_available() else [],
        "cuda_device_count": int(th.cuda.device_count()) if th.cuda.is_available() else 0,
        "model_action_space": _space_rng_state(model.action_space),
        "model_observation_space": _space_rng_state(model.observation_space),
        "action_noise_attribute_present": action_noise_present,
        "action_noise": cloudpickle.dumps(getattr(model, "action_noise", None)),
    }


def restore_rng_state(model: Any, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    th.set_rng_state(state["torch"])
    saved_cuda_count = int(state.get("cuda_device_count", 0))
    current_cuda_count = int(th.cuda.device_count()) if th.cuda.is_available() else 0
    if saved_cuda_count != current_cuda_count:
        raise RuntimeError(
            f"CUDA device-count mismatch on strict resume: saved={saved_cuda_count}, current={current_cuda_count}"
        )
    if saved_cuda_count:
        th.cuda.set_rng_state_all(state["cuda"])
    _restore_space_rng_state(model.action_space, state.get("model_action_space"))
    _restore_space_rng_state(model.observation_space, state.get("model_observation_space"))
    if bool(state.get("action_noise_attribute_present", True)):
        model.action_noise = cloudpickle.loads(state["action_noise"])


def _truncate_to_checkpoint(path: Path, size: int) -> None:
    if not path.exists():
        if int(size) != 0:
            raise RuntimeError(f"Strict-resume log is missing: {path}")
        return
    current_size = path.stat().st_size
    if current_size < int(size):
        raise RuntimeError(f"Strict-resume log is shorter than checkpoint: {path}")
    if current_size > int(size):
        with path.open("r+b") as handle:
            handle.truncate(int(size))


def _checkpoint_checksums(directory: Path, names: list[str]) -> dict[str, str]:
    return {name: file_sha256(directory / name) for name in names}


def validate_checkpoint_bundle(
    bundle: Path,
    expected_config_hash: str,
    *,
    expected_model_class: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = bundle / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Incomplete strict checkpoint (missing manifest): {bundle}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != PIPELINE_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported checkpoint schema in {bundle}")
    if str(manifest.get("training_config_hash")) != str(expected_config_hash):
        raise RuntimeError(
            "Strict-resume configuration hash mismatch: "
            f"saved={manifest.get('training_config_hash')} current={expected_config_hash}"
        )
    if str(manifest.get("model_class")) != str(expected_model_class):
        raise RuntimeError(
            "Strict-resume model-class mismatch: "
            f"saved={manifest.get('model_class')} current={expected_model_class}"
        )
    checksums = manifest.get("checksums")
    if not isinstance(checksums, dict):
        raise RuntimeError(f"Strict checkpoint has no checksum table: {bundle}")
    required_payloads = {
        CHECKPOINT_PAYLOADS["model"],
        CHECKPOINT_PAYLOADS["replay_buffer"],
        CHECKPOINT_PAYLOADS["base_environment"],
        CHECKPOINT_PAYLOADS["pipeline_state"],
    }
    allowed_payloads = required_payloads | {CHECKPOINT_PAYLOADS["vecnormalize"]}
    if not required_payloads.issubset(checksums) or not set(checksums).issubset(allowed_payloads):
        raise RuntimeError(
            "Strict checkpoint payload set mismatch: "
            f"required={sorted(required_payloads)} saved={sorted(checksums)}"
        )
    for name, expected in checksums.items():
        path = bundle / name
        if not path.exists() or file_sha256(path) != expected:
            raise RuntimeError(f"Checkpoint payload is missing or corrupt: {path}")
    with (bundle / CHECKPOINT_PAYLOADS["pipeline_state"]).open("rb") as handle:
        pipeline_state = pickle.load(handle)
    manifest_step = int(manifest.get("timestep", -1))
    if manifest_step < 0 or int(pipeline_state.get("timestep", -2)) != manifest_step:
        raise RuntimeError("Checkpoint timestep disagreement between manifest and pipeline state")
    if int(pipeline_state.get("schema_version", -1)) != PIPELINE_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported pipeline-state schema in {bundle}")
    if str(pipeline_state.get("training_config_hash")) != str(expected_config_hash):
        raise RuntimeError("Checkpoint pipeline-state configuration hash mismatch")
    if int(pipeline_state.get("n_updates", -1)) != int(manifest.get("n_updates", -2)):
        raise RuntimeError("Checkpoint update-count disagreement between manifest and pipeline state")
    return manifest, pipeline_state


class StrictCheckpointCallback(BaseCallback):
    """Save coherent model/replay/environment/RNG bundles between updates."""

    def __init__(
        self,
        *,
        variant_dir: Path,
        checkpoint_interval: int,
        training_config_hash: str,
        metrics_callback: TrainingMetricsCallback,
        tracked_log_paths: list[Path],
        resume_pipeline_state: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(verbose=0)
        self.variant_dir = Path(variant_dir)
        self.checkpoint_interval = int(checkpoint_interval)
        self.training_config_hash = str(training_config_hash)
        self.metrics_callback = metrics_callback
        self.tracked_log_paths = [Path(path) for path in tracked_log_paths]
        self.resume_pipeline_state = resume_pipeline_state
        self.next_checkpoint_step = self.checkpoint_interval
        self.pending_checkpoint = False
        self.last_saved_step = -1

    def _on_training_start(self) -> None:
        if self.resume_pipeline_state is None:
            self.next_checkpoint_step = (
                (int(self.model.num_timesteps) // self.checkpoint_interval) + 1
            ) * self.checkpoint_interval
            return
        callback_state = self.resume_pipeline_state.get("metrics_callback_state", {})
        self.metrics_callback.load_state_dict(callback_state)
        restore_rng_state(self.model, self.resume_pipeline_state["rng_state"])
        self.next_checkpoint_step = int(
            self.resume_pipeline_state.get(
                "next_checkpoint_step",
                ((int(self.model.num_timesteps) // self.checkpoint_interval) + 1) * self.checkpoint_interval,
            )
        )
        self.last_saved_step = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        if int(self.num_timesteps) >= int(self.next_checkpoint_step):
            self.pending_checkpoint = True
        return True

    def _on_rollout_start(self) -> None:
        if self.pending_checkpoint and int(self.model.num_timesteps) > self.last_saved_step:
            self._save_bundle()

    def _on_training_end(self) -> None:
        if int(self.model.num_timesteps) > self.last_saved_step:
            self._save_bundle()

    def _save_bundle(self) -> None:
        step = int(self.model.num_timesteps)
        # Keep this layout deliberately short.  The repository commonly lives
        # under a long OneDrive path, where verbose checkpoint names exceed the
        # legacy Windows MAX_PATH limit inside SB3's replay-buffer saver.
        checkpoints_dir = self.variant_dir / "ckpt"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        final_dir = checkpoints_dir / f"{step:09d}"
        temp_dir = checkpoints_dir / f".tmp_{step:09d}_{os.getpid()}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if final_dir.exists():
            raise RuntimeError(f"Refusing to overwrite strict checkpoint: {final_dir}")
        temp_dir.mkdir(parents=True)

        base_blob, environment_state = capture_environment_state(self.model.get_env())
        log_offsets = {
            str(path.resolve().relative_to(self.variant_dir.resolve())): int(path.stat().st_size) if path.exists() else 0
            for path in self.tracked_log_paths
        }
        pipeline_state = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "timestep": step,
            "n_updates": int(self.model._n_updates),
            "training_config_hash": self.training_config_hash,
            "environment_state": environment_state,
            "rng_state": capture_rng_state(self.model),
            "replay_buffer_state": {
                "class": _qualified_name(self.model.replay_buffer),
                "size": int(self.model.replay_buffer.size()),
                "position": int(self.model.replay_buffer.pos),
                "full": bool(self.model.replay_buffer.full),
            },
            "metrics_callback_state": self.metrics_callback.state_dict(),
            "next_checkpoint_step": (
                (step // self.checkpoint_interval) + 1
            ) * self.checkpoint_interval,
            "log_offsets": log_offsets,
        }
        self.model.save(str(temp_dir / CHECKPOINT_PAYLOADS["model"]))
        self.model.save_replay_buffer(str(temp_dir / CHECKPOINT_PAYLOADS["replay_buffer"]))
        (temp_dir / CHECKPOINT_PAYLOADS["base_environment"]).write_bytes(base_blob)
        with (temp_dir / CHECKPOINT_PAYLOADS["pipeline_state"]).open("wb") as handle:
            pickle.dump(pipeline_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

        payload_names = [
            CHECKPOINT_PAYLOADS["model"],
            CHECKPOINT_PAYLOADS["replay_buffer"],
            CHECKPOINT_PAYLOADS["base_environment"],
            CHECKPOINT_PAYLOADS["pipeline_state"],
        ]
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None:
            vec_normalize.save(str(temp_dir / CHECKPOINT_PAYLOADS["vecnormalize"]))
            payload_names.append(CHECKPOINT_PAYLOADS["vecnormalize"])
        manifest = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "timestep": step,
            "n_updates": int(self.model._n_updates),
            "training_config_hash": self.training_config_hash,
            "model_class": _qualified_name(self.model),
            "checksums": _checkpoint_checksums(temp_dir, payload_names),
        }
        (temp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temp_dir.replace(final_dir)

        latest_tmp = self.variant_dir / ".latest_checkpoint.json.tmp"
        latest = self.variant_dir / "latest_checkpoint.json"
        latest_tmp.write_text(
            json.dumps({"checkpoint": str(final_dir.relative_to(self.variant_dir)), "timestep": step}, indent=2),
            encoding="utf-8",
        )
        os.replace(latest_tmp, latest)
        self.last_saved_step = step
        self.next_checkpoint_step = ((step // self.checkpoint_interval) + 1) * self.checkpoint_interval
        self.pending_checkpoint = False
        print(f"[ablation] strict checkpoint seed/variant step={step:,}: {final_dir}", flush=True)


def bootstrap_notebook_namespace(project_root: Path) -> dict[str, Any]:
    """Provide the deleted notebook setup cell without modifying the notebook.

    The notebook is intentionally treated as a source of definitions, not an
    executable artifact.  This keeps the runner robust when prose cells are
    inserted or removed and preserves the user's working notebook state.
    """

    lane_free_dir = project_root / "laneless highway env"
    lane_free_dir_str = str(lane_free_dir)
    if lane_free_dir_str not in sys.path:
        sys.path.insert(0, lane_free_dir_str)
    import lane_free_env  # noqa: F401

    artifact_dir = project_root / "artifacts" / "lanelessKaralakou"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return {
        "__name__": "__main__",
        "Any": Any,
        "Optional": Optional,
        "Union": Union,
        "Path": Path,
        "datetime": datetime,
        "json": json,
        "os": os,
        "sys": sys,
        "time": time,
        "warnings": warnings,
        "gym": gym,
        "np": np,
        "pd": pd,
        "torch": th,
        "DDPG": DDPG,
        "BaseCallback": BaseCallback,
        "CallbackList": CallbackList,
        "Monitor": Monitor,
        "OrnsteinUhlenbeckActionNoise": OrnsteinUhlenbeckActionNoise,
        "VectorizedActionNoise": VectorizedActionNoise,
        "DummyVecEnv": DummyVecEnv,
        "SubprocVecEnv": SubprocVecEnv,
        "PROJECT_ROOT": project_root,
        "LANE_FREE_DIR": lane_free_dir,
        "ARTIFACT_DIR": artifact_dir,
    }


def exec_required_notebook_cells(notebook_path: Path, namespace: dict[str, Any]) -> None:
    """Execute only the code definitions required by this ablation.

    Historical scripts relied on fixed notebook indices.  The current notebook
    has a user-edited setup section, so marker-based selection is safer and
    avoids executing training, plotting, or rendering cells.
    """

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    selected: list[tuple[int, str]] = []
    found_markers: set[str] = set()
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        matching = [marker for marker in NOTEBOOK_SETUP_MARKERS if marker in source]
        if matching:
            selected.append((index, source))
            found_markers.update(matching)
    missing = sorted(set(NOTEBOOK_SETUP_MARKERS) - found_markers)
    if missing:
        raise RuntimeError(f"Notebook is missing required definitions: {missing}")
    for index, source in selected:
        print(f"[ablation] executing notebook definitions from cell {index}", flush=True)
        exec(compile(source, f"{notebook_path}:cell-{index}", "exec"), namespace)


def _finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _mean(values: list[float], default: float = np.nan) -> float:
    array = _finite(values)
    return float(np.mean(array)) if array.size else default


def _min(values: list[float], default: float = np.nan) -> float:
    array = _finite(values)
    return float(np.min(array)) if array.size else default


def _p95(values: list[float], default: float = np.nan) -> float:
    array = _finite(values)
    return float(np.percentile(array, 95)) if array.size else default


def _as_float(value: Any, default: float = np.nan) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return default
    return scalar if np.isfinite(scalar) else default


def seed_everything(seed: int) -> None:
    """Reset all learner-side RNGs before every paired training replicate."""

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    set_random_seed(seed, using_cuda=False)


def physical_bounds(env_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    bounds = env_config["bounds"]
    low = np.asarray([bounds["ax_min"], bounds["ay_min"]], dtype=np.float32)
    high = np.asarray([bounds["ax_max"], bounds["ay_max"]], dtype=np.float32)
    return low, high


def normalized_to_physical(action_normalized: np.ndarray, env_config: dict[str, Any]) -> np.ndarray:
    """Apply the lane-free environment's exact per-axis action map."""

    low, high = physical_bounds(env_config)
    action = np.clip(np.asarray(action_normalized, dtype=np.float32).reshape(-1)[:2], -1.0, 1.0)
    output = np.empty(2, dtype=np.float32)
    for index, (value, lower, upper) in enumerate(zip(action, low, high)):
        if float(lower) < 0.0 < float(upper):
            scale = float(upper) if float(value) >= 0.0 else abs(float(lower))
            output[index] = float(value) * max(scale, 1e-6)
        else:
            output[index] = float(lower) + 0.5 * (float(value) + 1.0) * float(upper - lower)
    return np.clip(output, low, high).astype(np.float32)


def physical_to_normalized(action_phys: np.ndarray, env_config: dict[str, Any]) -> np.ndarray:
    """Invert :func:`normalized_to_physical`, including asymmetric zero-crossing bounds."""

    low, high = physical_bounds(env_config)
    action_phys = np.asarray(action_phys, dtype=np.float32).reshape(-1)[:2]
    clipped = np.clip(action_phys, low, high)
    normalized = np.empty(2, dtype=np.float32)
    for index, (value, lower, upper) in enumerate(zip(clipped, low, high)):
        if float(lower) < 0.0 < float(upper):
            scale = float(upper) if float(value) >= 0.0 else abs(float(lower))
            normalized[index] = float(value) / max(scale, 1e-6)
        else:
            normalized[index] = 2.0 * (float(value) - float(lower)) / max(float(upper - lower), 1e-6) - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def normalized_delta_norm(raw_phys: np.ndarray, safe_phys: np.ndarray, env_config: dict[str, Any]) -> float:
    raw_normalized = physical_to_normalized(raw_phys, env_config)
    safe_normalized = physical_to_normalized(safe_phys, env_config)
    return float(np.linalg.norm(safe_normalized - raw_normalized))


def box_scaled_delta_norm(
    raw_phys: np.ndarray,
    safe_phys: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> float:
    """Measure a physical correction on an affine actor/filter box scale."""

    half_range = np.maximum(
        0.5
        * (
            np.asarray(high, dtype=np.float32).reshape(-1)[:2]
            - np.asarray(low, dtype=np.float32).reshape(-1)[:2]
        ),
        1e-6,
    )
    delta = (
        np.asarray(safe_phys, dtype=np.float32).reshape(-1)[:2]
        - np.asarray(raw_phys, dtype=np.float32).reshape(-1)[:2]
    )
    return float(np.linalg.norm(delta / half_range))


def model_action_to_physical(model: Any, action: np.ndarray, env_config: dict[str, Any]) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
    model_low = np.asarray(model.action_space.low, dtype=np.float32).reshape(-1)[:2]
    model_high = np.asarray(model.action_space.high, dtype=np.float32).reshape(-1)[:2]
    if np.allclose(model_low, -1.0, atol=1e-5) and np.allclose(model_high, 1.0, atol=1e-5):
        return normalized_to_physical(action, env_config)
    env_low, env_high = physical_bounds(env_config)
    return np.clip(action, np.maximum(model_low, env_low), np.minimum(model_high, env_high)).astype(np.float32)


def initial_state_hash(env: gym.Env) -> str:
    base = env.unwrapped
    rows = []
    for vehicle in base.road.vehicles:
        rows.append(
            {
                "x": round(float(vehicle.position[0]), 10),
                "y": round(float(vehicle.position[1]), 10),
                "vx": round(float(vehicle.vx), 10),
                "vy": round(float(getattr(vehicle, "vy", 0.0)), 10),
                "length": round(float(vehicle.length), 10),
                "width": round(float(vehicle.width), 10),
                "desired_speed": round(float(vehicle.desired_speed), 10),
                "profile": str(getattr(vehicle, "driver_profile", "")),
            }
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_base_reward_config(namespace: dict[str, Any]) -> dict[str, float]:
    """One common task reward for every learned variant and every deployment mode."""

    config = dict(namespace["REWARD_CONFIG"])
    config.update(
        {
            "use_current_potential": 1.0,
            "use_safety_potential": 0.0,
            "w_safe": 0.0,
        }
    )
    return config


def install_correction_reward_env(namespace: dict[str, Any]) -> None:
    """Install the exact correction reward on top of the notebook's shield.

    The notebook wrapper's original correction term is disabled.  The terms
    below use normalized action deltas, matching the replay-buffer actor loss.
    """

    base_wrapper = namespace["SafetyFilteredAccelerationWrapper"]

    class CorrectionRewardSafetyFilteredAccelerationWrapper(base_wrapper):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            *args,
            lambda_delta: float = 0.0,
            lambda_intervention: float = 0.0,
            correction_epsilon: float = 0.03,
            **kwargs,
        ) -> None:
            kwargs["lambda_filter"] = 0.0
            super().__init__(*args, **kwargs)
            self.lambda_delta = float(lambda_delta)
            self.lambda_intervention = float(lambda_intervention)
            self.correction_epsilon = float(correction_epsilon)

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            info = dict(info)
            raw_action = np.asarray(
                [info.get("cbf_a_rl_x", 0.0), info.get("cbf_a_rl_y", 0.0)], dtype=np.float32
            )
            safe_action = np.asarray(
                [info.get("cbf_a_safe_x", raw_action[0]), info.get("cbf_a_safe_y", raw_action[1])],
                dtype=np.float32,
            )
            low = np.asarray([self.ax_bounds[0], self.ay_bounds[0]], dtype=np.float32)
            high = np.asarray([self.ax_bounds[1], self.ay_bounds[1]], dtype=np.float32)
            half_range = np.maximum(0.5 * (high - low), 1e-6)
            correction_norm_normalized = float(np.linalg.norm((safe_action - raw_action) / half_range))
            event_intervened = bool(correction_norm_normalized > self.correction_epsilon)
            correction_reward = -(
                self.lambda_delta * correction_norm_normalized**2
                + self.lambda_intervention * float(event_intervened)
            )
            reward = float(reward) + correction_reward
            info.update(
                {
                    "raw_action_phys": raw_action,
                    "safe_action_phys": safe_action,
                    "intervention": event_intervened,
                    "cbf_event_intervened": event_intervened,
                    "cbf_event_intervention_threshold": self.correction_epsilon,
                    "cbf_correction_norm_normalized": correction_norm_normalized,
                    "cbf_filter_norm_reward_penalty": self.lambda_delta * correction_norm_normalized**2,
                    "cbf_filter_event_reward_penalty": self.lambda_intervention * float(event_intervened),
                    "cbf_filter_reward_penalty": -correction_reward,
                    "cbf_correction_reward": correction_reward,
                }
            )
            return obs, reward, terminated, truncated, info

    namespace["CorrectionRewardSafetyFilteredAccelerationWrapper"] = CorrectionRewardSafetyFilteredAccelerationWrapper


def make_raw_env(
    namespace: dict[str, Any],
    *,
    seed: int,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
) -> gym.Env:
    env = gym.make("lane-free-v0", render_mode=None, config=copy.deepcopy(env_config))
    env = namespace["KaralakouRewardWrapper"](env, reward_config=copy.deepcopy(reward_config))
    if namespace.get("NORMALIZE_RL_OBSERVATIONS", False):
        env = namespace["LaneFreeObservationNormalizationWrapper"](env, clip=namespace["OBSERVATION_CLIP"])
    if "KPIInfoWrapper" in namespace:
        env = namespace["KPIInfoWrapper"](env)
    return env


def make_cbf_env(
    namespace: dict[str, Any],
    *,
    seed: int,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    lambda_delta: float,
    lambda_intervention: float,
    correction_epsilon: float,
    k0: float,
    k1: float,
    eps_side: float,
) -> gym.Env:
    env = gym.make("lane-free-v0", render_mode=None, config=copy.deepcopy(env_config))
    env = namespace["KaralakouRewardWrapper"](env, reward_config=copy.deepcopy(reward_config))
    env = namespace["CorrectionRewardSafetyFilteredAccelerationWrapper"](
        env,
        lambda_delta=float(lambda_delta),
        lambda_intervention=float(lambda_intervention),
        correction_epsilon=float(correction_epsilon),
        eps_side=float(eps_side),
        k0=float(k0),
        k1=float(k1),
    )
    if namespace.get("NORMALIZE_RL_OBSERVATIONS", False):
        env = namespace["LaneFreeObservationNormalizationWrapper"](env, clip=namespace["OBSERVATION_CLIP"])
    if "KPIInfoWrapper" in namespace:
        env = namespace["KPIInfoWrapper"](env, intervention_threshold=float(correction_epsilon))
    return env


def make_training_env(
    namespace: dict[str, Any],
    *,
    filtered: bool,
    seed: int,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    args: argparse.Namespace,
    monitor_path: Path,
    append_monitor: bool,
) -> Any:
    def make_single(env_seed: int) -> gym.Env:
        if not filtered:
            env = make_raw_env(
                namespace,
                seed=env_seed,
                env_config=env_config,
                reward_config=reward_config,
            )
        else:
            env = make_cbf_env(
                namespace,
                seed=env_seed,
                env_config=env_config,
                reward_config=reward_config,
                lambda_delta=float(args.lambda_delta),
                lambda_intervention=float(args.lambda_intervention),
                correction_epsilon=float(args.correction_epsilon),
                k0=float(args.k0),
                k1=float(args.k1),
                eps_side=float(args.eps_side),
            )
        if not bool(env.unwrapped.config.get("terminate_on_collision", False)):
            raise RuntimeError("Training protocol requires terminate_on_collision=True")
        env = ProtocolMetricsWrapper(env)
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=TRAINING_MONITOR_INFO_KEYS,
            override_existing=not bool(append_monitor),
        )
        return env

    return namespace["_make_vectorized_env"](
        make_single,
        seed=int(seed),
        n_envs=int(args.n_envs),
        use_subproc=False,
        start_method=namespace["DDPG_SUBPROC_START_METHOD"],
    )


def model_kwargs(
    namespace: dict[str, Any],
    train_env: Any,
    seed: int,
    device: str,
    tensorboard_log: Path,
) -> dict[str, Any]:
    n_actions = int(train_env.action_space.shape[-1])
    return {
        "learning_rate": namespace["DDPG_LEARNING_RATE"],
        "buffer_size": namespace["DDPG_REPLAY_MEMORY"],
        "learning_starts": namespace["DDPG_LEARNING_STARTS"],
        "batch_size": namespace["DDPG_BATCH_SIZE"],
        "tau": namespace["DDPG_TAU"],
        "gamma": namespace["DDPG_GAMMA"],
        "train_freq": (1, "step"),
        "gradient_steps": 1,
        "action_noise": namespace["make_ou_action_noise"](n_actions, n_envs=1),
        "policy_kwargs": {"net_arch": [256, 128]},
        "tensorboard_log": str(tensorboard_log),
        "verbose": 0,
        "seed": int(seed),
        "device": device,
    }


def build_model(
    namespace: dict[str, Any],
    *,
    variant: str,
    train_env: Any,
    seed: int,
    args: argparse.Namespace,
    tensorboard_log: Path,
) -> Any:
    kwargs = model_kwargs(namespace, train_env, seed, args.device, tensorboard_log)
    spec = variant_spec(variant, args)
    if variant == "a_nominal":
        return DDPG("MlpPolicy", train_env, **kwargs)
    return GuidedCBFDDPG(
        "MlpPolicy",
        train_env,
        lambda_bc=float(args.lambda_bc) if spec["actor_loss"] else 0.0,
        bc_delta=float(args.correction_epsilon),
        bc_loss_mode="local_projection_mse",
        use_projected_q=False,
        projected_q_weight=0.0,
        critic_action_mode="raw",
        actor_action_mode="raw",
        **kwargs,
    )


def variant_spec(variant: str, args: argparse.Namespace) -> dict[str, Any]:
    if variant == "a_nominal":
        return {"filtered": False, "lambda_delta": 0.0, "lambda_intervention": 0.0, "actor_loss": False}
    if variant == "b_filtered":
        return {"filtered": True, "lambda_delta": 0.0, "lambda_intervention": 0.0, "actor_loss": False}
    if variant == "c_reward":
        return {
            "filtered": True,
            "lambda_delta": float(args.lambda_delta),
            "lambda_intervention": float(args.lambda_intervention),
            "actor_loss": False,
        }
    if variant == "d_loss":
        return {
            "filtered": True,
            "lambda_delta": 0.0,
            "lambda_intervention": 0.0,
            "actor_loss": True,
        }
    if variant == "e_reward_actor":
        return {
            "filtered": True,
            "lambda_delta": float(args.lambda_delta),
            "lambda_intervention": float(args.lambda_intervention),
            "actor_loss": True,
        }
    raise ValueError(f"Unknown variant: {variant}")


def model_class_for_variant(variant: str) -> type:
    return GuidedCBFDDPG if variant in set(FACTORIAL_VARIANTS.values()) else DDPG


def _package_versions() -> dict[str, str]:
    names = ("stable-baselines3", "torch", "gymnasium", "numpy", "scipy", "qpsolvers", "osqp", "cloudpickle")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    versions["python"] = sys.version
    return versions


def training_config_payload(
    namespace: dict[str, Any],
    *,
    project_root: Path,
    variant: str,
    seed: int,
    args: argparse.Namespace,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
) -> dict[str, Any]:
    spec = variant_spec(variant, args)
    source_paths = {
        "runner": Path(__file__).resolve(),
        "notebook": project_root / "notebooks" / "lanelessKaralakou.ipynb",
        "guided_cbf": project_root / "scripts" / "common" / "guided_cbf_minimal.py",
        "environment": project_root / "laneless highway env" / "lane_free_env.py",
        "script_config": project_root / "scripts" / "common" / "laneless_script_config.py",
    }
    low, high = physical_bounds(env_config)
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "study": "cbf_filter_internalization_ablation",
        "variant": variant,
        "variant_spec": spec,
        "training_seed": int(seed),
        "target_timesteps": int(args.timesteps),
        "n_envs": int(args.n_envs),
        "device": str(args.device),
        "env_config": env_config,
        "reward_config": reward_config,
        "cbf": {
            "k0": float(args.k0),
            "k1": float(args.k1),
            "eps_side": float(args.eps_side),
            "correction_epsilon": float(args.correction_epsilon),
            "lambda_delta": float(spec["lambda_delta"]),
            "lambda_intervention": float(spec["lambda_intervention"]),
            "lambda_bc_configured": float(args.lambda_bc),
            "lambda_bc_effective": float(args.lambda_bc) if spec["actor_loss"] else 0.0,
            "actor_loss_mode": "local_projection_mse" if spec["actor_loss"] else "disabled",
            "physical_action_low": low,
            "physical_action_high": high,
            "solver": str(namespace.get("CBF_QP_SOLVER", "osqp")),
            "max_neighbor_constraints": namespace.get("CBF_MAX_NEIGHBOR_CONSTRAINTS"),
        },
        "ddpg": {
            "learning_rate": float(namespace["DDPG_LEARNING_RATE"]),
            "buffer_size": int(namespace["DDPG_REPLAY_MEMORY"]),
            "learning_starts": int(namespace["DDPG_LEARNING_STARTS"]),
            "batch_size": int(namespace["DDPG_BATCH_SIZE"]),
            "tau": float(namespace["DDPG_TAU"]),
            "gamma": float(namespace["DDPG_GAMMA"]),
            "train_freq": [1, "step"],
            "gradient_steps": 1,
            "policy_net_arch": [256, 128],
            "ou_sigma": float(namespace["DDPG_OU_SIGMA"]),
            "ou_theta": 0.15,
            "ou_dt": 0.01,
            "replay_class": (
                "CBFGuidedReplayBuffer"
                if variant in set(FACTORIAL_VARIANTS.values())
                else "ReplayBuffer"
            ),
            "replay_action_semantics": {
                "actions": "nominal behavior action on actor scale",
                "safe_actions_side_channel": bool(
                    variant in set(FACTORIAL_VARIANTS.values())
                ),
                "projection_jacobians_side_channel": bool(
                    variant in set(FACTORIAL_VARIANTS.values())
                ),
                "actor_loss_target": (
                    "stop_gradient[safe_behavior + projection_jacobian_behavior "
                    "@ (current_actor - behavior_action)], gated to identity on "
                    "the recorded feasible side"
                    if spec["actor_loss"]
                    else "disabled"
                ),
                "actor_loss_mask": "recorded_intervention" if spec["actor_loss"] else "disabled",
                "executed_safe_action_replaces_nominal": False,
            },
            "critic_action_mode": "raw",
            "actor_action_mode": "raw",
        },
        "pipeline": {
            "terminate_on_collision": True,
            "timestep_based": True,
            "checkpoint_interval": int(args.checkpoint_interval),
            "monitor": True,
            "tensorboard": True,
            "tensorboard_resume_policy": "new lineage segment per process with preserved global timestep",
            "tensorboard_metrics": [
                "train/actor_loss",
                "train/critic_loss",
                "train/actor_g_q_norm",
                "train/actor_g_cbf_norm",
                "train/actor_g_cbf_to_g_q_ratio",
                "train/actor_g_q_g_cbf_cosine",
                "train/actor_g_q_g_cbf_cosine_valid_rate",
                "rollout/action_saturation_mean",
                "rollout/episode_return",
                "rollout/episode_length",
            ],
            "normalization_kind": "static" if namespace.get("NORMALIZE_RL_OBSERVATIONS", False) else "none",
        },
        "runtime_versions": _package_versions(),
        "torch_runtime": {
            "deterministic_algorithms": bool(th.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(th.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(th.backends.cudnn.benchmark),
            "num_threads": int(th.get_num_threads()),
            "num_interop_threads": int(th.get_num_interop_threads()),
        },
        "source_hashes": {name: file_sha256(path) for name, path in source_paths.items()},
        "native_thread_settings": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "TORCH_NUM_THREADS",
            )
        },
    }


def _latest_checkpoint_bundle(variant_dir: Path) -> Path:
    pointer = variant_dir / "latest_checkpoint.json"
    if not pointer.exists():
        legacy = variant_dir / "model_final.zip"
        if legacy.exists():
            raise RuntimeError(
                f"Legacy model is evaluation-only and cannot be strictly resumed: {legacy}"
            )
        raise RuntimeError(f"No strict checkpoint is available in {variant_dir}")
    data = json.loads(pointer.read_text(encoding="utf-8"))
    bundle = (variant_dir / str(data["checkpoint"])).resolve()
    if os.path.commonpath([str(bundle), str(variant_dir.resolve())]) != str(variant_dir.resolve()):
        raise RuntimeError(f"Invalid checkpoint pointer outside variant directory: {bundle}")
    pointer_step = int(data.get("timestep", -1))
    if pointer_step < 0 or bundle.name != f"{pointer_step:09d}":
        raise RuntimeError(f"Checkpoint pointer timestep/path mismatch: {pointer}")
    if not bundle.is_dir():
        raise RuntimeError(f"Checkpoint pointer target is missing: {bundle}")
    return bundle


def preflight_output(
    namespace: dict[str, Any],
    *,
    project_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
) -> None:
    """Validate every output cell before any metadata or model file is written."""

    if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Refusing to mix a fresh run with existing artifacts: {output_dir}"
        )
    for seed in args.seeds:
        for variant in VARIANTS:
            variant_dir = output_dir / f"seed_{int(seed)}" / variant
            payload = training_config_payload(
                namespace,
                project_root=project_root,
                variant=variant,
                seed=int(seed),
                args=args,
                env_config=env_config,
                reward_config=reward_config,
            )
            config_hash = canonical_config_hash(payload)
            if args.resume:
                bundle = _latest_checkpoint_bundle(variant_dir)
                validate_checkpoint_bundle(
                    bundle,
                    config_hash,
                    expected_model_class=_class_qualified_name(model_class_for_variant(variant)),
                )
            elif variant_dir.exists() and any(variant_dir.iterdir()):
                raise RuntimeError(
                    f"Refusing to overwrite existing training artifacts without --resume: {variant_dir}"
                )


def _observations_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_observations_equal(left[key], right[key]) for key in left)
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def train_variant(
    namespace: dict[str, Any],
    *,
    variant: str,
    seed: int,
    args: argparse.Namespace,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    output_dir: Path,
) -> Path:
    spec = variant_spec(variant, args)
    variant_dir = output_dir / f"seed_{seed}" / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    final_path = variant_dir / "model_final.zip"
    config_path = variant_dir / "run_config.json"
    monitor_path = variant_dir / "training.monitor.csv"
    training_metrics_path = variant_dir / "training_episodes.csv"
    project_root = Path(namespace["PROJECT_ROOT"])
    config_payload = training_config_payload(
        namespace,
        project_root=project_root,
        variant=variant,
        seed=seed,
        args=args,
        env_config=env_config,
        reward_config=reward_config,
    )
    config_hash = canonical_config_hash(config_payload)
    default_tb_root = project_root.parent / ".tb"
    tensorboard_root = Path(os.environ.get("CBF_ABLATION_TENSORBOARD_ROOT", str(default_tb_root)))
    output_identity = hashlib.sha256(str(variant_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    tensorboard_log = tensorboard_root / f"{config_hash[:12]}_{output_identity}"
    resume_bundle: Optional[Path] = None
    resume_manifest: Optional[dict[str, Any]] = None
    resume_pipeline_state: Optional[dict[str, Any]] = None
    if args.resume:
        resume_bundle = _latest_checkpoint_bundle(variant_dir)
        resume_manifest, resume_pipeline_state = validate_checkpoint_bundle(
            resume_bundle,
            config_hash,
            expected_model_class=_class_qualified_name(model_class_for_variant(variant)),
        )
        expected_logs = {
            str(monitor_path.relative_to(variant_dir)): monitor_path,
            str(training_metrics_path.relative_to(variant_dir)): training_metrics_path,
        }
        saved_offsets = resume_pipeline_state.get("log_offsets", {})
        if set(saved_offsets) != set(expected_logs):
            raise RuntimeError(
                f"Strict-resume log set mismatch: saved={sorted(saved_offsets)}, current={sorted(expected_logs)}"
            )
        for relative, size in saved_offsets.items():
            _truncate_to_checkpoint(expected_logs[relative], int(size))
    else:
        seed_everything(seed)
    session_parent_step = 0 if resume_manifest is None else int(resume_manifest["timestep"])
    tensorboard_session = (
        f"fresh_{time.time_ns()}"
        if resume_manifest is None
        else f"resume_{session_parent_step:09d}_{time.time_ns()}"
    )

    train_args = copy.copy(args)
    train_args.lambda_delta = float(spec["lambda_delta"])
    train_args.lambda_intervention = float(spec["lambda_intervention"])
    train_env = make_training_env(
        namespace,
        filtered=bool(spec["filtered"]),
        seed=seed,
        env_config=env_config,
        reward_config=reward_config,
        args=train_args,
        monitor_path=monitor_path,
        append_monitor=bool(args.resume),
    )
    if resume_bundle is not None:
        vecnormalize_path = resume_bundle / CHECKPOINT_PAYLOADS["vecnormalize"]
        if vecnormalize_path.exists():
            train_env = VecNormalize.load(str(vecnormalize_path), train_env)
        model_cls = model_class_for_variant(variant)
        model = model_cls.load(
            str(resume_bundle / CHECKPOINT_PAYLOADS["model"]),
            env=train_env,
            device=args.device,
            force_reset=False,
        )
        model.tensorboard_log = str(tensorboard_log)
        model.load_replay_buffer(
            str(resume_bundle / CHECKPOINT_PAYLOADS["replay_buffer"]),
            truncate_last_traj=False,
        )
        restore_environment_state(
            train_env,
            (resume_bundle / CHECKPOINT_PAYLOADS["base_environment"]).read_bytes(),
            resume_pipeline_state["environment_state"],
        )
        saved_step = int(resume_manifest["timestep"])
        if int(model.num_timesteps) != saved_step or int(resume_pipeline_state["timestep"]) != saved_step:
            raise RuntimeError("Checkpoint timestep disagreement between model, manifest, and pipeline state")
        if int(model._n_updates) != int(resume_pipeline_state["n_updates"]):
            raise RuntimeError("Checkpoint learner update count was not restored exactly")
        replay_state = resume_pipeline_state["replay_buffer_state"]
        actual_replay_state = {
            "class": _qualified_name(model.replay_buffer),
            "size": int(model.replay_buffer.size()),
            "position": int(model.replay_buffer.pos),
            "full": bool(model.replay_buffer.full),
        }
        if actual_replay_state != replay_state:
            raise RuntimeError(
                f"Replay-buffer state mismatch: saved={replay_state} restored={actual_replay_state}"
            )
        restored_obs = _base_vec_env(train_env)._obs_from_buf()
        if not _observations_equal(model._last_obs, restored_obs):
            raise RuntimeError("Saved model observation does not match restored environment observation")
        if np.asarray(model._last_episode_starts, dtype=bool).shape != (int(model.n_envs),):
            raise RuntimeError("Saved episode-start flags have the wrong vector-environment shape")
        # Restore immediately even when the target timestep has already been
        # reached.  learn() resets OU noise during setup, so the checkpoint
        # callback restores this same state once more at training start when
        # continuation is required.
        restore_rng_state(model, resume_pipeline_state["rng_state"])
    else:
        model = build_model(
            namespace,
            variant=variant,
            train_env=train_env,
            seed=seed,
            args=args,
            tensorboard_log=tensorboard_log,
        )

    metadata = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "training_config_hash": config_hash,
        "training_config": config_payload,
        "variant": variant,
        "training_seed": int(seed),
        "timesteps": int(args.timesteps),
        "filtered_training": bool(spec["filtered"]),
        "lambda_delta": float(spec["lambda_delta"]),
        "lambda_intervention": float(spec["lambda_intervention"]),
        "correction_epsilon_normalized": float(args.correction_epsilon),
        "actor_correction_loss": bool(spec["actor_loss"]),
        "actor_correction_loss_mode": "local_projection_mse" if spec["actor_loss"] else "none",
        "lambda_bc_effective": float(args.lambda_bc) if spec["actor_loss"] else 0.0,
        "critic_action_mode": "raw",
        "policy_external_action_semantics": (
            "normalized_base_command" if variant == "a_nominal" else "physical_acceleration_command"
        ),
        "base_reward_config": reward_config,
        "env_config": env_config,
        "resumed_from": None if resume_bundle is None else str(resume_bundle),
        "tensorboard_log": str(tensorboard_log),
        "tensorboard_session": tensorboard_session,
        "tensorboard_parent_timestep": int(session_parent_step),
        "tensorboard_resume_policy": "new lineage segment per process; global timesteps are preserved",
    }
    config_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    metrics_callback = TrainingMetricsCallback(
        path=training_metrics_path,
        training_seed=seed,
        variant=variant,
    )
    if resume_pipeline_state is not None:
        metrics_callback.load_state_dict(resume_pipeline_state.get("metrics_callback_state", {}))
    checkpoint_callback = StrictCheckpointCallback(
        variant_dir=variant_dir,
        checkpoint_interval=int(args.checkpoint_interval),
        training_config_hash=config_hash,
        metrics_callback=metrics_callback,
        tracked_log_paths=[monitor_path, training_metrics_path],
        resume_pipeline_state=resume_pipeline_state,
    )
    callbacks = CallbackList([metrics_callback, checkpoint_callback])
    started = time.perf_counter()
    try:
        remaining = int(args.timesteps) - int(model.num_timesteps)
        if remaining < 0:
            raise RuntimeError(
                f"Checkpoint timestep {model.num_timesteps} exceeds target {args.timesteps}"
            )
        if remaining > 0:
            model.learn(
                total_timesteps=remaining,
                callback=callbacks,
                reset_num_timesteps=False,
                tb_log_name=tensorboard_session,
                log_interval=1,
                progress_bar=False,
            )
    finally:
        train_env.close()
    model.save(str(final_path))
    metadata["elapsed_sec"] = time.perf_counter() - started
    config_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return final_path


def load_model(variant: str, model_path: Path, device: str) -> Any:
    model_cls = model_class_for_variant(variant)
    return model_cls.load(str(model_path), device=device)


def make_evaluation_env(
    namespace: dict[str, Any],
    *,
    mode: str,
    scenario_seed: int,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    args: argparse.Namespace,
) -> gym.Env:
    if mode == "raw":
        env = make_raw_env(
            namespace,
            seed=scenario_seed,
            env_config=env_config,
            reward_config=reward_config,
        )
    else:
        env = make_cbf_env(
            namespace,
            seed=scenario_seed,
            env_config=env_config,
            reward_config=reward_config,
            lambda_delta=0.0,
            lambda_intervention=0.0,
            correction_epsilon=float(args.correction_epsilon),
            k0=float(args.k0),
            k1=float(args.k1),
            eps_side=float(args.eps_side),
        )
    if not bool(env.unwrapped.config.get("terminate_on_collision", False)):
        raise RuntimeError("Evaluation protocol requires terminate_on_collision=True")
    return ProtocolMetricsWrapper(env)


def policy_action_physical(
    *,
    model: Any | None,
    obs: np.ndarray,
    env_config: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    action_phys, _ = policy_action_and_q_physical(
        model=model,
        obs=obs,
        env_config=env_config,
        rng=rng,
        compute_q=False,
    )
    return action_phys


def _model_observation(model: Any, obs: np.ndarray) -> np.ndarray:
    """Put a raw evaluation observation on the actor/critic input scale."""

    model_obs = np.asarray(obs).copy()
    vec_normalize = getattr(model, "get_vec_normalize_env", lambda: None)()
    if vec_normalize is not None:
        model_obs = vec_normalize.normalize_obs(model_obs)
    return np.asarray(model_obs)


def _critic_reward(model: Any, reward: float) -> tuple[float, str]:
    """Return reward on the scale used to train the critic, without updating stats."""

    vec_normalize = getattr(model, "get_vec_normalize_env", lambda: None)()
    if vec_normalize is None or not bool(getattr(vec_normalize, "norm_reward", False)):
        return float(reward), "environment_raw"
    normalized = vec_normalize.normalize_reward(np.asarray([reward], dtype=np.float32))
    return float(np.asarray(normalized).reshape(-1)[0]), "vecnormalize_frozen"


def policy_action_and_q_physical(
    *,
    model: Any | None,
    obs: np.ndarray,
    env_config: dict[str, Any],
    rng: np.random.Generator,
    compute_q: bool,
) -> tuple[np.ndarray, float]:
    """Use one deterministic actor command for both environment stepping and Q."""

    if model is None:
        low, high = physical_bounds(env_config)
        return rng.uniform(low=low, high=high).astype(np.float32), np.nan

    model_obs = _model_observation(model, obs)
    action, _ = model.predict(model_obs, deterministic=True)
    action_array = np.asarray(action, dtype=np.float32).reshape(1, -1)
    action_phys = model_action_to_physical(model, action_array, env_config)
    if not compute_q:
        return action_phys, np.nan

    buffer_action = model.policy.scale_action(action_array)
    obs_tensor, _ = model.policy.obs_to_tensor(model_obs)
    action_tensor = th.as_tensor(buffer_action, device=model.device, dtype=th.float32)
    with th.no_grad():
        q_values = model.critic(obs_tensor, action_tensor)
        q_value = th.min(th.cat(q_values, dim=1), dim=1).values[0]
    return action_phys, float(q_value.detach().cpu().item())


def policy_q_value(model: Any, obs: np.ndarray) -> float:
    """Evaluate Q(s, pi(s)) with the same normalization and action semantics as replay."""

    model_obs = _model_observation(model, obs)
    action, _ = model.predict(model_obs, deterministic=True)
    action_array = np.asarray(action, dtype=np.float32).reshape(1, -1)
    buffer_action = model.policy.scale_action(action_array)
    obs_tensor, _ = model.policy.obs_to_tensor(model_obs)
    action_tensor = th.as_tensor(buffer_action, device=model.device, dtype=th.float32)
    with th.no_grad():
        q_values = model.critic(obs_tensor, action_tensor)
        q_value = th.min(th.cat(q_values, dim=1), dim=1).values[0]
    return float(q_value.detach().cpu().item())


def discounted_return_to_go(rewards: list[float], gamma: float) -> np.ndarray:
    """Finite discounted reward-to-go for one uninterrupted episode segment."""

    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    returns = np.empty(len(rewards), dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + float(gamma) * running
        returns[index] = running
    return returns


def append_critic_calibration_segment(
    sink: list[dict[str, Any]],
    *,
    anchors: list[dict[str, Any]],
    rewards: list[float],
    gamma: float,
    terminal_observed: bool,
    truncated: bool,
    collision_terminal: bool,
    tail_q: float,
    reward_scale: str,
    variant: str,
    mode: str,
    training_seed: int | None,
    scenario_seed: int,
    segment_index: int,
    censor_reason: str,
) -> None:
    """Finish calibration anchors without treating a time limit as a terminal."""

    if not anchors:
        return
    returns = discounted_return_to_go(rewards, gamma)
    segment_length = len(rewards)
    for anchor in anchors:
        anchor_step = int(anchor["anchor_segment_step"])
        remaining_steps = segment_length - anchor_step
        partial_return = float(returns[anchor_step])
        gamma_tail = float(float(gamma) ** remaining_steps)
        exact = bool(terminal_observed)
        empirical_return = partial_return if exact else np.nan
        bootstrapped_return = (
            partial_return
            if exact
            else (
                partial_return + gamma_tail * float(tail_q)
                if np.isfinite(tail_q)
                else np.nan
            )
        )
        q_value = float(anchor["q_value"])
        error = q_value - empirical_return if exact else np.nan
        bootstrap_error = (
            q_value - bootstrapped_return if np.isfinite(bootstrapped_return) else np.nan
        )
        sink.append(
            {
                "variant": str(variant),
                "mode": str(mode),
                "training_seed": np.nan if training_seed is None else int(training_seed),
                "scenario_seed": int(scenario_seed),
                "segment_index": int(segment_index),
                "anchor_segment_step": anchor_step,
                "anchor_global_step": int(anchor["anchor_global_step"]),
                "steps_to_boundary": int(remaining_steps),
                "gamma": float(gamma),
                "gamma_tail": gamma_tail,
                "q_value": q_value,
                "partial_discounted_return": partial_return,
                "empirical_discounted_return": empirical_return,
                "tail_q_value": float(tail_q) if np.isfinite(tail_q) else np.nan,
                "bootstrapped_discounted_return": bootstrapped_return,
                "calibration_error": error,
                "calibration_abs_error": abs(error) if np.isfinite(error) else np.nan,
                "calibration_squared_error": error * error if np.isfinite(error) else np.nan,
                "bootstrap_error": bootstrap_error,
                "terminal_mc_included": int(exact),
                "right_censored": int(not exact),
                "terminated": int(bool(terminal_observed)),
                "truncated": int(bool(truncated)),
                "collision_terminal": int(bool(collision_terminal)),
                "censor_reason": "" if exact else str(censor_reason),
                "reward_scale": str(reward_scale),
                "target_kind": "terminal_mc" if exact else "bootstrapped_sensitivity",
            }
        )


def evaluation_scenario_count(args: argparse.Namespace) -> int:
    value = args.eval_scenarios if args.eval_scenarios is not None else args.eval_episodes
    return int(value)


def evaluation_timestep_budget(args: argparse.Namespace) -> int:
    value = args.eval_timesteps if args.eval_timesteps is not None else args.eval_horizon
    return int(value)


def _shadow_cbf_action(
    namespace: dict[str, Any],
    env: gym.Env,
    raw_phys: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any], float]:
    """Evaluate the common CBF at the current state without stepping it."""

    required = ("get_ego_state", "get_neighbor_states", "cbf_filter_2d")
    if not all(callable(namespace.get(name)) for name in required):
        return np.asarray(raw_phys, dtype=np.float32).reshape(-1)[:2], {}, 0.0
    ego = namespace["get_ego_state"](env)
    neighbors = namespace["get_neighbor_states"](
        env,
        neighbor_range=float(namespace.get("CBF_NEIGHBOR_RANGE", env.unwrapped.config.get("sensing_range", 90.0))),
    )
    safe, info = namespace["cbf_filter_2d"](
        np.asarray(raw_phys, dtype=np.float32).reshape(-1)[:2],
        ego,
        neighbors,
        float(env.unwrapped.config["road_width"]),
        ax_bounds=namespace.get("CBF_AX_BOUNDS"),
        ay_bounds=namespace.get("CBF_AY_BOUNDS"),
        eps_side=float(args.eps_side),
        k0=float(args.k0),
        k1=float(args.k1),
        max_neighbor_constraints=namespace.get("CBF_MAX_NEIGHBOR_CONSTRAINTS"),
    )
    env_low, env_high = physical_bounds(env.unwrapped.config)
    ax_bounds = namespace.get("CBF_AX_BOUNDS", (float(env_low[0]), float(env_high[0])))
    ay_bounds = namespace.get("CBF_AY_BOUNDS", (float(env_low[1]), float(env_high[1])))
    filter_low = np.asarray([float(ax_bounds[0]), float(ay_bounds[0])], dtype=np.float32)
    filter_high = np.asarray([float(ax_bounds[1]), float(ay_bounds[1])], dtype=np.float32)
    normalized_norm = box_scaled_delta_norm(raw_phys, safe, filter_low, filter_high)
    return np.asarray(safe, dtype=np.float32).reshape(-1)[:2], dict(info), normalized_norm


def linearized_ttc_from_barriers(
    h_and_dot: Iterable[tuple[float, float]],
    *,
    cap_s: float,
) -> float:
    """Return a finite, capped linearized time to the nearest CBF boundary.

    A violated constraint has zero TTC.  A valid constraint that is not
    approaching the boundary contributes the reporting cap rather than
    infinity, so scenario and seed means retain a stable, unconditional
    interpretation.
    """

    cap_s = float(cap_s)
    if not np.isfinite(cap_s) or cap_s <= 0.0:
        raise ValueError("cap_s must be finite and positive")
    pairs = [
        (float(h_value), float(h_dot))
        for h_value, h_dot in h_and_dot
        if np.isfinite(float(h_value)) and np.isfinite(float(h_dot))
    ]
    if not pairs:
        return float(np.nan)
    if any(h_value <= 0.0 for h_value, _ in pairs):
        return 0.0
    closing_times = [
        h_value / -h_dot
        for h_value, h_dot in pairs
        if h_dot < -1e-9
    ]
    return float(min(cap_s, min(closing_times))) if closing_times else cap_s


def cbf_state_occupancy_metrics(
    namespace: dict[str, Any],
    env: gym.Env,
    *,
    eps_side: float,
    ttc_cap_s: float = 30.0,
) -> dict[str, float]:
    """Compute pre-action occupancy diagnostics for one exact simulator state."""

    result = {
        "h_min": np.nan,
        "h_dot": np.nan,
        "ttc_cbf_linearized_s": np.nan,
        "vehicle_spacing_m": np.nan,
        "surface_clearance_m": np.nan,
        "neighbor_count": np.nan,
        "traffic_density_per_km": np.nan,
    }
    kpi_fn = namespace.get("kpi_neighbor_and_h_metrics")
    if callable(kpi_fn):
        kpi = kpi_fn(env, eps_side=float(eps_side))
        pair_h = _as_float(kpi.get("kpi_pairwise_h_min", kpi.get("kpi_h_min")))
        boundary_h = _as_float(kpi.get("kpi_boundary_h_min"))
        result.update(
            {
                "h_min": _min([pair_h, boundary_h]),
                "vehicle_spacing_m": _as_float(kpi.get("kpi_min_center_distance_m")),
                "surface_clearance_m": pair_h,
                "neighbor_count": _as_float(kpi.get("kpi_neighbor_count")),
                "traffic_density_per_km": _as_float(kpi.get("kpi_neighbor_density_per_km")),
            }
        )

    required = ("get_ego_state", "get_neighbor_states", "pairwise_relative_state", "centerline_barrier_derivatives")
    if not all(callable(namespace.get(name)) for name in required):
        return result
    ego = namespace["get_ego_state"](env)
    neighbors = namespace["get_neighbor_states"](
        env,
        neighbor_range=float(namespace.get("CBF_NEIGHBOR_RANGE", env.unwrapped.config.get("sensing_range", 90.0))),
    )
    h_and_dot: list[tuple[float, float]] = []
    center_distances: list[float] = []
    for neighbor in neighbors:
        try:
            dx, dy, dvx, dvy = namespace["pairwise_relative_state"](ego, neighbor)
            p = np.asarray([dx, dy], dtype=float)
            relative_velocity = np.asarray([dvx, dvy], dtype=float)
            h, gradient, _, center_distance, _, _ = namespace["centerline_barrier_derivatives"](
                p,
                ego,
                neighbor,
                float(eps_side),
            )
            h_value = float(h)
            h_dot = float(np.asarray(gradient, dtype=float) @ relative_velocity)
        except (FloatingPointError, OverflowError, TypeError, ValueError, ZeroDivisionError):
            continue
        if not np.isfinite(h_value) or not np.isfinite(h_dot):
            continue
        h_and_dot.append((h_value, h_dot))
        if np.isfinite(float(center_distance)):
            center_distances.append(float(center_distance))
    ego_y = float(ego.get("y", np.nan))
    ego_vy = float(ego.get("vy", np.nan))
    ego_half_width = 0.5 * float(ego.get("width", 0.0))
    road_width = float(env.unwrapped.config.get("road_width", np.nan))
    if np.isfinite(ego_y) and np.isfinite(ego_vy) and np.isfinite(road_width):
        h_and_dot.extend(
            [
                (ego_y - ego_half_width, ego_vy),
                (road_width - ego_half_width - ego_y, -ego_vy),
            ]
        )
    if h_and_dot:
        active_h, active_h_dot = min(h_and_dot, key=lambda pair: pair[0])
        result["h_min"] = float(active_h)
        result["h_dot"] = float(active_h_dot)
        result["ttc_cbf_linearized_s"] = linearized_ttc_from_barriers(
            h_and_dot,
            cap_s=float(ttc_cap_s),
        )
    if center_distances:
        result["vehicle_spacing_m"] = float(min(center_distances))
    return result


def evaluate_scenario(
    namespace: dict[str, Any],
    *,
    model: Any | None,
    variant: str,
    mode: str,
    scenario_seed: int,
    training_seed: int | None,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    args: argparse.Namespace,
    critic_calibration_samples: Optional[list[dict[str, Any]]] = None,
    critic_calibration_stride: int = 20,
    occupancy_samples: Optional[list[dict[str, Any]]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = make_evaluation_env(
        namespace,
        mode=mode,
        scenario_seed=scenario_seed,
        env_config=env_config,
        reward_config=reward_config,
        args=args,
    )
    try:
        obs, _ = env.reset(seed=int(scenario_seed))
        state_hash = initial_state_hash(env)
        rng = np.random.default_rng(int(scenario_seed) + 7_919)
        timestep_budget = evaluation_timestep_budget(args)
        policy_dt = _policy_dt(env)
        rewards: list[float] = []
        speed_errors: list[float] = []
        jerk_norms: list[float] = []
        h_values: list[float] = []
        h_dot_values: list[float] = []
        ttc_values: list[float] = []
        vehicle_spacings: list[float] = []
        surface_clearances: list[float] = []
        neighbor_counts: list[float] = []
        traffic_densities: list[float] = []
        intervention_events: list[float] = []
        delta_norms: list[float] = []
        shadow_intervention_events: list[float] = []
        shadow_delta_norms: list[float] = []
        shadow_delta_phys: list[float] = []
        shadow_delta_x: list[float] = []
        shadow_delta_y: list[float] = []
        qp_failures: list[float] = []
        qp_fallbacks: list[float] = []
        shadow_qp_failures: list[float] = []
        shadow_qp_fallbacks: list[float] = []
        nominal_saturations: list[float] = []
        safe_saturations: list[float] = []
        executed_saturations: list[float] = []
        common_form1_rewards: list[float] = []
        target_speed_errors: list[float] = []
        target_lateral_errors: list[float] = []
        formulation_cf_values: list[float] = []
        boundary_costs: list[float] = []
        normalized_acceleration_sq: list[float] = []
        normalized_acceleration_delta_sq: list[float] = []
        normalized_jerk_command_sq: list[float] = []
        policy_command_saturations: list[float] = []
        acceleration_integrator_clip_rates: list[float] = []
        reference_endpoint_rates: list[float] = []
        controller_saturation_rates: list[float] = []
        distinct_ego_collision_events = 0
        ego_collision_incidents = 0
        ego_collision_active_timesteps = 0
        distinct_all_pair_collision_events = 0
        active_collision_pair_timesteps = 0
        total_distance_m = 0.0
        collision_transition_return = 0.0
        collision_transition_timesteps = 0
        post_collision_return = 0.0
        post_collision_timesteps = 0
        correction_return = 0.0
        collision_survived_without_reset = 0
        active_collision_without_event = 0
        event_without_active_collision = 0
        first_collision_step: Optional[int] = None
        time_to_first_collision_s = np.nan
        distance_to_first_collision_m = np.nan
        reset_calls_total = 1
        resets_after_collision = 0
        resets_after_truncation_only = 0
        resets_after_other_terminal = 0
        segments: list[dict[str, Any]] = []
        segment_index = 0
        segment_steps = 0
        segment_return = 0.0
        segment_distance_m = 0.0
        segment_collision_events = 0
        # The simulator resets its executed acceleration to zero, so the first
        # transition's acceleration change is a real jerk event from zero.
        previous_acceleration: np.ndarray | None = np.zeros(2, dtype=float)
        calibration_enabled = critic_calibration_samples is not None and model is not None
        if int(critic_calibration_stride) <= 0:
            raise ValueError("critic_calibration_stride must be positive")
        calibration_gamma = float(getattr(model, "gamma", np.nan)) if model is not None else np.nan
        segment_calibration_rewards: list[float] = []
        segment_calibration_anchors: list[dict[str, Any]] = []
        calibration_reward_scale = "environment_raw"
        evaluation_action_hook = namespace.get("ppo_formulation_evaluation_action")

        for step in range(timestep_budget):
            calibration_anchor = bool(
                calibration_enabled and segment_steps % int(critic_calibration_stride) == 0
            )
            if callable(evaluation_action_hook):
                if mode != "raw":
                    raise RuntimeError(
                        "Formulation action hooks are supported only for raw evaluation"
                    )
                if calibration_anchor:
                    raise RuntimeError(
                        "Formulation action hooks do not support off-policy Q calibration"
                    )
                environment_action = np.asarray(
                    evaluation_action_hook(model=model, obs=obs, rng=rng),
                    dtype=np.float32,
                ).reshape(-1)[:2]
                raw_phys = np.full(2, np.nan, dtype=np.float32)
                anchor_q = np.nan
            else:
                environment_action = None
                raw_phys, anchor_q = policy_action_and_q_physical(
                    model=model,
                    obs=obs,
                    env_config=env_config,
                    rng=rng,
                    compute_q=calibration_anchor,
                )
            if calibration_anchor:
                segment_calibration_anchors.append(
                    {
                        "anchor_segment_step": int(segment_steps),
                        "anchor_global_step": int(step),
                        "q_value": float(anchor_q),
                    }
                )
            pre_state_metrics = cbf_state_occupancy_metrics(
                namespace,
                env,
                eps_side=float(args.eps_side),
                ttc_cap_s=float(getattr(args, "ttc_cap", 30.0)),
            )
            shadow_safe_phys = np.asarray(raw_phys, dtype=np.float32).reshape(-1)[:2]
            shadow_filter_info: dict[str, Any] = {}
            shadow_delta_norm = 0.0
            if environment_action is None and np.all(np.isfinite(raw_phys)):
                shadow_safe_phys, shadow_filter_info, shadow_delta_norm = _shadow_cbf_action(
                    namespace,
                    env,
                    raw_phys,
                    args,
                )
            shadow_delta = shadow_safe_phys - np.asarray(raw_phys, dtype=np.float32).reshape(-1)[:2]
            shadow_intervention = float(shadow_delta_norm > float(args.correction_epsilon))
            shadow_qp_success = bool(shadow_filter_info.get("qp_success", True))
            low, high = physical_bounds(env_config)
            physical_tolerance = 1e-3 * np.maximum(high - low, 1.0)
            if environment_action is None:
                nominal_saturations.append(
                    float(np.mean((np.abs(raw_phys - low) <= physical_tolerance) | (np.abs(raw_phys - high) <= physical_tolerance)))
                )
            if mode == "cbf":
                obs, reward, terminated, truncated, info = env.step(raw_phys)
                info = dict(info)
                safe_phys = np.asarray(
                    [
                        info.get("cbf_a_safe_x", raw_phys[0]),
                        info.get("cbf_a_safe_y", raw_phys[1]),
                    ],
                    dtype=np.float32,
                )
                delta_norm = _as_float(info.get("cbf_correction_norm_normalized"), default=np.nan)
                if not np.isfinite(delta_norm):
                    ax_bounds = namespace.get("CBF_AX_BOUNDS", (float(low[0]), float(high[0])))
                    ay_bounds = namespace.get("CBF_AY_BOUNDS", (float(low[1]), float(high[1])))
                    delta_norm = box_scaled_delta_norm(
                        raw_phys,
                        safe_phys,
                        np.asarray([ax_bounds[0], ay_bounds[0]], dtype=np.float32),
                        np.asarray([ax_bounds[1], ay_bounds[1]], dtype=np.float32),
                    )
                intervention = float(delta_norm > float(args.correction_epsilon))
                qp_success = bool(info.get("cbf_qp_success", True))
                qp_failures.append(float(not qp_success))
                qp_fallbacks.append(float(info.get("cbf_fallback_used", not qp_success)))
                # Prefer the wrapper's identical pre-state result for applied
                # diagnostics, while retaining the explicit shadow call for
                # parity with raw deployment.
                shadow_safe_phys = safe_phys.copy()
                shadow_delta = shadow_safe_phys - raw_phys
                shadow_delta_norm = float(delta_norm)
                shadow_intervention = float(intervention)
                shadow_qp_success = qp_success
            else:
                if environment_action is None:
                    obs, reward, terminated, truncated, info = env.step(
                        physical_to_normalized(raw_phys, env_config)
                    )
                else:
                    obs, reward, terminated, truncated, info = env.step(environment_action)
                info = dict(info)
                if environment_action is not None:
                    raw_phys = np.asarray(
                        [
                            info.get("formulation_executed_ax", np.nan),
                            info.get("formulation_executed_ay", np.nan),
                        ],
                        dtype=np.float32,
                    )
                    if not np.all(np.isfinite(raw_phys)):
                        executed_fallback = np.asarray(
                            getattr(env.unwrapped, "_last_accelerations", np.zeros((1, 2))),
                            dtype=float,
                        )
                        raw_phys = (
                            executed_fallback.reshape((-1, 2))[0, :2].astype(np.float32)
                            if executed_fallback.size >= 2
                            else np.zeros(2, dtype=np.float32)
                        )
                    nominal_saturations.append(
                        _as_float(
                            info.get("formulation_policy_command_saturation_rate"),
                            default=float(
                                np.mean(
                                    (np.abs(raw_phys - low) <= physical_tolerance)
                                    | (np.abs(raw_phys - high) <= physical_tolerance)
                                )
                            ),
                        )
                    )
                safe_phys = shadow_safe_phys.copy()
                delta_norm = 0.0
                intervention = 0.0

            shadow_delta_norms.append(float(shadow_delta_norm))
            shadow_delta_phys.append(float(np.linalg.norm(shadow_delta)))
            shadow_delta_x.append(float(shadow_delta[0]))
            shadow_delta_y.append(float(shadow_delta[1]))
            shadow_intervention_events.append(float(shadow_intervention))
            shadow_qp_failures.append(float(not shadow_qp_success))
            shadow_qp_fallbacks.append(
                float(shadow_filter_info.get("fallback_used", not shadow_qp_success))
            )

            base = env.unwrapped
            distance_step_m = _as_float(info.get("pipeline_distance_step_m"), default=0.0)
            total_distance_m += distance_step_m
            segment_distance_m += distance_step_m
            speed_errors.append(abs(float(base.vehicle.vx) - float(base.vehicle.desired_speed)))
            common_form1_rewards.append(
                _as_float(info.get("formulation_common_form1_reward"), default=np.nan)
            )
            target_speed_errors.append(
                _as_float(info.get("formulation_abs_target_speed_error"), default=np.nan)
            )
            target_lateral_errors.append(
                _as_float(info.get("formulation_abs_target_lateral_error_m"), default=np.nan)
            )
            formulation_cf_values.append(
                _as_float(info.get("formulation_cf"), default=np.nan)
            )
            boundary_costs.append(
                _as_float(info.get("formulation_boundary_cost"), default=np.nan)
            )
            normalized_acceleration_sq.append(
                _as_float(info.get("formulation_normalized_acceleration_sq"), default=np.nan)
            )
            normalized_acceleration_delta_sq.append(
                _as_float(
                    info.get("formulation_normalized_acceleration_delta_sq"),
                    default=np.nan,
                )
            )
            normalized_jerk_command_sq.append(
                _as_float(info.get("formulation_normalized_jerk_command_sq"), default=np.nan)
            )
            policy_command_saturations.append(
                _as_float(
                    info.get("formulation_policy_command_saturation_rate"),
                    default=np.nan,
                )
            )
            acceleration_integrator_clip_rates.append(
                _as_float(
                    info.get("formulation_acceleration_integrator_clip_rate"),
                    default=np.nan,
                )
            )
            reference_endpoint_rates.append(
                _as_float(info.get("formulation_reference_endpoint_rate"), default=np.nan)
            )
            controller_saturation_rates.append(
                _as_float(
                    info.get("formulation_controller_saturation_rate"),
                    default=np.nan,
                )
            )
            accelerations = np.asarray(getattr(base, "_last_accelerations", np.empty((0, 2))), dtype=float)
            if accelerations.ndim == 2 and accelerations.shape[0] > 0:
                acceleration = accelerations[0, :2]
                if previous_acceleration is not None:
                    jerk_norms.append(float(np.linalg.norm(acceleration - previous_acceleration) / max(policy_dt, 1e-6)))
                previous_acceleration = acceleration.copy()
            h_values.append(_as_float(pre_state_metrics.get("h_min")))
            h_dot_values.append(_as_float(pre_state_metrics.get("h_dot")))
            ttc_values.append(
                float(pre_state_metrics.get("ttc_cbf_linearized_s", np.inf))
            )
            vehicle_spacings.append(
                _as_float(pre_state_metrics.get("vehicle_spacing_m"))
            )
            surface_clearances.append(
                _as_float(pre_state_metrics.get("surface_clearance_m"))
            )
            neighbor_counts.append(_as_float(pre_state_metrics.get("neighbor_count")))
            traffic_densities.append(
                _as_float(pre_state_metrics.get("traffic_density_per_km"))
            )
            rewards.append(float(reward))
            if calibration_enabled:
                critic_reward, calibration_reward_scale = _critic_reward(model, float(reward))
                segment_calibration_rewards.append(float(critic_reward))
            segment_return += float(reward)
            segment_steps += 1
            delta_norms.append(float(delta_norm))
            intervention_events.append(intervention)
            events_step = max(int(info.get("ego_collision_events", 0)), 0)
            active_collision = bool(info.get("ego_collision", False))
            all_pair_events_step = max(int(info.get("collisions", 0)), 0)
            active_pairs_step = max(int(info.get("active_collisions", 0)), 0)
            distinct_ego_collision_events += events_step
            segment_collision_events += events_step
            ego_collision_incidents += int(events_step > 0)
            ego_collision_active_timesteps += int(active_collision)
            distinct_all_pair_collision_events += all_pair_events_step
            active_collision_pair_timesteps += active_pairs_step
            active_collision_without_event += int(active_collision and events_step == 0)
            event_without_active_collision += int(events_step > 0 and not active_collision)
            if bool(info.get("pipeline_collision_transition", False)):
                collision_transition_return += float(reward)
                collision_transition_timesteps += 1
            if bool(info.get("pipeline_post_collision_timestep", False)):
                post_collision_return += float(reward)
                post_collision_timesteps += 1
            collision_survived_without_reset += int(
                bool(info.get("pipeline_collision_survived_without_reset", False))
            )
            correction_return += _as_float(info.get("cbf_correction_reward"), default=0.0)
            if first_collision_step is None and (active_collision or events_step > 0):
                first_collision_step = step + 1
                time_to_first_collision_s = float((step + 1) * policy_dt)
                distance_to_first_collision_m = float(total_distance_m)

            safe_saturations.append(
                float(np.mean((np.abs(safe_phys - low) <= physical_tolerance) | (np.abs(safe_phys - high) <= physical_tolerance)))
            )
            executed = np.asarray(getattr(base, "_last_accelerations", np.zeros((1, 2))), dtype=float)
            executed_action = executed.reshape((-1, 2))[0, :2] if executed.size >= 2 else safe_phys
            executed_saturations.append(
                float(
                    np.mean(
                        (np.abs(executed_action - low) <= physical_tolerance)
                        | (np.abs(executed_action - high) <= physical_tolerance)
                    )
                )
            )
            if occupancy_samples is not None:
                active_types = shadow_filter_info.get("active_constraint_types", ())
                if isinstance(active_types, str):
                    active_type_label = active_types
                else:
                    active_type_label = "|".join(str(value) for value in active_types)
                occupancy_samples.append(
                    {
                        "variant": variant,
                        "mode": mode,
                        "training_seed": np.nan if training_seed is None else int(training_seed),
                        "scenario_seed": int(scenario_seed),
                        "step": int(step),
                        **{key: float(value) for key, value in pre_state_metrics.items()},
                        "near_boundary": float(
                            abs(_as_float(pre_state_metrics.get("h_min"), default=np.inf))
                            <= float(getattr(args, "near_boundary_h", 0.5))
                        ),
                        "raw_ax": float(raw_phys[0]),
                        "raw_ay": float(raw_phys[1]),
                        "shadow_safe_ax": float(shadow_safe_phys[0]),
                        "shadow_safe_ay": float(shadow_safe_phys[1]),
                        "executed_ax": float(executed_action[0]),
                        "executed_ay": float(executed_action[1]),
                        "shadow_delta_ax": float(shadow_delta[0]),
                        "shadow_delta_ay": float(shadow_delta[1]),
                        "shadow_delta_norm_physical": float(np.linalg.norm(shadow_delta)),
                        "shadow_delta_norm_scaled": float(shadow_delta_norm),
                        "applied_delta_norm_scaled": float(delta_norm),
                        "shadow_intervened": float(shadow_intervention),
                        "applied_intervened": float(intervention),
                        "active_constraint_type": active_type_label or "none",
                        "active_constraint_count": int(
                            shadow_filter_info.get("cbf_active_constraint_count", 0)
                        ),
                        "ego_collision_events": int(info.get("ego_collision_events", 0)),
                    }
                )

            if terminated or truncated:
                collision_terminal = bool(active_collision or events_step > 0)
                if calibration_enabled:
                    tail_q = (
                        np.nan if bool(terminated) else policy_q_value(model, np.asarray(obs))
                    )
                    append_critic_calibration_segment(
                        critic_calibration_samples,
                        anchors=segment_calibration_anchors,
                        rewards=segment_calibration_rewards,
                        gamma=calibration_gamma,
                        terminal_observed=bool(terminated),
                        truncated=bool(truncated),
                        collision_terminal=collision_terminal,
                        tail_q=tail_q,
                        reward_scale=calibration_reward_scale,
                        variant=variant,
                        mode=mode,
                        training_seed=training_seed,
                        scenario_seed=scenario_seed,
                        segment_index=segment_index,
                        censor_reason="environment_truncation" if truncated else "nonterminal_boundary",
                    )
                segments.append(
                    {
                        "variant": variant,
                        "mode": mode,
                        "training_seed": np.nan if training_seed is None else int(training_seed),
                        "scenario_seed": int(scenario_seed),
                        "segment_index": int(segment_index),
                        "steps": int(segment_steps),
                        "return": float(segment_return),
                        "distance_m": float(segment_distance_m),
                        "distinct_ego_collision_events": int(segment_collision_events),
                        "terminated": int(bool(terminated)),
                        "truncated": int(bool(truncated)),
                        "collision_terminal": int(collision_terminal),
                        "right_censored": 0,
                    }
                )
                segment_index += 1
                segment_steps = 0
                segment_return = 0.0
                segment_distance_m = 0.0
                segment_collision_events = 0
                previous_acceleration = np.zeros(2, dtype=float)
                segment_calibration_rewards = []
                segment_calibration_anchors = []
                if step + 1 < timestep_budget:
                    if collision_terminal:
                        resets_after_collision += 1
                    elif truncated:
                        resets_after_truncation_only += 1
                    else:
                        resets_after_other_terminal += 1
                    obs, _ = env.reset()
                    reset_calls_total += 1

        if segment_steps > 0:
            if calibration_enabled:
                append_critic_calibration_segment(
                    critic_calibration_samples,
                    anchors=segment_calibration_anchors,
                    rewards=segment_calibration_rewards,
                    gamma=calibration_gamma,
                    terminal_observed=False,
                    truncated=False,
                    collision_terminal=False,
                    tail_q=policy_q_value(model, np.asarray(obs)),
                    reward_scale=calibration_reward_scale,
                    variant=variant,
                    mode=mode,
                    training_seed=training_seed,
                    scenario_seed=scenario_seed,
                    segment_index=segment_index,
                    censor_reason="evaluation_budget",
                )
            segments.append(
                {
                    "variant": variant,
                    "mode": mode,
                    "training_seed": np.nan if training_seed is None else int(training_seed),
                    "scenario_seed": int(scenario_seed),
                    "segment_index": int(segment_index),
                    "steps": int(segment_steps),
                    "return": float(segment_return),
                    "distance_m": float(segment_distance_m),
                    "distinct_ego_collision_events": int(segment_collision_events),
                    "terminated": 0,
                    "truncated": 0,
                    "collision_terminal": 0,
                    "right_censored": 1,
                }
            )

        segment_lengths = [float(row["steps"]) for row in segments]
        completed_segments = [row for row in segments if not bool(row["right_censored"])]
        first_collision_observed = first_collision_step is not None
        total_return = float(np.sum(rewards))
        common_form1_total_return = float(np.nansum(common_form1_rewards))
        finite_h_values = _finite(h_values)
        near_boundary_threshold = float(getattr(args, "near_boundary_h", 0.5))
        near_boundary_steps = int(
            np.sum(np.abs(finite_h_values) <= near_boundary_threshold)
        )
        row = {
            "variant": variant,
            "mode": mode,
            "training_seed": np.nan if training_seed is None else int(training_seed),
            "scenario_seed": int(scenario_seed),
            "initial_state_hash": state_hash,
            "timestep_budget": int(timestep_budget),
            "timesteps": int(len(rewards)),
            "steps": int(len(rewards)),
            "total_time_s": float(len(rewards) * policy_dt),
            "total_return": total_return,
            "return": total_return,
            "task_return": float(total_return - correction_return),
            "correction_return": float(correction_return),
            "return_per_timestep": float(total_return / max(len(rewards), 1)),
            "common_form1_total_return": common_form1_total_return,
            "common_form1_return_per_timestep": _mean(
                common_form1_rewards, default=np.nan
            ),
            "distinct_ego_collision_events": int(distinct_ego_collision_events),
            "ego_collisions": int(distinct_ego_collision_events),
            "ego_collision_incidents": int(ego_collision_incidents),
            "ego_collision_active_timesteps": int(ego_collision_active_timesteps),
            "distinct_all_pair_collision_events": int(distinct_all_pair_collision_events),
            "active_collision_pair_timesteps": int(active_collision_pair_timesteps),
            "total_distance_m": float(total_distance_m),
            "distance_m": float(total_distance_m),
            "distance_per_collision_m": _distance_per_collision(
                total_distance_m, distinct_ego_collision_events
            ),
            "distance_per_collision_right_censored": int(distinct_ego_collision_events == 0),
            "distance_per_collision_exposure_bound_m": _distance_per_collision_exposure_bound(
                total_distance_m, distinct_ego_collision_events
            ),
            "collision_events_per_m": _ratio(distinct_ego_collision_events, total_distance_m),
            "ego_collisions_per_km": _collisions_per_km(
                distinct_ego_collision_events, total_distance_m
            ),
            "first_collision_observed": int(first_collision_observed),
            "first_collision_step": np.nan if first_collision_step is None else int(first_collision_step),
            "time_to_first_collision_s": float(time_to_first_collision_s),
            "distance_to_first_collision_m": float(distance_to_first_collision_m),
            "first_collision_censor_time_s": float(len(rewards) * policy_dt),
            "first_collision_censor_distance_m": float(total_distance_m),
            "collision_transition_timesteps": int(collision_transition_timesteps),
            "collision_transition_return": float(collision_transition_return),
            "post_collision_timesteps": int(post_collision_timesteps),
            "post_collision_return": float(post_collision_return),
            "reset_calls_total": int(reset_calls_total),
            "resets_after_collision": int(resets_after_collision),
            "resets_after_truncation_only": int(resets_after_truncation_only),
            "resets_after_other_terminal": int(resets_after_other_terminal),
            "episode_segments": int(len(segments)),
            "completed_segments": int(len(completed_segments)),
            "right_censored_segments": int(len(segments) - len(completed_segments)),
            "episode_length_sum": float(np.sum(segment_lengths)),
            "episode_length_mean": _mean(segment_lengths, default=np.nan),
            "completed_episode_length_mean": _mean(
                [float(item["steps"]) for item in completed_segments], default=np.nan
            ),
            "episode_return_mean": _mean([float(item["return"]) for item in segments], default=np.nan),
            "collision_survived_without_reset": int(collision_survived_without_reset),
            "active_collision_without_event": int(active_collision_without_event),
            "event_without_active_collision": int(event_without_active_collision),
            "h_min": _min(h_values),
            "h_violation_rate": _mean([float(value < 0.0) for value in _finite(h_values).tolist()], default=np.nan),
            "near_boundary_h_threshold": near_boundary_threshold,
            "near_boundary_steps": near_boundary_steps,
            "near_boundary_rate": float(near_boundary_steps / max(len(rewards), 1)),
            "time_near_boundary_s": float(near_boundary_steps * policy_dt),
            "mean_h_dot": _mean(h_dot_values),
            "min_ttc_s": _min(ttc_values, default=np.inf),
            "mean_ttc_s": _mean(ttc_values, default=np.inf),
            "mean_vehicle_spacing_m": _mean(vehicle_spacings),
            "min_vehicle_spacing_m": _min(vehicle_spacings),
            "mean_surface_clearance_m": _mean(surface_clearances),
            "mean_neighbor_count": _mean(neighbor_counts),
            "mean_traffic_density_per_km": _mean(traffic_densities),
            "mean_abs_speed_error": _mean(speed_errors, default=0.0),
            "mean_abs_target_speed_error": _mean(
                target_speed_errors, default=np.nan
            ),
            "mean_abs_target_lateral_error_m": _mean(
                target_lateral_errors, default=np.nan
            ),
            "mean_formulation_cf": _mean(formulation_cf_values, default=np.nan),
            "mean_boundary_cost": _mean(boundary_costs, default=np.nan),
            "mean_normalized_acceleration_sq": _mean(
                normalized_acceleration_sq, default=np.nan
            ),
            "mean_normalized_acceleration_delta_sq": _mean(
                normalized_acceleration_delta_sq, default=np.nan
            ),
            "mean_normalized_jerk_command_sq": _mean(
                normalized_jerk_command_sq, default=np.nan
            ),
            "policy_command_saturation_rate": _mean(
                policy_command_saturations, default=np.nan
            ),
            "acceleration_integrator_clip_rate": _mean(
                acceleration_integrator_clip_rates, default=np.nan
            ),
            "reference_endpoint_rate": _mean(
                reference_endpoint_rates, default=np.nan
            ),
            "controller_saturation_rate": _mean(
                controller_saturation_rates, default=np.nan
            ),
            "mean_jerk_norm": _mean(jerk_norms, default=0.0),
            "IR": _mean(intervention_events, default=0.0),
            "mean_delta_a": _mean(delta_norms, default=0.0),
            "p95_delta_a": _p95(delta_norms, default=0.0),
            "shadow_IR": _mean(shadow_intervention_events, default=0.0),
            "shadow_mean_delta_a": _mean(shadow_delta_norms, default=0.0),
            "shadow_p95_delta_a": _p95(shadow_delta_norms, default=0.0),
            "shadow_mean_delta_a_physical": _mean(shadow_delta_phys, default=0.0),
            "shadow_mean_delta_ax": _mean(shadow_delta_x, default=0.0),
            "shadow_mean_delta_ay": _mean(shadow_delta_y, default=0.0),
            "qp_failure_rate": _mean(qp_failures),
            "qp_fallback_rate": _mean(qp_fallbacks),
            "shadow_qp_failure_rate": _mean(shadow_qp_failures, default=0.0),
            "shadow_qp_fallback_rate": _mean(shadow_qp_fallbacks, default=0.0),
            "nominal_action_saturation_rate": _mean(nominal_saturations, default=0.0),
            "safe_action_saturation_rate": _mean(safe_saturations, default=0.0),
            "executed_action_saturation_rate": _mean(executed_saturations, default=0.0),
        }
        return row, segments
    finally:
        env.close()


def evaluate_models(
    namespace: dict[str, Any],
    *,
    models: dict[tuple[int, str], Any],
    args: argparse.Namespace,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    occupancy_rows: list[dict[str, Any]] = []
    scenario_seeds = [
        int(args.eval_seed_start) + index for index in range(evaluation_scenario_count(args))
    ]
    manifest = {
        "scenario_seeds": scenario_seeds,
        "timestep_budget_per_scenario": evaluation_timestep_budget(args),
        "terminate_on_collision": True,
        "reset_immediately_after_terminal": True,
        "primary_safety_metric": {
            "name": "distance_per_collision_m",
            "formula": "total_distance_m / distinct_ego_collision_events",
            "units": "m/collision",
            "zero_collision_value": "infinity (right-censored)",
            "finite_censored_companion": (
                "distance_per_collision_exposure_bound_m equals driven distance when collision-free"
            ),
        },
        "inverse_safety_metrics": {
            "collision_events_per_m": "distinct_ego_collision_events / total_distance_m",
            "ego_collisions_per_km": "1000 * distinct_ego_collision_events / total_distance_m",
        },
        "traffic_model": active_traffic_model(env_config),
        "env_config": env_config,
        "base_reward_config": reward_config,
        "notes": [
            "Reset states are paired across all cells; traffic trajectories become policy-dependent after the first action.",
            "IR/mean_delta_a are applied-filter metrics and are zero in raw mode; shadow_* fields evaluate the same CBF without changing raw execution.",
            "evaluation_occupancy_steps.csv records pre-action h, h_dot, capped linearized TTC, spacing, and density; action/outcome fields refer to the following transition.",
            "Raw, shadow-safe command, and actually executed acceleration are logged separately because environment dynamics may add boundary assistance.",
        ],
        "linearized_ttc_cap_s": float(args.ttc_cap),
    }
    (output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    expected_hashes: dict[int, str] = {}
    for training_seed in args.seeds:
        for variant in VARIANTS:
            model = models[(int(training_seed), variant)]
            for mode in MODES:
                for scenario_seed in scenario_seeds:
                    row, segments = evaluate_scenario(
                        namespace,
                        model=model,
                        variant=variant,
                        mode=mode,
                        scenario_seed=scenario_seed,
                        training_seed=int(training_seed),
                        env_config=env_config,
                        reward_config=reward_config,
                        args=args,
                        occupancy_samples=occupancy_rows,
                    )
                    expected = expected_hashes.setdefault(scenario_seed, str(row["initial_state_hash"]))
                    if str(row["initial_state_hash"]) != expected:
                        raise RuntimeError(f"Scenario {scenario_seed} did not reset to the paired initial state")
                    rows.append(row)
                    segment_rows.extend(segments)
    for mode in MODES:
        for scenario_seed in scenario_seeds:
            row, segments = evaluate_scenario(
                namespace,
                model=None,
                variant=RANDOM_VARIANT,
                mode=mode,
                scenario_seed=scenario_seed,
                training_seed=None,
                env_config=env_config,
                reward_config=reward_config,
                args=args,
                occupancy_samples=occupancy_rows,
            )
            expected = expected_hashes.setdefault(scenario_seed, str(row["initial_state_hash"]))
            if str(row["initial_state_hash"]) != expected:
                raise RuntimeError(f"Random-policy scenario {scenario_seed} did not reset to the paired initial state")
            rows.append(row)
            segment_rows.extend(segments)
    scenarios = pd.DataFrame(rows)
    scenarios.to_csv(output_dir / "evaluation_scenarios.csv", index=False)
    # Preserve the old filename for the existing report builder.  Each row is
    # now a fixed-budget scenario, not one collision-terminated episode.
    scenarios.to_csv(output_dir / "evaluation_episodes.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(output_dir / "evaluation_segments.csv", index=False)
    occupancy = pd.DataFrame(occupancy_rows)
    occupancy.to_csv(output_dir / "evaluation_occupancy_steps.csv", index=False)
    return scenarios, occupancy


def checkpoint_index(
    namespace: dict[str, Any],
    *,
    args: argparse.Namespace,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    output_dir: Path,
) -> pd.DataFrame:
    """Validate and index every callback checkpoint, including the final step."""

    rows: list[dict[str, Any]] = []
    project_root = Path(namespace["PROJECT_ROOT"])
    for training_seed in args.seeds:
        for variant in VARIANTS:
            variant_dir = output_dir / f"seed_{int(training_seed)}" / variant
            payload = training_config_payload(
                namespace,
                project_root=project_root,
                variant=variant,
                seed=int(training_seed),
                args=args,
                env_config=env_config,
                reward_config=reward_config,
            )
            config_hash = canonical_config_hash(payload)
            checkpoint_root = variant_dir / "ckpt"
            bundles = sorted(
                path for path in checkpoint_root.iterdir() if path.is_dir() and path.name.isdigit()
            )
            if not bundles:
                raise RuntimeError(f"No callback checkpoints found in {checkpoint_root}")
            for bundle in bundles:
                manifest, pipeline_state = validate_checkpoint_bundle(
                    bundle,
                    config_hash,
                    expected_model_class=_class_qualified_name(model_class_for_variant(variant)),
                )
                timestep = int(manifest["timestep"])
                rows.append(
                    {
                        "training_seed": int(training_seed),
                        "variant": variant,
                        "model_timestep": timestep,
                        "n_updates": int(manifest["n_updates"]),
                        "replay_size": int(pipeline_state["replay_buffer_state"]["size"]),
                        "is_final_checkpoint": int(timestep == int(args.timesteps)),
                        "training_config_hash": config_hash,
                        "bundle_path": str(bundle),
                        "model_path": str(bundle / CHECKPOINT_PAYLOADS["model"]),
                    }
                )
    index = pd.DataFrame(rows).sort_values(["training_seed", "variant", "model_timestep"])
    index.to_csv(output_dir / "checkpoint_index.csv", index=False)
    return index.reset_index(drop=True)


def evaluate_checkpoints(
    namespace: dict[str, Any],
    *,
    checkpoints: pd.DataFrame,
    args: argparse.Namespace,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    output_dir: Path,
) -> pd.DataFrame:
    """Evaluate callback checkpoints on the same fixed-timestep paired protocol."""

    rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    expected_hashes: dict[int, str] = {}
    scenario_seeds = [
        int(args.eval_seed_start) + index for index in range(evaluation_scenario_count(args))
    ]
    for checkpoint in checkpoints.itertuples(index=False):
        model = load_model(str(checkpoint.variant), Path(checkpoint.model_path), args.device)
        try:
            for mode in MODES:
                for scenario_seed in scenario_seeds:
                    row, segments = evaluate_scenario(
                        namespace,
                        model=model,
                        variant=str(checkpoint.variant),
                        mode=mode,
                        scenario_seed=scenario_seed,
                        training_seed=int(checkpoint.training_seed),
                        env_config=env_config,
                        reward_config=reward_config,
                        args=args,
                    )
                    expected = expected_hashes.setdefault(scenario_seed, str(row["initial_state_hash"]))
                    if str(row["initial_state_hash"]) != expected:
                        raise RuntimeError(
                            f"Checkpoint scenario {scenario_seed} did not reset to the paired initial state"
                        )
                    row["model_timestep"] = int(checkpoint.model_timestep)
                    for segment in segments:
                        segment["model_timestep"] = int(checkpoint.model_timestep)
                    rows.append(row)
                    segment_rows.extend(segments)
        finally:
            del model
    scenarios = pd.DataFrame(rows)
    scenarios.to_csv(output_dir / "checkpoint_evaluation_scenarios.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(
        output_dir / "checkpoint_evaluation_segments.csv", index=False
    )
    return scenarios


def summarize_checkpoint_evaluations(
    scenarios: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the scenario-within-seed hierarchy independently at each checkpoint."""

    seed_pieces: list[pd.DataFrame] = []
    summary_pieces: list[pd.DataFrame] = []
    comparison_pieces: list[pd.DataFrame] = []
    comparison_summary_pieces: list[pd.DataFrame] = []
    for model_timestep, group in scenarios.groupby("model_timestep", sort=True):
        scenario_group = group.drop(columns=["model_timestep"])
        seed_summary = summarize_within_training_seed(scenario_group)
        summary = summarize_across_training_seeds(seed_summary)
        comparisons = paired_comparisons(seed_summary)
        comparison_summary = summarize_paired_comparisons(comparisons)
        for frame in (seed_summary, summary, comparisons, comparison_summary):
            frame.insert(0, "model_timestep", int(model_timestep))
        seed_pieces.append(seed_summary)
        summary_pieces.append(summary)
        comparison_pieces.append(comparisons)
        comparison_summary_pieces.append(comparison_summary)
    return (
        pd.concat(seed_pieces, ignore_index=True) if seed_pieces else pd.DataFrame(),
        pd.concat(summary_pieces, ignore_index=True) if summary_pieces else pd.DataFrame(),
        pd.concat(comparison_pieces, ignore_index=True) if comparison_pieces else pd.DataFrame(),
        pd.concat(comparison_summary_pieces, ignore_index=True)
        if comparison_summary_pieces
        else pd.DataFrame(),
    )


def summarize_within_training_seed(
    episodes: pd.DataFrame,
    step_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    learned_seeds = sorted(
        int(value) for value in pd.to_numeric(episodes["training_seed"], errors="coerce").dropna().unique()
    )
    expanded_groups: list[tuple[int, str, str, pd.DataFrame]] = []
    for (training_seed, variant, mode), group in episodes.dropna(subset=["training_seed"]).groupby(
        ["training_seed", "variant", "mode"], dropna=False
    ):
        expanded_groups.append((int(training_seed), str(variant), str(mode), group))
    random_rows = episodes[episodes["variant"] == RANDOM_VARIANT]
    for seed in learned_seeds:
        for mode, group in random_rows.groupby("mode"):
            expanded_groups.append((int(seed), RANDOM_VARIANT, str(mode), group))

    rows: list[dict[str, Any]] = []
    mean_metrics = (
        "h_violation_rate",
        "mean_h_dot",
        "mean_ttc_s",
        "mean_vehicle_spacing_m",
        "min_vehicle_spacing_m",
        "mean_surface_clearance_m",
        "mean_neighbor_count",
        "mean_traffic_density_per_km",
        "mean_abs_speed_error",
        "mean_jerk_norm",
        "IR",
        "mean_delta_a",
        "shadow_IR",
        "shadow_mean_delta_a",
        "shadow_mean_delta_a_physical",
        "shadow_mean_delta_ax",
        "shadow_mean_delta_ay",
        "qp_failure_rate",
        "qp_fallback_rate",
        "shadow_qp_failure_rate",
        "shadow_qp_fallback_rate",
        "nominal_action_saturation_rate",
        "safe_action_saturation_rate",
        "executed_action_saturation_rate",
        "common_form1_return_per_timestep",
        "mean_abs_target_speed_error",
        "mean_abs_target_lateral_error_m",
        "mean_formulation_cf",
        "mean_boundary_cost",
        "mean_normalized_acceleration_sq",
        "mean_normalized_acceleration_delta_sq",
        "mean_normalized_jerk_command_sq",
        "policy_command_saturation_rate",
        "acceleration_integrator_clip_rate",
        "reference_endpoint_rate",
        "controller_saturation_rate",
    )
    sum_metrics = (
        "timesteps",
        "total_time_s",
        "total_return",
        "common_form1_total_return",
        "task_return",
        "correction_return",
        "distinct_ego_collision_events",
        "ego_collision_incidents",
        "ego_collision_active_timesteps",
        "distinct_all_pair_collision_events",
        "active_collision_pair_timesteps",
        "total_distance_m",
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
        "near_boundary_steps",
        "time_near_boundary_s",
        "collision_survived_without_reset",
        "active_collision_without_event",
        "event_without_active_collision",
    )
    for training_seed, variant, mode, group in expanded_groups:
        row: dict[str, Any] = {
            "training_seed": int(training_seed),
            "variant": variant,
            "mode": mode,
            "scenarios": int(len(group)),
        }
        for metric in sum_metrics:
            if metric in group:
                row[metric] = float(
                    pd.to_numeric(group[metric], errors="coerce").fillna(0.0).sum()
                )
        collisions = float(row["distinct_ego_collision_events"])
        distance_m = float(row["total_distance_m"])
        timesteps = float(row["timesteps"])
        row["return_per_timestep"] = _ratio(float(row["total_return"]), timesteps)
        if "common_form1_total_return" in row:
            row["common_form1_return_per_timestep"] = _ratio(
                float(row["common_form1_total_return"]), timesteps
            )
        row["distance_per_collision_m"] = _distance_per_collision(distance_m, collisions)
        row["distance_per_collision_right_censored"] = int(collisions == 0.0)
        row["distance_per_collision_exposure_bound_m"] = _distance_per_collision_exposure_bound(
            distance_m, collisions
        )
        row["collision_events_per_m"] = _ratio(collisions, distance_m)
        row["ego_collisions_per_km"] = _collisions_per_km(collisions, distance_m)
        scenario_collision_counts = pd.to_numeric(
            group["distinct_ego_collision_events"], errors="coerce"
        ).fillna(0.0)
        row["collision_free_scenarios"] = int((scenario_collision_counts == 0.0).sum())
        row["shared_random_baseline"] = int(variant == RANDOM_VARIANT)
        row["episode_length_mean"] = _ratio(
            float(row["episode_length_sum"]), float(row["episode_segments"])
        )
        row["h_min"] = float(pd.to_numeric(group["h_min"], errors="coerce").min())
        row["min_ttc_s"] = (
            float(pd.to_numeric(group["min_ttc_s"], errors="coerce").min())
            if "min_ttc_s" in group
            else np.nan
        )
        row["near_boundary_rate"] = _ratio(
            float(row.get("near_boundary_steps", 0.0)), timesteps
        )
        # A mean of per-scenario quantiles is not a pooled quantile.  Preserve
        # it under an explicit descriptive name and, when step rows are
        # available, compute the actual within-seed p95 below.
        for quantile_metric in ("p95_delta_a", "shadow_p95_delta_a"):
            if quantile_metric in group:
                row[f"mean_scenario_{quantile_metric}"] = float(
                    pd.to_numeric(group[quantile_metric], errors="coerce").mean()
                )
            row[quantile_metric] = np.nan
        if step_metrics is not None and not step_metrics.empty:
            step_mask = (
                step_metrics["variant"].astype(str).eq(variant)
                & step_metrics["mode"].astype(str).eq(mode)
            )
            step_training_seeds = pd.to_numeric(
                step_metrics["training_seed"], errors="coerce"
            )
            if variant == RANDOM_VARIANT:
                step_mask &= step_training_seeds.isna()
            else:
                step_mask &= step_training_seeds.eq(int(training_seed))
            matching_steps = step_metrics.loc[step_mask]
            quantile_sources = {
                "p95_delta_a": "applied_delta_norm_scaled",
                "shadow_p95_delta_a": "shadow_delta_norm_scaled",
            }
            for target, source in quantile_sources.items():
                if source not in matching_steps:
                    continue
                values = pd.to_numeric(matching_steps[source], errors="coerce").dropna()
                if not values.empty:
                    row[target] = float(values.quantile(0.95))
        for metric in mean_metrics:
            if metric in group and metric not in row:
                row[metric] = float(pd.to_numeric(group[metric], errors="coerce").mean())
        observed = pd.to_numeric(group["first_collision_observed"], errors="coerce").fillna(0.0) > 0.5
        row["first_collision_observed_rate"] = float(observed.mean())
        row["time_to_first_collision_observed_mean_s"] = float(
            pd.to_numeric(group.loc[observed, "time_to_first_collision_s"], errors="coerce").mean()
        )
        row["distance_to_first_collision_observed_mean_m"] = float(
            pd.to_numeric(group.loc[observed, "distance_to_first_collision_m"], errors="coerce").mean()
        )
        restricted_time = pd.to_numeric(group["time_to_first_collision_s"], errors="coerce").where(
            observed,
            pd.to_numeric(group["first_collision_censor_time_s"], errors="coerce"),
        )
        restricted_distance = pd.to_numeric(group["distance_to_first_collision_m"], errors="coerce").where(
            observed,
            pd.to_numeric(group["first_collision_censor_distance_m"], errors="coerce"),
        )
        row["time_to_first_collision_restricted_mean_s"] = float(restricted_time.mean())
        row["distance_to_first_collision_restricted_mean_m"] = float(restricted_distance.mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["variant", "mode", "training_seed"]).reset_index(drop=True)


def summarize_across_training_seeds(seed_summary: pd.DataFrame) -> pd.DataFrame:
    identifier_columns = {"training_seed", "variant", "mode"}
    metric_columns = [column for column in seed_summary.columns if column not in identifier_columns]
    rows: list[dict[str, Any]] = []
    for (variant, mode), group in seed_summary.groupby(["variant", "mode"], dropna=False):
        paired_seed_replicates = int(group["training_seed"].nunique())
        shared_random_baseline = str(variant) == RANDOM_VARIANT
        analysis_group = group.iloc[[0]] if shared_random_baseline else group
        row: dict[str, Any] = {
            "variant": variant,
            "mode": mode,
            "training_seeds": 0 if shared_random_baseline else paired_seed_replicates,
            "paired_training_seed_replicates": paired_seed_replicates,
        }
        for metric in metric_columns:
            values = pd.to_numeric(analysis_group[metric], errors="coerce")
            row[f"{metric}_seed_mean"] = float(values.mean())
            row[f"{metric}_seed_variance"] = float(values.var(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["variant", "mode"]).reset_index(drop=True)


def summarize_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    """Compatibility entry point: return the required across-seed hierarchy."""

    return summarize_across_training_seeds(summarize_within_training_seed(episodes))


def paired_comparisons(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "return_per_timestep",
        "ego_collisions_per_km",
        "distance_per_collision_m",
        "distance_per_collision_exposure_bound_m",
        "distance_per_collision_right_censored",
        "total_distance_m",
        "episode_length_mean",
        "reset_calls_total",
        "h_min",
        "near_boundary_rate",
        "time_near_boundary_s",
        "mean_h_dot",
        "min_ttc_s",
        "mean_vehicle_spacing_m",
        "mean_traffic_density_per_km",
        "mean_abs_speed_error",
        "mean_jerk_norm",
        "IR",
        "mean_delta_a",
        "p95_delta_a",
        "shadow_IR",
        "shadow_mean_delta_a",
        "shadow_p95_delta_a",
        "qp_failure_rate",
        "qp_fallback_rate",
    ]
    rows: list[dict[str, Any]] = []
    for comparison, (left_variant, left_mode, right_variant, right_mode) in COMPARISONS.items():
        left = seed_summary[(seed_summary["variant"] == left_variant) & (seed_summary["mode"] == left_mode)].copy()
        right = seed_summary[(seed_summary["variant"] == right_variant) & (seed_summary["mode"] == right_mode)].copy()
        if left.empty or right.empty:
            continue
        merged = left.merge(right, on=["training_seed"], suffixes=("_left", "_right"), how="inner")
        for _, pair in merged.iterrows():
            row: dict[str, Any] = {
                "comparison": comparison,
                "training_seed": int(pair["training_seed"]),
                "left": f"{left_variant}:{left_mode}",
                "right": f"{right_variant}:{right_mode}",
            }
            for metric in metrics:
                left_value = _as_float(pair.get(f"{metric}_left"))
                right_value = _as_float(pair.get(f"{metric}_right"))
                row[f"delta_{metric}"] = left_value - right_value if np.isfinite(left_value) and np.isfinite(right_value) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_paired_comparisons(comparisons: pd.DataFrame) -> pd.DataFrame:
    if comparisons.empty:
        return comparisons.copy()
    delta_columns = [column for column in comparisons.columns if column.startswith("delta_")]
    rows: list[dict[str, Any]] = []
    for comparison, group in comparisons.groupby("comparison"):
        row: dict[str, Any] = {
            "comparison": comparison,
            "training_seeds": int(group["training_seed"].nunique()),
            "left": group["left"].iloc[0],
            "right": group["right"].iloc[0],
        }
        for column in delta_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_seed_mean"] = float(values.mean())
            row[f"{column}_seed_variance"] = float(values.var(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


FACTORIAL_CONTRASTS: dict[str, dict[tuple[bool, bool], float]] = {
    # Average of the two conditional effects for each factor.
    "reward_main_effect": {
        (False, False): -0.5,
        (True, False): 0.5,
        (False, True): -0.5,
        (True, True): 0.5,
    },
    "actor_loss_main_effect": {
        (False, False): -0.5,
        (True, False): -0.5,
        (False, True): 0.5,
        (True, True): 0.5,
    },
    # Difference of differences: both - loss-only - reward-only + shield-only.
    "reward_actor_interaction": {
        (False, False): 1.0,
        (True, False): -1.0,
        (False, True): -1.0,
        (True, True): 1.0,
    },
}


def factorial_effects(seed_summary: pd.DataFrame) -> pd.DataFrame:
    """Return paired 2x2 reward/loss contrasts within each seed and mode.

    The nominal and random controls are intentionally excluded.  Every
    contrast therefore compares actors trained with the same runtime shield
    and differs only in the two registered treatment switches.
    """

    metrics = [
        "return_per_timestep",
        "ego_collisions_per_km",
        "distance_per_collision_m",
        "distance_per_collision_exposure_bound_m",
        "total_distance_m",
        "episode_length_mean",
        "h_min",
        "h_violation_rate",
        "near_boundary_rate",
        "mean_abs_speed_error",
        "mean_jerk_norm",
        "IR",
        "mean_delta_a",
        "p95_delta_a",
        "shadow_IR",
        "shadow_mean_delta_a",
        "mean_h_dot",
        "min_ttc_s",
        "mean_vehicle_spacing_m",
        "mean_traffic_density_per_km",
    ]
    available_metrics = [metric for metric in metrics if metric in seed_summary.columns]
    learned = seed_summary[
        seed_summary["variant"].isin(set(FACTORIAL_VARIANTS.values()))
    ].copy()
    rows: list[dict[str, Any]] = []
    for (training_seed, mode), group in learned.groupby(["training_seed", "mode"], dropna=False):
        by_variant = {
            str(row["variant"]): row
            for _, row in group.drop_duplicates(subset=["variant"], keep="last").iterrows()
        }
        if any(variant not in by_variant for variant in FACTORIAL_VARIANTS.values()):
            continue
        for effect, coefficients in FACTORIAL_CONTRASTS.items():
            result: dict[str, Any] = {
                "effect": effect,
                "training_seed": int(training_seed),
                "mode": str(mode),
                "formula": " + ".join(
                    f"{coefficient:+g}*{FACTORIAL_VARIANTS[cell]}"
                    for cell, coefficient in coefficients.items()
                ),
            }
            for metric in available_metrics:
                terms = [
                    (
                        float(coefficient),
                        _as_float(by_variant[FACTORIAL_VARIANTS[cell]].get(metric)),
                    )
                    for cell, coefficient in coefficients.items()
                ]
                result[f"effect_{metric}"] = (
                    float(sum(coefficient * value for coefficient, value in terms))
                    if all(np.isfinite(value) for _, value in terms)
                    else np.nan
                )
            rows.append(result)
    return pd.DataFrame(rows)


def summarize_factorial_effects(effects: pd.DataFrame) -> pd.DataFrame:
    if effects.empty:
        return effects.copy()
    metric_columns = [column for column in effects.columns if column.startswith("effect_")]
    rows: list[dict[str, Any]] = []
    for (effect, mode), group in effects.groupby(["effect", "mode"], dropna=False):
        row: dict[str, Any] = {
            "effect": str(effect),
            "mode": str(mode),
            "training_seeds": int(group["training_seed"].nunique()),
            "formula": str(group["formula"].iloc[0]),
        }
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_seed_mean"] = float(values.mean())
            row[f"{column}_seed_variance"] = (
                float(values.var(ddof=1)) if values.notna().sum() > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mode", "effect"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the controlled CBF-filter internalization ablation.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timesteps", type=int, default=20_000, help="Use 200000 for the main pre-registered study.")
    parser.add_argument("--checkpoint-interval", type=int, default=5_000)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAINING_SEEDS),
        help="Independent training seeds; every variant uses this same paired seed set.",
    )
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument(
        "--eval-scenarios",
        type=int,
        default=None,
        help="Number of independently seeded fixed-timestep evaluation scenarios.",
    )
    parser.add_argument(
        "--eval-timesteps",
        type=int,
        default=None,
        help="Timestep budget per evaluation scenario; collisions reset and do not end the budget.",
    )
    parser.add_argument("--eval-episodes", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--eval-seed-start", type=int, default=900_000)
    parser.add_argument("--eval-horizon", type=int, default=800, help=argparse.SUPPRESS)
    parser.add_argument("--k0", type=float, default=5.29)
    parser.add_argument("--k1", type=float, default=3.68)
    parser.add_argument("--eps-side", type=float, default=0.10)
    parser.add_argument("--lambda-delta", type=float, default=0.025)
    parser.add_argument("--lambda-intervention", type=float, default=0.02)
    parser.add_argument("--correction-epsilon", type=float, default=0.03)
    parser.add_argument(
        "--near-boundary-h",
        type=float,
        default=0.5,
        help="Report time with |h_min| at or below this threshold (metres).",
    )
    parser.add_argument(
        "--ttc-cap",
        type=float,
        default=30.0,
        help="Finite reporting cap for linearized time-to-CBF-boundary (seconds).",
    )
    parser.add_argument("--lambda-bc", type=float, default=0.03)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument(
        "--evaluate-checkpoints",
        action="store_true",
        help="Also evaluate every callback checkpoint and write timestep-indexed learning curves.",
    )
    add_env_config_args(parser)
    parser.set_defaults(traffic_model="mtm")
    return parser.parse_args()


def main() -> int:
    set_stable_native_defaults()
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()
    if int(args.n_envs) != 1:
        raise ValueError("Use --n-envs 1 for this paired, one-update-per-transition ablation.")
    if int(args.timesteps) <= 0 or int(args.checkpoint_interval) <= 0:
        raise ValueError("--timesteps and --checkpoint-interval must be positive")
    if len(set(int(seed) for seed in args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")
    if evaluation_timestep_budget(args) <= 0 or evaluation_scenario_count(args) <= 0:
        raise ValueError("Evaluation timestep budget and scenario count must be positive")
    if not 0.0 <= float(args.correction_epsilon):
        raise ValueError("--correction-epsilon must be non-negative")
    if not 0.0 <= float(args.near_boundary_h):
        raise ValueError("--near-boundary-h must be non-negative")
    if not np.isfinite(float(args.ttc_cap)) or float(args.ttc_cap) <= 0.0:
        raise ValueError("--ttc-cap must be finite and positive")
    if args.skip_evaluation and args.evaluate_checkpoints:
        raise ValueError("--evaluate-checkpoints cannot be combined with --skip-evaluation")

    project_root = find_project_root(args.project_root or Path.cwd())
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    namespace = bootstrap_notebook_namespace(project_root)
    exec_required_notebook_cells(notebook_path, namespace)
    namespace["DEVICE"] = args.device
    namespace["CBF_K0"] = float(args.k0)
    namespace["CBF_K1"] = float(args.k1)
    namespace["CBF_EPS_SIDE"] = float(args.eps_side)
    namespace["CBF_FILTER_REWARD_LAMBDA"] = 0.0
    install_minimal_guided_cbf(namespace)
    namespace["install_cbf_projection_reporting"]()
    install_correction_reward_env(namespace)

    env_config = env_config_from_args(args, namespace["ENV_CONFIG"])
    if active_traffic_model(env_config) == "mtm":
        deep_update(env_config, copy.deepcopy(MTM_CONGESTED_UNCERTAIN_UPDATES))
    reward_config = make_base_reward_config(namespace)
    output_dir = args.output_dir or (project_root / "artifacts" / "cbf_filter_ablation_pilot")
    output_dir = output_dir.resolve()
    preflight_output(
        namespace,
        project_root=project_root,
        output_dir=output_dir,
        args=args,
        env_config=env_config,
        reward_config=reward_config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_config = {
        "scenario_count": evaluation_scenario_count(args),
        "timestep_budget_per_scenario": evaluation_timestep_budget(args),
        "eval_seed_start": int(args.eval_seed_start),
        "modes": list(MODES),
        "terminate_on_collision": True,
        "reset_immediately_after_terminal": True,
        "deterministic_policy": True,
        "env_config": env_config,
        "reward_config": reward_config,
        "k0": float(args.k0),
        "k1": float(args.k1),
        "eps_side": float(args.eps_side),
        "near_boundary_h": float(args.near_boundary_h),
        "ttc_cap_s": float(args.ttc_cap),
    }
    run_config = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "study": "cbf_filter_internalization_ablation",
        "timesteps": int(args.timesteps),
        "seeds": [int(seed) for seed in args.seeds],
        "seed_protocol": "same seed across variants within a replicate; independent seeds across replicates",
        "variants": list(VARIANTS),
        "factorial_variants": {
            f"reward_{int(reward_on)}_actor_loss_{int(loss_on)}": variant
            for (reward_on, loss_on), variant in FACTORIAL_VARIANTS.items()
        },
        "contextual_controls": ["a_nominal", RANDOM_VARIANT],
        "evaluation_modes": list(MODES),
        "evaluation_protocol": evaluation_config,
        "evaluation_config_hash": canonical_config_hash(evaluation_config),
        "evaluate_checkpoints": bool(args.evaluate_checkpoints),
        "env_config": env_config,
        "base_reward_config": reward_config,
        "k0": float(args.k0),
        "k1": float(args.k1),
        "eps_side": float(args.eps_side),
        "lambda_delta": float(args.lambda_delta),
        "lambda_intervention": float(args.lambda_intervention),
        "correction_epsilon_normalized": float(args.correction_epsilon),
        "lambda_bc": float(args.lambda_bc),
        "actor_loss": (
            "lambda_bc * I_intervention(s,a_behavior) * "
            "||pi(s)-stop_gradient[local_F(s,pi(s))]||^2, where local_F uses "
            "the recorded projection Jacobian and is identity on the feasible side"
        ),
        "note": (
            "B--E form the filtered 2x2 reward-by-actor-loss study. A nominal and the random policy "
            "are contextual controls. All reported returns use the common base task reward; deployment "
            "evaluation applies no correction reward."
        ),
    }
    run_config_path = output_dir / "run_config.json"
    if args.resume and args.skip_evaluation and run_config_path.exists():
        previous_run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        for key in ("evaluation_protocol", "evaluation_config_hash", "evaluate_checkpoints"):
            if key in previous_run_config:
                run_config[key] = previous_run_config[key]
        run_config["evaluation_artifacts_preserved_on_resume"] = True
    else:
        run_config["evaluation_artifacts_preserved_on_resume"] = False
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    evaluation_label = (
        "skipped"
        if args.skip_evaluation
        else f"{evaluation_scenario_count(args)}x{evaluation_timestep_budget(args)} timesteps"
    )
    print(
        "[ablation] starting"
        f" output={output_dir}"
        f" traffic={active_traffic_model(env_config)}"
        f" timesteps={args.timesteps:,}"
        f" seeds={args.seeds}"
        f" eval={evaluation_label}",
        flush=True,
    )

    models: dict[tuple[int, str], Any] = {}
    model_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for variant in VARIANTS:
            model_path = train_variant(
                namespace,
                variant=variant,
                seed=int(seed),
                args=args,
                env_config=env_config,
                reward_config=reward_config,
                output_dir=output_dir,
            )
            models[(int(seed), variant)] = load_model(variant, model_path, args.device)
            spec = variant_spec(variant, args)
            model_rows.append(
                {
                    "training_seed": int(seed),
                    "variant": variant,
                    "model_path": str(model_path),
                    "action_semantics": (
                        "normalized" if variant == "a_nominal" else "physical"
                    ),
                    "reward_penalty": int(
                        float(spec["lambda_delta"]) > 0.0
                        or float(spec["lambda_intervention"]) > 0.0
                    ),
                    "actor_cbf_loss": int(bool(spec["actor_loss"])),
                }
            )
    pd.DataFrame(model_rows).to_csv(output_dir / "models.csv", index=False)
    checkpoints = checkpoint_index(
        namespace,
        args=args,
        env_config=env_config,
        reward_config=reward_config,
        output_dir=output_dir,
    )

    if not args.skip_evaluation:
        scenarios, occupancy_steps = evaluate_models(
            namespace,
            models=models,
            args=args,
            env_config=env_config,
            reward_config=reward_config,
            output_dir=output_dir,
        )
        seed_summary = summarize_within_training_seed(
            scenarios,
            step_metrics=occupancy_steps,
        )
        summary = summarize_across_training_seeds(seed_summary)
        comparisons = paired_comparisons(seed_summary)
        comparison_summary = summarize_paired_comparisons(comparisons)
        factorial = factorial_effects(seed_summary)
        factorial_summary = summarize_factorial_effects(factorial)
        seed_summary.to_csv(output_dir / "evaluation_seed_summary.csv", index=False)
        summary.to_csv(output_dir / "evaluation_summary.csv", index=False)
        comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
        comparison_summary.to_csv(output_dir / "paired_comparisons_summary.csv", index=False)
        factorial.to_csv(output_dir / "factorial_effects.csv", index=False)
        factorial_summary.to_csv(output_dir / "factorial_effects_summary.csv", index=False)
        if args.evaluate_checkpoints:
            checkpoint_scenarios = evaluate_checkpoints(
                namespace,
                checkpoints=checkpoints,
                args=args,
                env_config=env_config,
                reward_config=reward_config,
                output_dir=output_dir,
            )
            (
                checkpoint_seed_summary,
                checkpoint_summary,
                checkpoint_comparisons,
                checkpoint_comparison_summary,
            ) = summarize_checkpoint_evaluations(checkpoint_scenarios)
            checkpoint_seed_summary.to_csv(
                output_dir / "checkpoint_evaluation_seed_summary.csv", index=False
            )
            checkpoint_summary.to_csv(
                output_dir / "checkpoint_evaluation_summary.csv", index=False
            )
            checkpoint_comparisons.to_csv(
                output_dir / "checkpoint_paired_comparisons.csv", index=False
            )
            checkpoint_comparison_summary.to_csv(
                output_dir / "checkpoint_paired_comparisons_summary.csv", index=False
            )
        print("[ablation] evaluation summary", flush=True)
        report_columns = [
            "variant",
            "mode",
            "training_seeds",
            "return_per_timestep_seed_mean",
            "return_per_timestep_seed_variance",
            "distinct_ego_collision_events_seed_mean",
            "distinct_ego_collision_events_seed_variance",
            "total_distance_m_seed_mean",
            "total_distance_m_seed_variance",
            "distance_per_collision_m_seed_mean",
            "distance_per_collision_m_seed_variance",
            "distance_per_collision_exposure_bound_m_seed_mean",
            "distance_per_collision_exposure_bound_m_seed_variance",
            "distance_per_collision_right_censored_seed_mean",
            "ego_collisions_per_km_seed_mean",
            "ego_collisions_per_km_seed_variance",
            "episode_length_mean_seed_mean",
            "episode_length_mean_seed_variance",
            "reset_calls_total_seed_mean",
            "reset_calls_total_seed_variance",
        ]
        print(summary[report_columns].to_string(index=False), flush=True)
    print(f"[ablation] complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
