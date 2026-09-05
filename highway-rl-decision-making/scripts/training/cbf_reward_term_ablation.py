from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from scripts.training.cbf_lambda_event_bc_pilot_sweep import (
    exec_notebook_cells,
    find_project_root,
    install_event_penalty_env,
    set_stable_native_defaults,
)
from scripts.common.guided_cbf_minimal import install_minimal_guided_cbf
from scripts.common.laneless_script_config import active_traffic_model, add_env_config_args, env_config_from_args


warnings.filterwarnings("ignore", message="OSQP exited.*")

NOTEBOOK_DEPS = [2, 3, 5, 6, 8, 11, 33, 35, 37, 39, 41]

DEFAULT_TRIALS: list[dict[str, float | str | bool]] = [
    {
        "trial_name": "current_bc003",
        "use_current_potential": True,
        "use_safety_potential": False,
        "wy": 0.65,
        "wf": 1.0,
        "w_safe": 0.0,
        "lambda_bc": 0.03,
    },
    {
        "trial_name": "current_no_y_bc003",
        "use_current_potential": True,
        "use_safety_potential": False,
        "wy": 0.0,
        "wf": 1.0,
        "w_safe": 0.0,
        "lambda_bc": 0.03,
    },
    {
        "trial_name": "no_old_potential_bc003",
        "use_current_potential": False,
        "use_safety_potential": False,
        "wy": 0.65,
        "wf": 0.0,
        "w_safe": 0.0,
        "lambda_bc": 0.03,
    },
    {
        "trial_name": "safety_potential_bc003",
        "use_current_potential": False,
        "use_safety_potential": True,
        "wy": 0.65,
        "wf": 0.0,
        "w_safe": 0.80,
        "lambda_bc": 0.03,
    },
    {
        "trial_name": "safety_potential_no_y_bc003",
        "use_current_potential": False,
        "use_safety_potential": True,
        "wy": 0.0,
        "wf": 0.0,
        "w_safe": 0.80,
        "lambda_bc": 0.03,
    },
    {
        "trial_name": "safety_potential_no_bc",
        "use_current_potential": False,
        "use_safety_potential": True,
        "wy": 0.65,
        "wf": 0.0,
        "w_safe": 0.80,
        "lambda_bc": 0.0,
    },
]


def install_safety_set_reward_wrapper(namespace: dict[str, Any]) -> None:
    base_reward_wrapper = namespace["KaralakouRewardWrapper"]

    class SafetySetKaralakouRewardWrapper(base_reward_wrapper):  # type: ignore[misc, valid-type]
        """Karalakou reward with optional old potential and optional CBF-set potential."""

        def __init__(self, env: gym.Env, reward_config: dict[str, float] | None = None) -> None:
            super().__init__(env, reward_config=reward_config)
            self.reward_config.setdefault("use_current_potential", 1.0)
            self.reward_config.setdefault("use_safety_potential", 0.0)
            self.reward_config.setdefault("w_safe", 0.0)
            self.reward_config.setdefault("safety_potential_sigma_h", 2.0)
            self.reward_config.setdefault("safety_boundary_sigma", 1.0)
            self.reward_config.setdefault("safety_potential_eps_side", namespace["CBF_EPS_SIDE"])

        def _karalakou_reward(
            self,
            previous_dx: dict[int, float],
            previous_ego_x: float | None = None,
        ) -> tuple[float, dict[str, float]]:
            base = self.base_env
            ego = base.vehicle
            cfg = self.reward_config

            target_y, target_speed, zone_found = self._lateral_target_and_speed()
            ego_y = float(ego.position[1])
            ego_speed = float(ego.vx)
            desired_speed = float(ego.desired_speed)
            road_width = max(float(base.config["road_width"]), 1e-6)
            cx = abs(ego_speed - target_speed) / max(target_speed, 1e-6)
            cy = abs(ego_y - target_y) / road_width
            lat_y_error_m = abs(ego_y - target_y)
            lat_y_coherence = float(np.clip(1.0 - cy, 0.0, 1.0))
            current_cf = self._potential_field_cost() if bool(cfg.get("use_current_potential", 1.0)) else 0.0
            safety_cf = self._safety_set_potential_cost() if bool(cfg.get("use_safety_potential", 0.0)) else 0.0
            cay = self._lateral_acceleration_cost()
            overtakes = self._overtake_count(previous_dx)
            progress_m = self._forward_progress(previous_ego_x)
            progress_normalized = self._normalized_forward_progress(progress_m, desired_speed)
            progress_clipped = float(np.clip(progress_normalized, 0.0, float(cfg.get("progress_clip", 1.25))))
            progress_reward = float(cfg.get("progress_reward_weight", 0.0)) * progress_clipped
            jerk_vector, jerk_norm, jerk_cost = self._jerk_metrics()
            jerk_penalty = float(cfg.get("jerk_penalty_weight", 0.0)) * jerk_cost

            denom = (
                cfg["epsilon_r"]
                + cfg["wx"] * cx
                + cfg["wy"] * cy
                + cfg["wf"] * current_cf
                + cfg.get("w_safe", 0.0) * safety_cf
                + cfg.get("way", 0.0) * cay
            )
            reward = cfg["epsilon_r"] / max(denom, 1e-9) + progress_reward - jerk_penalty
            if bool(base._last_ego_collision):
                reward += cfg["collision_penalty"]
            elif overtakes > 0:
                reward += cfg["overtake_bonus"] * min(overtakes, 1)

            components = {
                "reward": float(reward),
                "cx": float(cx),
                "cy": float(cy),
                "cf": float(current_cf),
                "safety_cf": float(safety_cf),
                "cay": float(cay),
                "ay": float(self._ego_lateral_acceleration()),
                "ego_y": float(ego_y),
                "ego_speed": float(ego_speed),
                "desired_speed": float(desired_speed),
                "speed_deviation": float(ego_speed - desired_speed),
                "abs_speed_deviation": float(abs(ego_speed - desired_speed)),
                "target_speed_deviation": float(ego_speed - target_speed),
                "abs_target_speed_deviation": float(abs(ego_speed - target_speed)),
                "target_y": float(target_y),
                "lat_y_error_m": float(lat_y_error_m),
                "lat_y_coherence": float(lat_y_coherence),
                "progress_m": float(progress_m),
                "progress_normalized": float(progress_normalized),
                "progress_clipped": float(progress_clipped),
                "progress_reward": float(progress_reward),
                "applied_ax": float(self._current_applied_acceleration()[0]),
                "applied_ay": float(self._current_applied_acceleration()[1]),
                "jerk_ax": float(jerk_vector[0]),
                "jerk_ay": float(jerk_vector[1]),
                "jerk_norm": float(jerk_norm),
                "jerk_cost": float(jerk_cost),
                "jerk_penalty": float(jerk_penalty),
                "target_speed": float(target_speed),
                "zone_found": float(zone_found),
                "overtakes": float(overtakes),
                "ego_collision": float(base._last_ego_collision),
            }
            return float(reward), components

        def _safety_set_potential_cost(self) -> float:
            base = self.base_env
            cfg = self.reward_config
            road_width = max(float(base.config["road_width"]), 1e-6)
            neighbor_range = float(namespace.get("CBF_NEIGHBOR_RANGE", base.config["sensing_range"]))
            eps_side = float(cfg.get("safety_potential_eps_side", namespace["CBF_EPS_SIDE"]))
            sigma_h = max(float(cfg.get("safety_potential_sigma_h", 2.0)), 1e-6)
            sigma_boundary = max(float(cfg.get("safety_boundary_sigma", 1.0)), 1e-6)

            ego = namespace["get_ego_state"](self)
            neighbors = namespace["get_neighbor_states"](self, neighbor_range=neighbor_range)
            max_neighbors = namespace.get("CBF_MAX_NEIGHBOR_CONSTRAINTS")
            if max_neighbors is not None:
                neighbors = list(neighbors)[: int(max_neighbors)]

            risks: list[float] = []
            for other in neighbors:
                h, *_ = namespace["pairwise_cbf_geometry"](ego, other, eps_side=eps_side)
                risks.append(self._barrier_risk(float(h), sigma_h))

            ego_y = float(ego["y"])
            ego_half_width = 0.5 * float(ego["width"])
            left_h = ego_y - ego_half_width
            right_h = road_width - ego_half_width - ego_y
            risks.append(self._barrier_risk(float(min(left_h, right_h)), sigma_boundary))
            return float(np.clip(max(risks) if risks else 0.0, 0.0, 1.0))

        @staticmethod
        def _barrier_risk(h_value: float, sigma: float) -> float:
            if not np.isfinite(h_value):
                return 1.0
            if h_value <= 0.0:
                return 1.0
            return float(np.exp(-h_value / sigma))

    namespace["KaralakouRewardWrapper"] = SafetySetKaralakouRewardWrapper
    namespace["SafetySetKaralakouRewardWrapper"] = SafetySetKaralakouRewardWrapper


def make_reward_config(namespace: dict[str, Any], trial: dict[str, float | str | bool]) -> dict[str, float]:
    config = dict(namespace["REWARD_CONFIG"])
    config.update(
        {
            "wy": float(trial["wy"]),
            "wf": float(trial["wf"]),
            "w_safe": float(trial["w_safe"]),
            "use_current_potential": float(bool(trial["use_current_potential"])),
            "use_safety_potential": float(bool(trial["use_safety_potential"])),
            "safety_potential_sigma_h": 2.0,
            "safety_boundary_sigma": 1.0,
            "progress_reward_weight": 0.0,
        }
    )
    return config


def make_single_env(
    namespace: dict[str, Any],
    *,
    seed: int,
    reward_config: dict[str, float],
    env_config: dict[str, Any],
    lambda_norm: float,
    lambda_event: float,
    event_threshold: float,
    k0: float,
    k1: float,
    eps_side: float,
    render_mode: str | None = None,
    normalize_observation: bool | None = None,
) -> gym.Env:
    env = gym.make("lane-free-v0", render_mode=render_mode, config=env_config)
    env = namespace["KaralakouRewardWrapper"](env, reward_config=reward_config)
    env = namespace["EventPenaltySafetyFilteredAccelerationWrapper"](
        env,
        lambda_filter=float(lambda_norm),
        lambda_event=float(lambda_event),
        intervention_threshold=float(event_threshold),
        eps_side=float(eps_side),
        k0=float(k0),
        k1=float(k1),
    )
    normalize = namespace["NORMALIZE_RL_OBSERVATIONS"] if normalize_observation is None else normalize_observation
    if normalize:
        env = namespace["LaneFreeObservationNormalizationWrapper"](env, clip=namespace["OBSERVATION_CLIP"])
    if "KPIInfoWrapper" in namespace:
        env = namespace["KPIInfoWrapper"](env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def make_training_env(
    namespace: dict[str, Any],
    *,
    seed: int,
    reward_config: dict[str, float],
    env_config: dict[str, Any],
    lambda_norm: float,
    lambda_event: float,
    event_threshold: float,
    k0: float,
    k1: float,
    eps_side: float,
    n_envs: int,
) -> Any:
    def _single_env(env_seed: int) -> gym.Env:
        return make_single_env(
            namespace,
            seed=env_seed,
            reward_config=reward_config,
            env_config=env_config,
            lambda_norm=lambda_norm,
            lambda_event=lambda_event,
            event_threshold=event_threshold,
            k0=k0,
            k1=k1,
            eps_side=eps_side,
        )

    return namespace["_make_vectorized_env"](
        _single_env,
        seed=seed,
        n_envs=n_envs,
        use_subproc=bool(namespace.get("DDPG_USE_SUBPROC_VEC_ENV", True)),
        start_method=namespace["DDPG_SUBPROC_START_METHOD"],
    )


def evaluate_model(
    namespace: dict[str, Any],
    model: Any,
    *,
    episodes: int,
    seed: int,
    reward_config: dict[str, float],
    env_config: dict[str, Any],
    k0: float,
    k1: float,
    eps_side: float,
    event_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for episode in range(episodes):
        env = make_single_env(
            namespace,
            seed=seed + episode,
            reward_config=reward_config,
            env_config=env_config,
            lambda_norm=0.0,
            lambda_event=0.0,
            event_threshold=event_threshold,
            k0=k0,
            k1=k1,
            eps_side=eps_side,
        )
        namespace["configure_paper_evaluation_env"](env, steps=namespace["PAPER_EVAL_STEPS"])
        obs, _ = env.reset(seed=seed + episode)
        done = False
        step_count = 0
        rewards: list[float] = []
        speeds: list[float] = []
        abs_speed_errors: list[float] = []
        lat_y_errors: list[float] = []
        correction_norms: list[float] = []
        meaningful_correction_norms: list[float] = []
        event_interventions: list[float] = []
        numerical_interventions: list[float] = []
        qp_successes: list[float] = []
        min_h_values: list[float] = []
        old_potential_costs: list[float] = []
        safety_potential_costs: list[float] = []
        lateral_costs: list[float] = []
        kpi_info_rows: list[dict[str, Any]] = []
        ego_collisions = 0
        ego_collision_steps = 0
        all_collision_events = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            base = env.unwrapped
            speed = float(base.vehicle.vx)
            desired_speed = float(base.vehicle.desired_speed)
            correction = float(info.get("cbf_correction_norm", 0.0))
            meaningful_correction = float(info.get("cbf_meaningful_correction_norm", max(correction - event_threshold, 0.0)))

            rewards.append(float(reward))
            speeds.append(speed)
            abs_speed_errors.append(abs(speed - desired_speed))
            lat_error = float(info.get("karalakou_lat_y_error_m", np.nan))
            if np.isfinite(lat_error):
                lat_y_errors.append(lat_error)
            kpi_info_rows.append(dict(info))
            correction_norms.append(correction)
            meaningful_correction_norms.append(meaningful_correction)
            event_interventions.append(float(info.get("cbf_event_intervened", correction > event_threshold)))
            numerical_interventions.append(float(info.get("cbf_intervened", correction > 1e-6)))
            qp_successes.append(float(info.get("cbf_qp_success", True)))
            min_h_values.append(float(info.get("cbf_min_h", np.nan)))
            old_potential_costs.append(float(info.get("karalakou_cf", 0.0)))
            safety_potential_costs.append(float(info.get("karalakou_safety_cf", 0.0)))
            lateral_costs.append(float(info.get("karalakou_cy", 0.0)))
            all_collision_events += int(info.get("collisions", 0))
            ego_collisions += int(info.get("ego_collision_events", 0))
            if bool(info.get("ego_collision", False)):
                ego_collision_steps += 1
            step_count += 1
            done = bool(terminated or truncated)

        episode_row = {
            "episode": float(episode),
            "steps": float(step_count),
            "episode_length_steps": float(step_count),
            "return": float(np.sum(rewards)),
            "episode_return": float(np.sum(rewards)),
            "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            "mean_abs_speed_error": float(np.mean(abs_speed_errors)) if abs_speed_errors else 0.0,
            "mean_lat_y_error_m": float(np.mean(lat_y_errors)) if lat_y_errors else np.nan,
            "mean_correction_norm": float(np.mean(correction_norms)) if correction_norms else 0.0,
            "max_correction_norm": float(np.max(correction_norms)) if correction_norms else 0.0,
            "mean_meaningful_correction_norm": float(np.mean(meaningful_correction_norms))
            if meaningful_correction_norms
            else 0.0,
            "max_meaningful_correction_norm": float(np.max(meaningful_correction_norms))
            if meaningful_correction_norms
            else 0.0,
            "event_intervention_rate": float(np.mean(event_interventions)) if event_interventions else 0.0,
            "numerical_intervention_rate": float(np.mean(numerical_interventions)) if numerical_interventions else 0.0,
            "qp_failure_rate": float(1.0 - np.mean(qp_successes)) if qp_successes else 0.0,
            "min_h": float(np.nanmin(min_h_values)) if min_h_values and not np.all(np.isnan(min_h_values)) else np.nan,
            "mean_old_potential_cost": float(np.mean(old_potential_costs)) if old_potential_costs else 0.0,
            "mean_safety_potential_cost": float(np.mean(safety_potential_costs)) if safety_potential_costs else 0.0,
            "mean_lateral_y_cost": float(np.mean(lateral_costs)) if lateral_costs else 0.0,
            "ego_collisions": float(ego_collisions),
            "ego_collision_steps": float(ego_collision_steps),
            "total_collision_events": float(all_collision_events),
        }
        summarize_episode_kpis = namespace.get("summarize_episode_kpis")
        if callable(summarize_episode_kpis):
            episode_row.update(
                summarize_episode_kpis(
                    kpi_info_rows,
                    rewards=rewards,
                    task_completed=False,
                    fallback_steps=step_count,
                    fallback_distance_m=0.0,
                    fallback_dt_s=namespace["kpi_policy_dt"](env) if "kpi_policy_dt" in namespace else np.nan,
                )
            )
        rows.append(episode_row)
        env.close()
    return pd.DataFrame(rows)


def _metric_series(metrics: pd.DataFrame, column: str) -> pd.Series:
    if column not in metrics.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(metrics[column], errors="coerce").dropna()


def _metric_mean(metrics: pd.DataFrame, column: str, default: float = 0.0) -> float:
    values = _metric_series(metrics, column)
    return float(values.mean()) if not values.empty else float(default)


def _metric_min(metrics: pd.DataFrame, column: str, default: float = np.nan) -> float:
    values = _metric_series(metrics, column)
    return float(values.min()) if not values.empty else float(default)


def _metric_max(metrics: pd.DataFrame, column: str, default: float = 0.0) -> float:
    values = _metric_series(metrics, column)
    return float(values.max()) if not values.empty else float(default)


def summarize(metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "episodes": float(len(metrics)),
        "steps_mean": float(metrics["steps"].mean()),
        "episode_length_steps_mean": float(metrics.get("episode_length_steps", metrics["steps"]).mean()),
        "return_mean": float(metrics["return"].mean()),
        "episode_return_mean": float(metrics.get("episode_return", metrics["return"]).mean()),
        "completion_rate": float(metrics.get("task_completed", pd.Series([0.0])).mean()),
        "task_timeout_rate": float(metrics.get("task_timeout", pd.Series([0.0])).mean()),
        "task_distance_traveled_m_mean": float(metrics.get("task_distance_traveled_m", pd.Series([0.0])).mean()),
        "task_progress_ratio_mean": float(metrics.get("task_progress_ratio", pd.Series([0.0])).mean()),
        "episode_time_s_mean": _metric_mean(metrics, "episode_time_s", default=np.nan),
        "completion_time_s_mean": _metric_mean(metrics, "completion_time_s", default=np.nan),
        "distance_traveled_m_mean": _metric_mean(metrics, "distance_traveled_m"),
        "progress_rate_mps_mean": _metric_mean(metrics, "progress_rate_mps", default=np.nan),
        "return_std": float(metrics["return"].std()),
        "mean_speed": float(metrics["mean_speed"].mean()),
        "speed_std_mean": _metric_mean(metrics, "speed_std"),
        "mean_abs_speed_error": float(metrics["mean_abs_speed_error"].mean()),
        "mean_lat_y_error_m": float(metrics["mean_lat_y_error_m"].mean()),
        "mean_correction_norm": float(metrics["mean_correction_norm"].mean()),
        "max_correction_norm": float(metrics["max_correction_norm"].max()),
        "mean_meaningful_correction_norm": float(
            metrics.get("mean_meaningful_correction_norm", metrics["mean_correction_norm"]).mean()
        ),
        "max_meaningful_correction_norm": float(
            metrics.get("max_meaningful_correction_norm", metrics["max_correction_norm"]).max()
        ),
        "mean_raw_safe_gap_norm": _metric_mean(metrics, "mean_raw_safe_gap_norm"),
        "max_raw_safe_gap_norm": _metric_max(metrics, "max_raw_safe_gap_norm"),
        "event_intervention_rate": float(metrics["event_intervention_rate"].mean()),
        "numerical_intervention_rate": float(metrics["numerical_intervention_rate"].mean()),
        "qp_attempt_count_mean": _metric_mean(metrics, "qp_attempt_count"),
        "qp_failure_count_mean": _metric_mean(metrics, "qp_failure_count"),
        "qp_failure_rate": float(metrics["qp_failure_rate"].mean()),
        "min_h": _metric_min(metrics, "min_h"),
        "h_min": _metric_min(metrics, "h_min", default=_metric_min(metrics, "min_h")),
        "boundary_h_min": _metric_min(metrics, "boundary_h_min"),
        "mean_old_potential_cost": float(metrics["mean_old_potential_cost"].mean()),
        "mean_safety_potential_cost": float(metrics["mean_safety_potential_cost"].mean()),
        "mean_lateral_y_cost": float(metrics["mean_lateral_y_cost"].mean()),
        "ego_collisions_mean": float(metrics["ego_collisions"].mean()),
        "ego_collision_steps_mean": float(metrics["ego_collision_steps"].mean()),
        "total_collision_events_mean": float(metrics["total_collision_events"].mean()),
        "ego_collisions_per_km_mean": _metric_mean(metrics, "ego_collisions_per_km"),
        "total_collision_events_per_km_mean": _metric_mean(metrics, "total_collision_events_per_km"),
        "mean_abs_accel_x": _metric_mean(metrics, "mean_abs_accel_x"),
        "mean_abs_accel_y": _metric_mean(metrics, "mean_abs_accel_y"),
        "mean_accel_norm": _metric_mean(metrics, "mean_accel_norm"),
        "mean_abs_delta_accel_x": _metric_mean(metrics, "mean_abs_delta_accel_x"),
        "mean_abs_delta_accel_y": _metric_mean(metrics, "mean_abs_delta_accel_y"),
        "mean_delta_accel_norm": _metric_mean(metrics, "mean_delta_accel_norm"),
        "mean_abs_jerk_x": _metric_mean(metrics, "mean_abs_jerk_x"),
        "mean_abs_jerk_y": _metric_mean(metrics, "mean_abs_jerk_y"),
        "mean_jerk_norm": _metric_mean(metrics, "mean_jerk_norm"),
        "action_saturation_rate": _metric_mean(metrics, "action_saturation_rate"),
        "lateral_shift_total_m_mean": _metric_mean(metrics, "lateral_shift_total_m"),
        "lateral_shift_rate_mps_mean": _metric_mean(metrics, "lateral_shift_rate_mps", default=np.nan),
        "overtakes_count_mean": _metric_mean(metrics, "overtakes_count"),
        "overtaken_count_mean": _metric_mean(metrics, "overtaken_count"),
        "mean_neighbor_count": _metric_mean(metrics, "mean_neighbor_count"),
        "mean_neighbor_density_per_km": _metric_mean(metrics, "mean_neighbor_density_per_km"),
        "max_neighbor_density_per_km": _metric_max(metrics, "max_neighbor_density_per_km"),
    }


def behavior_score(row: dict[str, float | str | bool]) -> float:
    return float(
        300.0
        - 28.0 * float(row["mean_abs_speed_error"])
        - 18.0 * float(row["mean_lat_y_error_m"])
        - 120.0 * float(row["event_intervention_rate"])
        - 70.0 * float(row.get("mean_meaningful_correction_norm", row["mean_correction_norm"]))
        - 900.0 * float(row["qp_failure_rate"])
        - 450.0 * float(row["ego_collisions_mean"])
    )


class RewardAblationEvalCallback(BaseCallback):
    def __init__(
        self,
        namespace: dict[str, Any],
        *,
        trial_name: str,
        reward_config: dict[str, float],
        env_config: dict[str, Any],
        lambda_bc: float,
        event_threshold: float,
        k0: float,
        k1: float,
        eps_side: float,
        eval_freq: int,
        episodes: int,
        seed: int,
    ) -> None:
        super().__init__(verbose=1)
        self.namespace = namespace
        self.trial_name = trial_name
        self.reward_config = reward_config
        self.env_config = env_config
        self.lambda_bc = float(lambda_bc)
        self.event_threshold = float(event_threshold)
        self.k0 = float(k0)
        self.k1 = float(k1)
        self.eps_side = float(eps_side)
        self.eval_freq = int(eval_freq)
        self.episodes = int(episodes)
        self.seed = int(seed)
        self.records: list[dict[str, float | str]] = []
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        metrics = evaluate_model(
            self.namespace,
            self.model,
            episodes=self.episodes,
            seed=self.seed + self.num_timesteps,
            reward_config=self.reward_config,
            env_config=self.env_config,
            k0=self.k0,
            k1=self.k1,
            eps_side=self.eps_side,
            event_threshold=self.event_threshold,
        )
        row: dict[str, float | str] = {
            "trial_name": self.trial_name,
            "lambda_bc": self.lambda_bc,
            "timesteps": float(self.num_timesteps),
            **summarize(metrics),
        }
        row["behavior_score"] = behavior_score(row)
        self.records.append(row)
        print(
            "[reward-eval]"
            f" {self.trial_name}"
            f" steps={self.num_timesteps:,}"
            f" abs_speed={row['mean_abs_speed_error']:.3f}"
            f" lat_y={row['mean_lat_y_error_m']:.3f}"
            f" event_int={row['event_intervention_rate']:.2%}"
            f" corr={row['mean_correction_norm']:.3f}"
            f" score={row['behavior_score']:.1f}",
            flush=True,
        )
        return True


def plot_history(history: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.0))
    axes = axes.ravel()
    correction_column = (
        "mean_meaningful_correction_norm" if "mean_meaningful_correction_norm" in history.columns else "mean_correction_norm"
    )
    if not history.empty:
        axes[0].plot(history["timesteps"], history["mean_abs_speed_error"], marker="o")
        axes[1].plot(history["timesteps"], history["mean_lat_y_error_m"], marker="o")
        axes[2].plot(history["timesteps"], history["event_intervention_rate"], marker="o")
        axes[3].plot(history["timesteps"], history[correction_column], marker="o")
    axes[0].set_title("Abs Speed Error")
    axes[1].set_title("Lateral y Error")
    axes[2].set_title("Meaningful Intervention")
    axes[3].set_title("Meaningful Correction")
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    for axis in axes:
        axis.set_xlabel("Training timestep")
        axis.grid(True, alpha=0.28)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_aggregate(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    ranked = summary.sort_values("behavior_score", ascending=False).reset_index(drop=True)
    labels = ranked["trial_name"].tolist()
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 4, figsize=(18, max(4.8, 0.5 * len(labels))))
    correction_column = (
        "mean_meaningful_correction_norm" if "mean_meaningful_correction_norm" in ranked.columns else "mean_correction_norm"
    )
    panels = [
        ("mean_abs_speed_error", "Abs Speed Error", False),
        ("mean_lat_y_error_m", "Lateral y Error", False),
        ("event_intervention_rate", "Meaningful Intervention", True),
        (correction_column, "Meaningful Correction", False),
    ]
    for axis, (column, title, percent) in zip(axes, panels):
        axis.barh(y, ranked[column].to_numpy(dtype=float), color="#1f77b4")
        axis.set_yticks(y)
        axis.set_yticklabels(labels if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(True, axis="x", alpha=0.25)
        if percent:
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    fig.suptitle("Reward-Term and Actor-Loss Ablation", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate lateral-y reward, old potential, CBF safety potential, and actor BC loss.")
    parser.add_argument("--k0", type=float, default=5.29)
    parser.add_argument("--k1", type=float, default=3.68)
    parser.add_argument("--eps-side", type=float, default=0.149)
    parser.add_argument("--event-threshold", type=float, default=0.03)
    parser.add_argument("--lambda-norm", type=float, default=0.025)
    parser.add_argument("--lambda-event", type=float, default=0.02)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--train-eval-freq", type=int, default=5_000)
    parser.add_argument("--train-eval-episodes", type=int, default=2)
    parser.add_argument("--final-eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=623_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-envs", type=int, default=1)
    add_env_config_args(parser)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run only two tiny trials to validate plumbing.")
    return parser.parse_args()


def main() -> int:
    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()

    project_root = find_project_root(args.project_root or Path.cwd())
    env_dir = project_root / "laneless highway env"
    if env_dir.exists():
        sys.path.insert(0, str(env_dir))
        import lane_free_env  # noqa: F401
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "ARTIFACT_DIR": project_root / "artifacts" / "lanelessKaralakou",
        "np": np,
        "pd": pd,
        "gym": gym,
        "torch": torch,
        "Path": Path,
        "json": json,
        "os": os,
        "sys": sys,
        "time": time,
        "warnings": warnings,
        "datetime": datetime,
        "Any": Any,
        "Callable": Callable,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "Union": Union,
        "BaseCallback": BaseCallback,
        "CallbackList": CallbackList,
        "Monitor": Monitor,
    }
    exec_notebook_cells(notebook_path, NOTEBOOK_DEPS, namespace)
    install_minimal_guided_cbf(namespace)
    namespace["DEVICE"] = args.device
    install_safety_set_reward_wrapper(namespace)
    install_event_penalty_env(namespace)
    env_config = env_config_from_args(args, namespace["ENV_CONFIG"])
    traffic_model = active_traffic_model(env_config)

    trials = DEFAULT_TRIALS
    if args.smoke:
        trials = [DEFAULT_TRIALS[0], DEFAULT_TRIALS[3]]
        args.timesteps = min(args.timesteps, 1_000)
        args.train_eval_freq = min(args.train_eval_freq, 500)
        args.train_eval_episodes = min(args.train_eval_episodes, 1)
        args.final_eval_episodes = min(args.final_eval_episodes, 2)

    default_output_name = "cbf_reward_term_ablation_mtm" if traffic_model == "mtm" else "cbf_reward_term_ablation"
    output_dir = args.output_dir or (Path(namespace["ARTIFACT_DIR"]) / default_output_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trials).to_csv(output_dir / "trial_configs.csv", index=False)

    print(
        "[reward] starting ablation",
        {
            "trials": len(trials),
            "timesteps": args.timesteps,
            "lambda_norm": args.lambda_norm,
            "lambda_event": args.lambda_event,
            "traffic_model": traffic_model,
            "k0": args.k0,
            "k1": args.k1,
            "eps_side": args.eps_side,
            "output_dir": str(output_dir),
        },
        flush=True,
    )

    final_rows: list[dict[str, float | str | bool]] = []
    for index, trial in enumerate(trials, start=1):
        trial_name = str(trial["trial_name"])
        trial_dir = output_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        model_path = trial_dir / "model.zip"
        history_path = trial_dir / "train_eval_history.csv"
        final_episodes_path = trial_dir / "final_eval_episodes.csv"
        final_summary_path = trial_dir / "final_summary.csv"
        plot_path = trial_dir / "train_eval_history.png"

        if not args.no_resume and final_summary_path.exists() and model_path.exists():
            print(f"[reward] [{index}/{len(trials)}] {trial_name} complete; loading summary", flush=True)
            final_rows.append(pd.read_csv(final_summary_path).iloc[0].to_dict())
            continue

        reward_config = make_reward_config(namespace, trial)
        lambda_bc = float(trial["lambda_bc"])
        print(
            f"[reward] [{index}/{len(trials)}] {trial_name}"
            f" wy={reward_config['wy']:g}"
            f" old_potential={bool(trial['use_current_potential'])}"
            f" safety_potential={bool(trial['use_safety_potential'])}"
            f" w_safe={reward_config['w_safe']:g}"
            f" lambda_bc={lambda_bc:g}",
            flush=True,
        )
        train_env = make_training_env(
            namespace,
            seed=args.seed + index * 1_000,
            reward_config=reward_config,
            env_config=env_config,
            lambda_norm=args.lambda_norm,
            lambda_event=args.lambda_event,
            event_threshold=args.event_threshold,
            k0=args.k0,
            k1=args.k1,
            eps_side=args.eps_side,
            n_envs=args.n_envs,
        )
        print(
            f"[reward] vec_env={type(train_env).__name__} | n_envs={args.n_envs} | UTD=1",
            flush=True,
        )
        n_actions = train_env.action_space.shape[-1]
        action_noise = namespace["make_ou_action_noise"](n_actions, n_envs=args.n_envs)
        callback = RewardAblationEvalCallback(
            namespace,
            trial_name=trial_name,
            reward_config=reward_config,
            env_config=env_config,
            lambda_bc=lambda_bc,
            event_threshold=args.event_threshold,
            k0=args.k0,
            k1=args.k1,
            eps_side=args.eps_side,
            eval_freq=args.train_eval_freq,
            episodes=args.train_eval_episodes,
            seed=args.seed + index * 10_000,
        )
        model = namespace["GuidedCBFDDPG"](
            "MlpPolicy",
            train_env,
            learning_rate=namespace["DDPG_LEARNING_RATE"],
            buffer_size=namespace["DDPG_REPLAY_MEMORY"],
            learning_starts=namespace["DDPG_LEARNING_STARTS"],
            batch_size=namespace["DDPG_BATCH_SIZE"],
            tau=namespace["DDPG_TAU"],
            gamma=namespace["DDPG_GAMMA"],
            train_freq=(1, "step"),
            gradient_steps=int(namespace.get("DDPG_GRADIENT_STEPS", -1)),
            action_noise=action_noise,
            policy_kwargs={"net_arch": [256, 128]},
            tensorboard_log=None,
            verbose=0,
            seed=args.seed + index,
            device=args.device,
            lambda_bc=lambda_bc,
            bc_delta=namespace["GUIDED_CBF_BC_DELTA"],
            bc_action_scale=namespace["GUIDED_CBF_ACTION_SCALE"],
            bc_weight_max=namespace["GUIDED_CBF_WEIGHT_MAX"],
        )

        start = time.time()
        try:
            model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
        finally:
            train_env.close()
        elapsed_sec = time.time() - start
        model.save(str(model_path))

        history = pd.DataFrame(callback.records)
        history.to_csv(history_path, index=False)
        plot_history(history, plot_path, title=trial_name)

        print(f"[reward] evaluating {trial_name}", flush=True)
        final_metrics = evaluate_model(
            namespace,
            model,
            episodes=args.final_eval_episodes,
            seed=args.seed + index * 100_000,
            reward_config=reward_config,
            env_config=env_config,
            k0=args.k0,
            k1=args.k1,
            eps_side=args.eps_side,
            event_threshold=args.event_threshold,
        )
        final_metrics.to_csv(final_episodes_path, index=False)
        summary: dict[str, float | str | bool] = {
            "trial_name": trial_name,
            "model_path": str(model_path),
            "elapsed_sec": float(elapsed_sec),
            "timesteps": float(args.timesteps),
            "k0": float(args.k0),
            "k1": float(args.k1),
            "eps_side": float(args.eps_side),
            "event_threshold": float(args.event_threshold),
            "traffic_model": traffic_model,
            "lambda_norm": float(args.lambda_norm),
            "lambda_event": float(args.lambda_event),
            "lambda_bc": lambda_bc,
            "wy": float(reward_config["wy"]),
            "wf": float(reward_config["wf"]),
            "w_safe": float(reward_config["w_safe"]),
            "use_current_potential": bool(trial["use_current_potential"]),
            "use_safety_potential": bool(trial["use_safety_potential"]),
            **summarize(final_metrics),
        }
        summary["behavior_score"] = behavior_score(summary)
        pd.DataFrame([summary]).to_csv(final_summary_path, index=False)
        (trial_dir / "run_config.json").write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        final_rows.append(summary)
        print(
            "[reward-result]"
            f" {trial_name}"
            f" abs_speed={summary['mean_abs_speed_error']:.3f}"
            f" lat_y={summary['mean_lat_y_error_m']:.3f}"
            f" event_int={summary['event_intervention_rate']:.2%}"
            f" corr={summary['mean_correction_norm']:.3f}"
            f" qp_fail={summary['qp_failure_rate']:.2%}"
            f" score={summary['behavior_score']:.1f}",
            flush=True,
        )

    aggregate = pd.DataFrame(final_rows)
    if not aggregate.empty:
        aggregate = aggregate.sort_values("behavior_score", ascending=False).reset_index(drop=True)
    aggregate_path = output_dir / "reward_ablation_summary.csv"
    aggregate_plot_path = output_dir / "reward_ablation_comparison.png"
    aggregate.to_csv(aggregate_path, index=False)
    plot_aggregate(aggregate, aggregate_plot_path)
    print(f"[reward] wrote {aggregate_path}", flush=True)
    print(f"[reward] wrote {aggregate_plot_path}", flush=True)
    if not aggregate.empty:
        display_cols = [
            "trial_name",
            "wy",
            "wf",
            "w_safe",
            "lambda_bc",
            "mean_abs_speed_error",
            "mean_lat_y_error_m",
            "event_intervention_rate",
            "mean_correction_norm",
            "qp_failure_rate",
            "behavior_score",
        ]
        display_cols = [c for c in display_cols if c in aggregate.columns]
        print(aggregate[display_cols].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
