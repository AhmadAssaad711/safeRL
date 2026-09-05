from __future__ import annotations

import argparse
import faulthandler
import json
import os
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor

from scripts.training.cbf_lambda_event_bc_pilot_sweep import (
    exec_notebook_cells,
    find_project_root,
    install_event_penalty_env,
    set_stable_native_defaults,
)
from scripts.training.cbf_reward_term_ablation import (
    NOTEBOOK_DEPS,
    install_safety_set_reward_wrapper,
)
from scripts.common.guided_cbf_minimal import install_minimal_guided_cbf
from scripts.common.laneless_script_config import add_env_config_args
from scripts.training.train_safety_potential_variants import (
    MTM_CONGESTED_UNCERTAIN_UPDATES,
    SAFETY_REWARD_TRIAL,
    TB_VARIANT_RUN_NAMES,
    VARIANTS,
    deep_update,
    make_reward_config,
)


warnings.filterwarnings("ignore", message="OSQP exited.*")

MODE_LABELS = {
    "raw_actor": "Raw actor",
    "actor_cbf": "Actor + CBF",
    "random_cbf": "Random actor + CBF",
    "rule_cbf": "Rule actor + CBF",
}


def _as_float(value: Any, default: float = np.nan) -> float:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return default
    return scalar if np.isfinite(scalar) else default


def _series(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _mean(values: list[float], default: float = np.nan) -> float:
    array = _series(values)
    return float(np.mean(array)) if array.size else default


def _std(values: list[float], default: float = np.nan) -> float:
    array = _series(values)
    return float(np.std(array, ddof=0)) if array.size else default


def _min(values: list[float], default: float = np.nan) -> float:
    array = _series(values)
    return float(np.min(array)) if array.size else default


def _max(values: list[float], default: float = np.nan) -> float:
    array = _series(values)
    return float(np.max(array)) if array.size else default


def _p95(values: list[float], default: float = np.nan) -> float:
    array = _series(values)
    return float(np.percentile(array, 95)) if array.size else default


def normalized_to_physical(env: gym.Env, action: np.ndarray) -> np.ndarray:
    base = env.unwrapped
    action = np.asarray(action, dtype=float).reshape(-1)[:2]
    if hasattr(base, "_map_action"):
        return np.asarray(
            [
                base._map_action(float(action[0]), "longitudinal"),
                base._map_action(float(action[1]), "lateral"),
            ],
            dtype=np.float32,
        )
    bounds = base.config["bounds"]
    lows = np.asarray([bounds["ax_min"], bounds["ay_min"]], dtype=np.float32)
    highs = np.asarray([bounds["ax_max"], bounds["ay_max"]], dtype=np.float32)
    return (lows + 0.5 * (np.clip(action, -1.0, 1.0) + 1.0) * (highs - lows)).astype(np.float32)


def physical_bounds(env_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    bounds = env_config["bounds"]
    low = np.asarray([bounds["ax_min"], bounds["ay_min"]], dtype=np.float32)
    high = np.asarray([bounds["ax_max"], bounds["ay_max"]], dtype=np.float32)
    return low, high


def model_action_is_normalized(model: Any) -> bool:
    low = np.asarray(model.action_space.low, dtype=float).reshape(-1)[:2]
    high = np.asarray(model.action_space.high, dtype=float).reshape(-1)[:2]
    return bool(np.allclose(low, -1.0, atol=1e-5) and np.allclose(high, 1.0, atol=1e-5))


def model_action_to_physical(model: Any, env: gym.Env, action: np.ndarray, env_config: dict[str, Any]) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
    if model_action_is_normalized(model):
        return normalized_to_physical(env, action)
    low, high = physical_bounds(env_config)
    return np.clip(action, low, high).astype(np.float32)


def physical_to_normalized(namespace: dict[str, Any], env: gym.Env, action_phys: np.ndarray) -> np.ndarray:
    return np.asarray(namespace["_physical_to_normalized_action"](env, action_phys), dtype=np.float32).reshape(-1)[:2]


def make_raw_eval_env(
    namespace: dict[str, Any],
    *,
    seed: int,
    reward_config: dict[str, float],
    env_config: dict[str, Any],
    task_distance_m: float,
    task_max_steps: int,
) -> gym.Env:
    env = gym.make("lane-free-v0", render_mode=None, config=env_config)
    env = namespace["KaralakouRewardWrapper"](env, reward_config=reward_config)
    if namespace["NORMALIZE_RL_OBSERVATIONS"]:
        env = namespace["LaneFreeObservationNormalizationWrapper"](env, clip=namespace["OBSERVATION_CLIP"])
    if "KPIInfoWrapper" in namespace:
        env = namespace["KPIInfoWrapper"](env)
    env = Monitor(env)
    env = namespace["make_task_evaluation_wrapper"](
        env,
        task_distance_m=float(task_distance_m),
        max_steps=int(task_max_steps),
    )
    env.reset(seed=seed)
    return env


def make_cbf_eval_env(
    namespace: dict[str, Any],
    *,
    seed: int,
    reward_config: dict[str, float],
    env_config: dict[str, Any],
    event_threshold: float,
    k0: float,
    k1: float,
    eps_side: float,
    task_distance_m: float,
    task_max_steps: int,
) -> gym.Env:
    env = namespace["make_event_cbf_single_env"](
        seed=seed,
        lambda_norm=0.0,
        lambda_event=0.0,
        event_threshold=float(event_threshold),
        k0=float(k0),
        k1=float(k1),
        eps_side=float(eps_side),
        env_config=env_config,
        reward_config=reward_config,
    )
    env = namespace["make_task_evaluation_wrapper"](
        env,
        task_distance_m=float(task_distance_m),
        max_steps=int(task_max_steps),
    )
    env.reset(seed=seed)
    return env


def simple_rule_action(env: gym.Env, *, kv: float, ky: float, kvy: float, env_config: dict[str, Any]) -> np.ndarray:
    base = env.unwrapped
    ego = base.vehicle
    road_width = float(base.config["road_width"])
    y_error = float(ego.position[1] - 0.5 * road_width)
    ax = kv * float(ego.desired_speed - ego.vx)
    ay = -ky * y_error - kvy * float(getattr(ego, "vy", 0.0))
    low, high = physical_bounds(env_config)
    return np.clip(np.asarray([ax, ay], dtype=np.float32), low, high).astype(np.float32)


def policy_action_phys(
    *,
    mode: str,
    model: Any | None,
    env: gym.Env,
    obs: np.ndarray,
    env_config: dict[str, Any],
    rng: np.random.Generator,
    rule_kv: float,
    rule_ky: float,
    rule_kvy: float,
) -> np.ndarray:
    low, high = physical_bounds(env_config)
    if mode == "random_cbf":
        return rng.uniform(low=low, high=high).astype(np.float32)
    if mode == "rule_cbf":
        return simple_rule_action(env, kv=rule_kv, ky=rule_ky, kvy=rule_kvy, env_config=env_config)
    if model is None:
        raise ValueError(f"Mode {mode!r} requires a model")
    action, _ = model.predict(obs, deterministic=True)
    return model_action_to_physical(model, env, action, env_config)


def load_model_for_variant(namespace: dict[str, Any], variant: str, checkpoint_path: Path, device: str) -> Any:
    variant_cfg = next((item for item in VARIANTS if str(item["variant"]) == variant), {})
    model_cls = namespace["GuidedCBFDDPG"] if variant_cfg.get("model_class_name") == "GuidedCBFDDPG" else DDPG
    return model_cls.load(str(checkpoint_path), device=device)


def install_safe_reward_overtake_counter(namespace: dict[str, Any]) -> None:
    def _safe_overtakes(self, previous_dx):  # noqa: ANN001
        base = self.base_env
        ego = base.vehicle
        sensing_range = float(base.config["sensing_range"])
        road_length = float(base.config.get("road_length", 0.0))
        overtakes = 0
        for vehicle in list(base.road.vehicles):
            if vehicle is ego:
                continue
            old_dx = previous_dx.get(id(vehicle))
            if old_dx is None:
                continue
            new_dx = float(vehicle.position[0] - ego.position[0])
            if road_length > 1e-6:
                new_dx = float((new_dx + 0.5 * road_length) % road_length - 0.5 * road_length)
            if 0.0 < float(old_dx) < sensing_range and new_dx < -0.5 * float(ego.length):
                overtakes += 1
        return overtakes

    for class_key in ["KaralakouRewardWrapper", "SafetySetKaralakouRewardWrapper"]:
        cls = namespace.get(class_key)
        if cls is not None:
            setattr(cls, "_overtake_count", _safe_overtakes)


def find_checkpoint(source_dir: Path, variant: str, step: int) -> Path:
    candidates = [
        source_dir / variant / f"ckpt_{int(step):06d}.zip",
        source_dir / variant / f"checkpoint_{int(step):06d}.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing checkpoint for {variant} at {step:,}: {candidates[0]}")


def evaluate_one(
    namespace: dict[str, Any],
    *,
    model: Any | None,
    variant: str,
    mode: str,
    checkpoint_step: int,
    episodes: int,
    seed: int,
    reward_config: dict[str, float],
    env_config: dict[str, Any],
    event_threshold: float,
    k0: float,
    k1: float,
    eps_side: float,
    task_distance_m: float,
    task_max_steps: int,
    rule_kv: float,
    rule_ky: float,
    rule_kvy: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for episode in range(int(episodes)):
        episode_seed = int(seed) + int(checkpoint_step) * 10 + episode
        use_cbf = mode.endswith("_cbf")
        env = (
            make_cbf_eval_env(
                namespace,
                seed=episode_seed,
                reward_config=reward_config,
                env_config=env_config,
                event_threshold=event_threshold,
                k0=k0,
                k1=k1,
                eps_side=eps_side,
                task_distance_m=task_distance_m,
                task_max_steps=task_max_steps,
            )
            if use_cbf
            else make_raw_eval_env(
                namespace,
                seed=episode_seed,
                reward_config=reward_config,
                env_config=env_config,
                task_distance_m=task_distance_m,
                task_max_steps=task_max_steps,
            )
        )
        rng = np.random.default_rng(episode_seed + 9973)
        obs, _ = env.reset(seed=episode_seed)
        done = False
        step_count = 0
        rewards: list[float] = []
        speeds: list[float] = []
        abs_speed_errors: list[float] = []
        lat_y_errors: list[float] = []
        h_values: list[float] = []
        boundary_h_values: list[float] = []
        qp_attempts: list[float] = []
        qp_failures: list[float] = []
        delta_u_values: list[float] = []
        delta_ax_values: list[float] = []
        delta_ay_values: list[float] = []
        raw_action_norms: list[float] = []
        safe_action_norms: list[float] = []
        event_interventions: list[float] = []
        numerical_interventions: list[float] = []
        neighbor_constraints: list[float] = []
        info_rows: list[dict[str, Any]] = []
        last_task_info: dict[str, Any] = {}
        ego_collisions = 0
        total_collision_events = 0

        while not done:
            raw_phys = policy_action_phys(
                mode=mode,
                model=model,
                env=env,
                obs=obs,
                env_config=env_config,
                rng=rng,
                rule_kv=rule_kv,
                rule_ky=rule_ky,
                rule_kvy=rule_kvy,
            )
            if use_cbf:
                obs, reward, terminated, truncated, info = env.step(raw_phys)
                safe_phys = np.asarray(
                    [
                        _as_float(info.get("cbf_a_safe_x"), default=raw_phys[0]),
                        _as_float(info.get("cbf_a_safe_y"), default=raw_phys[1]),
                    ],
                    dtype=np.float32,
                )
                raw_logged = np.asarray(
                    [
                        _as_float(info.get("cbf_a_rl_x"), default=raw_phys[0]),
                        _as_float(info.get("cbf_a_rl_y"), default=raw_phys[1]),
                    ],
                    dtype=np.float32,
                )
            else:
                safe_phys = raw_phys.copy()
                raw_logged = raw_phys.copy()
                obs, reward, terminated, truncated, info = env.step(physical_to_normalized(namespace, env, raw_phys))

            info = dict(info)
            base = env.unwrapped
            ego = base.vehicle
            speed = float(ego.vx)
            desired_speed = float(ego.desired_speed)
            delta = safe_phys - raw_logged
            delta_abs = np.abs(delta)
            delta_norm = float(np.linalg.norm(delta))
            intervention = float(delta_norm > float(event_threshold))
            numerical_intervention = float(delta_norm > 1e-6)

            rewards.append(float(reward))
            speeds.append(speed)
            abs_speed_errors.append(abs(speed - desired_speed))
            lat_error = _as_float(info.get("karalakou_lat_y_error_m"))
            if np.isfinite(lat_error):
                lat_y_errors.append(lat_error)
            h_values.append(_as_float(info.get("kpi_h_min", info.get("cbf_min_h"))))
            boundary_h_values.append(_as_float(info.get("kpi_boundary_h_min", info.get("cbf_min_boundary_h"))))
            qp_attempt = float("cbf_qp_success" in info or "qp_success" in info)
            qp_success = bool(info.get("cbf_qp_success", info.get("qp_success", True)))
            qp_attempts.append(qp_attempt)
            qp_failures.append(float(qp_attempt > 0.5 and not qp_success))
            delta_u_values.append(delta_norm)
            delta_ax_values.append(float(delta_abs[0]))
            delta_ay_values.append(float(delta_abs[1]))
            raw_action_norms.append(float(np.linalg.norm(raw_logged)))
            safe_action_norms.append(float(np.linalg.norm(safe_phys)))
            event_interventions.append(intervention)
            numerical_interventions.append(numerical_intervention)
            neighbor_constraints.append(_as_float(info.get("cbf_num_neighbor_constraints")))
            info_rows.append(info)
            last_task_info = {
                "task_distance_m": _as_float(info.get("task_distance_m"), default=task_distance_m),
                "task_distance_traveled_m": _as_float(info.get("task_distance_traveled_m"), default=0.0),
                "task_progress_ratio": _as_float(info.get("task_progress_ratio"), default=0.0),
                "task_completed": bool(info.get("task_completed", False)),
                "task_timeout": bool(info.get("task_timeout", False)),
            }
            ego_collisions += int(info.get("ego_collision_events", info.get("kpi_ego_collision_events", 0)))
            if bool(info.get("ego_collision", False)) and int(info.get("ego_collision_events", 0)) <= 0:
                ego_collisions += 1
            total_collision_events += int(info.get("collisions", info.get("total_collision_events", 0)))
            step_count += 1
            done = bool(terminated or truncated)

        row: dict[str, Any] = {
            "variant": variant,
            "mode": mode,
            "mode_label": MODE_LABELS.get(mode, mode),
            "checkpoint_step": float(checkpoint_step),
            "episode": float(episode),
            "seed": float(episode_seed),
            "steps": float(step_count),
            "episode_length_steps": float(step_count),
            "return": float(np.sum(rewards)),
            "episode_return": float(np.sum(rewards)),
            "task_completed": float(last_task_info.get("task_completed", False)),
            "task_timeout": float(last_task_info.get("task_timeout", False)),
            "task_distance_m": float(last_task_info.get("task_distance_m", task_distance_m)),
            "task_distance_traveled_m": float(last_task_info.get("task_distance_traveled_m", 0.0)),
            "task_progress_ratio": float(last_task_info.get("task_progress_ratio", 0.0)),
            "mean_speed": _mean(speeds, default=0.0),
            "mean_abs_speed_error": _mean(abs_speed_errors, default=0.0),
            "mean_lat_y_error_m": _mean(lat_y_errors),
            "ego_collisions": float(ego_collisions),
            "total_collision_events": float(total_collision_events),
            "h_min": _min(h_values),
            "boundary_h_min": _min(boundary_h_values),
            "qp_attempt_count": float(np.sum(_series(qp_attempts))) if qp_attempts else 0.0,
            "qp_failure_count": float(np.sum(_series(qp_failures))) if qp_failures else 0.0,
            "qp_failure_rate": (
                float(np.sum(_series(qp_failures)) / max(np.sum(_series(qp_attempts)), 1e-9))
                if np.sum(_series(qp_attempts)) > 0.0
                else np.nan
            ),
            "event_intervention_rate": _mean(event_interventions, default=0.0),
            "numerical_intervention_rate": _mean(numerical_interventions, default=0.0),
            "mean_correction_norm": _mean(delta_u_values, default=0.0),
            "max_correction_norm": _max(delta_u_values, default=0.0),
            "p95_correction_norm": _p95(delta_u_values, default=0.0),
            "mean_delta_ax_abs": _mean(delta_ax_values, default=0.0),
            "p95_delta_ax_abs": _p95(delta_ax_values, default=0.0),
            "mean_delta_ay_abs": _mean(delta_ay_values, default=0.0),
            "p95_delta_ay_abs": _p95(delta_ay_values, default=0.0),
            "mean_raw_action_norm": _mean(raw_action_norms, default=0.0),
            "mean_safe_action_norm": _mean(safe_action_norms, default=0.0),
            "mean_neighbor_constraints": _mean(neighbor_constraints),
            "max_neighbor_constraints": _max(neighbor_constraints),
        }
        distance_km = max(float(row["task_distance_traveled_m"]) / 1000.0, 1e-9)
        row["ego_collisions_per_km"] = float(row["ego_collisions"]) / distance_km
        row["total_collision_events_per_km"] = float(row["total_collision_events"]) / distance_km
        summarize_episode_kpis = namespace.get("summarize_episode_kpis")
        if callable(summarize_episode_kpis):
            row.update(
                summarize_episode_kpis(
                    info_rows,
                    rewards=rewards,
                    task_completed=bool(last_task_info.get("task_completed", False)),
                    fallback_steps=step_count,
                    fallback_distance_m=float(last_task_info.get("task_distance_traveled_m", 0.0)),
                    fallback_dt_s=namespace["kpi_policy_dt"](env) if "kpi_policy_dt" in namespace else np.nan,
                )
            )
            row["event_intervention_rate"] = _mean(event_interventions, default=0.0)
            row["mean_correction_norm"] = _mean(delta_u_values, default=0.0)
            row["p95_correction_norm"] = _p95(delta_u_values, default=0.0)
            row["mean_delta_ax_abs"] = _mean(delta_ax_values, default=0.0)
            row["mean_delta_ay_abs"] = _mean(delta_ay_values, default=0.0)
        rows.append(row)
        env.close()
    return pd.DataFrame(rows)


def aggregate_episode_rows(episodes: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["variant", "checkpoint_step", "mode", "mode_label"]
    mean_cols = [
        "return",
        "episode_length_steps",
        "task_completed",
        "task_distance_traveled_m",
        "task_progress_ratio",
        "ego_collisions",
        "ego_collisions_per_km",
        "total_collision_events",
        "mean_abs_speed_error",
        "mean_lat_y_error_m",
        "h_min",
        "boundary_h_min",
        "qp_failure_rate",
        "event_intervention_rate",
        "numerical_intervention_rate",
        "mean_correction_norm",
        "max_correction_norm",
        "p95_correction_norm",
        "mean_delta_ax_abs",
        "p95_delta_ax_abs",
        "mean_delta_ay_abs",
        "p95_delta_ay_abs",
        "mean_raw_action_norm",
        "mean_safe_action_norm",
        "mean_neighbor_constraints",
        "max_neighbor_constraints",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in episodes.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["episodes"] = float(len(group))
        for column in mean_cols:
            if column not in group:
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            if values.notna().any():
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_std"] = float(values.std(ddof=0))
        for column in ["h_min", "boundary_h_min"]:
            if column in group:
                values = pd.to_numeric(group[column], errors="coerce")
                if values.notna().any():
                    row[column] = float(values.min())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["variant", "checkpoint_step", "mode"]).reset_index(drop=True)


def compute_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, checkpoint_step), group in summary.groupby(["variant", "checkpoint_step"], dropna=False):
        by_mode = {str(row["mode"]): row for _, row in group.iterrows()}
        raw = by_mode.get("raw_actor")
        filtered = by_mode.get("actor_cbf")
        random_cbf = by_mode.get("random_cbf")
        if raw is None or filtered is None:
            continue
        row: dict[str, Any] = {
            "variant": variant,
            "checkpoint_step": checkpoint_step,
            "delta_R_filtered_minus_raw": float(filtered.get("return_mean", np.nan) - raw.get("return_mean", np.nan)),
            "delta_C_raw_minus_filtered": float(
                raw.get("ego_collisions_mean", np.nan) - filtered.get("ego_collisions_mean", np.nan)
            ),
            "delta_Cpkm_raw_minus_filtered": float(
                raw.get("ego_collisions_per_km_mean", np.nan)
                - filtered.get("ego_collisions_per_km_mean", np.nan)
            ),
            "delta_Ev_raw_minus_filtered": float(
                raw.get("mean_abs_speed_error_mean", np.nan)
                - filtered.get("mean_abs_speed_error_mean", np.nan)
            ),
            "delta_Ey_raw_minus_filtered": float(
                raw.get("mean_lat_y_error_m_mean", np.nan)
                - filtered.get("mean_lat_y_error_m_mean", np.nan)
            ),
            "filtered_intervention_rate": float(filtered.get("event_intervention_rate_mean", np.nan)),
            "filtered_correction_norm": float(filtered.get("mean_correction_norm_mean", np.nan)),
            "filtered_p95_correction_norm": float(filtered.get("p95_correction_norm_mean", np.nan)),
            "filtered_h_min": float(filtered.get("h_min", np.nan)),
            "raw_h_min": float(raw.get("h_min", np.nan)),
            "raw_return": float(raw.get("return_mean", np.nan)),
            "filtered_return": float(filtered.get("return_mean", np.nan)),
            "raw_ego_collisions": float(raw.get("ego_collisions_mean", np.nan)),
            "filtered_ego_collisions": float(filtered.get("ego_collisions_mean", np.nan)),
        }
        if random_cbf is not None:
            row["delta_R_actor_cbf_minus_random_cbf"] = float(
                filtered.get("return_mean", np.nan) - random_cbf.get("return_mean", np.nan)
            )
            row["delta_C_random_cbf_minus_actor_cbf"] = float(
                random_cbf.get("ego_collisions_mean", np.nan) - filtered.get("ego_collisions_mean", np.nan)
            )
            row["random_cbf_return"] = float(random_cbf.get("return_mean", np.nan))
            row["random_cbf_ego_collisions"] = float(random_cbf.get("ego_collisions_mean", np.nan))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["variant", "checkpoint_step"])
    return pd.DataFrame(rows).sort_values(["variant", "checkpoint_step"]).reset_index(drop=True)


def write_tensorboard(namespace: dict[str, Any], summary: pd.DataFrame, diagnostics: pd.DataFrame, tb_root: Path) -> None:
    writer_cls = namespace.get("SummaryWriter")
    if writer_cls is None:
        from torch.utils.tensorboard import SummaryWriter as writer_cls  # type: ignore[no-redef]

    for _, row in summary.iterrows():
        variant = str(row["variant"])
        mode = str(row["mode"])
        run_name = f"{TB_VARIANT_RUN_NAMES.get(variant, variant)}/{mode}"
        run_dir = tb_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        writer = writer_cls(log_dir=str(run_dir))
        step = int(float(row["checkpoint_step"]))
        writer.add_scalar("00_reward/return_mean", _as_float(row.get("return_mean")), step)
        writer.add_scalar("01_task/episode_length_steps", _as_float(row.get("episode_length_steps_mean")), step)
        writer.add_scalar("01_task/completion_rate", _as_float(row.get("task_completed_mean")), step)
        writer.add_scalar("02_safety/ego_collisions", _as_float(row.get("ego_collisions_mean")), step)
        writer.add_scalar("02_safety/ego_collisions_per_km", _as_float(row.get("ego_collisions_per_km_mean")), step)
        writer.add_scalar("02_safety/h_min", _as_float(row.get("h_min")), step)
        writer.add_scalar("02_safety/boundary_h_min", _as_float(row.get("boundary_h_min")), step)
        writer.add_scalar("02_safety/qp_failure_rate", _as_float(row.get("qp_failure_rate_mean")), step)
        writer.add_scalar("03_efficiency/speed_error_abs", _as_float(row.get("mean_abs_speed_error_mean")), step)
        writer.add_scalar("05_filter/intervention_rate", _as_float(row.get("event_intervention_rate_mean")), step)
        writer.add_scalar("05_filter/correction_norm_mean", _as_float(row.get("mean_correction_norm_mean")), step)
        writer.add_scalar("05_filter/correction_norm_p95", _as_float(row.get("p95_correction_norm_mean")), step)
        writer.add_scalar("05_filter/delta_ax_abs_mean", _as_float(row.get("mean_delta_ax_abs_mean")), step)
        writer.add_scalar("05_filter/delta_ay_abs_mean", _as_float(row.get("mean_delta_ay_abs_mean")), step)
        writer.add_scalar("06_traffic/neighbor_constraints_mean", _as_float(row.get("mean_neighbor_constraints_mean")), step)
        writer.flush()
        writer.close()

    for _, row in diagnostics.iterrows():
        variant = str(row["variant"])
        run_name = f"{TB_VARIANT_RUN_NAMES.get(variant, variant)}/diagnostics"
        run_dir = tb_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        writer = writer_cls(log_dir=str(run_dir))
        step = int(float(row["checkpoint_step"]))
        for key, value in row.items():
            if key in {"variant", "checkpoint_step"}:
                continue
            writer.add_scalar(f"diagnostics/{key}", _as_float(value), step)
        writer.flush()
        writer.close()


def plot_summary(summary: pd.DataFrame, diagnostics: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    variants = list(dict.fromkeys(summary["variant"].astype(str)))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), sharex=False)
    for variant in variants:
        frame = summary[summary["variant"] == variant]
        for mode in ["raw_actor", "actor_cbf", "random_cbf"]:
            mode_frame = frame[frame["mode"] == mode].sort_values("checkpoint_step")
            if mode_frame.empty:
                continue
            label = f"{variant} / {MODE_LABELS.get(mode, mode)}"
            axes[0, 0].plot(mode_frame["checkpoint_step"], mode_frame["return_mean"], marker="o", label=label)
            axes[0, 1].plot(
                mode_frame["checkpoint_step"],
                mode_frame["ego_collisions_per_km_mean"],
                marker="o",
                label=label,
            )
            axes[1, 0].plot(
                mode_frame["checkpoint_step"],
                mode_frame["event_intervention_rate_mean"],
                marker="o",
                label=label,
            )
            axes[1, 1].plot(
                mode_frame["checkpoint_step"],
                mode_frame["mean_correction_norm_mean"],
                marker="o",
                label=label,
            )
    axes[0, 0].set_title("Return")
    axes[0, 1].set_title("Ego Collisions Per km")
    axes[1, 0].set_title("Meaningful Intervention Rate")
    axes[1, 1].set_title("Mean Correction Norm")
    for axis in axes.ravel():
        axis.set_xlabel("Checkpoint step")
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=1)
    fig.suptitle("Raw Actor vs Actor+CBF vs Random+CBF Checkpoint Attribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    if not diagnostics.empty:
        diag_path = output_path.parent / "deltas.png"
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharex=True)
        for variant in variants:
            frame = diagnostics[diagnostics["variant"] == variant].sort_values("checkpoint_step")
            if frame.empty:
                continue
            axes[0].plot(frame["checkpoint_step"], frame["delta_R_filtered_minus_raw"], marker="o", label=variant)
            axes[1].plot(frame["checkpoint_step"], frame["delta_Cpkm_raw_minus_filtered"], marker="o", label=variant)
            axes[2].plot(frame["checkpoint_step"], frame["delta_R_actor_cbf_minus_random_cbf"], marker="o", label=variant)
        axes[0].set_title("Delta R: Filtered - Raw")
        axes[1].set_title("Delta Collisions/km: Raw - Filtered")
        axes[2].set_title("Delta R: Actor+CBF - Random+CBF")
        for axis in axes:
            axis.set_xlabel("Checkpoint step")
            axis.grid(True, alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(diag_path, dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whether CBF-filtered performance comes from the actor or filter.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/sp3_mtm_e0p1_p0p0_kpi_full"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=[10_000, 25_000, 50_000])
    parser.add_argument("--variants", nargs="+", default=["ddpg", "ddpg_cbf_reward", "guided_ddpg_cbf"])
    parser.add_argument("--modes", nargs="+", default=["raw_actor", "actor_cbf", "random_cbf"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=307_000)
    parser.add_argument("--k0", type=float, default=None)
    parser.add_argument("--k1", type=float, default=None)
    parser.add_argument("--eps-side", type=float, default=None)
    parser.add_argument("--event-threshold", type=float, default=None)
    parser.add_argument("--task-distance-m", type=float, default=None)
    parser.add_argument("--task-max-steps", type=int, default=None)
    parser.add_argument("--skip-tensorboard", action="store_true")
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("artifacts/tb_filter_contribution"))
    parser.add_argument("--rule-kv", type=float, default=0.35)
    parser.add_argument("--rule-ky", type=float, default=0.40)
    parser.add_argument("--rule-kvy", type=float, default=0.60)
    parser.add_argument("--force-mtm-congested", action="store_true", default=True)
    add_env_config_args(parser)
    return parser.parse_args()


def main() -> int:
    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()
    project_root = find_project_root(args.project_root or Path.cwd())
    source_dir = (project_root / args.source_dir).resolve() if not args.source_dir.is_absolute() else args.source_dir.resolve()
    output_dir = args.output_dir or (source_dir / "filter_policy_contribution")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = source_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing source run config: {run_config_path}")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))

    namespace: dict[str, Any] = {"__name__": "__main__"}
    exec_notebook_cells(project_root / "notebooks" / "lanelessKaralakou.ipynb", NOTEBOOK_DEPS, namespace)
    namespace["DEVICE"] = args.device
    namespace["CBF_K0"] = float(args.k0 if args.k0 is not None else run_config.get("k0", 5.29))
    namespace["CBF_K1"] = float(args.k1 if args.k1 is not None else run_config.get("k1", 3.68))
    namespace["CBF_EPS_SIDE"] = float(args.eps_side if args.eps_side is not None else run_config.get("eps_side", 0.10))
    namespace["CBF_FILTER_REWARD_LAMBDA"] = 0.0
    install_minimal_guided_cbf(namespace)
    install_safety_set_reward_wrapper(namespace)
    install_safe_reward_overtake_counter(namespace)
    install_event_penalty_env(namespace)

    env_config = dict(run_config.get("env_config", namespace["ENV_CONFIG"]))
    if args.force_mtm_congested and str(env_config.get("traffic_model", "")) == "mtm":
        deep_update(env_config, MTM_CONGESTED_UNCERTAIN_UPDATES.copy())
    reward_config = dict(run_config.get("reward_config") or make_reward_config(namespace, SAFETY_REWARD_TRIAL))
    reward_config["progress_reward_weight"] = float(run_config.get("progress_reward_weight", reward_config.get("progress_reward_weight", 0.0)))
    reward_config["safety_potential_eps_side"] = float(namespace["CBF_EPS_SIDE"])

    event_threshold = float(args.event_threshold if args.event_threshold is not None else run_config.get("event_threshold", 0.03))
    task_distance_m = float(args.task_distance_m if args.task_distance_m is not None else run_config.get("task_distance_m", 1000.0))
    task_max_steps = int(args.task_max_steps if args.task_max_steps is not None else run_config.get("task_max_steps", 1200))
    k0 = float(namespace["CBF_K0"])
    k1 = float(namespace["CBF_K1"])
    eps_side = float(namespace["CBF_EPS_SIDE"])

    valid_variants = {str(item["variant"]) for item in VARIANTS}
    variants = [variant for variant in args.variants if variant in valid_variants]
    invalid_variants = sorted(set(args.variants) - valid_variants)
    if invalid_variants:
        raise ValueError(f"Unknown variants: {invalid_variants}")
    modes = list(dict.fromkeys(args.modes))
    invalid_modes = sorted(set(modes) - set(MODE_LABELS))
    if invalid_modes:
        raise ValueError(f"Unknown modes: {invalid_modes}")

    print(
        "[filter-contribution] starting"
        f" source={source_dir}"
        f" output={output_dir}"
        f" variants={variants}"
        f" checkpoints={args.checkpoint_steps}"
        f" modes={modes}"
        f" episodes={args.episodes}"
        f" eps={eps_side:g}"
        f" threshold={event_threshold:g}"
        f" task={task_distance_m:g}m/{task_max_steps}steps",
        flush=True,
    )

    episode_frames: list[pd.DataFrame] = []
    for variant in variants:
        for checkpoint_step in args.checkpoint_steps:
            checkpoint_path = find_checkpoint(source_dir, variant, int(checkpoint_step))
            model = load_model_for_variant(namespace, variant, checkpoint_path, args.device)
            for mode in modes:
                mode_model = None if mode in {"random_cbf", "rule_cbf"} else model
                metrics = evaluate_one(
                    namespace,
                    model=mode_model,
                    variant=variant,
                    mode=mode,
                    checkpoint_step=int(checkpoint_step),
                    episodes=int(args.episodes),
                    seed=int(args.seed),
                    reward_config=reward_config,
                    env_config=env_config,
                    event_threshold=event_threshold,
                    k0=k0,
                    k1=k1,
                    eps_side=eps_side,
                    task_distance_m=task_distance_m,
                    task_max_steps=task_max_steps,
                    rule_kv=float(args.rule_kv),
                    rule_ky=float(args.rule_ky),
                    rule_kvy=float(args.rule_kvy),
                )
                episode_frames.append(metrics)
                summary = aggregate_episode_rows(metrics)
                if not summary.empty:
                    row = summary.iloc[0]
                    print(
                        "[filter-contribution]"
                        f" {variant}"
                        f" ckpt={checkpoint_step:,}"
                        f" mode={mode}"
                        f" return={row.get('return_mean', np.nan):.2f}"
                        f" C/km={row.get('ego_collisions_per_km_mean', np.nan):.2f}"
                        f" IR={row.get('event_intervention_rate_mean', np.nan):.2%}"
                        f" du={row.get('mean_correction_norm_mean', np.nan):.3f}"
                        f" h_min={row.get('h_min', np.nan):.3f}",
                        flush=True,
                    )

    episodes = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()
    summary = aggregate_episode_rows(episodes)
    diagnostics = compute_diagnostics(summary)
    episodes_path = output_dir / "episodes.csv"
    summary_path = output_dir / "summary.csv"
    diagnostics_path = output_dir / "diagnostics.csv"
    plot_path = output_dir / "summary.png"
    config_path = output_dir / "study_config.json"
    episodes.to_csv(episodes_path, index=False)
    summary.to_csv(summary_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    plot_summary(summary, diagnostics, plot_path)
    config_path.write_text(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "output_dir": str(output_dir),
                "variants": variants,
                "checkpoint_steps": [int(step) for step in args.checkpoint_steps],
                "modes": modes,
                "episodes": int(args.episodes),
                "seed": int(args.seed),
                "traffic_model": env_config.get("traffic_model"),
                "env_config": env_config,
                "reward_config": reward_config,
                "k0": k0,
                "k1": k1,
                "eps_side": eps_side,
                "event_threshold": event_threshold,
                "task_distance_m": task_distance_m,
                "task_max_steps": task_max_steps,
                "rule": {"kv": float(args.rule_kv), "ky": float(args.rule_ky), "kvy": float(args.rule_kvy)},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    if not args.skip_tensorboard:
        tb_root = (
            (project_root / args.tensorboard_dir).resolve()
            if not args.tensorboard_dir.is_absolute()
            else args.tensorboard_dir.resolve()
        )
        write_tensorboard(namespace, summary, diagnostics, tb_root)
        print(f"[filter-contribution] tensorboard {tb_root}", flush=True)
    print(f"[filter-contribution] wrote {episodes_path}", flush=True)
    print(f"[filter-contribution] wrote {summary_path}", flush=True)
    print(f"[filter-contribution] wrote {diagnostics_path}", flush=True)
    print(f"[filter-contribution] wrote {plot_path}", flush=True)

    display_cols = [
        "variant",
        "checkpoint_step",
        "mode",
        "return_mean",
        "ego_collisions_per_km_mean",
        "event_intervention_rate_mean",
        "mean_correction_norm_mean",
        "p95_correction_norm_mean",
        "mean_delta_ax_abs_mean",
        "mean_delta_ay_abs_mean",
        "h_min",
        "qp_failure_rate_mean",
    ]
    display_cols = [column for column in display_cols if column in summary.columns]
    if display_cols:
        print(summary[display_cols].to_string(index=False), flush=True)
    if not diagnostics.empty:
        print(diagnostics.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
