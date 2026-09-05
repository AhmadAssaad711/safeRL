from __future__ import annotations

import argparse
import faulthandler
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from scripts.training.cbf_lambda_event_bc_pilot_sweep import (
    NOTEBOOK_DEPS,
    exec_notebook_cells,
    find_project_root,
    install_event_penalty_env,
    plot_aggregate,
    plot_trial_history,
    score_summary,
    set_stable_native_defaults,
    summarize,
)
from scripts.common.guided_cbf_minimal import install_minimal_guided_cbf
from scripts.common.laneless_script_config import active_traffic_model, add_env_config_args, env_config_from_args


warnings.filterwarnings("ignore", message="OSQP exited.*")

DEFAULT_VARIANTS: list[dict[str, float | str | bool]] = [
    {
        "variant": "baseline_best",
        "use_safety_obs": False,
        "lambda_norm": 0.025,
        "lambda_event": 0.02,
        "lambda_bc": 0.03,
    },
    {
        "variant": "safety_obs_best",
        "use_safety_obs": True,
        "lambda_norm": 0.025,
        "lambda_event": 0.02,
        "lambda_bc": 0.03,
    },
    {
        "variant": "baseline_no_bc",
        "use_safety_obs": False,
        "lambda_norm": 0.025,
        "lambda_event": 0.02,
        "lambda_bc": 0.00,
    },
    {
        "variant": "safety_obs_no_bc",
        "use_safety_obs": True,
        "lambda_norm": 0.025,
        "lambda_event": 0.02,
        "lambda_bc": 0.00,
    },
]

SAFETY_FEATURE_NAMES = [
    "min_pair_h_norm",
    "min_boundary_h_norm",
    "left_boundary_h_norm",
    "right_boundary_h_norm",
    "closest_pair_h_norm",
    "closest_distance_margin_norm",
    "closest_required_distance_norm",
    "closest_actual_distance_norm",
    "closest_relative_x_norm",
    "closest_relative_y_norm",
    "closest_relative_vx_norm",
    "closest_relative_vy_norm",
    "num_cbf_neighbors_norm",
]


def install_safety_observation_env(namespace: dict[str, Any]) -> None:
    if "make_event_cbf_single_env" not in namespace:
        install_event_penalty_env(namespace)

    class SafetySetObservationWrapper(gym.ObservationWrapper):
        """Append state-only CBF safety-set features to the policy observation."""

        def __init__(
            self,
            env: gym.Env,
            *,
            eps_side: float,
            neighbor_range: float,
            max_neighbor_constraints: int | None,
            clip: float = 5.0,
        ) -> None:
            super().__init__(env)
            self.eps_side = float(eps_side)
            self.neighbor_range = float(neighbor_range)
            self.max_neighbor_constraints = max_neighbor_constraints
            self.clip = float(clip)
            base_shape = self.env.observation_space.shape
            if base_shape is None or len(base_shape) != 1:
                raise ValueError(f"Expected flat observation space, got {self.env.observation_space}")
            feature_count = len(SAFETY_FEATURE_NAMES)
            self.observation_space = gym.spaces.Box(
                low=-self.clip,
                high=self.clip,
                shape=(int(base_shape[0]) + feature_count,),
                dtype=np.float32,
            )

        def safety_features(self) -> np.ndarray:
            base = self.env.unwrapped
            config = base.config
            road_width = max(float(config["road_width"]), 1e-6)
            sensing_range = max(float(config["sensing_range"]), 1e-6)
            obs_vmax = max(float(config.get("observation_vmax", namespace.get("OBS_VMAX", 24.0))), 1e-6)
            obs_vymax = max(float(config.get("observation_vymax", namespace.get("OBS_VYMAX", 7.2))), 1e-6)
            ego = namespace["get_ego_state"](self.env)
            neighbors = namespace["get_neighbor_states"](self.env, neighbor_range=self.neighbor_range)
            if self.max_neighbor_constraints is not None:
                neighbors = list(neighbors)[: int(self.max_neighbor_constraints)]

            ego_y = float(ego["y"])
            ego_half_width = 0.5 * float(ego["width"])
            left_boundary_h = ego_y - ego_half_width
            right_boundary_h = road_width - ego_half_width - ego_y
            min_boundary_h = min(left_boundary_h, right_boundary_h)

            min_pair_h = sensing_range
            closest_h = sensing_range
            closest_dx = 0.0
            closest_dy = 0.0
            closest_dvx = 0.0
            closest_dvy = 0.0
            closest_actual_distance = sensing_range
            closest_required_distance = 0.0

            if neighbors:
                scored: list[tuple[float, float, float, float, float, float, float]] = []
                for neighbor in neighbors:
                    h, dx, dy, dvx, dvy, actual_distance, required_distance = namespace["pairwise_cbf_geometry"](
                        ego,
                        neighbor,
                        eps_side=self.eps_side,
                    )
                    scored.append(
                        (
                            float(h),
                            float(dx),
                            float(dy),
                            float(dvx),
                            float(dvy),
                            float(actual_distance),
                            float(required_distance),
                        )
                    )
                closest_h, closest_dx, closest_dy, closest_dvx, closest_dvy, closest_actual_distance, closest_required_distance = min(
                    scored,
                    key=lambda item: item[0],
                )
                min_pair_h = closest_h

            max_constraints = float(self.max_neighbor_constraints or max(len(neighbors), 1))
            values = np.asarray(
                [
                    min_pair_h / sensing_range,
                    min_boundary_h / road_width,
                    left_boundary_h / road_width,
                    right_boundary_h / road_width,
                    closest_h / sensing_range,
                    (closest_actual_distance - closest_required_distance) / sensing_range,
                    closest_required_distance / sensing_range,
                    closest_actual_distance / sensing_range,
                    closest_dx / sensing_range,
                    closest_dy / road_width,
                    closest_dvx / obs_vmax,
                    closest_dvy / obs_vymax,
                    len(neighbors) / max(max_constraints, 1.0),
                ],
                dtype=np.float32,
            )
            return np.clip(values, -self.clip, self.clip).astype(np.float32)

        def observation(self, observation: np.ndarray) -> np.ndarray:
            obs = np.asarray(observation, dtype=np.float32).reshape(-1)
            return np.concatenate([obs, self.safety_features()], dtype=np.float32)

    def make_safety_obs_event_cbf_single_env(
        *,
        seed: int,
        lambda_norm: float,
        lambda_event: float,
        event_threshold: float,
        k0: float,
        k1: float,
        eps_side: float,
        use_safety_obs: bool,
        render_mode: str | None = None,
        env_config: dict[str, Any] | None = None,
        reward_config: dict[str, float] | None = None,
        normalize_observation: bool | None = None,
    ) -> gym.Env:
        env = gym.make(
            "lane-free-v0",
            render_mode=render_mode,
            config=env_config or namespace["ENV_CONFIG"],
        )
        env = namespace["KaralakouRewardWrapper"](env, reward_config=reward_config or namespace["REWARD_CONFIG"])
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
        if use_safety_obs:
            env = SafetySetObservationWrapper(
                env,
                eps_side=eps_side,
                neighbor_range=namespace["CBF_NEIGHBOR_RANGE"],
                max_neighbor_constraints=namespace.get("CBF_MAX_NEIGHBOR_CONSTRAINTS"),
                clip=5.0,
            )
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    def make_safety_obs_event_cbf_training_env(
        *,
        seed: int,
        lambda_norm: float,
        lambda_event: float,
        event_threshold: float,
        k0: float,
        k1: float,
        eps_side: float,
        use_safety_obs: bool,
        n_envs: int,
        use_subproc: bool = bool(namespace.get("DDPG_USE_SUBPROC_VEC_ENV", True)),
        env_config: dict[str, Any] | None = None,
        reward_config: dict[str, float] | None = None,
        normalize_observation: bool | None = None,
    ):
        def _single_env(env_seed: int) -> gym.Env:
            return make_safety_obs_event_cbf_single_env(
                seed=env_seed,
                lambda_norm=lambda_norm,
                lambda_event=lambda_event,
                event_threshold=event_threshold,
                k0=k0,
                k1=k1,
                eps_side=eps_side,
                use_safety_obs=use_safety_obs,
                env_config=env_config,
                reward_config=reward_config,
                normalize_observation=normalize_observation,
            )

        return namespace["_make_vectorized_env"](
            _single_env,
            seed=seed,
            n_envs=n_envs,
            use_subproc=use_subproc,
            start_method=namespace["DDPG_SUBPROC_START_METHOD"],
        )

    namespace.update(
        {
            "SafetySetObservationWrapper": SafetySetObservationWrapper,
            "make_safety_obs_event_cbf_single_env": make_safety_obs_event_cbf_single_env,
            "make_safety_obs_event_cbf_training_env": make_safety_obs_event_cbf_training_env,
            "SAFETY_FEATURE_NAMES": SAFETY_FEATURE_NAMES,
        }
    )


def evaluate_variant(
    namespace: dict[str, Any],
    model: Any,
    *,
    variant: str,
    use_safety_obs: bool,
    episodes: int,
    seed: int,
    k0: float,
    k1: float,
    eps_side: float,
    event_threshold: float,
    env_config: dict[str, Any] | None = None,
    use_distance_task: bool = True,
    task_distance_m: float = 1000.0,
    task_max_steps: int = 1200,
    collect_steps: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict[str, float]] = []
    step_rows: list[dict[str, float | str]] = []
    for episode in range(episodes):
        env = namespace["make_safety_obs_event_cbf_single_env"](
            seed=seed + episode,
            lambda_norm=0.0,
            lambda_event=0.0,
            event_threshold=event_threshold,
            k0=k0,
            k1=k1,
            eps_side=eps_side,
            use_safety_obs=use_safety_obs,
            env_config=env_config,
        )
        if use_distance_task and "make_task_evaluation_wrapper" in namespace:
            env = namespace["make_task_evaluation_wrapper"](
                env,
                task_distance_m=float(task_distance_m),
                max_steps=int(task_max_steps),
            )
        else:
            namespace["configure_paper_evaluation_env"](env, steps=namespace["PAPER_EVAL_STEPS"])
        obs, _ = env.reset(seed=seed + episode)
        done = False
        step_count = 0
        rewards: list[float] = []
        speeds: list[float] = []
        signed_speed_errors: list[float] = []
        abs_speed_errors: list[float] = []
        lat_y_errors: list[float] = []
        corrections: list[float] = []
        event_interventions: list[float] = []
        numerical_interventions: list[float] = []
        qp_successes: list[float] = []
        fallbacks: list[float] = []
        min_h_values: list[float] = []
        min_boundary_h_values: list[float] = []
        ego_collisions = 0
        ego_collision_steps = 0
        all_collision_events = 0
        last_task_info: dict[str, Any] = {}
        kpi_info_rows: list[dict[str, Any]] = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            base = env.unwrapped
            speed = float(base.vehicle.vx)
            desired_speed = float(base.vehicle.desired_speed)
            speed_error = speed - desired_speed
            lat_y_error = float(info.get("karalakou_lat_y_error_m", np.nan))
            correction = float(info.get("cbf_correction_norm", 0.0))
            event_intervention = float(info.get("cbf_event_intervened", correction > event_threshold))
            numerical_intervention = float(info.get("cbf_intervened", correction > 1e-6))
            qp_success = float(info.get("cbf_qp_success", True))
            fallback = float(info.get("cbf_fallback_used", False))
            min_h = float(info.get("cbf_min_h", np.nan))
            min_boundary_h = float(info.get("cbf_min_boundary_h", np.nan))

            rewards.append(float(reward))
            speeds.append(speed)
            signed_speed_errors.append(speed_error)
            abs_speed_errors.append(abs(speed_error))
            if np.isfinite(lat_y_error):
                lat_y_errors.append(lat_y_error)
            corrections.append(correction)
            event_interventions.append(event_intervention)
            numerical_interventions.append(numerical_intervention)
            qp_successes.append(qp_success)
            fallbacks.append(fallback)
            min_h_values.append(min_h)
            min_boundary_h_values.append(min_boundary_h)
            kpi_info_rows.append(dict(info))
            last_task_info = {
                "task_distance_m": float(info.get("task_distance_m", task_distance_m)),
                "task_distance_traveled_m": float(info.get("task_distance_traveled_m", 0.0)),
                "task_progress_ratio": float(info.get("task_progress_ratio", 0.0)),
                "task_completed": bool(info.get("task_completed", False)),
                "task_timeout": bool(info.get("task_timeout", False)),
            }
            all_collision_events += int(info.get("collisions", 0))
            ego_collisions += int(info.get("ego_collision_events", 0))
            if bool(info.get("ego_collision", False)):
                ego_collision_steps += 1
            if collect_steps:
                step_rows.append(
                    {
                        "variant": variant,
                        "episode": float(episode),
                        "step": float(step_count),
                        "reward": float(reward),
                        "correction_norm": correction,
                        "event_intervention": event_intervention,
                        "numerical_intervention": numerical_intervention,
                        "qp_success": qp_success,
                        "fallback_used": fallback,
                        "min_h": min_h,
                        "min_boundary_h": min_boundary_h,
                        "mean_abs_speed_error": abs(speed_error),
                        "lat_y_error_m": lat_y_error,
                        "ego_collision": float(bool(info.get("ego_collision", False))),
                        "task_distance_traveled_m": float(last_task_info.get("task_distance_traveled_m", 0.0)),
                        "task_progress_ratio": float(last_task_info.get("task_progress_ratio", 0.0)),
                    }
                )
            step_count += 1
            done = bool(terminated or truncated)

        episode_rows.append(
            {
                "episode": float(episode),
                "steps": float(step_count),
                "episode_length_steps": float(step_count),
                "return": float(np.sum(rewards)),
                "episode_return": float(np.sum(rewards)),
                "task_completed": float(last_task_info.get("task_completed", False)),
                "task_timeout": float(last_task_info.get("task_timeout", False)),
                "task_distance_m": float(last_task_info.get("task_distance_m", task_distance_m)),
                "task_distance_traveled_m": float(last_task_info.get("task_distance_traveled_m", 0.0)),
                "task_progress_ratio": float(last_task_info.get("task_progress_ratio", 0.0)),
                "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
                "mean_signed_speed_error": float(np.mean(signed_speed_errors)) if signed_speed_errors else 0.0,
                "mean_abs_speed_error": float(np.mean(abs_speed_errors)) if abs_speed_errors else 0.0,
                "mean_lat_y_error_m": float(np.mean(lat_y_errors)) if lat_y_errors else np.nan,
                "mean_correction_norm": float(np.mean(corrections)) if corrections else 0.0,
                "max_correction_norm": float(np.max(corrections)) if corrections else 0.0,
                "event_intervention_rate": float(np.mean(event_interventions)) if event_interventions else 0.0,
                "numerical_intervention_rate": float(np.mean(numerical_interventions)) if numerical_interventions else 0.0,
                "qp_failure_rate": float(1.0 - np.mean(qp_successes)) if qp_successes else 0.0,
                "fallback_rate": float(np.mean(fallbacks)) if fallbacks else 0.0,
                "min_h": float(np.nanmin(min_h_values)) if min_h_values and not np.all(np.isnan(min_h_values)) else np.nan,
                "min_boundary_h": float(np.nanmin(min_boundary_h_values))
                if min_boundary_h_values and not np.all(np.isnan(min_boundary_h_values))
                else np.nan,
                "ego_collisions": float(ego_collisions),
                "ego_collision_steps": float(ego_collision_steps),
                "total_collision_events": float(all_collision_events),
            }
        )
        summarize_episode_kpis = namespace.get("summarize_episode_kpis")
        if callable(summarize_episode_kpis):
            episode_rows[-1].update(
                summarize_episode_kpis(
                    kpi_info_rows,
                    rewards=rewards,
                    task_completed=bool(last_task_info.get("task_completed", False)),
                    fallback_steps=step_count,
                    fallback_distance_m=float(last_task_info.get("task_distance_traveled_m", 0.0)),
                    fallback_dt_s=namespace["kpi_policy_dt"](env) if "kpi_policy_dt" in namespace else np.nan,
                )
            )
        env.close()
    return pd.DataFrame(episode_rows), pd.DataFrame(step_rows)


class SafetyObsEvalCallback(BaseCallback):
    def __init__(
        self,
        namespace: dict[str, Any],
        *,
        variant: str,
        use_safety_obs: bool,
        lambda_norm: float,
        lambda_event: float,
        lambda_bc: float,
        event_threshold: float,
        k0: float,
        k1: float,
        eps_side: float,
        env_config: dict[str, Any] | None,
        task_distance_m: float,
        task_max_steps: int,
        eval_freq: int,
        episodes: int,
        seed: int,
    ) -> None:
        super().__init__(verbose=1)
        self.namespace = namespace
        self.variant = variant
        self.use_safety_obs = bool(use_safety_obs)
        self.lambda_norm = float(lambda_norm)
        self.lambda_event = float(lambda_event)
        self.lambda_bc = float(lambda_bc)
        self.event_threshold = float(event_threshold)
        self.k0 = float(k0)
        self.k1 = float(k1)
        self.eps_side = float(eps_side)
        self.env_config = env_config
        self.task_distance_m = float(task_distance_m)
        self.task_max_steps = int(task_max_steps)
        self.eval_freq = int(eval_freq)
        self.episodes = int(episodes)
        self.seed = int(seed)
        self.records: list[dict[str, float | str | bool]] = []
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        metrics, _ = evaluate_variant(
            self.namespace,
            self.model,
            variant=self.variant,
            use_safety_obs=self.use_safety_obs,
            episodes=self.episodes,
            seed=self.seed + self.num_timesteps,
            k0=self.k0,
            k1=self.k1,
            eps_side=self.eps_side,
            event_threshold=self.event_threshold,
            env_config=self.env_config,
            use_distance_task=True,
            task_distance_m=self.task_distance_m,
            task_max_steps=self.task_max_steps,
            collect_steps=False,
        )
        row: dict[str, float | str | bool] = {
            "variant": self.variant,
            "use_safety_obs": self.use_safety_obs,
            "lambda_norm": self.lambda_norm,
            "lambda_event": self.lambda_event,
            "lambda_bc": self.lambda_bc,
            "event_threshold": self.event_threshold,
            "timesteps": float(self.num_timesteps),
            **summarize(metrics),
        }
        self.records.append(row)
        print(
            "[safety-obs-eval]"
            f" {self.variant}"
            f" steps={self.num_timesteps:,}"
            f" return={row['return_mean']:.2f}"
            f" abs_speed={row['mean_abs_speed_error']:.3f}"
            f" event_int={row['event_intervention_rate']:.2%}"
            f" corr={row['mean_correction_norm']:.3f}"
            f" qp_fail={row['qp_failure_rate']:.2%}",
            flush=True,
        )
        return True


def plot_interpretability(step_data: pd.DataFrame, output_path: Path) -> None:
    if step_data.empty:
        return
    frame = step_data.copy()
    frame["min_h_clipped"] = pd.to_numeric(frame["min_h"], errors="coerce").clip(-2.0, 8.0)
    frame["min_boundary_h_clipped"] = pd.to_numeric(frame["min_boundary_h"], errors="coerce").clip(0.0, 5.0)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["min_h_clipped", "correction_norm"])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for variant, group in frame.groupby("variant"):
        group = group.copy()
        group["h_bin"] = pd.cut(group["min_h_clipped"], bins=np.linspace(-2.0, 8.0, 21), include_lowest=True)
        binned = (
            group.groupby("h_bin", observed=True)
            .agg(
                min_h=("min_h_clipped", "mean"),
                event_intervention=("event_intervention", "mean"),
                correction_norm=("correction_norm", "mean"),
                count=("correction_norm", "count"),
            )
            .reset_index(drop=True)
        )
        binned = binned[binned["count"] >= 10]
        axes[0].plot(binned["min_h"], binned["event_intervention"], marker="o", label=variant)
        axes[1].plot(binned["min_h"], binned["correction_norm"], marker="o", label=variant)

    axes[0].set_title("Intervention vs Pair Safety Margin")
    axes[0].set_xlabel("min pair h")
    axes[0].set_ylabel("Meaningful intervention rate")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    axes[1].set_title("Correction Norm vs Pair Safety Margin")
    axes[1].set_xlabel("min pair h")
    axes[1].set_ylabel("Mean correction norm")
    for axis in axes:
        axis.grid(True, alpha=0.28)
        axis.legend(fontsize=8)
    fig.suptitle("Safety Interpretability From Final Evaluation Steps", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B test state-only CBF safety-set observation features.")
    parser.add_argument("--k0", type=float, default=5.29)
    parser.add_argument("--k1", type=float, default=3.68)
    parser.add_argument("--eps-side", type=float, default=0.149)
    parser.add_argument("--event-threshold", type=float, default=0.03)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--train-eval-freq", type=int, default=5_000)
    parser.add_argument("--train-eval-episodes", type=int, default=2)
    parser.add_argument("--final-eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=711_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--task-distance-m", type=float, default=1000.0)
    parser.add_argument("--task-max-steps", type=int, default=1200)
    add_env_config_args(parser)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    args = parse_args()

    project_root = find_project_root(args.project_root or Path.cwd())
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    namespace: dict[str, Any] = {"__name__": "__main__"}
    exec_notebook_cells(notebook_path, NOTEBOOK_DEPS, namespace)
    namespace["DEVICE"] = args.device
    install_minimal_guided_cbf(namespace)
    install_event_penalty_env(namespace)
    install_safety_observation_env(namespace)
    env_config = env_config_from_args(args, namespace["ENV_CONFIG"])
    traffic_model = active_traffic_model(env_config)

    default_output_name = "cbf_safety_obs_experiment_mtm" if traffic_model == "mtm" else "cbf_safety_obs_experiment"
    output_dir = args.output_dir or (Path(namespace["ARTIFACT_DIR"]) / default_output_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(DEFAULT_VARIANTS).to_csv(output_dir / "variant_configs.csv", index=False)
    (output_dir / "safety_feature_names.json").write_text(
        json.dumps(SAFETY_FEATURE_NAMES, indent=2),
        encoding="utf-8",
    )

    print(
        "[safety-obs] starting experiment",
        {
            "variants": len(DEFAULT_VARIANTS),
            "timesteps": args.timesteps,
            "final_eval_episodes": args.final_eval_episodes,
            "event_threshold": args.event_threshold,
            "task_distance_m": args.task_distance_m,
            "task_max_steps": args.task_max_steps,
            "traffic_model": traffic_model,
            "k0": args.k0,
            "k1": args.k1,
            "eps_side": args.eps_side,
            "output_dir": str(output_dir),
        },
        flush=True,
    )

    final_rows: list[dict[str, float | str | bool]] = []
    final_step_frames: list[pd.DataFrame] = []
    for index, config in enumerate(DEFAULT_VARIANTS, start=1):
        variant = str(config["variant"])
        use_safety_obs = bool(config["use_safety_obs"])
        lambda_norm = float(config["lambda_norm"])
        lambda_event = float(config["lambda_event"])
        lambda_bc = float(config["lambda_bc"])
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        model_path = variant_dir / "model.zip"
        checkpoint_path = variant_dir / "checkpoint_model.zip"
        history_path = variant_dir / "train_eval_history.csv"
        final_episodes_path = variant_dir / "final_eval_episodes.csv"
        final_steps_path = variant_dir / "final_eval_steps.csv"
        final_summary_path = variant_dir / "final_summary.csv"
        plot_path = variant_dir / "train_eval_history.png"

        if not args.no_resume and final_summary_path.exists() and model_path.exists():
            print(f"[safety-obs] [{index}/{len(DEFAULT_VARIANTS)}] {variant} complete; loading summary", flush=True)
            final_rows.append(pd.read_csv(final_summary_path).iloc[0].to_dict())
            if final_steps_path.exists():
                final_step_frames.append(pd.read_csv(final_steps_path))
            continue

        print(
            f"[safety-obs] [{index}/{len(DEFAULT_VARIANTS)}] {variant}"
            f" safety_obs={use_safety_obs}"
            f" lambda_norm={lambda_norm:g}"
            f" lambda_event={lambda_event:g}"
            f" lambda_bc={lambda_bc:g}",
            flush=True,
        )
        train_env = namespace["make_safety_obs_event_cbf_training_env"](
            seed=args.seed + index * 1_000,
            lambda_norm=lambda_norm,
            lambda_event=lambda_event,
            event_threshold=args.event_threshold,
            k0=args.k0,
            k1=args.k1,
            eps_side=args.eps_side,
            use_safety_obs=use_safety_obs,
            n_envs=args.n_envs,
            use_subproc=bool(namespace.get("DDPG_USE_SUBPROC_VEC_ENV", True)),
            env_config=env_config,
        )
        print(
            f"[safety-obs] vec_env={type(train_env).__name__} | n_envs={args.n_envs} | UTD=1",
            flush=True,
        )
        n_actions = train_env.action_space.shape[-1]
        action_noise = namespace["make_ou_action_noise"](n_actions, n_envs=args.n_envs)
        history_records: list[dict[str, float | str | bool]] = []
        completed_steps = 0
        if not args.no_resume and checkpoint_path.exists() and history_path.exists():
            history = pd.read_csv(history_path)
            if not history.empty and "timesteps" in history.columns:
                completed_steps = int(float(history["timesteps"].max()))
                history_records = history.to_dict("records")
            print(
                f"[safety-obs] resuming {variant} from {completed_steps:,} steps",
                flush=True,
            )
            model = namespace["GuidedCBFDDPG"].load(str(checkpoint_path), env=train_env, device=args.device)
            model.action_noise = action_noise
        else:
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
            while completed_steps < args.timesteps:
                next_steps = min(args.timesteps, completed_steps + args.train_eval_freq)
                chunk_steps = int(next_steps - completed_steps)
                print(
                    f"[safety-obs] {variant} training chunk {completed_steps:,}->{next_steps:,}",
                    flush=True,
                )
                model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False, progress_bar=False)
                completed_steps = int(next_steps)
                metrics, _ = evaluate_variant(
                    namespace,
                    model,
                    variant=variant,
                    use_safety_obs=use_safety_obs,
                    episodes=args.train_eval_episodes,
                    seed=args.seed + index * 10_000 + completed_steps,
                    k0=args.k0,
                    k1=args.k1,
                    eps_side=args.eps_side,
                    event_threshold=args.event_threshold,
                    env_config=env_config,
                    use_distance_task=True,
                    task_distance_m=args.task_distance_m,
                    task_max_steps=args.task_max_steps,
                    collect_steps=False,
                )
                row: dict[str, float | str | bool] = {
                    "variant": variant,
                    "use_safety_obs": use_safety_obs,
                    "lambda_norm": lambda_norm,
                    "lambda_event": lambda_event,
                    "lambda_bc": lambda_bc,
                    "event_threshold": float(args.event_threshold),
                    "timesteps": float(completed_steps),
                    **summarize(metrics),
                }
                history_records.append(row)
                pd.DataFrame(history_records).to_csv(history_path, index=False)
                model.save(str(checkpoint_path))
                print(
                    "[safety-obs-eval]"
                    f" {variant}"
                    f" steps={completed_steps:,}"
                    f" return={row['return_mean']:.2f}"
                    f" abs_speed={row['mean_abs_speed_error']:.3f}"
                    f" event_int={row['event_intervention_rate']:.2%}"
                    f" corr={row['mean_correction_norm']:.3f}"
                    f" qp_fail={row['qp_failure_rate']:.2%}",
                    flush=True,
                )
        finally:
            train_env.close()
        elapsed_sec = time.time() - start
        model.save(str(model_path))

        history = pd.DataFrame(history_records)
        history.to_csv(history_path, index=False)
        plot_trial_history(history.rename(columns={"variant": "trial_name"}), plot_path, title=variant)

        print(f"[safety-obs] evaluating {variant}", flush=True)
        final_metrics, final_steps = evaluate_variant(
            namespace,
            model,
            variant=variant,
            use_safety_obs=use_safety_obs,
            episodes=args.final_eval_episodes,
            seed=args.seed + index * 100_000,
            k0=args.k0,
            k1=args.k1,
            eps_side=args.eps_side,
            event_threshold=args.event_threshold,
            env_config=env_config,
            use_distance_task=True,
            task_distance_m=args.task_distance_m,
            task_max_steps=args.task_max_steps,
            collect_steps=True,
        )
        final_metrics.to_csv(final_episodes_path, index=False)
        final_steps.to_csv(final_steps_path, index=False)
        final_step_frames.append(final_steps)
        summary: dict[str, float | str | bool] = {
            "variant": variant,
            "model_path": str(model_path),
            "elapsed_sec": float(elapsed_sec),
            "timesteps": float(args.timesteps),
            "use_safety_obs": use_safety_obs,
            "k0": float(args.k0),
            "k1": float(args.k1),
            "eps_side": float(args.eps_side),
            "event_threshold": float(args.event_threshold),
            "traffic_model": traffic_model,
            "task_distance_m": float(args.task_distance_m),
            "task_max_steps": int(args.task_max_steps),
            "lambda_norm": lambda_norm,
            "lambda_event": lambda_event,
            "lambda_bc": lambda_bc,
            **summarize(final_metrics),
        }
        summary["selection_score"] = score_summary(summary)
        pd.DataFrame([summary]).to_csv(final_summary_path, index=False)
        (variant_dir / "run_config.json").write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        final_rows.append(summary)
        print(
            "[safety-obs-result]"
            f" {variant}"
            f" return={summary['return_mean']:.2f}"
            f" abs_speed={summary['mean_abs_speed_error']:.3f}"
            f" event_int={summary['event_intervention_rate']:.2%}"
            f" num_int={summary['numerical_intervention_rate']:.2%}"
            f" corr={summary['mean_correction_norm']:.3f}"
            f" qp_fail={summary['qp_failure_rate']:.2%}"
            f" ego_col={summary['ego_collisions_mean']:.2f}",
            flush=True,
        )

    aggregate = pd.DataFrame(final_rows)
    if not aggregate.empty:
        aggregate = aggregate.sort_values("selection_score", ascending=False).reset_index(drop=True)
    aggregate_path = output_dir / "safety_obs_final_summary.csv"
    aggregate_plot_path = output_dir / "safety_obs_comparison.png"
    aggregate.to_csv(aggregate_path, index=False)
    plot_aggregate(aggregate.rename(columns={"variant": "trial_name"}), aggregate_plot_path)

    if final_step_frames:
        all_steps = pd.concat(final_step_frames, ignore_index=True)
        all_steps_path = output_dir / "safety_obs_final_steps.csv"
        interpretability_path = output_dir / "safety_obs_interpretability.png"
        all_steps.to_csv(all_steps_path, index=False)
        plot_interpretability(all_steps, interpretability_path)
        print(f"[safety-obs] wrote {all_steps_path}", flush=True)
        print(f"[safety-obs] wrote {interpretability_path}", flush=True)

    print(f"[safety-obs] wrote {aggregate_path}", flush=True)
    print(f"[safety-obs] wrote {aggregate_plot_path}", flush=True)
    if not aggregate.empty:
        display_cols = [
            "variant",
            "use_safety_obs",
            "lambda_bc",
            "return_mean",
            "mean_abs_speed_error",
            "event_intervention_rate",
            "mean_correction_norm",
            "qp_failure_rate",
            "ego_collisions_mean",
            "selection_score",
        ]
        print(aggregate[display_cols].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
