from __future__ import annotations

import argparse
import faulthandler
import hashlib
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
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from scripts.training.cbf_lambda_event_bc_pilot_sweep import (
    exec_notebook_cells,
    find_project_root,
    install_event_penalty_env,
    set_stable_native_defaults,
)
from scripts.training.cbf_reward_term_ablation import (
    NOTEBOOK_DEPS,
    behavior_score,
    install_safety_set_reward_wrapper,
    make_reward_config,
    summarize,
)
from scripts.common.guided_cbf_minimal import install_minimal_guided_cbf
from scripts.common.laneless_script_config import active_traffic_model, add_env_config_args, env_config_from_args
from scripts.rendering.render_laneless_policy_videos import (
    scenario_file_name,
    selected_scenarios,
    run_clip,
    variant_file_name,
)
from scripts.rendering.render_policy_scenarios import make_scenarios


warnings.filterwarnings("ignore", message="OSQP exited.*")

VARIANTS = [
    {
        "variant": "ddpg",
        "video_variant": "ddpg",
        "label": "DDPG (CBF safety reward)",
        "env_kind": "baseline",
        "model_class": DDPG,
        "lambda_bc": 0.0,
    },
    {
        "variant": "ddpg_cbf_reward",
        "video_variant": "ddpg-cbf",
        "label": "DDPG-CBF reward",
        "env_kind": "event_cbf",
        "model_class": DDPG,
        "lambda_bc": 0.0,
    },
    {
        "variant": "guided_ddpg_cbf",
        "video_variant": "guided-ddpg-cbf",
        "label": "DDPG-CBF reward + actor loss",
        "env_kind": "event_cbf",
        "model_class_name": "GuidedCBFDDPG",
        "lambda_bc": 0.03,
        "critic_action_mode": "raw",
        "actor_action_mode": "raw",
    },
    {
        "variant": "update_a_raw_q_raw_bc",
        "video_variant": "guided-ddpg-cbf",
        "label": "A: Q(s, raw) + actor raw + BC",
        "env_kind": "event_cbf",
        "model_class_name": "GuidedCBFDDPG",
        "lambda_bc": 0.03,
        "critic_action_mode": "raw",
        "actor_action_mode": "raw",
    },
    {
        "variant": "update_b_safe_critic_raw_actor",
        "video_variant": "guided-ddpg-cbf",
        "label": "B: Q(s, safe replay) + actor raw + BC",
        "env_kind": "event_cbf",
        "model_class_name": "GuidedCBFDDPG",
        "lambda_bc": 0.03,
        "critic_action_mode": "safe",
        "actor_action_mode": "raw",
    },
    {
        "variant": "update_c_safe_critic_diff_actor",
        "video_variant": "guided-ddpg-cbf",
        "label": "C: Q(s, safe replay) + actor diff-CBF",
        "env_kind": "event_cbf",
        "model_class_name": "GuidedCBFDDPG",
        "lambda_bc": 0.03,
        "critic_action_mode": "safe",
        "actor_action_mode": "diff_cbf",
    },
]

INITIAL_VARIANT_NAMES = ("ddpg", "ddpg_cbf_reward", "guided_ddpg_cbf")

TB_VARIANT_RUN_NAMES = {
    "ddpg": "ddpg",
    "ddpg_cbf_reward": "cbfr",
    "guided_ddpg_cbf": "guided",
    "update_a_raw_q_raw_bc": "updA_rawQ",
    "update_b_safe_critic_raw_actor": "updB_safeQ",
    "update_c_safe_critic_diff_actor": "updC_diff",
}

SAFETY_REWARD_TRIAL = {
    "trial_name": "safety_potential_bc003",
    "use_current_potential": False,
    "use_safety_potential": True,
    "wy": 0.65,
    "wf": 0.0,
    "w_safe": 0.80,
    "lambda_bc": 0.03,
}

# Historical reward formulation retained for controlled ablations.  This is
# the original Karalakou potential term (``cf``), not the CBF-clearance
# potential introduced by SAFETY_REWARD_TRIAL.
OLD_REWARD_TRIAL = {
    "trial_name": "current_bc003",
    "use_current_potential": True,
    "use_safety_potential": False,
    "wy": 0.65,
    "wf": 1.0,
    "w_safe": 0.0,
    "lambda_bc": 0.03,
}

REWARD_TRIALS = {
    "safety": SAFETY_REWARD_TRIAL,
    "old": OLD_REWARD_TRIAL,
}

MTM_CONGESTED_UNCERTAIN_UPDATES = {
    "traffic_model": "mtm",
    "ego_controlled": True,
    "road_length": 380.0,
    "vehicles_count": 55,
    "sensing_range": 90.0,
    "desired_speed_range": [15.0, 25.0],
    "initial_speed_fraction_range": [0.55, 1.10],
    "episode_steps": 800,
    "duration": 800,
    "terminate_on_collision": True,
    "show_trajectories": False,
    "mtm": {
        "leader_range": 90.0,
        "continuous_driver_aggressiveness": True,
        "profile_probabilities": {"normal": 0.25, "aggressive": 0.50, "cautious": 0.25},
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def make_baseline_single_env(
    namespace: dict[str, Any],
    *,
    seed: int,
    reward_config: dict[str, float],
    env_config: dict[str, Any],
    render_mode: str | None = None,
    normalize_observation: bool | None = None,
) -> gym.Env:
    env = gym.make("lane-free-v0", render_mode=render_mode, config=env_config)
    env = namespace["KaralakouRewardWrapper"](env, reward_config=reward_config)
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
    env_kind: str,
    seed: int,
    reward_config: dict[str, float],
    env_config: dict[str, Any],
    lambda_norm: float,
    lambda_event: float,
    lambda_raw_violation: float,
    event_threshold: float,
    k0: float,
    k1: float,
    eps_side: float,
    n_envs: int,
):
    def _single_env(env_seed: int) -> gym.Env:
        if env_kind == "baseline":
            return make_baseline_single_env(
                namespace,
                seed=env_seed,
                reward_config=reward_config,
                env_config=env_config,
            )
        return namespace["make_event_cbf_single_env"](
            seed=env_seed,
            lambda_norm=lambda_norm,
            lambda_event=lambda_event,
            lambda_raw_violation=lambda_raw_violation,
            event_threshold=event_threshold,
            k0=k0,
            k1=k1,
            eps_side=eps_side,
            env_config=env_config,
            reward_config=reward_config,
        )

    return namespace["_make_vectorized_env"](
        _single_env,
        seed=seed,
        n_envs=n_envs,
        use_subproc=True,
        start_method=namespace["DDPG_SUBPROC_START_METHOD"],
    )


def make_eval_env(
    namespace: dict[str, Any],
    *,
    env_kind: str,
    seed: int,
    reward_config: dict[str, float],
    env_config: dict[str, Any],
    event_threshold: float,
    k0: float,
    k1: float,
    eps_side: float,
    use_distance_task: bool,
    task_distance_m: float,
    task_max_steps: int,
) -> gym.Env:
    if env_kind == "baseline":
        env = make_baseline_single_env(
            namespace,
            seed=seed,
            reward_config=reward_config,
            env_config=env_config,
        )
    else:
        env = namespace["make_event_cbf_single_env"](
            seed=seed,
            lambda_norm=0.0,
            lambda_event=0.0,
            event_threshold=event_threshold,
            k0=k0,
            k1=k1,
            eps_side=eps_side,
            env_config=env_config,
            reward_config=reward_config,
        )
    if use_distance_task and "make_task_evaluation_wrapper" in namespace:
        env = namespace["make_task_evaluation_wrapper"](
            env,
            task_distance_m=float(task_distance_m),
            max_steps=int(task_max_steps),
        )
    else:
        namespace["configure_paper_evaluation_env"](env, steps=namespace["PAPER_EVAL_STEPS"])
    return env


def evaluate_model(
    namespace: dict[str, Any],
    model: Any,
    *,
    env_kind: str,
    episodes: int,
    seed: int,
    reward_config: dict[str, float],
    env_config: dict[str, Any],
    event_threshold: float,
    k0: float,
    k1: float,
    eps_side: float,
    use_distance_task: bool,
    task_distance_m: float,
    task_max_steps: int,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for episode in range(episodes):
        env = make_eval_env(
            namespace,
            env_kind=env_kind,
            seed=seed + episode,
            reward_config=reward_config,
            env_config=env_config,
            event_threshold=event_threshold,
            k0=k0,
            k1=k1,
            eps_side=eps_side,
            use_distance_task=use_distance_task,
            task_distance_m=task_distance_m,
            task_max_steps=task_max_steps,
        )
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
        progress_rewards: list[float] = []
        kpi_info_rows: list[dict[str, Any]] = []
        last_task_info: dict[str, Any] = {}
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
            progress_rewards.append(float(info.get("karalakou_progress_reward", 0.0)))
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
            step_count += 1
            done = bool(terminated or truncated)

        episode_row = {
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
            "min_h": float(np.nanmin(min_h_values))
            if min_h_values and not np.all(np.isnan(min_h_values))
            else np.nan,
            "mean_old_potential_cost": float(np.mean(old_potential_costs)) if old_potential_costs else 0.0,
            "mean_safety_potential_cost": float(np.mean(safety_potential_costs)) if safety_potential_costs else 0.0,
            "mean_lateral_y_cost": float(np.mean(lateral_costs)) if lateral_costs else 0.0,
            "mean_progress_reward": float(np.mean(progress_rewards)) if progress_rewards else 0.0,
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
                    task_completed=bool(last_task_info.get("task_completed", False)),
                    fallback_steps=step_count,
                    fallback_distance_m=float(last_task_info.get("task_distance_traveled_m", 0.0)),
                    fallback_dt_s=namespace["kpi_policy_dt"](env) if "kpi_policy_dt" in namespace else np.nan,
                )
            )
        rows.append(episode_row)
        env.close()
    return pd.DataFrame(rows)


class VariantEvalCallback(BaseCallback):
    def __init__(
        self,
        namespace: dict[str, Any],
        *,
        variant: str,
        env_kind: str,
        reward_config: dict[str, float],
        env_config: dict[str, Any],
        event_threshold: float,
        k0: float,
        k1: float,
        eps_side: float,
        use_distance_task: bool,
        task_distance_m: float,
        task_max_steps: int,
        eval_freq: int,
        episodes: int,
        seed: int,
    ) -> None:
        super().__init__(verbose=0)
        self.namespace = namespace
        self.variant = variant
        self.env_kind = env_kind
        self.reward_config = reward_config
        self.env_config = env_config
        self.event_threshold = float(event_threshold)
        self.k0 = float(k0)
        self.k1 = float(k1)
        self.eps_side = float(eps_side)
        self.use_distance_task = bool(use_distance_task)
        self.task_distance_m = float(task_distance_m)
        self.task_max_steps = int(task_max_steps)
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
            env_kind=self.env_kind,
            episodes=self.episodes,
            seed=self.seed + self.num_timesteps,
            reward_config=self.reward_config,
            env_config=self.env_config,
            event_threshold=self.event_threshold,
            k0=self.k0,
            k1=self.k1,
            eps_side=self.eps_side,
            use_distance_task=self.use_distance_task,
            task_distance_m=self.task_distance_m,
            task_max_steps=self.task_max_steps,
        )
        row: dict[str, float | str] = {
            "variant": self.variant,
            "timesteps": float(self.num_timesteps),
            **summarize(metrics),
        }
        row["behavior_score"] = behavior_score(row)
        self.records.append(row)
        recorder = self.namespace.get("record_tensorboard_row")
        if callable(recorder):
            recorder(self.logger, f"eval/{self.variant}", row, step=self.num_timesteps)
        print(
            "[safety-potential-eval]"
            f" {self.variant}"
            f" steps={self.num_timesteps:,}"
            f" return={row['return_mean']:.2f}"
            f" complete={row.get('completion_rate', 0.0):.2%}"
            f" abs_speed={row['mean_abs_speed_error']:.3f}"
            f" lat_y={row['mean_lat_y_error_m']:.3f}"
            f" event_int={row['event_intervention_rate']:.2%}"
            f" corr={row['mean_correction_norm']:.3f}"
            f" ego_coll={row['ego_collisions_mean']:.2f}",
            flush=True,
        )
        return True


def plot_history(history: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4))
    axes = axes.ravel()
    correction_column = (
        "mean_meaningful_correction_norm" if "mean_meaningful_correction_norm" in history.columns else "mean_correction_norm"
    )
    panels = [
        ("return_mean", "Return", False),
        ("mean_abs_speed_error", "Abs Speed Error", False),
        ("mean_lat_y_error_m", "Lateral y Error", False),
        ("event_intervention_rate", "Meaningful Intervention", True),
        (correction_column, "Meaningful Correction", False),
        ("ego_collisions_mean", "Ego Collisions", False),
    ]
    if not history.empty:
        for axis, (column, panel_title, percent) in zip(axes, panels):
            axis.plot(history["timesteps"], history[column], marker="o")
            axis.set_title(panel_title)
            if percent:
                axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
            axis.grid(True, alpha=0.28)
            axis.set_xlabel("Training timestep")
    for axis in axes:
        if not axis.has_data():
            axis.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    labels = summary["variant"].tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.8))
    correction_column = (
        "mean_meaningful_correction_norm" if "mean_meaningful_correction_norm" in summary.columns else "mean_correction_norm"
    )
    panels = [
        ("return_mean", "Return", False),
        ("mean_abs_speed_error", "Abs Speed Error", False),
        ("mean_lat_y_error_m", "Lateral y Error", False),
        ("event_intervention_rate", "Meaningful Intervention", True),
        (correction_column, "Meaningful Correction", False),
    ]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for axis, (column, title, percent) in zip(axes, panels):
        axis.bar(x, summary[column].to_numpy(dtype=float), color=colors[: len(labels)])
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
        if percent:
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    fig.suptitle("Safety-Set Potential Reward: Three Policy Variants", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_resilient(frame: pd.DataFrame, output_path: Path, *, index: bool = False) -> None:
    """Atomically write a CSV while tolerating transient OneDrive reparse races."""
    output_path = Path(output_path)
    last_error: OSError | None = None
    for attempt in range(8):
        temporary_path = output_path.with_name(
            f".{output_path.name}.{os.getpid()}.{attempt}.tmp"
        )
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(temporary_path, index=index)
            os.replace(temporary_path, output_path)
            return
        except OSError as exc:
            last_error = exc
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt + 1 < 8:
                delay = min(2.0, 0.25 * (2**attempt))
                print(
                    f"[safety-potential-io] retrying CSV write ({attempt + 1}/7) "
                    f"after {type(exc).__name__}: {output_path}",
                    flush=True,
                )
                time.sleep(delay)
    raise OSError(f"Could not write CSV after 8 attempts: {output_path}") from last_error


def plot_paired_deployment_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot the same CBF-trained policy with and without deployment shielding."""
    if summary.empty:
        return
    labels = summary["evaluation"].tolist()
    x = np.arange(len(labels))
    panels = [
        ("return_mean", "Return", False),
        ("completion_rate", "Completion", True),
        ("mean_abs_speed_error", "Abs Speed Error", False),
        ("mean_lat_y_error_m", "Lateral y Error", False),
        ("ego_collisions_mean", "Ego Collisions", False),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 4.8))
    colors = ["#0072B2", "#D55E00"]
    for axis, (column, title, percent) in zip(axes, panels):
        values = summary[column].to_numpy(dtype=float)
        axis.bar(x, values, color=colors[: len(values)])
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
        if percent:
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    fig.suptitle("Old Reward: CBF-Trained DDPG Deployment Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate_paired_deployments(
    namespace: dict[str, Any],
    args: argparse.Namespace,
    *,
    model_path: Path,
    reward_config: dict[str, float],
    env_config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """Evaluate one CBF-trained model with the shield enabled and disabled.

    Both conditions use the same old reward configuration, seeds, task
    protocol, and model.  Only the deployment-time CBF wrapper changes.
    """
    paired_dir = output_dir / "paired_evaluation"
    paired_dir.mkdir(parents=True, exist_ok=True)
    model_sha256 = _sha256_file(model_path)
    evaluation_seed = int(args.seed) + 200_000
    conditions = (
        ("old_reward_no_cbf", "baseline", evaluation_seed),
        ("old_reward_with_cbf", "event_cbf", evaluation_seed),
    )

    model = namespace["DDPG"].load(str(model_path), device=args.device)
    rows: list[dict[str, float | str | bool]] = []
    for condition, env_kind, seed in conditions:
        episodes_path = paired_dir / f"{condition}_episodes.csv"
        summary_path = paired_dir / f"{condition}_summary.csv"
        if episodes_path.exists() and summary_path.exists() and not args.no_resume:
            cached_summary = pd.read_csv(summary_path)
            cached_row = cached_summary.iloc[0] if not cached_summary.empty else None
            cached_hash = str(cached_row.get("model_sha256", "")) if cached_row is not None else ""
            cached_protocol_matches = (
                cached_row is not None
                and int(float(cached_row.get("episodes", -1))) == int(args.final_eval_episodes)
                and int(float(cached_row.get("seed", -1))) == int(seed)
                and abs(float(cached_row.get("task_distance_m", np.nan)) - float(args.task_distance_m)) < 1e-9
                and int(float(cached_row.get("task_max_steps", -1))) == int(args.task_max_steps)
                and str(cached_row.get("reward_term", "")) == str(args.reward_term)
            )
            if cached_hash == model_sha256 and cached_protocol_matches:
                print(f"[paired-eval] {condition} already complete; loading summary", flush=True)
                rows.append(cached_row.to_dict())
                continue
            print(f"[paired-eval] {condition} cache is stale; refreshing", flush=True)

        print(f"[paired-eval] evaluating {condition} env={env_kind}", flush=True)
        metrics = evaluate_model(
            namespace,
            model,
            env_kind=env_kind,
            episodes=args.final_eval_episodes,
            seed=seed,
            reward_config=reward_config,
            env_config=env_config,
            event_threshold=args.event_threshold,
            k0=args.k0,
            k1=args.k1,
            eps_side=args.eps_side,
            use_distance_task=not args.legacy_fixed_step_eval,
            task_distance_m=args.task_distance_m,
            task_max_steps=args.task_max_steps,
        )
        write_csv_resilient(metrics, episodes_path)
        summary: dict[str, float | str | bool] = {
            "evaluation": condition,
            "env_kind": env_kind,
            "reward_term": str(args.reward_term),
            "model_path": str(model_path),
            "model_sha256": model_sha256,
            "seed": float(seed),
            "k0": float(args.k0),
            "k1": float(args.k1),
            "eps_side": float(args.eps_side),
            "event_threshold": float(args.event_threshold),
            "use_distance_task_eval": bool(not args.legacy_fixed_step_eval),
            "task_distance_m": float(args.task_distance_m),
            "task_max_steps": float(args.task_max_steps),
            "wf": float(reward_config["wf"]),
            "w_safe": float(reward_config["w_safe"]),
            "use_current_potential": bool(reward_config["use_current_potential"]),
            "use_safety_potential": bool(reward_config["use_safety_potential"]),
            **summarize(metrics),
        }
        summary["behavior_score"] = behavior_score(summary)
        write_csv_resilient(pd.DataFrame([summary]), summary_path)
        rows.append(summary)

    summary_frame = pd.DataFrame(rows)
    write_csv_resilient(summary_frame, paired_dir / "comparison.csv")
    if len(summary_frame) == 2:
        no_cbf = summary_frame.iloc[0]
        with_cbf = summary_frame.iloc[1]
        delta: dict[str, float | str] = {
            "comparison": "with_cbf_minus_no_cbf",
            "from_evaluation": str(no_cbf["evaluation"]),
            "to_evaluation": str(with_cbf["evaluation"]),
        }
        for column in summary_frame.columns:
            if column in {"evaluation", "env_kind", "model_path", "model_sha256", "from_evaluation", "to_evaluation"}:
                continue
            if pd.api.types.is_number(no_cbf[column]) and pd.api.types.is_number(with_cbf[column]):
                delta[column] = float(with_cbf[column]) - float(no_cbf[column])
        write_csv_resilient(pd.DataFrame([delta]), paired_dir / "delta_with_cbf_minus_no_cbf.csv")
    plot_paired_deployment_summary(summary_frame, paired_dir / "comparison.png")
    manifest = {
        "schema_version": 1,
        "complete": len(summary_frame) == len(conditions),
        "reward_term": str(args.reward_term),
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "env_config": env_config,
        "reward_config": reward_config,
        "conditions": [
            {"evaluation": condition, "env_kind": env_kind, "seed": seed}
            for condition, env_kind, seed in conditions
        ],
        "episodes": int(args.final_eval_episodes),
        "task_distance_m": float(args.task_distance_m),
        "task_max_steps": int(args.task_max_steps),
    }
    temporary_manifest = paired_dir / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    temporary_manifest.replace(paired_dir / "manifest.json")
    print(f"[paired-eval] wrote {paired_dir / 'comparison.csv'}", flush=True)
    return summary_frame


def latest_checkpoint(variant_dir: Path) -> Path | None:
    checkpoints = sorted([*variant_dir.glob("ckpt_*.zip"), *variant_dir.glob("checkpoint_*.zip")])
    return checkpoints[-1] if checkpoints else None


def make_baseline_reward_config(
    namespace: dict[str, Any],
    *,
    progress_reward_weight: float,
    progress_clip: float,
    eps_side: float,
    reward_trial: dict[str, float | str | bool] = SAFETY_REWARD_TRIAL,
) -> dict[str, float]:
    config = make_reward_config(namespace, reward_trial)
    config.update(
        {
            "progress_reward_weight": float(progress_reward_weight),
            "progress_clip": float(progress_clip),
            "safety_potential_eps_side": float(eps_side),
        }
    )
    return config


def make_cbf_reward_config(
    namespace: dict[str, Any],
    args: argparse.Namespace,
    *,
    reward_trial: dict[str, float | str | bool] = SAFETY_REWARD_TRIAL,
) -> dict[str, float]:
    config = make_reward_config(namespace, reward_trial)
    config.update(
        {
            "progress_reward_weight": float(args.progress_reward_weight),
            "progress_clip": float(args.progress_clip),
            "safety_potential_eps_side": float(args.eps_side),
        }
    )
    return config


def reward_config_for_variant(
    variant_cfg: dict[str, Any],
    *,
    baseline_reward_config: dict[str, float],
    cbf_reward_config: dict[str, float],
) -> dict[str, float]:
    if str(variant_cfg.get("env_kind", "")) == "baseline":
        return baseline_reward_config
    return cbf_reward_config


def apply_guided_runtime_config(
    model: Any,
    namespace: dict[str, Any],
    variant_cfg: dict[str, Any],
    args: argparse.Namespace,
    env_config: dict[str, Any],
) -> None:
    if variant_cfg.get("model_class_name") != "GuidedCBFDDPG":
        return
    model.lambda_bc = float(variant_cfg["lambda_bc"])
    model.lambda_bc_initial = float(variant_cfg["lambda_bc"])
    model.lambda_bc_final = (
        None
        if namespace.get("GUIDED_CBF_LAMBDA_BC_FINAL") is None
        else float(namespace["GUIDED_CBF_LAMBDA_BC_FINAL"])
    )
    model.lambda_bc_decay_steps = int(namespace.get("GUIDED_CBF_LAMBDA_BC_DECAY_STEPS", 0))
    model.use_projected_q = bool(namespace["GUIDED_CBF_USE_PROJECTED_Q"])
    model.projected_q_weight = float(namespace["GUIDED_CBF_PROJECTED_Q_WEIGHT"])
    model.critic_action_mode = str(
        variant_cfg.get("critic_action_mode", namespace.get("GUIDED_CBF_CRITIC_ACTION_MODE", "raw"))
    ).strip().lower()
    model.actor_action_mode = str(
        variant_cfg.get("actor_action_mode", namespace.get("GUIDED_CBF_ACTOR_ACTION_MODE", "raw"))
    ).strip().lower()
    if model.critic_action_mode not in {"raw", "safe"}:
        raise ValueError(f"critic_action_mode must be 'raw' or 'safe', got {model.critic_action_mode!r}")
    if model.actor_action_mode not in {"raw", "diff_cbf"}:
        raise ValueError(f"actor_action_mode must be 'raw' or 'diff_cbf', got {model.actor_action_mode!r}")
    model.cbf_projection_steps = int(namespace.get("GUIDED_CBF_DIFF_PROJECTION_STEPS", 2))
    model.cbf_projection_fd_step = float(namespace.get("GUIDED_CBF_DIFF_PROJECTION_FD_STEP", 1e-3))
    model.cbf_road_width = float(env_config.get("road_width", namespace["ENV_CONFIG"].get("road_width", 10.2)))
    model.cbf_sensing_range = float(env_config.get("sensing_range", namespace["ENV_CONFIG"].get("sensing_range", 90.0)))
    model.cbf_obs_vmax = float(env_config.get("observation_vmax", namespace["ENV_CONFIG"].get("observation_vmax", 24.0)))
    model.cbf_obs_vymax = float(env_config.get("observation_vymax", namespace["ENV_CONFIG"].get("observation_vymax", 7.2)))
    model.cbf_eps_side = float(args.eps_side)
    model.cbf_k0 = float(args.k0)
    model.cbf_k1 = float(args.k1)


def _tb_scalar(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return np.nan
    return scalar if np.isfinite(scalar) else np.nan


def write_summarywriter_tensorboard_row(
    namespace: dict[str, Any],
    *,
    tb_root: Path,
    run_name: str,
    prefix: str,
    row: dict[str, Any],
    step: int,
) -> Path | None:
    writer_cls = namespace.get("SummaryWriter")
    if writer_cls is None:
        return None
    run_dir = Path(tb_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        writer = writer_cls(log_dir=str(run_dir))
        kpi_index = namespace.get("TENSORBOARD_KPI_INDEX_MARKDOWN")
        if kpi_index:
            writer.add_text("00_kpi_index", str(kpi_index), 0)
        specs = namespace.get("DEFAULT_TENSORBOARD_INFO_METRICS", {})
        logged_keys: set[str] = set()
        for tag, keys in specs.items():
            scalar = np.nan
            for key in keys:
                if key not in row:
                    continue
                scalar = _tb_scalar(row.get(key))
                if np.isfinite(scalar):
                    logged_keys.add(key)
                    break
            if np.isfinite(scalar):
                writer.add_scalar(f"{prefix}/{tag}", scalar, int(step))
        for key, value in row.items():
            if key in logged_keys or key in {"variant", "label", "model_path"}:
                continue
            scalar = _tb_scalar(value)
            if np.isfinite(scalar):
                writer.add_scalar(f"{prefix}/99_raw/{key}", scalar, int(step))
        writer.flush()
        writer.close()
    except Exception as exc:
        # TensorBoard is auxiliary output. A filesystem/extension failure
        # must not discard a completed model, evaluation, or CSV summary.
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        print(f"[safety-potential-tb] final-eval write skipped: {exc!r}", flush=True)
        return None
    return run_dir


def train_variant(
    namespace: dict[str, Any],
    args: argparse.Namespace,
    *,
    variant_cfg: dict[str, Any],
    baseline_reward_config: dict[str, float],
    cbf_reward_config: dict[str, float],
    env_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, float | str | bool]:
    variant = str(variant_cfg["variant"])
    variant_index = next(
        (index for index, item in enumerate(VARIANTS) if str(item["variant"]) == variant),
        0,
    )
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    model_path = variant_dir / "model.zip"
    history_path = variant_dir / "train_eval_history.csv"
    final_episodes_path = variant_dir / "final_eval_episodes.csv"
    final_summary_path = variant_dir / "final_summary.csv"
    plot_path = variant_dir / "train_eval_history.png"

    if final_summary_path.exists() and model_path.exists() and not args.no_resume:
        print(f"[safety-potential] {variant} already complete; loading summary", flush=True)
        return pd.read_csv(final_summary_path).iloc[0].to_dict()

    env_kind = str(variant_cfg["env_kind"])
    reward_config = reward_config_for_variant(
        variant_cfg,
        baseline_reward_config=baseline_reward_config,
        cbf_reward_config=cbf_reward_config,
    )
    train_env = make_training_env(
        namespace,
        env_kind=env_kind,
        seed=args.seed + 1_000 * (variant_index + 1),
        reward_config=reward_config,
        env_config=env_config,
        lambda_norm=args.lambda_norm,
        lambda_event=args.lambda_event if env_kind != "baseline" else 0.0,
        lambda_raw_violation=args.raw_violation_reward_weight if env_kind != "baseline" else 0.0,
        event_threshold=args.event_threshold,
        k0=args.k0,
        k1=args.k1,
        eps_side=args.eps_side,
        n_envs=args.n_envs,
    )
    callback = VariantEvalCallback(
        namespace,
        variant=variant,
        env_kind=env_kind,
        reward_config=reward_config,
        env_config=env_config,
        event_threshold=args.event_threshold,
        k0=args.k0,
        k1=args.k1,
        eps_side=args.eps_side,
        use_distance_task=not args.legacy_fixed_step_eval,
        task_distance_m=args.task_distance_m,
        task_max_steps=args.task_max_steps,
        eval_freq=args.train_eval_freq,
        episodes=args.train_eval_episodes,
        seed=args.seed + 10_000 * (variant_index + 1),
    )
    learn_callback: BaseCallback = callback
    tb_root = output_dir / "tb"
    tb_custom_root = tb_root / "custom"
    tb_run_name = TB_VARIANT_RUN_NAMES.get(variant, variant[:12])
    if not args.skip_tensorboard:
        tb_custom_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_tensorboard and "TensorBoardMetricsBridgeCallback" in namespace:
        tb_callback = namespace["TensorBoardMetricsBridgeCallback"](
            variant=variant,
            run_name=tb_run_name,
            tb_root=tb_custom_root,
            write_freq=int(args.tb_write_freq),
            flush_freq=int(args.tb_flush_freq),
            config={
                "phase": "train",
                "variant": variant,
                "traffic_model": active_traffic_model(env_config),
                "eps_side": float(args.eps_side),
                "k0": float(args.k0),
                "k1": float(args.k1),
                "lambda_norm": float(args.lambda_norm if env_kind != "baseline" else 0.0),
                "lambda_event": float(args.lambda_event if env_kind != "baseline" else 0.0),
                "lambda_raw_violation": float(args.raw_violation_reward_weight if env_kind != "baseline" else 0.0),
                "event_threshold": float(args.event_threshold),
                "projected_q": bool(namespace.get("GUIDED_CBF_USE_PROJECTED_Q", False))
                if variant_cfg.get("model_class_name") == "GuidedCBFDDPG"
                else False,
                "projected_q_weight": float(namespace.get("GUIDED_CBF_PROJECTED_Q_WEIGHT", 0.0))
                if variant_cfg.get("model_class_name") == "GuidedCBFDDPG"
                else 0.0,
                "task_distance_m": float(args.task_distance_m),
                "task_max_steps": int(args.task_max_steps),
            },
        )
        learn_callback = CallbackList([callback, tb_callback])
    n_actions = train_env.action_space.shape[-1]
    model_cls = variant_cfg.get("model_class") or namespace[str(variant_cfg["model_class_name"])]
    checkpoint = latest_checkpoint(variant_dir)
    if checkpoint is not None and not args.no_resume:
        print(f"[safety-potential] loading checkpoint for {variant}: {checkpoint}", flush=True)
        model = model_cls.load(str(checkpoint), env=train_env, device=args.device)
        apply_guided_runtime_config(model, namespace, variant_cfg, args, env_config)
        model.tensorboard_log = None
    else:
        action_noise = namespace["make_ou_action_noise"](n_actions, n_envs=args.n_envs)
        model_kwargs: dict[str, Any] = {}
        if variant_cfg.get("model_class_name") == "GuidedCBFDDPG":
            model_kwargs.update(
                {
                    "lambda_bc": float(variant_cfg["lambda_bc"]),
                    "lambda_bc_final": None
                    if namespace.get("GUIDED_CBF_LAMBDA_BC_FINAL") is None
                    else float(namespace["GUIDED_CBF_LAMBDA_BC_FINAL"]),
                    "lambda_bc_decay_steps": int(namespace.get("GUIDED_CBF_LAMBDA_BC_DECAY_STEPS", 0)),
                    "bc_delta": namespace["GUIDED_CBF_BC_DELTA"],
                    "bc_action_scale": namespace["GUIDED_CBF_ACTION_SCALE"],
                    "bc_weight_max": namespace["GUIDED_CBF_WEIGHT_MAX"],
                    "use_projected_q": namespace["GUIDED_CBF_USE_PROJECTED_Q"],
                    "projected_q_weight": namespace["GUIDED_CBF_PROJECTED_Q_WEIGHT"],
                    "critic_action_mode": str(
                        variant_cfg.get("critic_action_mode", namespace.get("GUIDED_CBF_CRITIC_ACTION_MODE", "raw"))
                    ),
                    "actor_action_mode": str(
                        variant_cfg.get("actor_action_mode", namespace.get("GUIDED_CBF_ACTOR_ACTION_MODE", "raw"))
                    ),
                    "cbf_projection_steps": int(namespace.get("GUIDED_CBF_DIFF_PROJECTION_STEPS", 2)),
                    "cbf_projection_fd_step": float(namespace.get("GUIDED_CBF_DIFF_PROJECTION_FD_STEP", 1e-3)),
                    "cbf_road_width": float(env_config.get("road_width", namespace["ENV_CONFIG"].get("road_width", 10.2))),
                    "cbf_sensing_range": float(
                        env_config.get("sensing_range", namespace["ENV_CONFIG"].get("sensing_range", 90.0))
                    ),
                    "cbf_obs_vmax": float(
                        env_config.get("observation_vmax", namespace["ENV_CONFIG"].get("observation_vmax", 24.0))
                    ),
                    "cbf_obs_vymax": float(
                        env_config.get("observation_vymax", namespace["ENV_CONFIG"].get("observation_vymax", 7.2))
                    ),
                    "cbf_eps_side": float(args.eps_side),
                    "cbf_k0": float(args.k0),
                    "cbf_k1": float(args.k1),
                }
            )
        model = model_cls(
            "MlpPolicy",
            train_env,
            learning_rate=namespace["DDPG_LEARNING_RATE"],
            buffer_size=namespace["DDPG_REPLAY_MEMORY"],
            learning_starts=namespace["DDPG_LEARNING_STARTS"],
            batch_size=namespace["DDPG_BATCH_SIZE"],
            tau=namespace["DDPG_TAU"],
            gamma=namespace["DDPG_GAMMA"],
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=action_noise,
            policy_kwargs={"net_arch": [256, 128]},
            tensorboard_log=None,
            verbose=0,
            seed=args.seed + variant_index + 1,
            device=args.device,
            **model_kwargs,
        )

    print(
        "[safety-potential] training"
        f" {variant}"
        f" timesteps={args.timesteps:,}"
        f" current={model.num_timesteps:,}"
        f" env={env_kind}",
        flush=True,
    )
    start = time.time()
    try:
        while int(model.num_timesteps) < int(args.timesteps):
            remaining = int(args.timesteps) - int(model.num_timesteps)
            chunk = min(int(args.chunk_timesteps), remaining)
            model.learn(
                total_timesteps=chunk,
                callback=learn_callback,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            checkpoint_path = variant_dir / f"ckpt_{int(model.num_timesteps):06d}.zip"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(checkpoint_path))
            print(f"[safety-potential] saved {checkpoint_path}", flush=True)
    finally:
        train_env.close()
    elapsed_sec = time.time() - start
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    history = pd.DataFrame(callback.records)
    write_csv_resilient(history, history_path)
    plot_history(history, plot_path, title=variant)

    print(f"[safety-potential] evaluating {variant}", flush=True)
    final_metrics = evaluate_model(
        namespace,
        model,
        env_kind=env_kind,
        episodes=args.final_eval_episodes,
        seed=args.seed + 100_000 * (variant_index + 1),
        reward_config=reward_config,
        env_config=env_config,
        event_threshold=args.event_threshold,
        k0=args.k0,
        k1=args.k1,
        eps_side=args.eps_side,
        use_distance_task=not args.legacy_fixed_step_eval,
        task_distance_m=args.task_distance_m,
        task_max_steps=args.task_max_steps,
    )
    write_csv_resilient(final_metrics, final_episodes_path)
    summary: dict[str, float | str | bool] = {
        "variant": variant,
        "label": str(variant_cfg["label"]),
        "reward_term": str(args.reward_term),
        "reward_trial": str(REWARD_TRIALS[str(args.reward_term)]["trial_name"]),
        "model_path": str(model_path),
        "elapsed_sec": float(elapsed_sec),
        "timesteps": float(args.timesteps),
        "traffic_model": active_traffic_model(env_config),
        "k0": float(args.k0),
        "k1": float(args.k1),
        "eps_side": float(args.eps_side),
        "lambda_norm": float(args.lambda_norm if env_kind != "baseline" else 0.0),
        "lambda_event": float(args.lambda_event if env_kind != "baseline" else 0.0),
        "lambda_bc": float(variant_cfg["lambda_bc"]),
        "critic_action_mode": str(variant_cfg.get("critic_action_mode", "raw")),
        "actor_action_mode": str(variant_cfg.get("actor_action_mode", "raw")),
        "event_threshold": float(args.event_threshold),
        "wy": float(reward_config["wy"]),
        "wf": float(reward_config["wf"]),
        "w_safe": float(reward_config["w_safe"]),
        "use_current_potential": bool(reward_config["use_current_potential"]),
        "use_safety_potential": bool(reward_config["use_safety_potential"]),
        "progress_reward_weight": float(reward_config.get("progress_reward_weight", 0.0)),
        "use_distance_task_eval": bool(not args.legacy_fixed_step_eval),
        "task_distance_m": float(args.task_distance_m),
        "task_max_steps": float(args.task_max_steps),
        **summarize(final_metrics),
    }
    summary["behavior_score"] = behavior_score(summary)
    write_csv_resilient(pd.DataFrame([summary]), final_summary_path)
    (variant_dir / "run_config.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if not args.skip_tensorboard:
        final_tb_dir = write_summarywriter_tensorboard_row(
            namespace,
            tb_root=tb_root / "final",
            run_name=tb_run_name,
            prefix=f"eval_final/{variant}",
            row=summary,
            step=int(args.timesteps),
        )
        if final_tb_dir is not None:
            print(f"[safety-potential-tb] final eval {variant}: {final_tb_dir}", flush=True)
    print(
        "[safety-potential-result]"
        f" {variant}"
        f" return={summary['return_mean']:.2f}"
        f" complete={summary.get('completion_rate', 0.0):.2%}"
        f" abs_speed={summary['mean_abs_speed_error']:.3f}"
        f" lat_y={summary['mean_lat_y_error_m']:.3f}"
        f" event_int={summary['event_intervention_rate']:.2%}"
        f" corr={summary['mean_correction_norm']:.3f}"
        f" qp_fail={summary['qp_failure_rate']:.2%}"
        f" ego_coll={summary['ego_collisions_mean']:.2f}",
        flush=True,
    )
    return summary


def export_videos(
    namespace: dict[str, Any],
    *,
    env_config: dict[str, Any],
    output_dir: Path,
    model_paths: dict[str, Path],
    args: argparse.Namespace,
) -> Path:
    namespace["ENV_CONFIG"] = env_config
    namespace["DDPG_MODEL_PATH"] = model_paths["ddpg"]
    namespace["DDPG_CBF_MODEL_PATH"] = model_paths["ddpg_cbf_reward"]
    namespace["GUIDED_DDPG_CBF_MODEL_PATH"] = model_paths["guided_ddpg_cbf"]
    variants = ["ddpg", "ddpg-cbf", "guided-ddpg-cbf"]
    from scripts.rendering.render_laneless_policy_videos import load_models

    models = load_models(namespace, variants)
    scenarios = selected_scenarios(make_scenarios(float(namespace["ENV_CONFIG"]["road_width"])), args.scenario)
    videos_dir = output_dir / "videos"
    rows: list[dict[str, Any]] = []
    if not args.skip_normal_video:
        normal_dir = videos_dir / "normal"
        for variant_index, variant in enumerate(variants):
            output_path = normal_dir / f"{variant_file_name(variant)}.mp4"
            print(f"[safety-potential-video] normal | {variant} -> {output_path}", flush=True)
            rows.append(
                run_clip(
                    namespace=namespace,
                    model=models[variant],
                    variant=variant,
                    scenario=None,
                    output_path=output_path,
                    seed=int(args.seed) + 100_000 * variant_index,
                    sim_seconds=float(args.video_seconds),
                    fps=int(args.video_fps),
                    repeat_frames=int(args.video_repeat_frames),
                )
            )
    if not args.skip_scenario_videos:
        scenario_dir = videos_dir / "scenarios"
        for scenario_index, scenario in enumerate(scenarios):
            for variant_index, variant in enumerate(variants):
                output_path = scenario_dir / scenario_file_name(scenario) / f"{variant_file_name(variant)}.mp4"
                print(f"[safety-potential-video] {scenario.name} | {variant} -> {output_path}", flush=True)
                rows.append(
                    run_clip(
                        namespace=namespace,
                        model=models[variant],
                        variant=variant,
                        scenario=scenario,
                        output_path=output_path,
                        seed=int(args.seed) + 10_000 * scenario_index + 100_000 * variant_index,
                        sim_seconds=float(args.video_seconds),
                        fps=int(args.video_fps),
                        repeat_frames=int(args.video_repeat_frames),
                    )
                )
    summary_path = videos_dir / "video_summary.csv"
    write_csv_resilient(pd.DataFrame(rows), summary_path)
    print(f"[safety-potential-video] wrote {summary_path}", flush=True)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPG variants with a selectable historical or CBF safety potential reward.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--chunk-timesteps", type=int, default=5_000)
    parser.add_argument("--train-eval-freq", type=int, default=10_000)
    parser.add_argument("--train-eval-episodes", type=int, default=2)
    parser.add_argument("--final-eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=307_000)
    parser.add_argument("--n-envs", type=int, default=20)
    parser.add_argument("--k0", type=float, default=5.29)
    parser.add_argument("--k1", type=float, default=3.68)
    parser.add_argument("--eps-side", type=float, default=0.10)
    parser.add_argument("--lambda-norm", type=float, default=0.025)
    parser.add_argument("--lambda-event", type=float, default=0.02)
    parser.add_argument("--raw-violation-reward-weight", type=float, default=0.0)
    parser.add_argument("--event-threshold", type=float, default=0.03)
    parser.add_argument("--guided-lambda-bc", type=float, default=None)
    parser.add_argument("--guided-lambda-bc-final", type=float, default=None)
    parser.add_argument("--guided-bc-decay-steps", type=int, default=0)
    parser.add_argument("--guided-projected-q-weight", type=float, default=0.0)
    parser.add_argument("--disable-guided-projected-q", action="store_true")
    parser.add_argument("--task-distance-m", type=float, default=1000.0)
    parser.add_argument("--task-max-steps", type=int, default=1200)
    parser.add_argument("--legacy-fixed-step-eval", action="store_true")
    parser.add_argument("--progress-reward-weight", type=float, default=0.0)
    parser.add_argument("--progress-clip", type=float, default=1.25)
    parser.add_argument(
        "--reward-term",
        choices=sorted(REWARD_TRIALS),
        default="safety",
        help="Reward safety term: 'safety' uses the CBF-clearance potential; 'old' uses the historical Karalakou potential.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-tensorboard", action="store_true")
    parser.add_argument(
        "--paired-eval",
        action="store_true",
        help="After training ddpg_cbf_reward, evaluate the same model with and without the CBF deployment shield.",
    )
    parser.add_argument("--tb-write-freq", type=int, default=100)
    parser.add_argument("--tb-flush-freq", type=int, default=500)
    parser.add_argument("--force-mtm-congested", action="store_true", default=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help=(
            "Optional subset of policy variants to train. "
            f"Default: {' '.join(INITIAL_VARIANT_NAMES)}."
        ),
    )
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-normal-video", action="store_true")
    parser.add_argument("--skip-scenario-videos", action="store_true")
    parser.add_argument("--video-seconds", type=float, default=20.0)
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--video-repeat-frames", type=int, default=3)
    parser.add_argument("--scenario", action="append", default=None)
    add_env_config_args(parser)
    return parser.parse_args()


def main() -> int:
    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()
    if args.paired_eval and args.reward_term != "old":
        raise ValueError("--paired-eval is reserved for the old-reward CBF deployment comparison.")
    if args.paired_eval and args.variants is not None and "ddpg_cbf_reward" not in args.variants:
        raise ValueError("--paired-eval requires ddpg_cbf_reward in --variants.")
    project_root = find_project_root(args.project_root or Path.cwd())
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    namespace: dict[str, Any] = {"__name__": "__main__"}
    exec_notebook_cells(notebook_path, NOTEBOOK_DEPS, namespace)
    namespace["DEVICE"] = args.device
    namespace["CBF_K0"] = float(args.k0)
    namespace["CBF_K1"] = float(args.k1)
    namespace["CBF_EPS_SIDE"] = float(args.eps_side)
    namespace["CBF_FILTER_REWARD_LAMBDA"] = float(args.lambda_norm)
    install_minimal_guided_cbf(namespace)
    namespace["GUIDED_CBF_PROJECTED_Q_WEIGHT"] = float(np.clip(args.guided_projected_q_weight, 0.0, 1.0))
    namespace["GUIDED_CBF_USE_PROJECTED_Q"] = (
        not bool(args.disable_guided_projected_q) and namespace["GUIDED_CBF_PROJECTED_Q_WEIGHT"] > 0.0
    )
    namespace["GUIDED_CBF_LAMBDA_BC_FINAL"] = args.guided_lambda_bc_final
    namespace["GUIDED_CBF_LAMBDA_BC_DECAY_STEPS"] = int(max(args.guided_bc_decay_steps, 0))
    if namespace["GUIDED_CBF_USE_PROJECTED_Q"] and namespace["GUIDED_CBF_PROJECTED_Q_WEIGHT"] > 0.0:
        namespace["install_cbf_projection_reporting"]()
    install_safety_set_reward_wrapper(namespace)
    install_event_penalty_env(namespace)

    env_config = env_config_from_args(args, namespace["ENV_CONFIG"])
    if args.force_mtm_congested and active_traffic_model(env_config) == "mtm":
        deep_update(env_config, MTM_CONGESTED_UNCERTAIN_UPDATES)
    traffic_model = active_traffic_model(env_config)

    reward_trial = REWARD_TRIALS[str(args.reward_term)]
    baseline_reward_config = make_baseline_reward_config(
        namespace,
        progress_reward_weight=float(args.progress_reward_weight),
        progress_clip=float(args.progress_clip),
        eps_side=float(args.eps_side),
        reward_trial=reward_trial,
    )
    cbf_reward_config = make_cbf_reward_config(namespace, args, reward_trial=reward_trial)
    eps_tag = str(args.eps_side).replace(".", "p")
    progress_tag = str(args.progress_reward_weight).replace(".", "p").replace("-", "m")
    reward_tag = "sp3" if args.reward_term == "safety" else "oldsp3"
    output_name = f"{reward_tag}_{traffic_model}_e{eps_tag}_p{progress_tag}"
    output_dir = args.output_dir or (Path(namespace["ARTIFACT_DIR"]) / output_name)
    output_dir = output_dir.resolve()
    if os.name == "nt":
        path_probe = output_dir / "ddpg_cbf_reward" / "train_eval_history.csv"
        if len(str(path_probe)) >= 248:
            raise ValueError(
                "The requested artifact directory is too deep for Windows file creation: "
                f"{path_probe}. Choose a shorter --output-dir."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_variants = [item for item in VARIANTS if str(item["variant"]) in INITIAL_VARIANT_NAMES]
    if args.variants is not None:
        valid_variants = {str(item["variant"]) for item in VARIANTS}
        requested_variants = list(dict.fromkeys(str(item) for item in args.variants))
        invalid_variants = sorted(set(requested_variants) - valid_variants)
        if invalid_variants:
            raise ValueError(f"Unknown variants: {invalid_variants}. Valid variants: {sorted(valid_variants)}")
        selected_variants = [item for item in VARIANTS if str(item["variant"]) in requested_variants]
    selected_variants = [dict(item) for item in selected_variants]
    if args.guided_lambda_bc is not None:
        for item in selected_variants:
            if item.get("model_class_name") == "GuidedCBFDDPG":
                item["lambda_bc"] = float(args.guided_lambda_bc)

    run_config = {
        "reward_term": str(args.reward_term),
        "reward_trial": reward_trial,
        "paired_eval": bool(args.paired_eval),
        "timesteps": int(args.timesteps),
        "chunk_timesteps": int(args.chunk_timesteps),
        "traffic_model": traffic_model,
        "env_config": env_config,
        "reward_config": cbf_reward_config,
        "baseline_reward_config": baseline_reward_config,
        "cbf_reward_config": cbf_reward_config,
        "variants": selected_variants,
        "k0": float(args.k0),
        "k1": float(args.k1),
        "eps_side": float(args.eps_side),
        "lambda_norm": float(args.lambda_norm),
        "lambda_event": float(args.lambda_event),
        "raw_violation_reward_weight": float(args.raw_violation_reward_weight),
        "event_threshold": float(args.event_threshold),
        "guided_lambda_bc": None if args.guided_lambda_bc is None else float(args.guided_lambda_bc),
        "guided_lambda_bc_final": None
        if args.guided_lambda_bc_final is None
        else float(args.guided_lambda_bc_final),
        "guided_bc_decay_steps": int(args.guided_bc_decay_steps),
        "guided_projected_q": bool(namespace["GUIDED_CBF_USE_PROJECTED_Q"]),
        "guided_projected_q_weight": float(namespace["GUIDED_CBF_PROJECTED_Q_WEIGHT"]),
        "use_distance_task_eval": bool(not args.legacy_fixed_step_eval),
        "task_distance_m": float(args.task_distance_m),
        "task_max_steps": int(args.task_max_steps),
        "progress_reward_weight": float(args.progress_reward_weight),
        "progress_clip": float(args.progress_clip),
        "tensorboard_enabled": bool(not args.skip_tensorboard),
        "tb_write_freq": int(args.tb_write_freq),
        "tb_flush_freq": int(args.tb_flush_freq),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    print(
        "[safety-potential] starting"
        f" output={output_dir}"
        f" traffic={traffic_model}"
        f" timesteps={args.timesteps:,}"
        f" eps={args.eps_side:g}"
        f" projected_q={namespace['GUIDED_CBF_USE_PROJECTED_Q']}:{namespace['GUIDED_CBF_PROJECTED_Q_WEIGHT']:g}"
        f" raw_violation={args.raw_violation_reward_weight:g}"
        f" task={args.task_distance_m:g}m/{args.task_max_steps}steps"
        f" progress={cbf_reward_config.get('progress_reward_weight', 0.0):g}",
        f" tensorboard={not args.skip_tensorboard}",
        flush=True,
    )

    summaries: list[dict[str, float | str | bool]] = []
    model_paths: dict[str, Path] = {}
    for variant_cfg in selected_variants:
        summary = train_variant(
            namespace,
            args,
            variant_cfg=variant_cfg,
            baseline_reward_config=baseline_reward_config,
            cbf_reward_config=cbf_reward_config,
            env_config=env_config,
            output_dir=output_dir,
        )
        summaries.append(summary)
        model_paths[str(variant_cfg["variant"])] = Path(str(summary["model_path"]))

    summary_frame = pd.DataFrame(summaries)
    summary_path = output_dir / "summary.csv"
    summary_plot_path = output_dir / "summary.png"
    write_csv_resilient(summary_frame, summary_path)
    plot_summary(summary_frame, summary_plot_path)
    print(f"[safety-potential] wrote {summary_path}", flush=True)
    print(f"[safety-potential] wrote {summary_plot_path}", flush=True)
    if not summary_frame.empty:
        display_cols = [
            "variant",
            "return_mean",
            "completion_rate",
            "episode_length_steps_mean",
            "distance_traveled_m_mean",
            "episode_time_s_mean",
            "mean_abs_speed_error",
            "speed_std_mean",
            "mean_lat_y_error_m",
            "ego_collisions_per_km_mean",
            "h_min",
            "boundary_h_min",
            "event_intervention_rate",
            "mean_correction_norm",
            "mean_raw_safe_gap_norm",
            "qp_failure_rate",
            "mean_jerk_norm",
            "action_saturation_rate",
            "mean_neighbor_density_per_km",
            "ego_collisions_mean",
            "mean_safety_potential_cost",
        ]
        display_cols = [column for column in display_cols if column in summary_frame.columns]
        print(summary_frame[display_cols].to_string(index=False), flush=True)

    if args.paired_eval:
        if str(args.reward_term) != "old":
            raise ValueError("--paired-eval is reserved for the old-reward CBF deployment comparison.")
        if "ddpg_cbf_reward" not in model_paths:
            raise ValueError("--paired-eval requires the ddpg_cbf_reward variant so one CBF-trained model is evaluated twice.")
        paired_summary = evaluate_paired_deployments(
            namespace,
            args,
            model_path=model_paths["ddpg_cbf_reward"],
            reward_config=cbf_reward_config,
            env_config=env_config,
            output_dir=output_dir,
        )
        print(
            paired_summary[
                [
                    "evaluation",
                    "return_mean",
                    "completion_rate",
                    "mean_abs_speed_error",
                    "mean_lat_y_error_m",
                    "ego_collisions_mean",
                    "event_intervention_rate",
                    "behavior_score",
                ]
            ].to_string(index=False),
            flush=True,
        )

    required_video_models = {"ddpg", "ddpg_cbf_reward", "guided_ddpg_cbf"}
    if not args.skip_videos and required_video_models.issubset(model_paths):
        video_summary_path = export_videos(
            namespace,
            env_config=env_config,
            output_dir=output_dir,
            model_paths=model_paths,
            args=args,
        )
        print(f"[safety-potential] videos summary {video_summary_path}", flush=True)
    elif not args.skip_videos:
        print("[safety-potential-video] skipped because only a variant subset was trained", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
