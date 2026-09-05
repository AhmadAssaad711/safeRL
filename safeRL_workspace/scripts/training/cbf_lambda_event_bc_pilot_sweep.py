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

from scripts.common.guided_cbf_minimal import install_minimal_guided_cbf
from scripts.common.laneless_script_config import active_traffic_model, add_env_config_args, env_config_from_args


warnings.filterwarnings("ignore", message="OSQP exited.*")

NOTEBOOK_DEPS = [2, 3, 5, 6, 8, 33, 35, 37, 39, 41, 43]

DEFAULT_TRIALS: list[tuple[str, float, float, float]] = [
    ("norm0025_event000_bc000", 0.025, 0.00, 0.00),
    ("norm0010_event002_bc000", 0.010, 0.02, 0.00),
    ("norm0025_event002_bc000", 0.025, 0.02, 0.00),
    ("norm0010_event002_bc001", 0.010, 0.02, 0.01),
    ("norm0025_event002_bc001", 0.025, 0.02, 0.01),
    ("norm0025_event005_bc001", 0.025, 0.05, 0.01),
    ("norm0025_event002_bc003", 0.025, 0.02, 0.03),
]


def set_stable_native_defaults() -> None:
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    ]:
        os.environ.setdefault(key, "1")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return candidate
        nested = candidate / "safeRL_workspace"
        if (nested / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return nested
    raise RuntimeError("Could not find project root containing notebooks/lanelessKaralakou.ipynb")


def exec_notebook_cells(notebook_path: Path, cell_indices: list[int], namespace: dict[str, Any]) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_index in cell_indices:
        cell = notebook["cells"][cell_index]
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        print(f"[cbf_lambda_event_bc_pilot_sweep] executing notebook cell {cell_index}", flush=True)
        exec(compile(source, f"{notebook_path}:cell-{cell_index}", "exec"), namespace)


def install_event_penalty_env(namespace: dict[str, Any]) -> None:
    base_wrapper = namespace["SafetyFilteredAccelerationWrapper"]

    class EventPenaltySafetyFilteredAccelerationWrapper(base_wrapper):  # type: ignore[misc, valid-type]
        """CBF wrapper with separate norm and meaningful-intervention reward penalties."""

        def __init__(
            self,
            *args,
            lambda_event: float = 0.0,
            lambda_raw_violation: float = 0.0,
            intervention_threshold: float = 0.03,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.lambda_event = float(lambda_event)
            self.lambda_raw_violation = float(lambda_raw_violation)
            self.intervention_threshold = float(intervention_threshold)

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            info = dict(info)
            correction_norm = float(info.get("cbf_correction_norm", 0.0))
            meaningful_correction_norm = float(max(correction_norm - self.intervention_threshold, 0.0))
            event_intervened = bool(correction_norm > self.intervention_threshold)
            raw_norm_penalty = float(info.get("cbf_filter_reward_penalty", self.lambda_filter * correction_norm**2))
            norm_penalty = float(self.lambda_filter * meaningful_correction_norm**2)
            event_penalty = float(self.lambda_event * float(event_intervened))
            raw_constraint_violation = float(max(float(info.get("cbf_max_constraint_violation_rl", 0.0)), 0.0))
            raw_violation_penalty = float(self.lambda_raw_violation * raw_constraint_violation**2)
            reward = float(reward) + raw_norm_penalty - norm_penalty - event_penalty - raw_violation_penalty

            raw_action = np.asarray(
                [info.get("cbf_a_rl_x", 0.0), info.get("cbf_a_rl_y", 0.0)],
                dtype=np.float32,
            )
            safe_action = np.asarray(
                [info.get("cbf_a_safe_x", raw_action[0]), info.get("cbf_a_safe_y", raw_action[1])],
                dtype=np.float32,
            )
            qp_success = bool(info.get("cbf_qp_success", True))
            fallback_used = bool(info.get("cbf_fallback_used", not qp_success))
            info.update(
                {
                    "raw_action_phys": raw_action,
                    "safe_action_phys": safe_action,
                    "correction_norm": correction_norm,
                    "meaningful_correction_norm": meaningful_correction_norm,
                    "intervention": event_intervened,
                    "qp_success": qp_success,
                    "fallback_used": fallback_used,
                    "cbf_event_intervened": event_intervened,
                    "cbf_event_intervention_threshold": float(self.intervention_threshold),
                    "cbf_meaningful_correction_norm": meaningful_correction_norm,
                    "cbf_filter_raw_norm_reward_penalty": raw_norm_penalty,
                    "cbf_filter_norm_reward_penalty": norm_penalty,
                    "cbf_filter_event_reward_penalty": event_penalty,
                    "cbf_filter_raw_violation": raw_constraint_violation,
                    "cbf_filter_raw_violation_reward_penalty": raw_violation_penalty,
                    "cbf_filter_reward_penalty": norm_penalty + event_penalty + raw_violation_penalty,
                }
            )
            return obs, reward, terminated, truncated, info

    def make_event_cbf_single_env(
        *,
        seed: int,
        lambda_norm: float,
        lambda_event: float,
        event_threshold: float,
        k0: float,
        k1: float,
        eps_side: float,
        lambda_raw_violation: float = 0.0,
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
        env = EventPenaltySafetyFilteredAccelerationWrapper(
            env,
            lambda_filter=float(lambda_norm),
            lambda_event=float(lambda_event),
            lambda_raw_violation=float(lambda_raw_violation),
            intervention_threshold=float(event_threshold),
            eps_side=float(eps_side),
            k0=float(k0),
            k1=float(k1),
        )
        normalize = namespace["NORMALIZE_RL_OBSERVATIONS"] if normalize_observation is None else normalize_observation
        if normalize:
            env = namespace["LaneFreeObservationNormalizationWrapper"](env, clip=namespace["OBSERVATION_CLIP"])
        if "KPIInfoWrapper" in namespace:
            env = namespace["KPIInfoWrapper"](env, intervention_threshold=float(event_threshold))
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    def make_event_cbf_training_env(
        *,
        seed: int,
        lambda_norm: float,
        lambda_event: float,
        event_threshold: float,
        k0: float,
        k1: float,
        eps_side: float,
        n_envs: int,
        lambda_raw_violation: float = 0.0,
        use_subproc: bool = bool(namespace.get("DDPG_USE_SUBPROC_VEC_ENV", True)),
        env_config: dict[str, Any] | None = None,
        reward_config: dict[str, float] | None = None,
        normalize_observation: bool | None = None,
    ):
        def _single_env(env_seed: int) -> gym.Env:
            return make_event_cbf_single_env(
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
            "EventPenaltySafetyFilteredAccelerationWrapper": EventPenaltySafetyFilteredAccelerationWrapper,
            "make_event_cbf_single_env": make_event_cbf_single_env,
            "make_event_cbf_training_env": make_event_cbf_training_env,
        }
    )


def evaluate_model(
    namespace: dict[str, Any],
    model: Any,
    *,
    episodes: int,
    seed: int,
    k0: float,
    k1: float,
    eps_side: float,
    event_threshold: float,
    env_config: dict[str, Any] | None = None,
    reward_config: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for episode in range(episodes):
        env = namespace["make_event_cbf_single_env"](
            seed=seed + episode,
            lambda_norm=0.0,
            lambda_event=0.0,
            event_threshold=event_threshold,
            k0=k0,
            k1=k1,
            eps_side=eps_side,
            env_config=env_config,
            reward_config=reward_config,
        )
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
        max_corrections: list[float] = []
        meaningful_corrections: list[float] = []
        event_interventions: list[float] = []
        numerical_interventions: list[float] = []
        qp_successes: list[float] = []
        fallbacks: list[float] = []
        min_h_values: list[float] = []
        norm_penalties: list[float] = []
        event_penalties: list[float] = []
        ego_collisions = 0
        ego_collision_steps = 0
        all_collision_events = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            base = env.unwrapped
            speed = float(base.vehicle.vx)
            desired_speed = float(base.vehicle.desired_speed)
            speed_error = speed - desired_speed
            lat_y_error = float(info.get("karalakou_lat_y_error_m", np.nan))
            correction = float(info.get("cbf_correction_norm", 0.0))
            meaningful_correction = float(info.get("cbf_meaningful_correction_norm", max(correction - event_threshold, 0.0)))

            rewards.append(float(reward))
            speeds.append(speed)
            signed_speed_errors.append(speed_error)
            abs_speed_errors.append(abs(speed_error))
            if np.isfinite(lat_y_error):
                lat_y_errors.append(lat_y_error)
            corrections.append(correction)
            max_corrections.append(correction)
            meaningful_corrections.append(meaningful_correction)
            event_interventions.append(float(info.get("cbf_event_intervened", correction > event_threshold)))
            numerical_interventions.append(float(info.get("cbf_intervened", correction > 1e-6)))
            qp_successes.append(float(info.get("cbf_qp_success", True)))
            fallbacks.append(float(info.get("cbf_fallback_used", False)))
            min_h_values.append(float(info.get("cbf_min_h", np.nan)))
            norm_penalties.append(float(info.get("cbf_filter_norm_reward_penalty", 0.0)))
            event_penalties.append(float(info.get("cbf_filter_event_reward_penalty", 0.0)))
            all_collision_events += int(info.get("collisions", 0))
            ego_collisions += int(info.get("ego_collision_events", 0))
            if bool(info.get("ego_collision", False)):
                ego_collision_steps += 1
            step_count += 1
            done = bool(terminated or truncated)

        rows.append(
            {
                "episode": float(episode),
                "steps": float(step_count),
                "return": float(np.sum(rewards)),
                "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
                "mean_signed_speed_error": float(np.mean(signed_speed_errors)) if signed_speed_errors else 0.0,
                "mean_abs_speed_error": float(np.mean(abs_speed_errors)) if abs_speed_errors else 0.0,
                "mean_lat_y_error_m": float(np.mean(lat_y_errors)) if lat_y_errors else np.nan,
                "mean_correction_norm": float(np.mean(corrections)) if corrections else 0.0,
                "max_correction_norm": float(np.max(max_corrections)) if max_corrections else 0.0,
                "mean_meaningful_correction_norm": float(np.mean(meaningful_corrections))
                if meaningful_corrections
                else 0.0,
                "max_meaningful_correction_norm": float(np.max(meaningful_corrections))
                if meaningful_corrections
                else 0.0,
                "event_intervention_rate": float(np.mean(event_interventions)) if event_interventions else 0.0,
                "numerical_intervention_rate": float(np.mean(numerical_interventions)) if numerical_interventions else 0.0,
                "qp_failure_rate": float(1.0 - np.mean(qp_successes)) if qp_successes else 0.0,
                "fallback_rate": float(np.mean(fallbacks)) if fallbacks else 0.0,
                "min_h": float(np.nanmin(min_h_values)) if min_h_values and not np.all(np.isnan(min_h_values)) else np.nan,
                "ego_collisions": float(ego_collisions),
                "ego_collision_steps": float(ego_collision_steps),
                "total_collision_events": float(all_collision_events),
                "mean_norm_reward_penalty": float(np.mean(norm_penalties)) if norm_penalties else 0.0,
                "mean_event_reward_penalty": float(np.mean(event_penalties)) if event_penalties else 0.0,
            }
        )
        env.close()
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "episodes": float(len(metrics)),
        "steps_mean": float(metrics["steps"].mean()),
        "return_mean": float(metrics["return"].mean()),
        "return_std": float(metrics["return"].std()),
        "mean_speed": float(metrics["mean_speed"].mean()),
        "mean_signed_speed_error": float(metrics["mean_signed_speed_error"].mean()),
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
        "event_intervention_rate": float(metrics["event_intervention_rate"].mean()),
        "numerical_intervention_rate": float(metrics["numerical_intervention_rate"].mean()),
        "qp_failure_rate": float(metrics["qp_failure_rate"].mean()),
        "fallback_rate": float(metrics["fallback_rate"].mean()),
        "min_h": float(metrics["min_h"].min()),
        "ego_collisions_mean": float(metrics["ego_collisions"].mean()),
        "ego_collision_steps_mean": float(metrics["ego_collision_steps"].mean()),
        "total_collision_events_mean": float(metrics["total_collision_events"].mean()),
    }


class PilotEvalCallback(BaseCallback):
    def __init__(
        self,
        namespace: dict[str, Any],
        *,
        trial_name: str,
        lambda_norm: float,
        lambda_event: float,
        lambda_bc: float,
        event_threshold: float,
        k0: float,
        k1: float,
        eps_side: float,
        env_config: dict[str, Any] | None,
        reward_config: dict[str, float] | None,
        eval_freq: int,
        episodes: int,
        seed: int,
    ) -> None:
        super().__init__(verbose=1)
        self.namespace = namespace
        self.trial_name = trial_name
        self.lambda_norm = float(lambda_norm)
        self.lambda_event = float(lambda_event)
        self.lambda_bc = float(lambda_bc)
        self.event_threshold = float(event_threshold)
        self.k0 = float(k0)
        self.k1 = float(k1)
        self.eps_side = float(eps_side)
        self.env_config = env_config
        self.reward_config = reward_config
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
            k0=self.k0,
            k1=self.k1,
            eps_side=self.eps_side,
            event_threshold=self.event_threshold,
            env_config=self.env_config,
            reward_config=self.reward_config,
        )
        row: dict[str, float | str] = {
            "trial_name": self.trial_name,
            "lambda_norm": self.lambda_norm,
            "lambda_event": self.lambda_event,
            "lambda_bc": self.lambda_bc,
            "event_threshold": self.event_threshold,
            "timesteps": float(self.num_timesteps),
            **summarize(metrics),
        }
        self.records.append(row)
        print(
            "[pilot-eval]"
            f" {self.trial_name}"
            f" steps={self.num_timesteps:,}"
            f" return={row['return_mean']:.2f}"
            f" abs_speed={row['mean_abs_speed_error']:.3f}"
            f" event_int={row['event_intervention_rate']:.2%}"
            f" corr={row['mean_correction_norm']:.3f}"
            f" qp_fail={row['qp_failure_rate']:.2%}",
            flush=True,
        )
        return True


def trial_config_rows() -> list[dict[str, float | str]]:
    return [
        {
            "trial_name": name,
            "lambda_norm": lambda_norm,
            "lambda_event": lambda_event,
            "lambda_bc": lambda_bc,
        }
        for name, lambda_norm, lambda_event, lambda_bc in DEFAULT_TRIALS
    ]


def plot_trial_history(history: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    correction_column = (
        "mean_meaningful_correction_norm" if "mean_meaningful_correction_norm" in history.columns else "mean_correction_norm"
    )
    if not history.empty:
        axes[0].plot(history["timesteps"], history["return_mean"], marker="o")
        axes[1].plot(history["timesteps"], history["event_intervention_rate"], marker="o")
        axes[2].plot(history["timesteps"], history[correction_column], marker="o")
    axes[0].set_title("Eval Return")
    axes[1].set_title("Meaningful Intervention Rate")
    axes[2].set_title("Meaningful Correction")
    axes[0].set_ylabel("Return")
    axes[1].set_ylabel("Rate")
    axes[2].set_ylabel("Correction")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
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
    ranked = summary.sort_values("selection_score", ascending=False).reset_index(drop=True)
    labels = ranked["trial_name"].tolist()
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4.5, 0.48 * len(labels))))
    correction_column = (
        "mean_meaningful_correction_norm" if "mean_meaningful_correction_norm" in ranked.columns else "mean_correction_norm"
    )
    panels = [
        ("return_mean", "Return", False),
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
    fig.suptitle("CBF Lambda/Event/BC Pilot Sweep", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def score_summary(row: dict[str, float | str]) -> float:
    collision_penalty = 450.0 * float(row["ego_collisions_mean"])
    qp_penalty = 900.0 * float(row["qp_failure_rate"])
    intervention_penalty = 90.0 * float(row["event_intervention_rate"])
    correction_penalty = 65.0 * float(row.get("mean_meaningful_correction_norm", row["mean_correction_norm"]))
    speed_penalty = 25.0 * float(row["mean_abs_speed_error"])
    return float(row["return_mean"]) - collision_penalty - qp_penalty - intervention_penalty - correction_penalty - speed_penalty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pilot sweep for CBF norm/event reward terms and guided BC lambda.")
    parser.add_argument("--k0", type=float, default=5.29)
    parser.add_argument("--k1", type=float, default=3.68)
    parser.add_argument("--eps-side", type=float, default=0.149)
    parser.add_argument("--event-threshold", type=float, default=0.03)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--train-eval-freq", type=int, default=5_000)
    parser.add_argument("--train-eval-episodes", type=int, default=2)
    parser.add_argument("--final-eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=511_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-envs", type=int, default=1)
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
    namespace["DEVICE"] = args.device
    install_minimal_guided_cbf(namespace)
    install_event_penalty_env(namespace)
    env_config = env_config_from_args(args, namespace["ENV_CONFIG"])
    traffic_model = active_traffic_model(env_config)

    default_output_name = "cbf_lambda_event_bc_pilot_mtm" if traffic_model == "mtm" else "cbf_lambda_event_bc_pilot"
    output_dir = args.output_dir or (Path(namespace["ARTIFACT_DIR"]) / default_output_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trial_configs.csv").write_text(pd.DataFrame(trial_config_rows()).to_csv(index=False), encoding="utf-8")

    print(
        "[pilot] starting sweep",
        {
            "trials": len(DEFAULT_TRIALS),
            "timesteps": args.timesteps,
            "final_eval_episodes": args.final_eval_episodes,
            "event_threshold": args.event_threshold,
            "traffic_model": traffic_model,
            "k0": args.k0,
            "k1": args.k1,
            "eps_side": args.eps_side,
            "output_dir": str(output_dir),
        },
        flush=True,
    )

    final_rows: list[dict[str, float | str]] = []
    for index, (trial_name, lambda_norm, lambda_event, lambda_bc) in enumerate(DEFAULT_TRIALS, start=1):
        trial_dir = output_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        model_path = trial_dir / "model.zip"
        history_path = trial_dir / "train_eval_history.csv"
        final_episodes_path = trial_dir / "final_eval_episodes.csv"
        final_summary_path = trial_dir / "final_summary.csv"
        plot_path = trial_dir / "train_eval_history.png"

        if not args.no_resume and final_summary_path.exists() and model_path.exists():
            print(f"[pilot] [{index}/{len(DEFAULT_TRIALS)}] {trial_name} complete; loading summary", flush=True)
            final_rows.append(pd.read_csv(final_summary_path).iloc[0].to_dict())
            continue

        print(
            f"[pilot] [{index}/{len(DEFAULT_TRIALS)}] {trial_name}"
            f" lambda_norm={lambda_norm:g}"
            f" lambda_event={lambda_event:g}"
            f" lambda_bc={lambda_bc:g}",
            flush=True,
        )
        train_env = namespace["make_event_cbf_training_env"](
            seed=args.seed + index * 1_000,
            lambda_norm=lambda_norm,
            lambda_event=lambda_event,
            event_threshold=args.event_threshold,
            k0=args.k0,
            k1=args.k1,
            eps_side=args.eps_side,
            n_envs=args.n_envs,
            use_subproc=bool(namespace.get("DDPG_USE_SUBPROC_VEC_ENV", True)),
            env_config=env_config,
        )
        print(
            f"[pilot] vec_env={type(train_env).__name__} | n_envs={args.n_envs} | UTD=1",
            flush=True,
        )
        n_actions = train_env.action_space.shape[-1]
        action_noise = namespace["make_ou_action_noise"](n_actions, n_envs=args.n_envs)
        callback = PilotEvalCallback(
            namespace,
            trial_name=trial_name,
            lambda_norm=lambda_norm,
            lambda_event=lambda_event,
            lambda_bc=lambda_bc,
            event_threshold=args.event_threshold,
            k0=args.k0,
            k1=args.k1,
            eps_side=args.eps_side,
            env_config=env_config,
            reward_config=None,
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
            lambda_bc=float(lambda_bc),
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
        plot_trial_history(history, plot_path, title=trial_name)

        print(f"[pilot] evaluating {trial_name}", flush=True)
        final_metrics = evaluate_model(
            namespace,
            model,
            episodes=args.final_eval_episodes,
            seed=args.seed + index * 100_000,
            k0=args.k0,
            k1=args.k1,
            eps_side=args.eps_side,
            event_threshold=args.event_threshold,
            env_config=env_config,
            reward_config=None,
        )
        final_metrics.to_csv(final_episodes_path, index=False)
        summary = {
            "trial_name": trial_name,
            "model_path": str(model_path),
            "elapsed_sec": float(elapsed_sec),
            "timesteps": float(args.timesteps),
            "k0": float(args.k0),
            "k1": float(args.k1),
            "eps_side": float(args.eps_side),
            "event_threshold": float(args.event_threshold),
            "traffic_model": traffic_model,
            "lambda_norm": float(lambda_norm),
            "lambda_event": float(lambda_event),
            "lambda_bc": float(lambda_bc),
            **summarize(final_metrics),
        }
        summary["selection_score"] = score_summary(summary)
        pd.DataFrame([summary]).to_csv(final_summary_path, index=False)
        (trial_dir / "run_config.json").write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        final_rows.append(summary)
        print(
            "[pilot-result]"
            f" {trial_name}"
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
    aggregate_path = output_dir / "pilot_final_summary.csv"
    aggregate_plot_path = output_dir / "pilot_comparison.png"
    aggregate.to_csv(aggregate_path, index=False)
    plot_aggregate(aggregate, aggregate_plot_path)
    print(f"[pilot] wrote {aggregate_path}", flush=True)
    print(f"[pilot] wrote {aggregate_plot_path}", flush=True)
    if not aggregate.empty:
        display_cols = [
            "trial_name",
            "lambda_norm",
            "lambda_event",
            "lambda_bc",
            "traffic_model",
            "return_mean",
            "mean_abs_speed_error",
            "event_intervention_rate",
            "mean_correction_norm",
            "qp_failure_rate",
            "ego_collisions_mean",
            "selection_score",
        ]
        display_cols = [c for c in display_cols if c in aggregate.columns]
        print(aggregate[display_cols].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
