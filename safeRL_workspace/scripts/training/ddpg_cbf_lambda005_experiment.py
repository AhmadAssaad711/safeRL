from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv


NOTEBOOK_PATH = Path(__file__).resolve().parents[2] / "notebooks" / "lanelessKaralakou.ipynb"
PAPER_EPISODE_STEPS = 800
TRAINING_EPISODES = 625
TOTAL_TIMESTEPS = TRAINING_EPISODES * PAPER_EPISODE_STEPS
DDPG_CBF_TOTAL_TIMESTEPS = 50_000
EVAL_EVERY = PAPER_EPISODE_STEPS
EVAL_EPISODES = 5
FINAL_EVAL_EPISODES = 50
PAPER_WINDOW_STEPS = PAPER_EPISODE_STEPS
DDPG_REPLAY_MEMORY = 100_000
DDPG_BATCH_SIZE = 64
DDPG_TAU = 0.001
DDPG_GAMMA = 0.98
DDPG_LEARNING_RATE = 0.001
DDPG_LEARNING_STARTS = 1_000
DDPG_OU_SIGMA = 0.1
LAMBDA_FILTER = 0.05
CBF_TARGET_PAIR_DY = 3.0
CBF_NEIGHBOR_RANGE = 60.0
CBF_K0 = 4.0
CBF_K1 = 4.0
MEANINGFUL_INTERVENTION_TOL = 1e-2


def _load_notebook_context() -> dict[str, Any]:
    """Reuse the notebook's environment, reward, and baseline evaluation code."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {}
    required_markers = [
        "def find_project_root",
        "class KaralakouRewardWrapper",
        "ENV_CONFIG =",
        "def make_single_env",
        "def evaluate_policy_with_metrics",
        "from qpsolvers import solve_qp",
        "def cbf_filter_2d",
        "def _lane_free_base",
        "class SafetyFilteredAccelerationWrapper",
    ]

    for marker in required_markers:
        for cell in notebook["cells"]:
            source = "".join(cell.get("source", []))
            if marker not in source:
                continue
            if marker == "def make_single_env":
                source = source.split("smoke_env = make_single_env")[0]
            exec(compile(source, f"{NOTEBOOK_PATH.name}:{marker}", "exec"), namespace)
            break
        else:
            raise RuntimeError(f"Could not find notebook cell containing {marker!r}.")

    return namespace


def make_cbf_single_env(ns: dict[str, Any], seed: int, lambda_filter: float = LAMBDA_FILTER) -> gym.Env:
    env = gym.make("lane-free-v0", render_mode=None, config=ns["ENV_CONFIG"])
    env = ns["KaralakouRewardWrapper"](env, reward_config=ns["REWARD_CONFIG"])
    env = ns["SafetyFilteredAccelerationWrapper"](
        env,
        lambda_filter=lambda_filter,
        neighbor_range=CBF_NEIGHBOR_RANGE,
        eps_side=float(ns.get("CBF_EPS_SIDE", 0.0)),
        k0=CBF_K0,
        k1=CBF_K1,
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def make_cbf_training_env(ns: dict[str, Any], seed: int, lambda_filter: float = LAMBDA_FILTER) -> DummyVecEnv:
    def _init() -> gym.Env:
        return make_cbf_single_env(ns, seed=seed, lambda_filter=lambda_filter)

    return DummyVecEnv([_init])


def configure_paper_evaluation_env(env: gym.Env, steps: int = PAPER_WINDOW_STEPS) -> None:
    """Run paper-style fixed-length evaluation without ending on ego collisions."""
    base = env.unwrapped
    if hasattr(base, "config"):
        base.config["episode_steps"] = int(steps)
        base.config["duration"] = int(steps)
        base.config["terminate_on_collision"] = False


def evaluate_policy_fixed_800_step_windows(
    model: Any,
    make_env,
    windows: int,
    seed: int,
    algorithm: str,
    deterministic: bool = True,
    include_cbf_metrics: bool = False,
) -> pd.DataFrame:
    """Evaluate fixed 800-step episodes while logging collision rewards/events."""
    rows: list[dict[str, float]] = []

    for window in range(windows):
        env = make_env(seed + window)
        configure_paper_evaluation_env(env, steps=PAPER_WINDOW_STEPS)
        obs, _ = env.reset(seed=seed + window)
        rewards: list[float] = []
        signed_deviations: list[float] = []
        abs_deviations: list[float] = []
        corrections: list[float] = []
        meaningful_interventions: list[float] = []
        qp_successes: list[float] = []
        min_h_values: list[float] = []
        ego_collisions = 0
        ego_collision_steps = 0
        total_collision_events = 0

        for step_count in range(PAPER_WINDOW_STEPS):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            base = env.unwrapped
            desired = float(base.vehicle.desired_speed)
            speed = float(base.vehicle.vx)
            deviation = speed - desired
            correction = float(info.get("cbf_correction_norm", 0.0))

            rewards.append(float(reward))
            signed_deviations.append(deviation)
            abs_deviations.append(abs(deviation))
            corrections.append(correction)
            meaningful_interventions.append(float(correction > MEANINGFUL_INTERVENTION_TOL))
            qp_successes.append(float(info.get("cbf_qp_success", True)))
            min_h_values.append(float(info.get("cbf_min_h", np.nan)))
            total_collision_events += int(info.get("collisions", 0))
            ego_collisions += int(info.get("ego_collision_events", 0))
            if bool(info.get("ego_collision", False)):
                ego_collision_steps += 1

        row = {
            "algorithm": algorithm,
            "window": float(window),
            "steps": float(PAPER_WINDOW_STEPS),
            "resets_inside_window": 0.0,
            "average_reward": float(np.mean(rewards)) if rewards else 0.0,
            "return": float(np.sum(rewards)),
            "mean_signed_speed_deviation": float(np.mean(signed_deviations)) if signed_deviations else 0.0,
            "mean_abs_speed_deviation": float(np.mean(abs_deviations)) if abs_deviations else 0.0,
            "average_ego_collisions_per_800_steps": float(ego_collisions),
            "ego_collision_steps_per_800_steps": float(ego_collision_steps),
            "total_collision_events_per_800_steps": float(total_collision_events),
        }
        if include_cbf_metrics:
            row.update(
                {
                    "mean_correction_norm": float(np.mean(corrections)) if corrections else 0.0,
                    "max_correction_norm": float(np.max(corrections)) if corrections else 0.0,
                    "meaningful_intervention_rate": float(np.mean(meaningful_interventions))
                    if meaningful_interventions
                    else 0.0,
                    "qp_failure_rate": float(1.0 - np.mean(qp_successes)) if qp_successes else 0.0,
                    "min_h": float(np.nanmin(min_h_values))
                    if min_h_values and not np.all(np.isnan(min_h_values))
                    else np.nan,
                }
            )
        rows.append(row)
        env.close()

    return pd.DataFrame(rows)


def evaluate_cbf_policy_with_paper_metrics(
    ns: dict[str, Any],
    model: Any,
    windows: int,
    seed: int,
    deterministic: bool = True,
    lambda_filter: float = LAMBDA_FILTER,
) -> pd.DataFrame:
    return evaluate_policy_fixed_800_step_windows(
        model,
        make_env=lambda env_seed: make_cbf_single_env(ns, seed=env_seed, lambda_filter=lambda_filter),
        windows=windows,
        seed=seed,
        algorithm="DDPG-CBF lambda=0.05",
        deterministic=deterministic,
        include_cbf_metrics=True,
    )


class CBFPaperMetricsCallback(BaseCallback):
    def __init__(self, ns: dict[str, Any], eval_freq: int, n_eval_episodes: int, seed: int) -> None:
        super().__init__(verbose=1)
        self.ns = ns
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.seed = int(seed)
        self.records: list[dict[str, float]] = []
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True

        self._last_eval_step = self.num_timesteps
        metrics = evaluate_cbf_policy_with_paper_metrics(
            self.ns,
            self.model,
            windows=self.n_eval_episodes,
            seed=self.seed + self.num_timesteps,
            deterministic=True,
            lambda_filter=LAMBDA_FILTER,
        )
        row = {
            "timesteps": float(self.num_timesteps),
            "average_reward": float(metrics["average_reward"].mean()),
            "return": float(metrics["return"].mean()),
            "mean_signed_speed_deviation": float(metrics["mean_signed_speed_deviation"].mean()),
            "mean_abs_speed_deviation": float(metrics["mean_abs_speed_deviation"].mean()),
            "average_ego_collisions_per_800_steps": float(
                metrics["average_ego_collisions_per_800_steps"].mean()
            ),
            "total_collision_events_per_800_steps": float(
                metrics["total_collision_events_per_800_steps"].mean()
            ),
            "average_resets_inside_800_step_window": float(metrics["resets_inside_window"].mean()),
            "mean_correction_norm": float(metrics["mean_correction_norm"].mean()),
            "meaningful_intervention_rate": float(metrics["meaningful_intervention_rate"].mean()),
            "qp_failure_rate": float(metrics["qp_failure_rate"].mean()),
            "min_h": float(metrics["min_h"].min()),
        }
        self.records.append(row)
        print(
            f"steps={self.num_timesteps:,} | "
            f"avg reward={row['average_reward']:.3f} | "
            f"collisions/800={row['average_ego_collisions_per_800_steps']:.2f} | "
            f"speed dev={row['mean_signed_speed_deviation']:.3f} m/s | "
            f"lambda={LAMBDA_FILTER:.3f}"
        )
        return True


def rename_ddpg_history(ddpg_history: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "algorithm": "DDPG",
            "timesteps": ddpg_history["timesteps"],
            "average_reward": ddpg_history["return"] / PAPER_WINDOW_STEPS,
            "return": ddpg_history["return"],
            "mean_signed_speed_deviation": ddpg_history["mean_signed_speed_deviation"],
            "mean_abs_speed_deviation": ddpg_history["mean_abs_speed_deviation"],
            "average_ego_collisions_per_800_steps": ddpg_history["ego_collisions"],
            "total_collision_events_per_800_steps": ddpg_history["total_collision_events"],
        }
    )


def add_algorithm(history: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    result = history.copy()
    result.insert(0, "algorithm", algorithm)
    return result


def evaluate_ddpg_final(ns: dict[str, Any], model: Any, windows: int, seed: int) -> pd.DataFrame:
    return evaluate_policy_fixed_800_step_windows(
        model,
        make_env=lambda env_seed: ns["make_single_env"](seed=env_seed, render_mode=None),
        windows=windows,
        seed=seed,
        algorithm="DDPG",
        deterministic=True,
        include_cbf_metrics=False,
    )


def plot_paper_metrics(combined_history: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    for algorithm, group in combined_history.groupby("algorithm"):
        axes[0].plot(group["timesteps"], group["average_reward"], marker="o", label=algorithm)
        axes[1].plot(
            group["timesteps"],
            group["average_ego_collisions_per_800_steps"],
            marker="o",
            label=algorithm,
        )
        axes[2].plot(group["timesteps"], group["mean_signed_speed_deviation"], marker="o", label=algorithm)

    axes[0].set_title("Average reward")
    axes[0].set_xlabel("Training timesteps")
    axes[0].set_ylabel("Mean reward per step")

    axes[1].set_title("Average collisions")
    axes[1].set_xlabel("Training timesteps")
    axes[1].set_ylabel("Ego collisions / 800 steps")

    axes[2].set_title("Speed deviation")
    axes[2].set_xlabel("Training timesteps")
    axes[2].set_ylabel("vx - vd (m/s)")
    axes[2].axhline(0.0, color="black", linewidth=1.0, alpha=0.4)

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ns = _load_notebook_context()
    artifact_dir = ns["ARTIFACT_DIR"]
    seed = int(ns["SEED"])
    device = ns["DEVICE"]
    ddpg_model_path = ns["DDPG_MODEL_PATH"]
    ddpg_history_path = ns["DDPG_HISTORY_PATH"]
    ddpg_cbf_model_path = artifact_dir / "ddpg_cbf_lambda005_laneless_karalakou.zip"
    ddpg_cbf_history_path = artifact_dir / "ddpg_cbf_lambda005_laneless_karalakou_eval_history.csv"
    combined_history_path = artifact_dir / "ddpg_vs_ddpg_cbf_lambda005_paper_training_metrics.csv"
    final_episode_path = artifact_dir / "ddpg_vs_ddpg_cbf_lambda005_final_episode_metrics.csv"
    final_summary_path = artifact_dir / "ddpg_vs_ddpg_cbf_lambda005_final_summary.csv"
    final_paper_summary_path = artifact_dir / "ddpg_vs_ddpg_cbf_lambda005_exact_paper_summary.csv"
    plot_path = artifact_dir / "ddpg_vs_ddpg_cbf_lambda005_paper_metrics.png"

    print("Experiment parameters:")
    print(f"  lambda_filter={LAMBDA_FILTER}")
    cbf_target_dy = float(ns.get("CBF_TARGET_PAIR_DY", CBF_TARGET_PAIR_DY))
    cbf_eps_side = float(ns.get("CBF_EPS_SIDE", 0.0))
    print(
        f"  CBF target_pair_Dy={cbf_target_dy}, eps_side={cbf_eps_side:.4f}, "
        f"range={CBF_NEIGHBOR_RANGE}, k0={CBF_K0}, k1={CBF_K1}"
    )
    print(f"  ddpg_cbf_total_timesteps={DDPG_CBF_TOTAL_TIMESTEPS}")
    print(f"  device={device}")

    train_env = make_cbf_training_env(ns, seed=seed, lambda_filter=LAMBDA_FILTER)
    n_actions = train_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions, dtype=np.float32),
        sigma=DDPG_OU_SIGMA * np.ones(n_actions, dtype=np.float32),
    )
    callback = CBFPaperMetricsCallback(
        ns,
        eval_freq=EVAL_EVERY,
        n_eval_episodes=EVAL_EPISODES,
        seed=seed + 70_000,
    )

    model = ns["DDPG"](
        "MlpPolicy",
        train_env,
        learning_rate=DDPG_LEARNING_RATE,
        buffer_size=DDPG_REPLAY_MEMORY,
        learning_starts=DDPG_LEARNING_STARTS,
        batch_size=DDPG_BATCH_SIZE,
        tau=DDPG_TAU,
        gamma=DDPG_GAMMA,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
        policy_kwargs={"net_arch": [256, 128]},
        tensorboard_log=str(artifact_dir / "tensorboard"),
        verbose=1,
        seed=seed,
        device=device,
    )

    start_time = time.time()
    model.learn(total_timesteps=DDPG_CBF_TOTAL_TIMESTEPS, callback=callback)
    elapsed = time.time() - start_time
    model.save(str(ddpg_cbf_model_path))
    train_env.close()

    cbf_history = pd.DataFrame(callback.records)
    cbf_history.to_csv(ddpg_cbf_history_path, index=False)
    print(f"Saved retrained DDPG-CBF model to {ddpg_cbf_model_path}")
    print(f"Saved retrained DDPG-CBF history to {ddpg_cbf_history_path}")
    print(f"Training time: {elapsed / 60:.1f} min")

    if not ddpg_model_path.exists():
        raise FileNotFoundError(f"Normal DDPG model not found: {ddpg_model_path}")
    if not ddpg_history_path.exists():
        raise FileNotFoundError(f"Normal DDPG history not found: {ddpg_history_path}")

    ddpg_history = pd.read_csv(ddpg_history_path)
    combined_history = pd.concat(
        [
            rename_ddpg_history(ddpg_history),
            add_algorithm(cbf_history, "DDPG-CBF lambda=0.05"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined_history.to_csv(combined_history_path, index=False)
    plot_paper_metrics(combined_history, plot_path)

    ddpg_model = ns["DDPG"].load(str(ddpg_model_path), device=device)
    cbf_model = ns["DDPG"].load(str(ddpg_cbf_model_path), device=device)
    eval_seed = seed + 90_000
    ddpg_final = evaluate_ddpg_final(ns, ddpg_model, windows=FINAL_EVAL_EPISODES, seed=eval_seed)
    cbf_final = evaluate_cbf_policy_with_paper_metrics(
        ns,
        cbf_model,
        windows=FINAL_EVAL_EPISODES,
        seed=eval_seed,
        deterministic=True,
        lambda_filter=LAMBDA_FILTER,
    )

    final_episode_metrics = pd.concat([ddpg_final, cbf_final], ignore_index=True, sort=False)
    final_episode_metrics.to_csv(final_episode_path, index=False)

    final_summary = final_episode_metrics.groupby("algorithm").mean(numeric_only=True).drop(columns=["window"])
    final_summary.to_csv(final_summary_path)
    final_paper_summary = final_episode_metrics.groupby("algorithm").agg(
        average_reward=("average_reward", "mean"),
        speed_deviation_mps=("mean_signed_speed_deviation", "mean"),
        average_ego_collisions_per_800_steps=("average_ego_collisions_per_800_steps", "mean"),
    )
    final_paper_summary.to_csv(final_paper_summary_path)

    print(f"Saved combined training metrics to {combined_history_path}")
    print(f"Saved final episode metrics to {final_episode_path}")
    print(f"Saved final summary to {final_summary_path}")
    print(f"Saved exact paper summary to {final_paper_summary_path}")
    print(f"Saved paper-style plot to {plot_path}")
    print("\nFinal 20-episode summary:")
    print(final_summary.to_string())
    print("\nExact paper metrics:")
    print(final_paper_summary.to_string())


if __name__ == "__main__":
    sys.exit(main())
