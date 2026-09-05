from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise


TRAINING_TIMESTEPS = int(os.environ.get("LANELESS_DDPG_COMPARE_TIMESTEPS", "50000"))
LEARNING_STARTS = int(os.environ.get("LANELESS_DDPG_COMPARE_LEARNING_STARTS", "1000"))
REPLAY_MEMORY = int(os.environ.get("LANELESS_DDPG_COMPARE_REPLAY_MEMORY", "100000"))
LOG_EVERY_STEPS = int(os.environ.get("LANELESS_DDPG_COMPARE_LOG_EVERY", "1000"))
ROLLING_WINDOW = int(os.environ.get("LANELESS_DDPG_COMPARE_ROLLING_WINDOW", "500"))
RUN_TAG = os.environ.get("LANELESS_DDPG_COMPARE_TAG")


warnings.filterwarnings("ignore", message="OSQP exited.*")
warnings.filterwarnings("ignore", message="Clarabel.rs terminated.*")


def find_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    for candidate in [script_path.parent, *script_path.parents]:
        notebook = candidate / "notebooks" / "lanelessKaralakou.ipynb"
        env_file = candidate / "laneless highway env" / "lane_free_env.py"
        if notebook.exists() and env_file.exists():
            return candidate
    raise RuntimeError("Could not find repo root containing notebooks/lanelessKaralakou.ipynb")


def load_notebook_namespace(repo_root: Path) -> dict[str, Any]:
    notebook_path = repo_root / "notebooks" / "lanelessKaralakou.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {}

    required_prefixes = [
        "from __future__ import annotations",
        "class KaralakouRewardWrapper",
        "ENV_CONFIG = {",
        "class LaneFreeObservationNormalizationWrapper",
        "try:\n    from qpsolvers import solve_qp",
        "CBF_AX_BOUNDS =",
        "def _lane_free_base",
        "class SafetyFilteredAccelerationWrapper",
        "# Tuned DDPG-CBF shield overrides",
        "def evaluate_cbf_policy_with_metrics",
    ]

    for prefix in required_prefixes:
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if source.startswith(prefix):
                exec(compile(source, f"{notebook_path}:cell_{index}", "exec"), namespace)
                break
        else:
            raise RuntimeError(f"Could not find notebook cell starting with {prefix!r}")

    return namespace


def first_float(info: dict[str, Any], keys: Iterable[str], default: float = np.nan) -> float:
    for key in keys:
        if key in info:
            value = info[key]
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
    return default


def first_bool(info: dict[str, Any], keys: Iterable[str], default: bool = False) -> bool:
    for key in keys:
        if key in info:
            return bool(info[key])
    return default


class StepwiseTrainingLogger(BaseCallback):
    def __init__(self, algorithm: str, log_every_steps: int = LOG_EVERY_STEPS) -> None:
        super().__init__(verbose=0)
        self.algorithm = algorithm
        self.log_every_steps = int(log_every_steps)
        self.started_at = time.perf_counter()
        self.current_episode = 0
        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_cbf_corrections: list[float] = []
        self.episode_cbf_interventions: list[float] = []
        self.step_records: list[dict[str, float | str]] = []
        self.episode_records: list[dict[str, float | str]] = []

    def _on_training_start(self) -> None:
        self.started_at = time.perf_counter()

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", []), dtype=float)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool)
        infos = self.locals.get("infos", [])
        if len(rewards) != 1:
            raise RuntimeError("This logger expects a single-env DummyVecEnv so every row is one real timestep.")

        reward = float(rewards[0])
        done = bool(dones[0])
        info = infos[0] if infos else {}
        elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        correction_norm = first_float(info, ["cbf_correction_norm", "correction_norm"], default=0.0)
        intervention = first_bool(info, ["cbf_intervened", "intervention"], default=correction_norm > 1e-2)
        qp_success = first_bool(info, ["cbf_qp_success", "qp_success"], default=True)

        self.episode_return += reward
        self.episode_length += 1
        self.episode_cbf_corrections.append(correction_norm)
        self.episode_cbf_interventions.append(float(intervention))

        completed_return = np.nan
        completed_length = np.nan
        if done:
            episode_info = info.get("episode", {})
            completed_return = float(episode_info.get("r", self.episode_return))
            completed_length = float(episode_info.get("l", self.episode_length))

        self.step_records.append(
            {
                "algorithm": self.algorithm,
                "timestep": float(self.num_timesteps),
                "episode": float(self.current_episode),
                "episode_step": float(self.episode_length),
                "reward": reward,
                "episode_return_so_far": float(self.episode_return),
                "episode_length_so_far": float(self.episode_length),
                "completed_episode_return": completed_return,
                "completed_episode_length": completed_length,
                "done": float(done),
                "elapsed_sec": float(elapsed),
                "timesteps_per_sec": float(self.num_timesteps / elapsed),
                "cbf_correction_norm": correction_norm,
                "cbf_intervention": float(intervention),
                "cbf_qp_success": float(qp_success),
                "cbf_min_h": first_float(info, ["cbf_min_h", "min_h"], default=np.nan),
                "ego_collision": float(first_bool(info, ["ego_collision"], default=False)),
                "ego_collision_events": first_float(info, ["ego_collision_events"], default=0.0),
                "total_collision_events": first_float(info, ["collisions"], default=0.0),
            }
        )

        if done:
            episode_elapsed = elapsed
            self.episode_records.append(
                {
                    "algorithm": self.algorithm,
                    "episode": float(self.current_episode),
                    "end_timestep": float(self.num_timesteps),
                    "length": completed_length,
                    "return": completed_return,
                    "elapsed_sec": float(episode_elapsed),
                    "timesteps_per_sec": float(self.num_timesteps / episode_elapsed),
                    "mean_cbf_correction_norm": float(np.mean(self.episode_cbf_corrections)),
                    "cbf_intervention_rate": float(np.mean(self.episode_cbf_interventions)),
                }
            )
            self.current_episode += 1
            self.episode_return = 0.0
            self.episode_length = 0
            self.episode_cbf_corrections.clear()
            self.episode_cbf_interventions.clear()

        if self.num_timesteps % self.log_every_steps == 0:
            print(
                f"{self.algorithm}: step={self.num_timesteps:,}/{TRAINING_TIMESTEPS:,} "
                f"episode={self.current_episode} "
                f"reward={reward:.3f} "
                f"return_so_far={self.episode_return:.2f} "
                f"tps={self.num_timesteps / elapsed:.1f}",
                flush=True,
            )

        return True


def make_baseline_env(namespace: dict[str, Any], seed: int):
    return namespace["make_training_env"](seed=seed)


def make_cbf_env(namespace: dict[str, Any], seed: int):
    lambda_filter = float(namespace.get("CBF_FILTER_REWARD_LAMBDA", namespace.get("LAMBDA_FILTER", 0.05)))
    try:
        return namespace["make_cbf_training_env"](
            seed=seed,
            lambda_filter=lambda_filter,
            n_envs=1,
            use_subproc=False,
        )
    except TypeError:
        return namespace["make_cbf_training_env"](seed=seed, lambda_filter=lambda_filter)


def make_model(namespace: dict[str, Any], env: Any, seed: int, device: str) -> DDPG:
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions, dtype=np.float32),
        sigma=float(namespace["DDPG_OU_SIGMA"]) * np.ones(n_actions, dtype=np.float32),
    )
    return DDPG(
        "MlpPolicy",
        env,
        learning_rate=float(namespace["DDPG_LEARNING_RATE"]),
        buffer_size=REPLAY_MEMORY,
        learning_starts=LEARNING_STARTS,
        batch_size=int(namespace["DDPG_BATCH_SIZE"]),
        tau=float(namespace["DDPG_TAU"]),
        gamma=float(namespace["DDPG_GAMMA"]),
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
        policy_kwargs={"net_arch": [256, 128]},
        tensorboard_log=str(namespace["ARTIFACT_DIR"] / "tensorboard"),
        verbose=0,
        seed=seed,
        device=device,
    )


def train_variant(
    namespace: dict[str, Any],
    algorithm: str,
    make_env,
    output_dir: Path,
    seed: int,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    env = make_env(namespace, seed)
    model = make_model(namespace, env, seed=seed, device=device)
    callback = StepwiseTrainingLogger(algorithm=algorithm)
    model_path = output_dir / "models" / f"{algorithm.lower().replace('-', '_')}.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Training {algorithm} for {TRAINING_TIMESTEPS:,} timesteps ===", flush=True)
    start = time.perf_counter()
    try:
        model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=callback, progress_bar=False)
    finally:
        env.close()
    elapsed = time.perf_counter() - start
    model.save(str(model_path))

    step_frame = pd.DataFrame(callback.step_records)
    episode_frame = pd.DataFrame(callback.episode_records)
    if not episode_frame.empty:
        episode_frame["final_elapsed_sec"] = float(elapsed)
    print(
        f"Finished {algorithm}: elapsed={elapsed / 60.0:.2f} min, "
        f"episodes={len(episode_frame)}, model={model_path}",
        flush=True,
    )
    return step_frame, episode_frame, model_path


def add_rolling_columns(step_frame: pd.DataFrame) -> pd.DataFrame:
    if step_frame.empty:
        return step_frame
    result = step_frame.sort_values(["algorithm", "timestep"]).copy()
    grouped = result.groupby("algorithm", group_keys=False)
    result["reward_rolling"] = grouped["reward"].transform(
        lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    result["tps_rolling"] = grouped["timesteps_per_sec"].transform(
        lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    return result


def plot_training_curves(step_frame: pd.DataFrame, episode_frame: pd.DataFrame, output_dir: Path) -> Path:
    step_frame = add_rolling_columns(step_frame)
    plot_path = output_dir / "ddpg_vs_ddpg_cbf_stepwise_training_curves.png"
    colors = {"DDPG": "#1f77b4", "DDPG-CBF": "#d62728"}

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.ravel()

    for algorithm, group in step_frame.groupby("algorithm"):
        color = colors.get(algorithm, None)
        axes[0].plot(group["timestep"], group["reward"], color=color, alpha=0.18, linewidth=0.7)
        axes[0].plot(group["timestep"], group["reward_rolling"], color=color, linewidth=1.7, label=algorithm)
        axes[1].plot(group["timestep"], group["episode_return_so_far"], color=color, linewidth=1.0, label=algorithm)
        axes[2].plot(group["timestep"], group["episode_length_so_far"], color=color, linewidth=1.0, label=algorithm)
        axes[4].plot(group["timestep"], group["tps_rolling"], color=color, linewidth=1.6, label=algorithm)

    for algorithm, group in episode_frame.groupby("algorithm"):
        color = colors.get(algorithm, None)
        axes[3].plot(group["end_timestep"], group["return"], marker="o", markersize=3.5, color=color, label=algorithm)
        axes[5].plot(group["end_timestep"], group["length"], marker="o", markersize=3.5, color=color, label=algorithm)

    axes[0].set_title(f"Reward at every timestep with {ROLLING_WINDOW}-step mean")
    axes[0].set_ylabel("Reward")
    axes[1].set_title("Episode return so far at every timestep")
    axes[1].set_ylabel("Cumulative reward")
    axes[2].set_title("Episode length so far at every timestep")
    axes[2].set_ylabel("Current episode step")
    axes[3].set_title("Completed episode return")
    axes[3].set_ylabel("Episode return")
    axes[4].set_title("Training throughput")
    axes[4].set_ylabel("Timesteps / second")
    axes[5].set_title("Completed episode length")
    axes[5].set_ylabel("Episode length")

    for axis in axes:
        axis.set_xlabel("Training timestep")
        axis.grid(True, alpha=0.28)
        axis.legend()

    fig.suptitle("DDPG vs DDPG-CBF Stepwise Training Comparison", fontsize=15)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def plot_cbf_filter(step_frame: pd.DataFrame, output_dir: Path) -> Path:
    plot_path = output_dir / "ddpg_cbf_filter_every_timestep.png"
    cbf = step_frame[step_frame["algorithm"] == "DDPG-CBF"].copy()
    if cbf.empty:
        return plot_path

    cbf["correction_rolling"] = cbf["cbf_correction_norm"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    cbf["intervention_rolling"] = cbf["cbf_intervention"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    cbf["qp_failure_rolling"] = (1.0 - cbf["cbf_qp_success"]).rolling(ROLLING_WINDOW, min_periods=1).mean()

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(cbf["timestep"], cbf["cbf_correction_norm"], alpha=0.18, linewidth=0.7, color="#d62728")
    axes[0].plot(cbf["timestep"], cbf["correction_rolling"], linewidth=1.8, color="#d62728")
    axes[0].set_title(f"CBF correction norm at every timestep with {ROLLING_WINDOW}-step mean")
    axes[0].set_ylabel("Correction norm")

    axes[1].plot(cbf["timestep"], cbf["intervention_rolling"], linewidth=1.8, color="#9467bd")
    axes[1].set_title("Rolling CBF intervention rate")
    axes[1].set_ylabel("Rate")

    axes[2].plot(cbf["timestep"], cbf["qp_failure_rolling"], linewidth=1.8, color="#8c564b")
    axes[2].set_title("Rolling QP failure rate")
    axes[2].set_ylabel("Rate")

    for axis in axes:
        axis.grid(True, alpha=0.28)
    axes[-1].set_xlabel("Training timestep")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def write_summary(
    output_dir: Path,
    step_frame: pd.DataFrame,
    episode_frame: pd.DataFrame,
    model_paths: dict[str, Path],
    device: str,
) -> Path:
    rows = []
    for algorithm in sorted(step_frame["algorithm"].unique()):
        steps = step_frame[step_frame["algorithm"] == algorithm]
        episodes = episode_frame[episode_frame["algorithm"] == algorithm]
        row: dict[str, Any] = {
            "algorithm": algorithm,
            "training_timesteps": TRAINING_TIMESTEPS,
            "completed_episodes": len(episodes),
            "mean_step_reward": float(steps["reward"].mean()),
            "final_rolling_step_reward": float(steps["reward_rolling"].iloc[-1]),
            "final_timesteps_per_sec": float(steps["timesteps_per_sec"].iloc[-1]),
            "model_path": str(model_paths[algorithm]),
            "device": device,
        }
        if not episodes.empty:
            row.update(
                {
                    "mean_episode_return": float(episodes["return"].mean()),
                    "last_episode_return": float(episodes["return"].iloc[-1]),
                    "mean_episode_length": float(episodes["length"].mean()),
                    "last_episode_length": float(episodes["length"].iloc[-1]),
                }
            )
        if algorithm == "DDPG-CBF":
            row.update(
                {
                    "mean_cbf_correction_norm": float(steps["cbf_correction_norm"].mean()),
                    "mean_cbf_intervention_rate": float(steps["cbf_intervention"].mean()),
                    "mean_qp_failure_rate": float((1.0 - steps["cbf_qp_success"]).mean()),
                }
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    repo_root = find_repo_root()
    namespace = load_notebook_namespace(repo_root)

    seed = int(namespace["SEED"])
    device = os.environ.get("LANELESS_DDPG_DEVICE", str(namespace["DEVICE"]))
    run_tag = RUN_TAG or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir: Path = namespace["ARTIFACT_DIR"] / "ddpg_stepwise_comparison" / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "training_timesteps_per_variant": TRAINING_TIMESTEPS,
        "learning_starts": LEARNING_STARTS,
        "replay_memory": REPLAY_MEMORY,
        "rolling_window": ROLLING_WINDOW,
        "seed": seed,
        "device": device,
        "output_dir": str(output_dir),
        "cbf_lambda_filter": float(namespace.get("CBF_FILTER_REWARD_LAMBDA", namespace.get("LAMBDA_FILTER", 0.05))),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

    print("Project root:", repo_root, flush=True)
    print("Run config:", json.dumps(run_config, default=str), flush=True)

    baseline_steps, baseline_episodes, baseline_model_path = train_variant(
        namespace,
        algorithm="DDPG",
        make_env=make_baseline_env,
        output_dir=output_dir,
        seed=seed,
        device=device,
    )
    cbf_steps, cbf_episodes, cbf_model_path = train_variant(
        namespace,
        algorithm="DDPG-CBF",
        make_env=make_cbf_env,
        output_dir=output_dir,
        seed=seed,
        device=device,
    )

    step_frame = add_rolling_columns(pd.concat([baseline_steps, cbf_steps], ignore_index=True, sort=False))
    episode_frame = pd.concat([baseline_episodes, cbf_episodes], ignore_index=True, sort=False)

    step_trace_path = output_dir / "step_trace_every_timestep.csv"
    episode_trace_path = output_dir / "episode_trace.csv"
    step_frame.to_csv(step_trace_path, index=False)
    episode_frame.to_csv(episode_trace_path, index=False)

    comparison_plot_path = plot_training_curves(step_frame, episode_frame, output_dir)
    cbf_plot_path = plot_cbf_filter(step_frame, output_dir)
    summary_path = write_summary(
        output_dir,
        step_frame,
        episode_frame,
        model_paths={"DDPG": baseline_model_path, "DDPG-CBF": cbf_model_path},
        device=device,
    )

    print("\nSaved outputs:", flush=True)
    print("  step trace:", step_trace_path, flush=True)
    print("  episode trace:", episode_trace_path, flush=True)
    print("  comparison plot:", comparison_plot_path, flush=True)
    print("  CBF filter plot:", cbf_plot_path, flush=True)
    print("  summary:", summary_path, flush=True)
    print(pd.read_csv(summary_path).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
