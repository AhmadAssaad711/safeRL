from __future__ import annotations

import argparse
import faulthandler
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback


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


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return candidate
        nested = candidate / "highway-rl-decision-making"
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
        print(f"[train_cbf_damped_gain] executing notebook cell {cell_index}", flush=True)
        exec(compile(source, f"{notebook_path}:cell-{cell_index}", "exec"), namespace)


def set_cbf_gains(env: Any, k0: float, k1: float) -> None:
    current = env
    while current is not None:
        if hasattr(current, "k0") and hasattr(current, "k1"):
            current.k0 = float(k0)
            current.k1 = float(k1)
            return
        current = getattr(current, "env", None)
    raise RuntimeError("Could not find SafetyFilteredAccelerationWrapper to set k0/k1.")


def set_vec_cbf_gains(vec_env: Any, k0: float, k1: float) -> None:
    envs = getattr(vec_env, "envs", None)
    if envs is None:
        set_cbf_gains(vec_env, k0, k1)
        return
    for env in envs:
        set_cbf_gains(env, k0, k1)


def make_eval_env(namespace: dict[str, Any], seed: int, k0: float, k1: float, eps_side: float) -> Any:
    env = namespace["make_cbf_single_env"](
        seed=seed,
        lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
        eps_side=eps_side,
    )
    set_cbf_gains(env, k0, k1)
    namespace["configure_paper_evaluation_env"](env, steps=namespace["PAPER_EVAL_STEPS"])
    return env


def evaluate_model(
    namespace: dict[str, Any],
    model: DDPG,
    k0: float,
    k1: float,
    eps_side: float,
    episodes: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for episode in range(episodes):
        env = make_eval_env(namespace, seed + episode, k0, k1, eps_side)
        obs, _ = env.reset(seed=seed + episode)
        done = False
        step_count = 0
        rewards: list[float] = []
        speeds: list[float] = []
        signed_speed_errors: list[float] = []
        abs_speed_errors: list[float] = []
        lat_y_errors: list[float] = []
        corrections: list[float] = []
        interventions: list[float] = []
        qp_successes: list[float] = []
        min_h_values: list[float] = []
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

            rewards.append(float(reward))
            speeds.append(speed)
            signed_speed_errors.append(speed_error)
            abs_speed_errors.append(abs(speed_error))
            if np.isfinite(lat_y_error):
                lat_y_errors.append(lat_y_error)
            corrections.append(float(info.get("cbf_correction_norm", 0.0)))
            interventions.append(float(info.get("cbf_intervened", False)))
            qp_successes.append(float(info.get("cbf_qp_success", True)))
            min_h_values.append(float(info.get("cbf_min_h", np.nan)))
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
                "max_correction_norm": float(np.max(corrections)) if corrections else 0.0,
                "intervention_rate": float(np.mean(interventions)) if interventions else 0.0,
                "qp_failure_rate": float(1.0 - np.mean(qp_successes)) if qp_successes else 0.0,
                "min_h": float(np.nanmin(min_h_values)) if min_h_values and not np.all(np.isnan(min_h_values)) else np.nan,
                "ego_collisions": float(ego_collisions),
                "ego_collision_steps": float(ego_collision_steps),
                "total_collision_events": float(all_collision_events),
            }
        )
        env.close()
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = {
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
        "intervention_rate": float(metrics["intervention_rate"].mean()),
        "qp_failure_rate": float(metrics["qp_failure_rate"].mean()),
        "min_h": float(metrics["min_h"].min()),
        "ego_collisions_mean": float(metrics["ego_collisions"].mean()),
        "ego_collision_steps_mean": float(metrics["ego_collision_steps"].mean()),
        "total_collision_events_mean": float(metrics["total_collision_events"].mean()),
    }
    return pd.DataFrame([summary])


class DampedGainEvalCallback(BaseCallback):
    def __init__(
        self,
        namespace: dict[str, Any],
        k0: float,
        k1: float,
        eps_side: float,
        eval_freq: int,
        episodes: int,
        seed: int,
    ) -> None:
        super().__init__(verbose=1)
        self.namespace = namespace
        self.k0 = float(k0)
        self.k1 = float(k1)
        self.eps_side = float(eps_side)
        self.eval_freq = int(eval_freq)
        self.episodes = int(episodes)
        self.seed = int(seed)
        self.records: list[dict[str, float]] = []
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        metrics = evaluate_model(
            self.namespace,
            self.model,
            self.k0,
            self.k1,
            self.eps_side,
            episodes=self.episodes,
            seed=self.seed + self.num_timesteps,
        )
        row = summarize(metrics).iloc[0].to_dict()
        row["timesteps"] = float(self.num_timesteps)
        row["k0"] = self.k0
        row["k1"] = self.k1
        row["eps_side"] = self.eps_side
        self.records.append(row)
        print(
            "[train-eval]"
            f" steps={self.num_timesteps:,}"
            f" return={row['return_mean']:.2f}"
            f" abs_speed={row['mean_abs_speed_error']:.3f}"
            f" lat_y={row['mean_lat_y_error_m']:.3f}"
            f" intervention={row['intervention_rate']:.2%}",
            flush=True,
        )
        return True


def plot_results(history: pd.DataFrame, final_summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    if not history.empty:
        axes[0].plot(history["timesteps"], history["return_mean"], marker="o")
        axes[1].plot(history["timesteps"], history["mean_abs_speed_error"], marker="o")
        axes[2].plot(history["timesteps"], history["intervention_rate"], marker="o")
    axes[0].set_title("Eval Return During Training")
    axes[1].set_title("Eval Abs Speed Error")
    axes[2].set_title("Eval CBF Intervention Rate")
    axes[0].set_ylabel("Return")
    axes[1].set_ylabel("m/s")
    axes[2].set_ylabel("Rate")
    axes[2].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.set_xlabel("Training timestep")
        axis.grid(True, alpha=0.3)
    final = final_summary.iloc[0]
    fig.suptitle(
        "Retrained DDPG-CBF "
        f"k0={float(final['k0']):.2f}, k1={float(final['k1']):.2f}, "
        f"eps={float(final['eps_side']):.3f}; "
        f"final return={float(final['return_mean']):.1f}, "
        f"intervention={float(final['intervention_rate']):.1%}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain DDPG-CBF with explicit damped-system CBF gains.")
    parser.add_argument("--k0", type=float, default=5.29)
    parser.add_argument("--k1", type=float, default=3.68)
    parser.add_argument("--eps-side", type=float, default=None)
    parser.add_argument("--lambda-filter", type=float, default=None)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--train-eval-episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--project-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    args = parse_args()

    project_root = find_project_root(args.project_root or Path.cwd())
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    namespace: dict[str, Any] = {"__name__": "__main__"}
    exec_notebook_cells(notebook_path, [2, 3, 5, 6, 8, 33, 35, 37, 39, 41, 43], namespace)
    namespace["DEVICE"] = args.device
    if args.lambda_filter is not None:
        namespace["CBF_FILTER_REWARD_LAMBDA"] = float(args.lambda_filter)
    lambda_filter = float(namespace["CBF_FILTER_REWARD_LAMBDA"])
    eps_side = float(namespace["CBF_EPS_SIDE"] if args.eps_side is None else args.eps_side)

    tag = f"k0_{args.k0:.2f}_k1_{args.k1:.2f}_lambda_{lambda_filter:.3f}_eps_{eps_side:.3f}".replace(".", "p")
    output_dir = namespace["ARTIFACT_DIR"] / "cbf_damped_retrain" / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.zip"
    history_path = output_dir / "train_eval_history.csv"
    final_metrics_path = output_dir / "final_eval_episodes.csv"
    final_summary_path = output_dir / "final_summary.csv"
    plot_path = output_dir / "training_eval_summary.png"

    print(
        "[train] starting retrain",
        {
            "k0": args.k0,
            "k1": args.k1,
            "eps_side": eps_side,
            "lambda_filter": lambda_filter,
            "timesteps": args.timesteps,
            "eval_episodes": args.eval_episodes,
            "n_envs": args.n_envs,
            "output_dir": str(output_dir),
        },
        flush=True,
    )

    train_env = namespace["make_cbf_training_env"](
        seed=args.seed,
        lambda_filter=lambda_filter,
        eps_side=eps_side,
        n_envs=args.n_envs,
        use_subproc=False,
    )
    set_vec_cbf_gains(train_env, args.k0, args.k1)
    n_actions = train_env.action_space.shape[-1]
    action_noise = namespace["make_ou_action_noise"](n_actions, n_envs=args.n_envs)
    callback = DampedGainEvalCallback(
        namespace,
        args.k0,
        args.k1,
        eps_side,
        eval_freq=int(namespace["TRAIN_EVAL_EVERY"]),
        episodes=args.train_eval_episodes,
        seed=args.seed + 70_000,
    )
    model = DDPG(
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
        tensorboard_log=str(namespace["ARTIFACT_DIR"] / "tensorboard"),
        verbose=1,
        seed=args.seed,
        device=args.device,
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

    print("[train] running final evaluation", flush=True)
    final_metrics = evaluate_model(
        namespace,
        model,
        args.k0,
        args.k1,
        eps_side,
        episodes=args.eval_episodes,
        seed=args.seed + 90_000,
    )
    final_metrics.to_csv(final_metrics_path, index=False)
    final_summary = summarize(final_metrics)
    final_summary.insert(0, "lambda_filter", lambda_filter)
    final_summary.insert(0, "eps_side", eps_side)
    final_summary.insert(0, "k1", float(args.k1))
    final_summary.insert(0, "k0", float(args.k0))
    final_summary.insert(0, "timesteps", float(args.timesteps))
    final_summary.insert(0, "elapsed_sec", float(elapsed_sec))
    final_summary.insert(0, "model_path", str(model_path))
    final_summary.to_csv(final_summary_path, index=False)
    plot_results(history, final_summary, plot_path)

    print(f"[train] wrote model {model_path}", flush=True)
    print(f"[train] wrote history {history_path}", flush=True)
    print(f"[train] wrote final episodes {final_metrics_path}", flush=True)
    print(f"[train] wrote final summary {final_summary_path}", flush=True)
    print(f"[train] wrote plot {plot_path}", flush=True)
    print(final_summary.T.to_string(header=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
