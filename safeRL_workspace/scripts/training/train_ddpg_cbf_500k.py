from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback


TRAINING_TIMESTEPS = int(os.environ.get("LANELESS_DDPG_CBF_TIMESTEPS", "500000"))
CHUNK_TIMESTEPS = int(os.environ.get("LANELESS_DDPG_CBF_CHUNK_TIMESTEPS", "50000"))
TRAIN_EVAL_FREQ = int(os.environ.get("LANELESS_DDPG_CBF_EVAL_FREQ", "10000"))
TRAIN_EVAL_EPISODES = int(os.environ.get("LANELESS_DDPG_CBF_EVAL_EPISODES", "2"))
FINAL_EVAL_EPISODES = int(os.environ.get("LANELESS_DDPG_CBF_FINAL_EPISODES", "50"))
REPLAY_MEMORY = int(os.environ.get("LANELESS_DDPG_CBF_REPLAY_MEMORY", "100000"))
LEARNING_STARTS = int(os.environ.get("LANELESS_DDPG_CBF_LEARNING_STARTS", "1000"))
CHECKPOINT_FREQ = int(os.environ.get("LANELESS_DDPG_CBF_CHECKPOINT_FREQ", "50000"))
USE_SUBPROC = bool(int(os.environ.get("LANELESS_DDPG_CBF_USE_SUBPROC", "0")))
RESUME = bool(int(os.environ.get("LANELESS_DDPG_CBF_RESUME", "0")))


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


def summarize(metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "steps": float(metrics["steps"].mean()),
        "return": float(metrics["return"].mean()),
        "average_reward": float(metrics["return"].mean() / metrics["steps"].mean()),
        "mean_speed": float(metrics["mean_speed"].mean()),
        "mean_signed_speed_deviation": float(metrics["mean_signed_speed_deviation"].mean()),
        "mean_abs_speed_deviation": float(metrics["mean_abs_speed_deviation"].mean()),
        "ego_collisions": float(metrics["ego_collisions"].mean()),
        "ego_collision_steps": float(metrics["ego_collision_steps"].mean()),
        "total_collision_events": float(metrics["total_collision_events"].mean()),
        "mean_correction_norm": float(metrics["mean_correction_norm"].mean()),
        "max_correction_norm": float(metrics["max_correction_norm"].max()),
        "intervention_rate": float(metrics["intervention_rate"].mean()),
        "qp_failure_rate": float(metrics["qp_failure_rate"].mean()),
        "min_h": float(metrics["min_h"].min()),
    }


def plot_history(history: pd.DataFrame, output_path: Path) -> None:
    if history.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].plot(history["timesteps"], history["return"], marker="o")
    axes[0].set_title("Evaluation return")
    axes[0].set_ylabel("Return / 800 steps")

    axes[1].plot(history["timesteps"], history["mean_abs_speed_deviation"], marker="o")
    axes[1].set_title("Speed tracking error")
    axes[1].set_ylabel("Mean abs deviation (m/s)")

    axes[2].plot(history["timesteps"], history["ego_collisions"], marker="o", label="ego")
    axes[2].plot(
        history["timesteps"],
        history["total_collision_events"],
        marker="o",
        label="all traffic",
    )
    axes[2].set_title("Collisions")
    axes[2].set_ylabel("Events / 800 steps")
    axes[2].legend()

    axes[3].plot(history["timesteps"], history["intervention_rate"], marker="o", label="intervention")
    axes[3].plot(history["timesteps"], history["qp_failure_rate"], marker="o", label="QP failure")
    axes[3].set_title("CBF filter")
    axes[3].set_ylabel("Rate")
    axes[3].legend()

    for axis in axes:
        axis.set_xlabel("Training timesteps")
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


class PersistentCBFEvalCallback(BaseCallback):
    def __init__(
        self,
        namespace: dict[str, Any],
        eval_freq: int,
        n_eval_episodes: int,
        seed: int,
        lambda_filter: float,
        history_path: Path,
        checkpoint_dir: Path,
        checkpoint_freq: int,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.namespace = namespace
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.seed = int(seed)
        self.lambda_filter = float(lambda_filter)
        self.history_path = history_path
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = int(checkpoint_freq)
        self.records: list[dict[str, float]] = []
        self._last_eval_step = 0
        self._last_checkpoint_step = 0
        if self.history_path.exists():
            existing = pd.read_csv(self.history_path)
            if not existing.empty:
                self.records = existing.to_dict("records")
                self._last_eval_step = int(float(existing["timesteps"].max()))
                self._last_checkpoint_step = (self._last_eval_step // self.checkpoint_freq) * self.checkpoint_freq

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = self.num_timesteps
            metrics = self.namespace["evaluate_cbf_policy_with_metrics"](
                self.model,
                episodes=self.n_eval_episodes,
                seed=self.seed + self.num_timesteps,
                deterministic=True,
                lambda_filter=self.lambda_filter,
            )
            row = {
                "timesteps": float(self.num_timesteps),
                "return": float(metrics["return"].mean()),
                "mean_abs_speed_deviation": float(metrics["mean_abs_speed_deviation"].mean()),
                "ego_collisions": float(metrics["ego_collisions"].mean()),
                "total_collision_events": float(metrics["total_collision_events"].mean()),
                "mean_correction_norm": float(metrics["mean_correction_norm"].mean()),
                "intervention_rate": float(metrics["intervention_rate"].mean()),
                "qp_failure_rate": float(metrics["qp_failure_rate"].mean()),
                "min_h": float(metrics["min_h"].min()),
            }
            self.records.append(row)
            pd.DataFrame(self.records).to_csv(self.history_path, index=False)
            if self.verbose:
                print(
                    f"steps={self.num_timesteps:,} | "
                    f"abs dev={row['mean_abs_speed_deviation']:.3f} m/s | "
                    f"intervention={row['intervention_rate']:.2%} | "
                    f"qp failures={row['qp_failure_rate']:.2%} | "
                    f"ego collisions={row['ego_collisions']:.2f}",
                    flush=True,
                )

        if self.num_timesteps - self._last_checkpoint_step >= self.checkpoint_freq:
            self._last_checkpoint_step = self.num_timesteps
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.checkpoint_dir / f"ddpg_cbf_500k_step_{self.num_timesteps:06d}.zip"))

        return True


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = sorted(checkpoint_dir.glob("ddpg_cbf_500k_*.zip"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def main() -> None:
    repo_root = find_repo_root()
    ns = load_notebook_namespace(repo_root)

    seed = int(ns["SEED"])
    device = os.environ.get("LANELESS_DDPG_DEVICE", str(ns["DEVICE"]))
    qp_solver = os.environ.get("LANELESS_CBF_QP_SOLVER", str(ns.get("CBF_QP_SOLVER", "osqp")))
    ns["CBF_QP_SOLVER"] = qp_solver
    artifact_dir: Path = ns["ARTIFACT_DIR"]
    model_path = artifact_dir / "ddpg_cbf_flat42_vmax24_noslack_tuned_500k_laneless_karalakou.zip"
    history_path = artifact_dir / "ddpg_cbf_flat42_vmax24_noslack_tuned_500k_laneless_karalakou_eval_history.csv"
    final_eval_path = artifact_dir / "ddpg_cbf_flat42_vmax24_noslack_tuned_500k_laneless_karalakou_final_eval.csv"
    summary_path = artifact_dir / "ddpg_cbf_flat42_vmax24_noslack_tuned_500k_laneless_karalakou_summary.csv"
    plot_path = artifact_dir / "ddpg_cbf_flat42_vmax24_noslack_tuned_500k_laneless_karalakou_metrics.png"
    config_path = artifact_dir / "ddpg_cbf_flat42_vmax24_noslack_tuned_500k_laneless_karalakou_config.json"
    checkpoint_dir = artifact_dir / "ddpg_cbf_flat42_vmax24_noslack_tuned_500k_checkpoints"
    replay_buffer_path = checkpoint_dir / "ddpg_cbf_500k_replay_buffer.pkl"

    train_env = ns["make_cbf_training_env"](
        seed=seed,
        lambda_filter=ns["CBF_FILTER_REWARD_LAMBDA"],
        n_envs=ns["DDPG_CBF_NUM_ENVS"],
        use_subproc=USE_SUBPROC,
    )
    n_envs = int(getattr(train_env, "num_envs", 1))
    n_actions = train_env.action_space.shape[-1]
    action_noise = ns["make_ou_action_noise"](n_actions, n_envs=n_envs)
    callback = PersistentCBFEvalCallback(
        namespace=ns,
        eval_freq=TRAIN_EVAL_FREQ,
        n_eval_episodes=TRAIN_EVAL_EPISODES,
        seed=seed + 70_000,
        lambda_filter=ns["CBF_FILTER_REWARD_LAMBDA"],
        history_path=history_path,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=CHECKPOINT_FREQ,
        verbose=1,
    )

    run_config: dict[str, Any] = {
        "timesteps": TRAINING_TIMESTEPS,
        "chunk_timesteps": CHUNK_TIMESTEPS,
        "train_eval_freq": TRAIN_EVAL_FREQ,
        "train_eval_episodes": TRAIN_EVAL_EPISODES,
        "final_eval_episodes": FINAL_EVAL_EPISODES,
        "replay_memory": REPLAY_MEMORY,
        "learning_starts": LEARNING_STARTS,
        "learning_rate": ns["DDPG_LEARNING_RATE"],
        "batch_size": ns["DDPG_BATCH_SIZE"],
        "tau": ns["DDPG_TAU"],
        "gamma": ns["DDPG_GAMMA"],
        "ou_sigma": ns["DDPG_OU_SIGMA"],
        "net_arch": [256, 128],
        "n_envs": n_envs,
        "subproc_requested": bool(ns["DDPG_USE_SUBPROC_VEC_ENV"]),
        "subproc_used": bool(USE_SUBPROC),
        "resume": bool(RESUME),
        "checkpoint_freq": CHECKPOINT_FREQ,
        "device": str(device),
        "seed": seed,
        "lambda_filter": ns["CBF_FILTER_REWARD_LAMBDA"],
        "cbf_qp_solver": qp_solver,
        "cbf_k0": ns["CBF_K0"],
        "cbf_k1": ns["CBF_K1"],
        "cbf_ax_bounds": ns["CBF_AX_BOUNDS"],
        "cbf_ay_bounds": ns["CBF_AY_BOUNDS"],
        "observation_vmax": ns["ENV_CONFIG"]["observation_vmax"],
        "normalize_observations": bool(ns["NORMALIZE_RL_OBSERVATIONS"]),
    }
    config_path.write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

    print("Project root:", repo_root, flush=True)
    print("Run config:", json.dumps(run_config, default=str), flush=True)
    print("Model path:", model_path, flush=True)

    resume_path = latest_checkpoint(checkpoint_dir) if RESUME else None
    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}", flush=True)
        model = DDPG.load(str(resume_path), env=train_env, device=device)
        model.action_noise = action_noise
        if replay_buffer_path.exists():
            model.load_replay_buffer(str(replay_buffer_path))
            print(f"Loaded replay buffer: {replay_buffer_path}", flush=True)
        else:
            model.learning_starts = int(model.num_timesteps) + LEARNING_STARTS
            print(
                f"No replay buffer in checkpoint; delaying resumed updates until "
                f"{model.learning_starts:,} timesteps.",
                flush=True,
            )
    else:
        model = DDPG(
            "MlpPolicy",
            train_env,
            learning_rate=ns["DDPG_LEARNING_RATE"],
            buffer_size=REPLAY_MEMORY,
            learning_starts=LEARNING_STARTS,
            batch_size=ns["DDPG_BATCH_SIZE"],
            tau=ns["DDPG_TAU"],
            gamma=ns["DDPG_GAMMA"],
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=action_noise,
            policy_kwargs={"net_arch": [256, 128]},
            tensorboard_log=str(artifact_dir / "tensorboard"),
            verbose=1,
            seed=seed,
            device=device,
        )

    start = time.time()
    print("Starting DDPG-CBF learn loop", flush=True)
    try:
        while model.num_timesteps < TRAINING_TIMESTEPS:
            remaining = TRAINING_TIMESTEPS - int(model.num_timesteps)
            chunk_steps = min(CHUNK_TIMESTEPS, remaining)
            target_steps = int(model.num_timesteps) + int(chunk_steps)
            print(
                f"Learning chunk: current={model.num_timesteps:,}, "
                f"chunk={chunk_steps:,}, target={target_steps:,}",
                flush=True,
            )
            model.learn(
                total_timesteps=chunk_steps,
                callback=callback,
                reset_num_timesteps=(model.num_timesteps == 0),
                progress_bar=False,
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(checkpoint_dir / f"ddpg_cbf_500k_chunk_{model.num_timesteps:06d}.zip"))
            model.save_replay_buffer(str(replay_buffer_path))
    finally:
        elapsed = time.time() - start
        train_env.close()
        print("Closed training env after %.1f min" % (elapsed / 60.0), flush=True)

    model.save(str(model_path))

    history = pd.DataFrame(callback.records)
    history.to_csv(history_path, index=False)
    plot_history(history, plot_path)

    final_metrics = ns["evaluate_cbf_policy_with_metrics"](
        model,
        episodes=FINAL_EVAL_EPISODES,
        seed=seed + 900_000,
        deterministic=True,
        lambda_filter=ns["CBF_FILTER_REWARD_LAMBDA"],
    )
    final_metrics.to_csv(final_eval_path, index=False)

    summary = {
        **run_config,
        "elapsed_min": float(elapsed / 60.0),
        "model_path": str(model_path),
        "history_path": str(history_path),
        "final_eval_path": str(final_eval_path),
        "plot_path": str(plot_path),
        **summarize(final_metrics),
    }
    summary_frame = pd.DataFrame([summary])
    summary_frame.to_csv(summary_path, index=False)

    print("Training complete in %.1f min" % (elapsed / 60.0), flush=True)
    print("Saved model:", model_path, flush=True)
    print("Saved history:", history_path, flush=True)
    print("Saved final eval:", final_eval_path, flush=True)
    print("Saved summary:", summary_path, flush=True)
    print("Saved plot:", plot_path, flush=True)
    print(summary_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
