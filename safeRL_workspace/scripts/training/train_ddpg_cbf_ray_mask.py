from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CallbackList

from scripts.common.cbf_ray_mask import install_ray_mask_cbf


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

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "class StepwiseTrainingMetricsCallback" in source:
            exec(compile(source, f"{notebook_path}:cell_{index}", "exec"), namespace)
            break
    else:
        raise RuntimeError("Could not find notebook cell containing StepwiseTrainingMetricsCallback")

    install_ray_mask_cbf(namespace)
    return namespace


def _info_float(info: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = info.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def evaluate_ray_mask_policy(
    namespace: dict[str, Any],
    model: DDPG,
    episodes: int,
    seed: int,
    lambda_filter: float,
    backup_qp: bool,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for episode in range(episodes):
        env = namespace["make_ray_mask_single_env"](
            seed=seed + episode,
            lambda_filter=lambda_filter,
            backup_qp=backup_qp,
        )
        namespace["configure_paper_evaluation_env"](env, steps=namespace["PAPER_EVAL_STEPS"])
        obs, _ = env.reset(seed=seed + episode)
        done = False
        step_count = 0
        rewards: list[float] = []
        speed_deviations: list[float] = []
        abs_speed_deviations: list[float] = []
        corrections: list[float] = []
        interventions: list[float] = []
        backup_uses: list[float] = []
        qp_successes: list[float] = []
        min_h_values: list[float] = []
        safe_violations: list[float] = []
        ego_collision_steps = 0
        ego_collision_events = 0
        total_collision_events = 0

        while not done and step_count < namespace["PAPER_EVAL_STEPS"]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            step_count += 1
            rewards.append(float(reward))
            speed_deviations.append(_info_float(info, "karalakou_speed_deviation", np.nan))
            abs_speed_deviations.append(_info_float(info, "karalakou_abs_speed_deviation", np.nan))
            corrections.append(_info_float(info, "cbf_correction_norm", 0.0))
            interventions.append(float(bool(info.get("cbf_intervened", False))))
            backup_uses.append(float(bool(info.get("cbf_ray_backup_qp_used", False))))
            qp_successes.append(float(bool(info.get("cbf_qp_success", True))))
            min_h_values.append(_info_float(info, "cbf_min_h", np.nan))
            safe_violations.append(_info_float(info, "cbf_max_constraint_violation_safe", np.nan))
            if bool(info.get("ego_collision", False)):
                ego_collision_steps += 1
            ego_collision_events += int(info.get("ego_collision_events", 0))
            total_collision_events += int(info.get("collisions", 0))

        env.close()
        rows.append(
            {
                "episode": float(episode),
                "steps": float(step_count),
                "return": float(np.sum(rewards)),
                "average_reward": float(np.mean(rewards)) if rewards else 0.0,
                "mean_speed_deviation": float(np.nanmean(speed_deviations)) if speed_deviations else np.nan,
                "mean_abs_speed_deviation": float(np.nanmean(abs_speed_deviations)) if abs_speed_deviations else np.nan,
                "mean_correction_norm": float(np.mean(corrections)) if corrections else 0.0,
                "max_correction_norm": float(np.max(corrections)) if corrections else 0.0,
                "intervention_rate": float(np.mean(interventions)) if interventions else 0.0,
                "ray_backup_rate": float(np.mean(backup_uses)) if backup_uses else 0.0,
                "qp_failure_rate": float(1.0 - np.mean(qp_successes)) if qp_successes else 0.0,
                "min_h": float(np.nanmin(min_h_values)) if min_h_values else np.nan,
                "max_safe_violation": float(np.nanmax(safe_violations)) if safe_violations else np.nan,
                "ego_collision_steps": float(ego_collision_steps),
                "ego_collision_events": float(ego_collision_events),
                "total_collision_events": float(total_collision_events),
            }
        )
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "episodes": float(len(metrics)),
        "return_mean": float(metrics["return"].mean()),
        "average_reward_mean": float(metrics["average_reward"].mean()),
        "mean_abs_speed_deviation": float(metrics["mean_abs_speed_deviation"].mean()),
        "mean_correction_norm": float(metrics["mean_correction_norm"].mean()),
        "intervention_rate": float(metrics["intervention_rate"].mean()),
        "ray_backup_rate": float(metrics["ray_backup_rate"].mean()),
        "qp_failure_rate": float(metrics["qp_failure_rate"].mean()),
        "min_h": float(metrics["min_h"].min()),
        "max_safe_violation": float(metrics["max_safe_violation"].max()),
        "ego_collision_events": float(metrics["ego_collision_events"].mean()),
        "total_collision_events": float(metrics["total_collision_events"].mean()),
    }


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("ray_mask_chunk_*.zip"):
        match = re.search(r"ray_mask_chunk_(\d+)\.zip$", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate DDPG with CBF ray-mask action mapping.")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--chunk-timesteps", type=int, default=5_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lambda-filter", type=float, default=None)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--train-eval-freq", type=int, default=10_000)
    parser.add_argument("--no-backup-qp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root()
    namespace = load_notebook_namespace(repo_root)
    namespace["DEVICE"] = args.device
    namespace["SEED"] = int(args.seed)
    if args.lambda_filter is not None:
        namespace["CBF_FILTER_REWARD_LAMBDA"] = float(args.lambda_filter)
    lambda_filter = float(namespace["CBF_FILTER_REWARD_LAMBDA"])
    backup_qp = not bool(args.no_backup_qp)

    artifact_dir: Path = namespace["ARTIFACT_DIR"]
    model_path = artifact_dir / "ddpg_cbf_ray_mask_flat42_vmax24_noslack_tuned_laneless_karalakou.zip"
    history_path = artifact_dir / "ddpg_cbf_ray_mask_flat42_vmax24_noslack_tuned_laneless_karalakou_eval_history.csv"
    final_metrics_path = artifact_dir / "ddpg_cbf_ray_mask_flat42_vmax24_noslack_tuned_laneless_karalakou_final_metrics.csv"
    summary_path = artifact_dir / "ddpg_cbf_ray_mask_flat42_vmax24_noslack_tuned_laneless_karalakou_summary.csv"
    step_trace_path = artifact_dir / "ddpg_cbf_ray_mask_training_step_trace.csv"
    episode_trace_path = artifact_dir / "ddpg_cbf_ray_mask_training_episode_trace.csv"
    checkpoint_dir = artifact_dir / "ddpg_cbf_ray_mask_checkpoints"
    replay_buffer_path = checkpoint_dir / "ray_mask_replay_buffer.pkl"

    train_env = namespace["make_ray_mask_training_env"](
        seed=args.seed,
        lambda_filter=lambda_filter,
        n_envs=args.n_envs,
        use_subproc=False,
        backup_qp=backup_qp,
    )
    n_envs = int(getattr(train_env, "num_envs", 1))
    n_actions = train_env.action_space.shape[-1]
    action_noise = namespace["make_ou_action_noise"](n_actions, n_envs=n_envs)
    stepwise_callback = namespace["StepwiseTrainingMetricsCallback"](
        variant="DDPG-CBF ray mask",
        step_trace_path=step_trace_path,
        episode_trace_path=episode_trace_path,
    )

    resume_path = latest_checkpoint(checkpoint_dir) if args.resume else None
    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}", flush=True)
        model = DDPG.load(str(resume_path), env=train_env, device=args.device)
        model.action_noise = action_noise
        if replay_buffer_path.exists():
            model.load_replay_buffer(str(replay_buffer_path))
            print(f"Loaded replay buffer: {replay_buffer_path}", flush=True)
    else:
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
            tensorboard_log=str(artifact_dir / "tensorboard"),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )

    print(
        "Starting DDPG-CBF ray-mask training:",
        {
            "timesteps": args.timesteps,
            "chunk_timesteps": args.chunk_timesteps,
            "n_envs": n_envs,
            "lambda_filter": lambda_filter,
            "backup_qp": backup_qp,
            "action_space": str(train_env.action_space),
        },
        flush=True,
    )
    start = time.time()
    try:
        while int(model.num_timesteps) < int(args.timesteps):
            remaining = int(args.timesteps) - int(model.num_timesteps)
            chunk_steps = min(max(1, int(args.chunk_timesteps)), remaining)
            print(
                f"Learning chunk: current={model.num_timesteps:,}, "
                f"chunk={chunk_steps:,}, target={model.num_timesteps + chunk_steps:,}",
                flush=True,
            )
            model.learn(
                total_timesteps=chunk_steps,
                callback=CallbackList([stepwise_callback]),
                reset_num_timesteps=(int(model.num_timesteps) == 0),
                progress_bar=False,
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(checkpoint_dir / f"ray_mask_chunk_{int(model.num_timesteps):06d}.zip"))
            model.save_replay_buffer(str(replay_buffer_path))
            print(f"Saved checkpoint at {model.num_timesteps:,} timesteps", flush=True)
    finally:
        train_env.close()

    elapsed = time.time() - start
    model.save(str(model_path))
    print(f"Saved model to {model_path}", flush=True)
    print(f"Training time: {elapsed / 60:.1f} min", flush=True)

    if stepwise_callback.step_records:
        history = pd.DataFrame(stepwise_callback.step_records)
        history.to_csv(history_path, index=False)
    else:
        pd.DataFrame().to_csv(history_path, index=False)

    final_metrics = evaluate_ray_mask_policy(
        namespace,
        model,
        episodes=int(args.eval_episodes),
        seed=args.seed + 90_000,
        lambda_filter=lambda_filter,
        backup_qp=backup_qp,
    )
    final_metrics.to_csv(final_metrics_path, index=False)
    summary = summarize(final_metrics)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print("Final summary:", json.dumps(summary, indent=2), flush=True)
    print(f"Saved final metrics to {final_metrics_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
