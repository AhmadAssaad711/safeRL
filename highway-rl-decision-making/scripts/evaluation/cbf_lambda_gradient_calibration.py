from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th


NOTEBOOK_DEPS = [2, 3, 5, 6, 8, 33, 35, 37, 39, 41, 43]


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
        print(f"[cbf_lambda_gradient_calibration] executing notebook cell {cell_index}", flush=True)
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


def default_model_path(namespace: dict[str, Any], k0: float, k1: float, lambda_filter: float, eps_side: float) -> Path:
    artifact_dir = Path(namespace["ARTIFACT_DIR"])
    tags = [
        f"k0_{k0:.2f}_k1_{k1:.2f}_lambda_{lambda_filter:.3f}_eps_{eps_side:.3f}".replace(".", "p"),
        f"k0_{k0:.2f}_k1_{k1:.2f}_lambda_{lambda_filter:.3f}".replace(".", "p"),
        f"k0_{k0:.2f}_k1_{k1:.2f}".replace(".", "p"),
    ]
    for tag in tags:
        candidate = artifact_dir / "cbf_damped_retrain" / tag / "model.zip"
        if candidate.exists():
            return candidate
    return Path(namespace["DDPG_CBF_MODEL_PATH"])


def parse_float_list(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def read_action(info: dict[str, Any], x_key: str, y_key: str, fallback: np.ndarray) -> np.ndarray:
    if x_key in info and y_key in info:
        action = np.asarray([info[x_key], info[y_key]], dtype=np.float32)
        if action.size >= 2 and np.all(np.isfinite(action[:2])):
            return action[:2].astype(np.float32)
    return np.asarray(fallback, dtype=np.float32).reshape(-1)[:2]


def to_actor_scale(action_phys: np.ndarray, action_space: Any) -> np.ndarray:
    low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)[:2]
    high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)[:2]
    action_phys = np.asarray(action_phys, dtype=np.float32).reshape(-1)[:2]
    scaled = 2.0 * ((np.clip(action_phys, low, high) - low) / np.maximum(high - low, 1e-6)) - 1.0
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def describe_values(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_p99": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p50": percentile(values, 50),
        f"{prefix}_p90": percentile(values, 90),
        f"{prefix}_p95": percentile(values, 95),
        f"{prefix}_p99": percentile(values, 99),
        f"{prefix}_max": float(np.max(values)),
    }


def collect_diagnostic_batch(
    namespace: dict[str, Any],
    model: Any,
    *,
    steps: int,
    seed: int,
    k0: float,
    k1: float,
    eps_side: float,
    episode_steps: int,
    reward_lambda_in_env: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, float]] = []
    observations: list[np.ndarray] = []
    raw_actions_scaled: list[np.ndarray] = []
    safe_actions_scaled: list[np.ndarray] = []
    episode = 0

    while len(rows) < steps:
        env = namespace["make_cbf_single_env"](
            seed=seed + episode,
            lambda_filter=reward_lambda_in_env,
            eps_side=eps_side,
        )
        set_cbf_gains(env, k0, k1)
        namespace["configure_paper_evaluation_env"](env, steps=episode_steps)
        obs, _ = env.reset(seed=seed + episode)
        done = False
        step_in_episode = 0

        while not done and len(rows) < steps:
            obs_for_actor = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
            action_phys, _ = model.predict(obs, deterministic=True)
            raw_action_input = np.asarray(action_phys, dtype=np.float32).reshape(-1)[:2]
            obs, reward, terminated, truncated, info = env.step(raw_action_input)

            raw_phys = read_action(info, "cbf_a_rl_x", "cbf_a_rl_y", raw_action_input)
            safe_phys = read_action(info, "cbf_a_safe_x", "cbf_a_safe_y", raw_phys)
            raw_scaled = to_actor_scale(raw_phys, env.action_space)
            safe_scaled = to_actor_scale(safe_phys, env.action_space)
            correction_norm = float(info.get("cbf_correction_norm", np.linalg.norm(safe_phys - raw_phys)))
            scaled_correction_norm = float(np.linalg.norm(safe_scaled - raw_scaled))
            intervention = bool(info.get("cbf_intervened", correction_norm > 1e-6))
            qp_success = bool(info.get("cbf_qp_success", True))
            fallback_used = bool(info.get("cbf_fallback_used", not qp_success))
            max_constraint_violation_rl = float(info.get("cbf_max_constraint_violation_rl", np.nan))
            max_constraint_violation_safe = float(info.get("cbf_max_constraint_violation_safe", np.nan))
            raw_feasible = bool(
                info.get(
                    "cbf_raw_feasible",
                    max_constraint_violation_rl <= float(namespace.get("CBF_QP_FEASIBILITY_TOL", 1e-3)),
                )
            )

            observations.append(obs_for_actor)
            raw_actions_scaled.append(raw_scaled)
            safe_actions_scaled.append(safe_scaled)
            rows.append(
                {
                    "global_step": float(len(rows)),
                    "episode": float(episode),
                    "episode_step": float(step_in_episode),
                    "base_reward": float(reward),
                    "abs_base_reward": float(abs(reward)),
                    "raw_ax": float(raw_phys[0]),
                    "raw_ay": float(raw_phys[1]),
                    "safe_ax": float(safe_phys[0]),
                    "safe_ay": float(safe_phys[1]),
                    "raw_scaled_ax": float(raw_scaled[0]),
                    "raw_scaled_ay": float(raw_scaled[1]),
                    "safe_scaled_ax": float(safe_scaled[0]),
                    "safe_scaled_ay": float(safe_scaled[1]),
                    "correction_norm": correction_norm,
                    "correction_norm_sq": float(correction_norm**2),
                    "scaled_correction_norm": scaled_correction_norm,
                    "intervention": float(intervention),
                    "qp_success": float(qp_success),
                    "fallback_used": float(fallback_used),
                    "raw_feasible": float(raw_feasible),
                    "strict_bc_mask_unit": float(
                        intervention and qp_success and not fallback_used and scaled_correction_norm > 0.0
                    ),
                    "min_h": float(info.get("cbf_min_h", np.nan)),
                    "max_constraint_violation_rl": max_constraint_violation_rl,
                    "max_constraint_violation_safe": max_constraint_violation_safe,
                    "ego_collision": float(bool(info.get("ego_collision", False))),
                    "ego_collision_events": float(info.get("ego_collision_events", 0)),
                    "total_collision_events": float(info.get("collisions", 0)),
                }
            )
            step_in_episode += 1
            done = bool(terminated or truncated)

        env.close()
        episode += 1

    return (
        pd.DataFrame(rows),
        np.asarray(observations, dtype=np.float32),
        np.asarray(raw_actions_scaled, dtype=np.float32),
        np.asarray(safe_actions_scaled, dtype=np.float32),
    )


def nearest_value(values: list[float], target: float) -> float:
    if not values:
        return float(target)
    return min(values, key=lambda value: abs(float(value) - float(target)))


def reward_scale_tables(
    samples: pd.DataFrame,
    lambda_norm_values: list[float],
    lambda_event_values: list[float],
    event_thresholds: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_abs_reward_mean = float(samples["abs_base_reward"].mean())
    base_reward_std = float(samples["base_reward"].std())
    norm_unit = samples["correction_norm_sq"].to_numpy(dtype=np.float64)
    correction_norm = samples["correction_norm"].to_numpy(dtype=np.float64)

    term_rows: list[dict[str, float | str]] = []
    for value in lambda_norm_values:
        penalty = float(value) * norm_unit
        row: dict[str, float | str] = {
            "term": "norm",
            "lambda": float(value),
            "event_threshold": np.nan,
            "event_rate": np.nan,
            "unit_mean": float(np.mean(norm_unit)),
            "base_abs_reward_mean": base_abs_reward_mean,
            "base_reward_std": base_reward_std,
            "mean_penalty_to_abs_reward": float(np.mean(penalty) / max(base_abs_reward_mean, 1e-9)),
        }
        row.update(describe_values(penalty, "penalty"))
        term_rows.append(row)

    event_threshold_rows: list[dict[str, float]] = []
    for threshold in event_thresholds:
        event_unit = (correction_norm > float(threshold)).astype(np.float64)
        event_threshold_rows.append(
            {
                "event_threshold": float(threshold),
                "event_rate": float(np.mean(event_unit)),
                "event_count": float(np.sum(event_unit)),
            }
        )
        for value in lambda_event_values:
            penalty = float(value) * event_unit
            row = {
                "term": "event",
                "lambda": float(value),
                "event_threshold": float(threshold),
                "event_rate": float(np.mean(event_unit)),
                "unit_mean": float(np.mean(event_unit)),
                "base_abs_reward_mean": base_abs_reward_mean,
                "base_reward_std": base_reward_std,
                "mean_penalty_to_abs_reward": float(np.mean(penalty) / max(base_abs_reward_mean, 1e-9)),
            }
            row.update(describe_values(penalty, "penalty"))
            term_rows.append(row)

    grid_rows: list[dict[str, float]] = []
    for event_threshold in event_thresholds:
        event_unit = (correction_norm > float(event_threshold)).astype(np.float64)
        for lambda_norm in lambda_norm_values:
            for lambda_event in lambda_event_values:
                total = float(lambda_norm) * norm_unit + float(lambda_event) * event_unit
                row = {
                    "lambda_norm": float(lambda_norm),
                    "lambda_event": float(lambda_event),
                    "event_threshold": float(event_threshold),
                    "event_rate": float(np.mean(event_unit)),
                    "base_abs_reward_mean": base_abs_reward_mean,
                    "base_reward_std": base_reward_std,
                    "mean_total_penalty_to_abs_reward": float(np.mean(total) / max(base_abs_reward_mean, 1e-9)),
                }
                row.update(describe_values(total, "total_penalty"))
                grid_rows.append(row)

    return pd.DataFrame(term_rows), pd.DataFrame(grid_rows), pd.DataFrame(event_threshold_rows)


def actor_grad_norm(model: Any, loss: th.Tensor) -> float:
    model.actor.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    total_sq = 0.0
    for parameter in model.actor.parameters():
        if parameter.grad is not None:
            total_sq += float(parameter.grad.detach().pow(2).sum().item())
    model.actor.optimizer.zero_grad(set_to_none=True)
    return float(total_sq**0.5)


def bc_loss_from_batch(
    model: Any,
    obs: th.Tensor,
    raw_actions: th.Tensor,
    safe_actions: th.Tensor,
    mask: th.Tensor,
    *,
    bc_action_scale: float,
    bc_weight_max: float,
) -> tuple[th.Tensor, float, float]:
    a_pred = model.actor(obs)
    scaled_corrections = th.norm(safe_actions - raw_actions, dim=1, keepdim=True)
    mask_float = mask.float()
    weights = 1.0 + th.clamp(scaled_corrections / max(float(bc_action_scale), 1e-6), min=0.0, max=float(bc_weight_max))
    bc_per_sample = ((a_pred - safe_actions) ** 2).sum(dim=1, keepdim=True)
    bc_loss = (mask_float * weights * bc_per_sample).sum() / (mask_float.sum() + 1e-6)
    mask_rate = float(mask_float.mean().item())
    if mask_float.sum().item() > 0.0:
        weight_mean = float((mask_float * weights).sum().item() / (mask_float.sum().item() + 1e-6))
    else:
        weight_mean = 0.0
    return bc_loss, mask_rate, weight_mean


def gradient_tables(
    model: Any,
    samples: pd.DataFrame,
    observations: np.ndarray,
    raw_actions_scaled: np.ndarray,
    safe_actions_scaled: np.ndarray,
    *,
    lambda_bc_values: list[float],
    bc_delta: float,
    bc_action_scale: float,
    bc_weight_max: float,
    batch_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    count = len(samples)
    batch_count = min(int(batch_size), count)
    batch_indices = np.sort(rng.choice(count, size=batch_count, replace=False))
    device = model.device

    obs = th.as_tensor(observations[batch_indices], device=device, dtype=th.float32)
    raw_actions = th.as_tensor(raw_actions_scaled[batch_indices], device=device, dtype=th.float32)
    safe_actions = th.as_tensor(safe_actions_scaled[batch_indices], device=device, dtype=th.float32)
    scaled_corrections = th.norm(safe_actions - raw_actions, dim=1, keepdim=True)
    minimal_mask = scaled_corrections > float(bc_delta)

    strict_mask_np = (
        (samples.iloc[batch_indices]["intervention"].to_numpy(dtype=np.float32) > 0.5)
        & (samples.iloc[batch_indices]["qp_success"].to_numpy(dtype=np.float32) > 0.5)
        & (samples.iloc[batch_indices]["fallback_used"].to_numpy(dtype=np.float32) < 0.5)
    )
    strict_mask = th.as_tensor(strict_mask_np.reshape(-1, 1), device=device, dtype=th.bool) & minimal_mask

    model.policy.set_training_mode(True)
    a_pred = model.actor(obs)
    rl_actor_loss = -model.critic.q1_forward(obs, a_pred).mean()
    rl_grad_norm = actor_grad_norm(model, rl_actor_loss)

    bc_minimal_loss, minimal_mask_rate, minimal_weight_mean = bc_loss_from_batch(
        model,
        obs,
        raw_actions,
        safe_actions,
        minimal_mask,
        bc_action_scale=bc_action_scale,
        bc_weight_max=bc_weight_max,
    )
    bc_minimal_grad_norm = actor_grad_norm(model, bc_minimal_loss)

    bc_strict_loss, strict_mask_rate, strict_weight_mean = bc_loss_from_batch(
        model,
        obs,
        raw_actions,
        safe_actions,
        strict_mask,
        bc_action_scale=bc_action_scale,
        bc_weight_max=bc_weight_max,
    )
    bc_strict_grad_norm = actor_grad_norm(model, bc_strict_loss)

    summary = pd.DataFrame(
        [
            {
                "batch_size": float(batch_count),
                "rl_actor_loss": float(rl_actor_loss.detach().cpu().item()),
                "rl_actor_grad_norm": rl_grad_norm,
                "minimal_bc_loss_unit": float(bc_minimal_loss.detach().cpu().item()),
                "minimal_bc_grad_norm_unit": bc_minimal_grad_norm,
                "minimal_bc_mask_rate": minimal_mask_rate,
                "minimal_bc_weight_mean": minimal_weight_mean,
                "strict_bc_loss_unit": float(bc_strict_loss.detach().cpu().item()),
                "strict_bc_grad_norm_unit": bc_strict_grad_norm,
                "strict_bc_mask_rate": strict_mask_rate,
                "strict_bc_weight_mean": strict_weight_mean,
                "bc_delta": float(bc_delta),
                "bc_action_scale": float(bc_action_scale),
                "bc_weight_max": float(bc_weight_max),
            }
        ]
    )

    rows: list[dict[str, float | str]] = []
    for lambda_bc in lambda_bc_values:
        for mask_name, grad_norm, loss_value, mask_rate in [
            (
                "minimal",
                bc_minimal_grad_norm,
                float(bc_minimal_loss.detach().cpu().item()),
                minimal_mask_rate,
            ),
            (
                "strict",
                bc_strict_grad_norm,
                float(bc_strict_loss.detach().cpu().item()),
                strict_mask_rate,
            ),
        ]:
            rows.append(
                {
                    "mask": mask_name,
                    "lambda_bc": float(lambda_bc),
                    "bc_loss_scaled": float(lambda_bc) * loss_value,
                    "bc_grad_norm_scaled": float(lambda_bc) * grad_norm,
                    "rl_actor_grad_norm": rl_grad_norm,
                    "bc_to_rl_grad_ratio": float(lambda_bc) * grad_norm / max(rl_grad_norm, 1e-12),
                    "mask_rate": mask_rate,
                }
            )
    return summary, pd.DataFrame(rows)


def recommend_ranges(reward_grid: pd.DataFrame, gradient_grid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    reward_candidates = reward_grid[
        (reward_grid["mean_total_penalty_to_abs_reward"] >= 0.005)
        & (reward_grid["mean_total_penalty_to_abs_reward"] <= 0.12)
    ].copy()
    if reward_candidates.empty:
        reward_candidates = reward_grid.copy()
    reward_candidates = reward_candidates.sort_values(
        ["mean_total_penalty_to_abs_reward", "total_penalty_p90"],
        ascending=[True, True],
    ).head(8)
    for _, row in reward_candidates.iterrows():
        rows.append(
            {
                "category": "reward_pair",
                "lambda_norm": float(row["lambda_norm"]),
                "lambda_event": float(row["lambda_event"]),
                "event_threshold": float(row["event_threshold"]),
                "lambda_bc": np.nan,
                "mask": "",
                "score_value": float(row["mean_total_penalty_to_abs_reward"]),
                "reason": "mean total penalty is in a modest fraction of mean absolute step reward",
            }
        )

    bc_candidates = gradient_grid[
        (gradient_grid["mask"] == "minimal")
        & (gradient_grid["bc_to_rl_grad_ratio"] >= 0.05)
        & (gradient_grid["bc_to_rl_grad_ratio"] <= 0.50)
    ].copy()
    if bc_candidates.empty:
        bc_candidates = gradient_grid[gradient_grid["mask"] == "minimal"].copy()
    bc_candidates = bc_candidates.sort_values("bc_to_rl_grad_ratio").head(6)
    for _, row in bc_candidates.iterrows():
        rows.append(
            {
                "category": "bc_lambda",
                "lambda_norm": np.nan,
                "lambda_event": np.nan,
                "event_threshold": np.nan,
                "lambda_bc": float(row["lambda_bc"]),
                "mask": str(row["mask"]),
                "score_value": float(row["bc_to_rl_grad_ratio"]),
                "reason": "BC actor gradient is a controlled fraction of RL actor gradient",
            }
        )
    return pd.DataFrame(rows)


def plot_outputs(
    samples: pd.DataFrame,
    reward_grid: pd.DataFrame,
    gradient_grid: pd.DataFrame,
    output_path: Path,
    plot_event_threshold: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hist(samples["correction_norm"], bins=60, alpha=0.82, color="#1f77b4")
    axes[0].set_title("Physical Correction Norm")
    axes[0].set_xlabel("||a_safe - a_raw||")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.25)

    threshold_values = sorted(reward_grid["event_threshold"].dropna().unique().tolist())
    selected_threshold = nearest_value(threshold_values, plot_event_threshold)
    plot_grid = reward_grid[np.isclose(reward_grid["event_threshold"], selected_threshold)]
    pivot = plot_grid.pivot(index="lambda_norm", columns="lambda_event", values="mean_total_penalty_to_abs_reward")
    image = axes[1].imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_title(f"Mean Reward Perturbation\nEvent threshold={selected_threshold:g}")
    axes[1].set_xlabel("lambda_event")
    axes[1].set_ylabel("lambda_norm")
    axes[1].set_xticks(range(len(pivot.columns)))
    axes[1].set_xticklabels([f"{value:g}" for value in pivot.columns], rotation=45, ha="right")
    axes[1].set_yticks(range(len(pivot.index)))
    axes[1].set_yticklabels([f"{value:g}" for value in pivot.index])
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="mean penalty / mean |reward|")

    minimal = gradient_grid[gradient_grid["mask"] == "minimal"]
    strict = gradient_grid[gradient_grid["mask"] == "strict"]
    axes[2].plot(minimal["lambda_bc"], minimal["bc_to_rl_grad_ratio"], marker="o", label="minimal mask")
    axes[2].plot(strict["lambda_bc"], strict["bc_to_rl_grad_ratio"], marker="s", label="strict mask")
    axes[2].axhspan(0.05, 0.50, color="#2ca02c", alpha=0.12)
    axes[2].set_title("BC Gradient vs RL Gradient")
    axes[2].set_xlabel("lambda_bc")
    axes[2].set_ylabel("||lambda_bc grad BC|| / ||grad RL||")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate CBF reward/loss lambda ranges from reward scales and gradients.")
    parser.add_argument("--k0", type=float, default=5.29)
    parser.add_argument("--k1", type=float, default=3.68)
    parser.add_argument("--eps-side", type=float, default=0.149)
    parser.add_argument("--model-lambda-filter", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--episode-steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=410_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lambda-norm-values", default="0,0.01,0.025,0.05,0.075,0.10")
    parser.add_argument("--lambda-event-values", default="0,0.001,0.0025,0.005,0.01,0.02")
    parser.add_argument("--lambda-bc-values", default="0,0.001,0.003,0.01,0.03,0.10,0.30,1.0")
    parser.add_argument("--event-thresholds", default="0.000001,0.001,0.01,0.03,0.05,0.10")
    parser.add_argument("--plot-event-threshold", type=float, default=0.03)
    parser.add_argument("--bc-delta", type=float, default=0.03)
    parser.add_argument("--bc-action-scale", type=float, default=1.0)
    parser.add_argument("--bc-weight-max", type=float, default=5.0)
    parser.add_argument("--gradient-batch-size", type=int, default=2048)
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

    model_path = args.model_path or default_model_path(
        namespace,
        k0=args.k0,
        k1=args.k1,
        lambda_filter=args.model_lambda_filter,
        eps_side=args.eps_side,
    )
    output_dir = args.output_dir or (
        Path(namespace["ARTIFACT_DIR"])
        / "cbf_lambda_gradient_calibration"
        / f"k0_{args.k0:.2f}_k1_{args.k1:.2f}_eps_{args.eps_side:.3f}".replace(".", "p")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[calibration] starting",
        {
            "model_path": str(model_path),
            "steps": args.steps,
            "k0": args.k0,
            "k1": args.k1,
            "eps_side": args.eps_side,
            "output_dir": str(output_dir),
        },
        flush=True,
    )
    model = namespace["DDPG"].load(str(model_path), device=args.device)

    samples, observations, raw_actions_scaled, safe_actions_scaled = collect_diagnostic_batch(
        namespace,
        model,
        steps=args.steps,
        seed=args.seed,
        k0=args.k0,
        k1=args.k1,
        eps_side=args.eps_side,
        episode_steps=args.episode_steps,
        reward_lambda_in_env=0.0,
    )

    lambda_norm_values = parse_float_list(args.lambda_norm_values)
    lambda_event_values = parse_float_list(args.lambda_event_values)
    lambda_bc_values = parse_float_list(args.lambda_bc_values)
    event_thresholds = parse_float_list(args.event_thresholds)

    term_scales, reward_grid, event_threshold_summary = reward_scale_tables(
        samples,
        lambda_norm_values,
        lambda_event_values,
        event_thresholds,
    )
    gradient_summary, gradient_grid = gradient_tables(
        model,
        samples,
        observations,
        raw_actions_scaled,
        safe_actions_scaled,
        lambda_bc_values=lambda_bc_values,
        bc_delta=args.bc_delta,
        bc_action_scale=args.bc_action_scale,
        bc_weight_max=args.bc_weight_max,
        batch_size=args.gradient_batch_size,
        seed=args.seed + 7,
    )
    recommendations = recommend_ranges(reward_grid, gradient_grid)

    sample_summary = pd.DataFrame(
        [
            {
                "steps": float(len(samples)),
                "episodes": float(samples["episode"].nunique()),
                "base_reward_mean": float(samples["base_reward"].mean()),
                "base_abs_reward_mean": float(samples["abs_base_reward"].mean()),
                "base_reward_std": float(samples["base_reward"].std()),
                "correction_norm_mean": float(samples["correction_norm"].mean()),
                "correction_norm_p90": percentile(samples["correction_norm"].to_numpy(dtype=float), 90),
                "scaled_correction_norm_mean": float(samples["scaled_correction_norm"].mean()),
                "intervention_rate": float(samples["intervention"].mean()),
                "raw_feasible_rate": float(samples["raw_feasible"].mean()),
                "qp_failure_rate": float(1.0 - samples["qp_success"].mean()),
                "fallback_rate": float(samples["fallback_used"].mean()),
                "min_h_min": float(samples["min_h"].min()),
                "ego_collision_steps": float(samples["ego_collision"].sum()),
            }
        ]
    )

    samples_path = output_dir / "diagnostic_samples.csv"
    sample_summary_path = output_dir / "sample_summary.csv"
    term_scales_path = output_dir / "term_scale_summary.csv"
    event_threshold_summary_path = output_dir / "event_threshold_summary.csv"
    reward_grid_path = output_dir / "reward_lambda_grid.csv"
    gradient_summary_path = output_dir / "gradient_summary.csv"
    gradient_grid_path = output_dir / "bc_gradient_lambda_grid.csv"
    recommendations_path = output_dir / "recommended_ranges.csv"
    plot_path = output_dir / "calibration_summary.png"
    config_path = output_dir / "run_config.json"

    samples.to_csv(samples_path, index=False)
    sample_summary.to_csv(sample_summary_path, index=False)
    term_scales.to_csv(term_scales_path, index=False)
    event_threshold_summary.to_csv(event_threshold_summary_path, index=False)
    reward_grid.to_csv(reward_grid_path, index=False)
    gradient_summary.to_csv(gradient_summary_path, index=False)
    gradient_grid.to_csv(gradient_grid_path, index=False)
    recommendations.to_csv(recommendations_path, index=False)
    plot_outputs(samples, reward_grid, gradient_grid, plot_path, args.plot_event_threshold)
    config_path.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "k0": args.k0,
                "k1": args.k1,
                "eps_side": args.eps_side,
                "model_lambda_filter": args.model_lambda_filter,
                "steps": args.steps,
                "episode_steps": args.episode_steps,
                "seed": args.seed,
                "lambda_norm_values": lambda_norm_values,
                "lambda_event_values": lambda_event_values,
                "lambda_bc_values": lambda_bc_values,
                "event_thresholds": event_thresholds,
                "plot_event_threshold": args.plot_event_threshold,
                "bc_delta": args.bc_delta,
                "bc_action_scale": args.bc_action_scale,
                "bc_weight_max": args.bc_weight_max,
                "gradient_batch_size": args.gradient_batch_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[calibration] wrote {samples_path}", flush=True)
    print(f"[calibration] wrote {term_scales_path}", flush=True)
    print(f"[calibration] wrote {event_threshold_summary_path}", flush=True)
    print(f"[calibration] wrote {reward_grid_path}", flush=True)
    print(f"[calibration] wrote {gradient_summary_path}", flush=True)
    print(f"[calibration] wrote {gradient_grid_path}", flush=True)
    print(f"[calibration] wrote {recommendations_path}", flush=True)
    print(f"[calibration] wrote {plot_path}", flush=True)
    print("[calibration] sample summary", flush=True)
    print(sample_summary.T.to_string(header=False), flush=True)
    print("[calibration] gradient summary", flush=True)
    print(gradient_summary.T.to_string(header=False), flush=True)
    print("[calibration] recommendations", flush=True)
    print(recommendations.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
