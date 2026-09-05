from __future__ import annotations

import argparse
import faulthandler
import json
import os
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th

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
from scripts.evaluation.evaluate_filter_policy_contribution import (
    _as_float,
    load_model_for_variant,
    make_cbf_eval_env,
    model_action_is_normalized,
    model_action_to_physical,
    physical_bounds,
    physical_to_normalized,
)
from scripts.common.guided_cbf_minimal import install_minimal_guided_cbf
from scripts.training.train_safety_potential_variants import (
    MTM_CONGESTED_UNCERTAIN_UPDATES,
    SAFETY_REWARD_TRIAL,
    deep_update,
    make_reward_config,
)


warnings.filterwarnings("ignore", message="OSQP exited.*")


DEFAULT_VARIANTS = [
    "update_a_raw_q_raw_bc",
    "update_b_safe_critic_raw_actor",
    "update_c_safe_critic_diff_actor",
]


def _finite_mean(values: pd.Series, default: float = np.nan) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.mean()) if len(numeric) else float(default)


def _finite_quantile(values: pd.Series, q: float, default: float = np.nan) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.quantile(q)) if len(numeric) else float(default)


def model_action_from_physical(namespace: dict[str, Any], model: Any, env: Any, action_phys: np.ndarray) -> np.ndarray:
    action_phys = np.asarray(action_phys, dtype=np.float32).reshape(-1)[:2]
    if model_action_is_normalized(model):
        return physical_to_normalized(namespace, env, action_phys)
    low = np.asarray(model.action_space.low, dtype=np.float32).reshape(-1)[:2]
    high = np.asarray(model.action_space.high, dtype=np.float32).reshape(-1)[:2]
    return np.clip(action_phys, low, high).astype(np.float32)


def critic_q(model: Any, obs: np.ndarray, action_model_space: np.ndarray) -> float:
    obs_array = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    action_array = np.asarray(action_model_space, dtype=np.float32).reshape(1, -1)[:, :2]
    with th.no_grad():
        obs_tensor = th.as_tensor(obs_array, device=model.device)
        action_tensor = th.as_tensor(action_array, device=model.device)
        return float(model.critic.q1_forward(obs_tensor, action_tensor).cpu().item())


def classify_projection(info: dict[str, Any], correction_norm: float, event_threshold: float) -> str:
    qp_success = bool(info.get("cbf_qp_success", True))
    fallback_used = bool(info.get("cbf_fallback_used", not qp_success))
    if fallback_used or not qp_success:
        return "fail"
    active_count = int(_as_float(info.get("cbf_active_constraint_count"), default=-1.0))
    if active_count < 0:
        return "inside" if correction_norm <= event_threshold else "projected"
    if active_count <= 0:
        return "inside"
    if active_count == 1:
        return "edge"
    return "vertex"


def evaluate_variant(
    namespace: dict[str, Any],
    *,
    model: Any,
    variant: str,
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    low_model = np.asarray(model.action_space.low, dtype=np.float32).reshape(-1)[:2]
    high_model = np.asarray(model.action_space.high, dtype=np.float32).reshape(-1)[:2]
    low_phys, high_phys = physical_bounds(env_config)
    zero_model_action = np.clip(np.zeros(2, dtype=np.float32), low_model, high_model)
    model.policy.set_training_mode(False)

    for episode in range(int(episodes)):
        episode_seed = int(seed) + int(checkpoint_step) * 10 + episode
        env = make_cbf_eval_env(
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
        rng = np.random.default_rng(episode_seed + 17)
        obs, _ = env.reset(seed=episode_seed)
        done = False
        step_index = 0
        episode_return = 0.0
        ego_collisions = 0
        total_collision_events = 0
        last_task_info: dict[str, Any] = {}

        while not done:
            action_model, _ = model.predict(obs, deterministic=True)
            action_model = np.asarray(action_model, dtype=np.float32).reshape(-1)[:2]
            raw_phys = model_action_to_physical(model, env, action_model, env_config)
            obs_before = np.asarray(obs, dtype=np.float32).copy()
            obs, reward, terminated, truncated, info_raw = env.step(raw_phys)
            info = dict(info_raw)
            raw_logged = np.asarray(
                [
                    _as_float(info.get("cbf_a_rl_x"), default=raw_phys[0]),
                    _as_float(info.get("cbf_a_rl_y"), default=raw_phys[1]),
                ],
                dtype=np.float32,
            )
            safe_phys = np.asarray(
                [
                    _as_float(info.get("cbf_a_safe_x"), default=raw_logged[0]),
                    _as_float(info.get("cbf_a_safe_y"), default=raw_logged[1]),
                ],
                dtype=np.float32,
            )
            safe_model = model_action_from_physical(namespace, model, env, safe_phys)
            safe_perturb_phys = np.clip(
                safe_phys + rng.normal(loc=0.0, scale=np.asarray([0.15, 0.10], dtype=np.float32)),
                low_phys,
                high_phys,
            ).astype(np.float32)
            safe_perturb_model = model_action_from_physical(namespace, model, env, safe_perturb_phys)
            random_model = rng.uniform(low=low_model, high=high_model).astype(np.float32)

            delta = safe_phys - raw_logged
            correction_norm = float(np.linalg.norm(delta))
            raw_violation = _as_float(info.get("cbf_max_constraint_violation_rl"), default=np.nan)
            safe_violation = _as_float(info.get("cbf_max_constraint_violation_safe"), default=np.nan)
            qp_success = bool(info.get("cbf_qp_success", True))
            fallback_used = bool(info.get("cbf_fallback_used", not qp_success))
            projection_bucket = classify_projection(info, correction_norm, event_threshold)

            q_raw = critic_q(model, obs_before, action_model)
            q_safe = critic_q(model, obs_before, safe_model)
            q_safe_perturb = critic_q(model, obs_before, safe_perturb_model)
            q_random = critic_q(model, obs_before, random_model)
            q_zero = critic_q(model, obs_before, zero_model_action)

            step_rows.append(
                {
                    "variant": variant,
                    "checkpoint_step": float(checkpoint_step),
                    "episode": float(episode),
                    "episode_seed": float(episode_seed),
                    "step": float(step_index),
                    "global_step": float(episode * task_max_steps + step_index),
                    "reward": float(reward),
                    "raw_ax": float(raw_logged[0]),
                    "raw_ay": float(raw_logged[1]),
                    "safe_ax": float(safe_phys[0]),
                    "safe_ay": float(safe_phys[1]),
                    "delta_ax": float(delta[0]),
                    "delta_ay": float(delta[1]),
                    "correction_norm": correction_norm,
                    "meaningful_correction_norm": float(max(correction_norm - event_threshold, 0.0)),
                    "event_intervened": float(correction_norm > event_threshold),
                    "numerical_intervened": float(correction_norm > 1e-6),
                    "raw_feasible": float(bool(info.get("cbf_raw_feasible", raw_violation <= 1e-6))),
                    "raw_constraint_violation": raw_violation,
                    "safe_constraint_violation": safe_violation,
                    "safe_infeasible": float(safe_violation > 1e-5) if np.isfinite(safe_violation) else np.nan,
                    "qp_success": float(qp_success),
                    "qp_failure": float(not qp_success),
                    "fallback_used": float(fallback_used),
                    "soft_qp_success": float(bool(info.get("cbf_soft_qp_success", False))),
                    "soft_qp_used": float(bool(info.get("cbf_soft_qp_used", False))),
                    "soft_qp_slack_l2": _as_float(info.get("cbf_soft_qp_slack_l2"), default=np.nan),
                    "soft_qp_slack_max": _as_float(info.get("cbf_soft_qp_slack_max"), default=np.nan),
                    "fallback_source": str(info.get("cbf_fallback_source", "none")),
                    "fallback_max_constraint_violation": _as_float(
                        info.get("cbf_fallback_max_constraint_violation"),
                        default=np.nan,
                    ),
                    "fallback_positive_violation_l2": _as_float(
                        info.get("cbf_fallback_positive_violation_l2"),
                        default=np.nan,
                    ),
                    "h_min": _as_float(info.get("cbf_min_h", info.get("kpi_h_min")), default=np.nan),
                    "boundary_h_min": _as_float(
                        info.get("cbf_min_boundary_h", info.get("kpi_boundary_h_min")),
                        default=np.nan,
                    ),
                    "active_constraint_count": _as_float(info.get("cbf_active_constraint_count"), default=np.nan),
                    "num_neighbor_constraints": _as_float(info.get("cbf_num_neighbor_constraints"), default=np.nan),
                    "projection_bucket": projection_bucket,
                    "q_raw": q_raw,
                    "q_safe": q_safe,
                    "q_safe_perturb": q_safe_perturb,
                    "q_random": q_random,
                    "q_zero": q_zero,
                    "q_safe_minus_raw": q_safe - q_raw,
                    "q_safe_perturb_minus_safe": q_safe_perturb - q_safe,
                    "q_safe_minus_zero": q_safe - q_zero,
                    "ego_collision_events": float(info.get("ego_collision_events", info.get("kpi_ego_collision_events", 0))),
                    "total_collision_events": float(info.get("collisions", info.get("total_collision_events", 0))),
                }
            )

            episode_return += float(reward)
            ego_collisions += int(info.get("ego_collision_events", info.get("kpi_ego_collision_events", 0)))
            total_collision_events += int(info.get("collisions", info.get("total_collision_events", 0)))
            last_task_info = {
                "task_completed": bool(info.get("task_completed", False)),
                "task_timeout": bool(info.get("task_timeout", False)),
                "task_distance_traveled_m": _as_float(info.get("task_distance_traveled_m"), default=0.0),
            }
            step_index += 1
            done = bool(terminated or truncated)

        distance_km = max(float(last_task_info.get("task_distance_traveled_m", 0.0)) / 1000.0, 1e-9)
        episode_rows.append(
            {
                "variant": variant,
                "checkpoint_step": float(checkpoint_step),
                "episode": float(episode),
                "episode_return": float(episode_return),
                "episode_length_steps": float(step_index),
                "task_completed": float(last_task_info.get("task_completed", False)),
                "task_timeout": float(last_task_info.get("task_timeout", False)),
                "task_distance_traveled_m": float(last_task_info.get("task_distance_traveled_m", 0.0)),
                "ego_collisions": float(ego_collisions),
                "ego_collisions_per_km": float(ego_collisions) / distance_km,
                "total_collision_events": float(total_collision_events),
            }
        )
        env.close()

    return pd.DataFrame(step_rows), pd.DataFrame(episode_rows)


def summarize_steps(steps: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, group in steps.groupby("variant", sort=False):
        ep_group = episodes[episodes["variant"] == variant]
        bucket_counts = group["projection_bucket"].value_counts(normalize=True)
        qp_success_mask = (pd.to_numeric(group["qp_success"], errors="coerce") > 0.5) & (
            pd.to_numeric(group["fallback_used"], errors="coerce") < 0.5
        )
        qp_fail_mask = ~qp_success_mask

        def masked_mean(column: str, mask: pd.Series, default: float = np.nan) -> float:
            if column not in group:
                return float(default)
            return _finite_mean(group.loc[mask, column], default=default)

        def masked_quantile(column: str, mask: pd.Series, q: float, default: float = np.nan) -> float:
            if column not in group:
                return float(default)
            return _finite_quantile(group.loc[mask, column], q=q, default=default)

        def masked_rate(column: str, mask: pd.Series, default: float = np.nan) -> float:
            if column not in group:
                return float(default)
            return _finite_mean(group.loc[mask, column], default=default)

        row = {
            "variant": variant,
            "steps": float(len(group)),
            "episodes": float(len(ep_group)),
            "return_mean": _finite_mean(ep_group["episode_return"]),
            "episode_length_steps_mean": _finite_mean(ep_group["episode_length_steps"]),
            "completion_rate": _finite_mean(ep_group["task_completed"]),
            "ego_collisions_mean": _finite_mean(ep_group["ego_collisions"]),
            "ego_collisions_per_km_mean": _finite_mean(ep_group["ego_collisions_per_km"]),
            "event_intervention_rate": _finite_mean(group["event_intervened"], default=0.0),
            "numerical_intervention_rate": _finite_mean(group["numerical_intervened"], default=0.0),
            "mean_correction_norm": _finite_mean(group["correction_norm"], default=0.0),
            "p50_correction_norm": _finite_quantile(group["correction_norm"], 0.50, default=0.0),
            "p95_correction_norm": _finite_quantile(group["correction_norm"], 0.95, default=0.0),
            "max_correction_norm": float(pd.to_numeric(group["correction_norm"], errors="coerce").max()),
            "qp_failure_rate": _finite_mean(group["qp_failure"], default=0.0),
            "fallback_rate": _finite_mean(group["fallback_used"], default=0.0),
            "soft_qp_used_rate": _finite_mean(group["soft_qp_used"], default=0.0),
            "soft_qp_success_rate": _finite_mean(group["soft_qp_success"], default=0.0),
            "soft_qp_slack_l2_mean": masked_mean("soft_qp_slack_l2", qp_fail_mask),
            "soft_qp_slack_max_mean": masked_mean("soft_qp_slack_max", qp_fail_mask),
            "fallback_max_constraint_violation_mean": masked_mean("fallback_max_constraint_violation", qp_fail_mask),
            "fallback_positive_violation_l2_mean": masked_mean("fallback_positive_violation_l2", qp_fail_mask),
            "raw_feasible_rate": _finite_mean(group["raw_feasible"], default=0.0),
            "raw_violation_rate": _finite_mean((group["raw_constraint_violation"] > 1e-5).astype(float), default=0.0),
            "safe_infeasible_rate": _finite_mean(group["safe_infeasible"], default=0.0),
            "raw_constraint_violation_mean": _finite_mean(group["raw_constraint_violation"]),
            "raw_constraint_violation_p95": _finite_quantile(group["raw_constraint_violation"], 0.95),
            "safe_constraint_violation_mean": _finite_mean(group["safe_constraint_violation"]),
            "safe_constraint_violation_p95": _finite_quantile(group["safe_constraint_violation"], 0.95),
            "h_min": float(pd.to_numeric(group["h_min"], errors="coerce").min()),
            "h_min_mean": _finite_mean(group["h_min"]),
            "active_constraint_count_mean": _finite_mean(group["active_constraint_count"]),
            "inside_rate": float(bucket_counts.get("inside", 0.0)),
            "edge_rate": float(bucket_counts.get("edge", 0.0)),
            "vertex_rate": float(bucket_counts.get("vertex", 0.0)),
            "fail_bucket_rate": float(bucket_counts.get("fail", 0.0)),
            "projected_bucket_rate": float(bucket_counts.get("projected", 0.0)),
            "q_raw_mean": _finite_mean(group["q_raw"]),
            "q_safe_mean": _finite_mean(group["q_safe"]),
            "q_safe_perturb_mean": _finite_mean(group["q_safe_perturb"]),
            "q_random_mean": _finite_mean(group["q_random"]),
            "q_zero_mean": _finite_mean(group["q_zero"]),
            "q_safe_minus_raw_mean": _finite_mean(group["q_safe_minus_raw"]),
            "q_raw_minus_safe_mean": _finite_mean(-group["q_safe_minus_raw"]),
            "q_safe_perturb_minus_safe_mean": _finite_mean(group["q_safe_perturb_minus_safe"]),
            "q_safe_minus_zero_mean": _finite_mean(group["q_safe_minus_zero"]),
            "q_safe_gt_raw_rate": _finite_mean((group["q_safe_minus_raw"] > 0.0).astype(float), default=0.0),
            "qp_success_step_rate": float(qp_success_mask.mean()) if len(group) else np.nan,
            "qp_success_step_reward_mean": masked_mean("reward", qp_success_mask),
            "qp_fail_step_reward_mean": masked_mean("reward", qp_fail_mask),
            "correction_norm_qp_success_mean": masked_mean("correction_norm", qp_success_mask, default=0.0),
            "correction_norm_qp_fail_mean": masked_mean("correction_norm", qp_fail_mask, default=0.0),
            "correction_norm_qp_success_p95": masked_quantile("correction_norm", qp_success_mask, 0.95, default=0.0),
            "correction_norm_qp_fail_p95": masked_quantile("correction_norm", qp_fail_mask, 0.95, default=0.0),
            "safe_infeasible_qp_success_rate": masked_rate("safe_infeasible", qp_success_mask),
            "safe_infeasible_qp_fail_rate": masked_rate("safe_infeasible", qp_fail_mask),
            "safe_constraint_violation_qp_success_mean": masked_mean("safe_constraint_violation", qp_success_mask),
            "safe_constraint_violation_qp_success_p95": masked_quantile(
                "safe_constraint_violation",
                qp_success_mask,
                0.95,
            ),
            "safe_constraint_violation_qp_fail_mean": masked_mean("safe_constraint_violation", qp_fail_mask),
            "safe_constraint_violation_qp_fail_p95": masked_quantile(
                "safe_constraint_violation",
                qp_fail_mask,
                0.95,
            ),
            "q_raw_qp_success_mean": masked_mean("q_raw", qp_success_mask),
            "q_safe_qp_success_mean": masked_mean("q_safe", qp_success_mask),
            "q_safe_perturb_qp_success_mean": masked_mean("q_safe_perturb", qp_success_mask),
            "q_zero_qp_success_mean": masked_mean("q_zero", qp_success_mask),
            "q_safe_minus_raw_qp_success_mean": masked_mean("q_safe_minus_raw", qp_success_mask),
            "q_raw_minus_safe_qp_success_mean": masked_mean("q_safe_minus_raw", qp_success_mask) * -1.0,
            "q_safe_minus_zero_qp_success_mean": masked_mean("q_safe_minus_zero", qp_success_mask),
            "q_safe_gt_raw_qp_success_rate": _finite_mean(
                (group.loc[qp_success_mask, "q_safe_minus_raw"] > 0.0).astype(float),
                default=np.nan,
            ),
            "q_safe_gt_zero_qp_success_rate": _finite_mean(
                (group.loc[qp_success_mask, "q_safe_minus_zero"] > 0.0).astype(float),
                default=np.nan,
            ),
        }
        source_counts = group.loc[qp_fail_mask, "fallback_source"].value_counts(normalize=True)
        for source_name in [
            "soft_qp",
            "linprog_minimax",
            "linprog_after_soft_qp_failure",
            "continuous_refinement",
            "continuous_after_soft_qp_failure",
            "grid_refinement",
            "grid_after_soft_qp_failure",
            "legacy_grid",
            "emergency_brake",
        ]:
            row[f"fallback_source_{source_name}_rate"] = float(source_counts.get(source_name, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_qp_fail_over_time(steps: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    for variant, group in steps.groupby("variant", sort=False):
        frame = group.sort_values(["episode", "step"]).reset_index(drop=True)
        rolling = frame["qp_failure"].rolling(window=75, min_periods=10).mean()
        axis.plot(np.arange(len(frame)), rolling, label=variant)
    axis.set_title("QP Failure Rate Over Evaluation Steps")
    axis.set_xlabel("Collected step")
    axis.set_ylabel("Rolling failure rate")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_correction_distribution(steps: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    bins = np.linspace(0.0, max(3.0, float(steps["correction_norm"].max())), 45)
    for variant, group in steps.groupby("variant", sort=False):
        axis.hist(group["correction_norm"], bins=bins, density=True, alpha=0.35, label=variant)
    axis.set_title("Correction Norm Distribution")
    axis.set_xlabel("|u_safe - u_raw|")
    axis.set_ylabel("Density")
    axis.grid(True, alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_q_raw_vs_safe(steps: pd.DataFrame, output_path: Path) -> None:
    variants = list(dict.fromkeys(steps["variant"].astype(str)))
    fig, axes = plt.subplots(1, len(variants), figsize=(5.1 * len(variants), 4.7), squeeze=False)
    all_q = pd.concat([steps["q_raw"], steps["q_safe"]], ignore_index=True)
    q_min = float(all_q.quantile(0.01))
    q_max = float(all_q.quantile(0.99))
    for axis, variant in zip(axes.ravel(), variants):
        group = steps[steps["variant"] == variant]
        axis.scatter(group["q_raw"], group["q_safe"], s=10, alpha=0.35)
        axis.plot([q_min, q_max], [q_min, q_max], color="black", linewidth=1.0, linestyle="--")
        axis.set_title(variant)
        axis.set_xlabel("Q(s, u_raw)")
        axis.set_ylabel("Q(s, u_safe)")
        axis.grid(True, alpha=0.2)
    fig.suptitle("Critic Values: Raw Action vs Filtered Action", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_projection_buckets(steps: pd.DataFrame, output_path: Path) -> None:
    buckets = ["inside", "edge", "vertex", "fail", "projected"]
    variants = list(dict.fromkeys(steps["variant"].astype(str)))
    values = []
    for variant in variants:
        counts = steps[steps["variant"] == variant]["projection_bucket"].value_counts(normalize=True)
        values.append([float(counts.get(bucket, 0.0)) for bucket in buckets])
    value_array = np.asarray(values, dtype=float)
    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    bottom = np.zeros(len(variants), dtype=float)
    x = np.arange(len(variants))
    for index, bucket in enumerate(buckets):
        axis.bar(x, value_array[:, index], bottom=bottom, label=bucket)
        bottom += value_array[:, index]
    axis.set_xticks(x, variants, rotation=15, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Fraction of steps")
    axis.set_title("Projection Geometry Buckets")
    axis.grid(True, axis="y", alpha=0.2)
    axis.legend(fontsize=8, ncol=len(buckets))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_success_failure_split(summary: pd.DataFrame, output_path: Path) -> None:
    variants = list(summary["variant"].astype(str))
    x = np.arange(len(variants))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    axes[0].bar(x - width / 2, summary["correction_norm_qp_success_mean"], width, label="QP success")
    axes[0].bar(x + width / 2, summary["correction_norm_qp_fail_mean"], width, label="QP fail/fallback")
    axes[0].set_title("Mean Correction Norm")
    axes[0].set_ylabel("|u_safe - u_raw|")
    axes[1].bar(x - width / 2, summary["safe_infeasible_qp_success_rate"], width, label="QP success")
    axes[1].bar(x + width / 2, summary["safe_infeasible_qp_fail_rate"], width, label="QP fail/fallback")
    axes[1].set_title("Safe Action Infeasible Rate")
    axes[1].set_ylabel("Fraction of steps")
    for axis in axes:
        axis.set_xticks(x, variants, rotation=15, ha="right")
        axis.grid(True, axis="y", alpha=0.22)
        axis.legend(fontsize=8)
    fig.suptitle("Normal QP Solves vs Fallback Steps", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_qp_success_q_values(summary: pd.DataFrame, output_path: Path) -> None:
    variants = list(summary["variant"].astype(str))
    columns = [
        ("q_raw_qp_success_mean", "raw"),
        ("q_safe_qp_success_mean", "safe"),
        ("q_safe_perturb_qp_success_mean", "safe + small perturb"),
        ("q_zero_qp_success_mean", "zero"),
    ]
    x = np.arange(len(variants))
    width = 0.18
    fig, axis = plt.subplots(figsize=(12.0, 4.8))
    for index, (column, label) in enumerate(columns):
        offsets = (index - (len(columns) - 1) / 2.0) * width
        axis.bar(x + offsets, summary[column], width, label=label)
    axis.set_xticks(x, variants, rotation=15, ha="right")
    axis.set_ylabel("Mean Q on QP-success steps")
    axis.set_title("Critic Values on In-Distribution Successful Safe Actions")
    axis.grid(True, axis="y", alpha=0.22)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_plots(steps: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_qp_fail_over_time(steps, output_dir / "01_qp_failure_over_time.png")
    plot_correction_distribution(steps, output_dir / "02_correction_norm_distribution.png")
    plot_q_raw_vs_safe(steps, output_dir / "03_q_raw_vs_q_safe.png")
    plot_projection_buckets(steps, output_dir / "04_projection_buckets.png")
    plot_success_failure_split(summary, output_dir / "05_success_vs_fallback.png")
    plot_qp_success_q_values(summary, output_dir / "06_qp_success_q_values.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug CBF filter and actor/critic credit-assignment diagnostics.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/update_signal_ablation_10k"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--checkpoint-step", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=307_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-distance-m", type=float, default=None)
    parser.add_argument("--task-max-steps", type=int, default=None)
    parser.add_argument("--event-threshold", type=float, default=None)
    parser.add_argument("--k0", type=float, default=None)
    parser.add_argument("--k1", type=float, default=None)
    parser.add_argument("--eps-side", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()

    project_root = find_project_root(args.project_root or Path.cwd())
    source_dir = (project_root / args.source_dir).resolve() if not args.source_dir.is_absolute() else args.source_dir.resolve()
    output_dir = args.output_dir or (source_dir / "pipeline_diagnostics")
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
    namespace["GUIDED_CBF_ENABLE_PROJECTION_REPORTING"] = True
    install_minimal_guided_cbf(namespace)
    if "install_cbf_projection_reporting" in namespace:
        namespace["install_cbf_projection_reporting"]()
    install_safety_set_reward_wrapper(namespace)
    install_event_penalty_env(namespace)

    env_config = dict(run_config.get("env_config", namespace["ENV_CONFIG"]))
    if str(env_config.get("traffic_model", "")) == "mtm":
        deep_update(env_config, MTM_CONGESTED_UNCERTAIN_UPDATES.copy())
    reward_config = dict(run_config.get("reward_config") or make_reward_config(namespace, SAFETY_REWARD_TRIAL))
    reward_config["progress_reward_weight"] = float(
        run_config.get("progress_reward_weight", reward_config.get("progress_reward_weight", 0.0))
    )
    reward_config["safety_potential_eps_side"] = float(namespace["CBF_EPS_SIDE"])

    event_threshold = float(args.event_threshold if args.event_threshold is not None else run_config.get("event_threshold", 0.03))
    task_distance_m = float(args.task_distance_m if args.task_distance_m is not None else run_config.get("task_distance_m", 1000.0))
    task_max_steps = int(args.task_max_steps if args.task_max_steps is not None else run_config.get("task_max_steps", 1200))
    k0 = float(namespace["CBF_K0"])
    k1 = float(namespace["CBF_K1"])
    eps_side = float(namespace["CBF_EPS_SIDE"])

    print(
        "[pipeline-diagnostics] starting"
        f" source={source_dir}"
        f" output={output_dir}"
        f" variants={args.variants}"
        f" checkpoint={args.checkpoint_step:,}"
        f" episodes={args.episodes}"
        f" eps={eps_side:g}"
        f" threshold={event_threshold:g}",
        flush=True,
    )

    step_frames: list[pd.DataFrame] = []
    episode_frames: list[pd.DataFrame] = []
    for variant in args.variants:
        checkpoint_path = source_dir / variant / f"ckpt_{int(args.checkpoint_step):06d}.zip"
        if not checkpoint_path.exists():
            checkpoint_path = source_dir / variant / "model.zip"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {variant}: {checkpoint_path}")
        model = load_model_for_variant(namespace, variant, checkpoint_path, args.device)
        steps, episodes = evaluate_variant(
            namespace,
            model=model,
            variant=variant,
            checkpoint_step=int(args.checkpoint_step),
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
        )
        step_frames.append(steps)
        episode_frames.append(episodes)
        row = summarize_steps(steps, episodes).iloc[0]
        print(
            "[pipeline-diagnostics]"
            f" {variant}"
            f" return={row['return_mean']:.2f}"
            f" qp_fail={row['qp_failure_rate']:.2%}"
            f" fallback={row['fallback_rate']:.2%}"
            f" safe_bad={row['safe_infeasible_rate']:.2%}"
            f" du_p95={row['p95_correction_norm']:.3f}"
            f" q_safe_minus_raw={row['q_safe_minus_raw_mean']:.3f}"
            f" vertex={row['vertex_rate']:.2%}",
            flush=True,
        )

    steps_all = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()
    episodes_all = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()
    summary = summarize_steps(steps_all, episodes_all)

    steps_path = output_dir / "step_diagnostics.csv"
    episodes_path = output_dir / "episode_diagnostics.csv"
    summary_path = output_dir / "summary.csv"
    config_path = output_dir / "diagnostic_config.json"
    steps_all.to_csv(steps_path, index=False)
    episodes_all.to_csv(episodes_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_plots(steps_all, summary, output_dir)
    config_path.write_text(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "output_dir": str(output_dir),
                "variants": list(args.variants),
                "checkpoint_step": int(args.checkpoint_step),
                "episodes": int(args.episodes),
                "seed": int(args.seed),
                "env_config": env_config,
                "reward_config": reward_config,
                "k0": k0,
                "k1": k1,
                "eps_side": eps_side,
                "event_threshold": event_threshold,
                "task_distance_m": task_distance_m,
                "task_max_steps": task_max_steps,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"[pipeline-diagnostics] wrote {steps_path}", flush=True)
    print(f"[pipeline-diagnostics] wrote {episodes_path}", flush=True)
    print(f"[pipeline-diagnostics] wrote {summary_path}", flush=True)
    display_cols = [
        "variant",
        "return_mean",
        "event_intervention_rate",
        "mean_correction_norm",
        "p95_correction_norm",
        "qp_failure_rate",
        "fallback_rate",
        "safe_infeasible_rate",
        "raw_violation_rate",
        "inside_rate",
        "edge_rate",
        "vertex_rate",
        "fail_bucket_rate",
        "q_raw_mean",
        "q_safe_mean",
        "q_safe_minus_raw_mean",
        "q_safe_gt_raw_rate",
    ]
    print(summary[display_cols].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
