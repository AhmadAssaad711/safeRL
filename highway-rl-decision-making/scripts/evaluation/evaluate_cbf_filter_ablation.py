from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DDPG

from scripts.evaluation.evaluate_laneless_karalakou import (
    TEN_KPI_SPECS,
    apply_cbf_overrides,
    exec_notebook_cells,
    find_project_root,
    set_stable_native_defaults,
)


POLICIES: dict[str, dict[str, str]] = {
    "DDPG without CBF": {"folder": "DDPG", "action_mode": "normalized", "run_id": "canonical_20260714"},
    "DDPG-CBF reward": {"folder": "DDPG_CBF_Reward", "action_mode": "physical", "run_id": "20260715_085620"},
    "DDPG-CBF reward + loss": {"folder": "DDPG_CBF_Reward_Loss", "action_mode": "physical", "run_id": "20260715_092430"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired test-time CBF filter ablation for the saved DDPG models.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=120_007)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k0", type=float, default=5.29)
    parser.add_argument("--k1", type=float, default=3.68)
    parser.add_argument("--eps-side", type=float, default=0.10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _normalised_to_physical(env: Any, action: np.ndarray) -> np.ndarray:
    base = env.unwrapped
    bounds = base.config["bounds"]
    values = np.asarray(action, dtype=float).reshape(-1)[:2]
    output: list[float] = []
    for value, low_key, high_key in zip(values, ("ax_min", "ay_min"), ("ax_max", "ay_max")):
        low = float(bounds[low_key])
        high = float(bounds[high_key])
        value = float(np.clip(value, -1.0, 1.0))
        if low < 0.0 < high:
            output.append(value * (high if value >= 0.0 else abs(low)))
        else:
            output.append(low + 0.5 * (value + 1.0) * (high - low))
    return np.asarray(output, dtype=np.float32)


def _physical_to_normalised(env: Any, action: np.ndarray) -> np.ndarray:
    base = env.unwrapped
    bounds = base.config["bounds"]
    values = np.asarray(action, dtype=float).reshape(-1)[:2]
    output: list[float] = []
    for value, low_key, high_key in zip(values, ("ax_min", "ay_min"), ("ax_max", "ay_max")):
        low = float(bounds[low_key])
        high = float(bounds[high_key])
        value = float(np.clip(value, low, high))
        if low < 0.0 < high:
            output.append(value / (high if value >= 0.0 else abs(low)))
        else:
            output.append(2.0 * (value - low) / max(high - low, 1e-6) - 1.0)
    return np.clip(np.asarray(output, dtype=np.float32), -1.0, 1.0)


def _make_ablation_env(namespace: dict[str, Any], seed: int):
    env = namespace["make_single_env"](
        seed=seed,
        render_mode=None,
        env_config=namespace["ENV_CONFIG"],
        reward_config=namespace["REWARD_CONFIG"],
        normalize_observation=bool(namespace.get("NORMALIZE_RL_OBSERVATIONS", False)),
    )
    if bool(namespace.get("USE_DISTANCE_TASK_EVALUATION", True)):
        env = namespace["make_task_evaluation_wrapper"](
            env,
            task_distance_m=float(namespace["TASK_DISTANCE_M"]),
            max_steps=int(namespace["TASK_MAX_STEPS"]),
        )
    else:
        namespace["configure_paper_evaluation_env"](env, steps=int(namespace["PAPER_EVAL_STEPS"]))
    return env


def _set_filter_info(
    info: dict[str, Any],
    *,
    raw_action: np.ndarray,
    safe_action: np.ndarray,
    filter_info: dict[str, Any],
    filter_enabled: bool,
    intervention_threshold: float,
) -> dict[str, Any]:
    shadow_correction = float(np.linalg.norm(np.asarray(safe_action) - np.asarray(raw_action)))
    applied_correction = shadow_correction if filter_enabled else 0.0
    applied_intervened = bool(filter_enabled and applied_correction > 1e-6)
    applied_meaningful = bool(filter_enabled and applied_correction > intervention_threshold)
    qp_success = bool(filter_info.get("qp_success", True))
    min_h = float(filter_info.get("min_h", np.nan))
    boundary_h = float(filter_info.get("min_boundary_h", np.nan))
    info = dict(info)
    info.update(
        {
            "cbf_a_rl_x": float(raw_action[0]),
            "cbf_a_rl_y": float(raw_action[1]),
            "cbf_a_safe_x": float(safe_action[0] if filter_enabled else raw_action[0]),
            "cbf_a_safe_y": float(safe_action[1] if filter_enabled else raw_action[1]),
            "cbf_correction_norm": applied_correction,
            "cbf_intervened": applied_intervened,
            "cbf_event_intervened": applied_meaningful,
            "cbf_qp_success": qp_success,
            "cbf_min_h": min_h,
            "cbf_min_boundary_h": boundary_h,
            "cbf_raw_feasible": bool(filter_info.get("raw_feasible", False)),
            "kpi_correction_norm": applied_correction,
            "kpi_meaningful_correction_norm": max(applied_correction - intervention_threshold, 0.0),
            "kpi_meaningful_intervention": float(applied_meaningful),
            "kpi_numerical_intervention": float(applied_intervened),
            "kpi_raw_safe_gap_norm": applied_correction,
            "kpi_qp_attempt": float(filter_enabled),
            "kpi_qp_failure": float(filter_enabled and not qp_success),
            "kpi_h_min": min_h,
            "kpi_boundary_h_min": boundary_h,
        }
    )
    return info


def evaluate_condition(
    namespace: dict[str, Any],
    model: Any,
    *,
    policy: str,
    action_mode: str,
    filter_enabled: bool,
    episodes: int,
    seed: int,
    k0: float,
    k1: float,
    eps_side: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_neighbors = namespace.get("CBF_MAX_NEIGHBOR_CONSTRAINTS")
    intervention_threshold = float(namespace.get("KPI_DEFAULT_INTERVENTION_THRESHOLD", 0.03))
    for episode in range(int(episodes)):
        episode_seed = int(seed) + episode
        env = _make_ablation_env(namespace, episode_seed)
        obs, _ = env.reset(seed=episode_seed)
        done = False
        rewards: list[float] = []
        info_rows: list[dict[str, Any]] = []
        shadow_corrections: list[float] = []
        shadow_interventions: list[float] = []
        shadow_raw_feasible: list[float] = []
        shadow_qp_failures: list[float] = []
        shadow_min_h: list[float] = []
        last_task_info: dict[str, Any] = {}
        steps = 0

        while not done:
            predicted_action, _ = model.predict(obs, deterministic=True)
            if action_mode == "normalized":
                raw_physical = _normalised_to_physical(env, predicted_action)
            else:
                raw_physical = np.asarray(predicted_action, dtype=np.float32).reshape(-1)[:2]
                raw_physical = np.clip(
                    raw_physical,
                    [float(namespace["CBF_AX_BOUNDS"][0]), float(namespace["CBF_AY_BOUNDS"][0])],
                    [float(namespace["CBF_AX_BOUNDS"][1]), float(namespace["CBF_AY_BOUNDS"][1])],
                )

            ego = namespace["get_ego_state"](env)
            neighbors = namespace["get_neighbor_states"](env, neighbor_range=float(namespace["CBF_NEIGHBOR_RANGE"]))
            safe_physical, filter_info = namespace["cbf_filter_2d"](
                raw_physical,
                ego,
                neighbors,
                float(env.unwrapped.config["road_width"]),
                ax_bounds=namespace["CBF_AX_BOUNDS"],
                ay_bounds=namespace["CBF_AY_BOUNDS"],
                eps_side=float(eps_side),
                k0=float(k0),
                k1=float(k1),
                max_neighbor_constraints=max_neighbors,
            )
            shadow_correction = float(np.linalg.norm(np.asarray(safe_physical) - np.asarray(raw_physical)))
            shadow_corrections.append(shadow_correction)
            shadow_interventions.append(float(shadow_correction > 1e-6))
            shadow_raw_feasible.append(float(bool(filter_info.get("raw_feasible", False))))
            shadow_qp_failures.append(float(not bool(filter_info.get("qp_success", True))))
            if np.isfinite(float(filter_info.get("min_h", np.nan))):
                shadow_min_h.append(float(filter_info["min_h"]))

            executed_physical = safe_physical if filter_enabled else raw_physical
            normalized_action = _physical_to_normalised(env, executed_physical)
            obs, reward, terminated, truncated, info = env.step(normalized_action)
            info = _set_filter_info(
                info,
                raw_action=raw_physical,
                safe_action=safe_physical,
                filter_info=filter_info,
                filter_enabled=filter_enabled,
                intervention_threshold=intervention_threshold,
            )
            info_rows.append(info)
            rewards.append(float(reward))
            last_task_info = {
                "task_distance_m": float(info.get("task_distance_m", namespace["TASK_DISTANCE_M"])),
                "task_distance_traveled_m": float(info.get("task_distance_traveled_m", 0.0)),
                "task_progress_ratio": float(info.get("task_progress_ratio", 0.0)),
                "task_completed": bool(info.get("task_completed", False)),
                "task_timeout": bool(info.get("task_timeout", False)),
            }
            steps += 1
            done = bool(terminated or truncated)

        summary = namespace["summarize_episode_kpis"](
            info_rows,
            rewards=rewards,
            task_completed=bool(last_task_info.get("task_completed", False)),
            fallback_steps=steps,
            fallback_distance_m=float(last_task_info.get("task_distance_traveled_m", 0.0)),
            fallback_dt_s=namespace["kpi_policy_dt"](env),
        )
        summary.update(
            {
                "policy": policy,
                "filter_enabled": int(filter_enabled),
                "filter_label": "CBF ON" if filter_enabled else "CBF OFF",
                "episode": episode,
                "seed": episode_seed,
                "action_mode": action_mode,
                "episode_length_steps": float(steps),
                "episode_return": float(np.sum(rewards)),
                "return": float(np.sum(rewards)),
                "task_completed": float(last_task_info.get("task_completed", False)),
                "task_timeout": float(last_task_info.get("task_timeout", False)),
                "task_distance_m": float(last_task_info.get("task_distance_m", namespace["TASK_DISTANCE_M"])),
                "task_distance_traveled_m": float(last_task_info.get("task_distance_traveled_m", 0.0)),
                "task_progress_ratio": float(last_task_info.get("task_progress_ratio", 0.0)),
                "shadow_intervention_rate": float(np.mean(shadow_interventions)) if shadow_interventions else 0.0,
                "shadow_mean_correction_norm": float(np.mean(shadow_corrections)) if shadow_corrections else 0.0,
                "shadow_max_correction_norm": float(np.max(shadow_corrections)) if shadow_corrections else 0.0,
                "shadow_raw_feasible_rate": float(np.mean(shadow_raw_feasible)) if shadow_raw_feasible else 0.0,
                "shadow_qp_failure_rate": float(np.mean(shadow_qp_failures)) if shadow_qp_failures else 0.0,
                "shadow_min_h": float(np.min(shadow_min_h)) if shadow_min_h else np.nan,
                "applied_intervention_rate": float(np.mean(shadow_interventions)) if filter_enabled and shadow_interventions else 0.0,
                "applied_mean_correction_norm": float(np.mean(shadow_corrections)) if filter_enabled and shadow_corrections else 0.0,
                "applied_qp_failure_rate": float(np.mean(shadow_qp_failures)) if filter_enabled and shadow_qp_failures else 0.0,
            }
        )
        rows.append(summary)
        env.close()
    return rows


def summarise_results(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "episode_return",
        "task_completed",
        "mean_abs_speed_deviation",
        "ego_collisions_per_km",
        "total_collision_events_per_km",
        "mean_jerk_norm",
        "action_saturation_rate",
        "mean_correction_norm",
        "event_intervention_rate",
        "qp_failure_rate",
        "h_min",
        "shadow_intervention_rate",
        "shadow_mean_correction_norm",
        "shadow_qp_failure_rate",
    ]
    grouped = episodes.groupby(["policy", "filter_label"], sort=False)
    rows: list[dict[str, Any]] = []
    for (policy, filter_label), group in grouped:
        row: dict[str, Any] = {"policy": policy, "filter_label": filter_label, "episodes": int(len(group))}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci95"] = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)

    paired_metrics = [
        "episode_return",
        "task_completed",
        "mean_abs_speed_deviation",
        "ego_collisions_per_km",
        "total_collision_events_per_km",
        "mean_jerk_norm",
        "action_saturation_rate",
        "h_min",
    ]
    index = ["policy", "episode", "seed"]
    wide = episodes.pivot_table(index=index, columns="filter_label", values=paired_metrics, aggfunc="first")
    paired_rows: list[dict[str, Any]] = []
    for (policy, episode, seed), row in wide.iterrows():
        paired: dict[str, Any] = {"policy": policy, "episode": int(episode), "seed": int(seed)}
        for metric in paired_metrics:
            off = row.get((metric, "CBF OFF"), np.nan)
            on = row.get((metric, "CBF ON"), np.nan)
            paired[f"{metric}_off"] = float(off) if pd.notna(off) else np.nan
            paired[f"{metric}_on"] = float(on) if pd.notna(on) else np.nan
            paired[f"{metric}_delta_on_minus_off"] = float(on - off) if pd.notna(on) and pd.notna(off) else np.nan
        paired_rows.append(paired)
    paired = pd.DataFrame(paired_rows)
    return summary, paired


PART1_POLICY = "DDPG without CBF"


def _policy_dt_from_env_config(env_config: dict[str, Any]) -> float:
    dt = float(env_config.get("dt", 0.05))
    simulation_frequency = float(env_config.get("simulation_frequency", 1.0))
    policy_frequency = float(env_config.get("policy_frequency", simulation_frequency))
    return float(dt * simulation_frequency / max(policy_frequency, 1e-9))


def part1_ten_kpi_table(episodes: pd.DataFrame, *, policy_dt_s: float) -> pd.DataFrame:
    """Return the two deployment rows for the frozen nominal-DDPG Part 1 test."""

    working = episodes.loc[episodes["policy"].astype(str) == PART1_POLICY].copy()
    if working.empty:
        raise ValueError(f"Part 1 policy {PART1_POLICY!r} is absent from the evaluation rows")
    if "episode_length_steps" not in working and "episode_time_s" in working:
        working["episode_length_steps"] = (
            pd.to_numeric(working["episode_time_s"], errors="coerce") / max(float(policy_dt_s), 1e-9)
        )

    missing = [column for _, column in TEN_KPI_SPECS if column not in working]
    if missing:
        raise KeyError(f"Part 1 evaluation is missing required KPI columns: {missing}")

    rows: list[dict[str, Any]] = []
    for deployment, group in working.groupby("filter_label", sort=False):
        row: dict[str, Any] = {
            "policy": PART1_POLICY,
            "deployment": str(deployment),
            "episodes": int(len(group)),
        }
        for _, column in TEN_KPI_SPECS:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)

    table = pd.DataFrame(rows)
    order = {"CBF OFF": 0, "CBF ON": 1}
    return table.assign(_deployment_order=table["deployment"].map(order).fillna(99)).sort_values(
        "_deployment_order"
    ).drop(columns="_deployment_order").reset_index(drop=True)


def part1_inline_kpi_table(kpi_table: pd.DataFrame) -> pd.DataFrame:
    """Make the Part 1 result readable in a notebook or terminal without opening a CSV."""

    by_deployment = kpi_table.set_index("deployment")
    rows: list[dict[str, str]] = []
    for label, column in TEN_KPI_SPECS:
        row = {"KPI": label}
        for deployment in ("CBF OFF", "CBF ON"):
            if deployment not in by_deployment.index:
                row[deployment] = "not evaluated"
                continue
            result = by_deployment.loc[deployment]
            row[deployment] = f"{float(result[f'{column}_mean']):.3f} +/- {float(result[f'{column}_std']):.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def write_and_print_part1_kpis(
    episodes: pd.DataFrame,
    *,
    output_path: Path,
    policy_dt_s: float,
) -> pd.DataFrame:
    table = part1_ten_kpi_table(episodes, policy_dt_s=policy_dt_s)
    table.to_csv(output_path, index=False)
    print(
        "[cbf-ablation] Part 1: frozen nominal DDPG, deployed raw versus CBF (mean +/- sample SD)",
        flush=True,
    )
    print(part1_inline_kpi_table(table).to_string(index=False), flush=True)
    return table


def make_figures(episodes: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    policy_order = list(POLICIES)
    colors = {"DDPG without CBF": "#4472C4", "DDPG-CBF reward": "#ED7D31", "DDPG-CBF reward + loss": "#70AD47"}
    short_labels = ["DDPG", "CBF reward", "CBF reward + loss"]
    metric_specs = [
        ("episode_return", "Mean episode return", "Return", "{:.2f}"),
        ("mean_abs_speed_deviation", "Speed-tracking error", "m/s", "{:.2f}"),
        ("ego_collisions_per_km", "Ego collisions", "collisions/km", "{:.2f}"),
        ("mean_jerk_norm", "Mean jerk", "m/s³", "{:.2f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), constrained_layout=True)
    fig.suptitle("Test-Time CBF Filter Ablation", fontsize=17, fontweight="bold")
    for axis, (metric, title, ylabel, fmt) in zip(axes.flat, metric_specs):
        x = np.arange(len(policy_order))
        width = 0.34
        off = summary[summary["filter_label"] == "CBF OFF"].set_index("policy").loc[policy_order]
        on = summary[summary["filter_label"] == "CBF ON"].set_index("policy").loc[policy_order]
        off_values = off[f"{metric}_mean"].astype(float).to_numpy()
        on_values = on[f"{metric}_mean"].astype(float).to_numpy()
        off_bars = axis.bar(x - width / 2, off_values, width, label="CBF OFF", color="#B8C4D6", edgecolor="white", linewidth=1.0)
        on_bars = axis.bar(x + width / 2, on_values, width, label="CBF ON", color=[colors[p] for p in policy_order], edgecolor="white", linewidth=1.0)
        axis.bar_label(off_bars, labels=[fmt.format(v) for v in off_values], padding=3, fontsize=8)
        axis.bar_label(on_bars, labels=[fmt.format(v) for v in on_values], padding=3, fontsize=8)
        axis.set_xticks(x, short_labels, rotation=8)
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        top = max(float(np.nanmax(np.r_[off_values, on_values])), 1e-6)
        axis.set_ylim(0.0, top * 1.22)
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")
    fig.savefig(output_dir / "performance_bars.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    activity = summary[summary["filter_label"] == "CBF ON"].set_index("policy").loc[policy_order]
    activity_specs = [
        ("shadow_intervention_rate_mean", "Would-intervene rate", "rate"),
        ("shadow_mean_correction_norm_mean", "Mean shadow correction", "physical action norm"),
        ("shadow_qp_failure_rate_mean", "QP failure rate", "rate"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    fig.suptitle("CBF Filter Activity (Shadow Diagnostics)", fontsize=16, fontweight="bold")
    for axis, (column, title, ylabel) in zip(axes, activity_specs):
        values = activity[column].astype(float).to_numpy()
        bars = axis.bar(short_labels, values, color=[colors[p] for p in policy_order], edgecolor="white", linewidth=1.0)
        axis.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, fontsize=9)
        axis.set_title(title, fontweight="semibold")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=8)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylim(0.0, max(float(np.nanmax(values)) * 1.22, 1e-6))
    fig.savefig(output_dir / "filter_activity_bars.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    set_stable_native_defaults()
    args = parse_args()
    project_root = find_project_root(args.project_root)
    os.chdir(project_root)
    artifact_dir = project_root / "artifacts" / "lanelessKaralakou"
    output_dir = artifact_dir / "PaperResults" / "CBF_Filter_Ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = {
        policy: output_dir.parent / spec["folder"] / "model.zip"
        for policy, spec in POLICIES.items()
    }
    model_hashes = {policy: sha256(path) for policy, path in model_paths.items()}
    configuration = {
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "deterministic": True,
        "evaluation_reward": "common Karalakou reward; filter penalty disabled during ablation",
        "filter_parameters": {"k0": float(args.k0), "k1": float(args.k1), "eps_side": float(args.eps_side)},
        "models": {policy: {"run_id": POLICIES[policy]["run_id"], "path": str(path.resolve()), "sha256": model_hashes[policy], "action_mode": POLICIES[policy]["action_mode"]} for policy, path in model_paths.items()},
    }
    manifest_path = output_dir / "manifest.json"
    episode_path = output_dir / "episode_metrics.csv"
    summary_path = output_dir / "summary.csv"
    paired_path = output_dir / "paired_deltas.csv"
    part1_kpi_path = output_dir / "part1_kpi_10.csv"

    namespace: dict[str, Any] = {"__name__": "__main__", "DDPG": DDPG}
    override_args = argparse.Namespace(lambda_filter=0.0, k0=args.k0, k1=args.k1, eps_side=args.eps_side)
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    exec_notebook_cells(notebook_path, [2, 3, 5, 6, 8, 33, 35, 37, 39], namespace, override_args)
    apply_cbf_overrides(namespace, override_args)
    policy_dt_s = _policy_dt_from_env_config(namespace["ENV_CONFIG"])
    # A result is valid only for the exact evaluation environment.  This also
    # prevents a legacy force-traffic cache from being reused after switching
    # the canonical notebook default to MTM.
    configuration["pipeline_schema_version"] = 2
    configuration["environment_config"] = namespace["ENV_CONFIG"]

    if not args.force and manifest_path.exists() and episode_path.exists() and summary_path.exists() and paired_path.exists():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached.get("configuration") == configuration:
                cached_episodes = pd.read_csv(episode_path)
                write_and_print_part1_kpis(
                    cached_episodes,
                    output_path=part1_kpi_path,
                    policy_dt_s=policy_dt_s,
                )
                print(f"[cbf-ablation] using cached results in {output_dir}", flush=True)
                return 0
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            print("[cbf-ablation] cached result lacks the standard ten-KPI schema; reevaluating", flush=True)

    all_rows: list[dict[str, Any]] = []
    for policy, spec in POLICIES.items():
        print(f"[cbf-ablation] loading {policy}: {model_paths[policy]}", flush=True)
        model = DDPG.load(str(model_paths[policy]), device=args.device)
        for filter_enabled in (False, True):
            label = "CBF ON" if filter_enabled else "CBF OFF"
            print(f"[cbf-ablation] evaluating {policy} | {label} | {args.episodes} episodes", flush=True)
            all_rows.extend(
                evaluate_condition(
                    namespace,
                    model,
                    policy=policy,
                    action_mode=spec["action_mode"],
                    filter_enabled=filter_enabled,
                    episodes=args.episodes,
                    seed=args.seed,
                    k0=args.k0,
                    k1=args.k1,
                    eps_side=args.eps_side,
                )
            )
        del model

    episode_frame = pd.DataFrame(all_rows)
    summary, paired = summarise_results(episode_frame)
    episode_frame.to_csv(episode_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    write_and_print_part1_kpis(
        episode_frame,
        output_path=part1_kpi_path,
        policy_dt_s=policy_dt_s,
    )
    make_figures(episode_frame, summary, output_dir)
    manifest = {
        "created_at": pd.Timestamp.now().isoformat(),
        "configuration": configuration,
        "output_files": {
            "episode_metrics": str(episode_path.resolve()),
            "summary": str(summary_path.resolve()),
            "paired_deltas": str(paired_path.resolve()),
            "part1_kpi_10": str(part1_kpi_path.resolve()),
            "performance_figure": str((output_dir / "performance_bars.png").resolve()),
            "activity_figure": str((output_dir / "filter_activity_bars.png").resolve()),
        },
        "notes": [
            "CBF OFF passes each policy's raw physical action to the common base environment.",
            "CBF ON applies the tuned no-slack CBF-QP filter before executing the action.",
            "CBF diagnostics are computed in shadow for both conditions; shadow actions are not applied in CBF OFF.",
            "The CBF reward penalty is disabled during this ablation so returns use the same common evaluation reward.",
            "The same episode seed is used for CBF OFF and CBF ON for every policy, enabling paired deltas.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[cbf-ablation] saved results to {output_dir}", flush=True)
    print(summary[["policy", "filter_label", "episodes", "episode_return_mean", "task_completed_mean", "ego_collisions_per_km_mean", "mean_jerk_norm_mean"]].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
