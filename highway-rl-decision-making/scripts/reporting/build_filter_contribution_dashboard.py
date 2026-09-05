from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


VARIANT_RUN_NAMES = {
    "ddpg": "ddpg",
    "ddpg_cbf_reward": "cbfr",
    "guided_ddpg_cbf": "guided",
}

VARIANT_LABELS = {
    "ddpg": "DDPG",
    "ddpg_cbf_reward": "CBF reward",
    "guided_ddpg_cbf": "Guided CBF",
}

VARIANT_COLORS = {
    "ddpg": "#1f77b4",
    "ddpg_cbf_reward": "#d62728",
    "guided_ddpg_cbf": "#2ca02c",
}

MODE_RUN_NAMES = {
    "raw_actor": "raw",
    "actor_cbf": "actor_cbf",
    "random_cbf": "random_cbf",
    "rule_cbf": "rule_cbf",
}

MODE_LABELS = {
    "raw_actor": "Raw actor",
    "actor_cbf": "Actor + CBF",
    "random_cbf": "Random + CBF",
    "rule_cbf": "Rule + CBF",
}

MODE_STYLES = {
    "raw_actor": "-",
    "actor_cbf": "--",
    "random_cbf": ":",
    "rule_cbf": "-.",
}

HEADLINE_KPIS: list[dict[str, Any]] = [
    {
        "tag": "01_performance/episode_return",
        "title": "Episode Return",
        "column": "return_mean",
        "direction": "higher",
    },
    {
        "tag": "01_performance/episode_length_steps",
        "title": "Episode Length",
        "column": "episode_length_steps_mean",
        "direction": "lower",
    },
    {
        "tag": "02_safety/ego_collisions_per_km",
        "title": "Ego Collisions Per km",
        "column": "ego_collisions_per_km_mean",
        "direction": "lower",
    },
    {
        "tag": "02_safety/h_min",
        "title": "Minimum h",
        "column": "h_min",
        "fallback_column": "h_min_mean",
        "direction": "higher",
    },
    {
        "tag": "02_safety/qp_failure_rate",
        "title": "QP Failure Rate",
        "column": "qp_failure_rate_mean",
        "direction": "lower",
    },
    {
        "tag": "03_efficiency/abs_speed_error_mps",
        "title": "Abs Speed Error",
        "column": "mean_abs_speed_error_mean",
        "direction": "lower",
    },
    {
        "tag": "03_efficiency/progress_rate_mps",
        "title": "Progress Rate",
        "column": "progress_rate_mps_mean",
        "direction": "higher",
    },
    {
        "tag": "04_control_filter/intervention_rate",
        "title": "Intervention Rate",
        "column": "event_intervention_rate_mean",
        "direction": "lower",
    },
    {
        "tag": "04_control_filter/correction_norm",
        "title": "Correction Norm",
        "column": "mean_correction_norm_mean",
        "direction": "lower",
    },
    {
        "tag": "05_traffic/neighbor_density_per_km",
        "title": "Neighbor Density",
        "column": "mean_neighbor_density_per_km_mean",
        "fallback_column": "mean_neighbor_constraints_mean",
        "direction": "context",
    },
]

DIAGNOSTIC_KPIS: list[dict[str, Any]] = [
    {
        "tag": "01_filter_lift/delta_return_filtered_minus_raw",
        "title": "Return Gain: Actor+CBF - Raw",
        "column": "delta_R_filtered_minus_raw",
        "direction": "higher means filter helps return",
    },
    {
        "tag": "01_filter_lift/delta_collisions_per_km_raw_minus_filtered",
        "title": "Collision Reduction: Raw - Actor+CBF",
        "column": "delta_Cpkm_raw_minus_filtered",
        "direction": "higher means filter removes collisions",
    },
    {
        "tag": "01_filter_lift/delta_speed_error_raw_minus_filtered",
        "title": "Speed Error Reduction: Raw - Actor+CBF",
        "column": "delta_Ev_raw_minus_filtered",
        "direction": "higher means filter improves speed tracking",
    },
    {
        "tag": "01_filter_lift/delta_lateral_error_raw_minus_filtered",
        "title": "Lateral Error Reduction: Raw - Actor+CBF",
        "column": "delta_Ey_raw_minus_filtered",
        "direction": "higher means filter improves lateral behavior",
    },
    {
        "tag": "02_filter_load/filtered_intervention_rate",
        "title": "Actor+CBF Intervention Rate",
        "column": "filtered_intervention_rate",
        "direction": "lower means actor needs less correction",
    },
    {
        "tag": "02_filter_load/filtered_correction_norm",
        "title": "Actor+CBF Correction Norm",
        "column": "filtered_correction_norm",
        "direction": "lower means smaller correction load",
    },
    {
        "tag": "02_filter_load/filtered_p95_correction_norm",
        "title": "Actor+CBF P95 Correction Norm",
        "column": "filtered_p95_correction_norm",
        "direction": "lower means fewer large corrections",
    },
    {
        "tag": "03_actor_vs_filter/delta_return_actor_cbf_minus_random_cbf",
        "title": "Actor+CBF Return - Random+CBF",
        "column": "delta_R_actor_cbf_minus_random_cbf",
        "direction": "higher means actor beats filter-only random actions",
    },
    {
        "tag": "03_actor_vs_filter/delta_collisions_random_cbf_minus_actor_cbf",
        "title": "Random+CBF Collisions - Actor+CBF",
        "column": "delta_C_random_cbf_minus_actor_cbf",
        "direction": "higher means actor is safer than random+filter",
    },
]

TRAIN_EPISODE_SOURCE_TAGS = {
    "return_mean": "01_task/episode_return",
    "episode_length_steps_mean": "01_task/episode_length_steps",
    "ego_collisions_per_km_mean": "02_safety/ego_collisions_per_km",
    "h_min": "02_safety/h_min",
    "qp_failure_rate_mean": "02_safety/qp_failure_rate",
    "mean_abs_speed_error_mean": "03_efficiency/abs_speed_deviation_mps",
    "event_intervention_rate_mean": "05_filter/intervention_rate",
    "mean_correction_norm_mean": "05_filter/correction_norm_mean",
    "mean_neighbor_density_per_km_mean": "06_traffic/neighbor_density_per_km",
    "_progress_distance_m": "03_efficiency/distance_traveled_m",
    "_progress_time_s": "01_task/episode_time_s",
}


def clean_value(value: Any) -> float | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if np.isfinite(scalar) else None


def read_training_episode_tensorboard(source_dir: Path) -> pd.DataFrame:
    candidate_roots = [
        source_dir / "tb" / "custom",
        source_dir.parent / "tb" / "custom",
    ]
    tb_root = next((root for root in candidate_roots if root.exists()), None)
    if tb_root is None:
        return pd.DataFrame()

    wanted_tags: dict[str, tuple[str, str]] = {}
    for variant in VARIANT_RUN_NAMES:
        for column, source_tag in TRAIN_EPISODE_SOURCE_TAGS.items():
            wanted_tags[f"train_episode/{variant}/{source_tag}"] = (variant, column)

    values_by_key: dict[tuple[str, int, str], list[float]] = {}
    for event_file in tb_root.rglob("events.out.tfevents*"):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for full_tag in set(accumulator.Tags().get("scalars", [])).intersection(wanted_tags):
            variant, column = wanted_tags[full_tag]
            for event in accumulator.Scalars(full_tag):
                values_by_key.setdefault((variant, int(event.step), column), []).append(float(event.value))

    if not values_by_key:
        return pd.DataFrame()

    rows_by_key: dict[tuple[str, int], dict[str, float | str]] = {}
    for (variant, step, column), values in values_by_key.items():
        key = (variant, step)
        rows_by_key.setdefault(key, {"variant": variant, "checkpoint_step": float(step)})
        rows_by_key[key][column] = float(np.mean(values))

    rows: list[dict[str, float | str]] = []
    for row in rows_by_key.values():
        distance = clean_value(row.get("_progress_distance_m"))
        time_s = clean_value(row.get("_progress_time_s"))
        if distance is not None and time_s is not None and time_s > 1e-9:
            row["progress_rate_mps_mean"] = float(distance / time_s)
        rows.append(row)

    frame = pd.DataFrame(rows)
    return frame.sort_values(["variant", "checkpoint_step"]).reset_index(drop=True)


def load_summary(source_dir: Path) -> pd.DataFrame:
    summary_path = source_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    summary["checkpoint_step"] = pd.to_numeric(summary["checkpoint_step"], errors="coerce")

    episodes_path = source_dir / "episodes.csv"
    if episodes_path.exists():
        episodes = pd.read_csv(episodes_path)
        episodes["checkpoint_step"] = pd.to_numeric(episodes["checkpoint_step"], errors="coerce")
        mean_columns = [
            "progress_rate_mps",
            "mean_neighbor_density_per_km",
        ]
        available = [column for column in mean_columns if column in episodes.columns]
        if available:
            aggregates = (
                episodes.groupby(["variant", "checkpoint_step", "mode"], dropna=False)[available]
                .mean(numeric_only=True)
                .reset_index()
                .rename(columns={column: f"{column}_mean" for column in available})
            )
            summary = summary.merge(aggregates, on=["variant", "checkpoint_step", "mode"], how="left")

    return summary.sort_values(["variant", "mode", "checkpoint_step"]).reset_index(drop=True)


def load_diagnostics(source_dir: Path) -> pd.DataFrame:
    diagnostics_path = source_dir / "diagnostics.csv"
    if not diagnostics_path.exists():
        return pd.DataFrame()
    diagnostics = pd.read_csv(diagnostics_path)
    diagnostics["checkpoint_step"] = pd.to_numeric(diagnostics["checkpoint_step"], errors="coerce")
    return diagnostics.sort_values(["variant", "checkpoint_step"]).reset_index(drop=True)


def row_value(row: pd.Series, spec: dict[str, Any]) -> float | None:
    value = clean_value(row.get(spec["column"]))
    if value is not None:
        return value
    fallback = spec.get("fallback_column")
    if fallback is None:
        return None
    return clean_value(row.get(fallback))


def write_tensorboard(
    during_training: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_text = "\n".join(
        [
            "# Filter Contribution Dashboard",
            "",
            "Runs are policy/execution-mode pairs; tags are shared, so TensorBoard overlays comparable lines on one KPI chart.",
            "",
            "Use `during_training/*` for real episode-level KPI traces emitted while each policy was learning.",
            "Use `checkpoint/*` for the 10 headline KPIs over checkpoint.",
            "Use `diagnostics/*` for raw-vs-filtered and actor-vs-random-CBF attribution.",
            "QP failure rate may be absent under `during_training/*` when the training episode stream did not record it; it remains available under checkpoint attribution.",
            "",
            "| Section | KPI | Direction |",
            "|---|---|---|",
            *[
                f"| during_training | `{spec['tag']}` | {spec['direction']} |"
                for spec in HEADLINE_KPIS
            ],
            *[
                f"| checkpoint | `{spec['tag']}` | {spec['direction']} |"
                for spec in HEADLINE_KPIS
            ],
            *[
                f"| diagnostics | `{spec['tag']}` | {spec['direction']} |"
                for spec in DIAGNOSTIC_KPIS
            ],
        ]
    )

    for variant, frame in during_training.groupby("variant", dropna=False):
        variant = str(variant)
        run_name = f"training/{VARIANT_RUN_NAMES.get(variant, variant)}"
        writer = SummaryWriter(log_dir=str(output_dir / run_name))
        for _, row in frame.sort_values("checkpoint_step").iterrows():
            step = int(clean_value(row.get("checkpoint_step")) or 0)
            for spec in HEADLINE_KPIS:
                value = row_value(row, spec)
                if value is not None:
                    writer.add_scalar(f"during_training/{spec['tag']}", value, step)
        writer.add_text("00_index", index_text, 0)
        writer.flush()
        writer.close()

    for (variant, mode), frame in summary.groupby(["variant", "mode"], dropna=False):
        variant = str(variant)
        mode = str(mode)
        run_name = f"{VARIANT_RUN_NAMES.get(variant, variant)}/{MODE_RUN_NAMES.get(mode, mode)}"
        writer = SummaryWriter(log_dir=str(output_dir / run_name))
        for _, row in frame.sort_values("checkpoint_step").iterrows():
            step = int(clean_value(row.get("checkpoint_step")) or 0)
            for spec in HEADLINE_KPIS:
                value = row_value(row, spec)
                if value is not None:
                    writer.add_scalar(f"checkpoint/{spec['tag']}", value, step)
        writer.add_text("00_index", index_text, 0)
        writer.flush()
        writer.close()

    for variant, frame in diagnostics.groupby("variant", dropna=False):
        variant = str(variant)
        run_name = f"diagnostics/{VARIANT_RUN_NAMES.get(variant, variant)}"
        writer = SummaryWriter(log_dir=str(output_dir / run_name))
        for _, row in frame.sort_values("checkpoint_step").iterrows():
            step = int(clean_value(row.get("checkpoint_step")) or 0)
            for spec in DIAGNOSTIC_KPIS:
                value = clean_value(row.get(spec["column"]))
                if value is not None:
                    writer.add_scalar(f"diagnostics/{spec['tag']}", value, step)
        writer.add_text("00_index", index_text, 0)
        writer.flush()
        writer.close()


def plot_during_training(during_training: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(14, 17), sharex=True)
    for axis, spec in zip(axes.ravel(), HEADLINE_KPIS):
        any_data = False
        for variant in VARIANT_RUN_NAMES:
            frame = during_training[during_training["variant"] == variant].sort_values("checkpoint_step")
            if frame.empty or spec["column"] not in frame.columns:
                continue
            values = pd.to_numeric(frame[spec["column"]], errors="coerce")
            if values.notna().sum() == 0:
                continue
            smoothed = values.rolling(window=25, min_periods=1).mean()
            axis.plot(
                frame["checkpoint_step"],
                smoothed,
                linewidth=1.8,
                color=VARIANT_COLORS.get(variant),
                label=VARIANT_LABELS.get(variant, variant),
            )
            any_data = True
        axis.set_title(spec["title"])
        axis.set_xlabel("Training timestep")
        axis.grid(True, alpha=0.25)
        if not any_data:
            axis.text(0.5, 0.5, "not in explicit training stream", ha="center", va="center", transform=axis.transAxes)
    axes[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle("CBF Isolation Study: KPIs During Policy Training", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_headline(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(14, 17), sharex=True)
    for axis, spec in zip(axes.ravel(), HEADLINE_KPIS):
        for variant in VARIANT_RUN_NAMES:
            for mode in MODE_RUN_NAMES:
                frame = summary[(summary["variant"] == variant) & (summary["mode"] == mode)].sort_values(
                    "checkpoint_step"
                )
                if frame.empty:
                    continue
                values = [row_value(row, spec) for _, row in frame.iterrows()]
                if all(value is None for value in values):
                    continue
                label = f"{VARIANT_LABELS.get(variant, variant)} / {MODE_LABELS.get(mode, mode)}"
                axis.plot(
                    frame["checkpoint_step"],
                    values,
                    marker="o",
                    linewidth=2.0,
                    markersize=4,
                    color=VARIANT_COLORS.get(variant),
                    linestyle=MODE_STYLES.get(mode, "-"),
                    label=label,
                )
        axis.set_title(spec["title"])
        axis.set_xlabel("Checkpoint timestep")
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=7)
    fig.suptitle("CBF Attribution: Headline KPIs By Checkpoint", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_diagnostics(diagnostics: pd.DataFrame, output_path: Path) -> None:
    if diagnostics.empty:
        return
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
    for axis, spec in zip(axes.ravel(), DIAGNOSTIC_KPIS):
        for variant in VARIANT_RUN_NAMES:
            frame = diagnostics[diagnostics["variant"] == variant].sort_values("checkpoint_step")
            if frame.empty or spec["column"] not in frame.columns:
                continue
            axis.plot(
                frame["checkpoint_step"],
                frame[spec["column"]],
                marker="o",
                linewidth=2.0,
                markersize=4,
                color=VARIANT_COLORS.get(variant),
                label=VARIANT_LABELS.get(variant, variant),
            )
        axis.set_title(spec["title"])
        axis.set_xlabel("Checkpoint timestep")
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("CBF Attribution Diagnostics", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_clean_csvs(
    during_training: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    training_cols = ["variant", "checkpoint_step"]
    for spec in HEADLINE_KPIS:
        for column in [spec["column"], spec.get("fallback_column")]:
            if column and column in during_training.columns and column not in training_cols:
                training_cols.append(column)
    headline_cols = ["variant", "checkpoint_step", "mode", "mode_label"]
    for spec in HEADLINE_KPIS:
        for column in [spec["column"], spec.get("fallback_column")]:
            if column and column in summary.columns and column not in headline_cols:
                headline_cols.append(column)
    during_training[training_cols].to_csv(output_dir / "during_training.csv", index=False)
    summary[headline_cols].to_csv(output_dir / "headline.csv", index=False)
    if not diagnostics.empty:
        diagnostics.to_csv(output_dir / "diagnostics.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean TensorBoard dashboard for the CBF attribution study.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("artifacts/sp3_mtm_e0p1_p0p0_kpi_full/filter_policy_contribution"),
    )
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("artifacts/tb_filter_contribution_clean"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = (args.output_dir or source_dir / "headline_kpis").resolve()
    tb_dir = args.tensorboard_dir.resolve()

    during_training = read_training_episode_tensorboard(source_dir)
    summary = load_summary(source_dir)
    diagnostics = load_diagnostics(source_dir)
    write_tensorboard(during_training, summary, diagnostics, tb_dir)
    plot_during_training(during_training, output_dir / "during_training.png")
    plot_headline(summary, output_dir / "headline.png")
    plot_diagnostics(diagnostics, output_dir / "diagnostics.png")
    write_clean_csvs(during_training, summary, diagnostics, output_dir)

    print(f"[filter-dashboard] tensorboard={tb_dir}")
    print(f"[filter-dashboard] output={output_dir}")
    print(f"[filter-dashboard] during_training_plot={output_dir / 'during_training.png'}")
    print(f"[filter-dashboard] headline_plot={output_dir / 'headline.png'}")
    print(f"[filter-dashboard] diagnostics_plot={output_dir / 'diagnostics.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
