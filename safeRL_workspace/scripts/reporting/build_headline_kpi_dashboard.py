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


RUN_NAMES = {
    "ddpg": "ddpg",
    "ddpg_cbf_reward": "cbfr",
    "guided_ddpg_cbf": "guided",
}

LABELS = {
    "ddpg": "DDPG",
    "ddpg_cbf_reward": "CBF reward",
    "guided_ddpg_cbf": "Guided CBF",
}

COLORS = {
    "ddpg": "#1f77b4",
    "ddpg_cbf_reward": "#d62728",
    "guided_ddpg_cbf": "#2ca02c",
}

HEADLINE_KPIS: list[dict[str, Any]] = [
    {
        "tag": "01_performance/episode_return",
        "title": "Episode Return",
        "column": "return_mean",
        "final_column": "return_mean",
        "direction": "higher",
    },
    {
        "tag": "01_performance/episode_length_steps",
        "title": "Episode Length",
        "column": "episode_length_steps_mean",
        "final_column": "episode_length_steps_mean",
        "direction": "lower",
    },
    {
        "tag": "02_safety/ego_collisions_per_km",
        "title": "Ego Collisions Per km",
        "column": "ego_collisions_per_km_mean",
        "final_column": "ego_collisions_per_km_mean",
        "direction": "lower",
    },
    {
        "tag": "02_safety/h_min",
        "title": "Minimum h",
        "column": "h_min",
        "final_column": "h_min",
        "direction": "higher",
    },
    {
        "tag": "02_safety/qp_failure_rate",
        "title": "QP Failure Rate",
        "column": "qp_failure_rate",
        "final_column": "qp_failure_rate",
        "direction": "lower",
    },
    {
        "tag": "03_efficiency/abs_speed_error_mps",
        "title": "Abs Speed Error",
        "column": "mean_abs_speed_error",
        "final_column": "mean_abs_speed_error",
        "direction": "lower",
    },
    {
        "tag": "03_efficiency/progress_rate_mps",
        "title": "Progress Rate",
        "column": "progress_rate_mps_mean",
        "final_column": "progress_rate_mps_mean",
        "direction": "higher",
    },
    {
        "tag": "04_control_filter/intervention_rate",
        "title": "Intervention Rate",
        "column": "event_intervention_rate",
        "final_column": "event_intervention_rate",
        "direction": "lower",
    },
    {
        "tag": "04_control_filter/correction_norm",
        "title": "Correction Norm",
        "column": "mean_correction_norm",
        "final_column": "mean_correction_norm",
        "direction": "lower",
    },
    {
        "tag": "05_traffic/neighbor_density_per_km",
        "title": "Neighbor Density",
        "column": "mean_neighbor_density_per_km",
        "final_column": "mean_neighbor_density_per_km",
        "direction": "context",
    },
]

SOURCE_EVAL_TAGS = {
    "return_mean": "01_task/episode_return",
    "episode_length_steps_mean": "01_task/episode_length_steps",
    "ego_collisions_per_km_mean": "02_safety/ego_collisions_per_km",
    "h_min": "02_safety/h_min",
    "qp_failure_rate": "02_safety/qp_failure_rate",
    "mean_abs_speed_error": "03_efficiency/abs_speed_deviation_mps",
    "progress_rate_mps_mean": "03_efficiency/progress_rate_mps",
    "event_intervention_rate": "05_filter/intervention_rate",
    "mean_correction_norm": "05_filter/correction_norm_mean",
    "mean_neighbor_density_per_km": "06_traffic/neighbor_density_per_km",
}

TRAIN_EPISODE_SOURCE_TAGS = {
    "return_mean": "01_task/episode_return",
    "episode_length_steps_mean": "01_task/episode_length_steps",
    "ego_collisions_per_km_mean": "02_safety/ego_collisions_per_km",
    "h_min": "02_safety/h_min",
    "qp_failure_rate": "02_safety/qp_failure_rate",
    "mean_abs_speed_error": "03_efficiency/abs_speed_deviation_mps",
    "event_intervention_rate": "05_filter/intervention_rate",
    "mean_correction_norm": "05_filter/correction_norm_mean",
    "mean_neighbor_density_per_km": "06_traffic/neighbor_density_per_km",
    "_progress_distance_m": "03_efficiency/distance_traveled_m",
    "_progress_time_s": "01_task/episode_time_s",
}


def read_history(source_dir: Path, variant: str) -> pd.DataFrame:
    path = source_dir / variant / "train_eval_history.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["variant"] = variant
    return frame


def read_source_tensorboard_eval(source_dir: Path) -> pd.DataFrame:
    tb_root = source_dir / "tb"
    if not tb_root.exists():
        return pd.DataFrame()
    event_files = list(tb_root.rglob("events.out.tfevents*"))
    wanted_tags: dict[str, tuple[str, str]] = {}
    for variant in RUN_NAMES:
        for column, source_tag in SOURCE_EVAL_TAGS.items():
            wanted_tags[f"eval/{variant}/{source_tag}"] = (variant, column)

    values_by_key: dict[tuple[str, int, str], list[float]] = {}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for full_tag in set(accumulator.Tags().get("scalars", [])).intersection(wanted_tags):
            variant, column = wanted_tags[full_tag]
            for event in accumulator.Scalars(full_tag):
                # Resume can create steps like 20,001. Round to the intended 5k eval checkpoint
                # so the compact dashboard does not show duplicate near-identical x positions.
                rounded_step = int(round(float(event.step) / 5000.0) * 5000)
                values_by_key.setdefault((variant, rounded_step, column), []).append(float(event.value))

    if not values_by_key:
        return pd.DataFrame()

    rows_by_key: dict[tuple[str, int], dict[str, float | str]] = {}
    for (variant, step, column), values in values_by_key.items():
        key = (variant, step)
        rows_by_key.setdefault(key, {"variant": variant, "timesteps": float(step)})
        rows_by_key[key][column] = float(np.mean(values))
    frame = pd.DataFrame(rows_by_key.values())
    return frame.sort_values(["variant", "timesteps"]).reset_index(drop=True)


def read_training_episode_tensorboard(source_dir: Path) -> pd.DataFrame:
    tb_root = source_dir / "tb" / "custom"
    if not tb_root.exists():
        return pd.DataFrame()

    wanted_tags: dict[str, tuple[str, str]] = {}
    for variant, run_name in RUN_NAMES.items():
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
        rows_by_key.setdefault(key, {"variant": variant, "timesteps": float(step)})
        rows_by_key[key][column] = float(np.mean(values))

    rows: list[dict[str, float | str]] = []
    for row in rows_by_key.values():
        distance = clean_value(row.get("_progress_distance_m"))
        time_s = clean_value(row.get("_progress_time_s"))
        if distance is not None and time_s is not None and time_s > 1e-9:
            row["progress_rate_mps_mean"] = float(distance / time_s)
        rows.append(row)

    frame = pd.DataFrame(rows)
    return frame.sort_values(["variant", "timesteps"]).reset_index(drop=True)


def load_train_eval(source_dir: Path) -> pd.DataFrame:
    tensorboard_frame = read_source_tensorboard_eval(source_dir)
    expected_columns = {"variant", "timesteps", *[spec["column"] for spec in HEADLINE_KPIS]}
    if not tensorboard_frame.empty and expected_columns.issubset(set(tensorboard_frame.columns)):
        return tensorboard_frame
    histories = [read_history(source_dir, variant) for variant in RUN_NAMES]
    return pd.concat(histories, ignore_index=True)


def clean_value(value: Any) -> float | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if np.isfinite(scalar) else None


def append_terminal_final_eval(train_history: pd.DataFrame, final_summary: pd.DataFrame) -> pd.DataFrame:
    candidate_steps: list[float] = []
    if "timesteps" in train_history.columns:
        train_max = clean_value(pd.to_numeric(train_history["timesteps"], errors="coerce").max())
        if train_max is not None:
            candidate_steps.append(train_max)
    if "timesteps" in final_summary.columns:
        final_max = clean_value(pd.to_numeric(final_summary["timesteps"], errors="coerce").max())
        if final_max is not None:
            candidate_steps.append(final_max)
    if not candidate_steps:
        return train_history

    target_step = float(max(candidate_steps))
    additions: list[dict[str, Any]] = []
    for variant in RUN_NAMES:
        history = train_history[train_history["variant"] == variant]
        final_rows = final_summary[final_summary["variant"] == variant]
        if history.empty or final_rows.empty:
            continue
        last_step = clean_value(pd.to_numeric(history["timesteps"], errors="coerce").max()) or 0.0
        if last_step >= target_step:
            continue

        final_row = final_rows.iloc[0]
        addition: dict[str, Any] = {
            "variant": variant,
            "timesteps": target_step,
            "source": "terminal_final_eval_backfill",
        }
        for spec in HEADLINE_KPIS:
            value = clean_value(final_row.get(spec["final_column"]))
            if value is not None:
                addition[spec["column"]] = value
        additions.append(addition)

    if not additions:
        return train_history
    return (
        pd.concat([train_history, pd.DataFrame(additions)], ignore_index=True)
        .sort_values(["variant", "timesteps"])
        .reset_index(drop=True)
    )


def write_tensorboard(
    *,
    during_training: pd.DataFrame,
    train_history: pd.DataFrame,
    final_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for variant, run_name in RUN_NAMES.items():
        writer = SummaryWriter(log_dir=str(output_dir / run_name))
        training = during_training[during_training["variant"] == variant].sort_values("timesteps")
        history = train_history[train_history["variant"] == variant].sort_values("timesteps")
        final_rows = final_summary[final_summary["variant"] == variant]
        if final_rows.empty:
            continue
        final_row = final_rows.iloc[0]
        final_step = int(clean_value(final_row.get("timesteps")) or clean_value(history["timesteps"].max()) or 0)

        for _, row in training.iterrows():
            step = int(clean_value(row.get("timesteps")) or 0)
            for spec in HEADLINE_KPIS:
                value = clean_value(row.get(spec["column"]))
                if value is not None:
                    writer.add_scalar(f"during_training/{spec['tag']}", value, step)

        for _, row in history.iterrows():
            step = int(clean_value(row.get("timesteps")) or 0)
            for spec in HEADLINE_KPIS:
                value = clean_value(row.get(spec["column"]))
                if value is not None:
                    writer.add_scalar(f"train_eval/{spec['tag']}", value, step)

        for spec in HEADLINE_KPIS:
            value = clean_value(final_row.get(spec["final_column"]))
            if value is not None:
                writer.add_scalar(f"final_eval/{spec['tag']}", value, final_step)

        writer.add_text(
            "00_index",
            "\n".join(
                [
                    "# Headline KPI Dashboard",
                    "",
                    "Runs are agents; tags are shared across agents, so TensorBoard overlays them on the same graph.",
                    "",
                    "Use `during_training/*` for real episode-level KPI traces emitted while learning.",
                    "Use `train_eval/*` to see evolution over training.",
                    "Use `final_eval/*` for the 50-episode final evaluation point.",
                    "If one run missed only the terminal training-eval checkpoint, the terminal point is filled from final eval so every line reaches the same endpoint.",
                    "QP failure rate may be absent under `during_training/*` when the training episode stream did not record it; it remains available under eval sections.",
                    "",
                    "| Group | KPI | Direction |",
                    "|---|---|---|",
                    *[
                        f"| `{spec['tag'].split('/')[0]}` | `{spec['tag'].split('/', 1)[1]}` | {spec['direction']} |"
                        for spec in HEADLINE_KPIS
                    ],
                ]
            ),
            0,
        )
        writer.flush()
        writer.close()


def plot_during_training(during_training: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(14, 17), sharex=True)
    for axis, spec in zip(axes.ravel(), HEADLINE_KPIS):
        any_data = False
        for variant in RUN_NAMES:
            history = during_training[during_training["variant"] == variant].sort_values("timesteps")
            if history.empty or spec["column"] not in history.columns:
                continue
            values = pd.to_numeric(history[spec["column"]], errors="coerce")
            if values.notna().sum() == 0:
                continue
            smoothed = values.rolling(window=25, min_periods=1).mean()
            axis.plot(
                history["timesteps"],
                smoothed,
                linewidth=1.8,
                color=COLORS.get(variant),
                label=LABELS.get(variant, variant),
            )
            any_data = True
        axis.set_title(spec["title"])
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("Training timesteps")
        if not any_data:
            axis.text(0.5, 0.5, "not in explicit training stream", ha="center", va="center", transform=axis.transAxes)
    axes[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle("Headline KPIs During Training Episodes (Rolling Mean)", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_training(train_history: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(14, 17), sharex=True)
    for axis, spec in zip(axes.ravel(), HEADLINE_KPIS):
        for variant in RUN_NAMES:
            history = train_history[train_history["variant"] == variant].sort_values("timesteps")
            if spec["column"] not in history.columns:
                continue
            axis.plot(
                history["timesteps"],
                history[spec["column"]],
                marker="o",
                linewidth=2.0,
                markersize=4,
                color=COLORS.get(variant),
                label=LABELS.get(variant, variant),
            )
        axis.set_title(spec["title"])
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("Training timesteps")
    axes[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle("Headline KPIs Over Training Evaluation", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_final(final_summary: pd.DataFrame, output_path: Path) -> None:
    variants = [variant for variant in RUN_NAMES if variant in set(final_summary["variant"])]
    x = np.arange(len(variants), dtype=float)
    fig, axes = plt.subplots(5, 2, figsize=(14, 17))
    labels = [LABELS.get(variant, variant) for variant in variants]
    colors = [COLORS.get(variant) for variant in variants]
    for axis, spec in zip(axes.ravel(), HEADLINE_KPIS):
        values = []
        for variant in variants:
            row = final_summary[final_summary["variant"] == variant].iloc[0]
            values.append(clean_value(row.get(spec["final_column"])) or 0.0)
        axis.bar(x, values, color=colors)
        axis.set_title(spec["title"])
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Headline KPIs: Final 50-Episode Evaluation", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_clean_csvs(
    during_training: pd.DataFrame,
    train_history: pd.DataFrame,
    final_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    training_cols = ["variant", "timesteps", *[spec["column"] for spec in HEADLINE_KPIS]]
    train_cols = ["variant", "timesteps", *[spec["column"] for spec in HEADLINE_KPIS]]
    final_cols = ["variant", "timesteps", *[spec["final_column"] for spec in HEADLINE_KPIS]]
    training_cols = list(dict.fromkeys([col for col in training_cols if col in during_training.columns]))
    train_cols = list(dict.fromkeys([col for col in train_cols if col in train_history.columns]))
    final_cols = list(dict.fromkeys([col for col in final_cols if col in final_summary.columns]))
    output_dir.mkdir(parents=True, exist_ok=True)
    during_training[training_cols].to_csv(output_dir / "headline_during_training.csv", index=False)
    train_history[train_cols].to_csv(output_dir / "headline_train_eval.csv", index=False)
    final_summary[final_cols].to_csv(output_dir / "headline_final_eval.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean overlaid TensorBoard dashboard for headline KPIs.")
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/sp3_mtm_e0p1_p0p0_kpi_full"))
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("artifacts/tb_headline_kpis"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = (args.output_dir or source_dir / "headline_kpis").resolve()
    tb_dir = args.tensorboard_dir.resolve()
    during_training = read_training_episode_tensorboard(source_dir)
    train_history = load_train_eval(source_dir)
    final_summary_path = source_dir / "summary.csv"
    if not final_summary_path.exists():
        raise FileNotFoundError(final_summary_path)
    final_summary = pd.read_csv(final_summary_path)
    train_history = append_terminal_final_eval(train_history, final_summary)

    write_tensorboard(
        during_training=during_training,
        train_history=train_history,
        final_summary=final_summary,
        output_dir=tb_dir,
    )
    plot_during_training(during_training, output_dir / "headline_during_training.png")
    plot_training(train_history, output_dir / "headline_train_eval.png")
    plot_final(final_summary, output_dir / "headline_final_eval.png")
    write_clean_csvs(during_training, train_history, final_summary, output_dir)

    print(f"[headline-kpis] tensorboard={tb_dir}")
    print(f"[headline-kpis] output={output_dir}")
    print(f"[headline-kpis] during_training_plot={output_dir / 'headline_during_training.png'}")
    print(f"[headline-kpis] train_plot={output_dir / 'headline_train_eval.png'}")
    print(f"[headline-kpis] final_plot={output_dir / 'headline_final_eval.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
