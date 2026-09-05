"""Generate PPO 500k plots from saved CSV and TensorBoard artifacts.

This script never trains or evaluates an agent.  It consumes the existing
500k study directories and writes per-agent reports plus one combined report.
"""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUNS = {
    "ppo_nominal": {
        "label": "Nominal PPO",
        "directory": "ppo_nominal_500k_seed307",
        "color": "#4C78A8",
    },
    "ppo_cbf_reward": {
        "label": "CBF-reward PPO",
        "directory": "ppo_cbf_reward_500k_seed307",
        "color": "#54A24B",
    },
    "ppo_cbf_projected": {
        "label": "Projected-CBF PPO",
        "directory": "ppo_cbf_projected_500k_seed307",
        "color": "#E45756",
    },
}

MODE_LABELS = {"raw": "CBF OFF", "cbf": "CBF ON"}
MODE_COLORS = {"raw": "#7F8C8D", "cbf": "#1677B9"}

POST_KPIS = [
    ("episode_return", "Episode return", "Return"),
    ("episode_length_steps", "Episode length (steps)", "Steps"),
    ("ego_collisions_per_km", "Ego collisions / km", "Collisions / km"),
    ("h_min", "Minimum h", "Minimum h"),
    ("qp_failure_rate", "QP failure rate", "QP failure rate"),
    ("mean_abs_speed_deviation", "Abs speed error (m/s)", "Speed error (m/s)"),
    ("mean_lat_y_error_m", "Mean lateral tracking error (m)", "Lateral error (m)"),
    ("event_intervention_rate", "Intervention rate", "Intervention rate"),
    ("mean_correction_norm", "Correction norm", "Correction norm"),
    ("mean_jerk_norm", "Mean jerk norm", "Jerk norm"),
]

TRAINING_KPIS = [
    ("episode_return", "Episode return"),
    ("episode_length", "Episode length"),
    ("return_per_timestep", "Return / timestep"),
    ("ego_collisions_per_km", "Ego collisions / km"),
    ("total_distance_m", "Distance (m)"),
    ("distinct_ego_collision_events", "Collision events"),
    ("action_saturation_mean", "Action saturation"),
    ("resets_after_collision", "Resets after collision"),
]

TB_ROLLOUT = [
    ("rollout/episode_return", "Episode return"),
    ("rollout/episode_length", "Episode length"),
    ("rollout/collisions_per_km", "Collisions / km"),
    ("rollout/distance_m", "Distance (m)"),
    ("rollout/return_per_timestep", "Return / timestep"),
    ("rollout/action_saturation_mean", "Action saturation"),
    ("rollout/reset_calls_total", "Reset calls"),
]

TB_TRAIN = [
    ("train/loss", "Total loss"),
    ("train/value_loss", "Value loss"),
    ("train/policy_gradient_loss", "Policy-gradient loss"),
    ("train/approx_kl", "Approx. KL"),
    ("train/clip_fraction", "Clip fraction"),
    ("train/entropy_loss", "Entropy loss"),
    ("train/explained_variance", "Explained variance"),
    ("train/std", "Policy std"),
    ("train/learning_rate", "Learning rate"),
]

TB_CBF = [
    ("train/cbf_mean_correction", "Mean CBF correction"),
    ("train/cbf_mean_infeasible_rate", "Mean infeasible rate"),
    ("train/cbf_mean_loss", "Mean CBF loss"),
    ("train/cbf_sample_correction", "Sample correction"),
    ("train/cbf_sample_infeasible_rate", "Sample infeasible rate"),
    ("train/cbf_sample_loss", "Sample CBF loss"),
    ("train/actor_g_ppo_norm", "PPO gradient norm"),
    ("train/actor_g_cbf_norm", "CBF gradient norm"),
    ("train/actor_g_cbf_to_g_ppo_ratio", "CBF/PPO gradient ratio"),
    ("train/actor_g_ppo_g_cbf_cosine", "PPO/CBF gradient cosine"),
    ("train/cbf_lambda_mean", "CBF lambda (mean)"),
    ("train/cbf_lambda_sample", "CBF lambda (sample)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=None,
        help="Combined-report directory; defaults to artifacts/ppo_500k_comparison.",
    )
    return parser.parse_args()


def _find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} under {directory}")
    return matches[0]


def _read_events(run_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    manifest_path = _find_one(run_dir, "**/tb.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event_names = manifest.get("new_event_files") or manifest.get("all_event_files") or []
    event_paths = [Path(name) for name in event_names if Path(name).exists()]
    if not event_paths:
        raise FileNotFoundError(f"No TensorBoard event files listed in {manifest_path}")

    rows: list[dict[str, object]] = []
    for event_path in event_paths:
        accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for scalar in accumulator.Scalars(tag):
                rows.append(
                    {
                        "tag": tag,
                        "step": scalar.step,
                        "value": scalar.value,
                    }
                )
    return pd.DataFrame(rows), [str(path) for path in event_paths]


def load_run(project_root: Path, variant: str) -> dict[str, object]:
    run_dir = project_root / "artifacts" / RUNS[variant]["directory"]
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    episodes_path = _find_one(run_dir, "**/pe/e.csv")
    blocks_path = _find_one(run_dir, "**/pe/b.csv")
    training_path = _find_one(run_dir, "**/training_episodes.csv")
    kpi_path = run_dir / "post_train_200ep_kpis.csv"

    episodes = pd.read_csv(episodes_path)
    blocks = pd.read_csv(blocks_path)
    training = pd.read_csv(training_path)
    summary = pd.read_csv(kpi_path)
    events, event_paths = _read_events(run_dir)
    if "variant" not in events:
        events.insert(0, "variant", variant)

    for frame in (episodes, blocks, training):
        if "variant" not in frame:
            frame.insert(0, "variant", variant)
    summary.insert(0, "variant", variant) if "variant" not in summary else None
    summary["mode"] = summary["external_cbf"].map({"OFF": "raw", "ON": "cbf"})

    return {
        "variant": variant,
        "run_dir": run_dir,
        "episodes": episodes,
        "blocks": blocks,
        "training": training,
        "summary": summary,
        "events": events,
        "event_paths": event_paths,
    }


def _title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=9, color="#475569")


def _style(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig: plt.Figure, path: Path, pdf: PdfPages) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.90))
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _labels(variants: list[str]) -> list[str]:
    return [RUNS[v]["label"] for v in variants]


def plot_kpi_bars(frame: pd.DataFrame, variants: list[str], path: Path, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(19, 8.5))
    _title(fig, "Post-training KPI comparison", "Saved 200-episode evaluations per CBF mode; bars show Mean +/- SD across ten 20-episode blocks.")
    x = np.arange(len(variants))
    width = 0.36 if len(variants) > 1 else 0.28
    for ax, (column, kpi, ylabel) in zip(axes.flat, POST_KPIS):
        for offset, mode in ((-width / 2, "raw"), (width / 2, "cbf")):
            rows = frame[(frame["mode"] == mode) & (frame["KPI"] == kpi)].set_index("variant").reindex(variants)
            values = pd.to_numeric(rows["Mean"], errors="coerce").to_numpy(float)
            errors = pd.to_numeric(rows["SD"], errors="coerce").fillna(0).to_numpy(float)
            bars = ax.bar(x + offset, values, width, yerr=errors, capsize=2.5, color=MODE_COLORS[mode], alpha=0.9, label=MODE_LABELS[mode])
            if ax is axes.flat[0]:
                ax.bar_label(bars, labels=[f"{v:.3g}" if np.isfinite(v) else "" for v in values], fontsize=6, padding=1)
        ax.set_title(kpi, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x, _labels(variants), rotation=18, ha="right", fontsize=7)
        _style(ax)
    axes.flat[0].legend(frameon=False, fontsize=8)
    _save(fig, path, pdf)


def plot_distributions(frame: pd.DataFrame, variants: list[str], path: Path, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(19, 8.5))
    _title(fig, "Post-training episode distributions", "Each box contains the saved complete episodes for one agent and external-CBF mode.")
    x = np.arange(len(variants))
    for ax, (column, kpi, ylabel) in zip(axes.flat, POST_KPIS):
        for mode, offset in (("raw", -0.19), ("cbf", 0.19)):
            data = [frame[(frame["variant"] == v) & (frame["mode"] == mode)][column].dropna().to_numpy(float) for v in variants]
            box = ax.boxplot(data, positions=x + offset, widths=0.30, patch_artist=True, showfliers=False)
            for patch in box["boxes"]:
                patch.set(facecolor=MODE_COLORS[mode], alpha=0.65)
            for median in box["medians"]:
                median.set(color="#111827", linewidth=1.1)
        ax.set_title(kpi, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x, _labels(variants), rotation=18, ha="right", fontsize=7)
        _style(ax)
    from matplotlib.patches import Patch

    axes.flat[0].legend(handles=[Patch(facecolor=MODE_COLORS["raw"], label="CBF OFF"), Patch(facecolor=MODE_COLORS["cbf"], label="CBF ON")], frameon=False, fontsize=8)
    _save(fig, path, pdf)


def plot_block_trends(frame: pd.DataFrame, variants: list[str], path: Path, pdf: PdfPages) -> None:
    specs = [
        ("episode_return", "Return"),
        ("ego_collisions_per_km", "Collisions / km"),
        ("h_min", "Minimum h"),
        ("event_intervention_rate", "Intervention rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    _title(fig, "Post-training block trends", "Ten saved 20-episode blocks; dashed lines are CBF OFF and solid lines are CBF ON.")
    for ax, (column, ylabel) in zip(axes.flat, specs):
        for variant in variants:
            for mode, linestyle in (("raw", "--"), ("cbf", "-")):
                rows = frame[(frame["variant"] == variant) & (frame["mode"] == mode)].sort_values("summary_block")
                if rows.empty:
                    continue
                ax.plot(rows["summary_block"], rows[column], color=RUNS[variant]["color"], linestyle=linestyle, linewidth=1.6, marker="o", markersize=2.5, label=RUNS[variant]["label"])
        ax.set_xlabel("20-episode block", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        _style(ax)
    from matplotlib.lines import Line2D

    handles = [Line2D([0], [0], color=RUNS[v]["color"], lw=2, label=RUNS[v]["label"]) for v in variants]
    handles += [Line2D([0], [0], color="#334155", lw=2, linestyle=ls, label=MODE_LABELS[m]) for m, ls in (("raw", "--"), ("cbf", "-"))]
    axes.flat[0].legend(handles=handles, fontsize=7, frameon=False, ncol=2)
    _save(fig, path, pdf)


def plot_training_traces(frame: pd.DataFrame, variants: list[str], path: Path, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    _title(fig, "Training episode traces", "Saved episode-level traces through the 500,000 training timesteps; solid lines are 25-episode rolling means.")
    for ax, (column, ylabel) in zip(axes.flat, TRAINING_KPIS):
        for variant in variants:
            rows = frame[frame["variant"] == variant].sort_values("global_timestep")
            if rows.empty or column not in rows:
                continue
            x = pd.to_numeric(rows["global_timestep"], errors="coerce")
            y = pd.to_numeric(rows[column], errors="coerce")
            ax.scatter(x, y, s=5, alpha=0.12, color=RUNS[variant]["color"])
            ax.plot(x, y.rolling(25, min_periods=1).mean(), color=RUNS[variant]["color"], linewidth=1.7, label=RUNS[variant]["label"])
        ax.set_xlabel("Training timestep", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        _style(ax)
    axes.flat[0].legend(frameon=False, fontsize=7, ncol=2)
    _save(fig, path, pdf)


def plot_tb(frame: pd.DataFrame, variants: list[str], specs: list[tuple[str, str]], path: Path, pdf: PdfPages, title: str) -> bool:
    if frame.empty or not frame["tag"].isin([tag for tag, _ in specs]).any():
        return False
    ncols = 3
    nrows = ceil(len(specs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(19, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    _title(fig, title, "Read directly from the saved TensorBoard event files; no retraining was performed.")
    for ax, (tag, ylabel) in zip(axes, specs):
        for variant in variants:
            rows = frame[(frame["variant"] == variant) & (frame["tag"] == tag)].sort_values("step")
            if rows.empty:
                continue
            ax.plot(rows["step"], rows["value"], color=RUNS[variant]["color"], linewidth=1.5, label=RUNS[variant]["label"])
        ax.set_xlabel("Training timestep", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(ylabel, fontsize=10)
        _style(ax)
    for ax in axes[len(specs):]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    _save(fig, path, pdf)
    return True


def inventory_page(data: dict[str, dict[str, object]], variants: list[str], path: Path, pdf: PdfPages) -> None:
    rows = []
    for variant in variants:
        run = data[variant]
        training = run["training"]
        rows.append(
            {
                "Agent": RUNS[variant]["label"],
                "Training steps": int(pd.to_numeric(training["global_timestep"], errors="coerce").max()),
                "Training rows": len(training),
                "Post episodes": len(run["episodes"]),
                "Post blocks": len(run["blocks"]),
                "TB event files": len(run["event_paths"]),
            }
        )
    table_frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")
    fig.suptitle("PPO 500k saved-data graph report", fontsize=22, fontweight="bold", y=0.94)
    fig.text(0.5, 0.885, "All plots were generated from existing CSV traces and TensorBoard event files; training/evaluation were not rerun.", ha="center", fontsize=11)
    table = ax.table(cellText=table_frame.astype(str).values, colLabels=list(table_frame.columns), loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1F4E79")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2:
            cell.set_facecolor("#EEF2F7")
    _save(fig, path, pdf)


def make_report(data: dict[str, dict[str, object]], variants: list[str], output: Path, report_name: str) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.concat([data[v]["summary"] for v in variants], ignore_index=True)
    episodes = pd.concat([data[v]["episodes"] for v in variants], ignore_index=True)
    blocks = pd.concat([data[v]["blocks"] for v in variants], ignore_index=True)
    training = pd.concat([data[v]["training"] for v in variants], ignore_index=True)
    events = pd.concat([data[v]["events"] for v in variants], ignore_index=True)

    generated: list[str] = []
    with PdfPages(output / report_name) as pdf:
        inventory_page(data, variants, output / "00_artifact_inventory.png", pdf)
        generated.append("00_artifact_inventory.png")
        plot_kpi_bars(summary, variants, output / "01_post_training_kpi_comparison.png", pdf)
        generated.append("01_post_training_kpi_comparison.png")
        plot_distributions(episodes, variants, output / "02_post_training_episode_distributions.png", pdf)
        generated.append("02_post_training_episode_distributions.png")
        plot_block_trends(blocks, variants, output / "03_post_training_block_trends.png", pdf)
        generated.append("03_post_training_block_trends.png")
        plot_training_traces(training, variants, output / "04_training_episode_traces.png", pdf)
        generated.append("04_training_episode_traces.png")
        if plot_tb(events, variants, TB_ROLLOUT, output / "05_tensorboard_rollout_scalars.png", pdf, "TensorBoard rollout scalars"):
            generated.append("05_tensorboard_rollout_scalars.png")
        if plot_tb(events, variants, TB_TRAIN, output / "06_tensorboard_optimization_scalars.png", pdf, "TensorBoard PPO optimization scalars"):
            generated.append("06_tensorboard_optimization_scalars.png")
        if plot_tb(events, variants, TB_CBF, output / "07_tensorboard_cbf_diagnostics.png", pdf, "TensorBoard CBF and projected-actor diagnostics"):
            generated.append("07_tensorboard_cbf_diagnostics.png")

    manifest = {
        "generated_from_existing_data": True,
        "training_rerun": False,
        "variants": variants,
        "source_rows": {
            v: {
                "training_episode_rows": len(data[v]["training"]),
                "post_training_episode_rows": len(data[v]["episodes"]),
                "post_training_block_rows": len(data[v]["blocks"]),
                "tensorboard_scalar_rows": len(data[v]["events"]),
            }
            for v in variants
        },
        "event_files": {v: data[v]["event_paths"] for v in variants},
        "figures": generated,
        "report_pdf": str((output / report_name).resolve()),
    }
    (output / "graph_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    data = {variant: load_run(project_root, variant) for variant in RUNS}

    manifests = {}
    for variant, run in data.items():
        output = run["run_dir"] / "figures_from_saved_data"
        manifests[variant] = make_report({variant: run}, [variant], output, "ppo_500k_graph_report.pdf")

    comparison_dir = (args.comparison_dir or project_root / "artifacts" / "ppo_500k_comparison").resolve()
    manifests["comparison"] = make_report(data, list(RUNS), comparison_dir, "ppo_500k_comparison_report.pdf")
    print(json.dumps(manifests, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
