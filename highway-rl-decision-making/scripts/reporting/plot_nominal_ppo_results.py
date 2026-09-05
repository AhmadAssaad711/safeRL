"""Plot the standalone elongated nominal-PPO study.

This report intentionally consumes only saved artifacts.  It produces the same
families of outputs as the canonical progression report (KPI bars, episode
distributions, pooled-block trends, training traces, and TensorBoard curves),
but does not require the other PPO variants or counterfactual files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


MODE_COLORS = {"raw": "#7F8C8D", "cbf": "#1677B9"}
MODE_LABELS = {"raw": "CBF OFF", "cbf": "CBF ON"}
VARIANT_COLOR = "#4C78A8"
VARIANT_LABEL = "Nominal PPO"

KPI_ORDER = [
    ("Episode return", "episode_return", "Return"),
    ("Episode length (steps)", "episode_length_steps", "Steps"),
    ("Ego collisions / km", "ego_collisions_per_km", "Collisions / km"),
    ("Minimum h", "h_min", "Minimum h"),
    ("QP failure rate", "qp_failure_rate", "QP failure rate"),
    ("Abs speed error (m/s)", "mean_abs_speed_deviation", "Speed error (m/s)"),
    ("Mean lateral tracking error (m)", "mean_lat_y_error_m", "Lateral error (m)"),
    ("Intervention rate", "event_intervention_rate", "Intervention rate"),
    ("Correction norm", "mean_correction_norm", "Correction norm"),
    ("Mean jerk norm", "mean_jerk_norm", "Jerk norm"),
]

TB_ROLLOUT = [
    ("rollout/episode_return", "Episode return"),
    ("rollout/episode_length", "Episode length"),
    ("rollout/collisions_per_km", "Collisions / km"),
    ("rollout/distinct_collision_events", "Collision events"),
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path("artifacts/ppo_nominal_500k_seed307"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _format_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=9, color="#475569")


def _save(fig: plt.Figure, path: Path, pdf: PdfPages) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.90))
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _mode_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "mode" not in out and "external_cbf" in out:
        out["mode"] = out["external_cbf"].map({"OFF": "raw", "ON": "cbf"})
    if "mode" in out:
        out["mode"] = out["mode"].astype(str).str.lower()
    return out


def _load_tensorboard(run_dir: Path, study_dir: Path) -> tuple[pd.DataFrame, Path | None]:
    roots: list[Path] = []
    manifest_path = run_dir / "tb.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("log_dir"):
                roots.append(Path(str(manifest["log_dir"])))
        except (OSError, json.JSONDecodeError):
            pass
    config_path = run_dir / "run_config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            tensorboard = config.get("tensorboard", {})
            if isinstance(tensorboard, dict) and tensorboard.get("log_dir"):
                roots.append(Path(str(tensorboard["log_dir"])))
        except (OSError, json.JSONDecodeError):
            pass
    roots.extend(
        [
            run_dir / "tensorboard",
            study_dir / "tb" / "ppo" / "n5_nom_307",
            study_dir.parents[1] / "artifacts" / "tb" / "ppo" / "n5_nom_307",
        ]
    )
    event_paths: dict[Path, None] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("events.out.tfevents.*"):
            event_paths[path.resolve()] = None
    rows: list[dict[str, float | int | str]] = []
    for event_path in sorted(event_paths, key=str):
        try:
            accumulator = EventAccumulator(
                str(event_path), size_guidance={"scalars": 0}
            )
            accumulator.Reload()
        except Exception:
            continue
        for tag in accumulator.Tags().get("scalars", []):
            for scalar in accumulator.Scalars(tag):
                rows.append(
                    {
                        "tag": str(tag),
                        "step": int(scalar.step),
                        "value": float(scalar.value),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["tag", "step", "value"]), None
    frame = pd.DataFrame(rows).sort_values(["tag", "step"])
    frame = frame.drop_duplicates(["tag", "step"], keep="last")
    return frame.reset_index(drop=True), next(iter(event_paths), None)


def _plot_kpi_table(table: pd.DataFrame, path: Path, pdf: PdfPages) -> None:
    columns = ["external_cbf", "KPI", "Mean", "SD", "N", "episodes_per_mode"]
    visible = table.loc[:, [column for column in columns if column in table]].copy()
    for column in ("Mean", "SD"):
        if column in visible:
            visible[column] = pd.to_numeric(visible[column], errors="coerce").map(
                lambda value: f"{value:.3f}" if np.isfinite(value) else ""
            )
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.axis("off")
    _title(
        fig,
        "Nominal PPO post-training KPI table",
        "400 complete episodes: 200 with external CBF OFF and 200 with external CBF ON; N is the number of pooled 20-episode blocks.",
    )
    rendered = ax.table(
        cellText=visible.astype(str).values,
        colLabels=list(visible.columns),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(8.5)
    rendered.scale(1, 1.55)
    for (row, _column), cell in rendered.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1F4E79")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2:
            cell.set_facecolor("#EEF2F7")
    _save(fig, path, pdf)


def _plot_kpi_bars(table: pd.DataFrame, path: Path, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(19, 8.5))
    _title(
        fig,
        "Nominal PPO post-training evaluation",
        "400 complete episodes total; bars show pooled-block Mean +/- SD for CBF OFF and CBF ON.",
    )
    table = _mode_frame(table)
    for ax, (kpi, _column, ylabel) in zip(axes.flat, KPI_ORDER):
        for offset, mode in ((-0.18, "raw"), (0.18, "cbf")):
            rows = table[(table["mode"] == mode) & (table["KPI"] == kpi)]
            mean = float(pd.to_numeric(rows["Mean"], errors="coerce").iloc[0]) if not rows.empty else np.nan
            sd = float(pd.to_numeric(rows["SD"], errors="coerce").iloc[0]) if not rows.empty else 0.0
            ax.bar(
                [offset],
                [mean],
                width=0.32,
                yerr=[sd],
                capsize=3,
                color=MODE_COLORS[mode],
                label=MODE_LABELS[mode],
            )
        ax.set_title(kpi, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks([])
        _format_axes(ax)
    axes.flat[0].legend(frameon=False, fontsize=8)
    _save(fig, path, pdf)


def _plot_episode_distributions(episodes: pd.DataFrame, path: Path, pdf: PdfPages) -> None:
    episodes = _mode_frame(episodes)
    fig, axes = plt.subplots(2, 5, figsize=(19, 8.5))
    _title(
        fig,
        "Nominal PPO episode-level evaluation distributions",
        "Each box contains 200 complete episodes for the indicated external-CBF mode; these are raw episode distributions.",
    )
    for ax, (kpi, column, ylabel) in zip(axes.flat, KPI_ORDER):
        data: list[np.ndarray] = []
        labels: list[str] = []
        for mode in ("raw", "cbf"):
            values = pd.to_numeric(
                episodes.loc[episodes["mode"] == mode, column], errors="coerce"
            ).dropna().to_numpy(float)
            data.append(values if values.size else np.asarray([np.nan]))
            labels.append(MODE_LABELS[mode])
        box = ax.boxplot(data, positions=[0, 1], widths=0.55, patch_artist=True, showfliers=False)
        for patch, mode in zip(box["boxes"], ("raw", "cbf")):
            patch.set(facecolor=MODE_COLORS[mode], alpha=0.65)
        for median in box["medians"]:
            median.set(color="#111827", linewidth=1.1)
        ax.set_title(kpi, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks([0, 1], ["OFF", "ON"], fontsize=8)
        _format_axes(ax)
    from matplotlib.patches import Patch

    axes.flat[0].legend(
        handles=[
            Patch(facecolor=MODE_COLORS["raw"], label="CBF OFF"),
            Patch(facecolor=MODE_COLORS["cbf"], label="CBF ON"),
        ],
        frameon=False,
        fontsize=8,
    )
    _save(fig, path, pdf)


def _plot_block_trends(blocks: pd.DataFrame, path: Path, pdf: PdfPages) -> None:
    blocks = _mode_frame(blocks)
    specs = [
        ("episode_return", "Return"),
        ("ego_collisions_per_km", "Ego collisions / km"),
        ("h_min", "Minimum h"),
        ("event_intervention_rate", "Intervention rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    _title(
        fig,
        "Nominal PPO post-training pooled-block trends",
        "Ten independent 20-episode blocks per mode; these are the block-level values used for the KPI SDs.",
    )
    for ax, (column, ylabel) in zip(axes.flat, specs):
        for mode, linestyle in (("raw", "--"), ("cbf", "-")):
            rows = blocks[blocks["mode"] == mode].sort_values("summary_block")
            if rows.empty or column not in rows:
                continue
            ax.plot(
                rows["summary_block"],
                pd.to_numeric(rows[column], errors="coerce"),
                color=MODE_COLORS[mode],
                linestyle=linestyle,
                linewidth=1.7,
                marker="o",
                markersize=3,
                label=MODE_LABELS[mode],
            )
        ax.set_xlabel("20-episode block", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        _format_axes(ax)
    axes.flat[0].legend(frameon=False, fontsize=8)
    _save(fig, path, pdf)


def _plot_training_traces(training: pd.DataFrame, path: Path, pdf: PdfPages) -> None:
    specs = [
        ("episode_return", "Episode return"),
        ("episode_length", "Episode length"),
        ("return_per_timestep", "Return / timestep"),
        ("ego_collisions_per_km", "Ego collisions / km"),
        ("total_distance_m", "Distance (m)"),
        ("distinct_ego_collision_events", "Collision events"),
        ("action_saturation_mean", "Action saturation"),
        ("resets_after_collision", "Resets after collision"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    _title(
        fig,
        "Nominal PPO training episode traces",
        "Saved training episodes across the 500,000-timestep run; points are episode values and the line is a 25-episode rolling mean.",
    )
    training = training.sort_values("global_timestep")
    x = pd.to_numeric(training["global_timestep"], errors="coerce")
    for ax, (column, ylabel) in zip(axes.flat, specs):
        if column in training:
            y = pd.to_numeric(training[column], errors="coerce")
            ax.scatter(x, y, s=5, alpha=0.14, color=VARIANT_COLOR)
            ax.plot(x, y.rolling(25, min_periods=1).mean(), color=VARIANT_COLOR, linewidth=1.7)
        ax.set_xlabel("Training timestep", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        _format_axes(ax)
    _save(fig, path, pdf)


def _plot_tensorboard(
    scalars: pd.DataFrame,
    specs: list[tuple[str, str]],
    path: Path,
    pdf: PdfPages,
    title: str,
    subtitle: str,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(19, 10))
    _title(fig, title, subtitle)
    for ax, (tag, ylabel) in zip(axes.flat, specs):
        rows = scalars[scalars["tag"] == tag].sort_values("step")
        if rows.empty:
            ax.text(0.5, 0.5, f"No saved values for\n{tag}", ha="center", va="center", fontsize=9, color="#64748B")
        else:
            ax.plot(rows["step"], rows["value"], color=VARIANT_COLOR, linewidth=1.6, label=VARIANT_LABEL)
        ax.set_xlabel("Training timestep", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(tag, fontsize=9)
        _format_axes(ax)
    for ax in axes.flat[len(specs) :]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=8)
    _save(fig, path, pdf)


def _inventory_page(study_dir: Path, output_dir: Path, files: Iterable[tuple[str, Path]], pdf: PdfPages) -> None:
    rows = []
    for label, path in files:
        rows.append(
            {
                "Dataset": label,
                "Rows": len(pd.read_csv(path)) if path.is_file() and path.suffix == ".csv" else "-",
                "File": str(path.relative_to(study_dir)) if path.is_relative_to(study_dir) else str(path),
            }
        )
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")
    _title(fig, "Nominal PPO 500k artifact inventory", "All figures and tables are generated from saved training, evaluation, and TensorBoard artifacts.")
    rendered = ax.table(cellText=frame.astype(str).values, colLabels=list(frame.columns), loc="center", cellLoc="left", colLoc="left")
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(10)
    rendered.scale(1, 1.8)
    for (row, _column), cell in rendered.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1F4E79")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2:
            cell.set_facecolor("#EEF2F7")
    _save(fig, output_dir / "00_artifact_inventory.png", pdf)


def main() -> int:
    args = _parse_args()
    study_dir = args.study_dir.resolve()
    output_dir = (args.output_dir or study_dir / "figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = study_dir / "ppo_nominal" / "seed_307"
    episodes_path = run_dir / "pe" / "e.csv"
    blocks_path = run_dir / "pe" / "b.csv"
    training_path = run_dir / "training_episodes.csv"
    kpi_path = study_dir / "post_train_200ep_kpis.csv"
    required = [episodes_path, blocks_path, training_path, kpi_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Nominal PPO graph inputs are missing: " + ", ".join(missing))

    episodes = pd.read_csv(episodes_path)
    blocks = pd.read_csv(blocks_path)
    training = pd.read_csv(training_path)
    kpis = _mode_frame(pd.read_csv(kpi_path))
    counts = episodes.groupby("mode", sort=False).size().to_dict()
    expected_modes = {"raw": 200, "cbf": 200}
    if {str(key).lower(): int(value) for key, value in counts.items()} != expected_modes:
        raise RuntimeError(f"Expected exactly 200 complete episodes per mode; observed {counts}")
    if set(kpis["variant"].astype(str)) != {"ppo_nominal"}:
        kpis = kpis.loc[kpis["variant"].astype(str).eq("ppo_nominal")].copy()
    scalars, event_path = _load_tensorboard(run_dir, study_dir)

    report_path = output_dir / "nominal_ppo_500k_graph_report.pdf"
    inventory_files = [
        ("Post-training episodes", episodes_path),
        ("Post-training pooled blocks", blocks_path),
        ("Post-training KPI table", kpi_path),
        ("Training episode traces", training_path),
        ("TensorBoard manifest", run_dir / "tb.json"),
    ]
    with PdfPages(report_path) as pdf:
        _inventory_page(study_dir, output_dir, inventory_files, pdf)
        _plot_kpi_table(kpis, output_dir / "01_post_training_kpi_table.png", pdf)
        _plot_kpi_bars(kpis, output_dir / "02_post_training_400_episode_kpis.png", pdf)
        _plot_episode_distributions(episodes, output_dir / "03_post_training_episode_distributions.png", pdf)
        _plot_block_trends(blocks, output_dir / "04_post_training_block_trends.png", pdf)
        _plot_training_traces(training, output_dir / "05_training_episode_traces.png", pdf)
        _plot_tensorboard(
            scalars,
            TB_ROLLOUT,
            output_dir / "06_tensorboard_rollout_scalars.png",
            pdf,
            "Nominal PPO TensorBoard rollout curves",
            "Saved TensorBoard scalars from the parallel 500,000-timestep training run.",
        )
        _plot_tensorboard(
            scalars,
            TB_TRAIN,
            output_dir / "07_tensorboard_optimization_scalars.png",
            pdf,
            "Nominal PPO TensorBoard optimization curves",
            "Stable-Baselines3 PPO optimization metrics from the saved event files.",
        )

    manifest = {
        "study_dir": str(study_dir),
        "output_dir": str(output_dir),
        "post_training_episode_counts": {key: int(value) for key, value in counts.items()},
        "tensorboard_event_file": str(event_path) if event_path else None,
        "tensorboard_scalar_rows": int(len(scalars)),
        "figures": sorted(path.name for path in output_dir.glob("*.png")),
        "report_pdf": str(report_path),
    }
    (output_dir / "graph_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
