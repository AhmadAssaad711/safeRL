"""Build graphs and a PDF report for the canonical PPO/CBF progression run.

The script consumes only the saved CSVs and TensorBoard event files.  It does not
rerun training or evaluation.  The output is written to ``figures`` inside the
study artifact directory.
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


VARIANTS = [
    "ppo_nominal",
    "ppo_cbf_shield_only",
    "ppo_cbf_reward",
    "ppo_cbf_projected_reward_off",
    "ppo_cbf_projected",
]
SHORT = {
    "ppo_nominal": "Nominal",
    "ppo_cbf_shield_only": "Shield-only",
    "ppo_cbf_reward": "CBF reward",
    "ppo_cbf_projected_reward_off": "Projected\nreward-off",
    "ppo_cbf_projected": "Projected",
}
COLORS = {
    "ppo_nominal": "#4C78A8",
    "ppo_cbf_shield_only": "#F58518",
    "ppo_cbf_reward": "#54A24B",
    "ppo_cbf_projected_reward_off": "#B279A2",
    "ppo_cbf_projected": "#E45756",
}
MODE_COLORS = {"raw": "#7F8C8D", "cbf": "#1677B9"}
MODE_LABELS = {"raw": "CBF OFF", "cbf": "CBF ON"}

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

TRAIN_METRICS = [
    ("episode_return", "Episode return"),
    ("episode_length", "Episode length"),
    ("return_per_timestep", "Return / timestep"),
    ("ego_collisions_per_km", "Ego collisions / km"),
    ("total_distance_m", "Distance (m)"),
    ("distinct_ego_collision_events", "Ego collision events"),
    ("action_saturation_mean", "Action saturation"),
    ("resets_after_collision", "Resets after collision"),
]

POST_TREND_METRICS = [
    ("episode_return", "Return"),
    ("ego_collisions_per_km", "Ego collisions / km"),
    ("h_min", "Minimum h"),
    ("event_intervention_rate", "Intervention rate"),
]

TB_ROLLOUT = [
    ("rollout/episode_return", "Episode return"),
    ("rollout/episode_length", "Episode length"),
    ("rollout/collisions_per_km", "Collisions / km"),
    ("rollout/distinct_collision_events", "Distinct collision events"),
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
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path("artifacts/ppo_cbf_progression_parallel_v3"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _concat_variant_files(study: Path, relative: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for variant in VARIANTS:
        path = study / variant / "seed_307" / relative
        if path.exists():
            frame = pd.read_csv(path)
            if "variant" not in frame:
                frame.insert(0, "variant", variant)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_tb(study: Path) -> pd.DataFrame:
    """Read the five PPO TensorBoard event files, including legacy migration paths."""
    names = {
        "nom_307": "ppo_nominal",
        "shld_307": "ppo_cbf_shield_only",
        "rwd_307": "ppo_cbf_reward",
        "pro0_307": "ppo_cbf_projected_reward_off",
        "pro_307": "ppo_cbf_projected",
    }
    roots = [
        study / "tb" / "ppo",
        # The canonical runner currently migrates the legacy PPO events to the
        # sibling (outer) project artifact tree, one project level above study.
        study.parents[2] / "artifacts" / "tb" / "ppo",
    ]
    paths: dict[Path, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for folder, variant in names.items():
            for event in (root / folder).rglob("events.out.tfevents.*"):
                paths[event.resolve()] = variant
    rows: list[dict[str, object]] = []
    for event_path, variant in sorted(paths.items(), key=lambda item: str(item[0])):
        try:
            accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
            accumulator.Reload()
        except Exception:
            continue
        for tag in accumulator.Tags().get("scalars", []):
            for scalar in accumulator.Scalars(tag):
                rows.append(
                    {
                        "variant": variant,
                        "tag": tag,
                        "step": scalar.step,
                        "value": scalar.value,
                    }
                )
    return pd.DataFrame(rows)


def _format_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig: plt.Figure, path: Path, pdf: PdfPages | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.90))
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _suptitle(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=9, color="#475569")


def _ordered(df: pd.DataFrame, column: str = "variant") -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_order"] = out[column].map({name: i for i, name in enumerate(VARIANTS)})
    return out.sort_values("_order").drop(columns="_order")


def plot_summary_bars(
    frame: pd.DataFrame,
    output: Path,
    pdf: PdfPages,
    title: str,
    subtitle: str,
    mode_column: str = "mode",
    modes: Iterable[str] = ("raw", "cbf"),
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(19, 8.5))
    _suptitle(fig, title, subtitle)
    x = np.arange(len(VARIANTS))
    width = 0.36
    for ax, (kpi, column, ylabel) in zip(axes.flat, KPI_ORDER):
        for offset, mode in ((-width / 2, list(modes)[0]), (width / 2, list(modes)[1])):
            rows = frame[
                (frame[mode_column].astype(str).str.lower() == mode)
                & (frame["KPI"].astype(str) == kpi)
            ]
            rows = rows.set_index("variant").reindex(VARIANTS)
            values = pd.to_numeric(rows["Mean"], errors="coerce").to_numpy(float)
            errors = pd.to_numeric(rows["SD"], errors="coerce").fillna(0).to_numpy(float)
            bars = ax.bar(
                x + offset,
                values,
                width,
                yerr=errors,
                capsize=2.5,
                color=MODE_COLORS[mode],
                alpha=0.9,
                label=MODE_LABELS[mode],
            )
            if ax in (axes.flat[0],):
                ax.bar_label(bars, labels=[f"{v:.2g}" if np.isfinite(v) else "" for v in values], fontsize=6, padding=1)
        ax.set_title(kpi, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x, [SHORT[v] for v in VARIANTS], rotation=18, ha="right", fontsize=7)
        _format_axes(ax)
    axes.flat[0].legend(frameon=False, fontsize=8, loc="best")
    _save(fig, output, pdf)


def plot_episode_distributions(
    episodes: pd.DataFrame,
    output: Path,
    pdf: PdfPages,
    title: str,
    subtitle: str,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(19, 8.5))
    _suptitle(fig, title, subtitle)
    x = np.arange(len(VARIANTS))
    for ax, (kpi, column, ylabel) in zip(axes.flat, KPI_ORDER):
        raw_data = [episodes.loc[(episodes["variant"] == v) & (episodes["mode"] == "raw"), column].dropna().to_numpy(float) for v in VARIANTS]
        cbf_data = [episodes.loc[(episodes["variant"] == v) & (episodes["mode"] == "cbf"), column].dropna().to_numpy(float) for v in VARIANTS]
        bp1 = ax.boxplot(raw_data, positions=x - 0.19, widths=0.30, patch_artist=True, showfliers=False)
        bp2 = ax.boxplot(cbf_data, positions=x + 0.19, widths=0.30, patch_artist=True, showfliers=False)
        for box in bp1["boxes"]:
            box.set(facecolor=MODE_COLORS["raw"], alpha=0.65)
        for box in bp2["boxes"]:
            box.set(facecolor=MODE_COLORS["cbf"], alpha=0.65)
        for med in bp1["medians"] + bp2["medians"]:
            med.set(color="#111827", linewidth=1.1)
        ax.set_title(kpi, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x, [SHORT[v] for v in VARIANTS], rotation=18, ha="right", fontsize=7)
        _format_axes(ax)
    from matplotlib.patches import Patch

    axes.flat[0].legend(
        handles=[Patch(facecolor=MODE_COLORS["raw"], label="CBF OFF"), Patch(facecolor=MODE_COLORS["cbf"], label="CBF ON")],
        frameon=False,
        fontsize=8,
    )
    _save(fig, output, pdf)


def plot_post_block_trends(blocks: pd.DataFrame, output: Path, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    _suptitle(
        fig,
        "Post-training pooled-block variation",
        "Ten independent 20-episode blocks per variant/mode; lines show the block-level values used for the KPI SDs.",
    )
    for ax, (column, ylabel) in zip(axes.flat, POST_TREND_METRICS):
        for variant in VARIANTS:
            for mode, linestyle in (("raw", "--"), ("cbf", "-")):
                rows = blocks[(blocks["variant"] == variant) & (blocks["mode"] == mode)].sort_values("summary_block")
                if rows.empty:
                    continue
                ax.plot(
                    rows["summary_block"],
                    rows[column],
                    color=COLORS[variant],
                    linestyle=linestyle,
                    linewidth=1.6,
                    marker="o",
                    markersize=2.5,
                    alpha=0.9,
                )
        ax.set_xlabel("20-episode block", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        _format_axes(ax)
    from matplotlib.lines import Line2D

    variant_handles = [Line2D([0], [0], color=COLORS[v], lw=2, label=SHORT[v].replace("\n", " ")) for v in VARIANTS]
    mode_handles = [Line2D([0], [0], color="#334155", lw=2, linestyle=ls, label=MODE_LABELS[m]) for m, ls in (("raw", "--"), ("cbf", "-"))]
    axes.flat[0].legend(handles=variant_handles + mode_handles, fontsize=7, frameon=False, ncol=2)
    _save(fig, output, pdf)


def plot_training_episodes(train: pd.DataFrame, output: Path, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    _suptitle(
        fig,
        "Training episode traces",
        "Episode-level CSV traces across the 50,000 training timesteps; faint points are saved episodes and solid lines are 25-episode rolling means.",
    )
    for ax, (column, ylabel) in zip(axes.flat, TRAIN_METRICS):
        for variant in VARIANTS:
            rows = train[train.variant == variant].sort_values("global_timestep").copy()
            if rows.empty or column not in rows:
                continue
            y = pd.to_numeric(rows[column], errors="coerce")
            x = pd.to_numeric(rows["global_timestep"], errors="coerce")
            ax.scatter(x, y, s=5, alpha=0.12, color=COLORS[variant])
            smooth = y.rolling(25, min_periods=1).mean()
            ax.plot(x, smooth, color=COLORS[variant], linewidth=1.7, label=SHORT[variant].replace("\n", " "))
        ax.set_xlabel("Training timestep", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        _format_axes(ax)
    axes.flat[0].legend(frameon=False, fontsize=7, ncol=2)
    _save(fig, output, pdf)


def plot_tb_scalars(tb: pd.DataFrame, specs: list[tuple[str, str]], output: Path, pdf: PdfPages, title: str, subtitle: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(19, 10))
    _suptitle(fig, title, subtitle)
    for ax, (tag, ylabel) in zip(axes.flat, specs):
        for variant in VARIANTS:
            rows = tb[(tb.variant == variant) & (tb.tag == tag)].sort_values("step")
            if rows.empty:
                continue
            ax.plot(rows.step, rows.value, color=COLORS[variant], linewidth=1.6, label=SHORT[variant].replace("\n", " "))
        ax.set_xlabel("Training timestep", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        _format_axes(ax)
    for ax in axes.flat[len(specs):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=7, ncol=2)
    _save(fig, output, pdf)


def plot_tb_cbf(tb: pd.DataFrame, output: Path, pdf: PdfPages) -> None:
    specs = TB_CBF
    fig, axes = plt.subplots(3, 4, figsize=(19, 10))
    _suptitle(
        fig,
        "CBF and projected-actor TensorBoard diagnostics",
        "Tags are read directly from the saved TensorBoard event files; missing tags are left blank for variants that do not use the projected actor layer.",
    )
    for ax, (tag, ylabel) in zip(axes.flat, specs):
        for variant in VARIANTS:
            rows = tb[(tb.variant == variant) & (tb.tag == tag)].sort_values("step")
            if rows.empty:
                continue
            ax.plot(rows.step, rows.value, color=COLORS[variant], linewidth=1.5, label=SHORT[variant].replace("\n", " "))
        ax.set_xlabel("Training timestep", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        _format_axes(ax)
    for ax in axes.flat[len(specs):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=7, ncol=2)
    _save(fig, output, pdf)


def plot_fixed_state_summary(summary: pd.DataFrame, output: Path, pdf: PdfPages) -> None:
    metrics = [
        ("intervention_probability", "Intervention probability"),
        ("mean_external_correction_box_norm", "External correction norm"),
        ("mean_internal_mean_correction_norm", "Internal correction norm"),
        ("raw_policy_feasible_probability", "Raw policy feasible"),
        ("sample_intervention_probability", "Sample intervention probability"),
        ("fallback_probability", "Fallback probability"),
    ]
    strata = ["all", "dense", "intervention", "near_boundary", "normal", "overtaking"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    _suptitle(fig, "Fixed-state counterfactual summary", "Rows are the saved state-bank strata; values compare each policy on the same 100 states.")
    for ax, (column, title) in zip(axes.flat, metrics):
        matrix = summary.pivot(index="stratum", columns="variant", values=column).reindex(index=strata, columns=VARIANTS)
        im = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="YlGnBu", vmin=0 if "probability" in column else None)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(len(VARIANTS)), [SHORT[v] for v in VARIANTS], rotation=25, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(strata)), strata, fontsize=8)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    _save(fig, output, pdf)


def plot_fixed_state_actions(actions: pd.DataFrame, output: Path, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    _suptitle(fig, "Fixed-state action and correction distributions", "500 rows = 5 variants × 100 identical states; correction quantities are in normalized action-box units unless noted.")
    for variant in VARIANTS:
        rows = actions[actions.variant == variant]
        axes[0, 0].scatter(rows.state_h_min, rows.correction_box_norm, s=12, alpha=0.45, color=COLORS[variant], label=SHORT[variant].replace("\n", " "))
    axes[0, 0].set_xlabel("State minimum h")
    axes[0, 0].set_ylabel("Correction box norm")
    _format_axes(axes[0, 0])
    for variant in VARIANTS:
        rows = actions[actions.variant == variant]
        axes[0, 1].hist(rows.correction_box_norm, bins=20, alpha=0.35, color=COLORS[variant], label=SHORT[variant].replace("\n", " "))
    axes[0, 1].set_xlabel("Correction box norm")
    axes[0, 1].set_ylabel("State count")
    _format_axes(axes[0, 1])
    intervention = actions.assign(intervention_numeric=actions.intervention.astype(bool).astype(int)).groupby(["stratum", "variant"], as_index=False).intervention_numeric.mean()
    for variant in VARIANTS:
        rows = intervention[intervention.variant == variant]
        axes[0, 2].plot(rows.stratum, rows.intervention_numeric, marker="o", color=COLORS[variant], label=SHORT[variant].replace("\n", " "))
    axes[0, 2].set_ylabel("Intervention fraction")
    axes[0, 2].tick_params(axis="x", rotation=25, labelsize=8)
    _format_axes(axes[0, 2])
    for col, title, ax in [("policy_action_ax", "Policy action ax", axes[1, 0]), ("policy_action_ay", "Policy action ay", axes[1, 1])]:
        for variant in VARIANTS:
            rows = actions[actions.variant == variant]
            ax.hist(rows[col], bins=22, alpha=0.35, color=COLORS[variant], label=SHORT[variant].replace("\n", " "))
        ax.set_xlabel(title)
        ax.set_ylabel("State count")
        _format_axes(ax)
    axes[1, 2].scatter(actions.internal_mean_correction_norm, actions.correction_physical_norm, c=actions.active_constraint_count, cmap="viridis", s=16, alpha=0.55)
    axes[1, 2].set_xlabel("Internal mean correction norm")
    axes[1, 2].set_ylabel("Physical correction norm")
    _format_axes(axes[1, 2])
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    _save(fig, output, pdf)


def plot_state_bank(bank: pd.DataFrame, output: Path, pdf: PdfPages) -> None:
    specs = [
        ("h_min", "State minimum h"),
        ("h_dot", "h dot"),
        ("ttc_s", "TTC (s)"),
        ("vehicle_spacing_m", "Vehicle spacing (m)"),
        ("traffic_density_per_km", "Traffic density / km"),
        ("neighbor_count", "Neighbor count"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    _suptitle(fig, "State-bank distributions", "The 100 fixed states are stratified into five 20-state conditions and reused across all counterfactual policies.")
    strata = ["normal", "near_boundary", "intervention", "dense", "overtaking"]
    for ax, (column, title) in zip(axes.flat, specs):
        data = [bank.loc[bank.stratum == s, column].dropna().to_numpy(float) for s in strata]
        bp = ax.boxplot(data, tick_labels=strata, patch_artist=True, showfliers=False)
        for patch, s in zip(bp["boxes"], strata):
            patch.set(facecolor=plt.cm.tab10(strata.index(s)), alpha=0.55)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        _format_axes(ax)
    _save(fig, output, pdf)


def plot_occupancy_summary(occupancy: pd.DataFrame, output: Path, pdf: PdfPages) -> None:
    specs = [
        ("mean_h_min", "Mean h minimum"),
        ("mean_h_dot", "Mean h dot"),
        ("mean_ttc_s", "Mean TTC (s)"),
        ("mean_vehicle_spacing_m", "Mean spacing (m)"),
        ("mean_traffic_density_per_km", "Traffic density / km"),
        ("mean_correction_box_norm", "Mean correction norm"),
        ("mean_intervention", "Intervention rate"),
        ("mean_fallback_used", "Fallback rate"),
        ("minimum_h_min", "Minimum observed h"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    _suptitle(fig, "On-policy occupancy summary", "One 800-step occupancy rollout per variant; bars summarize the saved occupancy_summary.csv.")
    x = np.arange(len(VARIANTS))
    rows = occupancy.set_index("variant").reindex(VARIANTS)
    for ax, (column, ylabel) in zip(axes.flat, specs):
        values = pd.to_numeric(rows[column], errors="coerce").to_numpy(float)
        bars = ax.bar(x, values, color=[COLORS[v] for v in VARIANTS], alpha=0.88)
        ax.set_title(ylabel, fontsize=10)
        ax.set_xticks(x, [SHORT[v] for v in VARIANTS], rotation=20, ha="right", fontsize=7)
        ax.bar_label(bars, labels=[f"{v:.2g}" if np.isfinite(v) else "" for v in values], fontsize=7, padding=2)
        _format_axes(ax)
    _save(fig, output, pdf)


def plot_occupancy_traces(occupancy: pd.DataFrame, output: Path, pdf: PdfPages) -> None:
    specs = [
        ("h_min", "h minimum"),
        ("h_dot", "h dot"),
        ("ttc_s", "TTC (s)"),
        ("vehicle_spacing_m", "Spacing (m)"),
        ("traffic_density_per_km", "Density / km"),
        ("neighbor_count", "Neighbor count"),
        ("correction_box_norm", "Correction norm"),
        ("intervention", "Intervention rate"),
        ("fallback_used", "Fallback rate"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    _suptitle(fig, "On-policy occupancy traces", "Mean trajectory by scenario step across the saved 4,000 occupancy rows; shaded bands show the 10th–90th percentile.")
    for ax, (column, ylabel) in zip(axes.flat, specs):
        for variant in VARIANTS:
            rows = occupancy[occupancy.source_variant == variant].copy()
            rows["value"] = rows[column].astype(bool).astype(float) if rows[column].dtype == bool else pd.to_numeric(rows[column], errors="coerce")
            grouped = rows.groupby("scenario_step")["value"]
            stats = grouped.agg(mean="mean", q10=lambda x: x.quantile(0.10), q90=lambda x: x.quantile(0.90)).reset_index()
            ax.plot(stats.scenario_step, stats["mean"], color=COLORS[variant], linewidth=1.5, label=SHORT[variant].replace("\n", " "))
            ax.fill_between(stats.scenario_step, stats.q10, stats.q90, color=COLORS[variant], alpha=0.06)
        ax.set_xlabel("Scenario step", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        _format_axes(ax)
    axes.flat[0].legend(frameon=False, fontsize=7, ncol=2)
    _save(fig, output, pdf)


def plot_active_constraints(active: pd.DataFrame, output: Path, pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    _suptitle(fig, "Projected-CBF active-constraint distribution", "The saved counterfactual active_constraint_distribution.csv covers the two projected architectures.")
    rows = active.set_index("variant").reindex(["ppo_cbf_projected_reward_off", "ppo_cbf_projected"])
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for column, label, color in [("neighbor", "Neighbor constraint", "#E45756"), ("none", "No active constraint", "#72B7B2")]:
        values = pd.to_numeric(rows[column], errors="coerce").fillna(0).to_numpy(float)
        ax.bar(x, values, bottom=bottom, color=color, label=label)
        for i, value in enumerate(values):
            if value > 0.04:
                ax.text(i, bottom[i] + value / 2, f"{value:.2f}", ha="center", va="center", fontsize=10)
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of counterfactual states")
    ax.set_xticks(x, [SHORT[v].replace("\n", " ") for v in rows.index])
    ax.legend(frameon=False)
    _format_axes(ax)
    _save(fig, output, pdf)


def inventory_page(study: Path, output: Path, pdf: PdfPages) -> None:
    rows: list[dict[str, object]] = []
    paths = [
        ("Original scenario evaluation", study / "evaluation_scenarios.csv"),
        ("Original KPI summary", study / "ten_kpi_summary.csv"),
        ("Post-training raw episodes", study / "ppo_nominal" / "seed_307" / "pe" / "e.csv"),
        ("Post-training pooled blocks", study / "ppo_nominal" / "seed_307" / "pe" / "b.csv"),
        ("Post-training KPI summary", study / "post_train_200ep_kpis.csv"),
        ("Counterfactual actions", study / "counterfactuals" / "fixed_state_actions.csv"),
        ("On-policy occupancy", study / "counterfactuals" / "on_policy_occupancy.csv"),
    ]
    for label, path in paths:
        frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
        rows.append({"Dataset": label, "Rows": len(frame), "Columns": len(frame.columns), "File": str(path.relative_to(study))})
    fig, ax = plt.subplots(figsize=(17, 9.5))
    ax.axis("off")
    fig.suptitle("Canonical PPO progression graph report", fontsize=22, fontweight="bold", y=0.94)
    fig.text(0.5, 0.885, "All plots are generated from saved artifacts; no training or evaluation was rerun.", ha="center", fontsize=11)
    table = ax.table(cellText=pd.DataFrame(rows).astype(str).values, colLabels=list(rows[0]), loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1F4E79")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2:
            cell.set_facecolor("#EEF2F7")
    _save(fig, output, pdf)


def main() -> int:
    args = parse_args()
    study = args.study_dir.resolve()
    output = (args.output_dir or study / "figures").resolve()
    output.mkdir(parents=True, exist_ok=True)

    scenarios = _read_csv(study / "evaluation_scenarios.csv")
    final_summary = _read_csv(study / "ten_kpi_summary.csv")
    post_summary = _read_csv(study / "post_train_200ep_kpis.csv")
    post = _concat_variant_files(study, "pe/e.csv")
    blocks = _concat_variant_files(study, "pe/b.csv")
    training = _concat_variant_files(study, "training_episodes.csv")
    fixed_summary = _read_csv(study / "counterfactuals" / "fixed_state_summary.csv")
    fixed_actions = _read_csv(study / "counterfactuals" / "fixed_state_actions.csv")
    state_bank = _read_csv(study / "counterfactuals" / "state_bank.csv")
    occupancy_summary = _read_csv(study / "counterfactuals" / "occupancy_summary.csv")
    occupancy = _read_csv(study / "counterfactuals" / "on_policy_occupancy.csv")
    active = _read_csv(study / "counterfactuals" / "active_constraint_distribution.csv")
    tb = _load_tb(study)

    with PdfPages(output / "ppo_progression_graph_report.pdf") as pdf:
        inventory_page(study, output / "00_report_inventory.png", pdf)
        plot_summary_bars(
            final_summary,
            output / "01_final_10_scenario_kpis.png",
            pdf,
            "Original final evaluation: all KPI comparisons",
            "Ten deterministic scenarios per variant/mode; bars are Mean ± SD across scenarios.",
        )
        plot_episode_distributions(
            scenarios,
            output / "01b_final_10_scenario_distributions.png",
            pdf,
            "Original final evaluation: scenario-level distributions",
            "Each box contains the ten saved deterministic scenarios per variant and CBF mode.",
        )
        post_plot = post_summary.copy()
        post_plot["mode"] = post_plot["external_cbf"].map({"OFF": "raw", "ON": "cbf"}).fillna(post_plot["mode"])
        plot_summary_bars(
            post_plot,
            output / "02_post_training_200_episode_kpis.png",
            pdf,
            "Post-training evaluation: 200 episodes per CBF mode",
            "Corrected pooled summaries: 10 blocks × 20 episodes; bars are Mean ± SD across blocks.",
        )
        plot_episode_distributions(
            post,
            output / "03_post_training_episode_distributions.png",
            pdf,
            "Post-training episode distributions",
            "Each box contains 200 complete episodes per variant and external-CBF mode; boxes show distributions, not pooled-rate estimates.",
        )
        plot_post_block_trends(blocks, output / "04_post_training_block_trends.png", pdf)
        plot_training_episodes(training, output / "05_training_episode_traces.png", pdf)
        if not tb.empty:
            plot_tb_scalars(tb, TB_ROLLOUT, output / "06_tensorboard_rollout_scalars.png", pdf, "TensorBoard rollout scalars", "Saved PPO event files, 50 scalar points per 50,000-timestep run.")
            plot_tb_scalars(tb, TB_TRAIN, output / "07_tensorboard_optimization_scalars.png", pdf, "TensorBoard optimization scalars", "Stable-Baselines3 PPO optimization metrics from the saved event files.")
            plot_tb_cbf(tb, output / "08_tensorboard_cbf_diagnostics.png", pdf)
        plot_fixed_state_summary(fixed_summary, output / "09_counterfactual_fixed_state_summary.png", pdf)
        plot_fixed_state_actions(fixed_actions, output / "10_counterfactual_action_distributions.png", pdf)
        plot_state_bank(state_bank, output / "11_counterfactual_state_bank.png", pdf)
        plot_occupancy_summary(occupancy_summary, output / "12_occupancy_summary.png", pdf)
        plot_occupancy_traces(occupancy, output / "13_occupancy_traces.png", pdf)
        plot_active_constraints(active, output / "14_active_constraint_distribution.png", pdf)

    manifest = {
        "study_dir": str(study),
        "output_dir": str(output),
        "source_rows": {
            "evaluation_scenarios": int(len(scenarios)),
            "ten_kpi_summary": int(len(final_summary)),
            "post_training_episodes": int(len(post)),
            "post_training_blocks": int(len(blocks)),
            "post_training_kpis": int(len(post_summary)),
            "training_episodes": int(len(training)),
            "counterfactual_fixed_state_summary": int(len(fixed_summary)),
            "counterfactual_fixed_state_actions": int(len(fixed_actions)),
            "counterfactual_state_bank": int(len(state_bank)),
            "counterfactual_occupancy_summary": int(len(occupancy_summary)),
            "counterfactual_on_policy_occupancy": int(len(occupancy)),
            "counterfactual_active_constraints": int(len(active)),
            "tensorboard_scalar_rows": int(len(tb)),
        },
        "figures": sorted(path.name for path in output.glob("*.png")),
        "report_pdf": str((output / "ppo_progression_graph_report.pdf").resolve()),
    }
    (output / "graph_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
