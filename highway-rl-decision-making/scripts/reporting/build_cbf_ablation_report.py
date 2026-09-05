"""Build a visual report from schema-v4 CBF ablation evaluation artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


PIPELINE_SCHEMA_VERSION = 4
FIGURE_DPI = 220

VARIANTS = [
    "a_nominal",
    "b_filtered",
    "c_reward",
    "d_loss",
    "e_reward_actor",
    "f_random",
]
FACTORIAL_VARIANTS = ["b_filtered", "c_reward", "d_loss", "e_reward_actor"]
SETUPS = {
    "a_nominal": "Nominal DDPG; no training shield",
    "b_filtered": "Shielded; reward off; actor loss off",
    "c_reward": "Shielded; reward on; actor loss off",
    "d_loss": "Shielded; reward off; actor loss on",
    "e_reward_actor": "Shielded; reward on; actor loss on",
    "f_random": "Shared random-policy reference",
}
CELL_LABELS = {
    (variant, mode): f"{variant[0].upper()}{'1' if mode == 'raw' else '2'}"
    for variant in VARIANTS
    for mode in ("raw", "cbf")
}
MODE_COLORS = {"raw": "#7b8794", "cbf": "#1677b9"}

EFFECT_ORDER = [
    "reward_main_effect",
    "actor_loss_main_effect",
    "reward_actor_interaction",
]
EFFECT_LABELS = {
    "reward_main_effect": "Reward main",
    "actor_loss_main_effect": "Actor-loss main",
    "reward_actor_interaction": "Reward × loss",
}
COMPARISON_ORDER = [
    "runtime_filter_a",
    "filtered_experience",
    "reward_effect_loss_off",
    "loss_effect_reward_off",
    "reward_effect_loss_on",
    "loss_effect_reward_on",
    "runtime_filter_b",
    "runtime_filter_c",
    "runtime_filter_d",
    "runtime_filter_e",
    "actor_vs_random_with_cbf",
]
COMPARISON_LABELS = {
    "runtime_filter_a": "A2 − A1  Runtime filter",
    "filtered_experience": "B1 − A1  Filtered experience",
    "reward_effect_loss_off": "C1 − B1  Reward effect (loss off)",
    "loss_effect_reward_off": "D1 − B1  Loss effect (reward off)",
    "reward_effect_loss_on": "E1 − D1  Reward effect (loss on)",
    "loss_effect_reward_on": "E1 − C1  Loss effect (reward on)",
    "runtime_filter_b": "B2 − B1  Runtime filter",
    "runtime_filter_c": "C2 − C1  Runtime filter",
    "runtime_filter_d": "D2 − D1  Runtime filter",
    "runtime_filter_e": "E2 − E1  Runtime filter",
    "actor_vs_random_with_cbf": "E2 − F2  Actor beyond filter",
}

OVERVIEW_METRICS = [
    ("return_per_timestep", "Return / timestep (higher is better)"),
    ("ego_collisions_per_km", "Collisions / km (lower is better)"),
    ("h_min", r"Minimum safety value $h_{min}$ (higher is better)"),
    ("near_boundary_rate", "Near-boundary timestep rate (lower is better)"),
    ("IR", "CBF-demand intervention rate"),
    ("mean_delta_a", r"CBF-demand mean $|\Delta a|$"),
    ("mean_abs_speed_error", "Longitudinal speed-tracking error"),
    ("mean_jerk_norm", "Mean jerk norm (lower is better)"),
]
OPTIONAL_LATERAL_TRACKING_METRIC = (
    "mean_abs_target_lateral_error_m",
    "Lateral tracking error (m)",
)
FACTORIAL_METRICS = [
    ("return_per_timestep", "Return / timestep (higher is better)"),
    ("ego_collisions_per_km", "Collisions / km (lower is better)"),
    ("h_min", r"Minimum safety value $h_{min}$ (higher is better)"),
    ("mean_abs_speed_error", "Mean absolute speed error (lower is better)"),
]
FACTORIAL_TITLES = {
    "return_per_timestep": "Return / timestep",
    "ego_collisions_per_km": "Collisions / km",
    "h_min": r"Minimum $h$",
    "mean_abs_speed_error": "Speed error",
}
FILTER_METRICS = [
    ("IR", "Intervention rate, IR"),
    ("mean_delta_a", r"Mean $|\Delta a|$"),
    ("p95_delta_a", r"P95 $|\Delta a|$"),
    ("qp_failure_rate", "QP failure rate"),
    ("qp_fallback_rate", "QP fallback rate"),
]
PAIRED_METRICS = FACTORIAL_METRICS


@dataclass(frozen=True)
class StudyMetadata:
    study_dir: Path
    schema_version: int | None
    training_seed_count: int | None
    scenario_count: int | None
    timestep_budget: int | None
    training_timesteps: int | None
    traffic_model: str | None

    def footer(self) -> str:
        parts = [f"Source: {self.study_dir.name}"]
        if self.schema_version is not None:
            parts.append(f"schema v{self.schema_version}")
        if self.training_seed_count is not None:
            suffix = "seed" if self.training_seed_count == 1 else "seeds"
            parts.append(f"{self.training_seed_count} training {suffix}")
        if self.scenario_count is not None:
            parts.append(f"{self.scenario_count} paired scenarios / seed")
        if self.timestep_budget is not None:
            parts.append(f"{self.timestep_budget:,} timesteps / scenario")
        return " | ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _metadata(study_dir: Path, summary: pd.DataFrame) -> StudyMetadata:
    run_config = _load_json(study_dir / "run_config.json")
    manifest = _load_json(study_dir / "evaluation_manifest.json")
    schema = run_config.get("schema_version")
    if schema is not None and int(schema) != PIPELINE_SCHEMA_VERSION:
        raise ValueError(
            f"Expected CBF ablation schema v{PIPELINE_SCHEMA_VERSION}, got v{schema}."
        )
    seeds = run_config.get("seeds")
    if isinstance(seeds, list):
        seed_count: int | None = len(seeds)
    elif "training_seeds" in summary:
        values = pd.to_numeric(summary["training_seeds"], errors="coerce")
        seed_count = int(values.max()) if values.notna().any() else None
    else:
        seed_count = None
    protocol = run_config.get("evaluation_protocol", {})
    if not isinstance(protocol, dict):
        protocol = {}
    scenario_seeds = manifest.get("scenario_seeds")
    scenario_count = protocol.get("scenario_count")
    if scenario_count is None and isinstance(scenario_seeds, list):
        scenario_count = len(scenario_seeds)
    timestep_budget = protocol.get(
        "timestep_budget_per_scenario", manifest.get("timestep_budget_per_scenario")
    )
    traffic_model = manifest.get("traffic_model")
    if traffic_model is None:
        env_config = run_config.get("env_config", {})
        if isinstance(env_config, dict):
            traffic_model = env_config.get("traffic_model")
    return StudyMetadata(
        study_dir=study_dir,
        schema_version=int(schema) if schema is not None else PIPELINE_SCHEMA_VERSION,
        training_seed_count=int(seed_count) if seed_count is not None else None,
        scenario_count=int(scenario_count) if scenario_count is not None else None,
        timestep_budget=int(timestep_budget) if timestep_budget is not None else None,
        training_timesteps=(
            int(run_config["timesteps"]) if run_config.get("timesteps") is not None else None
        ),
        traffic_model=str(traffic_model) if traffic_model is not None else None,
    )


def _require_columns(frame: pd.DataFrame, columns: list[str], artifact: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(
            f"{artifact} is not a schema-v{PIPELINE_SCHEMA_VERSION} artifact; "
            f"missing columns: {', '.join(missing)}"
        )


def load_inputs(study_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "evaluation_summary.csv": study_dir / "evaluation_summary.csv",
        "paired_comparisons.csv": study_dir / "paired_comparisons.csv",
        "factorial_effects_summary.csv": study_dir / "factorial_effects_summary.csv",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing report inputs: {', '.join(missing)}")
    summary = pd.read_csv(paths["evaluation_summary.csv"])
    comparisons = pd.read_csv(paths["paired_comparisons.csv"])
    factorial = pd.read_csv(paths["factorial_effects_summary.csv"])

    summary_metrics = sorted(
        {metric for metric, _ in OVERVIEW_METRICS + FILTER_METRICS}
    )
    summary_columns = ["variant", "mode", "training_seeds"]
    for metric in summary_metrics:
        summary_columns.extend([f"{metric}_seed_mean", f"{metric}_seed_variance"])
    _require_columns(summary, summary_columns, "evaluation_summary.csv")

    paired_columns = ["comparison", "training_seed"] + [
        f"delta_{metric}"
        for metric, _ in PAIRED_METRICS + [("mean_jerk_norm", "Jerk")]
    ]
    _require_columns(comparisons, paired_columns, "paired_comparisons.csv")

    factorial_columns = ["effect", "mode", "training_seeds"]
    for metric, _ in FACTORIAL_METRICS:
        factorial_columns.extend(
            [
                f"effect_{metric}_seed_mean",
                f"effect_{metric}_seed_variance",
            ]
        )
    _require_columns(factorial, factorial_columns, "factorial_effects_summary.csv")

    expected_cells = {(variant, mode) for variant in VARIANTS for mode in ("raw", "cbf")}
    actual_cells = set(zip(summary["variant"].astype(str), summary["mode"].astype(str)))
    absent_cells = sorted(expected_cells - actual_cells)
    if absent_cells:
        formatted = ", ".join(f"{variant}:{mode}" for variant, mode in absent_cells)
        raise ValueError(f"evaluation_summary.csv is missing design cells: {formatted}")
    expected_effect_cells = {
        (effect, mode) for effect in EFFECT_ORDER for mode in ("raw", "cbf")
    }
    actual_effect_cells = set(
        zip(factorial["effect"].astype(str), factorial["mode"].astype(str))
    )
    absent_effect_cells = sorted(expected_effect_cells - actual_effect_cells)
    if absent_effect_cells:
        formatted = ", ".join(
            f"{effect}:{mode}" for effect, mode in absent_effect_cells
        )
        raise ValueError(
            f"factorial_effects_summary.csv is missing effect cells: {formatted}"
        )

    variant_order = {variant: index for index, variant in enumerate(VARIANTS)}
    mode_order = {"raw": 0, "cbf": 1}
    summary = summary.assign(
        _variant_order=summary["variant"].map(variant_order),
        _mode_order=summary["mode"].map(mode_order),
    ).sort_values(["_variant_order", "_mode_order"])
    return summary.drop(columns=["_variant_order", "_mode_order"]), comparisons, factorial


def _rows(summary: pd.DataFrame, mode: str, variants: list[str] = VARIANTS) -> pd.DataFrame:
    return summary[summary["mode"] == mode].set_index("variant").reindex(variants)


def _has_summary_metric(summary: pd.DataFrame, metric: str) -> bool:
    return {
        f"{metric}_seed_mean",
        f"{metric}_seed_variance",
    }.issubset(summary.columns)


def _has_finite_summary_metric(summary: pd.DataFrame, metric: str) -> bool:
    return _has_summary_metric(summary, metric) and bool(
        pd.to_numeric(summary[f"{metric}_seed_mean"], errors="coerce").notna().any()
    )


def _overview_metrics(summary: pd.DataFrame) -> list[tuple[str, str]]:
    metrics = list(OVERVIEW_METRICS)
    lateral_metric, _ = OPTIONAL_LATERAL_TRACKING_METRIC
    if _has_finite_summary_metric(summary, lateral_metric):
        metrics.insert(-1, OPTIONAL_LATERAL_TRACKING_METRIC)
    return metrics


def _rollout_metric(summary: pd.DataFrame, metric: str, mode: str) -> str:
    """Use shadow-filter demand for raw rollouts and applied demand for CBF rollouts."""

    shadow_metric = {
        "IR": "shadow_IR",
        "mean_delta_a": "shadow_mean_delta_a",
    }.get(metric)
    if mode == "raw" and shadow_metric and _has_summary_metric(summary, shadow_metric):
        return shadow_metric
    return metric


def _seed_means(frame: pd.DataFrame, metric: str) -> np.ndarray:
    return pd.to_numeric(frame[f"{metric}_seed_mean"], errors="coerce").to_numpy(float)


def _seed_sd(frame: pd.DataFrame, metric: str) -> np.ndarray:
    variance = pd.to_numeric(
        frame[f"{metric}_seed_variance"], errors="coerce"
    ).to_numpy(float)
    return np.sqrt(np.clip(variance, 0.0, None))


def _normal_ci(values: pd.Series) -> tuple[float, float, int]:
    by_seed = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    n = len(by_seed)
    mean = float(np.mean(by_seed)) if n else np.nan
    if n < 2:
        return mean, np.nan, n
    sem = float(np.std(by_seed, ddof=1) / np.sqrt(n))
    return mean, 1.96 * sem, n


def _factorial_ci(row: pd.Series, metric: str) -> float:
    variance = pd.to_numeric(
        pd.Series([row.get(f"effect_{metric}_seed_variance")]), errors="coerce"
    ).iloc[0]
    n = pd.to_numeric(pd.Series([row.get("training_seeds")]), errors="coerce").iloc[0]
    if not np.isfinite(variance) or not np.isfinite(n) or n < 2:
        return np.nan
    return float(1.96 * np.sqrt(max(float(variance), 0.0) / float(n)))


def add_footer(fig: plt.Figure, metadata: StudyMetadata) -> None:
    fig.text(0.01, 0.012, metadata.footer(), fontsize=8, color="#4a5560")


def save(fig: plt.Figure, path: Path, pdf: PdfPages) -> None:
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def grouped_overview(
    summary: pd.DataFrame,
    figures: Path,
    pdf: PdfPages,
    metadata: StudyMetadata,
) -> None:
    metrics = _overview_metrics(summary)
    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    fig.suptitle(
        "Raw actor versus CBF-shielded rollout metrics",
        fontsize=18,
        fontweight="bold",
    )
    x = np.arange(len(VARIANTS))
    width = 0.36
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for offset, mode in ((-width / 2, "raw"), (width / 2, "cbf")):
            rows = _rows(summary, mode)
            displayed_metric = _rollout_metric(summary, metric, mode)
            ax.bar(
                x + offset,
                _seed_means(rows, displayed_metric),
                width,
                yerr=np.nan_to_num(_seed_sd(rows, displayed_metric), nan=0.0),
                capsize=3,
                color=MODE_COLORS[mode],
                alpha=0.9,
                label="Raw deployment" if mode == "raw" else "CBF deployment",
            )
        ax.axhline(0, color="#a0a8af", linewidth=0.7)
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks(x, [variant[0].upper() for variant in VARIANTS])
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.ravel()[len(metrics) :]:
        ax.axis("off")
    axes[0, 0].legend(frameon=False, loc="best")
    fig.text(
        0.5,
        0.938,
        "Bars are seed means ±1 SD. CBF-demand metrics use shadow filtering in raw rollouts and applied filtering in shielded rollouts.",
        ha="center",
        fontsize=9,
    )
    add_footer(fig, metadata)
    save(fig, figures / "01_raw_vs_cbf_deployment.png", pdf)


def factorial_figure(
    summary: pd.DataFrame,
    factorial: pd.DataFrame,
    figures: Path,
    pdf: PdfPages,
    metadata: StudyMetadata,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(19, 10))
    fig.suptitle("2×2 reward-penalty × actor-CBF-loss design", fontsize=18, fontweight="bold")
    raw = _rows(summary, "raw", FACTORIAL_VARIANTS)
    reward_x = np.asarray([0.0, 1.0])
    for column_index, (ax, (metric, _)) in enumerate(
        zip(axes[0], FACTORIAL_METRICS)
    ):
        short_title = FACTORIAL_TITLES[metric]
        means = _seed_means(raw, metric)
        errors = np.nan_to_num(_seed_sd(raw, metric), nan=0.0)
        ax.errorbar(
            reward_x,
            means[[0, 1]],
            yerr=errors[[0, 1]],
            marker="o",
            capsize=3,
            linewidth=2,
            color="#5b6470",
            label="Actor loss off (B, C)",
        )
        ax.errorbar(
            reward_x,
            means[[2, 3]],
            yerr=errors[[2, 3]],
            marker="o",
            capsize=3,
            linewidth=2,
            color="#d1495b",
            label="Actor loss on (D, E)",
        )
        ax.set_xticks(reward_x, ["Reward off", "Reward on"])
        ax.set_title(f"Raw cells: {short_title}", fontsize=10)
        ax.grid(axis="y", alpha=0.25)

        effect_rows = factorial[factorial["effect"].isin(EFFECT_ORDER)].copy()
        effect_rows["effect"] = pd.Categorical(
            effect_rows["effect"], EFFECT_ORDER, ordered=True
        )
        for offset, mode in ((-0.18, "raw"), (0.18, "cbf")):
            mode_rows = (
                effect_rows[effect_rows["mode"] == mode]
                .sort_values("effect")
                .set_index("effect")
                .reindex(EFFECT_ORDER)
            )
            values = pd.to_numeric(
                mode_rows[f"effect_{metric}_seed_mean"], errors="coerce"
            ).to_numpy(float)
            cis = np.asarray(
                [_factorial_ci(row, metric) for _, row in mode_rows.iterrows()],
                dtype=float,
            )
            axes[1, column_index].bar(
                np.arange(len(EFFECT_ORDER)) + offset,
                values,
                0.34,
                yerr=np.nan_to_num(cis, nan=0.0),
                capsize=3,
                color=MODE_COLORS[mode],
                alpha=0.9,
                label="Raw" if mode == "raw" else "CBF",
            )
        effect_ax = axes[1, column_index]
        effect_ax.axhline(0, color="#334155", linewidth=0.8)
        effect_ax.set_xticks(
            np.arange(len(EFFECT_ORDER)),
            [EFFECT_LABELS[effect] for effect in EFFECT_ORDER],
            rotation=13,
            ha="right",
        )
        effect_ax.set_title(f"Factorial effects: {short_title}", fontsize=10)
        effect_ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)
    fig.text(
        0.5,
        0.938,
        "Top: the four factorial cells. Bottom: reward and loss main effects plus E − D − C + B interaction; effect whiskers are paired 95% normal-approximation CIs across training seeds.",
        ha="center",
        fontsize=8.8,
    )
    add_footer(fig, metadata)
    save(fig, figures / "02_factorial_main_and_interaction_effects.png", pdf)


def filter_dependence(
    summary: pd.DataFrame,
    figures: Path,
    pdf: PdfPages,
    metadata: StudyMetadata,
) -> None:
    rows = _rows(summary, "cbf")
    x = np.arange(len(VARIANTS))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("Filter load under CBF-shielded deployment", fontsize=18, fontweight="bold")
    for ax, (metric, title) in zip(axes.ravel(), FILTER_METRICS):
        ax.bar(
            x,
            _seed_means(rows, metric),
            yerr=np.nan_to_num(_seed_sd(rows, metric), nan=0.0),
            capsize=4,
            color="#1677b9",
            alpha=0.88,
        )
        ax.set_xticks(x, [CELL_LABELS[(variant, "cbf")] for variant in VARIANTS])
        ax.set_title(title, fontsize=11)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[-1].axis("off")
    fig.text(
        0.5,
        0.938,
        "Lower intervention and correction magnitudes indicate less runtime dependence on the shield.",
        ha="center",
        fontsize=9,
    )
    add_footer(fig, metadata)
    save(fig, figures / "03_filter_load.png", pdf)


def paired_comparison_figure(
    comparisons: pd.DataFrame,
    figures: Path,
    pdf: PdfPages,
    metadata: StudyMetadata,
) -> pd.DataFrame:
    present = [
        comparison
        for comparison in COMPARISON_ORDER
        if comparison in set(comparisons["comparison"].astype(str))
    ]
    stat_rows: list[dict[str, float | int | str]] = []
    for comparison in present:
        frame = comparisons[comparisons["comparison"] == comparison]
        row: dict[str, float | int | str] = {
            "comparison": comparison,
            "effect": COMPARISON_LABELS[comparison],
            "training_seeds": int(frame["training_seed"].nunique()),
        }
        for metric, _ in PAIRED_METRICS + [("mean_jerk_norm", "Jerk")]:
            seed_values = frame.groupby("training_seed", sort=False)[
                f"delta_{metric}"
            ].mean()
            mean, ci, _ = _normal_ci(seed_values)
            row[f"delta_{metric}"] = mean
            row[f"delta_{metric}_ci95"] = ci
        stat_rows.append(row)
    stats = pd.DataFrame(stat_rows)
    if stats.empty:
        raise ValueError("paired_comparisons.csv contains no recognized schema-v4 comparisons")

    y = np.arange(len(stats))
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), sharey=True)
    fig.suptitle("Paired seed-level ablation comparisons (left minus right)", fontsize=18, fontweight="bold")
    for ax, (metric, title) in zip(axes.ravel(), PAIRED_METRICS):
        column = f"delta_{metric}"
        values = pd.to_numeric(stats[column], errors="coerce").to_numpy(float)
        errors = pd.to_numeric(stats[f"{column}_ci95"], errors="coerce").to_numpy(float)
        higher_is_better = metric in {"return_per_timestep", "h_min"}
        favorable = values >= 0 if higher_is_better else values <= 0
        colors = np.where(favorable, "#2a9d8f", "#d1495b")
        ax.barh(
            y,
            values,
            xerr=np.nan_to_num(errors, nan=0.0),
            color=colors,
            alpha=0.86,
            capsize=3,
        )
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.set_title(f"Δ {title}", fontsize=10.5)
        ax.grid(axis="x", alpha=0.25)
        ax.set_yticks(y, stats["effect"], fontsize=8)
    axes[0, 0].invert_yaxis()
    fig.text(
        0.5,
        0.938,
        "Each row first aggregates scenarios within a training seed; whiskers are paired 95% normal-approximation CIs across seeds and are omitted when not estimable.",
        ha="center",
        fontsize=8.8,
    )
    add_footer(fig, metadata)
    save(fig, figures / "04_paired_seed_effects.png", pdf)
    return stats


def table_page(
    pdf: PdfPages,
    title: str,
    subtitle: str,
    table: pd.DataFrame,
    metadata: StudyMetadata,
    fontsize: int = 8,
) -> None:
    fig, ax = plt.subplots(figsize=(17, 9.5))
    ax.axis("off")
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.97)
    fig.text(0.5, 0.928, subtitle, ha="center", fontsize=9)
    rendered = ax.table(
        cellText=table.astype(str).values,
        colLabels=table.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(fontsize)
    rendered.scale(1, 1.5)
    for (row, _), cell in rendered.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2:
            cell.set_facecolor("#f1f5f9")
    add_footer(fig, metadata)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _fmt_mean_sd(row: pd.Series, metric: str, decimals: int = 2) -> str:
    mean = float(row[f"{metric}_seed_mean"])
    variance = float(row[f"{metric}_seed_variance"])
    if np.isfinite(variance):
        return f"{mean:.{decimals}f} ± {np.sqrt(max(variance, 0.0)):.{decimals}f}"
    return f"{mean:.{decimals}f}"


def _fmt_effect(row: pd.Series, metric: str, decimals: int = 2) -> str:
    mean = float(row[f"effect_{metric}_seed_mean"])
    ci = _factorial_ci(row, metric)
    if np.isfinite(ci):
        return f"{mean:.{decimals}f} ± {ci:.{decimals}f}"
    return f"{mean:.{decimals}f}"


def _fmt_paired(row: pd.Series, metric: str, decimals: int = 2) -> str:
    mean = float(row[f"delta_{metric}"])
    ci = float(row[f"delta_{metric}_ci95"])
    if np.isfinite(ci):
        return f"{mean:.{decimals}f} ± {ci:.{decimals}f}"
    return f"{mean:.{decimals}f}"


def _title_page(pdf: PdfPages, metadata: StudyMetadata) -> None:
    fig = plt.figure(figsize=(17, 9.5))
    fig.suptitle("CBF Filter Internalization Ablation", fontsize=25, fontweight="bold", y=0.91)
    details: list[str] = ["2×2 reward penalty × actor CBF loss"]
    if metadata.training_timesteps is not None:
        details.append(f"{metadata.training_timesteps:,} training timesteps")
    if metadata.traffic_model is not None:
        details.append(f"{metadata.traffic_model.upper()} traffic")
    fig.text(0.5, 0.82, " | ".join(details), ha="center", fontsize=14)
    fig.text(0.10, 0.68, "Registered design", fontsize=16, fontweight="bold")
    fig.text(
        0.12,
        0.57,
        "B: reward off / loss off     C: reward on / loss off\n"
        "D: reward off / loss on      E: reward on / loss on\n"
        "A is the nominal-training contextual control; F is the shared random reference.",
        fontsize=13,
        linespacing=1.65,
    )
    fig.text(0.10, 0.39, "Statistical reading", fontsize=16, fontweight="bold")
    fig.text(
        0.12,
        0.29,
        "Scenarios are aggregated within each training seed. Main effects, interactions, and paired\n"
        "comparisons therefore use independent seed replicates rather than treating scenarios as\n"
        "independent policy-training replicates.",
        fontsize=12.5,
        linespacing=1.5,
    )
    fig.text(
        0.12,
        0.13,
        "Attribution caveat: the environment executes the shielded action while the critic is queried at\n"
        "the nominal action. The reward penalty corrects value attribution indirectly; the actor loss\n"
        "directly changes the nominal action map.",
        fontsize=11.5,
        color="#5b2c2c",
        linespacing=1.4,
    )
    add_footer(fig, metadata)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_report(study_dir: Path, output_dir: Path | None = None) -> Path:
    study_dir = study_dir.resolve()
    output_dir = (output_dir or study_dir / "visual_report").resolve()
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    summary, comparisons, factorial = load_inputs(study_dir)
    metadata = _metadata(study_dir, summary)
    report_path = output_dir / "cbf_filter_ablation_report.pdf"
    with PdfPages(report_path) as pdf:
        _title_page(pdf, metadata)
        grouped_overview(summary, figures, pdf, metadata)
        factorial_figure(summary, factorial, figures, pdf, metadata)
        filter_dependence(summary, figures, pdf, metadata)
        paired_stats = paired_comparison_figure(
            comparisons, figures, pdf, metadata
        )

        deployment_data: dict[str, Any] = {
            "Cell": [
                CELL_LABELS[(str(row.variant), str(row.mode))]
                for row in summary.itertuples()
            ],
            "Training setup": [SETUPS[str(value)] for value in summary["variant"]],
            "Deploy": [
                "Raw" if str(mode) == "raw" else "CBF" for mode in summary["mode"]
            ],
            "Seeds": [
                "Shared" if str(variant) == "f_random" else str(int(seed_count))
                for variant, seed_count in zip(
                    summary["variant"], summary["training_seeds"]
                )
            ],
            "Return / step": summary.apply(
                lambda row: _fmt_mean_sd(row, "return_per_timestep", 3), axis=1
            ),
            "Collisions / km": summary.apply(
                lambda row: _fmt_mean_sd(row, "ego_collisions_per_km", 2), axis=1
            ),
            "h_min": summary.apply(lambda row: _fmt_mean_sd(row, "h_min", 2), axis=1),
            "Near-boundary rate": summary.apply(
                lambda row: _fmt_mean_sd(row, "near_boundary_rate", 3), axis=1
            ),
            "CBF-demand IR": summary.apply(
                lambda row: _fmt_mean_sd(
                    row,
                    _rollout_metric(summary, "IR", str(row["mode"])),
                    3,
                ),
                axis=1,
            ),
            "CBF-demand mean |Δa|": summary.apply(
                lambda row: _fmt_mean_sd(
                    row,
                    _rollout_metric(summary, "mean_delta_a", str(row["mode"])),
                    3,
                ),
                axis=1,
            ),
            "Speed tracking": summary.apply(
                lambda row: _fmt_mean_sd(row, "mean_abs_speed_error", 2), axis=1
            ),
        }
        lateral_metric, _ = OPTIONAL_LATERAL_TRACKING_METRIC
        if _has_finite_summary_metric(summary, lateral_metric):
            deployment_data["Lateral tracking (m)"] = summary.apply(
                lambda row: _fmt_mean_sd(row, lateral_metric, 2), axis=1
            )
        deployment_data["Jerk"] = summary.apply(
            lambda row: _fmt_mean_sd(row, "mean_jerk_norm", 2), axis=1
        )
        deployment_table = pd.DataFrame(deployment_data)
        table_page(
            pdf,
            "All deployment results",
            "Mean ± seed SD; raw CBF-demand columns are shadow-filter values and shielded columns are applied values.",
            deployment_table,
            metadata,
            fontsize=5.8,
        )

        cbf = _rows(summary, "cbf").reset_index()
        filter_table = pd.DataFrame(
            {
                "Cell": [CELL_LABELS[(str(value), "cbf")] for value in cbf["variant"]],
                "Training setup": [SETUPS[str(value)] for value in cbf["variant"]],
                "IR": cbf.apply(lambda row: _fmt_mean_sd(row, "IR", 3), axis=1),
                "Mean |Δa|": cbf.apply(
                    lambda row: _fmt_mean_sd(row, "mean_delta_a", 3), axis=1
                ),
                "P95 |Δa|": cbf.apply(
                    lambda row: _fmt_mean_sd(row, "p95_delta_a", 3), axis=1
                ),
                "QP failure": cbf.apply(
                    lambda row: _fmt_mean_sd(row, "qp_failure_rate", 3), axis=1
                ),
                "QP fallback": cbf.apply(
                    lambda row: _fmt_mean_sd(row, "qp_fallback_rate", 3), axis=1
                ),
            }
        )
        table_page(
            pdf,
            "Filter-load results",
            "Reported for CBF-shielded deployment (A2–F2).",
            filter_table,
            metadata,
            fontsize=8,
        )

        factorial_sorted = factorial.copy()
        factorial_sorted["effect"] = pd.Categorical(
            factorial_sorted["effect"], EFFECT_ORDER, ordered=True
        )
        factorial_sorted["mode"] = pd.Categorical(
            factorial_sorted["mode"], ["raw", "cbf"], ordered=True
        )
        factorial_sorted = factorial_sorted.sort_values(["mode", "effect"])
        factorial_table = pd.DataFrame(
            {
                "Deployment": factorial_sorted["mode"].map(
                    {"raw": "Raw", "cbf": "CBF"}
                ),
                "Effect": factorial_sorted["effect"].map(EFFECT_LABELS),
                "Seeds": factorial_sorted["training_seeds"].astype(int),
                "Return / step [95% CI]": factorial_sorted.apply(
                    lambda row: _fmt_effect(row, "return_per_timestep", 3), axis=1
                ),
                "Collisions / km [95% CI]": factorial_sorted.apply(
                    lambda row: _fmt_effect(row, "ego_collisions_per_km", 2), axis=1
                ),
                "h_min [95% CI]": factorial_sorted.apply(
                    lambda row: _fmt_effect(row, "h_min", 2), axis=1
                ),
                "Speed error [95% CI]": factorial_sorted.apply(
                    lambda row: _fmt_effect(row, "mean_abs_speed_error", 2), axis=1
                ),
            }
        )
        table_page(
            pdf,
            "Factorial main and interaction effects",
            "Effects use paired 95% normal-approximation CIs; interaction is E − D − C + B.",
            factorial_table,
            metadata,
            fontsize=7.5,
        )

        paired_table = pd.DataFrame(
            {
                "Comparison (left − right)": paired_stats["effect"],
                "Seeds": paired_stats["training_seeds"],
                "Δ Return / step [95% CI]": paired_stats.apply(
                    lambda row: _fmt_paired(row, "return_per_timestep", 3), axis=1
                ),
                "Δ Collisions / km [95% CI]": paired_stats.apply(
                    lambda row: _fmt_paired(row, "ego_collisions_per_km", 2), axis=1
                ),
                "Δ h_min [95% CI]": paired_stats.apply(
                    lambda row: _fmt_paired(row, "h_min", 2), axis=1
                ),
                "Δ Speed error [95% CI]": paired_stats.apply(
                    lambda row: _fmt_paired(row, "mean_abs_speed_error", 2), axis=1
                ),
                "Δ Jerk [95% CI]": paired_stats.apply(
                    lambda row: _fmt_paired(row, "mean_jerk_norm", 2), axis=1
                ),
            }
        )
        table_page(
            pdf,
            "Paired seed-level ablation comparisons",
            "Paired 95% normal-approximation CIs; positive is favorable for return/h_min and negative for error metrics.",
            paired_table,
            metadata,
            fontsize=6.8,
        )

    deployment_table.to_csv(output_dir / "deployment_results_table.csv", index=False)
    filter_table.to_csv(output_dir / "filter_load_table.csv", index=False)
    factorial_table.to_csv(output_dir / "factorial_effects_table.csv", index=False)
    paired_table.to_csv(output_dir / "paired_effects_table.csv", index=False)
    print(f"Created {report_path}")
    print(f"Created figures in {figures}")
    return report_path


def main() -> None:
    args = parse_args()
    build_report(args.study_dir, args.output_dir)


if __name__ == "__main__":
    main()
