from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


import scripts.reporting.build_cbf_ablation_report as report


SUMMARY_METRICS = [
    "return_per_timestep",
    "ego_collisions_per_km",
    "h_min",
    "near_boundary_rate",
    "mean_abs_speed_error",
    "mean_abs_target_lateral_error_m",
    "mean_jerk_norm",
    "IR",
    "mean_delta_a",
    "shadow_IR",
    "shadow_mean_delta_a",
    "p95_delta_a",
    "qp_failure_rate",
    "qp_fallback_rate",
]
PAIRED_METRICS = [
    "return_per_timestep",
    "ego_collisions_per_km",
    "h_min",
    "mean_abs_speed_error",
    "mean_jerk_norm",
]


def _write_current_schema_artifacts(study_dir: Path) -> None:
    summary_rows: list[dict[str, float | int | str]] = []
    for variant_index, variant in enumerate(report.VARIANTS):
        for mode_index, mode in enumerate(("raw", "cbf")):
            row: dict[str, float | int | str] = {
                "variant": variant,
                "mode": mode,
                "training_seeds": 0 if variant == "f_random" else 3,
                "paired_training_seed_replicates": 3,
            }
            for metric_index, metric in enumerate(SUMMARY_METRICS):
                row[f"{metric}_seed_mean"] = (
                    0.1 * (variant_index + 1)
                    + 0.03 * mode_index
                    + 0.01 * metric_index
                )
                row[f"{metric}_seed_variance"] = (
                    float("nan") if variant == "f_random" else 0.0025
                )
            summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        study_dir / "evaluation_summary.csv", index=False
    )

    paired_rows: list[dict[str, float | int | str]] = []
    for comparison_index, comparison in enumerate(report.COMPARISON_ORDER):
        for seed_index, seed in enumerate((307, 1307, 2307)):
            row = {
                "comparison": comparison,
                "training_seed": seed,
                "left": "left:raw",
                "right": "right:raw",
            }
            for metric_index, metric in enumerate(PAIRED_METRICS):
                row[f"delta_{metric}"] = (
                    0.02 * comparison_index
                    - 0.01 * metric_index
                    + 0.005 * seed_index
                )
            paired_rows.append(row)
    pd.DataFrame(paired_rows).to_csv(
        study_dir / "paired_comparisons.csv", index=False
    )

    factorial_rows: list[dict[str, float | int | str]] = []
    for mode_index, mode in enumerate(("raw", "cbf")):
        for effect_index, effect in enumerate(report.EFFECT_ORDER):
            row = {
                "effect": effect,
                "mode": mode,
                "training_seeds": 3,
                "formula": "synthetic paired contrast",
            }
            for metric_index, metric in enumerate(
                [name for name, _ in report.FACTORIAL_METRICS]
            ):
                row[f"effect_{metric}_seed_mean"] = (
                    0.05 * effect_index + 0.02 * mode_index + 0.01 * metric_index
                )
                row[f"effect_{metric}_seed_variance"] = 0.0009
            factorial_rows.append(row)
    pd.DataFrame(factorial_rows).to_csv(
        study_dir / "factorial_effects_summary.csv", index=False
    )

    (study_dir / "run_config.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "timesteps": 1_234,
                "seeds": [307, 1307, 2307],
                "evaluation_protocol": {
                    "scenario_count": 4,
                    "timestep_budget_per_scenario": 25,
                },
                "env_config": {"traffic_model": "mtm"},
            }
        ),
        encoding="utf-8",
    )


def test_schema_v4_report_smoke(tmp_path, monkeypatch):
    study_dir = tmp_path / "synthetic_study"
    output_dir = tmp_path / "report"
    study_dir.mkdir()
    _write_current_schema_artifacts(study_dir)
    monkeypatch.setattr(report, "FIGURE_DPI", 45)

    report_path = report.build_report(study_dir, output_dir)

    assert report_path == output_dir / "cbf_filter_ablation_report.pdf"
    assert report_path.stat().st_size > 0
    assert {path.name for path in (output_dir / "figures").glob("*.png")} == {
        "01_raw_vs_cbf_deployment.png",
        "02_factorial_main_and_interaction_effects.png",
        "03_filter_load.png",
        "04_paired_seed_effects.png",
    }
    assert (output_dir / "factorial_effects_table.csv").exists()
    deployment = pd.read_csv(output_dir / "deployment_results_table.csv")
    assert set(deployment["Cell"]) == {
        f"{letter}{deployment_mode}"
        for letter in "ABCDEF"
        for deployment_mode in (1, 2)
    }
    assert {
        "Return / step",
        "Collisions / km",
        "h_min",
        "Near-boundary rate",
        "CBF-demand IR",
        "CBF-demand mean |Δa|",
        "Speed tracking",
        "Lateral tracking (m)",
    }.issubset(deployment.columns)
    paired = pd.read_csv(output_dir / "paired_effects_table.csv")
    assert set(paired["Seeds"]) == {3}
