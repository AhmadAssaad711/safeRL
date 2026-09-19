"""Build the safety-matrix 1M results tables: policy coverage plus two KPI tables.

Consumes ``post_train_200ep_kpis.csv`` written by
``scripts/ops/run_safety_matrix_study.py`` for each arm/seed of the five-arm,
seven-seed, 1,000,000-timestep CBF safety matrix. Produces three tables:

1. policy coverage -- which policies were evaluated under CBF-OFF/CBF-ON and
   with how many seeds/episodes;
2. safety/CBF KPIs (collisions, barrier margin, QP failures, intervention,
   correction norm);
3. task/comfort KPIs (return, episode length, completion, speed/lateral
   tracking, jerk).

Each per-seed KPI file already reports a pooled-block Mean/SD (10 blocks of 20
episodes). This script treats that per-seed Mean as one independent replicate
and aggregates across seeds as mean +/- SD of the seven seed means (n=7),
rather than pooling all 1,400 episodes together, so the reported spread
reflects seed-to-seed variation and not just within-seed evaluation noise --
the same distinction the safety-matrix study and the state-cost statistical
correction were built around.

The two ``C*`` arms (differentiable-projection variants) are excluded by
default: their deployed action is the CBF-projected one even under
"CBF-OFF" evaluation, so they are not comparable to the other three arms on
the same axis without the separate raw-actor-mean evaluation. Pass
``--arms`` to include them explicitly.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_STUDY_DIR = Path.home() / "AppData" / "Local" / "Temp" / "saferl_runs" / "safety_matrix_1M"
DEFAULT_SEEDS = [307, 327, 347, 367, 387, 407, 427]

# Arm label -> short display code used in the tables.
ALL_ARMS = {
    "A1_nominal": "A1",
    "B1_state_cost_raw": "B1",
    "B2_cbf_nd_reward_actor": "B2",
    "C1_cbf_diff_reward_only": "C1",
    "C2_cbf_projected": "C2",
}
DEFAULT_ARMS = ["A1_nominal", "B1_state_cost_raw", "B2_cbf_nd_reward_actor"]
EXCLUDED_BY_DEFAULT = ["C1_cbf_diff_reward_only", "C2_cbf_projected"]

MODE_LABELS = {"raw": "CBF-OFF", "cbf": "CBF-ON"}

SAFETY_KPIS = [
    "Ego collisions / km",
    "Minimum h",
    "QP failure rate",
    "Intervention rate",
    "Correction norm",
]
TASK_KPIS = [
    "Episode return",
    "Episode length (steps)",
    "Distance-based completion rate",
    "Abs speed error (m/s)",
    "Mean lateral tracking error (m)",
    "Mean jerk norm",
]

# Decimal places per KPI for the formatted "mean +/- sd" table cells.
KPI_DECIMALS = {
    "Episode return": 1,
    "Episode length (steps)": 1,
    "Ego collisions / km": 3,
    "Minimum h": 3,
    "QP failure rate": 3,
    "Abs speed error (m/s)": 3,
    "Mean lateral tracking error (m)": 3,
    "Intervention rate": 3,
    "Correction norm": 3,
    "Mean jerk norm": 3,
    "Distance-based completion rate": 3,
}


@dataclass
class LoadedRow:
    arm: str
    arm_code: str
    variant: str
    variant_label: str
    seed: int
    mode: str
    kpi: str
    mean: float
    sd: float
    n_blocks: int
    episodes_per_mode: int


def git_commit(project_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True,
            text=True, check=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def load_arm_seed(study_dir: Path, arm: str, seed: int) -> list[LoadedRow]:
    path = study_dir / arm / f"seed_{seed}" / "post_train_200ep_kpis.csv"
    if not path.is_file():
        print(f"  MISSING: {path}")
        return []
    frame = pd.read_csv(path)
    rows: list[LoadedRow] = []
    for _, r in frame.iterrows():
        rows.append(
            LoadedRow(
                arm=arm,
                arm_code=ALL_ARMS.get(arm, arm),
                variant=str(r["variant"]),
                variant_label=str(r["variant_label"]),
                seed=seed,
                mode=str(r["mode"]),
                kpi=str(r["KPI"]),
                mean=float(r["Mean"]),
                sd=float(r["SD"]),
                n_blocks=int(r["summary_blocks"]),
                episodes_per_mode=int(r["episodes_per_mode"]),
            )
        )
    return rows


def collect(study_dir: Path, arms: Sequence[str], seeds: Sequence[int]) -> pd.DataFrame:
    all_rows: list[LoadedRow] = []
    print(f"Reading {study_dir}")
    for arm in arms:
        for seed in seeds:
            all_rows.extend(load_arm_seed(study_dir, arm, seed))
    if not all_rows:
        raise SystemExit(f"No KPI files found under {study_dir} for arms {list(arms)}")
    return pd.DataFrame(r.__dict__ for r in all_rows)


def policy_label(arm: str, variant_label: str) -> str:
    code = ALL_ARMS.get(arm, arm)
    return f"{code} — {variant_label}"


def build_coverage_table(long_df: pd.DataFrame, expected_seeds: Sequence[int]) -> pd.DataFrame:
    grouped = (
        long_df.groupby(["arm", "variant", "variant_label", "mode"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            episodes_per_mode=("episodes_per_mode", "first"),
        )
    )
    grouped["mode_label"] = grouped["mode"].map(MODE_LABELS)
    grouped["seeds_x_episodes"] = grouped.apply(
        lambda r: f"{r['n_seeds']}/{len(expected_seeds)} seeds x {r['episodes_per_mode']} ep", axis=1
    )
    wide = grouped.pivot_table(
        index=["arm", "variant", "variant_label"],
        columns="mode_label",
        values="seeds_x_episodes",
        aggfunc="first",
    )
    wide = wide.reset_index()
    policy = [policy_label(a, l) for a, l in zip(wide["arm"], wide["variant_label"])]
    out = pd.DataFrame({"Policy": policy, "Variant": wide["variant"]})
    for mode_label in ("CBF-OFF", "CBF-ON"):
        out[mode_label] = wide[mode_label] if mode_label in wide else ""
    return out


def aggregate_across_seeds(long_df: pd.DataFrame, kpis: Sequence[str]) -> pd.DataFrame:
    subset = long_df[long_df["kpi"].isin(kpis)].copy()
    grouped = subset.groupby(["arm", "variant", "variant_label", "mode", "kpi"], as_index=False).agg(
        n_seeds=("mean", "count"),
        seed_mean=("mean", "mean"),
        seed_sd=("mean", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else float("nan")),
        seed_min=("mean", "min"),
        seed_max=("mean", "max"),
    )
    grouped["mode_label"] = grouped["mode"].map(MODE_LABELS)
    grouped["policy"] = [policy_label(a, l) for a, l in zip(grouped["arm"], grouped["variant_label"])]
    return grouped


def format_cell(mean: float, sd: float, decimals: int) -> str:
    if pd.isna(sd):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def build_formatted_kpi_table(agg_df: pd.DataFrame, kpis: Sequence[str]) -> pd.DataFrame:
    rows = []
    row_order = []
    for policy in agg_df["policy"].drop_duplicates():
        for mode in ("raw", "cbf"):
            row_order.append((policy, mode))
    for policy, mode in row_order:
        sub = agg_df[(agg_df["policy"] == policy) & (agg_df["mode"] == mode)]
        if sub.empty:
            continue
        row = {"Policy": policy, "CBF mode": MODE_LABELS[mode]}
        for kpi in kpis:
            match = sub[sub["kpi"] == kpi]
            if match.empty:
                row[kpi] = ""
                continue
            m = match.iloc[0]
            row[kpi] = format_cell(m["seed_mean"], m["seed_sd"], KPI_DECIMALS.get(kpi, 3))
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(frame: pd.DataFrame) -> str:
    """Render without the optional ``tabulate`` dependency."""
    columns = [str(c) for c in frame.columns]
    cells = [[("" if pd.isna(v) else str(v)) for v in row] for row in frame.itertuples(index=False)]
    widths = [len(c) for c in columns]
    for row in cells:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))
    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(values)) + " |"
    lines = [fmt_row(columns), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(fmt_row(row) for row in cells)
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR,
                         help="Root of the safety-matrix run (contains <arm>/seed_<n>/).")
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS,
                         help=f"Arms to include. Default excludes {EXCLUDED_BY_DEFAULT}.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--modes", nargs="+", choices=list(MODE_LABELS), default=list(MODE_LABELS),
                         help="Evaluation modes to include: raw (CBF-OFF), cbf (CBF-ON).")
    parser.add_argument("--out", type=Path,
                         default=Path("artifacts") / "reports" / "safety_matrix_1M",
                         help="Output directory for the CSV/Markdown tables.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    unknown = [a for a in args.arms if a not in ALL_ARMS]
    if unknown:
        raise SystemExit(f"Unknown arm(s) {unknown}; known arms: {list(ALL_ARMS)}")

    long_df = collect(args.study_dir, args.arms, args.seeds)
    long_df = long_df[long_df["mode"].isin(args.modes)]
    args.out.mkdir(parents=True, exist_ok=True)

    coverage = build_coverage_table(long_df, args.seeds)
    safety_long = aggregate_across_seeds(long_df, SAFETY_KPIS)
    task_long = aggregate_across_seeds(long_df, TASK_KPIS)
    safety_wide = build_formatted_kpi_table(safety_long, SAFETY_KPIS)
    task_wide = build_formatted_kpi_table(task_long, TASK_KPIS)

    coverage.to_csv(args.out / "table1_policy_coverage.csv", index=False)
    safety_long.to_csv(args.out / "table2_safety_cbf_kpis_long.csv", index=False)
    safety_wide.to_csv(args.out / "table2_safety_cbf_kpis.csv", index=False)
    task_long.to_csv(args.out / "table3_task_comfort_kpis_long.csv", index=False)
    task_wide.to_csv(args.out / "table3_task_comfort_kpis.csv", index=False)

    commit = git_commit(args.project_root)
    report = []
    report.append("# Safety-matrix 1M results tables")
    report.append("")
    report.append(f"- Source: `{args.study_dir}`")
    report.append(f"- Arms included: {', '.join(args.arms)}")
    report.append(f"- Arms excluded: {', '.join(a for a in ALL_ARMS if a not in args.arms)}")
    report.append(f"- Seeds: {', '.join(str(s) for s in args.seeds)} (n={len(args.seeds)})")
    report.append(f"- Modes: {', '.join(MODE_LABELS[m] for m in args.modes)}")
    report.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    report.append(f"- Git commit: {commit}")
    report.append(
        "- Aggregation: mean +/- SD across the per-seed pooled-block means "
        "(n = number of seeds), not a pool of all episodes across seeds."
    )
    report.append("")
    report.append("## Table 1 -- Evaluated policies")
    report.append("")
    report.append(to_markdown(coverage))
    report.append("")
    report.append("## Table 2 -- Safety / CBF KPIs")
    report.append("")
    report.append(to_markdown(safety_wide))
    report.append("")
    report.append("## Table 3 -- Task / comfort KPIs")
    report.append("")
    report.append(to_markdown(task_wide))
    report.append("")
    text = "\n".join(report)
    (args.out / "safety_matrix_results_tables.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWritten to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
