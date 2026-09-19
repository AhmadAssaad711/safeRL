"""Complete results tables for every variant tested: RAW vs CBF-ON.

Two tables, all 11 KPIs (5 safety + 6 task), mean +/- SD over 7 seeds:

- Table 1 (RAW): no protection at all. A1/B1/B2 = mode "raw" (no external
  filter). C1/C2 = raw-actor-mean (internal projection bypassed) -- their
  literal "raw" mode still runs the internal projection, so it is not a fair
  "no protection" comparison; raw-actor-mean is.
- Table 2 (CBF-ON): full protection. A1/B1/B2 = mode "cbf" (external QP
  filter). C1/C2 = mode "cbf" (internal projection + external filter, their
  actual deployed configuration).

B2/C2 are shown at lambda=1.0 (the best version found); A1/B1/C1 only exist
at lambda=0.1 (no lambda=1.0 follow-up was run for them).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HOME = Path.home()
SAFETY_MATRIX_DIR = HOME / "AppData" / "Local" / "Temp" / "saferl_runs" / "safety_matrix_1M"
LAMBDA_DETACHED_DIR = HOME / "AppData" / "Local" / "Temp" / "saferl_runs" / "lambda_detached_actor_1p0"
LAMBDA_MEAN_DIR = HOME / "AppData" / "Local" / "Temp" / "saferl_runs" / "lambda_mean_1p0"
SEEDS = [307, 327, 347, 367, 387, 407, 427]

SAFETY_KPIS = [
    "Ego collisions / km", "Minimum h", "QP failure rate",
    "Intervention rate", "Correction norm",
]
TASK_KPIS = [
    "Episode return", "Episode length (steps)", "Distance-based completion rate",
    "Abs speed error (m/s)", "Mean lateral tracking error (m)", "Mean jerk norm",
]
ALL_KPIS = SAFETY_KPIS + TASK_KPIS
KPI_DECIMALS = {
    "Episode return": 1, "Episode length (steps)": 1, "Ego collisions / km": 3,
    "Minimum h": 3, "QP failure rate": 3, "Abs speed error (m/s)": 3,
    "Mean lateral tracking error (m)": 3, "Intervention rate": 3,
    "Correction norm": 3, "Mean jerk norm": 3, "Distance-based completion rate": 3,
}

POLICY_LABELS = ["A1", "B1", "B2 (lambda=1.0)", "C1", "C2 (lambda=1.0)"]


def deployed_path(seed_dir: Path) -> Path:
    return seed_dir / "post_train_200ep_kpis.csv"


def raw_actor_mean_path(seed_dir: Path, seed: int, variant: str) -> Path:
    return seed_dir / "raw_actor_ablation" / variant / f"seed_{seed}" / "kpi.csv"


def load_mode(seed_dir_fn, seeds, mode: str, policy_label: str) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        path = deployed_path(seed_dir_fn(seed))
        if not path.is_file():
            print(f"  MISSING: {path}")
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["mode"] == mode].copy()
        frame["seed"] = seed
        frame["policy"] = policy_label
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_raw_actor_mean(seed_dir_fn, seeds, variant: str, policy_label: str) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        path = raw_actor_mean_path(seed_dir_fn(seed), seed, variant)
        if not path.is_file():
            print(f"  MISSING: {path}")
            continue
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frame["policy"] = policy_label
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def aggregate(long_df: pd.DataFrame, kpis: list[str], policy_order: list[str]) -> pd.DataFrame:
    subset = long_df[long_df["KPI"].isin(kpis)].copy()
    grouped = subset.groupby(["policy", "KPI"], as_index=False).agg(
        n_seeds=("Mean", "count"), seed_mean=("Mean", "mean"),
        seed_sd=("Mean", lambda s: float(s.std(ddof=1)) if len(s) > 1 else float("nan")),
    )
    rows = []
    for policy in policy_order:
        row = {"Policy": policy}
        for kpi in kpis:
            match = grouped[(grouped["policy"] == policy) & (grouped["KPI"] == kpi)]
            if match.empty:
                row[kpi] = ""
                continue
            m = match.iloc[0]
            dec = KPI_DECIMALS.get(kpi, 3)
            sd = m["seed_sd"]
            row[kpi] = f"{m['seed_mean']:.{dec}f} +/- {sd:.{dec}f}" if pd.notna(sd) else f"{m['seed_mean']:.{dec}f}"
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(c) for c in frame.columns]
    cells = [[("" if pd.isna(v) else str(v)) for v in row] for row in frame.itertuples(index=False)]
    widths = [len(c) for c in columns]
    for row in cells:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))
    def fmt(values):
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(values)) + " |"
    lines = [fmt(columns), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(fmt(row) for row in cells)
    return "\n".join(lines)


def main() -> int:
    out_dir = Path("artifacts") / "reports" / "lambda1p0_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    a1_dir = lambda s: SAFETY_MATRIX_DIR / "A1_nominal" / f"seed_{s}"
    b1_dir = lambda s: SAFETY_MATRIX_DIR / "B1_state_cost_raw" / f"seed_{s}"
    b2_dir = lambda s: LAMBDA_DETACHED_DIR / f"ldac1p0_s{s}"
    c1_dir = lambda s: SAFETY_MATRIX_DIR / "C1_cbf_diff_reward_only" / f"seed_{s}"
    c2_dir = lambda s: LAMBDA_MEAN_DIR / f"lam1p0_s{s}"

    # --- RAW table ---
    raw_frames = [
        load_mode(a1_dir, SEEDS, "raw", "A1"),
        load_mode(b1_dir, SEEDS, "raw", "B1"),
        load_mode(b2_dir, SEEDS, "raw", "B2 (lambda=1.0)"),
        load_raw_actor_mean(c1_dir, SEEDS, "ppo_cbf_diff_reward_only", "C1"),
        load_raw_actor_mean(c2_dir, SEEDS, "ppo_cbf_projected", "C2 (lambda=1.0)"),
    ]
    raw_long = pd.concat(raw_frames, ignore_index=True)
    raw_table = aggregate(raw_long, ALL_KPIS, POLICY_LABELS)

    # --- CBF-ON table ---
    cbf_frames = [
        load_mode(a1_dir, SEEDS, "cbf", "A1"),
        load_mode(b1_dir, SEEDS, "cbf", "B1"),
        load_mode(b2_dir, SEEDS, "cbf", "B2 (lambda=1.0)"),
        load_mode(c1_dir, SEEDS, "cbf", "C1"),
        load_mode(c2_dir, SEEDS, "cbf", "C2 (lambda=1.0)"),
    ]
    cbf_long = pd.concat(cbf_frames, ignore_index=True)
    cbf_table = aggregate(cbf_long, ALL_KPIS, POLICY_LABELS)

    raw_table.to_csv(out_dir / "table_raw_all_variants.csv", index=False)
    cbf_table.to_csv(out_dir / "table_cbf_on_all_variants.csv", index=False)

    report = []
    report.append("# Complete variant comparison: RAW vs CBF-ON")
    report.append("")
    report.append("- RAW: A1/B1/B2 = no external filter (mode=raw); C1/C2 = raw-actor-mean (internal projection bypassed)")
    report.append("- CBF-ON: A1/B1/B2 = external QP filter (mode=cbf); C1/C2 = deployed (internal projection + external filter)")
    report.append("- B2/C2 shown at lambda=1.0 (best found); A1/B1/C1 only exist at lambda=0.1")
    report.append("- Seeds: 307,327,347,367,387,407,427 (n=7); mean +/- SD across seed means")
    report.append("")
    report.append("## Table 1 -- RAW (no protection)")
    report.append("")
    report.append(to_markdown(raw_table))
    report.append("")
    report.append("## Table 2 -- CBF-ON (full protection)")
    report.append("")
    report.append(to_markdown(cbf_table))
    text = "\n".join(report)
    (out_dir / "final_variant_comparison.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
