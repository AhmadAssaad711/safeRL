"""Build evaluation results tables for the lambda=1.0 follow-up studies.

Reports the deployed and (for C2) raw-actor-mean evaluation KPIs of:

- B2 (ppo_cbf_nd_reward_actor), lambda_detached_actor_1p0 (lambda=1.0),
  7 seeds, vehicles_count=40.
- C2 (ppo_cbf_projected), lambda_mean_1p0 (lambda=1.0), 7 seeds,
  vehicles_count=40, under both the deployed evaluation (mode=raw/cbf,
  internal projection still active under "CBF-OFF") and the raw-actor-mean
  ablation (bypasses the internal projection layer entirely -- the only
  evaluation comparable to B2's CBF-OFF mode).

No lambda=0.1 data is included.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

HOME = Path.home()
SAFETY_MATRIX_DIR = HOME / "AppData" / "Local" / "Temp" / "saferl_runs" / "safety_matrix_1M"
LAMBDA_MEAN_DIR = HOME / "AppData" / "Local" / "Temp" / "saferl_runs" / "lambda_mean_1p0"
LAMBDA_DETACHED_DIR = HOME / "AppData" / "Local" / "Temp" / "saferl_runs" / "lambda_detached_actor_1p0"

SEEDS = [307, 327, 347, 367, 387, 407, 427]

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
ALL_KPIS = SAFETY_KPIS + TASK_KPIS
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
MODE_LABELS = {"raw": "CBF-OFF", "cbf": "CBF-ON", "raw_actor_mean": "Raw-actor-mean"}


@dataclass
class RunSpec:
    policy: str  # display label, e.g. "B2 (lambda=1.0)"
    path_for_seed: Callable[[int], Path]
    modes: Sequence[str]  # which "mode" values in the CSV to keep ("raw","cbf") or override
    mode_relabel: str | None = None  # if set, force this mode label regardless of CSV content


def deployed_csv(seed: int, seed_dir: Path) -> Path:
    return seed_dir / "post_train_200ep_kpis.csv"


def load_deployed(spec_seed_dir: Callable[[int], Path], seeds: Sequence[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        path = deployed_csv(seed, spec_seed_dir(seed))
        if not path.is_file():
            print(f"  MISSING: {path}")
            continue
        frame = pd.read_csv(path)
        frame["seed"] = seed
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_raw_actor_ablation(spec_seed_dir: Callable[[int], Path], seeds: Sequence[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        path = spec_seed_dir(seed) / "raw_actor_ablation" / "ppo_cbf_projected" / f"seed_{seed}" / "kpi.csv"
        if not path.is_file():
            print(f"  MISSING: {path}")
            continue
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frame["mode"] = "raw_actor_mean"
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate(long_df: pd.DataFrame, kpis: Sequence[str]) -> pd.DataFrame:
    subset = long_df[long_df["KPI"].isin(kpis)].copy()
    grouped = subset.groupby(["policy", "mode", "KPI"], as_index=False).agg(
        n_seeds=("Mean", "count"),
        seed_mean=("Mean", "mean"),
        seed_sd=("Mean", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else float("nan")),
    )
    return grouped


def format_cell(mean: float, sd: float, decimals: int) -> str:
    if pd.isna(sd):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} +/- {sd:.{decimals}f}"


def build_wide(agg_df: pd.DataFrame, kpis: Sequence[str], policy_order: Sequence[str], mode_order: Sequence[str]) -> pd.DataFrame:
    rows = []
    for policy in policy_order:
        for mode in mode_order:
            sub = agg_df[(agg_df["policy"] == policy) & (agg_df["mode"] == mode)]
            if sub.empty:
                continue
            row = {"Policy": policy, "Eval mode": MODE_LABELS.get(mode, mode)}
            for kpi in kpis:
                match = sub[sub["KPI"] == kpi]
                if match.empty:
                    row[kpi] = ""
                    continue
                m = match.iloc[0]
                row[kpi] = format_cell(m["seed_mean"], m["seed_sd"], KPI_DECIMALS.get(kpi, 3))
            rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(frame: pd.DataFrame) -> str:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "reports" / "lambda1p0_comparison")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, pd.DataFrame]] = []

    # B2 new (lambda=1.0, detached actor loss), deployed evaluation only.
    b2_new = load_deployed(lambda s: LAMBDA_DETACHED_DIR / f"ldac1p0_s{s}", SEEDS)
    if not b2_new.empty:
        b2_new["policy"] = "B2 (lambda=1.0, new)"
        runs.append(("B2 (lambda=1.0, new)", b2_new))

    # C2 new (lambda=1.0, mean actor loss), deployed evaluation (internal projection still active under CBF-OFF).
    c2_new_dep = load_deployed(lambda s: LAMBDA_MEAN_DIR / f"lam1p0_s{s}", SEEDS)
    if not c2_new_dep.empty:
        c2_new_dep["policy"] = "C2 (lambda=1.0, new)"
        runs.append(("C2 (lambda=1.0, new)", c2_new_dep))

    # C2 raw-actor-mean ablation (true unprotected action; comparable to A1/B1/B2 CBF-OFF).
    c2_new_raw = load_raw_actor_ablation(lambda s: LAMBDA_MEAN_DIR / f"lam1p0_s{s}", SEEDS)
    if not c2_new_raw.empty:
        c2_new_raw["policy"] = "C2 (lambda=1.0, new)"
        runs.append(("C2 raw-actor-mean (lambda=1.0)", c2_new_raw))

    long_df = pd.concat([d for _, d in runs], ignore_index=True)

    deployed_policy_order = [
        "B2 (lambda=1.0, new)",
        "C2 (lambda=1.0, new)",
    ]
    deployed_df = long_df[long_df["mode"].isin(["raw", "cbf"])]
    deployed_agg = aggregate(deployed_df, ALL_KPIS)
    deployed_wide = build_wide(deployed_agg, ALL_KPIS, deployed_policy_order, ["raw", "cbf"])

    raw_actor_policy_order = ["C2 (lambda=1.0, new)"]
    raw_actor_df = long_df[long_df["mode"] == "raw_actor_mean"]
    raw_actor_agg = aggregate(raw_actor_df, ALL_KPIS)
    raw_actor_wide = build_wide(raw_actor_agg, ALL_KPIS, raw_actor_policy_order, ["raw_actor_mean"])

    deployed_wide.to_csv(args.out / "table1_deployed_kpis.csv", index=False)
    raw_actor_wide.to_csv(args.out / "table2_c2_raw_actor_mean_kpis.csv", index=False)

    report = []
    report.append("# Lambda=1.0 follow-up: B2 / C2 evaluation results")
    report.append("")
    report.append(f"- B2 (lambda=1.0, detached actor loss): `{LAMBDA_DETACHED_DIR}`")
    report.append(f"- C2 (lambda=1.0, mean actor loss): `{LAMBDA_MEAN_DIR}`")
    report.append("- vehicles_count=40 for both studies. No lambda=0.1 data included.")
    report.append(f"- Seeds: {', '.join(str(s) for s in SEEDS)} (n=7 per policy)")
    report.append(
        "- Aggregation: mean +/- SD across the 7 per-seed pooled-block means "
        "(each seed's mean is itself pooled over 10 blocks of 20 episodes)."
    )
    report.append(
        "- Caveat: for C2, the deployed 'CBF-OFF' row still executes the "
        "internal differentiable projection (mu_safe) -- it is NOT the raw "
        "network output, so it is not directly comparable to B2's CBF-OFF. "
        "Use Table 2 (raw-actor-mean) for that comparison."
    )
    report.append("")
    report.append("## Table 1 -- Deployed KPIs (safety + task, all policies)")
    report.append("")
    report.append(to_markdown(deployed_wide))
    report.append("")
    report.append("## Table 2 -- C2 raw-actor-mean KPIs (projection bypassed)")
    report.append("")
    report.append(to_markdown(raw_actor_wide))
    report.append("")
    text = "\n".join(report)
    (args.out / "lambda1p0_comparison_tables.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWritten to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
