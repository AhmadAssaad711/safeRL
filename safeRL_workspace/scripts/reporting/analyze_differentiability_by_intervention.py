"""Does differentiating the CBF filter help more on scenarios that demand more
filter intervention?

The credit-assignment argument says B2 and C2 differ only through the
correction the filter applies: when the filter is inactive, mu_safe = mu_raw,
the projection Jacobian is the identity, and the two architectures produce the
same gradient. So any advantage must scale with

    E[ ||mu_safe - mu_raw|| * 1{QP feasible} ]

which rises with filter activity and collapses where the QP is infeasible (the
projection is undefined there and the gradient is detached to zero). This
script measures that as a slope rather than a two-arm mean.

Design
------
Both arms are evaluated on the SAME 200 episode seeds, so the comparison is
paired scenario-by-scenario, which removes the scenario variance that made the
unpaired d60 comparison hopeless. Each arm's value at a seed is the mean over
its 7 training seeds.

The stratifier has to be architecture-neutral. C2's own intervention rate is
identically zero -- its internal projection already emits safe actions, so the
external filter never fires -- which makes it useless as a difficulty measure
and would put every C2 episode in one bin. A1 (nominal) is used instead: it
trains with no CBF at all, so how hard its filter has to work on a scenario is
a property of the scenario, not of the B2/C2 contrast being tested.

Reported per difficulty stratum, and as a regression slope:

    D(s) = outcome_B2(s) - outcome_C2(s)

so for collisions a POSITIVE D means C2 is better. A positive slope of D
against difficulty is the claim "differentiability earns its keep on hard
jobs". A flat slope falsifies it at this density.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

HOME_RUNS = Path.home() / "AppData" / "Local" / "Temp" / "saferl_runs"

ARMS = {
    "B2": "lambda_detached_actor_1p0",
    "C2": "lambda_mean_1p0",
}
NEUTRAL_ARM_GLOB = "safety_matrix_1M/A1_nominal"


def load_episodes(root: Path, pattern: str) -> pd.DataFrame:
    files = glob.glob(str(root / pattern / "**" / "pe" / "e.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"no per-episode files under {root / pattern}")
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return frame[frame["mode"] == "cbf"].copy()


def per_seed_mean(
    frame: pd.DataFrame, columns: Sequence[str], keys: Sequence[str]
) -> pd.DataFrame:
    """Collapse to one row per key.

    With keys=("episode_seed",) this averages over training seeds, which is the
    right pairing for a scenario-level comparison but compresses the
    stratifier: A1's per-episode intervention spans 0.31-0.98 raw and only
    0.50-0.68 once averaged. With keys=("training_seed","episode_seed") no
    averaging happens, so the full difficulty contrast survives; the arms are
    then paired on the training seed too, which shares network init and the
    training scenario stream between architectures.
    """

    return frame.groupby(list(keys), as_index=False)[list(columns)].mean()


def bootstrap_slope(
    x: np.ndarray, y: np.ndarray, *, draws: int = 10000, seed: int = 0
) -> tuple[float, float, float, float]:
    """Slope with a paired bootstrap CI and a two-sided sign-flip p-value."""

    rng = np.random.default_rng(seed)
    slope = float(np.polyfit(x, y, 1)[0])
    n = len(x)
    boot = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        boot[i] = np.polyfit(x[idx], y[idx], 1)[0]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Null: no association. Permute the stratifier.
    null = np.empty(draws)
    for i in range(draws):
        null[i] = np.polyfit(rng.permutation(x), y, 1)[0]
    p = float(np.mean(np.abs(null) >= abs(slope)))
    return slope, float(lo), float(hi), p


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=HOME_RUNS)
    parser.add_argument("--strata", type=int, default=5)
    parser.add_argument(
        "--pair-level", choices=("scenario", "run"), default="scenario",
        help=(
            "scenario: average over training seeds, one row per episode seed. "
            "run: pair on (training_seed, episode_seed), keeping the full "
            "difficulty contrast that averaging otherwise smooths away."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    keys = (
        ["episode_seed"]
        if args.pair_level == "scenario"
        else ["training_seed", "episode_seed"]
    )

    metrics = [
        "distinct_ego_collision_events",
        "total_distance_m",
        "mean_jerk_norm",
        "episode_length_steps",
        "qp_failure_rate",
        "event_intervention_rate",
    ]

    arms = {
        name: per_seed_mean(load_episodes(args.runs_root, pattern), metrics, keys)
        for name, pattern in ARMS.items()
    }
    neutral = per_seed_mean(
        load_episodes(args.runs_root, NEUTRAL_ARM_GLOB), metrics, keys
    )[list(keys) + ["event_intervention_rate", "qp_failure_rate"]]
    neutral = neutral.rename(
        columns={
            "event_intervention_rate": "difficulty_intervention",
            "qp_failure_rate": "difficulty_qp_failure",
        }
    )

    merged = arms["B2"].merge(arms["C2"], on=keys, suffixes=("_B2", "_C2"))
    merged = merged.merge(neutral, on=keys)
    print(f"paired scenarios: {len(merged)}")

    # Positive => C2 better (fewer collisions, shorter distance is worse so flip)
    merged["d_collisions"] = (
        merged["distinct_ego_collision_events_B2"]
        - merged["distinct_ego_collision_events_C2"]
    )
    merged["d_distance"] = merged["total_distance_m_C2"] - merged["total_distance_m_B2"]
    merged["d_jerk"] = merged["mean_jerk_norm_B2"] - merged["mean_jerk_norm_C2"]

    print("\n=== overall paired means (positive = C2 better) ===")
    for column, label in [
        ("d_collisions", "collisions/episode  B2 - C2"),
        ("d_distance", "distance (m)        C2 - B2"),
        ("d_jerk", "jerk norm           B2 - C2"),
    ]:
        values = merged[column].to_numpy(dtype=float)
        sem = values.std(ddof=1) / np.sqrt(len(values))
        print(f"  {label}: {values.mean():+.4f} +/- {sem:.4f}")

    for stratifier, nice in [
        ("difficulty_intervention", "A1 intervention rate (filter demand)"),
        ("difficulty_qp_failure", "A1 QP failure rate (infeasibility)"),
    ]:
        print(f"\n=== stratified by {nice} ===")
        merged["_bin"] = pd.qcut(
            merged[stratifier], args.strata, labels=False, duplicates="drop"
        )
        print(
            "%-6s %9s %9s %9s %9s %11s"
            % ("bin", "diff_lo", "diff_hi", "B2 coll", "C2 coll", "B2-C2")
        )
        for b, group in merged.groupby("_bin"):
            print(
                "%-6d %9.3f %9.3f %9.4f %9.4f %+11.4f"
                % (
                    int(b),
                    group[stratifier].min(),
                    group[stratifier].max(),
                    group["distinct_ego_collision_events_B2"].mean(),
                    group["distinct_ego_collision_events_C2"].mean(),
                    group["d_collisions"].mean(),
                )
            )
        x = merged[stratifier].to_numpy(dtype=float)
        for column, label in [
            ("d_collisions", "collisions"),
            ("d_distance", "distance"),
            ("d_jerk", "jerk"),
        ]:
            slope, lo, hi, p = bootstrap_slope(
                x, merged[column].to_numpy(dtype=float)
            )
            verdict = "significant" if (lo > 0) or (hi < 0) else "n.s."
            print(
                f"  slope d_{label:<10} = {slope:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
                f"  p={p:.3f}  {verdict}"
            )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
