"""Lower-variance sample-efficiency test, plus a mediation check on the
gradient diagnostics that are already logged.

Two additions over the steps-to-threshold analysis:

1. Area under the learning curve. Steps-to-threshold reads one crossing off a
   noisy curve and discards the other ~999 points; AULC integrates the whole
   trajectory, so it estimates the same "learned faster" quantity with far less
   variance. Reported over the full budget and over the first half, where the
   measured gap actually lived (the 2400 reward bar was crossed at 205k-305k).

2. Mediation. If correct credit assignment is the mechanism, it should show up
   in the gradient diagnostics themselves -- actor_g_ppo_g_cbf_cosine and
   actor_g_cbf_to_g_ppo_ratio -- not only in the outcome. Both arms log them.
   A difference in outcome WITH a matching difference in gradient alignment,
   and a correlation between the two across runs, is a mechanism; a difference
   in outcome alone is a coincidence.

Everything reads TensorBoard scalars already on disk. No training.
"""

from __future__ import annotations

import argparse
import glob
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HOME_RUNS = Path.home() / "AppData" / "Local" / "Temp" / "saferl_runs"
ARMS = {"B2": "lambda_detached_actor_1p0", "C2": "lambda_mean_1p0"}

DIAGNOSTICS = [
    "train/actor_g_ppo_g_cbf_cosine",
    "train/actor_g_cbf_to_g_ppo_ratio",
    "train/actor_g_cbf_norm",
    "train/actor_g_ppo_norm",
    "train/cbf_mean_infeasible_rate",
    "train/approx_kl",
    "train/std",
]


def seed_of(path: str) -> int:
    match = re.search(r"s(\d{3})", path)
    return int(match.group(1)) if match else -1


def read_run(path: str, tags: Sequence[str]) -> dict[str, pd.DataFrame]:
    acc = EventAccumulator(path, size_guidance={"scalars": 0})
    acc.Reload()
    available = set(acc.Tags()["scalars"])
    out = {}
    for tag in tags:
        if tag in available:
            events = acc.Scalars(tag)
            out[tag] = pd.DataFrame(
                {"step": [e.step for e in events], "value": [e.value for e in events]}
            )
    return out


def aulc(frame: pd.DataFrame, *, max_step: float | None = None) -> float:
    """Mean of the curve up to max_step -- the area normalised by its width."""

    data = frame if max_step is None else frame[frame["step"] <= max_step]
    if len(data) < 2:
        return float("nan")
    return float(
        np.trapz(data["value"].to_numpy(), data["step"].to_numpy())
        / (data["step"].iloc[-1] - data["step"].iloc[0])
    )


def paired_report(label: str, b2: np.ndarray, c2: np.ndarray, seeds: Sequence[int]) -> np.ndarray:
    from scipy.stats import ttest_rel, wilcoxon

    diff = c2 - b2
    sem = diff.std(ddof=1) / math.sqrt(len(diff))
    wins = int(np.sum(diff > 0))
    try:
        w_p = wilcoxon(diff).pvalue
    except ValueError:
        w_p = float("nan")
    t_p = ttest_rel(c2, b2).pvalue
    print(
        f"  {label:<34} B2 {b2.mean():9.1f}  C2 {c2.mean():9.1f}  "
        f"C2-B2 {diff.mean():+9.1f} +/- {sem:7.1f}  {wins}/{len(diff)} seeds  "
        f"wilcoxon p={w_p:.4f}  t p={t_p:.4f}"
    )
    return diff


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=HOME_RUNS)
    parser.add_argument("--half-step", type=float, default=500_000)
    args = parser.parse_args(argv)

    runs: dict[str, dict[int, dict[str, pd.DataFrame]]] = {}
    for arm, folder in ARMS.items():
        runs[arm] = {}
        for path in sorted(
            glob.glob(str(args.runs_root / folder / "**" / "events.out.tfevents*"), recursive=True)
        ):
            runs[arm][seed_of(path)] = read_run(path, ["rollout/ep_rew_mean"] + DIAGNOSTICS)

    seeds = sorted(set(runs["B2"]) & set(runs["C2"]))
    print(f"paired seeds: {seeds}")

    print("\n=== area under learning curve (rollout/ep_rew_mean) ===")
    results = {}
    for label, cap in [("AULC full 1M", None), (f"AULC first {args.half_step/1000:.0f}k", args.half_step)]:
        b2 = np.array([aulc(runs["B2"][s]["rollout/ep_rew_mean"], max_step=cap) for s in seeds])
        c2 = np.array([aulc(runs["C2"][s]["rollout/ep_rew_mean"], max_step=cap) for s in seeds])
        results[label] = paired_report(label, b2, c2, seeds)
        if cap is None:
            results["_aulc_b2"], results["_aulc_c2"] = b2, c2

    print("\n=== gradient diagnostics (mean over training) ===")
    diag_means: dict[str, dict[str, np.ndarray]] = {}
    for tag in DIAGNOSTICS:
        present = all(tag in runs[a][s] for a in ARMS for s in seeds)
        if not present:
            print(f"  {tag:<34} not logged in both arms -- skipped")
            continue
        b2 = np.array([runs["B2"][s][tag]["value"].mean() for s in seeds])
        c2 = np.array([runs["C2"][s][tag]["value"].mean() for s in seeds])
        diag_means[tag] = {"B2": b2, "C2": c2}
        paired_report(tag.replace("train/", ""), b2, c2, seeds)

    print("\n=== mediation: does gradient alignment predict learning speed? ===")
    print("  (correlation across all 14 runs between a diagnostic and that run's AULC)")
    all_aulc = np.concatenate([results["_aulc_b2"], results["_aulc_c2"]])
    for tag, values in diag_means.items():
        x = np.concatenate([values["B2"], values["C2"]])
        if np.allclose(x.std(), 0.0):
            continue
        r = float(np.corrcoef(x, all_aulc)[0, 1])
        # within-arm, to separate "arm differs" from "diagnostic predicts"
        r_b2 = float(np.corrcoef(values["B2"], results["_aulc_b2"])[0, 1])
        r_c2 = float(np.corrcoef(values["C2"], results["_aulc_c2"])[0, 1])
        print(
            f"  {tag.replace('train/',''):<34} r_all={r:+.3f}   "
            f"r_within_B2={r_b2:+.3f}   r_within_C2={r_c2:+.3f}"
        )
    print(
        "\n  A large r_all with near-zero within-arm r means the diagnostic only\n"
        "  tracks which arm a run belongs to, which is not mediation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
