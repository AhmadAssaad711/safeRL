"""Does differentiating the CBF filter make the policy learn FASTER?

Every comparison so far has been of converged policies at 1M steps, and those
came out flat. But the credit-assignment argument is about learning, not about
the fixed point: correct attribution should get you to the same place sooner,
and two runs can converge identically at different rates. A final-KPI
comparison is blind to that by construction.

This reads the TensorBoard scalars already written by the lambda=1.0 d40 runs
(7 seeds x B2 and C2, no retraining) and measures steps-to-threshold, paired by
training seed. Training seeds are shared between the arms, so seed s pairs
against seed s: same network init and the same scenario stream, differing only
in whether the projection is differentiated.

Thresholds are ABSOLUTE and shared between arms rather than relative to each
run's own plateau, so a run is never rewarded for having a lower bar.

With 7 pairs the honest tests are a paired mean with its SEM and an exact sign
test -- 7/7 in one direction is p=0.016, 6/7 is p=0.125, anything less is
nothing. Reported as such rather than as a t-test on 7 points.
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


def load_scalar(path: str, tag: str) -> pd.DataFrame:
    acc = EventAccumulator(path, size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return pd.DataFrame(columns=["step", "value"])
    events = acc.Scalars(tag)
    return pd.DataFrame(
        {"step": [e.step for e in events], "value": [e.value for e in events]}
    )


def seed_of(path: str) -> int:
    match = re.search(r"s(\d{3})", path)
    if match:
        return int(match.group(1))
    match = re.search(r"seed_(\d+)", path)
    return int(match.group(1)) if match else -1


def collect(runs_root: Path, tag: str) -> dict[str, dict[int, pd.DataFrame]]:
    out: dict[str, dict[int, pd.DataFrame]] = {}
    for arm, folder in ARMS.items():
        out[arm] = {}
        for path in sorted(
            glob.glob(str(runs_root / folder / "**" / "events.out.tfevents*"), recursive=True)
        ):
            frame = load_scalar(path, tag)
            if not frame.empty:
                out[arm][seed_of(path)] = frame
    return out


def steps_to_threshold(
    frame: pd.DataFrame, threshold: float, *, window: int, rising: bool
) -> float:
    """First step where the smoothed curve crosses the threshold, else NaN."""

    smooth = frame["value"].rolling(window, min_periods=window).mean()
    hit = smooth >= threshold if rising else smooth <= threshold
    idx = np.flatnonzero(hit.to_numpy())
    return float(frame["step"].iloc[idx[0]]) if idx.size else float("nan")


def sign_test(diffs: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided sign test, ignoring exact ties."""

    pos = int(np.sum(diffs > 0))
    neg = int(np.sum(diffs < 0))
    n = pos + neg
    if n == 0:
        return 0, 0, 1.0
    k = max(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)
    return pos, neg, float(min(1.0, 2 * tail))


def analyse(
    curves: dict[str, dict[int, pd.DataFrame]],
    thresholds: Sequence[float],
    *,
    window: int,
    rising: bool,
    label: str,
) -> None:
    seeds = sorted(set(curves["B2"]) & set(curves["C2"]))
    print(f"\n=== {label} | paired seeds: {seeds} ===")
    print(
        "%-12s %12s %12s %12s %10s %14s"
        % ("threshold", "B2 steps", "C2 steps", "B2-C2", "n_pairs", "sign (pos/neg,p)")
    )
    for threshold in thresholds:
        b2 = np.array(
            [steps_to_threshold(curves["B2"][s], threshold, window=window, rising=rising) for s in seeds]
        )
        c2 = np.array(
            [steps_to_threshold(curves["C2"][s], threshold, window=window, rising=rising) for s in seeds]
        )
        ok = np.isfinite(b2) & np.isfinite(c2)
        if ok.sum() < 3:
            print("%-12.4g %12s %12s %12s %10d %14s" % (threshold, "-", "-", "-", int(ok.sum()), "too few"))
            continue
        diff = b2[ok] - c2[ok]
        pos, neg, p = sign_test(diff)
        sem = diff.std(ddof=1) / math.sqrt(len(diff)) if len(diff) > 1 else float("nan")
        print(
            "%-12.4g %12.0f %12.0f %+9.0f+-%-6.0f %10d %14s"
            % (threshold, b2[ok].mean(), c2[ok].mean(), diff.mean(), sem, int(ok.sum()), f"{pos}/{neg}, p={p:.3f}")
        )
    print("  positive B2-C2 => C2 reached the bar in FEWER steps (C2 faster)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=HOME_RUNS)
    parser.add_argument("--window", type=int, default=25)
    args = parser.parse_args(argv)

    reward = collect(args.runs_root, "rollout/ep_rew_mean")
    print("reward curves:", {a: sorted(v) for a, v in reward.items()})
    finals = {
        arm: np.array([frame["value"].tail(50).mean() for frame in seeds.values()])
        for arm, seeds in reward.items()
    }
    for arm, values in finals.items():
        print(f"  {arm} final ep_rew_mean: {values.mean():.2f} +/- {values.std(ddof=1)/math.sqrt(len(values)):.2f}")

    lo = min(f.min() for f in finals.values())
    ladder = [round(lo * frac, 1) for frac in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)]
    analyse(reward, ladder, window=args.window, rising=True, label="steps to reach ep_rew_mean")

    collisions = collect(args.runs_root, "rollout/collisions_per_km")
    if collisions["B2"] and collisions["C2"]:
        starts = np.concatenate(
            [f["value"].head(50).to_numpy() for s in collisions.values() for f in s.values()]
        )
        base = float(np.nanmedian(starts))
        ladder_c = [round(base * frac, 3) for frac in (0.9, 0.75, 0.6, 0.5, 0.4)]
        analyse(
            collisions, ladder_c, window=args.window, rising=False,
            label="steps to fall below training collisions/km",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
