"""Plot training reward curves for the safety-matrix 1M study.

Reads the ``rollout/ep_rew_mean`` scalar from each arm/seed's TensorBoard
event file (written by ``scripts/ops/run_safety_matrix_study.py``) and
produces:

1. one PNG per arm, all seven seeds overlaid plus their mean; and
2. one combined PNG comparing the mean curve of every included arm.

Static images rather than a live TensorBoard server, since a server started
from an agent's shell is not guaranteed to be reachable from the user's own
browser (different network namespace).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_STUDY_DIR = Path.home() / "AppData" / "Local" / "Temp" / "saferl_runs" / "safety_matrix_1M"
DEFAULT_SEEDS = [307, 327, 347, 367, 387, 407, 427]
ALL_ARMS = {
    "A1_nominal": ("A1", "ppo_nominal"),
    "B1_state_cost_raw": ("B1", "ppo_hocbf_state_cost_raw"),
    "B2_cbf_nd_reward_actor": ("B2", "ppo_cbf_nd_reward_actor"),
    "C1_cbf_diff_reward_only": ("C1", "ppo_cbf_diff_reward_only"),
    "C2_cbf_projected": ("C2", "ppo_cbf_projected"),
}
DEFAULT_ARMS = ["A1_nominal", "B1_state_cost_raw", "B2_cbf_nd_reward_actor"]
REWARD_TAG = "rollout/ep_rew_mean"
ARM_COLORS = {
    "A1_nominal": "#4472C4",
    "B1_state_cost_raw": "#ED7D31",
    "B2_cbf_nd_reward_actor": "#70AD47",
    "C1_cbf_diff_reward_only": "#9E480E",
    "C2_cbf_projected": "#997300",
}


def find_event_file(study_dir: Path, arm: str, seed: int) -> Path | None:
    variant = ALL_ARMS[arm][1]
    pattern = f"{arm}/seed_{seed}/{variant}/seed_{seed}/tensorboard/*/events.out.tfevents.*"
    matches = sorted(study_dir.glob(pattern))
    if not matches:
        matches = sorted((study_dir / arm / f"seed_{seed}").rglob("events.out.tfevents.*"))
    return matches[0] if matches else None


def load_curve(event_path: Path, tag: str) -> pd.DataFrame:
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return pd.DataFrame(columns=["step", "value"])
    scalars = accumulator.Scalars(tag)
    return pd.DataFrame({"step": [s.step for s in scalars], "value": [s.value for s in scalars]})


def resample(curves: dict[int, pd.DataFrame], n_points: int = 400) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Interpolate each seed's curve onto a shared step grid so a mean/band is defined."""
    max_common_step = min(c["step"].max() for c in curves.values() if not c.empty)
    grid = np.linspace(0, max_common_step, n_points)
    resampled = {
        seed: np.interp(grid, c["step"].to_numpy(), c["value"].to_numpy())
        for seed, c in curves.items()
        if not c.empty
    }
    return grid, resampled


def plot_arm(arm: str, curves: dict[int, pd.DataFrame], out_path: Path) -> None:
    code, variant = ALL_ARMS[arm]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for seed, curve in sorted(curves.items()):
        if curve.empty:
            continue
        ax.plot(curve["step"], curve["value"], color=ARM_COLORS[arm], alpha=0.28, linewidth=1.1,
                label=f"seed {seed}" if False else None)
    grid, resampled = resample(curves)
    if resampled:
        stacked = np.vstack(list(resampled.values()))
        mean = stacked.mean(axis=0)
        ax.plot(grid, mean, color=ARM_COLORS[arm], linewidth=2.6, label=f"mean of {len(resampled)} seeds")
    ax.set_title(f"{code} — {variant}: training reward ({REWARD_TAG})", fontweight="semibold")
    ax.set_xlabel("Training timestep")
    ax.set_ylabel("Episode return (rollout mean)")
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_comparison(all_curves: dict[str, dict[int, pd.DataFrame]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for arm, curves in all_curves.items():
        code, variant = ALL_ARMS[arm]
        grid, resampled = resample(curves)
        if not resampled:
            continue
        stacked = np.vstack(list(resampled.values()))
        mean = stacked.mean(axis=0)
        sd = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)
        color = ARM_COLORS[arm]
        ax.plot(grid, mean, color=color, linewidth=2.4, label=f"{code} {variant}")
        ax.fill_between(grid, mean - sd, mean + sd, color=color, alpha=0.15, linewidth=0)
    ax.set_title(f"Safety-matrix 1M: training reward by arm (mean ± SD across seeds)", fontweight="semibold")
    ax.set_xlabel("Training timestep")
    ax.set_ylabel("Episode return (rollout mean)")
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--tag", default=REWARD_TAG)
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "reports" / "safety_matrix_1M" / "figures")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    unknown = [a for a in args.arms if a not in ALL_ARMS]
    if unknown:
        raise SystemExit(f"Unknown arm(s) {unknown}; known arms: {list(ALL_ARMS)}")
    args.out.mkdir(parents=True, exist_ok=True)

    all_curves: dict[str, dict[int, pd.DataFrame]] = {}
    for arm in args.arms:
        curves: dict[int, pd.DataFrame] = {}
        for seed in args.seeds:
            event_path = find_event_file(args.study_dir, arm, seed)
            if event_path is None:
                print(f"  MISSING event file: {arm} seed {seed}")
                continue
            curves[seed] = load_curve(event_path, args.tag)
        all_curves[arm] = curves
        out_path = args.out / f"{ALL_ARMS[arm][0]}_{arm}_reward_curve.png"
        plot_arm(arm, curves, out_path)
        print(f"wrote {out_path}")

    comparison_path = args.out / "arms_reward_comparison.png"
    plot_comparison(all_curves, comparison_path)
    print(f"wrote {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
