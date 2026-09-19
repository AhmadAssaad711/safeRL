"""Training reward curves: B2/C2 at lambda=0.1 (baseline) vs lambda=1.0 (new).

Reads ``rollout/ep_rew_mean`` from each seed's TensorBoard event file (same
tag/mechanism as ``plot_safety_matrix_reward_curves.py``) for four run
groups, all 7 seeds each, and plots mean +/- SD per group plus one combined
comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HOME = Path.home()
SAFETY_MATRIX_DIR = HOME / "AppData" / "Local" / "Temp" / "saferl_runs" / "safety_matrix_1M"
LAMBDA_DETACHED_DIR = HOME / "AppData" / "Local" / "Temp" / "saferl_runs" / "lambda_detached_actor_1p0"
LAMBDA_MEAN_DIR = HOME / "AppData" / "Local" / "Temp" / "saferl_runs" / "lambda_mean_1p0"
SEEDS = [307, 327, 347, 367, 387, 407, 427]
REWARD_TAG = "rollout/ep_rew_mean"

GROUP_COLORS = {
    "B2 (lambda=0.1, baseline)": "#A9D18E",
    "B2 (lambda=1.0, new)": "#375623",
    "C2 (lambda=0.1, baseline)": "#FFD966",
    "C2 (lambda=1.0, new)": "#7F6000",
}


def find_event_file(seed_dir: Path) -> Path | None:
    matches = sorted(seed_dir.rglob("events.out.tfevents.*"))
    return matches[0] if matches else None


def load_curve(event_path: Path, tag: str) -> pd.DataFrame:
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return pd.DataFrame(columns=["step", "value"])
    scalars = accumulator.Scalars(tag)
    return pd.DataFrame({"step": [s.step for s in scalars], "value": [s.value for s in scalars]})


def resample(curves: dict[int, pd.DataFrame], n_points: int = 400) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    non_empty = {s: c for s, c in curves.items() if not c.empty}
    if not non_empty:
        return np.array([]), {}
    max_common_step = min(c["step"].max() for c in non_empty.values())
    grid = np.linspace(0, max_common_step, n_points)
    resampled = {
        seed: np.interp(grid, c["step"].to_numpy(), c["value"].to_numpy())
        for seed, c in non_empty.items()
    }
    return grid, resampled


def load_group(seed_dir_fn, seeds: list[int]) -> dict[int, pd.DataFrame]:
    curves: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        seed_dir = seed_dir_fn(seed)
        event_path = find_event_file(seed_dir)
        if event_path is None:
            print(f"  MISSING event file under {seed_dir}")
            continue
        curves[seed] = load_curve(event_path, REWARD_TAG)
    return curves


def main() -> int:
    out_dir = Path("artifacts") / "reports" / "lambda1p0_comparison" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = {
        "B2 (lambda=0.1, baseline)": load_group(
            lambda s: SAFETY_MATRIX_DIR / "B2_cbf_nd_reward_actor" / f"seed_{s}" / "ppo_cbf_nd_reward_actor" / f"seed_{s}",
            SEEDS,
        ),
        "B2 (lambda=1.0, new)": load_group(
            lambda s: LAMBDA_DETACHED_DIR / f"ldac1p0_s{s}" / "ppo_cbf_nd_reward_actor" / f"seed_{s}",
            SEEDS,
        ),
        "C2 (lambda=0.1, baseline)": load_group(
            lambda s: SAFETY_MATRIX_DIR / "C2_cbf_projected" / f"seed_{s}" / "ppo_cbf_projected" / f"seed_{s}",
            SEEDS,
        ),
        "C2 (lambda=1.0, new)": load_group(
            lambda s: LAMBDA_MEAN_DIR / f"lam1p0_s{s}" / "ppo_cbf_projected" / f"seed_{s}",
            SEEDS,
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, prefix, title in [(axes[0], "B2", "B2: lambda=0.1 vs lambda=1.0"), (axes[1], "C2", "C2: lambda=0.1 vs lambda=1.0")]:
        for label, curves in groups.items():
            if not label.startswith(prefix):
                continue
            grid, resampled = resample(curves)
            if not resampled:
                continue
            stacked = np.vstack(list(resampled.values()))
            mean = stacked.mean(axis=0)
            sd = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)
            color = GROUP_COLORS[label]
            ax.plot(grid, mean, color=color, linewidth=2.4, label=f"{label} (n={stacked.shape[0]})")
            ax.fill_between(grid, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
        ax.set_title(title, fontweight="semibold")
        ax.set_xlabel("Training timestep")
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    axes[0].set_ylabel("Episode return (rollout mean)")
    fig.suptitle("Training reward curves: baseline (lambda=0.1) vs new (lambda=1.0)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = out_dir / "lambda1p0_reward_curves.png"
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
