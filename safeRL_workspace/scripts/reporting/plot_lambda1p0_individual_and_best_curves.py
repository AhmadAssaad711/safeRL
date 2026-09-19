"""Standalone B2/C2 (lambda=1.0) reward curves, plus a 5-arm "current best"
comparison: A1/B1/C1 at their only trained point (lambda=0.1) alongside
B2/C2 at lambda=1.0 (the best version of each found so far).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.reporting.plot_lambda1p0_reward_curves import (
    LAMBDA_DETACHED_DIR,
    LAMBDA_MEAN_DIR,
    SAFETY_MATRIX_DIR,
    SEEDS,
    load_group,
    resample,
)

BEST_COLORS = {
    "A1 (nominal)": "#4472C4",
    "B1 (state cost)": "#ED7D31",
    "B2 (lambda=1.0)": "#70AD47",
    "C1 (diff reward only)": "#9E480E",
    "C2 (lambda=1.0)": "#997300",
}


def plot_individual(label: str, curves: dict, color: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for seed, curve in sorted(curves.items()):
        if curve.empty:
            continue
        ax.plot(curve["step"], curve["value"], color=color, alpha=0.28, linewidth=1.1)
    grid, resampled = resample(curves)
    if resampled:
        stacked = np.vstack(list(resampled.values()))
        mean = stacked.mean(axis=0)
        ax.plot(grid, mean, color=color, linewidth=2.6, label=f"mean of {len(resampled)} seeds")
    ax.set_title(f"{label}: training reward (rollout/ep_rew_mean)", fontweight="semibold")
    ax.set_xlabel("Training timestep")
    ax.set_ylabel("Episode return (rollout mean)")
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    out_dir = Path("artifacts") / "reports" / "lambda1p0_comparison" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    b2_new = load_group(
        lambda s: LAMBDA_DETACHED_DIR / f"ldac1p0_s{s}" / "ppo_cbf_nd_reward_actor" / f"seed_{s}", SEEDS
    )
    c2_new = load_group(
        lambda s: LAMBDA_MEAN_DIR / f"lam1p0_s{s}" / "ppo_cbf_projected" / f"seed_{s}", SEEDS
    )
    plot_individual("B2 (lambda=1.0, new)", b2_new, "#375623", out_dir / "B2_lambda1p0_reward_curve.png")
    plot_individual("C2 (lambda=1.0, new)", c2_new, "#7F6000", out_dir / "C2_lambda1p0_reward_curve.png")

    a1 = load_group(
        lambda s: SAFETY_MATRIX_DIR / "A1_nominal" / f"seed_{s}" / "ppo_nominal" / f"seed_{s}", SEEDS
    )
    b1 = load_group(
        lambda s: SAFETY_MATRIX_DIR / "B1_state_cost_raw" / f"seed_{s}" / "ppo_hocbf_state_cost_raw" / f"seed_{s}", SEEDS
    )
    c1 = load_group(
        lambda s: SAFETY_MATRIX_DIR / "C1_cbf_diff_reward_only" / f"seed_{s}" / "ppo_cbf_diff_reward_only" / f"seed_{s}", SEEDS
    )

    best = {
        "A1 (nominal)": a1,
        "B1 (state cost)": b1,
        "B2 (lambda=1.0)": b2_new,
        "C1 (diff reward only)": c1,
        "C2 (lambda=1.0)": c2_new,
    }
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for label, curves in best.items():
        grid, resampled = resample(curves)
        if not resampled:
            continue
        stacked = np.vstack(list(resampled.values()))
        mean = stacked.mean(axis=0)
        sd = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)
        color = BEST_COLORS[label]
        ax.plot(grid, mean, color=color, linewidth=2.4, label=f"{label} (n={stacked.shape[0]})")
        ax.fill_between(grid, mean - sd, mean + sd, color=color, alpha=0.15, linewidth=0)
    ax.set_title(
        "Current-best training reward by arm\n"
        "(A1/B1/C1 only trained at lambda=0.1; B2/C2 shown at lambda=1.0, the better of the two)",
        fontweight="semibold",
    )
    ax.set_xlabel("Training timestep")
    ax.set_ylabel("Episode return (rollout mean)")
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    out_path = out_dir / "current_best_arms_reward_comparison.png"
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
