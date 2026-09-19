"""Training-diagnostic plots for B2/C2, lambda=0.1 (baseline) vs lambda=1.0
(new): training-rollout signals, PPO stability, and CBF-field signals.

Mirrors ``plot_safety_matrix_training_diagnostics.py``'s tag set and figure
layout, but reads from the lambda1p0 follow-up studies' directory layout
(``lambda_detached_actor_1p0``/``lambda_mean_1p0``, no <arm> folder) as well
as the original ``safety_matrix_1M`` baseline, and applies TensorBoard-style
EMA smoothing to every curve before plotting (raw per-step scalars are noisy
enough to obscure the trend otherwise).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

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
SMOOTHING_WEIGHT = 0.85  # TensorBoard default is 0.6; a bit heavier since these are noisier signals.

GROUP_COLORS = {
    "B2 (lambda=0.1)": "#A9D18E",
    "B2 (lambda=1.0)": "#375623",
    "C2 (lambda=0.1)": "#FFD966",
    "C2 (lambda=1.0)": "#7F6000",
}
GROUPS = {
    "B2 (lambda=0.1)": lambda s: SAFETY_MATRIX_DIR / "B2_cbf_nd_reward_actor" / f"seed_{s}" / "ppo_cbf_nd_reward_actor" / f"seed_{s}",
    "B2 (lambda=1.0)": lambda s: LAMBDA_DETACHED_DIR / f"ldac1p0_s{s}" / "ppo_cbf_nd_reward_actor" / f"seed_{s}",
    "C2 (lambda=0.1)": lambda s: SAFETY_MATRIX_DIR / "C2_cbf_projected" / f"seed_{s}" / "ppo_cbf_projected" / f"seed_{s}",
    "C2 (lambda=1.0)": lambda s: LAMBDA_MEAN_DIR / f"lam1p0_s{s}" / "ppo_cbf_projected" / f"seed_{s}",
}


def find_event_file(seed_dir: Path) -> Path | None:
    matches = sorted(seed_dir.rglob("events.out.tfevents.*"))
    return matches[0] if matches else None


def load_curves(event_path: Path, tags: Sequence[str]) -> dict[str, pd.DataFrame]:
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    out: dict[str, pd.DataFrame] = {}
    for tag in tags:
        if tag not in available:
            continue
        scalars = accumulator.Scalars(tag)
        out[tag] = pd.DataFrame({"step": [s.step for s in scalars], "value": [s.value for s in scalars]})
    return out


def ema_smooth(values: np.ndarray, weight: float = SMOOTHING_WEIGHT) -> np.ndarray:
    """TensorBoard-style exponential moving average with bias correction."""
    if values.size == 0:
        return values
    smoothed = np.empty_like(values, dtype=float)
    last = 0.0
    for i, v in enumerate(values):
        last = last * weight + (1 - weight) * v
        smoothed[i] = last / (1 - weight ** (i + 1))
    return smoothed


def load_all(tags: Sequence[str]) -> dict[str, dict[str, dict[int, pd.DataFrame]]]:
    """tag -> group -> seed -> DataFrame(step, value), pre-smoothed."""
    result: dict[str, dict[str, dict[int, pd.DataFrame]]] = {tag: {g: {} for g in GROUPS} for tag in tags}
    for group, seed_dir_fn in GROUPS.items():
        for seed in SEEDS:
            seed_dir = seed_dir_fn(seed)
            event_path = find_event_file(seed_dir)
            if event_path is None:
                print(f"  MISSING event file under {seed_dir}")
                continue
            curves = load_curves(event_path, tags)
            for tag, curve in curves.items():
                if not curve.empty:
                    curve = curve.copy()
                    curve["value"] = ema_smooth(curve["value"].to_numpy())
                result[tag][group][seed] = curve
    return result


def resample_mean(curves: dict[int, pd.DataFrame], n_points: int = 300) -> tuple[np.ndarray, np.ndarray] | None:
    usable = {seed: c for seed, c in curves.items() if not c.empty}
    if not usable:
        return None
    max_common_step = min(c["step"].max() for c in usable.values())
    if max_common_step <= 0:
        return None
    grid = np.linspace(0, max_common_step, n_points)
    resampled = np.vstack(
        [np.interp(grid, c["step"].to_numpy(), c["value"].to_numpy()) for c in usable.values()]
    )
    return grid, resampled.mean(axis=0)


def plot_tag_grid(
    curves_by_tag: dict[str, dict[str, dict[int, pd.DataFrame]]],
    tags: Sequence[str],
    titles: Sequence[str],
    groups: Sequence[str],
    *,
    ncols: int,
    suptitle: str,
    out_path: Path,
) -> None:
    nrows = int(np.ceil(len(tags) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.5 * nrows), squeeze=False)
    flat_axes = list(axes.flat)
    for tag, title, ax in zip(tags, titles, flat_axes):
        any_line = False
        for group in groups:
            curves = curves_by_tag.get(tag, {}).get(group, {})
            result = resample_mean(curves)
            if result is None:
                continue
            grid, mean = result
            ax.plot(grid, mean, color=GROUP_COLORS[group], linewidth=2.0, label=group)
            any_line = True
        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Training timestep")
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if any_line:
            ax.legend(loc="best", frameon=False, fontsize=8)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="#999")
    for ax in flat_axes[len(tags):]:
        ax.axis("off")
    fig.suptitle(suptitle + f"\n(EMA-smoothed, weight={SMOOTHING_WEIGHT}; mean of 7 seeds)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    out_dir = Path("artifacts") / "reports" / "lambda1p0_comparison" / "figures"

    training_tags = ["rollout/ep_rew_mean", "rollout/collisions_per_km", "rollout/distance_m"]
    stability_tags = [
        "train/approx_kl", "train/clip_fraction", "train/explained_variance", "train/std",
        "train/value_loss", "train/policy_gradient_loss", "train/entropy_loss", "train/loss",
    ]
    cbf_field_tags = ["train/cbf_mean_loss", "train/cbf_mean_correction", "train/cbf_mean_infeasible_rate"]
    all_tags = training_tags + stability_tags + cbf_field_tags

    print("Reading TensorBoard scalars for B2/C2, lambda=0.1 and lambda=1.0")
    curves = load_all(all_tags)
    groups = list(GROUPS)

    plot_tag_grid(
        curves,
        ["rollout/ep_rew_mean", "rollout/collisions_per_km", "rollout/distance_m"],
        ["Episode return", "Collisions / km (rollout)", "Distance (m)"],
        groups, ncols=3,
        suptitle="Training-rollout signals: B2/C2, lambda=0.1 vs lambda=1.0",
        out_path=out_dir / "lambda1p0_training_rollout_signals.png",
    )
    plot_tag_grid(
        curves,
        ["train/approx_kl", "train/clip_fraction", "train/explained_variance", "train/std"],
        ["Approx. KL", "Clip fraction", "Explained variance", "Action std (train/std)"],
        groups, ncols=2,
        suptitle="PPO optimizer health: B2/C2, lambda=0.1 vs lambda=1.0",
        out_path=out_dir / "lambda1p0_stability_optimizer_health.png",
    )
    plot_tag_grid(
        curves,
        ["train/value_loss", "train/policy_gradient_loss", "train/entropy_loss", "train/loss"],
        ["Value loss", "Policy-gradient loss", "Entropy loss", "Total loss"],
        groups, ncols=2,
        suptitle="PPO loss curves: B2/C2, lambda=0.1 vs lambda=1.0",
        out_path=out_dir / "lambda1p0_stability_losses.png",
    )
    plot_tag_grid(
        curves,
        ["train/cbf_mean_loss", "train/cbf_mean_correction", "train/cbf_mean_infeasible_rate"],
        ["CBF mean loss", "CBF mean correction norm", "CBF mean infeasible rate"],
        groups, ncols=3,
        suptitle="CBF projection signal during training: B2/C2, lambda=0.1 vs lambda=1.0",
        out_path=out_dir / "lambda1p0_cbf_field_projection_signal.png",
    )

    print(f"\nAll figures written to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
