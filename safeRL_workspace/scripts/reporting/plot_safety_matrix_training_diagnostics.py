"""Training-diagnostic plots for the safety-matrix 1M study.

Scope (confirmed by user, 2026-09-17):

- Training signals: reward curve, collisions/km (training rollout), distance
  traveled (training rollout), and the final post-training 10-KPI comparison.
- RL stability: PPO optimizer health (approx_kl, clip_fraction,
  explained_variance, std) and loss curves (value/policy-gradient/entropy/total).
- CBF field: projection loss/correction/infeasibility (arms with CBF in the
  training loop only: B2, C1, C2) and the actor gradient signal
  (arms with an actor-loss CBF term only: B2, C2).

TensorBoard scalars for the training-signal and stability sections exist for
every arm; the CBF-field tags only exist where the mechanism they describe
is actually present, so those figures silently skip arms with no matching
tag rather than plotting an empty/zero line.
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

from scripts.reporting.build_safety_matrix_results_tables import (
    ALL_ARMS,
    DEFAULT_SEEDS,
    DEFAULT_STUDY_DIR,
    SAFETY_KPIS,
    TASK_KPIS,
    aggregate_across_seeds,
    collect,
)

ARM_COLORS = {
    "A1_nominal": "#4472C4",
    "B1_state_cost_raw": "#ED7D31",
    "B2_cbf_nd_reward_actor": "#70AD47",
    "C1_cbf_diff_reward_only": "#9E480E",
    "C2_cbf_projected": "#997300",
}
ALL_FIVE = list(ALL_ARMS)
CBF_TRAINING_ARMS = ["B2_cbf_nd_reward_actor", "C1_cbf_diff_reward_only", "C2_cbf_projected"]
ACTOR_GRADIENT_ARMS = ["B2_cbf_nd_reward_actor", "C2_cbf_projected"]


def find_event_file(study_dir: Path, arm: str, seed: int) -> Path | None:
    variant = ALL_ARMS[arm][1]
    pattern = f"{arm}/seed_{seed}/{variant}/seed_{seed}/tensorboard/*/events.out.tfevents.*"
    matches = sorted(study_dir.glob(pattern))
    if not matches:
        matches = sorted((study_dir / arm / f"seed_{seed}").rglob("events.out.tfevents.*"))
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
        out[tag] = pd.DataFrame(
            {"step": [s.step for s in scalars], "value": [s.value for s in scalars]}
        )
    return out


def load_all(
    study_dir: Path, arms: Sequence[str], seeds: Sequence[int], tags: Sequence[str]
) -> dict[str, dict[str, dict[int, pd.DataFrame]]]:
    """tag -> arm -> seed -> DataFrame(step, value)."""

    result: dict[str, dict[str, dict[int, pd.DataFrame]]] = {tag: {} for tag in tags}
    for arm in arms:
        for tag in tags:
            result[tag].setdefault(arm, {})
        for seed in seeds:
            event_path = find_event_file(study_dir, arm, seed)
            if event_path is None:
                print(f"  MISSING event file: {arm} seed {seed}")
                continue
            curves = load_curves(event_path, tags)
            for tag, curve in curves.items():
                result[tag][arm][seed] = curve
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
    arms: Sequence[str],
    *,
    ncols: int,
    suptitle: str,
    out_path: Path,
    ylabel: str = "value",
) -> None:
    nrows = int(np.ceil(len(tags) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.4 * nrows), squeeze=False)
    flat_axes = axes.flat
    for tag, title, ax in zip(tags, titles, flat_axes):
        any_line = False
        for arm in arms:
            curves = curves_by_tag.get(tag, {}).get(arm, {})
            result = resample_mean(curves)
            if result is None:
                continue
            grid, mean = result
            code = ALL_ARMS[arm][0]
            ax.plot(grid, mean, color=ARM_COLORS[arm], linewidth=2.0, label=code)
            any_line = True
        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Training timestep")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if any_line:
            ax.legend(loc="best", frameon=False, fontsize=8)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="#999")
    for ax in list(flat_axes)[len(tags):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_reward_curves(
    curves_by_tag: dict[str, dict[str, dict[int, pd.DataFrame]]], arms: Sequence[str], out_path: Path
) -> None:
    ncols = 3
    nrows = int(np.ceil(len(arms) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.6 * nrows), squeeze=False)
    flat_axes = list(axes.flat)
    for arm, ax in zip(arms, flat_axes):
        curves = curves_by_tag["rollout/ep_rew_mean"].get(arm, {})
        for seed, curve in sorted(curves.items()):
            if curve.empty:
                continue
            ax.plot(curve["step"], curve["value"], color=ARM_COLORS[arm], alpha=0.25, linewidth=1.0)
        result = resample_mean(curves)
        code, variant = ALL_ARMS[arm]
        if result is not None:
            grid, mean = result
            ax.plot(grid, mean, color=ARM_COLORS[arm], linewidth=2.4, label=f"mean of {len(curves)} seeds")
            ax.legend(loc="lower right", frameon=False, fontsize=8)
        ax.set_title(f"{code} — {variant}", fontsize=11, fontweight="semibold")
        ax.set_xlabel("Training timestep")
        ax.set_ylabel("Episode return (rollout mean)")
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in flat_axes[len(arms):]:
        ax.axis("off")
    fig.suptitle("Training reward, per policy (own seeds, own scale)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_combined_overlay(
    curves_by_tag: dict[str, dict[str, dict[int, pd.DataFrame]]],
    tag: str,
    arms: Sequence[str],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for arm in arms:
        curves = curves_by_tag[tag].get(arm, {})
        result = resample_mean(curves)
        if result is None:
            continue
        grid, mean = result
        code, variant = ALL_ARMS[arm]
        ax.plot(grid, mean, color=ARM_COLORS[arm], linewidth=2.2, label=f"{code} {variant}")
    ax.set_title(title, fontsize=13, fontweight="semibold")
    ax.set_xlabel("Training timestep")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_kpi_bars(
    agg_df: pd.DataFrame, kpis: Sequence[str], *, title: str, out_path: Path
) -> None:
    policies = list(agg_df["policy"].drop_duplicates())
    ncols = 3
    nrows = int(np.ceil(len(kpis) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.3 * nrows), squeeze=False)
    flat_axes = list(axes.flat)
    x = np.arange(len(policies))
    width = 0.36
    for kpi, ax in zip(kpis, flat_axes):
        for offset, mode, color, label in (
            (-width / 2, "raw", "#4472C4", "CBF-OFF"),
            (width / 2, "cbf", "#ED7D31", "CBF-ON"),
        ):
            means, sds = [], []
            for policy in policies:
                row = agg_df[
                    (agg_df["policy"] == policy)
                    & (agg_df["mode"] == mode)
                    & (agg_df["kpi"] == kpi)
                ]
                if row.empty:
                    means.append(np.nan)
                    sds.append(0.0)
                else:
                    means.append(float(row.iloc[0]["seed_mean"]))
                    sds.append(float(row.iloc[0]["seed_sd"]) if pd.notna(row.iloc[0]["seed_sd"]) else 0.0)
            ax.bar(x + offset, means, width, yerr=sds, capsize=3, color=color, label=label, alpha=0.9)
        ax.set_title(kpi, fontsize=10, fontweight="semibold")
        codes = [policy.split(" — ")[0] for policy in policies]
        ax.set_xticks(x)
        ax.set_xticklabels(codes, fontsize=9)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in flat_axes[len(kpis):]:
        ax.axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncols=2)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "reports" / "safety_matrix_1M" / "figures")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    arms = ALL_FIVE

    training_tags = ["rollout/ep_rew_mean", "rollout/collisions_per_km", "rollout/distance_m"]
    stability_tags = [
        "train/approx_kl", "train/clip_fraction", "train/explained_variance", "train/std",
        "train/value_loss", "train/policy_gradient_loss", "train/entropy_loss", "train/loss",
    ]
    cbf_field_tags = [
        "train/cbf_mean_loss", "train/cbf_mean_correction", "train/cbf_mean_infeasible_rate",
        "train/actor_g_cbf_norm", "train/actor_g_ppo_norm",
        "train/actor_g_cbf_to_g_ppo_ratio", "train/actor_g_ppo_g_cbf_cosine",
    ]
    all_tags = training_tags + stability_tags + cbf_field_tags

    print(f"Reading TensorBoard scalars from {args.study_dir}")
    curves = load_all(args.study_dir, arms, args.seeds, all_tags)

    # --- 1. Training signals -------------------------------------------------
    plot_reward_curves(curves, arms, args.out / "training_reward_curves.png")
    plot_combined_overlay(
        curves, "rollout/collisions_per_km", arms,
        title="Training-rollout collisions/km, all 5 arms",
        ylabel="Collisions / km (rollout)",
        out_path=args.out / "training_collisions_per_km.png",
    )
    plot_combined_overlay(
        curves, "rollout/distance_m", arms,
        title="Training-rollout episode distance, all 5 arms",
        ylabel="Distance (m)",
        out_path=args.out / "training_distance_m.png",
    )

    long_df = collect(args.study_dir, arms, args.seeds)
    safety_agg = aggregate_across_seeds(long_df, SAFETY_KPIS)
    task_agg = aggregate_across_seeds(long_df, TASK_KPIS)
    plot_kpi_bars(
        safety_agg, SAFETY_KPIS,
        title="Final post-training safety/CBF KPIs (mean ± SD across 7 seeds)",
        out_path=args.out / "final_kpi_safety_bars.png",
    )
    plot_kpi_bars(
        task_agg, TASK_KPIS,
        title="Final post-training task/comfort KPIs (mean ± SD across 7 seeds)",
        out_path=args.out / "final_kpi_task_bars.png",
    )

    # --- 2. RL stability -------------------------------------------------------
    plot_tag_grid(
        curves,
        ["train/approx_kl", "train/clip_fraction", "train/explained_variance", "train/std"],
        ["Approx. KL", "Clip fraction", "Explained variance", "Action std (train/std)"],
        arms, ncols=2, suptitle="PPO optimizer health, all 5 arms",
        out_path=args.out / "stability_optimizer_health.png",
    )
    plot_tag_grid(
        curves,
        ["train/value_loss", "train/policy_gradient_loss", "train/entropy_loss", "train/loss"],
        ["Value loss", "Policy-gradient loss", "Entropy loss", "Total loss"],
        arms, ncols=2, suptitle="PPO loss curves, all 5 arms",
        out_path=args.out / "stability_losses.png",
    )

    # --- 3. CBF field ------------------------------------------------------
    plot_tag_grid(
        curves,
        ["train/cbf_mean_loss", "train/cbf_mean_correction", "train/cbf_mean_infeasible_rate"],
        ["CBF mean loss", "CBF mean correction norm", "CBF mean infeasible rate"],
        CBF_TRAINING_ARMS, ncols=3,
        suptitle="CBF projection signal during training (B2 / C1 / C2 only)",
        out_path=args.out / "cbf_field_projection_signal.png",
    )
    plot_tag_grid(
        curves,
        [
            "train/actor_g_cbf_norm", "train/actor_g_ppo_norm",
            "train/actor_g_cbf_to_g_ppo_ratio", "train/actor_g_ppo_g_cbf_cosine",
        ],
        ["Actor grad norm (CBF term)", "Actor grad norm (PPO term)", "Grad ratio (CBF/PPO)", "Grad cosine (CBF, PPO)"],
        ACTOR_GRADIENT_ARMS, ncols=2,
        suptitle="Actor gradient signal (B2 / C2 only)",
        out_path=args.out / "cbf_field_actor_gradient.png",
    )

    print(f"\nAll figures written to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
