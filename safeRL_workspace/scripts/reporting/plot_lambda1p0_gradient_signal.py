"""Compare the actor gradient signal between B2 and C2 at lambda=0.1 vs 1.0.

Both variants log the same four diagnostics during training (see
scripts/common/projected_ppo_cbf.py): the norm of the actor-loss gradient
contributed by the CBF-consistency term, the norm of the ordinary PPO
policy-gradient term, their ratio, and their cosine similarity. B2's CBF
gradient comes from a plain loss against a detached (stop-gradient) target;
C2's comes from backpropagating through the differentiable projection's own
Jacobian (d mu_safe / d mu_raw) into mu_raw. Comparing these curves directly
answers what the Jacobian is actually contributing to the gradient the actor
receives, versus what B2's non-differentiable target contributes -- rather
than inferring it indirectly from downstream KPIs, which is all the earlier
comparison used.
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

RUNS = Path.home() / "AppData" / "Local" / "Temp" / "saferl_runs"
SEEDS = [307, 327, 347, 367, 387, 407, 427]

def matrix_rel(prefix: str, variant: str, seed: int) -> str:
    return f"{prefix}/seed_{seed}/{variant}/seed_{seed}"


def lambda_study_rel(prefix: str, variant: str, seed: int) -> str:
    return f"{prefix}_s{seed}/{variant}/seed_{seed}"


GROUPS = {
    "B2 lambda=0.1": ("B2_cbf_nd_reward_actor", "ppo_cbf_nd_reward_actor", RUNS / "safety_matrix_1M", matrix_rel),
    "B2 lambda=1.0": ("ldac1p0", "ppo_cbf_nd_reward_actor", RUNS / "lambda_detached_actor_1p0", lambda_study_rel),
    "C2 lambda=0.1": ("C2_cbf_projected", "ppo_cbf_projected", RUNS / "safety_matrix_1M", matrix_rel),
    "C2 lambda=1.0": ("lam1p0", "ppo_cbf_projected", RUNS / "lambda_mean_1p0", lambda_study_rel),
}
GROUP_COLORS = {
    "B2 lambda=0.1": "#70AD47",
    "B2 lambda=1.0": "#2E7D32",
    "C2 lambda=0.1": "#997300",
    "C2 lambda=1.0": "#B8860B",
}
GROUP_STYLES = {
    "B2 lambda=0.1": "--",
    "B2 lambda=1.0": "-",
    "C2 lambda=0.1": "--",
    "C2 lambda=1.0": "-",
}

TAGS = [
    "train/actor_g_cbf_norm",
    "train/actor_g_ppo_norm",
    "train/actor_g_cbf_to_g_ppo_ratio",
    "train/actor_g_ppo_g_cbf_cosine",
]
TAG_TITLES = {
    "train/actor_g_cbf_norm": "||grad from CBF-consistency term||",
    "train/actor_g_ppo_norm": "||grad from ordinary PPO term||",
    "train/actor_g_cbf_to_g_ppo_ratio": "Gradient ratio (CBF / PPO)",
    "train/actor_g_ppo_g_cbf_cosine": "Cosine(grad_PPO, grad_CBF)",
}


def find_event_file(base: Path, rel_fn, prefix: str, variant: str, seed: int) -> Path | None:
    rel = rel_fn(prefix, variant, seed)
    root = base / rel
    matches = sorted(root.glob("tensorboard/*/events.out.tfevents.*")) if root.exists() else []
    if not matches:
        top = base / rel.split("/")[0]
        matches = sorted(top.rglob("events.out.tfevents.*")) if top.exists() else []
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


def resample_mean(curves: dict[int, pd.DataFrame], n_points: int = 300) -> tuple[np.ndarray, np.ndarray] | None:
    usable = {seed: c for seed, c in curves.items() if not c.empty}
    if not usable:
        return None
    max_common_step = min(c["step"].max() for c in usable.values())
    if max_common_step <= 0:
        return None
    grid = np.linspace(0, max_common_step, n_points)
    resampled = np.vstack([np.interp(grid, c["step"].to_numpy(), c["value"].to_numpy()) for c in usable.values()])
    return grid, resampled.mean(axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "reports" / "safety_matrix_1M" / "figures")
    args = parser.parse_args()

    all_curves: dict[str, dict[str, dict[int, pd.DataFrame]]] = {tag: {g: {} for g in GROUPS} for tag in TAGS}
    for group_name, (prefix, variant, base, rel_fn) in GROUPS.items():
        for seed in SEEDS:
            event_path = find_event_file(base, rel_fn, prefix, variant, seed)
            if event_path is None:
                print(f"  MISSING event file: {group_name} seed {seed}")
                continue
            curves = load_curves(event_path, TAGS)
            for tag, curve in curves.items():
                all_curves[tag][group_name][seed] = curve

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    for tag, ax in zip(TAGS, axes.flat):
        for group_name in GROUPS:
            result = resample_mean(all_curves[tag].get(group_name, {}))
            if result is None:
                continue
            grid, mean = result
            ax.plot(
                grid, mean,
                color=GROUP_COLORS[group_name], linestyle=GROUP_STYLES[group_name],
                linewidth=2.2, label=group_name,
            )
        if "ratio" in tag:
            ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle=":", label="ratio = 1")
        if "cosine" in tag:
            ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle=":")
        ax.set_title(TAG_TITLES[tag], fontsize=11, fontweight="semibold")
        ax.set_xlabel("Training timestep")
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="best", frameon=False, fontsize=8)

    fig.suptitle(
        "Actor gradient signal: CBF-consistency term vs. ordinary PPO term\n"
        "solid = lambda 1.0, dashed = lambda 0.1 (baseline); mean of available seeds",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "gradient_signal_b2_vs_c2_lambda_comparison.png"
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
