"""Publication collage: HOCBF margin for a few scenarios, side by side.

Reads the ``hocbf_margin_series.csv`` that
``scripts.reporting.plot_hocbf_margin_aggregate`` already wrote and re-renders
it -- no rollouts, no environment, no policy. The figure is sized to sit across
the top of an A4 page and carries no titles or captions, on the assumption that
the document's own caption names the panels; the only text is the axes and one
legend.

The mean at each time is taken over the rollouts still running at that time
(the environment ends an episode at its collision, so a crashed rollout simply
has no later rows). That is the ``--censor drop`` convention of the aggregate
script, and the ribbon is the standard error of that mean.
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

from scripts.reporting.plot_hocbf_safety_margin import ARM_COLORS

DEFAULT_CONDITIONS = [
    "staggered_gap_selection",
    "rear_pressure_escape",
    "boxed_in_hold_position",
]
DEFAULT_POLICIES = ["A1_nominal", "B1_state_cost_raw", "B2_cbf_nd_reward_actor"]


def mean_and_sem(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = frame.groupby("time_s")["s_hocbf"]
    mean = grouped.mean()
    count = grouped.count()
    sem = grouped.std(ddof=1) / np.sqrt(count.clip(lower=1))
    return mean.index.to_numpy(), mean.to_numpy(), sem.fillna(0.0).to_numpy()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--series", type=Path, action="append", required=True,
                   help="hocbf_margin_series.csv to read; repeatable, and later "
                        "files win for a condition present in several.")
    p.add_argument("--condition", action="append", default=None,
                   help=f"Panels, left to right. Default: {DEFAULT_CONDITIONS}.")
    p.add_argument("--policy", action="append", default=None,
                   help=f"Curves per panel. Default: {DEFAULT_POLICIES}.")
    p.add_argument("--width", type=float, default=7.2,
                   help="Figure width in inches; 7.2 fits A4 with 2 cm margins.")
    p.add_argument("--height", type=float, default=2.45,
                   help="Figure height in inches.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--band", choices=("sem", "none"), default="sem")
    p.add_argument("--out", type=Path,
                   default=Path("artifacts") / "Results" / "HOCBF margin"
                   / "margin_scenario_collage.png")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    conditions = args.condition or DEFAULT_CONDITIONS
    policies = args.policy or DEFAULT_POLICIES

    frames = [pd.read_csv(path) for path in args.series]
    data = pd.concat(frames, ignore_index=True)
    # Later files win, so a re-run can supersede an earlier condition.
    data = data.drop_duplicates(
        subset=["condition", "policy", "episode_seed", "policy_step"], keep="last"
    )
    missing = [c for c in conditions if c not in set(data["condition"])]
    if missing:
        raise SystemExit(f"no rows for condition(s) {missing}; known: "
                         f"{sorted(set(data['condition']))}")

    fig, axes = plt.subplots(
        1, len(conditions), figsize=(args.width, args.height), sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, condition in zip(axes, conditions):
        panel = data[data["condition"] == condition]
        for label in policies:
            runs = panel[panel["policy"] == label]
            if runs.empty:
                continue
            t, mean, sem = mean_and_sem(runs)
            color = ARM_COLORS[label]
            if args.band == "sem":
                ax.fill_between(t, mean - sem, mean + sem, color=color,
                                alpha=0.16, linewidth=0.0, zorder=2)
            ax.plot(t, mean, color=color, linewidth=1.5, label=label, zorder=3)
        ax.axhline(0.0, color="#333333", linewidth=0.9, linestyle="--", zorder=1)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"mean $S_{HOCBF}(t)$", fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.995),
               frameon=False, fontsize=7.5, ncols=len(labels),
               columnspacing=1.1, handlelength=1.6, handletextpad=0.5)
    # Room for the legend strip only -- no titles, no captions.
    fig.subplots_adjust(top=0.845, bottom=0.215, left=0.082, right=0.995,
                        wspace=0.12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=int(args.dpi), facecolor="white")
    plt.close(fig)

    n_runs = (data[data["condition"] == conditions[0]]
              .groupby("policy")["episode_seed"].nunique().to_dict())
    print(f"wrote {args.out} ({args.width} x {args.height} in at {args.dpi} dpi)")
    print(f"panels, left to right: {', '.join(conditions)}")
    print(f"rollouts per policy in panel 1: {n_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
