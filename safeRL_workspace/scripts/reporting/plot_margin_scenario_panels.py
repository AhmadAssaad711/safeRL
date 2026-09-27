"""Re-render the standalone per-scenario HOCBF margin figures from saved data.

Same layout as the ``per_scenario/`` figures that
``scripts.reporting.plot_hocbf_margin_aggregate`` writes -- scene strip on top,
margin curves below, legend above them -- but built from that run's saved
``hocbf_margin_series.csv`` rather than from fresh rollouts, so an arm can be
dropped or the text changed without touching the experiment. The curves are
therefore bit-identical to the originals for the arms that are kept.

The scene strip is not in the CSV, so it is cropped out of the original figure
rather than re-rendered: re-driving the environment today would not necessarily
reproduce the frame the original figure shows.

No title: the document's caption names the scenario. The legend stays, since it
is the only thing identifying the arms.
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
from PIL import Image

from scripts.reporting.plot_hocbf_safety_margin import ARM_COLORS

DEFAULT_POLICIES = ["A1_nominal", "B1_state_cost_raw", "B2_cbf_nd_reward_actor"]


def crop_scene_strip(path: Path, *, search_fraction: float = 0.55) -> np.ndarray | None:
    """Cut the rendered road strip out of an existing per-scenario figure.

    The strip is the only wide block of mid-grey in the upper part of the page;
    title and legend are thin dark text on white and never fill a row.
    """

    if not path.is_file():
        return None
    image = np.asarray(Image.open(path).convert("RGB"))
    top = image[: int(image.shape[0] * search_fraction)]
    grey = (
        (np.abs(top[:, :, 0].astype(int) - top[:, :, 1]) < 12)
        & (np.abs(top[:, :, 1].astype(int) - top[:, :, 2]) < 12)
        & (top[:, :, 0] > 40) & (top[:, :, 0] < 170)
    )
    rows = np.flatnonzero(grey.sum(axis=1) > 0.2 * top.shape[1])
    cols = np.flatnonzero(grey.sum(axis=0) > 0.2 * max(len(rows), 1))
    if rows.size == 0 or cols.size == 0:
        return None
    return image[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def mean_and_sem(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = frame.groupby("time_s")["s_hocbf"]
    mean = grouped.mean()
    count = grouped.count()
    sem = (grouped.std(ddof=1) / np.sqrt(count.clip(lower=1))).fillna(0.0)
    return mean.index.to_numpy(), mean.to_numpy(), sem.to_numpy()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--series", type=Path, required=True,
                   help="hocbf_margin_series.csv from the run to re-render.")
    p.add_argument("--scenario", action="append", required=True,
                   help="Scenario condition to render; repeatable.")
    p.add_argument("--policy", action="append", default=None,
                   help=f"Arms to keep. Default: {DEFAULT_POLICIES}.")
    p.add_argument("--strip-from", type=Path, default=None,
                   help="Directory of the original <scenario>.png figures, used "
                        "only to crop the scene strip. Omit to draw no strip.")
    p.add_argument("--band", choices=("sem", "none"), default="sem")
    p.add_argument("--out", type=Path, required=True, help="Output directory.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    policies = args.policy or DEFAULT_POLICIES
    data = pd.read_csv(args.series)
    known = set(data["condition"])
    missing = [s for s in args.scenario if s not in known]
    if missing:
        raise SystemExit(f"no rows for {missing}; known: {sorted(known)}")

    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.scenario:
        panel = data[data["condition"] == name]
        strip = (crop_scene_strip(args.strip_from / f"{name}.png")
                 if args.strip_from else None)
        has_strip = strip is not None

        heights = ([1.15] if has_strip else []) + [3.0]
        fig = plt.figure(figsize=(8.6, 5.0 + (1.5 if has_strip else 0.0)))
        gs = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.26)
        offset = 0
        if has_strip:
            ax_strip = fig.add_subplot(gs[0, 0])
            ax_strip.axis("off")
            ax_strip.imshow(strip, interpolation="antialiased")
            offset = 1

        ax = fig.add_subplot(gs[offset, 0])
        n_runs = {}
        for label in policies:
            runs = panel[panel["policy"] == label]
            if runs.empty:
                continue
            n_runs[label] = int(runs["episode_seed"].nunique())
            t, mean, sem = mean_and_sem(runs)
            color = ARM_COLORS[label]
            ax.plot(t, mean, color=color, linewidth=2.4, label=label, zorder=4)
            if args.band != "none":
                ax.fill_between(t, mean - sem, mean + sem, color=color,
                                alpha=0.15, linewidth=0, zorder=2)
        ax.axhline(0.0, color="#333333", linewidth=1.2, linestyle="--", zorder=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"mean $S_{HOCBF}(t)$")
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, 0.965), frameon=False,
                   fontsize=8.5, ncols=len(labels))
        fig.subplots_adjust(top=0.88, bottom=0.10, left=0.11, right=0.975)
        path = args.out / f"{name}.png"
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)
        print(f"wrote {path}  strip={'yes' if has_strip else 'no'}  rollouts={n_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
