"""Quantify the gap between shielded safety and the policy's own safety.

The shield gap of a policy is its ego collision rate with the HOCBF shield off
(CBF-OFF, the policy alone) divided by its rate with the shield on (CBF-ON).
A gap of 15x means the deployed system looks fifteen times safer than the
network driving it. Three panels:

A. the gap for every paired CBF-OFF/CBF-ON measurement recorded in the
   experiment docs;
B. C2, whose internal projection runs even under the standard CBF-OFF
   evaluation, so that evaluation hides the gap;
C. one fixed policy under five shield configurations: the reported safety
   moves with the shield alone.

Every number is transcribed from a committed document named in its ``source``
field; nothing is recomputed. Collision rates are ego collisions per km,
averaged over seeds where a document reports several.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

TRIAL_LOG = "docs/experiments/trial_log.md"
STATE_COST = "docs/experiments/hocbf_state_cost_20260915.md"
D60_LOG = "docs/congested_d60_experiment_log.md"
GAIN_ABLATION = "docs/experiments/cbf_gain_density_ablation_20260911.md"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8985"
GRID = "#e6e5e1"
POLICY_ALONE = "#eb6834"  # CBF-OFF
WITH_SHIELD = "#2a78d6"  # CBF-ON


@dataclass(frozen=True)
class PairedRate:
    group: str
    label: str
    cbf_off: float
    cbf_on: float
    source: str

    @property
    def gap(self) -> float:
        return self.cbf_off / self.cbf_on


@dataclass(frozen=True)
class Bar:
    label: str
    rate: float
    sd: float | None
    shielded: bool


PAIRS = [
    PairedRate("A1 nominal, 1M, 3 seeds", "collision penalty −2.5", 3.501, 0.277,
               f"{TRIAL_LOG} §4b"),
    PairedRate("A1 nominal, 1M, 3 seeds", "collision penalty −25", 4.915, 0.263,
               f"{TRIAL_LOG} §4b"),
    PairedRate("A1 nominal, 1M, 3 seeds", "collision penalty −60", 5.413, 0.324,
               f"{TRIAL_LOG} §4b"),
    PairedRate("A1 nominal, 1M, 3 seeds", "Huber + split clip, penalty −250", 5.199,
               0.176, f"{TRIAL_LOG} §4d"),
    PairedRate("A1 nominal, 3M, 5 seeds", "desired speed 20 m/s", 4.476, 0.300,
               f"{TRIAL_LOG} §3"),
    PairedRate("State-cost study, 1M, 3 seeds", "A1 control", 3.712, 0.268,
               STATE_COST),
    PairedRate("State-cost study, 1M, 3 seeds", "B1-type state cost (ellipse)", 4.196,
               0.320, STATE_COST),
    PairedRate("60 vehicles, CBF in training", "B2, λ = 1.0", 7.473, 2.859, D60_LOG),
    PairedRate("60 vehicles, CBF in training", "C2, λ = 1.0, projection bypassed",
               8.105, 2.865, D60_LOG),
]

# C2 at 60 vehicles under its three evaluations (SD over 10 blocks of 20 episodes).
C2_EVALUATIONS = [
    Bar("Network alone\n(projection bypassed)", 8.105, 0.639, shielded=False),
    Bar("Standard CBF-OFF\n(internal projection on)", 2.865, 0.016, shielded=True),
    Bar("CBF-ON\n(projection + shield)", 2.865, 0.033, shielded=True),
]

# One nominal policy (one training seed, trained without a CBF), 40 vehicles,
# 200 paired episodes per cell; SD over 10 blocks of 20 episodes.
SHIELD_CONFIGS = [
    Bar("Shield off", 4.90, 1.72, shielded=False),
    Bar("Former gains\nk0 5.29, k1 3.68", 2.71, 0.70, shielded=True),
    Bar("(c1, c2) = (2.3, 2.3)", 1.71, 0.37, shielded=True),
    Bar("(c1, c2) = (0.5, 3.5)", 0.63, 0.18, shielded=True),
    Bar("(c1, c2) = (0.5, 5)", 0.51, 0.19, shielded=True),
    Bar("(c1, c2) = (0.5, 8)\ncanonical", 0.50, 0.16, shielded=True),
]


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_MUTED)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def panel_title(ax: plt.Axes, letter: str, title: str, subtitle: str) -> None:
    ax.set_title(f"{letter}   {title}", loc="left", fontsize=11.5,
                 fontweight="bold", color=TEXT, pad=26)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=9,
            color=TEXT_SECONDARY, va="bottom")


def plot_gap_dumbbell(ax: plt.Axes) -> None:
    rows: list[tuple[float, PairedRate | str]] = []
    y = 0.0
    previous_group = None
    for pair in PAIRS:
        if pair.group != previous_group:
            if previous_group is not None:
                y += 0.55
            rows.append((y, pair.group))
            y += 0.8
            previous_group = pair.group
        rows.append((y, pair))
        y += 1.0

    bottom = y
    label_at = blended_transform_factory(ax.transAxes, ax.transData)
    for y, row in rows:
        if isinstance(row, str):
            ax.text(-0.02, y, row.upper(), transform=label_at, fontsize=7.8,
                    fontweight="bold", color=TEXT_MUTED, va="center", ha="right")
            continue
        ax.plot([row.cbf_on, row.cbf_off], [y, y], color="#c9c8c3",
                linewidth=2.0, solid_capstyle="round", zorder=2)
        ax.scatter(row.cbf_on, y, s=70, color=WITH_SHIELD, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.scatter(row.cbf_off, y, s=70, color=POLICY_ALONE, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.text(row.cbf_off * 1.13, y, f"{row.gap:.1f}×", fontsize=10,
                fontweight="bold", color=TEXT, va="center")
        ax.text(-0.02, y, row.label, transform=label_at, fontsize=9, color=TEXT,
                va="center", ha="right")

    ax.set_xscale("log")
    ax.set_xlim(0.1, 14)
    ax.set_ylim(bottom - 0.4, -0.6)
    ax.set_yticks([])
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10])
    ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10"])
    ax.set_xlabel("Ego collisions per km (log scale)", fontsize=9.5,
                  color=TEXT_SECONDARY)
    style_axis(ax)

    first = next(row for _, row in rows if isinstance(row, PairedRate))
    first_y = next(y for y, row in rows if row is first)
    ax.annotate("with shield", (first.cbf_on, first_y), xytext=(0, 11),
                textcoords="offset points", ha="center", fontsize=8.5,
                color=TEXT_SECONDARY)
    ax.annotate("policy alone", (first.cbf_off, first_y), xytext=(0, 11),
                textcoords="offset points", ha="center", fontsize=8.5,
                color=TEXT_SECONDARY)


def plot_bars(ax: plt.Axes, bars: list[Bar], reference: float, effect: str) -> None:
    """Horizontal bars; shielded bars note their reduction from ``reference``.

    ``effect`` is a format string that receives that reduction as a fraction.
    """
    ys = list(range(len(bars)))
    for y, bar in zip(ys, bars):
        color = WITH_SHIELD if bar.shielded else POLICY_ALONE
        ax.barh(y, bar.rate, height=0.56, color=color, zorder=2)
        if bar.sd is not None:
            ax.errorbar(bar.rate, y, xerr=bar.sd, color=TEXT_SECONDARY,
                        elinewidth=1.0, capsize=2.5, zorder=3)
        end = bar.rate + (bar.sd or 0.0)
        note = f"{bar.rate:.2f}"
        if bar.shielded:
            note += "   " + effect.format(1 - bar.rate / reference)
        ax.text(end + 0.12, y, note, fontsize=8.8, color=TEXT, va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([bar.label for bar in bars], fontsize=8.8, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("Ego collisions per km", fontsize=9.5, color=TEXT_SECONDARY)
    style_axis(ax)


def build_figure() -> plt.Figure:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(14.5, 8.6), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0],
                            height_ratios=[0.78, 1.22], wspace=0.5, hspace=0.62,
                            left=0.215, right=0.975, top=0.83, bottom=0.13)

    ax_gap = fig.add_subplot(grid[:, 0])
    plot_gap_dumbbell(ax_gap)
    panel_title(ax_gap, "A", "The shield gap",
                "Collision rate of the policy alone ÷ rate with the shield on")

    ax_c2 = fig.add_subplot(grid[0, 1])
    plot_bars(ax_c2, C2_EVALUATIONS, reference=C2_EVALUATIONS[0].rate,
              effect="{:.0%} below the network alone")
    ax_c2.set_xlim(0, 11.5)
    panel_title(ax_c2, "B", "The standard CBF-OFF number can hide it",
                "C2 carries its own projection; only bypassing it measures the network")

    ax_gain = fig.add_subplot(grid[1, 1])
    plot_bars(ax_gain, SHIELD_CONFIGS, reference=SHIELD_CONFIGS[0].rate,
              effect="shield removes {:.0%}")
    ax_gain.set_xlim(0, 9.5)
    spread = SHIELD_CONFIGS[1].rate / SHIELD_CONFIGS[-1].rate
    panel_title(ax_gain, "C", "Same policy, different shields",
                f"One nominal policy; the shield alone moves its reported rate {spread:.1f}×")

    forty = [p.gap for p in PAIRS if not p.group.startswith("60")]
    fig.text(0.03, 0.955, "Shielded safety overstates what the policy learned",
             fontsize=17, fontweight="bold", color=TEXT)
    fig.text(0.03, 0.935,
             f"In every paired 40-vehicle measurement in the repository, the policy alone "
             f"crashes {min(forty):.0f}–{max(forty):.0f}× more often than the shielded "
             "system.\nA collision rate measured with the shield on mostly describes the "
             "shield.",
             fontsize=10.5, color=TEXT_SECONDARY, va="top", linespacing=1.5)

    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markersize=8, color=POLICY_ALONE,
               label="Policy alone (CBF-OFF)"),
        Line2D([], [], marker="o", linestyle="", markersize=8, color=WITH_SHIELD,
               label="With the HOCBF shield (CBF-ON)"),
    ], loc="upper right", bbox_to_anchor=(0.975, 0.965), frameon=False,
        fontsize=9.5, ncols=2)

    fig.text(0.03, 0.025,
             "Sources in safeRL_workspace/docs/:  A  experiments/trial_log.md, "
             "experiments/hocbf_state_cost_20260915.md, congested_d60_experiment_log.md"
             "   B  congested_d60_experiment_log.md"
             "   C  experiments/cbf_gain_density_ablation_20260911.md\n"
             "All rows except the 60-vehicle ones use 40 vehicles. Rates are means "
             "over the stated seeds; error bars are SD over 10 blocks of 20 episodes.",
             fontsize=7.8, color=TEXT_MUTED, va="bottom")
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True,
                        help="Output image path; the extension sets the format.")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    fig = build_figure()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=SURFACE)
    plt.close(fig)
    for pair in PAIRS:
        print(f"{pair.group:32s} {pair.label:36s} gap {pair.gap:5.1f}x  ({pair.source})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
