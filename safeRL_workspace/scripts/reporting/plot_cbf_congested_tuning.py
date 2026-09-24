"""Report the congested (60-vehicle) CBF pole-placement tuning sweep.

The sweep parameterises the HOCBF cascade by its closed-loop poles rather than
by raw gains:

    psi2 = h_ddot + k1*h_dot + k0*h >= 0     ->     s^2 + k1 s + k0

so with natural frequency omega_n and damping ratio zeta,

    k0 = omega_n^2        k1 = 2 * zeta * omega_n

and the critically damped locus (zeta = 1, repeated real pole at -omega_n) is

    k1 = 2 * sqrt(k0)

which is drawn on every gain-plane panel. Below it the poles are complex
(underdamped, oscillatory barrier response); above it they are real and
distinct (overdamped, sluggish).

Consumes the sweep's own summary.csv; it never re-runs the sweep.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    FixedFormatter,
    FixedLocator,
    NullFormatter,
    NullLocator,
)

# KPI column -> (axis label, lower-is-better)
PANELS: list[tuple[str, str, bool]] = [
    ("collisions_per_km", "Ego collisions / km", True),
    ("qp_failure_rate", "QP failure rate", True),
    ("intervention_rate", "Intervention rate", True),
    ("correction_norm", "Correction norm", True),
    ("min_h", "Minimum h", False),
    ("jerk_norm", "Mean jerk norm", True),
]

OMEGA_COLORS = ["#4472C4", "#ED7D31", "#70AD47", "#9E480E", "#997300", "#7F7F7F"]


def read_summary(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("is_probe", "kpi_ok"):
                r[k] = str(v).strip().lower() == "true"
            else:
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return rows


def plot_vs_zeta(rows: list[dict], out: Path, *, title_suffix: str) -> None:
    """KPI against damping ratio, one line per natural frequency."""

    omegas = sorted({r["omega_n"] for r in rows if not r["is_probe"]})
    colors = {w: OMEGA_COLORS[i % len(OMEGA_COLORS)] for i, w in enumerate(omegas)}
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6), squeeze=False)
    for (column, label, lower_better), ax in zip(PANELS, axes.flat):
        for w in omegas:
            pts = sorted(
                (r for r in rows if r["omega_n"] == w and not r["is_probe"]),
                key=lambda r: r["zeta"],
            )
            if not pts:
                continue
            z = [p["zeta"] for p in pts]
            y = [p[column] for p in pts]
            sd_col = f"{column}_sd"
            ax.plot(z, y, marker="o", markersize=4.5, linewidth=1.9,
                    color=colors[w], label=f"$\\omega_n$ = {w:g}", zorder=4)
            if sd_col in pts[0]:
                sd = np.asarray([p.get(sd_col, 0.0) for p in pts], dtype=float)
                ax.fill_between(z, np.asarray(y) - sd, np.asarray(y) + sd,
                                color=colors[w], alpha=0.10, linewidth=0, zorder=2)
        ax.axvline(1.0, color="#C00000", linewidth=1.8, linestyle="--", zorder=3)
        ax.text(1.0, ax.get_ylim()[1], " critically damped\n $\\zeta$ = 1",
                color="#C00000", fontsize=8.5, va="top", ha="left")
        ax.set_xscale("log")
        ticks = [0.5, 0.7, 1.0, 1.4, 2.125, 3.0]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
        # A log axis otherwise draws its own decade minor labels straight
        # through the sweep's tick labels.
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"damping ratio $\zeta$")
        ax.set_ylabel(label + ("  (lower better)" if lower_better else "  (higher better)"))
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="#C00000", linestyle="--", linewidth=1.8))
    labels.append(r"critical damping $\zeta = 1$")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               frameon=False, ncols=len(labels), fontsize=9.5)
    fig.suptitle(f"CBF pole-placement sweep -- {title_suffix}",
                 fontsize=15.0, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.855, bottom=0.075, left=0.05, right=0.985,
                        hspace=0.32, wspace=0.26)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def plot_gain_plane(rows: list[dict], out: Path, *, title_suffix: str) -> None:
    """The (k0, k1) plane with the critical-damping locus k1 = 2*sqrt(k0)."""

    live = [r for r in rows if not r["is_probe"]]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4), squeeze=False)
    for (column, label, lower_better), ax in zip(PANELS[:3], axes.flat):
        k0 = np.asarray([r["k0"] for r in live], dtype=float)
        k1 = np.asarray([r["k1"] for r in live], dtype=float)
        v = np.asarray([r[column] for r in live], dtype=float)
        cmap = "viridis_r" if lower_better else "viridis"
        sc = ax.scatter(k0, k1, c=v, s=190, cmap=cmap, edgecolor="white",
                        linewidth=0.8, zorder=4)
        best = int(np.argmin(v) if lower_better else np.argmax(v))
        ax.scatter([k0[best]], [k1[best]], s=430, facecolor="none",
                   edgecolor="#C00000", linewidth=2.2, zorder=5)
        # Critical damping: zeta = 1  <=>  k1 = 2*sqrt(k0). Below = complex
        # poles (underdamped), above = real distinct poles (overdamped).
        grid = np.linspace(max(k0.min() * 0.6, 1e-3), k0.max() * 1.15, 400)
        ax.plot(grid, 2.0 * np.sqrt(grid), color="#C00000", linewidth=2.0,
                linestyle="--", zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$k_0 = \omega_n^2$")
        ax.set_ylabel(r"$k_1 = 2\zeta\omega_n$")
        ax.set_title(label, fontsize=11.0, fontweight="semibold")
        ax.grid(alpha=0.25, linewidth=0.8, which="both")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    handles = [
        Line2D([], [], color="#C00000", linestyle="--", linewidth=2.0),
        Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="#C00000",
               markersize=12, markeredgewidth=2.0, linestyle="none"),
    ]
    fig.legend(handles,
               [r"critical damping  $k_1 = 2\sqrt{k_0}$  ($\zeta=1$)", "best cell"],
               loc="upper center", bbox_to_anchor=(0.5, 0.95), frameon=False,
               ncols=2, fontsize=10)
    fig.suptitle(f"Gain plane -- {title_suffix}", fontsize=15.0,
                 fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.775, bottom=0.115, left=0.045, right=0.985, wspace=0.28)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


TABLE_COLUMNS: list[tuple[str, str, int]] = [
    ("omega_n", "omega_n", 3), ("zeta", "zeta", 3),
    ("k0", "k0", 2), ("k1", "k1", 2),
    ("collisions_per_km", "coll/km", 3), ("collisions_per_km_sd", "+/-", 3),
    ("qp_failure_rate", "QPfail", 3), ("intervention_rate", "interv", 3),
    ("correction_norm", "corr", 3), ("min_h", "min_h", 3),
    ("episode_length", "ep_len", 1), ("episode_return", "return", 1),
    ("completion_rate", "compl", 3), ("abs_speed_error", "spd_err", 2),
    ("lateral_error", "lat_err", 2), ("jerk_norm", "jerk", 2),
    ("feasible_fraction", "feas", 3), ("binding_fraction", "bind", 3),
]


def write_table(rows: list[dict], out_md: Path, out_csv: Path) -> str:
    ordered = sorted(rows, key=lambda r: (r["is_probe"], r["omega_n"], r["zeta"]))
    header = [label for _c, label, _d in TABLE_COLUMNS] + ["poles"]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    csv_rows = []
    for r in ordered:
        zeta = r["zeta"]
        poles = ("critically damped" if abs(zeta - 1.0) < 1e-9
                 else "underdamped (complex)" if zeta < 1.0
                 else "overdamped (real)")
        if r["is_probe"]:
            poles += " [probe]"
        cells = []
        for col, _label, dec in TABLE_COLUMNS:
            val = r.get(col)
            cells.append(f"{val:.{dec}f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(cells + [poles]) + " |")
        csv_rows.append({label: cell for (_c, label, _d), cell
                         in zip(TABLE_COLUMNS, cells)} | {"poles": poles})
    md = "\n".join(lines)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md + "\n", encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"wrote {out_md}")
    print(f"wrote {out_csv}")
    return md


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True,
                   help="summary.csv from the pole-placement sweep.")
    p.add_argument("--label", default="congested MTM, 60 vehicles",
                   help="Short description used in figure titles.")
    p.add_argument("--out", type=Path,
                   default=Path("artifacts") / "Results" / "CBF tuning")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_summary(args.summary)
    live = [r for r in rows if not r["is_probe"]]
    print(f"{len(rows)} cells ({len(live)} swept, {len(rows) - len(live)} probe)")
    plot_vs_zeta(live, args.out / "cbf_tuning_vs_damping.png", title_suffix=args.label)
    plot_gain_plane(rows, args.out / "cbf_tuning_gain_plane.png", title_suffix=args.label)
    md = write_table(rows, args.out / "cbf_tuning_complete_table.md",
                     args.out / "cbf_tuning_complete_table.csv")
    print()
    print(md)
    best = min(live, key=lambda r: r["collisions_per_km"])
    print(f"\nbest collisions/km: omega_n={best['omega_n']:g} zeta={best['zeta']:g} "
          f"(k0={best['k0']:.2f}, k1={best['k1']:.2f}) -> "
          f"{best['collisions_per_km']:.3f} +/- {best['collisions_per_km_sd']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
