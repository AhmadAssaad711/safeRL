"""Does the congested-environment CBF tuning actually behave like the
second-order damped system its gains describe?

Consumes ``summary.csv`` and ``traces/*.csv`` from
``scripts/evaluation/cbf_congested_tuning_sweep.py``.

The placed poles are the roots of ``s^2 + k1*s + k0`` with ``k0 = omega_n^2``
and ``k1 = 2*zeta*omega_n``. A binding HOCBF row forces
``psi_2 = b - A@u = 0``, i.e. ``hddot + k1*h_dot + k0*h = 0``, so on binding
steps the placed poles are correct by construction. What is *not* guaranteed is
that the realised trajectory expresses them: the binding neighbour switches,
the barrier is forced by other vehicles' accelerations, the actuator saturates,
and the filter is applied discretely at 20 Hz.

So the measured poles here are obtained by system identification that never
uses k0 or k1. Inside each window where one neighbour barrier is continuously
the binding constraint, with a feasible QP and no saturation, this fits

    hddot = -a0*h - a1*h_dot

by least squares, taking hddot from finite differences of the analytically
logged h_dot. The measured poles are the roots of ``s^2 + a1*s + a0``. If the
identified (a0, a1) reproduce the placed (k0, k1), the second-order
description survives discretisation and switching; where they diverge, it does
not, and the divergence is the result.

Figures:

  1  pole map in the s-plane, constant-zeta rays and constant-omega_n arcs
  2  identified vs placed poles, the actual test
  3  KPI surfaces over (omega_n, zeta) with the zeta=1 critical line
  4  h(t) in binding windows against the predicted envelope
  5  domain of validity: how much of an episode the model can govern at all
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DT = 0.05  # policy/CBF period, 20 Hz


def identify_windows(
    trace: pd.DataFrame, *, min_window: int
) -> list[pd.DataFrame]:
    """Contiguous stretches where one barrier is continuously the binding one."""

    if trace.empty:
        return []
    usable = (
        trace["cbf_qp_success"].to_numpy(dtype=bool)
        & trace.get(
            "cbf_min_is_active", pd.Series(False, index=trace.index)
        ).to_numpy(dtype=bool)
        & (trace["actuator_clip_norm"].to_numpy(dtype=float) <= 1e-9)
    )
    index = trace["cbf_min_h_index"].to_numpy(dtype=float)
    episode = trace["episode_seed"].to_numpy(dtype=float)
    step = trace["step"].to_numpy(dtype=float)
    # A window breaks when the barrier stops binding, the binding pair changes,
    # the episode changes, or a step is skipped.
    new_window = np.ones(len(trace), dtype=bool)
    new_window[1:] = (
        (index[1:] != index[:-1])
        | (episode[1:] != episode[:-1])
        | (step[1:] != step[:-1] + 1)
    )
    windows: list[pd.DataFrame] = []
    start = None
    for position in range(len(trace)):
        if not usable[position]:
            if start is not None and position - start >= min_window:
                windows.append(trace.iloc[start:position])
            start = None
            continue
        if start is None or new_window[position]:
            if start is not None and position - start >= min_window:
                windows.append(trace.iloc[start:position])
            start = position
    if start is not None and len(trace) - start >= min_window:
        windows.append(trace.iloc[start:])
    return windows


def fit_second_order(windows: Sequence[pd.DataFrame]) -> dict[str, float]:
    """Least-squares (a0, a1) in hddot = -a0*h - a1*h_dot, pooled over windows.

    k0 and k1 are deliberately not used anywhere in this fit.
    """

    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for window in windows:
        h = window["cbf_min_h"].to_numpy(dtype=float)
        h_dot = window["cbf_min_h_dot"].to_numpy(dtype=float)
        if h.size < 3 or not np.all(np.isfinite(h)) or not np.all(np.isfinite(h_dot)):
            continue
        # Central differences, so the interior points only.
        hddot = (h_dot[2:] - h_dot[:-2]) / (2.0 * DT)
        rows.append(np.column_stack([h[1:-1], h_dot[1:-1]]))
        targets.append(hddot)
    if not rows:
        return {
            "a0": float("nan"),
            "a1": float("nan"),
            "n_windows": 0.0,
            "n_samples": 0.0,
            "r2": float("nan"),
        }
    design = np.vstack(rows)
    target = np.concatenate(targets)
    # hddot = -a0*h - a1*h_dot  =>  target = design @ [-a0, -a1]
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ solution
    residual = target - predicted
    denominator = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else float("nan")
    return {
        "a0": float(-solution[0]),
        "a1": float(-solution[1]),
        "n_windows": float(len(rows)),
        "n_samples": float(design.shape[0]),
        "r2": float(r2),
    }


def poles_from_coefficients(a0: float, a1: float) -> tuple[complex, complex]:
    if not np.isfinite(a0) or not np.isfinite(a1):
        return complex("nan"), complex("nan")
    disc = complex(a1 * a1 / 4.0 - a0, 0.0) ** 0.5
    return (-a1 / 2.0 + disc, -a1 / 2.0 - disc)


def plot_pole_map(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    finite = summary[np.isfinite(summary.get("collisions_per_km", np.nan))]
    for _, row in summary.iterrows():
        omega_n, zeta = float(row["omega_n"]), float(row["zeta"])
        disc = complex(zeta**2 - 1.0, 0.0) ** 0.5
        for pole in (-zeta * omega_n + omega_n * disc, -zeta * omega_n - omega_n * disc):
            colour = row.get("collisions_per_km", np.nan)
            marker = "^" if bool(row.get("is_probe", False)) else "o"
            scatter = ax.scatter(
                pole.real, pole.imag, c=[colour], cmap="viridis",
                vmin=float(finite["collisions_per_km"].min()) if not finite.empty else 0.0,
                vmax=float(finite["collisions_per_km"].max()) if not finite.empty else 1.0,
                s=70, marker=marker, edgecolor="#222222", linewidth=0.5, zorder=3,
            )
    for zeta_ray in (0.5, 0.7, 1.0):
        if zeta_ray >= 1.0:
            continue
        angle = np.arccos(zeta_ray)
        radius = float(summary["omega_n"].max()) * 1.1
        ax.plot(
            [0, -radius * zeta_ray], [0, radius * np.sin(angle)],
            color="#888888", linestyle=":", linewidth=1.0, zorder=1,
        )
        ax.plot(
            [0, -radius * zeta_ray], [0, -radius * np.sin(angle)],
            color="#888888", linestyle=":", linewidth=1.0, zorder=1,
        )
        ax.annotate(f"ζ={zeta_ray}", (-radius * zeta_ray * 0.95, radius * np.sin(angle) * 0.95),
                    fontsize=8, color="#666666")
    for omega_arc in sorted(set(summary["omega_n"].tolist())):
        angles = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
        ax.plot(omega_arc * np.cos(angles), omega_arc * np.sin(angles),
                color="#cccccc", linewidth=0.8, zorder=0)
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.axvline(0.0, color="#333333", linewidth=0.9)
    ax.set_xlabel("Re(s)   [1/s]")
    ax.set_ylabel("Im(s)   [1/s]")
    ax.set_title(
        "Placed CBF poles, s² + k₁s + k₀, coloured by collisions/km\n"
        "triangles are deliberate out-of-validity probes (ωₙΔt ≫ 0.2)",
        fontsize=11, fontweight="bold",
    )
    fig.colorbar(scatter, ax=ax, label="Ego collisions / km")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_identified_vs_placed(identified: pd.DataFrame, out_path: Path) -> None:
    usable = identified[identified["n_samples"] > 0]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    pairs = [
        ("k0", "a0", "k₀  (=ωₙ²)", "identified a₀"),
        ("k1", "a1", "k₁  (=2ζωₙ)", "identified a₁"),
    ]
    for axis, (placed_col, ident_col, xlabel, ylabel) in zip(axes, pairs):
        if usable.empty:
            axis.text(0.5, 0.5, "no binding windows found", ha="center", va="center")
            continue
        colour = usable["omega_n_dt"].to_numpy(dtype=float)
        scatter = axis.scatter(
            usable[placed_col], usable[ident_col], c=colour, cmap="plasma",
            s=64, edgecolor="#222222", linewidth=0.5, zorder=3,
        )
        limit = [
            float(min(usable[placed_col].min(), usable[ident_col].min())),
            float(max(usable[placed_col].max(), usable[ident_col].max())),
        ]
        axis.plot(limit, limit, color="#cc3311", linestyle="--", linewidth=1.4,
                  label="identified = placed", zorder=2)
        axis.set_xlabel(f"placed {xlabel}")
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False, fontsize=9)
        axis.grid(alpha=0.25)
        fig.colorbar(scatter, ax=axis, label="ωₙ·Δt")
    axis = axes[2]
    if not usable.empty:
        axis.scatter(usable["omega_n_dt"], usable["r2"], s=64, color="#4472C4",
                     edgecolor="#222222", linewidth=0.5)
        axis.axvline(0.2, color="#cc3311", linestyle="--", linewidth=1.3)
        axis.annotate("ωₙΔt = 0.2\ncontinuous-time\nvalidity limit", (0.21, 0.15),
                      fontsize=8, color="#cc3311")
    axis.set_xlabel("ωₙ·Δt")
    axis.set_ylabel("fit R² of hddot = -a₀h - a₁ḣ")
    axis.grid(alpha=0.25)
    fig.suptitle(
        "P1/P4: poles identified from the realised barrier trajectory, "
        "without using k₀ or k₁ in the fit",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=190, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_kpi_surfaces(summary: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("collisions_per_km", "Ego collisions / km"),
        ("qp_failure_rate", "QP failure rate"),
        ("intervention_rate", "Intervention rate"),
        ("correction_norm", "Correction norm"),
        ("episode_length", "Episode length (steps)"),
        ("governed_fraction", "Governed fraction (P3)"),
    ]
    grid = summary[~summary["is_probe"].astype(bool)] if "is_probe" in summary else summary
    omegas = sorted(set(grid["omega_n"].tolist()))
    zetas = sorted(set(grid["zeta"].tolist()))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for axis, (column, title) in zip(axes.ravel(), metrics):
        if column not in grid.columns:
            axis.set_visible(False)
            continue
        matrix = np.full((len(zetas), len(omegas)), np.nan)
        for _, row in grid.iterrows():
            i = zetas.index(float(row["zeta"]))
            j = omegas.index(float(row["omega_n"]))
            matrix[i, j] = float(row[column])
        image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(omegas)))
        axis.set_xticklabels([f"{value:g}" for value in omegas])
        axis.set_yticks(range(len(zetas)))
        axis.set_yticklabels([f"{value:g}" for value in zetas])
        axis.set_xlabel("ωₙ  [rad/s]")
        axis.set_ylabel("ζ")
        axis.set_title(title, fontsize=10)
        if 1.0 in zetas:
            axis.axhline(zetas.index(1.0), color="#cc3311", linestyle="--", linewidth=1.5)
        fig.colorbar(image, ax=axis)
    fig.suptitle(
        "P2: KPI surfaces over the damping parameterisation "
        "(red line = critical damping ζ=1)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=185, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_binding_windows(
    traces: dict[tuple[float, float], pd.DataFrame],
    picks: Sequence[tuple[float, float]],
    out_path: Path,
    *,
    min_window: int,
) -> None:
    fig, axes = plt.subplots(1, len(picks), figsize=(6.0 * len(picks), 5.0))
    if len(picks) == 1:
        axes = [axes]
    for axis, key in zip(axes, picks):
        trace = traces.get(key)
        omega_n, zeta = key
        if trace is None or trace.empty:
            axis.text(0.5, 0.5, "no trace", ha="center", va="center")
            continue
        windows = identify_windows(trace, min_window=min_window)
        windows = sorted(windows, key=len, reverse=True)[:6]
        for window in windows:
            h = window["cbf_min_h"].to_numpy(dtype=float)
            t = np.arange(h.size) * DT
            axis.plot(t, h, linewidth=1.5, alpha=0.85)
        if windows:
            longest = max(len(w) for w in windows)
            t = np.arange(longest) * DT
            h0 = float(windows[0]["cbf_min_h"].to_numpy(dtype=float)[0])
            axis.plot(
                t, h0 * np.exp(-zeta * omega_n * t), color="#cc3311",
                linestyle="--", linewidth=1.8, label=r"$e^{-\zeta\omega_n t}$ envelope",
            )
            axis.legend(frameon=False, fontsize=9)
        axis.axhline(0.0, color="#333333", linewidth=1.0, linestyle=":")
        regime = "underdamped" if zeta < 1 else ("critical" if zeta == 1 else "overdamped")
        axis.set_title(
            f"ωₙ={omega_n:g}, ζ={zeta:g}  ({regime})\n"
            f"{len(windows)} binding windows ≥ {min_window} steps",
            fontsize=10,
        )
        axis.set_xlabel("time in binding window (s)")
        axis.set_ylabel("h")
        axis.grid(alpha=0.25)
    fig.suptitle(
        "P1: barrier trajectories while one pair is continuously the binding constraint",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=185, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_validity(summary: pd.DataFrame, out_path: Path) -> None:
    grid = summary.sort_values(["omega_n", "zeta"]).reset_index(drop=True)
    labels = [f"{row.omega_n:g}/{row.zeta:g}" for row in grid.itertuples()]
    fig, axis = plt.subplots(figsize=(max(11.0, 0.42 * len(labels)), 6.0))
    x = np.arange(len(grid))
    for column, colour, label in (
        ("feasible_fraction", "#4472C4", "QP feasible"),
        ("active_fraction", "#ED7D31", "min-h barrier binding"),
        ("governed_fraction", "#70AD47", "governed (feasible ∧ binding ∧ unsaturated)"),
        ("saturated_fraction", "#C00000", "actuator saturated"),
    ):
        if column in grid.columns:
            axis.plot(x, grid[column], marker="o", markersize=4.5, linewidth=1.6,
                      color=colour, label=label)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=90, fontsize=7)
    axis.set_xlabel("ωₙ / ζ")
    axis.set_ylabel("fraction of traced steps")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=9)
    axis.set_title(
        "P3: how much of an episode the second-order description can govern at all",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=185, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--min-window", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = pd.read_csv(args.sweep_root / "summary.csv")
    out_dir = args.out or (args.sweep_root / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    traces: dict[tuple[float, float], pd.DataFrame] = {}
    for path in sorted((args.sweep_root / "traces").glob("*.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        key = (float(frame["omega_n"].iloc[0]), float(frame["zeta"].iloc[0]))
        traces[key] = frame

    records: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        key = (float(row["omega_n"]), float(row["zeta"]))
        trace = traces.get(key)
        fit = (
            fit_second_order(identify_windows(trace, min_window=int(args.min_window)))
            if trace is not None
            else fit_second_order([])
        )
        placed_a, placed_b = (
            -row["zeta"] * row["omega_n"]
            + row["omega_n"] * complex(row["zeta"] ** 2 - 1.0, 0.0) ** 0.5,
            -row["zeta"] * row["omega_n"]
            - row["omega_n"] * complex(row["zeta"] ** 2 - 1.0, 0.0) ** 0.5,
        )
        identified_a, identified_b = poles_from_coefficients(fit["a0"], fit["a1"])
        records.append(
            {
                "omega_n": float(row["omega_n"]),
                "zeta": float(row["zeta"]),
                "k0": float(row["k0"]),
                "k1": float(row["k1"]),
                "omega_n_dt": float(row.get("omega_n_dt", row["omega_n"] * DT)),
                "is_probe": bool(row.get("is_probe", False)),
                **fit,
                "placed_pole_dominant": float(max(placed_a.real, placed_b.real)),
                "identified_pole_dominant": float(
                    max(identified_a.real, identified_b.real)
                )
                if np.isfinite(identified_a.real)
                else float("nan"),
            }
        )
    identified = pd.DataFrame(records)
    identified.to_csv(out_dir / "identified_poles.csv", index=False)
    print(f"wrote {out_dir / 'identified_poles.csv'}")

    plot_pole_map(summary, out_dir / "fig1_pole_map.png")
    plot_identified_vs_placed(identified, out_dir / "fig2_identified_vs_placed.png")
    plot_kpi_surfaces(summary, out_dir / "fig3_kpi_surfaces.png")

    available = sorted(traces)
    picks = [
        key
        for key in (
            next((k for k in available if k[1] < 1.0), None),
            next((k for k in available if k[1] == 1.0), None),
            next((k for k in available if k[1] > 1.5), None),
        )
        if key is not None
    ]
    if picks:
        plot_binding_windows(
            traces, picks, out_dir / "fig4_binding_windows.png",
            min_window=int(args.min_window),
        )
    plot_validity(summary, out_dir / "fig5_validity.png")

    usable = identified[identified["n_samples"] > 0]
    print("\n=== identification summary ===")
    if usable.empty:
        print("No binding windows of the requested length were found.")
    else:
        print(
            usable[
                ["omega_n", "zeta", "k0", "a0", "k1", "a1", "r2", "n_windows", "n_samples"]
            ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
