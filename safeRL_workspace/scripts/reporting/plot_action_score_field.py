"""Freeze a state whose HOCBF safe set is a K-sided polygon, and draw the
actor's score field over the action box.

The laneless action space is the 2D box ``ax, ay in [-3, 3]`` m/s^2, and every
HOCBF row is affine in the action (``A_i . a <= b_i``, from
``CBFContextPhysicalActionWrapper._constraint_system``), so at a frozen state
the safe set is an exactly computable convex polygon: the action box clipped by
one half-plane per neighbour plus the two road-boundary rows. This script walks
the driven trajectory, computes that polygon at every step, and freezes the
first state whose polygon has exactly ``--sides`` edges -- preferring one whose
edges all come from CBF rows, so the polygon is strictly inscribed in the box
rather than leaning on it.

Over that polygon it draws the actor's **score function**

    g(a) = grad_a log pi(a | s) = (mu - a) / sigma^2

for the diagonal-Gaussian PPO policy: the direction PPO scales by the advantage
when it reinforces or discourages the action it actually sampled. It vanishes at
mu and grows linearly outward, skewed per axis by 1/sigma_i^2, so it is *not*
simply "toward mu" unless the two sigmas agree. Direction is drawn by the arrow,
magnitude by arrow colour -- a single-hue sequential ramp, since magnitude is a
one-ended quantity.

Nothing here retrains or re-evaluates: it drives the environment to reach a
state, reads the constraint system the wrapper itself builds, and queries the
loaded policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, Polygon

from scripts.rendering.render_policy_scenarios import (
    ScenarioSpec,
    VehicleSpec,
    find_repo_root,
    make_scenarios,
)
from scripts.rendering.render_ppo_scenario_gifs import build_namespace, load_policy
from scripts.reporting.plot_action_space_polytope import (
    CONGESTED,
    actor_readout,
    drive_and_freeze,
    feasible,
    replay_capture,
)
from scripts.reporting.plot_hocbf_safety_margin import ACTION_SOURCE, ARM_COLORS
from scripts.training.run_ppo_cbf_progression import load_model

EDGE_TOL = 1e-7
SAFE_FILL = "#70AD47"


def box_polygon(box: float) -> np.ndarray:
    return np.array([[-box, -box], [box, -box], [box, box], [-box, box]], dtype=float)


def clip_halfplane(poly: np.ndarray, row: np.ndarray, bound: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex polygon by ``row . a <= bound``."""

    if poly.size == 0:
        return poly
    slack = bound - poly @ row
    out: list[np.ndarray] = []
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        s_i, s_j = float(slack[i]), float(slack[j])
        if s_i >= 0.0:
            out.append(poly[i])
        if (s_i > 0.0) != (s_j > 0.0) and abs(s_i - s_j) > 1e-15:
            t = s_i / (s_i - s_j)
            out.append(poly[i] + t * (poly[j] - poly[i]))
    return np.asarray(out, dtype=float).reshape(-1, 2)


def dedupe(poly: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Drop vertices that repeat, including the wrap-around pair."""

    if len(poly) < 2:
        return poly
    keep = [poly[0]]
    for point in poly[1:]:
        if np.linalg.norm(point - keep[-1]) > tol:
            keep.append(point)
    if len(keep) > 1 and np.linalg.norm(keep[0] - keep[-1]) <= tol:
        keep.pop()
    return np.asarray(keep, dtype=float).reshape(-1, 2)


def safe_polygon(rows: np.ndarray, bounds: np.ndarray, box: float) -> np.ndarray:
    """The action box clipped by every HOCBF row: the exact safe set."""

    poly = box_polygon(box)
    for row, bound in zip(rows, bounds):
        poly = clip_halfplane(poly, np.asarray(row, dtype=float), float(bound))
        if poly.size == 0:
            return poly
    return dedupe(poly)


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def classify_edges(
    poly: np.ndarray, rows: np.ndarray, bounds: np.ndarray, box: float,
    *, min_edge: float,
) -> dict[str, Any]:
    """Which constraint owns each edge, and is any edge a wall of the box?

    An edge is attributed to whichever half-plane it is tight on at its own
    midpoint. Edges shorter than ``min_edge`` are numerical slivers from nearly
    parallel rows, not faces a reader would count, so they are dropped.
    """

    if len(poly) < 3:
        return {"edges": 0, "owners": [], "box_edges": 0, "cbf_edges": 0}
    box_rows = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    box_bounds = np.array([box, box, box, box], dtype=float)
    owners: list[str] = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        if float(np.linalg.norm(b - a)) < min_edge:
            continue
        mid = 0.5 * (a + b)
        cbf_slack = np.abs(np.asarray(bounds, dtype=float) - rows @ mid)
        box_slack = np.abs(box_bounds - box_rows @ mid)
        best_cbf = int(np.argmin(cbf_slack)) if len(cbf_slack) else -1
        if len(box_slack) and box_slack.min() <= min(
            cbf_slack.min() if len(cbf_slack) else np.inf, EDGE_TOL
        ):
            owners.append(f"box[{int(np.argmin(box_slack))}]")
        elif best_cbf >= 0 and cbf_slack[best_cbf] <= EDGE_TOL:
            owners.append(f"cbf[{best_cbf}]")
        else:
            owners.append("?")
    return {
        "edges": len(owners),
        "owners": owners,
        "box_edges": sum(1 for o in owners if o.startswith("box")),
        "cbf_edges": sum(1 for o in owners if o.startswith("cbf")),
    }


def score_field(mu: np.ndarray, sigma: np.ndarray, box: float, n: int):
    """grad_a log pi(a|s) on a grid: (mu - a) / sigma^2, per axis."""

    grid = np.linspace(-box, box, n)
    AX, AY = np.meshgrid(grid, grid)
    var = np.maximum(np.asarray(sigma, dtype=float), 1e-6) ** 2
    GX = (mu[0] - AX) / var[0]
    GY = (mu[1] - AY) / var[1]
    return AX, AY, GX, GY


CONSTRUCTED = "constructed_ring"


def ring_scenario(
    angles: np.ndarray, radii: np.ndarray, *, road_width: float, ego_speed: float,
    x_scale: float, y_scale: float,
) -> ScenarioSpec:
    """Ego ringed by one neighbour per angle, each at its own radius.

    The collision ellipse is twice as long as it is wide, so the ring is drawn
    in ellipse coordinates -- a neighbour at the same radius sits roughly
    ``x_scale / y_scale`` times further ahead than it does to the side.
    """

    center = 0.5 * road_width
    vehicles = [VehicleSpec("ego", 0.0, center, ego_speed, role="ego",
                            desired_speed=ego_speed)]
    for i, (angle, radius) in enumerate(zip(angles, radii)):
        dx = float(radius * x_scale * np.cos(angle))
        y = float(np.clip(center + radius * y_scale * np.sin(angle),
                          1.15, road_width - 1.15))
        vehicles.append(
            VehicleSpec(f"n{i}", dx, y, ego_speed, role="traffic",
                        desired_speed=ego_speed)
        )
    return ScenarioSpec(
        name=CONSTRUCTED,
        title=f"Constructed ring of {len(angles)} neighbours",
        expected="A deliberately many-faced HOCBF safe set.",
        vehicles=vehicles,
    )


def face_window(rows: np.ndarray, bounds: np.ndarray, box: float) -> np.ndarray:
    """Per-row signed slack at the action box's most favourable corner.

    ``A_i . a <= b_i`` cuts the box only while ``b_i`` sits between the row's
    extreme values over the box, ``+/- (|A_ix| + |A_iy|) * box``. Outside that
    window the row is either slack everywhere (no face) or excludes the box
    entirely (empty set). This returns ``b_i`` normalised into that window:
    **1 = the row just grazes the box's far corner** (the shallowest possible
    face), **0 = it has swallowed the box** (empty set), and only rows inside
    [0, 1] can contribute a face at all.

    Seven faces therefore need seven rows near the *top* of their windows, not
    the middle: seven half-planes each cutting halfway across the box have an
    empty intersection, however well each one is placed on its own.
    """

    rows = np.asarray(rows, dtype=float).reshape(-1, 2)
    reach = (np.abs(rows[:, 0]) + np.abs(rows[:, 1])) * box
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.asarray(bounds, dtype=float) + reach) / (2.0 * reach)


def pick_state(
    systems: Sequence[tuple[np.ndarray, np.ndarray]], *, sides: int, box: float,
    min_area_frac: float, min_edge: float,
) -> tuple[int | None, list[dict[str, Any]]]:
    """First step whose safe set has exactly ``sides`` edges.

    A polygon whose edges are all CBF faces is preferred over one that borrows a
    wall of the action box: the first is a safe set the constraints alone carve
    out, the second is partly just the box redrawn.
    """

    box_area = (2.0 * box) ** 2
    report: list[dict[str, Any]] = []
    for index, (rows, bounds) in enumerate(systems):
        poly = safe_polygon(rows, bounds, box)
        area = polygon_area(poly)
        info = classify_edges(poly, rows, bounds, box, min_edge=min_edge)
        report.append({
            "step": index + 1, "edges": info["edges"], "box_edges": info["box_edges"],
            "cbf_edges": info["cbf_edges"], "area_frac": area / box_area,
            "n_rows": int(np.asarray(rows).reshape(-1, 2).shape[0]),
        })
    eligible = [r for r in report
                if r["edges"] == sides and r["area_frac"] >= min_area_frac]
    interior = [r for r in eligible if r["box_edges"] == 0]
    chosen = (interior or eligible)
    if not chosen:
        return None, report
    # Largest safe set among the qualifying states: the faces stay far enough
    # apart to be told apart by eye.
    best = max(chosen, key=lambda r: r["area_frac"])
    return int(best["step"]), report


def draw(
    ax, poly: np.ndarray, rows: np.ndarray, bounds: np.ndarray,
    readout: dict[str, Any], *, box: float, view: float, n_arrows: int,
    label: str, owners: Sequence[str],
) -> None:
    mu = np.asarray(readout["mu_raw"], dtype=float)
    sigma = np.asarray(readout["sigma"], dtype=float)
    arm_color = ARM_COLORS.get(label, "#111111")

    AX, AY, GX, GY = score_field(mu, sigma, box, n_arrows)
    magnitude = np.hypot(GX, GY)
    # Direction is the arrow, magnitude is the colour: the field spans orders of
    # magnitude between mu and the corners, so true-length arrows would be a
    # blank centre and an unreadable rim.
    with np.errstate(invalid="ignore", divide="ignore"):
        UX, UY = GX / magnitude, GY / magnitude
    quiver = ax.quiver(
        AX, AY, UX, UY, magnitude, cmap="Blues", pivot="mid",
        scale=n_arrows * 1.25, width=0.004, zorder=5,
        clim=(0.0, float(np.nanmax(magnitude))),
    )

    if len(poly) >= 3:
        ax.add_patch(Polygon(poly, closed=True, facecolor=SAFE_FILL, alpha=0.20,
                             edgecolor=SAFE_FILL, linewidth=2.2, zorder=3))
    ax.add_patch(plt.Rectangle((-box, -box), 2 * box, 2 * box, fill=False,
                               edgecolor="#111111", linewidth=1.4,
                               linestyle=(0, (5, 3)), zorder=4))
    for k, alpha in ((1.0, 0.55), (2.0, 0.30)):
        ax.add_patch(Ellipse(tuple(mu), 2 * k * sigma[0], 2 * k * sigma[1],
                             facecolor="none", edgecolor=arm_color,
                             linewidth=1.4, alpha=alpha, zorder=6))
    ax.plot(*mu, marker="o", markersize=10, color=arm_color,
            markeredgecolor="white", markeredgewidth=1.4, zorder=7)

    ax.set_xlim(-view, view)
    ax.set_ylim(-view, view)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$a_x$  (m/s$^2$)")
    ax.set_ylabel(r"$a_y$  (m/s$^2$)")
    ax.grid(alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Patch(facecolor=SAFE_FILL, alpha=0.20, edgecolor=SAFE_FILL, linewidth=2.0,
              label=f"HOCBF safe set ({len(owners)} faces)"),
        Line2D([], [], color="#111111", linestyle=(0, (5, 3)), linewidth=1.4,
               label=r"action box $[-3,3]^2$"),
        Line2D([], [], color=arm_color, marker="o", linestyle="none", markersize=9,
               markeredgecolor="white", label=r"$\mu$ (actor mean)"),
        Line2D([], [], color=arm_color, linewidth=1.4, alpha=0.5,
               label=r"1$\sigma$ / 2$\sigma$"),
    ]
    # The field fills the whole box, so the legend needs its own ground to sit on.
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.92,
              edgecolor="none", fontsize=8.5)
    return quiver


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", required=True, metavar="LABEL=RUN_CONFIG_JSON",
                   help="One arm, e.g. B2_cbf_nd_reward_actor=<run_config.json>.")
    p.add_argument("--scenario", default=CONGESTED)
    p.add_argument("--construct", action="store_true",
                   help="Instead of searching a driven trajectory, place a ring "
                        "of neighbours around the ego and search placements "
                        "until the safe set has --sides faces.")
    p.add_argument("--construct-trials", type=int, default=400,
                   help="Placements tried in --construct mode. Default %(default)s.")
    p.add_argument("--construct-seed", type=int, default=0,
                   help="RNG seed for the placement search. Default %(default)s.")
    p.add_argument("--face-depth", type=float, default=0.14,
                   help="How deep each constructed face cuts into the action "
                        "box, as a fraction of the row's full span over it. "
                        "Small keeps the safe set large and every face "
                        "non-redundant; large empties it. Default %(default)s.")
    p.add_argument("--neighbors", type=int, default=7,
                   help="Neighbours placed around the ego in --construct mode. "
                        "Capped by the env's neighbors_count. Default %(default)s.")
    p.add_argument("--sides", type=int, default=7,
                   help="Edge count the frozen safe set must have. Default %(default)s.")
    p.add_argument("--horizon", type=int, default=300,
                   help="Policy steps searched for such a state. Default %(default)s.")
    p.add_argument("--min-area-fraction", type=float, default=0.02,
                   help="Smallest safe-set share of the action box that still "
                        "counts. Default %(default)s.")
    p.add_argument("--min-edge", type=float, default=0.02,
                   help="Edges shorter than this (m/s^2) are slivers, not faces. "
                        "Default %(default)s.")
    p.add_argument("--arrows", type=int, default=21,
                   help="Score-field arrows per axis. Default %(default)s.")
    p.add_argument("--seed", type=int, default=1100011)
    p.add_argument("--box", type=float, default=3.0, help="Action box half-width.")
    p.add_argument("--view-scale", type=float, default=1.08)
    p.add_argument("--ease-gap-scale", type=float, default=1.0)
    p.add_argument("--ease-closing-speed-scale", type=float, default=1.0)
    p.add_argument("--ease-boundary-scale", type=float, default=1.0)
    p.add_argument("--out", type=Path,
                   default=Path("artifacts") / "Results" / "Action space")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = find_repo_root()
    scenarios_by_name = {s.name: s for s in make_scenarios(10.2)}
    label, path = args.policy.split("=", 1)
    if label not in ACTION_SOURCE:
        raise SystemExit(f"Unknown policy label {label!r}")

    policy = load_policy(Path(path).resolve())
    policy["namespace"] = build_namespace(root, policy["cbf"])
    policy["model"] = load_model(policy["variant"], policy["model_path"], "cpu")
    box = float(args.box)
    bounds_cfg = policy["env_config"].get("bounds", {})
    print(f"[score-field] {label}: action box "
          f"ax [{bounds_cfg.get('ax_min')}, {bounds_cfg.get('ax_max')}] "
          f"ay [{bounds_cfg.get('ay_min')}, {bounds_cfg.get('ay_max')}] m/s^2")
    ease = (args.ease_gap_scale, args.ease_closing_speed_scale,
            args.ease_boundary_scale)

    if args.construct:
        road_width = float(policy["env_config"]["road_width"])
        ego_speed = float(policy["env_config"].get("ego_desired_speed", 16.0))
        n_neighbors = min(int(args.neighbors),
                          int(policy["env_config"].get("neighbors_count", 7)))
        geometry = policy["env_config"].get("cbf_geometry", {})
        x_scale = float(geometry.get("relative_ellipse_a_m", 5.09))
        y_scale = float(geometry.get("relative_ellipse_b_m", 2.55))
        # Random placement does not find this state: a row cuts the action box
        # only while its bound sits inside a narrow window (see face_window),
        # and every row has to be in that window at once. So drive each
        # neighbour's radius toward its own window instead of sampling. Rows
        # follow the neighbour ordering by distance, so the radii are kept
        # sorted and matched to the rows in the same order.
        angles = np.linspace(0.0, 2.0 * np.pi, n_neighbors, endpoint=False)
        angles = angles + float(args.construct_seed) * 0.11
        radii = np.full(n_neighbors, 2.2, dtype=float)
        target = 1.0 - float(args.face_depth)
        counts: dict[int, int] = {}
        state = None
        step = 0
        for trial in range(int(args.construct_trials)):
            order = np.argsort(radii)
            scenario = ring_scenario(
                angles[order], radii[order], road_width=road_width,
                ego_speed=ego_speed, x_scale=x_scale, y_scale=y_scale,
            )
            frozen = drive_and_freeze(
                policy["namespace"], policy, ACTION_SOURCE[label], CONSTRUCTED,
                steps=0, seed=int(args.seed),
                scenarios_by_name={CONSTRUCTED: scenario}, ease=ease,
                replay=None, render=False, record_systems=False,
            )
            rows_t = np.asarray(frozen["rows"], dtype=float).reshape(-1, 2)
            bounds_t = np.asarray(frozen["bounds"], dtype=float).reshape(-1)
            poly_t = safe_polygon(rows_t, bounds_t, box)
            info_t = classify_edges(poly_t, rows_t, bounds_t, box,
                                    min_edge=float(args.min_edge))
            counts[info_t["edges"]] = counts.get(info_t["edges"], 0) + 1
            area_frac = polygon_area(poly_t) / (2 * box) ** 2
            window = face_window(rows_t, bounds_t, box)[:n_neighbors]
            print(f"[score-field] trial {trial}: {info_t['edges']} faces "
                  f"({info_t['cbf_edges']} CBF, {info_t['box_edges']} box), "
                  f"area {area_frac:6.1%}, in-window rows "
                  f"{int(np.sum((window >= 0.0) & (window <= 1.0)))}/{n_neighbors}",
                  flush=True)
            if (info_t["edges"] == int(args.sides)
                    and area_frac >= float(args.min_area_fraction)):
                state = frozen
                break
            # A row above its window is slack (push the neighbour closer, which
            # lowers the bound); below it the row has swallowed the box (push
            # that neighbour away). One multiplicative step per neighbour.
            for slot in range(min(n_neighbors, len(window))):
                index = int(order[slot])
                w = float(window[slot])
                if not np.isfinite(w):
                    continue
                gap = w - target
                radii[index] = float(np.clip(
                    radii[index] * float(np.exp(-0.45 * np.tanh(gap))),
                    1.05, 6.0,
                ))
        print(f"[score-field] face counts over {sum(counts.values())} placements: "
              + ", ".join(f"{k}-gon x{v}" for k, v in sorted(counts.items())))
        if state is None:
            raise SystemExit(
                f"no placement in {args.construct_trials} steps gives a "
                f"{args.sides}-sided safe set with area >= "
                f"{args.min_area_fraction:.0%} of the box; try more trials, a "
                f"different --construct-seed, or --min-area-fraction"
            )
    else:
        scout = drive_and_freeze(
            policy["namespace"], policy, ACTION_SOURCE[label], args.scenario,
            steps=int(args.horizon), seed=int(args.seed),
            scenarios_by_name=scenarios_by_name, ease=ease, replay=None,
            render=False, record_systems=True,
        )
        step, report = pick_state(
            scout["systems"], sides=int(args.sides), box=box,
            min_area_frac=float(args.min_area_fraction),
            min_edge=float(args.min_edge),
        )
        counts = {}
        for r in report:
            counts[r["edges"]] = counts.get(r["edges"], 0) + 1
        print(f"[score-field] edge counts over {len(report)} driven steps: "
              + ", ".join(f"{k}-gon x{v}" for k, v in sorted(counts.items())))
        if step is None:
            raise SystemExit(
                f"no driven step gives a {args.sides}-sided safe set with area >= "
                f"{args.min_area_fraction:.0%} of the box; try --horizon, "
                f"--min-area-fraction, --construct, or a different --seed"
            )

        captured = replay_capture(
            policy["namespace"], policy, args.scenario, replay=scout["executed"],
            capture_steps=[step], seed=int(args.seed),
            scenarios_by_name=scenarios_by_name, ease=ease, render=False,
        )
        if step not in captured:
            raise SystemExit(f"replay did not reach step {step}")
        state = captured[step]
    rows = np.asarray(state["rows"], dtype=float).reshape(-1, 2)
    row_bounds = np.asarray(state["bounds"], dtype=float).reshape(-1)
    poly = safe_polygon(rows, row_bounds, box)
    info = classify_edges(poly, rows, row_bounds, box, min_edge=float(args.min_edge))
    if info["edges"] != int(args.sides):
        raise SystemExit(
            f"the frozen state gives a {info['edges']}-sided set, not "
            f"{args.sides}: the search and the replayed state disagree"
        )
    readout = actor_readout(policy, state["obs"])
    mu = np.asarray(readout["mu_raw"], dtype=float)
    sigma = np.asarray(readout["sigma"], dtype=float)
    inside = bool(feasible(rows, row_bounds, mu.reshape(1, 2))[0])
    where = ("constructed placement" if args.construct
             else f"step {step} (t = {step / 20:.2f} s)")
    print(f"[score-field] {where}: "
          f"{info['edges']} faces ({info['cbf_edges']} CBF, {info['box_edges']} box), "
          f"area {polygon_area(poly) / (2 * box) ** 2:.1%} of the box, "
          f"{rows.shape[0]} rows in the system")
    print(f"[score-field] mu = ({mu[0]:+.3f}, {mu[1]:+.3f}) m/s^2, "
          f"sigma = ({sigma[0]:.3f}, {sigma[1]:.3f}), mu inside safe set: {inside}")

    view = box * float(args.view_scale)
    fig, ax = plt.subplots(figsize=(8.4, 7.6))
    quiver = draw(ax, poly, rows, row_bounds, readout, box=box, view=view,
                  n_arrows=int(args.arrows), label=label, owners=info["owners"])
    bar = fig.colorbar(quiver, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label(r"$\|\nabla_a \log \pi(a|s)\|$", fontsize=9.5)
    bar.outline.set_visible(False)
    scene = CONSTRUCTED if args.construct else args.scenario
    ax.set_title(
        f"Actor score field over the {info['edges']}-sided HOCBF safe set\n"
        f"{label} - seed {policy['training_seed']}, {scene}, {where}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    stem = (f"score_field_{label}_{info['edges']}gon_"
            + ("constructed" if args.construct else f"step{step}"))
    fig.savefig(args.out / f"{stem}.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out / f'{stem}.png'}")

    provenance = {
        "policy": label,
        "run_config": str(Path(path).resolve()),
        "training_seed": policy["training_seed"],
        "scenario": args.scenario,
        "drive_seed": int(args.seed),
        "step": step,
        "action_box": {"ax": [-box, box], "ay": [-box, box]},
        "constraint_rows": rows.tolist(),
        "constraint_bounds": row_bounds.tolist(),
        "system_hash": state.get("hash"),
        "polygon_vertices": poly.tolist(),
        "edge_owners": info["owners"],
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "mu_inside_safe_set": inside,
    }
    (args.out / f"{stem}.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out / f'{stem}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
