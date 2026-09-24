"""Freeze one state and draw every policy's action against the HOCBF polytope.

The laneless action space is 2D (ax, ay in [-3, 3] m/s^2), and every HOCBF row
is affine in the action:

    A_i . a <= b_i          (rows/bounds from CBFContextPhysicalActionWrapper)

so at a frozen state the safe action set is an exactly computable polytope and

    S_HOCBF(a) = min_i ( b_i - A_i . a )

is concave piecewise-linear over the plane. The background field and its zero
contour are therefore exact, not sampled: the contour *is* the safe-set
boundary.

Every arm is compared at the *same physical state*. A reference policy drives
the environment and its executed physical actions are recorded; each arm then
replays that identical action sequence in its own environment (deterministic
given seed + actions), so all arms reach the same state while still receiving
their own correctly-shaped observation. The constraint system is verified
identical across arms before anything is plotted.

Per arm the panel shows:
  - mu_raw       the actor mean, the network's own proposal
  - mu_safe      after the internal differentiable projection (C arms only;
                 identical to mu_raw for A/B arms)
  - 1 and 2 sigma ellipses of the actor's Gaussian
  - P(safe)      Monte-Carlo probability that a sampled action is feasible
  - d(mu_raw)    distance from the actor mean to the polytope, 0 if inside
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch

from scripts.common.ppo_cbf_env import CBFContextPhysicalActionWrapper
from scripts.rendering.render_policy_scenarios import find_repo_root, make_scenarios
from scripts.rendering.render_ppo_scenario_gifs import (
    build_namespace,
    load_policy,
    reposition_vehicles,
    scenario_env_config,
)
from scripts.reporting.plot_hocbf_safety_margin import ACTION_SOURCE, ARM_COLORS
from scripts.training.run_ppo_cbf_progression import (
    _predict_evaluation_action,
    load_model,
    make_ppo_cbf_env,
)

CONGESTED = "congested"


def find_wrapper(env: Any) -> CBFContextPhysicalActionWrapper | None:
    node = env
    while node is not None:
        if isinstance(node, CBFContextPhysicalActionWrapper):
            return node
        node = getattr(node, "env", None)
    return None


def build_env(
    namespace: dict[str, Any],
    policy: dict[str, Any],
    scenario_name: str,
    steps: int,
    scenarios_by_name: dict[str, Any],
) -> tuple[Any, Any]:
    if scenario_name == CONGESTED:
        cfg = copy.deepcopy(policy["env_config"])
        sub = max(1, round(float(cfg["simulation_frequency"])
                           / float(cfg["policy_frequency"])))
        cfg["episode_steps"] = (int(steps) + 21) * sub
        cfg["duration"] = cfg["episode_steps"]
        scenario = None
    else:
        scenario = scenarios_by_name[scenario_name]
        cfg = scenario_env_config(policy["env_config"], scenario, steps)
    cfg["terminate_on_collision"] = True
    env = make_ppo_cbf_env(
        namespace, env_config=cfg, reward_config=policy["reward_config"],
        project_inputs=False, lambda_delta=0.0, lambda_intervention=0.0,
        correction_epsilon=policy["correction_epsilon"], monitor_path=None,
    )
    return env, scenario


def drive_and_freeze(
    namespace: dict[str, Any],
    policy: dict[str, Any],
    action_source: str,
    scenario_name: str,
    *,
    steps: int,
    seed: int,
    scenarios_by_name: dict[str, Any],
    ease: tuple[float, float, float],
    replay: np.ndarray | None,
    render: bool,
    record_systems: bool = False,
) -> dict[str, Any]:
    """Advance to `steps`, then return the frozen observation and constraints.

    With `replay` given, the recorded physical actions are executed instead of
    the policy's own, so a different arm lands on the identical state.
    """

    env, scenario = build_env(namespace, policy, scenario_name, steps,
                              scenarios_by_name)
    if render:
        env.unwrapped.render_mode = "rgb_array"
        env.unwrapped.config["offscreen_rendering"] = True
        env.unwrapped.config["real_time_rendering"] = False
    wrapper = find_wrapper(env)
    if wrapper is None:
        env.close()
        raise RuntimeError("CBFContextPhysicalActionWrapper not found in env stack")
    executed: list[np.ndarray] = []
    margins: list[float] = []
    systems: list[tuple[np.ndarray, np.ndarray]] = []
    frame = None
    try:
        obs, _info = env.reset(seed=int(seed))
        terminated = truncated = False
        if scenario is not None:
            reposition_vehicles(env, scenario, ease_gap_scale=ease[0],
                                ease_closing_speed_scale=ease[1],
                                ease_boundary_scale=ease[2])
            obs, _r, terminated, truncated, _i = env.step(
                np.zeros(2, dtype=np.float32)
            )
        for index in range(int(steps)):
            if terminated or truncated:
                break
            if replay is not None:
                if index >= len(replay):
                    break
                action = replay[index]
            else:
                action = _predict_evaluation_action(
                    policy["model"], obs, action_source=action_source
                )
            executed.append(np.asarray(action, dtype=float).reshape(-1)[:2].copy())
            obs, _r, terminated, truncated, info = env.step(action)
            margins.append(float(info.get("cbf_hocbf_raw_min_margin", np.nan)))
            if record_systems:
                sys_i = wrapper._constraint_system()
                systems.append((
                    np.asarray(sys_i["cbf_rows"], dtype=float).reshape(-1, 2),
                    np.asarray(sys_i["cbf_bounds"], dtype=float).reshape(-1),
                ))
        system = wrapper._constraint_system()
        if render:
            frame = np.asarray(env.render(), dtype=np.uint8)
    finally:
        env.close()
    return {
        "obs": obs,
        "rows": np.asarray(system["cbf_rows"], dtype=float).reshape(-1, 2),
        "bounds": np.asarray(system["cbf_bounds"], dtype=float).reshape(-1),
        "hash": system.get("hash"),
        "executed": np.asarray(executed, dtype=float),
        "margins": np.asarray(margins, dtype=float),
        "systems": systems,
        "frame": frame,
        "terminated": bool(terminated or truncated),
    }


def replay_capture(
    namespace: dict[str, Any],
    policy: dict[str, Any],
    scenario_name: str,
    *,
    replay: np.ndarray,
    capture_steps: Sequence[int],
    seed: int,
    scenarios_by_name: dict[str, Any],
    ease: tuple[float, float, float],
    render: bool,
) -> dict[int, dict[str, Any]]:
    """Replay one action sequence, capturing state at each requested step.

    One pass per arm rather than one pass per (arm, state): the prefix is
    identical for every capture, so re-driving it K times only burns time.
    """

    wanted = sorted(set(int(s) for s in capture_steps))
    horizon = max(wanted)
    env, scenario = build_env(namespace, policy, scenario_name, horizon,
                              scenarios_by_name)
    if render:
        env.unwrapped.render_mode = "rgb_array"
        env.unwrapped.config["offscreen_rendering"] = True
        env.unwrapped.config["real_time_rendering"] = False
    wrapper = find_wrapper(env)
    if wrapper is None:
        env.close()
        raise RuntimeError("CBFContextPhysicalActionWrapper not found")
    out: dict[int, dict[str, Any]] = {}
    try:
        obs, _info = env.reset(seed=int(seed))
        terminated = truncated = False
        if scenario is not None:
            reposition_vehicles(env, scenario, ease_gap_scale=ease[0],
                                ease_closing_speed_scale=ease[1],
                                ease_boundary_scale=ease[2])
            obs, _r, terminated, truncated, _i = env.step(
                np.zeros(2, dtype=np.float32)
            )
        for index in range(horizon):
            if terminated or truncated or index >= len(replay):
                break
            obs, _r, terminated, truncated, _i = env.step(replay[index])
            step = index + 1
            if step in wanted:
                system = wrapper._constraint_system()
                out[step] = {
                    "obs": obs.copy() if hasattr(obs, "copy") else obs,
                    "rows": np.asarray(system["cbf_rows"],
                                       dtype=float).reshape(-1, 2),
                    "bounds": np.asarray(system["cbf_bounds"],
                                         dtype=float).reshape(-1),
                    "hash": system.get("hash"),
                    "frame": (np.asarray(env.render(), dtype=np.uint8)
                              if render else None),
                }
    finally:
        env.close()
    return out


def pick_states(
    fracs: np.ndarray, *, count: int, min_frac: float, min_gap: int,
    mode: str = "variety",
) -> list[int]:
    """Choose frozen states, kept apart in time.

    Consecutive steps describe almost the same geometry, so without a minimum
    separation the selection returns several near-duplicates of one moment.

    ``tight`` takes the smallest non-empty safe sets, which all look alike --
    a thin sliver in the same corner. ``variety`` instead spreads the choice
    across the range of safe-set sizes, so the polytope takes a visibly
    different shape in each row: a sliver, a wedge, a half-plane, a large
    region with several binding faces.
    """

    eligible = [i for i in range(len(fracs)) if fracs[i] >= min_frac]
    if not eligible:
        return []
    if mode == "tight":
        order = sorted(eligible, key=lambda i: fracs[i])
    else:
        # Walk target quantiles of the safe-set size, nearest-first, so the
        # chosen rows span small -> large rather than clustering at one end.
        vals = np.array([fracs[i] for i in eligible], dtype=float)
        targets = np.quantile(vals, np.linspace(0.05, 0.95, max(count, 1)))
        order = []
        for t in targets:
            for i in sorted(eligible, key=lambda j: abs(fracs[j] - t)):
                if i not in order:
                    order.append(i)
                    break
        order += sorted(eligible, key=lambda i: fracs[i])
    chosen: list[int] = []
    for i in order:
        if all(abs(i - j) >= min_gap for j in chosen):
            chosen.append(int(i))
        if len(chosen) == count:
            break
    return sorted(s + 1 for s in chosen)


def actor_readout(policy: dict[str, Any], obs: np.ndarray) -> dict[str, Any]:
    """mu_raw, mu_safe and the actor's per-dimension sigma, in m/s^2."""

    model = policy["model"]
    stages = model.predict_action_stages(obs, deterministic=True)
    mu_raw = np.asarray(stages["mu_raw"], dtype=float).reshape(-1)[:2]
    mu_safe = np.asarray(stages.get("mu_safe", stages["mu_raw"]),
                         dtype=float).reshape(-1)[:2]
    obs_t, _vec = model.policy.obs_to_tensor(obs)
    with th.no_grad():
        dist = model.policy.get_distribution(obs_t)
        sigma = dist.distribution.stddev.cpu().numpy().reshape(-1)[:2]
    return {"mu_raw": mu_raw, "mu_safe": mu_safe, "sigma": np.abs(sigma)}


def margin_field(rows: np.ndarray, bounds: np.ndarray, lim: float, n: int = 420):
    grid = np.linspace(-lim, lim, n)
    AX, AY = np.meshgrid(grid, grid)
    flat = np.stack([AX.ravel(), AY.ravel()], axis=1)
    # rows @ a <= bounds, so slack_i(a) = b_i - A_i . a ; S = min_i slack_i
    slack = bounds[None, :] - flat @ rows.T
    return AX, AY, slack.min(axis=1).reshape(AX.shape)


def feasible(rows: np.ndarray, bounds: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (bounds[None, :] - pts @ rows.T).min(axis=1) >= 0.0


def distance_to_polytope(
    rows: np.ndarray, bounds: np.ndarray, mu: np.ndarray, lim: float, n: int = 260
) -> float:
    """0 inside; otherwise the distance to the nearest feasible action."""

    if bool(feasible(rows, bounds, mu.reshape(1, 2))[0]):
        return 0.0
    grid = np.linspace(-lim, lim, n)
    AX, AY = np.meshgrid(grid, grid)
    pts = np.stack([AX.ravel(), AY.ravel()], axis=1)
    ok = feasible(rows, bounds, pts)
    if not np.any(ok):
        return float("nan")
    return float(np.min(np.linalg.norm(pts[ok] - mu[None, :], axis=1)))


def p_safe(
    rows: np.ndarray, bounds: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
    lim: float, samples: int = 20000, rng_seed: int = 0,
) -> float:
    rng = np.random.default_rng(rng_seed)
    pts = rng.normal(mu[None, :], np.maximum(sigma, 1e-6)[None, :],
                     size=(samples, 2))
    pts = np.clip(pts, -lim, lim)
    return float(np.mean(feasible(rows, bounds, pts)))


def draw_panel(
    ax, rows, bounds, readout, label, *, lim: float, box: float,
    show_ylabel: bool, show_title: bool = True
) -> dict[str, float]:
    AX, AY, S = margin_field(rows, bounds, lim)
    vmax = float(np.nanmax(np.abs(S)))
    im = ax.pcolormesh(AX, AY, S, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                       shading="auto", zorder=1)
    ax.contour(AX, AY, S, levels=[0.0], colors="#111111", linewidths=2.0,
               zorder=3)
    ax.contourf(AX, AY, S, levels=[-1e9, 0.0], colors="none", hatches=["////"],
                zorder=2, alpha=0.0)
    # Each half-plane boundary, so the polytope's faces are identifiable.
    t = np.linspace(-lim, lim, 2)
    for A_i, b_i in zip(rows, bounds):
        if abs(A_i[1]) > 1e-9:
            ax.plot(t, (b_i - A_i[0] * t) / A_i[1], color="#333333",
                    linewidth=0.6, alpha=0.35, zorder=3)
        elif abs(A_i[0]) > 1e-9:
            ax.axvline(b_i / A_i[0], color="#333333", linewidth=0.6,
                       alpha=0.35, zorder=3)

    mu_raw, mu_safe, sigma = readout["mu_raw"], readout["mu_safe"], readout["sigma"]
    color = ARM_COLORS.get(label, "#000000")
    for k, alpha in ((1.0, 0.55), (2.0, 0.30)):
        ax.add_patch(Ellipse(tuple(mu_raw), 2 * k * sigma[0], 2 * k * sigma[1],
                             facecolor="none", edgecolor=color, linewidth=1.5,
                             alpha=alpha, zorder=5))
    moved = float(np.linalg.norm(mu_safe - mu_raw))
    if moved > 1e-6:
        ax.annotate("", xy=tuple(mu_safe), xytext=tuple(mu_raw), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="#111111",
                                    linewidth=2.0, shrinkA=0, shrinkB=0))
        ax.plot(*mu_safe, marker="X", markersize=11, color="#111111",
                markeredgecolor="white", markeredgewidth=1.2, zorder=7)
    ax.plot(*mu_raw, marker="o", markersize=10, color=color,
            markeredgecolor="white", markeredgewidth=1.4, zorder=7)

    # The physical action box. mu_raw is the actor mean *before* box clipping,
    # so it can legitimately sit outside; the panel must show that rather than
    # silently crop the marker away.
    ax.add_patch(plt.Rectangle((-box, -box), 2 * box, 2 * box, fill=False,
                               edgecolor="#111111", linewidth=1.4,
                               linestyle=(0, (5, 3)), zorder=4))
    stats = {
        "p_safe": p_safe(rows, bounds, mu_raw, sigma, box),
        "d_mu_raw": distance_to_polytope(rows, bounds, mu_raw, box),
        "projection_shift": moved,
        "s_at_mu_raw": float((bounds - rows @ mu_raw).min()),
    }
    if show_title:
        ax.set_title(label, fontsize=10.5, fontweight="semibold", color=color)
    ax.text(
        0.5, -0.17,
        f"P(safe) = {stats['p_safe']:.2f}    d($\\mu_{{raw}}$) = "
        f"{stats['d_mu_raw']:.2f}    $S(\\mu_{{raw}})$ = {stats['s_at_mu_raw']:+.2f}"
        + (f"\nprojection shift = {moved:.2f} m/s$^2$" if moved > 1e-6
           else "\nno internal projection"),
        transform=ax.transAxes, ha="center", va="top", fontsize=8.6,
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$a_x$  (m/s$^2$)")
    if show_ylabel:
        ax.set_ylabel(r"$a_y$  (m/s$^2$)")
    else:
        ax.tick_params(labelleft=False)
    return stats | {"im": im}


def policy_density(
    mu: np.ndarray, sigma: np.ndarray, box: float, n: int = 320
):
    """pi(a|s) over the action box for a diagonal-Gaussian PPO policy.

    Returned already normalised to the distribution's own peak (1.0 at mu),
    so a policy whose mean sits outside the box shows only a dim tail -- which
    is the honest picture, not an artefact.
    """

    g = np.linspace(-box, box, n)
    AX, AY = np.meshgrid(g, g)
    sx = max(float(sigma[0]), 1e-6)
    sy = max(float(sigma[1]), 1e-6)
    z = ((AX - mu[0]) / sx) ** 2 + ((AY - mu[1]) / sy) ** 2
    return AX, AY, np.exp(-0.5 * z)


def draw_density_panel(
    ax, rows, bounds, readout: dict[str, Any], label: str, *, box: float,
    show_ylabel: bool,
) -> dict[str, float]:
    """One policy's action distribution, with the safe set drawn over it.

    Background is pi(a|s): bright where this policy would actually act. The
    polytope is identical across panels -- it depends only on the state -- so
    the only thing that moves between panels is where the probability sits.
    """

    mu_raw, mu_safe, sigma = readout["mu_raw"], readout["mu_safe"], readout["sigma"]
    color = ARM_COLORS.get(label, "#333333")
    AX, AY, D = policy_density(mu_raw, sigma, box)
    cmap = LinearSegmentedColormap.from_list(f"{label}_d", ["#ffffff", color])
    ax.pcolormesh(AX, AY, D, cmap=cmap, vmin=0.0, vmax=1.0, shading="auto",
                  zorder=1)
    # Density contours at the 1 and 2 sigma levels of a 2D Gaussian.
    ax.contour(AX, AY, D, levels=[np.exp(-2.0), np.exp(-0.5)],
               colors=[color], linewidths=[0.9, 1.3], alpha=0.85, zorder=3)

    # The safe set: mute everything outside it, then outline it hard, so the
    # polygon reads as a shape rather than as a colour transition.
    GX, GY, S = margin_field(rows, bounds, box)
    ax.contourf(GX, GY, S, levels=[-1e9, 0.0], colors=["#9aa0a6"], alpha=0.34,
                zorder=4)
    ax.contour(GX, GY, S, levels=[0.0], colors="#111111", linewidths=2.6,
               zorder=5)

    inside = bool(np.abs(mu_raw).max() <= box)
    shown = np.clip(mu_raw, -box, box)
    ax.plot(*shown, marker=("o" if inside else "^"), markersize=11,
            color=color, markeredgecolor="#111111", markeredgewidth=1.5,
            zorder=8)
    moved = float(np.linalg.norm(mu_safe - mu_raw))
    if moved > 1e-6:
        ax.annotate("", xy=tuple(np.clip(mu_safe, -box, box)),
                    xytext=tuple(shown), zorder=7,
                    arrowprops=dict(arrowstyle="-|>", color="#111111",
                                    linewidth=2.0, shrinkA=0, shrinkB=0))
        ax.plot(*np.clip(mu_safe, -box, box), marker="X", markersize=12,
                color="#111111", markeredgecolor="white", markeredgewidth=1.2,
                zorder=9)
    stats = {
        "p_safe": p_safe(rows, bounds, mu_raw, sigma, box),
        "d_mu_raw": distance_to_polytope(rows, bounds, mu_raw, box),
        "s_at_mu_raw": float((bounds - rows @ mu_raw).min()),
        "projection_shift": moved,
        "mu_raw_ax": float(mu_raw[0]), "mu_raw_ay": float(mu_raw[1]),
        "sigma_ax": float(sigma[0]), "sigma_ay": float(sigma[1]),
    }
    ax.set_title(f"{label}\nP(safe) = {stats['p_safe']:.2f}", fontsize=10.5,
                 fontweight="semibold", color=color)
    if not inside:
        # Marker is pinned to the box edge; say where the mean actually is,
        # inside the panel so it cannot collide with the axis label.
        ax.text(0.5, 0.025,
                f"$\\mu_{{raw}}$ off-box at ({mu_raw[0]:.1f}, {mu_raw[1]:.1f})",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.6, color="#111111", zorder=11,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#bbbbbb", alpha=0.92))
    ax.set_xlim(-box, box)
    ax.set_ylim(-box, box)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$a_x$  (m/s$^2$)")
    if show_ylabel:
        ax.set_ylabel(r"$a_y$  (m/s$^2$)")
    else:
        ax.tick_params(labelleft=False)
    return stats


def draw_state_panel(
    ax, rows, bounds, readouts: dict[str, dict[str, Any]],
    order: Sequence[str], *, lim: float, box: float, detail: bool = False,
) -> tuple[Any, list[dict[str, Any]]]:
    """One polytope, every arm's action on it.

    All arms are frozen at the same state, so they share one constraint
    system. Drawing the field once and overlaying the arms puts the whole
    comparison in a single frame instead of asking the reader to eye-track
    the same background across five copies.
    """

    AX, AY, S = margin_field(rows, bounds, lim)
    vmax = float(np.nanmax(np.abs(S))) or 1.0
    im = ax.pcolormesh(AX, AY, S, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                       shading="auto", zorder=1)
    ax.contour(AX, AY, S, levels=[0.0], colors="#111111", linewidths=2.2,
               zorder=4)
    if detail:
        # Hatching duplicates what the red already says, and one thin line per
        # constraint row is mostly noise outside the box, so both are opt-in.
        ax.contourf(AX, AY, S, levels=[-1e9, 0.0], colors="none",
                    hatches=["////"], zorder=2, alpha=0.0)
        for A_i, b_i in zip(rows, bounds):
            if abs(A_i[1]) > 1e-9:
                t = np.linspace(-lim, lim, 2)
                ax.plot(t, (b_i - A_i[0] * t) / A_i[1], color="#333333",
                        linewidth=0.6, alpha=0.30, zorder=3)
            elif abs(A_i[0]) > 1e-9:
                ax.axvline(b_i / A_i[0], color="#333333", linewidth=0.6,
                           alpha=0.30, zorder=3)

    ax.add_patch(plt.Rectangle((-box, -box), 2 * box, 2 * box, fill=False,
                               edgecolor="#111111", linewidth=1.8,
                               linestyle=(0, (5, 3)), zorder=6))

    stats: list[dict[str, Any]] = []
    for label in order:
        r = readouts[label]
        mu_raw, mu_safe, sigma = r["mu_raw"], r["mu_safe"], r["sigma"]
        color = ARM_COLORS.get(label, "#000000")
        if detail:
            for k, alpha in ((1.0, 0.5), (2.0, 0.26)):
                ax.add_patch(Ellipse(tuple(mu_raw), 2 * k * sigma[0],
                                     2 * k * sigma[1], facecolor="none",
                                     edgecolor=color, linewidth=1.3,
                                     alpha=alpha, zorder=7))
        moved = float(np.linalg.norm(mu_safe - mu_raw))
        if moved > 1e-6:
            ax.annotate("", xy=tuple(mu_safe), xytext=tuple(mu_raw), zorder=8,
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        linewidth=2.0, shrinkA=0, shrinkB=0))
            ax.plot(*mu_safe, marker="X", markersize=12, color=color,
                    markeredgecolor="#111111", markeredgewidth=1.3, zorder=9)
        ax.plot(*mu_raw, marker="o", markersize=11, color=color,
                markeredgecolor="white", markeredgewidth=1.6, zorder=10)
        stats.append({
            "policy": label,
            "p_safe": p_safe(rows, bounds, mu_raw, sigma, box),
            "d_mu_raw": distance_to_polytope(rows, bounds, mu_raw, box),
            "s_at_mu_raw": float((bounds - rows @ mu_raw).min()),
            "projection_shift": moved,
        })
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$a_x$  longitudinal acceleration  (m/s$^2$)")
    ax.set_ylabel(r"$a_y$  lateral acceleration  (m/s$^2$)")
    return im, stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", action="append", required=True,
                   metavar="LABEL=RUN_CONFIG_JSON")
    p.add_argument("--driver", default=None,
                   help="Arm that drives to the frozen state. Default: B2 if present.")
    p.add_argument("--scenario", default=CONGESTED,
                   help=f"Scenario name, or '{CONGESTED}' for the 40-vehicle env.")
    p.add_argument("--freeze-at", choices=("tight-feasible", "tightest", "step"),
                   default="tight-feasible",
                   help="'tight-feasible' (default) picks the smallest "
                        "non-empty safe set -- the state that actually "
                        "separates the arms; 'tightest' picks minimum "
                        "S_HOCBF, which is usually QP-infeasible; 'step' "
                        "uses --freeze-step verbatim.")
    p.add_argument("--min-feasible-fraction", type=float, default=0.02,
                   help="Smallest safe-set share of the action box that still "
                        "counts as non-empty. Default %(default)s.")
    p.add_argument("--states", type=int, default=4,
                   help="How many frozen states to show, one per row. Default %(default)s.")
    p.add_argument("--min-state-gap", type=int, default=12,
                   help="Minimum policy steps between chosen states, so the "
                        "rows are not near-duplicates. Default %(default)s.")
    p.add_argument("--state-mode", choices=("variety", "tight"), default="variety",
                   help="'variety' (default) spreads the chosen states across "
                        "the range of safe-set sizes, so the polytope takes a "
                        "different shape in each row; 'tight' takes the K "
                        "smallest non-empty safe sets.")
    p.add_argument("--freeze-step", type=int, default=40,
                   help="Policy step to freeze at when --freeze-at step.")
    p.add_argument("--horizon", type=int, default=200,
                   help="Steps searched when --freeze-at tightest. Default %(default)s.")
    p.add_argument("--seed", type=int, default=1100011)
    p.add_argument("--lim", type=float, default=3.0, help="Action box half-width.")
    p.add_argument("--detail", action="store_true",
                   help="Restore the optional layers: violation hatching, "
                        "per-constraint boundary lines, and the 1/2 sigma "
                        "ellipses. Off by default to keep the plane readable.")
    p.add_argument("--view-scale", type=float, default=1.15,
                   help="Plot half-width as a multiple of the action box, so "
                        "the polytope is visible beyond the box rather than "
                        "cropped at it. Default %(default)s.")
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
    if args.scenario != CONGESTED and args.scenario not in scenarios_by_name:
        raise SystemExit(f"Unknown scenario {args.scenario!r}")

    policies: dict[str, dict[str, Any]] = {}
    for spec in args.policy:
        label, path = spec.split("=", 1)
        if label not in ACTION_SOURCE:
            raise SystemExit(f"Unknown policy label {label!r}")
        policies[label] = load_policy(Path(path).resolve())
    order = list(policies)
    driver = args.driver or ("B2_cbf_nd_reward_actor"
                             if "B2_cbf_nd_reward_actor" in policies else order[0])
    ease = (args.ease_gap_scale, args.ease_closing_speed_scale,
            args.ease_boundary_scale)

    for label, pol in policies.items():
        pol["namespace"] = build_namespace(root, pol["cbf"])
        pol["model"] = load_model(pol["variant"], pol["model_path"], "cpu")

    # Pass 1: the driver explores, so a genuinely tight state can be located.
    # A slack state makes every arm read P(safe)=1 and separates nothing.
    d = policies[driver]
    search = args.freeze_at != "step"
    search_steps = int(args.horizon) if search else int(args.freeze_step)
    scout = drive_and_freeze(
        d["namespace"], d, ACTION_SOURCE[driver], args.scenario,
        steps=search_steps, seed=int(args.seed),
        scenarios_by_name=scenarios_by_name, ease=ease, replay=None,
        render=False, record_systems=search,
    )
    if args.freeze_at == "tightest":
        m = scout["margins"]
        if not np.any(np.isfinite(m)):
            raise SystemExit("no finite margins recorded; cannot pick a state")
        freeze_steps = [
            int(np.nanargmin(np.where(np.isfinite(m), m, np.inf))) + 1
        ]
        print(f"[polytope] tightest state at step {freeze_steps[0]} "
              f"(t = {freeze_steps[0] / 20:.2f} s), "
              f"S_HOCBF = {m[freeze_steps[0] - 1]:+.2f}")
    elif args.freeze_at == "tight-feasible":
        # The interesting state is not the worst one -- that is QP-infeasible
        # and reads P(safe)=0 for every arm alike. It is the state whose safe
        # set is smallest while still non-empty, because that is where the
        # arms' action distributions can actually be separated.
        lim = float(args.lim)
        g = np.linspace(-lim, lim, 60)
        GX, GY = np.meshgrid(g, g)
        probe = np.stack([GX.ravel(), GY.ravel()], axis=1)
        fracs = np.array([
            float(np.mean(feasible(r, b, probe)))
            for r, b in scout["systems"]
        ])
        freeze_steps = pick_states(
            fracs, count=int(args.states),
            min_frac=float(args.min_feasible_fraction),
            min_gap=int(args.min_state_gap),
            mode=str(args.state_mode),
        )
        if not freeze_steps:
            raise SystemExit(
                f"no driven step keeps a feasible fraction >= "
                f"{args.min_feasible_fraction}; loosen it or drive longer"
            )
        print(f"[polytope] feasible fraction over {len(fracs)} steps: "
              f"min {100 * fracs.min():.1f}%, median {100 * np.median(fracs):.1f}%, "
              f"fully infeasible {int(np.sum(fracs == 0.0))}")
        for s in freeze_steps:
            print(f"           state t = {s / 20:.2f} s (step {s}): safe set "
                  f"{100 * fracs[s - 1]:.1f}% of box, "
                  f"S_HOCBF = {scout['margins'][s - 1]:+.2f}")
    else:
        freeze_steps = [int(args.freeze_step)]

    # Pass 2: one replay per arm, capturing every chosen state on the way.
    prefix = scout["executed"][: max(freeze_steps)]
    captures: dict[str, dict[int, dict[str, Any]]] = {}
    for label, pol in policies.items():
        captures[label] = replay_capture(
            pol["namespace"], pol, args.scenario,
            replay=prefix, capture_steps=freeze_steps, seed=int(args.seed),
            scenarios_by_name=scenarios_by_name, ease=ease,
            render=(label == driver),
        )
    ref_caps = captures[driver]
    usable = [s for s in freeze_steps if all(s in captures[l] for l in order)]
    if not usable:
        raise SystemExit("no chosen state was reached by every arm")
    for s in usable:
        for label in order:
            if captures[label][s]["hash"] != ref_caps[s]["hash"]:
                print(f"  WARNING: {label} constraint hash differs from driver "
                      f"at step {s}; states diverged.")

    readouts: dict[int, dict[str, dict[str, Any]]] = {
        s: {label: actor_readout(policies[label], captures[label][s]["obs"])
            for label in order}
        for s in usable
    }

    box = float(args.lim)
    reach = max(
        float(np.max(np.abs(np.concatenate([r["mu_raw"], r["mu_safe"]]))))
        for per_state in readouts.values() for r in per_state.values()
    )
    lim = max(box * float(args.view_scale), reach * 1.08)
    print(f"[polytope] view half-width {lim:.2f} m/s^2 "
          f"(action box {box:g}, furthest actor mean {reach:.2f})")

    args.out.mkdir(parents=True, exist_ok=True)
    stats_rows: list[dict[str, Any]] = []
    for step in usable:
        cap = ref_caps[step]
        # The rendered frame is ~900x260: laying it beside a square plot
        # letterboxes it into a sliver. Across the top it keeps its own
        # aspect and the polytope gets a full square panel.
        ncols = len(order)
        fig = plt.figure(figsize=(3.35 * ncols + 0.7, 7.6))
        gs = fig.add_gridspec(2, ncols, height_ratios=[0.50, 1.0],
                              hspace=0.32, wspace=0.10)
        ax_img = fig.add_subplot(gs[0, :])
        if cap["frame"] is not None:
            ax_img.imshow(cap["frame"])
        ax_img.axis("off")
        ax_img.set_title("simulator state", fontsize=11.0,
                         fontweight="semibold")
        stats = []
        for c, label in enumerate(order):
            stats.append(draw_density_panel(
                fig.add_subplot(gs[1, c]), cap["rows"], cap["bounds"],
                readouts[step][label], label, box=box, show_ylabel=(c == 0),
            ) | {"policy": label})
        frac = 100.0 * float(np.mean(feasible(
            cap["rows"], cap["bounds"],
            np.stack(np.meshgrid(np.linspace(-box, box, 60),
                                 np.linspace(-box, box, 60)),
                     axis=-1).reshape(-1, 2))))
        fig.legend(
            handles=[
                Patch(facecolor="#9aa0a6", alpha=0.34, edgecolor="none",
                      label="HOCBF-violating region"),
                Line2D([], [], color="#111111", linewidth=2.6,
                       label="safe set boundary (polytope)"),
                Line2D([], [], marker="o", linestyle="none", markersize=9,
                       color="#888888", markeredgecolor="#111111",
                       label=r"$\mu_{raw}$ (actor mean)"),
                Line2D([], [], marker="X", linestyle="none", markersize=10,
                       color="#111111", markeredgecolor="white",
                       label=r"$\mu_{safe}$ after internal projection"),
            ],
            loc="lower center", bbox_to_anchor=(0.5, 0.005), frameon=False,
            ncols=4, fontsize=9.2,
        )
        fig.suptitle(
            f"Where each policy would act -- {args.scenario}, "
            f"t = {step / 20:.2f} s, seed {args.seed}\n"
            r"shading = $\pi(a\,|\,s)$, the policy's own probability over "
            "actions (bright = likely)\n"
            f"safe set = {frac:.1f}% of the action box",
            fontsize=12.0, fontweight="bold", y=0.995,
        )
        fig.subplots_adjust(top=0.825, bottom=0.115, left=0.045, right=0.99)
        path = args.out / (f"polytope_{args.scenario}_s{args.seed}"
                           f"_t{step:03d}.png")
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)
        print(f"wrote {path}")
        for s in stats:
            stats_rows.append(
                {"step": step, "time_s": round(step / 20, 3),
                 "safe_set_pct_of_box": round(frac, 2)}
                | {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in s.items()}
            )

    stem = (f"polytope_{args.scenario}_s{args.seed}_"
            f"{len(usable)}states_{args.state_mode}")
    csv_path = args.out / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stats_rows[0]))
        w.writeheader()
        w.writerows(stats_rows)
    print(f"wrote {csv_path}")
    print()
    print(f"{'t (s)':>7}  {'policy':<26}{'P(safe)':>9}{'d(mu_raw)':>11}"
          f"{'S(mu_raw)':>11}{'proj shift':>12}")
    for row in stats_rows:
        print(f"{row['time_s']:>7.2f}  {row['policy']:<26}{row['p_safe']:>9.2f}"
              f"{row['d_mu_raw']:>11.2f}{row['s_at_mu_raw']:>11.2f}"
              f"{row['projection_shift']:>12.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
