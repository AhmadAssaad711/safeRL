"""Seed-averaged HOCBF safety margin per policy, congested env and scenarios.

Companion to ``plot_hocbf_safety_margin.py``. That script plots one thin line
per rollout; this one averages across rollouts and produces the two figures:

1. ``hocbf_margin_congested_mean.png`` -- the real 40-vehicle MTM environment
   these policies were trained and evaluated in (no scripted geometry), every
   policy on the same episode seeds for the same wall time.
2. ``hocbf_margin_scenarios_mean.png`` -- the same aggregation, broken out per
   scripted diagnostic scenario.

The plotted quantity is unchanged from the companion script:

    S_HOCBF(t) = min_i [ b_i(t) - A_i(t) u_raw(t) ]

the minimum slack across every pairwise-vehicle and road-boundary HOCBF row,
excluding the physical action-box rows, evaluated at the policy's *raw*
action. It is read straight off the ``cbf_hocbf_raw_min_margin`` info field, so
u_raw correctness is entirely a question of which action reaches ``env.step()``:
A1/B1/B2 use ``action_source="policy"``; C1/C2 carry an internal projection and
therefore use ``action_source="raw_actor_mean"`` to recover the genuine raw
actor mean. Interpretation: S_HOCBF > 0 means the raw action satisfies every
HOCBF condition; < 0 means it violates at least one.

Censoring
---------
``terminate_on_collision`` is on, so a rollout that crashes contributes no
samples after its crash step. A plain mean across rollouts is therefore taken
over *survivors only* and is biased optimistic for exactly the policies that
crash most -- the opposite of what a safety figure should do. Two guards:

- the surviving-rollout count n(t) is carried through to the CSV and drawn as
  a companion trace, and the mean line is dimmed once n(t) drops below the
  full sample, so a late-time mean built from a handful of lucky rollouts is
  visibly marked as such;
- ``--censor hold`` additionally carries each crashed rollout's final margin
  forward to the full horizon, which keeps the crashing policy in the average
  at the value it crashed at. ``--censor drop`` (default) is the survivors-only
  mean.

Neither is a substitute for reading the survival rate, which is printed and
written into the summary table alongside every mean.
"""

from __future__ import annotations

import argparse
import copy
import csv
import warnings
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.rendering.render_policy_scenarios import find_repo_root, make_scenarios
from scripts.rendering.render_ppo_scenario_gifs import (
    SIX_SCENES,
    build_namespace,
    load_policy,
)
from scripts.reporting.plot_hocbf_safety_margin import (
    ACTION_SOURCE,
    ARM_COLORS,
    check_shared_gains,
    rollout_margin,
)
from scripts.training.run_ppo_cbf_progression import (
    _predict_evaluation_action,
    load_model,
    make_ppo_cbf_env,
)

CONGESTED = "congested_mtm_40"
CONGESTED_TITLE = "Congested MTM traffic (40 vehicles, no scripted geometry)"


def congested_env_config(base_env_config: dict[str, Any], steps: int) -> dict[str, Any]:
    """The policy's own training/evaluation environment, horizon-extended.

    Unlike ``scenario_env_config`` this keeps ``vehicles_count`` untouched --
    the whole point of this condition is that it is the real congested
    environment, not a scripted scene with a handful of placed cars.
    """

    config = copy.deepcopy(base_env_config)
    substeps_per_policy_step = max(
        1,
        round(
            float(base_env_config["simulation_frequency"])
            / float(base_env_config["policy_frequency"])
        ),
    )
    physics_steps = (int(steps) + 21) * substeps_per_policy_step
    config["episode_steps"] = physics_steps
    config["duration"] = physics_steps
    config["terminate_on_collision"] = True
    return config


def rollout_margin_congested(
    namespace: dict[str, Any],
    policy: dict[str, Any],
    action_source: str,
    *,
    steps: int,
    seed: int,
) -> tuple[np.ndarray, int | None]:
    """``rollout_margin`` for the unscripted congested environment.

    Mirrors the companion script's scenario rollout exactly -- same external
    CBF OFF, same bootstrap-step convention, same info field -- minus the
    vehicle repositioning, so the two conditions stay comparable.
    """

    env_config = congested_env_config(policy["env_config"], steps)
    env = make_ppo_cbf_env(
        namespace,
        env_config=env_config,
        reward_config=policy["reward_config"],
        project_inputs=False,  # external CBF OFF throughout
        lambda_delta=0.0,
        lambda_intervention=0.0,
        correction_epsilon=policy["correction_epsilon"],
        monitor_path=None,
    )
    margins: list[float] = []
    first_collision_step: int | None = None
    try:
        obs, _info = env.reset(seed=int(seed))
        terminated = truncated = False
        for step_index in range(1, int(steps) + 1):
            if terminated or truncated:
                break
            action = _predict_evaluation_action(
                policy["model"], obs, action_source=action_source
            )
            obs, _reward, terminated, truncated, info = env.step(action)
            margins.append(float(info["cbf_hocbf_raw_min_margin"]))
            if (
                first_collision_step is None
                and int(info.get("ego_collision_events", 0)) > 0
            ):
                first_collision_step = step_index
    finally:
        env.close()
    return np.asarray(margins, dtype=float), first_collision_step


def capture_filmstrip(
    namespace: dict[str, Any],
    policy: dict[str, Any],
    action_source: str,
    condition: str,
    *,
    steps: int,
    seed: int,
    n_frames: int,
    scenarios_by_name: dict[str, Any],
    ease_gap_scale: float,
    ease_closing_speed_scale: float,
    ease_boundary_scale: float,
) -> tuple[Image.Image | None, list[float]]:
    """Render one rollout and return (horizontal strip, frame times in s).

    Same rollout contract as the margin passes -- external CBF off, same
    bootstrap-step convention, same action source -- so the strip shows the
    manoeuvre that produced the curve above it, not a different rollout.
    Frames are sampled evenly across whatever the rollout actually lasted, so
    the last frame is the crash for a rollout that crashed.
    """

    from scripts.rendering.render_ppo_scenario_gifs import (
        reposition_vehicles,
        scenario_env_config,
    )

    if condition == CONGESTED:
        env_config = congested_env_config(policy["env_config"], steps)
        scenario = None
    else:
        scenario = scenarios_by_name[condition]
        env_config = scenario_env_config(policy["env_config"], scenario, steps)
        env_config["terminate_on_collision"] = True
    env = make_ppo_cbf_env(
        namespace,
        env_config=env_config,
        reward_config=policy["reward_config"],
        project_inputs=False,
        lambda_delta=0.0,
        lambda_intervention=0.0,
        correction_epsilon=policy["correction_epsilon"],
        monitor_path=None,
    )
    env.unwrapped.render_mode = "rgb_array"
    env.unwrapped.config["offscreen_rendering"] = True
    env.unwrapped.config["real_time_rendering"] = False
    frames: list[np.ndarray] = []
    policy_frequency_hz = float(policy["env_config"]["policy_frequency"])
    try:
        obs, _info = env.reset(seed=int(seed))
        terminated = truncated = False
        if scenario is not None:
            reposition_vehicles(
                env, scenario,
                ease_gap_scale=ease_gap_scale,
                ease_closing_speed_scale=ease_closing_speed_scale,
                ease_boundary_scale=ease_boundary_scale,
            )
            obs, _reward, terminated, truncated, _info = env.step(
                np.zeros(2, dtype=np.float32)
            )
        for _step_index in range(1, int(steps) + 1):
            if terminated or truncated:
                break
            action = _predict_evaluation_action(
                policy["model"], obs, action_source=action_source
            )
            obs, _reward, terminated, truncated, _info = env.step(action)
            frames.append(np.asarray(env.render(), dtype=np.uint8))
    finally:
        env.close()
    if not frames:
        return None, []
    picks = np.unique(
        np.linspace(0, len(frames) - 1, num=int(n_frames)).round().astype(int)
    )
    images = [Image.fromarray(frames[i]) for i in picks]
    times = [float((i + 1) / policy_frequency_hz) for i in picks]
    gap = 6
    width = sum(im.width for im in images) + gap * (len(images) - 1)
    height = max(im.height for im in images)
    strip = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for im in images:
        strip.paste(im, (x, 0))
        x += im.width + gap
    return strip, times


def draw_filmstrip(
    ax: plt.Axes,
    strip: Image.Image | None,
    times: Sequence[float],
    *,
    caption: str,
) -> None:
    ax.axis("off")
    if strip is None:
        return
    ax.imshow(np.asarray(strip), interpolation="antialiased")
    n = len(times)
    # A single frame is scene context, not a sequence: a lone timestamp is
    # clutter. Label the frames only when there are several to order.
    if n > 1:
        for index, t in enumerate(times):
            ax.text(
                (index + 0.5) / n, -0.06, f"t = {t:.1f}s",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
                color="#333333",
            )
    if caption:
        ax.set_title(caption, fontsize=8.0, color="#333333", pad=2.0)


def stack_runs(
    runs: Sequence[tuple[np.ndarray, int | None]],
    *,
    horizon: int,
    censor: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, sd, n_contributing) over a common time grid."""

    grid = np.full((len(runs), horizon), np.nan, dtype=float)
    for row, (series, _first_collision) in enumerate(runs):
        take = min(series.size, horizon)
        grid[row, :take] = series[:take]
        if censor == "hold" and 0 < take < horizon:
            grid[row, take:] = series[take - 1]
    # An all-NaN column is expected and meaningful: every rollout has already
    # crashed by that time index. It becomes a gap in the plotted line.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"(Mean|Degrees) of", RuntimeWarning)
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        mean = np.nanmean(grid, axis=0)
        sd = np.nanstd(grid, axis=0, ddof=0)
    n = np.sum(np.isfinite(grid), axis=0)
    return mean, sd, n


def draw_condition(
    ax: plt.Axes,
    runs_by_policy: dict[str, list[tuple[np.ndarray, int | None]]],
    policy_order: Sequence[str],
    *,
    horizon: int,
    policy_frequency_hz: float,
    censor: str,
    band: str,
    fade_censored: bool = True,
) -> None:
    t = np.arange(1, horizon + 1) / policy_frequency_hz
    for label in policy_order:
        runs = runs_by_policy.get(label, [])
        if not runs:
            continue
        mean, sd, n = stack_runs(runs, horizon=horizon, censor=censor)
        color = ARM_COLORS[label]
        if fade_censored:
            full = n.max() if n.size else 0
            complete = n >= full
            # Solid where every rollout still contributes; faded once the
            # sample thins out, so a late-time mean over survivors cannot be
            # misread.
            ax.plot(t, np.where(complete, mean, np.nan), color=color,
                    linewidth=2.0, label=label, zorder=4)
            ax.plot(t, mean, color=color, linewidth=1.3, alpha=0.35, zorder=3)
        else:
            # Uniformly bold. Safe to drop the fade only where the companion
            # panel already shows how many rollouts remain at each time.
            ax.plot(t, mean, color=color, linewidth=2.4, label=label, zorder=4)
        if band != "none":
            # SD is the spread of individual rollouts and at n=7 it dwarfs the
            # between-arm difference; SEM is the uncertainty of the plotted
            # mean, which is what the curve is actually claiming.
            half = sd if band == "sd" else sd / np.sqrt(np.maximum(n, 1))
            ax.fill_between(t, mean - half, mean + half, color=color,
                            alpha=0.15, linewidth=0, zorder=2)
    ax.axhline(0.0, color="#333333", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"mean $S_{HOCBF}(t)$")
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def robust_ylim(
    runs_by_policy: dict[str, list[tuple[np.ndarray, int | None]]],
    *,
    low: float = 2.0,
    high: float = 98.0,
) -> tuple[float, float] | None:
    """Percentile limits so one arm's deep excursions cannot squash the rest.

    This only changes the viewport, never the plotted quantity -- the lines
    themselves are the raw S_HOCBF.
    """

    samples = [s for runs in runs_by_policy.values() for s, _fc in runs if s.size]
    if not samples:
        return None
    pool = np.concatenate(samples)
    lo, hi = np.percentile(pool, [low, high])
    pad = 0.08 * max(hi - lo, 1e-6)
    return float(lo - pad), float(hi + pad)


def mean_curve_ylim(
    runs_by_policy: dict[str, list[tuple[np.ndarray, int | None]]],
    policy_order: Sequence[str],
    *,
    horizon: int,
    censor: str,
    band: str,
) -> tuple[float, float] | None:
    """Fit the viewport to the averaged curves, not to individual excursions.

    The seed-average panel exists to separate the arms; scaling it to the
    worst single rollout is what flattens that separation into a band of
    overlapping lines near the top of the axes.
    """

    lows: list[float] = []
    highs: list[float] = []
    for label in policy_order:
        runs = runs_by_policy.get(label, [])
        if not runs:
            continue
        mean, sd, n = stack_runs(runs, horizon=horizon, censor=censor)
        if band == "none":
            half = np.zeros_like(mean)
        elif band == "sd":
            half = sd
        else:
            half = sd / np.sqrt(np.maximum(n, 1))
        lo = mean - half
        hi = mean + half
        if np.any(np.isfinite(lo)):
            lows.append(float(np.nanmin(lo)))
        if np.any(np.isfinite(hi)):
            highs.append(float(np.nanmax(hi)))
    if not lows or not highs:
        return None
    lo, hi = min(lows), max(highs)
    pad = 0.10 * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def draw_rollouts(
    ax: plt.Axes,
    runs_by_policy: dict[str, list[tuple[np.ndarray, int | None]]],
    policy_order: Sequence[str],
    *,
    policy_frequency_hz: float,
) -> None:
    """Every individual rollout, one thin line each, x at its collision."""

    for label in policy_order:
        for index, (series, first_collision) in enumerate(
            runs_by_policy.get(label, [])
        ):
            if series.size == 0:
                continue
            t = np.arange(1, series.size + 1) / policy_frequency_hz
            ax.plot(t, series, color=ARM_COLORS[label], linewidth=0.9, alpha=0.5,
                    label=label if index == 0 else None, zorder=3)
            if first_collision is not None:
                ax.plot(
                    first_collision / policy_frequency_hz, series[-1],
                    marker="x", markersize=6, markeredgewidth=1.6,
                    color=ARM_COLORS[label], alpha=0.85, zorder=5,
                )
    ax.axhline(0.0, color="#333333", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_xlabel("Time (s)")
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def smooth_monotone(values: np.ndarray, window: int) -> np.ndarray:
    """Hann-smooth a survival curve, then restore monotonicity.

    Survival can only fall, but a symmetric smoother will lift points just
    after a drop. Re-imposing a running minimum keeps the curve readable and
    still truthful about the direction.
    """

    if window < 3 or values.size < 3:
        return values
    window = min(int(window) | 1, values.size if values.size % 2 else values.size - 1)
    if window < 3:
        return values
    kernel = np.hanning(window)
    kernel /= kernel.sum()
    padded = np.pad(values, window // 2, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")[: values.size]
    return np.minimum.accumulate(smoothed)


def draw_survival(
    ax: plt.Axes,
    runs_by_policy: dict[str, list[tuple[np.ndarray, int | None]]],
    policy_order: Sequence[str],
    *,
    horizon: int,
    policy_frequency_hz: float,
    smooth_s: float = 0.0,
) -> None:
    t = np.arange(1, horizon + 1) / policy_frequency_hz
    window = int(round(float(smooth_s) * policy_frequency_hz))
    for label in policy_order:
        runs = runs_by_policy.get(label, [])
        if not runs:
            continue
        _mean, _sd, n = stack_runs(runs, horizon=horizon, censor="drop")
        pct = 100.0 * n / len(runs)
        if window >= 3:
            ax.plot(t, pct, color=ARM_COLORS[label], linewidth=0.9, alpha=0.25,
                    zorder=2)
            pct = smooth_monotone(pct, window)
        ax.plot(t, pct, color=ARM_COLORS[label], linewidth=2.4, label=label,
                zorder=4)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("% rollouts without a collision")
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def summarize(
    results: dict[str, dict[str, list[tuple[np.ndarray, int | None]]]],
    policy_order: Sequence[str],
    *,
    horizon: int,
    policy_frequency_hz: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, runs_by_policy in results.items():
        for label in policy_order:
            runs = runs_by_policy.get(label, [])
            if not runs:
                continue
            all_samples = np.concatenate(
                [s for s, _ in runs if s.size]
            ) if any(s.size for s, _ in runs) else np.array([])
            survived = [fc is None for _s, fc in runs]
            crash_times = [
                fc / policy_frequency_hz for _s, fc in runs if fc is not None
            ]
            rows.append(
                {
                    "condition": condition,
                    "policy": label,
                    "rollouts": len(runs),
                    "survival_rate_at_horizon": round(
                        float(np.mean(survived)), 4
                    ),
                    "mean_margin": round(float(np.mean(all_samples)), 4)
                    if all_samples.size
                    else float("nan"),
                    "median_margin": round(float(np.median(all_samples)), 4)
                    if all_samples.size
                    else float("nan"),
                    "worst_margin": round(float(np.min(all_samples)), 4)
                    if all_samples.size
                    else float("nan"),
                    "frac_steps_margin_negative": round(
                        float(np.mean(all_samples < 0.0)), 4
                    )
                    if all_samples.size
                    else float("nan"),
                    "mean_time_to_collision_s": round(float(np.mean(crash_times)), 3)
                    if crash_times
                    else "",
                    "recorded_steps": int(all_samples.size),
                }
            )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", action="append", required=True, metavar="LABEL=RUN_CONFIG_JSON",
        help="Repeatable, and repeatable with the SAME label to pool several "
             "training seeds of one arm. LABEL must be a key of ACTION_SOURCE.",
    )
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument(
        "--no-congested", action="store_true",
        help="Skip the unscripted 40-vehicle condition (figure 1).",
    )
    parser.add_argument(
        "--no-scenarios", action="store_true",
        help="Skip every scripted scenario (figure 2 and its rollouts), leaving "
             "only the congested condition. Each scenario costs another "
             "len(seeds) x len(policies) rollouts, so this is what makes a "
             "high seed count affordable.",
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Policy steps per rollout; 200 = 10 s at 20 Hz. Default %(default)s.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[1100011, 1100012, 1100013, 1100014, 1100015, 1100016, 1100017],
        help="Episode seeds, identical for every policy. Default %(default)s.",
    )
    parser.add_argument(
        "--censor", choices=("drop", "hold"), default="drop",
        help="How to treat a crashed rollout after its crash step. "
             "drop = survivors-only mean (default); hold = carry its final "
             "margin forward to the horizon.",
    )
    parser.add_argument(
        "--no-per-scenario", action="store_true",
        help="Skip the standalone one-figure-per-scenario files.",
    )
    parser.add_argument(
        "--band", choices=("none", "sd", "sem"), default="sem",
        help="Ribbon around the mean: sem = standard error of the plotted "
             "mean (default), sd = spread of individual rollouts, none.",
    )
    parser.add_argument(
        "--filmstrip-frames", type=int, default=3,
        help="Rendered frames shown above each scenario panel; 0 disables. "
             "The congested figure never carries one. Default %(default)s.",
    )
    parser.add_argument(
        "--survival-smooth-s", type=float, default=1.0,
        help="Hann window, in seconds, for the collision-free curve; 0 leaves "
             "the raw step function. Default %(default)s.",
    )
    parser.add_argument(
        "--filmstrip-policy", default=None,
        help="Which arm's manoeuvre the filmstrip shows. Default: B2 if "
             "present, else the first --policy given.",
    )
    parser.add_argument(
        "--filmstrip-seed", type=int, default=None,
        help="Episode seed for the filmstrip rollout. Default: the first --seeds entry.",
    )
    parser.add_argument("--ease-gap-scale", type=float, default=1.0)
    parser.add_argument("--ease-closing-speed-scale", type=float, default=1.0)
    parser.add_argument("--ease-boundary-scale", type=float, default=1.0)
    parser.add_argument(
        "--out", type=Path,
        default=Path("artifacts") / "Results" / "HOCBF margin",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = find_repo_root()
    scenarios_by_name = {s.name: s for s in make_scenarios(10.2)}
    selected_scenarios = [] if args.no_scenarios else (args.scenario or list(SIX_SCENES))
    unknown = [n for n in selected_scenarios if n not in scenarios_by_name]
    if unknown:
        raise SystemExit(
            f"Unknown scenario(s) {unknown}; known: {sorted(scenarios_by_name)}"
        )

    # label -> [policy dicts], one per training seed of that arm.
    policies: dict[str, list[dict[str, Any]]] = {}
    for spec in args.policy:
        if "=" not in spec:
            raise SystemExit(f"--policy must be LABEL=RUN_CONFIG_JSON, got {spec!r}")
        label, config_path_str = spec.split("=", 1)
        if label not in ACTION_SOURCE:
            raise SystemExit(
                f"Unknown policy label {label!r}; expected one of {list(ACTION_SOURCE)}"
            )
        policies.setdefault(label, []).append(
            load_policy(Path(config_path_str).resolve())
        )
    check_shared_gains({
        f"{label}#{p['training_seed']}": p
        for label, group in policies.items()
        for p in group
    })

    policy_order = list(policies)
    seeds = [int(s) for s in args.seeds]
    horizon = int(args.steps)
    conditions = ([] if args.no_congested else [CONGESTED]) + selected_scenarios
    results: dict[str, dict[str, list[tuple[np.ndarray, int | None]]]] = {
        c: {} for c in conditions
    }
    series_rows: list[dict[str, Any]] = []
    policy_frequency_hz = 20.0

    for label, group in policies.items():
        action_source = ACTION_SOURCE[label]
        for policy in group:
            namespace = build_namespace(project_root, policy["cbf"])
            policy["model"] = load_model(policy["variant"], policy["model_path"], "cpu")
            policy_frequency_hz = float(policy["env_config"]["policy_frequency"])
            print(
                f"[hocbf-aggregate] {label} train_seed={policy['training_seed']} "
                f"variant={policy['variant']} action_source={action_source}",
                flush=True,
            )
            for condition in conditions:
                for seed in seeds:
                    if condition == CONGESTED:
                        series, first_collision = rollout_margin_congested(
                            namespace, policy, action_source,
                            steps=horizon, seed=seed,
                        )
                    else:
                        series, first_collision = rollout_margin(
                            namespace, policy, action_source, condition,
                            steps=horizon, seed=seed,
                            ease_gap_scale=float(args.ease_gap_scale),
                            ease_closing_speed_scale=float(
                                args.ease_closing_speed_scale
                            ),
                            ease_boundary_scale=float(args.ease_boundary_scale),
                        )
                    results[condition].setdefault(label, []).append(
                        (series, first_collision)
                    )
                    for step_index, value in enumerate(series, start=1):
                        series_rows.append({
                            "condition": condition,
                            "policy": label,
                            "training_seed": policy["training_seed"],
                            "episode_seed": seed,
                            "policy_step": step_index,
                            "time_s": step_index / policy_frequency_hz,
                            "s_hocbf": value,
                        })
                survived = sum(
                    1 for _s, fc in results[condition][label][-len(seeds):]
                    if fc is None
                )
                print(
                    f"    {condition}: survived {survived}/{len(seeds)} seeds",
                    flush=True,
                )

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Filmstrips -----------------------------------------------------
    strips: dict[str, tuple[Image.Image | None, list[float]]] = {}
    strip_label = ""
    if int(args.filmstrip_frames) > 0:
        strip_label = args.filmstrip_policy or (
            "B2_cbf_nd_reward_actor"
            if "B2_cbf_nd_reward_actor" in policies
            else policy_order[0]
        )
        if strip_label not in policies:
            raise SystemExit(
                f"--filmstrip-policy {strip_label!r} is not among the given "
                f"policies {policy_order}"
            )
        strip_policy = policies[strip_label][0]
        strip_seed = int(
            args.filmstrip_seed if args.filmstrip_seed is not None else seeds[0]
        )
        strip_namespace = build_namespace(project_root, strip_policy["cbf"])
        print(
            f"[hocbf-aggregate] filmstrip: {strip_label} "
            f"train_seed={strip_policy['training_seed']} episode_seed={strip_seed}",
            flush=True,
        )
        # The congested figure shows no simulator frame, so do not pay for one.
        for condition in [c for c in conditions if c != CONGESTED]:
            strips[condition] = capture_filmstrip(
                strip_namespace, strip_policy, ACTION_SOURCE[strip_label],
                condition,
                steps=horizon, seed=strip_seed,
                n_frames=int(args.filmstrip_frames),
                scenarios_by_name=scenarios_by_name,
                ease_gap_scale=float(args.ease_gap_scale),
                ease_closing_speed_scale=float(args.ease_closing_speed_scale),
                ease_boundary_scale=float(args.ease_boundary_scale),
            )
    strip_caption = (
        f"{strip_label} manoeuvre, raw action, external CBF off"
        if strip_label else ""
    )
    has_strips = bool(strips)

    n_roll = len(seeds) * max(len(g) for g in policies.values())

    # ---- Figure 1: congested only, one panel per file --------------------
    # The mean curve and the survival curve are read separately -- each is a
    # figure in its own right -- so they are written as separate images rather
    # than as a two-panel strip.
    if not args.no_congested:
        def congested_figure(draw_fn, subtitle: str, filename: str,
                             ylim: tuple[float, float] | None = None) -> None:
            fig = plt.figure(figsize=(7.6, 5.6))
            ax = fig.add_subplot(1, 1, 1)
            draw_fn(ax)
            ax.set_title(subtitle, fontsize=10.5, fontweight="semibold")
            if ylim is not None:
                ax.set_ylim(*ylim)
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.905),
                frameon=False, fontsize=9.5, ncols=min(len(policy_order), 3),
            )
            fig.suptitle(
                "HOCBF safety margin -- congested highway",
                fontsize=15.0, fontweight="bold", y=0.985,
            )
            fig.subplots_adjust(top=0.790, bottom=0.105, left=0.105, right=0.975)
            path = out_dir / filename
            fig.savefig(path, dpi=200, facecolor="white")
            plt.close(fig)
            print(f"wrote {path}")

        # The survival figure reports the thinning sample directly, so the mean
        # curve does not also have to encode it by fading.
        congested_figure(
            lambda ax: draw_condition(
                ax, results[CONGESTED], policy_order,
                horizon=horizon, policy_frequency_hz=policy_frequency_hz,
                censor=args.censor, band=args.band, fade_censored=False,
            ),
            f"Seed average ({n_roll} rollouts per policy)",
            "hocbf_margin_congested_mean.png",
            ylim=mean_curve_ylim(
                results[CONGESTED], policy_order,
                horizon=horizon, censor=args.censor, band=args.band,
            ),
        )
        congested_figure(
            lambda ax: draw_survival(
                ax, results[CONGESTED], policy_order,
                horizon=horizon, policy_frequency_hz=policy_frequency_hz,
                smooth_s=float(args.survival_smooth_s),
            ),
            "Collision-free rollouts",
            "hocbf_margin_congested_survival.png",
        )

    # ---- Figure 2: scenario grid ---------------------------------------
    if selected_scenarios:
        ncols = 3
        nrows = int(np.ceil(len(selected_scenarios) / ncols))
        rows_per_cell = 2 if has_strips else 1
        cell_h = 4.4 + (1.15 if has_strips else 0.0)
        fig = plt.figure(figsize=(6.4 * ncols, cell_h * nrows))
        gs = fig.add_gridspec(
            nrows * rows_per_cell, ncols,
            height_ratios=([0.8, 3.0] if has_strips else [3.0]) * nrows,
            hspace=0.55 if has_strips else 0.40,
            wspace=0.26,
        )
        first_ax = None
        for index, name in enumerate(selected_scenarios):
            r, c = divmod(index, ncols)
            if has_strips:
                draw_filmstrip(
                    fig.add_subplot(gs[r * 2, c]), *strips[name],
                    caption=scenarios_by_name[name].title,
                )
            ax = fig.add_subplot(gs[r * rows_per_cell + rows_per_cell - 1, c])
            draw_condition(
                ax, results[name], policy_order,
                horizon=horizon, policy_frequency_hz=policy_frequency_hz,
                censor=args.censor, band=args.band, fade_censored=False,
            )
            if not has_strips:
                ax.set_title(
                    scenarios_by_name[name].title,
                    fontsize=10.5, fontweight="semibold",
                )
            first_ax = first_ax or ax
        handles, labels = first_ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
            frameon=False, fontsize=9.5, ncols=len(policy_order),
        )
        fig.suptitle(
            "HOCBF safety margin -- diagnostic scenarios",
            fontsize=15.0, fontweight="bold", y=0.995,
        )
        fig.subplots_adjust(top=0.895 if has_strips else 0.88,
                            bottom=0.055, left=0.05, right=0.985)
        path = out_dir / "hocbf_margin_scenarios_mean.png"
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)
        print(f"wrote {path}")

    # ---- One standalone figure per scenario -----------------------------
    if selected_scenarios and not args.no_per_scenario:
        per_dir = out_dir / "per_scenario"
        per_dir.mkdir(parents=True, exist_ok=True)
        for name in selected_scenarios:
            heights = ([1.15] if has_strips else []) + [3.0]
            fig = plt.figure(figsize=(8.6, 5.0 + (1.5 if has_strips else 0.0)))
            gs = fig.add_gridspec(len(heights), 1, height_ratios=heights,
                                  hspace=0.26)
            offset = 0
            if has_strips:
                draw_filmstrip(fig.add_subplot(gs[0, 0]), *strips[name],
                               caption="")
                offset = 1
            ax = fig.add_subplot(gs[offset, 0])
            draw_condition(
                ax, results[name], policy_order,
                horizon=horizon, policy_frequency_hz=policy_frequency_hz,
                censor=args.censor, band=args.band, fade_censored=False,
            )
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center",
                       bbox_to_anchor=(0.5, 0.905), frameon=False,
                       fontsize=8.5, ncols=3)
            fig.suptitle(scenarios_by_name[name].title, fontsize=13.5,
                         fontweight="bold", y=0.985)
            fig.subplots_adjust(top=0.80, bottom=0.10, left=0.11, right=0.975)
            path = per_dir / f"{name}.png"
            fig.savefig(path, dpi=200, facecolor="white")
            plt.close(fig)
            print(f"wrote {path}")

    # ---- Tables ---------------------------------------------------------
    summary = summarize(
        results, policy_order,
        horizon=horizon, policy_frequency_hz=policy_frequency_hz,
    )
    summary_path = out_dir / "hocbf_margin_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {summary_path}")

    series_path = out_dir / "hocbf_margin_series.csv"
    with series_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(series_rows[0]))
        writer.writeheader()
        writer.writerows(series_rows)
    print(f"wrote {series_path} ({len(series_rows)} rows)")

    print("\n=== Summary ===")
    header = (
        "condition".ljust(38) + "policy".ljust(26) + "surv".rjust(6)
        + "mean_S".rjust(10) + "worst_S".rjust(10) + "%neg".rjust(8)
    )
    print(header)
    for row in summary:
        print(
            str(row["condition"]).ljust(38)
            + str(row["policy"]).ljust(26)
            + f"{row['survival_rate_at_horizon']:.2f}".rjust(6)
            + f"{row['mean_margin']:.3f}".rjust(10)
            + f"{row['worst_margin']:.3f}".rjust(10)
            + f"{100 * row['frac_steps_margin_negative']:.1f}".rjust(8)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
