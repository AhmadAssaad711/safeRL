"""Time-varying HOCBF safety margin on the real 40-vehicle highway traffic
environment (the actual training/evaluation distribution), not the six
scripted diagnostic scenes.

    S_HOCBF(t) = min_i [ b_i(t) - A_i(t) u_raw(t) ]

CBF-off for every arm: mode="raw" (no external filter), and for C1/C2 --
which carry an internal projection that stays active even under mode="raw"
-- action_source="raw_actor_mean" to bypass that architecture and obtain the
genuine unprotected action, same mapping used in ``plot_hocbf_safety_margin.py``.

Caveat (established earlier): B2/C1/C2 all train with ``execution_mode="cbf"``
(filter/projection always active during training rollouts), so this raw
condition is out-of-distribution for them -- their raw behavior was never
directly optimized. A1/B1 train with ``execution_mode="box"`` (raw dynamics
only), so raw is exactly their trained regime. This is not a fair ranking of
"which policy is safest" -- it's each policy's behavior with its own safety
mechanism removed, on the real highway distribution instead of six scripted
scenes.

Multiple episode seeds are rolled out per policy and all plotted as separate
thin lines (never averaged into one line), so a single lucky/unlucky episode
can't be mistaken for typical behavior.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.rendering.render_ppo_scenario_gifs import build_namespace, find_repo_root, load_policy
from scripts.training.run_ppo_cbf_progression import (
    _predict_evaluation_action,
    load_model,
    make_evaluation_env,
)

ARM_COLORS = {
    "A1_nominal": "#4472C4",
    "B1_state_cost_raw": "#ED7D31",
    "B2_cbf_nd_reward_actor": "#70AD47",
    "C1_cbf_diff_reward_only": "#9E480E",
    "C2_cbf_projected": "#997300",
}
ACTION_SOURCE = {
    "A1_nominal": "policy",
    "B1_state_cost_raw": "policy",
    "B2_cbf_nd_reward_actor": "policy",
    "C1_cbf_diff_reward_only": "raw_actor_mean",
    "C2_cbf_projected": "raw_actor_mean",
}

_SAFETY_MATRIX_DIR = Path.home() / "AppData" / "Local" / "Temp" / "saferl_runs" / "safety_matrix_1M"
_LAMBDA_DETACHED_DIR = Path.home() / "AppData" / "Local" / "Temp" / "saferl_runs" / "lambda_detached_actor_1p0"
_LAMBDA_MEAN_DIR = Path.home() / "AppData" / "Local" / "Temp" / "saferl_runs" / "lambda_mean_1p0"

# Each arm's own best training seed (lowest raw-mode collisions/km), not a shared default.
RUN_CONFIGS = {
    "A1_nominal": _SAFETY_MATRIX_DIR / "A1_nominal" / "seed_347" / "ppo_nominal" / "seed_347" / "run_config.json",
    "B1_state_cost_raw": _SAFETY_MATRIX_DIR / "B1_state_cost_raw" / "seed_367" / "ppo_hocbf_state_cost_raw" / "seed_367" / "run_config.json",
    "B2_cbf_nd_reward_actor": _LAMBDA_DETACHED_DIR / "ldac1p0_s347" / "ppo_cbf_nd_reward_actor" / "seed_347" / "run_config.json",
    "C1_cbf_diff_reward_only": _SAFETY_MATRIX_DIR / "C1_cbf_diff_reward_only" / "seed_327" / "ppo_cbf_diff_reward_only" / "seed_327" / "run_config.json",
    "C2_cbf_projected": _LAMBDA_MEAN_DIR / "lam1p0_s347" / "ppo_cbf_projected" / "seed_347" / "run_config.json",
}


def rollout_margin(
    namespace: dict[str, Any],
    policy: dict[str, Any],
    action_source: str,
    *,
    steps: int,
    episode_seed: int,
) -> tuple[np.ndarray, int | None]:
    env = make_evaluation_env(
        namespace,
        mode="raw",
        env_config=policy["env_config"],
        reward_config=policy["reward_config"],
        correction_epsilon=policy["correction_epsilon"],
    )
    margins: list[float] = []
    first_collision_step: int | None = None
    terminated = False
    truncated = False
    try:
        obs, _ = env.reset(seed=int(episode_seed))
        for step_index in range(1, int(steps) + 1):
            if terminated or truncated:
                break
            action = _predict_evaluation_action(policy["model"], obs, action_source=action_source)
            obs, _reward, terminated, truncated, info = env.step(action)
            margins.append(float(info["cbf_hocbf_raw_min_margin"]))
            if first_collision_step is None and int(info.get("ego_collision_events", 0)) > 0:
                first_collision_step = step_index
    finally:
        env.close()
    return np.asarray(margins, dtype=float), first_collision_step


def plot_highway_grid(
    margins: dict[str, list[tuple[np.ndarray, int | None]]],
    policy_order: Sequence[str],
    *,
    policy_frequency_hz: float,
    n_seeds: int,
    out_path: Path,
    n_resample_points: int = 300,
) -> None:
    """Plot the mean +/- SD margin curve per policy across its episode seeds.

    Episodes end at different times (collision or truncation), so each is
    resampled onto a shared grid up to the shortest-surviving episode's
    length for that policy before averaging -- the same convention used
    elsewhere in this project (resample_mean in the training-diagnostic
    scripts) rather than padding/extrapolating past a real collision.
    """
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    for label in policy_order:
        runs = margins.get(label, [])
        usable = [series for series, _ in runs if series.size > 1]
        if not usable:
            continue
        min_len = min(s.size for s in usable)
        grid_steps = np.linspace(1, min_len, n_resample_points)
        stacked = np.vstack([
            np.interp(grid_steps, np.arange(1, s.size + 1), s) for s in usable
        ])
        mean = stacked.mean(axis=0)
        sd = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)
        t = grid_steps / policy_frequency_hz
        color = ARM_COLORS[label]
        n_survived = sum(1 for _, fc in runs if fc is None)
        ax.plot(t, mean, color=color, linewidth=2.4,
                 label=f"{label} ({n_survived}/{len(runs)} survived to {min_len} steps)")
        ax.fill_between(t, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
    ax.axhline(0.0, color="#333333", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$S_{HOCBF}(t)$")
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(
        r"HOCBF safety margin $S_{HOCBF}(t)=\min_i[b_i(t)-A_i(t)u_{raw}(t)]$ "
        "-- real 40-vehicle highway traffic, CBF OFF for every arm (raw action; "
        "raw-actor-mean for C1/C2)\n"
        rf"mean $\pm$ SD across {n_seeds} episode seeds per policy, each policy's own "
        "best training seed; curve stops at the shortest-surviving episode's length\n"
        "B2/C1/C2 never trained on this raw regime -- see script docstring.",
        fontsize=11.5, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument(
        "--episode-seeds", type=int, nargs="+",
        default=[1_100_000 + i for i in range(20)],
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "Results" / "highway")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    project_root = find_repo_root()

    policies: dict[str, dict[str, Any]] = {}
    margins: dict[str, list[tuple[np.ndarray, int | None]]] = {}
    survival: dict[str, list[bool]] = {}
    policy_frequency_hz = None

    for label, run_config_path in RUN_CONFIGS.items():
        policy = load_policy(run_config_path)
        namespace = build_namespace(project_root, policy["cbf"])
        policy["model"] = load_model(policy["variant"], policy["model_path"], "cpu")
        policies[label] = policy
        policy_frequency_hz = float(policy["env_config"]["policy_frequency"])
        print(
            f"[highway-margin] {label}: variant={policy['variant']} "
            f"seed={policy['training_seed']}",
            flush=True,
        )
        action_source = ACTION_SOURCE[label]
        runs: list[tuple[np.ndarray, int | None]] = []
        survived_flags: list[bool] = []
        for episode_seed in args.episode_seeds:
            series, first_collision_step = rollout_margin(
                namespace, policy, action_source, steps=int(args.steps), episode_seed=int(episode_seed)
            )
            runs.append((series, first_collision_step))
            survived_flags.append(first_collision_step is None)
        margins[label] = runs
        survival[label] = survived_flags
        n_survived = sum(survived_flags)
        worsts = [float(np.min(s)) for s, _ in runs if s.size]
        collision_times = [fc / policy_frequency_hz for _, fc in runs if fc is not None]
        worst_str = f"{min(worsts):.2f}" if worsts else "nan"
        ct_str = f"{np.mean(collision_times):.2f}s" if collision_times else "n/a"
        print(
            f"  survived {n_survived}/{len(args.episode_seeds)} seeds, "
            f"worst_margin={worst_str}, mean_collision_time={ct_str}",
            flush=True,
        )

    plot_highway_grid(
        margins, list(policies),
        policy_frequency_hz=policy_frequency_hz or 20.0,
        n_seeds=len(args.episode_seeds),
        out_path=args.out / "hocbf_safety_margin_highway.png",
    )

    print(f"\n=== Survival summary (out of {len(args.episode_seeds)} seeds), CBF OFF for every arm ===")
    for label in policies:
        n = sum(survival[label])
        print(f"  {label}: {n}/{len(args.episode_seeds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
