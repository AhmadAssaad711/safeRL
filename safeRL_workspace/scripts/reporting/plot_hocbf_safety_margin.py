"""Time-varying HOCBF safety margin across the six diagnostic scenarios.

    S_HOCBF(t) = min_i [ b_i(t) - A_i(t) u_raw(t) ]

the minimum slack across every pairwise-vehicle and road-boundary HOCBF row
(psi2 = h_ddot + k1*h_dot + k0*h >= 0, rewritten as row @ a <= bound),
excluding the physical action-box rows, evaluated at the policy's *raw*
action -- never the CBF-corrected one.

This is exactly the ``cbf_hocbf_raw_min_margin`` info field
``CBFContextPhysicalActionWrapper.step()`` already computes
(scripts/common/ppo_cbf_env.py:``_hocbf_diagnostics``), from the box-clipped
action the caller passed into ``env.step()`` *before* any CBF projection --
so getting u_raw right is entirely about which action is handed to
``env.step()``, not a new derivation of A_i/b_i:

- A1/B1/B2 carry no internal projection: ``action_source="policy"`` already
  is u_raw.
- C1/C2 carry an internal differentiable projection (``projected_mean``),
  so their ``"policy"`` action is already the projected mu_safe;
  ``action_source="raw_actor_mean"`` bypasses that architecture to obtain the
  genuine raw actor mean.

Interpretation:
    S_HOCBF(t) >  0: the raw action satisfies every HOCBF condition.
    S_HOCBF(t) =  0: the worst constraint is exactly active.
    S_HOCBF(t) <  0: the raw action violates at least one HOCBF condition.
This is an HOCBF-compliance margin for the policy's own action map, not a
complete measure of collision safety (e.g. it says nothing about the
executed/shielded action, or about ex-post outcomes).

Every one of the five run_config.json files already resolves to identical
gains (k0=4.0, k1=8.5, eps_side=0.1, neighbor_range=90, max_neighbor_
constraints=7, psi1_gain=2.3) -- verified before writing this script, not
overridden here, so a stale local default cannot silently diverge from what
each checkpoint actually trained/evaluated under.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.rendering.render_policy_scenarios import find_repo_root, make_scenarios
from scripts.rendering.render_ppo_scenario_gifs import (
    SIX_SCENES,
    build_namespace,
    load_policy,
    reposition_vehicles,
    scenario_env_config,
)
from scripts.training.run_ppo_cbf_progression import (
    _predict_evaluation_action,
    load_model,
    make_ppo_cbf_env,
)

ARM_COLORS = {
    "A1_nominal": "#4472C4",
    "B1_state_cost_raw": "#ED7D31",
    "B2_cbf_nd_reward_actor": "#70AD47",
    "C1_cbf_diff_reward_only": "#9E480E",
    "C2_cbf_projected": "#997300",
    "C0_cmdp_collision": "#7030A0",
}
ACTION_SOURCE = {
    "A1_nominal": "policy",
    "B1_state_cost_raw": "policy",
    "B2_cbf_nd_reward_actor": "policy",
    "C1_cbf_diff_reward_only": "raw_actor_mean",
    "C2_cbf_projected": "raw_actor_mean",
    "C0_cmdp_collision": "policy",
}
REQUIRED_CBF_GAINS = {
    "CBF_K0": 4.0,
    "CBF_K1": 8.5,
    "CBF_EPS_SIDE": 0.1,
    "CBF_NEIGHBOR_RANGE": 90.0,
    "CBF_MAX_NEIGHBOR_CONSTRAINTS": 7,
    "CBF_PSI1_GAIN": 2.3,
}


def check_shared_gains(policies: dict[str, dict[str, Any]]) -> None:
    for label, policy in policies.items():
        cbf = policy["cbf"]
        for key, expected in REQUIRED_CBF_GAINS.items():
            actual = float(cbf[key])
            if not np.isclose(actual, float(expected)):
                raise SystemExit(
                    f"{label} has {key}={actual}, expected {expected} -- the "
                    "policies do not share the same HOCBF gains; the "
                    "resulting margins would not be comparable across arms."
                )


def rollout_margin(
    namespace: dict[str, Any],
    policy: dict[str, Any],
    action_source: str,
    scenario_name: str,
    *,
    steps: int,
    seed: int,
    ease_gap_scale: float = 1.0,
    ease_closing_speed_scale: float = 1.0,
    ease_boundary_scale: float = 1.0,
) -> tuple[np.ndarray, int | None]:
    """Return (margins, first_collision_step).

    ``terminate_on_collision`` is on for these rollouts: a collision ends the
    episode immediately (``terminated=True``), so nothing post-collision is
    ever simulated or fed back into the margin computation.
    ``first_collision_step`` still lets the caller mark the plotted line's
    endpoint with a collision marker.
    """

    scenario = {s.name: s for s in make_scenarios(10.2)}[scenario_name]
    env_config = scenario_env_config(policy["env_config"], scenario, steps)
    env_config["terminate_on_collision"] = True
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
        env.reset(seed=int(seed))
        reposition_vehicles(
            env, scenario,
            ease_gap_scale=ease_gap_scale,
            ease_closing_speed_scale=ease_closing_speed_scale,
            ease_boundary_scale=ease_boundary_scale,
        )
        # Bootstrap step (zero action) only refreshes the observation after
        # directly mutating vehicle state; its margin is not policy behavior
        # and is not recorded.
        obs, _reward, terminated, truncated, _info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        for step_index in range(1, int(steps) + 1):
            if terminated or truncated:
                break
            action = _predict_evaluation_action(
                policy["model"], obs, action_source=action_source
            )
            obs, _reward, terminated, truncated, info = env.step(action)
            margins.append(float(info["cbf_hocbf_raw_min_margin"]))
            if first_collision_step is None and int(info.get("ego_collision_events", 0)) > 0:
                first_collision_step = step_index
    finally:
        env.close()
    return np.asarray(margins, dtype=float), first_collision_step


def plot_scenario_grid(
    margins: dict[str, dict[str, list[tuple[np.ndarray, int | None]]]],
    scenario_names: Sequence[str],
    scenario_titles: dict[str, str],
    policy_order: Sequence[str],
    *,
    policy_frequency_hz: float,
    out_path: Path,
    n_seeds: int,
) -> None:
    ncols = 3
    nrows = int(np.ceil(len(scenario_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.6 * nrows), squeeze=False)
    flat_axes = list(axes.flat)
    for scenario_name, ax in zip(scenario_names, flat_axes):
        for label in policy_order:
            runs = margins[scenario_name].get(label, [])
            for run_index, (series, first_collision_step) in enumerate(runs):
                if series.size == 0:
                    continue
                # terminate_on_collision is on: the environment itself stops
                # the episode at the collision step, so ``series`` already
                # ends there -- nothing post-collision is ever recorded.
                t = np.arange(1, series.size + 1) / policy_frequency_hz
                ax.plot(
                    t, series, color=ARM_COLORS[label], linewidth=1.1, alpha=0.55,
                    label=label if run_index == 0 else None,
                )
                if first_collision_step is not None:
                    ax.plot(
                        first_collision_step / policy_frequency_hz, series[-1],
                        marker="x", markersize=7, markeredgewidth=1.8,
                        color=ARM_COLORS[label], alpha=0.7, zorder=5,
                    )
        ax.axhline(0.0, color="#333333", linewidth=1.2, linestyle="--", zorder=1)
        ax.set_title(scenario_titles.get(scenario_name, scenario_name), fontsize=10.5, fontweight="semibold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$S_{HOCBF}(t)$")
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncols=1, fontsize=9)
    for ax in flat_axes[len(scenario_names):]:
        ax.axis("off")
    fig.suptitle(
        r"HOCBF safety margin $S_{HOCBF}(t)=\min_i[b_i(t)-A_i(t)u_{raw}(t)]$ "
        "-- raw policy action, box constraints excluded\n"
        rf"{n_seeds} scenario seeds per policy (thin lines, each ending at its own "
        r"collision $\times$ if any); terminate_on_collision is on",
        fontsize=12.5, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        metavar="LABEL=RUN_CONFIG_JSON",
        help="One policy to include, repeatable. LABEL must be a key of ACTION_SOURCE.",
    )
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[307, 327, 347, 367, 387, 407, 427],
        help="Scenario seeds to roll out per policy per scenario (varies background-vehicle "
             "personality draw, not the scripted geometry). Multiple seeds are all plotted "
             "as separate thin lines so a single noisy draw can't be mistaken for the policy's "
             "typical behavior.",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "reports" / "safety_matrix_1M" / "figures")
    parser.add_argument(
        "--ease-gap-scale", type=float, default=1.0,
        help="Scale every non-ego vehicle's longitudinal offset from ego by this factor.",
    )
    parser.add_argument(
        "--ease-closing-speed-scale", type=float, default=1.0,
        help="Scale every non-ego vehicle's initial velocity gap from ego by this factor.",
    )
    parser.add_argument(
        "--ease-boundary-scale", type=float, default=1.0,
        help="Pull ego's initial y toward the road's lateral center by this factor.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = find_repo_root()
    scenarios_by_name = {scenario.name: scenario for scenario in make_scenarios(10.2)}
    selected_scenarios = args.scenario or SIX_SCENES
    unknown = [name for name in selected_scenarios if name not in scenarios_by_name]
    if unknown:
        raise SystemExit(f"Unknown scenario(s) {unknown}; known: {sorted(scenarios_by_name)}")
    scenario_titles = {name: scenarios_by_name[name].title for name in selected_scenarios}

    policies: dict[str, dict[str, Any]] = {}
    for spec in args.policy:
        if "=" not in spec:
            raise SystemExit(f"--policy must be LABEL=RUN_CONFIG_JSON, got {spec!r}")
        label, config_path_str = spec.split("=", 1)
        if label not in ACTION_SOURCE:
            raise SystemExit(f"Unknown policy label {label!r}; expected one of {list(ACTION_SOURCE)}")
        policies[label] = load_policy(Path(config_path_str).resolve())
    check_shared_gains(policies)

    seeds = [int(s) for s in args.seeds]
    policy_frequency_hz = None
    margins: dict[str, dict[str, list[tuple[np.ndarray, int | None]]]] = {
        name: {} for name in selected_scenarios
    }
    survival: dict[str, dict[str, list[bool]]] = {name: {} for name in selected_scenarios}
    for label, policy in policies.items():
        namespace = build_namespace(project_root, policy["cbf"])
        policy["model"] = load_model(policy["variant"], policy["model_path"], "cpu")
        policy_frequency_hz = float(policy["env_config"]["policy_frequency"])
        action_source = ACTION_SOURCE[label]
        print(
            f"[hocbf-margin] {label}: variant={policy['variant']} "
            f"seed={policy['training_seed']} action_source={action_source}",
            flush=True,
        )
        for scenario_name in selected_scenarios:
            runs: list[tuple[np.ndarray, int | None]] = []
            survived_flags: list[bool] = []
            for seed in seeds:
                series, first_collision_step = rollout_margin(
                    namespace, policy, action_source, scenario_name,
                    steps=int(args.steps), seed=seed,
                    ease_gap_scale=float(args.ease_gap_scale),
                    ease_closing_speed_scale=float(args.ease_closing_speed_scale),
                    ease_boundary_scale=float(args.ease_boundary_scale),
                )
                runs.append((series, first_collision_step))
                survived_flags.append(first_collision_step is None)
            margins[scenario_name][label] = runs
            survival[scenario_name][label] = survived_flags
            n_survived = sum(survived_flags)
            worsts = [float(np.min(s)) for s, _ in runs if s.size]
            collision_times = [
                fc / policy_frequency_hz for s, fc in runs if fc is not None
            ]
            worst_str = f"{min(worsts):.2f}" if worsts else "nan"
            ct_str = f"{np.mean(collision_times):.2f}s" if collision_times else "n/a"
            print(
                f"  {scenario_name}: survived {n_survived}/{len(seeds)} seeds, "
                f"worst_margin={worst_str}, mean_collision_time={ct_str}",
                flush=True,
            )

    plot_scenario_grid(
        margins, selected_scenarios, scenario_titles, list(policies),
        policy_frequency_hz=policy_frequency_hz or 20.0,
        out_path=args.out / "hocbf_safety_margin_all_scenarios.png",
        n_seeds=len(seeds),
    )

    print("\n=== Survival summary (out of {} seeds) ===".format(len(seeds)))
    header = "Scenario".ljust(36) + "".join(label.ljust(26) for label in policies)
    print(header)
    for scenario_name in selected_scenarios:
        row = scenario_name.ljust(36)
        for label in policies:
            n = sum(survival[scenario_name].get(label, []))
            row += f"{n}/{len(seeds)}".ljust(26)
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
