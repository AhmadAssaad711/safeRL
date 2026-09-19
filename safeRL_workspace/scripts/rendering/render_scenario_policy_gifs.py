"""Render all 5 safety-matrix policies (A1, B1, B2, C1, C2), CBF on/off (plus
the C1/C2 raw-actor-mean true-unprotected ablation), across the six canonical
scripted diagnostic scenarios (see docs/diagnostic_scenarios.md).

Companion to ``render_policy_traffic_gifs.py`` (same policies/conditions, but
against the real random 40-vehicle traffic environment instead of these six
fixed scripted scenes). Each scenario is kept as its own separate GIF file
(never concatenated) so individual maneuvers stay easy to inspect.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from scripts.rendering.render_policy_scenarios import find_repo_root, make_scenarios
from scripts.rendering.render_ppo_scenario_gifs import (
    build_namespace,
    load_policy,
    render_scenario_gif,
)
from scripts.training.run_ppo_cbf_progression import load_model

SIX_SCENES = [
    "safe_overtake_open_upper_gap",
    "unsafe_overtake_fast_closing_upper",
    "boxed_in_hold_position",
    "rear_pressure_escape",
    "boundary_recovery_no_upper_squeeze",
    "sudden_lead_slowdown",
]

# Each entry: (policy_key, mode, action_source, output subfolder)
CONDITION_PLAN: list[tuple[str, str, str, str]] = [
    ("A1", "raw", "policy", "A1_nominal/cbf_off"),
    ("A1", "cbf", "policy", "A1_nominal/cbf_on"),
    ("B1", "raw", "policy", "B1_state_cost_raw/cbf_off"),
    ("B1", "cbf", "policy", "B1_state_cost_raw/cbf_on"),
    ("B2", "raw", "policy", "B2_lambda1p0/cbf_off"),
    ("B2", "cbf", "policy", "B2_lambda1p0/cbf_on"),
    ("C1", "raw", "policy", "C1_baseline/cbf_off_literal_projected"),
    ("C1", "cbf", "policy", "C1_baseline/cbf_on"),
    ("C1", "raw", "raw_actor_mean", "C1_baseline/cbf_off_raw_actor_mean"),
    ("C2", "raw", "policy", "C2_lambda1p0/cbf_off_literal_projected"),
    ("C2", "cbf", "policy", "C2_lambda1p0/cbf_on"),
    ("C2", "raw", "raw_actor_mean", "C2_lambda1p0/cbf_off_raw_actor_mean"),
]

_SAFETY_MATRIX_DIR = Path.home() / "AppData" / "Local" / "Temp" / "saferl_runs" / "safety_matrix_1M"

RUN_CONFIGS = {
    "A1": _SAFETY_MATRIX_DIR / "A1_nominal" / "seed_307" / "ppo_nominal" / "seed_307" / "run_config.json",
    "B1": _SAFETY_MATRIX_DIR / "B1_state_cost_raw" / "seed_307" / "ppo_hocbf_state_cost_raw" / "seed_307" / "run_config.json",
    "B2": Path.home() / "AppData" / "Local" / "Temp" / "saferl_runs" / "lambda_detached_actor_1p0"
    / "ldac1p0_s307" / "ppo_cbf_nd_reward_actor" / "seed_307" / "run_config.json",
    "C1": _SAFETY_MATRIX_DIR / "C1_cbf_diff_reward_only" / "seed_307" / "ppo_cbf_diff_reward_only" / "seed_307" / "run_config.json",
    "C2": Path.home() / "AppData" / "Local" / "Temp" / "saferl_runs" / "lambda_mean_1p0"
    / "lam1p0_s307" / "ppo_cbf_projected" / "seed_307" / "run_config.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ease-gap-scale", type=float, default=1.0)
    parser.add_argument("--ease-closing-speed-scale", type=float, default=1.0)
    parser.add_argument("--ease-boundary-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    project_root = (args.project_root or find_repo_root()).resolve()
    scenarios_by_name = {scenario.name: scenario for scenario in make_scenarios(10.2)}
    selected_names = args.scenario or SIX_SCENES
    unknown = [name for name in selected_names if name not in scenarios_by_name]
    if unknown:
        raise SystemExit(f"Unknown scenario(s) {unknown}; known: {sorted(scenarios_by_name)}")

    policies: dict[str, dict[str, Any]] = {}
    summary: list[dict[str, Any]] = []
    for policy_key, run_config_path in RUN_CONFIGS.items():
        policy = load_policy(run_config_path)
        namespace = build_namespace(project_root, policy["cbf"])
        policy["model"] = load_model(policy["variant"], policy["model_path"], "cpu")
        policy["namespace"] = namespace
        policies[policy_key] = policy
        print(
            f"[scenario-gif] loaded {policy_key}: variant={policy['variant']} "
            f"seed={policy['training_seed']} model={policy['model_path']}",
            flush=True,
        )

    for policy_key, mode, action_source, subfolder in CONDITION_PLAN:
        policy = policies[policy_key]
        for name in selected_names:
            scenario = scenarios_by_name[name]
            out_path = args.out_dir / subfolder / f"{name}.gif"
            frame_count = render_scenario_gif(
                policy["namespace"],
                policy,
                scenario,
                steps=int(args.steps),
                seed=int(args.seed),
                fps=int(args.fps),
                out_path=out_path,
                mode=mode,
                action_source=action_source,
                ease_gap_scale=float(args.ease_gap_scale),
                ease_closing_speed_scale=float(args.ease_closing_speed_scale),
                ease_boundary_scale=float(args.ease_boundary_scale),
            )
            record = {
                "policy": policy_key,
                "mode": mode,
                "action_source": action_source,
                "scenario": name,
                "frames": frame_count,
                "policy_steps_rendered": frame_count - 2,  # minus the two bootstrap frames
                "out_path": str(out_path),
            }
            summary.append(record)
            print(f"  {subfolder}/{name}.gif: {frame_count} frames", flush=True)

    (args.out_dir / "render_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
