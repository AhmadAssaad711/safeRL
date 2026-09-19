"""Render the six canonical diagnostic scenarios against saved PPO/CBF policies.

The six scenes are defined in ``docs/diagnostic_scenarios.md`` (source: E.2 of
lanelessKaralakou.ipynb) and were previously only wired to the three legacy
DDPG policies via ``render_policy_scenarios.py``. This module reuses that
file's scenario geometry (``make_scenarios``, ``apply_scenario``) but builds
the environment and loads the model through
``scripts.training.run_ppo_cbf_progression``, so a PPO/CBF-progression
checkpoint (nominal, state-cost, CBF-reward, differentiable-projection, ...)
gets the exact wrapper stack, CBF gains, and reward config it was trained and
evaluated with -- read from that run's own ``run_config.json``, never
reconstructed from scratch.

One GIF per (policy, scenario), external CBF OFF (mode "raw", i.e. the
deployed policy's own behavior, matching how these checkpoints are compared
throughout the safety-matrix study).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import scripts.training.run_cbf_filter_ablation as protocol
from scripts.rendering.render_policy_scenarios import (
    ScenarioSpec,
    find_repo_root,
    make_scenarios,
)
from scripts.training.run_ppo_cbf_progression import (
    _predict_evaluation_action,
    load_model,
    make_ppo_cbf_env,
)

SIX_SCENES = [
    "safe_overtake_open_upper_gap",
    "unsafe_overtake_fast_closing_upper",
    "boxed_in_hold_position",
    "rear_pressure_escape",
    "boundary_recovery_no_upper_squeeze",
    "sudden_lead_slowdown",
]


def load_policy(run_config_path: Path) -> dict[str, Any]:
    payload = json.loads(run_config_path.read_text(encoding="utf-8"))
    signature = payload["training_signature"]
    return {
        "variant": str(payload["variant"]),
        "training_seed": int(payload["training_seed"]),
        "label": str(payload["variant_spec"]["label"]),
        "env_config": copy.deepcopy(signature["env_config"]),
        "reward_config": copy.deepcopy(payload["reward_config"]),
        "cbf": dict(signature["cbf"]),
        "correction_epsilon": float(signature.get("correction_epsilon", 0.03)),
        "model_path": run_config_path.parent / "model_final.zip",
    }


def build_namespace(project_root: Path, cbf: dict[str, Any]) -> dict[str, Any]:
    namespace = protocol.bootstrap_notebook_namespace(project_root)
    protocol.exec_required_notebook_cells(
        project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    namespace["DEVICE"] = "cpu"
    namespace["CBF_K0"] = float(cbf["CBF_K0"])
    namespace["CBF_K1"] = float(cbf["CBF_K1"])
    namespace["CBF_EPS_SIDE"] = float(cbf["CBF_EPS_SIDE"])
    namespace["CBF_NEIGHBOR_RANGE"] = float(cbf["CBF_NEIGHBOR_RANGE"])
    namespace["CBF_MAX_NEIGHBOR_CONSTRAINTS"] = int(cbf["CBF_MAX_NEIGHBOR_CONSTRAINTS"])
    namespace["CBF_PSI1_GAIN"] = float(cbf["CBF_PSI1_GAIN"])
    namespace["CBF_AX_BOUNDS"] = tuple(cbf["CBF_AX_BOUNDS"])
    namespace["CBF_AY_BOUNDS"] = tuple(cbf["CBF_AY_BOUNDS"])
    return namespace


def reposition_vehicles(
    env: Any,
    scenario: ScenarioSpec,
    ego_x: float = 100.0,
    *,
    ease_gap_scale: float = 1.0,
    ease_closing_speed_scale: float = 1.0,
    ease_boundary_scale: float = 1.0,
) -> None:
    """Place the scenario's fixed vehicles, mirroring render_policy_scenarios.apply_scenario.

    That function's own observation reconstruction (``policy_observation``,
    a generic ``_augment_observation`` wrapper walk) does not cover this
    PPO stack: the reward wrapper builds the target-y and previous-action
    features inside its own ``reset``/``step``, not a separately named hook,
    so a manual walk silently skips it and undercounts the observation
    width. This function only repositions vehicles; the caller obtains a
    valid observation by taking a real step afterward.

    These scenes were tuned for the legacy DDPG diagnostic path under the
    old (complex-root, softer) HOCBF gains. Under the current cascade
    (k0=4.0, k1=8.5), several of them start with S_HOCBF already deep
    negative at t=0 -- e.g. a fast-closing hazard placed only ~14 m away at
    +10 m/s relative speed gives under 1.5 s to contact before the policy
    ever gets to act, which the larger k1 amplifies heavily through h_dot.
    ``ease_gap_scale`` widens every non-ego vehicle's longitudinal offset
    from ego (proportionally, so relative geometry/intent is preserved);
    ``ease_closing_speed_scale`` shrinks each non-ego vehicle's velocity gap
    from ego's initial 20 m/s. Neither one touches "boundary recovery" scenes
    (their danger is ego's own initial distance to the road edge, not another
    vehicle's dx/vx), so ``ease_boundary_scale`` separately pulls the ego's
    initial y toward the road's lateral center (1.0 = unchanged, 0.0 = exact
    center). All three default to 1.0 (scenario unchanged). This function is
    PPO-path-only and never touches render_policy_scenarios.make_scenarios,
    so the legacy DDPG scenes stay exactly as originally tuned.
    """

    base = env.unwrapped
    road_length = float(base.config["road_length"])
    road_width = float(base.config["road_width"])
    center_y = 0.5 * road_width
    vehicles = list(base.road.vehicles)
    if len(vehicles) < len(scenario.vehicles):
        raise RuntimeError(
            f"Environment has {len(vehicles)} vehicles but scenario needs {len(scenario.vehicles)}"
        )
    ego_vx = float(scenario.vehicles[0].vx)
    ego_y0 = float(scenario.vehicles[0].y)
    # Shift the whole lateral formation by however far easing would move ego
    # toward center, not just ego -- moving only ego toward center changes
    # its distance to every other vehicle's y band as a side effect (it can
    # walk ego straight into a "hazard" lane that was positioned assuming
    # ego stays at the edge). Translating everyone together preserves every
    # inter-vehicle relative gap exactly and only changes ego's distance to
    # the boundary.
    y_shift = (center_y + (ego_y0 - center_y) * float(ease_boundary_scale)) - ego_y0
    for index, (vehicle, spec) in enumerate(zip(vehicles, scenario.vehicles)):
        dx = float(spec.dx)
        vx = float(spec.vx)
        y = float(spec.y) + y_shift
        if index > 0:
            dx *= float(ease_gap_scale)
            vx = ego_vx + (vx - ego_vx) * float(ease_closing_speed_scale)
        vehicle.position[0] = float((ego_x + dx) % road_length)
        vehicle.position[1] = y
        vehicle.vx = float(vx)
        vehicle.vy = float(spec.vy)
        vehicle.length = float(spec.length)
        vehicle.width = float(spec.width)
        vehicle.LENGTH = vehicle.length
        vehicle.WIDTH = vehicle.width
        vehicle.desired_speed = float(
            spec.desired_speed if spec.desired_speed is not None else max(vx, 1.0)
        )
        vehicle.is_ego = index == 0
        vehicle.crashed = False
        vehicle.hit = False
        vehicle._sync_graphics_fields()
    base.vehicle = vehicles[0]
    base.controlled_vehicles = [vehicles[0]]
    base._last_action = np.zeros(2, dtype=np.float32)
    base._last_accelerations = np.zeros((len(vehicles), 2), dtype=float)
    base._last_boundary_violations = 0
    base._last_collision_count = 0
    base._last_active_collision_count = 0
    base._last_ego_collision_count = 0
    base._last_ego_collision = False
    base._cumulative_collision_count = 0
    base._active_collision_pairs = set()
    base._flow_count = 0


def scenario_env_config(
    base_env_config: dict[str, Any], scenario: ScenarioSpec, steps: int
) -> dict[str, Any]:
    config = copy.deepcopy(base_env_config)
    # The scenario positions only its own len(scenario.vehicles) cars; the rest
    # keep their randomly (safely) spawned positions as distant background
    # traffic. This must be at least neighbors_count + 1 (ego + neighbors), or
    # the base observation comes out narrower than the model's declared
    # observation_space -- fewer real vehicles than neighbor slots is an edge
    # case the environment's neighbor padding does not fully cover.
    min_vehicles = int(base_env_config.get("neighbors_count", 0)) + 1
    config["vehicles_count"] = max(len(scenario.vehicles), min_vehicles)
    # episode_steps/duration are physics-step counts, not policy-step counts:
    # simulation_frequency/policy_frequency physics substeps run per env.step()
    # call (5 at the canonical 100/20 Hz). Convert the requested policy-step
    # budget (plus a buffer, plus the one bootstrap step) so the rollout never
    # truncates on the horizon itself -- only collision or the fixed step loop
    # ends a clip.
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
    config["terminate_on_collision"] = False
    config["screen_width"] = 1200
    config["screen_height"] = 320
    config["centering_position"] = [0.5, 0.5]
    config["scaling"] = 9.8
    config["offscreen_rendering"] = True
    config["real_time_rendering"] = False
    return config


def render_scenario_gif(
    namespace: dict[str, Any],
    policy: dict[str, Any],
    scenario: ScenarioSpec,
    *,
    steps: int,
    seed: int,
    fps: int,
    out_path: Path,
    mode: str = "raw",
    action_source: str = "policy",
    ease_gap_scale: float = 1.0,
    ease_closing_speed_scale: float = 1.0,
    ease_boundary_scale: float = 1.0,
) -> int:
    env_config = scenario_env_config(policy["env_config"], scenario, steps)
    env = make_ppo_cbf_env(
        namespace,
        env_config=env_config,
        reward_config=policy["reward_config"],
        project_inputs=mode == "cbf",
        lambda_delta=0.0,
        lambda_intervention=0.0,
        correction_epsilon=policy["correction_epsilon"],
        monitor_path=None,
    )
    env.unwrapped.render_mode = "rgb_array"
    env.unwrapped.config["offscreen_rendering"] = True

    frames: list[Image.Image] = []
    try:
        env.reset(seed=int(seed))
        reposition_vehicles(
            env, scenario,
            ease_gap_scale=ease_gap_scale,
            ease_closing_speed_scale=ease_closing_speed_scale,
            ease_boundary_scale=ease_boundary_scale,
        )
        frames.append(Image.fromarray(np.asarray(env.render(), dtype=np.uint8)))
        # A neutral bootstrap step is the only way to obtain a validly
        # wrapped observation after directly mutating vehicle state (see
        # reposition_vehicles); it does not use the policy and is not
        # counted against --steps.
        obs, _reward, terminated, truncated, _info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        frames.append(Image.fromarray(np.asarray(env.render(), dtype=np.uint8)))
        for _ in range(int(steps)):
            if terminated or truncated:
                break
            action = _predict_evaluation_action(
                policy["model"], obs, action_source=action_source
            )
            obs, _reward, terminated, truncated, _info = env.step(action)
            frames.append(Image.fromarray(np.asarray(env.render(), dtype=np.uint8)))
    finally:
        env.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        optimize=True,
    )
    return len(frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        metavar="LABEL=RUN_CONFIG_JSON",
        help="One policy to render, repeatable. LABEL is used for the output subfolder.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Scenario name to render. Repeat for multiple. Default: the six canonical scenes.",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--ease-gap-scale", type=float, default=1.0,
        help="Scale every non-ego vehicle's longitudinal offset from ego by this factor "
             "(>1 widens gaps). These scenes were tuned under older, softer HOCBF gains; "
             "under the current cascade several start already deep HOCBF-negative at t=0.",
    )
    parser.add_argument(
        "--ease-closing-speed-scale", type=float, default=1.0,
        help="Scale every non-ego vehicle's initial velocity gap from ego by this factor "
             "(<1 shrinks closing speed toward ego's own speed).",
    )
    parser.add_argument(
        "--ease-boundary-scale", type=float, default=1.0,
        help="Pull ego's initial y toward the road's lateral center by this factor "
             "(1.0 = unchanged, 0.0 = exact center). Only this affects boundary-recovery scenes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    project_root = find_repo_root()
    scenarios_by_name = {scenario.name: scenario for scenario in make_scenarios(10.2)}
    selected_names = args.scenario or SIX_SCENES
    unknown = [name for name in selected_names if name not in scenarios_by_name]
    if unknown:
        raise SystemExit(f"Unknown scenario(s) {unknown}; known: {sorted(scenarios_by_name)}")

    for spec in args.policy:
        if "=" not in spec:
            raise SystemExit(f"--policy must be LABEL=RUN_CONFIG_JSON, got {spec!r}")
        label, config_path_str = spec.split("=", 1)
        policy = load_policy(Path(config_path_str).resolve())
        namespace = build_namespace(project_root, policy["cbf"])
        policy["model"] = load_model(
            policy["variant"], policy["model_path"], "cpu"
        )
        print(
            f"[ppo-scenario-gif] {label}: variant={policy['variant']} "
            f"seed={policy['training_seed']} model={policy['model_path']}",
            flush=True,
        )
        for name in selected_names:
            scenario = scenarios_by_name[name]
            out_path = args.out_dir / label / f"{name}.gif"
            frame_count = render_scenario_gif(
                namespace,
                policy,
                scenario,
                steps=int(args.steps),
                seed=int(args.seed),
                fps=int(args.fps),
                out_path=out_path,
                ease_gap_scale=float(args.ease_gap_scale),
                ease_closing_speed_scale=float(args.ease_closing_speed_scale),
                ease_boundary_scale=float(args.ease_boundary_scale),
            )
            print(f"  {name}: {frame_count} frames -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
