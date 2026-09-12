"""Counterfactual replay of the reward's speed target on a finished PPO run.

Read-only diagnostic.  It replays a trained policy deterministically and, at
every policy step, records both

* the speed target the reward actually used (the fixed nominal
  ``ego_desired_speed``, read from ``info["karalakou_target_speed"]``), and
* the speed target a *feasibility-capped* rule would have produced from the
  same post-step traffic state.

Nothing is trained and no reward is changed; the executed actions are exactly
those of the source run, so the recorded trajectories are the real ones.  The
question the output answers is whether the proposed rule would bind often
enough, and by enough, to matter.

The proposed rule caps the free-flow target by a constant-time-headway speed
for each vehicle ahead, weighted by a smooth lateral-relevance kernel and
combined with a softmin so that the target is continuous in the traffic
configuration:

    v_allow_i = max((dx_i - d0) / T, 0)
    w_i       = sigmoid( -( |dy_i| - 0.5*(W_ego + W_i) ) / sigma_lat )
    v_eff_i   = w_i * v_allow_i + (1 - w_i) * v_nom
    v_target  = min( v_nom, min_i v_eff_i )

The lateral weight is applied *inside* each vehicle's allowance, blending it
toward "no constraint", rather than as a weight on an aggregation.  A car that
is laterally irrelevant therefore has ``v_eff_i -> v_nom`` and drops out of the
minimum on its own.  Weighting an aggregation from outside fails badly here: a
car 2 m ahead but laterally clear has ``v_allow = 0``, and under a softmin its
exponential weight swamps any lateral damping.

The minimum is continuous in the traffic configuration even though it is not
smooth.  Vehicles enter and leave the constrained set through ``w_i``, which is
continuous, and the sensing-range boundary is never active: the cap binds only
below ``dx = d0 + T*v_nom`` (35 m at ``v_nom = 20``), well inside the 90 m
range.  That is what the earlier hard blocker-selection rule lacked, and it is
the discontinuity the fixed target was introduced to remove.

The cost the reward would see is normalized by the fixed free-flow speed,

    cx = |v_ego - v_target| / v_nom

and not by the live target: a dynamic target in the denominator would make the
penalty diverge whenever traffic forces the target toward zero.
"""

from __future__ import annotations

import os

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import scripts.training.run_ppo_cbf_progression as progression
from scripts.evaluation.evaluate_cbf_random_actions import _git_provenance
from scripts.evaluation.evaluate_ppo_cbf_overrides import (
    apply_env_overrides,
    build_namespace,
    load_source_run,
    prepare_output_dir,
    write_manifest,
)

DEFAULT_EPISODES = 30
DEFAULT_SEED_START = 1_100_000
DEFAULT_V_NOMINAL = 20.0
DEFAULT_TIMEGAP = 1.5
DEFAULT_STANDSTILL_GAP = 5.0
DEFAULT_SIGMA_LAT = 0.45
DEFAULT_BETA = 0.0


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def proposed_speed_target(
    base_env: Any,
    *,
    v_nominal: float,
    timegap: float,
    standstill_gap: float,
    sigma_lat: float,
    beta: float,
) -> dict[str, float]:
    """Return the feasibility-capped target and its supporting quantities.

    ``base_env`` is the unwrapped lane-free environment, read at the state the
    reward itself is evaluated at (i.e. after the simulator step).
    """

    ego = base_env.vehicle
    sensing_range = float(base_env.config["sensing_range"])

    # Minimum over per-vehicle allowances, each already faded toward the
    # free-flow speed by its lateral relevance.  ``beta > 0`` replaces the
    # minimum with a softmax-weighted average over the same faded allowances.
    v_target = float(v_nominal)
    weight_sum = 1.0
    weighted_speed_sum = float(v_nominal)
    binding_weight = 0.0
    hard_min_allow = float(v_nominal)
    nearest_gap = float("inf")
    nearest_weight = 0.0
    nearest_speed = float("nan")
    relevant_count = 0.0

    for vehicle in base_env.road.vehicles:
        if vehicle is ego:
            continue
        dx = float(base_env._forward_distance(ego.position[0], vehicle.position[0]))
        if not (0.0 < dx < sensing_range):
            continue
        dy = float(vehicle.position[1] - ego.position[1])
        clearance = abs(dy) - 0.5 * (float(ego.width) + float(vehicle.width))
        weight = _sigmoid(-clearance / max(sigma_lat, 1e-6))
        if weight <= 1e-6:
            continue
        relevant_count += weight
        v_allow = max((dx - standstill_gap) / max(timegap, 1e-6), 0.0)
        # Fade the constraint toward "no constraint" by lateral relevance.
        v_effective = weight * v_allow + (1.0 - weight) * float(v_nominal)
        v_target = min(v_target, v_effective)
        candidate_weight = math.exp(-beta * (v_effective - v_nominal))
        weight_sum += candidate_weight
        weighted_speed_sum += candidate_weight * v_effective
        if weight > 0.5 and v_allow < hard_min_allow:
            hard_min_allow = v_allow
        if v_allow < v_nominal:
            binding_weight = max(binding_weight, weight)
        if dx < nearest_gap and weight > 0.5:
            nearest_gap = dx
            nearest_weight = weight
            nearest_speed = float(vehicle.vx)

    if beta > 0.0:
        v_target = weighted_speed_sum / max(weight_sum, 1e-12)
    v_target = float(np.clip(v_target, 0.0, float(v_nominal)))
    return {
        "proposed_target_speed": v_target,
        "proposed_hard_min_allow": float(hard_min_allow),
        "proposed_binding_weight": float(binding_weight),
        "relevant_vehicle_weight": float(relevant_count),
        "nearest_corridor_gap_m": float(nearest_gap) if math.isfinite(nearest_gap) else float("nan"),
        "nearest_corridor_weight": float(nearest_weight),
        "nearest_corridor_speed": float(nearest_speed),
    }


def replay_episode(
    namespace: dict[str, Any],
    *,
    model: Any,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    mode: str,
    scenario_seed: int,
    correction_epsilon: float,
    task_distance_m: float,
    task_max_policy_steps: int,
    rule: dict[str, float],
) -> list[dict[str, Any]]:
    """Replay one deterministic episode and return its per-step rows."""

    env = progression.make_evaluation_env(
        namespace,
        mode=mode,
        env_config=env_config,
        reward_config=reward_config,
        correction_epsilon=float(correction_epsilon),
        task_distance_m=float(task_distance_m),
        task_max_policy_steps=int(task_max_policy_steps),
    )
    rows: list[dict[str, Any]] = []
    try:
        observation, _ = env.reset(seed=int(scenario_seed))
        base_env = env.unwrapped
        for step in range(int(task_max_policy_steps)):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            info = dict(info)

            # The reward is evaluated after the simulator step, so the
            # counterfactual target is read from the same post-step state.
            proposed = proposed_speed_target(base_env, **rule)
            v_nominal = float(rule["v_nominal"])
            actual_target = float(info.get("karalakou_target_speed", float("nan")))
            ego_speed = float(info.get("karalakou_ego_speed", float("nan")))
            proposed_target = proposed["proposed_target_speed"]

            rows.append(
                {
                    "seed": int(scenario_seed),
                    "step": int(step),
                    "reward": float(reward),
                    "ego_speed": ego_speed,
                    "ego_y": float(info.get("karalakou_ego_y", float("nan"))),
                    "actual_target_speed": actual_target,
                    "actual_cx": float(info.get("karalakou_cx", float("nan"))),
                    "cy": float(info.get("karalakou_cy", float("nan"))),
                    "cf": float(info.get("karalakou_cf", float("nan"))),
                    "zone_found": float(info.get("karalakou_zone_found", float("nan"))),
                    # Both costs are normalized by the same fixed free-flow
                    # speed so the two rules are compared on one scale; the
                    # live target never enters a denominator.
                    "proposed_cx": abs(ego_speed - proposed_target) / v_nominal,
                    "actual_cx_common_norm": abs(ego_speed - actual_target) / v_nominal,
                    "collision": float(info.get("karalakou_ego_collision", 0.0)),
                    **proposed,
                }
            )
            if bool(terminated) or bool(truncated):
                break
    finally:
        env.close()
    return rows


def _pre_collision_overspeed(steps: pd.DataFrame, window: int) -> dict[str, Any]:
    """Does the proposed target anticipate the collisions the fixed one missed?

    For every episode that ends in a collision, compare the ego's excess over
    each target in the last ``window`` steps against the same episode's overall
    mean.  A rule that spikes before impact is one the policy could have acted
    on; a flat rule carries no warning.
    """

    proposed_late: list[float] = []
    proposed_all: list[float] = []
    actual_late: list[float] = []
    episodes = 0
    for _, episode in steps.groupby("seed", sort=False):
        if float(episode["collision"].sum()) <= 0.0:
            continue
        episodes += 1
        tail = episode.tail(int(window))
        proposed_late.append(float((tail["ego_speed"] - tail["proposed_target_speed"]).mean()))
        proposed_all.append(float((episode["ego_speed"] - episode["proposed_target_speed"]).mean()))
        actual_late.append(float((tail["ego_speed"] - tail["actual_target_speed"]).mean()))
    if not episodes:
        return {"collision_episodes": 0}
    return {
        "collision_episodes": int(episodes),
        "pre_collision_window": int(window),
        "proposed_overspeed_episode_mean": float(np.mean(proposed_all)),
        "proposed_overspeed_pre_collision": float(np.mean(proposed_late)),
        "proposed_overspeed_rise": float(np.mean(proposed_late) - np.mean(proposed_all)),
        "actual_overspeed_pre_collision": float(np.mean(actual_late)),
    }


def summarize(steps: pd.DataFrame, *, v_nominal: float, bind_tolerance: float = 0.25,
              pre_collision_window: int = 10) -> dict[str, Any]:
    """Pooled summary of how often and how hard the proposed cap would bind."""

    binding = steps["proposed_target_speed"] < (v_nominal - bind_tolerance)
    # cx at the ego's actual speed, under each target.  The actual run's cx is
    # recomputed against the fixed 16 m/s target it really used.
    return {
        "episodes": int(steps["seed"].nunique()),
        "steps": int(len(steps)),
        "collisions": int(steps["collision"].sum()),
        "mean_ego_speed": float(steps["ego_speed"].mean()),
        "actual_target_speed": float(steps["actual_target_speed"].mean()),
        "binding_rate": float(binding.mean()),
        "mean_proposed_target": float(steps["proposed_target_speed"].mean()),
        "mean_proposed_target_when_binding": (
            float(steps.loc[binding, "proposed_target_speed"].mean()) if binding.any() else float("nan")
        ),
        "p10_proposed_target": float(steps["proposed_target_speed"].quantile(0.10)),
        "p50_proposed_target": float(steps["proposed_target_speed"].quantile(0.50)),
        "p90_proposed_target": float(steps["proposed_target_speed"].quantile(0.90)),
        # The run's own cost, exactly as the reward paid it (normalized by the
        # 16 m/s target it used).
        "mean_actual_cx_as_paid": float(steps["actual_cx"].mean()),
        # Like-for-like: both rules normalized by the same free-flow speed.
        "mean_actual_cx": float(steps["actual_cx_common_norm"].mean()),
        "mean_proposed_cx": float(steps["proposed_cx"].mean()),
        "mean_actual_cx_when_binding": (
            float(steps.loc[binding, "actual_cx_common_norm"].mean()) if binding.any() else float("nan")
        ),
        "mean_proposed_cx_when_binding": (
            float(steps.loc[binding, "proposed_cx"].mean()) if binding.any() else float("nan")
        ),
        "mean_hard_min_allow": float(steps["proposed_hard_min_allow"].mean()),
        # How often the ego is slower than its target: with a fixed target this
        # is the "punished for being blocked" regime the change is meant to fix.
        "actual_below_target_rate": float((steps["ego_speed"] < steps["actual_target_speed"]).mean()),
        "proposed_below_target_rate": float((steps["ego_speed"] < steps["proposed_target_speed"]).mean()),
        "zone_found_rate": float(steps["zone_found"].mean()),
        "mean_nearest_corridor_gap_m": float(steps["nearest_corridor_gap_m"].mean(skipna=True)),
        "corridor_occupied_rate": float(steps["nearest_corridor_weight"].gt(0.5).mean()),
        **_pre_collision_overspeed(steps, int(pre_collision_window)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True, help="Ladder seed directory with run_config.json and model_final.zip.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--mode", choices=tuple(progression.EVALUATION_MODES), default="raw",
                        help="raw = CBF OFF (the unshielded baseline path); cbf = filter in the loop.")
    parser.add_argument("--vehicles", type=int, default=None, help="Traffic density override (canonical: 40).")
    parser.add_argument("--ttc-cap", type=float, default=30.0)
    parser.add_argument("--v-nominal", type=float, default=DEFAULT_V_NOMINAL,
                        help="Free-flow target the cap applies to (default 20 m/s).")
    parser.add_argument("--timegap", type=float, default=DEFAULT_TIMEGAP)
    parser.add_argument("--standstill-gap", type=float, default=DEFAULT_STANDSTILL_GAP)
    parser.add_argument("--sigma-lat", type=float, default=DEFAULT_SIGMA_LAT)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA,
                        help="0 = hard minimum over faded allowances (default); >0 smooths it.")
    parser.add_argument("--allow-dirty-exploratory", action="store_true")
    args = parser.parse_args()

    project_root = progression.protocol.find_project_root(args.project_root or Path.cwd()).resolve()
    run_dir = (args.run_dir if args.run_dir.is_absolute() else project_root / args.run_dir).resolve()
    run_config, study_config = load_source_run(run_dir)
    output_dir = prepare_output_dir(project_root, args.output_dir)
    source = _git_provenance(project_root, output_dir, allow_dirty=bool(args.allow_dirty_exploratory))

    env_config, max_policy_steps, changes = apply_env_overrides(
        run_config["env_config"],
        max_policy_steps=int(study_config["evaluation_task_max_policy_steps"]),
        vehicles=args.vehicles,
    )
    namespace = build_namespace(project_root, run_config)
    reward_config = copy.deepcopy(run_config["reward_config"])
    rule = {
        "v_nominal": float(args.v_nominal),
        "timegap": float(args.timegap),
        "standstill_gap": float(args.standstill_gap),
        "sigma_lat": float(args.sigma_lat),
        "beta": float(args.beta),
    }
    model_path = run_dir / "model_final.zip"
    manifest: dict[str, Any] = {
        "status": "running",
        "evaluation_kind": "speed_target_counterfactual_replay",
        "entrypoint": "scripts.evaluation.replay_speed_target_counterfactual",
        "source_run_dir": str(run_dir),
        "variant": run_config["variant"],
        "model_path": str(model_path),
        "model_sha256": progression.protocol.file_sha256(model_path),
        "mode": str(args.mode),
        "env_config_changes": changes,
        "episodes": int(args.episodes),
        "episode_seed_start": int(args.seed_start),
        "reward_speed_target_actual": float(reward_config.get("ego_desired_speed", float("nan"))),
        "proposed_rule": rule,
        "source": source,
    }
    manifest_path = output_dir / "manifest.json"
    write_manifest(manifest_path, manifest)
    print(f"[speed-target] {run_config['variant']} mode={args.mode} changes={changes} rule={rule}", flush=True)

    model = progression.load_model(run_config["variant"], model_path, device="cpu")
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for index in range(int(args.episodes)):
        seed = int(args.seed_start) + index
        episode_rows = replay_episode(
            namespace,
            model=model,
            env_config=env_config,
            reward_config=reward_config,
            mode=str(args.mode),
            scenario_seed=seed,
            correction_epsilon=float(run_config["training_signature"]["correction_epsilon"]),
            task_distance_m=float(study_config["evaluation_task_distance_m"]),
            task_max_policy_steps=int(max_policy_steps),
            rule=rule,
        )
        rows.extend(episode_rows)
        print(f"  seed {seed}: {len(episode_rows)} steps (total {len(rows)})", flush=True)

    elapsed = time.perf_counter() - started
    steps = pd.DataFrame(rows)
    steps.to_csv(output_dir / "steps.csv", index=False)
    summary = summarize(steps, v_nominal=float(args.v_nominal))
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    manifest.update(summary, status="complete", elapsed_s=elapsed)
    write_manifest(manifest_path, manifest)

    print("\n[speed-target] pooled summary")
    for key, value in summary.items():
        print(f"  {key:38s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
