"""Replay CBF-ON episodes and classify QP infeasibility and ego collisions.

Before every step it rebuilds the exact constraint system the shield uses and
records whether the QP is jointly feasible and whether any single neighbor row
is impossible inside the action box on its own.  At every ego collision it
records which vehicle was hit (ahead / behind / side), the relative velocity,
the ego position and acceleration, and how long the QP had been infeasible.
It also counts barrier crossings (min h going from >= 0 to < 0) and whether
the QP was feasible on the step that produced them.
"""

from __future__ import annotations

import os

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import argparse
import copy
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scripts.training.run_ppo_cbf_progression as progression
from scripts.evaluation.evaluate_cbf_random_actions import _git_provenance
from scripts.evaluation.evaluate_ppo_cbf_overrides import (
    add_override_arguments,
    apply_env_overrides,
    build_namespace,
    hocbf_gains,
    load_source_run,
    resolve_run,
)

_WORKER: dict[str, Any] = {}
_TOL = 1e-3


def _init_worker(project_root: str, run_dir: str, env_config: dict[str, Any], gains, max_policy_steps: int) -> None:
    run_config, study_config = load_source_run(Path(run_dir))
    _WORKER.update(
        namespace=build_namespace(Path(project_root), run_config, gains),
        run_config=run_config,
        env_config=env_config,
        task_distance_m=float(study_config["evaluation_task_distance_m"]),
        max_policy_steps=int(max_policy_steps),
        model=progression.load_model(run_config["variant"], Path(run_dir) / "model_final.zip", "cpu"),
    )


def _replay(seed: int) -> dict[str, Any]:
    ns, run_config = _WORKER["namespace"], _WORKER["run_config"]
    env = progression.make_evaluation_env(
        ns, mode="cbf", env_config=_WORKER["env_config"],
        reward_config=copy.deepcopy(run_config["reward_config"]),
        correction_epsilon=float(run_config["training_signature"]["correction_epsilon"]),
        task_distance_m=_WORKER["task_distance_m"], task_max_policy_steps=_WORKER["max_policy_steps"],
    )
    base = env.unwrapped
    semi_a, semi_b = float(ns["CBF_RELATIVE_ELLIPSE_A"]), float(ns["CBF_RELATIVE_ELLIPSE_B"])
    psi1_gain = float(ns["CBF_PSI1_GAIN"])
    low = np.array([ns["CBF_AX_BOUNDS"][0], ns["CBF_AY_BOUNDS"][0]], dtype=float)
    high = np.array([ns["CBF_AX_BOUNDS"][1], ns["CBF_AY_BOUNDS"][1]], dtype=float)
    lateral_ratio = float(base.config["bounds"]["max_lateral_speed_ratio"])
    road_length = float(base.config["road_length"])
    observation, _ = env.reset(seed=int(seed))
    failures, collisions, crossings = [], [], []
    step, previous_failed, previous_min_h, previous_feasible = 0, False, None, None
    feasible_steps = feasible_negative_h = 0
    while True:
        ego = ns["get_ego_state"](env)
        neighbors = ns["get_neighbor_states"](env, neighbor_range=float(ns["CBF_NEIGHBOR_RANGE"]))
        neighbors = neighbors[: int(ns["CBF_MAX_NEIGHBOR_CONSTRAINTS"])]
        system = env.get_wrapper_attr("current_constraint_system")()
        rows = np.asarray(system["cbf_rows"], dtype=float).reshape(-1, 2)
        bounds = np.asarray(system["cbf_bounds"], dtype=float)
        feasible = ns["project_action_to_linear_constraints_2d"](np.zeros(2), list(rows), list(bounds)) is not None
        min_h = float(system["min_h"]) if np.isfinite(system["min_h"]) else np.inf
        feasible_steps += int(feasible)
        feasible_negative_h += int(feasible and min_h < 0.0)
        if previous_min_h is not None and previous_min_h >= 0.0 and min_h < 0.0:
            crossings.append({"step": step, "prev_step_feasible": bool(previous_feasible), "h_before": previous_min_h, "h_after": min_h})
        previous_min_h, previous_feasible = min_h, feasible
        if not feasible:
            # min over the box of a.x for each row, compared with its bound
            single_gap = np.minimum(rows * low, rows * high).sum(axis=1) - bounds
            record = {
                "step": step, "onset": not previous_failed, "ego_vx": ego["vx"], "ego_vy": ego["vy"],
                "n_single_infeasible": int((single_gap[: len(neighbors)] > _TOL).sum()),
                "boundary_single_infeasible": bool((single_gap[len(neighbors):] > _TOL).any()),
            }
            if neighbors:
                worst = int(np.argmax(single_gap[: len(neighbors)]))
                other = neighbors[worst]
                dx, dy = float(other["signed_dx"]), float(other["y"] - ego["y"])
                dvx, dvy = float(other["vx"] - ego["vx"]), float(other["vy"] - ego["vy"])
                h_value = (dx / semi_a) ** 2 + (dy / semi_b) ** 2 - 1.0
                h_dot = 2.0 * dx * dvx / semi_a**2 + 2.0 * dy * dvy / semi_b**2
                record.update(worst_dx=dx, worst_dy=dy, worst_dvx=dvx, worst_dvy=dvy, worst_h=h_value,
                              worst_psi1=h_dot + psi1_gain * h_value, worst_gap=float(single_gap[worst]))
            failures.append(record)
        previous_failed = not feasible
        observation, _reward, terminated, truncated, info = env.step(
            progression._predict_evaluation_action(_WORKER["model"], observation)
        )
        step += 1
        if int(info.get("ego_collision_events", 0)) > 0 or bool(info.get("ego_collision", False)):
            state = base._current_state_arrays()
            ego_index = int(np.flatnonzero(state["is_ego"])[0])
            dxs = ((state["x"] - state["x"][ego_index] + 0.5 * road_length) % road_length) - 0.5 * road_length
            dys = state["y"] - state["y"][ego_index]
            overlap = (np.abs(dxs) < 0.5 * (state["lengths"] + state["lengths"][ego_index])) & (
                np.abs(dys) < 0.5 * (state["widths"] + state["widths"][ego_index])
            )
            hit = [index for index in np.flatnonzero(overlap) if index != ego_index]
            other = hit[0] if hit else None
            ego_vx, ego_vy = float(state["vx"][ego_index]), float(state["vy"][ego_index])
            run = 0
            for failure in reversed(failures):
                if failure["step"] != step - 1 - run:
                    break
                run += 1
            nan = float("nan")
            collisions.append({
                "step": step, "qp_fail_last_step": previous_failed, "consecutive_fail_steps_before": run,
                "hit_dx": float(dxs[other]) if other is not None else nan,
                "hit_dy": float(dys[other]) if other is not None else nan,
                "hit_dvx": float(state["vx"][other] - ego_vx) if other is not None else nan,
                "hit_dvy": float(state["vy"][other] - ego_vy) if other is not None else nan,
                "hit_other_ax": float(base._last_accelerations[other, 0]) if other is not None else nan,
                "ego_vx": ego_vx, "ego_vy": ego_vy, "ego_y": float(state["y"][ego_index]),
                "ego_ax": float(base._last_accelerations[ego_index, 0]),
                "ego_vy_at_clamp": abs(ego_vy) >= lateral_ratio * max(ego_vx, 1.0) - 1e-6,
            })
        if terminated or truncated:
            break
    env.close()
    return {"seed": int(seed), "steps": step, "fails": failures, "collisions": collisions, "crossings": crossings,
            "n_feasible": feasible_steps, "n_feasible_hneg": feasible_negative_h}


def summarize(results: list[dict[str, Any]], half_length_m: float) -> dict[str, Any]:
    steps = sum(result["steps"] for result in results)
    failures = pd.DataFrame([dict(item, seed=r["seed"]) for r in results for item in r["fails"]])
    collisions = pd.DataFrame([dict(item, seed=r["seed"]) for r in results for item in r["collisions"]])
    crossings = pd.DataFrame([dict(item, seed=r["seed"]) for r in results for item in r["crossings"]])
    summary: dict[str, Any] = {"episodes": len(results), "steps": steps,
                               "infeasible_step_rate": len(failures) / max(steps, 1),
                               "barrier_crossings": len(crossings),
                               "crossings_with_feasible_qp": int(crossings["prev_step_feasible"].sum()) if len(crossings) else 0}
    if len(failures):
        onsets = failures[failures["onset"]]
        single = onsets["n_single_infeasible"] > 0
        summary.update(infeasible_onsets=len(onsets), onset_single_row_rate=float(single.mean()),
                       onset_single_row_behind=int((single & (onsets["worst_dx"] < 0)).sum()),
                       onset_joint_only_rate=float((~single & ~onsets["boundary_single_infeasible"]).mean()))
    if len(collisions):
        where = np.where(collisions["hit_dx"].abs() <= half_length_m, "side",
                         np.where(collisions["hit_dx"] > 0, "ahead", "behind"))
        summary.update(collisions=len(collisions), collisions_by_direction=pd.Series(where).value_counts().to_dict(),
                       collision_qp_failed_previous_step=float(collisions["qp_fail_last_step"].mean()))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_override_arguments(parser)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file for per-episode records and the summary.")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--c1", type=float, default=None)
    parser.add_argument("--c2", type=float, default=None)
    args = parser.parse_args()
    if (args.c1 is None) != (args.c2 is None):
        parser.error("--c1 and --c2 must be given together")
    project_root, run_dir, run_config, study_config = resolve_run(args)
    output = args.output if args.output.is_absolute() else project_root / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source = _git_provenance(project_root, output.parent, allow_dirty=bool(args.allow_dirty_exploratory))
    gains = hocbf_gains(args.c1, args.c2) if args.c1 is not None else None
    env_config, max_policy_steps, changes = apply_env_overrides(
        run_config["env_config"], max_policy_steps=int(study_config["evaluation_task_max_policy_steps"]),
        vehicles=args.vehicles, hz=args.hz, spawn_psi1_gain=args.spawn_psi1_gain,
        require_initial_safe_set=None if args.require_initial_safe_set is None else args.require_initial_safe_set == "true",
    )
    seeds = [int(args.seed_start) + index for index in range(int(args.episodes))]
    with ProcessPoolExecutor(
        int(args.workers), mp_context=mp.get_context("spawn"), initializer=_init_worker,
        initargs=(str(project_root), str(run_dir), env_config, gains, max_policy_steps),
    ) as executor:
        results = list(executor.map(_replay, seeds))
    half_length = 0.5 * float(run_config["env_config"]["ego_dimensions"][0])
    summary = summarize(results, half_length)
    payload = {"entrypoint": "scripts.evaluation.diagnose_cbf_qp_failures", "source_run_dir": str(run_dir),
               "env_config_changes": changes, "gains": gains, "seeds": seeds, "source": source,
               "summary": summary, "episodes": results}
    output.write_text(json.dumps(payload, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
