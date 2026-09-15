"""Measure the degenerate behaviours the post-training KPI table cannot show.

The canonical KPI table reports collisions, return, completion, tracking
error, speed error, jerk and intervention. None of those distinguish a policy
that drives well from one that bought its safety by **hugging the sideline**
or by **lagging** behind the traffic, and both failure modes have appeared in
this repository before (``docs/hocbf_reward_investigation.md``: a term whose
median ego y sat at 9.30 on a 10.2 m road, and two seeds of five that
collapsed to 9.45 and 11.02 m/s).

This probe rolls a checkpoint deterministically through the *evaluation*
environment for the run it belongs to and reports:

* sideline hugging -- ego y distribution, distance to the nearer boundary, and
  the fraction of steps spent within a threshold of an edge;
* lagging -- mean speed and the **signed** deficit against the run's own speed
  target, so crawling and speeding are distinguishable rather than folded into
  one absolute error;
* where a state cost is active, the vehicle/boundary split of its ``max``, which
  says whether the wall channel is crowding out the vehicle channel.

Edge geometry follows the environment clamp: the ego centre is confined to
``[width/2, road_width - width/2]`` (``lane_free_env.py``), so the distance to
the nearer edge is ``min(y - y_min, y_max - y)``.

Results are generated output; pass an explicit ``--out`` and keep them out of
the repository.
"""

from __future__ import annotations

import os

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np

import scripts.training.run_ppo_cbf_progression as progression
from scripts.evaluation.evaluate_ppo_cbf_overrides import (
    apply_env_overrides,
    build_namespace,
    load_source_run,
)

DEFAULT_SEED_START = 1_100_000
DEFAULT_EPISODES = 10
DEFAULT_EDGE_THRESHOLD_M = 0.5


def _episode_records(
    namespace,
    run_config,
    study_config,
    *,
    mode: str,
    seed: int,
    step_cap: int,
    env_config,
    max_policy_steps: int,
    model,
) -> list[dict]:
    env = progression.make_evaluation_env(
        namespace,
        mode=mode,
        env_config=copy.deepcopy(env_config),
        reward_config=copy.deepcopy(run_config["reward_config"]),
        correction_epsilon=float(
            run_config["training_signature"]["correction_epsilon"]
        ),
        task_distance_m=float(study_config["evaluation_task_distance_m"]),
        task_max_policy_steps=int(max_policy_steps),
    )
    records: list[dict] = []
    try:
        observation, _info = env.reset(seed=seed)
        for _step in range(step_cap):
            action, _ = model.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, info = env.step(action)
            records.append(dict(info))
            if terminated or truncated:
                break
    finally:
        env.close()
    return records


def summarize(records: list[dict], *, road_width: float, ego_width: float,
              edge_threshold_m: float) -> dict:
    y_min = 0.5 * ego_width
    y_max = road_width - 0.5 * ego_width

    ego_y = np.asarray(
        [float(r.get("karalakou_ego_y", np.nan)) for r in records], dtype=float
    )
    speed = np.asarray(
        [float(r.get("karalakou_ego_speed", np.nan)) for r in records], dtype=float
    )
    target = np.asarray(
        [float(r.get("karalakou_target_speed", np.nan)) for r in records], dtype=float
    )
    ego_y = ego_y[np.isfinite(ego_y)]
    finite_speed = np.isfinite(speed) & np.isfinite(target)
    edge_distance = np.minimum(ego_y - y_min, y_max - ego_y)

    classes = Counter(
        str(r["cbf_state_cost_argmax_class"])
        for r in records
        if "cbf_state_cost_argmax_class" in r
    )
    total_classes = sum(classes.values())

    summary = {
        "steps": int(len(records)),
        "road_width_m": float(road_width),
        "ego_y_min_m": round(float(y_min), 3),
        "ego_y_max_m": round(float(y_max), 3),
        # Sideline hugging.
        "ego_y_median": round(float(np.median(ego_y)), 3) if ego_y.size else None,
        "ego_y_p05": round(float(np.percentile(ego_y, 5)), 3) if ego_y.size else None,
        "ego_y_p95": round(float(np.percentile(ego_y, 95)), 3) if ego_y.size else None,
        "edge_distance_mean_m": (
            round(float(edge_distance.mean()), 3) if ego_y.size else None
        ),
        "edge_threshold_m": float(edge_threshold_m),
        "frac_steps_near_edge": (
            round(float(np.mean(edge_distance <= edge_threshold_m)), 4)
            if ego_y.size
            else None
        ),
        # Lagging. The deficit is signed on purpose: crawling and speeding are
        # different failures and abs speed error hides both.
        "mean_speed_mps": (
            round(float(speed[finite_speed].mean()), 3) if finite_speed.any() else None
        ),
        "mean_target_speed_mps": (
            round(float(target[finite_speed].mean()), 3) if finite_speed.any() else None
        ),
        "mean_signed_speed_deficit_mps": (
            round(float((target[finite_speed] - speed[finite_speed]).mean()), 3)
            if finite_speed.any()
            else None
        ),
    }
    if total_classes:
        summary["state_cost_argmax_counts"] = dict(classes)
        summary["state_cost_argmax_boundary_share"] = round(
            classes.get("boundary", 0) / total_classes, 4
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Seed directory holding run_config.json and model_final.zip.",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(progression.EVALUATION_MODES),
        default="raw",
        help="raw = external CBF OFF (the unshielded policy); cbf = filter on.",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--edge-threshold-m", type=float, default=DEFAULT_EDGE_THRESHOLD_M
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    project_root = progression.protocol.find_project_root(
        args.project_root or Path.cwd()
    ).resolve()
    run_dir = (
        args.run_dir
        if args.run_dir.is_absolute()
        else project_root / args.run_dir
    ).resolve()
    run_config, study_config = load_source_run(run_dir)
    env_config, max_policy_steps, _changes = apply_env_overrides(
        run_config["env_config"],
        max_policy_steps=int(study_config["evaluation_task_max_policy_steps"]),
        vehicles=None,
    )
    namespace = build_namespace(project_root, run_config)
    model_path = run_dir / "model_final.zip"
    model = progression.load_model(
        run_config["variant"], model_path, device=str(args.device)
    )
    step_cap = int(args.steps) if args.steps is not None else int(max_policy_steps)

    road_width = float(env_config["road_width"])
    ego_dimensions = env_config.get("ego_dimensions", [3.5, 1.8])
    ego_width = float(ego_dimensions[1])

    records: list[dict] = []
    per_episode: list[dict] = []
    for index in range(int(args.episodes)):
        seed = int(args.seed_start) + index
        episode = _episode_records(
            namespace,
            run_config,
            study_config,
            mode=str(args.mode),
            seed=seed,
            step_cap=step_cap,
            env_config=env_config,
            max_policy_steps=int(max_policy_steps),
            model=model,
        )
        records.extend(episode)
        episode_summary = summarize(
            episode,
            road_width=road_width,
            ego_width=ego_width,
            edge_threshold_m=float(args.edge_threshold_m),
        )
        episode_summary["seed"] = seed
        per_episode.append(episode_summary)

    summary = summarize(
        records,
        road_width=road_width,
        ego_width=ego_width,
        edge_threshold_m=float(args.edge_threshold_m),
    )
    summary.update(
        {
            "run_dir": str(run_dir),
            "variant": run_config["variant"],
            "mode": str(args.mode),
            "episodes": int(args.episodes),
            "seed_start": int(args.seed_start),
            "model_sha256": progression.protocol.file_sha256(model_path),
            "per_episode": per_episode,
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"{run_config['variant']:26s} mode={args.mode} "
        f"y_med={summary['ego_y_median']} "
        f"edge<={args.edge_threshold_m}m={summary['frac_steps_near_edge']} "
        f"edge_dist={summary['edge_distance_mean_m']} "
        f"v={summary['mean_speed_mps']}/{summary['mean_target_speed_mps']} "
        f"deficit={summary['mean_signed_speed_deficit_mps']} "
        f"steps={summary['steps']}"
    )
    if "state_cost_argmax_boundary_share" in summary:
        print(
            f"  state cost argmax boundary share: "
            f"{summary['state_cost_argmax_boundary_share']}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
