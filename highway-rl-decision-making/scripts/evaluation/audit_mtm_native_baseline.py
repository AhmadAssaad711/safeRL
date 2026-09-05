"""Native-MTM controllability and background-collision baseline.

The ego is assigned the same MTM controller as the surrounding vehicles by
setting ``ego_controlled=False``.  No PPO policy, CBF wrapper, or safety filter
is used.  The experiment runs a fixed horizon so traffic-only collisions are
also counted after an ego collision.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import scripts.evaluation.audit_nominal_mtm_collision_provenance as provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1_300_000)
    parser.add_argument(
        "--horizon-policy-steps",
        type=int,
        default=400,
        help="Fixed nominal horizon; MTM policy dt is normally 0.1 s.",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _finite_mean(rows: list[dict[str, Any]], field: str) -> float:
    values = np.asarray([row[field] for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def _native_episode(
    namespace: dict[str, Any],
    *,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    seed: int,
    horizon_policy_steps: int,
) -> dict[str, Any]:
    native_config = copy.deepcopy(env_config)
    # Reset with the original controlled-ego setting so the MTM profile draw
    # cannot change the RNG sequence and therefore cannot change the initial
    # traffic geometry.  Convert the already-reset ego to native MTM below.
    native_config["ego_controlled"] = True
    # Keep simulating after an ego collision so background traffic events are
    # not hidden by the normal terminal condition.
    native_config["terminate_on_collision"] = False

    env = provenance.basics.make_nominal_env(
        namespace, env_config=native_config, reward_config=reward_config
    )
    try:
        observation, _ = env.reset(seed=int(seed))
        base = provenance.basics._base(env)
        base.config["ego_controlled"] = False
        base.vehicle.driver_profile = "normal"
        initial_snapshot = provenance._snapshot(base)
        initial = provenance._initial_geometry(
            base,
            initial_snapshot,
            eps_side=float(reward_config.get("safety_potential_eps_side", 0.10)),
        )
        ego_index = provenance._ego_index(initial_snapshot)
        policy_dt = provenance._policy_dt(native_config)
        previous_x = float(base.vehicle.position[0])
        distance_m = 0.0
        total_collision_events = 0
        ego_collision_events = 0
        background_only_events = 0
        any_collision_seen = False
        ego_collision_seen = False
        first_any_collision_s = np.nan
        first_ego_collision_s = np.nan
        abs_ax: list[float] = []
        abs_ay: list[float] = []
        mtm_leader_gaps: list[float] = []
        steps = 0

        for _ in range(int(horizon_policy_steps)):
            observation, _, terminated, truncated, info = env.step(
                np.zeros(2, dtype=np.float32)
            )
            base = provenance.basics._base(env)
            distance_m += provenance.basics._distance_step(base, previous_x)
            previous_x = float(base.vehicle.position[0])
            steps += 1

            info = dict(info)
            step_total = max(int(info.get("collisions", 0)), 0)
            step_ego = max(int(info.get("ego_collision_events", 0)), 0)
            step_background = max(step_total - step_ego, 0)
            total_collision_events += step_total
            ego_collision_events += step_ego
            background_only_events += step_background

            if step_total > 0 and not any_collision_seen:
                first_any_collision_s = float(base.time)
                any_collision_seen = True
            if step_ego > 0 and not ego_collision_seen:
                first_ego_collision_s = float(base.time)
                ego_collision_seen = True

            accelerations = np.asarray(
                getattr(base, "_last_accelerations", np.empty((0, 2))),
                dtype=float,
            )
            if accelerations.ndim == 2 and accelerations.shape[0] > ego_index:
                abs_ax.append(abs(float(accelerations[ego_index, 0])))
                abs_ay.append(abs(float(accelerations[ego_index, 1])))
            leader_gap = float(info.get("mtm_mean_leader_gap", np.nan))
            if np.isfinite(leader_gap):
                mtm_leader_gaps.append(leader_gap)

            if terminated or truncated:
                break

        return {
            "seed": int(seed),
            "steps": int(steps),
            "horizon_s": float(steps * policy_dt),
            "distance_m": float(distance_m),
            "any_collision": int(any_collision_seen),
            "ego_collision": int(ego_collision_seen),
            "total_collision_events": int(total_collision_events),
            "ego_collision_events": int(ego_collision_events),
            "background_only_collision_events": int(background_only_events),
            "first_any_collision_s": float(first_any_collision_s),
            "first_ego_collision_s": float(first_ego_collision_s),
            "mean_abs_ego_ax_mps2": float(np.mean(abs_ax)) if abs_ax else np.nan,
            "mean_abs_ego_ay_mps2": float(np.mean(abs_ay)) if abs_ay else np.nan,
            "mean_mtm_leader_gap_m": (
                float(np.mean(mtm_leader_gaps)) if mtm_leader_gaps else np.nan
            ),
            **initial,
        }
    finally:
        env.close()


def _aggregate(rows: list[dict[str, Any]], *, horizon_policy_steps: int, policy_dt: float) -> dict[str, Any]:
    total_distance = float(sum(float(row["distance_m"]) for row in rows))
    total_events = int(sum(int(row["total_collision_events"]) for row in rows))
    total_ego_events = int(sum(int(row["ego_collision_events"]) for row in rows))
    total_background_events = int(
        sum(int(row["background_only_collision_events"]) for row in rows)
    )
    initial_clearances = np.asarray(
        [row["initial_min_rect_separation_m"] for row in rows], dtype=float
    )
    finite_initial_clearances = initial_clearances[np.isfinite(initial_clearances)]
    return {
        "episodes": int(len(rows)),
        "horizon_policy_steps": int(horizon_policy_steps),
        "horizon_s": float(horizon_policy_steps * policy_dt),
        "policy_dt_s": float(policy_dt),
        "initial_overlap_rate": float(
            np.mean([row["initial_ego_collision"] for row in rows])
        ),
        "initial_min_rect_separation_mean_m": _finite_mean(
            rows, "initial_min_rect_separation_m"
        ),
        "initial_min_rect_separation_min_m": (
            float(np.min(finite_initial_clearances))
            if finite_initial_clearances.size
            else np.nan
        ),
        "initial_min_ellipse_h_mean": _finite_mean(rows, "initial_min_ellipse_h"),
        "any_collision_episode_rate": float(
            np.mean([row["any_collision"] for row in rows])
        ),
        "ego_collision_episode_rate": float(
            np.mean([row["ego_collision"] for row in rows])
        ),
        "mean_total_collision_events": float(
            np.mean([row["total_collision_events"] for row in rows])
        ),
        "mean_ego_collision_events": float(
            np.mean([row["ego_collision_events"] for row in rows])
        ),
        "mean_background_only_collision_events": float(
            np.mean([row["background_only_collision_events"] for row in rows])
        ),
        "total_collisions_per_km": float(
            1000.0 * total_events / max(total_distance, 1e-9)
        ),
        "ego_collisions_per_km": float(
            1000.0 * total_ego_events / max(total_distance, 1e-9)
        ),
        "background_only_collisions_per_km": float(
            1000.0 * total_background_events / max(total_distance, 1e-9)
        ),
        "mean_first_any_collision_s": _finite_mean(rows, "first_any_collision_s"),
        "mean_first_ego_collision_s": _finite_mean(rows, "first_ego_collision_s"),
        "mean_distance_m": float(np.mean([row["distance_m"] for row in rows])),
        "mean_abs_native_ego_ax_mps2": _finite_mean(rows, "mean_abs_ego_ax_mps2"),
        "mean_abs_native_ego_ay_mps2": _finite_mean(rows, "mean_abs_ego_ay_mps2"),
        "mean_mtm_leader_gap_m": _finite_mean(rows, "mean_mtm_leader_gap_m"),
    }


def main() -> int:
    args = parse_args()
    if int(args.episodes) <= 0 or int(args.horizon_policy_steps) <= 0:
        raise ValueError("episodes and horizon-policy-steps must be positive")

    provenance.basics.notebook_pipeline.set_stable_native_defaults()
    project_root = provenance.basics.notebook_pipeline.find_project_root(
        args.project_root or Path.cwd()
    )
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    env_config = copy.deepcopy(config["env_config"])
    reward_config = copy.deepcopy(config["reward_config"])
    traffic_model = str(env_config.get("traffic_model", "")).strip().lower()
    if traffic_model != "mtm":
        raise RuntimeError(
            "This baseline is intentionally MTM-only; "
            f"received traffic_model={traffic_model!r}"
        )

    namespace = provenance._bootstrap_nominal_namespace(project_root)
    seeds = [int(args.seed_start) + index for index in range(int(args.episodes))]
    policy_dt = provenance._policy_dt(env_config)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[native-mtm] seed={seed}", flush=True)
        rows.append(
            _native_episode(
                namespace,
                env_config=env_config,
                reward_config=reward_config,
                seed=seed,
                horizon_policy_steps=int(args.horizon_policy_steps),
            )
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / "native_mtm_episode_results.csv"
    summary_path = output_dir / "native_mtm_summary.json"
    summary = _aggregate(
        rows,
        horizon_policy_steps=int(args.horizon_policy_steps),
        policy_dt=policy_dt,
    )
    _write_csv(episode_path, rows)
    summary_path.write_text(
        json.dumps(
            {
                "traffic_model": traffic_model,
                "ego_controlled": False,
                "terminate_on_collision": False,
                "seed_start": int(args.seed_start),
                "summary": summary,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    print(f"[native-mtm] wrote {episode_path}", flush=True)
    print(f"[native-mtm] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
