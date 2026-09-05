"""Native-MTM density/stability ladder.

Only ``vehicles_count`` changes between conditions.  The ego receives native
MTM control after reset, and each episode runs for a fixed horizon with
collision termination disabled so both ego and background traffic events are
counted.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import scripts.evaluation.audit_mtm_native_baseline as native_baseline


DEFAULT_COUNTS = (1, 10, 20, 35, 55)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1_300_000)
    parser.add_argument(
        "--vehicle-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_COUNTS),
    )
    parser.add_argument("--horizon-policy-steps", type=int, default=400)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_density(
    rows: list[dict[str, Any]],
    *,
    vehicle_count: int,
    horizon_policy_steps: int,
    policy_dt: float,
) -> dict[str, Any]:
    total_distance = float(sum(float(row["distance_m"]) for row in rows))
    total_events = int(sum(int(row["total_collision_events"]) for row in rows))
    total_ego_events = int(sum(int(row["ego_collision_events"]) for row in rows))
    total_background_events = int(
        sum(int(row["background_only_collision_events"]) for row in rows)
    )
    initial_clearances = np.asarray(
        [row["initial_min_rect_separation_m"] for row in rows], dtype=float
    )
    initial_clearances = initial_clearances[np.isfinite(initial_clearances)]
    initial_h = np.asarray(
        [row["initial_min_ellipse_h"] for row in rows], dtype=float
    )
    initial_h = initial_h[np.isfinite(initial_h)]
    return {
        "vehicles_count": int(vehicle_count),
        "episodes": int(len(rows)),
        "horizon_s": float(horizon_policy_steps * policy_dt),
        "initial_overlap_rate": float(
            np.mean([row["initial_ego_collision"] for row in rows])
        ),
        "initial_min_rect_separation_mean_m": (
            float(np.mean(initial_clearances)) if initial_clearances.size else np.nan
        ),
        "initial_min_rect_separation_min_m": (
            float(np.min(initial_clearances)) if initial_clearances.size else np.nan
        ),
        "initial_min_ellipse_h_mean": (
            float(np.mean(initial_h)) if initial_h.size else np.nan
        ),
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
        "mean_first_any_collision_s": native_baseline._finite_mean(
            rows, "first_any_collision_s"
        ),
        "mean_first_ego_collision_s": native_baseline._finite_mean(
            rows, "first_ego_collision_s"
        ),
        "mean_distance_m": float(np.mean([row["distance_m"] for row in rows])),
        "mean_native_ego_abs_ax_mps2": native_baseline._finite_mean(
            rows, "mean_abs_ego_ax_mps2"
        ),
        "mean_native_ego_abs_ay_mps2": native_baseline._finite_mean(
            rows, "mean_abs_ego_ay_mps2"
        ),
    }


def main() -> int:
    args = parse_args()
    if int(args.episodes) <= 0 or int(args.horizon_policy_steps) <= 0:
        raise ValueError("episodes and horizon-policy-steps must be positive")
    counts = [int(count) for count in args.vehicle_counts]
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("vehicle-counts must contain positive integers")

    native_baseline.provenance.basics.notebook_pipeline.set_stable_native_defaults()
    project_root = native_baseline.provenance.basics.notebook_pipeline.find_project_root(
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
            "This density ladder is intentionally MTM-only; "
            f"received traffic_model={traffic_model!r}"
        )

    namespace = native_baseline.provenance._bootstrap_nominal_namespace(project_root)
    seeds = [int(args.seed_start) + index for index in range(int(args.episodes))]
    policy_dt = native_baseline.provenance._policy_dt(env_config)
    episode_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for vehicle_count in counts:
        density_config = copy.deepcopy(env_config)
        density_config["vehicles_count"] = int(vehicle_count)
        density_rows: list[dict[str, Any]] = []
        for seed in seeds:
            print(
                f"[density] vehicles={vehicle_count} seed={seed}",
                flush=True,
            )
            row = native_baseline._native_episode(
                namespace,
                env_config=density_config,
                reward_config=reward_config,
                seed=seed,
                horizon_policy_steps=int(args.horizon_policy_steps),
            )
            row["vehicles_count"] = int(vehicle_count)
            density_rows.append(row)
            episode_rows.append(row)
        summaries.append(
            _aggregate_density(
                density_rows,
                vehicle_count=vehicle_count,
                horizon_policy_steps=int(args.horizon_policy_steps),
                policy_dt=policy_dt,
            )
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / "mtm_density_episode_results.csv"
    summary_path = output_dir / "mtm_density_summary.json"
    _write_csv(episode_path, episode_rows)
    summary_path.write_text(
        json.dumps(
            {
                "traffic_model": traffic_model,
                "vehicle_counts": counts,
                "episodes_per_count": int(args.episodes),
                "seed_start": int(args.seed_start),
                "horizon_policy_steps": int(args.horizon_policy_steps),
                "policy_dt_s": float(policy_dt),
                "ego_controlled": False,
                "terminate_on_collision": False,
                "summary": summaries,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2, allow_nan=True), flush=True)
    print(f"[density] wrote {episode_path}", flush=True)
    print(f"[density] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
