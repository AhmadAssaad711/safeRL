"""Nominal-only MTM collision-provenance audit.

This audit does not construct a CBF wrapper or apply a safety filter.  It
replays the same simple nominal controllers used by
``audit_nominal_ppo_basics.py`` and records the first ego collision in detail:

* whether the reset already contained an overlapping ego pair;
* the road-list index and MTM profile of the first collision partner;
* pre-impact geometry, relative motion, and longitudinal TTC;
* elapsed time and forward distance before the first impact.

The MTM collision detector is symmetric, so ``closing_direction`` is an
inference from relative motion, not a simulator-provided causal label.
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

import scripts.evaluation.audit_nominal_ppo_basics as basics


CONTROLLERS = ("zero", "full_brake", "speed_only", "reactive_nominal")
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1_300_000)
    return parser.parse_args()


def _policy_dt(env_config: dict[str, Any]) -> float:
    simulation_frequency = max(float(env_config.get("simulation_frequency", 1.0)), EPS)
    policy_frequency = max(float(env_config.get("policy_frequency", 1.0)), EPS)
    frames = max(1, int(round(simulation_frequency / policy_frequency)))
    dt = float(env_config.get("dt", 1.0 / simulation_frequency))
    return float(frames * dt)


def _bootstrap_nominal_namespace(project_root: Path) -> dict[str, Any]:
    """Load only the nominal reward wrapper, never the notebook CBF cells."""

    namespace = basics.notebook_pipeline.bootstrap_notebook_namespace(project_root)
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    matches = []
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "class KaralakouRewardWrapper" in source:
            matches.append((index, source))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one nominal KaralakouRewardWrapper definition; "
            f"found {len(matches)}"
        )
    index, source = matches[0]
    print(f"[provenance] executing nominal reward definition from cell {index}", flush=True)
    exec(compile(source, f"{notebook_path}:cell-{index}", "exec"), namespace)
    return namespace


def _snapshot(base: Any) -> list[dict[str, float | str | bool]]:
    return [
        {
            "x": float(vehicle.position[0]),
            "y": float(vehicle.position[1]),
            "vx": float(vehicle.vx),
            "vy": float(vehicle.vy),
            "length": float(vehicle.length),
            "width": float(vehicle.width),
            "desired_speed": float(vehicle.desired_speed),
            "profile": str(getattr(vehicle, "driver_profile", "unknown")),
            "is_ego": bool(vehicle.is_ego),
        }
        for vehicle in base.road.vehicles
    ]


def _ego_index(snapshot: list[dict[str, Any]]) -> int:
    for index, state in enumerate(snapshot):
        if bool(state["is_ego"]):
            return index
    return 0


def _geometry(
    base: Any,
    ego: dict[str, Any],
    other: dict[str, Any],
    *,
    eps_side: float,
) -> dict[str, float]:
    dx = float(base._signed_distance(float(ego["x"]), float(other["x"])))
    dy = float(other["y"] - ego["y"])
    x_gap = abs(dx) - 0.5 * (float(ego["length"]) + float(other["length"]))
    y_gap = abs(dy) - 0.5 * (float(ego["width"]) + float(other["width"]))
    ego_a = float(ego["length"]) / np.sqrt(2.0) + 2.0 * eps_side
    ego_b = float(ego["width"]) / np.sqrt(2.0) + 2.0 * eps_side
    other_a = float(other["length"]) / np.sqrt(2.0) + 2.0 * eps_side
    other_b = float(other["width"]) / np.sqrt(2.0) + 2.0 * eps_side
    A = max(ego_a + other_a, EPS)
    B = max(ego_b + other_b, EPS)
    ellipse_h = (dx / A) ** 2 + (dy / B) ** 2 - 1.0
    return {
        "dx": dx,
        "dy": dy,
        "x_gap": float(x_gap),
        "y_gap": float(y_gap),
        # Positive means the axis-aligned rectangles are separated in at
        # least one dimension; negative means overlap in both dimensions.
        "rect_separation": float(max(x_gap, y_gap)),
        "ellipse_h": float(ellipse_h),
        "range": float(np.hypot(dx, dy)),
    }


def _initial_geometry(
    base: Any,
    snapshot: list[dict[str, Any]],
    *,
    eps_side: float,
) -> dict[str, Any]:
    ego_index = _ego_index(snapshot)
    ego = snapshot[ego_index]
    pairs: list[tuple[int, dict[str, float]]] = []
    for index, other in enumerate(snapshot):
        if index == ego_index:
            continue
        pairs.append((index, _geometry(base, ego, other, eps_side=eps_side)))
    if not pairs:
        return {
            "initial_ego_collision": False,
            "initial_overlap_pair_count": 0,
            "initial_overlap_partner_indices": [],
            "initial_min_rect_separation_m": np.nan,
            "initial_min_ellipse_h": np.nan,
            "initial_nearest_rect_partner_index": np.nan,
            "initial_nearest_ellipse_partner_index": np.nan,
        }
    initial_overlaps = [
        index
        for index, geometry in pairs
        if geometry["x_gap"] < 0.0 and geometry["y_gap"] < 0.0
    ]
    nearest_rect = min(pairs, key=lambda item: item[1]["rect_separation"])
    nearest_ellipse = min(pairs, key=lambda item: item[1]["ellipse_h"])
    return {
        "initial_ego_collision": bool(initial_overlaps),
        "initial_overlap_pair_count": int(len(initial_overlaps)),
        "initial_overlap_partner_indices": [int(index) for index in initial_overlaps],
        "initial_min_rect_separation_m": float(nearest_rect[1]["rect_separation"]),
        "initial_min_ellipse_h": float(nearest_ellipse[1]["ellipse_h"]),
        "initial_nearest_rect_partner_index": int(nearest_rect[0]),
        "initial_nearest_ellipse_partner_index": int(nearest_ellipse[0]),
    }


def _active_ego_pairs(base: Any, ego_index: int) -> set[tuple[int, int]]:
    pairs = set(getattr(base, "_active_collision_pairs", set()))
    return {
        (int(first), int(second))
        for first, second in pairs
        if int(first) == ego_index or int(second) == ego_index
    }


def _partner_index(pair: tuple[int, int], ego_index: int) -> int:
    first, second = pair
    return int(second if first == ego_index else first)


def _closing_rates(
    pre_geometry: dict[str, float],
    post_geometry: dict[str, float],
    pre_ego: dict[str, Any],
    pre_other: dict[str, Any],
    policy_dt: float,
) -> tuple[float, float, float, float, str]:
    dx = float(pre_geometry["dx"])
    dy = float(pre_geometry["dy"])
    ego_vx = float(pre_ego["vx"])
    other_vx = float(pre_other["vx"])
    ego_vy = float(pre_ego["vy"])
    other_vy = float(pre_other["vy"])

    if dx > 0.0:
        longitudinal_closing = ego_vx - other_vx
        front_back_label = "ego_closing_into_front"
    elif dx < 0.0:
        longitudinal_closing = other_vx - ego_vx
        front_back_label = "traffic_closing_from_rear"
    else:
        longitudinal_closing = 0.0
        front_back_label = "longitudinally_aligned"

    if dy > 0.0:
        lateral_closing = ego_vy - other_vy
    elif dy < 0.0:
        lateral_closing = other_vy - ego_vy
    else:
        lateral_closing = 0.0

    range_closing = (
        float(pre_geometry["range"]) - float(post_geometry["range"])
    ) / max(policy_dt, EPS)
    if longitudinal_closing > 0.05 and abs(dx) > 0.5:
        direction = front_back_label
    elif lateral_closing > 0.05:
        direction = "lateral_closing"
    elif range_closing > 0.05:
        direction = "closing_but_direction_ambiguous"
    else:
        direction = "low_relative_closing_or_post_contact"

    longitudinal_gap = max(float(pre_geometry["x_gap"]), 0.0)
    longitudinal_ttc = (
        longitudinal_gap / longitudinal_closing
        if longitudinal_closing > 0.05
        else math.inf
    )
    return (
        float(longitudinal_closing),
        float(lateral_closing),
        float(range_closing),
        float(longitudinal_ttc),
        direction,
    )


def _action(
    controller_name: str,
    observation: np.ndarray,
    env_config: dict[str, Any],
) -> np.ndarray:
    if controller_name == "zero":
        return basics.ACTION_SWEEP["zero"]
    if controller_name == "full_brake":
        return basics.ACTION_SWEEP["negative_ax"]
    if controller_name == "speed_only":
        return basics._controller_from_observation(
            observation, env_config=env_config, reactive=False
        )
    if controller_name == "reactive_nominal":
        return basics._controller_from_observation(
            observation, env_config=env_config, reactive=True
        )
    raise ValueError(f"Unknown controller {controller_name!r}")


def _empty_collision_fields() -> dict[str, Any]:
    return {
        "collision_step": np.nan,
        "collision_time_s": np.nan,
        "time_before_collision_s": np.nan,
        "distance_before_collision_m": np.nan,
        "partner_road_index": np.nan,
        "partner_profile": "no_collision",
        "partner_desired_speed_mps": np.nan,
        "pre_dx_m": np.nan,
        "pre_dy_m": np.nan,
        "pre_ego_vx_mps": np.nan,
        "pre_partner_vx_mps": np.nan,
        "pre_ego_vy_mps": np.nan,
        "pre_partner_vy_mps": np.nan,
        "post_dx_m": np.nan,
        "post_dy_m": np.nan,
        "pre_x_gap_m": np.nan,
        "pre_y_gap_m": np.nan,
        "pre_rect_separation_m": np.nan,
        "pre_ellipse_h": np.nan,
        "pre_range_m": np.nan,
        "longitudinal_closing_rate_mps": np.nan,
        "lateral_closing_rate_mps": np.nan,
        "range_closing_rate_mps": np.nan,
        "longitudinal_ttc_s": np.nan,
        "closing_direction": "no_collision",
        "ego_ax_mps2": np.nan,
        "ego_ay_mps2": np.nan,
        "partner_ax_mps2": np.nan,
        "partner_ay_mps2": np.nan,
    }


def run_controller_provenance(
    namespace: dict[str, Any],
    *,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    seeds: list[int],
    controller_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eps_side = float(reward_config.get("safety_potential_eps_side", 0.10))
    policy_dt = _policy_dt(env_config)
    max_steps = basics._max_policy_steps(env_config) + 2

    for seed in seeds:
        env = basics.make_nominal_env(
            namespace, env_config=env_config, reward_config=reward_config
        )
        try:
            observation, _ = env.reset(seed=int(seed))
            base = basics._base(env)
            initial_snapshot = _snapshot(base)
            ego_index = _ego_index(initial_snapshot)
            initial = _initial_geometry(
                base, initial_snapshot, eps_side=eps_side
            )
            previous_x = float(base.vehicle.position[0])
            distance = 0.0
            steps = 0
            events = 0
            previous_active_pairs: set[tuple[int, int]] = set()
            first_collision = _empty_collision_fields()

            for _ in range(max_steps):
                base = basics._base(env)
                pre_snapshot = _snapshot(base)
                pre_time = float(base.time)
                pre_distance = float(distance)
                action = _action(controller_name, np.asarray(observation), env_config)
                observation, _, terminated, truncated, info = env.step(action)
                base = basics._base(env)
                post_snapshot = _snapshot(base)
                distance += basics._distance_step(base, previous_x)
                previous_x = float(base.vehicle.position[0])
                steps += 1
                step_events = basics._collision_events(base, dict(info))
                events += step_events

                active_pairs = _active_ego_pairs(base, ego_index)
                new_ego_pairs = active_pairs - previous_active_pairs
                collision_now = bool(
                    step_events > 0
                    or bool(getattr(base, "_last_ego_collision", False))
                )
                if collision_now and first_collision["closing_direction"] == "no_collision":
                    if new_ego_pairs:
                        pair = sorted(new_ego_pairs)[0]
                    elif active_pairs:
                        pair = sorted(active_pairs)[0]
                    else:
                        # Fallback for a pair that separated within a policy
                        # step: choose the smallest post-step rectangle gap.
                        candidate_indices = [
                            index for index in range(len(post_snapshot)) if index != ego_index
                        ]
                        partner = min(
                            candidate_indices,
                            key=lambda index: _geometry(
                                base,
                                post_snapshot[ego_index],
                                post_snapshot[index],
                                eps_side=eps_side,
                            )["rect_separation"],
                        )
                        pair = (ego_index, int(partner))
                    partner_index = _partner_index(pair, ego_index)
                    pre_geometry = _geometry(
                        base,
                        pre_snapshot[ego_index],
                        pre_snapshot[partner_index],
                        eps_side=eps_side,
                    )
                    post_geometry = _geometry(
                        base,
                        post_snapshot[ego_index],
                        post_snapshot[partner_index],
                        eps_side=eps_side,
                    )
                    (
                        longitudinal_closing,
                        lateral_closing,
                        range_closing,
                        longitudinal_ttc,
                        direction,
                    ) = _closing_rates(
                        pre_geometry,
                        post_geometry,
                        pre_snapshot[ego_index],
                        pre_snapshot[partner_index],
                        policy_dt,
                    )
                    accelerations = np.asarray(
                        getattr(base, "_last_accelerations", np.empty((0, 2))),
                        dtype=float,
                    )
                    ego_acc = (
                        accelerations[ego_index]
                        if accelerations.ndim == 2 and accelerations.shape[0] > ego_index
                        else np.asarray([np.nan, np.nan])
                    )
                    partner_acc = (
                        accelerations[partner_index]
                        if accelerations.ndim == 2 and accelerations.shape[0] > partner_index
                        else np.asarray([np.nan, np.nan])
                    )
                    first_collision.update(
                        {
                            "collision_step": int(steps),
                            "collision_time_s": float(base.time),
                            "time_before_collision_s": float(pre_time),
                            "distance_before_collision_m": float(pre_distance),
                            "partner_road_index": int(partner_index),
                            "partner_profile": str(
                                post_snapshot[partner_index]["profile"]
                            ),
                            "partner_desired_speed_mps": float(
                                pre_snapshot[partner_index]["desired_speed"]
                            ),
                            "pre_dx_m": float(pre_geometry["dx"]),
                            "pre_dy_m": float(pre_geometry["dy"]),
                            "pre_ego_vx_mps": float(pre_snapshot[ego_index]["vx"]),
                            "pre_partner_vx_mps": float(
                                pre_snapshot[partner_index]["vx"]
                            ),
                            "pre_ego_vy_mps": float(pre_snapshot[ego_index]["vy"]),
                            "pre_partner_vy_mps": float(
                                pre_snapshot[partner_index]["vy"]
                            ),
                            "post_dx_m": float(post_geometry["dx"]),
                            "post_dy_m": float(post_geometry["dy"]),
                            "pre_x_gap_m": float(pre_geometry["x_gap"]),
                            "pre_y_gap_m": float(pre_geometry["y_gap"]),
                            "pre_rect_separation_m": float(
                                pre_geometry["rect_separation"]
                            ),
                            "pre_ellipse_h": float(pre_geometry["ellipse_h"]),
                            "pre_range_m": float(pre_geometry["range"]),
                            "longitudinal_closing_rate_mps": longitudinal_closing,
                            "lateral_closing_rate_mps": lateral_closing,
                            "range_closing_rate_mps": range_closing,
                            "longitudinal_ttc_s": longitudinal_ttc,
                            "closing_direction": direction,
                            "ego_ax_mps2": float(ego_acc[0]),
                            "ego_ay_mps2": float(ego_acc[1]),
                            "partner_ax_mps2": float(partner_acc[0]),
                            "partner_ay_mps2": float(partner_acc[1]),
                        }
                    )
                previous_active_pairs = active_pairs
                if terminated or truncated:
                    break

            row = {
                "controller": controller_name,
                "seed": int(seed),
                "steps": int(steps),
                "collision": int(first_collision["closing_direction"] != "no_collision"),
                "collision_events": int(events),
                "distance_total_m": float(distance),
                **initial,
                **first_collision,
            }
            rows.append(row)
        finally:
            env.close()
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_mean(rows: list[dict[str, Any]], field: str) -> float:
    values = np.asarray([row[field] for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for controller in CONTROLLERS:
        subset = [row for row in rows if row["controller"] == controller]
        collisions = [row for row in subset if int(row["collision"]) == 1]
        direction_counts: dict[str, int] = {}
        profile_counts: dict[str, int] = {}
        for row in collisions:
            direction = str(row["closing_direction"])
            profile = str(row["partner_profile"])
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
            profile_counts[profile] = profile_counts.get(profile, 0) + 1
        summaries.append(
            {
                "controller": controller,
                "episodes": int(len(subset)),
                "initial_overlap_rate": float(
                    np.mean([row["initial_ego_collision"] for row in subset])
                ),
                "initial_min_rect_separation_mean_m": _finite_mean(
                    subset, "initial_min_rect_separation_m"
                ),
                "initial_min_ellipse_h_mean": _finite_mean(
                    subset, "initial_min_ellipse_h"
                ),
                "collision_episode_rate": float(
                    np.mean([row["collision"] for row in subset])
                ),
                "mean_collision_time_s": _finite_mean(collisions, "collision_time_s"),
                "mean_time_before_collision_s": _finite_mean(
                    collisions, "time_before_collision_s"
                ),
                "mean_distance_before_collision_m": _finite_mean(
                    collisions, "distance_before_collision_m"
                ),
                "mean_pre_rect_separation_m": _finite_mean(
                    collisions, "pre_rect_separation_m"
                ),
                "mean_pre_ellipse_h": _finite_mean(collisions, "pre_ellipse_h"),
                "mean_pre_range_closing_rate_mps": _finite_mean(
                    collisions, "range_closing_rate_mps"
                ),
                "mean_longitudinal_ttc_s_finite": _finite_mean(
                    [
                        row
                        for row in collisions
                        if np.isfinite(float(row["longitudinal_ttc_s"]))
                    ],
                    "longitudinal_ttc_s",
                ),
                "direction_counts": json.dumps(direction_counts, sort_keys=True),
                "partner_profile_counts": json.dumps(profile_counts, sort_keys=True),
            }
        )
    return summaries


def main() -> int:
    args = parse_args()
    if int(args.episodes) <= 0:
        raise ValueError("episodes must be positive")

    basics.notebook_pipeline.set_stable_native_defaults()
    project_root = basics.notebook_pipeline.find_project_root(
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
            "This provenance audit is intentionally MTM-only; "
            f"received traffic_model={traffic_model!r}"
        )

    namespace = _bootstrap_nominal_namespace(project_root)
    seeds = [int(args.seed_start) + index for index in range(int(args.episodes))]
    all_rows: list[dict[str, Any]] = []
    for controller in CONTROLLERS:
        print(
            f"[provenance] controller={controller} episodes={len(seeds)} traffic_model=mtm",
            flush=True,
        )
        all_rows.extend(
            run_controller_provenance(
                namespace,
                env_config=env_config,
                reward_config=reward_config,
                seeds=seeds,
                controller_name=controller,
            )
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / "collision_provenance_episodes.csv"
    summary_path = output_dir / "collision_provenance_summary.csv"
    json_path = output_dir / "collision_provenance_summary.json"
    summaries = _summary(all_rows)
    _write_csv(episode_path, all_rows)
    _write_csv(summary_path, summaries)
    json_path.write_text(
        json.dumps(
            {
                "traffic_model": traffic_model,
                "episodes_per_controller": int(len(seeds)),
                "seed_start": int(args.seed_start),
                "policy_dt_s": _policy_dt(env_config),
                "controllers": list(CONTROLLERS),
                "causal_note": (
                    "MTM detects symmetric collision pairs. The reported "
                    "closing_direction is inferred from pre-impact relative "
                    "motion and is not a simulator causal label."
                ),
                "summary": summaries,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2, allow_nan=True), flush=True)
    print(f"[provenance] wrote {episode_path}", flush=True)
    print(f"[provenance] wrote {summary_path}", flush=True)
    print(f"[provenance] wrote {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
