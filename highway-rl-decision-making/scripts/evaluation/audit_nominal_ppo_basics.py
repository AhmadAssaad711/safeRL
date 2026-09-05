"""Nominal PPO basics audit for the MTM lane-free environment.

This diagnostic deliberately excludes the CBF action/context wrapper.  It
builds the simulator, the Karalakou reward wrapper, and a small physical-action
adapter so that the action interface matches nominal PPO while the underlying
environment remains the raw MTM simulator.

The audit covers three questions:

1. Does each action produce the expected ego-state change?
2. Are the observations finite, scaled, deterministic, and informative about
   the vehicles used by the safety calculation?
3. Can simple nominal controllers make progress in the same MTM scenarios?
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np

import scripts.training.run_cbf_filter_ablation as notebook_pipeline


ACTION_SWEEP = {
    "zero": np.asarray([0.0, 0.0], dtype=np.float32),
    "positive_ax": np.asarray([3.0, 0.0], dtype=np.float32),
    "negative_ax": np.asarray([-3.0, 0.0], dtype=np.float32),
    "positive_ay": np.asarray([0.0, 3.0], dtype=np.float32),
    "negative_ay": np.asarray([0.0, -3.0], dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1_300_000)
    parser.add_argument("--scan-steps", type=int, default=80)
    parser.add_argument(
        "--observation-mode",
        choices=("full", "minimal"),
        default=None,
        help="Optional observation-layout override for the audit.",
    )
    return parser.parse_args()


class NominalPhysicalActionAdapter(gym.Wrapper):
    """Expose physical acceleration actions without any CBF logic."""

    def __init__(self, env: gym.Env, env_config: dict[str, Any]) -> None:
        super().__init__(env)
        bounds = env_config["bounds"]
        self.physical_low = np.asarray(
            [float(bounds["ax_min"]), float(bounds["ay_min"])], dtype=np.float32
        )
        self.physical_high = np.asarray(
            [float(bounds["ax_max"]), float(bounds["ay_max"])], dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=self.physical_low,
            high=self.physical_high,
            dtype=np.float32,
        )

    def physical_to_normalized(self, action: Any) -> np.ndarray:
        physical = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        physical = np.clip(physical, self.physical_low, self.physical_high)
        normalized = np.zeros(2, dtype=np.float32)
        for index, value in enumerate(physical):
            if value >= 0.0:
                scale = max(float(self.physical_high[index]), 1e-6)
            else:
                scale = max(abs(float(self.physical_low[index])), 1e-6)
            normalized[index] = float(value) / scale
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def step(self, action: Any):
        return self.env.step(self.physical_to_normalized(action))


def make_nominal_env(
    namespace: dict[str, Any],
    *,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
) -> gym.Env:
    """Build the raw MTM environment; no CBF wrapper is inserted."""

    env = gym.make(
        "lane-free-v0",
        render_mode=None,
        config=copy.deepcopy(env_config),
    )
    env = namespace["KaralakouRewardWrapper"](
        env,
        reward_config=copy.deepcopy(reward_config),
    )
    return NominalPhysicalActionAdapter(env, env_config)


def _base(env: gym.Env) -> Any:
    return env.unwrapped


def _max_policy_steps(env_config: dict[str, Any]) -> int:
    physics_steps = float(env_config.get("episode_steps", env_config.get("duration", 800)))
    simulation_frequency = max(float(env_config.get("simulation_frequency", 1.0)), 1e-6)
    policy_frequency = max(float(env_config.get("policy_frequency", 1.0)), 1e-6)
    frames = max(1, int(round(simulation_frequency / policy_frequency)))
    return int(math.ceil(physics_steps / frames))


def _state(base: Any) -> np.ndarray:
    ego = base.vehicle
    return np.asarray(
        [
            float(ego.position[0]),
            float(ego.position[1]),
            float(ego.vx),
            float(ego.vy),
        ],
        dtype=float,
    )


def _distance_step(base: Any, previous_x: float) -> float:
    return float(max(base._signed_distance(previous_x, float(base.vehicle.position[0])), 0.0))


def _collision_events(base: Any, info: dict[str, Any]) -> int:
    return max(
        int(getattr(base, "_last_ego_collision_count", 0)),
        int(info.get("ego_collision_events", 0)),
    )


def _ellipse_h(base: Any, vehicle: Any, eps_side: float) -> float:
    ego = base.vehicle
    dx = float(base._signed_distance(ego.position[0], vehicle.position[0]))
    dy = float(vehicle.position[1] - ego.position[1])
    ego_a = float(ego.length) / np.sqrt(2.0) + 2.0 * eps_side
    ego_b = float(ego.width) / np.sqrt(2.0) + 2.0 * eps_side
    other_a = float(vehicle.length) / np.sqrt(2.0) + 2.0 * eps_side
    other_b = float(vehicle.width) / np.sqrt(2.0) + 2.0 * eps_side
    A = max(ego_a + other_a, 1e-6)
    B = max(ego_b + other_b, 1e-6)
    return float((dx / A) ** 2 + (dy / B) ** 2 - 1.0)


def _neighbor_visibility(
    base: Any,
    *,
    neighbor_count: int,
    sensing_range: float,
    eps_side: float,
) -> dict[str, Any]:
    ego = base.vehicle
    entries: list[tuple[float, Any, float, float, float]] = []
    for vehicle in base.road.vehicles:
        if vehicle is ego:
            continue
        dx = float(base._signed_distance(ego.position[0], vehicle.position[0]))
        dy = float(vehicle.position[1] - ego.position[1])
        distance_squared = dx * dx + dy * dy
        entries.append(
            (distance_squared, vehicle, dx, dy, _ellipse_h(base, vehicle, eps_side))
        )
    entries.sort(key=lambda item: item[0])
    visible = entries[: int(neighbor_count)]
    sensed = [entry for entry in entries if abs(entry[2]) <= float(sensing_range)]
    dangerous = min(sensed, key=lambda item: item[4]) if sensed else None
    visible_ids = {id(entry[1]) for entry in visible}
    return {
        "order": tuple(id(entry[1]) for entry in visible),
        "visible_ids": visible_ids,
        "sensed_count": len(sensed),
        "dangerous_visible": bool(dangerous is not None and id(dangerous[1]) in visible_ids),
        "dangerous_h": np.nan if dangerous is None else float(dangerous[4]),
    }


def _observation_feature_names(env_config: dict[str, Any]) -> tuple[str, ...]:
    mode = str(env_config.get("observation_mode", "full")).strip().lower()
    if mode == "full":
        return ("dx", "dy", "vx", "vy", "length", "width", "desired_speed")
    if mode == "minimal":
        return ("dx", "dy", "vx", "vy", "desired_speed")
    raise ValueError(f"Unsupported observation_mode={mode!r}")


def _controller_from_observation(
    observation: np.ndarray,
    *,
    env_config: dict[str, Any],
    reactive: bool,
) -> np.ndarray:
    feature_names = _observation_feature_names(env_config)
    rows = np.asarray(observation, dtype=np.float32).reshape(-1, len(feature_names))
    road_width = float(env_config["road_width"])
    sensing_range = float(env_config["sensing_range"])
    observation_vmax = float(env_config.get("observation_vmax", 30.0))
    observation_vymax = float(env_config.get("observation_vymax", 9.0))
    ego_vx = float(rows[0, 2]) * observation_vmax
    desired_speed = float(rows[0, feature_names.index("desired_speed")]) * observation_vmax
    ego_y = 0.5 * road_width + 0.5 * road_width * float(rows[0, 0])
    ax = float(np.clip(0.8 * (desired_speed - ego_vx), -3.0, 3.0))
    ay = float(np.clip(0.5 * (0.5 * road_width - ego_y), -3.0, 3.0))
    if not reactive:
        return np.asarray([ax, 0.0], dtype=np.float32)

    candidates: list[tuple[float, float, float]] = []
    for row in rows[1:]:
        desired_speed_index = feature_names.index("desired_speed")
        if float(row[desired_speed_index]) <= 0.0:
            continue
        dx = float(row[0]) * sensing_range
        dy = float(row[1]) * road_width
        if dx <= 0.0 or abs(dy) > 2.5:
            continue
        neighbor_vx = float(row[2]) * observation_vmax
        closing_speed = ego_vx - neighbor_vx
        gap = dx - 0.5 * (3.5 + 3.5)
        ttc = gap / max(closing_speed, 1e-6)
        risk_score = min(dx, 25.0) + 2.0 * abs(dy)
        candidates.append((risk_score, dy, ttc))
    if not candidates:
        return np.asarray([ax, ay], dtype=np.float32)

    _, dy, ttc = min(candidates, key=lambda item: item[0])
    if ttc < 4.0 or abs(dy) < 1.0:
        ax = -3.0
        if abs(dy) >= 0.25:
            direction = -float(np.sign(dy))
        else:
            direction = float(np.sign(0.5 * road_width - ego_y))
        if ego_y < 1.0:
            direction = 1.0
        elif ego_y > road_width - 1.0:
            direction = -1.0
        ay = 3.0 * direction
    return np.asarray([ax, ay], dtype=np.float32)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_action_response(
    namespace: dict[str, Any],
    *,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, action in ACTION_SWEEP.items():
        for seed in seeds:
            env = make_nominal_env(
                namespace, env_config=env_config, reward_config=reward_config
            )
            try:
                env.reset(seed=int(seed))
                base = _base(env)
                before = _state(base)
                _, _, terminated, truncated, info = env.step(action)
                after = _state(base)
                acceleration = np.asarray(base._last_accelerations[0], dtype=float)
                rows.append(
                    {
                        "action": label,
                        "seed": int(seed),
                        # MTM uses a periodic longitudinal road coordinate.  Use
                        # the environment's signed-distance helper rather than
                        # subtracting wrapped x-coordinates directly.
                        "delta_x_m": float(_distance_step(base, before[0])),
                        "delta_y_m": float(after[1] - before[1]),
                        "delta_vx_mps": float(after[2] - before[2]),
                        "delta_vy_mps": float(after[3] - before[3]),
                        "actual_ax_mps2": float(acceleration[0]),
                        "actual_ay_mps2": float(acceleration[1]),
                        "collision": int(_collision_events(base, dict(info)) > 0),
                        "terminated": int(bool(terminated)),
                        "truncated": int(bool(truncated)),
                    }
                )
            finally:
                env.close()
    aggregates: list[dict[str, Any]] = []
    for label in ACTION_SWEEP:
        subset = [row for row in rows if row["action"] == label]
        aggregates.append(
            {
                "action": label,
                "episodes": len(subset),
                "mean_delta_x_m": float(np.mean([row["delta_x_m"] for row in subset])),
                "mean_delta_y_m": float(np.mean([row["delta_y_m"] for row in subset])),
                "mean_delta_vx_mps": float(np.mean([row["delta_vx_mps"] for row in subset])),
                "mean_delta_vy_mps": float(np.mean([row["delta_vy_mps"] for row in subset])),
                "mean_actual_ax_mps2": float(np.mean([row["actual_ax_mps2"] for row in subset])),
                "mean_actual_ay_mps2": float(np.mean([row["actual_ay_mps2"] for row in subset])),
                "first_step_collision_rate": float(np.mean([row["collision"] for row in subset])),
            }
        )
    return aggregates


def run_observation_audit(
    namespace: dict[str, Any],
    *,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    seeds: list[int],
    scan_steps: int,
) -> dict[str, Any]:
    observations: list[np.ndarray] = []
    order_changes = 0
    set_changes = 0
    order_transitions = 0
    dangerous_count = 0
    dangerous_visible_count = 0
    sensed_counts: list[int] = []
    dangerous_h: list[float] = []
    reset_determinism_errors: list[float] = []
    valid_count = 0
    observation_space_shape: list[int] | None = None
    observation_space_low: list[float] | None = None
    observation_space_high: list[float] | None = None
    neighbor_count = int(env_config["neighbors_count"])
    feature_names_by_row = _observation_feature_names(env_config)
    sensing_range = float(env_config["sensing_range"])
    eps_side = float(reward_config.get("safety_potential_eps_side", 0.10))

    for seed in seeds:
        env = make_nominal_env(
            namespace, env_config=env_config, reward_config=reward_config
        )
        try:
            if observation_space_shape is None:
                observation_space_shape = list(env.observation_space.shape)
                observation_space_low = np.asarray(
                    env.observation_space.low, dtype=np.float64
                ).tolist()
                observation_space_high = np.asarray(
                    env.observation_space.high, dtype=np.float64
                ).tolist()
            first, _ = env.reset(seed=int(seed))
            second, _ = env.reset(seed=int(seed))
            reset_determinism_errors.append(
                float(np.max(np.abs(np.asarray(first) - np.asarray(second))))
            )
            observation = np.asarray(first, dtype=np.float32)
            previous_order: tuple[int, ...] | None = None
            previous_set: set[int] | None = None
            for _ in range(int(scan_steps)):
                observations.append(observation.copy())
                valid_count += int(bool(env.observation_space.contains(observation)))
                base = _base(env)
                visibility = _neighbor_visibility(
                    base,
                    neighbor_count=neighbor_count,
                    sensing_range=sensing_range,
                    eps_side=eps_side,
                )
                current_order = visibility["order"]
                current_set = set(visibility["visible_ids"])
                if previous_order is not None:
                    order_transitions += 1
                    order_changes += int(current_order != previous_order)
                    set_changes += int(current_set != previous_set)
                previous_order = current_order
                previous_set = current_set
                sensed_counts.append(int(visibility["sensed_count"]))
                if np.isfinite(visibility["dangerous_h"]):
                    dangerous_count += 1
                    dangerous_visible_count += int(visibility["dangerous_visible"])
                    dangerous_h.append(float(visibility["dangerous_h"]))
                action = _controller_from_observation(
                    observation, env_config=env_config, reactive=True
                )
                observation, _, terminated, truncated, _ = env.step(action)
                observation = np.asarray(observation, dtype=np.float32)
                if terminated or truncated:
                    break
        finally:
            env.close()

    feature_names: list[str] = [
        "ego_y_centered",
        "ego_target_hidden",
        *[f"ego_{name}" for name in feature_names_by_row[2:]],
    ]
    neighbor_features = list(feature_names_by_row)
    for row_index in range(1, 1 + neighbor_count):
        feature_names.extend([f"neighbor_{row_index}_{name}" for name in neighbor_features])
    matrix = (
        np.vstack(observations)
        if observations
        else np.empty((0, len(feature_names)))
    )
    feature_stats = []
    for index, name in enumerate(feature_names):
        values = matrix[:, index] if matrix.size else np.asarray([], dtype=float)
        feature_stats.append(
            {
                "feature": name,
                "min": float(np.min(values)) if values.size else np.nan,
                "max": float(np.max(values)) if values.size else np.nan,
                "mean": float(np.mean(values)) if values.size else np.nan,
                "std": float(np.std(values)) if values.size else np.nan,
                "p99_abs": float(np.percentile(np.abs(values), 99)) if values.size else np.nan,
                "near_constant": bool(values.size and np.std(values) < 1e-8),
            }
        )
    return {
        "observation_mode": str(env_config.get("observation_mode", "full")),
        "observation_shape": list(matrix.shape[1:]) if matrix.ndim == 2 else [],
        "declared_observation_space_shape": observation_space_shape or [],
        "declared_observation_space_low_min": float(
            np.min(observation_space_low)
        ) if observation_space_low else np.nan,
        "declared_observation_space_high_max": float(
            np.max(observation_space_high)
        ) if observation_space_high else np.nan,
        "declared_observation_space_is_finite": bool(
            observation_space_low
            and observation_space_high
            and np.isfinite(observation_space_low).all()
            and np.isfinite(observation_space_high).all()
        ),
        "sample_count": int(len(matrix)),
        "finite_fraction": float(np.isfinite(matrix).mean()) if matrix.size else 0.0,
        "observation_space_contains_fraction": float(valid_count / max(len(matrix), 1)),
        "reset_determinism_max_abs_error": float(max(reset_determinism_errors, default=np.nan)),
        "mean_sensed_vehicle_count": float(np.mean(sensed_counts)) if sensed_counts else np.nan,
        "dangerous_vehicle_visibility_fraction": float(
            dangerous_visible_count / max(dangerous_count, 1)
        ),
        "dangerous_vehicle_count": int(dangerous_count),
        "mean_dangerous_h": float(np.mean(dangerous_h)) if dangerous_h else np.nan,
        "neighbor_order_change_rate": float(order_changes / max(order_transitions, 1)),
        "neighbor_set_change_rate": float(set_changes / max(order_transitions, 1)),
        "feature_stats": feature_stats,
    }


def run_controller(
    namespace: dict[str, Any],
    *,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    seeds: list[int],
    controller_name: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        env = make_nominal_env(
            namespace, env_config=env_config, reward_config=reward_config
        )
        try:
            observation, _ = env.reset(seed=int(seed))
            base = _base(env)
            min_h = np.inf
            distance = 0.0
            events = 0
            steps = 0
            collided = False
            previous_x = float(base.vehicle.position[0])
            for _ in range(_max_policy_steps(env_config) + 2):
                if controller_name == "zero":
                    action = ACTION_SWEEP["zero"]
                elif controller_name == "full_brake":
                    action = ACTION_SWEEP["negative_ax"]
                elif controller_name == "speed_only":
                    action = _controller_from_observation(
                        observation, env_config=env_config, reactive=False
                    )
                elif controller_name == "reactive_nominal":
                    action = _controller_from_observation(
                        observation, env_config=env_config, reactive=True
                    )
                else:
                    raise ValueError(f"Unknown controller {controller_name!r}")
                base = _base(env)
                min_h = min(
                    min_h,
                    *[
                        _ellipse_h(base, vehicle, float(reward_config.get("safety_potential_eps_side", 0.10)))
                        for vehicle in base.road.vehicles
                        if vehicle is not base.vehicle
                    ],
                )
                observation, _, terminated, truncated, info = env.step(action)
                base = _base(env)
                distance += _distance_step(base, previous_x)
                previous_x = float(base.vehicle.position[0])
                step_events = _collision_events(base, dict(info))
                events += step_events
                collided = collided or step_events > 0 or bool(getattr(base, "_last_ego_collision", False))
                steps += 1
                if terminated or truncated:
                    break
            rows.append(
                {
                    "seed": int(seed),
                    "steps": int(steps),
                    "collision": int(collided),
                    "events": int(events),
                    "distance_m": float(distance),
                    "min_h": float(min_h),
                }
            )
        finally:
            env.close()
    total_distance = float(sum(row["distance_m"] for row in rows))
    return {
        "controller": controller_name,
        "episodes": len(rows),
        "safe_episodes": int(sum(row["collision"] == 0 for row in rows)),
        "collision_episode_rate": float(np.mean([row["collision"] for row in rows])),
        "collision_events": int(sum(row["events"] for row in rows)),
        "collisions_per_km": float(
            1000.0 * sum(row["events"] for row in rows) / max(total_distance, 1e-9)
        ),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_distance_m": float(np.mean([row["distance_m"] for row in rows])),
        "mean_min_h": float(np.mean([row["min_h"] for row in rows])),
    }


def main() -> int:
    notebook_pipeline.set_stable_native_defaults()
    args = parse_args()
    if int(args.episodes) <= 0 or int(args.scan_steps) <= 0:
        raise ValueError("episodes and scan-steps must be positive")
    project_root = notebook_pipeline.find_project_root(
        args.project_root or Path.cwd()
    )
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    env_config = copy.deepcopy(config["env_config"])
    reward_config = copy.deepcopy(config["reward_config"])
    if args.observation_mode is not None:
        env_config["observation_mode"] = str(args.observation_mode)
    traffic_model = str(env_config.get("traffic_model", "")).strip().lower()
    if traffic_model != "mtm":
        raise RuntimeError(
            "This nominal PPO basics audit is intentionally MTM-only; "
            f"received traffic_model={traffic_model!r}"
        )
    namespace = notebook_pipeline.bootstrap_notebook_namespace(project_root)
    notebook_pipeline.exec_required_notebook_cells(
        project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    seeds = [int(args.seed_start) + index for index in range(int(args.episodes))]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    action_response = run_action_response(
        namespace, env_config=env_config, reward_config=reward_config, seeds=seeds
    )
    observation_audit = run_observation_audit(
        namespace,
        env_config=env_config,
        reward_config=reward_config,
        seeds=seeds,
        scan_steps=int(args.scan_steps),
    )
    controller_results = [
        run_controller(
            namespace,
            env_config=env_config,
            reward_config=reward_config,
            seeds=seeds,
            controller_name=name,
        )
        for name in ("zero", "full_brake", "speed_only", "reactive_nominal")
    ]

    _write_csv(output_dir / "action_response_summary.csv", action_response)
    _write_csv(output_dir / "nominal_controller_summary.csv", controller_results)
    (output_dir / "observation_audit.json").write_text(
        json.dumps(observation_audit, indent=2, allow_nan=True), encoding="utf-8"
    )
    summary = {
        "traffic_model": traffic_model,
        "observation_mode": str(env_config.get("observation_mode", "full")),
        "episodes": int(args.episodes),
        "seed_start": int(args.seed_start),
        "run_config": str(config_path),
        "action_response": action_response,
        "observation_audit": observation_audit,
        "nominal_controller_results": controller_results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
