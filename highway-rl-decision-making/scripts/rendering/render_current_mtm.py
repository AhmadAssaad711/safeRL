"""Render the current MTM-only traffic simulator to an MP4.

The reference vehicle is also MTM-controlled: no RL policy, CBF wrapper, or
force-model traffic is involved.  Environment settings are loaded from a run
configuration so the video uses the same physical setup as the experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import cv2
import gymnasium as gym
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1_100_000)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--profile-mix",
        type=float,
        nargs=3,
        metavar=("CAUTIOUS", "NORMAL", "AGGRESSIVE"),
        help="Override continuous-personality population masses.",
    )
    return parser.parse_args()


def install_local_environment(project_root: Path) -> None:
    environment_root = project_root / "laneless highway env"
    vendored_highway_env = environment_root / "HighwayEnv"
    for path in (vendored_highway_env, environment_root):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    import lane_free_env  # noqa: F401


def read_env_config(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document.get("env_config"), dict):
        return copy.deepcopy(document["env_config"])
    training_signature = document.get("training_signature", {})
    if isinstance(training_signature, dict) and isinstance(
        training_signature.get("env_config"), dict
    ):
        return copy.deepcopy(training_signature["env_config"])
    raise RuntimeError(f"No env_config found in {path}")


def output_stem(config: dict[str, Any], seed: int) -> str:
    guard = bool(config.get("traffic_safety", {}).get("dynamics_guard", False))
    vehicles = int(config.get("vehicles_count", 0))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"current_mtm_guard_{'on' if guard else 'off'}_"
        f"{vehicles}veh_seed{seed}_{timestamp}"
    )


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.fps <= 0:
        raise ValueError("--steps and --fps must be positive")

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    project_root = args.project_root.resolve()
    install_local_environment(project_root)
    config_path = args.config.resolve()
    env_config = read_env_config(config_path)
    traffic_model = str(env_config.get("traffic_model", "")).strip().lower()
    if traffic_model != "mtm":
        raise RuntimeError(
            "This renderer is MTM-only; "
            f"the supplied config selects traffic_model={traffic_model!r}"
        )
    # This script renders the current experiment, so opt in explicitly even
    # when the supplied signature predates the continuous-personality flag.
    env_config.setdefault("mtm", {})["continuous_driver_aggressiveness"] = True
    if args.profile_mix is not None:
        cautious, normal, aggressive = (float(value) for value in args.profile_mix)
        if min(cautious, normal, aggressive) < 0.0:
            raise ValueError("--profile-mix values must be nonnegative")
        total = cautious + normal + aggressive
        if total <= 0.0:
            raise ValueError("--profile-mix must have a positive total")
        env_config["mtm"]["profile_probabilities"] = {
            "cautious": cautious / total,
            "normal": normal / total,
            "aggressive": aggressive / total,
        }

    # Make vehicle 0 a normal member of the MTM traffic stream.  The action
    # passed to step() is consequently ignored by the simulator dynamics.
    env_config["ego_controlled"] = False
    env_config["terminate_on_collision"] = False
    env_config["real_time_rendering"] = False
    env_config["offscreen_rendering"] = True
    configured_policy_frequency = float(env_config["policy_frequency"])
    frames_per_video_frame = max(
        1,
        int(
            round(
                float(env_config["simulation_frequency"])
                / configured_policy_frequency
            )
        ),
    )
    required_physics_steps = int(args.steps) * frames_per_video_frame
    env_config["episode_steps"] = max(
        int(env_config.get("episode_steps", required_physics_steps)),
        required_physics_steps,
    )
    # Step at the physics frequency so collision events that begin during the
    # first half of a 10 Hz policy interval are not hidden by the second half.
    # Two 20 Hz physics steps are still combined into each 10 Hz video frame.
    env_config["policy_frequency"] = float(env_config["simulation_frequency"])

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(env_config, int(args.seed))
    video_path = output_dir / f"{stem}.mp4"
    preview_path = output_dir / f"{stem}_preview.png"
    summary_path = output_dir / f"{stem}_summary.json"

    env = gym.make("lane-free-v0", render_mode="rgb_array", config=env_config)
    writer: cv2.VideoWriter | None = None
    collision_events = 0
    ego_collision_events = 0
    max_active_collisions = 0
    mean_speeds: list[float] = []
    guard_constraints = 0
    guard_brakes = 0
    guard_lateral_yields = 0
    last_info: dict[str, Any] = {}
    rendered_steps = 0
    physics_steps = 0

    try:
        env.reset(seed=int(args.seed))
        for _ in range(int(args.steps)):
            terminated = False
            truncated = False
            for _ in range(frames_per_video_frame):
                _, _, terminated, truncated, info = env.step(
                    np.zeros(2, dtype=np.float32)
                )
                last_info = dict(info)
                physics_steps += 1
                collision_events += max(int(info.get("collisions", 0)), 0)
                ego_collision_events += max(
                    int(info.get("ego_collision_events", 0)), 0
                )
                max_active_collisions = max(
                    max_active_collisions,
                    max(int(info.get("active_collisions", 0)), 0),
                )
                guard_constraints += max(
                    int(info.get("traffic_guard_constraints", 0)), 0
                )
                guard_brakes += max(
                    int(info.get("traffic_guard_brakes", 0)), 0
                )
                guard_lateral_yields += max(
                    int(info.get("traffic_guard_lateral_yields", 0)), 0
                )
                if terminated or truncated:
                    break

            frame = np.asarray(env.render(), dtype=np.uint8)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(f"Unexpected render frame shape: {frame.shape}")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if writer is None:
                height, width = frame_bgr.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(args.fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open MP4 writer: {video_path}")
                if not cv2.imwrite(str(preview_path), frame_bgr):
                    raise RuntimeError(f"Could not write preview: {preview_path}")
            writer.write(frame_bgr)

            rendered_steps += 1
            mean_speeds.append(float(last_info.get("mean_speed", np.nan)))
            if terminated or truncated:
                break
    finally:
        if writer is not None:
            writer.release()
        env.close()

    if rendered_steps == 0 or not video_path.is_file():
        raise RuntimeError("No video frames were produced")

    physics_dt = float(env_config.get("dt", 1.0 / env_config["simulation_frequency"]))
    summary = {
        "video_path": str(video_path),
        "preview_path": str(preview_path),
        "config_path": str(config_path),
        "seed": int(args.seed),
        "traffic_model": traffic_model,
        "all_vehicles_mtm_controlled": True,
        "continuous_driver_aggressiveness": bool(
            env_config.get("mtm", {}).get(
                "continuous_driver_aggressiveness", False
            )
        ),
        "profile_probabilities": dict(
            env_config.get("mtm", {}).get("profile_probabilities", {})
        ),
        "realized_profile_counts": {
            name: int(last_info.get(f"mtm_profile_count_{name}", 0))
            for name in ("cautious", "normal", "aggressive")
        },
        "traffic_dynamics_guard": bool(
            env_config.get("traffic_safety", {}).get("dynamics_guard", False)
        ),
        "safe_spawn": bool(
            env_config.get("traffic_safety", {}).get("safe_spawn", False)
        ),
        "vehicles": int(env_config["vehicles_count"]),
        "road_length_m": float(env_config["road_length"]),
        "road_width_m": float(env_config["road_width"]),
        "physics_dt_s": physics_dt,
        "configured_policy_frequency_hz": configured_policy_frequency,
        "video_fps": int(args.fps),
        "policy_steps_rendered": rendered_steps,
        "physics_steps_rendered": physics_steps,
        "simulated_seconds": physics_steps * physics_dt,
        "collision_events": collision_events,
        "ego_collision_events": ego_collision_events,
        "max_active_collisions": max_active_collisions,
        "mean_speed_mps_over_video": float(np.nanmean(mean_speeds)),
        "final_mean_speed_mps": float(last_info.get("mean_speed", np.nan)),
        "guard_constraints_observed": guard_constraints,
        "guard_brakes_observed": guard_brakes,
        "guard_lateral_yields_observed": guard_lateral_yields,
        "mtm_aggressiveness_mean": float(
            last_info.get("mtm_aggressiveness_mean", np.nan)
        ),
        "mtm_aggressiveness_std": float(
            last_info.get("mtm_aggressiveness_std", np.nan)
        ),
        "mtm_aggressiveness_min": float(
            last_info.get("mtm_aggressiveness_min", np.nan)
        ),
        "mtm_aggressiveness_max": float(
            last_info.get("mtm_aggressiveness_max", np.nan)
        ),
        "mtm_aggressiveness_unique": int(
            last_info.get("mtm_aggressiveness_unique", 0)
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    print(f"[render-current-mtm] wrote {video_path}", flush=True)
    print(f"[render-current-mtm] wrote {preview_path}", flush=True)
    print(f"[render-current-mtm] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
