"""Run the saved current MTM simulator in a live pygame window.

This entry point intentionally does not create a video writer or save frames.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import numpy as np


DEFAULT_CONFIG = Path("configs") / "current_mtm_live.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=1_100_000)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Maximum live steps; 0 keeps the simulator running until stopped.",
    )
    parser.add_argument("--vehicles", type=int, default=None)
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
    if not isinstance(document.get("env_config"), dict):
        raise RuntimeError(f"No env_config found in {path}")
    return copy.deepcopy(document["env_config"])


def apply_profile_mix(config: dict[str, Any], values: tuple[float, float, float]) -> None:
    cautious, normal, aggressive = (float(value) for value in values)
    if min(cautious, normal, aggressive) < 0.0:
        raise ValueError("--profile-mix values must be nonnegative")
    total = cautious + normal + aggressive
    if total <= 0.0:
        raise ValueError("--profile-mix must have a positive total")
    config.setdefault("mtm", {})["profile_probabilities"] = {
        "cautious": cautious / total,
        "normal": normal / total,
        "aggressive": aggressive / total,
    }


def profile_counts(info: dict[str, Any]) -> dict[str, int]:
    return {
        name: int(info.get(f"mtm_profile_count_{name}", 0))
        for name in ("cautious", "normal", "aggressive")
    }


def main() -> int:
    args = parse_args()
    if args.steps < 0:
        raise ValueError("--steps must be nonnegative")
    if args.vehicles is not None and args.vehicles <= 0:
        raise ValueError("--vehicles must be positive")

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    project_root = args.project_root.resolve()
    install_local_environment(project_root)
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    env_config = read_env_config(config_path.resolve())

    if str(env_config.get("traffic_model", "")).strip().lower() != "mtm":
        raise RuntimeError("The current live simulator requires traffic_model='mtm'.")
    if args.vehicles is not None:
        env_config["vehicles_count"] = int(args.vehicles)
    if args.profile_mix is not None:
        apply_profile_mix(env_config, tuple(args.profile_mix))

    env_config.setdefault("mtm", {})["continuous_driver_aggressiveness"] = True
    env_config.setdefault("traffic_safety", {})["safe_spawn"] = True
    env_config["traffic_safety"]["dynamics_guard"] = True
    env_config["ego_controlled"] = False
    env_config["terminate_on_collision"] = False
    env_config["real_time_rendering"] = True
    env_config["offscreen_rendering"] = False
    if args.steps > 0:
        env_config["episode_steps"] = max(
            int(env_config.get("episode_steps", args.steps)), args.steps
        )

    env = gym.make("lane-free-v0", render_mode="human", config=env_config)
    print(
        "[current-mtm-live] live display active; no video recording is enabled.",
        flush=True,
    )
    print(
        json.dumps(
            {
                "config": str(config_path.resolve()),
                "vehicles": int(env_config["vehicles_count"]),
                "profile_probabilities": env_config["mtm"]["profile_probabilities"],
                "safe_spawn": True,
                "dynamics_guard": True,
            },
            indent=2,
        ),
        flush=True,
    )

    action = np.zeros(2, dtype=np.float32)
    total_steps = 0
    episode = 0
    try:
        obs, info = env.reset(seed=int(args.seed))
        print(f"[current-mtm-live] realized profiles: {profile_counts(info)}", flush=True)
        while args.steps == 0 or total_steps < args.steps:
            obs, _, terminated, truncated, info = env.step(action)
            total_steps += 1
            if terminated or truncated:
                episode += 1
                obs, info = env.reset(seed=int(args.seed) + episode)
                print(
                    f"[current-mtm-live] reset episode {episode}; "
                    f"realized profiles: {profile_counts(info)}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("[current-mtm-live] stopped by user.", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
