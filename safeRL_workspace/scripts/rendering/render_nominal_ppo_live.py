"""Live-render a saved nominal PPO policy for a bounded number of episodes.

This renderer is intentionally for the raw pure-RL policy used by the nominal
pilot.  It rebuilds the saved MTM environment and Karalakou observation/reward
assembly, loads the PPO checkpoint, and does not install the incompatible
CBF-context/physical-action wrapper used by older 500k CBF renderers.

The effective observation contract is taken from the checkpoint itself.  The
current social-only seed-307 checkpoint is a 30-D target-y policy, even though
some historical run-config flags retain the previous-action compatibility
setting.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import scripts.training.run_cbf_filter_ablation as pipeline
from scripts.common.ppo_observation_variants import install_previous_action_observation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1_100_000)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Write an RGB-array MP4 instead of opening a live human window.",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


def _load_namespace(project_root: Path, expected_dimension: int) -> dict[str, Any]:
    namespace = pipeline.bootstrap_notebook_namespace(project_root)
    pipeline.exec_required_notebook_cells(
        project_root / "notebooks" / "lanelessKaralakou.ipynb",
        namespace,
    )
    if expected_dimension == 32:
        install_previous_action_observation(namespace)
    elif expected_dimension != 30:
        raise RuntimeError(
            "This live renderer supports the 30-D target-y or 32-D "
            f"target-y+previous-action policy, got {expected_dimension}D."
        )
    return namespace


def _make_env(
    namespace: dict[str, Any],
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    *,
    video: bool,
) -> gym.Env:
    render_config = copy.deepcopy(env_config)
    render_config["real_time_rendering"] = not video
    render_config["offscreen_rendering"] = video
    env: gym.Env = gym.make(
        "lane-free-v0",
        render_mode="rgb_array" if video else "human",
        config=render_config,
    )
    env = namespace["KaralakouRewardWrapper"](
        env,
        reward_config=copy.deepcopy(reward_config),
    )
    if namespace.get("NORMALIZE_RL_OBSERVATIONS", False):
        env = namespace["LaneFreeObservationNormalizationWrapper"](
            env,
            clip=namespace["OBSERVATION_CLIP"],
        )
    if "KPIInfoWrapper" in namespace:
        env = namespace["KPIInfoWrapper"](env)
    return env


def _collision_count(info: dict[str, Any]) -> int:
    return max(
        int(
            info.get(
                "pipeline_distinct_ego_collision_events",
                info.get("ego_collision_events", 0),
            )
        ),
        0,
    )


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    project_root = args.project_root.resolve()
    model_path = args.model_path.resolve()
    run_config_path = args.run_config.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not run_config_path.is_file():
        raise FileNotFoundError(run_config_path)

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    model = PPO.load(str(model_path), device=args.device)
    expected_dimension = int(np.prod(model.observation_space.shape))
    namespace = _load_namespace(project_root, expected_dimension)
    env_config = copy.deepcopy(run_config["env_config"])
    reward_config = copy.deepcopy(run_config["reward_config"])
    max_steps = int(args.steps or env_config.get("episode_steps", 160))
    video_path = args.video_path.resolve() if args.video_path is not None else None
    if video_path is not None:
        if video_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing video: {video_path}")
        video_path.parent.mkdir(parents=True, exist_ok=True)
    env = _make_env(
        namespace,
        env_config,
        reward_config,
        video=video_path is not None,
    )
    actual_dimension = int(np.prod(env.observation_space.shape))
    if actual_dimension != expected_dimension:
        env.close()
        raise RuntimeError(
            f"Policy expects {expected_dimension} observations but the rendered "
            f"environment provides {actual_dimension}."
        )

    print(f"[render-live] model={model_path}", flush=True)
    print(
        f"[render-live] episodes={args.episodes} steps={max_steps} "
        f"seed_start={args.seed} CBF=OFF traffic_guard="
        f"{bool(env_config.get('traffic_safety', {}).get('dynamics_guard', False))}",
        flush=True,
    )
    print(f"[render-live] observation_space={env.observation_space}", flush=True)
    if video_path is not None:
        print(f"[render-live] video={video_path}", flush=True)

    writer: cv2.VideoWriter | None = None

    def render_frame() -> None:
        nonlocal writer
        frame = env.render()
        if video_path is None:
            return
        if frame is None:
            raise RuntimeError("RGB-array rendering returned no frame")
        frame_array = np.asarray(frame, dtype=np.uint8)
        if frame_array.ndim != 3 or frame_array.shape[2] != 3:
            raise RuntimeError(f"Unexpected render frame shape: {frame_array.shape}")
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        height, width = frame_bgr.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(args.fps),
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open MP4 writer: {video_path}")
        writer.write(frame_bgr)

    try:
        for episode_index in range(args.episodes):
            episode_seed = int(args.seed) + episode_index
            observation, _ = env.reset(seed=episode_seed)
            render_frame()
            total_reward = 0.0
            collisions = 0
            rendered_steps = 0
            for _ in range(max_steps):
                action, _ = model.predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                observation, reward, terminated, truncated, info = env.step(action)
                render_frame()
                total_reward += float(reward)
                collisions += _collision_count(info)
                rendered_steps += 1
                if terminated or truncated:
                    break
            print(
                f"[render-live] episode {episode_index + 1:02d}/{args.episodes}: "
                f"seed={episode_seed} steps={rendered_steps} "
                f"return={total_reward:.3f} collisions={collisions}",
                flush=True,
            )
    finally:
        env.close()
        if writer is not None:
            writer.release()
    if video_path is not None:
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise RuntimeError(f"Video was not written: {video_path}")
        print(f"[render-live] wrote {video_path}", flush=True)
    print("[render-live] finished", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    raise SystemExit(main())
