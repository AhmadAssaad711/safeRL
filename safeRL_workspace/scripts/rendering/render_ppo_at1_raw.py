"""Render the final target-y + previous-action PPO policy with CBF disabled.

This renderer rebuilds the saved MTM environment and reward/observation wrappers,
then feeds the PPO actor's normalized command directly to the simulator.  No
CBF wrapper or projection is installed.  The traffic dynamics guard remains
whatever was recorded in the saved run configuration (it is a traffic-model
setting, not an ego CBF shield).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO

import scripts.training.run_cbf_filter_ablation as pipeline
from scripts.common.ppo_observation_variants import install_previous_action_observation


DEFAULT_SOURCE_DIR = Path("artifacts/ppo_y_desired_at1_50k_cuda8_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--seed",
        type=int,
        default=900001,
        help="Held-out evaluation seed; 900001 has a representative raw collision.",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _resolve(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _source_paths(project_root: Path, source_dir: Path) -> tuple[Path, dict[str, Any], Path]:
    source = _resolve(project_root, source_dir)
    config_path = source / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run configuration: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("observation_variant") != "target_y_plus_previous_action":
        raise RuntimeError(
            "This renderer is for the target-y + a[t-1] policy; "
            f"found observation_variant={config.get('observation_variant')!r}"
        )
    training_seed = int(config["training_seeds"][0])
    variant = str(config["selected_configs"][0])
    target = int(config["target_timesteps"])
    model_path = source / f"seed_{training_seed}" / variant / "model_checkpoints" / f"{target:09d}.zip"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing final PPO checkpoint: {model_path}")
    return source, config, model_path


def _build_runtime(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    namespace = pipeline.bootstrap_notebook_namespace(project_root)
    pipeline.exec_required_notebook_cells(
        project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    install_previous_action_observation(namespace)
    return namespace


def _make_env(
    namespace: dict[str, Any],
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
) -> gym.Env:
    render_config = copy.deepcopy(env_config)
    # Display-only overrides; physics, MTM personality, and guard settings are
    # retained from the saved run configuration.
    render_config["real_time_rendering"] = False
    render_config["offscreen_rendering"] = True
    env = gym.make("lane-free-v0", render_mode="rgb_array", config=render_config)
    env = namespace["KaralakouRewardWrapper"](
        env, reward_config=copy.deepcopy(reward_config)
    )
    if namespace.get("NORMALIZE_RL_OBSERVATIONS", False):
        env = namespace["LaneFreeObservationNormalizationWrapper"](
            env, clip=namespace["OBSERVATION_CLIP"]
        )
    if "KPIInfoWrapper" in namespace:
        env = namespace["KPIInfoWrapper"](env)
    return pipeline.ProtocolMetricsWrapper(env)


def _as_number(info: dict[str, Any], *keys: str, default: float = float("nan")) -> float:
    for key in keys:
        try:
            value = float(info.get(key, default))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return float(default)


def _annotate(
    frame_rgb: np.ndarray,
    *,
    step: int,
    seed: int,
    policy_dt: float,
    action_normalized: np.ndarray,
    action_physical: np.ndarray,
    info: dict[str, Any],
    collisions: int,
    guard_enabled: bool,
    cbf_enabled: bool = False,
) -> np.ndarray:
    frame = cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    border = 66
    frame = cv2.copyMakeBorder(
        frame, border, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20)
    )
    ego_speed = _as_number(
        info, "ego_speed", "karalakou_ego_speed", "speed", default=float("nan")
    )
    ego_y = _as_number(info, "ego_y", "karalakou_ego_y", default=float("nan"))
    target_y = _as_number(info, "target_y", "karalakou_target_y", default=float("nan"))
    desired = _as_number(
        info,
        "desired_speed",
        "karalakou_desired_speed",
        "ego_desired_speed",
        default=float("nan"),
    )
    label_1 = (
        f"PPO target-y + a[t-1] | CBF {'ON' if cbf_enabled else 'OFF'} | "
        f"traffic guard {'ON' if guard_enabled else 'OFF'} | seed {seed}"
    )
    label_2 = (
        f"t={step * policy_dt:5.1f}s  v={ego_speed:5.2f}  v_des={desired:5.2f}  "
        f"y={ego_y:5.2f}  y_target={target_y:5.2f}  collisions={collisions}"
    )
    label_3 = (
        f"a_norm=[{float(action_normalized[0]):+.2f}, {float(action_normalized[1]):+.2f}]  "
        f"a_phys=[{float(action_physical[0]):+.2f}, {float(action_physical[1]):+.2f}]"
    )
    for y, text in ((20, label_1), (42, label_2), (62, label_3)):
        cv2.putText(
            frame,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return frame


def main() -> int:
    args = parse_args()
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if str(args.device).lower().startswith("cuda") and not th.cuda.is_available():
        raise RuntimeError("CUDA requested for PPO inference, but CUDA is unavailable")

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    project_root = pipeline.find_project_root(
        args.project_root or Path(__file__).resolve().parents[2]
    ).resolve()
    source_dir, run_config, model_path = _source_paths(project_root, args.source_dir)
    scenario_seed = int(args.seed)
    allowed_seeds = {int(value) for value in run_config.get("eval_seeds", [])}
    if allowed_seeds and scenario_seed not in allowed_seeds:
        raise ValueError(f"seed {scenario_seed} is not in saved eval_seeds={sorted(allowed_seeds)}")
    env_config = copy.deepcopy(run_config["env_config"])
    reward_config = copy.deepcopy(run_config["reward_config"])
    steps = int(args.steps if args.steps is not None else run_config["eval_timesteps"])
    if steps > int(env_config.get("episode_steps", steps)):
        raise ValueError("Requested steps exceed the saved episode horizon")

    namespace = _build_runtime(project_root, run_config)
    model = PPO.load(str(model_path), device=str(args.device))
    expected_dim = int(np.prod(model.observation_space.shape))
    if expected_dim != int(run_config.get("observation_dimensions", 32)):
        raise RuntimeError(
            f"Checkpoint observation dimension {expected_dim} does not match saved run config"
        )
    env = _make_env(namespace, env_config, reward_config)
    output_dir = (
        _resolve(project_root, args.output_dir)
        if args.output_dir is not None
        else (source_dir / "renders").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"ppo_at1_raw_cbf_off_seed{scenario_seed}_{steps}steps_{stamp}"
    video_path = output_dir / f"{stem}.mp4"
    preview_path = output_dir / f"{stem}_preview.png"
    summary_path = output_dir / f"{stem}_summary.json"

    writer: cv2.VideoWriter | None = None
    collisions = 0
    active_collision_steps = 0
    rendered_steps = 0
    terminated = False
    truncated = False
    last_info: dict[str, Any] = {}
    started = time.perf_counter()
    policy_dt = float(env_config.get("dt", 0.05)) * max(
        1,
        int(round(float(env_config.get("simulation_frequency", 20)) / float(env_config.get("policy_frequency", 10)))),
    )
    guard_enabled = bool(env_config.get("traffic_safety", {}).get("dynamics_guard", False))

    try:
        obs, _ = env.reset(seed=scenario_seed)
        for step_index in range(1, steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            action_normalized = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
            action_physical = pipeline.normalized_to_physical(action_normalized, env_config)
            obs, _, terminated, truncated, info = env.step(action_normalized)
            last_info = dict(info)
            event_count = max(
                int(info.get("pipeline_distinct_ego_collision_events", info.get("ego_collision_events", 0))),
                0,
            )
            collisions += event_count
            active_collision_steps += int(bool(info.get("ego_collision", False)))
            frame = _annotate(
                np.asarray(env.render(), dtype=np.uint8),
                step=step_index,
                seed=scenario_seed,
                policy_dt=policy_dt,
                action_normalized=action_normalized,
                action_physical=action_physical,
                info=last_info,
                collisions=collisions,
                guard_enabled=guard_enabled,
            )
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(args.fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open MP4 writer: {video_path}")
                if not cv2.imwrite(str(preview_path), frame):
                    raise RuntimeError(f"Could not write preview: {preview_path}")
            writer.write(frame)
            rendered_steps += 1
            if terminated or truncated:
                break
    finally:
        if writer is not None:
            writer.release()
        env.close()
        del model
        if th.cuda.is_available():
            th.cuda.empty_cache()

    if rendered_steps == 0 or not video_path.is_file():
        raise RuntimeError("No video frames were produced")
    summary = {
        "video_path": str(video_path),
        "preview_path": str(preview_path),
        "summary_path": str(summary_path),
        "source_dir": str(source_dir),
        "model_path": str(model_path),
        "seed": scenario_seed,
        "observation_variant": run_config["observation_variant"],
        "observation_dimension": expected_dim,
        "model_timestep": int(run_config["target_timesteps"]),
        "cbf_enabled": False,
        "traffic_dynamics_guard": guard_enabled,
        "steps_requested": steps,
        "steps_rendered": rendered_steps,
        "simulated_seconds": rendered_steps * policy_dt,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "distinct_ego_collision_events": collisions,
        "active_collision_timesteps": active_collision_steps,
        "final_info": {
            str(key): value
            for key, value in last_info.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
        "render_elapsed_seconds": time.perf_counter() - started,
    }
    # OneDrive can briefly rehydrate a newly-created directory while the MP4
    # handle is being released.  Retry the metadata write so a completed video
    # is never reported as failed just because its optional sidecar lagged.
    summary_json = json.dumps(summary, indent=2, default=str)
    summary_written = False
    for _attempt in range(12):
        try:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(summary_json, encoding="utf-8")
            summary_written = True
            break
        except FileNotFoundError:
            time.sleep(0.25)
    if not summary_written:
        print(f"Warning: could not write optional summary sidecar: {summary_path}", file=sys.stderr)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
