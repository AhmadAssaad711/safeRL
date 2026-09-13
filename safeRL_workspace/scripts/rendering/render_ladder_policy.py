"""Render a ladder policy episode through the evaluation environment.

This renderer drives the *same* environment stack the post-training evaluation
uses -- ``run_ppo_cbf_progression.make_evaluation_env`` with the run's saved
env, reward, and CBF configuration -- so a rendered episode reproduces the
measured one step for step on the same seed.  ``--mode raw`` renders with the
external CBF disabled and ``--mode cbf`` with the filter in the loop.

The evaluation factory hardcodes ``render_mode=None`` because training never
renders.  Rather than rebuild the wrapper chain here, which would risk drifting
from the evaluated one, this script installs a narrow shim around the
``gym.make`` call inside that factory for the duration of the build.
"""

from __future__ import annotations

import os

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import argparse
import contextlib
import copy
import json
import time
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

import scripts.training.run_ppo_cbf_progression as progression
from scripts.evaluation.evaluate_ppo_cbf_overrides import (
    apply_env_overrides,
    build_namespace,
    load_source_run,
)

DEFAULT_SEED = 1_100_000
DEFAULT_FPS = 20
OVERLAY_HEIGHT = 96


@contextlib.contextmanager
def _rgb_array_rendering() -> Iterator[None]:
    """Force ``lane-free-v0`` to build with ``render_mode='rgb_array'``."""

    original_make = progression.gym.make

    def patched_make(env_id: str, *args: Any, **kwargs: Any):
        if env_id == "lane-free-v0":
            kwargs["render_mode"] = "rgb_array"
        return original_make(env_id, *args, **kwargs)

    progression.gym.make = patched_make
    try:
        yield
    finally:
        progression.gym.make = original_make


def _overlay(frame_rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    frame = cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    frame = cv2.copyMakeBorder(
        frame, OVERLAY_HEIGHT, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20)
    )
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (12, 26 + 24 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Ladder seed directory with run_config.json and model_final.zip.")
    parser.add_argument("--mode", choices=tuple(progression.EVALUATION_MODES), default="raw",
                        help="raw = external CBF OFF (default); cbf = filter in the loop.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Evaluation scenario seed; the 200-episode evaluation starts at 1100000.")
    parser.add_argument("--episodes", type=int, default=1, help="Consecutive seeds to render, one file each.")
    parser.add_argument("--steps", type=int, default=None, help="Cap the rendered policy steps.")
    parser.add_argument("--vehicles", type=int, default=None, help="Traffic density override.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    project_root = progression.protocol.find_project_root(args.project_root or Path.cwd()).resolve()
    run_dir = (args.run_dir if args.run_dir.is_absolute() else project_root / args.run_dir).resolve()
    run_config, study_config = load_source_run(run_dir)
    output_dir = (
        (args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir)
        if args.output_dir is not None
        else run_dir / "renders"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env_config, max_policy_steps, changes = apply_env_overrides(
        run_config["env_config"],
        max_policy_steps=int(study_config["evaluation_task_max_policy_steps"]),
        vehicles=args.vehicles,
    )
    env_config = copy.deepcopy(env_config)
    env_config["offscreen_rendering"] = True
    env_config["real_time_rendering"] = False
    step_cap = int(args.steps) if args.steps is not None else int(max_policy_steps)

    namespace = build_namespace(project_root, run_config)
    model_path = run_dir / "model_final.zip"
    model = progression.load_model(run_config["variant"], model_path, device=str(args.device))
    label = "CBF OFF (raw)" if args.mode == "raw" else "CBF ON"
    print(f"[render] {run_config['variant']} mode={args.mode} changes={changes} steps<={step_cap}", flush=True)

    results: list[dict[str, Any]] = []
    for index in range(int(args.episodes)):
        seed = int(args.seed) + index
        started = time.perf_counter()
        with _rgb_array_rendering():
            env = progression.make_evaluation_env(
                namespace,
                mode=str(args.mode),
                env_config=env_config,
                reward_config=copy.deepcopy(run_config["reward_config"]),
                correction_epsilon=float(run_config["training_signature"]["correction_epsilon"]),
                task_distance_m=float(study_config["evaluation_task_distance_m"]),
                task_max_policy_steps=int(max_policy_steps),
            )
        video_path = output_dir / f"{run_config['variant']}_{args.mode}_seed{seed}.mp4"
        writer: cv2.VideoWriter | None = None
        collisions = 0
        distance_m = 0.0
        step = 0
        try:
            observation, _ = env.reset(seed=seed)
            for step in range(step_cap):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                info = dict(info)
                collisions += max(int(info.get("ego_collision_events", 0)), 0)
                distance_m += float(
                    info.get("task_distance_step_m", info.get("pipeline_distance_step_m", 0.0))
                )
                margin = float(info.get("cbf_hocbf_raw_min_margin", float("nan")))
                frame = _overlay(
                    np.asarray(env.render(), dtype=np.uint8),
                    [
                        f"{run_config['variant']}  |  {label}  |  seed {seed}",
                        (
                            f"step {step:4d}   v {float(info.get('karalakou_ego_speed', float('nan'))):5.1f}"
                            f" -> {float(info.get('karalakou_target_speed', float('nan'))):4.1f} m/s"
                            f"   y {float(info.get('karalakou_ego_y', float('nan'))):4.1f} m"
                            f"   dist {distance_m:6.1f} m"
                        ),
                        (
                            f"cx {float(info.get('karalakou_cx', float('nan'))):4.2f}"
                            f"  cy {float(info.get('karalakou_cy', float('nan'))):4.2f}"
                            f"  cf {float(info.get('karalakou_cf', float('nan'))):4.2f}"
                            f"  r {float(reward):6.2f}"
                            f"  HOCBF margin {margin:6.2f}"
                            f"  collisions {collisions}"
                        ),
                    ],
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
                        raise RuntimeError(f"Could not open a video writer at {video_path}")
                writer.write(frame)
                if bool(terminated) or bool(truncated):
                    break
        finally:
            if writer is not None:
                writer.release()
            env.close()

        record = {
            "seed": seed,
            "mode": str(args.mode),
            "steps": int(step) + 1,
            "distance_m": round(distance_m, 1),
            "collisions": int(collisions),
            "video": str(video_path),
            "render_seconds": round(time.perf_counter() - started, 1),
        }
        results.append(record)
        print(f"  {record}", flush=True)

    summary_path = output_dir / f"render_{args.mode}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "variant": run_config["variant"],
                "model_sha256": progression.protocol.file_sha256(model_path),
                "mode": str(args.mode),
                "env_config_changes": changes,
                "episodes": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[render] wrote {len(results)} video(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
