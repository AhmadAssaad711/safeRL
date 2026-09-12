"""Render episodes in the highway-env window with the reward's lateral target as a red line.

The episode runs through the *same* stack the post-training evaluation uses --
``run_ppo_cbf_progression.make_evaluation_env`` with the run's saved env,
reward, and CBF configuration -- so an episode rendered without reward-variant
flags reproduces the measured one step for step on the same seed.

The red line is drawn into the road surface itself, by wrapping the
``RoadGraphics.display`` hook the lane-free environment already installs, so it
sits under the vehicles and spans the whole road: it is the lateral position
the reward is paying the ego to reach at that instant. A red tick marks the
ego's own longitudinal position on that line.

Reward-variant flags change which target is drawn, which is the point of this
script: the same policy can be shown against the notebook target
(``--lateral-blockers all``, the default) and against a candidate one
(``--lateral-blockers closing``). Note that the target also occupies an
observation slot, so a variant changes what the policy reads and the episode
diverges from the evaluated one -- that is expected, and the seeds and traffic
are still identical.

``--display`` opens the live pygame window (the default). ``--save-frames N``
writes every Nth frame as a PNG next to the run so a still can be inspected
afterwards, and ``--no-display`` renders offscreen for exactly that purpose.
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

import numpy as np
import pygame
from highway_env.road.graphics import RoadGraphics

import scripts.training.run_ppo_cbf_progression as progression
from saferl.rewards import (
    add_reward_variant_arguments,
    apply_reward_variant_arguments,
    reward_variant_summary,
)
from scripts.evaluation.evaluate_ppo_cbf_overrides import (
    apply_env_overrides,
    build_namespace,
    load_source_run,
)

DEFAULT_SEED = 1_100_000
TARGET_COLOR = (225, 30, 30)
TARGET_WIDTH_M = 0.35


@contextlib.contextmanager
def _render_mode(mode: str) -> Iterator[None]:
    """Force ``lane-free-v0`` to build with a given render mode."""

    original_make = progression.gym.make

    def patched_make(env_id: str, *args: Any, **kwargs: Any):
        if env_id == "lane-free-v0":
            kwargs["render_mode"] = mode
        return original_make(env_id, *args, **kwargs)

    progression.gym.make = patched_make
    try:
        yield
    finally:
        progression.gym.make = original_make


@contextlib.contextmanager
def _target_line(state: dict[str, float]) -> Iterator[None]:
    """Draw the current lateral target as a red line on the road surface.

    The lane-free environment installs its own ``RoadGraphics.display``; this
    wraps whatever is installed, so the road, its edges, and the vehicles are
    drawn exactly as they normally are.
    """

    installed = RoadGraphics.display

    def display_with_target(road, surface) -> None:
        installed(road, surface)
        target_y = state.get("target_y")
        if target_y is None:
            return
        length = float(getattr(road, "lane_free_length", 1000.0))
        # The road is periodic, so the target y holds at every x. Span well
        # past the window in both directions: a line drawn only over
        # [0, length] stops at the ring seam, leaving the wrapped part of the
        # view without a line.
        centre = float(state.get("ego_x", 0.5 * length))
        start = surface.pos2pix(centre - 2.0 * length, float(target_y))
        end = surface.pos2pix(centre + 2.0 * length, float(target_y))
        thickness = max(2, int(surface.pix(TARGET_WIDTH_M)))
        pygame.draw.line(surface, TARGET_COLOR, start, end, thickness)
        ego_x = state.get("ego_x")
        if ego_x is not None:
            tick_top = surface.pos2pix(float(ego_x), float(target_y) - 0.9)
            tick_bottom = surface.pos2pix(float(ego_x), float(target_y) + 0.9)
            pygame.draw.line(surface, TARGET_COLOR, tick_top, tick_bottom, thickness)

    RoadGraphics.display = staticmethod(display_with_target)
    try:
        yield
    finally:
        RoadGraphics.display = installed


def _frame(env: Any) -> np.ndarray | None:
    viewer = getattr(env.unwrapped, "viewer", None)
    if viewer is None or not hasattr(viewer, "get_image"):
        return None
    return np.asarray(viewer.get_image(), dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Ladder seed directory with run_config.json and model_final.zip.",
    )
    parser.add_argument("--mode", choices=tuple(progression.EVALUATION_MODES), default="raw")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--episodes", type=int, default=1, help="Consecutive seeds to render.")
    parser.add_argument("--steps", type=int, default=None, help="Cap the rendered policy steps.")
    parser.add_argument("--vehicles", type=int, default=None)
    parser.add_argument("--fps", type=float, default=20.0, help="Playback rate; 0 renders unpaced.")
    parser.add_argument("--display", dest="display", action="store_true", default=True)
    parser.add_argument(
        "--no-display",
        dest="display",
        action="store_false",
        help="Render offscreen, for --save-frames without a window.",
    )
    parser.add_argument(
        "--save-frames",
        type=int,
        default=0,
        help="Write every Nth frame as a PNG (0 writes none).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--label", default=None, help="Name for this configuration in the output.")
    parser.add_argument("--device", default="cpu")
    add_reward_variant_arguments(parser)
    arguments = parser.parse_args()

    project_root = progression.protocol.find_project_root(
        arguments.project_root or Path.cwd()
    ).resolve()
    run_dir = (
        arguments.run_dir if arguments.run_dir.is_absolute() else project_root / arguments.run_dir
    ).resolve()
    run_config, study_config = load_source_run(run_dir)
    label = arguments.label or run_config["variant"]
    output_dir = (
        (
            arguments.output_dir
            if arguments.output_dir.is_absolute()
            else project_root / arguments.output_dir
        )
        if arguments.output_dir is not None
        else run_dir / "renders" / f"target_line_{label}"
    )
    if arguments.save_frames:
        output_dir.mkdir(parents=True, exist_ok=True)

    env_config, max_policy_steps, changes = apply_env_overrides(
        run_config["env_config"],
        max_policy_steps=int(study_config["evaluation_task_max_policy_steps"]),
        vehicles=arguments.vehicles,
    )
    env_config = copy.deepcopy(env_config)
    env_config["offscreen_rendering"] = not arguments.display
    env_config["real_time_rendering"] = False
    step_cap = int(arguments.steps) if arguments.steps is not None else int(max_policy_steps)

    reward_config = copy.deepcopy(run_config["reward_config"])
    apply_reward_variant_arguments(arguments, reward_config)
    variants = reward_variant_summary(reward_config)
    namespace = build_namespace(project_root, run_config)
    model_path = run_dir / "model_final.zip"
    model = progression.load_model(run_config["variant"], model_path, device=str(arguments.device))
    delay = 0.0 if float(arguments.fps) <= 0.0 else 1.0 / float(arguments.fps)

    print(
        f"[target-line] {label}: {run_config['variant']} mode={arguments.mode} "
        f"vehicles={env_config['vehicles_count']} target={variants['lateral_target']} "
        f"blockers={variants['lateral_blockers']} "
        f"fallback={variants['lateral_target_fallback']} wf={variants['potential_field_weight']}",
        flush=True,
    )

    state: dict[str, float] = {}
    episodes: list[dict[str, Any]] = []
    for index in range(int(arguments.episodes)):
        seed = int(arguments.seed) + index
        state.clear()
        with _render_mode("human" if arguments.display else "rgb_array"):
            env = progression.make_evaluation_env(
                namespace,
                mode=str(arguments.mode),
                env_config=env_config,
                reward_config=reward_config,
                correction_epsilon=float(run_config["training_signature"]["correction_epsilon"]),
                task_distance_m=float(study_config["evaluation_task_distance_m"]),
                task_max_policy_steps=int(max_policy_steps),
            )
        started = time.perf_counter()
        gap_steps = 0
        collisions = 0
        step = 0
        errors: list[float] = []
        try:
            with _target_line(state):
                observation, _ = env.reset(seed=seed)
                for step in range(step_cap):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, _reward, terminated, truncated, info = env.step(action)
                    info = dict(info)
                    # The environment renders inside step(), before this info
                    # exists, so that frame carries the previous target; render
                    # again with the fresh one so the visible frame matches.
                    state["target_y"] = float(info.get("karalakou_target_y", float("nan")))
                    state["ego_x"] = float(env.unwrapped.vehicle.position[0])
                    env.render()
                    gap_steps += int(float(info.get("karalakou_zone_found", 0.0)) > 0.5)
                    collisions += max(int(info.get("ego_collision_events", 0)), 0)
                    errors.append(float(info.get("karalakou_lat_y_error_m", float("nan"))))
                    if arguments.save_frames and step % int(arguments.save_frames) == 0:
                        frame = _frame(env)
                        if frame is not None:
                            path = output_dir / f"seed{seed}_step{step:04d}.png"
                            pygame.image.save(
                                pygame.surfarray.make_surface(frame.transpose(1, 0, 2)), str(path)
                            )
                    if delay:
                        time.sleep(delay)
                    if bool(terminated) or bool(truncated):
                        break
        finally:
            env.close()

        record = {
            "seed": seed,
            "steps": int(step) + 1,
            "gap_found_rate": round(gap_steps / max(int(step) + 1, 1), 3),
            "mean_lateral_error_m": round(float(np.nanmean(errors)) if errors else float("nan"), 2),
            "collisions": int(collisions),
            "render_seconds": round(time.perf_counter() - started, 1),
        }
        episodes.append(record)
        print(f"  {record}", flush=True)

    if arguments.save_frames:
        summary = {
            "run_dir": str(run_dir),
            "variant": run_config["variant"],
            "model_sha256": progression.protocol.file_sha256(model_path),
            "mode": str(arguments.mode),
            "env_config_changes": changes,
            "reward_variants": variants,
            "episodes": episodes,
        }
        (output_dir / "render_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"[target-line] frames and summary in {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
