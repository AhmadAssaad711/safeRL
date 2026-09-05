"""Render the final target-y + previous-action PPO policy with CBF enabled.

The actor command is converted to the physical acceleration units expected by
the fixed CBF shield.  The shield's safe physical action is then passed to the
same normalized simulator interface used by the saved CBF deployment eval.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO

import scripts.evaluation.evaluate_ppo_cbf_deployment as cbf_deployment
import scripts.rendering.render_ppo_at1_raw as raw_renderer
import scripts.training.run_cbf_filter_ablation as pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("artifacts/ppo_y_desired_at1_50k_cuda8_v2"),
    )
    parser.add_argument("--seed", type=int, default=900006)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _make_cbf_render_env(
    namespace: dict[str, Any],
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    snapshot: dict[str, Any],
) -> gym.Env:
    render_config = copy.deepcopy(env_config)
    render_config["real_time_rendering"] = False
    render_config["offscreen_rendering"] = True
    env = gym.make("lane-free-v0", render_mode="rgb_array", config=render_config)
    env = namespace["KaralakouRewardWrapper"](
        env, reward_config=copy.deepcopy(reward_config)
    )
    env = namespace["CorrectionRewardSafetyFilteredAccelerationWrapper"](
        env,
        lambda_delta=0.0,
        lambda_intervention=0.0,
        correction_epsilon=0.03,
        eps_side=float(snapshot["eps_side"]),
        k0=float(snapshot["k0"]),
        k1=float(snapshot["k1"]),
    )
    if namespace.get("NORMALIZE_RL_OBSERVATIONS", False):
        env = namespace["LaneFreeObservationNormalizationWrapper"](
            env, clip=namespace["OBSERVATION_CLIP"]
        )
    if "KPIInfoWrapper" in namespace:
        env = namespace["KPIInfoWrapper"](env, intervention_threshold=0.03)
    return pipeline.ProtocolMetricsWrapper(env)


def _write_summary(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, default=str)
    for _attempt in range(12):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")
            return
        except FileNotFoundError:
            time.sleep(0.25)
    print(f"Warning: could not write optional summary sidecar: {path}", flush=True)


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
    source_dir, run_config, model_path = raw_renderer._source_paths(
        project_root, args.source_dir
    )
    scenario_seed = int(args.seed)
    allowed_seeds = {int(value) for value in run_config.get("eval_seeds", [])}
    if allowed_seeds and scenario_seed not in allowed_seeds:
        raise ValueError(f"seed {scenario_seed} is not in saved eval_seeds={sorted(allowed_seeds)}")
    steps = int(args.steps if args.steps is not None else run_config["eval_timesteps"])
    env_config = copy.deepcopy(run_config["env_config"])
    reward_config = copy.deepcopy(run_config["reward_config"])
    if steps > int(env_config.get("episode_steps", steps)):
        raise ValueError("Requested steps exceed the saved episode horizon")
    snapshot = dict(run_config["fixed_cbf_snapshot"])

    namespace = cbf_deployment._build_runtime(
        project_root, run_config, device=str(args.device)
    )
    model = PPO.load(str(model_path), device=str(args.device))
    expected_dim = int(np.prod(model.observation_space.shape))
    if expected_dim != int(run_config.get("observation_dimensions", 32)):
        raise RuntimeError("Checkpoint observation dimension does not match run config")
    env = _make_cbf_render_env(namespace, env_config, reward_config, snapshot)

    output_dir = (
        raw_renderer._resolve(project_root, args.output_dir)
        if args.output_dir is not None
        else (source_dir / "renders").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"ppo_at1_cbf_on_seed{scenario_seed}_{steps}steps_{stamp}"
    video_path = output_dir / f"{stem}.mp4"
    preview_path = output_dir / f"{stem}_preview.png"
    summary_path = output_dir / f"{stem}_summary.json"

    writer: cv2.VideoWriter | None = None
    rendered_steps = 0
    raw_interventions = 0
    meaningful_interventions = 0
    qp_failures = 0
    correction_norms: list[float] = []
    active_collision_steps = 0
    collision_events = 0
    terminated = False
    truncated = False
    last_info: dict[str, Any] = {}
    started = time.perf_counter()
    policy_dt = float(env_config.get("dt", 0.05)) * max(
        1,
        int(
            round(
                float(env_config.get("simulation_frequency", 20))
                / float(env_config.get("policy_frequency", 10))
            )
        ),
    )
    guard_enabled = bool(env_config.get("traffic_safety", {}).get("dynamics_guard", False))

    try:
        obs, _ = env.reset(seed=scenario_seed)
        for step_index in range(1, steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            action_normalized = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
            raw_physical = pipeline.model_action_to_physical(
                model, action_normalized, env_config
            )
            obs, _, terminated, truncated, info = env.step(raw_physical)
            last_info = dict(info)
            safe_physical = np.asarray(
                [
                    info.get("cbf_a_safe_x", raw_physical[0]),
                    info.get("cbf_a_safe_y", raw_physical[1]),
                ],
                dtype=np.float32,
            )
            correction = float(info.get("cbf_correction_norm", 0.0))
            correction_norms.append(correction)
            raw_interventions += int(bool(info.get("cbf_intervened", False)))
            meaningful_interventions += int(bool(info.get("cbf_event_intervened", False)))
            qp_failures += int(bool(info.get("cbf_qp_failure", False)))
            active_collision_steps += int(bool(info.get("ego_collision", False)))
            collision_events += max(
                int(info.get("pipeline_distinct_ego_collision_events", 0)), 0
            )
            frame = raw_renderer._annotate(
                np.asarray(env.render(), dtype=np.uint8),
                step=step_index,
                seed=scenario_seed,
                policy_dt=policy_dt,
                action_normalized=action_normalized,
                action_physical=safe_physical,
                info=last_info,
                collisions=collision_events,
                guard_enabled=guard_enabled,
                cbf_enabled=True,
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
        "cbf_enabled": True,
        "cbf_snapshot": snapshot,
        "traffic_dynamics_guard": guard_enabled,
        "steps_requested": steps,
        "steps_rendered": rendered_steps,
        "simulated_seconds": rendered_steps * policy_dt,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "distinct_ego_collision_events": collision_events,
        "active_collision_timesteps": active_collision_steps,
        "cbf_raw_intervention_steps": raw_interventions,
        "cbf_meaningful_intervention_steps": meaningful_interventions,
        "cbf_qp_failure_steps": qp_failures,
        "cbf_mean_correction_norm": float(np.mean(correction_norms)) if correction_norms else 0.0,
        "cbf_max_correction_norm": float(np.max(correction_norms)) if correction_norms else 0.0,
        "final_info": {
            str(key): value
            for key, value in last_info.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
        "render_elapsed_seconds": time.perf_counter() - started,
    }
    _write_summary(summary_path, summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
