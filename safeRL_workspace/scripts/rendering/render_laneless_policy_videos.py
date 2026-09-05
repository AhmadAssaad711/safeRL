from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from stable_baselines3 import DDPG

from scripts.common.laneless_script_config import active_traffic_model, env_config_from_args
from scripts.rendering.render_laneless_policy_scenario import (
    apply_cbf_overrides,
    apply_env_and_artifact_overrides,
    find_project_root,
    normalize_artifact_suffix,
)
from scripts.rendering.render_policy_scenarios import POLICIES, ScenarioSpec, apply_scenario, load_notebook_namespace, make_scenarios


VARIANT_TO_POLICY_KEY = {
    "ddpg": "baseline",
    "ddpg-cbf": "ddpg_cbf_reward",
    "guided-ddpg-cbf": "guided_ddpg_cbf",
}

VARIANT_LABELS = {
    "ddpg": "DDPG",
    "ddpg-cbf": "DDPG + CBF reward",
    "guided-ddpg-cbf": "DDPG + CBF reward + actor loss",
}

VARIANT_FILE_NAMES = {
    "ddpg": "ddpg",
    "ddpg-cbf": "cbf_reward",
    "guided-ddpg-cbf": "cbf_reward_loss",
}

SCENARIO_FILE_NAMES = {
    "open_road_no_neighbors": "open_road",
    "safe_overtake_open_upper_gap": "open_pass",
    "unsafe_overtake_fast_closing_upper": "fast_closing",
    "narrow_gap_wait_or_upper_escape": "tight_slot",
    "boundary_recovery_no_upper_squeeze": "upper_edge",
    "opposite_edge_recovery": "lower_edge",
    "boxed_in_hold_position": "boxed_in",
    "rear_pressure_escape": "rear_pressure",
    "sudden_lead_slowdown": "lead_slowdown",
    "staggered_gap_selection": "staggered_gap",
}


def set_stable_native_defaults() -> None:
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    ]:
        os.environ.setdefault(key, "1")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")


def policy_for_variant(variant: str):
    policy_key = VARIANT_TO_POLICY_KEY[variant]
    for policy in POLICIES:
        if policy.key == policy_key:
            return policy
    raise RuntimeError(f"Missing policy for {variant}")


def slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def variant_file_name(variant: str) -> str:
    return VARIANT_FILE_NAMES.get(variant, slug(variant))


def scenario_file_name(scenario: ScenarioSpec) -> str:
    return SCENARIO_FILE_NAMES.get(scenario.name, slug(scenario.name)[:24])


def video_env_config(base_config: dict[str, Any], *, steps: int, scenario: ScenarioSpec | None) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    if scenario is not None:
        config["vehicles_count"] = len(scenario.vehicles)
        config["neighbors_count"] = 5
    config.update(
        {
            "episode_steps": int(steps),
            "duration": int(steps),
            "terminate_on_collision": False,
            "show_trajectories": False,
            "screen_width": 1200,
            "screen_height": 320,
            "centering_position": [0.5, 0.5],
            "scaling": 9.8,
            "offscreen_rendering": True,
            "real_time_rendering": False,
        }
    )
    return config


def make_policy_env(namespace: dict[str, Any], policy, env_config: dict[str, Any], seed: int):
    if policy.env_kind == "baseline":
        return namespace["make_single_env"](seed=seed, render_mode="rgb_array", env_config=env_config)
    if policy.env_kind == "cbf":
        return namespace["make_cbf_single_env"](
            seed=seed,
            render_mode="rgb_array",
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
            eps_side=namespace["CBF_EPS_SIDE"],
            env_config=env_config,
        )
    if policy.env_kind == "guided":
        return namespace["make_guided_cbf_single_env"](
            seed=seed,
            render_mode="rgb_array",
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
            eps_side=namespace["CBF_EPS_SIDE"],
            env_config=env_config,
        )
    raise ValueError(f"Unknown policy env kind: {policy.env_kind}")


def load_models(namespace: dict[str, Any], variants: list[str]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for variant in variants:
        policy = policy_for_variant(variant)
        path = Path(namespace[policy.model_key])
        if not path.exists():
            raise FileNotFoundError(f"Missing model for {VARIANT_LABELS[variant]}: {path}")
        model_cls = namespace["GuidedCBFDDPG"] if variant == "guided-ddpg-cbf" else DDPG
        models[variant] = model_cls.load(str(path), device=namespace["DEVICE"])
        print(f"[video-export] loaded {variant}: {path}", flush=True)
    return models


def rendered_frame(env) -> np.ndarray:
    frame = env.render()
    if frame is None:
        raise RuntimeError("Environment render returned None; expected an rgb_array frame.")
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"Unexpected render frame shape: {frame.shape}")
    return frame


def annotate_frame(
    frame_rgb: np.ndarray,
    *,
    title: str,
    subtitle: str,
    step: int,
    steps: int,
    info: dict[str, Any],
    cumulative_ego_events: int = 0,
    cumulative_ego_steps: int = 0,
) -> np.ndarray:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    annotated = cv2.copyMakeBorder(frame_bgr, 64, 34, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
    cv2.putText(
        annotated,
        title,
        (18, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        subtitle,
        (18, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    speed = float(info.get("speed", np.nan))
    current_event = int(info.get("ego_collision_events", 0))
    active_ego_collision = bool(info.get("ego_collision", False))
    correction = float(info.get("cbf_correction_norm", 0.0))
    min_h = info.get("cbf_min_h", None)
    footer = (
        f"step {step:03d}/{steps:03d} | speed {speed:5.2f} m/s | "
        f"ego active {int(active_ego_collision)} | "
        f"ego events {cumulative_ego_events} | ego steps {cumulative_ego_steps}"
    )
    if current_event:
        footer += f" | new event +{current_event}"
    if min_h is not None:
        footer += f" | correction {correction:.3f} | min h {float(min_h):.3f}"
    cv2.putText(
        annotated,
        footer,
        (18, annotated.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return annotated


def open_writer(path: Path, frame_shape: tuple[int, int, int], fps: int):
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {path}")
    return writer


def write_repeated(writer, frame_bgr: np.ndarray, repeat: int) -> None:
    for _ in range(max(1, int(repeat))):
        writer.write(frame_bgr)


def run_clip(
    *,
    namespace: dict[str, Any],
    model: Any,
    variant: str,
    scenario: ScenarioSpec | None,
    output_path: Path,
    seed: int,
    sim_seconds: float,
    fps: int,
    repeat_frames: int,
) -> dict[str, Any]:
    policy = policy_for_variant(variant)
    policy_frequency = float(namespace["ENV_CONFIG"].get("policy_frequency", 4))
    steps = max(1, int(round(float(sim_seconds) * policy_frequency)))
    env_config = video_env_config(namespace["ENV_CONFIG"], steps=steps, scenario=scenario)
    env = make_policy_env(namespace, policy, env_config, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_reward = 0.0
    ego_collision_events = 0
    ego_collision_steps = 0
    active_ego_collision_max = 0
    correction_norms: list[float] = []
    min_h_values: list[float] = []
    info: dict[str, Any] = {}
    writer = None
    try:
        obs, info = env.reset(seed=seed)
        if scenario is not None:
            obs = apply_scenario(env, scenario)
            info = {}

        title = VARIANT_LABELS[variant]
        if scenario is None:
            title = f"{title} | normal {active_traffic_model(namespace['ENV_CONFIG']).upper()} traffic"
            subtitle = "Random highway rollout from the selected active environment"
        else:
            title = f"{title} | {scenario.title}"
            subtitle = scenario.expected

        first = annotate_frame(
            rendered_frame(env),
            title=title,
            subtitle=subtitle,
            step=0,
            steps=steps,
            info=info,
            cumulative_ego_events=ego_collision_events,
            cumulative_ego_steps=ego_collision_steps,
        )
        writer = open_writer(output_path, first.shape, fps)
        write_repeated(writer, first, repeat_frames)

        for step in range(1, steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32).reshape(-1)[:2])
            total_reward += float(reward)
            ego_collision_events += int(info.get("ego_collision_events", 0))
            if bool(info.get("ego_collision", False)):
                ego_collision_steps += 1
                active_ego_collision_max = 1
            if "cbf_correction_norm" in info:
                correction_norms.append(float(info.get("cbf_correction_norm", 0.0)))
            if "cbf_min_h" in info:
                min_h_values.append(float(info.get("cbf_min_h", np.nan)))
            frame = annotate_frame(
                rendered_frame(env),
                title=title,
                subtitle=subtitle,
                step=step,
                steps=steps,
                info=info,
                cumulative_ego_events=ego_collision_events,
                cumulative_ego_steps=ego_collision_steps,
            )
            write_repeated(writer, frame, repeat_frames)
            if terminated or truncated:
                break
    finally:
        if writer is not None:
            writer.release()
        env.close()

    min_h = float(np.nanmin(min_h_values)) if min_h_values and not np.all(np.isnan(min_h_values)) else np.nan
    mean_correction = float(np.mean(correction_norms)) if correction_norms else 0.0
    return {
        "file": str(output_path),
        "kind": "normal" if scenario is None else "scenario",
        "scenario": "" if scenario is None else scenario.name,
        "policy": variant,
        "steps": step if "step" in locals() else 0,
        "sim_seconds": float(sim_seconds),
        "fps": int(fps),
        "repeat_frames": int(repeat_frames),
        "return": float(total_reward),
        "ego_collision_events": int(ego_collision_events),
        "ego_collision_steps": int(ego_collision_steps),
        "active_ego_collision_max": int(active_ego_collision_max),
        "mean_correction_norm": mean_correction,
        "min_h": min_h,
    }


def selected_scenarios(all_scenarios: list[ScenarioSpec], selection: list[str] | None) -> list[ScenarioSpec]:
    if not selection:
        return all_scenarios
    by_name = {scenario.name: scenario for scenario in all_scenarios}
    missing = [name for name in selection if name not in by_name]
    if missing:
        raise ValueError(f"Unknown scenario(s): {missing}")
    return [by_name[name] for name in selection]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MP4 videos for laneless DDPG policies and diagnostics.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sim-seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--repeat-frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--variant", action="append", choices=sorted(VARIANT_TO_POLICY_KEY), default=None)
    parser.add_argument("--scenario", action="append", default=None, help="Scenario to render. Repeat to select.")
    parser.add_argument("--skip-normal", action="store_true")
    parser.add_argument("--skip-scenarios", action="store_true")
    parser.add_argument("--lambda-filter", type=float, default=None)
    parser.add_argument("--k0", type=float, default=None)
    parser.add_argument("--k1", type=float, default=None)
    parser.add_argument("--eps-side", type=float, default=None)
    parser.add_argument("--traffic-model", choices=["force", "mtm"], default=None)
    parser.add_argument("--env-config-json", default=None)
    parser.add_argument("--env-config-file", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    set_stable_native_defaults()
    args = parse_args()
    project_root = find_project_root(args.project_root)
    namespace = load_notebook_namespace(project_root)
    namespace["DEVICE"] = args.device
    apply_cbf_overrides(namespace, args)
    apply_env_and_artifact_overrides(namespace, args)

    variants = args.variant or ["ddpg", "ddpg-cbf", "guided-ddpg-cbf"]
    artifact_suffix = normalize_artifact_suffix(args.artifact_suffix) or active_traffic_model(namespace["ENV_CONFIG"])
    output_dir = args.output_dir or (
        project_root / "artifacts" / "lanelessKaralakou" / f"vids_{artifact_suffix}_20s"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    models = load_models(namespace, variants)
    scenarios = selected_scenarios(make_scenarios(float(namespace["ENV_CONFIG"]["road_width"])), args.scenario)

    rows: list[dict[str, Any]] = []
    if not args.skip_normal:
        normal_dir = output_dir / "normal"
        for variant_index, variant in enumerate(variants):
            output_path = normal_dir / f"{variant_file_name(variant)}.mp4"
            print(f"[video-export] normal highway | {variant} -> {output_path}", flush=True)
            rows.append(
                run_clip(
                    namespace=namespace,
                    model=models[variant],
                    variant=variant,
                    scenario=None,
                    output_path=output_path,
                    seed=int(args.seed) + 100_000 * variant_index,
                    sim_seconds=float(args.sim_seconds),
                    fps=int(args.fps),
                    repeat_frames=int(args.repeat_frames),
                )
            )

    if not args.skip_scenarios:
        scenario_dir = output_dir / "scenarios"
        for scenario_index, scenario in enumerate(scenarios):
            for variant_index, variant in enumerate(variants):
                output_path = scenario_dir / scenario_file_name(scenario) / f"{variant_file_name(variant)}.mp4"
                print(f"[video-export] {scenario.name} | {variant} -> {output_path}", flush=True)
                rows.append(
                    run_clip(
                        namespace=namespace,
                        model=models[variant],
                        variant=variant,
                        scenario=scenario,
                        output_path=output_path,
                        seed=int(args.seed) + 10_000 * scenario_index + 100_000 * variant_index,
                        sim_seconds=float(args.sim_seconds),
                        fps=int(args.fps),
                        repeat_frames=int(args.repeat_frames),
                    )
                )

    summary_path = output_dir / "video_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[video-export] wrote {len(rows)} videos to {output_dir}", flush=True)
    print(f"[video-export] summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
