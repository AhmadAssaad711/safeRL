"""Live-render the saved 16 m/s PPO policy with optional runtime CBF filtering.

This is intentionally a display-only renderer: it opens the simulator's
human pygame window and never creates a video, image, or metrics artifact.
The defaults point at the current 50k Q1 checkpoint, so repeated renders can
be launched with simply::

    python -m scripts.rendering.render_ppo_ego16_live
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO

import scripts.training.run_cbf_filter_ablation as pipeline
from scripts.common.ppo_observation_variants import install_previous_action_observation


DEFAULT_RUN_CONFIG = Path(
    "artifacts/ppo_ego16_abs_target_50k_cuda8/seed_307/Q1_stable/run_config.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=900001)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument(
        "--cbf",
        action="store_true",
        help="enable the saved CBF shield (off by default)",
    )
    return parser.parse_args()


def resolve(project_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def load_run_config(project_root: Path, path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    config_path = resolve(project_root, path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    training = payload.get("training_config", payload)
    if not isinstance(training, dict):
        raise TypeError("run configuration has no training_config mapping")
    return config_path, payload, training


def build_env(project_root: Path, training: dict[str, Any], *, cbf: bool = False) -> gym.Env:
    namespace = pipeline.bootstrap_notebook_namespace(project_root)
    pipeline.exec_required_notebook_cells(
        project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    # The checkpoint expects the 30-D target-y state plus a[t-1].
    install_previous_action_observation(namespace)
    snapshot = dict(training.get("fixed_cbf_snapshot", {}))
    if cbf:
        pipeline.install_minimal_guided_cbf(namespace)
        pipeline.install_correction_reward_env(namespace)
    env_config = copy.deepcopy(training["env_config"])
    env_config["real_time_rendering"] = True
    env_config["offscreen_rendering"] = False
    env_config["terminate_on_collision"] = True
    env = gym.make("lane-free-v0", render_mode="human", config=env_config)
    env = namespace["KaralakouRewardWrapper"](
        env, reward_config=copy.deepcopy(training["reward_config"])
    )
    if cbf:
        env = namespace["CorrectionRewardSafetyFilteredAccelerationWrapper"](
            env,
            lambda_delta=0.0,
            lambda_intervention=0.0,
            correction_epsilon=0.03,
            eps_side=float(snapshot.get("eps_side", 0.1)),
            k0=float(snapshot.get("k0", 5.29)),
            k1=float(snapshot.get("k1", 3.68)),
        )
    if namespace.get("NORMALIZE_RL_OBSERVATIONS", False):
        env = namespace["LaneFreeObservationNormalizationWrapper"](
            env, clip=namespace["OBSERVATION_CLIP"]
        )
    if "KPIInfoWrapper" in namespace:
        if cbf:
            env = namespace["KPIInfoWrapper"](env, intervention_threshold=0.03)
        else:
            env = namespace["KPIInfoWrapper"](env)
    return env


def main() -> int:
    args = parse_args()
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.episodes <= 0 or args.print_every <= 0:
        raise ValueError("--episodes and --print-every must be positive")

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    project_root = pipeline.find_project_root(
        args.project_root or Path(__file__).resolve().parents[2]
    ).resolve()
    config_path, payload, training = load_run_config(project_root, args.run_config)
    model_path = resolve(project_root, args.model_path) if args.model_path else config_path.parent / "model_final.zip"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing PPO checkpoint: {model_path}")

    device = str(args.device)
    if device.lower().startswith("cuda") and not th.cuda.is_available():
        print("[render] CUDA unavailable; using CPU inference", flush=True)
        device = "cpu"
    model = PPO.load(str(model_path), device=device)
    expected_shape = tuple(model.observation_space.shape or ())
    if expected_shape != (32,):
        raise RuntimeError(f"Expected the 32-D target-y+a[t-1] observation, got {expected_shape}")

    env = build_env(project_root, training, cbf=bool(args.cbf))
    requested_steps = int(args.steps if args.steps is not None else training.get("evaluation", {}).get("timestep_budget", 800))
    max_steps = int(training["env_config"].get("episode_steps", requested_steps))
    if requested_steps > max_steps:
        raise ValueError(f"--steps={requested_steps} exceeds saved episode horizon {max_steps}")

    policy_dt = float(training["env_config"].get("dt", 0.05)) * max(
        1,
        int(
            round(
                float(training["env_config"].get("simulation_frequency", 20))
                / float(training["env_config"].get("policy_frequency", 10))
            )
        ),
    )
    print(
        f"[render] CBF {'ON' if args.cbf else 'OFF'} | live pygame window | no video recording",
        flush=True,
    )
    print(f"[render] model={model_path}", flush=True)
    print(f"[render] v_desired={training['reward_config'].get('ego_desired_speed')} m/s | obs={expected_shape}", flush=True)

    started = time.perf_counter()
    try:
        for episode in range(int(args.episodes)):
            seed = int(args.seed) + episode
            obs, _ = env.reset(seed=seed)
            collision_events = 0
            for step in range(1, requested_steps + 1):
                action, _ = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
                if args.cbf:
                    raw_physical = pipeline.model_action_to_physical(
                        model, action, training["env_config"]
                    )
                    obs, _, terminated, truncated, info = env.step(raw_physical)
                else:
                    raw_physical = None
                    obs, _, terminated, truncated, info = env.step(action)
                event_count = max(
                    int(info.get("kpi_ego_collision_events", info.get("ego_collision_events", 0))),
                    0,
                )
                if event_count == 0 and bool(info.get("ego_collision", False)):
                    event_count = 1
                collision_events += event_count
                if step == 1 or step % int(args.print_every) == 0 or terminated or truncated:
                    ego = env.unwrapped.vehicle
                    line = (
                        f"[render] episode={episode + 1} step={step} "
                        f"t={step * policy_dt:.1f}s v={float(ego.vx):.2f} "
                        f"v_des={float(ego.desired_speed):.2f}"
                    )
                    if args.cbf and raw_physical is not None:
                        line += (
                            f" ay_raw={float(raw_physical[1]):+.2f}"
                            f" ay_safe={float(info.get('cbf_a_safe_y', raw_physical[1])):+.2f}"
                            f" intervened={bool(info.get('cbf_intervened', False))}"
                        )
                    print(f"{line} collisions={collision_events}", flush=True)
                if terminated or truncated:
                    reason = "collision/terminal" if terminated else "timeout/truncated"
                    print(f"[render] episode ended ({reason})", flush=True)
                    break
    except KeyboardInterrupt:
        print("[render] interrupted by user", flush=True)
    finally:
        env.close()
        del model
        if th.cuda.is_available():
            th.cuda.empty_cache()
    print(f"[render] finished in {time.perf_counter() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
