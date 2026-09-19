"""Render short GIFs of trained PPO/CBF policies driving in the real,
non-scripted 40-vehicle MTM traffic environment (the same environment they
were trained and evaluated in -- not the six sparse scripted diagnostic
scenes rendered by ``render_ppo_scenario_gifs.py``).

Reuses ``make_evaluation_env`` / ``load_model`` / ``_predict_evaluation_action``
from ``run_ppo_cbf_progression`` so each render sees the exact wrapper stack,
CBF gains, and reward config the checkpoint was trained and evaluated with,
read from that run's own ``run_config.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.rendering.render_ppo_scenario_gifs import build_namespace, load_policy
from scripts.training.run_ppo_cbf_progression import (
    _predict_evaluation_action,
    load_model,
    make_evaluation_env,
)


def render_traffic_gif(
    namespace: dict[str, Any],
    policy: dict[str, Any],
    *,
    mode: str,
    action_source: str,
    steps: int,
    episode_seed: int,
    fps: int,
    out_path: Path,
) -> dict[str, Any]:
    env = make_evaluation_env(
        namespace,
        mode=mode,
        env_config=policy["env_config"],
        reward_config=policy["reward_config"],
        correction_epsilon=policy["correction_epsilon"],
    )
    env.unwrapped.render_mode = "rgb_array"
    env.unwrapped.config["offscreen_rendering"] = True
    env.unwrapped.config["real_time_rendering"] = False

    frames: list[Image.Image] = []
    collisions = 0
    terminated = False
    truncated = False
    try:
        obs, _ = env.reset(seed=int(episode_seed))
        frames.append(Image.fromarray(np.asarray(env.render(), dtype=np.uint8)))
        for _ in range(int(steps)):
            if terminated or truncated:
                break
            action = _predict_evaluation_action(
                policy["model"], obs, action_source=action_source
            )
            obs, _reward, terminated, truncated, info = env.step(action)
            collisions += max(int(info.get("ego_collision_events", 0)), 0)
            frames.append(Image.fromarray(np.asarray(env.render(), dtype=np.uint8)))
    finally:
        env.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        optimize=True,
    )
    return {
        "frames": len(frames),
        "policy_steps_rendered": len(frames) - 1,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "ego_collision_events": collisions,
    }


# Each entry: (policy_key, mode, action_source, output subpath)
RENDER_PLAN: list[tuple[str, str, str, str]] = [
    ("A1", "raw", "policy", "A1_nominal/cbf_off.gif"),
    ("A1", "cbf", "policy", "A1_nominal/cbf_on.gif"),
    ("B1", "raw", "policy", "B1_state_cost_raw/cbf_off.gif"),
    ("B1", "cbf", "policy", "B1_state_cost_raw/cbf_on.gif"),
    ("B2", "raw", "policy", "B2_lambda1p0/cbf_off.gif"),
    ("B2", "cbf", "policy", "B2_lambda1p0/cbf_on.gif"),
    ("C1", "raw", "policy", "C1_baseline/cbf_off_literal_projected.gif"),
    ("C1", "cbf", "policy", "C1_baseline/cbf_on.gif"),
    ("C1", "raw", "raw_actor_mean", "C1_baseline/cbf_off_raw_actor_mean.gif"),
    ("C2", "raw", "policy", "C2_lambda1p0/cbf_off_literal_projected.gif"),
    ("C2", "cbf", "policy", "C2_lambda1p0/cbf_on.gif"),
    ("C2", "raw", "raw_actor_mean", "C2_lambda1p0/cbf_off_raw_actor_mean.gif"),
]

_SAFETY_MATRIX_DIR = Path.home() / "AppData" / "Local" / "Temp" / "saferl_runs" / "safety_matrix_1M"

RUN_CONFIGS = {
    "A1": _SAFETY_MATRIX_DIR / "A1_nominal" / "seed_307" / "ppo_nominal" / "seed_307" / "run_config.json",
    "B1": _SAFETY_MATRIX_DIR / "B1_state_cost_raw" / "seed_307" / "ppo_hocbf_state_cost_raw" / "seed_307" / "run_config.json",
    "B2": Path.home() / "AppData" / "Local" / "Temp" / "saferl_runs" / "lambda_detached_actor_1p0"
    / "ldac1p0_s307" / "ppo_cbf_nd_reward_actor" / "seed_307" / "run_config.json",
    "C1": _SAFETY_MATRIX_DIR / "C1_cbf_diff_reward_only" / "seed_307" / "ppo_cbf_diff_reward_only" / "seed_307" / "run_config.json",
    "C2": Path.home() / "AppData" / "Local" / "Temp" / "saferl_runs" / "lambda_mean_1p0"
    / "lam1p0_s307" / "ppo_cbf_projected" / "seed_307" / "run_config.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--episode-seed", type=int, default=1_100_011)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from scripts.rendering.render_ppo_scenario_gifs import find_repo_root

    project_root = (args.project_root or find_repo_root()).resolve()

    policies: dict[str, dict[str, Any]] = {}
    summary: list[dict[str, Any]] = []
    for policy_key, run_config_path in RUN_CONFIGS.items():
        policy = load_policy(run_config_path)
        namespace = build_namespace(project_root, policy["cbf"])
        policy["model"] = load_model(policy["variant"], policy["model_path"], "cpu")
        policy["namespace"] = namespace
        policies[policy_key] = policy
        print(
            f"[traffic-gif] loaded {policy_key}: variant={policy['variant']} "
            f"seed={policy['training_seed']} model={policy['model_path']}",
            flush=True,
        )

    for policy_key, mode, action_source, rel_path in RENDER_PLAN:
        policy = policies[policy_key]
        out_path = args.out_dir / rel_path
        result = render_traffic_gif(
            policy["namespace"],
            policy,
            mode=mode,
            action_source=action_source,
            steps=int(args.steps),
            episode_seed=int(args.episode_seed),
            fps=int(args.fps),
            out_path=out_path,
        )
        result.update(
            policy=policy_key,
            mode=mode,
            action_source=action_source,
            episode_seed=int(args.episode_seed),
            out_path=str(out_path),
        )
        summary.append(result)
        print(
            f"  {rel_path}: {result['policy_steps_rendered']} steps, "
            f"collisions={result['ego_collision_events']}, "
            f"terminated={result['terminated']}",
            flush=True,
        )

    (args.out_dir / "render_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
