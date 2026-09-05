"""Live-render a saved nominal PPO checkpoint with optional external CBF.

The nominal 500k checkpoint was trained with the 114-dimensional physical-action
plus CBF-context observation contract.  This renderer preserves that contract,
and can either pass through the actor action or project it through the CBF.
"""

from __future__ import annotations

import argparse
import copy
import faulthandler
import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import scripts.training.run_cbf_filter_ablation as protocol
from scripts.common.ppo_cbf_env import CBFContextPhysicalActionWrapper
from scripts.common.projected_ppo_cbf import LatentActionPPO, ProjectedCBFPPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1_100_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--cbf",
        action="store_true",
        help="Project each actor action through the external CBF before stepping.",
    )
    return parser.parse_args()


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


def make_render_env(
    namespace: dict[str, Any],
    *,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    cbf_snapshot: dict[str, Any],
    action_rate_penalty_lambda: float,
    project_inputs: bool,
) -> gym.Env:
    namespace.update(cbf_snapshot)
    base = gym.make(
        "lane-free-v0",
        render_mode="human",
        config=copy.deepcopy(env_config),
    )
    env: gym.Env = namespace["KaralakouRewardWrapper"](
        base,
        reward_config=copy.deepcopy(reward_config),
    )
    if namespace.get("NORMALIZE_RL_OBSERVATIONS", False):
        env = namespace["LaneFreeObservationNormalizationWrapper"](
            env,
            clip=namespace["OBSERVATION_CLIP"],
        )

    # This is the same observation/action wrapper used by the PPO run.
    # ``project_inputs`` selects raw actor deployment or external CBF deployment.
    env = CBFContextPhysicalActionWrapper(
        env,
        namespace=namespace,
        ax_bounds=namespace["CBF_AX_BOUNDS"],
        ay_bounds=namespace["CBF_AY_BOUNDS"],
        neighbor_range=float(namespace["CBF_NEIGHBOR_RANGE"]),
        eps_side=float(namespace["CBF_EPS_SIDE"]),
        k0=float(namespace["CBF_K0"]),
        k1=float(namespace["CBF_K1"]),
        max_neighbor_constraints=int(namespace["CBF_MAX_NEIGHBOR_CONSTRAINTS"]),
        base_observation_dim=int(np.prod(env.observation_space.shape)),
        project_inputs=bool(project_inputs),
        lambda_delta=0.0,
        lambda_intervention=0.0,
        correction_epsilon=0.03,
        action_rate_penalty_lambda=float(action_rate_penalty_lambda),
    )
    if "KPIInfoWrapper" in namespace:
        env = namespace["KPIInfoWrapper"](env, intervention_threshold=0.03)
    return protocol.ProtocolMetricsWrapper(env)


def main() -> int:
    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    args = parse_args()
    project_root = args.project_root.resolve()
    os.chdir(project_root)

    model_path = args.model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    run_config_path = (
        args.run_config.resolve()
        if args.run_config is not None
        else model_path.parent / "run_config.json"
    )
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    training_signature = run_config.get("training_signature", {})
    cbf_snapshot = dict(training_signature.get("cbf", {}))
    if not cbf_snapshot:
        raise RuntimeError(f"No saved CBF observation-context settings in {run_config_path}")

    namespace = protocol.bootstrap_notebook_namespace(project_root)
    protocol.exec_required_notebook_cells(
        project_root / "notebooks" / "lanelessKaralakou.ipynb",
        namespace,
    )
    namespace["DEVICE"] = args.device

    env_config = copy.deepcopy(run_config["env_config"])
    env_config["real_time_rendering"] = True
    reward_config = copy.deepcopy(run_config["reward_config"])
    action_rate_penalty_lambda = float(run_config.get("action_rate_penalty", 0.0))
    max_steps = int(args.steps or env_config.get("episode_steps", 800))

    env = make_render_env(
        namespace,
        env_config=env_config,
        reward_config=reward_config,
        cbf_snapshot=cbf_snapshot,
        action_rate_penalty_lambda=action_rate_penalty_lambda,
        project_inputs=bool(args.cbf),
    )
    model_class = (
        ProjectedCBFPPO
        if run_config.get("variant") == "ppo_cbf_projected"
        else LatentActionPPO
    )
    model = model_class.load(str(model_path), device=args.device, env=env)
    print(f"[render-ppo] loaded {model_path}", flush=True)
    print(
        f"[render-ppo] CBF {'ON' if args.cbf else 'OFF'}: "
        f"project_inputs={bool(args.cbf)} | episodes={args.episodes} | "
        f"max_steps={max_steps} | seed_start={args.seed}",
        flush=True,
    )
    print(f"[render-ppo] observation_space={env.observation_space}", flush=True)

    try:
        for episode in range(int(args.episodes)):
            observation, _ = env.reset(seed=int(args.seed) + episode)
            total_reward = 0.0
            collision_events = 0
            steps = 0
            for _ in range(max_steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                collision_events += max(int(info.get("ego_collision_events", 0)), 0)
                if terminated or truncated:
                    break
            print(
                f"[render-ppo] episode {episode + 1:02d}/{args.episodes}: "
                f"steps={steps} return={total_reward:.3f} "
                f"collision_events={collision_events}",
                flush=True,
            )
    finally:
        env.close()
    print("[render-ppo] finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
