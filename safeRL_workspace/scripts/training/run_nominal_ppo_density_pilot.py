"""Clean nominal-PPO pilot at a selected MTM traffic density.

The reward and PPO hyperparameters are copied from an existing PPO run
configuration.  The default experiment change is ``vehicles_count``; the
observation mode and scales can be overridden explicitly.  The environment
is raw MTM plus the nominal reward wrapper and physical-action adapter; no CBF
context, constraint, projection, or filter is constructed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

import scripts.evaluation.audit_nominal_mtm_collision_provenance as provenance


DEFAULT_EVAL_EPISODES = 20
DEFAULT_EVAL_SEED_START = 1_300_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vehicles-count", type=int, default=10)
    parser.add_argument("--training-seed", type=int, default=307)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument(
        "--observation-mode",
        choices=("full", "minimal"),
        default=None,
        help="Optional observation layout override.",
    )
    parser.add_argument(
        "--observation-vmax",
        type=float,
        default=None,
        help="Optional longitudinal observation scale override (m/s).",
    )
    parser.add_argument(
        "--observation-vymax",
        type=float,
        default=None,
        help="Optional lateral observation scale override (m/s).",
    )
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--eval-seed-start", type=int, default=DEFAULT_EVAL_SEED_START)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


class ProgressCallback(BaseCallback):
    def __init__(self, report_interval: int, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.report_interval = int(report_interval)
        self.next_report = int(report_interval)

    def _on_rollout_end(self) -> None:
        step = int(self.model.num_timesteps)
        while step >= self.next_report:
            print(
                f"[nominal-ppo] rollout boundary reached: {self.next_report:,} steps",
                flush=True,
            )
            self.next_report += self.report_interval

    def _on_step(self) -> bool:
        return True


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_env(
    namespace: dict[str, Any],
    *,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    monitor_path: Path | None = None,
) -> gym.Env:
    env = provenance.basics.make_nominal_env(
        namespace,
        env_config=env_config,
        reward_config=reward_config,
    )
    if monitor_path is not None:
        monitor_path.parent.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(monitor_path))
    else:
        env = Monitor(env)
    return env


def _make_vec_env(
    namespace: dict[str, Any],
    *,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    n_envs: int,
    training_seed: int,
    output_dir: Path,
) -> DummyVecEnv:
    def factory(rank: int):
        def make() -> gym.Env:
            return _make_env(
                namespace,
                env_config=env_config,
                reward_config=reward_config,
                monitor_path=output_dir / f"monitor_{rank}.csv",
            )

        return make

    vec_env = DummyVecEnv([factory(rank) for rank in range(int(n_envs))])
    vec_env.seed(int(training_seed))
    return vec_env


def _finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else np.nan


def evaluate_model(
    model: PPO,
    namespace: dict[str, Any],
    *,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_steps = provenance.basics._max_policy_steps(env_config) + 2
    eps_side = float(reward_config.get("safety_potential_eps_side", 0.10))
    for seed in seeds:
        env = _make_env(
            namespace,
            env_config=env_config,
            reward_config=reward_config,
        )
        try:
            observation, _ = env.reset(seed=int(seed))
            base = provenance.basics._base(env)
            previous_x = float(base.vehicle.position[0])
            distance_m = 0.0
            episode_return = 0.0
            steps = 0
            collision_events = 0
            collided = False
            first_collision_s = np.nan
            min_h = np.inf
            abs_actions: list[float] = []
            saturated_actions = 0
            action_components = 0
            for _ in range(max_steps):
                for vehicle in base.road.vehicles:
                    if vehicle is not base.vehicle:
                        min_h = min(
                            min_h,
                            provenance.basics._ellipse_h(
                                base, vehicle, eps_side
                            ),
                        )
                action, _ = model.predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
                abs_actions.extend(np.abs(action).astype(float).tolist())
                saturated_actions += int(np.any(np.abs(action) >= 2.99))
                action_components += 2
                observation, reward, terminated, truncated, info = env.step(action)
                info = dict(info)
                base = provenance.basics._base(env)
                distance_m += provenance.basics._distance_step(base, previous_x)
                previous_x = float(base.vehicle.position[0])
                episode_return += float(reward)
                steps += 1
                step_events = provenance.basics._collision_events(base, info)
                collision_events += step_events
                if step_events > 0 or bool(info.get("ego_collision", False)):
                    if not collided:
                        first_collision_s = float(base.time)
                    collided = True
                if terminated or truncated:
                    break
            rows.append(
                {
                    "seed": int(seed),
                    "steps": int(steps),
                    "episode_return": float(episode_return),
                    "distance_m": float(distance_m),
                    "collision": int(collided),
                    "collision_events": int(collision_events),
                    "first_collision_s": float(first_collision_s),
                    "min_ellipse_h": float(min_h),
                    "mean_abs_action": _finite_mean(abs_actions),
                    "action_saturation_rate": float(
                        saturated_actions / max(steps, 1)
                    ),
                }
            )
        finally:
            env.close()
    return rows


def aggregate_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_distance = float(sum(row["distance_m"] for row in rows))
    total_collisions = int(sum(row["collision_events"] for row in rows))
    return {
        "episodes": int(len(rows)),
        "collision_episode_rate": float(np.mean([row["collision"] for row in rows])),
        "mean_collision_events": float(
            np.mean([row["collision_events"] for row in rows])
        ),
        "collisions_per_km": float(
            1000.0 * total_collisions / max(total_distance, 1e-9)
        ),
        "mean_episode_return": float(
            np.mean([row["episode_return"] for row in rows])
        ),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_distance_m": float(np.mean([row["distance_m"] for row in rows])),
        "mean_first_collision_s": _finite_mean(
            [row["first_collision_s"] for row in rows]
        ),
        "mean_min_ellipse_h": float(np.mean([row["min_ellipse_h"] for row in rows])),
        "mean_abs_action": float(np.mean([row["mean_abs_action"] for row in rows])),
        "mean_action_saturation_rate": float(
            np.mean([row["action_saturation_rate"] for row in rows])
        ),
    }


def main() -> int:
    args = parse_args()
    if int(args.vehicles_count) <= 0:
        raise ValueError("vehicles-count must be positive")
    if int(args.timesteps) <= 0 or int(args.n_envs) <= 0:
        raise ValueError("timesteps and n-envs must be positive")
    if int(args.timesteps) % (250 * int(args.n_envs)) != 0:
        raise ValueError("timesteps must be divisible by n_steps*n_envs = 250*n_envs")

    source_run_dir = args.source_run_dir.resolve()
    source_config_path = source_run_dir / "run_config.json"
    if not source_config_path.is_file():
        raise FileNotFoundError(f"Missing source run config: {source_config_path}")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    env_config = copy.deepcopy(source_config["env_config"])
    reward_config = copy.deepcopy(source_config["reward_config"])
    ppo_config = copy.deepcopy(source_config["ppo_config"])
    if str(env_config.get("traffic_model", "")).strip().lower() != "mtm":
        raise RuntimeError("This pilot is intentionally MTM-only")
    env_config["vehicles_count"] = int(args.vehicles_count)
    env_config["ego_controlled"] = True
    env_config["terminate_on_collision"] = True
    if args.observation_mode is not None:
        env_config["observation_mode"] = str(args.observation_mode)
    if args.observation_vmax is not None:
        if float(args.observation_vmax) <= 0.0:
            raise ValueError("observation-vmax must be positive")
        env_config["observation_vmax"] = float(args.observation_vmax)
    if args.observation_vymax is not None:
        if float(args.observation_vymax) <= 0.0:
            raise ValueError("observation-vymax must be positive")
        env_config["observation_vymax"] = float(args.observation_vymax)

    if int(ppo_config.get("n_steps", 250)) != 250:
        raise RuntimeError("Source PPO configuration does not use n_steps=250")
    if int(ppo_config.get("batch_size", 200)) != 200:
        raise RuntimeError("Source PPO configuration does not use batch_size=200")

    project_root = provenance.basics.notebook_pipeline.find_project_root(
        args.project_root or Path.cwd()
    )
    provenance.basics.notebook_pipeline.set_stable_native_defaults()
    namespace = provenance._bootstrap_nominal_namespace(project_root)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(int(args.training_seed))
    th.manual_seed(int(args.training_seed))

    n_steps = int(ppo_config["n_steps"])
    n_envs = int(args.n_envs)
    vec_env = _make_vec_env(
        namespace,
        env_config=env_config,
        reward_config=reward_config,
        n_envs=n_envs,
        training_seed=int(args.training_seed),
        output_dir=output_dir,
    )
    tensorboard_dir = output_dir / "tensorboard"
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=float(ppo_config["learning_rate"]),
        n_steps=n_steps,
        batch_size=int(ppo_config["batch_size"]),
        n_epochs=int(ppo_config["n_epochs"]),
        gamma=float(ppo_config["gamma"]),
        gae_lambda=float(ppo_config["gae_lambda"]),
        clip_range=float(ppo_config["clip_range"]),
        ent_coef=float(ppo_config["ent_coef"]),
        vf_coef=float(ppo_config["vf_coef"]),
        max_grad_norm=float(ppo_config["max_grad_norm"]),
        use_sde=False,
        policy_kwargs={
            "net_arch": {"pi": [256, 128], "vf": [256, 128]},
            "activation_fn": th.nn.Tanh,
            "ortho_init": True,
            "log_std_init": float(ppo_config["log_std_init"]),
        },
        tensorboard_log=str(tensorboard_dir),
        verbose=0,
        seed=int(args.training_seed),
        device=str(args.device),
    )
    started = time.perf_counter()
    print(
        f"[nominal-ppo] training vehicles={args.vehicles_count} "
        f"timesteps={args.timesteps:,} n_envs={n_envs} "
        "reward=current-source-config cbf_runtime=False",
        flush=True,
    )
    model.learn(
        total_timesteps=int(args.timesteps),
        callback=ProgressCallback(report_interval=n_steps * n_envs),
        progress_bar=False,
    )
    elapsed_s = float(time.perf_counter() - started)
    model_path = output_dir / "model_final.zip"
    model.save(str(model_path))
    vec_env.close()

    model = PPO.load(str(model_path), device=str(args.device))
    eval_seeds = [
        int(args.eval_seed_start) + index for index in range(int(args.eval_episodes))
    ]
    print(
        f"[nominal-ppo] evaluating {len(eval_seeds)} episodes at vehicles={args.vehicles_count}",
        flush=True,
    )
    evaluation_rows = evaluate_model(
        model,
        namespace,
        env_config=env_config,
        reward_config=reward_config,
        seeds=eval_seeds,
    )
    evaluation_summary = aggregate_evaluation(evaluation_rows)
    _write_csv(output_dir / "evaluation_episodes.csv", evaluation_rows)
    config_payload = {
        "source_run_config": str(source_config_path),
        "traffic_model": "mtm",
        "vehicles_count": int(args.vehicles_count),
        "training_seed": int(args.training_seed),
        "timesteps": int(args.timesteps),
        "n_envs": int(n_envs),
        "n_steps": int(n_steps),
        "global_rollout_steps": int(n_steps * n_envs),
        "eval_seeds": eval_seeds,
        "cbf_runtime": False,
        "environment": "raw MTM + Karalakou reward + nominal physical-action adapter",
        "env_config": env_config,
        "reward_config": reward_config,
        "ppo_config": ppo_config,
        "elapsed_training_s": elapsed_s,
        "evaluation_summary": evaluation_summary,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config_payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(evaluation_summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(evaluation_summary, indent=2, allow_nan=True), flush=True)
    print(f"[nominal-ppo] wrote {model_path}", flush=True)
    print(f"[nominal-ppo] elapsed_training_s={elapsed_s:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
