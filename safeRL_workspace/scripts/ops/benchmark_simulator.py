"""Deterministic throughput benchmark for the lane-free simulator.

The benchmark intentionally uses the base Gymnasium environment and a fixed
action stream.  It reports policy-step and physics-frame throughput for the
traffic/guard combinations that matter during training, together with a
short state checksum that makes accidental behavioral drift easy to spot.

Examples::

    python -m scripts.ops.benchmark_simulator --vehicles 55 --steps 250
    python -m scripts.ops.benchmark_simulator --traffic-model all --guard both
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def find_project_root(start: Path) -> Path:
    """Find the organized project root containing the lane-free environment."""

    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "laneless highway env" / "lane_free_env.py").exists():
            return candidate
        nested = candidate / "safeRL_workspace"
        if (nested / "laneless highway env" / "lane_free_env.py").exists():
            return nested
    raise RuntimeError(
        "Could not find project root containing laneless highway env/lane_free_env.py"
    )


def _load_environment(project_root: Path):
    env_root = project_root / "laneless highway env"
    env_root_string = str(env_root)
    if env_root_string not in sys.path:
        sys.path.insert(0, env_root_string)
    import lane_free_env  # noqa: F401

    return lane_free_env.LaneFreeTrafficEnv


def _state_checksum(snapshot: np.ndarray) -> str:
    """Return a compact checksum of the final raw simulator state."""

    return hashlib.sha256(np.asarray(snapshot, dtype=np.float64).tobytes()).hexdigest()[:16]


def run_benchmark(
    project_root: Path,
    *,
    traffic_model: str,
    guard: bool,
    vehicles: int,
    steps: int,
    warmup: int,
    seed: int,
    simulation_frequency: int = 100,
    policy_frequency: int = 10,
) -> dict[str, Any]:
    """Run one fixed-seed benchmark and return JSON-serializable metrics."""

    if vehicles < 1:
        raise ValueError("vehicles must be positive")
    if steps < 1:
        raise ValueError("steps must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if simulation_frequency < 1 or policy_frequency < 1:
        raise ValueError("simulation_frequency and policy_frequency must be positive")

    environment_class = _load_environment(project_root)
    config = {
        "traffic_model": str(traffic_model),
        "vehicles_count": int(vehicles),
        # Keep the episode alive for the entire timed section.
        "episode_steps": int(warmup + steps + 1),
        "duration": int(warmup + steps + 1),
        "terminate_on_collision": False,
        "simulation_frequency": int(simulation_frequency),
        "policy_frequency": int(policy_frequency),
        "traffic_safety": {"dynamics_guard": bool(guard)},
    }
    env = environment_class(config=config)
    actions = np.random.default_rng(int(seed)).uniform(
        -1.0, 1.0, size=(warmup + steps, 2)
    ).astype(np.float32)
    try:
        env.reset(seed=int(seed))
        for action in actions[:warmup]:
            env.step(action)

        start = time.perf_counter()
        rewards: list[float] = []
        for action in actions[warmup:]:
            _obs, reward, _terminated, _truncated, _info = env.step(action)
            rewards.append(float(reward))
        elapsed = max(time.perf_counter() - start, np.finfo(float).tiny)
        physics_frames = int(
            steps
            * max(1, int(round(float(simulation_frequency) / float(policy_frequency))))
        )
        return {
            "traffic_model": str(traffic_model),
            "guard": bool(guard),
            "vehicles": int(vehicles),
            "warmup_steps": int(warmup),
            "timed_steps": int(steps),
            "physics_frames": physics_frames,
            "elapsed_s": float(elapsed),
            "policy_steps_per_s": float(steps / elapsed),
            "physics_frames_per_s": float(physics_frames / elapsed),
            "mean_reward": float(np.mean(rewards)),
            "final_state_sha256": _state_checksum(env.snapshot()),
        }
    finally:
        env.close()


def _make_subprocess_worker(
    project_root: str, config: dict[str, Any]
):
    """Return a cloudpickle-friendly factory for one spawned base simulator."""

    def factory():
        environment_class = _load_environment(Path(project_root))
        return environment_class(config=copy.deepcopy(config))

    return factory


def run_parallel_benchmark(
    project_root: Path,
    *,
    traffic_model: str,
    guard: bool,
    vehicles: int,
    steps: int,
    warmup: int,
    seed: int,
    workers: int,
    simulation_frequency: int = 100,
    policy_frequency: int = 10,
) -> dict[str, Any]:
    """Benchmark aggregate transition throughput across spawned workers.

    Process creation and notebook import are intentionally outside the timed
    section.  The result therefore measures the steady-state rollout cost
    that contributes to PPO sample collection, not one-time startup latency.
    """

    if workers < 2:
        raise ValueError("parallel benchmark requires at least two workers")
    if vehicles < 1 or steps < 1 or warmup < 0:
        raise ValueError("vehicles/steps must be positive and warmup non-negative")
    if simulation_frequency < 1 or policy_frequency < 1:
        raise ValueError("simulation_frequency and policy_frequency must be positive")

    from stable_baselines3.common.vec_env import SubprocVecEnv

    config = {
        "traffic_model": str(traffic_model),
        "vehicles_count": int(vehicles),
        "episode_steps": int(warmup + steps + 1),
        "duration": int(warmup + steps + 1),
        "terminate_on_collision": False,
        "simulation_frequency": int(simulation_frequency),
        "policy_frequency": int(policy_frequency),
        "traffic_safety": {"dynamics_guard": bool(guard)},
    }
    root_text = str(Path(project_root).resolve())
    env_fns = [
        _make_subprocess_worker(root_text, copy.deepcopy(config))
        for _ in range(int(workers))
    ]
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    actions = np.random.default_rng(int(seed)).uniform(
        -1.0, 1.0, size=(warmup + steps, int(workers), 2)
    ).astype(np.float32)
    try:
        vec_env.seed(int(seed))
        vec_env.reset()
        for action in actions[:warmup]:
            vec_env.step(action)

        start = time.perf_counter()
        rewards: list[float] = []
        observations = None
        for action in actions[warmup:]:
            observations, step_rewards, _dones, _infos = vec_env.step(action)
            rewards.extend(float(value) for value in np.asarray(step_rewards).reshape(-1))
        elapsed = max(time.perf_counter() - start, np.finfo(float).tiny)
        frames_per_step = max(
            1,
            int(round(float(simulation_frequency) / float(policy_frequency))),
        )
        aggregate_steps = int(steps * workers)
        aggregate_frames = int(aggregate_steps * frames_per_step)
        observation_bytes = np.asarray(observations, dtype=np.float32).tobytes()
        return {
            "backend": "subproc",
            "workers": int(workers),
            "traffic_model": str(traffic_model),
            "guard": bool(guard),
            "vehicles": int(vehicles),
            "warmup_steps": int(warmup),
            "timed_steps": int(steps),
            "aggregate_policy_steps": aggregate_steps,
            "physics_frames": aggregate_frames,
            "elapsed_s": float(elapsed),
            "vector_policy_ticks_per_s": float(steps / elapsed),
            "aggregate_policy_steps_per_s": float(aggregate_steps / elapsed),
            "aggregate_physics_frames_per_s": float(aggregate_frames / elapsed),
            "mean_reward": float(np.mean(rewards)),
            "final_observation_sha256": hashlib.sha256(observation_bytes).hexdigest()[:16],
        }
    finally:
        vec_env.close()


def _choices(value: str, allowed: Iterable[str], *, label: str) -> list[str]:
    if value == "all":
        return list(allowed)
    if value not in allowed:
        raise ValueError(f"{label} must be one of {', '.join((*allowed, 'all'))}")
    return [value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark deterministic lane-free simulator throughput."
    )
    parser.add_argument("--vehicles", type=int, default=55)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--traffic-model", choices=("force", "mtm", "all"), default="mtm")
    parser.add_argument("--guard", choices=("on", "off", "both"), default="on")
    # Match the canonical notebook/training protocol (10 policy Hz, 100
    # dynamics Hz) unless a caller deliberately requests another ratio.
    parser.add_argument("--simulation-frequency", type=int, default=100)
    parser.add_argument("--policy-frequency", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Use direct execution with 1 (default), or benchmark this many spawned workers.",
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for the JSON result (human-readable output is still printed).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = find_project_root(args.project_root or Path.cwd())
    models = _choices(args.traffic_model, ("force", "mtm"), label="traffic-model")
    if args.guard == "both":
        guards = [False, True]
    else:
        guards = [args.guard == "on"]
    results = [
        (
            run_benchmark(
                project_root,
                traffic_model=model,
                guard=guard,
                vehicles=args.vehicles,
                steps=args.steps,
                warmup=args.warmup,
                seed=args.seed,
                simulation_frequency=args.simulation_frequency,
                policy_frequency=args.policy_frequency,
            )
            if int(args.workers) == 1
            else run_parallel_benchmark(
                project_root,
                traffic_model=model,
                guard=guard,
                vehicles=args.vehicles,
                steps=args.steps,
                warmup=args.warmup,
                seed=args.seed,
                workers=int(args.workers),
                simulation_frequency=args.simulation_frequency,
                policy_frequency=args.policy_frequency,
            )
        )
        for model in models
        for guard in guards
    ]

    print("backend workers traffic_model guard vehicles ticks aggregate_steps elapsed_s ticks/s aggregate_steps/s aggregate_frames/s mean_reward checksum")
    for result in results:
        workers = int(result.get("workers", 1))
        ticks_per_s = float(
            result.get("vector_policy_ticks_per_s")
            or result.get("policy_steps_per_s", 0.0)
        )
        aggregate_steps_per_s = float(
            result.get("aggregate_policy_steps_per_s")
            or result.get("policy_steps_per_s", 0.0)
        )
        aggregate_frames_per_s = float(
            result.get("aggregate_physics_frames_per_s")
            or result.get("physics_frames_per_s", 0.0)
        )
        checksum = result.get(
            "final_observation_sha256", result.get("final_state_sha256", "")
        )
        print(
            f"{result.get('backend', 'direct'):>7} {workers:>7}"
            f" {result['traffic_model']:>13} {str(result['guard']):>5}"
            f" {result['vehicles']:>8} {result['timed_steps']:>5}"
            f" {result.get('aggregate_policy_steps', result['timed_steps']):>16}"
            f" {result['elapsed_s']:>9.4f} {ticks_per_s:>9.2f}"
            f" {aggregate_steps_per_s:>17.2f} {aggregate_frames_per_s:>18.2f}"
            f" {result['mean_reward']:>11.5f} {checksum}"
        )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"[sim-benchmark] wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
