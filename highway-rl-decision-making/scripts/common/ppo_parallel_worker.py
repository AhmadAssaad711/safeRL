"""Importable SubprocVecEnv workers for the nominal PPO pilot.

The notebook namespace contains native objects that cannot be pickled on
Windows.  Each worker therefore builds its own namespace after it has spawned;
only plain configuration data crosses the process boundary.
"""

from __future__ import annotations

import copy
import functools
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import scripts.training.run_cbf_filter_ablation as pipeline
from scripts.common.ppo_observation_variants import install_previous_action_observation


def make_parallel_worker_env(
    *,
    project_root: str,
    seed: int,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    observation_at1: bool,
    monitor_path: str,
) -> gym.Env:
    """Build one complete worker environment inside the spawned process."""

    root = Path(project_root)
    namespace = pipeline.bootstrap_notebook_namespace(root)
    pipeline.exec_required_notebook_cells(
        root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    if observation_at1:
        install_previous_action_observation(namespace)
    environment = pipeline.make_raw_env(
        namespace,
        seed=int(seed),
        env_config=copy.deepcopy(env_config),
        reward_config=copy.deepcopy(reward_config),
    )
    if not bool(environment.unwrapped.config.get("terminate_on_collision", False)):
        environment.close()
        raise RuntimeError("Parallel PPO training requires terminate_on_collision=True")
    environment = pipeline.ProtocolMetricsWrapper(environment)
    return Monitor(
        environment,
        filename=str(monitor_path),
        info_keywords=pipeline.TRAINING_MONITOR_INFO_KEYS,
        override_existing=True,
    )


def make_parallel_subproc_training_env(
    *,
    project_root: Path,
    seed: int,
    n_envs: int,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    observation_at1: bool,
    monitor_path: Path,
) -> SubprocVecEnv:
    """Return real parallel workers without serializing the notebook namespace."""

    n_workers = int(n_envs)
    if n_workers < 2:
        raise ValueError("SubprocVecEnv requires at least two workers")
    root_text = str(Path(project_root).resolve())
    base_monitor = Path(monitor_path)
    env_fns = []
    for worker_index in range(n_workers):
        worker_seed = int(seed) + worker_index
        worker_monitor = base_monitor.with_name(
            f"{base_monitor.stem}.env_{worker_index}{base_monitor.suffix}"
        )
        env_fns.append(
            functools.partial(
                make_parallel_worker_env,
                project_root=root_text,
                seed=worker_seed,
                env_config=copy.deepcopy(env_config),
                reward_config=copy.deepcopy(reward_config),
                observation_at1=bool(observation_at1),
                monitor_path=str(worker_monitor),
            )
        )
    return SubprocVecEnv(env_fns, start_method="spawn")
