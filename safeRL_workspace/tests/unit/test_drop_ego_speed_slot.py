"""Contract tests for dropping the constant ego desired-speed observation slot."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pytest

from saferl.observations import (
    ego_speed_slot_index,
    resolve_drop_ego_speed_slot,
)
from scripts.training import run_ppo_cbf_progression as progression
from scripts.training.run_cbf_filter_ablation import (
    bootstrap_notebook_namespace,
    exec_required_notebook_cells,
)


def _notebook_namespace() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    namespace = bootstrap_notebook_namespace(root)
    exec_required_notebook_cells(root / "notebooks" / "lanelessKaralakou.ipynb", namespace)
    return namespace


def _observations(namespace, env_config, reward_config, steps: int = 5):
    env = progression._base_environment(
        namespace,
        env_config=copy.deepcopy(env_config),
        reward_config=copy.deepcopy(reward_config),
    )
    rng = np.random.default_rng(7)
    observations = []
    try:
        observation, _info = env.reset(seed=11)
        observations.append(np.asarray(observation, dtype=np.float32))
        declared = int(np.prod(env.observation_space.shape))
        for _ in range(steps):
            observation, _r, terminated, truncated, _info = env.step(
                rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            )
            observations.append(np.asarray(observation, dtype=np.float32))
            if terminated or truncated:
                break
    finally:
        env.close()
    return observations, declared


def test_slot_index_is_the_last_feature_of_the_ego_row():
    # The ego is row 0, so the slot does not move with neighbors_count.
    assert ego_speed_slot_index({"observation_include_vehicle_dimensions": False}) == 4
    assert ego_speed_slot_index({"observation_include_vehicle_dimensions": True}) == 6
    assert (
        ego_speed_slot_index(
            {"observation_include_vehicle_dimensions": False, "neighbors_count": 7}
        )
        == 4
    )


def test_flag_defaults_to_off():
    assert resolve_drop_ego_speed_slot({}) is False
    assert resolve_drop_ego_speed_slot({"drop_ego_speed_slot": True}) is True


@pytest.mark.parametrize("neighbors_count, kept, dropped", [(5, 32, 31), (7, 42, 41)])
def test_declared_width_loses_exactly_one_feature(neighbors_count, kept, dropped):
    env_config = {
        "neighbors_count": neighbors_count,
        "observation_include_vehicle_dimensions": False,
        "ppo_append_previous_action": True,
    }
    assert progression._base_observation_dim(env_config) == kept
    assert (
        progression._base_observation_dim({**env_config, "drop_ego_speed_slot": True})
        == dropped
    )


@pytest.mark.slow
def test_dropping_removes_the_ego_speed_feature_and_nothing_else():
    namespace = _notebook_namespace()
    env_config = copy.deepcopy(namespace["ENV_CONFIG"])
    env_config["neighbors_count"] = 7
    reward_config = copy.deepcopy(namespace["REWARD_CONFIG"])
    reward_config["ego_desired_speed"] = 20.0

    kept, kept_declared = _observations(namespace, env_config, reward_config)
    dropped, dropped_declared = _observations(
        namespace, {**env_config, "drop_ego_speed_slot": True}, reward_config
    )

    slot = ego_speed_slot_index(env_config)
    assert kept_declared == 42
    assert dropped_declared == 41
    assert len(kept) == len(dropped)
    for full, narrow in zip(kept, dropped):
        assert full.size == 42
        assert narrow.size == 41
        # The constant slot is gone; every other feature is untouched, which
        # includes the neighbour rows' own desired speeds.
        np.testing.assert_allclose(np.delete(full, slot), narrow, rtol=0, atol=0)


@pytest.mark.slow
def test_the_dropped_slot_really_was_the_constant_speed_target():
    namespace = _notebook_namespace()
    env_config = copy.deepcopy(namespace["ENV_CONFIG"])
    reward_config = copy.deepcopy(namespace["REWARD_CONFIG"])
    reward_config["ego_desired_speed"] = 20.0
    observation_vmax = float(env_config.get("observation_vmax", 24.0))

    kept, _declared = _observations(namespace, env_config, reward_config)
    slot = ego_speed_slot_index(env_config)
    values = np.asarray([observation[slot] for observation in kept], dtype=float)
    np.testing.assert_allclose(values, 20.0 / observation_vmax, rtol=1e-6, atol=1e-6)


def test_the_flag_requires_the_nominal_speed_target():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop-ego-speed-slot", action="store_true")
    args = parser.parse_args(["--drop-ego-speed-slot"])
    assert args.drop_ego_speed_slot is True
