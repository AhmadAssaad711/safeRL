from __future__ import annotations

import copy

import pytest

from scripts.evaluation.evaluate_hocbf_gain_grid import gain_grid_cells
from scripts.evaluation.evaluate_ppo_cbf_overrides import apply_env_overrides, hocbf_gains


def _env_config() -> dict:
    return {
        "simulation_frequency": 100,
        "policy_frequency": 20,
        "cbf_frequency": 20,
        "dt": 0.01,
        "vehicles_count": 55,
        "neighbors_count": 5,
        "cbf_substep_filtering": False,
        "cbf_require_initial_safe_set": True,
        "traffic_safety": {"spawn_cbf_psi1_gain": 2.3, "spawn_cbf_k1": 2.3, "safe_spawn": True},
    }


def test_hocbf_gains_follow_the_class_k_cascade_and_are_symmetric():
    gains = hocbf_gains(0.5, 8.0)
    assert gains == {"CBF_K0": 4.0, "CBF_K1": 8.5, "CBF_PSI1_GAIN": 0.5}
    assert hocbf_gains(8.0, 0.5) == gains
    assert gains["CBF_K1"] ** 2 >= 4.0 * gains["CBF_K0"]


@pytest.mark.parametrize("rates", [(0.0, 1.0), (-0.5, 2.0), (float("nan"), 1.0)])
def test_hocbf_gains_reject_non_positive_rates(rates):
    with pytest.raises(ValueError):
        hocbf_gains(*rates)


def test_gain_grid_evaluates_each_unordered_pair_once():
    cells = gain_grid_cells((0.5, 1.0, 8.0))
    assert [label for label, _ in cells] == [
        "c1=0.5_c2=0.5", "c1=0.5_c2=1", "c1=0.5_c2=8", "c1=1_c2=1", "c1=1_c2=8", "c1=8_c2=8",
    ]
    explicit = gain_grid_cells(pairs=[(8.0, 0.5), (0.5, 8.0), (2.3, 2.3)])
    assert [label for label, _ in explicit] == ["c1=0.5_c2=8", "c1=2.3_c2=2.3"]


def test_frequency_override_sets_all_rates_and_keeps_the_task_horizon():
    source = _env_config()
    original = copy.deepcopy(source)
    config, max_steps, changes = apply_env_overrides(source, max_policy_steps=3000, hz=100)
    assert source == original
    assert (config["simulation_frequency"], config["policy_frequency"], config["cbf_frequency"]) == (100, 100, 100)
    assert config["dt"] == pytest.approx(0.01)
    assert max_steps == 15000
    assert changes["task_max_policy_steps"] == [3000, 15000]

    config, max_steps, _ = apply_env_overrides(source, max_policy_steps=3000, hz=20)
    assert config["dt"] == pytest.approx(0.05)
    assert max_steps == 3000


def test_density_and_reset_overrides():
    config, _, changes = apply_env_overrides(
        _env_config(), max_policy_steps=3000, vehicles=40, spawn_psi1_gain=0.5, require_initial_safe_set=False
    )
    assert config["vehicles_count"] == 40
    assert config["traffic_safety"]["spawn_cbf_psi1_gain"] == 0.5
    assert config["traffic_safety"]["spawn_cbf_k1"] == 0.5
    assert config["traffic_safety"]["safe_spawn"] is True
    assert config["cbf_require_initial_safe_set"] is False
    assert changes["vehicles_count"] == [55, 40]


def test_overrides_reject_invalid_requests():
    with pytest.raises(ValueError):
        apply_env_overrides(_env_config(), max_policy_steps=3000, vehicles=5)
    substep = dict(_env_config(), cbf_substep_filtering=True)
    with pytest.raises(ValueError):
        apply_env_overrides(substep, max_policy_steps=3000, hz=100)
