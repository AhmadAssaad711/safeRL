from __future__ import annotations

import numpy as np

from lane_free_env import LaneFreeTrafficEnv, resolve_frequency_plan


def test_requested_frequency_plan_is_100_20_20():
    plan = resolve_frequency_plan(
        {
            "dt": 0.01,
            "simulation_frequency": 100,
            "policy_frequency": 20,
            "cbf_frequency": 20,
            "cbf_substep_filtering": False,
        }
    )

    assert plan["physics_frequency_hz"] == 100.0
    assert plan["policy_frequency_hz"] == 20.0
    assert plan["cbf_frequency_hz"] == 20.0
    assert plan["physics_frames_per_policy_action"] == 5
    assert plan["physics_frames_per_cbf_update"] == 5
    assert plan["cbf_evaluations_per_policy_action"] == 1.0
    assert plan["cbf_schedule"] == "policy"


def test_policy_rate_callback_is_rejected_as_ambiguous():
    try:
        resolve_frequency_plan(
            {
                "dt": 0.01,
                "simulation_frequency": 100,
                "policy_frequency": 20,
                "cbf_frequency": 20,
                "cbf_substep_filtering": True,
            }
        )
    except ValueError as exc:
        assert "cbf_frequency" in str(exc)
    else:  # pragma: no cover - the assertion above is the test contract
        raise AssertionError("physics-rate callback mode accepted a 20 Hz CBF")


def test_info_reports_frequencies_and_policy_step_frames():
    env = LaneFreeTrafficEnv(
        config={
            "dt": 0.01,
            "simulation_frequency": 100,
            "policy_frequency": 20,
            "cbf_frequency": 20,
            "cbf_substep_filtering": False,
            "vehicles_count": 1,
            "neighbors_count": 0,
            "ego_boundary_force": False,
        }
    )
    try:
        _observation, reset_info = env.reset(seed=307)
        _observation, _reward, _terminated, _truncated, info = env.step(
            np.zeros(2, dtype=np.float32)
        )
    finally:
        env.close()

    for report in (reset_info, info):
        assert report["physics_frequency_hz"] == 100.0
        assert report["policy_frequency_hz"] == 20.0
        assert report["cbf_frequency_hz"] == 20.0
        assert report["physics_frames_per_policy_action"] == 5
        assert report["physics_frames_per_cbf_update"] == 5
        assert report["cbf_schedule"] == "policy"
    assert info["physics_frames_per_policy_action"] == 5
