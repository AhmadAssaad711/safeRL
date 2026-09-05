from __future__ import annotations

import numpy as np


from lane_free_env import LaneFreeTrafficEnv


def _patched_env(monkeypatch, collision_frames):
    env = LaneFreeTrafficEnv(
        config={
            "simulation_frequency": 20,
            "policy_frequency": 10,
            "dt": 0.05,
            "vehicles_count": 1,
            "neighbors_count": 0,
            "episode_steps": 100,
            "duration": 100,
            "terminate_on_collision": True,
        }
    )
    env.reset(seed=123)

    frame_iter = iter(collision_frames)

    monkeypatch.setattr(
        env,
        "_compute_accelerations",
        lambda: np.zeros((len(env.road.vehicles), 2), dtype=float),
    )
    monkeypatch.setattr(env, "_integrate", lambda _accelerations, _dt: None)

    def fake_detect_collisions():
        collision_count, active_count, ego_event_count, ego_active = next(frame_iter)
        env._last_collision_count = int(collision_count)
        env._last_active_collision_count = int(active_count)
        env._last_ego_collision_count = int(ego_event_count)
        env._last_ego_collision = bool(ego_active)

    monkeypatch.setattr(env, "_detect_collisions", fake_detect_collisions)
    return env


def test_persistent_collision_event_is_not_overwritten_by_final_physics_frame(
    monkeypatch,
):
    env = _patched_env(
        monkeypatch,
        [
            (1, 1, 1, True),
            (0, 1, 0, True),
        ],
    )
    try:
        _obs, _reward, terminated, truncated, info = env.step(np.zeros(2))
    finally:
        env.close()

    assert terminated
    assert not truncated
    assert info["collisions"] == 1
    assert info["active_collisions"] == 1
    assert info["ego_collision_events"] == 1
    assert info["ego_collision"] is True


def test_collision_that_disappears_before_policy_step_end_still_terminates(
    monkeypatch,
):
    env = _patched_env(
        monkeypatch,
        [
            (1, 1, 1, True),
            (0, 0, 0, False),
        ],
    )
    try:
        _obs, _reward, terminated, truncated, info = env.step(np.zeros(2))
    finally:
        env.close()

    assert terminated
    assert not truncated
    assert info["collisions"] == 1
    assert info["active_collisions"] == 1
    assert info["ego_collision_events"] == 1
    assert info["ego_collision"] is True
