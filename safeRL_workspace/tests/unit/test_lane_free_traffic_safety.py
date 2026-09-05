from __future__ import annotations

import numpy as np


from lane_free_env import LaneFreeTrafficEnv


def _dense_env() -> LaneFreeTrafficEnv:
    return LaneFreeTrafficEnv(
        config={
            "road_length": 380.0,
            "road_width": 10.2,
            "dt": 0.05,
            "simulation_frequency": 20,
            "policy_frequency": 10,
            "vehicles_count": 55,
            "desired_speed_range": [15.0, 25.0],
            "initial_speed_fraction_range": [0.55, 1.10],
            "bounds": {
                "ax_min": -3.0,
                "ax_max": 3.0,
                "ay_min": -3.0,
                "ay_max": 3.0,
            },
        }
    )


def test_safe_spawn_has_no_overlapping_pairs_across_dense_seeds():
    for seed in range(12):
        env = _dense_env()
        try:
            _obs, info = env.reset(seed=seed)
            env._detect_collisions()
            assert env._last_active_collision_count == 0
            assert env._last_ego_collision_count == 0
            assert info["traffic_safe_spawn"] is True
        finally:
            env.close()


def test_guard_brakes_social_follower_without_changing_ego_action():
    env = LaneFreeTrafficEnv(
        config={
            "road_length": 380.0,
            "vehicles_count": 2,
            "bounds": {
                "ax_min": -3.0,
                "ax_max": 3.0,
                "ay_min": -3.0,
                "ay_max": 3.0,
            },
        }
    )
    try:
        env.reset(seed=1)
        ego, social = env.road.vehicles
        ego.position[:] = [100.0, 5.1]
        ego.vx, ego.vy = 15.0, 0.0
        social.position[:] = [90.0, 5.1]
        social.vx, social.vy = 25.0, 0.0
        requested = np.asarray([[1.2, -0.5], [2.0, 0.0]], dtype=float)

        guarded = env._apply_traffic_safety_guard(requested, dt=0.05)

        assert np.allclose(guarded[0], requested[0])
        assert guarded[1, 0] == -3.0
        assert env._last_traffic_safety_diagnostics["traffic_brakes"] == 1.0
    finally:
        env.close()


def test_guard_brakes_a_social_follower_for_social_traffic():
    env = LaneFreeTrafficEnv(
        config={
            "road_length": 380.0,
            "vehicles_count": 3,
            "bounds": {
                "ax_min": -3.0,
                "ax_max": 3.0,
                "ay_min": -3.0,
                "ay_max": 3.0,
            },
        }
    )
    try:
        env.reset(seed=3)
        ego, follower, leader = env.road.vehicles
        ego.position[:] = [250.0, 1.0]
        follower.position[:] = [90.0, 5.1]
        leader.position[:] = [100.0, 5.1]
        follower.vx, follower.vy = 25.0, 0.0
        leader.vx, leader.vy = 15.0, 0.0
        requested = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]], dtype=float)

        guarded = env._apply_traffic_safety_guard(requested, dt=0.05)

        assert np.allclose(guarded[0], requested[0])
        assert guarded[1, 0] == -3.0
        assert env._last_traffic_safety_diagnostics["traffic_brakes"] == 1.0
    finally:
        env.close()


def test_guard_makes_social_leader_yield_instead_of_overwriting_ego_action():
    env = LaneFreeTrafficEnv(
        config={
            "road_length": 380.0,
            "vehicles_count": 2,
            "bounds": {
                "ax_min": -3.0,
                "ax_max": 3.0,
                "ay_min": -3.0,
                "ay_max": 3.0,
            },
        }
    )
    try:
        env.reset(seed=2)
        ego, social = env.road.vehicles
        ego.position[:] = [90.0, 5.1]
        ego.vx, ego.vy = 25.0, 0.0
        social.position[:] = [100.0, 5.1]
        social.vx, social.vy = 15.0, 0.0
        requested = np.asarray([[-2.0, 0.7], [-3.0, 0.0]], dtype=float)

        guarded = env._apply_traffic_safety_guard(requested, dt=0.05)

        assert np.allclose(guarded[0], requested[0])
        assert guarded[1, 0] > requested[1, 0]
        assert env._last_traffic_safety_diagnostics["ego_leader_yields"] == 1.0
    finally:
        env.close()


def test_side_contact_projection_preserves_common_mode_lateral_acceleration():
    env = LaneFreeTrafficEnv(
        config={
            "road_length": 380.0,
            "vehicles_count": 3,
            "bounds": {
                "ax_min": -3.0,
                "ax_max": 3.0,
                "ay_min": -3.0,
                "ay_max": 3.0,
            },
        }
    )
    try:
        env.reset(seed=4)
        ego, lower, upper = env.road.vehicles
        ego.position[:] = [250.0, 1.0]
        lower.position[:] = [100.0, 4.0]
        upper.position[:] = [101.0, 6.5]
        lower.vx = upper.vx = 15.0
        lower.vy, upper.vy = 1.0, -1.0
        requested = np.zeros((3, 2), dtype=float)

        guarded = env._apply_traffic_safety_guard(requested, dt=0.05)

        assert guarded[1, 1] < 0.0
        assert guarded[2, 1] > 0.0
        assert np.isclose(guarded[1, 1] + guarded[2, 1], 0.0)
        assert env._last_traffic_safety_diagnostics["side_constraints"] == 1.0
    finally:
        env.close()


def test_side_contact_projection_does_not_block_nonoverlapping_cut_in():
    env = LaneFreeTrafficEnv(
        config={
            "road_length": 380.0,
            "vehicles_count": 3,
            "bounds": {
                "ax_min": -3.0,
                "ax_max": 3.0,
                "ay_min": -3.0,
                "ay_max": 3.0,
            },
        }
    )
    try:
        env.reset(seed=5)
        ego, lower, upper = env.road.vehicles
        ego.position[:] = [250.0, 1.0]
        lower.position[:] = [100.0, 4.0]
        upper.position[:] = [130.0, 6.5]
        lower.vx = upper.vx = 15.0
        lower.vy, upper.vy = 1.0, -1.0
        requested = np.asarray(
            [[0.0, 0.0], [0.0, 0.4], [0.0, -0.2]], dtype=float
        )

        guarded = env._apply_traffic_safety_guard(requested, dt=0.05)

        assert np.allclose(guarded[:, 1], requested[:, 1])
        assert env._last_traffic_safety_diagnostics["side_constraints"] == 0.0
    finally:
        env.close()


def test_side_contact_projection_never_overwrites_controlled_ego_action():
    env = LaneFreeTrafficEnv(
        config={
            "road_length": 380.0,
            "vehicles_count": 2,
            "bounds": {
                "ax_min": -3.0,
                "ax_max": 3.0,
                "ay_min": -3.0,
                "ay_max": 3.0,
            },
        }
    )
    try:
        env.reset(seed=6)
        ego, social = env.road.vehicles
        ego.position[:] = [100.0, 4.0]
        social.position[:] = [101.0, 6.5]
        ego.vx = social.vx = 15.0
        ego.vy, social.vy = 1.0, -1.0
        requested = np.asarray([[0.4, 0.7], [0.0, 0.0]], dtype=float)

        guarded = env._apply_traffic_safety_guard(requested, dt=0.05)

        assert np.allclose(guarded[0], requested[0])
        assert guarded[1, 1] > requested[1, 1]
        assert env._last_traffic_safety_diagnostics["side_constraints"] == 1.0
    finally:
        env.close()
