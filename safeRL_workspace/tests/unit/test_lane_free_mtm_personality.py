from __future__ import annotations

import numpy as np


from lane_free_env import LaneFreeTrafficEnv, LaneFreeVehicle


def test_continuous_personality_interpolates_through_existing_anchors():
    env = LaneFreeTrafficEnv(config={"traffic_model": "mtm", "vehicles_count": 1})
    try:
        expected = {
            0.0: (0.28, 1.40, 0.50, 0.160, 0.900, 1.30),
            0.25: (0.34, 1.20, 0.35, 0.180, 0.950, 1.15),
            0.5: (0.40, 1.00, 0.20, 0.200, 1.000, 1.00),
            0.75: (0.50, 0.85, 0.10, 0.225, 1.075, 0.85),
            1.0: (0.60, 0.70, 0.00, 0.250, 1.150, 0.70),
        }
        keys = (
            "lambda",
            "tau",
            "p",
            "theta",
            "desired_speed_multiplier",
            "min_gap_multiplier",
        )
        for score, values in expected.items():
            personality = env._interpolate_mtm_personality(score)
            assert np.allclose([personality[key] for key in keys], values)
    finally:
        env.close()


def test_continuous_personalities_are_unique_deterministic_and_persistent():
    config = {
        "traffic_model": "mtm",
        "ego_controlled": False,
        "vehicles_count": 24,
        "road_length": 380.0,
        "show_trajectories": True,
        "mtm": {
            "continuous_driver_aggressiveness": True,
            "profile_probabilities": {
                "cautious": 0.25,
                "normal": 0.25,
                "aggressive": 0.50,
            },
        },
    }
    first = LaneFreeTrafficEnv(config=config)
    second = LaneFreeTrafficEnv(config=config)
    try:
        _, first_info = first.reset(seed=17)
        second.reset(seed=17)
        first_scores = np.asarray(
            [vehicle.driver_aggressiveness for vehicle in first.road.vehicles],
            dtype=float,
        )
        second_scores = np.asarray(
            [vehicle.driver_aggressiveness for vehicle in second.road.vehicles],
            dtype=float,
        )

        assert np.array_equal(first_scores, second_scores)
        assert np.all((0.0 <= first_scores) & (first_scores <= 1.0))
        assert np.unique(first_scores).size == len(first_scores)
        assert int(first_info["mtm_aggressiveness_unique"]) == len(first_scores)
        assert np.isclose(first_info["mtm_aggressiveness_mean"], first_scores.mean())

        initial_scores = first_scores.copy()
        first.step(np.zeros(2, dtype=np.float32))
        stepped_scores = np.asarray(
            [vehicle.driver_aggressiveness for vehicle in first.road.vehicles],
            dtype=float,
        )
        assert np.array_equal(stepped_scores, initial_scores)
        copied = LaneFreeVehicle.create_from(first.road.vehicles[3])
        assert copied.driver_aggressiveness == first.road.vehicles[3].driver_aggressiveness
    finally:
        first.close()
        second.close()


def test_categorical_mode_remains_available_for_legacy_reproduction():
    env = LaneFreeTrafficEnv(
        config={
            "traffic_model": "mtm",
            "ego_controlled": False,
            "vehicles_count": 4,
            "mtm": {
                "continuous_driver_aggressiveness": False,
                "profile_probabilities": {
                    "cautious": 0.0,
                    "normal": 0.0,
                    "aggressive": 1.0,
                },
            },
        }
    )
    try:
        env.reset(seed=9)
        for vehicle in env.road.vehicles:
            assert vehicle.driver_profile == "aggressive"
            assert vehicle.driver_aggressiveness is None
            params = env._mtm_params_for_vehicle(vehicle)
            assert params["lambda"] == 0.6
            assert params["tau"] == 0.7
            assert params["p"] == 0.0
            assert params["theta"] == 0.25
            assert params["min_gap"] == 1.4
    finally:
        env.close()
