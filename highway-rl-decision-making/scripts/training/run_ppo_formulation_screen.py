"""50k PPO screen for lane-free MDP formulations P0--P4.

The screen freezes the Q0 PPO optimizer and the congested/uncertain MTM
environment.  Only the observation, reward, and action interpretation described
by each formulation changes.  Training and evaluation both use the corrected
fixed-timestep, immediate-collision-reset protocol from the nominal PPO pilot.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
import pandas as pd

import scripts.training.run_cbf_filter_ablation as pipeline
import scripts.training.run_nominal_ppo_parameter_pilot as ppo_base
from scripts.common.laneless_script_config import active_traffic_model, env_config_from_args
from scripts.training.train_safety_potential_variants import MTM_CONGESTED_UNCERTAIN_UPDATES, deep_update


FORMULATION_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIR = Path(r"C:\agv_ppo_formulation_50k")
DEFAULT_TIMESTEPS = 50_000
DEFAULT_CHECKPOINT_INTERVAL = 10_000
DEFAULT_TRAINING_SEED = 307
DEFAULT_EVAL_SEEDS = tuple(range(900_000, 900_010))
DEFAULT_EVAL_TIMESTEPS = 800

ACCELERATION_SCALE = 3.0
JERK_SCALE = 6.0
JERK_LATENT_LIMIT = 8.0  # tanh(8) differs from one by less than 2.3e-7.
REFERENCE_GAINS = {"k_v": 1.0, "k_p": 1.0, "k_d": 2.0}

FORMULATIONS: dict[str, dict[str, Any]] = {
    "P0_current": {
        "label": "Current formulation",
        "observation": "current_nearest_42",
        "observation_dim": 42,
        "reward": "current_reciprocal",
        "action": "current_normalized_acceleration",
        "action_low": [-1.0, -1.0],
        "action_high": [1.0, 1.0],
        "ego_boundary_force": True,
    },
    "P1_reward": {
        "label": "Formulation-1 reward only",
        "observation": "current_nearest_42",
        "observation_dim": 42,
        "reward": "formulation_1",
        "action": "current_normalized_acceleration",
        "action_low": [-1.0, -1.0],
        "action_high": [1.0, 1.0],
        "ego_boundary_force": True,
        "partially_observed_previous_acceleration": True,
    },
    "P2_observed": {
        "label": "Formulation-1 reward plus explicit Markov observation",
        "observation": "semantic_42_plus_explicit_7",
        "observation_dim": 49,
        "reward": "formulation_1",
        "action": "current_normalized_acceleration",
        "action_low": [-1.0, -1.0],
        "action_high": [1.0, 1.0],
        "ego_boundary_force": True,
    },
    "P3_jerk": {
        "label": "Fully observed jerk-control formulation",
        "observation": "semantic_42_plus_explicit_7",
        "observation_dim": 49,
        "reward": "formulation_1_with_commanded_jerk_penalty",
        "action": "wide_gaussian_jerk_latent",
        "action_low": [-JERK_LATENT_LIMIT, -JERK_LATENT_LIMIT],
        "action_high": [JERK_LATENT_LIMIT, JERK_LATENT_LIMIT],
        "jerk_scale_mps3": JERK_SCALE,
        "ego_boundary_force": False,
    },
    "P4_reference": {
        "label": "Fully observed reference-command formulation",
        "observation": "semantic_42_plus_explicit_7",
        "observation_dim": 49,
        "reward": "formulation_1",
        "action": "speed_and_lateral_reference",
        "action_low": [-1.0, -1.0],
        "action_high": [1.0, 1.0],
        "reference_gains": REFERENCE_GAINS,
        "ego_boundary_force": False,
    },
}

FORMULATION_ORDER = tuple(FORMULATIONS)
Q0_PPO_PARAMETERS = copy.deepcopy(ppo_base.PPO_CONFIGS["Q0_current_aligned"])
PPO_CONFIGS = {
    formulation: copy.deepcopy(Q0_PPO_PARAMETERS) for formulation in FORMULATION_ORDER
}

OBSERVATION_SCHEMA = {
    "base_rows": ["ego", "front", "front_left", "front_right", "rear_left", "rear_right"],
    "row_features": [
        "signed_dx/sensing_range",
        "relative_y/road_width",
        "vx/24",
        "vy/7.2",
        "length/5.15",
        "width/1.84",
        "desired_speed/24",
    ],
    "explicit_features": [
        "target_speed/24",
        "2*target_y/road_width-1",
        "left_body_clearance/road_width",
        "right_body_clearance/road_width",
        "previous_executed_ax/3",
        "previous_executed_ay/3",
        "current_potential_field_cost",
    ],
    "semantic_slot_rule": {
        "sensing_cutoff": "abs(wrapped_dx) <= sensing_range",
        "front": "nearest Euclidean ahead vehicle with lateral body overlap; nearest ahead fallback",
        "front_left": "nearest remaining ahead vehicle with positive relative y",
        "front_right": "nearest remaining ahead vehicle with negative relative y",
        "rear_left": "nearest behind vehicle with nonnegative relative y",
        "rear_right": "nearest behind vehicle with negative relative y",
        "tie_break": "stable base vehicle index",
        "missing_slot": "seven zeros",
        "left_definition": "positive relative y",
    },
}

REWARD_SPEC = {
    "speed_tracking_weight": 0.45,
    "speed_tracking_scale": 0.20,
    "lateral_tracking_weight": 0.20,
    "lateral_tracking_scale": 0.25,
    "speed_progress_weight": 0.35,
    "potential_field_weight": -1.5,
    "boundary_cost_weight": -0.5,
    "acceleration_effort_weight": -0.03,
    "acceleration_delta_weight": -0.08,
    "jerk_command_weight": -0.05,
    "collision_penalty": -20.0,
    "overtake_bonus": 0.0,
    "boundary_cost": (
        "clip(sum_side(max(0, 1 - nonnegative_body_edge_clearance / "
        "ego_half_width)^2), 0, 1)"
    ),
}


_ORIGINAL_PPO_CONFIG_PAYLOAD = ppo_base.ppo_config_payload
_ACTIVE_FORMULATION: Optional[str] = None


def _finite_vector(value: Any, *, size: int = 2) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size < size or not np.all(np.isfinite(array[:size])):
        raise ValueError(f"Expected {size} finite values, got {value!r}")
    return array[:size].copy()


def _current_ego_acceleration(base: Any) -> np.ndarray:
    values = np.asarray(getattr(base, "_last_accelerations", np.empty((0, 2))), dtype=float)
    if values.ndim == 2 and values.shape[0] > 0 and np.all(np.isfinite(values[0, :2])):
        return values[0, :2].astype(np.float32)
    return np.zeros(2, dtype=np.float32)


def _policy_dt(base: Any) -> float:
    simulation_frequency = max(float(base.config.get("simulation_frequency", 4.0)), 1e-6)
    policy_frequency = max(float(base.config.get("policy_frequency", 4.0)), 1e-6)
    frames = max(1, int(round(simulation_frequency / policy_frequency)))
    return float(frames * float(base.config.get("dt", 1.0 / simulation_frequency)))


def _physical_to_base_action(base: Any, acceleration: np.ndarray) -> np.ndarray:
    acceleration = _finite_vector(acceleration)
    bounds = base.config["bounds"]
    lows = np.asarray([bounds["ax_min"], bounds["ay_min"]], dtype=np.float32)
    highs = np.asarray([bounds["ax_max"], bounds["ay_max"]], dtype=np.float32)
    clipped = np.clip(acceleration, lows, highs)
    result = np.zeros(2, dtype=np.float32)
    for index, physical in enumerate(clipped):
        if physical >= 0.0:
            result[index] = physical / max(float(highs[index]), 1e-6)
        else:
            result[index] = physical / max(abs(float(lows[index])), 1e-6)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def boundary_state(base: Any) -> tuple[float, float, float]:
    ego = base.vehicle
    road_width = max(float(base.config["road_width"]), 1e-6)
    half_width = max(0.5 * float(ego.width), 1e-6)
    left = max(float(ego.position[1]) - half_width, 0.0)
    right = max(road_width - half_width - float(ego.position[1]), 0.0)
    left_risk = max(0.0, 1.0 - left / half_width) ** 2
    right_risk = max(0.0, 1.0 - right / half_width) ** 2
    return left, right, float(np.clip(left_risk + right_risk, 0.0, 1.0))


def _semantic_observation(wrapper: Any) -> np.ndarray:
    base = wrapper.base_env
    ego = base.vehicle
    road_width = max(float(base.config["road_width"]), 1e-6)
    sensing_range = max(float(base.config["sensing_range"]), 1e-6)
    rows = np.zeros((6, 7), dtype=np.float32)
    rows[0] = base._observation_row(ego, ego)
    rows[0, 0] = np.clip(
        (float(ego.position[1]) - 0.5 * road_width) / (0.5 * road_width),
        -1.0,
        1.0,
    )
    rows[0, 1] = 0.0

    records: list[dict[str, Any]] = []
    for index, vehicle in enumerate(base.road.vehicles):
        if vehicle is ego:
            continue
        dx = float(base._signed_distance(ego.position[0], vehicle.position[0]))
        if abs(dx) > sensing_range:
            continue
        dy = float(vehicle.position[1] - ego.position[1])
        records.append(
            {
                "index": int(index),
                "vehicle": vehicle,
                "dx": dx,
                "dy": dy,
                "distance_sq": dx * dx + dy * dy,
                "overlap": abs(dy) <= 0.5 * (float(ego.width) + float(vehicle.width)),
            }
        )

    used: set[int] = set()

    def select(predicate: Callable[[dict[str, Any]], bool]) -> Any | None:
        candidates = [
            item for item in records if item["index"] not in used and predicate(item)
        ]
        if not candidates:
            return None
        chosen = min(candidates, key=lambda item: (item["distance_sq"], item["index"]))
        used.add(int(chosen["index"]))
        return chosen["vehicle"]

    front = select(lambda item: item["dx"] > 0.0 and bool(item["overlap"]))
    if front is None:
        front = select(lambda item: item["dx"] > 0.0)
    slots = [
        front,
        select(lambda item: item["dx"] > 0.0 and item["dy"] >= 0.0),
        select(lambda item: item["dx"] > 0.0 and item["dy"] < 0.0),
        select(lambda item: item["dx"] < 0.0 and item["dy"] >= 0.0),
        select(lambda item: item["dx"] < 0.0 and item["dy"] < 0.0),
    ]
    for row_index, vehicle in enumerate(slots, start=1):
        if vehicle is not None:
            rows[row_index] = base._observation_row(vehicle, ego)

    target_y, target_speed, _ = wrapper._lateral_target_and_speed()
    left, right, _ = boundary_state(base)
    previous_acceleration = _current_ego_acceleration(base)
    field_cost = float(np.clip(wrapper._potential_field_cost(), 0.0, 1.0))
    explicit = np.asarray(
        [
            float(target_speed) / 24.0,
            np.clip(2.0 * float(target_y) / road_width - 1.0, -1.0, 1.0),
            np.clip(left / road_width, 0.0, 1.0),
            np.clip(right / road_width, 0.0, 1.0),
            np.clip(previous_acceleration[0] / ACCELERATION_SCALE, -1.0, 1.0),
            np.clip(previous_acceleration[1] / ACCELERATION_SCALE, -1.0, 1.0),
            field_cost,
        ],
        dtype=np.float32,
    )
    return np.concatenate((rows.reshape(-1), explicit)).astype(np.float32)


def _formulation_components(
    wrapper: Any,
    *,
    previous_acceleration: np.ndarray,
    policy_action: np.ndarray,
    commanded_jerk: Optional[np.ndarray],
    integrator_clip_rate: float,
    reference_endpoint_rate: float,
    reference_speed: float,
    reference_y: float,
) -> dict[str, float]:
    base = wrapper.base_env
    ego = base.vehicle
    acceleration = _current_ego_acceleration(base)
    target_y, target_speed, zone_found = wrapper._lateral_target_and_speed()
    target_speed = max(float(target_speed), 1e-6)
    road_width = max(float(base.config["road_width"]), 1e-6)
    e_v = (float(ego.vx) - target_speed) / target_speed
    e_y = 2.0 * (float(ego.position[1]) - float(target_y)) / road_width
    field_cost = float(np.clip(wrapper._potential_field_cost(), 0.0, 1.0))
    left, right, boundary_cost = boundary_state(base)
    normalized_acceleration_sq = float(
        np.sum(np.square(acceleration / ACCELERATION_SCALE))
    )
    normalized_delta_sq = float(
        np.sum(
            np.square(
                (acceleration - previous_acceleration) / ACCELERATION_SCALE
            )
        )
    )
    speed_tracking = REWARD_SPEC["speed_tracking_weight"] * math.exp(
        -((e_v / REWARD_SPEC["speed_tracking_scale"]) ** 2)
    )
    lateral_tracking = REWARD_SPEC["lateral_tracking_weight"] * math.exp(
        -((e_y / REWARD_SPEC["lateral_tracking_scale"]) ** 2)
    )
    speed_progress = REWARD_SPEC["speed_progress_weight"] * float(
        np.clip(float(ego.vx) / target_speed, 0.0, 1.0)
    )
    collision_event = float(
        int(getattr(base, "_last_ego_collision_count", 0)) > 0
        or bool(getattr(base, "_last_ego_collision", False))
    )
    reward_without_delta = float(
        speed_tracking
        + lateral_tracking
        + speed_progress
        + REWARD_SPEC["potential_field_weight"] * field_cost
        + REWARD_SPEC["boundary_cost_weight"] * boundary_cost
        + REWARD_SPEC["acceleration_effort_weight"] * normalized_acceleration_sq
        + REWARD_SPEC["collision_penalty"] * collision_event
    )
    common_form1_reward = float(
        reward_without_delta
        + REWARD_SPEC["acceleration_delta_weight"] * normalized_delta_sq
    )
    if commanded_jerk is None:
        normalized_jerk_sq = float("nan")
        native_reward = common_form1_reward
    else:
        normalized_jerk_sq = float(
            np.sum(np.square(commanded_jerk / JERK_SCALE))
        )
        native_reward = float(
            reward_without_delta
            + REWARD_SPEC["jerk_command_weight"] * normalized_jerk_sq
        )
    bounds = base.config["bounds"]
    low = np.asarray([bounds["ax_min"], bounds["ay_min"]], dtype=float)
    high = np.asarray([bounds["ax_max"], bounds["ay_max"]], dtype=float)
    tolerance = 1e-3 * np.maximum(high - low, 1.0)
    controller_saturation = float(
        np.mean(
            (np.abs(acceleration - low) <= tolerance)
            | (np.abs(acceleration - high) <= tolerance)
        )
    )
    if commanded_jerk is None:
        command_saturation = float(np.mean(np.abs(policy_action) >= 1.0 - 1e-3))
    else:
        command_saturation = float(
            np.mean(np.abs(commanded_jerk / JERK_SCALE) >= 0.99)
        )
    return {
        "native_reward": native_reward,
        "common_form1_reward": common_form1_reward,
        "speed_tracking_reward": float(speed_tracking),
        "lateral_tracking_reward": float(lateral_tracking),
        "speed_progress_reward": float(speed_progress),
        "target_speed": target_speed,
        "target_y": float(target_y),
        "zone_found": float(zone_found),
        "e_v": float(e_v),
        "e_y": float(e_y),
        "cf": field_cost,
        "boundary_cost": boundary_cost,
        "d_left": left,
        "d_right": right,
        "normalized_acceleration_sq": normalized_acceleration_sq,
        "normalized_acceleration_delta_sq": normalized_delta_sq,
        "normalized_jerk_command_sq": normalized_jerk_sq,
        "collision_event": collision_event,
        "executed_ax": float(acceleration[0]),
        "executed_ay": float(acceleration[1]),
        "policy_command_saturation_rate": command_saturation,
        "controller_saturation_rate": controller_saturation,
        "acceleration_integrator_clip_rate": float(integrator_clip_rate),
        "reference_endpoint_rate": float(reference_endpoint_rate),
        "reference_speed": float(reference_speed),
        "reference_y": float(reference_y),
        "abs_target_speed_error": abs(float(ego.vx) - target_speed),
        "abs_target_lateral_error_m": abs(float(ego.position[1]) - float(target_y)),
    }


def _add_formulation_info(
    info: dict[str, Any], components: dict[str, float], formulation: str
) -> dict[str, Any]:
    result = dict(info)
    result["formulation_id"] = formulation
    result.update(
        {f"formulation_{key}": value for key, value in components.items()}
    )
    # Preserve the notebook's familiar diagnostic names for common plotting.
    result.update(
        {
            "karalakou_reward": float(components["native_reward"]),
            "karalakou_cf": float(components["cf"]),
            "karalakou_target_speed": float(components["target_speed"]),
            "karalakou_target_y": float(components["target_y"]),
            "karalakou_abs_target_speed_deviation": float(
                components["abs_target_speed_error"]
            ),
            "karalakou_lat_y_error_m": float(
                components["abs_target_lateral_error_m"]
            ),
            "karalakou_ego_collision": float(components["collision_event"]),
        }
    )
    return result


def make_formulation_wrapper_class(
    original_wrapper: type[gym.Wrapper], formulation: str
) -> type[gym.Wrapper]:
    spec = copy.deepcopy(FORMULATIONS[formulation])
    explicit_observation = spec["observation"] == "semantic_42_plus_explicit_7"

    class KaralakouRewardWrapper(original_wrapper):  # type: ignore[misc, valid-type]
        def __init__(self, env: gym.Env, reward_config: Optional[dict[str, float]] = None) -> None:
            super().__init__(env, reward_config=reward_config)
            self.formulation_id = formulation
            self.base_env.config["ego_boundary_force"] = bool(
                spec["ego_boundary_force"]
            )
            if formulation == "P3_jerk":
                self.action_space = gym.spaces.Box(
                    low=-JERK_LATENT_LIMIT,
                    high=JERK_LATENT_LIMIT,
                    shape=(2,),
                    dtype=np.float32,
                )
            if explicit_observation:
                self.observation_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(49,),
                    dtype=np.float32,
                )

        def reset(self, **kwargs):
            observation, info = super().reset(**kwargs)
            self.base_env.config["ego_boundary_force"] = bool(
                spec["ego_boundary_force"]
            )
            if explicit_observation:
                observation = _semantic_observation(self)
            return np.asarray(observation, dtype=np.float32), info

        def step(self, action):
            previous_acceleration = _current_ego_acceleration(self.base_env)
            policy_action = _finite_vector(action)
            commanded_jerk: Optional[np.ndarray] = None
            integrator_clip_rate = float("nan")
            reference_endpoint_rate = float("nan")
            reference_speed = float("nan")
            reference_y = float("nan")

            if formulation == "P0_current":
                observation, native_reward, terminated, truncated, info = super().step(
                    policy_action
                )
                components = _formulation_components(
                    self,
                    previous_acceleration=previous_acceleration,
                    policy_action=np.clip(policy_action, -1.0, 1.0),
                    commanded_jerk=None,
                    integrator_clip_rate=integrator_clip_rate,
                    reference_endpoint_rate=reference_endpoint_rate,
                    reference_speed=reference_speed,
                    reference_y=reference_y,
                )
                components["native_reward"] = float(native_reward)
                return (
                    np.asarray(observation, dtype=np.float32),
                    float(native_reward),
                    terminated,
                    truncated,
                    _add_formulation_info(info, components, formulation),
                )

            if formulation in {"P1_reward", "P2_observed"}:
                environment_action = np.clip(policy_action, -1.0, 1.0).astype(
                    np.float32
                )
            elif formulation == "P3_jerk":
                latent = np.clip(
                    policy_action, -JERK_LATENT_LIMIT, JERK_LATENT_LIMIT
                )
                commanded_jerk = (JERK_SCALE * np.tanh(latent)).astype(np.float32)
                unconstrained = previous_acceleration + _policy_dt(
                    self.base_env
                ) * commanded_jerk
                bounds = self.base_env.config["bounds"]
                low = np.asarray([bounds["ax_min"], bounds["ay_min"]], dtype=float)
                high = np.asarray([bounds["ax_max"], bounds["ay_max"]], dtype=float)
                desired_acceleration = np.clip(unconstrained, low, high).astype(
                    np.float32
                )
                integrator_clip_rate = float(
                    np.mean(np.abs(unconstrained - desired_acceleration) > 1e-7)
                )
                environment_action = _physical_to_base_action(
                    self.base_env, desired_acceleration
                )
            elif formulation == "P4_reference":
                bounded = np.clip(policy_action, -1.0, 1.0)
                ego = self.base_env.vehicle
                road_width = float(self.base_env.config["road_width"])
                y_min = 0.5 * float(ego.width)
                y_max = road_width - 0.5 * float(ego.width)
                reference_speed = float(12.0 * (float(bounded[0]) + 1.0))
                reference_y = float(
                    y_min + 0.5 * (float(bounded[1]) + 1.0) * (y_max - y_min)
                )
                desired_acceleration = np.asarray(
                    [
                        REFERENCE_GAINS["k_v"]
                        * (reference_speed - float(ego.vx)),
                        REFERENCE_GAINS["k_p"]
                        * (reference_y - float(ego.position[1]))
                        - REFERENCE_GAINS["k_d"] * float(ego.vy),
                    ],
                    dtype=np.float32,
                )
                bounds = self.base_env.config["bounds"]
                low = np.asarray([bounds["ax_min"], bounds["ay_min"]], dtype=float)
                high = np.asarray([bounds["ax_max"], bounds["ay_max"]], dtype=float)
                desired_acceleration = np.clip(
                    desired_acceleration, low, high
                ).astype(np.float32)
                reference_endpoint_rate = float(np.mean(np.abs(bounded) >= 0.99))
                environment_action = _physical_to_base_action(
                    self.base_env, desired_acceleration
                )
            else:  # pragma: no cover - registry validation prevents this path.
                raise RuntimeError(f"Unknown PPO formulation: {formulation}")

            base_observation, _, terminated, truncated, info = self.env.step(
                environment_action
            )
            components = _formulation_components(
                self,
                previous_acceleration=previous_acceleration,
                policy_action=(
                    np.tanh(policy_action)
                    if formulation == "P3_jerk"
                    else np.clip(policy_action, -1.0, 1.0)
                ),
                commanded_jerk=commanded_jerk,
                integrator_clip_rate=integrator_clip_rate,
                reference_endpoint_rate=reference_endpoint_rate,
                reference_speed=reference_speed,
                reference_y=reference_y,
            )
            native_reward = float(components["native_reward"])
            observation = (
                _semantic_observation(self)
                if explicit_observation
                else self._augment_observation(base_observation)
            )
            return (
                np.asarray(observation, dtype=np.float32),
                native_reward,
                terminated,
                truncated,
                _add_formulation_info(info, components, formulation),
            )

    # Strict checkpointing recognizes the historical wrapper name as stateless.
    # All apparent memory (previous acceleration) lives in the serialized base env.
    KaralakouRewardWrapper.__name__ = "KaralakouRewardWrapper"
    KaralakouRewardWrapper.__qualname__ = "KaralakouRewardWrapper"
    KaralakouRewardWrapper.__module__ = __name__
    return KaralakouRewardWrapper


def formulation_evaluation_action(*, model: Any, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if model is None:
        low = np.asarray([-1.0, -1.0], dtype=np.float32)
        high = np.asarray([1.0, 1.0], dtype=np.float32)
        return rng.uniform(low=low, high=high).astype(np.float32)
    model_obs = pipeline._model_observation(model, obs)
    action, _ = model.predict(model_obs, deterministic=True)
    return np.asarray(action, dtype=np.float32).reshape(-1)[:2]


def validate_formulation_action_space(model_or_env: Any) -> None:
    if _ACTIVE_FORMULATION not in FORMULATIONS:
        raise RuntimeError("No active PPO formulation is selected")
    action_space = model_or_env.action_space
    expected = FORMULATIONS[str(_ACTIVE_FORMULATION)]
    if tuple(action_space.shape) != (2,):
        raise RuntimeError(f"Expected two formulation actions, got {action_space}")
    if not np.allclose(action_space.low, expected["action_low"]) or not np.allclose(
        action_space.high, expected["action_high"]
    ):
        raise RuntimeError(
            f"{_ACTIVE_FORMULATION} action space disagrees with its formulation: {action_space}"
        )


def formulation_config_payload(**kwargs: Any) -> dict[str, Any]:
    payload = _ORIGINAL_PPO_CONFIG_PAYLOAD(**kwargs)
    formulation = str(kwargs["pilot_config"])
    project_root = Path(kwargs["project_root"])
    payload["schema_version"] = FORMULATION_SCHEMA_VERSION
    payload["study"] = "ppo_mdp_formulation_screen_50k"
    payload["formulation"] = copy.deepcopy(FORMULATIONS[formulation])
    payload["observation_schema"] = copy.deepcopy(OBSERVATION_SCHEMA)
    payload["reward_spec"] = copy.deepcopy(REWARD_SPEC)
    payload["fixed_training"]["action_space"] = {
        "low": FORMULATIONS[formulation]["action_low"],
        "high": FORMULATIONS[formulation]["action_high"],
    }
    payload["fixed_training"]["cbf_active"] = False
    payload["source_hashes"]["ppo_formulation_screen"] = pipeline.file_sha256(
        Path(__file__).resolve()
    )
    payload["source_hashes"]["lane_free_environment"] = pipeline.file_sha256(
        project_root / "laneless highway env" / "lane_free_env.py"
    )
    return payload


def install_base_runner_hooks() -> None:
    ppo_base.PPO_CONFIGS = copy.deepcopy(PPO_CONFIGS)
    ppo_base.ppo_config_payload = formulation_config_payload
    ppo_base.validate_normalized_action_space = validate_formulation_action_space
    if "common_form1_total_return" not in ppo_base.FINAL_WINDOW_SUM_METRICS:
        ppo_base.FINAL_WINDOW_SUM_METRICS = (
            *ppo_base.FINAL_WINDOW_SUM_METRICS,
            "common_form1_total_return",
        )


def make_formulation_namespace(
    base_namespace: dict[str, Any], formulation: str
) -> dict[str, Any]:
    namespace = dict(base_namespace)
    namespace["KaralakouRewardWrapper"] = make_formulation_wrapper_class(
        base_namespace["KaralakouRewardWrapper"], formulation
    )
    namespace["ppo_formulation_evaluation_action"] = formulation_evaluation_action
    # The legacy optional normalizer assumes four-feature groups and must not
    # reinterpret the 42/49D formulation states.
    namespace["NORMALIZE_RL_OBSERVATIONS"] = False
    return namespace


def rank_formulations(across_seed: pd.DataFrame) -> pd.DataFrame:
    ranked = across_seed.copy()
    criteria = {
        "ego_collisions_per_km_seed_mean": True,
        "distance_per_collision_exposure_bound_m_seed_mean": False,
        "common_form1_return_per_timestep_seed_mean": False,
        "executed_action_saturation_rate_seed_mean": True,
        "mean_jerk_norm_seed_mean": True,
        "mean_abs_target_speed_error_seed_mean": True,
    }
    for metric, ascending in criteria.items():
        if metric not in ranked:
            raise RuntimeError(f"Formulation ranking metric is missing: {metric}")
        rank_name = f"rank_{metric.removesuffix('_seed_mean')}"
        ranked[rank_name] = pd.to_numeric(ranked[metric], errors="coerce").rank(
            method="min", ascending=ascending, na_option="bottom"
        )
    ranked["control_mean_rank"] = ranked[
        ["rank_executed_action_saturation_rate", "rank_mean_jerk_norm"]
    ].mean(axis=1)
    ranked["priority_weighted_rank"] = (
        4.0 * ranked["rank_ego_collisions_per_km"]
        + 4.0 * ranked["rank_distance_per_collision_exposure_bound_m"]
        + 2.0 * ranked["rank_common_form1_return_per_timestep"]
        + ranked["control_mean_rank"]
        + ranked["rank_mean_abs_target_speed_error"]
    ) / 12.0
    ranked = ranked.sort_values(
        [
            "priority_weighted_rank",
            "rank_ego_collisions_per_km",
            "rank_distance_per_collision_exposure_bound_m",
        ]
    ).reset_index(drop=True)
    ranked.insert(0, "overall_rank", np.arange(1, len(ranked) + 1))
    return ranked


def write_formulation_summaries(
    *, output_dir: Path, scenarios: pd.DataFrame, diagnostics: pd.DataFrame
) -> pd.DataFrame:
    checkpoint = ppo_base.build_checkpoint_seed_summary(scenarios, diagnostics)
    checkpoint.to_csv(output_dir / "checkpoint_seed_summary.csv", index=False)
    seed_averages = ppo_base.final_three_seed_averages(checkpoint)
    seed_averages.to_csv(output_dir / "final_three_seed_averages.csv", index=False)
    across_seed = ppo_base.across_seed_final_three(seed_averages)
    across_seed.to_csv(output_dir / "final_three_across_seeds.csv", index=False)
    ranking = rank_formulations(across_seed)
    ranking.to_csv(output_dir / "ranking_final_three.csv", index=False)
    selection = {
        "selection_rule": (
            "Final 30k/40k/50k checkpoints; scenario aggregation within training seed; "
            "safety ranks weighted first, then common Formulation-1 score, physical control "
            "burden, and target-speed error."
        ),
        "native_return_warning": (
            "P0, P1/P2/P4, and P3 use different native reward definitions; native "
            "return_per_timestep is reported but not used for cross-formulation ranking."
        ),
        "single_seed_warning": (
            "This 50k screening run has one training seed; across-seed variance is undefined."
        ),
        "selected_configs": ranking["pilot_config"].head(2).tolist(),
    }
    (output_dir / "selected_best_two.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    return ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the exact 50k PPO P0--P4 MDP-formulation screen."
    )
    parser.add_argument("--stage", choices=("screen", "summarize"), default="screen")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument(
        "--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_TRAINING_SEED])
    parser.add_argument("--configs", nargs="+", default=list(FORMULATION_ORDER))
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=list(DEFAULT_EVAL_SEEDS)
    )
    parser.add_argument("--eval-timesteps", type=int, default=DEFAULT_EVAL_TIMESTEPS)
    parser.add_argument("--strict-checkpoint-retention", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-envs", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--eval-episodes", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--eval-horizon", type=int, default=DEFAULT_EVAL_TIMESTEPS, help=argparse.SUPPRESS
    )
    parser.add_argument("--k0", type=float, default=5.29, help=argparse.SUPPRESS)
    parser.add_argument("--k1", type=float, default=3.68, help=argparse.SUPPRESS)
    parser.add_argument("--eps-side", type=float, default=0.10, help=argparse.SUPPRESS)
    parser.add_argument(
        "--correction-epsilon", type=float, default=0.03, help=argparse.SUPPRESS
    )
    parser.set_defaults(
        traffic_model="mtm", env_config_json=None, env_config_file=None
    )
    return parser.parse_args()


def main() -> int:
    global _ACTIVE_FORMULATION

    args = parse_args()
    project_root = (
        Path(args.project_root).resolve()
        if args.project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    output_dir = Path(args.output_dir).resolve()
    selected = [str(value) for value in args.configs]
    unknown = sorted(set(selected) - set(FORMULATIONS))
    if unknown:
        raise ValueError(f"Unknown formulations: {unknown}")
    if len(selected) != len(set(selected)):
        raise ValueError("Formulation list must not contain duplicates")
    if int(args.n_envs) != 1:
        raise ValueError("The strict formulation screen requires one environment")
    if not args.eval_seeds or len(args.eval_seeds) != len(set(args.eval_seeds)):
        raise ValueError("Evaluation seeds must be non-empty and unique")
    args.eval_seeds = [int(seed) for seed in args.eval_seeds]
    args.eval_scenarios = len(args.eval_seeds)
    args.eval_episodes = len(args.eval_seeds)
    args.eval_horizon = int(args.eval_timesteps)

    install_base_runner_hooks()
    namespace = pipeline.bootstrap_notebook_namespace(project_root)
    pipeline.exec_required_notebook_cells(
        project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    namespace["DEVICE"] = args.device
    args.cbf_snapshot = ppo_base.pilot_common.fixed_cbf_snapshot(namespace)
    args.k0 = float(args.cbf_snapshot["k0"])
    args.k1 = float(args.cbf_snapshot["k1"])
    args.eps_side = float(args.cbf_snapshot["eps_side"])
    env_config = env_config_from_args(args, namespace["ENV_CONFIG"])
    if active_traffic_model(env_config) == "mtm":
        deep_update(env_config, copy.deepcopy(MTM_CONGESTED_UNCERTAIN_UPDATES))
    if not bool(env_config.get("terminate_on_collision", False)):
        raise RuntimeError("Formulation screen requires terminate_on_collision=True")
    env_config["ego_boundary_force"] = True
    reward_config = pipeline.make_base_reward_config(namespace)
    run_specs = [
        (int(training_seed), formulation)
        for training_seed in args.seeds
        for formulation in selected
    ]
    for _, formulation in run_specs:
        ppo_base.validate_rollout_alignment(
            PPO_CONFIGS[formulation],
            n_envs=1,
            target_timesteps=int(args.timesteps),
            checkpoint_interval=int(args.checkpoint_interval),
        )

    if args.stage == "screen":
        ppo_base.preflight_runs(
            output_dir=output_dir,
            run_specs=run_specs,
            args=args,
            project_root=project_root,
            env_config=env_config,
            reward_config=reward_config,
            target_timesteps=int(args.timesteps),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        root_config = {
            "schema_version": FORMULATION_SCHEMA_VERSION,
            "study": "ppo_mdp_formulation_screen_50k",
            "selected_formulations": selected,
            "training_seeds": [int(seed) for seed in args.seeds],
            "target_timesteps": int(args.timesteps),
            "checkpoint_interval": int(args.checkpoint_interval),
            "eval_seeds": args.eval_seeds,
            "eval_timesteps": int(args.eval_timesteps),
            "ppo_parameters_frozen_to_q0": Q0_PPO_PARAMETERS,
            "formulations": {name: FORMULATIONS[name] for name in selected},
            "observation_schema": OBSERVATION_SCHEMA,
            "reward_spec": REWARD_SPEC,
            "environment_parameters_changed": False,
            "ego_boundary_force_is_formulation_action_semantics": True,
            "cbf_active": False,
            "fixed_cbf_snapshot": args.cbf_snapshot,
            "env_config": env_config,
            "base_reward_config": reward_config,
        }
        (output_dir / "run_config.json").write_text(
            json.dumps(root_config, indent=2, default=str), encoding="utf-8"
        )
        print(
            "[ppo-formulation] starting"
            f" configs={selected} seed={args.seeds} timesteps={int(args.timesteps):,}"
            f" eval_every={int(args.checkpoint_interval):,}"
            f" eval={len(args.eval_seeds)}x{int(args.eval_timesteps)}",
            flush=True,
        )
        model_rows: list[dict[str, Any]] = []
        for training_seed, formulation in run_specs:
            _ACTIVE_FORMULATION = formulation
            formulation_namespace = make_formulation_namespace(namespace, formulation)
            print(
                f"[ppo-formulation] train {formulation} seed={training_seed}"
                f" obs={FORMULATIONS[formulation]['observation_dim']}"
                f" action={FORMULATIONS[formulation]['action']}",
                flush=True,
            )
            model_rows.append(
                ppo_base.train_one_run(
                    formulation_namespace,
                    pilot_config=formulation,
                    training_seed=training_seed,
                    target_timesteps=int(args.timesteps),
                    args=args,
                    env_config=env_config,
                    reward_config=reward_config,
                    output_dir=output_dir,
                )
            )
        pd.DataFrame(model_rows).to_csv(output_dir / "model_manifest.csv", index=False)

    scenarios, diagnostics = ppo_base.collect_run_outputs(
        output_dir,
        run_specs,
        target_timesteps=int(args.timesteps),
        checkpoint_interval=int(args.checkpoint_interval),
        eval_seeds=args.eval_seeds,
        eval_timesteps=int(args.eval_timesteps),
    )
    ranking = write_formulation_summaries(
        output_dir=output_dir, scenarios=scenarios, diagnostics=diagnostics
    )
    display_columns = [
        "overall_rank",
        "pilot_config",
        "ego_collisions_per_km_seed_mean",
        "distance_per_collision_m_seed_mean",
        "common_form1_return_per_timestep_seed_mean",
        "return_per_timestep_seed_mean",
        "mean_abs_target_speed_error_seed_mean",
        "executed_action_saturation_rate_seed_mean",
        "mean_jerk_norm_seed_mean",
    ]
    print(ranking[display_columns].to_string(index=False), flush=True)
    print(f"[ppo-formulation] complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
