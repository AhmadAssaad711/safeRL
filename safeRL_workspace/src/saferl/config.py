"""Central configuration contracts for new laneless research code.

This module deliberately does not replace the frozen notebook contract or
migrate ``scripts/common``. It gives new experiments one explicit source for
shared constants and derived invariants while historical workflows remain
unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


# Reference dimensions for the CBF footprint model.  The minimum-area ellipse
# enclosing an axis-aligned rectangle has semi-axes equal to each rectangle
# side divided by sqrt(2).
REFERENCE_VEHICLE_LENGTH_M = 3.6
REFERENCE_VEHICLE_WIDTH_M = 1.8
MINIMUM_ENCLOSING_ELLIPSE_A_M = REFERENCE_VEHICLE_LENGTH_M / math.sqrt(2.0)
MINIMUM_ENCLOSING_ELLIPSE_B_M = REFERENCE_VEHICLE_WIDTH_M / math.sqrt(2.0)


@dataclass(frozen=True)
class FrequencyConfig:
    """Simulation timing for the canonical laneless experiment."""

    physics_hz: int = 100
    policy_hz: int = 20
    cbf_hz: int = 20

    def __post_init__(self) -> None:
        values = {
            "physics_hz": self.physics_hz,
            "policy_hz": self.policy_hz,
            "cbf_hz": self.cbf_hz,
        }
        for name, value in values.items():
            if int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.physics_hz % self.policy_hz != 0:
            raise ValueError("physics_hz must be divisible by policy_hz")
        if self.physics_hz % self.cbf_hz != 0:
            raise ValueError("physics_hz must be divisible by cbf_hz")

    @property
    def dt_s(self) -> float:
        return 1.0 / float(self.physics_hz)

    @property
    def physics_frames_per_policy_action(self) -> int:
        return self.physics_hz // self.policy_hz

    @property
    def physics_frames_per_cbf_update(self) -> int:
        return self.physics_hz // self.cbf_hz


@dataclass(frozen=True)
class ActionConfig:
    """Normalized policy and physical acceleration bounds."""

    normalized_low: float = -1.0
    normalized_high: float = 1.0
    physical_low: tuple[float, float] = (-3.0, -3.0)
    physical_high: tuple[float, float] = (3.0, 3.0)

    def __post_init__(self) -> None:
        if len(self.physical_low) != 2 or len(self.physical_high) != 2:
            raise ValueError("physical action bounds must have two components")
        for low, high in zip(self.physical_low, self.physical_high):
            if float(low) >= float(high):
                raise ValueError("each physical action lower bound must be below its upper bound")


@dataclass(frozen=True)
class EnvironmentConfig:
    """Resolved environment values shared by new research experiments."""

    environment_id: str = "lane-free-v0"
    road_length_m: float = 380.0
    road_width_m: float = 10.2
    vehicles_count: int = 55
    neighbors_count: int = 5
    sensing_range_m: float = 90.0
    episode_steps: int = 30_000
    traffic_model: str = "mtm"
    ego_desired_speed_mps: float = 16.0
    desired_speed_range_mps: tuple[float, float] = (15.0, 25.0)
    initial_speed_fraction_range: tuple[float, float] = (0.55, 1.10)
    terminate_on_collision: bool = True
    ego_boundary_force: bool = False
    cbf_substep_filtering: bool = False
    include_vehicle_dimensions: bool = False
    append_previous_executed_action: bool = True
    safe_spawn: bool = True
    spawn_cbf_safe_set: bool = True
    mtm_leader_range_m: float = 90.0
    mtm_profile_probabilities: tuple[tuple[str, float], ...] = (
        ("normal", 0.25),
        ("aggressive", 0.50),
        ("cautious", 0.25),
    )

    def __post_init__(self) -> None:
        if self.environment_id != "lane-free-v0":
            raise ValueError("new laneless research code must use lane-free-v0")
        if self.vehicles_count < 1 or self.neighbors_count < 0:
            raise ValueError("vehicle and neighbor counts are invalid")
        if self.neighbors_count >= self.vehicles_count:
            raise ValueError("neighbors_count must be smaller than vehicles_count")
        if self.road_length_m <= 0.0 or self.road_width_m <= 0.0:
            raise ValueError("road dimensions must be positive")


@dataclass(frozen=True)
class ObservationConfig:
    """The canonical PPO observation definition."""

    vehicle_row_features: tuple[str, ...] = ("dx", "dy", "vx", "vy", "v_desired")
    neighbors_count: int = 5
    append_previous_executed_action: bool = True
    expose_target_y: bool = True
    include_vehicle_dimensions: bool = False

    @property
    def vehicle_rows(self) -> int:
        return 1 + self.neighbors_count

    @property
    def base_dimension(self) -> int:
        return self.vehicle_rows * len(self.vehicle_row_features)

    @property
    def previous_action_dimension(self) -> int:
        return 2 if self.append_previous_executed_action else 0

    @property
    def expected_dimension(self) -> int:
        return self.base_dimension + self.previous_action_dimension

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "vehicle_row_features": list(self.vehicle_row_features),
            "vehicle_rows": self.vehicle_rows,
            "base_dimension": self.base_dimension,
            "append_previous_executed_action": self.append_previous_executed_action,
            "previous_action_dimension": self.previous_action_dimension,
            "expose_target_y": self.expose_target_y,
            "include_vehicle_dimensions": self.include_vehicle_dimensions,
            "expected_dimension": self.expected_dimension,
        }


@dataclass(frozen=True)
class RewardConfig:
    """Canonical Karalakou reward constants for new-compatible experiments."""

    epsilon_r: float = 0.4
    wx: float = 1.0
    wy: float = 0.65
    wf: float = 1.0
    way: float = 0.0
    collision_penalty: float = -2.5
    overtake_bonus: float = 0.5
    jerk_penalty_weight: float = 0.02
    jerk_scale: float = 10.0
    timegap: float = 1.5
    field_magnitude: float = 0.5
    field_px: float = 2.0
    field_py: float = 2.0
    field_pt: float = 1.0
    zone_margin: float = 0.15
    progress_reward_weight: float = 0.05
    progress_clip: float = 1.25
    progress_distance_scale_m: float = 2.0
    expose_target_y: bool = True


@dataclass(frozen=True)
class SafetyConfig:
    """Fixed HOCBF and CBF-footprint safety constants."""

    eps_side: float = 0.10
    k0: float = 5.29
    k1: float = 3.68
    psi1_gain: float = 2.30
    relative_ellipse_a_m: float = MINIMUM_ENCLOSING_ELLIPSE_A_M
    relative_ellipse_b_m: float = MINIMUM_ENCLOSING_ELLIPSE_B_M
    max_neighbor_constraints: int = 12
    neighbor_range_m: float = 90.0

    def __post_init__(self) -> None:
        if self.relative_ellipse_a_m <= 0.0 or self.relative_ellipse_b_m <= 0.0:
            raise ValueError("CBF ellipse semi-axes must be positive")
        if self.relative_ellipse_a_m < self.relative_ellipse_b_m:
            raise ValueError("CBF ellipse semi-major axis must be at least the semi-minor axis")


@dataclass(frozen=True)
class EvaluationProtocolConfig:
    """Strict task-evaluation contract."""

    task_distance_m: float = 1_000.0
    max_policy_steps: int = 3_000
    episodes: int = 200
    strict_collision_free: bool = True


@dataclass(frozen=True)
class LanelessResearchConfig:
    """Single resolved configuration object for new laneless experiments."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    frequencies: FrequencyConfig = field(default_factory=FrequencyConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    evaluation: EvaluationProtocolConfig = field(default_factory=EvaluationProtocolConfig)

    def __post_init__(self) -> None:
        if self.environment.neighbors_count != self.observation.neighbors_count:
            raise ValueError(
                "environment.neighbors_count and observation.neighbors_count must match"
            )
        if self.environment.append_previous_executed_action != self.observation.append_previous_executed_action:
            raise ValueError(
                "environment and observation must agree on previous executed action"
            )

    @property
    def expected_observation_shape(self) -> tuple[int]:
        return (self.observation.expected_dimension,)

    def resolved_environment_config(self) -> dict[str, Any]:
        """Return the explicit environment dictionary used by new factories."""

        env = self.environment
        safety = self.safety
        return {
            "road_length": env.road_length_m,
            "road_width": env.road_width_m,
            "dt": self.frequencies.dt_s,
            "simulation_frequency": self.frequencies.physics_hz,
            "policy_frequency": self.frequencies.policy_hz,
            "cbf_frequency": self.frequencies.cbf_hz,
            "vehicles_count": env.vehicles_count,
            "neighbors_count": env.neighbors_count,
            "sensing_range": env.sensing_range_m,
            "episode_steps": env.episode_steps,
            "duration": env.episode_steps,
            "terminate_on_collision": env.terminate_on_collision,
            "traffic_model": env.traffic_model,
            "ego_desired_speed": env.ego_desired_speed_mps,
            "desired_speed_range": list(env.desired_speed_range_mps),
            "initial_speed_fraction_range": list(env.initial_speed_fraction_range),
            "ego_boundary_force": env.ego_boundary_force,
            "cbf_substep_filtering": env.cbf_substep_filtering,
            "observation_include_vehicle_dimensions": env.include_vehicle_dimensions,
            "ppo_append_previous_action": env.append_previous_executed_action,
            "bounds": {
                "ax_min": self.action.physical_low[0],
                "ax_max": self.action.physical_high[0],
                "ay_min": self.action.physical_low[1],
                "ay_max": self.action.physical_high[1],
            },
            "cbf_geometry": {
                "model": "minimum_area_enclosing_vehicle_ellipse",
                "reference_vehicle_length_m": REFERENCE_VEHICLE_LENGTH_M,
                "reference_vehicle_width_m": REFERENCE_VEHICLE_WIDTH_M,
                "relative_ellipse_a_m": safety.relative_ellipse_a_m,
                "relative_ellipse_b_m": safety.relative_ellipse_b_m,
                "full_major_axis_m": 2.0 * safety.relative_ellipse_a_m,
                "full_minor_axis_m": 2.0 * safety.relative_ellipse_b_m,
            },
            "traffic_safety": {
                "safe_spawn": env.safe_spawn,
                "spawn_cbf_safe_set": env.spawn_cbf_safe_set,
                "spawn_cbf_eps_side": safety.eps_side,
                "spawn_cbf_psi1_gain": safety.psi1_gain,
                "spawn_cbf_k1": safety.psi1_gain,
            },
            "mtm": {
                "leader_range": env.mtm_leader_range_m,
                "profile_probabilities": dict(env.mtm_profile_probabilities),
            },
        }

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible resolved configuration snapshot."""

        data = asdict(self)
        data["observation"]["definition"] = self.observation.definition
        data["derived"] = {
            "dt_s": self.frequencies.dt_s,
            "physics_frames_per_policy_action": self.frequencies.physics_frames_per_policy_action,
            "physics_frames_per_cbf_update": self.frequencies.physics_frames_per_cbf_update,
            "expected_observation_shape": list(self.expected_observation_shape),
            "resolved_environment_config": self.resolved_environment_config(),
        }
        return data


DEFAULT_LANELESS_RESEARCH_CONFIG = LanelessResearchConfig()
