from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.graphics import RoadGraphics, WorldSurface
from highway_env.road.road import Road, RoadNetwork


LANE_FREE_ENV_ID = "lane-free-v0"


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _install_lane_free_road_renderer() -> None:
    """Extend highway-env's RoadGraphics at runtime, without editing its source."""
    if not hasattr(RoadGraphics, "_lane_free_original_display"):
        RoadGraphics._lane_free_original_display = RoadGraphics.display

    def display_lane_free_aware(road: Road, surface: WorldSurface) -> None:
        if not getattr(road, "lane_free", False):
            RoadGraphics._lane_free_original_display(road, surface)
            return

        surface.fill(surface.GREY)
        length = float(getattr(road, "lane_free_length", 1000.0))
        width = float(getattr(road, "lane_free_width", 10.2))
        top_left = surface.pos2pix(0.0, 0.0)
        rect = pygame.Rect(
            top_left[0],
            top_left[1],
            max(1, surface.pix(length)),
            max(1, surface.pix(width)),
        )
        pygame.draw.rect(surface, (70, 70, 70), rect)
        pygame.draw.line(surface, (245, 245, 245), rect.topleft, rect.topright, 2)
        pygame.draw.line(surface, (245, 245, 245), rect.bottomleft, rect.bottomright, 2)

    RoadGraphics.display = staticmethod(display_lane_free_aware)


def register_lane_free_env() -> None:
    """Register this extension module as a Gymnasium/highway-env environment."""
    existing_spec = registry.get(LANE_FREE_ENV_ID)
    if existing_spec is not None and existing_spec.entry_point != "lane_free_env:LaneFreeTrafficEnv":
        del registry[LANE_FREE_ENV_ID]
    if LANE_FREE_ENV_ID not in registry:
        register(id=LANE_FREE_ENV_ID, entry_point="lane_free_env:LaneFreeTrafficEnv")


@dataclass
class LaneFreeVehicleState:
    x: float
    y: float
    vx: float
    vy: float
    length: float
    width: float
    desired_speed: float
    driver_profile: str = "normal"
    # Persistent MTM personality coordinate: 0=cautious, 0.5=normal,
    # 1=aggressive.  ``None`` keeps legacy categorical fixtures compatible.
    driver_aggressiveness: float | None = None


class LaneFreeVehicle:
    """A lane-free vehicle compatible with highway-env's vehicle graphics."""

    HISTORY_SIZE = 30
    PROFILE_COLORS = {
        "normal": (80, 170, 255),
        "aggressive": (235, 80, 70),
        "cautious": (245, 195, 65),
        "ego": (50, 200, 0),
    }

    def __init__(self, road: Road, state: LaneFreeVehicleState, *, is_ego: bool = False) -> None:
        # Do not inherit from highway_env.vehicle.kinematics.Vehicle: its base
        # constructor immediately queries the closest lane.
        self.road = road
        self.position = np.array([state.x, state.y], dtype=np.float64)
        self.heading = 0.0
        self.speed = float(state.vx)
        self.vx = float(state.vx)
        self.vy = float(state.vy)
        self.length = float(state.length)
        self.width = float(state.width)
        self.LENGTH = self.length
        self.WIDTH = self.width
        self.diagonal = float(np.sqrt(self.length**2 + self.width**2))
        self.desired_speed = float(state.desired_speed)
        self.driver_profile = str(state.driver_profile)
        if state.driver_aggressiveness is None:
            self.driver_aggressiveness = None
        else:
            aggressiveness = float(state.driver_aggressiveness)
            if not np.isfinite(aggressiveness):
                raise ValueError("driver_aggressiveness must be finite")
            self.driver_aggressiveness = float(np.clip(aggressiveness, 0.0, 1.0))
        self.is_ego = bool(is_ego)

        self.lane_index = None
        self.lane = None
        self.target_lane_index = None
        self.route = None
        self.action = {"ax": 0.0, "ay": 0.0, "steering": 0.0, "acceleration": 0.0}
        self.ax = 0.0
        self.ay = 0.0

        self.collidable = True
        self.solid = True
        self.check_collisions = False
        self.crashed = False
        self.hit = False
        self.impact = np.zeros(2, dtype=float)
        self.log = []
        self.history = deque(maxlen=self.HISTORY_SIZE)
        self.color = self._profile_color()
        self._sync_graphics_fields()

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy], dtype=float)

    def act(self, action: dict | None = None) -> None:
        if action:
            self.ax = float(action.get("ax", self.ax))
            self.ay = float(action.get("ay", self.ay))
            self.action.update({"ax": self.ax, "ay": self.ay})

    def step(self, dt: float) -> None:
        self.position[0] += self.vx * dt + 0.5 * self.ax * dt * dt
        self.position[1] += self.vy * dt + 0.5 * self.ay * dt * dt
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        self._sync_graphics_fields()
        self.on_state_update()

    def on_state_update(self) -> None:
        if self.road and self.road.record_history:
            self.history.appendleft(self.create_from(self))

    def _sync_graphics_fields(self) -> None:
        self.speed = float(self.vx)
        self.heading = float(np.arctan2(self.vy, max(self.vx, 1e-6)))
        self.LENGTH = self.length
        self.WIDTH = self.width
        self.action["acceleration"] = self.ax
        self.action["steering"] = 0.0

    def _profile_color(self) -> tuple[int, int, int]:
        profile = str(getattr(self, "driver_profile", "normal")).strip().lower()
        if profile == "ego":
            return self.PROFILE_COLORS["ego"]

        score = getattr(self, "driver_aggressiveness", None)
        if score is None or not np.isfinite(float(score)):
            return self.PROFILE_COLORS.get(profile, self.PROFILE_COLORS["normal"])

        # Make the continuous personality visible: yellow -> blue -> red.
        score = float(np.clip(float(score), 0.0, 1.0))
        if score <= 0.5:
            low = np.asarray(self.PROFILE_COLORS["cautious"], dtype=float)
            high = np.asarray(self.PROFILE_COLORS["normal"], dtype=float)
            blend = 2.0 * score
        else:
            low = np.asarray(self.PROFILE_COLORS["normal"], dtype=float)
            high = np.asarray(self.PROFILE_COLORS["aggressive"], dtype=float)
            blend = 2.0 * score - 1.0
        color = np.rint(low + blend * (high - low)).astype(int)
        return tuple(int(channel) for channel in color)

    @classmethod
    def create_from(cls, vehicle: "LaneFreeVehicle") -> "LaneFreeVehicle":
        state = LaneFreeVehicleState(
            x=float(vehicle.position[0]),
            y=float(vehicle.position[1]),
            vx=float(vehicle.vx),
            vy=float(vehicle.vy),
            length=float(vehicle.length),
            width=float(vehicle.width),
            desired_speed=float(vehicle.desired_speed),
            driver_profile=str(getattr(vehicle, "driver_profile", "normal")),
            driver_aggressiveness=getattr(vehicle, "driver_aggressiveness", None),
        )
        copy = cls(vehicle.road, state, is_ego=vehicle.is_ego)
        copy.crashed = vehicle.crashed
        copy.color = vehicle.color
        return copy


class LaneFreeTrafficEnv(AbstractEnv):
    """A highway-env extension for lane-free ring-road traffic.

    The environment uses highway-env's ``AbstractEnv`` and native ``EnvViewer``.
    All lane-free additions live in this module; highway-env source files stay
    untouched.
    """

    # Keep the controlled vehicle footprint identical to the surrounding
    # traffic footprint.  The same dimensions drive collision geometry,
    # CBF clearances, observations, and the native renderer.
    VEHICLE_DIMENSIONS = np.array([[3.50, 1.80], [3.50, 1.80]], dtype=float)
    EGO_DIMENSIONS = VEHICLE_DIMENSIONS[0].copy()

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        config = super().default_config()
        config.update(
            {
                "road_length": 500.0,
                "road_width": 10.2,
                "dt": 0.25,
                "simulation_frequency": 4,
                "policy_frequency": 4,
                "vehicles_count": 35,
                "sensing_range": 80.0,
                "episode_steps": 800,
                "duration": 800,
                "terminate_on_collision": True,
                "gamma_nudge": 0.0,
                "ego_controlled": True,
                # A CBF wrapper can register a physical-action callback that
                # is evaluated at every simulator frame.  It is deliberately
                # opt-in so legacy policies retain their original action
                # semantics and runtime cost.
                "cbf_substep_filtering": False,
                # Keep the historical lateral boundary-force assist by default.
                # Formulation experiments can disable only the ego-side assist
                # while retaining the same traffic and physical road model.
                "ego_boundary_force": True,
                # MTM is the canonical traffic model. The historical force
                # model remains available only when explicitly selected.
                "traffic_model": "mtm",
                "neighbors_count": 5,
                "ego_dimensions": cls.EGO_DIMENSIONS.tolist(),
                "vehicle_dimensions": cls.VEHICLE_DIMENSIONS.tolist(),
                "placeholder_neighbor": [80.0, 10.2, 0.0, 0.0],
                "desired_speed_range": [18.0, 22.0],
                "initial_speed_fraction_range": [0.75, 1.0],
                "observation_vmax": 24.0,
                "observation_vymax": 7.2,
                "observation_include_vehicle_dimensions": True,
                "force": {
                    "k_target_x": 2.5,
                    "k_target_y": 2.0,
                    "k_rep": 8.0,
                    "k_nudge": 0.18,
                    "k_boundary": 0.18,
                    "sigma_x": 5.0,
                    "sigma_y": 1.0,
                    "T_x": 1.2,
                    "T_y": 0.8,
                },
                # Generate only collision-free, buffered initial scenes. The
                # traffic controller still produces genuine
                # interaction during an episode; this prevents reset from
                # starting the ego in an already-overlapping configuration.
                "traffic_safety": {
                    "safe_spawn": True,
                    "spawn_max_attempts": 1_000,
                    # Values below are clearances between vehicle footprints,
                    # not centre-to-centre distances.
                    "spawn_longitudinal_clearance": 6.0,
                    "spawn_lateral_clearance": 0.35,
                    "spawn_ego_longitudinal_clearance": 18.0,
                    "spawn_ego_lateral_clearance": 0.75,
                    "spawn_time_headway": 0.75,
                    # Optional CBF-consistent spawn condition.  When enabled,
                    # the reset state satisfies both the inflated-ellipse
                    # safe set h >= 0 and psi_1 = h_dot + k1 h >= 0.
                    "spawn_cbf_safe_set": False,
                    "spawn_cbf_eps_side": 0.10,
                    "spawn_cbf_margin_m": 0.0,
                    "spawn_cbf_k1": 3.68,
                    "spawn_cbf_psi_margin": 0.0,
                    # The guard controls surrounding traffic only. It never
                    # overwrites the controlled ego action, so an unsafe ego
                    # action can still cause a meaningful collision.
                    "dynamics_guard": True,
                    "guard_horizon_s": 1.5,
                    "guard_range_m": 120.0,
                    "guard_longitudinal_clearance": 1.5,
                    "guard_lateral_clearance": 0.25,
                    "guard_time_headway": 0.25,
                    "guard_lateral_conflict_range": 12.0,
                    "guard_lateral_yield_acceleration": 3.0,
                    # The side-contact shield is deliberately much narrower
                    # than the traffic comfort envelope above. It preserves
                    # aggressive MTM cut-ins and changes only relative lateral
                    # motion that is predicted to create physical overlap.
                    "guard_side_contact_margin": 0.05,
                    "guard_side_prediction_horizon_s": 0.75,
                },
                "mtm": {
                    "theta": 0.2,
                    "s_y0": 0.15,
                    "tilde_s_y0": 0.30,
                    "tau": 1.0,
                    "lambda": 0.4,
                    "lambda_delta_vy": 0.7,
                    "p": 0.2,
                    "a_max": 1.4,
                    "comfortable_decel": 2.0,
                    "time_gap": 1.2,
                    "min_gap": 2.0,
                    "leader_range": 80.0,
                    # Social MTM drivers live on a persistent continuous
                    # cautious-to-aggressive scale.  The probabilities below
                    # shape how much population occupies each diagnostic band;
                    # they no longer select one of three parameter presets.
                    "continuous_driver_aggressiveness": True,
                    "profile_probabilities": {
                        "normal": 0.25,
                        "aggressive": 0.50,
                        "cautious": 0.25,
                    },
                    "profiles": {
                        "normal": {
                            "lambda": 0.4,
                            "tau": 1.0,
                            "p": 0.2,
                            "theta": 0.2,
                            "desired_speed_multiplier": 1.0,
                            "min_gap_multiplier": 1.0,
                        },
                        "aggressive": {
                            "lambda": 0.6,
                            "tau": 0.7,
                            "p": 0.0,
                            "theta": 0.25,
                            "desired_speed_multiplier": 1.15,
                            "min_gap_multiplier": 0.7,
                        },
                        "cautious": {
                            "lambda": 0.28,
                            "tau": 1.4,
                            "p": 0.5,
                            "theta": 0.16,
                            "desired_speed_multiplier": 0.9,
                            "min_gap_multiplier": 1.3,
                        },
                    },
                },
                "bounds": {
                    "ax_min": -6.0,
                    "ax_max": 3.0,
                    "ay_min": -4.0,
                    "ay_max": 4.0,
                    "max_lateral_speed_ratio": 0.3,
                    "max_speed": 45.0,
                },
                "action": {"type": "LaneFreeContinuousAction"},
                "observation": {"type": "LaneFreeKinematics"},
                "screen_width": 900,
                "screen_height": 260,
                "centering_position": [0.3, 0.5],
                "scaling": 5.5,
                "real_time_rendering": True,
            }
        )
        return config

    def configure(self, config: dict | None) -> None:
        if config:
            _deep_update(self.config, config)
        vehicle_dimensions = np.asarray(
            self.config.get("vehicle_dimensions", self.VEHICLE_DIMENSIONS),
            dtype=float,
        )
        if vehicle_dimensions.ndim != 2 or vehicle_dimensions.shape[1] != 2 or len(vehicle_dimensions) == 0:
            raise ValueError(
                "vehicle_dimensions must contain at least one [length, width] pair"
            )
        # Enforce one canonical footprint for ego and surrounding traffic even
        # when a caller passes a legacy ego_dimensions override.
        self.config["vehicle_dimensions"] = vehicle_dimensions.tolist()
        self.config["ego_dimensions"] = vehicle_dimensions[0].tolist()

    def define_spaces(self) -> None:
        rows = 1 + int(self.config.get("neighbors_count", 5))
        include_vehicle_dimensions = bool(
            self.config.get("observation_include_vehicle_dimensions", True)
        )
        features = rows * (7 if include_vehicle_dimensions else 5)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(features,), dtype=np.float32)
        self.action_type = None
        self.observation_type = None

    def set_ego_substep_action_filter(
        self,
        callback: Callable[[np.ndarray, int], Any] | None,
    ) -> Callable[[np.ndarray, int], Any] | None:
        """Install a temporary physical-action filter for simulator substeps.

        The callback receives the *actual proposed ego acceleration* after
        the optional boundary-force contribution, plus its zero-based index
        within the policy step.  It may return either a two-element physical
        acceleration or ``(acceleration, diagnostics)``.  The latter is
        folded into the regular step ``info`` dictionary, allowing a CBF
        wrapper to prove its h/psi constraints at the 100 Hz dynamics rate.

        This is intentionally a narrow callback rather than a second action
        API: normal Gym users and legacy experiments are unchanged unless a
        wrapper explicitly enables ``cbf_substep_filtering``.
        """

        previous = getattr(self, "_ego_substep_action_filter", None)
        self._ego_substep_action_filter = callback
        return previous

    @staticmethod
    def _substep_filter_summary(
        records: list[dict[str, Any]], *, enabled: bool
    ) -> dict[str, Any]:
        """Reduce per-frame CBF records to a compact policy-step summary."""

        if not records:
            return {
                "enabled": bool(enabled),
                "steps": 0,
                "mean_correction_norm_physical": 0.0,
                "max_correction_norm_physical": 0.0,
                "mean_correction_norm_normalized": 0.0,
                "max_correction_norm_normalized": 0.0,
                "intervention_steps": 0,
                "fallback_steps": 0,
                "min_hocbf_margin": float("nan"),
                "max_hocbf_violation_safe": float("nan"),
                "hocbf_condition_satisfied": True,
            }

        def _finite_values(key: str) -> np.ndarray:
            values = [record.get(key, np.nan) for record in records]
            array = np.asarray(values, dtype=float).reshape(-1)
            return array[np.isfinite(array)]

        physical = _finite_values("correction_norm_physical")
        normalized = _finite_values("correction_norm_normalized")
        margins = _finite_values("hocbf_margin")
        violations = _finite_values("max_hocbf_violation_safe")
        satisfied = [
            bool(record.get("hocbf_condition_satisfied", True))
            for record in records
        ]
        return {
            "enabled": bool(enabled),
            "steps": int(len(records)),
            "mean_correction_norm_physical": (
                float(np.mean(physical)) if physical.size else 0.0
            ),
            "max_correction_norm_physical": (
                float(np.max(physical)) if physical.size else 0.0
            ),
            "mean_correction_norm_normalized": (
                float(np.mean(normalized)) if normalized.size else 0.0
            ),
            "max_correction_norm_normalized": (
                float(np.max(normalized)) if normalized.size else 0.0
            ),
            "intervention_steps": int(
                sum(bool(record.get("intervened", False)) for record in records)
            ),
            "fallback_steps": int(
                sum(bool(record.get("fallback_used", False)) for record in records)
            ),
            "min_hocbf_margin": (
                float(np.min(margins)) if margins.size else float("nan")
            ),
            "max_hocbf_violation_safe": (
                float(np.max(violations)) if violations.size else float("nan")
            ),
            "hocbf_condition_satisfied": bool(all(satisfied)),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        if options and "config" in options:
            self.configure(options["config"])
        self.update_metadata()
        self.define_spaces()
        self.time = 0.0
        self.steps = 0
        self.done = False
        self._last_action = np.zeros(2, dtype=np.float32)
        self._last_accelerations = np.zeros((int(self.config["vehicles_count"]), 2), dtype=float)
        self._last_boundary_violations = 0
        self._last_collision_count = 0
        self._last_active_collision_count = 0
        self._last_ego_collision_count = 0
        self._last_ego_collision = False
        self._cumulative_collision_count = 0
        self._active_collision_pairs: set[tuple[int, int]] = set()
        self._flow_count = 0
        self._last_mtm_diagnostics: dict[str, float] = {}
        self._mtm_profile_counts: dict[str, int] = {}
        self._mtm_aggressiveness_stats: dict[str, float] = {}
        self._last_spawn_diagnostics: dict[str, float] = {}
        self._last_traffic_safety_diagnostics: dict[str, float] = {}
        # A wrapper owns this callback for one Gym step and clears it in a
        # finally block.  Reset clears stale state in case a caller aborted a
        # prior step midway through an exception.
        self._ego_substep_action_filter: Callable[[np.ndarray, int], Any] | None = None
        self._last_cbf_substep_diagnostics: dict[str, Any] = self._substep_filter_summary(
            [], enabled=False
        )
        self._reset()
        obs = self._observe()
        return obs, self._info(obs, self._last_action)

    def _reset(self) -> None:
        self.road = Road(
            network=RoadNetwork(),
            vehicles=[],
            np_random=self.np_random,
            record_history=bool(self.config.get("show_trajectories", False)),
        )
        self.road.lane_free = True
        self.road.lane_free_length = float(self.config["road_length"])
        self.road.lane_free_width = float(self.config["road_width"])
        self._create_vehicles()

    def _create_vehicles(self) -> None:
        count = int(self.config["vehicles_count"])
        road_length = float(self.config["road_length"])
        road_width = float(self.config["road_width"])
        desired_low, desired_high = self.config["desired_speed_range"]
        speed_low, speed_high = self.config["initial_speed_fraction_range"]
        mtm_config = self.config.get("mtm", {})
        use_continuous_personality = (
            str(self.config.get("traffic_model", "force")).strip().lower() == "mtm"
            and bool(mtm_config.get("continuous_driver_aggressiveness", True))
        )

        vehicle_specs: list[tuple[LaneFreeVehicleState, bool]] = []
        profile_counts: dict[str, int] = {}
        aggressiveness_scores: list[float] = []

        vehicle_dimensions = np.asarray(self.config.get("vehicle_dimensions", self.VEHICLE_DIMENSIONS), dtype=float)
        ego_length, ego_width = np.asarray(self.config.get("ego_dimensions", self.EGO_DIMENSIONS), dtype=float)

        for index in range(count):
            if index == 0:
                vehicle_length, vehicle_width = ego_length, ego_width
            else:
                vehicle_length, vehicle_width = vehicle_dimensions[
                    self.np_random.integers(0, len(vehicle_dimensions))
                ]
            ego_controlled = index == 0 and bool(self.config.get("ego_controlled", True))
            if ego_controlled:
                driver_profile = "ego"
                driver_aggressiveness = None
                desired_multiplier = 1.0
            elif use_continuous_personality:
                driver_aggressiveness = self._sample_mtm_aggressiveness()
                driver_profile = self._mtm_profile_label(driver_aggressiveness)
                personality = self._interpolate_mtm_personality(
                    driver_aggressiveness
                )
                desired_multiplier = float(
                    personality.get("desired_speed_multiplier", 1.0)
                )
                aggressiveness_scores.append(driver_aggressiveness)
            else:
                driver_profile = self._sample_mtm_profile()
                driver_aggressiveness = None
                profiles = mtm_config.get("profiles", {})
                profile = (
                    profiles.get(driver_profile, {})
                    if isinstance(profiles, dict)
                    else {}
                )
                desired_multiplier = float(
                    profile.get("desired_speed_multiplier", 1.0)
                )
            desired_speed = float(self.np_random.uniform(desired_low, desired_high)) * desired_multiplier
            speed_fraction = float(self.np_random.uniform(speed_low, speed_high))
            vehicle_specs.append(
                (
                    LaneFreeVehicleState(
                        x=0.0,
                        y=0.0,
                        vx=speed_fraction * desired_speed,
                        vy=0.0,
                        length=float(vehicle_length),
                        width=float(vehicle_width),
                        desired_speed=desired_speed,
                        driver_profile=driver_profile,
                        driver_aggressiveness=driver_aggressiveness,
                    ),
                    index == 0,
                )
            )
            profile_counts[driver_profile] = profile_counts.get(driver_profile, 0) + 1

        if bool(self._traffic_safety_config().get("safe_spawn", True)):
            initial_states = self._sample_safe_initial_states(vehicle_specs)
        else:
            initial_states = self._sample_legacy_initial_states(vehicle_specs)

        vehicles = [
            LaneFreeVehicle(self.road, state, is_ego=is_ego)
            for state, is_ego in initial_states
        ]

        self.road.vehicles = vehicles
        self.vehicle = vehicles[0]
        self.controlled_vehicles = [self.vehicle]
        self._mtm_profile_counts = profile_counts
        if aggressiveness_scores:
            scores = np.asarray(aggressiveness_scores, dtype=float)
            self._mtm_aggressiveness_stats = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "unique": float(np.unique(scores).size),
            }
        else:
            self._mtm_aggressiveness_stats = {}

    def _traffic_safety_config(self) -> dict[str, Any]:
        """Return the simulator-side traffic safety settings."""

        config = self.config.get("traffic_safety", {})
        return config if isinstance(config, dict) else {}

    def _sample_legacy_initial_states(
        self, vehicle_specs: list[tuple[LaneFreeVehicleState, bool]]
    ) -> list[tuple[LaneFreeVehicleState, bool]]:
        """Reproduce independent placement for legacy experiments."""

        count = len(vehicle_specs)
        road_length = float(self.config["road_length"])
        road_width = float(self.config["road_width"])
        x_slots = np.linspace(0.0, road_length, count, endpoint=False)
        self.np_random.shuffle(x_slots)
        states: list[tuple[LaneFreeVehicleState, bool]] = []
        for index, (spec, is_ego) in enumerate(vehicle_specs):
            x = float(
                (
                    x_slots[index]
                    + self.np_random.uniform(-0.35, 0.35)
                    * road_length
                    / max(count, 1)
                )
                % road_length
            )
            y = float(
                self.np_random.uniform(
                    spec.width / 2.0, road_width - spec.width / 2.0
                )
            )
            states.append(
                (
                    LaneFreeVehicleState(
                        x=x,
                        y=y,
                        vx=spec.vx,
                        vy=spec.vy,
                        length=spec.length,
                        width=spec.width,
                        desired_speed=spec.desired_speed,
                        driver_profile=spec.driver_profile,
                        driver_aggressiveness=spec.driver_aggressiveness,
                    ),
                    is_ego,
                )
            )
        self._last_spawn_diagnostics = {"safe_spawn": 0.0, "rejections": 0.0}
        return states

    def _sample_safe_initial_states(
        self, vehicle_specs: list[tuple[LaneFreeVehicleState, bool]]
    ) -> list[tuple[LaneFreeVehicleState, bool]]:
        """Sample a collision-free reset with an enlarged ego escape buffer."""

        safety = self._traffic_safety_config()
        road_length = float(self.config["road_length"])
        road_width = float(self.config["road_width"])
        max_attempts = max(1, int(safety.get("spawn_max_attempts", 1_000)))
        placed: list[tuple[LaneFreeVehicleState, bool]] = []
        rejected = 0

        # The ego is placed first. Every later vehicle must respect its larger
        # buffer, so reset does not start in an immediately unresponsive state.
        ordered_specs = sorted(vehicle_specs, key=lambda item: not item[1])
        for spec, is_ego in ordered_specs:
            accepted = False
            for _attempt in range(max_attempts):
                candidate = LaneFreeVehicleState(
                    x=float(self.np_random.uniform(0.0, road_length)),
                    y=float(
                        self.np_random.uniform(
                            spec.width / 2.0, road_width - spec.width / 2.0
                        )
                    ),
                    vx=spec.vx,
                    vy=spec.vy,
                    length=spec.length,
                    width=spec.width,
                    desired_speed=spec.desired_speed,
                    driver_profile=spec.driver_profile,
                    driver_aggressiveness=spec.driver_aggressiveness,
                )
                if all(
                    self._spawn_pair_is_safe(candidate, is_ego, other, other_is_ego)
                    for other, other_is_ego in placed
                ):
                    placed.append((candidate, is_ego))
                    accepted = True
                    break
                rejected += 1
            if not accepted:
                raise RuntimeError(
                    "Unable to construct a collision-free initial traffic scene; "
                    "reduce vehicles_count or relax traffic_safety spawn clearances."
                )

        # Keep the controlled vehicle at index zero: downstream action,
        # reward, observation, and CBF code rely on that contract.
        placed.sort(key=lambda item: not item[1])
        self._last_spawn_diagnostics = {
            "safe_spawn": 1.0,
            "rejections": float(rejected),
            "placed_vehicles": float(len(placed)),
        }
        return placed

    def _spawn_pair_is_safe(
        self,
        first: LaneFreeVehicleState,
        first_is_ego: bool,
        second: LaneFreeVehicleState,
        second_is_ego: bool,
    ) -> bool:
        """Check footprint and closing-speed clearance for one spawn pair."""

        safety = self._traffic_safety_config()
        road_length = float(self.config["road_length"])
        forward_first = float((second.x - first.x) % road_length)
        forward_second = float((first.x - second.x) % road_length)
        if forward_first <= forward_second:
            follower, leader = first, second
            centre_distance = forward_first
        else:
            follower, leader = second, first
            centre_distance = forward_second
        longitudinal_clearance = centre_distance - 0.5 * (first.length + second.length)
        lateral_clearance = abs(first.y - second.y) - 0.5 * (first.width + second.width)

        ego_pair = bool(first_is_ego or second_is_ego)
        required_lateral = float(
            safety.get(
                "spawn_ego_lateral_clearance" if ego_pair else "spawn_lateral_clearance",
                0.75 if ego_pair else 0.35,
            )
        )
        physical_safe = lateral_clearance >= required_lateral
        if not physical_safe:
            base_longitudinal = float(
                safety.get(
                    "spawn_ego_longitudinal_clearance"
                    if ego_pair
                    else "spawn_longitudinal_clearance",
                    18.0 if ego_pair else 6.0,
                )
            )
            closing_speed = max(float(follower.vx - leader.vx), 0.0)
            required_longitudinal = base_longitudinal + float(
                safety.get("spawn_time_headway", 0.75)
            ) * closing_speed
            physical_safe = longitudinal_clearance >= required_longitudinal
        if not physical_safe:
            return False

        # The rectangle guard prevents physical contact, but the CBF operates
        # on an inflated ellipse.  When requested, reset states must satisfy
        # both h >= 0 and psi_1 = h_dot + k1*h >= 0 before the first action.
        if bool(safety.get("spawn_cbf_safe_set", False)):
            eps = max(float(safety.get("spawn_cbf_eps_side", 0.10)), 0.0)
            dx = float(
                (second.x - first.x + 0.5 * road_length) % road_length
                - 0.5 * road_length
            )
            dy = float(second.y - first.y)
            center_distance = float(np.hypot(dx, dy))
            angle = float(np.arctan2(dy, dx)) if center_distance > 1e-9 else 0.0

            def _inflated_radius_terms(
                state: LaneFreeVehicleState,
            ) -> tuple[float, float]:
                a = float(state.length) / np.sqrt(2.0) + 2.0 * eps
                b = float(state.width) / np.sqrt(2.0) + 2.0 * eps
                denominator = np.sqrt(
                    (b * np.cos(angle)) ** 2 + (a * np.sin(angle)) ** 2
                )
                radius = a * b / max(float(denominator), 1e-9)
                d_radius = (
                    -a
                    * b
                    * (a * a - b * b)
                    * np.sin(angle)
                    * np.cos(angle)
                    / max(float(denominator) ** 3, 1e-9)
                )
                return float(radius), float(d_radius)

            first_radius, first_d_radius = _inflated_radius_terms(first)
            second_radius, second_d_radius = _inflated_radius_terms(second)
            h_value = center_distance - first_radius - second_radius
            if h_value < float(safety.get("spawn_cbf_margin_m", 0.0)):
                return False

            radial = np.asarray([dx, dy], dtype=float) / max(center_distance, 1e-9)
            tangent = np.asarray([-radial[1], radial[0]], dtype=float)
            gradient = radial + (
                -(first_d_radius + second_d_radius)
                / max(center_distance, 1e-9)
            ) * tangent
            relative_velocity = np.asarray(
                [float(second.vx - first.vx), float(second.vy - first.vy)],
                dtype=float,
            )
            h_dot = float(gradient @ relative_velocity)
            k1 = float(safety.get("spawn_cbf_k1", 3.68))
            psi_margin = float(safety.get("spawn_cbf_psi_margin", 0.0))
            return bool(h_dot + k1 * h_value >= psi_margin)
        return True

    def _sample_mtm_profile(self) -> str:
        mtm_config = self.config.get("mtm", {})
        probabilities = mtm_config.get("profile_probabilities", {"normal": 1.0})
        if not isinstance(probabilities, dict) or not probabilities:
            return "normal"
        names = list(probabilities.keys())
        weights = np.asarray([max(float(probabilities[name]), 0.0) for name in names], dtype=float)
        total = float(weights.sum())
        if total <= 0.0:
            return names[0]
        weights /= total
        return str(self.np_random.choice(names, p=weights))

    @staticmethod
    def _mtm_profile_label(aggressiveness: float) -> str:
        """Return a compatibility/diagnostic label, not a dynamics selector."""

        score = float(np.clip(float(aggressiveness), 0.0, 1.0))
        if score < 0.25:
            return "cautious"
        if score < 0.75:
            return "normal"
        return "aggressive"

    def _sample_mtm_aggressiveness(self) -> float:
        """Sample one continuous personality while preserving population mix."""

        mtm_config = self.config.get("mtm", {})
        probabilities = mtm_config.get("profile_probabilities", {})
        uniform_quantile = float(self.np_random.random())
        if not isinstance(probabilities, dict) or not probabilities:
            return uniform_quantile

        bands = (
            ("cautious", 0.0, 0.25),
            ("normal", 0.25, 0.75),
            ("aggressive", 0.75, 1.0),
        )
        weights = np.asarray(
            [max(float(probabilities.get(name, 0.0)), 0.0) for name, _, _ in bands],
            dtype=float,
        )
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            return uniform_quantile
        weights /= total

        cumulative = 0.0
        for index, ((_, low, high), weight) in enumerate(zip(bands, weights)):
            next_cumulative = cumulative + float(weight)
            if uniform_quantile < next_cumulative or index == len(bands) - 1:
                if weight <= 0.0:
                    cumulative = next_cumulative
                    continue
                local_quantile = (uniform_quantile - cumulative) / float(weight)
                return float(low + np.clip(local_quantile, 0.0, 1.0) * (high - low))
            cumulative = next_cumulative
        return uniform_quantile

    def _interpolate_mtm_personality(
        self, aggressiveness: float
    ) -> dict[str, float]:
        """Interpolate the existing profile anchors on a continuous scale."""

        score = float(np.clip(float(aggressiveness), 0.0, 1.0))
        if score <= 0.5:
            lower_name, upper_name, blend = "cautious", "normal", 2.0 * score
        else:
            lower_name, upper_name, blend = (
                "normal",
                "aggressive",
                2.0 * score - 1.0,
            )

        mtm_config = self.config.get("mtm", {})
        profiles = mtm_config.get("profiles", {})
        profiles = profiles if isinstance(profiles, dict) else {}
        lower = profiles.get(lower_name, {})
        upper = profiles.get(upper_name, {})
        lower = lower if isinstance(lower, dict) else {}
        upper = upper if isinstance(upper, dict) else {}
        base_defaults = {
            "lambda": float(mtm_config.get("lambda", 0.4)),
            "tau": float(mtm_config.get("tau", 1.0)),
            "p": float(mtm_config.get("p", 0.2)),
            "theta": float(mtm_config.get("theta", 0.2)),
            "desired_speed_multiplier": 1.0,
            "min_gap_multiplier": 1.0,
        }
        personality: dict[str, float] = {}
        for key, default in base_defaults.items():
            low_value = float(lower.get(key, default))
            high_value = float(upper.get(key, default))
            personality[key] = float(
                low_value + blend * (high_value - low_value)
            )
        return personality

    def step(self, action: np.ndarray | list[float] | None) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.zeros(2, dtype=np.float32) if action is None else np.asarray(action, dtype=np.float32)
        action_array = np.clip(action_array, -1.0, 1.0)
        self._last_action = action_array

        frames = max(1, int(round(float(self.config["simulation_frequency"]) / float(self.config["policy_frequency"]))))
        dt = float(self.config.get("dt", 1.0 / float(self.config["simulation_frequency"])))
        substep_filter = getattr(self, "_ego_substep_action_filter", None)
        substep_enabled = bool(
            self.config.get("cbf_substep_filtering", False)
            and callable(substep_filter)
        )
        substep_records: list[dict[str, Any]] = []
        for frame_index in range(frames):
            accelerations = self._compute_accelerations()
            if bool(self.config["ego_controlled"]):
                ego_ax = self._map_action(action_array[0], "longitudinal")
                ego_boundary_force = (
                    self._boundary_force(self.vehicle)
                    if bool(self.config.get("ego_boundary_force", True))
                    else 0.0
                )
                proposed_ego_acceleration = np.asarray(
                    [
                        ego_ax,
                        self._map_action(action_array[1], "lateral")
                        + ego_boundary_force,
                    ],
                    dtype=float,
                )
                if substep_enabled:
                    filtered = substep_filter(
                        proposed_ego_acceleration.copy(), int(frame_index)
                    )
                    if isinstance(filtered, tuple):
                        if len(filtered) != 2:
                            raise ValueError(
                                "ego substep filter must return action or (action, diagnostics)"
                            )
                        safe_acceleration, diagnostics = filtered
                    else:
                        safe_acceleration, diagnostics = filtered, {}
                    safe_acceleration = np.asarray(
                        safe_acceleration, dtype=float
                    ).reshape(-1)
                    if safe_acceleration.size < 2 or not np.all(
                        np.isfinite(safe_acceleration[:2])
                    ):
                        raise ValueError(
                            "ego substep filter returned a non-finite physical action"
                        )
                    accelerations[0, :2] = safe_acceleration[:2]
                    if isinstance(diagnostics, dict):
                        substep_records.append(dict(diagnostics))
                    else:
                        substep_records.append({})
                else:
                    accelerations[0, 0] = ego_ax
                    accelerations[0, 1] = proposed_ego_acceleration[1]
            accelerations = self._apply_traffic_safety_guard(accelerations, dt)
            accelerations = self._clip_accelerations(accelerations)
            self._last_accelerations = accelerations
            self._integrate(accelerations, dt)
            self._detect_collisions()
            self.steps += 1
            self.time += dt

        self._last_cbf_substep_diagnostics = self._substep_filter_summary(
            substep_records, enabled=substep_enabled
        )

        obs = self._observe()
        reward = self._reward(action_array)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action_array)
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _map_action(self, value: float, axis: str) -> float:
        bounds = self.config["bounds"]
        if axis == "longitudinal":
            low, high = float(bounds["ax_min"]), float(bounds["ax_max"])
        else:
            low, high = float(bounds["ay_min"]), float(bounds["ay_max"])
        value = min(max(float(value), -1.0), 1.0)
        if low < 0.0 < high:
            scale = high if value >= 0.0 else abs(low)
            return value * max(scale, 1e-6)
        return low + 0.5 * (value + 1.0) * (high - low)

    def _compute_accelerations(self) -> np.ndarray:
        traffic_model = str(self.config.get("traffic_model", "force")).strip().lower()
        if traffic_model == "mtm":
            return self._compute_mtm_accelerations()
        return self._compute_force_accelerations()

    def _compute_force_accelerations(self) -> np.ndarray:
        vehicles = self.road.vehicles
        force = self.config['force']
        count = len(vehicles)
        accelerations = np.zeros((count, 2), dtype=float)
        self._last_mtm_diagnostics = {}

        if count == 0:
            return accelerations

        x = np.fromiter((vehicle.position[0] for vehicle in vehicles), dtype=float, count=count)
        y = np.fromiter((vehicle.position[1] for vehicle in vehicles), dtype=float, count=count)
        vx = np.fromiter((vehicle.vx for vehicle in vehicles), dtype=float, count=count)
        vy = np.fromiter((vehicle.vy for vehicle in vehicles), dtype=float, count=count)
        lengths = np.fromiter((vehicle.length for vehicle in vehicles), dtype=float, count=count)
        widths = np.fromiter((vehicle.width for vehicle in vehicles), dtype=float, count=count)
        desired_speeds = np.fromiter((vehicle.desired_speed for vehicle in vehicles), dtype=float, count=count)

        accelerations[:, 0] = float(force["k_target_x"]) * np.tanh(
            (desired_speeds - vx) / max(float(force["sigma_x"]), 1e-6)
        )
        accelerations[:, 1] = -float(force["k_target_y"]) * np.tanh(
            vy / max(float(force["sigma_y"]), 1e-6)
        )
        road_width = float(self.config["road_width"])
        boundary_eps = 1e-3
        left_clearance = np.maximum(y - 0.5 * widths, boundary_eps)
        right_clearance = np.maximum(road_width - 0.5 * widths - y, boundary_eps)
        accelerations[:, 1] += float(force["k_boundary"]) * (
            1.0 / (left_clearance + boundary_eps) ** 2
            - 1.0 / (right_clearance + boundary_eps) ** 2
        )

        sensing_range = float(self.config["sensing_range"])
        gamma_nudge = float(self.config["gamma_nudge"])
        k_rep = float(force["k_rep"])
        k_nudge = float(force["k_nudge"])
        t_x = float(force["T_x"])
        t_y = float(force["T_y"])

        road_length = float(self.config["road_length"])
        dx = (x[None, :] - x[:, None]) % road_length
        dy = y[None, :] - y[:, None]
        valid_pairs = (dx > 0.0) & (dx < sensing_range) & ~np.eye(count, dtype=bool)

        dx_scale = (
            0.5 * (lengths[:, None] + lengths[None, :])
            + t_x * np.maximum(vx[:, None] - vx[None, :], 0.0)
        )
        dvy = vy[None, :] - vy[:, None]
        v_close_y = np.maximum(0.0, -(dy * dvy) / (np.abs(dy) + 1e-6))
        dy_scale = 0.5 * (widths[:, None] + widths[None, :]) + t_y * v_close_y
        dx_scale = np.maximum(dx_scale, 1e-6)
        dy_scale = np.maximum(dy_scale, 1e-6)

        potential = np.exp(-((dx / dx_scale) ** 2) - ((dy / dy_scale) ** 2))
        potential *= valid_pairs
        norm = np.sqrt(dx * dx + dy * dy) + 1e-6
        ux = dx / norm
        uy = dy / norm
        longitudinal_effect = potential * ux
        lateral_effect = potential * uy

        accelerations[:, 0] -= k_rep * np.sum(longitudinal_effect, axis=1)
        accelerations[:, 1] -= k_rep * np.sum(lateral_effect, axis=1)

        if gamma_nudge != 0.0 and k_nudge != 0.0:
            blocked = np.maximum(desired_speeds - vx, 0.0)[:, None]
            nudge_scale = gamma_nudge * k_nudge
            accelerations[:, 0] += nudge_scale * np.sum(blocked * longitudinal_effect, axis=0)
            accelerations[:, 1] += nudge_scale * np.sum(blocked * lateral_effect, axis=0)

        return accelerations

    def _compute_mtm_accelerations(self) -> np.ndarray:
        vehicles = self.road.vehicles
        count = len(vehicles)
        if count == 0:
            self._last_mtm_diagnostics = {}
            return np.zeros((0, 2), dtype=float)

        x = np.fromiter((vehicle.position[0] for vehicle in vehicles), dtype=float, count=count)
        y = np.fromiter((vehicle.position[1] for vehicle in vehicles), dtype=float, count=count)
        vx = np.fromiter((vehicle.vx for vehicle in vehicles), dtype=float, count=count)
        vy = np.fromiter((vehicle.vy for vehicle in vehicles), dtype=float, count=count)
        lengths = np.fromiter((vehicle.length for vehicle in vehicles), dtype=float, count=count)
        widths = np.fromiter((vehicle.width for vehicle in vehicles), dtype=float, count=count)
        desired_speeds = np.fromiter((vehicle.desired_speed for vehicle in vehicles), dtype=float, count=count)
        is_ego = np.fromiter((vehicle.is_ego for vehicle in vehicles), dtype=bool, count=count)

        parameter_names = (
            "theta",
            "s_y0",
            "tilde_s_y0",
            "tau",
            "lambda",
            "lambda_delta_vy",
            "p",
            "a_max",
            "comfortable_decel",
            "time_gap",
            "min_gap",
            "leader_range",
        )
        parameter_rows = [self._mtm_params_for_vehicle(vehicle) for vehicle in vehicles]
        parameters = {
            name: np.fromiter((row[name] for row in parameter_rows), dtype=float, count=count)
            for name in parameter_names
        }
        state = np.column_stack((x, y, vx, vy, lengths, widths, desired_speeds))
        parameter_matrix = np.column_stack(tuple(parameters.values()))
        if not (np.all(np.isfinite(state)) and np.all(np.isfinite(parameter_matrix))):
            return self._compute_mtm_accelerations_scalar()

        desired_safe = np.maximum(desired_speeds, 1e-6)
        a_max = np.maximum(parameters["a_max"], 1e-6)
        comfortable_decel = np.maximum(parameters["comfortable_decel"], 1e-6)
        free_accel = a_max * (1.0 - (np.maximum(vx, 0.0) / desired_safe) ** 4)

        road_length = float(self.config["road_length"])
        dx = (x[None, :] - x[:, None]) % road_length
        dy = y[None, :] - y[:, None]
        valid = (
            (dx > 0.0)
            & (dx < parameters["leader_range"][:, None])
            & ~np.eye(count, dtype=bool)
        )
        gap_x = np.maximum(dx - 0.5 * (lengths[:, None] + lengths[None, :]), 0.05)
        lateral_gap = np.maximum(
            np.abs(dy) - parameters["theta"][:, None] * 0.5 * (widths[:, None] + widths[None, :]),
            0.0,
        )
        alpha = np.minimum(
            np.exp(-lateral_gap / np.maximum(parameters["s_y0"][:, None], 1e-6)),
            1.0,
        )
        alpha_tilde = np.minimum(
            np.exp(-lateral_gap / np.maximum(parameters["tilde_s_y0"][:, None], 1e-6)),
            1.0,
        )

        delta_v = vx[:, None] - vx[None, :]
        dynamic_gap = (
            vx[:, None] * parameters["time_gap"][:, None]
            + vx[:, None] * delta_v / (2.0 * np.sqrt(a_max * comfortable_decel)[:, None])
        )
        desired_gap = parameters["min_gap"][:, None] + np.maximum(dynamic_gap, 0.0)
        speed_ratio = np.clip(np.maximum(vx, 0.0) / desired_safe, 0.0, 10.0)
        gap_term = (desired_gap / gap_x) ** 2
        car_following = np.clip(
            a_max[:, None] * (1.0 - speed_ratio[:, None] ** 4 - gap_term),
            -50.0,
            50.0,
        )
        interaction = car_following - free_accel[:, None]
        traffic = alpha * interaction
        candidate = valid & ((alpha > 1e-6) | (alpha_tilde > 1e-6))

        traffic_magnitude = np.where(candidate, np.abs(traffic), -np.inf)
        strongest_indices = np.argmax(traffic_magnitude, axis=1)
        has_leader = np.any(candidate, axis=1)
        row_indices = np.arange(count)
        strongest_traffic = np.where(has_leader, traffic[row_indices, strongest_indices], 0.0)
        leader_gaps = np.where(has_leader, gap_x[row_indices, strongest_indices], 0.0)

        obstruction = np.maximum(-interaction, 0.0)
        away_direction = np.where(
            np.abs(dy) > 1e-6,
            -np.sign(dy),
            np.where(y[:, None] <= 0.5 * float(self.config["road_width"]), 1.0, -1.0),
        )
        dy_sign = np.where(dy > 0.0, 1.0, np.where(dy < 0.0, -1.0, 0.0))
        relative_vy = vy[None, :] - vy[:, None]
        relative_lateral_factor = np.clip(
            1.0 - parameters["lambda_delta_vy"][:, None] * relative_vy * dy_sign,
            0.0,
            2.5,
        )
        politeness_scale = np.clip(1.0 - 0.25 * parameters["p"], 0.5, 1.2)
        lateral_increments = (
            parameters["lambda"][:, None]
            * politeness_scale[:, None]
            * alpha_tilde
            * obstruction
            * away_direction
            * relative_lateral_factor
        )
        lateral_increments = np.where(candidate & (obstruction > 0.0), lateral_increments, 0.0)
        desired_vy = np.sum(lateral_increments, axis=1)

        boundary_forces = np.fromiter(
            (self._boundary_force(vehicle) for vehicle in vehicles),
            dtype=float,
            count=count,
        )
        accelerations = np.column_stack(
            (
                free_accel + strongest_traffic,
                (desired_vy - vy) / np.maximum(parameters["tau"], 1e-6) + boundary_forces,
            )
        )
        ego_controlled = bool(self.config["ego_controlled"])
        driven = ~(is_ego & ego_controlled)
        accelerations[~driven, 0] = 0.0
        accelerations[~driven, 1] = boundary_forces[~driven]

        driven_count = max(int(np.count_nonzero(driven)), 1)
        driven_has_leader = has_leader & driven
        driven_desired_vy = desired_vy[driven]
        driven_vy = np.abs(vy[driven])
        self._last_mtm_diagnostics = {
            "active_leader_rate": float(np.count_nonzero(driven_has_leader) / driven_count),
            "mean_abs_vy": float(np.mean(driven_vy)) if driven_vy.size else 0.0,
            "mean_abs_desired_vy": float(np.mean(np.abs(driven_desired_vy))) if driven_desired_vy.size else 0.0,
            "mean_leader_gap": float(np.mean(leader_gaps[driven_has_leader])) if np.any(driven_has_leader) else 0.0,
            "max_abs_desired_vy": float(np.max(np.abs(driven_desired_vy))) if driven_desired_vy.size else 0.0,
        }
        return accelerations

    def _compute_mtm_accelerations_scalar(self) -> np.ndarray:
        vehicles = self.road.vehicles
        accelerations = np.zeros((len(vehicles), 2), dtype=float)
        desired_vy_values: list[float] = []
        active_leaders = 0
        leader_distances: list[float] = []

        for i, vehicle in enumerate(vehicles):
            if vehicle.is_ego and bool(self.config["ego_controlled"]):
                accelerations[i, 1] = self._boundary_force(vehicle)
                continue

            ax, ay, diagnostics = self._mtm_controller(vehicle, vehicles)
            accelerations[i, 0] = ax
            accelerations[i, 1] = ay
            desired_vy_values.append(float(diagnostics["desired_vy"]))
            active_leaders += int(bool(diagnostics["has_leader"]))
            if bool(diagnostics["has_leader"]):
                leader_distances.append(float(diagnostics["leader_gap"]))

        ego_controlled = bool(self.config["ego_controlled"])
        mtm_driven_vehicles = [vehicle for vehicle in vehicles if not (vehicle.is_ego and ego_controlled)]
        mtm_vehicle_count = max(len(mtm_driven_vehicles), 1)
        mtm_vy = [abs(float(vehicle.vy)) for vehicle in mtm_driven_vehicles]
        self._last_mtm_diagnostics = {
            "active_leader_rate": float(active_leaders / mtm_vehicle_count),
            "mean_abs_vy": float(np.mean(mtm_vy)) if mtm_vy else 0.0,
            "mean_abs_desired_vy": float(np.mean(np.abs(desired_vy_values))) if desired_vy_values else 0.0,
            "mean_leader_gap": float(np.mean(leader_distances)) if leader_distances else 0.0,
            "max_abs_desired_vy": float(np.max(np.abs(desired_vy_values))) if desired_vy_values else 0.0,
        }
        return accelerations

    def _mtm_controller(self, vehicle: LaneFreeVehicle, vehicles: list[LaneFreeVehicle]) -> tuple[float, float, dict[str, float]]:
        params = self._mtm_params_for_vehicle(vehicle)
        free_accel = self._mtm_free_acceleration(vehicle, params)
        if not np.isfinite(free_accel):
            free_accel = 0.0
        strongest_traffic = 0.0
        desired_vy = 0.0
        best_abs_traffic = -np.inf
        best_gap = 0.0
        has_leader = False
        leader_range = float(params["leader_range"])

        for other in vehicles:
            if other is vehicle:
                continue
            if not (
                np.isfinite(vehicle.position[0])
                and np.isfinite(vehicle.position[1])
                and np.isfinite(other.position[0])
                and np.isfinite(other.position[1])
                and np.isfinite(vehicle.vx)
                and np.isfinite(vehicle.vy)
                and np.isfinite(other.vx)
                and np.isfinite(other.vy)
            ):
                continue
            dx = self._forward_distance(vehicle.position[0], other.position[0])
            if not (0.0 < dx < leader_range):
                continue

            gap_x = max(dx - 0.5 * (vehicle.length + other.length), 0.05)
            dy = float(other.position[1] - vehicle.position[1])
            lateral_gap = self._mtm_lateral_gap(vehicle, other, params, theta_key="theta")
            if not (np.isfinite(gap_x) and np.isfinite(dy) and np.isfinite(lateral_gap)):
                continue
            alpha = min(float(np.exp(-lateral_gap / max(float(params["s_y0"]), 1e-6))), 1.0)
            alpha_tilde = min(float(np.exp(-lateral_gap / max(float(params["tilde_s_y0"]), 1e-6))), 1.0)
            if not (np.isfinite(alpha) and np.isfinite(alpha_tilde)):
                continue
            if alpha <= 1e-6 and alpha_tilde <= 1e-6:
                continue

            car_following = self._mtm_car_following_acceleration(vehicle, other, gap_x, params)
            if not np.isfinite(car_following):
                continue
            interaction = car_following - free_accel
            if not np.isfinite(interaction):
                continue
            traffic = alpha * interaction
            if not np.isfinite(traffic):
                continue
            if abs(traffic) > best_abs_traffic:
                strongest_traffic = traffic
                best_abs_traffic = abs(traffic)
                best_gap = gap_x
                has_leader = True

            obstruction = max(-interaction, 0.0)
            if obstruction <= 0.0:
                continue
            away_direction = self._mtm_away_direction(vehicle, other, dy)
            dy_sign = 1.0 if dy > 0.0 else -1.0 if dy < 0.0 else 0.0
            relative_vy = float(other.vy) - float(vehicle.vy)
            lambda_delta_vy = float(params.get("lambda_delta_vy", 0.0))
            if not all(np.isfinite(value) for value in [away_direction, relative_vy, lambda_delta_vy, dy_sign]):
                continue
            relative_lateral_factor = 1.0 - lambda_delta_vy * relative_vy * dy_sign
            relative_lateral_factor = float(min(max(relative_lateral_factor, 0.0), 2.5))
            politeness_scale = float(np.clip(1.0 - 0.25 * float(params["p"]), 0.5, 1.2))
            lateral_vy_increment = (
                float(params["lambda"])
                * politeness_scale
                * alpha_tilde
                * obstruction
                * away_direction
                * relative_lateral_factor
            )
            if np.isfinite(lateral_vy_increment):
                desired_vy += lateral_vy_increment

        ax = free_accel + strongest_traffic
        ay = (desired_vy - float(vehicle.vy)) / max(float(params["tau"]), 1e-6) + self._boundary_force(vehicle)
        return (
            float(ax),
            float(ay),
            {
                "desired_vy": float(desired_vy),
                "has_leader": float(has_leader),
                "leader_gap": float(best_gap),
                "free_accel": float(free_accel),
                "traffic_accel": float(strongest_traffic),
            },
        )

    def _mtm_params_for_vehicle(self, vehicle: LaneFreeVehicle) -> dict[str, float]:
        mtm_config = self.config.get("mtm", {})
        profile_name = str(getattr(vehicle, "driver_profile", "normal"))
        profiles = mtm_config.get("profiles", {}) if isinstance(mtm_config, dict) else {}
        profile = profiles.get(profile_name, {}) if isinstance(profiles, dict) else {}
        aggressiveness = getattr(vehicle, "driver_aggressiveness", None)
        if aggressiveness is not None:
            profile = self._interpolate_mtm_personality(float(aggressiveness))
        params = {
            "theta": float(mtm_config.get("theta", 0.2)),
            "s_y0": float(mtm_config.get("s_y0", 0.15)),
            "tilde_s_y0": float(mtm_config.get("tilde_s_y0", 0.30)),
            "tau": float(mtm_config.get("tau", 1.0)),
            "lambda": float(mtm_config.get("lambda", 0.4)),
            "lambda_delta_vy": float(mtm_config.get("lambda_delta_vy", 0.7)),
            "p": float(mtm_config.get("p", 0.2)),
            "a_max": float(mtm_config.get("a_max", 1.4)),
            "comfortable_decel": float(mtm_config.get("comfortable_decel", 2.0)),
            "time_gap": float(mtm_config.get("time_gap", 1.2)),
            "min_gap": float(mtm_config.get("min_gap", 2.0)),
            "leader_range": float(mtm_config.get("leader_range", self.config["sensing_range"])),
        }
        for key in ["theta", "s_y0", "tilde_s_y0", "tau", "lambda", "lambda_delta_vy", "p", "a_max", "comfortable_decel", "time_gap"]:
            if key in profile:
                params[key] = float(profile[key])
        if "min_gap" in profile:
            params["min_gap"] = float(profile["min_gap"])
        elif "min_gap_multiplier" in profile:
            params["min_gap"] *= float(profile["min_gap_multiplier"])
        return params

    def _mtm_free_acceleration(self, vehicle: LaneFreeVehicle, params: dict[str, float]) -> float:
        desired_speed = max(float(vehicle.desired_speed), 1e-6)
        speed_ratio = max(float(vehicle.vx), 0.0) / desired_speed
        return float(params["a_max"] * (1.0 - speed_ratio**4))

    def _mtm_car_following_acceleration(
        self,
        vehicle: LaneFreeVehicle,
        leader: LaneFreeVehicle,
        gap_x: float,
        params: dict[str, float],
    ) -> float:
        desired_speed = max(float(vehicle.desired_speed), 1e-6)
        a_max = max(float(params["a_max"]), 1e-6)
        comfortable_decel = max(float(params["comfortable_decel"]), 1e-6)
        vx = float(vehicle.vx)
        leader_vx = float(leader.vx)
        gap_x = float(gap_x)
        time_gap = float(params["time_gap"])
        min_gap = float(params["min_gap"])
        if not all(np.isfinite(value) for value in [desired_speed, a_max, comfortable_decel, vx, leader_vx, gap_x, time_gap, min_gap]):
            return 0.0
        delta_v = vx - leader_vx
        dynamic_gap = vx * time_gap + (
            vx * delta_v / (2.0 * np.sqrt(a_max * comfortable_decel))
        )
        if not np.isfinite(dynamic_gap):
            dynamic_gap = 0.0
        desired_gap = min_gap + max(0.0, dynamic_gap)
        if not np.isfinite(desired_gap):
            desired_gap = min_gap
        speed_ratio = np.clip(max(vx, 0.0) / desired_speed, 0.0, 10.0)
        speed_term = float(speed_ratio**4)
        gap_term = (desired_gap / max(float(gap_x), 0.05)) ** 2
        if not np.isfinite(gap_term):
            gap_term = 1e6
        return float(np.clip(a_max * (1.0 - speed_term - gap_term), -50.0, 50.0))

    def _mtm_lateral_gap(
        self,
        vehicle: LaneFreeVehicle,
        other: LaneFreeVehicle,
        params: dict[str, float],
        *,
        theta_key: str,
    ) -> float:
        dy = abs(float(other.position[1] - vehicle.position[1]))
        lateral_body = float(params[theta_key]) * 0.5 * (vehicle.width + other.width)
        return float(max(dy - lateral_body, 0.0))

    def _mtm_away_direction(self, vehicle: LaneFreeVehicle, other: LaneFreeVehicle, dy: float) -> float:
        if abs(dy) > 1e-6:
            return float(-np.sign(dy))
        road_width = float(self.config["road_width"])
        if vehicle.position[1] <= 0.5 * road_width:
            return 1.0
        return -1.0

    def _boundary_force(self, vehicle: LaneFreeVehicle) -> float:
        road_width = float(self.config["road_width"])
        k_boundary = float(self.config["force"]["k_boundary"])
        eps = 1e-3
        left_clearance = max(float(vehicle.position[1] - vehicle.width / 2.0), eps)
        right_clearance = max(float(road_width - vehicle.width / 2.0 - vehicle.position[1]), eps)
        return k_boundary * (1.0 / (left_clearance + eps) ** 2 - 1.0 / (right_clearance + eps) ** 2)

    def _apply_traffic_safety_guard(
        self, accelerations: np.ndarray, dt: float
    ) -> np.ndarray:
        """Keep social traffic from creating an unavoidable ego conflict.

        The controlled ego acceleration is deliberately immutable here.  The
        guard asks surrounding vehicles to brake or yield laterally whenever
        their proposed motion would consume the ego's emergency-braking
        envelope, and applies the same rear-end rule to traffic-traffic pairs.
        This is a simulator traffic rule, not a hidden ego safety filter.
        """

        safety = self._traffic_safety_config()
        guarded = np.asarray(accelerations, dtype=float).copy()
        diagnostics = {
            "enabled": float(bool(safety.get("dynamics_guard", True))),
            "constraints": 0.0,
            "traffic_brakes": 0.0,
            "ego_leader_yields": 0.0,
            "lateral_yields": 0.0,
            "side_constraints": 0.0,
            "side_infeasible": 0.0,
            "side_projection_yields": 0.0,
        }
        vehicles = self.road.vehicles
        count = len(vehicles)
        if not bool(safety.get("dynamics_guard", True)) or count < 2:
            self._last_traffic_safety_diagnostics = diagnostics
            return guarded

        road_length = float(self.config["road_length"])
        bounds = self.config["bounds"]
        ax_min = float(bounds["ax_min"])
        ax_max = float(bounds["ax_max"])
        max_braking = max(abs(ax_min), 1e-6)
        horizon = max(float(safety.get("guard_horizon_s", 1.5)), float(dt), 1e-3)
        guard_range = max(float(safety.get("guard_range_m", 120.0)), 0.0)
        min_longitudinal = max(
            float(safety.get("guard_longitudinal_clearance", 1.5)), 0.0
        )
        min_lateral = max(float(safety.get("guard_lateral_clearance", 0.25)), 0.0)
        time_headway = max(float(safety.get("guard_time_headway", 0.25)), 0.0)
        lateral_conflict_range = max(
            float(safety.get("guard_lateral_conflict_range", 12.0)), 0.0
        )
        ego_controlled = bool(self.config.get("ego_controlled", True))
        controlled = np.asarray(
            [bool(vehicle.is_ego and ego_controlled) for vehicle in vehicles],
            dtype=bool,
        )

        x = np.fromiter(
            (float(vehicle.position[0]) for vehicle in vehicles),
            dtype=float,
            count=count,
        )
        y = np.fromiter(
            (float(vehicle.position[1]) for vehicle in vehicles),
            dtype=float,
            count=count,
        )
        vx = np.fromiter(
            (float(vehicle.vx) for vehicle in vehicles), dtype=float, count=count
        )
        vy = np.fromiter(
            (float(vehicle.vy) for vehicle in vehicles), dtype=float, count=count
        )
        lengths = np.fromiter(
            (float(vehicle.length) for vehicle in vehicles),
            dtype=float,
            count=count,
        )
        widths = np.fromiter(
            (float(vehicle.width) for vehicle in vehicles),
            dtype=float,
            count=count,
        )

        # Matrix rows are followers and columns are leaders.  Keeping this
        # calculation vectorized is essential: the 55-vehicle experiment has
        # 3,000 ordered pairs per physics frame.
        forward_distance = (x[None, :] - x[:, None]) % road_length
        longitudinal_gap = forward_distance - 0.5 * (
            lengths[:, None] + lengths[None, :]
        )
        lateral_offset = y[None, :] - y[:, None]
        lateral_gap_now = np.abs(lateral_offset) - 0.5 * (
            widths[:, None] + widths[None, :]
        )
        predicted_lateral_offset = (
            lateral_offset
            + horizon * (vy[None, :] - vy[:, None])
            + 0.5
            * horizon
            * horizon
            * (guarded[None, :, 1] - guarded[:, None, 1])
        )
        lateral_gap_predicted = np.abs(predicted_lateral_offset) - 0.5 * (
            widths[:, None] + widths[None, :]
        )
        valid_pair = (
            (forward_distance > 1e-9)
            & (forward_distance <= guard_range)
            & ~np.eye(count, dtype=bool)
        )
        lateral_conflict = np.minimum(
            lateral_gap_now, lateral_gap_predicted
        ) < min_lateral
        stopping_distance_delta = np.maximum(
            vx[:, None] ** 2 - vx[None, :] ** 2, 0.0
        ) / (2.0 * max_braking)
        required_gap = (
            min_longitudinal
            + time_headway * np.maximum(vx[:, None], 0.0)
            + stopping_distance_delta
        )
        needs_guard = valid_pair & lateral_conflict & (
            longitudinal_gap < required_gap
        )
        diagnostics["constraints"] = float(np.count_nonzero(needs_guard))

        # Social followers brake for either social traffic or the ego ahead.
        brake_mask = (~controlled) & np.any(needs_guard, axis=1)
        changing_brake = brake_mask & (guarded[:, 0] > ax_min + 1e-9)
        guarded[changing_brake, 0] = ax_min
        diagnostics["traffic_brakes"] = float(np.count_nonzero(changing_brake))

        # The ego remains fully controlled. If it is the follower, require a
        # social leader to accelerate/yield instead of rewriting the action.
        ego_indices = np.flatnonzero(controlled)
        if ego_indices.size:
            ego_index = int(ego_indices[0])
            ego_leader_mask = needs_guard[ego_index] & ~controlled
            leader_indices = np.flatnonzero(ego_leader_mask)
            if leader_indices.size:
                needed_leader_ax = ax_min + 2.0 * (
                    required_gap[ego_index, leader_indices]
                    - longitudinal_gap[ego_index, leader_indices]
                    - horizon
                    * (vx[leader_indices] - vx[ego_index])
                ) / (horizon * horizon)
                leader_targets = np.minimum(needed_leader_ax, ax_max)
                previous = guarded[leader_indices, 0].copy()
                guarded[leader_indices, 0] = np.maximum(previous, leader_targets)
                diagnostics["ego_leader_yields"] = float(
                    np.count_nonzero(
                        guarded[leader_indices, 0] > previous + 1e-9
                    )
                )

        # Restore the original broad lateral interaction drive as the nominal
        # source of cut-ins and zig-zag motion. It is intentionally applied
        # before the narrow contact projection below, which is responsible
        # only for trimming the final inward component that would overlap.
        lateral_yield_pairs = needs_guard & (
            longitudinal_gap
            <= np.maximum(required_gap, lateral_conflict_range)
        )
        participant = np.any(lateral_yield_pairs, axis=1) | np.any(
            lateral_yield_pairs, axis=0
        )
        offset_sign = np.sign(lateral_offset)
        lateral_score = (
            np.sum(-offset_sign * lateral_yield_pairs, axis=1)
            + np.sum(offset_sign * lateral_yield_pairs, axis=0)
        )
        lateral_direction = np.sign(lateral_score)
        undecided = participant & (np.abs(lateral_direction) < 1e-9)
        upper_room = float(self.config["road_width"]) - 0.5 * widths - y
        lower_room = y - 0.5 * widths
        lateral_direction[undecided] = np.where(
            upper_room[undecided] >= lower_room[undecided], 1.0, -1.0
        )
        yield_magnitude = min(
            max(float(safety.get("guard_lateral_yield_acceleration", 3.0)), 0.0),
            max(abs(float(bounds["ay_min"])), abs(float(bounds["ay_max"]))),
        )
        lateral_mask = participant & ~controlled & (
            np.abs(lateral_direction) > 1e-9
        )
        previous_lateral = guarded[:, 1].copy()
        positive = lateral_mask & (lateral_direction > 0.0)
        negative = lateral_mask & (lateral_direction < 0.0)
        guarded[positive, 1] = np.maximum(
            guarded[positive, 1], yield_magnitude
        )
        guarded[negative, 1] = np.minimum(
            guarded[negative, 1], -yield_magnitude
        )
        lateral_drive_yields = int(
            np.count_nonzero(
                lateral_mask
                & (np.abs(guarded[:, 1] - previous_lateral) > 1e-9)
            )
        )

        (
            guarded,
            side_constraints,
            side_projection_yields,
            side_infeasible,
        ) = self._project_lateral_contact_accelerations(
            guarded,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            lengths=lengths,
            widths=widths,
            controlled=controlled,
            dt=dt,
            safety=safety,
        )
        diagnostics["side_constraints"] = float(side_constraints)
        diagnostics["lateral_yields"] = float(lateral_drive_yields)
        diagnostics["side_projection_yields"] = float(side_projection_yields)
        diagnostics["side_infeasible"] = float(side_infeasible)

        self._last_traffic_safety_diagnostics = diagnostics
        return guarded

    def _project_lateral_contact_accelerations(
        self,
        accelerations: np.ndarray,
        *,
        x: np.ndarray,
        y: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        lengths: np.ndarray,
        widths: np.ndarray,
        controlled: np.ndarray,
        dt: float,
        safety: dict[str, Any],
    ) -> tuple[np.ndarray, int, int, int]:
        """Minimally remove imminent inward motion at a side contact.

        MTM remains the nominal controller. A constraint is created only when
        the unaccelerated lateral time-to-contact is short *and* the vehicle
        footprints are predicted to overlap longitudinally at that time. The
        projection changes the pair's relative lateral acceleration while
        preserving its common-mode acceleration whenever both vehicles are
        social traffic. A controlled ego action is never modified.
        """

        projected = np.asarray(accelerations, dtype=float).copy()
        count = len(projected)
        if count < 2:
            return projected, 0, 0, 0

        bounds = self.config["bounds"]
        ay_min = float(bounds["ay_min"])
        ay_max = float(bounds["ay_max"])
        road_length = float(self.config["road_length"])
        contact_margin = max(
            float(safety.get("guard_side_contact_margin", 0.05)), 0.0
        )
        prediction_horizon = max(
            float(safety.get("guard_side_prediction_horizon_s", 0.75)),
            float(dt),
        )
        constraints: list[tuple[int, int, float, float]] = []
        infeasible = 0

        for first in range(count - 1):
            for second in range(first + 1, count):
                lateral_offset = float(y[second] - y[first])
                if abs(lateral_offset) > 1e-9:
                    separation_sign = float(np.sign(lateral_offset))
                else:
                    # This tie-break is deterministic and chooses the ordering
                    # with more combined room before any overlap has occurred.
                    first_lower_room = float(y[first] - 0.5 * widths[first])
                    first_upper_room = float(
                        self.config["road_width"]
                        - 0.5 * widths[first]
                        - y[first]
                    )
                    second_lower_room = float(y[second] - 0.5 * widths[second])
                    second_upper_room = float(
                        self.config["road_width"]
                        - 0.5 * widths[second]
                        - y[second]
                    )
                    separation_sign = (
                        1.0
                        if second_upper_room + first_lower_room
                        >= second_lower_room + first_upper_room
                        else -1.0
                    )

                half_width_sum = 0.5 * float(widths[first] + widths[second])
                physical_lateral_clearance = (
                    abs(lateral_offset) - half_width_sum
                )
                separation_rate = separation_sign * float(
                    vy[second] - vy[first]
                )
                if separation_rate >= -1e-9:
                    continue

                closing_speed = -separation_rate
                physical_time_to_contact = max(
                    physical_lateral_clearance, 0.0
                ) / max(closing_speed, 1e-9)
                if physical_time_to_contact > prediction_horizon:
                    continue

                # Check longitudinal occupancy at the predicted side-contact
                # instant. Sampling +/- one physics step makes the gate robust
                # to discrete integration without creating a comfort buffer.
                signed_dx = float(
                    (
                        x[second]
                        - x[first]
                        + 0.5 * road_length
                    )
                    % road_length
                    - 0.5 * road_length
                )
                relative_vx = float(vx[second] - vx[first])
                relative_ax = float(
                    projected[second, 0] - projected[first, 0]
                )
                sample_times = np.unique(
                    np.clip(
                        np.asarray(
                            [
                                physical_time_to_contact - float(dt),
                                physical_time_to_contact,
                                physical_time_to_contact + float(dt),
                            ],
                            dtype=float,
                        ),
                        0.0,
                        prediction_horizon,
                    )
                )
                predicted_dx = (
                    signed_dx
                    + relative_vx * sample_times
                    + 0.5 * relative_ax * sample_times**2
                )
                predicted_dx = (
                    predicted_dx + 0.5 * road_length
                ) % road_length - 0.5 * road_length
                half_length_sum = 0.5 * float(
                    lengths[first] + lengths[second]
                )
                if float(np.min(np.abs(predicted_dx))) >= half_length_sum:
                    continue

                usable_clearance = max(
                    physical_lateral_clearance - contact_margin, 1e-6
                )
                required_separation_acceleration = (
                    closing_speed**2 / (2.0 * usable_clearance)
                )

                first_controlled = bool(controlled[first])
                second_controlled = bool(controlled[second])
                if first_controlled and second_controlled:
                    infeasible += 1
                    continue
                if separation_sign > 0.0:
                    maximum_first = (
                        projected[first, 1] if first_controlled else ay_min
                    )
                    maximum_second = (
                        projected[second, 1] if second_controlled else ay_max
                    )
                else:
                    maximum_first = (
                        projected[first, 1] if first_controlled else ay_max
                    )
                    maximum_second = (
                        projected[second, 1] if second_controlled else ay_min
                    )
                maximum_separation_acceleration = separation_sign * float(
                    maximum_second - maximum_first
                )
                if (
                    required_separation_acceleration
                    > maximum_separation_acceleration + 1e-9
                ):
                    infeasible += 1
                target = min(
                    required_separation_acceleration,
                    max(maximum_separation_acceleration, 0.0),
                )
                constraints.append(
                    (first, second, separation_sign, float(target))
                )

        original_lateral = projected[:, 1].copy()
        # Repeated half-space projections resolve vehicles participating in
        # more than one pair while retaining the smallest practical change.
        for _ in range(6):
            any_change = False
            for first, second, separation_sign, target in constraints:
                current = separation_sign * float(
                    projected[second, 1] - projected[first, 1]
                )
                deficit = target - current
                if deficit <= 1e-9:
                    continue

                first_coefficient = -separation_sign
                second_coefficient = separation_sign

                def capacity(index: int, coefficient: float) -> float:
                    if bool(controlled[index]):
                        return 0.0
                    if coefficient > 0.0:
                        return max(ay_max - float(projected[index, 1]), 0.0)
                    return max(float(projected[index, 1]) - ay_min, 0.0)

                first_capacity = capacity(first, first_coefficient)
                second_capacity = capacity(second, second_coefficient)
                correction = min(deficit, first_capacity + second_capacity)
                if correction <= 1e-12:
                    continue

                first_share = min(0.5 * correction, first_capacity)
                second_share = min(0.5 * correction, second_capacity)
                remainder = correction - first_share - second_share
                if remainder > 0.0:
                    first_extra = min(
                        remainder, first_capacity - first_share
                    )
                    first_share += first_extra
                    remainder -= first_extra
                if remainder > 0.0:
                    second_share += min(
                        remainder, second_capacity - second_share
                    )

                projected[first, 1] += first_coefficient * first_share
                projected[second, 1] += second_coefficient * second_share
                any_change = True
            if not any_change:
                break

        projected[:, 1] = np.clip(projected[:, 1], ay_min, ay_max)
        lateral_yields = int(
            np.count_nonzero(
                np.abs(projected[:, 1] - original_lateral) > 1e-9
            )
        )
        unsatisfied = sum(
            int(
                separation_sign
                * float(projected[second, 1] - projected[first, 1])
                < target - 1e-7
            )
            for first, second, separation_sign, target in constraints
        )
        return projected, len(constraints), lateral_yields, max(infeasible, unsatisfied)

    def _clip_accelerations(self, accelerations: np.ndarray) -> np.ndarray:
        bounds = self.config["bounds"]
        accelerations[:, 0] = np.clip(accelerations[:, 0], float(bounds["ax_min"]), float(bounds["ax_max"]))
        accelerations[:, 1] = np.clip(accelerations[:, 1], float(bounds["ay_min"]), float(bounds["ay_max"]))
        return accelerations

    def _integrate(self, accelerations: np.ndarray, dt: float) -> None:
        road_length = float(self.config["road_length"])
        road_width = float(self.config["road_width"])
        max_speed = float(self.config["bounds"]["max_speed"])
        lateral_ratio = float(self.config["bounds"]["max_lateral_speed_ratio"])
        old_x = np.array([vehicle.position[0] for vehicle in self.road.vehicles], dtype=float)
        boundary_violations = 0

        for vehicle, (ax, ay) in zip(self.road.vehicles, accelerations):
            vehicle.ax = float(ax)
            vehicle.ay = float(ay)
            vehicle.step(dt)
            vehicle.position[0] = float(vehicle.position[0] % road_length)
            vehicle.vx = float(min(max(vehicle.vx, 0.0), max_speed))
            max_lateral_speed = lateral_ratio * max(vehicle.vx, 1.0)
            vehicle.vy = float(min(max(vehicle.vy, -max_lateral_speed), max_lateral_speed))

            y_min = vehicle.width / 2.0
            y_max = road_width - vehicle.width / 2.0
            if vehicle.position[1] < y_min or vehicle.position[1] > y_max:
                boundary_violations += 1
                vehicle.position[1] = float(min(max(vehicle.position[1], y_min), y_max))
                vehicle.vy = 0.0
            vehicle._sync_graphics_fields()

        for previous_x, vehicle in zip(old_x, self.road.vehicles):
            if previous_x + max(vehicle.vx, 0.0) * dt >= road_length and vehicle.position[0] < previous_x:
                self._flow_count += 1
        self._last_boundary_violations = boundary_violations

    def _detect_collisions(self) -> None:
        vehicles = self.road.vehicles
        count = len(vehicles)
        snapshots = np.full((count, 4), np.nan, dtype=float)
        is_ego = np.zeros(count, dtype=bool)
        for index, vehicle in enumerate(vehicles):
            is_ego[index] = bool(vehicle.is_ego)
            try:
                snapshots[index] = (
                    float(vehicle.position[0]),
                    float(vehicle.position[1]),
                    float(vehicle.length),
                    float(vehicle.width),
                )
            except (TypeError, ValueError, IndexError):
                continue

        valid = np.all(np.isfinite(snapshots), axis=1)
        x, y, lengths, widths = snapshots.T
        road_length = float(self.config["road_length"])
        signed_dx = (
            (x[None, :] - x[:, None] + 0.5 * road_length) % road_length
        ) - 0.5 * road_length
        dx = np.abs(signed_dx)
        dy = np.abs(y[None, :] - y[:, None])
        colliding = (
            valid[:, None]
            & valid[None, :]
            & (dx < 0.5 * (lengths[:, None] + lengths[None, :]))
            & (dy < 0.5 * (widths[:, None] + widths[None, :]))
        )
        colliding = np.triu(colliding, k=1)
        pair_indices = np.argwhere(colliding)
        active_pairs = {(int(first), int(second)) for first, second in pair_indices}
        crashed = np.any(colliding, axis=0) | np.any(colliding, axis=1)
        for index, vehicle in enumerate(vehicles):
            vehicle.crashed = bool(crashed[index])

        ego_collision = bool(np.any(crashed & is_ego))

        new_pairs = active_pairs - self._active_collision_pairs
        self._last_collision_count = len(new_pairs)
        self._last_active_collision_count = len(active_pairs)
        self._last_ego_collision_count = sum(
            int(vehicles[first].is_ego or vehicles[second].is_ego)
            for first, second in new_pairs
        )
        self._last_ego_collision = ego_collision
        self._cumulative_collision_count += len(new_pairs)
        self._active_collision_pairs = active_pairs

    def _observe(self) -> np.ndarray:
        ego = self.vehicle
        vehicles = self.road.vehicles
        neighbor_count = int(self.config["neighbors_count"])
        include_vehicle_dimensions = bool(
            self.config.get("observation_include_vehicle_dimensions", True)
        )
        rows = np.zeros(
            (1 + neighbor_count, 7 if include_vehicle_dimensions else 5),
            dtype=np.float32,
        )
        if not vehicles:
            return rows.reshape(-1)

        count = len(vehicles)
        x = np.fromiter((vehicle.position[0] for vehicle in vehicles), dtype=float, count=count)
        y = np.fromiter((vehicle.position[1] for vehicle in vehicles), dtype=float, count=count)
        vx = np.fromiter((vehicle.vx for vehicle in vehicles), dtype=float, count=count)
        vy = np.fromiter((vehicle.vy for vehicle in vehicles), dtype=float, count=count)
        if include_vehicle_dimensions:
            lengths = np.fromiter(
                (vehicle.length for vehicle in vehicles), dtype=float, count=count
            )
            widths = np.fromiter(
                (vehicle.width for vehicle in vehicles), dtype=float, count=count
            )
        desired_speeds = np.fromiter((vehicle.desired_speed for vehicle in vehicles), dtype=float, count=count)
        ego_index = next((index for index, vehicle in enumerate(vehicles) if vehicle is ego), 0)

        road_length = float(self.config["road_length"])
        signed_dx = ((x - x[ego_index] + 0.5 * road_length) % road_length) - 0.5 * road_length
        relative_y = y - y[ego_index]
        distance_squared = signed_dx * signed_dx + relative_y * relative_y
        candidate_indices = np.delete(np.arange(count), ego_index)
        if candidate_indices.size:
            order = np.argsort(distance_squared[candidate_indices], kind="stable")
            chosen_neighbors = candidate_indices[order[:neighbor_count]]
        else:
            chosen_neighbors = np.empty(0, dtype=int)
        selected = np.concatenate((np.asarray([ego_index], dtype=int), chosen_neighbors))
        selected_count = len(selected)

        road_width = max(float(self.config["road_width"]), 1e-6)
        sensing_range = max(float(self.config["sensing_range"]), 1e-6)
        observation_vmax = max(float(self.config.get("observation_vmax", 24.0)), 1e-6)
        observation_vymax = max(float(self.config.get("observation_vymax", 0.3 * observation_vmax)), 1e-6)
        rows[:selected_count, 0] = np.clip(signed_dx[selected] / sensing_range, -1.0, 1.0)
        rows[:selected_count, 1] = np.clip(relative_y[selected] / road_width, -1.0, 1.0)
        rows[:selected_count, 2] = vx[selected] / observation_vmax
        rows[:selected_count, 3] = vy[selected] / observation_vymax
        if include_vehicle_dimensions:
            rows[:selected_count, 4] = lengths[selected] / 5.15
            rows[:selected_count, 5] = widths[selected] / 1.84
            rows[:selected_count, 6] = desired_speeds[selected] / observation_vmax
        else:
            rows[:selected_count, 4] = desired_speeds[selected] / observation_vmax
        rows[0, :2] = 0.0
        return rows.reshape(-1)

    def _observation_row(self, vehicle: LaneFreeVehicle, ego: LaneFreeVehicle) -> np.ndarray:
        include_vehicle_dimensions = bool(
            self.config.get("observation_include_vehicle_dimensions", True)
        )
        road_width = float(self.config["road_width"])
        sensing_range = float(self.config["sensing_range"])
        observation_vmax = max(float(self.config.get("observation_vmax", 24.0)), 1e-6)
        observation_vymax = max(float(self.config.get("observation_vymax", 0.3 * observation_vmax)), 1e-6)
        signed_dx = 0.0 if vehicle is ego else self._signed_distance(ego.position[0], vehicle.position[0])
        dy = 0.0 if vehicle is ego else float(vehicle.position[1] - ego.position[1])
        values = [
            np.clip(signed_dx / sensing_range, -1.0, 1.0),
            np.clip(dy / road_width, -1.0, 1.0),
            vehicle.vx / observation_vmax,
            vehicle.vy / observation_vymax,
        ]
        if include_vehicle_dimensions:
            values.extend([vehicle.length / 5.15, vehicle.width / 1.84])
        values.append(vehicle.desired_speed / observation_vmax)
        return np.asarray(values, dtype=np.float32)

    def _reward(self, action: np.ndarray) -> float:
        ego = self.vehicle
        ay_ego = float(self._last_accelerations[0, 1]) if len(self._last_accelerations) else 0.0
        boundary_penalty = 2.0 if self._last_boundary_violations > 0 else 0.0
        return float(
            ego.vx / max(ego.desired_speed, 1e-6)
            - 5.0 * float(self._last_ego_collision)
            - 0.05 * abs(ay_ego)
            - 0.02 * abs(ego.vy)
            - boundary_penalty
        )

    def _is_terminated(self) -> bool:
        return bool(self.config.get("terminate_on_collision", True) and self._last_ego_collision)

    def _is_truncated(self) -> bool:
        return self.steps >= int(self.config.get("episode_steps", self.config.get("duration", 2000)))

    def _info(self, obs: np.ndarray, action: np.ndarray | None = None) -> dict[str, Any]:
        elapsed_hours = max(self.time / 3600.0, 1e-9)
        info = {
            "traffic_model": str(self.config.get("traffic_model", "force")),
            "speed": float(self.vehicle.vx),
            "mean_speed": self.mean_speed,
            "collisions": int(self._last_collision_count),
            "active_collisions": int(self._last_active_collision_count),
            "ego_collision_events": int(self._last_ego_collision_count),
            "cumulative_collisions": int(self._cumulative_collision_count),
            "ego_collision": bool(self._last_ego_collision),
            "boundary_violations": int(self._last_boundary_violations),
            "flow_count": int(self._flow_count),
            "flow_per_hour": float(self._flow_count / elapsed_hours),
        }
        spawn = self._last_spawn_diagnostics or {}
        traffic_safety = self._last_traffic_safety_diagnostics or {}
        substep_cbf = self._last_cbf_substep_diagnostics or {}
        info.update(
            {
                "traffic_safe_spawn": bool(spawn.get("safe_spawn", 0.0)),
                "traffic_spawn_rejections": int(spawn.get("rejections", 0.0)),
                "traffic_guard_constraints": int(
                    traffic_safety.get("constraints", 0.0)
                ),
                "traffic_guard_brakes": int(
                    traffic_safety.get("traffic_brakes", 0.0)
                ),
                "traffic_guard_ego_leader_yields": int(
                    traffic_safety.get("ego_leader_yields", 0.0)
                ),
                "traffic_guard_lateral_yields": int(
                    traffic_safety.get("lateral_yields", 0.0)
                ),
                "traffic_guard_side_constraints": int(
                    traffic_safety.get("side_constraints", 0.0)
                ),
                "traffic_guard_side_infeasible": int(
                    traffic_safety.get("side_infeasible", 0.0)
                ),
                "traffic_guard_side_projection_yields": int(
                    traffic_safety.get("side_projection_yields", 0.0)
                ),
                "cbf_substep_filter_enabled": bool(
                    substep_cbf.get("enabled", False)
                ),
                "cbf_substep_count": int(substep_cbf.get("steps", 0)),
                "cbf_substep_mean_correction_norm": float(
                    substep_cbf.get("mean_correction_norm_physical", 0.0)
                ),
                "cbf_substep_max_correction_norm": float(
                    substep_cbf.get("max_correction_norm_physical", 0.0)
                ),
                "cbf_substep_mean_correction_norm_normalized": float(
                    substep_cbf.get("mean_correction_norm_normalized", 0.0)
                ),
                "cbf_substep_max_correction_norm_normalized": float(
                    substep_cbf.get("max_correction_norm_normalized", 0.0)
                ),
                "cbf_substep_intervention_steps": int(
                    substep_cbf.get("intervention_steps", 0)
                ),
                "cbf_substep_fallback_steps": int(
                    substep_cbf.get("fallback_steps", 0)
                ),
                "cbf_substep_min_hocbf_margin": float(
                    substep_cbf.get("min_hocbf_margin", float("nan"))
                ),
                "cbf_substep_max_hocbf_violation_safe": float(
                    substep_cbf.get("max_hocbf_violation_safe", float("nan"))
                ),
                "cbf_substep_hocbf_condition_satisfied": bool(
                    substep_cbf.get("hocbf_condition_satisfied", True)
                ),
            }
        )
        if str(self.config.get("traffic_model", "force")).strip().lower() == "mtm":
            diagnostics = self._last_mtm_diagnostics or {}
            info.update(
                {
                    "mtm_active_leader_rate": float(diagnostics.get("active_leader_rate", 0.0)),
                    "mtm_mean_abs_vy": float(diagnostics.get("mean_abs_vy", 0.0)),
                    "mtm_mean_abs_desired_vy": float(diagnostics.get("mean_abs_desired_vy", 0.0)),
                    "mtm_max_abs_desired_vy": float(diagnostics.get("max_abs_desired_vy", 0.0)),
                    "mtm_mean_leader_gap": float(diagnostics.get("mean_leader_gap", 0.0)),
                }
            )
            for profile_name, count in self._mtm_profile_counts.items():
                info[f"mtm_profile_count_{profile_name}"] = int(count)
            for statistic, value in self._mtm_aggressiveness_stats.items():
                info[f"mtm_aggressiveness_{statistic}"] = (
                    int(value) if statistic == "unique" else float(value)
                )
        return info

    @property
    def mean_speed(self) -> float:
        return float(np.mean([vehicle.vx for vehicle in self.road.vehicles])) if self.road else 0.0

    def snapshot(self) -> np.ndarray:
        return np.asarray(
            [
                [
                    vehicle.position[0],
                    vehicle.position[1],
                    vehicle.vx,
                    vehicle.vy,
                    vehicle.length,
                    vehicle.width,
                    vehicle.desired_speed,
                    float(vehicle.is_ego),
                    float(vehicle.crashed),
                ]
                for vehicle in self.road.vehicles
            ],
            dtype=float,
        )

    def _forward_distance(self, x_from: float, x_to: float) -> float:
        road_length = float(self.config["road_length"])
        if not (math.isfinite(x_from) and math.isfinite(x_to) and math.isfinite(road_length) and road_length > 1e-9):
            return 0.0
        return float((x_to - x_from) % road_length)

    def _signed_distance(self, x_from: float, x_to: float) -> float:
        road_length = float(self.config["road_length"])
        if not (math.isfinite(x_from) and math.isfinite(x_to) and math.isfinite(road_length) and road_length > 1e-9):
            return 0.0
        return float(((x_to - x_from + 0.5 * road_length) % road_length) - 0.5 * road_length)

    def _distance_to_ego(self, vehicle: LaneFreeVehicle) -> float:
        dx = self._signed_distance(self.vehicle.position[0], vehicle.position[0])
        dy = float(vehicle.position[1] - self.vehicle.position[1])
        return float(np.sqrt(dx * dx + dy * dy))


_install_lane_free_road_renderer()
register_lane_free_env()


__all__ = ["LaneFreeTrafficEnv", "LaneFreeVehicle", "LaneFreeVehicleState", "register_lane_free_env"]
