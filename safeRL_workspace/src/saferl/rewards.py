"""Linear alternative to the reciprocal Karalakou tracking term.

The canonical notebook reward is

    r = eps / (eps + wx*cx + wy*cy + wf*cf + way*cay)
        + progress_reward - jerk_penalty + event

The pull of each cost, ``|dr/dc_i| = eps * w_i / denom**2``, depends on every
other cost: it is strongest near perfect tracking and collapses once any cost
is large, so raising one weight past ``eps + sum(other costs)`` weakens its
own pull. The linear mode replaces only the first term with a weighted mean of
per-cost tracking scores,

    r_track = sum_i w_i * (1 - clip(c_i, 0, 1)) / sum_i w_i,

so each pull is the constant ``w_i / sum_i w_i`` and ``r_track`` stays in
``[0, 1]``. The clip is inactive for the canonical costs (cf and cay are
already capped at 1, cy <= 0.82 on the 10.2 m road, and cx <= 1 below twice
the 16 m/s target); it only guarantees the bound. Keeping ``r_track``
non-negative matters because episodes terminate on collision: a sustained
negative per-step reward would make crashing the return-maximizing choice.
``epsilon_r`` is unused in this mode. Progress, jerk, collision, and overtake
terms are unchanged.

Lateral target fallback. The notebook's ``_lateral_target_and_speed`` targets
the centre of the nearest free gap among the vehicles ahead and, when no gap
exists, the lateral position of the fastest vehicle ahead. At the canonical
55-vehicle density no gap exists on about 99% of steps, so that fallback is a
jumping target (on the ego's opposite road half about half the time), which
leaves ``cy`` without a consistent pull. ``lateral_target_fallback="center"``
replaces only the no-gap fallback with the road centre; the gap search, the
speed target, and every reward term are unchanged. The same method feeds the
exposed target-y observation, so the policy observes the target it is
rewarded for. It composes with either reward mode.

Potential-field weight. The canonical tracking denominator carries the
neighbour potential-field cost as ``wf * cf``, where cf is the capped sum of
two ellipsoid kernels per in-range vehicle. ``--potential-field-weight 0``
sets wf to zero, which removes the term from the denominator while cf is
still computed and published, so a run stays comparable against one that
scored it. With wf = 0 the reciprocal term tracks speed and lateral position
only, and every incentive to keep clear of neighbours comes from the
collision penalty and, when it is in the loop, the CBF. The flag composes
with either reward mode; in the linear mode it drops cf out of the weighted
mean as well, since the weights are normalized by their own sum.

Overtake detection. The notebook's ``_overtake_count`` pays the bonus only
when a vehicle goes from ahead (dx > 0) to more than half an ego length behind
within one policy step. At 20 Hz that needs a relative speed above 35 m/s, so
the bonus is never paid (0 bonuses for 3 real passes in 3057 steps, 40
vehicles, 2026-09-11). ``overtake_detection="latched"`` counts a vehicle once
it has been seen ahead within sensing range at any earlier step and is now
more than half an ego length behind (and still within sensing range, which
excludes ring wrap-around). Each vehicle counts at most once per episode, so
dropping back and re-passing the same vehicle cannot farm the bonus. The
bonus value and the one-bonus-per-step cap are unchanged.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Mapping

import numpy

RECIPROCAL_REWARD_MODE = "reciprocal"
LINEAR_REWARD_MODE = "linear"
SUPPORTED_REWARD_MODES = (RECIPROCAL_REWARD_MODE, LINEAR_REWARD_MODE)

FASTEST_BLOCKER_LATERAL_FALLBACK = "fastest_blocker"
CENTER_LATERAL_FALLBACK = "center"
SUPPORTED_LATERAL_FALLBACKS = (FASTEST_BLOCKER_LATERAL_FALLBACK, CENTER_LATERAL_FALLBACK)

STEP_OVERTAKE_DETECTION = "step"
LATCHED_OVERTAKE_DETECTION = "latched"
SUPPORTED_OVERTAKE_DETECTIONS = (STEP_OVERTAKE_DETECTION, LATCHED_OVERTAKE_DETECTION)

GAP_LATERAL_TARGET = "gap"
FIELD_LATERAL_TARGET = "field"
SUPPORTED_LATERAL_TARGETS = (GAP_LATERAL_TARGET, FIELD_LATERAL_TARGET)

DEFAULT_LATERAL_FIELD_GRID_STEP_M = 0.1
DEFAULT_LATERAL_FIELD_SIGMA_M = 0.45
DEFAULT_LATERAL_FIELD_BETA = 6.0
DEFAULT_LATERAL_FIELD_TRAVEL_WEIGHT = 0.5
DEFAULT_LATERAL_FIELD_BASIN_MARGIN = 0.25
DEFAULT_LATERAL_FIELD_HYSTERESIS = 0.0
# Wall repulsion. The occupancy field sums contributions from vehicles only, so
# a road boundary reads as free space and the target is pulled toward it: with
# the field target the measured target sits in the outer 1.5 m of a 10.2 m road
# ~23% of the time. These give each boundary an occupancy contribution of the
# same functional form as a vehicle, so "the wall occupies space" is expressed
# in the same units the rest of the field already uses. Default 0.0 keeps the
# historical behaviour; the feature is opt-in.
DEFAULT_LATERAL_FIELD_WALL_WEIGHT = 0.0
DEFAULT_LATERAL_FIELD_WALL_SIGMA_M = 0.9

ALL_LATERAL_BLOCKERS = "all"
CLOSING_LATERAL_BLOCKERS = "closing"
CLOSING_FORWARD_LATERAL_BLOCKERS = "closing_forward"
SUPPORTED_LATERAL_BLOCKERS = (
    ALL_LATERAL_BLOCKERS,
    CLOSING_LATERAL_BLOCKERS,
    CLOSING_FORWARD_LATERAL_BLOCKERS,
)

DEFAULT_LATERAL_BLOCKER_HORIZON_S = 3.0
DEFAULT_LATERAL_BLOCKER_STANDSTILL_GAP_M = 5.0

NOMINAL_SPEED_TARGET = "nominal"
HEADWAY_SPEED_TARGET = "headway"
SUPPORTED_SPEED_TARGETS = (NOMINAL_SPEED_TARGET, HEADWAY_SPEED_TARGET)

DEFAULT_SPEED_TARGET_NOMINAL = 20.0
DEFAULT_SPEED_TARGET_STANDSTILL_GAP_M = 5.0
DEFAULT_SPEED_TARGET_LATERAL_SIGMA_M = 0.45

# (component name published by the notebook wrapper, reward-config weight key)
_TRACKING_COSTS = (("cx", "wx"), ("cy", "wy"), ("cf", "wf"), ("cay", "way"))


def resolve_reward_mode(reward_config: Mapping[str, Any]) -> str:
    """Return the configured reward mode, rejecting modes nothing implements."""

    mode = str(reward_config.get("reward_mode", RECIPROCAL_REWARD_MODE)).strip().lower()
    if mode not in SUPPORTED_REWARD_MODES:
        raise ValueError(
            f"Unsupported reward_mode {mode!r}; the notebook reward wrapper "
            f"implements only {SUPPORTED_REWARD_MODES}"
        )
    return mode


def resolve_lateral_target_fallback(reward_config: Mapping[str, Any]) -> str:
    """Return the configured no-gap lateral target, defaulting to the notebook's."""

    fallback = (
        str(reward_config.get("lateral_target_fallback", FASTEST_BLOCKER_LATERAL_FALLBACK))
        .strip()
        .lower()
    )
    if fallback not in SUPPORTED_LATERAL_FALLBACKS:
        raise ValueError(
            f"Unsupported lateral_target_fallback {fallback!r}; expected one of "
            f"{SUPPORTED_LATERAL_FALLBACKS}"
        )
    return fallback


def resolve_overtake_detection(reward_config: Mapping[str, Any]) -> str:
    """Return the configured overtake detector, defaulting to the notebook's."""

    detection = (
        str(reward_config.get("overtake_detection", STEP_OVERTAKE_DETECTION)).strip().lower()
    )
    if detection not in SUPPORTED_OVERTAKE_DETECTIONS:
        raise ValueError(
            f"Unsupported overtake_detection {detection!r}; expected one of "
            f"{SUPPORTED_OVERTAKE_DETECTIONS}"
        )
    return detection


def resolve_lateral_blockers(reward_config: Mapping[str, Any]) -> str:
    """Return the configured blocker set, defaulting to the notebook's."""

    blockers = (
        str(reward_config.get("lateral_blockers", ALL_LATERAL_BLOCKERS)).strip().lower()
    )
    if blockers not in SUPPORTED_LATERAL_BLOCKERS:
        raise ValueError(
            f"Unsupported lateral_blockers {blockers!r}; expected one of "
            f"{SUPPORTED_LATERAL_BLOCKERS}"
        )
    return blockers


def lateral_blocker_parameters(reward_config: Mapping[str, Any]) -> dict[str, float]:
    """Return the closing-horizon parameters, validated."""

    parameters = {
        "horizon_s": float(
            reward_config.get(
                "lateral_blocker_horizon_s", DEFAULT_LATERAL_BLOCKER_HORIZON_S
            )
        ),
        "standstill_gap_m": float(
            reward_config.get(
                "lateral_blocker_standstill_gap_m",
                DEFAULT_LATERAL_BLOCKER_STANDSTILL_GAP_M,
            )
        ),
    }
    for key, value in parameters.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"lateral blocker parameter {key} must be finite and positive"
            )
    return parameters


def resolve_speed_target(reward_config: Mapping[str, Any]) -> str:
    """Return the configured speed target, defaulting to the notebook's fixed one."""

    target = (
        str(reward_config.get("speed_target", NOMINAL_SPEED_TARGET)).strip().lower()
    )
    if target not in SUPPORTED_SPEED_TARGETS:
        raise ValueError(
            f"Unsupported speed_target {target!r}; expected one of {SUPPORTED_SPEED_TARGETS}"
        )
    return target


def speed_target_parameters(reward_config: Mapping[str, Any]) -> dict[str, float]:
    """Return the headway-cap parameters, validated."""

    parameters = {
        "v_nominal": float(
            reward_config.get("speed_target_nominal", DEFAULT_SPEED_TARGET_NOMINAL)
        ),
        "timegap": float(
            reward_config.get("speed_target_timegap", reward_config.get("timegap", 1.5))
        ),
        "standstill_gap_m": float(
            reward_config.get(
                "speed_target_standstill_gap_m", DEFAULT_SPEED_TARGET_STANDSTILL_GAP_M
            )
        ),
        "lateral_sigma_m": float(
            reward_config.get(
                "speed_target_lateral_sigma_m", DEFAULT_SPEED_TARGET_LATERAL_SIGMA_M
            )
        ),
    }
    for key, value in parameters.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"speed target parameter {key} must be finite and positive")
    return parameters


def _lateral_relevance(clearance_m: float, sigma_m: float) -> float:
    """Smooth membership of the ego's corridor; 1 when overlapping, 0 when clear.

    ``clearance_m`` is ``|dy| - 0.5*(W_ego + W_other)``: negative when the two
    footprints overlap laterally, positive once they are clear. A vehicle whose
    footprint overlaps constrains the ego fully, so the kernel is exactly 1
    there and decays as a Gaussian in the clearance beyond it. A plain sigmoid
    would leak: at ``dy = 0`` it returns 0.982, so the cap would never apply at
    full strength directly behind a leader.
    """

    clearance = float(clearance_m)
    if clearance <= 0.0:
        return 1.0
    return math.exp(-((clearance / float(sigma_m)) ** 2))


def linear_tracking_weights(reward_config: Mapping[str, Any]) -> dict[str, float]:
    """Return the normalized per-cost weights ``w_i / sum_i w_i``."""

    weights = {cost: float(reward_config.get(key, 0.0)) for cost, key in _TRACKING_COSTS}
    for cost, weight in weights.items():
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Linear tracking weight for {cost} must be finite and >= 0")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("Linear tracking weights must have a positive sum")
    return {cost: weight / total for cost, weight in weights.items()}


def linear_tracking_reward(
    costs: Mapping[str, float], reward_config: Mapping[str, Any]
) -> float:
    """Weighted mean of ``1 - clip(c_i, 0, 1)``; always in ``[0, 1]``."""

    return float(
        sum(
            alpha * (1.0 - min(max(float(costs[cost]), 0.0), 1.0))
            for cost, alpha in linear_tracking_weights(reward_config).items()
        )
    )


def event_reward(components: Mapping[str, float], reward_config: Mapping[str, Any]) -> float:
    """Collision penalty, else a single overtake bonus, as in the notebook."""

    if float(components["ego_collision"]) > 0.5:
        return float(reward_config["collision_penalty"])
    overtakes = float(components["overtakes"])
    if overtakes > 0.0:
        return float(reward_config["overtake_bonus"]) * min(overtakes, 1.0)
    return 0.0


def linear_reward_terms(
    components: Mapping[str, float], reward_config: Mapping[str, Any]
) -> dict[str, float]:
    """Return the linear-mode reward and its additive terms."""

    tracking = linear_tracking_reward(components, reward_config)
    progress = float(components["progress_reward"])
    jerk = float(components["jerk_penalty"])
    event = event_reward(components, reward_config)
    return {
        "linear_tracking_reward": tracking,
        "event_reward": event,
        "reward": tracking + progress - jerk + event,
    }


def make_linear_reward_wrapper(base_wrapper: type) -> type:
    """Subclass a Karalakou reward wrapper so it returns the linear reward.

    ``base_wrapper`` must expose ``_karalakou_reward(previous_dx,
    previous_ego_x)`` returning ``(reward, components)`` with the notebook's
    component names. Every base component is kept; ``reward`` is replaced and
    the base reciprocal total is kept as ``reciprocal_mode_reward``.
    """

    class LinearKaralakouRewardWrapper(base_wrapper):  # type: ignore[misc, valid-type]
        def _karalakou_reward(self, previous_dx, previous_ego_x=None):
            reciprocal_total, components = super()._karalakou_reward(
                previous_dx, previous_ego_x
            )
            terms = linear_reward_terms(components, self.reward_config)
            components = dict(components)
            components["reciprocal_mode_reward"] = float(reciprocal_total)
            components.update(terms)
            return terms["reward"], components

    LinearKaralakouRewardWrapper.__qualname__ = (
        f"LinearKaralakouRewardWrapper[{base_wrapper.__qualname__}]"
    )
    return LinearKaralakouRewardWrapper


def resolve_lateral_target(reward_config: Mapping[str, Any]) -> str:
    """Return how the lateral target is computed, defaulting to the notebook's."""

    target = (
        str(reward_config.get("lateral_target", GAP_LATERAL_TARGET)).strip().lower()
    )
    if target not in SUPPORTED_LATERAL_TARGETS:
        raise ValueError(
            f"Unsupported lateral_target {target!r}; expected one of "
            f"{SUPPORTED_LATERAL_TARGETS}"
        )
    return target


def lateral_field_parameters(reward_config: Mapping[str, Any]) -> dict[str, float]:
    """Return the field-target parameters, validated.

    The closing horizon and standstill gap are shared with the gap target's
    blocker test, so a run cannot weight relevance one way in one target and
    another way in the other.
    """

    shared = lateral_blocker_parameters(reward_config)
    parameters = {
        "horizon_s": shared["horizon_s"],
        "standstill_gap_m": shared["standstill_gap_m"],
        "grid_step_m": float(
            reward_config.get(
                "lateral_field_grid_step_m", DEFAULT_LATERAL_FIELD_GRID_STEP_M
            )
        ),
        "sigma_m": float(
            reward_config.get("lateral_field_sigma_m", DEFAULT_LATERAL_FIELD_SIGMA_M)
        ),
        "beta": float(reward_config.get("lateral_field_beta", DEFAULT_LATERAL_FIELD_BETA)),
        "wall_sigma_m": float(
            reward_config.get(
                "lateral_field_wall_sigma_m", DEFAULT_LATERAL_FIELD_WALL_SIGMA_M
            )
        ),
    }
    for key, value in parameters.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"lateral field parameter {key} must be finite and positive")
    for key, default in (
        ("travel_weight", DEFAULT_LATERAL_FIELD_TRAVEL_WEIGHT),
        ("basin_margin", DEFAULT_LATERAL_FIELD_BASIN_MARGIN),
        ("hysteresis", DEFAULT_LATERAL_FIELD_HYSTERESIS),
        ("wall_weight", DEFAULT_LATERAL_FIELD_WALL_WEIGHT),
    ):
        value = float(reward_config.get(f"lateral_field_{key}", default))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"lateral field parameter {key} must be finite and non-negative"
            )
        parameters[key] = value
    return parameters


def make_field_target_wrapper(base_wrapper: type) -> type:
    """Subclass a Karalakou reward wrapper so the lateral target is a smooth field.

    The gap target answers a yes/no question -- is there an interval clear of
    every blocker -- and needs a fallback for the answer "no", which at 40
    vehicles is 94% of steps with the notebook blocker set and 2.5% with the
    closing set. This target answers a graded question instead: for every
    lateral position, how blocked is it right now, and how far would the ego
    have to travel to reach it. There is no branch and no fallback.

    For each vehicle in sensing range, its relevance decays with distance
    measured against the same closing reach the gap target uses,

        reach_i     = d0 + T * max(closing speed toward the ego, 0)
        relevance_i = exp(-(|dx_i| / reach_i)^2)

    which is smooth in both the distance and the speeds: a vehicle pulling away
    fades out, one closing fast fades in, and nothing enters or leaves at a
    range boundary. Its lateral footprint spreads over the road as

        occupancy_i(y) = relevance_i * exp(-(clearance_i(y) / sigma)^2)

    with ``clearance_i(y) = max(|y - y_i| - (W_ego + W_i)/2, 0)``, so a position
    the footprints overlap costs the full relevance and the cost decays outside
    it. Summed over vehicles this is an occupancy field over the road. The
    score adds a travel term, ``travel_weight * ((y - y_ego)/road_width)^2``,
    which is what breaks the tie between two equally free sides: without it the
    weighted mean of two symmetric free zones is the blocked middle between
    them.

    The target is the softmin of that score, restricted to the basin around its
    best point: the grid point with the lowest score, extended left and right
    while the score stays within ``basin_margin`` of it, then averaged with
    weights ``exp(-beta * score)``. The restriction is what keeps the target
    out of a blocked middle; a very large ``basin_margin`` recovers the
    unrestricted softmin over the whole road. Inside a basin the target moves
    continuously with the traffic -- sweeping a leader across the road moves it
    in steps under 0.05 m -- but switching basins is a discrete event and the
    target jumps the width of the road when the freest side swaps, including
    when a receding leader stops being worth avoiding. That is the residual
    discontinuity of this target, measured at 0.89% of steps against 2.13% for
    the closing gap target and 2.36% for the notebook one (30 episodes, CBF
    OFF, 2026-09-12).

    The worst case is exact symmetry: with the field symmetric about the ego the
    two sides are tied and the choice flip-flops on floating-point noise, ten
    times over a synthetic sweep of a leader receding from 5 m to 60 m.
    ``hysteresis`` fixes that by charging for moving the target away from where
    it already is, which removed every flip in that sweep and in three
    asymmetric variants of it at weight 2.0, holding the largest step at 0.08 m.
    It makes the target depend on the previous target, so the wrapper carries it
    across a step and clears it on ``reset``; that is Markov because the target
    is itself part of the observation. It defaults to 0, which keeps the target a
    pure function of the current state.

    ``zone_found`` is always True, so the ``--lateral-target-fallback`` and
    ``--lateral-blockers`` settings are inert under this target. The speed
    target is the notebook's nominal ``ego.desired_speed``.
    """

    class FieldTargetKaralakouRewardWrapper(base_wrapper):  # type: ignore[misc, valid-type]
        def reset(self, **kwargs):
            self._field_previous_target_y: float | None = None
            return super().reset(**kwargs)

        def _lateral_occupancy_field(self, grid):
            parameters = lateral_field_parameters(self.reward_config)
            base = self.base_env
            ego = base.vehicle
            ego_x = float(ego.position[0])
            ego_vx = float(ego.vx)
            ego_width = float(ego.width)
            sensing_range = float(base.config["sensing_range"])
            horizon = parameters["horizon_s"]
            standstill_gap = parameters["standstill_gap_m"]
            sigma = parameters["sigma_m"]
            occupancy = numpy.zeros_like(grid)
            for vehicle in base.road.vehicles:
                if vehicle is ego:
                    continue
                dx = float(base._signed_distance(ego_x, float(vehicle.position[0])))
                if abs(dx) > sensing_range:
                    continue
                other_vx = float(vehicle.vx)
                closing = ego_vx - other_vx if dx > 0.0 else other_vx - ego_vx
                reach = standstill_gap + horizon * max(closing, 0.0)
                relevance = math.exp(-((abs(dx) / reach) ** 2))
                half_span = 0.5 * (ego_width + float(vehicle.width))
                clearance = numpy.maximum(
                    numpy.abs(grid - float(vehicle.position[1])) - half_span, 0.0
                )
                occupancy += relevance * numpy.exp(-((clearance / sigma) ** 2))
            wall_weight = parameters["wall_weight"]
            if wall_weight > 0.0:
                # A boundary is an obstacle the field could not previously see.
                # Clearance is measured from the ego's own edge to the wall, the
                # same way a vehicle's clearance is measured edge-to-edge, so the
                # two contributions are directly comparable.
                road_width = float(base.config["road_width"])
                half_width = 0.5 * ego_width
                wall_sigma = parameters["wall_sigma_m"]
                left_clearance = numpy.maximum(grid - half_width, 0.0)
                right_clearance = numpy.maximum(road_width - half_width - grid, 0.0)
                occupancy += wall_weight * (
                    numpy.exp(-((left_clearance / wall_sigma) ** 2))
                    + numpy.exp(-((right_clearance / wall_sigma) ** 2))
                )
            return occupancy

        def _field_lateral_target(self) -> float:
            parameters = lateral_field_parameters(self.reward_config)
            base = self.base_env
            ego = base.vehicle
            road_width = float(base.config["road_width"])
            y_min = float(ego.width) / 2.0
            y_max = road_width - float(ego.width) / 2.0
            grid = numpy.arange(
                y_min, y_max + 0.5 * parameters["grid_step_m"], parameters["grid_step_m"]
            )
            ego_y = float(ego.position[1])
            score = self._lateral_occupancy_field(grid) + parameters["travel_weight"] * (
                ((grid - ego_y) / road_width) ** 2
            )
            previous = self.__dict__.get("_field_previous_target_y")
            if parameters["hysteresis"] > 0.0 and previous is not None:
                # Commitment: moving the target away from where it already is
                # costs something, so two nearly tied sides cannot swap back and
                # forth. Without this the choice flip-flops whenever the field is
                # symmetric about the ego.
                score = score + parameters["hysteresis"] * (
                    ((grid - float(previous)) / road_width) ** 2
                )
            best = int(numpy.argmin(score))
            threshold = float(score[best]) + parameters["basin_margin"]
            low = best
            while low > 0 and score[low - 1] <= threshold:
                low -= 1
            high = best
            while high + 1 < score.size and score[high + 1] <= threshold:
                high += 1
            basin = slice(low, high + 1)
            local = score[basin]
            weights = numpy.exp(-parameters["beta"] * (local - local.min()))
            target_y = float(numpy.dot(weights, grid[basin]) / weights.sum())
            target_y = float(min(max(target_y, y_min), y_max))
            self._field_previous_target_y = target_y
            return target_y

        def _lateral_target_and_speed(self):
            ego = self.base_env.vehicle
            target_speed = float(max(float(ego.desired_speed), 0.0))
            return self._field_lateral_target(), target_speed, True

    FieldTargetKaralakouRewardWrapper.__qualname__ = (
        f"FieldTargetKaralakouRewardWrapper[{base_wrapper.__qualname__}]"
    )
    return FieldTargetKaralakouRewardWrapper


def make_closing_blockers_wrapper(base_wrapper: type) -> type:
    """Subclass a Karalakou reward wrapper so only closing vehicles block a gap.

    The notebook treats every vehicle with ``0 < dx < sensing_range`` as a
    blocker and subtracts a band of ``ego.width + other.width + 2*zone_margin``
    from the free corridor for each one. At 40 vehicles on the 380 m ring about
    9.2 vehicles sit inside the 90 m forward window, so roughly 36 m of band is
    subtracted from an 8.4 m corridor and no gap survives on about 94% of steps
    (6.4% measured, 30 episodes, CBF OFF, 2026-09-12). The search asks for a
    lateral position clear of every vehicle in the next 90 m at once, which is a
    far stronger condition than "there is somewhere to go now".

    This wrapper keeps the band geometry, the nearest-gap choice, and the
    fallback, and changes only which vehicles are in the blocker set: a vehicle
    counts while the gap to it is closing inside ``horizon_s``,

        keep ahead  iff   dx < d0 + T * max(v_ego - v_other, 0)
        keep behind iff  -dx < d0 + T * max(v_other - v_ego, 0)

    so a vehicle ahead that is pulling away stops constraining the target and a
    faster vehicle overtaking from behind starts constraining it. The rear half
    matters because the ego averages 12 to 14 m/s against traffic at 15 to
    25 m/s, and its collisions come mostly from behind or from the side rather
    than from a leader it drove into. ``closing_forward`` keeps the
    forward-only view of the notebook, for the ablation.

    ``d0`` keeps a vehicle alongside relevant when the closing speed is zero.
    The blocker set is a step function of the state, so the selected gap can
    still change discretely. The speed target is reproduced exactly as the
    notebook computes it, the nominal ``ego.desired_speed``; the tests assert
    that against the base wrapper on a real rollout.
    """

    class ClosingBlockersKaralakouRewardWrapper(base_wrapper):  # type: ignore[misc, valid-type]
        def _relevant_blockers(self) -> list:
            parameters = lateral_blocker_parameters(self.reward_config)
            horizon = parameters["horizon_s"]
            standstill_gap = parameters["standstill_gap_m"]
            include_rear = (
                resolve_lateral_blockers(self.reward_config)
                == CLOSING_LATERAL_BLOCKERS
            )
            base = self.base_env
            ego = base.vehicle
            ego_x = float(ego.position[0])
            ego_vx = float(ego.vx)
            sensing_range = float(base.config["sensing_range"])
            blockers = []
            for vehicle in base.road.vehicles:
                if vehicle is ego:
                    continue
                other_vx = float(vehicle.vx)
                ahead = float(base._forward_distance(ego_x, float(vehicle.position[0])))
                if 0.0 < ahead < sensing_range:
                    reach = standstill_gap + horizon * max(ego_vx - other_vx, 0.0)
                    if ahead < reach:
                        blockers.append((ahead, vehicle))
                    continue
                if not include_rear:
                    continue
                behind = float(base._forward_distance(float(vehicle.position[0]), ego_x))
                if 0.0 < behind < sensing_range:
                    reach = standstill_gap + horizon * max(other_vx - ego_vx, 0.0)
                    if behind < reach:
                        blockers.append((behind, vehicle))
            return blockers

        def _lateral_target_and_speed(self):
            base = self.base_env
            ego = base.vehicle
            config = self.reward_config
            road_width = float(base.config["road_width"])
            target_speed = float(max(float(ego.desired_speed), 0.0))
            blockers = self._relevant_blockers()
            if not blockers:
                return road_width / 2.0, target_speed, True

            y_min = float(ego.width) / 2.0
            y_max = road_width - float(ego.width) / 2.0
            margin = float(config.get("zone_margin", 0.15))
            free_intervals = [(y_min, y_max)]
            for _, vehicle in blockers:
                half_span = 0.5 * (float(ego.width) + float(vehicle.width)) + margin
                blocked_low = max(float(vehicle.position[1]) - half_span, y_min)
                blocked_high = min(float(vehicle.position[1]) + half_span, y_max)
                free_intervals = self._subtract_interval(
                    free_intervals, blocked_low, blocked_high
                )
            free_intervals = [(lo, hi) for lo, hi in free_intervals if hi - lo > 0.05]

            if free_intervals:
                ego_y = float(ego.position[1])

                def interval_distance(interval):
                    low, high = interval
                    if low <= ego_y <= high:
                        return 0.0
                    return min(abs(ego_y - low), abs(ego_y - high))

                best_low, best_high = min(free_intervals, key=interval_distance)
                return 0.5 * (best_low + best_high), target_speed, True

            fastest = max(blockers, key=lambda item: float(item[1].vx))[1]
            target_y = min(max(float(fastest.position[1]), y_min), y_max)
            return float(target_y), target_speed, False

    ClosingBlockersKaralakouRewardWrapper.__qualname__ = (
        f"ClosingBlockersKaralakouRewardWrapper[{base_wrapper.__qualname__}]"
    )
    return ClosingBlockersKaralakouRewardWrapper


def make_center_fallback_wrapper(base_wrapper: type) -> type:
    """Subclass a Karalakou reward wrapper so a missing gap targets the road centre.

    ``base_wrapper`` must expose ``_lateral_target_and_speed()`` returning
    ``(target_y, target_speed, zone_found)`` and ``base_env.config["road_width"]``.
    Only ``target_y`` on no-gap steps changes; ``zone_found`` stays ``False``
    so logs still show which steps used the fallback.
    """

    class CenterFallbackKaralakouRewardWrapper(base_wrapper):  # type: ignore[misc, valid-type]
        def _lateral_target_and_speed(self):
            target_y, target_speed, zone_found = super()._lateral_target_and_speed()
            if not zone_found:
                target_y = 0.5 * float(self.base_env.config["road_width"])
            return float(target_y), target_speed, zone_found

    CenterFallbackKaralakouRewardWrapper.__qualname__ = (
        f"CenterFallbackKaralakouRewardWrapper[{base_wrapper.__qualname__}]"
    )
    return CenterFallbackKaralakouRewardWrapper


def make_headway_speed_target_wrapper(base_wrapper: type) -> type:
    """Subclass a Karalakou reward wrapper so the speed target is feasibility-capped.

    ``base_wrapper`` must expose ``_lateral_target_and_speed()``,
    ``_karalakou_reward(previous_dx, previous_ego_x)``, and ``base_env`` with
    ``vehicle``, ``road.vehicles``, ``_forward_distance`` and
    ``config["sensing_range"]``, as the notebook wrapper does.

    Two things change and nothing else:

    1. ``target_speed`` becomes ``min(v_nom, min_i v_eff_i)`` over the vehicles
       ahead, where ``v_allow_i = max((dx_i - d0)/T, 0)`` is the
       constant-time-headway speed and ``v_eff_i = w_i*v_allow_i +
       (1-w_i)*v_nom`` fades that constraint out by lateral relevance ``w_i``,
       which is 1 while the footprints overlap and Gaussian in the clearance
       beyond them. Applying the weight
       inside the allowance, rather than to an aggregation over vehicles, is
       what keeps a laterally irrelevant vehicle from constraining the target:
       a car 2 m ahead but laterally clear has ``v_allow = 0``, and under a
       softmin its weight swamps any lateral damping.
    2. ``cx`` is normalized by the fixed ``v_nom`` instead of by the live
       target. A moving target in the denominator diverges exactly when traffic
       forces the target toward zero.

    The target is continuous in the traffic configuration. Vehicles enter and
    leave through ``w_i``, which is continuous, and the cap can only bind below
    ``dx = d0 + T*v_nom`` (35 m at the defaults), well inside the 90 m sensing
    range, so nothing changes at the range boundary. That discontinuity is why
    the earlier blocker-derived target was replaced by a fixed one.

    ``ego.desired_speed`` is left alone, so spawn and reset states are
    identical to a nominal-target run on the same seed.
    """

    class HeadwaySpeedTargetKaralakouRewardWrapper(base_wrapper):  # type: ignore[misc, valid-type]
        def _headway_speed_target(self) -> float:
            parameters = speed_target_parameters(self.reward_config)
            v_nominal = parameters["v_nominal"]
            base = self.base_env
            ego = base.vehicle
            sensing_range = float(base.config["sensing_range"])
            target = v_nominal
            for vehicle in base.road.vehicles:
                if vehicle is ego:
                    continue
                dx = float(base._forward_distance(ego.position[0], vehicle.position[0]))
                if not (0.0 < dx < sensing_range):
                    continue
                dy = float(vehicle.position[1] - ego.position[1])
                clearance = abs(dy) - 0.5 * (float(ego.width) + float(vehicle.width))
                weight = _lateral_relevance(clearance, parameters["lateral_sigma_m"])
                allowed = max(
                    (dx - parameters["standstill_gap_m"]) / parameters["timegap"], 0.0
                )
                target = min(target, weight * allowed + (1.0 - weight) * v_nominal)
            return float(min(max(target, 0.0), v_nominal))

        def _lateral_target_and_speed(self):
            target_y, _, zone_found = super()._lateral_target_and_speed()
            return float(target_y), self._headway_speed_target(), zone_found

        def _karalakou_reward(self, previous_dx, previous_ego_x=None):
            _, components = super()._karalakou_reward(previous_dx, previous_ego_x)
            config = self.reward_config
            v_nominal = speed_target_parameters(config)["v_nominal"]
            components = dict(components)
            # The base wrapper normalized cx by the live target; keep that value
            # for provenance and replace cx with the fixed-normalizer form.
            components["target_normalized_cx"] = float(components["cx"])
            components["speed_target_nominal"] = float(v_nominal)
            cx = abs(
                float(components["ego_speed"]) - float(components["target_speed"])
            ) / v_nominal
            components["cx"] = float(cx)
            denominator = (
                float(config["epsilon_r"])
                + float(config["wx"]) * cx
                + float(config["wy"]) * float(components["cy"])
                + float(config["wf"]) * float(components["cf"])
                + float(config.get("way", 0.0)) * float(components["cay"])
            )
            reward = (
                float(config["epsilon_r"]) / max(denominator, 1e-9)
                + float(components["progress_reward"])
                - float(components["jerk_penalty"])
                + event_reward(components, config)
            )
            components["reward"] = float(reward)
            return float(reward), components

    HeadwaySpeedTargetKaralakouRewardWrapper.__qualname__ = (
        f"HeadwaySpeedTargetKaralakouRewardWrapper[{base_wrapper.__qualname__}]"
    )
    return HeadwaySpeedTargetKaralakouRewardWrapper


def make_latched_overtake_wrapper(base_wrapper: type) -> type:
    """Subclass a Karalakou reward wrapper with the latched overtake detector.

    ``base_wrapper`` must expose ``reset``, ``_overtake_count(previous_dx)``,
    and ``base_env`` with ``vehicle``, ``road.vehicles``, ``_signed_distance``
    and ``config["sensing_range"]``, as the notebook wrapper does.
    """

    class LatchedOvertakeKaralakouRewardWrapper(base_wrapper):  # type: ignore[misc, valid-type]
        def reset(self, **kwargs):
            self._overtake_seen_ahead: set[int] = set()
            self._overtake_counted: set[int] = set()
            return super().reset(**kwargs)

        def _overtake_count(self, previous_dx):
            base = self.base_env
            ego = base.vehicle
            sensing_range = float(base.config["sensing_range"])
            seen_ahead = self.__dict__.setdefault("_overtake_seen_ahead", set())
            counted = self.__dict__.setdefault("_overtake_counted", set())
            seen_ahead.update(
                key for key, dx in previous_dx.items() if 0.0 < float(dx) < sensing_range
            )
            overtakes = 0
            for vehicle in base.road.vehicles:
                if vehicle is ego:
                    continue
                key = id(vehicle)
                if key not in seen_ahead or key in counted:
                    continue
                new_dx = float(base._signed_distance(ego.position[0], vehicle.position[0]))
                if -sensing_range < new_dx < -0.5 * float(ego.length):
                    counted.add(key)
                    overtakes += 1
            return overtakes

    LatchedOvertakeKaralakouRewardWrapper.__qualname__ = (
        f"LatchedOvertakeKaralakouRewardWrapper[{base_wrapper.__qualname__}]"
    )
    return LatchedOvertakeKaralakouRewardWrapper


def wrap_reward_variants(base_wrapper: type, reward_config: Mapping[str, Any]) -> type:
    """Apply every opt-in variant in ``reward_config`` to the notebook wrapper."""

    wrapper = base_wrapper
    # Innermost: the centre fallback below reads the zone_found this produces.
    if resolve_lateral_target(reward_config) == FIELD_LATERAL_TARGET:
        # The field target has no gap search, so the blocker set does not apply.
        wrapper = make_field_target_wrapper(wrapper)
    elif resolve_lateral_blockers(reward_config) != ALL_LATERAL_BLOCKERS:
        wrapper = make_closing_blockers_wrapper(wrapper)
    if resolve_lateral_target_fallback(reward_config) == CENTER_LATERAL_FALLBACK:
        wrapper = make_center_fallback_wrapper(wrapper)
    if resolve_overtake_detection(reward_config) == LATCHED_OVERTAKE_DETECTION:
        wrapper = make_latched_overtake_wrapper(wrapper)
    # Before the linear wrapper: the linear tracking term reads components["cx"],
    # which this wrapper renormalizes, so it must already have run.
    if resolve_speed_target(reward_config) == HEADWAY_SPEED_TARGET:
        wrapper = make_headway_speed_target_wrapper(wrapper)
    if resolve_reward_mode(reward_config) == LINEAR_REWARD_MODE:
        wrapper = make_linear_reward_wrapper(wrapper)
    return wrapper


def add_reward_variant_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the opt-in reward-variant flags; omitted flags keep the notebook reward."""

    parser.add_argument(
        "--reward-mode",
        choices=SUPPORTED_REWARD_MODES,
        default=None,
        help=(
            "Tracking term of the base reward: the notebook reciprocal "
            "eps/(eps+sum w_i c_i), or the linear weighted mean "
            "sum w_i (1-c_i)/sum w_i from saferl.rewards; omitted means reciprocal."
        ),
    )
    parser.add_argument(
        "--lateral-target-fallback",
        choices=SUPPORTED_LATERAL_FALLBACKS,
        default=None,
        help=(
            "Lateral target when no free gap exists ahead: the notebook's "
            "fastest-blocker y, or the road center; omitted means fastest_blocker."
        ),
    )
    parser.add_argument(
        "--overtake-detection",
        choices=SUPPORTED_OVERTAKE_DETECTIONS,
        default=None,
        help=(
            "Overtake detector for the overtake bonus: the notebook's one-step "
            "test (never fires at 20 Hz), or latched (seen ahead earlier, now "
            "half an ego length behind, once per vehicle per episode); omitted "
            "means step."
        ),
    )
    parser.add_argument(
        "--speed-target",
        choices=SUPPORTED_SPEED_TARGETS,
        default=None,
        help=(
            "Speed target of the cx cost: the notebook's fixed ego_desired_speed, "
            "or headway, which caps a free-flow nominal by the constant-time-headway "
            "speed of the vehicles ahead (faded by lateral relevance) and normalizes "
            "cx by that nominal; omitted means nominal."
        ),
    )
    parser.add_argument(
        "--speed-target-nominal",
        type=float,
        default=None,
        help=(
            "Free-flow speed the headway cap applies to, in m/s "
            f"(default {DEFAULT_SPEED_TARGET_NOMINAL}). Requires --speed-target headway."
        ),
    )
    parser.add_argument(
        "--lateral-target",
        choices=SUPPORTED_LATERAL_TARGETS,
        default=None,
        help=(
            "How the lateral target is computed: gap, the notebook's nearest free "
            "interval with a fallback, or field, a smooth softmin of a "
            "relevance-weighted occupancy field with no fallback. Under field the "
            "--lateral-blockers and --lateral-target-fallback settings are inert; "
            "omitted means gap."
        ),
    )
    parser.add_argument(
        "--lateral-field-sigma-m",
        type=float,
        default=None,
        help=(
            "Lateral decay of a vehicle's occupancy outside its footprint, in m "
            f"(default {DEFAULT_LATERAL_FIELD_SIGMA_M}). Requires --lateral-target field."
        ),
    )
    parser.add_argument(
        "--lateral-field-beta",
        type=float,
        default=None,
        help=(
            "Softmin sharpness of the field target "
            f"(default {DEFAULT_LATERAL_FIELD_BETA}). Requires --lateral-target field."
        ),
    )
    parser.add_argument(
        "--lateral-field-wall-weight",
        type=float,
        default=None,
        help=(
            "Occupancy each road boundary contributes to the field, in the same "
            "units as a vehicle's. Without it the field sums vehicles only, so a "
            "wall reads as free space and the target is pulled into it "
            f"(default {DEFAULT_LATERAL_FIELD_WALL_WEIGHT}, i.e. off). "
            "Requires --lateral-target field."
        ),
    )
    parser.add_argument(
        "--lateral-field-wall-sigma-m",
        type=float,
        default=None,
        help=(
            "Decay of the boundary occupancy with clearance from the ego's edge, "
            f"in m (default {DEFAULT_LATERAL_FIELD_WALL_SIGMA_M}). "
            "Requires --lateral-target field."
        ),
    )
    parser.add_argument(
        "--lateral-field-travel-weight",
        type=float,
        default=None,
        help=(
            "Weight of the squared normalized distance from the ego's own y, which "
            f"breaks ties between equally free sides (default "
            f"{DEFAULT_LATERAL_FIELD_TRAVEL_WEIGHT}). Requires --lateral-target field."
        ),
    )
    parser.add_argument(
        "--lateral-field-hysteresis",
        type=float,
        default=None,
        help=(
            "Weight of the squared normalized distance from the previous target, "
            "which stops the target swapping between two nearly tied sides; 0 "
            f"keeps the target a pure function of the state (default "
            f"{DEFAULT_LATERAL_FIELD_HYSTERESIS}, 2.0 removes every flip in the "
            "sweeps). Requires --lateral-target field."
        ),
    )
    parser.add_argument(
        "--lateral-field-basin-margin",
        type=float,
        default=None,
        help=(
            "Score margin defining the basin the softmin averages over; large "
            "values recover the unrestricted softmin over the whole road "
            f"(default {DEFAULT_LATERAL_FIELD_BASIN_MARGIN}). Requires "
            "--lateral-target field."
        ),
    )
    parser.add_argument(
        "--lateral-blockers",
        choices=SUPPORTED_LATERAL_BLOCKERS,
        default=None,
        help=(
            "Which vehicles block a lateral gap: all in-range vehicles ahead, as "
            "the notebook has it, or only the ones the gap is closing on within "
            "the horizon (closing includes vehicles overtaking from behind, "
            "closing_forward keeps the forward-only view); omitted means all."
        ),
    )
    parser.add_argument(
        "--lateral-blocker-horizon-s",
        type=float,
        default=None,
        help=(
            "Closing horizon T of the blocker test, in seconds (default "
            f"{DEFAULT_LATERAL_BLOCKER_HORIZON_S}). Requires --lateral-blockers "
            "closing or closing_forward."
        ),
    )
    parser.add_argument(
        "--lateral-blocker-standstill-gap-m",
        type=float,
        default=None,
        help=(
            "Distance d0 inside which a vehicle blocks regardless of closing "
            f"speed, in m (default {DEFAULT_LATERAL_BLOCKER_STANDSTILL_GAP_M}). "
            "Requires --lateral-blockers closing or closing_forward."
        ),
    )
    parser.add_argument(
        "--potential-field-weight",
        type=float,
        default=None,
        help=(
            "Weight wf of the neighbour potential-field cost cf in the tracking "
            "denominator; 0 removes the term, leaving the reward purely a task "
            "reward. Omitted keeps the notebook value."
        ),
    )


def apply_reward_variant_arguments(args: argparse.Namespace, reward_config: dict[str, Any]) -> None:
    """Copy the reward-variant flags that were given into ``reward_config``."""

    for key in (
        "reward_mode",
        "lateral_target_fallback",
        "lateral_target",
        "lateral_blockers",
        "overtake_detection",
        "speed_target",
    ):
        value = getattr(args, key, None)
        if value is not None:
            reward_config[key] = str(value)
    nominal = getattr(args, "speed_target_nominal", None)
    if nominal is not None:
        if resolve_speed_target(reward_config) != HEADWAY_SPEED_TARGET:
            raise ValueError(
                "--speed-target-nominal has no effect without --speed-target headway"
            )
        reward_config["speed_target_nominal"] = float(nominal)
    for name, key in (
        ("lateral_blocker_horizon_s", "lateral_blocker_horizon_s"),
        ("lateral_blocker_standstill_gap_m", "lateral_blocker_standstill_gap_m"),
    ):
        value = getattr(args, name, None)
        if value is None:
            continue
        if resolve_lateral_blockers(reward_config) == ALL_LATERAL_BLOCKERS:
            raise ValueError(
                f"--{name.replace('_', '-')} has no effect without --lateral-blockers"
            )
        reward_config[key] = float(value)
    for name in (
        "lateral_field_sigma_m",
        "lateral_field_beta",
        "lateral_field_travel_weight",
        "lateral_field_basin_margin",
        "lateral_field_hysteresis",
        "lateral_field_wall_weight",
        "lateral_field_wall_sigma_m",
    ):
        value = getattr(args, name, None)
        if value is None:
            continue
        if resolve_lateral_target(reward_config) != FIELD_LATERAL_TARGET:
            raise ValueError(
                f"--{name.replace('_', '-')} has no effect without --lateral-target field"
            )
        reward_config[name] = float(value)
    potential_field_weight = getattr(args, "potential_field_weight", None)
    if potential_field_weight is not None:
        weight = float(potential_field_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("--potential-field-weight must be finite and >= 0")
        reward_config["wf"] = weight


def reward_variant_summary(reward_config: Mapping[str, Any]) -> dict[str, str]:
    """The effective reward variants, for run logs and study configs."""

    summary = {
        "reward_mode": str(reward_config.get("reward_mode", RECIPROCAL_REWARD_MODE)),
        "lateral_target_fallback": resolve_lateral_target_fallback(reward_config),
        "lateral_target": resolve_lateral_target(reward_config),
        "lateral_blockers": resolve_lateral_blockers(reward_config),
        "overtake_detection": resolve_overtake_detection(reward_config),
        "speed_target": resolve_speed_target(reward_config),
        "potential_field_weight": str(float(reward_config.get("wf", 0.0))),
    }
    if summary["lateral_target"] == FIELD_LATERAL_TARGET:
        summary.update(
            {
                f"lateral_field_{key}": str(value)
                for key, value in lateral_field_parameters(reward_config).items()
            }
        )
    if summary["lateral_blockers"] != ALL_LATERAL_BLOCKERS:
        summary.update(
            {
                f"lateral_blocker_{key}": str(value)
                for key, value in lateral_blocker_parameters(reward_config).items()
            }
        )
    if summary["speed_target"] == HEADWAY_SPEED_TARGET:
        summary.update(
            {
                f"speed_target_{key}": str(value)
                for key, value in speed_target_parameters(reward_config).items()
            }
        )
    return summary
