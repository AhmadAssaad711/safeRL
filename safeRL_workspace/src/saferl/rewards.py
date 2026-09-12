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

RECIPROCAL_REWARD_MODE = "reciprocal"
LINEAR_REWARD_MODE = "linear"
SUPPORTED_REWARD_MODES = (RECIPROCAL_REWARD_MODE, LINEAR_REWARD_MODE)

FASTEST_BLOCKER_LATERAL_FALLBACK = "fastest_blocker"
CENTER_LATERAL_FALLBACK = "center"
SUPPORTED_LATERAL_FALLBACKS = (FASTEST_BLOCKER_LATERAL_FALLBACK, CENTER_LATERAL_FALLBACK)

STEP_OVERTAKE_DETECTION = "step"
LATCHED_OVERTAKE_DETECTION = "latched"
SUPPORTED_OVERTAKE_DETECTIONS = (STEP_OVERTAKE_DETECTION, LATCHED_OVERTAKE_DETECTION)

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

    for key in ("reward_mode", "lateral_target_fallback", "overtake_detection", "speed_target"):
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
        "overtake_detection": resolve_overtake_detection(reward_config),
        "speed_target": resolve_speed_target(reward_config),
        "potential_field_weight": str(float(reward_config.get("wf", 0.0))),
    }
    if summary["speed_target"] == HEADWAY_SPEED_TARGET:
        summary.update(
            {
                f"speed_target_{key}": str(value)
                for key, value in speed_target_parameters(reward_config).items()
            }
        )
    return summary
