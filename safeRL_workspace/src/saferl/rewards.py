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
"""

from __future__ import annotations

import math
from typing import Any, Mapping

RECIPROCAL_REWARD_MODE = "reciprocal"
LINEAR_REWARD_MODE = "linear"
SUPPORTED_REWARD_MODES = (RECIPROCAL_REWARD_MODE, LINEAR_REWARD_MODE)

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
