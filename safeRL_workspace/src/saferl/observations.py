"""Opt-in observation variants for the laneless PPO interface.

Two variants, both off by default so no historical run changes.

**Dropping the ego row's desired-speed slot.** The ego row's final feature is
the *target* speed, which the reward wrapper writes as
``target_speed / observation_vmax``. Under the nominal speed target that value
is ``ego.desired_speed``, rewritten from the reward config on every reset, so
with a fixed ``ego_desired_speed`` the feature is the same constant on every
step of every episode. A constant input is absorbed exactly into the first
layer's bias, so it carries no information; dropping it removes the dead column
and makes the fixed-speed assumption explicit in the observation contract.

This is only valid while the speed target is constant. Under
``--speed-target headway`` the slot is dynamic and must be kept, which
``resolve_drop_ego_speed_slot`` refuses to let a caller get wrong silently.

**Exposing the state cost's two hinge terms.** The HOCBF state cost charges the
policy for proximity and closing rate but never tells it either quantity, so the
policy has to infer the penalty from the raw neighbour rows. These features hand
it the two terms the penalty is actually built from -- the squared hinged margin
on ``h`` and on ``psi1`` at the worst constraint, whose sum is ``C(s)``.

Two properties of that feature matter and are easy to get wrong:

*Timing.* The reward evaluates the cost on the state the action produced,
``s(t+1)``. An observation must instead be a function of ``s(t)``, so it is
computed from the same constraint system the observation itself is built from.
Reusing the reward's post-step value would leak the next state into the policy
input and quietly invalidate the comparison.

*Scale.* The hinge terms are squared and unbounded above, and the policy is a
Tanh MLP, so a rare near-collision would otherwise saturate the first layer and
swamp every other feature. They are clipped to an explicit bound that is
recorded in the manifest rather than left to chance.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np

from saferl.safety.state_cost import StateCostParameters, barrier_state_cost

DROP_EGO_SPEED_SLOT_KEY = "drop_ego_speed_slot"

STATE_COST_OBSERVE_KEY = "state_cost_observe"
STATE_COST_FEATURE_CLIP_KEY = "state_cost_feature_clip"
STATE_COST_FEATURE_PARAMS_KEY = "state_cost_feature_params"
STATE_COST_FEATURE_COUNT = 2
STATE_COST_FEATURE_NAMES = ("state_cost_h_term", "state_cost_psi1_term")
# A hinge of one scale past the margin gives 1.0, two scales gives 4.0. Clipping
# there keeps ordinary risk unsaturated while bounding the near-collision tail,
# which is the range where the feature stops carrying usable gradient anyway.
DEFAULT_STATE_COST_FEATURE_CLIP = 4.0


def resolve_state_cost_observe(env_config: Mapping[str, Any]) -> bool:
    """Return whether the state cost's hinge terms are exposed to the policy."""

    return bool(env_config.get(STATE_COST_OBSERVE_KEY, False))


def resolve_state_cost_feature_clip(env_config: Mapping[str, Any]) -> float:
    """Return the declared clip bound for the state-cost features."""

    value = env_config.get(STATE_COST_FEATURE_CLIP_KEY)
    clip = DEFAULT_STATE_COST_FEATURE_CLIP if value is None else float(value)
    if not np.isfinite(clip) or clip <= 0.0:
        raise ValueError(f"state-cost feature clip must be finite and positive; got {clip}")
    return clip


def resolve_state_cost_feature_params(
    env_config: Mapping[str, Any], fallback: StateCostParameters | None = None
) -> StateCostParameters:
    """Return the margins and scales the observation features are built from.

    These live in the environment configuration rather than being taken from
    whichever state cost the reward happens to be using, because evaluation
    deliberately runs with the penalty switched off. Reading the reward's copy
    would make an evaluation rebuild the features from default margins while
    training used the configured ones -- same observation width, different
    numbers in it, and nothing would fail loudly.
    """

    stored = env_config.get(STATE_COST_FEATURE_PARAMS_KEY)
    if stored:
        # Imported here to keep the module's import graph shallow for callers
        # that only need the drop-slot variant.
        from saferl.safety.state_cost import state_cost_parameters_from_mapping

        return state_cost_parameters_from_mapping(dict(stored))
    if fallback is not None:
        return fallback
    raise ValueError(
        f"{STATE_COST_FEATURE_PARAMS_KEY!r} is absent from the environment config "
        "and no fallback was supplied, so the state-cost features cannot be built "
        "with the margins the run was configured with"
    )


def state_cost_observation_features(
    h: Sequence[float] | np.ndarray,
    h_dot: Sequence[float] | np.ndarray,
    params: StateCostParameters,
    clip: float = DEFAULT_STATE_COST_FEATURE_CLIP,
) -> np.ndarray:
    """The state cost's two hinge terms at the worst constraint, clipped.

    ``h`` and ``h_dot`` must come from the constraint system of the *current*
    state, so that the features describe ``s(t)`` rather than the post-step
    system the reward is charged on.

    The parameters are the same margins and scales the reward uses, so the
    features and the penalty always agree about what counts as risk -- passing a
    different parameter set would describe a cost the policy is not paying.
    """

    if not np.isfinite(clip) or clip <= 0.0:
        raise ValueError(f"state-cost feature clip must be finite and positive; got {clip}")
    _cost, diagnostics = barrier_state_cost(h, h_dot, params)
    return np.clip(
        np.asarray(
            [diagnostics["h_term"], diagnostics["psi1_term"]], dtype=np.float32
        ),
        0.0,
        float(clip),
    )


def ego_speed_slot_index(env_config: Mapping[str, Any]) -> int:
    """Flat index of the ego row's desired-speed feature.

    The ego is row 0, so the slot is the last feature of the first row
    regardless of ``neighbors_count``.
    """

    include_dimensions = bool(
        env_config.get("observation_include_vehicle_dimensions", True)
    )
    features_per_row = 7 if include_dimensions else 5
    return features_per_row - 1


def resolve_drop_ego_speed_slot(env_config: Mapping[str, Any]) -> bool:
    """Return whether the ego desired-speed slot should be dropped."""

    return bool(env_config.get(DROP_EGO_SPEED_SLOT_KEY, False))


def make_drop_ego_speed_slot_wrapper(base_wrapper: type, env_config: Mapping[str, Any]) -> type:
    """Subclass a reward wrapper so the ego desired-speed slot is removed.

    The deletion happens inside ``_augment_observation``, i.e. after the base
    wrapper has written the ego y, target y and target speed slots, and before
    any previous-action features are appended. The observation space is
    narrowed by the same one element.
    """

    slot = int(ego_speed_slot_index(env_config))

    class DropEgoSpeedSlotWrapper(base_wrapper):  # type: ignore[misc, valid-type]
        ego_speed_slot_index = slot

        def __init__(self, env: Any, reward_config: dict[str, float] | None = None) -> None:
            super().__init__(env, reward_config=reward_config)
            low = np.asarray(self.observation_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(self.observation_space.high, dtype=np.float32).reshape(-1)
            if low.size <= slot:
                raise ValueError(
                    f"observation of width {low.size} has no ego speed slot at {slot}"
                )
            self.observation_space = gym.spaces.Box(
                low=np.delete(low, slot),
                high=np.delete(high, slot),
                dtype=np.float32,
            )

        def _augment_observation(self, obs: np.ndarray) -> np.ndarray:
            augmented = np.asarray(
                super()._augment_observation(obs), dtype=np.float32
            ).reshape(-1)
            if augmented.size <= slot:
                raise ValueError(
                    f"augmented observation of width {augmented.size} has no ego "
                    f"speed slot at {slot}"
                )
            return np.delete(augmented, slot).astype(np.float32)

    DropEgoSpeedSlotWrapper.__module__ = __name__
    DropEgoSpeedSlotWrapper.__name__ = "DropEgoSpeedSlotWrapper"
    return DropEgoSpeedSlotWrapper
