"""Opt-in observation variants for the laneless PPO interface.

Currently one variant: dropping the ego row's desired-speed slot.

The ego row's final feature is the *target* speed, which the reward wrapper
writes as ``target_speed / observation_vmax``. Under the nominal speed target
that value is ``ego.desired_speed``, rewritten from the reward config on every
reset, so with a fixed ``ego_desired_speed`` the feature is the same constant on
every step of every episode. A constant input is absorbed exactly into the first
layer's bias, so it carries no information; dropping it removes the dead column
and makes the fixed-speed assumption explicit in the observation contract.

This is only valid while the speed target is constant. Under
``--speed-target headway`` the slot is dynamic and must be kept, which
``resolve_drop_ego_speed_slot`` refuses to let a caller get wrong silently.
"""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import numpy as np

DROP_EGO_SPEED_SLOT_KEY = "drop_ego_speed_slot"


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
