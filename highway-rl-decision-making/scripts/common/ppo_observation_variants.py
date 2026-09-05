"""Small, picklable observation variants shared by PPO pilot workers."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np


def install_previous_action_observation(namespace: dict[str, Any]) -> None:
    """Append the previous normalized executed ego command to the y-target state."""

    original_wrapper = namespace["KaralakouRewardWrapper"]

    class PreviousActionObservationWrapper(original_wrapper):  # type: ignore[misc, valid-type]
        def __init__(
            self, env: Any, reward_config: Optional[dict[str, float]] = None
        ) -> None:
            super().__init__(env, reward_config=reward_config)
            low = np.asarray(self.observation_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(self.observation_space.high, dtype=np.float32).reshape(-1)
            self.observation_space = gym.spaces.Box(
                low=np.concatenate((low, np.full(2, -1.0, dtype=np.float32))),
                high=np.concatenate((high, np.full(2, 1.0, dtype=np.float32))),
                dtype=np.float32,
            )

        @staticmethod
        def _append_previous_action(
            observation: np.ndarray, previous_action: np.ndarray
        ) -> np.ndarray:
            base_observation = np.asarray(observation, dtype=np.float32).reshape(-1)
            action = np.asarray(previous_action, dtype=np.float32).reshape(-1)
            if action.size < 2:
                action = np.pad(action, (0, 2 - action.size))
            return np.concatenate(
                (base_observation, np.clip(action[:2], -1.0, 1.0))
            ).astype(np.float32)

        def reset(self, **kwargs):
            observation, info = super().reset(**kwargs)
            return self._append_previous_action(
                observation, np.zeros(2, dtype=np.float32)
            ), info

        def _last_executed_normalized_action(
            self, fallback_action: np.ndarray
        ) -> np.ndarray:
            """Return the final physical acceleration actually integrated.

            In a CBF rollout the action handed to the simulator is the raw
            policy command, whereas the 100 Hz filter may execute a different
            safe acceleration at every substep.  The at-1 observation must
            report the last of those *executed* commands, not the stale raw
            policy request.  The lane-free traffic guard never rewrites a
            controlled ego acceleration, so ``_last_accelerations[0]`` is the
            authoritative final physics-frame value.
            """

            accelerations = np.asarray(
                getattr(self.base_env, "_last_accelerations", np.empty((0, 2))),
                dtype=float,
            )
            if accelerations.ndim != 2 or accelerations.shape[0] == 0:
                return np.clip(
                    np.asarray(fallback_action, dtype=np.float32).reshape(-1)[:2],
                    -1.0,
                    1.0,
                )
            physical = accelerations[0, :2]
            bounds = self.base_env.config["bounds"]
            pairs = (
                (float(bounds["ax_min"]), float(bounds["ax_max"])),
                (float(bounds["ay_min"]), float(bounds["ay_max"])),
            )
            normalized = np.empty(2, dtype=np.float32)
            for index, (value, (low, high)) in enumerate(zip(physical, pairs)):
                clipped = float(np.clip(value, low, high))
                if low < 0.0 < high:
                    scale = high if clipped >= 0.0 else abs(low)
                    normalized[index] = clipped / max(scale, 1e-6)
                else:
                    normalized[index] = float(
                        2.0 * (clipped - low) / max(high - low, 1e-6) - 1.0
                    )
            return np.clip(normalized, -1.0, 1.0)

        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            previous_action = self._last_executed_normalized_action(
                np.asarray(action, dtype=np.float32)
            )
            info = dict(info)
            info.update(
                {
                    "observation_at1_ax": float(previous_action[0]),
                    "observation_at1_ay": float(previous_action[1]),
                    "observation_at1_norm": float(np.linalg.norm(previous_action)),
                    "observation_at1_source": "last_executed_physics_action",
                }
            )
            return (
                self._append_previous_action(observation, previous_action),
                reward,
                terminated,
                truncated,
                info,
            )

    PreviousActionObservationWrapper.__module__ = __name__
    namespace["KaralakouRewardWrapper"] = PreviousActionObservationWrapper
    namespace["PPO_OBSERVATION_VARIANT"] = "target_y_plus_previous_action"
