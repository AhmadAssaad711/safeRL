"""Shared physical/normalized action-unit conversions for laneless PPO."""

from __future__ import annotations

from typing import Any

import numpy as np


ACTION_DIM = 2


def _bounds_arrays(
    low: Any, high: Any
) -> tuple[np.ndarray, np.ndarray]:
    lows = np.asarray(low, dtype=np.float32).reshape(-1)[:ACTION_DIM]
    highs = np.asarray(high, dtype=np.float32).reshape(-1)[:ACTION_DIM]
    if (
        lows.size != ACTION_DIM
        or highs.size != ACTION_DIM
        or not np.all(np.isfinite(lows))
        or not np.all(np.isfinite(highs))
        or np.any(lows > highs)
    ):
        raise ValueError("action bounds must be finite two-dimensional intervals")
    return lows, highs


def physical_action_bounds_from_config(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Read the simulator's physical acceleration bounds from its config."""

    bounds = config["bounds"]
    return _bounds_arrays(
        [bounds["ax_min"], bounds["ay_min"]],
        [bounds["ax_max"], bounds["ay_max"]],
    )


def validate_matching_physical_action_bounds(
    cbf_low: Any,
    cbf_high: Any,
    env_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Require CBF and simulator physical action boxes to be identical."""

    configured_low, configured_high = physical_action_bounds_from_config(env_config)
    filter_low, filter_high = _bounds_arrays(cbf_low, cbf_high)
    if tuple(filter_low.tolist()) != tuple(configured_low.tolist()):
        raise ValueError(
            "CBF lower action bounds disagree with env_config['bounds']: "
            f"cbf={tuple(filter_low.tolist())}, "
            f"env={tuple(configured_low.tolist())}"
        )
    if tuple(filter_high.tolist()) != tuple(configured_high.tolist()):
        raise ValueError(
            "CBF upper action bounds disagree with env_config['bounds']: "
            f"cbf={tuple(filter_high.tolist())}, "
            f"env={tuple(configured_high.tolist())}"
        )
    return configured_low, configured_high


def physical_to_normalized_action(
    action_phys: Any,
    low: Any,
    high: Any,
    *,
    clip: bool = True,
) -> np.ndarray:
    """Apply the simulator's inverse physical-action map.

    For zero-crossing asymmetric bounds, positive and negative values use their
    respective actuator magnitudes.  ``clip=True`` is the canonical inverse
    for complete physical actions; ``clip=False`` is only for measuring how
    far an unbounded actor latent lies outside the actuator box.
    """

    lows, highs = _bounds_arrays(low, high)
    values = np.asarray(action_phys, dtype=np.float32).reshape(-1)
    if values.size < ACTION_DIM or not np.all(np.isfinite(values[:ACTION_DIM])):
        raise ValueError("action must contain two finite physical components")
    values = values[:ACTION_DIM].copy()
    if clip:
        values = np.clip(values, lows, highs)
    normalized = np.empty(ACTION_DIM, dtype=np.float32)
    for index, (value, lower, upper) in enumerate(zip(values, lows, highs)):
        if float(lower) < 0.0 < float(upper):
            scale = float(upper) if float(value) >= 0.0 else abs(float(lower))
            normalized[index] = float(value) / max(scale, 1e-6)
        else:
            normalized[index] = float(
                2.0 * (float(value) - float(lower))
                / max(float(upper - lower), 1e-6)
                - 1.0
            )
    return (
        np.clip(normalized, -1.0, 1.0)
        if clip
        else normalized
    ).astype(np.float32)


def normalized_to_physical_action(
    action_normalized: Any,
    low: Any,
    high: Any,
) -> np.ndarray:
    """Apply the simulator's normalized-to-physical action map."""

    lows, highs = _bounds_arrays(low, high)
    values = np.asarray(action_normalized, dtype=np.float32).reshape(-1)
    if values.size < ACTION_DIM or not np.all(np.isfinite(values[:ACTION_DIM])):
        raise ValueError("action must contain two finite normalized components")
    values = np.clip(values[:ACTION_DIM], -1.0, 1.0)
    physical = np.empty(ACTION_DIM, dtype=np.float32)
    for index, (value, lower, upper) in enumerate(zip(values, lows, highs)):
        if float(lower) < 0.0 < float(upper):
            scale = float(upper) if float(value) >= 0.0 else abs(float(lower))
            physical[index] = float(value) * max(scale, 1e-6)
        else:
            physical[index] = float(
                float(lower) + 0.5 * (float(value) + 1.0) * float(upper - lower)
            )
    return np.clip(physical, lows, highs).astype(np.float32)


def normalized_action_delta_norm(
    first_phys: Any,
    second_phys: Any,
    low: Any,
    high: Any,
) -> float:
    """Return the norm of two complete actions in normalized coordinates."""

    first = physical_to_normalized_action(first_phys, low, high)
    second = physical_to_normalized_action(second_phys, low, high)
    return float(np.linalg.norm(first - second))


def unbounded_action_clip_norm(
    latent_phys: Any,
    box_clipped_phys: Any,
    low: Any,
    high: Any,
) -> float:
    """Measure actuator clipping without treating it as a CBF correction."""

    latent = physical_to_normalized_action(latent_phys, low, high, clip=False)
    clipped = physical_to_normalized_action(
        box_clipped_phys, low, high, clip=False
    )
    return float(np.linalg.norm(latent - clipped))
