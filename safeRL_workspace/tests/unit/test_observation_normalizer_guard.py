from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from types import SimpleNamespace

from scripts.training.run_cbf_filter_ablation import (
    _apply_legacy_observation_normalizer,
)


def _env(width: int) -> gym.Env:
    return SimpleNamespace(
        observation_space=gym.spaces.Box(
            -np.inf, np.inf, shape=(width,), dtype=np.float32
        )
    )  # type: ignore[return-value]


def test_legacy_normalizer_rejects_canonical_30d_and_32d_states():
    namespace = {
        "NORMALIZE_RL_OBSERVATIONS": True,
        "OBSERVATION_CLIP": 5.0,
        "LaneFreeObservationNormalizationWrapper": object,
    }
    for width in (30, 32):
        with pytest.raises(ValueError, match="legacy-only"):
            _apply_legacy_observation_normalizer(namespace, _env(width))  # type: ignore[arg-type]


def test_legacy_normalizer_remains_available_for_noncanonical_schema():
    calls: list[tuple[object, float]] = []

    def wrapper(env, *, clip):
        calls.append((env, clip))
        return "legacy-wrapper"

    namespace = {
        "NORMALIZE_RL_OBSERVATIONS": True,
        "OBSERVATION_CLIP": 5.0,
        "LaneFreeObservationNormalizationWrapper": wrapper,
    }
    env = _env(42)
    assert _apply_legacy_observation_normalizer(namespace, env) == "legacy-wrapper"
    assert calls == [(env, 5.0)]
