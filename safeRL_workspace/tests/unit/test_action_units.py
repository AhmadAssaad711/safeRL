from __future__ import annotations

import numpy as np
import pytest

from scripts.common.action_units import (
    normalized_action_delta_norm,
    normalized_to_physical_action,
    physical_to_normalized_action,
    unbounded_action_clip_norm,
    validate_matching_physical_action_bounds,
)


def test_asymmetric_zero_crossing_round_trip_uses_sign_specific_scales():
    low = np.asarray([-6.0, -4.0], dtype=np.float32)
    high = np.asarray([3.0, 4.0], dtype=np.float32)
    physical = np.asarray([-6.0, 3.0], dtype=np.float32)

    normalized = physical_to_normalized_action(physical, low, high)
    np.testing.assert_allclose(normalized, [-1.0, 0.75])
    np.testing.assert_allclose(
        normalized_to_physical_action(normalized, low, high), physical
    )


def test_cbf_correction_is_measured_after_box_clipping():
    low = np.asarray([-6.0, -4.0], dtype=np.float32)
    high = np.asarray([3.0, 4.0], dtype=np.float32)
    actor_latent = np.asarray([4.0, 0.0], dtype=np.float32)
    box_clipped = np.asarray([3.0, 0.0], dtype=np.float32)

    assert unbounded_action_clip_norm(actor_latent, box_clipped, low, high) == pytest.approx(
        1.0 / 3.0
    )
    assert normalized_action_delta_norm(box_clipped, box_clipped, low, high) == pytest.approx(
        0.0
    )


def test_bounds_validation_rejects_cbf_environment_mismatch():
    config = {
        "bounds": {
            "ax_min": -6.0,
            "ax_max": 3.0,
            "ay_min": -4.0,
            "ay_max": 4.0,
        }
    }
    with pytest.raises(ValueError, match="disagree"):
        validate_matching_physical_action_bounds(
            (-3.0, -3.0), (3.0, 3.0), config
        )
