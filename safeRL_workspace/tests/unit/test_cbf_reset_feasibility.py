from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.common.ppo_cbf_env import CBFContextPhysicalActionWrapper


class _CandidateEnvironment:
    def __init__(self) -> None:
        self.seeds: list[int | None] = []

    def reset(self, **kwargs):
        self.seeds.append(kwargs.get("seed"))
        return np.asarray([float(len(self.seeds))]), {"candidate": len(self.seeds)}


def _wrapper(config: dict[str, object], safe_by_attempt: list[bool]):
    wrapper = object.__new__(CBFContextPhysicalActionWrapper)
    wrapper.env = _CandidateEnvironment()
    wrapper.namespace = {
        "_lane_free_base": lambda _wrapper: SimpleNamespace(config=config),
    }
    wrapper._pending_projection = object()
    wrapper._previous_executed_action_normalized = object()
    wrapper._last_reset_info = {}
    wrapper._augment_observation = lambda observation, _system: observation
    wrapper._constraint_system = lambda: {
        "hash": "fixed",
        "rows": np.zeros((0, 2), dtype=float),
    }
    attempts = iter(safe_by_attempt)

    def diagnostics():
        safe = next(attempts)
        return {
            "cbf_initial_min_h": 1.0 if safe else -0.1,
            "cbf_initial_min_psi1": 1.0 if safe else -0.1,
            "cbf_initial_safe_set": safe,
        }

    wrapper._initial_safety_diagnostics = diagnostics
    wrapper.psi1_gain = 2.3
    return wrapper


def test_opt_in_reset_feasibility_retries_deterministic_candidate_seeds():
    wrapper = _wrapper(
        {"cbf_reset_feasibility": {"enabled": True, "max_attempts": 3}},
        [False, True],
    )

    _observation, info = wrapper.reset(seed=41)

    assert wrapper.env.seeds == [41, 42]
    assert info["cbf_reset_feasibility_enabled"] is True
    assert info["cbf_reset_max_attempts"] == 3
    assert info["cbf_reset_retry_count"] == 1
    assert info["cbf_reset_candidate_seed"] == 42
    assert wrapper.last_reset_info == {
        "cbf_initial_min_h": 1.0,
        "cbf_initial_min_psi1": 1.0,
        "cbf_initial_safe_set": True,
        "cbf_reset_feasibility_enabled": True,
        "cbf_reset_max_attempts": 3,
        "cbf_reset_retry_count": 1,
        "cbf_reset_candidate_seed": 42,
    }


def test_reset_feasibility_is_disabled_by_default():
    wrapper = _wrapper({}, [False])

    _observation, info = wrapper.reset(seed=41)

    assert wrapper.env.seeds == [41]
    assert info["cbf_reset_feasibility_enabled"] is False
    assert info["cbf_reset_retry_count"] == 0
    assert info["cbf_reset_candidate_seed"] is None
