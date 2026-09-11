from __future__ import annotations

from types import SimpleNamespace

import pytest

from saferl.rewards import make_latched_overtake_wrapper, resolve_overtake_detection


def test_detection_defaults_to_notebook_and_rejects_unknown_values():
    assert resolve_overtake_detection({}) == "step"
    assert resolve_overtake_detection({"overtake_detection": " Latched "}) == "latched"
    with pytest.raises(ValueError, match="window"):
        resolve_overtake_detection({"overtake_detection": "window"})


class _Road:
    """Straight-line stand-in for the lane-free env: dx = other_x - ego_x."""

    def __init__(self, *xs: float) -> None:
        self.ego = SimpleNamespace(position=[0.0, 5.1], length=3.5)
        self.others = [SimpleNamespace(position=[x, 5.1]) for x in xs]
        self.road = SimpleNamespace(vehicles=[self.ego, *self.others])
        self.vehicle = self.ego
        self.config = {"sensing_range": 90.0}

    @staticmethod
    def _signed_distance(ego_x: float, other_x: float) -> float:
        return float(other_x - ego_x)

    def dx(self) -> dict[int, float]:
        return {id(v): v.position[0] - self.ego.position[0] for v in self.others}


class _NotebookLikeWrapper:
    def __init__(self, base_env: _Road) -> None:
        self.base_env = base_env

    def reset(self, **kwargs):
        return "obs", {}

    def _overtake_count(self, previous_dx):
        return 0


def _step(wrapper, road: _Road, ego_x: float) -> int:
    previous = road.dx()
    road.ego.position[0] = ego_x
    return wrapper._overtake_count(previous)


def test_latched_detector_counts_a_gradual_pass_once():
    road = _Road(10.0)
    wrapper = make_latched_overtake_wrapper(_NotebookLikeWrapper)(road)
    wrapper.reset()
    # Closing at 1 m/step (20 m/s relative at 20 Hz is already fast); the
    # notebook's one-step test would never fire on this pass.
    counts = [_step(wrapper, road, x) for x in range(1, 16)]
    assert sum(counts) == 1
    assert counts.index(1) == 11  # first step with dx < -1.75 m (ego_x = 12)


def test_latched_detector_does_not_repay_a_repass_or_count_unseen_vehicles():
    road = _Road(5.0, -20.0)
    wrapper = make_latched_overtake_wrapper(_NotebookLikeWrapper)(road)
    wrapper.reset()
    assert sum(_step(wrapper, road, x) for x in (2.0, 4.0, 8.0)) == 1
    # Drop back behind the first vehicle and pass it again: no second bonus.
    assert sum(_step(wrapper, road, x) for x in (0.0, 3.0, 9.0)) == 0
    # The vehicle that started behind was never ahead, so it never counts.
    assert _step(wrapper, road, -30.0) == 0


def test_latched_detector_ignores_ring_wraparound_and_resets_per_episode():
    road = _Road(80.0)
    wrapper = make_latched_overtake_wrapper(_NotebookLikeWrapper)(road)
    wrapper.reset()
    _step(wrapper, road, 0.0)
    # A leader that pulls away and reappears "behind" beyond sensing range.
    road.others[0].position[0] = -150.0
    assert _step(wrapper, road, 0.0) == 0
    road.others[0].position[0] = 5.0
    wrapper.reset()
    assert sum(_step(wrapper, road, x) for x in (4.0, 8.0)) == 1
