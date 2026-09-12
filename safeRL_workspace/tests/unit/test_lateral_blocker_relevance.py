from __future__ import annotations

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from saferl.rewards import (
    DEFAULT_LATERAL_BLOCKER_HORIZON_S,
    add_reward_variant_arguments,
    apply_reward_variant_arguments,
    lateral_blocker_parameters,
    make_closing_blockers_wrapper,
    resolve_lateral_blockers,
    reward_variant_summary,
)
from scripts.training import run_ppo_cbf_progression as progression
from scripts.training.run_cbf_filter_ablation import (
    bootstrap_notebook_namespace,
    exec_required_notebook_cells,
)

ROAD_WIDTH = 10.2
SENSING_RANGE = 90.0
EGO_WIDTH = 1.8
EGO_SPEED = 14.0


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_reward_variant_arguments(parser)
    return parser.parse_args(argv)


def test_blockers_default_to_the_notebook_and_reject_unknown_values():
    assert resolve_lateral_blockers({}) == "all"
    assert resolve_lateral_blockers({"lateral_blockers": " Closing "}) == "closing"
    with pytest.raises(ValueError, match="nearest"):
        resolve_lateral_blockers({"lateral_blockers": "nearest"})


def test_parameters_default_and_reject_non_positive_values():
    assert lateral_blocker_parameters({})["horizon_s"] == DEFAULT_LATERAL_BLOCKER_HORIZON_S
    assert lateral_blocker_parameters({"lateral_blocker_horizon_s": 2.0})["horizon_s"] == 2.0
    with pytest.raises(ValueError, match="standstill_gap_m"):
        lateral_blocker_parameters({"lateral_blocker_standstill_gap_m": 0.0})


def test_the_horizon_flags_need_a_non_default_blocker_set():
    reward_config: dict = {}
    with pytest.raises(ValueError, match="lateral-blocker-horizon-s"):
        apply_reward_variant_arguments(_parse(["--lateral-blocker-horizon-s", "2"]), reward_config)
    reward_config = {}
    apply_reward_variant_arguments(
        _parse(["--lateral-blockers", "closing", "--lateral-blocker-horizon-s", "2"]),
        reward_config,
    )
    assert reward_config["lateral_blockers"] == "closing"
    assert reward_config["lateral_blocker_horizon_s"] == 2.0
    summary = reward_variant_summary(reward_config)
    assert summary["lateral_blockers"] == "closing"
    assert summary["lateral_blocker_horizon_s"] == "2.0"


class _StubWrapper:
    """Minimal stand-in exposing what the closing-blocker wrapper reads."""

    def __init__(self, vehicles, reward_config=None, ego_y=5.1, ego_vx=EGO_SPEED) -> None:
        ego = SimpleNamespace(
            position=np.array([100.0, ego_y]),
            width=EGO_WIDTH,
            vx=ego_vx,
            desired_speed=16.0,
        )
        self.reward_config = {"lateral_blockers": "closing", **(reward_config or {})}
        self.base_env = SimpleNamespace(
            vehicle=ego,
            road=SimpleNamespace(vehicles=[ego, *vehicles]),
            config={"sensing_range": SENSING_RANGE, "road_width": ROAD_WIDTH},
            # A straight line, not a ring: the stub never wraps around.
            _forward_distance=lambda origin, other: float(other - origin),
        )

    @staticmethod
    def _subtract_interval(intervals, low, high):
        if high <= low:
            return intervals
        remaining = []
        for interval_low, interval_high in intervals:
            if high <= interval_low or low >= interval_high:
                remaining.append((interval_low, interval_high))
                continue
            if low > interval_low:
                remaining.append((interval_low, low))
            if high < interval_high:
                remaining.append((high, interval_high))
        return remaining


def _car(dx: float, y: float, vx: float):
    return SimpleNamespace(position=np.array([100.0 + dx, y]), width=1.8, vx=vx)


def _wrapper(vehicles, reward_config=None, **kwargs):
    return make_closing_blockers_wrapper(_StubWrapper)(vehicles, reward_config, **kwargs)


def _blocker_positions(vehicles, reward_config=None, **kwargs):
    wrapper = _wrapper(vehicles, reward_config, **kwargs)
    return sorted(float(vehicle.position[0]) - 100.0 for _, vehicle in wrapper._relevant_blockers())


def test_a_vehicle_ahead_that_is_pulling_away_does_not_block():
    # Faster than the ego, so the gap to it only grows: irrelevant at any range
    # beyond the standstill gap.
    assert _blocker_positions([_car(dx=20.0, y=5.1, vx=20.0)]) == []
    # Inside d0 = 5 m it blocks regardless of the closing speed.
    assert _blocker_positions([_car(dx=3.0, y=5.1, vx=20.0)]) == [3.0]


def test_a_slower_vehicle_blocks_only_inside_the_closing_horizon():
    # Closing speed 4 m/s, T = 3 s: reach = 5 + 12 = 17 m.
    assert _blocker_positions([_car(dx=16.0, y=5.1, vx=10.0)]) == [16.0]
    assert _blocker_positions([_car(dx=18.0, y=5.1, vx=10.0)]) == []
    # A longer horizon takes in more of the road.
    assert _blocker_positions(
        [_car(dx=18.0, y=5.1, vx=10.0)], {"lateral_blocker_horizon_s": 6.0}
    ) == [18.0]


def test_only_the_bidirectional_mode_sees_an_overtaker_from_behind():
    overtaker = [_car(dx=-15.0, y=5.1, vx=20.0)]
    # Closing at 6 m/s from 15 m back: reach = 5 + 18 = 23 m, so it blocks.
    assert _blocker_positions(overtaker) == [-15.0]
    assert _blocker_positions(overtaker, {"lateral_blockers": "closing_forward"}) == []
    # A vehicle behind that is slower than the ego never blocks.
    assert _blocker_positions([_car(dx=-15.0, y=5.1, vx=10.0)]) == []


def test_the_target_is_the_nearest_gap_among_the_relevant_vehicles_only():
    # One slow car dead ahead in the ego's corridor, one fast car alongside it
    # laterally that is pulling away and must not shrink the corridor.
    vehicles = [_car(dx=10.0, y=5.1, vx=10.0), _car(dx=30.0, y=8.0, vx=22.0)]
    target_y, target_speed, zone_found = _wrapper(vehicles)._lateral_target_and_speed()
    assert zone_found is True
    assert target_speed == pytest.approx(16.0)
    # Blocked band is 5.1 +- 1.95, so the free intervals are [0.9, 3.15] and
    # [7.05, 9.3]; the ego at 5.1 is equidistant and the lower one is first.
    assert target_y == pytest.approx(0.5 * (0.9 + 3.15))


def test_no_relevant_vehicle_leaves_the_road_centre_and_a_found_zone():
    assert _wrapper([_car(dx=40.0, y=5.1, vx=20.0)])._lateral_target_and_speed() == (
        ROAD_WIDTH / 2.0,
        16.0,
        True,
    )


def test_the_fallback_still_fires_when_the_relevant_set_fills_the_road():
    # Three slow cars 10 m ahead, spread across the road: 3 x 3.9 m of band
    # covers the 8.4 m corridor, so no gap survives and the fastest of them
    # becomes the target, exactly as the notebook does it.
    vehicles = [
        _car(dx=10.0, y=2.0, vx=10.0),
        _car(dx=10.0, y=5.1, vx=11.0),
        _car(dx=10.0, y=8.2, vx=9.0),
    ]
    target_y, _, zone_found = _wrapper(vehicles)._lateral_target_and_speed()
    assert zone_found is False
    assert target_y == pytest.approx(5.1)


def _notebook_namespace() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    namespace = bootstrap_notebook_namespace(root)
    exec_required_notebook_cells(root / "notebooks" / "lanelessKaralakou.ipynb", namespace)
    return namespace


def _rollout(namespace, reward_config, steps: int = 120):
    # _base_environment applies wrap_reward_variants itself, from this config.
    env = progression._base_environment(
        namespace,
        env_config=copy.deepcopy(namespace["ENV_CONFIG"]),
        reward_config=copy.deepcopy(reward_config),
    )
    env.reset(seed=11)
    rng = np.random.default_rng(11)
    records = []
    try:
        for _ in range(steps):
            _observation, _reward, terminated, truncated, info = env.step(
                rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            )
            records.append(dict(info))
            if terminated or truncated:
                break
    finally:
        env.close()
    return records


def test_a_long_horizon_still_drops_vehicles_that_are_pulling_away():
    # The filter is a closing-speed test, not a range test: no horizon makes a
    # faster vehicle ahead relevant again, which is why an unbounded horizon
    # does not reproduce the notebook blocker set.
    faster = [_car(dx=60.0, y=5.1, vx=20.0)]
    slower = [_car(dx=60.0, y=5.1, vx=10.0)]
    parameters = {"lateral_blocker_horizon_s": 1e6}
    assert _blocker_positions(faster, parameters) == []
    assert _blocker_positions(slower, parameters) == [60.0]
    # And the relevant set is always a subset of the notebook's forward set.
    mixed = faster + slower + [_car(dx=-20.0, y=5.1, vx=10.0)]
    forward_in_range = [60.0, 60.0]
    assert set(_blocker_positions(mixed, parameters)).issubset(forward_in_range)


@pytest.mark.slow
def test_the_closing_horizon_finds_gaps_the_notebook_misses():
    namespace = _notebook_namespace()
    baseline = _rollout(namespace, namespace["REWARD_CONFIG"])
    closing = _rollout(
        namespace,
        {**copy.deepcopy(namespace["REWARD_CONFIG"]), "lateral_blockers": "closing"},
    )
    assert baseline and len(baseline) == len(closing)
    baseline_found = np.mean([info["karalakou_zone_found"] for info in baseline])
    closing_found = np.mean([info["karalakou_zone_found"] for info in closing])
    assert baseline_found < 0.5, "the notebook blocker set should rarely find a gap"
    assert closing_found > baseline_found + 0.3
    # Dropping blockers can only widen the free intervals, so a step where the
    # notebook found a gap must still find one here. Both rollouts see the same
    # states: the reward never feeds back into the dynamics.
    for base_info, closing_info in zip(baseline, closing):
        if base_info["karalakou_zone_found"] > 0.5:
            assert closing_info["karalakou_zone_found"] > 0.5
    # The speed target must be untouched by a lateral-only change.
    for info in closing:
        assert info["karalakou_target_speed"] == pytest.approx(
            float(namespace["REWARD_CONFIG"]["ego_desired_speed"])
        )
