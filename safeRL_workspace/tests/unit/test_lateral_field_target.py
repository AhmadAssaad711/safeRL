from __future__ import annotations

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from saferl.rewards import (
    DEFAULT_LATERAL_FIELD_BETA,
    add_reward_variant_arguments,
    apply_reward_variant_arguments,
    lateral_field_parameters,
    make_field_target_wrapper,
    resolve_lateral_target,
    reward_variant_summary,
)
from scripts.training import run_ppo_cbf_progression as progression
from scripts.training.run_cbf_filter_ablation import (
    bootstrap_notebook_namespace,
    exec_required_notebook_cells,
)

ROAD_WIDTH = 10.2
CENTRE = 0.5 * ROAD_WIDTH
EGO_SPEED = 14.0


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_reward_variant_arguments(parser)
    return parser.parse_args(argv)


def test_target_mode_defaults_to_the_gap_search_and_rejects_unknown_values():
    assert resolve_lateral_target({}) == "gap"
    assert resolve_lateral_target({"lateral_target": " Field "}) == "field"
    with pytest.raises(ValueError, match="softmin"):
        resolve_lateral_target({"lateral_target": "softmin"})


def test_parameters_default_and_validate():
    parameters = lateral_field_parameters({})
    assert parameters["beta"] == DEFAULT_LATERAL_FIELD_BETA
    # The closing horizon is shared with the gap target's blocker test.
    assert parameters["horizon_s"] == 3.0
    assert lateral_field_parameters({"lateral_field_travel_weight": 0.0})["travel_weight"] == 0.0
    with pytest.raises(ValueError, match="sigma_m"):
        lateral_field_parameters({"lateral_field_sigma_m": 0.0})
    with pytest.raises(ValueError, match="travel_weight"):
        lateral_field_parameters({"lateral_field_travel_weight": -1.0})


def test_the_field_flags_need_the_field_target():
    with pytest.raises(ValueError, match="lateral-field-beta"):
        apply_reward_variant_arguments(_parse(["--lateral-field-beta", "3"]), {})
    reward_config: dict = {}
    apply_reward_variant_arguments(
        _parse(["--lateral-target", "field", "--lateral-field-beta", "3"]), reward_config
    )
    summary = reward_variant_summary(reward_config)
    assert summary["lateral_target"] == "field"
    assert summary["lateral_field_beta"] == "3.0"


class _StubWrapper:
    """Minimal stand-in exposing what the field-target wrapper reads."""

    def __init__(self, vehicles, reward_config=None, ego_y=CENTRE, ego_vx=EGO_SPEED) -> None:
        ego = SimpleNamespace(
            position=np.array([100.0, ego_y]), width=1.8, vx=ego_vx, desired_speed=16.0
        )
        self.reward_config = dict(reward_config or {})
        self.base_env = SimpleNamespace(
            vehicle=ego,
            road=SimpleNamespace(vehicles=[ego, *vehicles]),
            config={"sensing_range": 90.0, "road_width": ROAD_WIDTH},
            # A straight line, not a ring: the stub never wraps around.
            _signed_distance=lambda origin, other: float(other - origin),
        )


def _car(dx: float, y: float, vx: float = EGO_SPEED):
    return SimpleNamespace(position=np.array([100.0 + dx, y]), width=1.8, vx=vx)


def _target(vehicles, reward_config=None, **kwargs) -> float:
    wrapper = make_field_target_wrapper(_StubWrapper)(vehicles, reward_config, **kwargs)
    target_y, target_speed, zone_found = wrapper._lateral_target_and_speed()
    # There is no gap search under this target, so there is nothing to fall back
    # from: zone_found is always true and the speed target is the notebook's.
    assert zone_found is True
    assert target_speed == pytest.approx(16.0)
    return float(target_y)


def test_an_empty_road_targets_the_centre():
    assert _target([]) == pytest.approx(CENTRE, abs=1e-6)


def test_only_vehicles_the_gap_is_closing_on_move_the_target():
    # Same speed as the ego, 10 m ahead: the reach is the 5 m standstill gap, so
    # its relevance is small and the target barely moves.
    assert _target([_car(dx=10.0, y=CENTRE, vx=EGO_SPEED)]) == pytest.approx(CENTRE, abs=0.5)
    # Faster and pulling away: irrelevant.
    assert _target([_car(dx=10.0, y=CENTRE, vx=22.0)]) == pytest.approx(CENTRE, abs=1e-6)
    # Slower and being caught: the target leaves its footprint.
    assert abs(_target([_car(dx=10.0, y=CENTRE, vx=8.0)]) - CENTRE) > 2.0
    # Overtaking from behind: also moves the target out of the way.
    assert abs(_target([_car(dx=-8.0, y=CENTRE, vx=22.0)]) - CENTRE) > 2.0


def test_the_target_never_lands_in_the_blocked_middle_between_two_vehicles():
    # Two slow cars straddling the ego's lane leave a nominal 4.2 m "gap" at the
    # centre that neither footprint reaches but both occupancy tails do. An
    # unrestricted weighted mean of the two free sides returns that middle; the
    # basin restriction must not.
    vehicles = [_car(dx=10.0, y=3.0, vx=8.0), _car(dx=10.0, y=7.2, vx=8.0)]
    assert abs(_target(vehicles) - CENTRE) > 1.5
    # A large basin margin recovers the unrestricted softmin, which averages the
    # two sides back into the middle: that is what the margin is for.
    unrestricted = _target(vehicles, {"lateral_field_basin_margin": 1e6})
    assert unrestricted == pytest.approx(CENTRE, abs=0.6)


def test_the_travel_term_breaks_the_tie_toward_the_ego_side():
    vehicles = [_car(dx=10.0, y=CENTRE, vx=8.0)]
    low = _target(vehicles, ego_y=3.0)
    high = _target(vehicles, ego_y=7.2)
    assert low < CENTRE < high, "the target should resolve to whichever side the ego is on"
    # With no travel term the two must agree, since the field alone is symmetric.
    no_travel = {"lateral_field_travel_weight": 0.0}
    assert _target(vehicles, {**no_travel}, ego_y=3.0) == pytest.approx(
        _target(vehicles, {**no_travel}, ego_y=7.2), abs=1e-6
    )


def _flip_count(targets: np.ndarray, tolerance: float = 0.25) -> int:
    return int((np.abs(np.diff(targets)) > tolerance).sum())


def test_the_target_moves_continuously_except_when_the_basin_switches():
    # Sweep a slow leader across the road. Inside a basin the target tracks it
    # smoothly; when the freest side stops being the freest, the choice flips
    # and the target jumps once. That single flip is the residual discontinuity
    # of this target, and it is what the diagnostic measures as a jump.
    offsets = np.linspace(4.0, 9.0, 400)
    targets = np.array([_target([_car(dx=12.0, y=float(y), vx=8.0)]) for y in offsets])
    assert _flip_count(targets) == 1, "one side swap, not a jittering target"
    smooth = np.sort(np.abs(np.diff(targets)))[:-1]
    assert float(smooth.max()) < 0.05, "every other step is continuous"
    assert targets.std() > 0.3, "and the target actually tracks the leader"


def test_relevance_fades_as_a_leader_recedes():
    # Ego offset from the leader, as real traffic almost always is: the target
    # relaxes back toward the centre with at most one side swap on the way.
    distances = np.linspace(5.0, 60.0, 400)
    targets = np.array(
        [_target([_car(dx=float(dx), y=CENTRE, vx=8.0)], ego_y=3.0) for dx in distances]
    )
    assert _flip_count(targets) <= 1
    # Far away the leader stops mattering and the target settles between the
    # ego and the centre, where the travel term puts it.
    assert 3.0 - 0.2 <= targets[-1] <= CENTRE + 0.2


def test_hysteresis_removes_the_flip_flop_between_two_tied_sides():
    # Exact symmetry is the worst case: the ego shares the leader's y, so both
    # sides are equally free and the memoryless target swaps between them as
    # the leader recedes.
    distances = np.linspace(5.0, 60.0, 400)

    def sweep(reward_config):
        wrapper = make_field_target_wrapper(_StubWrapper)([], reward_config)
        ego = wrapper.base_env.vehicle
        targets = []
        for dx in distances:
            wrapper.base_env.road.vehicles = [ego, _car(dx=float(dx), y=CENTRE, vx=8.0)]
            targets.append(float(wrapper._lateral_target_and_speed()[0]))
        return np.array(targets)

    memoryless = sweep({})
    latched = sweep({"lateral_field_hysteresis": 2.0})
    assert _flip_count(memoryless) > 1, "the tie is what hysteresis is for"
    assert _flip_count(latched) == 0
    assert float(np.abs(np.diff(latched)).max()) < 0.1


def _notebook_namespace() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    namespace = bootstrap_notebook_namespace(root)
    exec_required_notebook_cells(root / "notebooks" / "lanelessKaralakou.ipynb", namespace)
    return namespace


def _rollout(namespace, reward_config, steps: int = 120):
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


@pytest.mark.slow
def test_the_real_environment_target_is_always_defined_and_closer_than_the_notebook_one():
    namespace = _notebook_namespace()
    baseline = _rollout(namespace, namespace["REWARD_CONFIG"])
    field = _rollout(
        namespace, {**copy.deepcopy(namespace["REWARD_CONFIG"]), "lateral_target": "field"}
    )
    assert baseline and len(baseline) == len(field)
    # Random actions drive both rollouts, so the states are identical and the
    # two targets are measured against the same traffic.
    assert all(info["karalakou_zone_found"] > 0.5 for info in field)
    for info in field:
        assert 0.9 - 1e-6 <= info["karalakou_target_y"] <= 9.3 + 1e-6
        assert info["karalakou_target_speed"] == pytest.approx(
            float(namespace["REWARD_CONFIG"]["ego_desired_speed"])
        )
    baseline_error = np.mean([info["karalakou_lat_y_error_m"] for info in baseline])
    field_error = np.mean([info["karalakou_lat_y_error_m"] for info in field])
    assert field_error < baseline_error
    targets = np.array([info["karalakou_target_y"] for info in field])
    assert targets.std() > 0.1, "the target must carry information, not sit still"
