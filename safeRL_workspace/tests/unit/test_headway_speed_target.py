from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from saferl.rewards import (
    DEFAULT_SPEED_TARGET_NOMINAL,
    make_headway_speed_target_wrapper,
    resolve_speed_target,
    speed_target_parameters,
)
from scripts.training import run_ppo_cbf_progression as progression
from scripts.training.run_cbf_filter_ablation import (
    bootstrap_notebook_namespace,
    exec_required_notebook_cells,
)

EGO_WIDTH = 1.8
CAR_WIDTH = 1.8
SENSING_RANGE = 90.0


def test_speed_target_defaults_to_notebook_and_rejects_unknown_values():
    assert resolve_speed_target({}) == "nominal"
    assert resolve_speed_target({"speed_target": " Headway "}) == "headway"
    with pytest.raises(ValueError, match="blocker"):
        resolve_speed_target({"speed_target": "blocker"})


def test_speed_target_parameters_reject_non_positive_values():
    assert speed_target_parameters({})["v_nominal"] == DEFAULT_SPEED_TARGET_NOMINAL
    assert speed_target_parameters({"timegap": 1.5})["timegap"] == 1.5
    with pytest.raises(ValueError, match="standstill_gap_m"):
        speed_target_parameters({"speed_target_standstill_gap_m": 0.0})


class _StubWrapper:
    """Minimal stand-in exposing what the headway wrapper reads."""

    def __init__(self, vehicles, reward_config=None) -> None:
        ego = SimpleNamespace(position=np.array([0.0, 5.1]), width=EGO_WIDTH, vx=12.0)
        self.reward_config = {"speed_target_nominal": 20.0, "timegap": 1.5, **(reward_config or {})}
        self.base_env = SimpleNamespace(
            vehicle=ego,
            road=SimpleNamespace(vehicles=[ego, *vehicles]),
            config={"sensing_range": SENSING_RANGE, "road_width": 10.2},
            _forward_distance=lambda origin, other: float(other - origin),
        )

    def _lateral_target_and_speed(self):
        return 5.1, 16.0, False


def _car(dx: float, dy: float, vx: float = 12.0):
    return SimpleNamespace(position=np.array([dx, 5.1 + dy]), width=CAR_WIDTH, vx=vx)


def _target(vehicles, reward_config=None) -> float:
    wrapper_class = make_headway_speed_target_wrapper(_StubWrapper)
    return wrapper_class(vehicles, reward_config)._headway_speed_target()


def test_empty_road_and_distant_leaders_leave_the_free_flow_target():
    assert _target([]) == pytest.approx(20.0)
    # The cap can only bind below d0 + T*v_nom = 5 + 1.5*20 = 35 m.
    assert _target([_car(dx=40.0, dy=0.0)]) == pytest.approx(20.0)
    assert _target([_car(dx=35.0, dy=0.0)]) == pytest.approx(20.0)


def test_a_close_leader_caps_the_target_at_its_headway_speed():
    # v_allow = (dx - d0) / T, at full lateral relevance.
    assert _target([_car(dx=25.0, dy=0.0)]) == pytest.approx(13.333, abs=1e-3)
    assert _target([_car(dx=15.0, dy=0.0)]) == pytest.approx(6.667, abs=1e-3)
    # Inside the standstill gap the allowance floors at zero.
    assert _target([_car(dx=3.0, dy=0.0)]) == pytest.approx(0.0, abs=1e-6)


def test_vehicles_behind_and_laterally_clear_vehicles_do_not_constrain():
    assert _target([_car(dx=-20.0, dy=0.0)]) == pytest.approx(20.0)
    # 4 m of lateral offset is far outside the 1.8 m half-width sum plus sigma.
    assert _target([_car(dx=2.0, dy=4.0)]) == pytest.approx(20.0, abs=1e-3)


def test_the_binding_vehicle_is_the_most_constraining_one():
    vehicles = [_car(dx=30.0, dy=0.0), _car(dx=20.0, dy=0.0), _car(dx=45.0, dy=0.0)]
    assert _target(vehicles) == pytest.approx(10.0, abs=1e-3)
    # Adding a second car at the same distance must not lower the target; a
    # log-sum-exp softmin would, which is why the aggregation is a minimum.
    assert _target(vehicles + [_car(dx=20.0, dy=0.3)]) == pytest.approx(10.0, abs=1e-2)


def test_full_lateral_overlap_applies_the_cap_at_full_strength():
    # Directly behind a leader the relevance kernel must be exactly 1, so the
    # target is the headway speed itself with no leak toward the free-flow one.
    assert _target([_car(dx=25.0, dy=0.0)]) == pytest.approx((25.0 - 5.0) / 1.5, abs=1e-9)
    # Still fully overlapping at 1.7 m of offset (half-width sum is 1.8 m).
    assert _target([_car(dx=25.0, dy=1.7)]) == pytest.approx((25.0 - 5.0) / 1.5, abs=1e-9)


def test_the_target_is_continuous_as_a_vehicle_drifts_out_of_the_corridor():
    offsets = np.linspace(0.0, 5.0, 501)
    targets = np.array([_target([_car(dx=10.0, dy=float(dy))]) for dy in offsets])
    assert targets[0] == pytest.approx(3.333, abs=1e-3)
    assert targets[-1] == pytest.approx(20.0, abs=1e-3)
    assert np.all(np.diff(targets) >= -1e-9), "target must rise monotonically with clearance"
    assert float(np.max(np.abs(np.diff(targets)))) < 0.5, "no jump as relevance fades"


def _notebook_namespace() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    namespace = bootstrap_notebook_namespace(root)
    exec_required_notebook_cells(root / "notebooks" / "lanelessKaralakou.ipynb", namespace)
    return namespace


def _rollout(namespace, reward_config, steps: int = 60):
    env = progression._base_environment(
        namespace,
        env_config=copy.deepcopy(namespace["ENV_CONFIG"]),
        reward_config=copy.deepcopy(reward_config),
    )
    observation, _ = env.reset(seed=11)
    rng = np.random.default_rng(11)
    records = []
    try:
        for _ in range(steps):
            observation, reward, terminated, truncated, info = env.step(
                rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            )
            records.append((np.asarray(observation, dtype=float), float(reward), dict(info)))
            if terminated or truncated:
                break
    finally:
        env.close()
    return records


@pytest.mark.slow
def test_real_environment_target_varies_and_cx_uses_the_fixed_nominal():
    namespace = _notebook_namespace()
    reward_config = copy.deepcopy(namespace["REWARD_CONFIG"])
    reward_config.update({"speed_target": "headway", "speed_target_nominal": 20.0})
    records = _rollout(namespace, reward_config)
    assert records, "the rollout produced no steps"

    targets = np.array([info["karalakou_target_speed"] for _, _, info in records])
    # The notebook target is the constant 16.0; this one must move.
    assert targets.std() > 0.1
    assert targets.max() <= 20.0 + 1e-6 and targets.min() >= -1e-9

    for _, _, info in records:
        expected = abs(info["karalakou_ego_speed"] - info["karalakou_target_speed"]) / 20.0
        assert info["karalakou_cx"] == pytest.approx(expected, abs=1e-9)
        assert info["karalakou_speed_target_nominal"] == pytest.approx(20.0)

    # Observation slot 4 is the ego row's v_d feature and must follow the target.
    observation_vmax = float(namespace["ENV_CONFIG"].get("observation_vmax", 24.0))
    for observation, _, info in records:
        assert observation[4] == pytest.approx(
            info["karalakou_target_speed"] / observation_vmax, abs=1e-6
        )


@pytest.mark.slow
def test_real_environment_reward_matches_its_published_components():
    namespace = _notebook_namespace()
    reward_config = copy.deepcopy(namespace["REWARD_CONFIG"])
    reward_config.update({"speed_target": "headway", "speed_target_nominal": 20.0})
    for _, reward, info in _rollout(namespace, reward_config):
        denominator = (
            reward_config["epsilon_r"]
            + reward_config["wx"] * info["karalakou_cx"]
            + reward_config["wy"] * info["karalakou_cy"]
            + reward_config["wf"] * info["karalakou_cf"]
            + reward_config.get("way", 0.0) * info["karalakou_cay"]
        )
        expected = (
            reward_config["epsilon_r"] / max(denominator, 1e-9)
            + info["karalakou_progress_reward"]
            - info["karalakou_jerk_penalty"]
        )
        if info["karalakou_ego_collision"] > 0.5:
            expected += reward_config["collision_penalty"]
        elif info["karalakou_overtakes"] > 0:
            expected += reward_config["overtake_bonus"]
        assert reward == pytest.approx(expected, abs=1e-6)


@pytest.mark.slow
def test_nominal_speed_target_is_bit_identical_to_the_notebook_reward():
    namespace = _notebook_namespace()
    baseline = _rollout(namespace, copy.deepcopy(namespace["REWARD_CONFIG"]))
    explicit = _rollout(
        namespace, {**copy.deepcopy(namespace["REWARD_CONFIG"]), "speed_target": "nominal"}
    )
    assert len(baseline) == len(explicit)
    for (_, left, left_info), (_, right, right_info) in zip(baseline, explicit):
        assert left == pytest.approx(right, abs=0.0)
        assert left_info["karalakou_target_speed"] == right_info["karalakou_target_speed"]
