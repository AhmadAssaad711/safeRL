from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from saferl.rewards import (
    linear_reward_terms,
    linear_tracking_reward,
    linear_tracking_weights,
    make_linear_reward_wrapper,
    resolve_reward_mode,
)
from scripts.training import run_ppo_cbf_progression as progression
from scripts.training.run_cbf_filter_ablation import (
    bootstrap_notebook_namespace,
    exec_required_notebook_cells,
)

CANONICAL_WEIGHTS = {"wx": 1.0, "wy": 0.65, "wf": 1.0, "way": 0.0}
EVENT_CONFIG = {**CANONICAL_WEIGHTS, "collision_penalty": -2.5, "overtake_bonus": 0.5}


def _components(**overrides: float) -> dict[str, float]:
    components = {
        "cx": 0.34,
        "cy": 0.41,
        "cf": 0.0,
        "cay": 0.0,
        "progress_reward": 0.02,
        "jerk_penalty": 0.005,
        "ego_collision": 0.0,
        "overtakes": 0.0,
    }
    components.update(overrides)
    return components


def test_reward_mode_defaults_to_reciprocal_and_rejects_unimplemented_modes():
    assert resolve_reward_mode({}) == "reciprocal"
    assert resolve_reward_mode({"reward_mode": " Linear "}) == "linear"
    with pytest.raises(ValueError, match="additive"):
        resolve_reward_mode({"reward_mode": "additive"})


def test_linear_weights_are_normalized_canonical_weights():
    weights = linear_tracking_weights(CANONICAL_WEIGHTS)
    assert weights["cx"] == pytest.approx(1.0 / 2.65)
    assert weights["cy"] == pytest.approx(0.65 / 2.65)
    assert weights["cay"] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        linear_tracking_weights({"wx": 0.0, "wy": 0.0, "wf": 0.0, "way": 0.0})
    with pytest.raises(ValueError):
        linear_tracking_weights({**CANONICAL_WEIGHTS, "wy": -1.0})


def test_linear_tracking_is_bounded_and_has_constant_pull():
    assert linear_tracking_reward(_components(cx=0.0, cy=0.0), CANONICAL_WEIGHTS) == 1.0
    worst = _components(cx=5.0, cy=5.0, cf=5.0, cay=5.0)
    assert linear_tracking_reward(worst, CANONICAL_WEIGHTS) == pytest.approx(0.0)
    # The lateral pull does not depend on the other costs.
    step = 0.1
    for other in (0.0, 0.5, 1.0):
        low = linear_tracking_reward(_components(cx=other, cf=other, cy=0.2), CANONICAL_WEIGHTS)
        high = linear_tracking_reward(_components(cx=other, cf=other, cy=0.2 + step), CANONICAL_WEIGHTS)
        assert (low - high) / step == pytest.approx(0.65 / 2.65)


def test_linear_reward_terms_sum_to_returned_reward():
    for components, event in (
        (_components(), 0.0),
        (_components(ego_collision=1.0, overtakes=2.0), -2.5),
        (_components(overtakes=3.0), 0.5),
    ):
        terms = linear_reward_terms(components, EVENT_CONFIG)
        assert terms["event_reward"] == event
        assert terms["reward"] == pytest.approx(
            terms["linear_tracking_reward"]
            + components["progress_reward"]
            - components["jerk_penalty"]
            + terms["event_reward"]
        )


class _FixedReciprocalWrapper:
    def __init__(self, reward_config: dict[str, float]) -> None:
        self.reward_config = reward_config

    def _karalakou_reward(self, previous_dx, previous_ego_x=None):
        return 0.123, _components(reward=0.123)


def test_linear_wrapper_replaces_reward_and_keeps_reciprocal_total():
    wrapper = make_linear_reward_wrapper(_FixedReciprocalWrapper)(EVENT_CONFIG)
    reward, components = wrapper._karalakou_reward({}, 0.0)
    assert components["reciprocal_mode_reward"] == 0.123
    assert components["reward"] == reward
    assert reward == pytest.approx(linear_reward_terms(_components(), EVENT_CONFIG)["reward"])


def _notebook_namespace() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    namespace = bootstrap_notebook_namespace(root)
    exec_required_notebook_cells(root / "notebooks" / "lanelessKaralakou.ipynb", namespace)
    return namespace


def _rollout(namespace, reward_config, steps: int = 40):
    env_config = copy.deepcopy(namespace["ENV_CONFIG"])
    env = progression._base_environment(
        namespace, env_config=env_config, reward_config=reward_config
    )
    env.reset(seed=11)
    rng = np.random.default_rng(11)
    records = []
    for _ in range(steps):
        _obs, reward, terminated, truncated, info = env.step(
            rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        )
        records.append((float(reward), info))
        if terminated or truncated:
            break
    env.close()
    return env, records


def test_canonical_env_linear_reward_matches_declared_components():
    namespace = _notebook_namespace()
    reward_config = {**namespace["REWARD_CONFIG"], "reward_mode": "linear"}
    env, records = _rollout(namespace, reward_config)
    assert records
    for reward, info in records:
        tracking = info["karalakou_linear_tracking_reward"]
        assert 0.0 <= tracking <= 1.0
        assert reward == pytest.approx(info["karalakou_reward"])
        assert reward == pytest.approx(
            tracking
            + info["karalakou_progress_reward"]
            - info["karalakou_jerk_penalty"]
            + info["karalakou_event_reward"]
        )
        denom = 0.4 + info["karalakou_cx"] + 0.65 * info["karalakou_cy"] + info["karalakou_cf"]
        assert info["karalakou_reciprocal_mode_reward"] == pytest.approx(
            0.4 / denom
            + info["karalakou_progress_reward"]
            - info["karalakou_jerk_penalty"]
            + info["karalakou_event_reward"]
        )


def test_canonical_env_default_mode_keeps_notebook_reciprocal_wrapper():
    namespace = _notebook_namespace()
    env, records = _rollout(namespace, dict(namespace["REWARD_CONFIG"]), steps=5)
    assert "karalakou_linear_tracking_reward" not in records[0][1]
    with pytest.raises(ValueError, match="additive"):
        progression._base_environment(
            namespace,
            env_config=copy.deepcopy(namespace["ENV_CONFIG"]),
            reward_config={**namespace["REWARD_CONFIG"], "reward_mode": "additive"},
        )
