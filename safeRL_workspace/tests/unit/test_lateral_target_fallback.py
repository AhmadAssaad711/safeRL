from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from saferl.rewards import make_center_fallback_wrapper, resolve_lateral_target_fallback
from scripts.training import run_ppo_cbf_progression as progression
from scripts.training.run_cbf_filter_ablation import (
    bootstrap_notebook_namespace,
    exec_required_notebook_cells,
)


def test_fallback_defaults_to_notebook_and_rejects_unknown_values():
    assert resolve_lateral_target_fallback({}) == "fastest_blocker"
    assert resolve_lateral_target_fallback({"lateral_target_fallback": " Center "}) == "center"
    with pytest.raises(ValueError, match="edge"):
        resolve_lateral_target_fallback({"lateral_target_fallback": "edge"})


class _FixedTargetWrapper:
    def __init__(self, zone_found: bool) -> None:
        self.zone_found = zone_found
        self.base_env = SimpleNamespace(config={"road_width": 10.2})

    def _lateral_target_and_speed(self):
        return 8.7, 16.0, self.zone_found


def test_center_fallback_replaces_only_the_no_gap_target():
    wrapper_class = make_center_fallback_wrapper(_FixedTargetWrapper)
    assert wrapper_class(zone_found=False)._lateral_target_and_speed() == (5.1, 16.0, False)
    assert wrapper_class(zone_found=True)._lateral_target_and_speed() == (8.7, 16.0, True)


def _notebook_namespace() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    namespace = bootstrap_notebook_namespace(root)
    exec_required_notebook_cells(root / "notebooks" / "lanelessKaralakou.ipynb", namespace)
    return namespace


def _rollout(namespace, reward_config, steps: int = 40):
    env = progression._base_environment(
        namespace,
        env_config=copy.deepcopy(namespace["ENV_CONFIG"]),
        reward_config=reward_config,
    )
    observation, _ = env.reset(seed=11)
    rng = np.random.default_rng(11)
    records = []
    for _ in range(steps):
        observation, reward, terminated, truncated, info = env.step(
            rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        )
        records.append((observation, float(reward), info))
        if terminated or truncated:
            break
    env.close()
    return records


def test_canonical_env_center_fallback_targets_center_in_reward_and_observation():
    namespace = _notebook_namespace()
    reward_config = {**namespace["REWARD_CONFIG"], "lateral_target_fallback": "center"}
    records = _rollout(namespace, reward_config)
    road_width = float(namespace["ENV_CONFIG"]["road_width"])
    no_gap = [(obs, info) for obs, _reward, info in records if info["karalakou_zone_found"] < 0.5]
    assert no_gap, "the canonical density should produce no-gap steps"
    for observation, info in no_gap:
        assert info["karalakou_target_y"] == pytest.approx(0.5 * road_width)
        assert info["karalakou_cy"] == pytest.approx(
            abs(info["karalakou_ego_y"] - 0.5 * road_width) / road_width
        )
        # Normalized target y in the observation is 0 at the road centre.
        assert observation[1] == pytest.approx(0.0, abs=1e-6)
    # The reciprocal notebook reward is otherwise unchanged.
    for _obs, reward, info in records:
        assert "karalakou_linear_tracking_reward" not in info
        denom = 0.4 + info["karalakou_cx"] + 0.65 * info["karalakou_cy"] + info["karalakou_cf"]
        event = -2.5 if info["karalakou_ego_collision"] > 0.5 else 0.5 * min(info["karalakou_overtakes"], 1.0)
        assert reward == pytest.approx(
            0.4 / denom + info["karalakou_progress_reward"] - info["karalakou_jerk_penalty"] + event
        )


def test_canonical_env_default_fallback_is_unchanged_notebook_target():
    namespace = _notebook_namespace()
    records = _rollout(namespace, dict(namespace["REWARD_CONFIG"]))
    offcenter = [
        info for _obs, _reward, info in records
        if info["karalakou_zone_found"] < 0.5
        and abs(info["karalakou_target_y"] - 0.5 * float(namespace["ENV_CONFIG"]["road_width"])) > 1e-6
    ]
    assert offcenter, "the notebook fallback should target a blocker, not the centre"
