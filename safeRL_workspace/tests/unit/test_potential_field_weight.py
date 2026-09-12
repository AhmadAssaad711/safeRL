from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pytest

from saferl.rewards import (
    add_reward_variant_arguments,
    apply_reward_variant_arguments,
    reward_variant_summary,
)
from scripts.training import run_ppo_cbf_progression as progression
from scripts.training.run_cbf_filter_ablation import (
    bootstrap_notebook_namespace,
    exec_required_notebook_cells,
)

# The two overrides the canonical ladder launches with.
LADDER_FLAGS = ["--lateral-target-fallback", "center", "--potential-field-weight", "0"]


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_reward_variant_arguments(parser)
    return parser.parse_args(argv)


def test_the_flag_zeroes_wf_and_leaves_every_other_weight_alone():
    reward_config = {"wx": 1.0, "wy": 0.65, "wf": 1.0, "way": 0.0}
    apply_reward_variant_arguments(_parse(LADDER_FLAGS), reward_config)
    assert reward_config["wf"] == 0.0
    assert reward_config["lateral_target_fallback"] == "center"
    assert (reward_config["wx"], reward_config["wy"], reward_config["way"]) == (1.0, 0.65, 0.0)


def test_omitting_the_flag_keeps_the_notebook_weight():
    reward_config = {"wf": 1.0}
    apply_reward_variant_arguments(_parse([]), reward_config)
    assert reward_config["wf"] == 1.0
    assert "lateral_target_fallback" not in reward_config


def test_negative_and_non_finite_weights_are_rejected():
    for value in ("-0.5", "nan", "inf"):
        with pytest.raises(ValueError, match="potential-field-weight"):
            apply_reward_variant_arguments(_parse(["--potential-field-weight", value]), {})


def test_the_summary_records_both_ladder_overrides():
    reward_config = {"wf": 1.0}
    apply_reward_variant_arguments(_parse(LADDER_FLAGS), reward_config)
    summary = reward_variant_summary(reward_config)
    assert summary["lateral_target_fallback"] == "center"
    assert summary["potential_field_weight"] == "0.0"
    # The variants this run does not use must still read as the notebook's.
    assert summary["reward_mode"] == "reciprocal"
    assert summary["overtake_detection"] == "step"
    assert summary["speed_target"] == "nominal"


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
    env.reset(seed=11)
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


def _ladder_reward_config(namespace) -> dict:
    reward_config = copy.deepcopy(namespace["REWARD_CONFIG"])
    apply_reward_variant_arguments(_parse(LADDER_FLAGS), reward_config)
    return reward_config


@pytest.mark.slow
def test_zero_weight_drops_the_field_term_from_the_real_reward():
    namespace = _notebook_namespace()
    baseline = _rollout(namespace, namespace["REWARD_CONFIG"])
    stripped = _rollout(namespace, _ladder_reward_config(namespace))
    assert baseline and len(baseline) == len(stripped)

    cf_values = np.array([info["karalakou_cf"] for _, _, info in baseline])
    assert cf_values.max() > 0.05, "the field cost must bind somewhere for this to be a test"

    config = namespace["REWARD_CONFIG"]
    for (_, _, base_info), (_, reward, info) in zip(baseline, stripped):
        # The reward never feeds back into the dynamics, so both rollouts see
        # the same states; only the lateral target and wf differ.
        assert info["karalakou_cf"] == pytest.approx(base_info["karalakou_cf"], abs=1e-12)
        denominator = (
            config["epsilon_r"]
            + config["wx"] * info["karalakou_cx"]
            + config["wy"] * info["karalakou_cy"]
            + config.get("way", 0.0) * info["karalakou_cay"]
        )
        expected = (
            config["epsilon_r"] / max(denominator, 1e-9)
            + info["karalakou_progress_reward"]
            - info["karalakou_jerk_penalty"]
        )
        if info["karalakou_ego_collision"] > 0.5:
            expected += config["collision_penalty"]
        elif info["karalakou_overtakes"] > 0:
            expected += config["overtake_bonus"]
        assert reward == pytest.approx(expected, abs=1e-9)


@pytest.mark.slow
def test_the_ladder_configuration_tracks_the_road_centre_with_no_field_term():
    namespace = _notebook_namespace()
    records = _rollout(namespace, _ladder_reward_config(namespace))
    road_width = float(namespace["ENV_CONFIG"]["road_width"])
    no_gap = [
        (observation, info)
        for observation, _reward, info in records
        if info["karalakou_zone_found"] < 0.5
    ]
    assert no_gap, "the canonical density should produce no-gap steps"
    for observation, info in no_gap:
        assert info["karalakou_target_y"] == pytest.approx(0.5 * road_width)
        assert observation[1] == pytest.approx(0.0, abs=1e-6)
    # cf is still published for provenance even though it no longer scores.
    assert any(info["karalakou_cf"] > 0.0 for _, _, info in records)
