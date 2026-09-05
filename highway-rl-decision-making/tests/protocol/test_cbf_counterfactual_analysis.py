from __future__ import annotations

import numpy as np
import pandas as pd


from scripts.evaluation.evaluate_cbf_counterfactuals import (
    EXPECTED_FACTORIAL_VARIANTS,
    compute_factorial_contrasts,
    filter_physical_bounds,
    make_typed_constraint_system,
    normal_tangent_decomposition,
    stable_state_hash,
    stratify_state_bank,
    typed_feasible_mask,
)


def test_stable_state_hash_is_order_and_dtype_invariant_but_state_sensitive() -> None:
    observation = np.asarray([1.0, 2.0, -3.0], dtype=np.float32)
    ego_a = {"x": np.float32(10.0), "y": 2.0, "vx": 20.0}
    ego_b = {"vx": 20.0, "y": np.float64(2.0), "x": 10.0}
    neighbors_a = [{"x": 14.0, "y": 3.0, "vx": 18.0}]
    neighbors_b = [{"vx": np.float32(18.0), "y": 3.0, "x": 14.0}]

    first = stable_state_hash(observation, ego_a, neighbors_a, 12.0)
    second = stable_state_hash(observation.astype(np.float64), ego_b, neighbors_b, np.float32(12.0))
    changed = stable_state_hash(observation + np.asarray([0.0, 0.0, 0.01]), ego_b, neighbors_b, 12.0)

    assert first == second
    assert changed != first


def _fake_candidate(index: int, category: str) -> dict[str, object]:
    values: dict[str, object] = {
        "observation": np.asarray([float(index), float(index + 1)], dtype=np.float32),
        "ego": {"x": float(index), "y": 4.0, "vx": 20.0, "vy": 0.0, "width": 2.0},
        "neighbors": [{"x": float(index + 10), "y": 4.0}],
        "road_width": 12.0,
        "h_min": 5.0,
        "intervention": False,
        "overtaking": False,
        "traffic_density_per_km": 1.0,
        "source_training_seed": 1,
        "source_variant": "b_filtered",
        "scenario_seed": 100,
        "scenario_index": 0,
        "scenario_step": index,
        "h_dot": 0.0,
        "ttc_s": 30.0,
        "vehicle_spacing_m": 5.0,
        "neighbor_count": 1,
    }
    if category == "near_boundary":
        values["h_min"] = 0.1
    elif category == "intervention":
        values["intervention"] = True
    elif category == "dense":
        values["traffic_density_per_km"] = 25.0
    elif category == "overtaking":
        values["overtaking"] = True
    values["state_hash"] = stable_state_hash(
        values["observation"], values["ego"], values["neighbors"], values["road_width"]
    )
    return values


def test_stratified_bank_is_deterministic_unique_and_covers_all_categories() -> None:
    candidates = []
    index = 0
    for category in ("normal", "near_boundary", "intervention", "dense", "overtaking"):
        for _ in range(5):
            candidates.append(_fake_candidate(index, category))
            index += 1

    first, first_metadata = stratify_state_bank(
        candidates,
        2,
        seed=123,
        near_boundary_margin=1.0,
        dense_threshold=10.0,
    )
    second, second_metadata = stratify_state_bank(
        list(reversed(candidates)),
        2,
        seed=123,
        near_boundary_margin=1.0,
        dense_threshold=10.0,
    )

    assert [(item["stratum"], item["state_hash"]) for item in first] == [
        (item["stratum"], item["state_hash"]) for item in second
    ]
    assert len({item["state_hash"] for item in first}) == len(first)
    assert {item["stratum"] for item in first} == {
        "normal",
        "near_boundary",
        "intervention",
        "dense",
        "overtaking",
    }
    assert first_metadata["selection_counts"] == second_metadata["selection_counts"]
    assert all(value == 2 for value in first_metadata["selection_counts"].values())
    assert first_metadata["bank_hash"] == second_metadata["bank_hash"]


def test_normal_tangent_decomposition_handles_single_and_rank_deficient_rows() -> None:
    single = normal_tangent_decomposition([3.0, 4.0], [[1.0, 0.0]])
    np.testing.assert_allclose(single["normal"], [3.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(single["tangent"], [0.0, 4.0], atol=1e-12)
    assert single["rank"] == 1
    assert single["reconstruction_error"] < 1e-12
    assert single["orthogonality_error"] < 1e-12

    duplicated = normal_tangent_decomposition([3.0, 4.0], [[1.0, 0.0], [2.0, 0.0]])
    np.testing.assert_allclose(duplicated["normal"], [3.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(duplicated["tangent"], [0.0, 4.0], atol=1e-12)
    assert duplicated["rank"] == 1

    full_rank = normal_tangent_decomposition([3.0, 4.0], [[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(full_rank["normal"], [3.0, 4.0], atol=1e-12)
    np.testing.assert_allclose(full_rank["tangent"], [0.0, 0.0], atol=1e-12)
    assert full_rank["rank"] == 2


def test_typed_feasible_mask_reports_overall_and_semantic_masks() -> None:
    # x <= 1 and y >= 0
    system = make_typed_constraint_system(
        [[1.0, 0.0], [0.0, -1.0]],
        [1.0, 0.0],
        ["neighbor_front", "road_left"],
    )
    ax = np.asarray([[0.0, 2.0], [0.0, 2.0]])
    ay = np.asarray([[1.0, 1.0], [-1.0, -1.0]])
    masks = typed_feasible_mask(ax, ay, system)

    assert masks["all"].dtype == np.bool_
    np.testing.assert_array_equal(masks["neighbor_front"], [[True, False], [True, False]])
    np.testing.assert_array_equal(masks["road_left"], [[True, True], [False, False]])
    np.testing.assert_array_equal(masks["all"], [[True, False], [False, False]])


def test_filter_bounds_prefer_cbf_configuration_over_environment_bounds() -> None:
    namespace = {"CBF_AX_BOUNDS": (-2.5, 1.5), "CBF_AY_BOUNDS": (-0.75, 0.9)}
    run_config = {
        "env_config": {
            "bounds": {"ax_min": -10.0, "ax_max": 10.0, "ay_min": -5.0, "ay_max": 5.0}
        }
    }
    low, high = filter_physical_bounds(namespace, run_config)
    np.testing.assert_allclose(low, [-2.5, -0.75])
    np.testing.assert_allclose(high, [1.5, 0.9])


def test_factorial_contrast_math_uses_paired_main_and_interaction_coefficients() -> None:
    values = {
        (False, False): 1.0,
        (True, False): 3.0,
        (False, True): 5.0,
        (True, True): 11.0,
    }
    summary = pd.DataFrame(
        [
            {"training_seed": 7, "variant": EXPECTED_FACTORIAL_VARIANTS[cell], "metric": value}
            for cell, value in values.items()
        ]
    )
    effects = compute_factorial_contrasts(summary, ["metric"])
    estimates = effects.set_index("effect")["estimate"].to_dict()

    assert estimates["reward_main_effect"] == 4.0
    assert estimates["actor_loss_main_effect"] == 6.0
    assert estimates["reward_actor_interaction"] == 4.0
