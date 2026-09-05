import pandas as pd
import pytest


import scripts.evaluation.compare_nominal_ppo_ddpg as comparison


def _scenario_row(step: int, seed: int, *, offset: float = 0.0) -> dict:
    return {
        "model_timestep": step,
        "scenario_seed": seed,
        "initial_state_hash": f"state-{seed}",
        "timesteps": 10,
        "total_return": 10.0 + offset,
        "total_distance_m": 20.0,
        "distinct_ego_collision_events": 1 if seed == 2 else 0,
        "ego_collisions_per_km": 0.0,
        "distance_per_collision_m": 20.0,
        "distance_per_collision_exposure_bound_m": 20.0,
        "mean_abs_speed_error": 2.0 + offset,
        "episode_segments": 1,
        "episode_length_sum": 10.0,
        "episode_length_mean": 10.0,
        "nominal_action_saturation_rate": 0.1,
    }


def test_aggregate_checkpoint_uses_exposure_totals():
    result = comparison.aggregate_checkpoint(
        pd.DataFrame([_scenario_row(10, 1), _scenario_row(10, 2)])
    )

    assert result["timesteps"] == 20
    assert result["return_per_timestep"] == pytest.approx(1.0)
    assert result["total_distance_m"] == pytest.approx(40.0)
    assert result["distinct_ego_collision_events"] == 1
    assert result["ego_collisions_per_km"] == pytest.approx(25.0)
    assert result["distance_per_collision_m"] == pytest.approx(40.0)


def test_build_comparison_tables_requires_paired_reset_states(tmp_path):
    ppo_path = tmp_path / "ppo.csv"
    ddpg_path = tmp_path / "ddpg.csv"
    rows = [_scenario_row(step, seed) for step in (10, 20, 30) for seed in (1, 2)]
    pd.DataFrame(rows).to_csv(ppo_path, index=False)
    pd.DataFrame(rows).to_csv(ddpg_path, index=False)

    curve, final = comparison.build_comparison_tables(
        ppo_scenarios=ppo_path,
        ddpg_scenarios=ddpg_path,
        timesteps=30,
        checkpoint_interval=10,
        eval_seeds=[1, 2],
    )

    assert set(curve["algorithm"]) == {"PPO", "DDPG"}
    assert set(curve["model_timestep"]) == {10, 20, 30}
    assert list(final["algorithm"]) == ["PPO", "DDPG"]

    altered = pd.DataFrame(rows)
    altered.loc[altered["scenario_seed"] == 1, "initial_state_hash"] = "different"
    altered.to_csv(ddpg_path, index=False)
    with pytest.raises(RuntimeError, match="reset states differ"):
        comparison.build_comparison_tables(
            ppo_scenarios=ppo_path,
            ddpg_scenarios=ddpg_path,
            timesteps=30,
            checkpoint_interval=10,
            eval_seeds=[1, 2],
        )


def test_formulation_signature_changes_when_reward_changes():
    env = {"bounds": {"ax_min": -3.0, "ax_max": 3.0}}
    reward = {"reward_mode": "reciprocal", "use_current_potential": 1.0}
    first = comparison.formulation_signature(env, reward)
    second = comparison.formulation_signature(env, {**reward, "epsilon_r": 0.1})
    assert first != second
