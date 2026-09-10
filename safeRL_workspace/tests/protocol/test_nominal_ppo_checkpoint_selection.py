from __future__ import annotations

import pandas as pd

from scripts.evaluation.evaluate_nominal_ppo_checkpoints import (
    rank_checkpoint_candidates,
)


def _candidate(seed: int, step: int, completion: float, collisions: float, reward: float) -> dict:
    return {
        "training_seed": seed,
        "pilot_config": "Q0_current_aligned",
        "model_timestep": step,
        "collision_free_completion_rate": completion,
        "collision_episode_rate": collisions,
        "ego_collisions_per_km_mean": collisions,
        "episode_return_mean": reward,
        "mean_abs_speed_deviation_mean": 1.0,
    }


def test_nominal_checkpoint_selection_is_per_seed_and_prefers_completion_then_return():
    candidates = pd.DataFrame(
        [
            _candidate(307, 50_000, 0.50, 0.50, 100.0),
            _candidate(307, 100_000, 1.00, 0.00, 10.0),
            _candidate(307, 150_000, 1.00, 0.00, 20.0),
            _candidate(308, 50_000, 0.00, 1.00, 500.0),
            _candidate(308, 100_000, 0.50, 0.50, 1.0),
        ]
    )

    ranked = rank_checkpoint_candidates(candidates)
    selected = ranked[ranked["selection_rank"] == 1]

    assert selected.set_index("training_seed")["model_timestep"].to_dict() == {
        307: 150_000,
        308: 100_000,
    }
