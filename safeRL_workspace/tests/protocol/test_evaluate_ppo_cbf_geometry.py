from __future__ import annotations

import json

from scripts.evaluation import evaluate_ppo_cbf_geometry as evaluator


def test_nested_nominal_pilot_config_requires_explicit_variant(tmp_path):
    path = tmp_path / "run_config.json"
    path.write_text(
        json.dumps(
            {
                "training_config": {
                    "training_seed": 307,
                    "target_timesteps": 500_000,
                    "env_config": {"simulation_frequency": 100},
                    "reward_config": {"wx": 1.0},
                },
                "completed_timesteps": 500_000,
            }
        ),
        encoding="utf-8",
    )

    resolved = evaluator._load_run_config(path, explicit_variant="ppo_nominal")

    assert resolved["variant"] == "ppo_nominal"
    assert resolved["training_seed"] == 307
    assert resolved["training_steps"] == 500_000


def test_current_geometry_preserves_legacy_axes_and_applies_requested_gains():
    namespace = {
        "CBF_RELATIVE_ELLIPSE_A": 9.0,
        "CBF_RELATIVE_ELLIPSE_B": 8.0,
        "CBF_K0": 5.29,
        "CBF_K1": 3.68,
        "CBF_PSI1_GAIN": 2.3,
        "CBF_AX_BOUNDS": [-3.0, 3.0],
        "CBF_AY_BOUNDS": [-3.0, 3.0],
    }
    run_config = {
        "source": {
            "fixed_cbf_snapshot": {
                "k0": 6.0,
                "k1": 7.0,
                "eps_side": 0.2,
            },
            "training_signature": {
                "cbf": {
                    "CBF_RELATIVE_ELLIPSE_A": 1.0,
                    "CBF_RELATIVE_ELLIPSE_B": 2.0,
                }
            },
        }
    }

    effective, ignored = evaluator._apply_saved_cbf_snapshot(
        namespace, run_config, k0=2.5, k1=1.5
    )

    assert namespace["CBF_RELATIVE_ELLIPSE_A"] == 9.0
    assert namespace["CBF_RELATIVE_ELLIPSE_B"] == 8.0
    assert ignored == {"CBF_RELATIVE_ELLIPSE_A": 1.0, "CBF_RELATIVE_ELLIPSE_B": 2.0}
    assert effective["CBF_K0"] == 2.5
    assert effective["CBF_K1"] == 1.5
    assert effective["CBF_EPS_SIDE"] == 0.2
    assert effective["CBF_PSI1_GAIN"] == 2.3
