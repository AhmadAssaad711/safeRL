from __future__ import annotations

from pathlib import Path

from scripts.ops.benchmark_simulator import run_benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_is_seed_deterministic_and_reports_physics_rate():
    kwargs = {
        "project_root": PROJECT_ROOT,
        "traffic_model": "mtm",
        "guard": True,
        "vehicles": 8,
        "steps": 3,
        "warmup": 1,
        "seed": 307,
        "simulation_frequency": 4,
        "policy_frequency": 4,
    }
    first = run_benchmark(**kwargs)
    second = run_benchmark(**kwargs)

    assert first["physics_frames"] == 3
    assert first["final_state_sha256"] == second["final_state_sha256"]
    assert first["mean_reward"] == second["mean_reward"]
    assert first["policy_steps_per_s"] > 0.0
    assert first["physics_frames_per_s"] == first["policy_steps_per_s"]
