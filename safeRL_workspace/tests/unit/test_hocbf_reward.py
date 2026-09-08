from __future__ import annotations

import json

import gymnasium as gym
import numpy as np

from scripts.common.ppo_cbf_env import CBFContextPhysicalActionWrapper
from scripts.training import run_ppo_cbf_progression as progression


class _BaseEnv(gym.Env):
    metadata = {}

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(42,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.config = {
            "road_width": 10.2,
            "cbf_substep_filtering": False,
            "bounds": {
                "ax_min": -3.0,
                "ax_max": 3.0,
                "ay_min": -3.0,
                "ay_max": 3.0,
            },
        }
        self.vehicle = object()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(42, dtype=np.float32), {}

    def step(self, action):
        assert np.all(np.abs(np.asarray(action, dtype=float)) <= 1.0 + 1e-6)
        return np.zeros(42, dtype=np.float32), 1.0, False, False, {}


def _namespace(base: _BaseEnv) -> dict[str, object]:
    return {
        "get_ego_state": lambda _env: {
            "y": 9.0,
            "vy": 0.0,
            "width": 1.8,
        },
        "get_neighbor_states": lambda _env, neighbor_range: [],
        "_lane_free_base": lambda _env: base,
        "_physical_to_normalized_action": lambda _wrapper, action: (
            np.asarray(action, dtype=np.float32) / 3.0
        ),
        "CBF_QP_FEASIBILITY_TOL": 1e-5,
    }


def test_direct_hocbf_reward_uses_second_order_action_residual():
    base = _BaseEnv()
    env = CBFContextPhysicalActionWrapper(
        base,
        namespace=_namespace(base),
        ax_bounds=(-3.0, 3.0),
        ay_bounds=(-3.0, 3.0),
        neighbor_range=90.0,
        eps_side=0.10,
        k0=1.0,
        k1=1.0,
        max_neighbor_constraints=12,
        base_observation_dim=42,
        project_inputs=False,
        hocbf_reward_lambda=2.0,
        hocbf_reward_scale=1.0,
    )
    try:
        observation, _ = env.reset(seed=7)
        assert observation.shape[0] > 42
        _, reward, _, _, info = env.step(np.asarray([0.0, 3.0], dtype=np.float32))

        # At y=9.0, the right boundary has h=0.3.  With vy=0, k0=k1=1,
        # psi2 = h - ay = -2.7 for the raw physical action.
        np.testing.assert_allclose(info["cbf_hocbf_raw_min_margin"], -2.7)
        np.testing.assert_allclose(info["cbf_hocbf_raw_max_violation"], 2.7)
        np.testing.assert_allclose(
            info["cbf_hocbf_reward_penalty"], 2.0 * 2.7**2
        )
        np.testing.assert_allclose(reward, 1.0 - 2.0 * 2.7**2)
    finally:
        env.close()


class _CalibrationEnv:
    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []
        self.closed = False
        self._step = 0

    def reset(self, *, seed=None):
        self.seed = seed
        self._step = 0
        return np.zeros(42, dtype=np.float32), {}

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self._step += 1
        return (
            np.zeros(42, dtype=np.float32),
            0.0,
            False,
            False,
            {
                "cbf_hocbf_raw_max_abs_residual": 10.0 * self._step,
                "cbf_hocbf_raw_max_violation": float(self._step),
                "raw_action_phys": np.asarray(action, dtype=np.float32),
            },
        )

    def close(self):
        self.closed = True


class _CalibrationModel:
    def predict(self, observation, deterministic=False):
        assert deterministic
        return np.asarray([0.25, -0.5], dtype=np.float32), None


def test_hocbf_scale_calibration_uses_fixed_nominal_policy_rollout(
    monkeypatch, tmp_path
):
    env = _CalibrationEnv()
    model = _CalibrationModel()
    model_path = tmp_path / "nominal_model.zip"
    model_path.write_bytes(b"nominal calibration checkpoint")
    loaded = {}

    def fake_load_model(variant, path, device, env=None):
        loaded.update(variant=variant, path=path, device=device, env=env)
        return model

    monkeypatch.setattr(progression, "load_model", fake_load_model)
    monkeypatch.setattr(
        progression,
        "make_ppo_cbf_env",
        lambda *args, **kwargs: env,
    )
    output_path = tmp_path / "hocbf_scale_calibration.json"

    scale = progression.calibrate_hocbf_reward_scale(
        {},
        nominal_model_path=model_path,
        model_device="cpu",
        nominal_training_seed=307,
        env_config={
            "simulation_frequency": 100,
            "policy_frequency": 20,
        },
        reward_config={},
        seed=901,
        steps=2,
        output_path=output_path,
    )

    np.testing.assert_allclose(env.actions, [[0.25, -0.5], [0.25, -0.5]])
    assert env.closed
    assert loaded == {
        "variant": "ppo_nominal",
        "path": model_path,
        "device": "cpu",
        "env": None,
    }
    np.testing.assert_allclose(scale, 19.5)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["calibration_policy"] == "ppo_nominal"
    assert payload["action_source"] == "nominal_policy_deterministic"
    assert payload["nominal_training_seed"] == 307
    assert payload["calibration_rollout_seed"] == 901
    assert payload["observed_policy_steps"] == 2
    np.testing.assert_allclose(payload["selected_psi_scale"], 19.5)
