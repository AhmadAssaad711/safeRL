from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from scripts.common.ppo_cbf_env import CBFContextPhysicalActionWrapper
from scripts.training.run_cbf_filter_ablation import cbf_state_occupancy_metrics


def _test_namespace() -> dict[str, object]:
    ego = {"x": 0.0, "y": 50.0, "vx": 16.0, "vy": 0.0, "width": 2.0}
    neighbor = {"x": 0.0, "y": 0.0, "vx": 19.0, "vy": 0.0, "width": 2.0}

    def get_ego_state(_env):
        return ego

    def get_neighbor_states(_env, *, neighbor_range):
        del neighbor_range
        return [neighbor]

    def pairwise_cbf_geometry(_ego, _neighbor, *, eps_side):
        del eps_side
        return (1.0,)

    def pairwise_relative_state(_ego, _neighbor):
        return (0.0, 0.0, -3.0, 0.0)

    def centerline_barrier_derivatives(_position, _ego, _neighbor, _eps_side):
        return (
            1.0,
            np.asarray([1.0, 0.0]),
            np.zeros((2, 2)),
            10.0,
            1.0,
            1.0,
        )

    return {
        "get_ego_state": get_ego_state,
        "get_neighbor_states": get_neighbor_states,
        "pairwise_cbf_geometry": pairwise_cbf_geometry,
        "pairwise_relative_state": pairwise_relative_state,
        "centerline_barrier_derivatives": centerline_barrier_derivatives,
        "_lane_free_base": lambda _env: SimpleNamespace(
            config={"road_width": 100.0, "sensing_range": 90.0}
        ),
        "CBF_QP_FEASIBILITY_TOL": 1e-5,
        "CBF_PSI1_GAIN": 2.3,
    }


def _wrapper(namespace: dict[str, object]) -> CBFContextPhysicalActionWrapper:
    wrapper = object.__new__(CBFContextPhysicalActionWrapper)
    wrapper.namespace = namespace
    wrapper.neighbor_range = 90.0
    wrapper.eps_side = 0.1
    wrapper.psi1_gain = 2.3
    return wrapper


def test_reset_psi1_uses_critical_first_level_gain_not_hocbf_k1():
    namespace = _test_namespace()
    diagnostics = _wrapper(namespace)._initial_safety_diagnostics()

    # h_dot=-3 and h=1: h_dot + 2.3*h = -0.7. Reusing HOCBF k1=4.6
    # would incorrectly report +1.6 for this same state.
    np.testing.assert_allclose(diagnostics["cbf_initial_min_psi1"], -0.7)
    np.testing.assert_allclose(diagnostics["cbf_initial_min_psi"], -0.7)
    np.testing.assert_allclose(diagnostics["cbf_psi1_gain"], 2.3)
    assert diagnostics["cbf_initial_safe_set"] is False


def test_runtime_psi1_monitor_uses_the_same_critical_gain():
    namespace = _test_namespace()
    env = SimpleNamespace(unwrapped=SimpleNamespace(config={"road_width": 100.0}))

    metrics = cbf_state_occupancy_metrics(
        namespace,
        env,  # type: ignore[arg-type]
        eps_side=0.1,
        psi1_gain=2.3,
    )

    np.testing.assert_allclose(metrics["psi1_min"], -0.7)
    np.testing.assert_allclose(metrics["psi1_gain"], 2.3)


def test_wrapper_constructor_accepts_separate_psi1_gain():
    class BareEnv(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    wrapper = CBFContextPhysicalActionWrapper(
        BareEnv(),
        namespace={"CBF_PSI1_GAIN": 2.3},
        ax_bounds=(-3.0, 3.0),
        ay_bounds=(-3.0, 3.0),
        neighbor_range=90.0,
        eps_side=0.1,
        k0=5.29,
        k1=4.6,
        max_neighbor_constraints=1,
        psi1_gain=2.3,
        base_observation_dim=2,
    )
    assert wrapper.psi1_gain == 2.3
