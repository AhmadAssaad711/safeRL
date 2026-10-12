"""Contract tests for the PPO-Lagrangian CMDP trainer.

Runs against a tiny synthetic environment carrying the real CBF context
layout, so the whole loop -- cost scoring, cost GAE, dual update, two critics
-- is exercised deterministically without the simulator.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

import gymnasium as gym

from scripts.common.cbf_projection import CBFContextLayout
from saferl.algorithms.cmdp import CMDPConfig
from saferl.algorithms.ppo_lagrangian import (
    COST_SOURCE_ACTION,
    COST_SOURCE_STATE,
    CMDPActorCriticPolicy,
    CostRolloutBuffer,
    PPOLagrangian,
)
from saferl.safety.state_cost import StateCostParameters

BASE_DIM = 8
MAX_CONSTRAINTS = 4
LAYOUT = CBFContextLayout(base_observation_dim=BASE_DIM, max_constraints=MAX_CONSTRAINTS)
OBS_DIM = LAYOUT.observation_dim


class _ContextEnv(gym.Env):
    """Synthetic env with a CBF context and a reachable constraint system.

    One active half-space ``a_x <= -1`` keeps a sizeable part of the action box
    infeasible, so the action cost is non-zero on most steps and the dual
    variable actually moves inside a short test.
    """

    metadata: dict = {}

    def __init__(self, seed: int = 0) -> None:
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-3.0, high=3.0, shape=(2,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)
        self._steps = 0

    def _observation(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[:BASE_DIM] = self._rng.normal(size=BASE_DIM).astype(np.float32)
        # Row 0: [1, 0] . u <= -1.0, active. Remaining rows masked off.
        obs[LAYOUT.rows_start] = 1.0
        obs[LAYOUT.rows_start + 1] = 0.0
        obs[LAYOUT.bounds_start] = -1.0
        obs[LAYOUT.mask_start] = 1.0
        return obs

    def current_constraint_system(self) -> dict:
        """Per-barrier h / h_dot, as the real wrapper exposes them."""

        return {
            "barrier_h": np.array([0.4, 2.0]),
            "barrier_h_dot": np.array([-0.3, 0.1]),
            "rows": np.array([[1.0, 0.0]]),
            "bounds": np.array([-1.0]),
            "hash": "synthetic",
        }

    def set_projection_record(self, *args, **kwargs) -> None:
        """Accepted and ignored; the real wrapper records diagnostics here."""

    def reset(self, *, seed=None, options=None):
        self._steps = 0
        return self._observation(), {}

    def step(self, action):
        self._steps += 1
        reward = float(-np.square(np.asarray(action, dtype=float)).sum())
        terminated = False
        truncated = self._steps >= 16
        return self._observation(), reward, terminated, truncated, {}


def _make_env(n_envs: int = 2) -> DummyVecEnv:
    return DummyVecEnv([lambda i=i: _ContextEnv(seed=i) for i in range(n_envs)])


def _make_model(cost_source: str, **config_kwargs) -> PPOLagrangian:
    action_cost_form = config_kwargs.pop("action_cost_form", "projection")
    config = CMDPConfig(
        cost_limit=config_kwargs.pop("cost_limit", 0.01),
        gamma_c=0.98,
        gae_lambda_c=0.95,
        dual_lr=config_kwargs.pop("dual_lr", 0.5),
        **config_kwargs,
    )
    return PPOLagrangian(
        CMDPActorCriticPolicy,
        _make_env(),
        cmdp_config=config,
        cost_source=cost_source,
        action_cost_form=action_cost_form,
        state_cost_params=StateCostParameters(lambda_s=1.0, boundary_count=1),
        n_steps=16,
        batch_size=16,
        n_epochs=2,
        seed=307,
        device="cpu",
        # The algorithm keeps its own layout for the rollout-time projection,
        # separate from the policy's.
        cbf_base_observation_dim=BASE_DIM,
        cbf_max_constraints=MAX_CONSTRAINTS,
        policy_kwargs={
            "cbf_base_observation_dim": BASE_DIM,
            "cost_critic_arch": [16, 16],
            "net_arch": {"pi": [16], "vf": [16]},
        },
    )


# --------------------------------------------------------------------------
# Construction contracts
# --------------------------------------------------------------------------


def test_filter_must_be_off():
    """The CMDP arms execute the raw action; 'cbf' execution is a design error."""

    with pytest.raises(ValueError, match="execution_mode must be 'box'"):
        PPOLagrangian(
            CMDPActorCriticPolicy,
            _make_env(),
            cmdp_config=CMDPConfig(cost_limit=0.01),
            execution_mode="cbf",
            n_steps=16,
            batch_size=16,
            device="cpu",
            policy_kwargs={"cbf_base_observation_dim": BASE_DIM},
        )


def test_unknown_cost_source_is_rejected():
    with pytest.raises(ValueError, match="cost_source"):
        _make_model("reward")


def test_cmdp_config_is_required_for_a_real_model():
    """cost_limit has no default, so a missing config must not be invented.

    The only path allowed to omit it is SB3's load(), which passes
    _init_setup_model=False and restores the config from the pickle.
    """

    with pytest.raises(ValueError, match="cmdp_config is required"):
        PPOLagrangian(
            CMDPActorCriticPolicy,
            _make_env(),
            n_steps=16,
            batch_size=16,
            device="cpu",
            cbf_base_observation_dim=BASE_DIM,
            cbf_max_constraints=MAX_CONSTRAINTS,
            policy_kwargs={"cbf_base_observation_dim": BASE_DIM},
        )


def test_cost_critic_is_independent_of_the_reward_critic():
    """No parameter may be shared between V_R and V_C.

    A shared trunk would let cost regression reshape the reward critic's
    representation, so a CMDP effect could be representation interference.
    """

    model = _make_model(COST_SOURCE_ACTION)
    cost_ids = {id(p) for p in model.policy.cost_critic.parameters()}
    reward_ids = {id(p) for p in model.policy.value_net.parameters()}
    extractor_ids = {id(p) for p in model.policy.mlp_extractor.parameters()}
    assert cost_ids
    assert not (cost_ids & reward_ids)
    assert not (cost_ids & extractor_ids)


def test_cost_critic_is_in_the_optimizer():
    """An unoptimized cost critic would leave V_C at its initialization."""

    model = _make_model(COST_SOURCE_ACTION)
    optimized = {
        id(p) for group in model.policy.optimizer.param_groups for p in group["params"]
    }
    assert all(id(p) in optimized for p in model.policy.cost_critic.parameters())


def test_cost_critic_is_grouped_with_the_value_parameters():
    """The cost critic must not be clipped together with the actor.

    The base split groups by '"value_net" in name'. The projected policy's
    extra head is called safety_value_net and matches by coincidence; this one
    is cost_critic and would otherwise land in the policy group, so enabling
    separate_grad_clip would scale the policy gradient by the cost critic's
    gradient -- the opposite of the intended fix.
    """

    model = _make_model(COST_SOURCE_ACTION)
    model.learn(total_timesteps=32)
    policy_params, value_params = model._gradient_parameter_groups()
    cost_ids = {id(p) for p in model.policy.cost_critic.parameters()}
    assert cost_ids & {id(p) for p in value_params} == cost_ids
    assert not (cost_ids & {id(p) for p in policy_params})


def test_separate_grad_clip_leaves_the_actor_group_cost_free():
    """With the split on, grad_norm_pi must exclude the cost critic."""

    model = _make_model(COST_SOURCE_ACTION)
    model.separate_grad_clip = True
    model.learn(total_timesteps=32)
    policy_params, _ = model._gradient_parameter_groups()
    actor_names = {
        name
        for name, param in model.policy.named_parameters()
        if any(param is p for p in policy_params)
    }
    assert not any(name.startswith("cost_critic") for name in actor_names)


def test_cost_critic_head_is_unconstrained():
    """No softplus: the head must be able to emit negative values for GAE."""

    model = _make_model(COST_SOURCE_ACTION)
    with th.no_grad():
        for module in model.policy.cost_critic:
            if isinstance(module, th.nn.Linear):
                module.bias.fill_(-5.0)
        obs = th.zeros((2, OBS_DIM))
        assert float(model.policy.predict_cost_values(obs).mean()) < 0.0


def test_cost_critic_sees_only_the_base_observation_by_default():
    model = _make_model(COST_SOURCE_ACTION)
    assert model.policy.cost_critic_sees_context is False
    first = model.policy.cost_critic[0]
    assert first.in_features == BASE_DIM


# --------------------------------------------------------------------------
# Buffer contracts
# --------------------------------------------------------------------------


def test_cost_buffer_round_trips_costs_through_a_minibatch():
    buffer = CostRolloutBuffer(
        4,
        spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32),
        spaces.Box(low=-3.0, high=3.0, shape=(2,), dtype=np.float32),
        gamma_c=1.0,
        gae_lambda_c=1.0,
        n_envs=1,
    )
    for step in range(4):
        buffer.add(
            np.zeros((1, OBS_DIM), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            th.zeros(1),
            th.zeros(1),
            costs=float(step),
            cost_values=0.0,
        )
    assert buffer.mean_raw_cost() == pytest.approx(1.5)
    buffer.compute_cost_returns(last_cost_values=th.zeros(1), dones=np.ones(1))
    # Undiscounted, no bootstrap: the first target is the whole cost sum.
    assert buffer.cost_returns[0, 0] == pytest.approx(6.0)
    batch = next(buffer.get(4))
    assert batch.costs.shape == (4,)
    assert batch.cost_advantages.shape == (4,)


def test_cost_buffer_validates_per_env_width():
    buffer = CostRolloutBuffer(
        2,
        spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32),
        spaces.Box(low=-3.0, high=3.0, shape=(2,), dtype=np.float32),
        n_envs=2,
    )
    with pytest.raises(ValueError, match="one value per environment"):
        buffer.add(
            np.zeros((2, OBS_DIM), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            th.zeros(2),
            th.zeros(2),
            costs=[1.0],
        )


# --------------------------------------------------------------------------
# End-to-end contracts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cost_source", [COST_SOURCE_ACTION, COST_SOURCE_STATE])
def test_learn_runs_and_records_the_dual_state(cost_source):
    model = _make_model(cost_source)
    model.learn(total_timesteps=32)
    assert model.cmdp_training_diagnostics
    record = model.cmdp_training_diagnostics[-1]
    for key in (
        "cmdp/lambda",
        "cmdp/cost_objective",
        "cmdp/constraint_violation",
        "cost_mean_raw",
        "cost_value_loss",
        "cost_surrogate",
    ):
        assert key in record
    assert record["cost_source"] == cost_source


def test_violating_the_constraint_drives_lambda_up():
    """With d = 0 and a genuinely costly policy, lambda must leave zero."""

    model = _make_model(COST_SOURCE_ACTION, cost_limit=0.0, dual_lr=1.0)
    assert model.dual.value == 0.0
    model.learn(total_timesteps=32)
    assert model.dual.value > 0.0
    assert model.cmdp_training_diagnostics[-1]["cmdp/constraint_violation"] > 0.0


def test_a_slack_constraint_leaves_lambda_at_zero():
    """A limit far above the achievable cost must keep the cost term off."""

    model = _make_model(COST_SOURCE_ACTION, cost_limit=1e6, dual_lr=1.0)
    model.learn(total_timesteps=32)
    assert model.dual.value == 0.0
    assert model.cmdp_training_diagnostics[-1]["cmdp/constraint_satisfied"] == 1.0


def test_action_cost_is_zero_when_every_action_is_feasible():
    """c_a must vanish on a policy whose actions already satisfy the rows."""

    model = _make_model(COST_SOURCE_ACTION)
    # Row 0 is a_x <= -1, so a_x = -3 is strictly feasible.
    box = np.array([[-3.0, 0.0], [-3.0, 1.0]], dtype=np.float32)
    obs = np.stack([_ContextEnv(seed=0)._observation() for _ in range(2)])
    costs, diagnostics = model._action_costs(box, obs)
    assert np.allclose(costs, 0.0)
    assert diagnostics["active_fraction"] == 0.0


def test_action_cost_grows_with_distance_outside_the_safe_set():
    model = _make_model(COST_SOURCE_ACTION)
    obs = np.stack([_ContextEnv(seed=0)._observation() for _ in range(2)])
    near = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    far = np.array([[3.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    near_cost, _ = model._action_costs(near, obs)
    far_cost, _ = model._action_costs(far, obs)
    # a_x <= -1, so the gaps are 1.0 and 4.0 giving 0.5 and 8.0.
    assert near_cost[0] == pytest.approx(0.5)
    assert far_cost[0] == pytest.approx(8.0)


def test_violation_form_needs_no_projection_and_stays_graded():
    """The reformulated action cost must order actions on a trapped state.

    The synthetic env's single row is a_x <= -1, which is satisfiable, so here
    the two forms agree; the point of the test is that the violation path runs
    end to end through the trainer and remains action-sensitive.
    """

    model = _make_model(COST_SOURCE_ACTION, action_cost_form="violation")
    obs = np.stack([_ContextEnv(seed=0)._observation() for _ in range(2)])
    safe = np.array([[-2.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
    near = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    far = np.array([[3.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    assert model._action_costs(safe, obs)[0][0] == pytest.approx(0.0)
    assert model._action_costs(near, obs)[0][0] == pytest.approx(0.5)
    assert model._action_costs(far, obs)[0][0] == pytest.approx(8.0)


def test_violation_form_trains_end_to_end():
    model = _make_model(COST_SOURCE_ACTION, action_cost_form="violation", cost_limit=0.0)
    model.learn(total_timesteps=32)
    assert model.cmdp_training_diagnostics
    assert model.dual.value > 0.0


def test_unknown_action_cost_form_is_rejected():
    with pytest.raises(ValueError, match="action_cost_form"):
        _make_model(COST_SOURCE_ACTION, action_cost_form="hinge")


def test_projection_remains_the_default_form():
    """The six completed seeds used the projection form; keep it the default."""

    assert _make_model(COST_SOURCE_ACTION).action_cost_form == "projection"


def test_state_cost_reads_the_post_step_constraint_system():
    model = _make_model(COST_SOURCE_STATE)
    env = _make_env()
    env.reset()
    costs, diagnostics = model._state_costs(env)
    assert costs.shape == (2,)
    assert np.all(costs >= 0.0)
    assert diagnostics["sample_count"] == 2


def test_state_cmdp_requires_an_enabled_cost_definition():
    with pytest.raises(ValueError, match="lambda_s > 0"):
        PPOLagrangian(
            CMDPActorCriticPolicy,
            _make_env(),
            cmdp_config=CMDPConfig(cost_limit=0.01),
            cost_source=COST_SOURCE_STATE,
            state_cost_params=StateCostParameters(lambda_s=0.0),
            n_steps=16,
            batch_size=16,
            device="cpu",
            policy_kwargs={"cbf_base_observation_dim": BASE_DIM},
        )


def test_dual_variable_survives_save_and_load(tmp_path):
    """lambda is part of the solution and must round-trip with the model."""

    model = _make_model(COST_SOURCE_ACTION, cost_limit=0.0, dual_lr=1.0)
    model.learn(total_timesteps=32)
    saved = model.dual.value
    assert saved > 0.0

    path = tmp_path / "cmdp_model.zip"
    model.save(str(path))
    restored = PPOLagrangian.load(str(path), env=_make_env(), device="cpu")
    assert restored.dual.value == pytest.approx(saved)
    assert restored.cmdp_config == model.cmdp_config
    assert restored.cost_source == model.cost_source


def test_model_loads_without_an_environment(tmp_path):
    """Evaluation paths load a checkpoint with env=None.

    On that path SB3 never sets an action space before __init__ runs, so an
    unconditional space check breaks every evaluation of a CMDP checkpoint.
    """

    model = _make_model(COST_SOURCE_ACTION)
    path = tmp_path / "no_env.zip"
    model.save(str(path))
    restored = PPOLagrangian.load(str(path), device="cpu")
    assert restored.cost_source == COST_SOURCE_ACTION
    assert restored.cmdp_config == model.cmdp_config
