"""Contract tests for the CMDP core: cost GAE, constraint estimate, dual."""

from __future__ import annotations

import numpy as np
import pytest

from saferl.algorithms.cmdp import (
    BATCH_MEAN_ESTIMATOR,
    EPISODE_SUM_ESTIMATOR,
    EPISODE_START_ESTIMATOR,
    VALUE_MEAN_ESTIMATOR,
    CMDPConfig,
    DualVariable,
    cmdp_config_from_mapping,
    cost_gae,
    cost_objective_estimate,
)

CONFIG = CMDPConfig(cost_limit=0.05, gamma_c=0.98, gae_lambda_c=0.95, dual_lr=0.1)


def _monte_carlo_cost_returns(costs, episode_starts, last_values, dones, gamma_c):
    """Reference bootstrapped Monte-Carlo cost return.

    Mirrors the recursion the projected-PPO safety buffer uses, so the
    ``gae_lambda_c = 1`` equivalence can be checked against it directly.
    """

    costs = np.asarray(costs, dtype=np.float64)
    returns = np.zeros_like(costs)
    next_returns = np.asarray(last_values, dtype=np.float64).reshape(-1)
    n_steps = costs.shape[0]
    for step in reversed(range(n_steps)):
        if step == n_steps - 1:
            next_non_terminal = 1.0 - np.asarray(dones, dtype=np.float64)
        else:
            next_non_terminal = 1.0 - np.asarray(episode_starts, dtype=np.float64)[step + 1]
        returns[step] = costs[step] + gamma_c * next_non_terminal * next_returns
        next_returns = returns[step]
    return returns


# --------------------------------------------------------------------------
# Configuration contracts
# --------------------------------------------------------------------------


def test_cost_limit_has_no_default():
    """d must be chosen from measured distributions, never inherited."""

    with pytest.raises(TypeError):
        CMDPConfig()  # type: ignore[call-arg]


def test_locked_starting_configuration():
    """The four locked decisions must be the defaults, not opt-ins.

    Normalizing A_C would obscure the lambda / cost-scale / d relationship;
    1/(1+lambda) changes the gradient magnitude and so interacts with Adam,
    entropy and clipping; and a shared critic trunk would let cost regression
    reshape the reward critic's representation.
    """

    config = CMDPConfig(cost_limit=0.05)
    assert config.normalize_cost_advantage is False
    assert config.use_penalty_scaling is False
    assert config.separate_cost_critic is True
    assert config.cost_estimator == BATCH_MEAN_ESTIMATOR
    # Average-cost J_C leaves the dual error in raw per-step units, so the
    # critic path must not be pre-scaled behind the caller's back.
    assert config.normalize_cost_objective is False
    assert config.cost_scale == 1.0


def test_cost_scale_is_one_minus_gamma_c_only_when_asked():
    normalized = CMDPConfig(
        cost_limit=1.0, gamma_c=0.98, normalize_cost_objective=True
    )
    assert normalized.cost_scale == pytest.approx(0.02)
    assert CONFIG.cost_scale == 1.0


def test_scale_costs_is_identity_under_the_locked_default():
    assert CONFIG.scale_costs([1.0, 2.0]) == pytest.approx([1.0, 2.0])
    normalized = CMDPConfig(
        cost_limit=1.0, gamma_c=0.98, normalize_cost_objective=True
    )
    assert normalized.scale_costs([1.0, 2.0]) == pytest.approx([0.02, 0.04])


def test_invalid_configurations_are_rejected():
    with pytest.raises(ValueError, match="cost_limit"):
        CMDPConfig(cost_limit=-1.0)
    with pytest.raises(ValueError, match="gamma_c"):
        CMDPConfig(cost_limit=0.1, gamma_c=1.5)
    with pytest.raises(ValueError, match="dual_lr"):
        CMDPConfig(cost_limit=0.1, dual_lr=0.0)
    with pytest.raises(ValueError, match="dual_init"):
        CMDPConfig(cost_limit=0.1, dual_init=-0.5)
    with pytest.raises(ValueError, match="dual_init must not exceed"):
        CMDPConfig(cost_limit=0.1, dual_init=5.0, dual_max=1.0)
    with pytest.raises(ValueError, match="cost_estimator"):
        CMDPConfig(cost_limit=0.1, cost_estimator="whatever")
    with pytest.raises(ValueError, match="gamma_c < 1"):
        CMDPConfig(cost_limit=0.1, gamma_c=1.0, normalize_cost_objective=True)


def test_config_round_trips_through_a_mapping():
    restored = cmdp_config_from_mapping(CONFIG.as_dict())
    assert restored == CONFIG


def test_stored_config_must_record_the_cost_limit():
    with pytest.raises(ValueError, match="cost_limit"):
        cmdp_config_from_mapping({"gamma_c": 0.98})


# --------------------------------------------------------------------------
# Cost GAE contracts
# --------------------------------------------------------------------------


def test_cost_gae_matches_hand_computed_two_step_case():
    costs = np.array([[1.0], [2.0]])
    values = np.array([[0.5], [1.5]])
    starts = np.array([[1.0], [0.0]])
    adv, targets = cost_gae(
        costs, values, starts,
        last_cost_values=[0.0], dones=[1.0],
        gamma_c=1.0, gae_lambda_c=1.0,
    )
    # Terminal step: delta = 2 + 0 - 1.5 = 0.5
    # First step:    delta = 1 + 1.5 - 0.5 = 2.0, A = 2.0 + 0.5 = 2.5
    assert adv[1, 0] == pytest.approx(0.5)
    assert adv[0, 0] == pytest.approx(2.5)
    assert targets[0, 0] == pytest.approx(3.0)
    assert targets[1, 0] == pytest.approx(2.0)


def test_lambda_one_reproduces_monte_carlo_returns():
    """gae_lambda_c = 1 is the existing Monte-Carlo cost return exactly.

    This is the claim that justifies replacing that recursion with GAE: the
    old behaviour must remain reachable as a special case.
    """

    rng = np.random.default_rng(307)
    costs = rng.random((6, 3))
    values = rng.random((6, 3))
    starts = np.zeros((6, 3))
    starts[0, :] = 1.0
    starts[3, 1] = 1.0
    last_values = rng.random(3)
    dones = np.array([0.0, 1.0, 0.0])

    _, targets = cost_gae(
        costs, values, starts,
        last_cost_values=last_values, dones=dones,
        gamma_c=0.98, gae_lambda_c=1.0,
    )
    reference = _monte_carlo_cost_returns(costs, starts, last_values, dones, 0.98)
    assert targets == pytest.approx(reference)


def test_cost_gae_does_not_bleed_across_episode_boundaries():
    """An episode start must cut the bootstrap, not carry cost backwards."""

    costs = np.array([[0.0], [10.0]])
    values = np.zeros((2, 1))
    starts = np.array([[0.0], [1.0]])  # step 1 begins a new episode
    adv, _ = cost_gae(
        costs, values, starts,
        last_cost_values=[0.0], dones=[0.0],
        gamma_c=0.99, gae_lambda_c=1.0,
    )
    # Step 0 must not see step 1's cost, because step 1 is a new episode.
    assert adv[0, 0] == pytest.approx(0.0)
    assert adv[1, 0] == pytest.approx(10.0)


def test_zero_costs_give_zero_advantage_with_zero_values():
    adv, targets = cost_gae(
        np.zeros((4, 2)), np.zeros((4, 2)), np.zeros((4, 2)),
        last_cost_values=np.zeros(2), dones=np.zeros(2),
        gamma_c=0.98, gae_lambda_c=0.95,
    )
    assert np.allclose(adv, 0.0)
    assert np.allclose(targets, 0.0)


def test_cost_gae_validates_shapes():
    with pytest.raises(ValueError, match="n_steps, n_envs"):
        cost_gae(
            np.zeros(4), np.zeros(4), np.zeros(4),
            last_cost_values=[0.0], dones=[0.0],
            gamma_c=0.98, gae_lambda_c=0.95,
        )
    with pytest.raises(ValueError, match="one prediction per environment"):
        cost_gae(
            np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)),
            last_cost_values=[0.0], dones=np.zeros(3),
            gamma_c=0.98, gae_lambda_c=0.95,
        )


# --------------------------------------------------------------------------
# Constraint-estimate contracts -- the unit-discipline trap
# --------------------------------------------------------------------------


def test_batch_mean_estimator_uses_raw_costs_without_rescaling():
    """J_C is a weighted average of RAW costs.

    Applying (1 - gamma_c) here as well would shrink every constraint
    violation by 50x at gamma_c = 0.98 and the constraint would look satisfied
    when it is not. This test is the guard against that double scaling.
    """

    raw = np.full(10, 0.2)
    estimate = cost_objective_estimate(
        BATCH_MEAN_ESTIMATOR, gamma_c=0.98, normalize=True, raw_costs=raw
    )
    assert estimate == pytest.approx(0.2)


def test_value_mean_estimator_is_already_per_step_when_normalized():
    values = np.full(5, 0.3)
    assert cost_objective_estimate(
        VALUE_MEAN_ESTIMATOR, gamma_c=0.98, normalize=True, cost_values=values
    ) == pytest.approx(0.3)
    # Unnormalized critics learn the undiscounted sum, so the factor applies.
    assert cost_objective_estimate(
        VALUE_MEAN_ESTIMATOR, gamma_c=0.98, normalize=False, cost_values=values
    ) == pytest.approx(0.006)


def test_episode_start_estimator_reads_only_episode_starts():
    values = np.array([[1.0, 5.0], [9.0, 9.0]])
    starts = np.array([[True, False], [False, True]])
    estimate = cost_objective_estimate(
        EPISODE_START_ESTIMATOR, gamma_c=0.98, normalize=True,
        cost_values=values, episode_starts=starts,
    )
    assert estimate == pytest.approx(5.0)


def test_episode_start_estimator_refuses_a_batch_with_no_starts():
    with pytest.raises(ValueError, match="no episode starts"):
        cost_objective_estimate(
            EPISODE_START_ESTIMATOR, gamma_c=0.98,
            cost_values=np.zeros((2, 2)),
            episode_starts=np.zeros((2, 2), dtype=bool),
        )


def test_estimators_require_their_inputs():
    with pytest.raises(ValueError, match="requires raw_costs"):
        cost_objective_estimate(BATCH_MEAN_ESTIMATOR, gamma_c=0.98)
    with pytest.raises(ValueError, match="requires cost_values"):
        cost_objective_estimate(VALUE_MEAN_ESTIMATOR, gamma_c=0.98)


# --------------------------------------------------------------------------
# Dual variable contracts
# --------------------------------------------------------------------------


def test_violation_raises_lambda_by_the_learning_rate_times_the_violation():
    dual = DualVariable(CONFIG)
    assert dual.value == 0.0
    info = dual.update(0.25)  # limit 0.05, so violation 0.20
    assert info["cmdp/constraint_violation"] == pytest.approx(0.20)
    assert dual.value == pytest.approx(0.02)  # 0.0 + 0.1 * 0.20
    assert info["cmdp/constraint_satisfied"] == 0.0


def test_satisfied_constraint_projects_lambda_to_zero_not_negative():
    """A safer-than-required policy must switch the cost term off.

    Letting lambda go negative would turn the constraint into a bonus for
    extra safety, which is not the problem being solved.
    """

    dual = DualVariable(CONFIG)
    dual.update(0.25)
    assert dual.value > 0.0
    for _ in range(50):
        dual.update(0.0)  # far below the limit
    assert dual.value == 0.0
    assert dual.update(0.0)["cmdp/at_lower_bound"] == 1.0


def test_lambda_respects_its_optional_cap():
    config = CMDPConfig(cost_limit=0.0, dual_lr=1.0, dual_max=0.5)
    dual = DualVariable(config)
    for _ in range(10):
        dual.update(1.0)
    assert dual.value == pytest.approx(0.5)


def test_penalty_scale_is_inert_under_the_locked_default():
    """The clean Lagrangian L_R - lambda * L_C is the starting objective."""

    config = CMDPConfig(cost_limit=0.0, dual_lr=1.0)
    dual = DualVariable(config)
    dual.update(1.0)
    assert dual.value == pytest.approx(1.0)
    assert dual.penalty_scale == 1.0


def test_penalty_scale_is_one_over_one_plus_lambda_when_enabled():
    """Available as a later stabilization experiment, off by default."""

    config = CMDPConfig(cost_limit=0.0, dual_lr=1.0, use_penalty_scaling=True)
    dual = DualVariable(config)
    assert dual.penalty_scale == pytest.approx(1.0)
    dual.update(1.0)  # lambda becomes 1.0
    assert dual.value == pytest.approx(1.0)
    assert dual.penalty_scale == pytest.approx(0.5)


def test_pid_with_zero_gains_matches_plain_ascent():
    """Kp = Kd = 0 reduces the PID controller to the baseline integrator."""

    ascent = DualVariable(CONFIG)
    pid = DualVariable(
        CMDPConfig(
            cost_limit=0.05, gamma_c=0.98, gae_lambda_c=0.95, dual_lr=0.1,
            dual_controller="pid",
        )
    )
    for cost in (0.25, 0.4, 0.01, 0.0, 0.3):
        ascent.update(cost)
        pid.update(cost)
        assert pid.value == pytest.approx(ascent.value)


def test_pid_integrator_cannot_wind_up_past_the_cap():
    """An unreachable limit must not leave lambda stuck high after recovery."""

    config = CMDPConfig(
        cost_limit=0.0, dual_lr=1.0, dual_max=2.0, dual_controller="pid"
    )
    dual = DualVariable(config)
    for _ in range(100):
        dual.update(1.0)  # limit unreachable for 100 rollouts
    assert dual.value == pytest.approx(2.0)
    # With the integrator capped at dual_max / Ki, two satisfied rollouts of
    # slack 1.0 bring lambda to zero. Uncapped it would take ~100.
    dual.update(-1.0)
    dual.update(-1.0)
    assert dual.value == 0.0


def test_pid_derivative_only_reacts_to_a_rising_cost():
    config = CMDPConfig(
        cost_limit=1.0, dual_lr=1e-9, dual_controller="pid", pid_kd=10.0,
        pid_ema=0.0,
    )
    dual = DualVariable(config)
    dual.update(1.0)
    rising = dual.update(1.5)
    assert rising["cmdp/pid_d"] == pytest.approx(5.0)
    falling = dual.update(0.5)
    assert falling["cmdp/pid_d"] == 0.0


def test_pid_proportional_term_acts_immediately():
    config = CMDPConfig(
        cost_limit=0.1, dual_lr=1e-9, dual_controller="pid", pid_kp=2.0,
        pid_ema=0.0,
    )
    dual = DualVariable(config)
    dual.update(0.6)
    assert dual.value == pytest.approx(1.0, rel=1e-6)  # 2.0 * (0.6 - 0.1)


def test_pid_config_round_trips_and_rejects_bad_values():
    config = CMDPConfig(
        cost_limit=0.1, dual_controller="pid", pid_kp=1.0, pid_kd=0.5,
        pid_ema=0.8,
    )
    assert cmdp_config_from_mapping(config.as_dict()) == config
    with pytest.raises(ValueError, match="dual_controller"):
        CMDPConfig(cost_limit=0.1, dual_controller="adam")
    with pytest.raises(ValueError, match="pid_kp"):
        CMDPConfig(cost_limit=0.1, pid_kp=-1.0)
    with pytest.raises(ValueError, match="pid_ema"):
        CMDPConfig(cost_limit=0.1, pid_ema=1.0)


def test_pid_state_round_trips_and_old_checkpoints_resume():
    config = CMDPConfig(
        cost_limit=0.0, dual_lr=0.5, dual_controller="pid", pid_kp=1.0
    )
    dual = DualVariable(config)
    dual.update(0.4)
    dual.update(0.2)
    restored = DualVariable(config)
    restored.load_state_dict(dual.state_dict())
    assert restored.update(0.3)["cmdp/lambda"] == pytest.approx(
        dual.update(0.3)["cmdp/lambda"]
    )
    legacy = DualVariable(config)
    legacy.load_state_dict({"value": 3.0, "updates": 7})  # pre-PID checkpoint
    assert legacy.value == 3.0


def test_dual_state_round_trips_for_checkpointing():
    """lambda is part of the solution, so it must survive a checkpoint."""

    dual = DualVariable(CONFIG)
    dual.update(0.5)
    dual.update(0.4)
    state = dual.state_dict()

    restored = DualVariable(CONFIG)
    restored.load_state_dict(state)
    assert restored.value == pytest.approx(dual.value)
    assert restored.updates == dual.updates


def test_restoring_a_negative_lambda_is_rejected():
    dual = DualVariable(CONFIG)
    with pytest.raises(ValueError, match="non-negative"):
        dual.load_state_dict({"value": -1.0})


def test_non_finite_cost_estimate_is_rejected():
    dual = DualVariable(CONFIG)
    with pytest.raises(ValueError, match="finite"):
        dual.update(float("nan"))


# --------------------------------------------------------------------------
# Episodic-sum estimator: the canonical formulation E[sum_t c_t] <= d.
# --------------------------------------------------------------------------


def test_episode_sum_is_cost_per_episode_not_per_step():
    """The distinction that makes d = 0.05 mean 5% of episodes.

    With a collision indicator and termination on first collision, the
    per-episode sum is 0 or 1, so J_C is the collision probability. The
    batch_mean estimator would instead report collisions per STEP, smaller by
    roughly the episode length, and a budget written for one is meaningless
    under the other.
    """

    # 20 steps, 1 env, 4 episodes, 1 collision in total.
    costs = np.zeros((20, 1))
    costs[7, 0] = 1.0
    starts = np.zeros((20, 1))
    starts[[0, 5, 10, 15], 0] = 1.0

    episode_sum = cost_objective_estimate(
        EPISODE_SUM_ESTIMATOR, gamma_c=1.0, normalize=False,
        raw_costs=costs, episode_starts=starts,
    )
    batch_mean = cost_objective_estimate(
        BATCH_MEAN_ESTIMATOR, gamma_c=1.0, normalize=False, raw_costs=costs,
    )
    assert episode_sum == pytest.approx(0.25)   # 1 collision / 4 episodes
    assert batch_mean == pytest.approx(0.05)    # 1 collision / 20 steps
    assert episode_sum != pytest.approx(batch_mean)


def test_episode_sum_reads_as_a_collision_probability():
    """Every episode colliding exactly once must give J_C = 1.0."""

    costs = np.zeros((12, 1))
    starts = np.zeros((12, 1))
    for i in (0, 4, 8):
        starts[i, 0] = 1.0
        costs[i + 3, 0] = 1.0
    assert cost_objective_estimate(
        EPISODE_SUM_ESTIMATOR, gamma_c=1.0, normalize=False,
        raw_costs=costs, episode_starts=starts,
    ) == pytest.approx(1.0)


def test_episode_sum_applies_no_discount_renormalization():
    """The episodic objective is undiscounted; a (1-gamma_c) factor is wrong."""

    costs = np.zeros((10, 1)); costs[2, 0] = 1.0
    starts = np.zeros((10, 1)); starts[[0, 5], 0] = 1.0
    for gamma in (0.9, 0.99, 1.0):
        assert cost_objective_estimate(
            EPISODE_SUM_ESTIMATOR, gamma_c=gamma, normalize=False,
            raw_costs=costs, episode_starts=starts,
        ) == pytest.approx(0.5)


def test_episode_sum_refuses_a_batch_with_no_episode_boundary():
    with pytest.raises(ValueError, match="no episode starts"):
        cost_objective_estimate(
            EPISODE_SUM_ESTIMATOR, gamma_c=1.0, normalize=False,
            raw_costs=np.ones((4, 1)), episode_starts=np.zeros((4, 1)),
        )


def test_episode_sum_requires_both_inputs():
    with pytest.raises(ValueError, match="requires raw_costs and"):
        cost_objective_estimate(EPISODE_SUM_ESTIMATOR, gamma_c=1.0)

