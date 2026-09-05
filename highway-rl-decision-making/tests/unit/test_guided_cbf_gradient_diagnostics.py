from __future__ import annotations

import math

import pytest
import torch as th


from scripts.common.guided_cbf_minimal import (
    _actor_gradient_diagnostics,
    _local_projection_target,
    _positive_kkt_support,
)


def test_actor_gradient_diagnostics_reports_weighted_components_without_touching_grad_buffers():
    actor = th.nn.Linear(2, 1, bias=False)
    with th.no_grad():
        actor.weight.copy_(th.tensor([[0.25, -0.50]], dtype=th.float32))

    existing_gradient = th.tensor([[7.0, -4.0]], dtype=th.float32)
    actor.weight.grad = existing_gradient.clone()
    q_loss = actor(th.tensor([[1.0, 0.0]], dtype=th.float32)).sum()
    weighted_cbf_loss = 2.0 * actor(th.tensor([[0.0, 1.0]], dtype=th.float32)).sum()

    diagnostics = _actor_gradient_diagnostics(actor, q_loss, weighted_cbf_loss)

    assert diagnostics["g_q_norm"] == pytest.approx(1.0)
    assert diagnostics["g_cbf_norm"] == pytest.approx(2.0)
    assert diagnostics["g_cbf_to_g_q_ratio"] == pytest.approx(2.0)
    assert diagnostics["g_q_g_cbf_cosine"] == pytest.approx(0.0)
    assert diagnostics["g_q_g_cbf_cosine_valid"] == pytest.approx(1.0)
    assert th.equal(actor.weight.grad, existing_gradient)


def test_actor_gradient_diagnostics_reports_opposed_and_disabled_cbf_terms():
    actor = th.nn.Linear(2, 1, bias=False)
    action = actor(th.tensor([[1.0, 0.0]], dtype=th.float32)).sum()

    opposed = _actor_gradient_diagnostics(actor, action, -3.0 * action)
    assert opposed["g_q_norm"] == pytest.approx(1.0)
    assert opposed["g_cbf_norm"] == pytest.approx(3.0)
    assert opposed["g_cbf_to_g_q_ratio"] == pytest.approx(3.0)
    assert opposed["g_q_g_cbf_cosine"] == pytest.approx(-1.0)
    assert opposed["g_q_g_cbf_cosine_valid"] == pytest.approx(1.0)

    disabled = _actor_gradient_diagnostics(actor, action, 0.0 * action)
    assert disabled["g_cbf_norm"] == pytest.approx(0.0)
    assert disabled["g_cbf_to_g_q_ratio"] == pytest.approx(0.0)
    assert math.isnan(disabled["g_q_g_cbf_cosine"])
    assert disabled["g_q_g_cbf_cosine_valid"] == pytest.approx(0.0)

    zero_q = _actor_gradient_diagnostics(actor, 0.0 * action, action)
    assert zero_q["g_q_norm"] == pytest.approx(0.0)
    assert zero_q["g_cbf_norm"] == pytest.approx(1.0)
    assert math.isnan(zero_q["g_q_g_cbf_cosine"])
    assert zero_q["g_q_g_cbf_cosine_valid"] == pytest.approx(0.0)


def test_local_projection_target_cancels_tangential_behavior_noise_and_tracks_current_actor():
    current_action = th.tensor([[0.8, -0.4]], dtype=th.float32, requires_grad=True)
    projection_jacobian = th.tensor(
        [[[0.0, 0.0], [0.0, 1.0]]],
        dtype=th.float32,
    )
    behavior_action_a = th.tensor([[1.0, 0.7]], dtype=th.float32)
    safe_behavior_a = th.tensor([[0.2, 0.7]], dtype=th.float32)
    behavior_action_b = th.tensor([[1.0, -0.9]], dtype=th.float32)
    safe_behavior_b = th.tensor([[0.2, -0.9]], dtype=th.float32)

    target_a = _local_projection_target(
        current_action,
        behavior_action_a,
        safe_behavior_a,
        projection_jacobian,
    )
    target_b = _local_projection_target(
        current_action,
        behavior_action_b,
        safe_behavior_b,
        projection_jacobian,
    )

    th.testing.assert_close(target_a, th.tensor([[0.2, -0.4]], dtype=th.float32))
    th.testing.assert_close(target_b, target_a)
    assert not target_a.requires_grad

    changed_current_action = th.tensor([[0.8, 0.3]], dtype=th.float32)
    changed_target = _local_projection_target(
        changed_current_action,
        behavior_action_a,
        safe_behavior_a,
        projection_jacobian,
    )
    th.testing.assert_close(changed_target, th.tensor([[0.2, 0.3]], dtype=th.float32))

    feasible_current_action = th.tensor([[0.0, 0.3]], dtype=th.float32)
    feasible_target = _local_projection_target(
        feasible_current_action,
        behavior_action_a,
        safe_behavior_a,
        projection_jacobian,
    )
    th.testing.assert_close(feasible_target, feasible_current_action)

    local_projection_loss = (current_action - target_a).square().sum()
    local_projection_loss.backward()
    th.testing.assert_close(current_action.grad, th.tensor([[1.2, 0.0]], dtype=th.float32))


def test_positive_kkt_support_excludes_tight_zero_multiplier_rows():
    rows = [[1.0, 0.0], [0.0, 1.0]]
    support, multipliers, residual = _positive_kkt_support(rows, [1.0, 0.0])
    assert support.tolist() == [0]
    assert multipliers[0] == pytest.approx(1.0)
    assert multipliers[1] == pytest.approx(0.0)
    assert residual == pytest.approx(0.0)

    corner_support, _, corner_residual = _positive_kkt_support(rows, [1.0, 2.0])
    assert corner_support.tolist() == [0, 1]
    assert corner_residual == pytest.approx(0.0)


def test_actor_gradient_diagnostics_do_not_change_the_combined_optimizer_step():
    diagnosed_actor = th.nn.Sequential(
        th.nn.Linear(2, 4),
        th.nn.Tanh(),
        th.nn.Linear(4, 2),
    )
    control_actor = th.nn.Sequential(
        th.nn.Linear(2, 4),
        th.nn.Tanh(),
        th.nn.Linear(4, 2),
    )
    control_actor.load_state_dict(diagnosed_actor.state_dict())
    observations = th.tensor(
        [[0.2, -0.4], [1.0, 0.5], [-0.3, 0.7]],
        dtype=th.float32,
    )
    safe_targets = th.tensor(
        [[-0.1, 0.2], [0.3, -0.5], [0.0, 0.4]],
        dtype=th.float32,
    )

    diagnosed_actions = diagnosed_actor(observations)
    diagnosed_q_loss = -diagnosed_actions.square().sum(dim=1).mean()
    diagnosed_cbf_loss = 0.3 * (diagnosed_actions - safe_targets).square().sum(dim=1).mean()
    diagnostics = _actor_gradient_diagnostics(
        diagnosed_actor,
        diagnosed_q_loss,
        diagnosed_cbf_loss,
    )
    diagnosed_optimizer = th.optim.SGD(diagnosed_actor.parameters(), lr=0.05)
    diagnosed_optimizer.zero_grad(set_to_none=True)
    (diagnosed_q_loss + diagnosed_cbf_loss).backward()
    diagnosed_optimizer.step()

    control_actions = control_actor(observations)
    control_q_loss = -control_actions.square().sum(dim=1).mean()
    control_cbf_loss = 0.3 * (control_actions - safe_targets).square().sum(dim=1).mean()
    control_optimizer = th.optim.SGD(control_actor.parameters(), lr=0.05)
    control_optimizer.zero_grad(set_to_none=True)
    (control_q_loss + control_cbf_loss).backward()
    control_optimizer.step()

    assert all(th.isfinite(th.tensor(value)) for value in diagnostics.values())
    for diagnosed_parameter, control_parameter in zip(
        diagnosed_actor.parameters(),
        control_actor.parameters(),
    ):
        assert th.equal(diagnosed_parameter, control_parameter)
