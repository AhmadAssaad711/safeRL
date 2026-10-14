"""PPO-Lagrangian trainer for the CMDP arms.

Solves ``max J_R  s.t.  J_C <= d`` with a separate cost critic and a dual
variable, rather than folding safety into the reward. Two cost sources share
the machinery:

* ``"state"``  -- CMDP-State. ``c_s(s_{t+1})`` from the margin-hinged HOCBF
  barriers, read from the post-step constraint system so the cost credits the
  action that produced the state.
* ``"action"`` -- CMDP-Action. ``c_a(s_t, a_t) = 1/2 ||a - Pi_Usafe(a)||^2``
  from the pre-step constraint system, i.e. the context the action was chosen
  in.

**The filter is OFF.** ``execution_mode`` is ``"box"``: the environment
executes the raw box-clipped policy action. For CMDP-Action the projection is
evaluated purely as a measurement and never alters the transition, which is
what distinguishes this arm from the detached actor loss.

**No reward shaping.** The environment's own ``state_cost`` must stay disabled
(``lambda_s = 0``) so nothing is subtracted from the reward; this trainer
evaluates the barrier cost itself from the public constraint system. Enabling
the environment's state cost *and* this trainer would penalize the same
quantity twice, once in the reward and once in the constraint.

Design note on layering. This module imports ``LatentActionPPO`` from
``scripts.common.projected_ppo_cbf``, which inverts the usual
``common -> training`` direction. That is deliberate: the CMDP arms have to
share the execution path, gradient-clip handling and rollout bookkeeping of
the arms they are compared against, and reimplementing those here would risk
silent divergence in exactly the places that decide whether a comparison is
valid. Nothing in ``scripts/common`` is modified.
"""

from __future__ import annotations

from typing import Any, Generator, NamedTuple, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from scripts.common.cbf_projection import project_polytope_2d_numpy
from scripts.common.projected_ppo_cbf import LatentActionPPO, split_cbf_context_numpy

from saferl.algorithms.cmdp import (
    CMDPConfig,
    DualVariable,
    cost_gae,
    cost_objective_estimate,
)
from saferl.safety.action_cost import (
    ACTION_COST_FORMS,
    BOX_BOUND_RULE,
    PROJECTION_COST_FORM,
    VIOLATION_COST_FORM,
    action_distance_cost,
    constraint_violation_cost,
)
from saferl.safety.state_cost import (
    StateCostParameters,
    barrier_state_cost,
    barrier_violation_cost,
)

# How the state cost is defined.
#
# "margin_hinge"    max_i of the margin-hinged h / psi_1 squares. Fires on
#                   technically safe states near the margin, so it is largely a
#                   traffic-proximity measure: measured floor ~0.49 and
#                   corr(d, collisions) = -0.968, i.e. tightening the budget
#                   made collisions worse.
# "psi1_violation"  1{min_i psi_1 < 0}. Exactly the forward-invariance
#                   condition, zero free parameters, and d is a fraction of
#                   steps. The violation rate also separated collision from
#                   collision-free episodes at AUC 0.875-0.999.
MARGIN_HINGE_STATE_COST = "margin_hinge"
PSI1_VIOLATION_STATE_COST = "psi1_violation"
STATE_COST_FORMS = (MARGIN_HINGE_STATE_COST, PSI1_VIOLATION_STATE_COST)

COST_SOURCE_STATE = "state"
COST_SOURCE_ACTION = "action"
SUPPORTED_COST_SOURCES = (COST_SOURCE_STATE, COST_SOURCE_ACTION)


class CMDPRolloutBufferSamples(NamedTuple):
    """A PPO minibatch plus the cost channel."""

    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    costs: th.Tensor
    cost_values: th.Tensor
    cost_advantages: th.Tensor
    cost_returns: th.Tensor


class CostRolloutBuffer(RolloutBuffer):
    """Rollout storage with a second, independently estimated cost channel.

    The reward targets are untouched. Costs are stored raw: the dual update
    needs raw per-step costs to form the average-cost constraint estimate, and
    any scaling is applied explicitly by the caller.
    """

    def __init__(
        self,
        *args: Any,
        gamma_c: float = 0.98,
        gae_lambda_c: float = 0.95,
        **kwargs: Any,
    ) -> None:
        self.gamma_c = float(gamma_c)
        self.gae_lambda_c = float(gae_lambda_c)
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        shape = (self.buffer_size, self.n_envs)
        self.costs = np.zeros(shape, dtype=np.float32)
        self.cost_values = np.zeros(shape, dtype=np.float32)
        self.cost_advantages = np.zeros(shape, dtype=np.float32)
        self.cost_returns = np.zeros(shape, dtype=np.float32)

    def add(  # type: ignore[override]
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        *,
        costs: Any = 0.0,
        cost_values: Any = 0.0,
    ) -> None:
        position = self.pos
        super().add(obs, action, reward, episode_start, value, log_prob)
        self.costs[position] = self._env_vector(costs, name="costs")
        self.cost_values[position] = self._env_vector(
            cost_values, name="cost_values"
        )

    def _env_vector(self, value: Any, *, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 0:
            array = np.full(self.n_envs, float(array), dtype=np.float32)
        array = array.reshape(-1)
        if int(array.size) != int(self.n_envs):
            raise ValueError(
                f"{name} must supply one value per environment "
                f"({array.size} != {self.n_envs})"
            )
        return array.astype(np.float32, copy=False)

    def compute_cost_returns(
        self, *, last_cost_values: th.Tensor, dones: np.ndarray
    ) -> None:
        """Fill the cost advantages and targets by GAE."""

        last_values = (
            last_cost_values.detach().clone().cpu().numpy().reshape(-1)
        )
        advantages, targets = cost_gae(
            self.costs,
            self.cost_values,
            self.episode_starts,
            last_cost_values=last_values,
            dones=np.asarray(dones, dtype=np.float32),
            gamma_c=self.gamma_c,
            gae_lambda_c=self.gae_lambda_c,
        )
        self.cost_advantages = advantages.astype(np.float32, copy=False)
        self.cost_returns = targets.astype(np.float32, copy=False)

    def mean_raw_cost(self) -> float:
        """Average per-step cost over the whole rollout."""

        return float(np.asarray(self.costs, dtype=np.float64).mean())

    def get(  # type: ignore[override]
        self, batch_size: Optional[int] = None
    ) -> Generator[CMDPRolloutBufferSamples, None, None]:
        assert self.full, "the rollout buffer must be full before sampling"
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for tensor in (
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "costs",
                "cost_values",
                "cost_advantages",
                "cost_returns",
            ):
                self.__dict__[tensor] = self.swap_and_flatten(
                    self.__dict__[tensor]
                )
            self.generator_ready = True
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        start = 0
        while start < self.buffer_size * self.n_envs:
            batch = indices[start : start + batch_size]
            yield CMDPRolloutBufferSamples(
                observations=self.to_torch(self.observations[batch]),
                actions=self.to_torch(self.actions[batch]),
                old_values=self.to_torch(self.values[batch].flatten()),
                old_log_prob=self.to_torch(self.log_probs[batch].flatten()),
                advantages=self.to_torch(self.advantages[batch].flatten()),
                returns=self.to_torch(self.returns[batch].flatten()),
                costs=self.to_torch(self.costs[batch].flatten()),
                cost_values=self.to_torch(self.cost_values[batch].flatten()),
                cost_advantages=self.to_torch(
                    self.cost_advantages[batch].flatten()
                ),
                cost_returns=self.to_torch(self.cost_returns[batch].flatten()),
            )
            start += batch_size


class CMDPActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy with an *independent* cost-value network.

    The cost critic does not share the reward critic's trunk. Sharing would let
    the cost regression reshape the representation the reward critic uses, so a
    measured CMDP effect could come partly from representation interference
    rather than from constrained optimization.

    The cost critic reads the base observation by default, i.e. the same view
    the actor gets, with the CBF context stripped. ``cost_critic_sees_context``
    grants it the full augmented observation instead.
    """

    def __init__(
        self,
        *args: Any,
        cbf_base_observation_dim: int = 42,
        cost_critic_arch: Optional[list[int]] = None,
        cost_critic_sees_context: bool = False,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        self.cbf_base_observation_dim = int(cbf_base_observation_dim)
        self.cost_critic_sees_context = bool(cost_critic_sees_context)
        self._cost_critic_arch = (
            [256, 128] if cost_critic_arch is None else list(cost_critic_arch)
        )
        # Bounds on the exploration scale. SB3 leaves log_std unbounded, and
        # measured on three independent cost definitions that is enough to end
        # the run: a Lagrangian penalty on a STATE cost rewards shrinking sigma
        # (less exploration means fewer excursions into costly states), so
        # lambda growth drove sigma to 0.087 and the policy locked into a poor
        # deterministic behaviour before it could satisfy the budget legitimately.
        # The action cost failed the other way -- its advantage was noise, so the
        # entropy bonus won and sigma diverged to 94 on a +/-3 action box.
        self.sigma_min = None if sigma_min is None else float(sigma_min)
        self.sigma_max = None if sigma_max is None else float(sigma_max)
        if (
            self.sigma_min is not None
            and self.sigma_max is not None
            and self.sigma_min >= self.sigma_max
        ):
            raise ValueError("sigma_min must be below sigma_max")
        for name in ("sigma_min", "sigma_max"):
            v = getattr(self, name)
            if v is not None and (not np.isfinite(v) or v <= 0.0):
                raise ValueError(f"{name} must be finite and positive")
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Any) -> None:
        super()._build(lr_schedule)
        observation_width = int(np.prod(self.observation_space.shape))
        width = (
            observation_width
            if self.cost_critic_sees_context
            else self.cbf_base_observation_dim
        )
        layers: list[th.nn.Module] = []
        last = int(width)
        for units in self._cost_critic_arch:
            layers.append(th.nn.Linear(last, int(units)))
            layers.append(th.nn.Tanh())
            last = int(units)
        # Unconstrained head on purpose: the targets are non-negative but a
        # softplus would bias the critic and interact badly with GAE.
        layers.append(th.nn.Linear(last, 1))
        self.cost_critic = th.nn.Sequential(*layers)
        if self.ortho_init:
            for module in self.cost_critic:
                if isinstance(module, th.nn.Linear):
                    th.nn.init.orthogonal_(module.weight, gain=1.0)
                    th.nn.init.constant_(module.bias, 0.0)
        # The base class builds its optimizer before this network exists.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def bounded_log_std(self) -> th.Tensor:
        """``log_std`` clamped to the configured sigma band.

        Clamping at use rather than reparameterizing keeps the parameter's
        meaning and checkpoint layout unchanged. The clamp also zeroes the
        gradient outside the band, so the optimizer stops pushing against it.
        """

        log_std = self.log_std
        if self.sigma_min is not None:
            log_std = th.clamp(log_std, min=float(np.log(self.sigma_min)))
        if self.sigma_max is not None:
            log_std = th.clamp(log_std, max=float(np.log(self.sigma_max)))
        return log_std

    def clamp_log_std_(self) -> None:
        """Clamp the stored parameter, called after each optimizer step.

        The use-time clamp alone leaves the raw parameter free to drift far
        outside the band on Adam momentum, which would make the bound invisible
        in the checkpoint and slow to release if it were ever widened.
        """

        if self.sigma_min is None and self.sigma_max is None:
            return
        with th.no_grad():
            lo = None if self.sigma_min is None else float(np.log(self.sigma_min))
            hi = None if self.sigma_max is None else float(np.log(self.sigma_max))
            self.log_std.clamp_(min=lo, max=hi)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):  # type: ignore[override]
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(
            mean_actions, self.bounded_log_std()
        )

    def _cost_critic_input(self, obs: th.Tensor) -> th.Tensor:
        if self.cost_critic_sees_context:
            return obs
        return obs[..., : self.cbf_base_observation_dim]

    def predict_cost_values(self, obs: th.Tensor) -> th.Tensor:
        return self.cost_critic(self._cost_critic_input(obs))

    def evaluate_actions_with_cost(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor], th.Tensor]:
        values, log_prob, entropy = self.evaluate_actions(obs, actions)
        return values, log_prob, entropy, self.predict_cost_values(obs)


class PPOLagrangian(LatentActionPPO):
    """PPO with a cost critic, a cost GAE and a projected dual variable."""

    def __init__(
        self,
        *args: Any,
        cmdp_config: Optional[CMDPConfig] = None,
        cost_source: str = COST_SOURCE_ACTION,
        state_cost_params: Optional[StateCostParameters] = None,
        infeasible_rule: str = BOX_BOUND_RULE,
        action_cost_form: str = PROJECTION_COST_FORM,
        state_cost_form: str = MARGIN_HINGE_STATE_COST,
        cost_vf_coef: float = 0.5,
        **kwargs: Any,
    ) -> None:
        self.cost_source = str(cost_source).strip().lower()
        if self.cost_source not in SUPPORTED_COST_SOURCES:
            raise ValueError(
                f"cost_source must be one of {SUPPORTED_COST_SOURCES}"
            )
        # SB3's load() reconstructs with only policy/env/device and
        # _init_setup_model=False, then overwrites __dict__ from the pickle, so
        # the config legitimately arrives later on that one path. Everywhere
        # else a missing config is a mistake: cost_limit has no default by
        # design and must not be silently invented.
        self._config_is_placeholder = cmdp_config is None
        if cmdp_config is None:
            if kwargs.get("_init_setup_model", True):
                raise ValueError(
                    "cmdp_config is required; cost_limit must be chosen from "
                    "measured cost distributions"
                )
            cmdp_config = CMDPConfig(cost_limit=0.0)
        if not isinstance(cmdp_config, CMDPConfig):
            raise TypeError("cmdp_config must be a CMDPConfig")
        self.cmdp_config = cmdp_config
        self.infeasible_rule = str(infeasible_rule)
        self.action_cost_form = str(action_cost_form).strip().lower()
        if self.action_cost_form not in ACTION_COST_FORMS:
            raise ValueError(
                f"action_cost_form must be one of {ACTION_COST_FORMS}"
            )
        self.state_cost_form = str(state_cost_form).strip().lower()
        if self.state_cost_form not in STATE_COST_FORMS:
            raise ValueError(
                f"state_cost_form must be one of {STATE_COST_FORMS}"
            )
        self.cost_vf_coef = float(cost_vf_coef)
        # lambda_s plays no part here: the trainer uses the raw cost and the
        # environment's own reward penalty must stay off.
        self.state_cost_params = (
            StateCostParameters(lambda_s=1.0)
            if state_cost_params is None
            else state_cost_params
        )
        if self.cost_source == COST_SOURCE_STATE and not self.state_cost_params.enabled:
            raise ValueError(
                "CMDP-State needs state_cost_params with lambda_s > 0 to enable "
                "the cost definition; it never reaches the reward"
            )
        self.dual = DualVariable(cmdp_config)
        self.cmdp_training_diagnostics: list[dict[str, float]] = []

        requested = str(kwargs.get("execution_mode", "box")).strip().lower()
        if requested != "box":
            raise ValueError(
                "the CMDP arms train with the HOCBF filter OFF, so "
                "execution_mode must be 'box'"
            )
        kwargs["execution_mode"] = "box"
        buffer_class = kwargs.pop("rollout_buffer_class", None)
        if buffer_class not in (None, CostRolloutBuffer):
            raise ValueError("PPOLagrangian requires CostRolloutBuffer")
        kwargs["rollout_buffer_class"] = CostRolloutBuffer
        buffer_kwargs = dict(kwargs.pop("rollout_buffer_kwargs", {}) or {})
        buffer_kwargs.update(
            {
                "gamma_c": cmdp_config.gamma_c,
                "gae_lambda_c": cmdp_config.gae_lambda_c,
            }
        )
        kwargs["rollout_buffer_kwargs"] = buffer_kwargs
        super().__init__(*args, **kwargs)
        # SB3's load() path constructs before restoring the spaces and the
        # policy from the pickle, and load(env=None) never sets an action
        # space at all, so both checks are conditional on the attribute
        # actually existing yet.
        if hasattr(self, "policy") and not isinstance(
            self.policy, CMDPActorCriticPolicy
        ):
            raise TypeError("PPOLagrangian requires CMDPActorCriticPolicy")
        if getattr(self, "action_space", None) is not None and not isinstance(
            self.action_space, spaces.Box
        ):
            raise TypeError("PPOLagrangian requires a Box action space")

    # ------------------------------------------------------------------
    # Gradient grouping
    # ------------------------------------------------------------------

    def _gradient_parameter_groups(self) -> tuple[list, list]:
        """Split parameters into policy and value groups, cost critic included.

        The base implementation groups by ``"value_net" in name``. The
        projected policy's extra head is called ``safety_value_net`` and so
        lands in the value group by substring coincidence; this one is called
        ``cost_critic`` and would land in the **policy** group.

        That matters specifically when ``separate_grad_clip`` is on: the cost
        critic's gradient would then be clipped together with the actor, so a
        large cost-regression gradient would scale the policy gradient down.
        It would look like the shared-budget fix had been applied while making
        the coupling worse, and ``train/grad_norm_pi`` would be mislabelled.
        """

        policy_params, value_params = [], []
        for name, param in self.policy.named_parameters():
            if param.grad is None:
                continue
            if "value_net" in name or name.startswith("cost_critic"):
                value_params.append(param)
            else:
                policy_params.append(param)
        return policy_params, value_params

    # ------------------------------------------------------------------
    # Cost evaluation
    # ------------------------------------------------------------------

    def _action_costs(
        self, box_clipped: np.ndarray, observations: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """``c_a`` against the pre-step constraint system.

        The projection here is a measurement: the executed action is the raw
        box-clipped one, and nothing about the transition depends on this call.
        """

        _, rows, bounds, mask = split_cbf_context_numpy(
            np.asarray(observations, dtype=np.float32), layout=self.cbf_layout
        )
        if self.action_cost_form == VIOLATION_COST_FORM:
            # No projection at all: the squared constraint violation is defined
            # by the same formula on feasible and infeasible states, so there
            # is no actionless branch and no solver call.
            return constraint_violation_cost(box_clipped, rows, bounds, mask)
        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        safe = np.empty_like(box_clipped)
        feasible = np.ones(box_clipped.shape[0], dtype=bool)
        for index in range(box_clipped.shape[0]):
            projection = project_polytope_2d_numpy(
                box_clipped[index],
                rows[index],
                bounds[index],
                mask[index],
                feasibility_tol=self.cbf_feasibility_tol,
                action_low=low,
                action_high=high,
            )
            safe[index] = projection.action
            feasible[index] = bool(projection.feasible)
        return action_distance_cost(
            box_clipped,
            safe,
            feasible=feasible,
            infeasible_rule=self.infeasible_rule,
            action_low=low,
            action_high=high,
        )

    def _state_costs(self, env: VecEnv) -> tuple[np.ndarray, dict[str, Any]]:
        """``c_s`` from the post-step constraint systems.

        One broadcast ``env_method`` per step rather than one call per
        environment, so the cost adds a single IPC round trip.
        """

        systems = env.env_method("current_constraint_system")
        costs = np.zeros(len(systems), dtype=np.float64)
        argmax_boundary = 0
        min_psi1: list[float] = []
        scorer = (
            barrier_violation_cost
            if self.state_cost_form == PSI1_VIOLATION_STATE_COST
            else barrier_state_cost
        )
        for index, system in enumerate(systems):
            h = np.asarray(system.get("barrier_h", ()), dtype=float)
            h_dot = np.asarray(system.get("barrier_h_dot", ()), dtype=float)
            if h.size == 0 or h_dot.size != h.size:
                # A silently zeroed safety cost is worse than a hard failure.
                raise RuntimeError(
                    "CMDP-State requires per-barrier h_dot from "
                    "batch_pairwise_hocbf_constraints"
                )
            cost, diagnostics = scorer(h, h_dot, self.state_cost_params)
            costs[index] = cost
            argmax_boundary += int(diagnostics["argmax_class"] == "boundary")
            if "min_psi1" in diagnostics:
                min_psi1.append(float(diagnostics["min_psi1"]))
        out = {
            "cost_mean": float(costs.mean()) if costs.size else 0.0,
            "cost_max": float(costs.max()) if costs.size else 0.0,
            # Under the psi1-violation form this IS the violation rate, i.e.
            # the quantity the constraint budgets.
            "active_fraction": float((costs > 0.0).mean()) if costs.size else 0.0,
            "boundary_argmax_fraction": (
                float(argmax_boundary) / float(len(systems)) if systems else 0.0
            ),
            "sample_count": int(len(systems)),
        }
        if min_psi1:
            out["min_psi1_mean"] = float(np.mean(min_psi1))
        return costs, out

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect a rollout, scoring the safety cost alongside the reward."""

        assert self._last_obs is not None, "No previous observation was provided"
        if not isinstance(rollout_buffer, CostRolloutBuffer):
            raise TypeError("PPOLagrangian requires CostRolloutBuffer")
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()
        cost_diagnostics: list[dict[str, Any]] = []

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions_tensor, values, log_probs = self.policy(obs_tensor)
                cost_values = self.policy.predict_cost_values(obs_tensor)
            actions = actions_tensor.cpu().numpy()
            executed_actions, records = self._execution_actions(
                actions, np.asarray(self._last_obs)
            )
            box_clipped = np.asarray(
                [record["box_clipped_phys"] for record in records],
                dtype=np.float32,
            )

            # c_a is a property of (s_t, a_t) and must be scored before the
            # step; c_s is a property of s_{t+1} and is scored after it.
            if self.cost_source == COST_SOURCE_ACTION:
                costs, diagnostics = self._action_costs(
                    box_clipped, np.asarray(self._last_obs)
                )
            else:
                costs, diagnostics = None, None

            for env_index, record in enumerate(records):
                env.env_method(
                    "set_projection_record",
                    actions[env_index],
                    executed_actions[env_index],
                    box_clipped_phys=box_clipped[env_index],
                    feasible=bool(record["feasible"]),
                    fallback_used=bool(record["fallback_used"]),
                    projection_source=str(record["projection_source"]),
                    max_constraint_violation_safe=float(
                        record["max_constraint_violation_safe"]
                    ),
                    active_indices=record["active_indices"],
                    constraint_hash=str(record["constraint_hash"]),
                    cbf_applied=bool(record["cbf_applied"]),
                    indices=env_index,
                )

            clipped_actions = box_clipped
            new_obs, rewards, dones, infos = env.step(executed_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1

            if self.cost_source == COST_SOURCE_STATE:
                costs, diagnostics = self._state_costs(env)
            cost_diagnostics.append(dict(diagnostics or {}))

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                costs=np.asarray(costs, dtype=np.float32),
                cost_values=cost_values.detach().cpu().numpy().reshape(-1),
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            final_obs = obs_as_tensor(new_obs, self.device)
            values = self.policy.predict_values(final_obs)
            final_cost_values = self.policy.predict_cost_values(final_obs)
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        rollout_buffer.compute_cost_returns(
            last_cost_values=final_cost_values, dones=dones
        )
        self._last_cost_diagnostics = _mean_of_dicts(cost_diagnostics)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def train(self) -> None:
        """PPO with a dual-weighted cost surrogate and two critics."""

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = (
            self.clip_range_vf(self._current_progress_remaining)
            if self.clip_range_vf is not None
            else None
        )

        buffer = self.rollout_buffer
        assert isinstance(buffer, CostRolloutBuffer)
        if getattr(self, "_config_is_placeholder", False):
            # Only reachable if someone trains a model built through the
            # load() path without a restored config.
            raise RuntimeError(
                "this model carries a placeholder CMDP config; set cmdp_config "
                "before training"
            )

        # One dual step per rollout, before the epochs, held fixed across them:
        # updating per minibatch makes lambda chase minibatch noise. The
        # estimate is the average raw per-step cost, matching the average-cost
        # definition of J_C.
        cost_objective = cost_objective_estimate(
            self.cmdp_config.cost_estimator,
            gamma_c=self.cmdp_config.gamma_c,
            normalize=self.cmdp_config.normalize_cost_objective,
            raw_costs=buffer.costs,
            cost_values=buffer.cost_values,
            episode_starts=np.asarray(buffer.episode_starts, dtype=bool),
        )
        dual_info = self.dual.update(cost_objective)
        lambda_value = self.dual.value
        penalty_scale = self.dual.penalty_scale

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        cost_pg_losses: list[float] = []
        value_losses: list[float] = []
        cost_value_losses: list[float] = []
        clip_fractions: list[float] = []
        approx_kls: list[float] = []
        continue_training = True

        for epoch in range(self.n_epochs):
            for rollout_data in buffer.get(self.batch_size):
                actions = rollout_data.actions
                values, log_prob, entropy, cost_values = (
                    self.policy.evaluate_actions_with_cost(
                        rollout_data.observations, actions
                    )
                )
                values = values.flatten()
                cost_values = cost_values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )
                # Cost advantages are deliberately NOT normalized: it would
                # destroy the relationship between lambda, the physical cost
                # scale and the cost limit.
                cost_advantages = rollout_data.cost_advantages

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                clipped_ratio = th.clamp(ratio, 1 - clip_range, 1 + clip_range)

                # Reward surrogate: pessimistic lower bound, so min.
                reward_surrogate = th.min(
                    ratio * advantages, clipped_ratio * advantages
                ).mean()
                # Cost surrogate: cost is driven down, so the pessimistic bound
                # is the upper one. Using min here would silently invert the
                # constraint.
                cost_surrogate = th.max(
                    ratio * cost_advantages, clipped_ratio * cost_advantages
                ).mean()

                policy_loss = -penalty_scale * (
                    reward_surrogate - lambda_value * cost_surrogate
                )
                pg_losses.append(float(-reward_surrogate.detach().cpu().item()))
                cost_pg_losses.append(float(cost_surrogate.detach().cpu().item()))
                clip_fractions.append(
                    float(
                        th.mean(
                            (th.abs(ratio - 1) > clip_range).float()
                        ).detach().cpu().item()
                    )
                )

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = self._value_loss(rollout_data.returns, values_pred)
                cost_value_loss = self._value_loss(
                    rollout_data.cost_returns, cost_values
                )
                value_losses.append(float(value_loss.detach().cpu().item()))
                cost_value_losses.append(
                    float(cost_value_loss.detach().cpu().item())
                )

                entropy_loss = (
                    -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                )
                entropy_losses.append(float(entropy_loss.detach().cpu().item()))

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.cost_vf_coef * cost_value_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio
                    ).cpu().item()
                    approx_kls.append(float(approx_kl))
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                self._clip_and_log_gradients()
                self.policy.optimizer.step()
                self.policy.clamp_log_std_()

            self._n_updates += 1
            if not continue_training:
                break

        explained = explained_variance(
            buffer.values.flatten(), buffer.returns.flatten()
        )
        cost_explained = explained_variance(
            buffer.cost_values.flatten(), buffer.cost_returns.flatten()
        )
        record = {
            "num_timesteps": float(self.num_timesteps),
            "n_updates": float(self._n_updates),
            "cost_source": self.cost_source,
            "entropy_loss": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
            "policy_gradient_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
            "cost_surrogate": (
                float(np.mean(cost_pg_losses)) if cost_pg_losses else 0.0
            ),
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "cost_value_loss": (
                float(np.mean(cost_value_losses)) if cost_value_losses else 0.0
            ),
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction": (
                float(np.mean(clip_fractions)) if clip_fractions else 0.0
            ),
            "explained_variance": float(explained),
            "cost_explained_variance": float(cost_explained),
            "cost_mean_raw": buffer.mean_raw_cost(),
            # Logged every run: sigma collapse and sigma divergence were both
            # invisible until read out of a checkpoint by hand.
            "sigma_mean": float(
                th.exp(self.policy.bounded_log_std().detach()).mean().cpu().item()
            ),
            "sigma_min_component": float(
                th.exp(self.policy.bounded_log_std().detach()).min().cpu().item()
            ),
            **dual_info,
            **{
                f"cmdp/rollout_{key}": value
                for key, value in (getattr(self, "_last_cost_diagnostics", {}) or {}).items()
                if isinstance(value, (int, float))
            },
        }
        self.cmdp_training_diagnostics.append(record)
        for key, value in record.items():
            if isinstance(value, (int, float)):
                self.logger.record(
                    key if key.startswith("cmdp/") else f"train/{key}", value
                )


def _mean_of_dicts(records: list[dict[str, Any]]) -> dict[str, float]:
    """Average the numeric entries of a list of diagnostic dicts."""

    if not records:
        return {}
    keys = [
        key
        for key, value in records[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {
        key: float(np.mean([float(record[key]) for record in records if key in record]))
        for key in keys
    }
