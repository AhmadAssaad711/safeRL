"""PPO variants with hard CBF execution and optional actor integration.

Data semantics are intentionally explicit:

* a detached-feedback policy uses ``Normal(mu_raw, sigma)``;
* a differentiable projected policy uses ``Normal(mu_safe, sigma)``;
* the rollout buffer action is the latent Gaussian sample ``z``;
* its stored log probability is ``log pi(z | s)``;
* the simulator receives the separate hard projection ``P_s(z)``.

The constraint context is appended to observations by :mod:`ppo_cbf_env`.
Only the original state features enter the learned actor/value networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, NamedTuple, Optional

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import DiagGaussianDistribution, Distribution
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from scripts.common.cbf_projection import (
    CBFContextLayout,
    TorchProjection2D,
    project_polytope_2d_numpy,
    project_polytope_2d_torch,
    split_cbf_context_numpy,
    split_cbf_context_torch,
)
from scripts.common.ppo_cbf_env import constraint_system_hash


class CBFBaseFeaturesExtractor(BaseFeaturesExtractor):
    """Expose only the original state to actor and value networks."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        *,
        base_observation_dim: int = 42,
    ) -> None:
        self.base_observation_dim = int(base_observation_dim)
        if int(np.prod(observation_space.shape)) < self.base_observation_dim:
            raise ValueError("Observation is narrower than the requested base state")
        super().__init__(observation_space, features_dim=self.base_observation_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations[..., : self.base_observation_dim]


class CBFSafetyRolloutBufferSamples(NamedTuple):
    """A PPO minibatch plus targets for the auxiliary CBF safety critic."""

    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    safety_costs: th.Tensor
    safety_returns: th.Tensor
    safety_fallbacks: th.Tensor


class CBFSafetyRolloutBuffer(RolloutBuffer):
    """Rollout storage for a discounted, executed-CBF correction cost.

    The ordinary PPO reward/value targets remain untouched.  This buffer stores
    the observed hard-filter correction cost separately so the Level-3 safety
    critic can learn its own discounted value without redefining the task
    critic's return semantics.
    """

    def __init__(
        self,
        *args,
        safety_gamma: float = 0.99,
        safety_cost_clip: float = 1.0,
        **kwargs,
    ) -> None:
        self.safety_gamma = float(safety_gamma)
        self.safety_cost_clip = float(safety_cost_clip)
        if not np.isfinite(self.safety_gamma) or not 0.0 <= self.safety_gamma <= 1.0:
            raise ValueError("safety_gamma must be finite and lie in [0, 1]")
        if not np.isfinite(self.safety_cost_clip) or self.safety_cost_clip <= 0.0:
            raise ValueError("safety_cost_clip must be finite and positive")
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self.safety_costs = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.safety_returns = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.safety_fallbacks = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )

    def _as_env_vector(self, value: Any, *, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 0:
            array = np.full(self.n_envs, float(array), dtype=np.float32)
        else:
            array = array.reshape(-1)
        if int(array.size) != int(self.n_envs):
            raise ValueError(
                f"{name} must supply one value per environment "
                f"({array.size} != {self.n_envs})"
            )
        return array.astype(np.float32, copy=False)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        *,
        safety_costs: Any = 0.0,
        safety_fallbacks: Any = 0.0,
    ) -> None:
        position = int(self.pos)
        super().add(obs, action, reward, episode_start, value, log_prob)
        costs = self._as_env_vector(safety_costs, name="safety_costs")
        fallbacks = self._as_env_vector(safety_fallbacks, name="safety_fallbacks")
        self.safety_costs[position] = np.clip(
            costs, 0.0, self.safety_cost_clip
        )
        self.safety_fallbacks[position] = np.clip(fallbacks, 0.0, 1.0)

    def compute_safety_returns(
        self, *, last_safety_values: th.Tensor, dones: np.ndarray
    ) -> None:
        """Compute bootstrapped Monte-Carlo correction-cost returns."""

        last_values = (
            last_safety_values.detach().clone().cpu().numpy().reshape(-1)
        )
        if int(last_values.size) != int(self.n_envs):
            raise ValueError(
                "last_safety_values must contain one prediction per environment"
            )
        next_returns = last_values.astype(np.float32, copy=False)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - np.asarray(dones, dtype=np.float32)
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
            current = self.safety_costs[step] + (
                self.safety_gamma * next_non_terminal * next_returns
            )
            self.safety_returns[step] = current.astype(np.float32, copy=False)
            next_returns = self.safety_returns[step]

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[CBFSafetyRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for tensor in (
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "safety_costs",
                "safety_returns",
                "safety_fallbacks",
            ):
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        start_index = 0
        while start_index < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_index : start_index + batch_size])
            start_index += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Any = None,
    ) -> CBFSafetyRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.safety_costs[batch_inds].flatten(),
            self.safety_returns[batch_inds].flatten(),
            self.safety_fallbacks[batch_inds].flatten(),
        )
        return CBFSafetyRolloutBufferSamples(*tuple(map(self.to_torch, data)))


@dataclass(frozen=True)
class ProjectedPolicyEvaluation:
    values: th.Tensor
    safety_values: Optional[th.Tensor]
    log_prob: th.Tensor
    entropy: Optional[th.Tensor]
    distribution: DiagGaussianDistribution
    mu_raw: th.Tensor
    mu_safe: th.Tensor
    projection: TorchProjection2D


@dataclass(frozen=True)
class DetachedPolicyEvaluation:
    """Standard PPO outputs plus the unprojected actor mean."""

    values: th.Tensor
    log_prob: th.Tensor
    entropy: Optional[th.Tensor]
    distribution: DiagGaussianDistribution
    mu_raw: th.Tensor


class DetachedCBFActorCriticPolicy(ActorCriticPolicy):
    """Ordinary Gaussian actor that exposes its mean for detached CBF feedback.

    The CBF context is available to the algorithm when it constructs the safe
    target, but it is deliberately excluded from the learned actor and value
    features.  Unlike :class:`ProjectedCBFActorCriticPolicy`, this policy never
    places the projection in its forward graph or changes its distribution.
    """

    def __init__(
        self,
        *args,
        cbf_base_observation_dim: int = 42,
        **kwargs,
    ) -> None:
        if kwargs.get("use_sde", False):
            raise ValueError("DetachedCBFActorCriticPolicy does not support gSDE")
        extractor_class = kwargs.pop(
            "features_extractor_class", CBFBaseFeaturesExtractor
        )
        if extractor_class is not CBFBaseFeaturesExtractor:
            raise ValueError(
                "Detached CBF actor feedback requires CBFBaseFeaturesExtractor "
                "so context cannot leak into the learned state representation"
            )
        extractor_kwargs = dict(kwargs.pop("features_extractor_kwargs", {}) or {})
        extractor_kwargs["base_observation_dim"] = int(cbf_base_observation_dim)
        kwargs["features_extractor_class"] = CBFBaseFeaturesExtractor
        kwargs["features_extractor_kwargs"] = extractor_kwargs
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise TypeError(
                "Detached CBF actor feedback requires a diagonal Gaussian action distribution"
            )
        if not isinstance(self.action_space, spaces.Box) or tuple(
            self.action_space.shape
        ) != (2,):
            raise TypeError(
                "Detached CBF actor feedback requires a two-dimensional Box action space"
            )

    def _latents(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        pi_features, vf_features = features
        return (
            self.mlp_extractor.forward_actor(pi_features),
            self.mlp_extractor.forward_critic(vf_features),
        )

    def evaluate_actions_with_mean(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> DetachedPolicyEvaluation:
        """Evaluate stored latent actions without projecting the distribution."""

        latent_pi, latent_vf = self._latents(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if not isinstance(distribution, DiagGaussianDistribution):
            raise TypeError("Expected a diagonal Gaussian policy distribution")
        values = self.value_net(latent_vf)
        return DetachedPolicyEvaluation(
            values=values,
            log_prob=distribution.log_prob(actions),
            entropy=distribution.entropy(),
            distribution=distribution,
            mu_raw=distribution.distribution.mean,
        )


class ProjectedCBFActorCriticPolicy(ActorCriticPolicy):
    """Diagonal-Gaussian policy whose mean is an exact CBF-QP projection."""

    def __init__(
        self,
        *args,
        cbf_base_observation_dim: int = 42,
        cbf_max_constraints: int = 18,
        cbf_feasibility_tol: float = 1e-6,
        use_safety_critic: bool = True,
        **kwargs,
    ) -> None:
        if kwargs.get("use_sde", False):
            raise ValueError("ProjectedCBFActorCriticPolicy does not support gSDE")
        extractor_class = kwargs.pop(
            "features_extractor_class", CBFBaseFeaturesExtractor
        )
        if extractor_class is not CBFBaseFeaturesExtractor:
            raise ValueError(
                "Projected CBF policy requires CBFBaseFeaturesExtractor so context "
                "cannot leak into the learned state representation"
            )
        extractor_kwargs = dict(kwargs.pop("features_extractor_kwargs", {}) or {})
        extractor_kwargs["base_observation_dim"] = int(cbf_base_observation_dim)
        kwargs["features_extractor_class"] = CBFBaseFeaturesExtractor
        kwargs["features_extractor_kwargs"] = extractor_kwargs
        self.cbf_layout = CBFContextLayout(
            base_observation_dim=int(cbf_base_observation_dim),
            max_constraints=int(cbf_max_constraints),
        )
        self.cbf_feasibility_tol = float(cbf_feasibility_tol)
        # This must exist before ActorCriticPolicy.__init__ calls our _build().
        self.use_safety_critic = bool(use_safety_critic)
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise TypeError("Projected CBF PPO requires a diagonal Gaussian action distribution")
        if not isinstance(self.action_space, spaces.Box) or tuple(self.action_space.shape) != (2,):
            raise TypeError("Projected CBF PPO requires a two-dimensional Box action space")

    def _build(self, lr_schedule) -> None:
        """Optionally add a separate safety-value head to PPO's value pathway."""

        super()._build(lr_schedule)
        if not self.use_safety_critic:
            return
        self.safety_value_net = th.nn.Linear(
            int(self.mlp_extractor.latent_dim_vf), 1
        )
        if self.ortho_init:
            th.nn.init.orthogonal_(self.safety_value_net.weight, gain=1.0)
            th.nn.init.constant_(self.safety_value_net.bias, 0.0)
        # ActorCriticPolicy creates its optimizer before this custom head
        # exists. Rebuild it once so the safety critic is trainable and saved
        # with the same optimizer contract as the rest of the value pathway.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _latents(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        pi_features, vf_features = features
        return (
            self.mlp_extractor.forward_actor(pi_features),
            self.mlp_extractor.forward_critic(vf_features),
        )

    def project_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> TorchProjection2D:
        _, rows, bounds, mask = split_cbf_context_torch(
            obs, layout=self.cbf_layout
        )
        return project_polytope_2d_torch(
            actions,
            rows,
            bounds,
            mask,
            feasibility_tol=self.cbf_feasibility_tol,
            action_low=th.as_tensor(
                self.action_space.low, dtype=obs.dtype, device=obs.device
            ),
            action_high=th.as_tensor(
                self.action_space.high, dtype=obs.dtype, device=obs.device
            ),
        )

    def _distribution_and_stages(
        self, obs: th.Tensor
    ) -> tuple[
        DiagGaussianDistribution,
        th.Tensor,
        Optional[th.Tensor],
        th.Tensor,
        th.Tensor,
        TorchProjection2D,
    ]:
        latent_pi, latent_vf = self._latents(obs)
        values = self.value_net(latent_vf)
        safety_values: Optional[th.Tensor] = None
        if self.use_safety_critic:
            # The optional target is a discounted sum of squared hard-CBF
            # corrections, so a softplus keeps it non-negative.
            safety_values = F.softplus(self.safety_value_net(latent_vf))
        mu_raw = self.action_net(latent_pi)
        projection = self.project_actions(obs, mu_raw)
        # No mathematical projection exists when the no-slack set is empty.
        # Use the shared labelled fallback for behavior, but do not claim or
        # propagate an optimization-layer Jacobian through that fallback.
        mu_safe = th.where(
            projection.feasible.unsqueeze(1),
            projection.action,
            projection.action.detach(),
        )
        distribution = self.action_dist.proba_distribution(mu_safe, self.log_std)
        assert isinstance(distribution, DiagGaussianDistribution)
        return distribution, values, safety_values, mu_raw, mu_safe, projection

    def predict_safety_values(self, obs: th.Tensor) -> th.Tensor:
        """Predict discounted future CBF-correction cost from the value branch."""

        if not self.use_safety_critic:
            raise RuntimeError("This projected policy has no auxiliary safety critic")
        _, latent_vf = self._latents(obs)
        return F.softplus(self.safety_value_net(latent_vf))

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        distribution, values, _, _, _, _ = self._distribution_and_stages(obs)
        latent_z = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(latent_z)
        latent_z = latent_z.reshape((-1, *self.action_space.shape))
        return latent_z, values, log_prob

    def evaluate_actions_with_projection(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> ProjectedPolicyEvaluation:
        distribution, values, safety_values, mu_raw, mu_safe, projection = (
            self._distribution_and_stages(obs)
        )
        return ProjectedPolicyEvaluation(
            values=values,
            safety_values=safety_values,
            log_prob=distribution.log_prob(actions),
            entropy=distribution.entropy(),
            distribution=distribution,
            mu_raw=mu_raw,
            mu_safe=mu_safe,
            projection=projection,
        )

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        result = self.evaluate_actions_with_projection(obs, actions)
        return result.values, result.log_prob, result.entropy

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        distribution, _, _, _, _, _ = self._distribution_and_stages(obs)
        return distribution

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.get_distribution(observation).get_actions(
            deterministic=deterministic
        )

    def action_stages(
        self,
        obs: th.Tensor,
        *,
        deterministic: bool = True,
    ) -> dict[str, th.Tensor]:
        """Return raw mean, safe mean, latent z, and final hard projection."""

        distribution, _, _, mu_raw, mu_safe, mean_projection = (
            self._distribution_and_stages(obs)
        )
        latent_z = distribution.get_actions(deterministic=deterministic)
        executed_projection = self.project_actions(obs, latent_z)
        return {
            "mu_raw": mu_raw,
            "mu_safe": mu_safe,
            "latent_z": latent_z,
            "executed_action": executed_projection.action,
            "mean_feasible": mean_projection.feasible,
            "sample_feasible": executed_projection.feasible,
        }


def context_ignoring_policy_kwargs(
    *,
    base_observation_dim: int = 42,
    policy_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Attach the no-context-leak feature extractor to a standard PPO policy."""

    result = dict(policy_kwargs or {})
    result["features_extractor_class"] = CBFBaseFeaturesExtractor
    extractor_kwargs = dict(result.get("features_extractor_kwargs", {}) or {})
    extractor_kwargs["base_observation_dim"] = int(base_observation_dim)
    result["features_extractor_kwargs"] = extractor_kwargs
    return result


class LatentActionPPO(PPO):
    """PPO collector that stores ``z`` but executes a separate hard action."""

    def __init__(
        self,
        *args,
        execution_mode: str = "box",
        cbf_base_observation_dim: int = 42,
        cbf_max_constraints: int = 18,
        cbf_feasibility_tol: float = 1e-6,
        **kwargs,
    ) -> None:
        self.execution_mode = str(execution_mode).strip().lower()
        if self.execution_mode not in {"box", "cbf"}:
            raise ValueError("execution_mode must be 'box' or 'cbf'")
        self.cbf_layout = CBFContextLayout(
            base_observation_dim=int(cbf_base_observation_dim),
            max_constraints=int(cbf_max_constraints),
        )
        self.cbf_feasibility_tol = float(cbf_feasibility_tol)
        super().__init__(*args, **kwargs)

    def _execution_actions(
        self, latent_actions: np.ndarray, observations: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        low = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)
        action_dim = int(low.size)
        latent_array = np.asarray(latent_actions, dtype=np.float32)
        if latent_array.size == 0 or latent_array.size % action_dim != 0:
            raise ValueError(
                f"latent actions contain {latent_array.size} values, which cannot "
                f"form actions of width {action_dim}"
            )
        latent_actions = latent_array.reshape((-1, action_dim))
        observation_batch = np.asarray(observations, dtype=np.float32)
        if observation_batch.ndim == 1:
            observation_batch = observation_batch.reshape(1, -1)
        elif observation_batch.ndim < 1:
            raise ValueError("observations must include an observation dimension")
        else:
            observation_batch = observation_batch.reshape(
                (-1, observation_batch.shape[-1])
            )
        batch_size = int(latent_actions.shape[0])
        if int(observation_batch.shape[0]) != batch_size:
            raise ValueError(
                "latent-action and observation batch sizes differ "
                f"({batch_size} != {observation_batch.shape[0]})"
            )
        _, rows, bounds, mask = split_cbf_context_numpy(
            observation_batch, layout=self.cbf_layout
        )
        executed = np.empty_like(latent_actions)
        records: list[dict[str, Any]] = []
        for env_index in range(batch_size):
            raw = latent_actions[env_index]
            active = np.asarray(mask[env_index] > 0.5, dtype=bool)
            active_rows = np.asarray(rows[env_index][active], dtype=np.float32)
            active_bounds = np.asarray(bounds[env_index][active], dtype=np.float32)
            context_hash = constraint_system_hash(active_rows, active_bounds)
            if self.execution_mode == "cbf":
                projection = project_polytope_2d_numpy(
                    raw,
                    rows[env_index],
                    bounds[env_index],
                    mask[env_index],
                    feasibility_tol=self.cbf_feasibility_tol,
                    action_low=low,
                    action_high=high,
                )
                safe = projection.action
                record = {
                    "feasible": projection.feasible,
                    "fallback_used": projection.fallback_used,
                    "projection_source": projection.source,
                    "max_constraint_violation_safe": projection.max_violation,
                    "active_indices": projection.active_indices,
                    "constraint_hash": context_hash,
                    "cbf_applied": True,
                }
            else:
                safe = np.clip(raw, low, high).astype(np.float32)
                record = {
                    "feasible": True,
                    "fallback_used": False,
                    "projection_source": "box",
                    "max_constraint_violation_safe": 0.0,
                    "active_indices": np.zeros(0, dtype=np.int64),
                    "constraint_hash": context_hash,
                    "cbf_applied": False,
                }
            executed[env_index] = safe
            records.append(record)
        return executed, records

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect latent actions while the environment executes ``P_s(z)``."""

        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                latent_actions_tensor, values, log_probs = self.policy(obs_tensor)
            latent_actions = latent_actions_tensor.cpu().numpy()
            executed_actions, projection_records = self._execution_actions(
                latent_actions, np.asarray(self._last_obs)
            )
            for env_index, record in enumerate(projection_records):
                env.env_method(
                    "set_projection_record",
                    latent_actions[env_index],
                    executed_actions[env_index],
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

            # The callback sees both quantities.  Crucially, RolloutBuffer.add
            # below receives latent_actions, never executed_actions.
            actions = latent_actions
            clipped_actions = executed_actions
            new_obs, rewards, dones, infos = env.step(executed_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)
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
                latent_actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                **(
                    {
                        "safety_costs": np.asarray(
                            [
                                float(
                                    np.clip(
                                        float(
                                            info.get(
                                                "cbf_correction_norm_normalized",
                                                0.0,
                                            )
                                        )
                                        ** 2,
                                        0.0,
                                        rollout_buffer.safety_cost_clip,
                                    )
                                )
                                for info in infos
                            ],
                            dtype=np.float32,
                        ),
                        "safety_fallbacks": np.asarray(
                            [
                                float(bool(info.get("cbf_fallback_used", False)))
                                for info in infos
                            ],
                            dtype=np.float32,
                        ),
                    }
                    if isinstance(rollout_buffer, CBFSafetyRolloutBuffer)
                    else {}
                ),
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(
                obs_as_tensor(new_obs, self.device)
            )
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        if isinstance(rollout_buffer, CBFSafetyRolloutBuffer):
            if not isinstance(self.policy, ProjectedCBFActorCriticPolicy):
                raise TypeError(
                    "CBFSafetyRolloutBuffer requires ProjectedCBFActorCriticPolicy"
                )
            with th.no_grad():
                safety_values = self.policy.predict_safety_values(
                    obs_as_tensor(new_obs, self.device)
                )
            rollout_buffer.compute_safety_returns(
                last_safety_values=safety_values, dones=dones
            )
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def predict_action_stages(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> dict[str, np.ndarray]:
        """Diagnostic API that never confuses latent and executed actions."""

        obs_tensor, vectorized = self.policy.obs_to_tensor(observation)
        with th.no_grad():
            if isinstance(self.policy, ProjectedCBFActorCriticPolicy):
                stages = self.policy.action_stages(
                    obs_tensor, deterministic=deterministic
                )
                result = {
                    key: value.detach().cpu().numpy() for key, value in stages.items()
                }
            else:
                distribution = self.policy.get_distribution(obs_tensor)
                latent = distribution.get_actions(deterministic=deterministic)
                latent_np = latent.detach().cpu().numpy()
                executed, _ = self._execution_actions(
                    latent_np, obs_tensor.detach().cpu().numpy()
                )
                result = {
                    "mu_raw": distribution.distribution.mean.detach().cpu().numpy(),
                    "mu_safe": distribution.distribution.mean.detach().cpu().numpy(),
                    "latent_z": latent_np,
                    "executed_action": executed,
                }
        if not vectorized:
            result = {key: np.asarray(value).squeeze(axis=0) for key, value in result.items()}
        return result


def _gradient_pair_diagnostics(
    parameters: list[th.nn.Parameter],
    primary_loss: th.Tensor,
    auxiliary_loss: th.Tensor,
) -> dict[str, float]:
    """Measure PPO/CBF actor gradient norms without mutating ``.grad``."""

    primary = th.autograd.grad(
        primary_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    auxiliary = th.autograd.grad(
        auxiliary_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    primary_norm_sq = th.zeros((), device=primary_loss.device)
    auxiliary_norm_sq = th.zeros((), device=primary_loss.device)
    dot = th.zeros((), device=primary_loss.device)
    for primary_grad, auxiliary_grad in zip(primary, auxiliary):
        if primary_grad is not None:
            primary_norm_sq = primary_norm_sq + primary_grad.detach().square().sum()
        if auxiliary_grad is not None:
            auxiliary_norm_sq = auxiliary_norm_sq + auxiliary_grad.detach().square().sum()
        if primary_grad is not None and auxiliary_grad is not None:
            dot = dot + (primary_grad.detach() * auxiliary_grad.detach()).sum()
    primary_norm = th.sqrt(primary_norm_sq)
    auxiliary_norm = th.sqrt(auxiliary_norm_sq)
    denominator = primary_norm * auxiliary_norm
    cosine = dot / denominator if float(denominator) > 1e-12 else th.full_like(dot, th.nan)
    ratio = auxiliary_norm / primary_norm.clamp_min(1e-12)
    return {
        "g_ppo_norm": float(primary_norm.cpu().item()),
        "g_cbf_norm": float(auxiliary_norm.cpu().item()),
        "g_cbf_to_g_ppo_ratio": float(ratio.cpu().item()),
        "g_ppo_g_cbf_cosine": float(cosine.cpu().item()),
    }


class DetachedCBFActorPPO(LatentActionPPO):
    """Ordinary PPO plus a stopped-gradient hard-CBF actor target.

    The policy distribution remains ``Normal(mu_raw, sigma)``.  At every PPO
    minibatch update, the current mean is projected with the hard CBF solver
    under ``no_grad`` and the actor minimizes ``||mu_raw - stopgrad(P(mu_raw))||^2``.
    PPO still evaluates the original latent rollout action and its original
    log probability, so the non-differentiable filter remains part of the
    environment rather than being mistaken for a transformed distribution.
    """

    policy_aliases = {
        **PPO.policy_aliases,
        "DetachedCBFPolicy": DetachedCBFActorCriticPolicy,
    }

    def __init__(
        self,
        *args,
        lambda_actor: float = 0.10,
        **kwargs,
    ) -> None:
        self.lambda_actor = float(lambda_actor)
        if not np.isfinite(self.lambda_actor) or self.lambda_actor < 0.0:
            raise ValueError("lambda_actor must be finite and non-negative")
        requested_execution_mode = str(
            kwargs.get("execution_mode", "cbf")
        ).strip().lower()
        if requested_execution_mode != "cbf":
            raise ValueError(
                "DetachedCBFActorPPO requires execution_mode='cbf' so its "
                "feedback target matches hard-CBF training execution"
            )
        kwargs["execution_mode"] = "cbf"
        self.cbf_training_diagnostics: list[dict[str, float]] = []
        super().__init__(*args, **kwargs)
        # SB3 load() first constructs with _init_setup_model=False and restores
        # the serialized policy afterward, so only validate an initialized one.
        if hasattr(self, "policy") and not isinstance(
            self.policy, DetachedCBFActorCriticPolicy
        ):
            raise TypeError(
                "DetachedCBFActorPPO must use DetachedCBFActorCriticPolicy"
            )

    def project_actor_mean_detached(
        self,
        observations: th.Tensor,
        mean_actions: th.Tensor,
    ) -> TorchProjection2D:
        """Return a current-mean CBF target with no solver gradient path."""

        with th.no_grad():
            _, rows, bounds, mask = split_cbf_context_torch(
                observations, layout=self.cbf_layout
            )
            return project_polytope_2d_torch(
                mean_actions.detach(),
                rows,
                bounds,
                mask,
                feasibility_tol=self.cbf_feasibility_tol,
                action_low=th.as_tensor(
                    self.action_space.low,
                    dtype=observations.dtype,
                    device=observations.device,
                ),
                action_high=th.as_tensor(
                    self.action_space.high,
                    dtype=observations.dtype,
                    device=observations.device,
                ),
            )

    @staticmethod
    def detached_actor_loss(
        mean_actions: th.Tensor,
        projection: TorchProjection2D,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Compute the feasible-target loss, correction, and infeasible rate."""

        safe_target = projection.action.detach()
        delta = mean_actions - safe_target
        feasible = projection.feasible.to(delta.dtype)
        denominator = feasible.sum().clamp_min(1.0)
        loss = (delta.square().sum(dim=1) * feasible).sum() / denominator
        correction = (
            th.linalg.vector_norm(delta.detach(), dim=1) * feasible
        ).sum() / denominator
        infeasible_rate = (~projection.feasible).float().mean()
        return loss, correction, infeasible_rate

    def train(self) -> None:
        """Run standard PPO with one detached, actor-only CBF regularizer."""

        if not isinstance(self.policy, DetachedCBFActorCriticPolicy):
            raise TypeError(
                "DetachedCBFActorPPO requires DetachedCBFActorCriticPolicy"
            )
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        mean_losses: list[float] = []
        mean_corrections: list[float] = []
        mean_infeasible_rates: list[float] = []
        gradient_rows: list[dict[str, float]] = []
        all_approx_kl_divs: list[float] = []
        continue_training = True
        last_loss = th.zeros((), device=self.device)

        actor_parameters = [
            parameter
            for name, parameter in self.policy.named_parameters()
            if parameter.requires_grad
            and not name.startswith(("value_net", "mlp_extractor.value_net"))
        ]

        for epoch in range(self.n_epochs):
            epoch_approx_kl_divs: list[float] = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                evaluation = self.policy.evaluate_actions_with_mean(
                    rollout_data.observations, rollout_data.actions
                )
                values = evaluation.values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                ratio = th.exp(evaluation.log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(float(policy_loss.detach().cpu().item()))
                clip_fractions.append(
                    float(th.mean((th.abs(ratio - 1) > clip_range).float()).item())
                )

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(float(value_loss.detach().cpu().item()))

                if evaluation.entropy is None:
                    entropy_loss = -th.mean(-evaluation.log_prob)
                else:
                    entropy_loss = -th.mean(evaluation.entropy)
                entropy_losses.append(float(entropy_loss.detach().cpu().item()))

                projection = self.project_actor_mean_detached(
                    rollout_data.observations, evaluation.mu_raw
                )
                mean_loss, mean_correction, mean_infeasible_rate = (
                    self.detached_actor_loss(evaluation.mu_raw, projection)
                )
                mean_losses.append(float(mean_loss.detach().cpu().item()))
                mean_corrections.append(
                    float(mean_correction.detach().cpu().item())
                )
                mean_infeasible_rates.append(
                    float(mean_infeasible_rate.detach().cpu().item())
                )

                actor_primary_loss = policy_loss + self.ent_coef * entropy_loss
                actor_auxiliary_loss = self.lambda_actor * mean_loss
                # One representative gradient comparison per PPO train() call
                # is enough for diagnostics.  Computing it for every minibatch
                # would add two extra autograd passes throughout a 1M run.
                if self.lambda_actor != 0.0 and not gradient_rows:
                    gradient_rows.append(
                        _gradient_pair_diagnostics(
                            actor_parameters,
                            actor_primary_loss,
                            actor_auxiliary_loss,
                        )
                    )
                loss = (
                    actor_primary_loss
                    + self.vf_coef * value_loss
                    + actor_auxiliary_loss
                )
                last_loss = loss

                with th.no_grad():
                    log_ratio = evaluation.log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio
                    ).cpu().item()
                    epoch_approx_kl_divs.append(float(approx_kl_div))
                    all_approx_kl_divs.append(float(approx_kl_div))
                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching "
                            f"max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(all_approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", float(last_loss.detach().cpu().item()))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/cbf_detached_actor_lambda", self.lambda_actor)
        self.logger.record("train/cbf_mean_loss", np.mean(mean_losses))
        self.logger.record(
            "train/cbf_mean_correction", np.mean(mean_corrections)
        )
        self.logger.record(
            "train/cbf_mean_infeasible_rate", np.mean(mean_infeasible_rates)
        )
        gradient_summary: dict[str, float] = {}
        if gradient_rows:
            for key in gradient_rows[0]:
                finite = [
                    row[key] for row in gradient_rows if np.isfinite(row[key])
                ]
                gradient_summary[key] = (
                    float(np.mean(finite)) if finite else np.nan
                )
                self.logger.record(f"train/actor_{key}", gradient_summary[key])
        self.cbf_training_diagnostics.append(
            {
                "n_updates": float(self._n_updates),
                "num_timesteps": float(self.num_timesteps),
                "mean_loss": float(np.mean(mean_losses)),
                "mean_correction": float(np.mean(mean_corrections)),
                "mean_infeasible_rate": float(np.mean(mean_infeasible_rates)),
                **gradient_summary,
            }
        )
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", th.exp(self.policy.log_std).mean().item()
            )
        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard"
        )
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


class ProjectedCBFPPO(LatentActionPPO):
    """PPO with a differentiable projected actor and ordinary reward critic.

    A separate CBF safety critic remains available only for loading historical
    checkpoints or an explicit nonzero ``lambda_critic``.  The canonical study
    uses the standard PPO rollout buffer and value loss exclusively.
    """

    policy_aliases = {
        **PPO.policy_aliases,
        "ProjectedCBFPolicy": ProjectedCBFActorCriticPolicy,
    }

    def __init__(
        self,
        *args,
        lambda_mean: float = 0.10,
        lambda_sample: float = 0.0,
        lambda_critic: float = 0.0,
        safety_gamma: float = 0.99,
        safety_cost_clip: float = 1.0,
        **kwargs,
    ) -> None:
        self.lambda_mean = float(lambda_mean)
        self.lambda_sample = float(lambda_sample)
        self.lambda_critic = float(lambda_critic)
        self.safety_gamma = float(safety_gamma)
        self.safety_cost_clip = float(safety_cost_clip)
        for name, value in (
            ("lambda_mean", self.lambda_mean),
            ("lambda_sample", self.lambda_sample),
            ("lambda_critic", self.lambda_critic),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(self.safety_gamma) or not 0.0 <= self.safety_gamma <= 1.0:
            raise ValueError("safety_gamma must be finite and lie in [0, 1]")
        if not np.isfinite(self.safety_cost_clip) or self.safety_cost_clip <= 0.0:
            raise ValueError("safety_cost_clip must be finite and positive")
        self.cbf_training_diagnostics: list[dict[str, float]] = []
        kwargs.setdefault("execution_mode", "cbf")
        provided_buffer_class = kwargs.pop("rollout_buffer_class", None)
        policy_kwargs = dict(kwargs.pop("policy_kwargs", {}) or {})
        requested_safety_critic = policy_kwargs.get("use_safety_critic")
        if requested_safety_critic is None:
            # Historical projected checkpoints did not save this policy flag,
            # but their saved buffer class identifies the old safety path.
            self.use_safety_critic = bool(
                self.lambda_critic > 0.0
                or provided_buffer_class is CBFSafetyRolloutBuffer
            )
        else:
            self.use_safety_critic = bool(requested_safety_critic)
        if self.lambda_critic > 0.0 and not self.use_safety_critic:
            raise ValueError(
                "A nonzero lambda_critic requires use_safety_critic=True"
            )
        policy_kwargs["use_safety_critic"] = self.use_safety_critic
        kwargs["policy_kwargs"] = policy_kwargs
        buffer_kwargs = dict(kwargs.pop("rollout_buffer_kwargs", {}) or {})
        if self.use_safety_critic:
            if provided_buffer_class not in (None, CBFSafetyRolloutBuffer):
                raise ValueError(
                    "The optional safety critic requires CBFSafetyRolloutBuffer"
                )
            buffer_kwargs.update(
                {
                    "safety_gamma": self.safety_gamma,
                    "safety_cost_clip": self.safety_cost_clip,
                }
            )
            kwargs["rollout_buffer_class"] = CBFSafetyRolloutBuffer
        else:
            if provided_buffer_class not in (None, RolloutBuffer):
                raise ValueError(
                    "Projected PPO without a safety critic requires RolloutBuffer"
                )
            if buffer_kwargs:
                raise ValueError(
                    "Plain projected PPO does not accept safety rollout-buffer kwargs"
                )
            kwargs["rollout_buffer_class"] = RolloutBuffer
        kwargs["rollout_buffer_kwargs"] = buffer_kwargs
        super().__init__(*args, **kwargs)
        # SB3 load() constructs once with _init_setup_model=False and restores
        # the policy immediately afterward.
        if hasattr(self, "policy") and not isinstance(
            self.policy, ProjectedCBFActorCriticPolicy
        ):
            raise TypeError(
                "ProjectedCBFPPO must be constructed with ProjectedCBFActorCriticPolicy"
            )
        if hasattr(self, "policy") and bool(
            self.policy.use_safety_critic
        ) != bool(self.use_safety_critic):
            raise RuntimeError(
                "Projected policy and rollout buffer disagree on safety-critic usage"
            )

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        safety_critic_losses: list[float] = []
        safety_target_means: list[float] = []
        safety_prediction_rmses: list[float] = []
        safety_cost_means: list[float] = []
        safety_fallback_rates: list[float] = []
        clip_fractions: list[float] = []
        mean_losses: list[float] = []
        sample_losses: list[float] = []
        mean_corrections: list[float] = []
        sample_corrections: list[float] = []
        mean_infeasible_rates: list[float] = []
        sample_infeasible_rates: list[float] = []
        gradient_rows: list[dict[str, float]] = []
        all_approx_kl_divs: list[float] = []
        continue_training = True
        last_loss = th.zeros((), device=self.device)

        actor_parameters = [
            parameter
            for name, parameter in self.policy.named_parameters()
            if parameter.requires_grad
            and not name.startswith(
                ("value_net", "safety_value_net", "mlp_extractor.value_net")
            )
        ]

        for epoch in range(self.n_epochs):
            epoch_approx_kl_divs: list[float] = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                evaluation = self.policy.evaluate_actions_with_projection(
                    rollout_data.observations, actions
                )
                values = evaluation.values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )
                ratio = th.exp(evaluation.log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(float(policy_loss.detach().cpu().item()))
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(float(clip_fraction))

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(float(value_loss.detach().cpu().item()))

                safety_critic_loss = th.zeros((), device=self.device)
                if self.use_safety_critic:
                    if evaluation.safety_values is None or not isinstance(
                        self.rollout_buffer, CBFSafetyRolloutBuffer
                    ):
                        raise RuntimeError(
                            "Enabled safety critic is missing its value head or rollout targets"
                        )
                    safety_values = evaluation.safety_values.flatten()
                    safety_targets = rollout_data.safety_returns.detach()
                    safety_critic_loss = F.smooth_l1_loss(
                        safety_values, safety_targets
                    )
                    safety_critic_losses.append(
                        float(safety_critic_loss.detach().cpu().item())
                    )
                    safety_target_means.append(
                        float(safety_targets.mean().detach().cpu().item())
                    )
                    safety_prediction_rmses.append(
                        float(
                            th.sqrt(
                                F.mse_loss(
                                    safety_values.detach(), safety_targets
                                )
                            )
                            .cpu()
                            .item()
                        )
                    )
                    safety_cost_means.append(
                        float(
                            rollout_data.safety_costs.mean().detach().cpu().item()
                        )
                    )
                    safety_fallback_rates.append(
                        float(
                            rollout_data.safety_fallbacks.mean()
                            .detach()
                            .cpu()
                            .item()
                        )
                    )

                if evaluation.entropy is None:
                    entropy_loss = -th.mean(-evaluation.log_prob)
                else:
                    entropy_loss = -th.mean(evaluation.entropy)
                entropy_losses.append(float(entropy_loss.detach().cpu().item()))

                mean_target = th.where(
                    evaluation.projection.feasible.unsqueeze(1),
                    evaluation.mu_safe,
                    evaluation.mu_safe.detach(),
                )
                mean_delta = evaluation.mu_raw - mean_target
                mean_feasible = evaluation.projection.feasible.to(mean_delta.dtype)
                mean_denominator = mean_feasible.sum().clamp_min(1.0)
                mean_loss = (
                    mean_delta.square().sum(dim=1) * mean_feasible
                ).sum() / mean_denominator
                mean_losses.append(float(mean_loss.detach().cpu().item()))
                mean_corrections.append(
                    float(
                        (
                            th.linalg.vector_norm(mean_delta.detach(), dim=1)
                            * mean_feasible
                        ).sum()
                        .div(mean_denominator)
                        .cpu()
                        .item()
                    )
                )
                mean_infeasible_rates.append(
                    float((~evaluation.projection.feasible).float().mean().cpu().item())
                )

                sample_loss = th.zeros((), device=self.device)
                if self.lambda_sample != 0.0:
                    # A stored rollout z is constant during this update.  Use a
                    # fresh reparameterized current-policy sample so this term
                    # can train both the mean and log standard deviation.
                    fresh_z = evaluation.distribution.distribution.rsample()
                    fresh_projection = self.policy.project_actions(
                        rollout_data.observations, fresh_z
                    )
                    fresh_target = th.where(
                        fresh_projection.feasible.unsqueeze(1),
                        fresh_projection.action,
                        fresh_projection.action.detach(),
                    )
                    sample_delta = fresh_z - fresh_target
                    sample_feasible = fresh_projection.feasible.to(
                        sample_delta.dtype
                    )
                    sample_denominator = sample_feasible.sum().clamp_min(1.0)
                    sample_loss = (
                        sample_delta.square().sum(dim=1) * sample_feasible
                    ).sum() / sample_denominator
                    sample_corrections.append(
                        float(
                            (
                                th.linalg.vector_norm(sample_delta.detach(), dim=1)
                                * sample_feasible
                            ).sum()
                            .div(sample_denominator)
                            .cpu()
                            .item()
                        )
                    )
                    sample_infeasible_rates.append(
                        float((~fresh_projection.feasible).float().mean().cpu().item())
                    )
                sample_losses.append(float(sample_loss.detach().cpu().item()))

                actor_auxiliary_loss = (
                    self.lambda_mean * mean_loss
                    + self.lambda_sample * sample_loss
                )
                critic_auxiliary_loss = self.lambda_critic * safety_critic_loss
                if (
                    (self.lambda_mean != 0.0 or self.lambda_sample != 0.0)
                    and not gradient_rows
                ):
                    actor_primary_loss = (
                        policy_loss + self.ent_coef * entropy_loss
                    )
                    gradient_rows.append(
                        _gradient_pair_diagnostics(
                            actor_parameters,
                            actor_primary_loss,
                            actor_auxiliary_loss,
                        )
                    )
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + actor_auxiliary_loss
                    + critic_auxiliary_loss
                )
                last_loss = loss

                with th.no_grad():
                    log_ratio = evaluation.log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio
                    ).cpu().item()
                    epoch_approx_kl_divs.append(float(approx_kl_div))
                    all_approx_kl_divs.append(float(approx_kl_div))
                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching "
                            f"max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(all_approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", float(last_loss.detach().cpu().item()))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/cbf_lambda_mean", self.lambda_mean)
        self.logger.record("train/cbf_lambda_sample", self.lambda_sample)
        self.logger.record("train/cbf_mean_loss", np.mean(mean_losses))
        self.logger.record("train/cbf_sample_loss", np.mean(sample_losses))
        if self.use_safety_critic:
            self.logger.record("train/cbf_lambda_critic", self.lambda_critic)
            self.logger.record(
                "train/cbf_safety_critic_loss", np.mean(safety_critic_losses)
            )
            self.logger.record(
                "train/cbf_safety_target_mean", np.mean(safety_target_means)
            )
            self.logger.record(
                "train/cbf_safety_prediction_rmse",
                np.mean(safety_prediction_rmses),
            )
            self.logger.record(
                "train/cbf_safety_cost_mean", np.mean(safety_cost_means)
            )
            self.logger.record(
                "train/cbf_safety_fallback_rate",
                np.mean(safety_fallback_rates),
            )
        self.logger.record("train/cbf_mean_correction", np.mean(mean_corrections))
        self.logger.record("train/cbf_mean_infeasible_rate", np.mean(mean_infeasible_rates))
        self.logger.record(
            "train/cbf_sample_correction",
            np.mean(sample_corrections) if sample_corrections else 0.0,
        )
        self.logger.record(
            "train/cbf_sample_infeasible_rate",
            np.mean(sample_infeasible_rates) if sample_infeasible_rates else 0.0,
        )
        gradient_summary: dict[str, float] = {}
        if gradient_rows:
            for key in gradient_rows[0]:
                finite = [row[key] for row in gradient_rows if np.isfinite(row[key])]
                gradient_summary[key] = float(np.mean(finite)) if finite else np.nan
                self.logger.record(
                    f"train/actor_{key}", gradient_summary[key]
                )
        diagnostic_row = {
            "n_updates": float(self._n_updates),
            "num_timesteps": float(self.num_timesteps),
            "mean_loss": float(np.mean(mean_losses)),
            "sample_loss": float(np.mean(sample_losses)),
            "mean_correction": float(np.mean(mean_corrections)),
            "mean_infeasible_rate": float(np.mean(mean_infeasible_rates)),
            "sample_correction": float(
                np.mean(sample_corrections) if sample_corrections else 0.0
            ),
            "sample_infeasible_rate": float(
                np.mean(sample_infeasible_rates)
                if sample_infeasible_rates
                else 0.0
            ),
            **gradient_summary,
        }
        if self.use_safety_critic:
            diagnostic_row.update(
                {
                    "safety_critic_loss": float(
                        np.mean(safety_critic_losses)
                    ),
                    "safety_target_mean": float(
                        np.mean(safety_target_means)
                    ),
                    "safety_prediction_rmse": float(
                        np.mean(safety_prediction_rmses)
                    ),
                    "safety_cost_mean": float(np.mean(safety_cost_means)),
                    "safety_fallback_rate": float(
                        np.mean(safety_fallback_rates)
                    ),
                }
            )
        self.cbf_training_diagnostics.append(diagnostic_row)
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", th.exp(self.policy.log_std).mean().item()
            )
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
