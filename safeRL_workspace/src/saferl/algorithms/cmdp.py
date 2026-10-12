"""CMDP core: cost returns, the constraint estimate, and the dual variable.

The constrained problem is

    max_theta  J_R(theta)   s.t.   J_C(theta) <= d

with a task return ``J_R`` under ``gamma`` and a *separate* safety objective
defined as the **average per-step cost under the policy's occupancy measure**

    J_C(pi) = E_{(s, a) ~ d_pi}[ c(s, a) ]

Reward and safety are two signals. This is deliberately **not** the reward
shaping form ``r' = r - lambda * c``: the penalty weight here is a dual
variable driven by constraint violation, not a fixed hyperparameter.

**Why an average and not an episodic sum.** Episodes here have very different
lengths and a collision terminates one, so an unnormalized cost sum would
reward a policy for crashing early and accumulating less cost. The average
form removes that pathology and puts ``d`` in the same units as the
instantaneous cost.

**The dual estimator and the advantage estimator are not the same object.**
The dual variable is driven by the average per-step cost above, while the
policy gradient uses a ``gamma_c``-discounted cost GAE. Nothing requires those
to coincide, and keeping them separate is what makes ``d`` interpretable while
still giving the actor a horizon-aware cost signal.

Unit discipline, which is the easiest thing to get wrong here:

* ``cost_objective_estimate`` takes **raw** (unscaled) costs for the
  ``"batch_mean"`` method, because ``J_C`` is itself an average of raw
  per-step costs. Feeding it scaled costs introduces a spurious factor of
  ``(1 - gamma_c)`` and silently shrinks every constraint violation.
* ``normalize_cost_objective`` is therefore **off** by default. It pre-scales
  stored costs by ``(1 - gamma_c)``, which was needed only when ``J_C`` was
  defined as a discounted-and-renormalized return. Under the average-cost
  definition it would leave the dual error in raw per-step units while the
  cost advantages are 50x smaller at ``gamma_c = 0.98``, forcing ``lambda`` to
  absorb the mismatch. The flag is retained for the normalized-critic variant
  but must be a deliberate choice.

This module is framework-free on purpose: no torch, no Stable-Baselines3, no
policy. The clipped surrogates belong to the trainer that owns the autograd
graph; everything here is contract-testable in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

# How the scalar constraint estimate ``J_C_hat`` driving the dual update is
# formed from a rollout. ``J_C`` is defined at the initial-state distribution,
# but a rollout of a few tens of thousands of steps contains few episode
# starts, so the faithful estimator is the noisiest one.
#
# "batch_mean"    mean raw per-step cost over the batch. Exactly ``J_C`` in the
#                 gamma_c -> 1 limit and an occupancy-weighted proxy below it.
#                 Lowest variance, needs no episode boundaries, keeps ``d`` in
#                 per-step units. The default.
# "value_mean"    mean of the cost critic over visited states. Low variance but
#                 carries the critic's bias, and weights by the visited-state
#                 occupancy rather than by ``rho_0``.
# "episode_start" mean of the cost critic at episode-start states only. Closest
#                 to the definition of ``J_C``; high variance when a rollout
#                 holds few starts.
BATCH_MEAN_ESTIMATOR = "batch_mean"
VALUE_MEAN_ESTIMATOR = "value_mean"
EPISODE_START_ESTIMATOR = "episode_start"
SUPPORTED_COST_ESTIMATORS = (
    BATCH_MEAN_ESTIMATOR,
    VALUE_MEAN_ESTIMATOR,
    EPISODE_START_ESTIMATOR,
)


@dataclass(frozen=True)
class CMDPConfig:
    """Resolved CMDP and Lagrangian settings.

    ``cost_limit`` is deliberately required with no default. It is the single
    most consequential setting here and it has to be chosen from measured cost
    distributions, not inherited from a placeholder.
    """

    cost_limit: float
    gamma_c: float = 0.98
    gae_lambda_c: float = 0.95
    dual_lr: float = 0.01
    dual_init: float = 0.0
    dual_max: Optional[float] = None
    # Off by default: see the module docstring. Under the average-cost
    # definition of J_C this would put the dual error and the cost advantages
    # in units differing by 1 / (1 - gamma_c).
    normalize_cost_objective: bool = False
    # Locked off: normalizing the cost advantage destroys the relationship
    # between lambda, the physical cost scale and the cost limit.
    normalize_cost_advantage: bool = False
    # Off initially. Scaling the combined actor objective by 1 / (1 + lambda)
    # preserves the gradient direction at fixed lambda but changes its
    # magnitude, so it interacts with Adam, the entropy term, clipping and the
    # learning rate and makes interpretation less clean. Revisit only if a
    # grown lambda is measured to destabilize optimization.
    use_penalty_scaling: bool = False
    # Separate reward and cost critic parameters. Sharing a trunk would let the
    # cost regression reshape the representation the reward critic uses, so any
    # CMDP effect could come partly from representation interference rather
    # than from constrained optimization.
    separate_cost_critic: bool = True
    cost_estimator: str = BATCH_MEAN_ESTIMATOR

    def __post_init__(self) -> None:
        object.__setattr__(self, "cost_limit", float(self.cost_limit))
        object.__setattr__(self, "gamma_c", float(self.gamma_c))
        object.__setattr__(self, "gae_lambda_c", float(self.gae_lambda_c))
        object.__setattr__(self, "dual_lr", float(self.dual_lr))
        object.__setattr__(self, "dual_init", float(self.dual_init))
        object.__setattr__(
            self, "cost_estimator", str(self.cost_estimator).strip().lower()
        )
        if self.dual_max is not None:
            object.__setattr__(self, "dual_max", float(self.dual_max))

        if not math.isfinite(self.cost_limit) or self.cost_limit < 0.0:
            raise ValueError("cost_limit must be finite and non-negative")
        for name in ("gamma_c", "gae_lambda_c"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        if self.normalize_cost_objective and self.gamma_c >= 1.0:
            # The (1 - gamma_c) weights would be degenerate.
            raise ValueError(
                "normalize_cost_objective requires gamma_c < 1"
            )
        if not math.isfinite(self.dual_lr) or self.dual_lr <= 0.0:
            raise ValueError("dual_lr must be finite and positive")
        if not math.isfinite(self.dual_init) or self.dual_init < 0.0:
            raise ValueError("dual_init must be finite and non-negative")
        if self.dual_max is not None:
            if not math.isfinite(self.dual_max) or self.dual_max <= 0.0:
                raise ValueError("dual_max must be finite and positive")
            if self.dual_init > self.dual_max:
                raise ValueError("dual_init must not exceed dual_max")
        if self.cost_estimator not in SUPPORTED_COST_ESTIMATORS:
            raise ValueError(
                f"Unsupported cost_estimator {self.cost_estimator!r}; expected "
                f"one of {SUPPORTED_COST_ESTIMATORS}"
            )

    @property
    def cost_scale(self) -> float:
        """The ``(1 - gamma_c)`` factor applied to stored per-step costs."""

        return (1.0 - self.gamma_c) if self.normalize_cost_objective else 1.0

    def scale_costs(self, costs: Any) -> np.ndarray:
        """Return per-step costs in the units the cost critic learns."""

        return np.asarray(costs, dtype=np.float64) * self.cost_scale

    def as_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe record for signatures and manifests."""

        return {
            "cost_limit": self.cost_limit,
            "gamma_c": self.gamma_c,
            "gae_lambda_c": self.gae_lambda_c,
            "dual_lr": self.dual_lr,
            "dual_init": self.dual_init,
            "dual_max": self.dual_max,
            "normalize_cost_objective": self.normalize_cost_objective,
            "normalize_cost_advantage": self.normalize_cost_advantage,
            "use_penalty_scaling": self.use_penalty_scaling,
            "separate_cost_critic": self.separate_cost_critic,
            "cost_estimator": self.cost_estimator,
            "cost_scale": self.cost_scale,
        }


def cmdp_config_from_mapping(config: Mapping[str, Any] | None) -> CMDPConfig:
    """Build a config from a flat mapping such as a stored run config."""

    if not config or "cost_limit" not in config:
        raise ValueError("a stored CMDP config must record cost_limit")
    defaults = CMDPConfig(cost_limit=float(config["cost_limit"]))
    return CMDPConfig(
        cost_limit=float(config["cost_limit"]),
        gamma_c=float(config.get("gamma_c", defaults.gamma_c)),
        gae_lambda_c=float(config.get("gae_lambda_c", defaults.gae_lambda_c)),
        dual_lr=float(config.get("dual_lr", defaults.dual_lr)),
        dual_init=float(config.get("dual_init", defaults.dual_init)),
        dual_max=(
            None if config.get("dual_max") is None else float(config["dual_max"])
        ),
        normalize_cost_objective=bool(
            config.get("normalize_cost_objective", defaults.normalize_cost_objective)
        ),
        normalize_cost_advantage=bool(
            config.get("normalize_cost_advantage", defaults.normalize_cost_advantage)
        ),
        use_penalty_scaling=bool(
            config.get("use_penalty_scaling", defaults.use_penalty_scaling)
        ),
        separate_cost_critic=bool(
            config.get("separate_cost_critic", defaults.separate_cost_critic)
        ),
        cost_estimator=str(config.get("cost_estimator", defaults.cost_estimator)),
    )


def cost_gae(
    costs: Any,
    cost_values: Any,
    episode_starts: Any,
    *,
    last_cost_values: Any,
    dones: Any,
    gamma_c: float,
    gae_lambda_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cost advantages and cost value targets by GAE.

    Mirrors the reward-side recursion so the two channels are estimated the
    same way::

        delta_t = c_t + gamma_c * V_C(s_{t+1}) * (1 - done) - V_C(s_t)
        A_t     = delta_t + gamma_c * lambda_c * (1 - done) * A_{t+1}
        target  = A_t + V_C(s_t)

    This replaces a bootstrapped Monte-Carlo cost return, which is the
    ``gae_lambda_c = 1`` special case and gives the cost critic no variance
    reduction at all.

    ``costs`` must already be in the units the critic learns, i.e. passed
    through ``CMDPConfig.scale_costs`` if the objective is normalized. All
    array arguments are ``(n_steps, n_envs)``; ``episode_starts[t]`` flags that
    ``t`` begins a new episode, matching the rollout-buffer convention.
    """

    gamma = float(gamma_c)
    lam = float(gae_lambda_c)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma_c must be finite and lie in [0, 1]")
    if not math.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError("gae_lambda_c must be finite and lie in [0, 1]")

    cost_array = np.asarray(costs, dtype=np.float64)
    value_array = np.asarray(cost_values, dtype=np.float64)
    start_array = np.asarray(episode_starts, dtype=np.float64)
    if cost_array.ndim != 2:
        raise ValueError("costs must be a (n_steps, n_envs) array")
    if value_array.shape != cost_array.shape:
        raise ValueError(
            "cost_values must match costs in shape "
            f"({value_array.shape} != {cost_array.shape})"
        )
    if start_array.shape != cost_array.shape:
        raise ValueError(
            "episode_starts must match costs in shape "
            f"({start_array.shape} != {cost_array.shape})"
        )

    n_steps, n_envs = cost_array.shape
    last_values = np.asarray(last_cost_values, dtype=np.float64).reshape(-1)
    done_flags = np.asarray(dones, dtype=np.float64).reshape(-1)
    if last_values.size != n_envs:
        raise ValueError(
            "last_cost_values must supply one prediction per environment "
            f"({last_values.size} != {n_envs})"
        )
    if done_flags.size != n_envs:
        raise ValueError(
            f"dones must supply one flag per environment ({done_flags.size} != {n_envs})"
        )

    advantages = np.zeros_like(cost_array)
    running = np.zeros(n_envs, dtype=np.float64)
    for step in reversed(range(n_steps)):
        if step == n_steps - 1:
            next_non_terminal = 1.0 - done_flags
            next_values = last_values
        else:
            next_non_terminal = 1.0 - start_array[step + 1]
            next_values = value_array[step + 1]
        delta = (
            cost_array[step]
            + gamma * next_values * next_non_terminal
            - value_array[step]
        )
        running = delta + gamma * lam * next_non_terminal * running
        advantages[step] = running
    targets = advantages + value_array
    return advantages, targets


def cost_objective_estimate(
    method: str,
    *,
    gamma_c: float,
    normalize: bool = True,
    raw_costs: Optional[Any] = None,
    cost_values: Optional[Any] = None,
    episode_starts: Optional[Any] = None,
) -> float:
    """Return the scalar ``J_C_hat`` that drives the dual update.

    ``raw_costs`` must be the **unscaled** per-step cost: ``J_C`` is a weighted
    average of raw costs, so scaling them first would apply ``(1 - gamma_c)``
    twice. ``cost_values`` are critic outputs and are therefore already in the
    critic's units -- per-step if the objective is normalized, undiscounted-sum
    if it is not, in which case the ``(1 - gamma_c)`` factor is applied here.
    """

    name = str(method).strip().lower()
    if name not in SUPPORTED_COST_ESTIMATORS:
        raise ValueError(
            f"Unsupported cost estimator {name!r}; expected one of "
            f"{SUPPORTED_COST_ESTIMATORS}"
        )
    scale = (1.0 - float(gamma_c)) if not normalize else 1.0

    if name == BATCH_MEAN_ESTIMATOR:
        if raw_costs is None:
            raise ValueError("the 'batch_mean' estimator requires raw_costs")
        costs = np.asarray(raw_costs, dtype=np.float64).reshape(-1)
        if costs.size == 0:
            raise ValueError("raw_costs must be non-empty")
        # Already a per-step average; the (1 - gamma_c) weights sum to one.
        return float(costs.mean())

    if cost_values is None:
        raise ValueError(f"the {name!r} estimator requires cost_values")
    values = np.asarray(cost_values, dtype=np.float64)

    if name == VALUE_MEAN_ESTIMATOR:
        if values.size == 0:
            raise ValueError("cost_values must be non-empty")
        return float(values.reshape(-1).mean()) * scale

    if episode_starts is None:
        raise ValueError(
            "the 'episode_start' estimator requires episode_starts"
        )
    starts = np.asarray(episode_starts, dtype=bool)
    if starts.shape != values.shape:
        raise ValueError(
            "episode_starts must match cost_values in shape "
            f"({starts.shape} != {values.shape})"
        )
    selected = values[starts]
    if selected.size == 0:
        raise ValueError(
            "the 'episode_start' estimator found no episode starts in this "
            "batch; use 'batch_mean' for short rollouts"
        )
    return float(selected.mean()) * scale


class DualVariable:
    """The Lagrange multiplier under projected dual ascent.

        lambda <- [ lambda + eta * (J_C_hat - d) ]_+

    Plain ascent, updated once per rollout and held fixed across the epochs of
    that rollout: updating per minibatch makes ``lambda`` chase minibatch noise
    rather than the constraint. A PID controller on the same error is a
    strictly later experiment -- it would add a confound to the first
    measurement of whether the constraint mechanism works at all.

    The projection onto ``lambda >= 0`` is what encodes the inequality: a
    satisfied constraint drives the multiplier to the boundary and switches the
    cost term off, rather than turning it into a bonus for extra safety.
    """

    def __init__(self, config: CMDPConfig) -> None:
        self._config = config
        self._value = float(config.dual_init)
        self._updates = 0

    @property
    def value(self) -> float:
        return self._value

    @property
    def updates(self) -> int:
        return self._updates

    @property
    def penalty_scale(self) -> float:
        """``1 / (1 + lambda)``, or 1 when penalty scaling is disabled."""

        if not self._config.use_penalty_scaling:
            return 1.0
        return 1.0 / (1.0 + self._value)

    def update(self, cost_objective: float) -> dict[str, float]:
        """Apply one dual ascent step and return its diagnostics."""

        estimate = float(cost_objective)
        if not math.isfinite(estimate):
            raise ValueError("cost_objective must be finite")
        violation = estimate - self._config.cost_limit
        previous = self._value
        updated = previous + self._config.dual_lr * violation
        updated = max(updated, 0.0)
        if self._config.dual_max is not None:
            updated = min(updated, self._config.dual_max)
        self._value = float(updated)
        self._updates += 1
        return {
            "cmdp/lambda": self._value,
            "cmdp/lambda_previous": previous,
            "cmdp/cost_objective": estimate,
            "cmdp/cost_limit": float(self._config.cost_limit),
            "cmdp/constraint_violation": float(violation),
            "cmdp/constraint_satisfied": float(violation <= 0.0),
            "cmdp/penalty_scale": self.penalty_scale,
            "cmdp/at_lower_bound": float(self._value == 0.0),
            "cmdp/dual_updates": float(self._updates),
        }

    def state_dict(self) -> dict[str, Any]:
        """Checkpointable state: the multiplier is part of the solution."""

        return {"value": self._value, "updates": self._updates}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        value = float(state["value"])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("a restored lambda must be finite and non-negative")
        self._value = value
        self._updates = int(state.get("updates", 0))
