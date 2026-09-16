"""Contract tests for the learning-rate override and decay schedule.

The load-bearing assertion is the first one: adding these options must leave the
resolved config for a default run byte-identical, because the training signature
hashes that dict and every existing checkpoint's identity depends on it. An
option that silently changed the default hash would orphan every model already
on disk.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import scripts.training.run_ppo_cbf_progression as progression
from scripts.training.run_nominal_ppo_parameter_pilot import PPO_CONFIGS


def _args(**overrides):
    values = {
        "ppo_config": "Q1_stable",
        "n_steps": 1000,
        "batch_size": 100,
        "n_epochs": 10,
        "ent_coef": None,
        "gamma": None,
        "learning_rate": None,
        "lr_schedule": progression.CONSTANT_LR_SCHEDULE,
        "n_envs": 20,
        "use_subproc": None,
        "timesteps": 1_000_000,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestDefaultIsUnchanged:
    def test_default_config_gains_no_key(self) -> None:
        """The historical resolved config must be untouched by these options."""

        config = progression.resolved_ppo_config(_args())
        assert "lr_schedule" not in config
        assert config["learning_rate"] == PPO_CONFIGS["Q1_stable"]["learning_rate"]

    def test_default_hash_is_stable(self) -> None:
        a = progression.resolved_ppo_config(_args())
        b = progression.resolved_ppo_config(_args())
        assert progression.protocol.canonical_config_hash(a) == (
            progression.protocol.canonical_config_hash(b)
        )

    def test_constant_schedule_returns_a_plain_float(self) -> None:
        rate = progression.resolved_learning_rate(progression.resolved_ppo_config(_args()))
        assert isinstance(rate, float)
        assert rate == pytest.approx(1e-4)


class TestCheckpointIdentity:
    def test_schedule_changes_the_hash(self) -> None:
        """A scheduled run must never reuse a constant-rate checkpoint."""

        constant = progression.resolved_ppo_config(_args())
        linear = progression.resolved_ppo_config(_args(lr_schedule="linear"))
        assert progression.protocol.canonical_config_hash(constant) != (
            progression.protocol.canonical_config_hash(linear)
        )

    def test_rate_override_changes_the_hash(self) -> None:
        base = progression.resolved_ppo_config(_args())
        raised = progression.resolved_ppo_config(_args(learning_rate=3e-4))
        assert progression.protocol.canonical_config_hash(base) != (
            progression.protocol.canonical_config_hash(raised)
        )

    def test_override_touches_nothing_else(self) -> None:
        base = progression.resolved_ppo_config(_args())
        raised = progression.resolved_ppo_config(_args(learning_rate=3e-4))
        assert raised["learning_rate"] == pytest.approx(3e-4)
        assert {k: v for k, v in raised.items() if k != "learning_rate"} == {
            k: v for k, v in base.items() if k != "learning_rate"
        }


class TestLinearDecay:
    def test_anneals_from_the_configured_rate_to_zero(self) -> None:
        config = progression.resolved_ppo_config(_args(lr_schedule="linear"))
        schedule = progression.resolved_learning_rate(config)
        assert callable(schedule)
        # SB3 passes progress_remaining: 1.0 at the first update, 0.0 at the last.
        assert schedule(1.0) == pytest.approx(1e-4)
        assert schedule(0.5) == pytest.approx(0.5e-4)
        assert schedule(0.0) == pytest.approx(0.0)

    def test_is_monotone(self) -> None:
        schedule = progression.resolved_learning_rate(
            progression.resolved_ppo_config(_args(lr_schedule="linear"))
        )
        values = [schedule(p) for p in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)]
        assert values == sorted(values, reverse=True)

    def test_decay_follows_the_overridden_rate(self) -> None:
        config = progression.resolved_ppo_config(
            _args(lr_schedule="linear", learning_rate=3e-4)
        )
        schedule = progression.resolved_learning_rate(config)
        assert schedule(1.0) == pytest.approx(3e-4)


class TestValidation:
    @pytest.mark.parametrize("bad", [0.0, -1e-4, float("nan"), float("inf")])
    def test_non_positive_or_non_finite_rate_is_rejected(self, bad: float) -> None:
        """A frozen policy trains for an hour and looks like a legitimate run."""

        with pytest.raises(ValueError):
            progression.resolved_ppo_config(_args(learning_rate=bad))

    def test_unknown_schedule_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            progression.resolved_learning_rate({"learning_rate": 1e-4, "lr_schedule": "cosine"})
