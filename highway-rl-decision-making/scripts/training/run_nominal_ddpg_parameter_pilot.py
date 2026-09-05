"""Short nominal-DDPG core-parameter pilot.

The safety environment and tuned CBF implementation are deliberately not tuned
here.  Screening trains only the unfiltered nominal policy.  Each candidate is
evaluated on the same seeded, fixed-timestep scenarios every checkpoint, and
selection uses the final three checkpoints rather than a best single point.

Stages
------
screen
    P0--P3, 50k timesteps, one common paired training seed.
confirm
    The best two screen configurations, 150k timesteps, three training seeds.
summarize
    Rebuild summaries/ranking from an already completed output directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import socket
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import update_learning_rate
from stable_baselines3.common.vec_env import VecNormalize

import scripts.training.run_cbf_filter_ablation as pipeline
from scripts.common.laneless_script_config import (
    active_traffic_model,
    add_env_config_args,
    env_config_from_args,
)
from scripts.training.train_safety_potential_variants import MTM_CONGESTED_UNCERTAIN_UPDATES, deep_update


PILOT_SCHEMA_VERSION = 2
DEFAULT_SCREEN_SEED = 307
DEFAULT_CONFIRM_SEEDS = (307, 1307, 2307)
DEFAULT_CONFIRM_CONFIGS = ("P0_current", "P2_more_exploration")
DEFAULT_EVAL_SEED_START = 900_000
DEFAULT_EVAL_SCENARIOS = 10
MIN_CRITIC_CALIBRATION_EXACT_COVERAGE = 0.20


class OutputDirectoryRunLock:
    """Hold an OS-level lock for one pilot output directory.

    Lock files live in the user's temporary directory, keyed by the resolved
    output path.  This also works for outputs directly under a Windows drive
    root, where creating a sibling file may be forbidden.  The OS releases the
    byte-range lock if the process exits or is killed, so a stale file never
    blocks a later run.
    """

    def __init__(self, output_dir: Path):
        resolved = output_dir.resolve()
        identity = hashlib.sha256(os.path.normcase(str(resolved)).encode("utf-8")).hexdigest()[:16]
        lock_root = Path(tempfile.gettempdir()) / "nominal_ddpg_pilot_locks"
        self.path = lock_root / f"{identity}.run.lock"
        self._handle: Any = None

    def acquire(self) -> "OutputDirectoryRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            owner = self._owner_description()
            raise RuntimeError(
                f"Another pilot invocation already holds the output lock for {self.path}: {owner}"
            ) from exc

        self._handle = handle
        metadata = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "acquired_unix_s": time.time(),
            },
            sort_keys=True,
        ).encode("utf-8")
        handle.seek(1)
        handle.truncate()
        handle.write(metadata)
        handle.flush()
        return self

    def _owner_description(self) -> str:
        try:
            with self.path.open("rb") as handle:
                handle.seek(1)
                metadata = json.loads(handle.read().decode("utf-8"))
            return f"pid={metadata.get('pid')} host={metadata.get('host')}"
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return "owner metadata unavailable"

    def release(self) -> None:
        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> "OutputDirectoryRunLock":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()

PILOT_CONFIGS: dict[str, dict[str, float | int | str]] = {
    "P0_current": {
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "batch_size": 64,
        "gamma": 0.98,
        "tau": 0.001,
        "learning_starts": 1_000,
        "buffer_size": 100_000,
        "ou_sigma": 0.1,
    },
    "P1_stable": {
        "actor_lr": 1e-4,
        "critic_lr": 1e-3,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "learning_starts": 5_000,
        "buffer_size": 500_000,
        "ou_sigma": 0.1,
    },
    "P2_more_exploration": {
        "actor_lr": 1e-4,
        "critic_lr": 1e-3,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "learning_starts": 5_000,
        "buffer_size": 500_000,
        "ou_sigma": 0.2,
    },
    "P3_slower_critic": {
        "actor_lr": 1e-4,
        "critic_lr": 3e-4,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "learning_starts": 5_000,
        "buffer_size": 500_000,
        "ou_sigma": 0.1,
    },
}


def fixed_cbf_snapshot(namespace: dict[str, Any]) -> dict[str, Any]:
    """Record the notebook's tuned shield without activating it in this pilot."""

    max_constraints = namespace.get("CBF_MAX_NEIGHBOR_CONSTRAINTS")
    return {
        "active_in_nominal_pilot": False,
        "k0": float(namespace["CBF_K0"]),
        "k1": float(namespace["CBF_K1"]),
        "eps_side": float(namespace["CBF_EPS_SIDE"]),
        "correction_reward_lambda": float(namespace["CBF_FILTER_REWARD_LAMBDA"]),
        "ax_bounds": [float(value) for value in namespace["CBF_AX_BOUNDS"]],
        "ay_bounds": [float(value) for value in namespace["CBF_AY_BOUNDS"]],
        "solver": str(namespace.get("CBF_QP_SOLVER", "osqp")),
        "max_neighbor_constraints": (
            None if max_constraints is None else int(max_constraints)
        ),
        "qp_feasibility_tolerance": float(
            namespace.get("CBF_QP_FEASIBILITY_TOL", 1e-3)
        ),
    }


class SplitLearningRateDDPG(DDPG):
    """DDPG with independent constant actor and critic Adam learning rates."""

    def __init__(
        self,
        *args,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        **kwargs,
    ) -> None:
        self.actor_learning_rate = float(actor_learning_rate)
        self.critic_learning_rate = float(critic_learning_rate)
        kwargs["learning_rate"] = float(self.critic_learning_rate)
        super().__init__(*args, **kwargs)
        if hasattr(self, "actor") and hasattr(self, "critic"):
            self._set_split_learning_rates()

    def _set_split_learning_rates(self) -> None:
        update_learning_rate(self.actor.optimizer, float(self.actor_learning_rate))
        update_learning_rate(self.critic.optimizer, float(self.critic_learning_rate))

    def _update_learning_rate(self, optimizers) -> None:  # noqa: ARG002
        self._set_split_learning_rates()
        self.logger.record("train/actor_learning_rate", float(self.actor_learning_rate))
        self.logger.record("train/critic_learning_rate", float(self.critic_learning_rate))


def _append_frame(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)


def _finite_mean(series: pd.Series, default: float = np.nan) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float(default)


def critic_diagnostics(model: SplitLearningRateDDPG, batch_size: int, sample_seed: int) -> dict[str, float]:
    """Measure Bellman error and Q magnitude without perturbing training RNG state."""

    replay_size = int(model.replay_buffer.size())
    if replay_size <= 0:
        return {
            "diagnostic_batch_size": 0.0,
            "critic_mse": np.nan,
            "td_abs_mean": np.nan,
            "td_abs_p95": np.nan,
            "td_abs_max": np.nan,
            "q_mean": np.nan,
            "q_abs_mean": np.nan,
            "q_abs_p95": np.nan,
            "q_abs_max": np.nan,
            "target_q_abs_mean": np.nan,
            "reward_abs_max": np.nan,
            "q_scale_reference": np.nan,
            "q_scale_ratio": np.nan,
            "q_scale_excess_log10": np.nan,
            "q_target_magnitude_log_gap": np.nan,
            "q_nonfinite_rate": np.nan,
        }

    sample_size = min(int(batch_size), replay_size)
    np.random.seed(int(sample_seed) % (2**32 - 1))
    replay_data = model.replay_buffer.sample(sample_size, env=model._vec_normalize_env)
    with th.no_grad():
        next_actions = model.actor_target(replay_data.next_observations).clamp(-1.0, 1.0)
        next_q_values = th.cat(model.critic_target(replay_data.next_observations, next_actions), dim=1)
        next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
        discounts = getattr(replay_data, "discounts", None)
        if discounts is None:
            discounts = model.gamma
        target_q = replay_data.rewards + (1.0 - replay_data.dones) * discounts * next_q_values
        current_q_values = model.critic(replay_data.observations, replay_data.actions)
        current_q = current_q_values[0]
        td_error = target_q - current_q
        critic_mse = sum(F.mse_loss(value, target_q) for value in current_q_values)

    q = current_q.detach().cpu().numpy().reshape(-1)
    target = target_q.detach().cpu().numpy().reshape(-1)
    td = td_error.detach().cpu().numpy().reshape(-1)
    reward = replay_data.rewards.detach().cpu().numpy().reshape(-1)
    finite_q = q[np.isfinite(q)]
    finite_td = np.abs(td[np.isfinite(td)])
    finite_target = np.abs(target[np.isfinite(target)])
    finite_reward = np.abs(reward[np.isfinite(reward)])
    q_abs_mean = float(np.mean(np.abs(finite_q))) if finite_q.size else np.nan
    q_abs_p95 = float(np.percentile(np.abs(finite_q), 95)) if finite_q.size else np.nan
    q_abs_max = float(np.max(np.abs(finite_q))) if finite_q.size else np.nan
    target_q_abs_mean = float(np.mean(finite_target)) if finite_target.size else np.nan
    reward_abs_max = float(np.max(finite_reward)) if finite_reward.size else np.nan
    q_scale_reference = (
        reward_abs_max / max(1.0 - float(model.gamma), 1e-6)
        if np.isfinite(reward_abs_max) and reward_abs_max > 0.0
        else np.nan
    )
    q_scale_ratio = (
        q_abs_max / q_scale_reference
        if np.isfinite(q_abs_max) and np.isfinite(q_scale_reference) and q_scale_reference > 0.0
        else np.nan
    )
    q_scale_excess_log10 = (
        float(max(np.log10(max(q_scale_ratio, 1.0)), 0.0))
        if np.isfinite(q_scale_ratio)
        else np.nan
    )
    q_target_magnitude_log_gap = (
        float(abs(np.log((q_abs_mean + 1e-8) / (target_q_abs_mean + 1e-8))))
        if np.isfinite(q_abs_mean) and np.isfinite(target_q_abs_mean)
        else np.nan
    )
    return {
        "diagnostic_batch_size": float(sample_size),
        "critic_mse": float(critic_mse.item()),
        "td_abs_mean": float(np.mean(finite_td)) if finite_td.size else np.nan,
        "td_abs_p95": float(np.percentile(finite_td, 95)) if finite_td.size else np.nan,
        "td_abs_max": float(np.max(finite_td)) if finite_td.size else np.nan,
        "q_mean": float(np.mean(finite_q)) if finite_q.size else np.nan,
        "q_abs_mean": q_abs_mean,
        "q_abs_p95": q_abs_p95,
        "q_abs_max": q_abs_max,
        "target_q_abs_mean": target_q_abs_mean,
        "reward_abs_max": reward_abs_max,
        "q_scale_reference": q_scale_reference,
        "q_scale_ratio": q_scale_ratio,
        "q_scale_excess_log10": q_scale_excess_log10,
        "q_target_magnitude_log_gap": q_target_magnitude_log_gap,
        "q_nonfinite_rate": float(1.0 - finite_q.size / max(q.size, 1)),
    }


CALIBRATION_SCENARIO_MEAN_METRICS = (
    "critic_calibration_q_mean",
    "critic_calibration_empirical_return_mean",
    "critic_calibration_bias_mean",
    "critic_calibration_mae",
    "critic_calibration_rmse",
    "critic_calibration_normalized_bias",
    "critic_calibration_normalized_mae",
    "critic_calibration_normalized_rmse",
    "critic_calibration_overestimation_rate",
    "critic_calibration_pearson_r",
    "critic_calibration_empirical_on_q_slope",
    "critic_calibration_empirical_on_q_intercept",
    "critic_calibration_quantile_ece",
    "critic_calibration_normalized_quantile_ece",
    "critic_calibration_bootstrap_mae",
    "critic_calibration_censored_gamma_tail_mean",
    "critic_calibration_censored_gamma_tail_max",
)


def summarize_critic_calibration_samples(samples: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize one evaluation scenario's terminal-MC calibration anchors."""

    frame = pd.DataFrame(samples)
    total = int(len(frame))
    if total == 0:
        result = {
            "critic_calibration_anchor_count": 0.0,
            "critic_calibration_exact_anchor_count": 0.0,
            "critic_calibration_censored_anchor_count": 0.0,
            "critic_calibration_finite_exact_anchor_count": 0.0,
            "critic_calibration_exact_coverage": np.nan,
            "critic_calibration_q_nonfinite_rate": np.nan,
        }
        result.update({metric: np.nan for metric in CALIBRATION_SCENARIO_MEAN_METRICS})
        return result

    exact_mask = pd.to_numeric(frame["terminal_mc_included"], errors="coerce").fillna(0).eq(1)
    censored_mask = ~exact_mask
    q_all = pd.to_numeric(frame["q_value"], errors="coerce").to_numpy(dtype=float)
    q_nonfinite_rate = float(np.mean(~np.isfinite(q_all)))
    exact = frame.loc[exact_mask].copy()
    q = pd.to_numeric(exact.get("q_value"), errors="coerce").to_numpy(dtype=float)
    empirical = pd.to_numeric(
        exact.get("empirical_discounted_return"), errors="coerce"
    ).to_numpy(dtype=float)
    finite = np.isfinite(q) & np.isfinite(empirical)
    q = q[finite]
    empirical = empirical[finite]
    count = int(q.size)

    result: dict[str, float] = {
        "critic_calibration_anchor_count": float(total),
        "critic_calibration_exact_anchor_count": float(exact_mask.sum()),
        "critic_calibration_censored_anchor_count": float(censored_mask.sum()),
        "critic_calibration_finite_exact_anchor_count": float(count),
        "critic_calibration_exact_coverage": float(exact_mask.sum() / total),
        "critic_calibration_q_nonfinite_rate": q_nonfinite_rate,
    }
    if count == 0:
        result.update({metric: np.nan for metric in CALIBRATION_SCENARIO_MEAN_METRICS})
    else:
        error = q - empirical
        empirical_abs_scale = max(float(np.mean(np.abs(empirical))), 1e-8)
        empirical_rms_scale = max(float(np.sqrt(np.mean(np.square(empirical)))), 1e-8)
        if count >= 2 and float(np.std(q)) > 1e-12 and float(np.std(empirical)) > 1e-12:
            pearson = float(np.corrcoef(q, empirical)[0, 1])
        else:
            pearson = np.nan
        q_variance = float(np.var(q))
        if count >= 2 and q_variance > 1e-12:
            slope = float(np.mean((q - np.mean(q)) * (empirical - np.mean(empirical))) / q_variance)
            intercept = float(np.mean(empirical) - slope * np.mean(q))
        else:
            slope = np.nan
            intercept = np.nan

        quantile_count = min(5, count)
        if quantile_count > 0:
            order = pd.Series(q).rank(method="first")
            bins = pd.qcut(order, q=quantile_count, labels=False, duplicates="drop")
            bin_frame = pd.DataFrame({"q": q, "empirical": empirical, "bin": bins})
            bin_summary = bin_frame.groupby("bin", observed=True).agg(
                q_mean=("q", "mean"),
                empirical_mean=("empirical", "mean"),
                count=("q", "size"),
            )
            quantile_ece = float(
                np.sum(
                    np.abs(bin_summary["q_mean"] - bin_summary["empirical_mean"])
                    * bin_summary["count"]
                )
                / count
            )
        else:
            quantile_ece = np.nan
        result.update(
            {
                "critic_calibration_q_mean": float(np.mean(q)),
                "critic_calibration_empirical_return_mean": float(np.mean(empirical)),
                "critic_calibration_bias_mean": float(np.mean(error)),
                "critic_calibration_mae": float(np.mean(np.abs(error))),
                "critic_calibration_rmse": float(np.sqrt(np.mean(np.square(error)))),
                "critic_calibration_normalized_bias": float(np.mean(error) / empirical_abs_scale),
                "critic_calibration_normalized_mae": float(
                    np.mean(np.abs(error)) / empirical_abs_scale
                ),
                "critic_calibration_normalized_rmse": float(
                    np.sqrt(np.mean(np.square(error))) / empirical_rms_scale
                ),
                "critic_calibration_overestimation_rate": float(np.mean(error > 0.0)),
                "critic_calibration_pearson_r": pearson,
                "critic_calibration_empirical_on_q_slope": slope,
                "critic_calibration_empirical_on_q_intercept": intercept,
                "critic_calibration_quantile_ece": quantile_ece,
                "critic_calibration_normalized_quantile_ece": float(
                    quantile_ece / empirical_abs_scale
                ),
            }
        )

    boot_q = pd.to_numeric(frame.get("q_value"), errors="coerce").to_numpy(dtype=float)
    boot_target = pd.to_numeric(
        frame.get("bootstrapped_discounted_return"), errors="coerce"
    ).to_numpy(dtype=float)
    finite_boot = np.isfinite(boot_q) & np.isfinite(boot_target)
    result["critic_calibration_bootstrap_mae"] = (
        float(np.mean(np.abs(boot_q[finite_boot] - boot_target[finite_boot])))
        if finite_boot.any()
        else np.nan
    )
    censored_gamma = pd.to_numeric(
        frame.loc[censored_mask, "gamma_tail"], errors="coerce"
    ).to_numpy(dtype=float)
    censored_gamma = censored_gamma[np.isfinite(censored_gamma)]
    result["critic_calibration_censored_gamma_tail_mean"] = (
        float(np.mean(censored_gamma)) if censored_gamma.size else np.nan
    )
    result["critic_calibration_censored_gamma_tail_max"] = (
        float(np.max(censored_gamma)) if censored_gamma.size else np.nan
    )
    return result


def aggregate_calibration_scenario_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Give each fixed evaluation scenario equal weight within a checkpoint."""

    frame = pd.DataFrame(rows)
    count_columns = (
        "critic_calibration_anchor_count",
        "critic_calibration_exact_anchor_count",
        "critic_calibration_censored_anchor_count",
        "critic_calibration_finite_exact_anchor_count",
    )
    result = {
        metric: float(pd.to_numeric(frame[metric], errors="coerce").fillna(0.0).sum())
        for metric in count_columns
    }
    total = result["critic_calibration_anchor_count"]
    exact = result["critic_calibration_exact_anchor_count"]
    result["critic_calibration_exact_coverage"] = exact / total if total > 0 else np.nan
    result["critic_calibration_q_nonfinite_rate"] = _finite_mean(
        frame["critic_calibration_q_nonfinite_rate"]
    )
    for metric in CALIBRATION_SCENARIO_MEAN_METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        result[metric] = (
            float(np.max(values))
            if metric == "critic_calibration_censored_gamma_tail_max" and values.size
            else (float(np.mean(values)) if values.size else np.nan)
        )
    return result


def build_critic_calibration_bins(samples: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Build a checkpoint-level Q-quantile calibration curve for plotting."""

    output: list[dict[str, Any]] = []
    if samples.empty:
        return pd.DataFrame()
    keys = ["pilot_config", "training_seed", "model_timestep"]
    for group_key, group in samples.groupby(keys, sort=True):
        exact = group[pd.to_numeric(group["terminal_mc_included"], errors="coerce").eq(1)].copy()
        exact["q_value"] = pd.to_numeric(exact["q_value"], errors="coerce")
        exact["empirical_discounted_return"] = pd.to_numeric(
            exact["empirical_discounted_return"], errors="coerce"
        )
        exact = exact[np.isfinite(exact["q_value"]) & np.isfinite(exact["empirical_discounted_return"])]
        if exact.empty:
            continue
        quantile_count = min(int(bins), len(exact))
        exact["q_bin"] = pd.qcut(
            exact["q_value"].rank(method="first"),
            q=quantile_count,
            labels=False,
            duplicates="drop",
        )
        for q_bin, bin_group in exact.groupby("q_bin", sort=True, observed=True):
            output.append(
                {
                    "pilot_config": str(group_key[0]),
                    "training_seed": int(group_key[1]),
                    "model_timestep": int(group_key[2]),
                    "q_bin": int(q_bin),
                    "anchor_count": int(len(bin_group)),
                    "q_mean": float(bin_group["q_value"].mean()),
                    "empirical_return_mean": float(
                        bin_group["empirical_discounted_return"].mean()
                    ),
                    "bias_mean": float(
                        (bin_group["q_value"] - bin_group["empirical_discounted_return"]).mean()
                    ),
                }
            )
    return pd.DataFrame(output)


def aggregate_checkpoint_scenarios(rows: list[dict[str, Any]]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    timesteps = float(pd.to_numeric(frame["timesteps"], errors="coerce").sum())
    total_return = float(pd.to_numeric(frame["total_return"], errors="coerce").sum())
    distance_m = float(pd.to_numeric(frame["total_distance_m"], errors="coerce").sum())
    collisions = float(
        pd.to_numeric(frame["distinct_ego_collision_events"], errors="coerce").sum()
    )
    episode_length_sum = float(pd.to_numeric(frame["episode_length_sum"], errors="coerce").sum())
    episode_segments = float(pd.to_numeric(frame["episode_segments"], errors="coerce").sum())
    return {
        "return_per_timestep": pipeline._ratio(total_return, timesteps),
        "total_distance_m": distance_m,
        "distinct_ego_collision_events": collisions,
        "distance_per_collision_m": pipeline._distance_per_collision(distance_m, collisions),
        "distance_per_collision_exposure_bound_m": pipeline._distance_per_collision_exposure_bound(
            distance_m, collisions
        ),
        "ego_collisions_per_km": pipeline._collisions_per_km(collisions, distance_m),
        "mean_abs_speed_error": _finite_mean(frame["mean_abs_speed_error"]),
        "episode_length_mean": pipeline._ratio(episode_length_sum, episode_segments),
        "nominal_action_saturation_rate": _finite_mean(frame["nominal_action_saturation_rate"]),
    }


def sb3_resume_learn_target_timesteps(target_timesteps: int, current_timesteps: int) -> tuple[int, int]:
    """Return (remaining, learn_total_timesteps) for reset_num_timesteps=False."""
    remaining = int(target_timesteps) - int(current_timesteps)
    if remaining < 0:
        raise RuntimeError(f"Checkpoint timestep {current_timesteps} exceeds target {target_timesteps}")
    return remaining, remaining


class PilotEvaluationCallback(BaseCallback):
    """Evaluate at coherent post-update checkpoint boundaries with fixed seeds."""

    def __init__(
        self,
        *,
        namespace: dict[str, Any],
        pilot_config: str,
        training_seed: int,
        config_values: dict[str, Any],
        args: argparse.Namespace,
        env_config: dict[str, Any],
        reward_config: dict[str, float],
        scenario_path: Path,
        diagnostics_path: Path,
        calibration_path: Path,
        interval: int,
        target_timesteps: int,
    ) -> None:
        super().__init__(verbose=0)
        self.namespace = namespace
        self.pilot_config = str(pilot_config)
        self.training_seed = int(training_seed)
        self.config_values = dict(config_values)
        self.args = args
        self.env_config = copy.deepcopy(env_config)
        self.reward_config = copy.deepcopy(reward_config)
        self.scenario_path = Path(scenario_path)
        self.diagnostics_path = Path(diagnostics_path)
        self.calibration_path = Path(calibration_path)
        self.interval = int(interval)
        self.target_timesteps = int(target_timesteps)
        self.pending = False
        self.evaluated_steps: set[int] = set()
        if self.scenario_path.exists() and self.scenario_path.stat().st_size:
            existing = pd.read_csv(self.scenario_path, usecols=["model_timestep"])
            self.evaluated_steps = {
                int(value) for value in pd.to_numeric(existing["model_timestep"], errors="coerce").dropna()
            }
        self.next_eval_step = self.interval

    def _on_training_start(self) -> None:
        current = int(self.model.num_timesteps)
        self.next_eval_step = ((current // self.interval) + 1) * self.interval

    def _on_step(self) -> bool:
        if int(self.num_timesteps) >= int(self.next_eval_step):
            self.pending = True
        return True

    def _on_rollout_start(self) -> None:
        if self.pending:
            self._evaluate_current_checkpoint()

    def _on_training_end(self) -> None:
        step = int(self.model.num_timesteps)
        if step % self.interval == 0 and step not in self.evaluated_steps:
            self._evaluate_current_checkpoint()

    def _evaluate_current_checkpoint(self) -> None:
        step = int(self.model.num_timesteps)
        if step in self.evaluated_steps:
            self.pending = False
            self.next_eval_step = ((step // self.interval) + 1) * self.interval
            return
        if step % self.interval != 0:
            raise RuntimeError(f"Evaluation checkpoint is not aligned to interval: {step}")

        rng_state = pipeline.capture_rng_state(self.model)
        rows: list[dict[str, Any]] = []
        checkpoint_calibration_samples: list[dict[str, Any]] = []
        try:
            for scenario_seed in self.args.eval_seeds:
                scenario_calibration_samples: Optional[list[dict[str, Any]]] = (
                    [] if bool(self.args.critic_calibration) else None
                )
                row, _ = pipeline.evaluate_scenario(
                    self.namespace,
                    model=self.model,
                    variant=self.pilot_config,
                    mode="raw",
                    scenario_seed=int(scenario_seed),
                    training_seed=self.training_seed,
                    env_config=self.env_config,
                    reward_config=self.reward_config,
                    args=self.args,
                    critic_calibration_samples=scenario_calibration_samples,
                    critic_calibration_stride=int(self.args.critic_calibration_stride),
                )
                if scenario_calibration_samples is not None:
                    row.update(summarize_critic_calibration_samples(scenario_calibration_samples))
                    for sample in scenario_calibration_samples:
                        sample.update(
                            {
                                "pilot_config": self.pilot_config,
                                "model_timestep": step,
                                **self.config_values,
                            }
                        )
                    checkpoint_calibration_samples.extend(scenario_calibration_samples)
                row.update(
                    {
                        "pilot_config": self.pilot_config,
                        "model_timestep": step,
                        **self.config_values,
                    }
                )
                rows.append(row)

            diagnostics = critic_diagnostics(
                self.model,
                batch_size=int(self.args.diagnostic_batch_size),
                sample_seed=self.training_seed * 1_000_003 + step,
            )
            if bool(self.args.critic_calibration):
                diagnostics.update(aggregate_calibration_scenario_rows(rows))
            diagnostics.update(
                {
                    "pilot_config": self.pilot_config,
                    "variant": self.pilot_config,
                    "training_seed": self.training_seed,
                    "model_timestep": step,
                    "replay_size": int(self.model.replay_buffer.size()),
                    "n_updates": int(self.model._n_updates),
                    "latest_train_actor_loss": pipeline._as_float(
                        self.model.logger.name_to_value.get("train/actor_loss")
                    ),
                    "latest_train_critic_loss": pipeline._as_float(
                        self.model.logger.name_to_value.get("train/critic_loss")
                    ),
                    **self.config_values,
                }
            )
            _append_frame(self.scenario_path, pd.DataFrame(rows))
            _append_frame(self.diagnostics_path, pd.DataFrame([diagnostics]))
            if checkpoint_calibration_samples:
                _append_frame(
                    self.calibration_path,
                    pd.DataFrame(checkpoint_calibration_samples),
                )

            checkpoint_metrics = aggregate_checkpoint_scenarios(rows)
            for name, value in checkpoint_metrics.items():
                self.logger.record(f"eval/{name}", value)
            for name in (
                "critic_mse",
                "td_abs_mean",
                "td_abs_p95",
                "q_abs_mean",
                "q_abs_p95",
                "q_abs_max",
                "q_scale_ratio",
                "q_scale_excess_log10",
                "q_target_magnitude_log_gap",
                "q_nonfinite_rate",
            ):
                self.logger.record(f"diagnostics/{name}", diagnostics[name])
            if bool(self.args.critic_calibration):
                for name in (
                    "critic_calibration_exact_coverage",
                    "critic_calibration_bias_mean",
                    "critic_calibration_mae",
                    "critic_calibration_rmse",
                    "critic_calibration_normalized_bias",
                    "critic_calibration_normalized_mae",
                    "critic_calibration_overestimation_rate",
                    "critic_calibration_pearson_r",
                ):
                    self.logger.record(f"calibration/{name}", diagnostics[name])
            self.logger.dump(step=step)
        finally:
            pipeline.restore_rng_state(self.model, rng_state)

        self.evaluated_steps.add(step)
        self.pending = False
        self.next_eval_step = ((step // self.interval) + 1) * self.interval
        print(
            f"[nominal-pilot] evaluated {self.pilot_config} seed={self.training_seed} step={step:,}",
            flush=True,
        )


class RetainedStrictCheckpointCallback(pipeline.StrictCheckpointCallback):
    """Keep all model snapshots but only recent full replay-resume bundles."""

    def __init__(self, *, strict_retention: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.strict_retention = max(int(strict_retention), 1)

    def _save_bundle(self) -> None:
        super()._save_bundle()
        step = int(self.last_saved_step)
        bundle = self.variant_dir / "ckpt" / f"{step:09d}"
        snapshot_dir = self.variant_dir / "model_checkpoints"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_dir / f"{step:09d}.zip"
        source = bundle / pipeline.CHECKPOINT_PAYLOADS["model"]
        if snapshot.exists():
            if pipeline.file_sha256(snapshot) != pipeline.file_sha256(source):
                raise RuntimeError(f"Model checkpoint collision at {snapshot}")
        else:
            shutil.copy2(source, snapshot)
        metadata_path = snapshot_dir / f"{step:09d}.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": PILOT_SCHEMA_VERSION,
                    "timestep": step,
                    "model_sha256": pipeline.file_sha256(snapshot),
                    "strict_bundle_at_save": str(bundle.relative_to(self.variant_dir)),
                    "strict_bundle_may_be_pruned_by_retention": True,
                    "training_config_hash": self.training_config_hash,
                    "contains_ou_state": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        bundles = sorted(
            path for path in (self.variant_dir / "ckpt").iterdir() if path.is_dir() and path.name.isdigit()
        )
        for old_bundle in bundles[: max(len(bundles) - self.strict_retention, 0)]:
            shutil.rmtree(old_bundle)


def pilot_config_payload(
    *,
    project_root: Path,
    pilot_config: str,
    config_values: dict[str, Any],
    training_seed: int,
    target_timesteps: int,
    args: argparse.Namespace,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
) -> dict[str, Any]:
    source_paths = {
        "pilot_runner": Path(__file__).resolve(),
        "pipeline_runner": project_root / "scripts" / "training" / "run_cbf_filter_ablation.py",
        "script_config": project_root / "scripts" / "common" / "laneless_script_config.py",
        "mtm_training_config": project_root / "scripts" / "training" / "train_safety_potential_variants.py",
        "notebook": project_root / "notebooks" / "lanelessKaralakou.ipynb",
        "environment": project_root / "laneless highway env" / "lane_free_env.py",
    }
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "study": "nominal_ddpg_core_parameter_pilot",
        "stage": str(args.stage),
        "pilot_config": pilot_config,
        "training_seed": int(training_seed),
        "target_timesteps": int(target_timesteps),
        "parameters": config_values,
        "fixed_training": {
            "algorithm": "SplitLearningRateDDPG",
            "filtered_training": False,
            "critic_semantics": "Q_nominal(s, a_RL); CBF is absent during this pilot",
            "shielded_critic_semantics_preserved_for_later_ablation": (
                "Q_shielded(s, a_RL): replay stores the nominal command while the transition "
                "is generated by the environment executing F(s, a_RL)"
            ),
            "train_freq": [1, "step"],
            "gradient_steps": 1,
            "policy_net_arch": [256, 128],
            "ou_theta": 0.15,
            "ou_dt": 0.01,
            "terminate_on_collision": True,
            "episode_reset_reseed": False,
        },
        "evaluation": {
            "mode": "raw",
            "scenario_seeds": [int(seed) for seed in args.eval_seeds],
            "timestep_budget": int(args.eval_timesteps),
            "checkpoint_interval": int(args.checkpoint_interval),
            "deterministic": True,
            "terminate_on_collision": True,
            "reset_immediately_after_collision": True,
            "critic_calibration": {
                "enabled": bool(args.critic_calibration),
                "anchor_stride_within_episode": int(args.critic_calibration_stride),
                "action_semantics": "deterministic actor command in SB3 replay-buffer scale",
                "primary_target": "discounted terminal Monte Carlo return",
                "gamma": float(config_values["gamma"]),
                "censoring": (
                    "environment truncations and evaluation-budget tails are excluded from "
                    "primary calibration"
                ),
                "secondary_target": "critic-bootstrapped sensitivity return for censored tails",
                "scenario_weighting": "equal within each training seed and checkpoint",
                "low_exact_coverage_warning_threshold": MIN_CRITIC_CALIBRATION_EXACT_COVERAGE,
            },
        },
        "env_config": env_config,
        "reward_config": reward_config,
        "fixed_cbf_snapshot": dict(args.cbf_snapshot),
        "device": str(args.device),
        "strict_checkpoint_retention": int(args.strict_checkpoint_retention),
        "diagnostic_batch_size": int(args.diagnostic_batch_size),
        "runtime_versions": pipeline._package_versions(),
        "source_hashes": {name: pipeline.file_sha256(path) for name, path in source_paths.items()},
    }


def _run_dir(output_dir: Path, training_seed: int, pilot_config: str) -> Path:
    return output_dir / f"seed_{int(training_seed)}" / pilot_config


def preflight_runs(
    *,
    output_dir: Path,
    run_specs: list[tuple[int, str]],
    args: argparse.Namespace,
    project_root: Path,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    target_timesteps: int,
) -> None:
    if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to mix a fresh pilot with existing artifacts: {output_dir}")
    for training_seed, pilot_config in run_specs:
        run_dir = _run_dir(output_dir, training_seed, pilot_config)
        if not args.resume:
            continue
        pointer = run_dir / "latest_checkpoint.json"
        if not pointer.exists():
            if run_dir.exists() and any(run_dir.iterdir()):
                raise RuntimeError(f"Incomplete run has no strict checkpoint: {run_dir}")
            continue
        payload = pilot_config_payload(
            project_root=project_root,
            pilot_config=pilot_config,
            config_values=PILOT_CONFIGS[pilot_config],
            training_seed=training_seed,
            target_timesteps=target_timesteps,
            args=args,
            env_config=env_config,
            reward_config=reward_config,
        )
        bundle = pipeline._latest_checkpoint_bundle(run_dir)
        pipeline.validate_checkpoint_bundle(
            bundle,
            pipeline.canonical_config_hash(payload),
            expected_model_class=pipeline._class_qualified_name(SplitLearningRateDDPG),
        )


def build_pilot_model(
    *,
    train_env: Any,
    config_values: dict[str, Any],
    training_seed: int,
    device: str,
    tensorboard_log: Path,
) -> SplitLearningRateDDPG:
    n_actions = int(train_env.action_space.shape[-1])
    noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions, dtype=np.float32),
        sigma=float(config_values["ou_sigma"]) * np.ones(n_actions, dtype=np.float32),
        theta=0.15,
        dt=0.01,
    )
    return SplitLearningRateDDPG(
        "MlpPolicy",
        train_env,
        actor_learning_rate=float(config_values["actor_lr"]),
        critic_learning_rate=float(config_values["critic_lr"]),
        buffer_size=int(config_values["buffer_size"]),
        learning_starts=int(config_values["learning_starts"]),
        batch_size=int(config_values["batch_size"]),
        tau=float(config_values["tau"]),
        gamma=float(config_values["gamma"]),
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=noise,
        policy_kwargs={"net_arch": [256, 128]},
        tensorboard_log=str(tensorboard_log),
        verbose=0,
        seed=int(training_seed),
        device=device,
    )


def train_one_run(
    namespace: dict[str, Any],
    *,
    pilot_config: str,
    training_seed: int,
    target_timesteps: int,
    args: argparse.Namespace,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    output_dir: Path,
) -> dict[str, Any]:
    config_values = dict(PILOT_CONFIGS[pilot_config])
    run_dir = _run_dir(output_dir, training_seed, pilot_config)
    run_dir.mkdir(parents=True, exist_ok=True)
    monitor_path = run_dir / "training.monitor.csv"
    training_metrics_path = run_dir / "training_episodes.csv"
    scenario_path = run_dir / "evaluation_scenarios.csv"
    diagnostics_path = run_dir / "checkpoint_diagnostics.csv"
    calibration_path = run_dir / "critic_calibration_samples.csv"
    tracked_logs = [
        monitor_path,
        training_metrics_path,
        scenario_path,
        diagnostics_path,
        calibration_path,
    ]
    project_root = Path(namespace["PROJECT_ROOT"])
    payload = pilot_config_payload(
        project_root=project_root,
        pilot_config=pilot_config,
        config_values=config_values,
        training_seed=training_seed,
        target_timesteps=target_timesteps,
        args=args,
        env_config=env_config,
        reward_config=reward_config,
    )
    config_hash = pipeline.canonical_config_hash(payload)
    pointer = run_dir / "latest_checkpoint.json"
    resume_this_run = bool(args.resume and pointer.exists())
    resume_bundle: Optional[Path] = None
    resume_manifest: Optional[dict[str, Any]] = None
    resume_pipeline_state: Optional[dict[str, Any]] = None

    if resume_this_run:
        resume_bundle = pipeline._latest_checkpoint_bundle(run_dir)
        resume_manifest, resume_pipeline_state = pipeline.validate_checkpoint_bundle(
            resume_bundle,
            config_hash,
            expected_model_class=pipeline._class_qualified_name(SplitLearningRateDDPG),
        )
        expected_logs = {str(path.relative_to(run_dir)): path for path in tracked_logs}
        saved_offsets = resume_pipeline_state.get("log_offsets", {})
        if set(saved_offsets) != set(expected_logs):
            raise RuntimeError(
                f"Strict-resume log set mismatch: saved={sorted(saved_offsets)} current={sorted(expected_logs)}"
            )
        for relative, size in saved_offsets.items():
            pipeline._truncate_to_checkpoint(expected_logs[relative], int(size))
    else:
        pipeline.seed_everything(training_seed)

    output_identity = hashlib.sha256(str(run_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    tensorboard_root = Path(
        os.environ.get("NOMINAL_DDPG_PILOT_TENSORBOARD_ROOT", str(output_dir / "tensorboard"))
    )
    tensorboard_log = tensorboard_root / f"{config_hash[:12]}_{output_identity}"
    parent_step = 0 if resume_manifest is None else int(resume_manifest["timestep"])
    tensorboard_session = (
        f"fresh_{time.time_ns()}"
        if not resume_this_run
        else f"resume_{parent_step:09d}_{time.time_ns()}"
    )

    train_args = copy.copy(args)
    train_args.n_envs = 1
    train_env = pipeline.make_training_env(
        namespace,
        filtered=False,
        seed=training_seed,
        env_config=env_config,
        reward_config=reward_config,
        args=train_args,
        monitor_path=monitor_path,
        append_monitor=resume_this_run,
    )

    if resume_bundle is not None:
        vecnormalize_path = resume_bundle / pipeline.CHECKPOINT_PAYLOADS["vecnormalize"]
        if vecnormalize_path.exists():
            train_env = VecNormalize.load(str(vecnormalize_path), train_env)
        model = SplitLearningRateDDPG.load(
            str(resume_bundle / pipeline.CHECKPOINT_PAYLOADS["model"]),
            env=train_env,
            device=args.device,
            force_reset=False,
        )
        model.tensorboard_log = str(tensorboard_log)
        model.load_replay_buffer(
            str(resume_bundle / pipeline.CHECKPOINT_PAYLOADS["replay_buffer"]),
            truncate_last_traj=False,
        )
        pipeline.restore_environment_state(
            train_env,
            (resume_bundle / pipeline.CHECKPOINT_PAYLOADS["base_environment"]).read_bytes(),
            resume_pipeline_state["environment_state"],
        )
        saved_step = int(resume_manifest["timestep"])
        if int(model.num_timesteps) != saved_step or int(resume_pipeline_state["timestep"]) != saved_step:
            raise RuntimeError("Checkpoint timestep disagreement")
        if int(model._n_updates) != int(resume_pipeline_state["n_updates"]):
            raise RuntimeError("Checkpoint update-count disagreement")
        replay_state = resume_pipeline_state["replay_buffer_state"]
        actual_replay_state = {
            "class": pipeline._qualified_name(model.replay_buffer),
            "size": int(model.replay_buffer.size()),
            "position": int(model.replay_buffer.pos),
            "full": bool(model.replay_buffer.full),
        }
        if actual_replay_state != replay_state:
            raise RuntimeError(
                f"Replay-buffer state mismatch: saved={replay_state} restored={actual_replay_state}"
            )
        restored_obs = pipeline._base_vec_env(train_env)._obs_from_buf()
        if not pipeline._observations_equal(model._last_obs, restored_obs):
            raise RuntimeError("Saved model observation does not match restored environment observation")
        pipeline.restore_rng_state(model, resume_pipeline_state["rng_state"])
    else:
        model = build_pilot_model(
            train_env=train_env,
            config_values=config_values,
            training_seed=training_seed,
            device=args.device,
            tensorboard_log=tensorboard_log,
        )

    metadata = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "training_config_hash": config_hash,
        "training_config": payload,
        "pilot_config": pilot_config,
        "training_seed": int(training_seed),
        "target_timesteps": int(target_timesteps),
        "parameters": config_values,
        "resumed_from": None if resume_bundle is None else str(resume_bundle),
        "tensorboard_log": str(tensorboard_log),
        "tensorboard_session": tensorboard_session,
    }
    config_path = run_dir / "run_config.json"
    config_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    metrics_callback = pipeline.TrainingMetricsCallback(
        path=training_metrics_path,
        training_seed=training_seed,
        variant=pilot_config,
    )
    if resume_pipeline_state is not None:
        metrics_callback.load_state_dict(resume_pipeline_state.get("metrics_callback_state", {}))
    evaluation_callback = PilotEvaluationCallback(
        namespace=namespace,
        pilot_config=pilot_config,
        training_seed=training_seed,
        config_values=config_values,
        args=args,
        env_config=env_config,
        reward_config=reward_config,
        scenario_path=scenario_path,
        diagnostics_path=diagnostics_path,
        calibration_path=calibration_path,
        interval=int(args.checkpoint_interval),
        target_timesteps=target_timesteps,
    )
    checkpoint_callback = RetainedStrictCheckpointCallback(
        variant_dir=run_dir,
        checkpoint_interval=int(args.checkpoint_interval),
        training_config_hash=config_hash,
        metrics_callback=metrics_callback,
        tracked_log_paths=tracked_logs,
        resume_pipeline_state=resume_pipeline_state,
        strict_retention=int(args.strict_checkpoint_retention),
    )
    callbacks = CallbackList([metrics_callback, evaluation_callback, checkpoint_callback])

    started = time.perf_counter()
    try:
        remaining, learn_total_timesteps = sb3_resume_learn_target_timesteps(
            target_timesteps, model.num_timesteps
        )
        if remaining > 0:
            model.learn(
                total_timesteps=learn_total_timesteps,
                callback=callbacks,
                reset_num_timesteps=False,
                tb_log_name=tensorboard_session,
                log_interval=1,
                progress_bar=False,
            )
    finally:
        train_env.close()

    final_path = run_dir / "model_final.zip"
    model.save(str(final_path))
    metadata["elapsed_sec_this_process"] = float(time.perf_counter() - started)
    metadata["completed_timesteps"] = int(model.num_timesteps)
    metadata["n_updates"] = int(model._n_updates)
    config_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {
        "pilot_config": pilot_config,
        "training_seed": int(training_seed),
        "model_path": str(final_path),
        "run_dir": str(run_dir),
        "training_config_hash": config_hash,
    }


def expected_checkpoint_steps(target_timesteps: int, checkpoint_interval: int) -> list[int]:
    if int(target_timesteps) <= 0 or int(checkpoint_interval) <= 0:
        raise ValueError("Target timesteps and checkpoint interval must be positive")
    if int(target_timesteps) % int(checkpoint_interval) != 0:
        raise ValueError("Target timesteps must be divisible by checkpoint interval")
    return list(range(int(checkpoint_interval), int(target_timesteps) + 1, int(checkpoint_interval)))


def validate_run_evaluation_coverage(
    scenarios: pd.DataFrame,
    diagnostics: pd.DataFrame,
    calibration_samples: pd.DataFrame,
    *,
    pilot_config: str,
    training_seed: int,
    checkpoint_steps: list[int],
    eval_seeds: list[int],
    eval_timesteps: int,
    calibration_enabled: bool,
) -> None:
    """Reject incomplete, duplicated, or mispaired checkpoint evaluation output."""

    if set(scenarios["pilot_config"].astype(str)) != {str(pilot_config)} or set(
        pd.to_numeric(scenarios["training_seed"], errors="raise").astype(int)
    ) != {int(training_seed)}:
        raise RuntimeError(f"Evaluation identity mismatch for {pilot_config} seed={training_seed}")
    if not pd.to_numeric(scenarios["timesteps"], errors="raise").eq(int(eval_timesteps)).all():
        raise RuntimeError(
            f"Evaluation timestep budget mismatch for {pilot_config} seed={training_seed}"
        )
    scenario_keys = ["pilot_config", "training_seed", "model_timestep", "scenario_seed"]
    if scenarios.duplicated(scenario_keys).any():
        raise RuntimeError(f"Duplicate evaluation scenario keys for {pilot_config} seed={training_seed}")
    actual_scenarios = {
        (int(step), int(seed))
        for step, seed in scenarios[["model_timestep", "scenario_seed"]].itertuples(index=False)
    }
    expected_scenarios = {(int(step), int(seed)) for step in checkpoint_steps for seed in eval_seeds}
    if actual_scenarios != expected_scenarios:
        missing = sorted(expected_scenarios - actual_scenarios)[:10]
        extra = sorted(actual_scenarios - expected_scenarios)[:10]
        raise RuntimeError(
            f"Evaluation coverage mismatch for {pilot_config} seed={training_seed}: "
            f"missing={missing} extra={extra}"
        )

    diagnostic_keys = ["pilot_config", "training_seed", "model_timestep"]
    if set(diagnostics["pilot_config"].astype(str)) != {str(pilot_config)} or set(
        pd.to_numeric(diagnostics["training_seed"], errors="raise").astype(int)
    ) != {int(training_seed)}:
        raise RuntimeError(f"Diagnostics identity mismatch for {pilot_config} seed={training_seed}")
    if diagnostics.duplicated(diagnostic_keys).any():
        raise RuntimeError(f"Duplicate diagnostics keys for {pilot_config} seed={training_seed}")
    actual_diagnostic_steps = set(
        pd.to_numeric(diagnostics["model_timestep"], errors="raise").astype(int).tolist()
    )
    if actual_diagnostic_steps != set(checkpoint_steps):
        raise RuntimeError(
            f"Diagnostics coverage mismatch for {pilot_config} seed={training_seed}: "
            f"actual={sorted(actual_diagnostic_steps)} expected={checkpoint_steps}"
        )

    if not calibration_enabled:
        return
    if calibration_samples.empty:
        raise RuntimeError(f"Critic calibration samples are missing for {pilot_config} seed={training_seed}")
    if set(calibration_samples["pilot_config"].astype(str)) != {str(pilot_config)} or set(
        pd.to_numeric(calibration_samples["training_seed"], errors="raise").astype(int)
    ) != {int(training_seed)}:
        raise RuntimeError(f"Calibration identity mismatch for {pilot_config} seed={training_seed}")
    required_calibration_diagnostics = {
        "critic_calibration_bias_mean",
        "critic_calibration_mae",
        "critic_calibration_exact_coverage",
    }
    missing_diagnostics = required_calibration_diagnostics - set(diagnostics.columns)
    if missing_diagnostics:
        raise RuntimeError(
            f"Calibration diagnostics are missing for {pilot_config} seed={training_seed}: "
            f"{sorted(missing_diagnostics)}"
        )
    calibration_keys = [
        "pilot_config",
        "training_seed",
        "model_timestep",
        "scenario_seed",
        "segment_index",
        "anchor_segment_step",
    ]
    if calibration_samples.duplicated(calibration_keys).any():
        raise RuntimeError(
            f"Duplicate critic calibration anchors for {pilot_config} seed={training_seed}"
        )
    actual_calibration_pairs = {
        (int(step), int(seed))
        for step, seed in calibration_samples[["model_timestep", "scenario_seed"]].itertuples(
            index=False
        )
    }
    if actual_calibration_pairs != expected_scenarios:
        raise RuntimeError(
            f"Critic calibration coverage mismatch for {pilot_config} seed={training_seed}"
        )


def collect_run_outputs(
    output_dir: Path,
    run_specs: list[tuple[int, str]],
    *,
    target_timesteps: int,
    checkpoint_interval: int,
    eval_seeds: list[int],
    eval_timesteps: int,
    calibration_enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    checkpoint_steps = expected_checkpoint_steps(target_timesteps, checkpoint_interval)
    for training_seed, pilot_config in run_specs:
        run_dir = _run_dir(output_dir, training_seed, pilot_config)
        scenario_path = run_dir / "evaluation_scenarios.csv"
        diagnostics_path = run_dir / "checkpoint_diagnostics.csv"
        calibration_path = run_dir / "critic_calibration_samples.csv"
        if not scenario_path.exists() or not diagnostics_path.exists():
            raise RuntimeError(f"Run is missing checkpoint evaluation outputs: {run_dir}")
        scenarios_run = pd.read_csv(scenario_path)
        diagnostics_run = pd.read_csv(diagnostics_path)
        calibration_run = (
            pd.read_csv(calibration_path)
            if calibration_path.exists() and calibration_path.stat().st_size > 0
            else pd.DataFrame()
        )
        validate_run_evaluation_coverage(
            scenarios_run,
            diagnostics_run,
            calibration_run,
            pilot_config=pilot_config,
            training_seed=training_seed,
            checkpoint_steps=checkpoint_steps,
            eval_seeds=[int(seed) for seed in eval_seeds],
            eval_timesteps=int(eval_timesteps),
            calibration_enabled=bool(calibration_enabled),
        )
        scenario_frames.append(scenarios_run)
        diagnostic_frames.append(diagnostics_run)
        if not calibration_run.empty:
            calibration_frames.append(calibration_run)
    scenarios = pd.concat(scenario_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    calibration_samples = (
        pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()
    )
    if "initial_state_hash" in scenarios:
        hash_counts = scenarios.groupby("scenario_seed")["initial_state_hash"].nunique(dropna=False)
        mismatched = hash_counts[hash_counts.ne(1)]
        if not mismatched.empty:
            raise RuntimeError(
                f"Fixed evaluation seeds did not reproduce paired initial states: "
                f"{mismatched.index.astype(int).tolist()}"
            )
    scenarios.to_csv(output_dir / "evaluation_scenarios.csv", index=False)
    diagnostics.to_csv(output_dir / "checkpoint_diagnostics.csv", index=False)
    if not calibration_samples.empty:
        calibration_samples.to_csv(output_dir / "critic_calibration_samples.csv", index=False)
    return scenarios, diagnostics, calibration_samples


def build_checkpoint_seed_summary(scenarios: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for timestep, group in scenarios.groupby("model_timestep", sort=True):
        seed_summary = pipeline.summarize_within_training_seed(
            group.drop(columns=["model_timestep", "pilot_config"], errors="ignore")
        )
        seed_summary.insert(0, "model_timestep", int(timestep))
        seed_summary["pilot_config"] = seed_summary["variant"]
        pieces.append(seed_summary)
    summary = pd.concat(pieces, ignore_index=True)
    diagnostic_columns = [
        column
        for column in diagnostics.columns
        if column
        not in {
            "variant",
            "actor_lr",
            "critic_lr",
            "batch_size",
            "gamma",
            "tau",
            "learning_starts",
            "buffer_size",
            "ou_sigma",
        }
    ]
    diagnostics_merge = diagnostics[diagnostic_columns].copy()
    merged = summary.merge(
        diagnostics_merge,
        on=["pilot_config", "training_seed", "model_timestep"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(summary) or len(merged) != len(diagnostics_merge):
        raise RuntimeError("Checkpoint summary and diagnostics keys are not identical")
    return merged


def final_three_seed_averages(checkpoint_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_mean_metrics = (
        "mean_abs_speed_error",
        "nominal_action_saturation_rate",
        "first_collision_observed_rate",
        "time_to_first_collision_observed_mean_s",
        "distance_to_first_collision_observed_mean_m",
        "time_to_first_collision_restricted_mean_s",
        "distance_to_first_collision_restricted_mean_m",
        "critic_mse",
        "td_abs_mean",
        "td_abs_p95",
        "q_mean",
        "q_abs_mean",
        "q_abs_p95",
        "q_abs_max",
        "target_q_abs_mean",
        "reward_abs_max",
        "q_scale_reference",
        "q_scale_ratio",
        "q_scale_excess_log10",
        "q_target_magnitude_log_gap",
        "q_nonfinite_rate",
        "latest_train_actor_loss",
        "latest_train_critic_loss",
    )
    calibration_mean_metrics = (
        "critic_calibration_q_mean",
        "critic_calibration_empirical_return_mean",
        "critic_calibration_bias_mean",
        "critic_calibration_mae",
        "critic_calibration_rmse",
        "critic_calibration_normalized_bias",
        "critic_calibration_normalized_mae",
        "critic_calibration_normalized_rmse",
        "critic_calibration_overestimation_rate",
        "critic_calibration_pearson_r",
        "critic_calibration_empirical_on_q_slope",
        "critic_calibration_empirical_on_q_intercept",
        "critic_calibration_quantile_ece",
        "critic_calibration_normalized_quantile_ece",
        "critic_calibration_bootstrap_mae",
        "critic_calibration_censored_gamma_tail_mean",
        "critic_calibration_censored_gamma_tail_max",
        "critic_calibration_q_nonfinite_rate",
    )
    base_sum_metrics = (
        "scenarios",
        "timesteps",
        "total_time_s",
        "total_return",
        "task_return",
        "correction_return",
        "collision_free_scenarios",
        "collision_transition_timesteps",
        "collision_transition_return",
        "post_collision_timesteps",
        "post_collision_return",
        "reset_calls_total",
        "resets_after_collision",
        "resets_after_truncation_only",
        "resets_after_other_terminal",
        "episode_segments",
        "completed_segments",
        "right_censored_segments",
        "episode_length_sum",
    )
    calibration_sum_metrics = (
        "critic_calibration_anchor_count",
        "critic_calibration_exact_anchor_count",
        "critic_calibration_censored_anchor_count",
        "critic_calibration_finite_exact_anchor_count",
    )
    mean_metrics = tuple(
        metric
        for metric in (*base_mean_metrics, *calibration_mean_metrics)
        if metric in checkpoint_summary.columns
    )
    sum_metrics = tuple(
        metric
        for metric in (*base_sum_metrics, *calibration_sum_metrics)
        if metric in checkpoint_summary.columns
    )
    grouped_steps = {
        (str(config), int(seed)): tuple(
            sorted(int(value) for value in group["model_timestep"].unique())
        )
        for (config, seed), group in checkpoint_summary.groupby(
            ["pilot_config", "training_seed"], sort=True
        )
    }
    if len(set(grouped_steps.values())) > 1:
        raise RuntimeError(f"Checkpoint steps are not aligned across runs: {grouped_steps}")
    for (pilot_config, training_seed), group in checkpoint_summary.groupby(
        ["pilot_config", "training_seed"], sort=True
    ):
        steps = sorted(int(value) for value in group["model_timestep"].unique())
        if len(steps) < 3:
            raise RuntimeError(
                f"{pilot_config} seed={training_seed} has only {len(steps)} evaluated checkpoints; need three"
            )
        selected_steps = steps[-3:]
        final = group[group["model_timestep"].isin(selected_steps)].copy()
        total_distance = float(pd.to_numeric(final["total_distance_m"], errors="coerce").sum())
        collisions = float(
            pd.to_numeric(final["distinct_ego_collision_events"], errors="coerce").sum()
        )
        totals = {
            metric: float(pd.to_numeric(final[metric], errors="coerce").fillna(0.0).sum())
            for metric in sum_metrics
        }
        row: dict[str, Any] = {
            "pilot_config": str(pilot_config),
            "training_seed": int(training_seed),
            "checkpoint_count": 3,
            "checkpoint_steps": ",".join(str(step) for step in selected_steps),
            **totals,
            "total_distance_m": total_distance,
            "distinct_ego_collision_events": collisions,
            "distance_per_collision_m": pipeline._distance_per_collision(total_distance, collisions),
            "distance_per_collision_right_censored": int(collisions == 0.0),
            "distance_per_collision_exposure_bound_m": pipeline._distance_per_collision_exposure_bound(
                total_distance, collisions
            ),
            "ego_collisions_per_km": pipeline._collisions_per_km(collisions, total_distance),
            "return_per_timestep": pipeline._ratio(
                totals["total_return"], totals["timesteps"]
            ),
            "episode_length_mean": pipeline._ratio(
                totals["episode_length_sum"], totals["episode_segments"]
            ),
        }
        for metric in mean_metrics:
            row[metric] = _finite_mean(final[metric])
        if "critic_calibration_anchor_count" in totals:
            row["critic_calibration_exact_coverage"] = pipeline._ratio(
                totals["critic_calibration_exact_anchor_count"],
                totals["critic_calibration_anchor_count"],
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pilot_config", "training_seed"]).reset_index(drop=True)


def across_seed_final_three(seed_averages: pd.DataFrame) -> pd.DataFrame:
    identifiers = {"pilot_config", "training_seed", "checkpoint_steps"}
    metrics = [column for column in seed_averages.columns if column not in identifiers]
    rows: list[dict[str, Any]] = []
    for pilot_config, group in seed_averages.groupby("pilot_config", sort=True):
        row: dict[str, Any] = {
            "pilot_config": str(pilot_config),
            "training_seeds": int(group["training_seed"].nunique()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_seed_mean"] = float(values.mean())
            row[f"{metric}_seed_variance"] = (
                float(values.var(ddof=1)) if values.notna().sum() > 1 else np.nan
            )
            row[f"{metric}_seed_min"] = float(values.min()) if values.notna().any() else np.nan
            row[f"{metric}_seed_max"] = float(values.max()) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def rank_final_three(across_seed: pd.DataFrame) -> pd.DataFrame:
    ranked = across_seed.copy()
    criteria = {
        "distance_per_collision_exposure_bound_m_seed_mean": False,
        "return_per_timestep_seed_mean": False,
        "mean_abs_speed_error_seed_mean": True,
        "episode_length_mean_seed_mean": False,
        "nominal_action_saturation_rate_seed_mean": True,
        "critic_mse_seed_mean": True,
        "td_abs_mean_seed_mean": True,
        "q_scale_excess_log10_seed_mean": True,
    }
    rank_columns: list[str] = []
    for metric, ascending in criteria.items():
        if metric not in ranked:
            raise RuntimeError(f"Ranking metric is missing: {metric}")
        rank_name = f"rank_{metric.removesuffix('_seed_mean')}"
        values = pd.to_numeric(ranked[metric], errors="coerce")
        if values.notna().sum() == 0:
            ranked[rank_name] = np.nan
        else:
            ranked[rank_name] = values.rank(method="min", ascending=ascending, na_option="bottom")
        rank_columns.append(rank_name)
    ranked["mean_rank"] = ranked[rank_columns].mean(axis=1)
    ranked["stability_nonfinite"] = (
        pd.to_numeric(ranked["q_nonfinite_rate_seed_mean"], errors="coerce").fillna(1.0) > 0.0
    ).astype(int)
    ranked = ranked.sort_values(
        ["stability_nonfinite", "mean_rank", "rank_distance_per_collision_exposure_bound_m"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    ranked.insert(0, "overall_rank", np.arange(1, len(ranked) + 1))
    return ranked


def rank_confirmation_rollout(across_seed: pd.DataFrame) -> pd.DataFrame:
    """Rank confirmation by rollout means, then worst seed and variance.

    Critic scale, Bellman error, and calibration are reported diagnostics.  They
    do not outweigh consistently better fixed-seed rollout performance.
    """

    ranked = across_seed.copy()
    mean_criteria = {
        "distance_per_collision_exposure_bound_m_seed_mean": False,
        "return_per_timestep_seed_mean": False,
        "mean_abs_speed_error_seed_mean": True,
        "episode_length_mean_seed_mean": False,
        "nominal_action_saturation_rate_seed_mean": True,
    }
    worst_criteria = {
        "distance_per_collision_exposure_bound_m_seed_min": False,
        "return_per_timestep_seed_min": False,
        "mean_abs_speed_error_seed_max": True,
        "episode_length_mean_seed_min": False,
        "nominal_action_saturation_rate_seed_max": True,
    }
    variance_criteria = {
        metric.replace("_seed_mean", "_seed_variance"): True for metric in mean_criteria
    }

    def add_ranks(criteria: dict[str, bool], prefix: str) -> list[str]:
        columns: list[str] = []
        for metric, ascending in criteria.items():
            if metric not in ranked:
                raise RuntimeError(f"Confirmation ranking metric is missing: {metric}")
            rank_name = f"rank_{prefix}_{metric.rsplit('_seed_', 1)[0]}"
            ranked[rank_name] = pd.to_numeric(ranked[metric], errors="coerce").rank(
                method="min", ascending=ascending, na_option="bottom"
            )
            columns.append(rank_name)
        return columns

    mean_ranks = add_ranks(mean_criteria, "mean")
    worst_ranks = add_ranks(worst_criteria, "worst")
    variance_ranks = add_ranks(variance_criteria, "variance")
    ranked["rollout_mean_rank"] = ranked[mean_ranks].mean(axis=1)
    ranked["rollout_worst_seed_rank"] = ranked[worst_ranks].mean(axis=1)
    ranked["rollout_seed_variance_rank"] = ranked[variance_ranks].mean(axis=1)
    calibration_nonfinite = (
        pd.to_numeric(
            ranked["critic_calibration_q_nonfinite_rate_seed_mean"], errors="coerce"
        ).fillna(0.0)
        if "critic_calibration_q_nonfinite_rate_seed_mean" in ranked
        else pd.Series(0.0, index=ranked.index)
    )
    calibration_pairs = (
        pd.to_numeric(
            ranked["critic_calibration_finite_exact_anchor_count_seed_mean"], errors="coerce"
        )
        if "critic_calibration_finite_exact_anchor_count_seed_mean" in ranked
        else pd.Series(np.nan, index=ranked.index)
    )
    calibration_coverage = (
        pd.to_numeric(
            ranked["critic_calibration_exact_coverage_seed_mean"], errors="coerce"
        )
        if "critic_calibration_exact_coverage_seed_mean" in ranked
        else pd.Series(np.nan, index=ranked.index)
    )
    ranked["critic_calibration_warning"] = (
        (calibration_nonfinite > 0.0)
        | calibration_pairs.fillna(0.0).le(0.0)
        | calibration_coverage.fillna(0.0).lt(MIN_CRITIC_CALIBRATION_EXACT_COVERAGE)
    ).astype(int)
    ranked = ranked.sort_values(
        ["rollout_mean_rank", "rollout_worst_seed_rank", "rollout_seed_variance_rank"],
        ascending=True,
    ).reset_index(drop=True)
    ranked.insert(0, "overall_rank", np.arange(1, len(ranked) + 1))
    return ranked


def paired_seed_differences(
    seed_averages: pd.DataFrame,
    *,
    minuend: str = "P2_more_exploration",
    subtrahend: str = "P0_current",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return paired P2-P0 differences and their across-seed mean/variance."""

    indexed = seed_averages.set_index(["training_seed", "pilot_config"])
    seeds_a = set(
        seed_averages.loc[seed_averages["pilot_config"].eq(minuend), "training_seed"].astype(int)
    )
    seeds_b = set(
        seed_averages.loc[seed_averages["pilot_config"].eq(subtrahend), "training_seed"].astype(int)
    )
    if seeds_a != seeds_b or not seeds_a:
        raise RuntimeError(
            f"Paired configurations do not have identical seeds: {minuend}={seeds_a}, "
            f"{subtrahend}={seeds_b}"
        )
    identifiers = {"pilot_config", "training_seed", "checkpoint_steps"}
    numeric_metrics = [
        column
        for column in seed_averages.columns
        if column not in identifiers and pd.api.types.is_numeric_dtype(seed_averages[column])
    ]
    rows: list[dict[str, Any]] = []
    for seed in sorted(seeds_a):
        row: dict[str, Any] = {
            "training_seed": int(seed),
            "comparison": f"{minuend}_minus_{subtrahend}",
        }
        a = indexed.loc[(seed, minuend)]
        b = indexed.loc[(seed, subtrahend)]
        for metric in numeric_metrics:
            row[f"delta_{metric}"] = float(a[metric]) - float(b[metric])
        rows.append(row)
    paired = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "comparison": f"{minuend}_minus_{subtrahend}",
        "training_seeds": len(seeds_a),
    }
    for column in paired.columns:
        if not column.startswith("delta_"):
            continue
        values = pd.to_numeric(paired[column], errors="coerce")
        summary[f"{column}_mean"] = float(values.mean())
        summary[f"{column}_variance"] = (
            float(values.var(ddof=1)) if values.notna().sum() > 1 else np.nan
        )
    return paired, pd.DataFrame([summary])


def write_summaries(
    *,
    output_dir: Path,
    scenarios: pd.DataFrame,
    diagnostics: pd.DataFrame,
    calibration_samples: Optional[pd.DataFrame] = None,
    stage: str = "screen",
) -> pd.DataFrame:
    checkpoint_summary = build_checkpoint_seed_summary(scenarios, diagnostics)
    checkpoint_summary.to_csv(output_dir / "checkpoint_seed_summary.csv", index=False)
    seed_averages = final_three_seed_averages(checkpoint_summary)
    seed_averages.to_csv(output_dir / "final_three_seed_averages.csv", index=False)
    across_seed = across_seed_final_three(seed_averages)
    across_seed.to_csv(output_dir / "final_three_across_seeds.csv", index=False)
    ranking = (
        rank_confirmation_rollout(across_seed)
        if str(stage) == "confirm"
        else rank_final_three(across_seed)
    )
    ranking.to_csv(output_dir / "ranking_final_three.csv", index=False)
    if calibration_samples is not None and not calibration_samples.empty:
        build_critic_calibration_bins(calibration_samples).to_csv(
            output_dir / "critic_calibration_bins.csv", index=False
        )
    if str(stage) == "confirm":
        paired, paired_summary = paired_seed_differences(seed_averages)
        paired.to_csv(output_dir / "paired_seed_differences.csv", index=False)
        paired_summary.to_csv(output_dir / "paired_difference_summary.csv", index=False)
    best_two = ranking["pilot_config"].head(2).tolist()
    (output_dir / "selected_best_two.json").write_text(
        json.dumps(
            {
                "selection_rule": (
                    "rollout seed means, then worst-seed rollout and seed variance; critic "
                    "diagnostics and calibration do not enter the confirmation rank"
                    if str(stage) == "confirm"
                    else "equal-weight mean rank over final three checkpoints"
                ),
                "distance_per_collision_note": (
                    "Exact distance/collision remains infinite when collision-free; finite exposure lower bound is used for ranking."
                ),
                "q_magnitude_note": (
                    "Raw Q magnitudes are reported; ranking uses only excess above the sampled "
                    "reward/(1-gamma) reference so different gamma values are not rewarded for "
                    "artificially smaller Q scales."
                ),
                "critic_calibration_note": (
                    "Positive Q-minus-empirical-return bias means overestimation. Only true "
                    "terminal Monte Carlo anchors enter primary calibration; truncated and "
                    "evaluation-budget tails are right-censored and reported separately."
                ),
                "critic_calibration_low_coverage_warning_threshold": (
                    MIN_CRITIC_CALIBRATION_EXACT_COVERAGE
                ),
                "selected_configs": best_two,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ranking


def resolve_selected_configs(args: argparse.Namespace, project_root: Path) -> list[str]:
    if args.stage == "screen":
        selected = list(args.configs or PILOT_CONFIGS.keys())
    elif args.selected_configs:
        selected = list(args.selected_configs)
    elif args.screen_ranking is None:
        selected = list(DEFAULT_CONFIRM_CONFIGS)
    else:
        ranking_path = args.screen_ranking
        ranking = pd.read_csv(ranking_path)
        selected = ranking["pilot_config"].head(2).astype(str).tolist()
    unknown = [name for name in selected if name not in PILOT_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown pilot configurations: {unknown}")
    if args.stage == "confirm" and len(selected) != 2:
        raise ValueError("Confirmation requires exactly two selected configurations")
    if args.stage == "confirm" and set(selected) != set(DEFAULT_CONFIRM_CONFIGS):
        raise ValueError(
            f"Confirmation is locked to the screened finalists: {list(DEFAULT_CONFIRM_CONFIGS)}"
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the nominal-DDPG P0--P3 parameter pilot.")
    parser.add_argument("--stage", choices=("screen", "confirm", "summarize"), default="screen")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--configs", nargs="+", default=None, help="Optional screen subset, mainly for smoke tests.")
    parser.add_argument("--selected-configs", nargs="+", default=None)
    parser.add_argument("--screen-ranking", type=Path, default=None)
    parser.add_argument("--eval-seed-start", type=int, default=DEFAULT_EVAL_SEED_START)
    parser.add_argument("--eval-scenarios", type=int, default=DEFAULT_EVAL_SCENARIOS)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--eval-timesteps", type=int, default=800)
    parser.add_argument("--diagnostic-batch-size", type=int, default=256)
    parser.add_argument(
        "--critic-calibration",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable terminal Monte Carlo Q calibration (enabled by default for confirm).",
    )
    parser.add_argument("--critic-calibration-stride", type=int, default=20)
    parser.add_argument("--strict-checkpoint-retention", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-envs", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--eval-episodes", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--eval-horizon", type=int, default=800, help=argparse.SUPPRESS)
    parser.add_argument("--k0", type=float, default=5.29, help=argparse.SUPPRESS)
    parser.add_argument("--k1", type=float, default=3.68, help=argparse.SUPPRESS)
    parser.add_argument("--eps-side", type=float, default=0.10, help=argparse.SUPPRESS)
    parser.add_argument("--correction-epsilon", type=float, default=0.03, help=argparse.SUPPRESS)
    add_env_config_args(parser)
    parser.set_defaults(
        traffic_model="mtm",
        env_config_json=None,
        env_config_file=None,
    )
    return parser.parse_args()


def _main_resolved(args: argparse.Namespace, project_root: Path, output_dir: Path) -> int:
    if args.stage == "summarize":
        root_config_path = output_dir / "run_config.json"
        root_config = (
            json.loads(root_config_path.read_text(encoding="utf-8"))
            if root_config_path.exists()
            else {}
        )
        summary_stage = str(root_config.get("stage", "screen"))
        scenarios = pd.read_csv(output_dir / "evaluation_scenarios.csv")
        diagnostics = pd.read_csv(output_dir / "checkpoint_diagnostics.csv")
        calibration_path = output_dir / "critic_calibration_samples.csv"
        calibration_samples = (
            pd.read_csv(calibration_path)
            if calibration_path.exists() and calibration_path.stat().st_size > 0
            else pd.DataFrame()
        )
        ranking = write_summaries(
            output_dir=output_dir,
            scenarios=scenarios,
            diagnostics=diagnostics,
            calibration_samples=calibration_samples,
            stage=summary_stage,
        )
        print(ranking.to_string(index=False), flush=True)
        return 0

    target_timesteps = int(args.timesteps or (50_000 if args.stage == "screen" else 150_000))
    if target_timesteps <= 0 or int(args.checkpoint_interval) <= 0:
        raise ValueError("Timesteps and checkpoint interval must be positive")
    if target_timesteps % int(args.checkpoint_interval) != 0:
        raise ValueError("Target timesteps must be divisible by checkpoint interval")
    if target_timesteps < 3 * int(args.checkpoint_interval):
        raise ValueError("At least three evaluation checkpoints are required")
    if int(args.n_envs) != 1:
        raise ValueError("This pilot requires --n-envs 1")
    if args.eval_seeds is None:
        args.eval_seeds = [int(args.eval_seed_start) + index for index in range(int(args.eval_scenarios))]
    else:
        args.eval_seeds = [int(seed) for seed in args.eval_seeds]
    if not args.eval_seeds or len(set(args.eval_seeds)) != len(args.eval_seeds):
        raise ValueError("Evaluation seeds must be non-empty and unique")
    if int(args.eval_timesteps) <= 0:
        raise ValueError("Evaluation timestep budget must be positive")
    if args.critic_calibration is None:
        args.critic_calibration = args.stage == "confirm"
    if int(args.critic_calibration_stride) <= 0:
        raise ValueError("Critic calibration stride must be positive")
    args.eval_scenarios = len(args.eval_seeds)
    args.eval_episodes = len(args.eval_seeds)
    args.eval_horizon = int(args.eval_timesteps)

    selected_configs = resolve_selected_configs(args, project_root)
    if args.seeds is None:
        training_seeds = [DEFAULT_SCREEN_SEED] if args.stage == "screen" else list(DEFAULT_CONFIRM_SEEDS)
    else:
        training_seeds = [int(seed) for seed in args.seeds]
    if len(set(training_seeds)) != len(training_seeds):
        raise ValueError("Training seeds must be unique")
    if args.stage == "screen" and len(training_seeds) != 1:
        raise ValueError("Screening requires one common training seed")
    if args.stage == "confirm" and len(training_seeds) != 3:
        raise ValueError("Confirmation requires exactly three independent training seeds")

    namespace = pipeline.bootstrap_notebook_namespace(project_root)
    pipeline.exec_required_notebook_cells(project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace)
    namespace["DEVICE"] = args.device
    args.cbf_snapshot = fixed_cbf_snapshot(namespace)
    args.k0 = float(args.cbf_snapshot["k0"])
    args.k1 = float(args.cbf_snapshot["k1"])
    args.eps_side = float(args.cbf_snapshot["eps_side"])
    env_config = env_config_from_args(args, namespace["ENV_CONFIG"])
    if active_traffic_model(env_config) == "mtm":
        deep_update(env_config, copy.deepcopy(MTM_CONGESTED_UNCERTAIN_UPDATES))
    if not bool(env_config.get("terminate_on_collision", False)):
        raise RuntimeError("Pilot protocol requires terminate_on_collision=True")
    reward_config = pipeline.make_base_reward_config(namespace)
    run_specs = [(seed, config) for seed in training_seeds for config in selected_configs]
    preflight_runs(
        output_dir=output_dir,
        run_specs=run_specs,
        args=args,
        project_root=project_root,
        env_config=env_config,
        reward_config=reward_config,
        target_timesteps=target_timesteps,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    root_config = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "study": "nominal_ddpg_core_parameter_pilot",
        "stage": args.stage,
        "selected_configs": selected_configs,
        "training_seeds": training_seeds,
        "target_timesteps": target_timesteps,
        "checkpoint_interval": int(args.checkpoint_interval),
        "eval_seeds": args.eval_seeds,
        "eval_timesteps": int(args.eval_timesteps),
        "environment_and_cbf_tuning_changed": False,
        "fixed_cbf_snapshot": args.cbf_snapshot,
        "filtered_training": False,
        "critic_calibration": {
            "enabled": bool(args.critic_calibration),
            "anchor_stride_within_episode": int(args.critic_calibration_stride),
            "primary_target": "terminal_mc",
            "right_censored_tails_excluded": True,
            "secondary_target": "bootstrapped_sensitivity",
            "low_exact_coverage_warning_threshold": MIN_CRITIC_CALIBRATION_EXACT_COVERAGE,
        },
        "episode_reset_reseed": False,
        "configurations": {name: PILOT_CONFIGS[name] for name in selected_configs},
        "env_config": env_config,
        "reward_config": reward_config,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(root_config, indent=2, default=str), encoding="utf-8"
    )
    print(
        "[nominal-pilot] starting"
        f" stage={args.stage} configs={selected_configs} seeds={training_seeds}"
        f" timesteps={target_timesteps:,} eval_every={args.checkpoint_interval:,}"
        f" eval={len(args.eval_seeds)}x{args.eval_timesteps}",
        flush=True,
    )

    model_rows: list[dict[str, Any]] = []
    for training_seed, pilot_config in run_specs:
        print(
            f"[nominal-pilot] train {pilot_config} seed={training_seed} parameters={PILOT_CONFIGS[pilot_config]}",
            flush=True,
        )
        model_rows.append(
            train_one_run(
                namespace,
                pilot_config=pilot_config,
                training_seed=training_seed,
                target_timesteps=target_timesteps,
                args=args,
                env_config=env_config,
                reward_config=reward_config,
                output_dir=output_dir,
            )
        )
    pd.DataFrame(model_rows).to_csv(output_dir / "models.csv", index=False)
    scenarios, diagnostics, calibration_samples = collect_run_outputs(
        output_dir,
        run_specs,
        target_timesteps=target_timesteps,
        checkpoint_interval=int(args.checkpoint_interval),
        eval_seeds=[int(seed) for seed in args.eval_seeds],
        eval_timesteps=int(args.eval_timesteps),
        calibration_enabled=bool(args.critic_calibration),
    )
    ranking = write_summaries(
        output_dir=output_dir,
        scenarios=scenarios,
        diagnostics=diagnostics,
        calibration_samples=calibration_samples,
        stage=args.stage,
    )
    print("[nominal-pilot] final-three-checkpoint ranking", flush=True)
    report_columns = [
        "overall_rank",
        "pilot_config",
        "training_seeds",
        *( ["rollout_mean_rank", "rollout_worst_seed_rank", "rollout_seed_variance_rank"]
           if args.stage == "confirm" else ["mean_rank"] ),
        "distance_per_collision_m_seed_mean",
        "distance_per_collision_right_censored_seed_mean",
        "ego_collisions_per_km_seed_mean",
        "total_distance_m_seed_mean",
        "distinct_ego_collision_events_seed_mean",
        "reset_calls_total_seed_mean",
        "return_per_timestep_seed_mean",
        "mean_abs_speed_error_seed_mean",
        "episode_length_mean_seed_mean",
        "time_to_first_collision_restricted_mean_s_seed_mean",
        "distance_to_first_collision_restricted_mean_m_seed_mean",
        "nominal_action_saturation_rate_seed_mean",
        "critic_mse_seed_mean",
        "td_abs_mean_seed_mean",
        "q_abs_mean_seed_mean",
        "q_abs_max_seed_mean",
        "q_scale_excess_log10_seed_mean",
    ]
    if args.stage == "confirm":
        report_columns.extend(
            [
                "critic_calibration_exact_coverage_seed_mean",
                "critic_calibration_bias_mean_seed_mean",
                "critic_calibration_normalized_bias_seed_mean",
                "critic_calibration_mae_seed_mean",
                "critic_calibration_pearson_r_seed_mean",
                "critic_calibration_warning",
            ]
        )
    print(ranking[report_columns].to_string(index=False), flush=True)
    print(f"[nominal-pilot] complete: {output_dir}", flush=True)
    return 0


def main() -> int:
    pipeline.set_stable_native_defaults()
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()
    project_root = pipeline.find_project_root(args.project_root or Path.cwd())
    default_stage_dir = project_root / "artifacts" / "nominal_ddpg_parameter_pilot" / args.stage
    output_dir = (args.output_dir or default_stage_dir).resolve()

    with OutputDirectoryRunLock(output_dir):
        return _main_resolved(args, project_root, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
