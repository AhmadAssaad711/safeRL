"""Audit nominal PPO behavior and reward components without changing training.

The audit evaluates a saved nominal PPO checkpoint with the external CBF OFF
and compares it with a constant-action policy on the same scenario seeds.  It
uses the reward wrapper's logged state terms to separate the reciprocal main
reward, progress, overtaking, collision, and comfort contributions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from itertools import permutations
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scripts.training.run_ppo_cbf_progression as progression


DEFAULT_SEED_START = 1_300_000
DEFAULT_EPISODES = 100
DEFAULT_GAMMA = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit nominal PPO reward and behavior with external CBF OFF."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="ppo_nominal/seed_<seed> directory containing model_final.zip and run_config.json.",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help="Discount factor used for the PPO-scale diagnostic; default=%(default)s.",
    )
    parser.add_argument(
        "--constant-ax",
        type=float,
        default=0.0,
        help="Physical longitudinal acceleration for the simple comparator.",
    )
    parser.add_argument(
        "--constant-ay",
        type=float,
        default=0.0,
        help="Physical lateral acceleration for the simple comparator.",
    )
    return parser.parse_args()


def _load_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run configuration: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("variant") != "ppo_nominal":
        raise ValueError(
            "This audit expects a nominal PPO run; observed "
            f"variant={config.get('variant')!r}."
        )
    for key in ("env_config", "reward_config", "training_signature"):
        if key not in config:
            raise KeyError(f"Run configuration is missing {key!r}: {config_path}")
    return config


def _override_cbf_snapshot(namespace: dict[str, Any], config: dict[str, Any]) -> None:
    snapshot = config.get("training_signature", {}).get("cbf", {})
    if not isinstance(snapshot, dict):
        raise TypeError("training_signature.cbf must be a JSON object")
    for key, value in snapshot.items():
        if str(key).startswith("CBF_"):
            namespace[str(key)] = value


def _as_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _shapley_denominator_losses(
    weighted_costs: dict[str, float], epsilon: float
) -> dict[str, float]:
    """Allocate the reciprocal reward loss across its coupled denominator costs.

    The current main reward is ``epsilon / (epsilon + sum(costs))``.  Since the
    costs are coupled, their raw denominator terms are not additive reward
    contributions.  Shapley allocation gives a symmetric decomposition of the
    loss from the no-cost reward (which is one), and the allocations sum exactly
    to ``1 - main_reward``.
    """

    names = tuple(weighted_costs)
    values = np.asarray([max(float(weighted_costs[name]), 0.0) for name in names])
    result = np.zeros(len(names), dtype=float)

    def loss(indices: tuple[int, ...]) -> float:
        denominator = float(epsilon) + float(np.sum(values[list(indices)])) if indices else float(epsilon)
        return float(1.0 - float(epsilon) / max(denominator, 1e-9))

    for order in permutations(range(len(names))):
        present: tuple[int, ...] = ()
        previous_loss = 0.0
        for index in order:
            expanded = present + (index,)
            current_loss = loss(expanded)
            result[index] += current_loss - previous_loss
            present = expanded
            previous_loss = current_loss
    result /= float(math.factorial(len(names)))
    return {name: float(result[index]) for index, name in enumerate(names)}


def _reward_terms(info: dict[str, Any], reward_config: dict[str, Any]) -> dict[str, float]:
    """Reconstruct the additive reward terms and expose denominator costs."""

    reward_mode = str(reward_config.get("reward_mode", "reciprocal")).strip().lower()
    epsilon = float(reward_config["epsilon_r"])
    wx = float(reward_config["wx"])
    wy = float(reward_config["wy"])
    wf = float(reward_config["wf"])
    way = float(reward_config.get("way", 0.0))
    cx = _as_float(info.get("karalakou_cx"), 0.0)
    cy = _as_float(info.get("karalakou_cy"), 0.0)
    cf = _as_float(info.get("karalakou_cf"), 0.0)
    cay = _as_float(info.get("karalakou_cay"), 0.0)
    denominator = epsilon + wx * cx + wy * cy + wf * cf + way * cay
    main_reward = epsilon / max(denominator, 1e-9)
    weighted_costs = {
        "speed": float(wx * cx),
        "lateral": float(wy * cy),
        "potential": float(wf * cf),
        "comfort": float(way * cay),
    }
    shapley_losses = _shapley_denominator_losses(weighted_costs, epsilon)
    collision = bool(_as_float(info.get("karalakou_ego_collision"), 0.0) > 0.5)
    overtakes = max(_as_float(info.get("karalakou_overtakes"), 0.0), 0.0)
    progress_reward = _as_float(info.get("karalakou_progress_reward"), 0.0)
    reciprocal_reward = _as_float(
        info.get("karalakou_reciprocal_reward"), main_reward
    )
    speed_tracking_reward = _as_float(
        info.get("karalakou_speed_tracking_reward"), 0.0
    )
    lateral_tracking_reward = _as_float(
        info.get("karalakou_lateral_tracking_reward"), 0.0
    )
    additive_speed_reward = _as_float(
        info.get("karalakou_additive_speed_reward"),
        float(reward_config.get("speed_reward_weight", 0.25))
        * speed_tracking_reward,
    )
    additive_lateral_reward = _as_float(
        info.get("karalakou_additive_lateral_reward"),
        float(reward_config.get("lateral_reward_weight", 0.25))
        * lateral_tracking_reward,
    )
    additive_risk_penalty = _as_float(
        info.get("karalakou_additive_risk_penalty"),
        -float(reward_config.get("risk_penalty_weight", 0.5))
        * float(np.clip(cf, 0.0, 1.0)),
    )
    safety_potential_penalty = _as_float(
        info.get("karalakou_safety_potential_penalty"), 0.0
    )
    safety_ellipse_cost = _as_float(info.get("karalakou_safety_ellipse_cost"), 0.0)
    safety_ttc_cost = _as_float(info.get("karalakou_safety_ttc_cost"), 0.0)
    safety_ellipse_penalty = _as_float(
        info.get("karalakou_safety_ellipse_penalty"),
        -float(reward_config.get("safety_ellipse_weight", 0.0)) * safety_ellipse_cost,
    )
    safety_ttc_penalty = _as_float(
        info.get("karalakou_safety_ttc_penalty"),
        -float(reward_config.get("safety_ttc_weight", 0.0)) * safety_ttc_cost,
    )
    safety_total_penalty = _as_float(
        info.get("karalakou_safety_total_penalty"),
        safety_potential_penalty + safety_ellipse_penalty + safety_ttc_penalty,
    )
    potential_shaping_reward = _as_float(
        info.get("karalakou_potential_shaping_reward"), 0.0
    )
    additive_base_reward = (
        progress_reward
        + additive_speed_reward
        + additive_lateral_reward
        + additive_risk_penalty
        + safety_total_penalty
        + potential_shaping_reward
    )
    if reward_mode == "additive":
        main_reward = additive_base_reward
        reciprocal_loss_total = 0.0
        shapley_losses = {name: 0.0 for name in shapley_losses}
    else:
        reciprocal_loss_total = float(1.0 - reciprocal_reward)
    overtake_bonus = (
        float(reward_config.get("overtake_bonus", 0.0)) * min(overtakes, 1.0)
        if not collision
        else 0.0
    )
    collision_penalty = (
        float(reward_config.get("collision_penalty", 0.0)) if collision else 0.0
    )
    collision_reward_override = bool(
        reward_config.get("collision_reward_override", False)
    )
    non_event_reward = (
        main_reward
        if reward_mode == "additive"
        else main_reward + progress_reward + potential_shaping_reward
    )
    reward_reconstructed = (
        collision_penalty
        if collision and collision_reward_override
        else non_event_reward + overtake_bonus + collision_penalty
    )
    return {
        "main_reward": float(main_reward),
        "reward_mode_additive": float(reward_mode == "additive"),
        "reciprocal_reward": float(reciprocal_reward),
        "speed_tracking_reward": float(speed_tracking_reward),
        "lateral_tracking_reward": float(lateral_tracking_reward),
        "additive_speed_reward": float(additive_speed_reward),
        "additive_lateral_reward": float(additive_lateral_reward),
        "additive_risk_penalty": float(additive_risk_penalty),
        "safety_ellipse_cost": float(safety_ellipse_cost),
        "safety_ttc_cost": float(safety_ttc_cost),
        "safety_potential_penalty": float(safety_potential_penalty),
        "safety_ellipse_penalty": float(safety_ellipse_penalty),
        "safety_ttc_penalty": float(safety_ttc_penalty),
        "safety_total_penalty": float(safety_total_penalty),
        "potential_shaping_reward": float(potential_shaping_reward),
        "progress_reward": float(progress_reward),
        "overtake_bonus": float(overtake_bonus),
        "collision_penalty": float(collision_penalty),
        "explicit_survival_reward": 0.0,
        "comfort_denominator_cost": float(way * cay),
        "speed_denominator_cost": float(wx * cx),
        "lateral_denominator_cost": float(wy * cy),
        "potential_denominator_cost": float(wf * cf),
        "reciprocal_loss_total": float(reciprocal_loss_total),
        "shapley_speed_loss": float(shapley_losses["speed"]),
        "shapley_lateral_loss": float(shapley_losses["lateral"]),
        "shapley_potential_loss": float(shapley_losses["potential"]),
        "shapley_comfort_loss": float(shapley_losses["comfort"]),
        "collision_reward_override": float(collision_reward_override),
        "reward_reconstructed": float(reward_reconstructed),
    }


def _action_metrics(action: np.ndarray, ax_bound: float = 3.0, ay_bound: float = 3.0) -> dict[str, float]:
    action = np.asarray(action, dtype=float).reshape(-1)[:2]
    near_saturated = np.abs(action) >= 0.95 * np.asarray([ax_bound, ay_bound])
    exact_saturated = np.abs(action) >= np.asarray([ax_bound, ay_bound]) - 1e-6
    return {
        "action_ax": float(action[0]),
        "action_ay": float(action[1]),
        "near_saturation_fraction": float(np.mean(near_saturated)),
        "exact_saturation_fraction": float(np.mean(exact_saturated)),
        "near_ax_positive_fraction": float(action[0] >= 0.95 * ax_bound),
        "near_ax_negative_fraction": float(action[0] <= -0.95 * ax_bound),
        "near_ay_saturation_fraction": float(abs(action[1]) >= 0.95 * ay_bound),
        "near_zero_action_fraction": float(np.linalg.norm(action) <= 0.10),
    }


def _run_episode(
    *,
    namespace: dict[str, Any],
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    correction_epsilon: float,
    gamma: float,
    episode_index: int,
    scenario_seed: int,
    policy_name: str,
    action_provider: Callable[[np.ndarray], np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = progression.make_evaluation_env(
        namespace,
        mode="raw",
        env_config=env_config,
        reward_config=reward_config,
        correction_epsilon=float(correction_epsilon),
    )
    rows: list[dict[str, Any]] = []
    try:
        observation, _ = env.reset(seed=int(scenario_seed))
        total_return = 0.0
        terminated = truncated = False
        max_steps = int(env_config.get("episode_steps", env_config.get("duration", 800)))
        for step in range(max_steps + 2):
            action = np.asarray(action_provider(observation), dtype=np.float32).reshape(-1)[:2]
            observation, reward, terminated, truncated, info = env.step(action)
            info = dict(info)
            terms = _reward_terms(info, reward_config)
            action_stats = _action_metrics(action)
            discount_factor = float(gamma ** step)
            base = env.unwrapped
            total_return += float(reward)
            row = {
                "policy": policy_name,
                "episode_index": int(episode_index),
                "scenario_seed": int(scenario_seed),
                "step": int(step + 1),
                "reward": float(reward),
                "discount_factor": discount_factor,
                "discounted_reward": float(discount_factor * float(reward)),
                "total_return_so_far": float(total_return),
                "ego_speed": _as_float(info.get("karalakou_ego_speed"), _as_float(getattr(base.vehicle, "vx", np.nan))),
                "desired_speed": _as_float(info.get("karalakou_desired_speed"), _as_float(getattr(base.vehicle, "desired_speed", np.nan))),
                "target_speed": _as_float(info.get("karalakou_target_speed")),
                "ego_y": _as_float(info.get("karalakou_ego_y"), _as_float(base.vehicle.position[1])),
                "target_y": _as_float(info.get("karalakou_target_y")),
                "progress_m": _as_float(info.get("karalakou_progress_m"), 0.0),
                "progress_normalized": _as_float(info.get("karalakou_progress_normalized"), 0.0),
                "cx": _as_float(info.get("karalakou_cx"), 0.0),
                "cy": _as_float(info.get("karalakou_cy"), 0.0),
                "cf": _as_float(info.get("karalakou_cf"), 0.0),
                "cay": _as_float(info.get("karalakou_cay"), 0.0),
                "overtakes": _as_float(info.get("karalakou_overtakes"), 0.0),
                # The reward wrapper charges its penalty from active contact;
                # the protocol KPI separately counts distinct collision events.
                "reward_collision_trigger": float(
                    _as_float(info.get("karalakou_ego_collision"), 0.0) > 0.5
                ),
                "distinct_collision_events": float(
                    max(
                        int(
                            _as_float(
                                info.get(
                                    "pipeline_distinct_ego_collision_events",
                                    info.get("ego_collision_events", 0),
                                ),
                                0.0,
                            )
                        ),
                        0,
                    )
                ),
                "pipeline_collision_transition": _as_float(info.get("pipeline_collision_transition"), 0.0),
                "pipeline_action_saturation": _as_float(info.get("pipeline_action_saturation"), 0.0),
                **terms,
                **action_stats,
            }
            for component in (
                "main_reward",
                "reciprocal_reward",
                "additive_speed_reward",
                "additive_lateral_reward",
                "additive_risk_penalty",
                "safety_ellipse_penalty",
                "safety_ttc_penalty",
                "safety_total_penalty",
                "safety_potential_penalty",
                "potential_shaping_reward",
                "progress_reward",
                "overtake_bonus",
                "collision_penalty",
                "reciprocal_loss_total",
                "shapley_speed_loss",
                "shapley_lateral_loss",
                "shapley_potential_loss",
                "shapley_comfort_loss",
            ):
                row[f"discounted_{component}"] = float(
                    discount_factor * float(terms[component])
                )
            rows.append(row)
            if terminated or truncated:
                break
        else:
            raise RuntimeError(
                f"Episode seed={scenario_seed} did not terminate within {max_steps + 2} steps"
            )
    finally:
        env.close()

    frame = pd.DataFrame(rows)
    collision_positions = np.flatnonzero(
        frame["reward_collision_trigger"].to_numpy() > 0.5
    )
    collision_position = (
        int(collision_positions[0]) if collision_positions.size else None
    )
    collision_step = (
        int(frame.iloc[collision_position]["step"])
        if collision_position is not None
        else np.nan
    )
    pre_collision_frame = (
        frame.iloc[:collision_position]
        if collision_position is not None
        else frame
    )
    pre_collision_main = float(pre_collision_frame["main_reward"].sum())
    pre_collision_discounted_main = float(
        pre_collision_frame["discounted_main_reward"].sum()
    )
    mean_pre_collision_main = float(pre_collision_frame["main_reward"].mean())
    collision_penalty_magnitude = abs(float(frame["collision_penalty"].sum()))
    summary = {
        "policy": policy_name,
        "episode_index": int(episode_index),
        "scenario_seed": int(scenario_seed),
        "steps": int(len(frame)),
        "return": float(frame["reward"].sum()),
        "discounted_return": float(frame["discounted_reward"].sum()),
        "mean_reward": float(frame["reward"].mean()),
        "main_reward_sum": float(frame["main_reward"].sum()),
        "main_reward_before_collision_sum": pre_collision_main,
        "discounted_main_reward_sum": float(frame["discounted_main_reward"].sum()),
        "discounted_main_reward_before_collision_sum": pre_collision_discounted_main,
        "pre_collision_steps": int(len(pre_collision_frame)),
        "mean_pre_collision_main_reward": mean_pre_collision_main,
        "normal_steps_to_cancel_collision": (
            float(collision_penalty_magnitude / mean_pre_collision_main)
            if mean_pre_collision_main > 1e-9 and collision_penalty_magnitude > 0.0
            else np.nan
        ),
        "collision_penalty_to_pre_collision_main_ratio": (
            float(collision_penalty_magnitude / pre_collision_main)
            if pre_collision_main > 1e-9 and collision_penalty_magnitude > 0.0
            else np.nan
        ),
        "progress_reward_sum": float(frame["progress_reward"].sum()),
        "reciprocal_reward_sum": float(frame["reciprocal_reward"].sum()),
        "additive_speed_reward_sum": float(frame["additive_speed_reward"].sum()),
        "additive_lateral_reward_sum": float(frame["additive_lateral_reward"].sum()),
        "additive_risk_penalty_sum": float(frame["additive_risk_penalty"].sum()),
        "safety_ellipse_cost_mean": float(frame["safety_ellipse_cost"].mean()),
        "safety_ttc_cost_mean": float(frame["safety_ttc_cost"].mean()),
        "safety_potential_penalty_sum": float(frame["safety_potential_penalty"].sum()),
        "safety_ellipse_penalty_sum": float(frame["safety_ellipse_penalty"].sum()),
        "safety_ttc_penalty_sum": float(frame["safety_ttc_penalty"].sum()),
        "safety_total_penalty_sum": float(frame["safety_total_penalty"].sum()),
        "potential_shaping_reward_sum": float(frame["potential_shaping_reward"].sum()),
        "overtake_bonus_sum": float(frame["overtake_bonus"].sum()),
        "collision_penalty_sum": float(frame["collision_penalty"].sum()),
        "discounted_reciprocal_reward_sum": float(
            frame["discounted_reciprocal_reward"].sum()
        ),
        "discounted_additive_speed_reward_sum": float(
            frame["discounted_additive_speed_reward"].sum()
        ),
        "discounted_additive_lateral_reward_sum": float(
            frame["discounted_additive_lateral_reward"].sum()
        ),
        "discounted_additive_risk_penalty_sum": float(
            frame["discounted_additive_risk_penalty"].sum()
        ),
        "discounted_safety_ellipse_penalty_sum": float(
            frame["discounted_safety_ellipse_penalty"].sum()
        ),
        "discounted_safety_ttc_penalty_sum": float(
            frame["discounted_safety_ttc_penalty"].sum()
        ),
        "discounted_safety_total_penalty_sum": float(
            frame["discounted_safety_total_penalty"].sum()
        ),
        "discounted_safety_potential_penalty_sum": float(
            frame["discounted_safety_potential_penalty"].sum()
        ),
        "discounted_potential_shaping_reward_sum": float(
            frame["discounted_potential_shaping_reward"].sum()
        ),
        "discounted_progress_reward_sum": float(frame["discounted_progress_reward"].sum()),
        "discounted_overtake_bonus_sum": float(frame["discounted_overtake_bonus"].sum()),
        "discounted_collision_penalty_sum": float(frame["discounted_collision_penalty"].sum()),
        "explicit_survival_reward_sum": float(frame["explicit_survival_reward"].sum()),
        "comfort_denominator_cost_sum": float(frame["comfort_denominator_cost"].sum()),
        "speed_denominator_cost_mean": float(frame["speed_denominator_cost"].mean()),
        "lateral_denominator_cost_mean": float(frame["lateral_denominator_cost"].mean()),
        "potential_denominator_cost_mean": float(frame["potential_denominator_cost"].mean()),
        "shapley_speed_loss_sum": float(frame["shapley_speed_loss"].sum()),
        "shapley_lateral_loss_sum": float(frame["shapley_lateral_loss"].sum()),
        "shapley_potential_loss_sum": float(frame["shapley_potential_loss"].sum()),
        "shapley_comfort_loss_sum": float(frame["shapley_comfort_loss"].sum()),
        "discounted_shapley_speed_loss_sum": float(
            frame["discounted_shapley_speed_loss"].sum()
        ),
        "discounted_shapley_lateral_loss_sum": float(
            frame["discounted_shapley_lateral_loss"].sum()
        ),
        "discounted_shapley_potential_loss_sum": float(
            frame["discounted_shapley_potential_loss"].sum()
        ),
        "discounted_shapley_comfort_loss_sum": float(
            frame["discounted_shapley_comfort_loss"].sum()
        ),
        "reconstruction_error_max": float(
            np.max(np.abs(frame["reward"] - frame["reward_reconstructed"]))
        ),
        "mean_speed": float(frame["ego_speed"].mean()),
        "mean_abs_speed_error": float(
            np.mean(np.abs(frame["ego_speed"] - frame["desired_speed"]))
        ),
        "mean_abs_lateral_action": float(np.mean(np.abs(frame["action_ay"]))),
        "mean_action_saturation": float(frame["near_saturation_fraction"].mean()),
        "exact_action_saturation": float(frame["exact_saturation_fraction"].mean()),
        "positive_ax_near_saturation_rate": float(frame["near_ax_positive_fraction"].mean()),
        "negative_ax_near_saturation_rate": float(frame["near_ax_negative_fraction"].mean()),
        "lateral_near_saturation_rate": float(frame["near_ay_saturation_fraction"].mean()),
        "near_zero_action_rate": float(frame["near_zero_action_fraction"].mean()),
        "lateral_action_sign_changes": int(
            np.sum(
                np.sign(frame["action_ay"].to_numpy()[1:])
                != np.sign(frame["action_ay"].to_numpy()[:-1])
            )
        )
        if len(frame) > 1
        else 0,
        "overtakes": float(frame["overtakes"].sum()),
        "reward_collision_trigger": int(
            frame["reward_collision_trigger"].max() > 0.5
        ),
        "collision": int(frame["distinct_collision_events"].sum() > 0),
        "distinct_collision_events": int(frame["distinct_collision_events"].sum()),
        "collision_step": collision_step,
        "collision_fraction_of_episode": (
            float(collision_step / len(frame)) if np.isfinite(collision_step) else np.nan
        ),
    }
    return summary, rows


def _plot_trajectories(trajectories: pd.DataFrame, output_path: Path) -> None:
    policies = list(trajectories["policy"].drop_duplicates())
    fig, axes = plt.subplots(4, len(policies), figsize=(7.0 * len(policies), 11.0), squeeze=False)
    for column, policy in enumerate(policies):
        subset = trajectories.loc[trajectories["policy"] == policy]
        for episode_index, episode in subset.groupby("episode_index", sort=True):
            label = f"ep {int(episode_index)}"
            axes[0, column].plot(episode["step"], episode["ego_speed"], label=label, alpha=0.8)
            axes[1, column].plot(episode["step"], episode["action_ax"], alpha=0.8)
            axes[1, column].plot(episode["step"], episode["action_ay"], linestyle="--", alpha=0.8)
            axes[2, column].plot(episode["step"], episode["ego_y"], alpha=0.8)
            axes[2, column].plot(episode["step"], episode["target_y"], linestyle="--", alpha=0.6)
            axes[3, column].plot(episode["step"], episode["main_reward"], alpha=0.7)
            axes[3, column].plot(episode["step"], episode["collision_penalty"], linestyle="--", alpha=0.8)
        axes[0, column].set_title(policy)
        axes[0, column].set_ylabel("ego speed (m/s)")
        axes[1, column].set_ylabel("ax solid / ay dashed")
        axes[2, column].set_ylabel("ego y solid / target y dashed")
        axes[3, column].set_ylabel("main reward solid / collision dashed")
        for row in range(4):
            axes[row, column].set_xlabel("policy step")
            axes[row, column].grid(alpha=0.25)
        if column == 0:
            axes[0, column].legend(fontsize=8, ncol=2)
    fig.suptitle("Nominal PPO reward/behavior audit — external CBF OFF")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    progression.protocol.set_stable_native_defaults()
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()
    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be positive")
    if not np.isfinite(float(args.gamma)) or not 0.0 < float(args.gamma) <= 1.0:
        raise ValueError("--gamma must be finite and in (0, 1]")
    run_dir = args.run_dir.resolve()
    config = _load_config(run_dir)
    model_path = (args.model_path or (run_dir / "model_final.zip")).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing PPO model: {model_path}")
    output_dir = (args.output_dir or (run_dir / "reward_audit_100ep")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not np.all(np.isfinite([args.constant_ax, args.constant_ay])):
        raise ValueError("constant action must be finite")
    if abs(float(args.constant_ax)) > 3.0 or abs(float(args.constant_ay)) > 3.0:
        raise ValueError("constant action must lie in the physical [-3, 3] bounds")

    project_root = progression.protocol.find_project_root(
        args.project_root or Path.cwd()
    )
    namespace = progression.protocol.bootstrap_notebook_namespace(project_root)
    progression.protocol.exec_required_notebook_cells(
        project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    _override_cbf_snapshot(namespace, config)
    model = progression.load_model("ppo_nominal", model_path, str(args.device))
    reward_config = {
        str(key): (
            value
            if isinstance(value, (bool, str))
            else float(value)
        )
        for key, value in config["reward_config"].items()
    }
    env_config = config["env_config"]
    correction_epsilon = float(config.get("correction_epsilon", 0.03))
    seeds = [int(args.seed_start) + index for index in range(int(args.episodes))]
    trajectories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for episode_index, scenario_seed in enumerate(seeds, start=1):
        ppo_summary, ppo_rows = _run_episode(
            namespace=namespace,
            env_config=env_config,
            reward_config=reward_config,
            correction_epsilon=correction_epsilon,
            gamma=float(args.gamma),
            episode_index=episode_index,
            scenario_seed=scenario_seed,
            policy_name="ppo_nominal_raw",
            action_provider=lambda observation: np.asarray(
                model.predict(observation, deterministic=True)[0]
            ),
        )
        constant_summary, constant_rows = _run_episode(
            namespace=namespace,
            env_config=env_config,
            reward_config=reward_config,
            correction_epsilon=correction_epsilon,
            gamma=float(args.gamma),
            episode_index=episode_index,
            scenario_seed=scenario_seed,
            policy_name="constant_ax0_ay0",
            action_provider=lambda _observation: np.asarray(
                [args.constant_ax, args.constant_ay], dtype=np.float32
            ),
        )
        summaries.extend([ppo_summary, constant_summary])
        trajectories.extend(ppo_rows)
        trajectories.extend(constant_rows)

    summary_frame = pd.DataFrame(summaries)
    trajectory_frame = pd.DataFrame(trajectories)
    summary_frame.to_csv(output_dir / "episode_summary.csv", index=False)
    trajectory_frame.to_csv(output_dir / "trajectory_trace.csv", index=False)

    aggregate_rows: list[dict[str, Any]] = []
    for policy, frame in summary_frame.groupby("policy", sort=False):
        aggregate_rows.append(
            {
                "policy": policy,
                "episodes": int(len(frame)),
                "mean_return": float(frame["return"].mean()),
                "sd_return": float(frame["return"].std(ddof=1)),
                "mean_discounted_return": float(frame["discounted_return"].mean()),
                "mean_steps": float(frame["steps"].mean()),
                "collision_episode_rate": float(frame["collision"].mean()),
                "collisions": int(frame["collision"].sum()),
                "distinct_collision_events": int(frame["distinct_collision_events"].sum()),
                "reward_collision_trigger_rate": float(
                    frame["reward_collision_trigger"].mean()
                ),
                "reward_collision_triggers": int(
                    frame["reward_collision_trigger"].sum()
                ),
                "mean_main_reward_sum": float(frame["main_reward_sum"].mean()),
                "mean_main_reward_before_collision_sum": float(
                    frame["main_reward_before_collision_sum"].mean()
                ),
                "mean_discounted_main_reward_before_collision_sum": float(
                    frame["discounted_main_reward_before_collision_sum"].mean()
                ),
                "mean_pre_collision_steps": float(frame["pre_collision_steps"].mean()),
                "mean_normal_steps_to_cancel_collision": float(
                    frame["normal_steps_to_cancel_collision"].mean()
                ),
                "mean_collision_penalty_to_pre_collision_main_ratio": float(
                    frame["collision_penalty_to_pre_collision_main_ratio"].mean()
                ),
                "mean_progress_reward_sum": float(frame["progress_reward_sum"].mean()),
                "mean_reciprocal_reward_sum": float(frame["reciprocal_reward_sum"].mean()),
                "mean_additive_speed_reward_sum": float(
                    frame["additive_speed_reward_sum"].mean()
                ),
                "mean_additive_lateral_reward_sum": float(
                    frame["additive_lateral_reward_sum"].mean()
                ),
                "mean_additive_risk_penalty_sum": float(
                    frame["additive_risk_penalty_sum"].mean()
                ),
                "mean_safety_ellipse_cost": float(frame["safety_ellipse_cost_mean"].mean()),
                "mean_safety_ttc_cost": float(frame["safety_ttc_cost_mean"].mean()),
                "mean_safety_potential_penalty_sum": float(
                    frame["safety_potential_penalty_sum"].mean()
                ),
                "mean_safety_ellipse_penalty_sum": float(
                    frame["safety_ellipse_penalty_sum"].mean()
                ),
                "mean_safety_ttc_penalty_sum": float(
                    frame["safety_ttc_penalty_sum"].mean()
                ),
                "mean_safety_total_penalty_sum": float(
                    frame["safety_total_penalty_sum"].mean()
                ),
                "mean_potential_shaping_reward_sum": float(
                    frame["potential_shaping_reward_sum"].mean()
                ),
                "mean_overtake_bonus_sum": float(frame["overtake_bonus_sum"].mean()),
                "mean_collision_penalty_sum": float(frame["collision_penalty_sum"].mean()),
                "mean_discounted_collision_penalty_sum": float(
                    frame["discounted_collision_penalty_sum"].mean()
                ),
                "mean_discounted_reciprocal_reward_sum": float(
                    frame["discounted_reciprocal_reward_sum"].mean()
                ),
                "mean_discounted_additive_speed_reward_sum": float(
                    frame["discounted_additive_speed_reward_sum"].mean()
                ),
                "mean_discounted_additive_lateral_reward_sum": float(
                    frame["discounted_additive_lateral_reward_sum"].mean()
                ),
                "mean_discounted_additive_risk_penalty_sum": float(
                    frame["discounted_additive_risk_penalty_sum"].mean()
                ),
                "mean_discounted_safety_ellipse_penalty_sum": float(
                    frame["discounted_safety_ellipse_penalty_sum"].mean()
                ),
                "mean_discounted_safety_ttc_penalty_sum": float(
                    frame["discounted_safety_ttc_penalty_sum"].mean()
                ),
                "mean_discounted_safety_total_penalty_sum": float(
                    frame["discounted_safety_total_penalty_sum"].mean()
                ),
                "mean_discounted_safety_potential_penalty_sum": float(
                    frame["discounted_safety_potential_penalty_sum"].mean()
                ),
                "mean_discounted_potential_shaping_reward_sum": float(
                    frame["discounted_potential_shaping_reward_sum"].mean()
                ),
                "mean_explicit_survival_reward_sum": float(frame["explicit_survival_reward_sum"].mean()),
                "mean_comfort_denominator_cost_sum": float(frame["comfort_denominator_cost_sum"].mean()),
                "mean_shapley_speed_loss_sum": float(frame["shapley_speed_loss_sum"].mean()),
                "mean_shapley_lateral_loss_sum": float(frame["shapley_lateral_loss_sum"].mean()),
                "mean_shapley_potential_loss_sum": float(frame["shapley_potential_loss_sum"].mean()),
                "mean_shapley_comfort_loss_sum": float(frame["shapley_comfort_loss_sum"].mean()),
                "mean_discounted_shapley_speed_loss_sum": float(
                    frame["discounted_shapley_speed_loss_sum"].mean()
                ),
                "mean_discounted_shapley_lateral_loss_sum": float(
                    frame["discounted_shapley_lateral_loss_sum"].mean()
                ),
                "mean_discounted_shapley_potential_loss_sum": float(
                    frame["discounted_shapley_potential_loss_sum"].mean()
                ),
                "mean_discounted_shapley_comfort_loss_sum": float(
                    frame["discounted_shapley_comfort_loss_sum"].mean()
                ),
                "mean_speed": float(frame["mean_speed"].mean()),
                "mean_abs_speed_error": float(frame["mean_abs_speed_error"].mean()),
                "mean_action_saturation": float(frame["mean_action_saturation"].mean()),
                "positive_ax_near_saturation_rate": float(frame["positive_ax_near_saturation_rate"].mean()),
                "negative_ax_near_saturation_rate": float(frame["negative_ax_near_saturation_rate"].mean()),
                "lateral_near_saturation_rate": float(frame["lateral_near_saturation_rate"].mean()),
                "near_zero_action_rate": float(frame["near_zero_action_rate"].mean()),
                "mean_overtakes": float(frame["overtakes"].mean()),
                "mean_collision_fraction_of_episode": float(frame["collision_fraction_of_episode"].mean()),
                "return_steps_correlation": float(
                    frame["return"].corr(frame["steps"])
                    if len(frame) > 1
                    else np.nan
                ),
            }
        )
    aggregate_frame = pd.DataFrame(aggregate_rows)
    aggregate_frame.to_csv(output_dir / "aggregate_summary.csv", index=False)
    _plot_trajectories(trajectory_frame, output_dir / "behavior_traces.png")
    (output_dir / "audit_config.json").write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "run_dir": str(run_dir),
                "external_cbf": "OFF",
                "scenario_seeds": seeds,
                "gamma": float(args.gamma),
                "constant_action_phys": [float(args.constant_ax), float(args.constant_ay)],
                "reward_config": reward_config,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("[reward-audit] aggregate summary", flush=True)
    print(aggregate_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)
    print(f"[reward-audit] outputs={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
