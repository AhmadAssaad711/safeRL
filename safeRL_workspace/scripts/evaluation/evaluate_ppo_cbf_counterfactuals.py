"""Common-state and occupancy diagnostics for the PPO-to-CBF progression.

The rollout comparison and the fixed-state comparison answer different
questions.  Shielded rollouts reveal which states each trained policy visits;
the common state bank holds the state fixed and reveals how its action map
changed.  PPO has a state-value critic, so this module deliberately does not
invent DDPG-style Q(s, a) contours.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th

from scripts.common.cbf_projection import (
    project_polytope_2d_numpy,
    split_cbf_context_numpy,
)
from scripts.evaluation.evaluate_cbf_counterfactuals import (
    _is_overtaking_state,
    normal_tangent_decomposition,
    occupancy_metrics,
    stable_state_hash,
    stratify_state_bank,
)


def _finite_mean(values: Any, default: float = np.nan) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float(default)


def _action_stages(model: Any, observation: np.ndarray) -> dict[str, np.ndarray]:
    stages = model.predict_action_stages(observation, deterministic=True)
    low = np.asarray(model.action_space.low, dtype=np.float32).reshape(2)
    high = np.asarray(model.action_space.high, dtype=np.float32).reshape(2)
    mu_raw = np.asarray(stages["mu_raw"], dtype=np.float32).reshape(2)
    mu_safe = np.asarray(stages["mu_safe"], dtype=np.float32).reshape(2)
    # This is the action executed in the RAW deployment mode.  For a projected
    # policy it includes the architectural mean projection but excludes the
    # final hard sample projection.  mu_raw is retained separately to measure
    # internalization by the underlying network.
    policy_action = np.clip(mu_safe, low, high).astype(np.float32)
    return {
        "mu_raw": mu_raw,
        "mu_safe": mu_safe,
        "policy_action": policy_action,
    }


def _distribution_parameters(
    model: Any, observation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    tensor, _ = model.policy.obs_to_tensor(observation)
    with th.no_grad():
        distribution = model.policy.get_distribution(tensor)
        normal = distribution.distribution
        mean = normal.mean.detach().cpu().numpy().reshape(2)
        std = normal.stddev.detach().cpu().numpy().reshape(2)
    return mean.astype(np.float32), std.astype(np.float32)


def _constraint_type(index: int, neighbor_count: int) -> str:
    if index < int(neighbor_count):
        return "neighbor"
    offset = int(index) - int(neighbor_count)
    labels = (
        "road_left",
        "road_right",
        "box_ax_max",
        "box_ax_min",
        "box_ay_max",
        "box_ay_min",
    )
    return labels[offset] if 0 <= offset < len(labels) else "unknown"


def _write_state_bank(bank: list[dict[str, Any]], output_dir: Path) -> None:
    records = []
    for item in bank:
        records.append(
            {
                "bank_index": int(item["bank_index"]),
                "state_hash": str(item["state_hash"]),
                "stratum": str(item["stratum"]),
                "categories": "|".join(map(str, item["categories"])),
                "source_training_seed": int(item["source_training_seed"]),
                "source_variant": str(item["source_variant"]),
                "scenario_seed": int(item["scenario_seed"]),
                "scenario_step": int(item["scenario_step"]),
                "h_min": float(item["h_min"]),
                "h_dot": float(item["h_dot"]),
                "ttc_s": float(item["ttc_s"]),
                "vehicle_spacing_m": float(item["vehicle_spacing_m"]),
                "traffic_density_per_km": float(item["traffic_density_per_km"]),
                "source_intervention": bool(item["intervention"]),
                "overtaking": bool(item["overtaking"]),
                "neighbor_count": int(item["neighbor_count"]),
            }
        )
    pd.DataFrame(records).to_csv(output_dir / "state_bank.csv", index=False)
    np.savez_compressed(
        output_dir / "state_bank_observations.npz",
        observations=np.stack(
            [np.asarray(item["observation"], dtype=np.float32) for item in bank]
        ),
        state_hashes=np.asarray([item["state_hash"] for item in bank]),
    )
    with (output_dir / "state_bank.jsonl").open("w", encoding="utf-8") as handle:
        for item in bank:
            payload = {
                "bank_index": int(item["bank_index"]),
                "state_hash": str(item["state_hash"]),
                "stratum": str(item["stratum"]),
                "categories": list(item["categories"]),
                "observation": np.asarray(item["observation"], dtype=float).tolist(),
                "ego": item["ego"],
                "neighbors": item["neighbors"],
                "road_width": float(item["road_width"]),
            }
            handle.write(json.dumps(payload, sort_keys=True, default=float) + "\n")


def collect_state_candidates(
    namespace: Mapping[str, Any],
    models: Mapping[tuple[int, str], Any],
    *,
    make_env: Callable[..., Any],
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    correction_epsilon: float,
    scenario_seeds: list[int],
    steps_per_scenario: int,
    neighbor_range: float,
    eps_side: float,
    k0: float,
    k1: float,
    ttc_cap: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    candidates: list[dict[str, Any]] = []
    occupancy_rows: list[dict[str, Any]] = []
    for (training_seed, variant), model in sorted(models.items()):
        for scenario_seed in scenario_seeds:
            env = make_env(
                namespace,
                mode="cbf",
                env_config=env_config,
                reward_config=reward_config,
                correction_epsilon=float(correction_epsilon),
            )
            try:
                observation, _ = env.reset(seed=int(scenario_seed))
                episode_index = 0
                for scenario_step in range(int(steps_per_scenario)):
                    ego = dict(namespace["get_ego_state"](env))
                    neighbors = [
                        dict(item)
                        for item in namespace["get_neighbor_states"](
                            env, neighbor_range=float(neighbor_range)
                        )
                    ]
                    road_width = float(env.unwrapped.config["road_width"])
                    stages = _action_stages(model, np.asarray(observation))
                    policy_action = stages["policy_action"]
                    _, rows, bounds, mask = split_cbf_context_numpy(observation)
                    projection = project_polytope_2d_numpy(
                        policy_action,
                        rows,
                        bounds,
                        mask,
                        action_low=model.action_space.low,
                        action_high=model.action_space.high,
                    )
                    half_range = np.maximum(
                        0.5
                        * (
                            np.asarray(model.action_space.high, dtype=float)
                            - np.asarray(model.action_space.low, dtype=float)
                        ),
                        1e-6,
                    )
                    correction = float(
                        np.linalg.norm((projection.action - policy_action) / half_range)
                    )
                    metrics = occupancy_metrics(
                        namespace,
                        ego,
                        neighbors,
                        road_width,
                        neighbor_range=float(neighbor_range),
                        eps_side=float(eps_side),
                        k0=float(k0),
                        k1=float(k1),
                        ttc_cap=float(ttc_cap),
                    )
                    next_observation, _, terminated, truncated, step_info = env.step(
                        policy_action
                    )
                    overtaking = _is_overtaking_state(ego, neighbors, step_info)
                    state_hash = stable_state_hash(
                        observation, ego, neighbors, road_width
                    )
                    common = {
                        "source_training_seed": int(training_seed),
                        "source_variant": str(variant),
                        "scenario_seed": int(scenario_seed),
                        "episode_index": int(episode_index),
                        "scenario_step": int(scenario_step),
                        "state_hash": state_hash,
                        **metrics,
                        "overtaking": bool(overtaking),
                        "intervention": bool(
                            correction > float(correction_epsilon)
                        ),
                        "correction_box_norm": correction,
                        "qp_success": bool(projection.feasible),
                        "fallback_used": bool(projection.fallback_used),
                    }
                    candidates.append(
                        {
                            **common,
                            "observation": np.asarray(
                                observation, dtype=np.float32
                            ).copy(),
                            "ego": ego,
                            "neighbors": neighbors,
                            "road_width": road_width,
                        }
                    )
                    occupancy_rows.append(common)
                    observation = next_observation
                    if terminated or truncated:
                        episode_index += 1
                        observation, _ = env.reset(
                            seed=int(scenario_seed) + 100_003 * episode_index
                        )
            finally:
                env.close()
    return candidates, pd.DataFrame(occupancy_rows)


def evaluate_fixed_state_bank(
    models: Mapping[tuple[int, str], Any],
    bank: list[dict[str, Any]],
    *,
    correction_epsilon: float,
    stochastic_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows_out: list[dict[str, Any]] = []
    for (training_seed, variant), model in sorted(models.items()):
        low = np.asarray(model.action_space.low, dtype=float).reshape(2)
        high = np.asarray(model.action_space.high, dtype=float).reshape(2)
        half_range = np.maximum(0.5 * (high - low), 1e-6)
        rng = np.random.default_rng(int(seed) + 1009 * int(training_seed))
        for state in bank:
            observation = np.asarray(state["observation"], dtype=np.float32)
            _, padded_rows, padded_bounds, mask = split_cbf_context_numpy(
                observation
            )
            active_mask = np.asarray(mask > 0.5, dtype=bool)
            system_rows = np.asarray(padded_rows[active_mask], dtype=float)
            system_bounds = np.asarray(padded_bounds[active_mask], dtype=float)
            stages = _action_stages(model, observation)
            mu_raw = stages["mu_raw"].astype(float)
            mu_safe = stages["mu_safe"].astype(float)
            policy_action = stages["policy_action"].astype(float)
            projection = project_polytope_2d_numpy(
                policy_action,
                system_rows,
                system_bounds,
                action_low=low,
                action_high=high,
            )
            hard_safe = projection.action.astype(float)
            delta = hard_safe - policy_action
            delta_scaled = delta / half_range
            active_indices = projection.active_indices.astype(int)
            active_rows = system_rows[active_indices]
            active_scaled = active_rows * half_range.reshape(1, 2)
            basis_source = "active_constraints"
            if active_rows.shape[0] == 0 and np.linalg.norm(delta_scaled) > 1e-10:
                active_rows = delta.reshape(1, 2)
                active_scaled = delta_scaled.reshape(1, 2)
                basis_source = "fallback_correction_direction"
            physical = normal_tangent_decomposition(delta, active_rows)
            scaled = normal_tangent_decomposition(delta_scaled, active_scaled)
            neighbor_count = int(state["neighbor_count"])
            types = tuple(
                _constraint_type(index, neighbor_count) for index in active_indices
            )
            active_type = "+".join(dict.fromkeys(types)) if types else "none"

            distribution_mean, distribution_std = _distribution_parameters(
                model, observation
            )
            sample_corrections: list[float] = []
            sample_interventions: list[float] = []
            sample_fallbacks: list[float] = []
            for latent_z in rng.normal(
                loc=distribution_mean,
                scale=distribution_std,
                size=(int(stochastic_samples), 2),
            ):
                sample_projection = project_polytope_2d_numpy(
                    latent_z,
                    system_rows,
                    system_bounds,
                    action_low=low,
                    action_high=high,
                )
                sample_correction = float(
                    np.linalg.norm(
                        (sample_projection.action - latent_z) / half_range
                    )
                )
                sample_corrections.append(sample_correction)
                sample_interventions.append(
                    float(sample_correction > float(correction_epsilon))
                )
                sample_fallbacks.append(float(sample_projection.fallback_used))

            internal_delta = mu_safe - mu_raw
            normal = np.asarray(physical["normal"], dtype=float)
            tangent = np.asarray(physical["tangent"], dtype=float)
            rows_out.append(
                {
                    "training_seed": int(training_seed),
                    "variant": str(variant),
                    "bank_index": int(state["bank_index"]),
                    "state_hash": str(state["state_hash"]),
                    "stratum": str(state["stratum"]),
                    "categories": "|".join(map(str, state["categories"])),
                    "mu_raw_ax": float(mu_raw[0]),
                    "mu_raw_ay": float(mu_raw[1]),
                    "mu_safe_ax": float(mu_safe[0]),
                    "mu_safe_ay": float(mu_safe[1]),
                    "policy_action_ax": float(policy_action[0]),
                    "policy_action_ay": float(policy_action[1]),
                    "safe_ax": float(hard_safe[0]),
                    "safe_ay": float(hard_safe[1]),
                    "internal_mean_delta_ax": float(internal_delta[0]),
                    "internal_mean_delta_ay": float(internal_delta[1]),
                    "internal_mean_correction_norm": float(
                        np.linalg.norm(internal_delta / half_range)
                    ),
                    "delta_ax": float(delta[0]),
                    "delta_ay": float(delta[1]),
                    "correction_physical_norm": float(np.linalg.norm(delta)),
                    "correction_box_norm": float(np.linalg.norm(delta_scaled)),
                    "intervention": bool(
                        np.linalg.norm(delta_scaled) > float(correction_epsilon)
                    ),
                    "raw_policy_feasible": bool(
                        np.max(system_rows @ policy_action - system_bounds)
                        <= 1e-6
                    ),
                    "qp_success": bool(projection.feasible),
                    "fallback_used": bool(projection.fallback_used),
                    "active_constraint_type": active_type,
                    "active_constraint_count": int(active_rows.shape[0]),
                    "decomposition_basis_source": basis_source,
                    "normal_delta_ax": float(normal[0]),
                    "normal_delta_ay": float(normal[1]),
                    "normal_correction_physical_norm": float(
                        physical["normal_norm"]
                    ),
                    "tangent_delta_ax": float(tangent[0]),
                    "tangent_delta_ay": float(tangent[1]),
                    "tangent_correction_physical_norm": float(
                        physical["tangent_norm"]
                    ),
                    "normal_correction_box_norm": float(scaled["normal_norm"]),
                    "tangent_correction_box_norm": float(scaled["tangent_norm"]),
                    "latent_std_ax": float(distribution_std[0]),
                    "latent_std_ay": float(distribution_std[1]),
                    "sample_intervention_probability": _finite_mean(
                        sample_interventions, 0.0
                    ),
                    "sample_mean_correction_box_norm": _finite_mean(
                        sample_corrections, 0.0
                    ),
                    "sample_fallback_probability": _finite_mean(
                        sample_fallbacks, 0.0
                    ),
                    "state_h_min": float(state["h_min"]),
                    "state_h_dot": float(state["h_dot"]),
                    "state_ttc_s": float(state["ttc_s"]),
                    "state_vehicle_spacing_m": float(
                        state["vehicle_spacing_m"]
                    ),
                    "state_traffic_density_per_km": float(
                        state["traffic_density_per_km"]
                    ),
                }
            )
    return pd.DataFrame(rows_out).sort_values(
        ["training_seed", "variant", "bank_index"]
    )


def summarize_fixed_actions(actions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frames = [("all", actions)] + [
        (str(stratum), group)
        for stratum, group in actions.groupby("stratum", sort=True)
    ]
    for stratum, frame in frames:
        for (training_seed, variant), group in frame.groupby(
            ["training_seed", "variant"], sort=True
        ):
            rows.append(
                {
                    "stratum": stratum,
                    "training_seed": int(training_seed),
                    "variant": str(variant),
                    "states": int(len(group)),
                    "mean_policy_ax": float(group["policy_action_ax"].mean()),
                    "mean_policy_ay": float(group["policy_action_ay"].mean()),
                    "intervention_probability": float(group["intervention"].mean()),
                    "mean_external_correction_box_norm": float(
                        group["correction_box_norm"].mean()
                    ),
                    "mean_internal_mean_correction_norm": float(
                        group["internal_mean_correction_norm"].mean()
                    ),
                    "mean_normal_correction_box_norm": float(
                        group["normal_correction_box_norm"].mean()
                    ),
                    "mean_tangent_correction_box_norm": float(
                        group["tangent_correction_box_norm"].mean()
                    ),
                    "raw_policy_feasible_probability": float(
                        group["raw_policy_feasible"].mean()
                    ),
                    "sample_intervention_probability": float(
                        group["sample_intervention_probability"].mean()
                    ),
                    "sample_mean_correction_box_norm": float(
                        group["sample_mean_correction_box_norm"].mean()
                    ),
                    "fallback_probability": float(group["fallback_used"].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_occupancy(occupancy: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "h_min",
        "h_dot",
        "ttc_s",
        "vehicle_spacing_m",
        "traffic_density_per_km",
        "correction_box_norm",
        "intervention",
        "fallback_used",
    )
    rows: list[dict[str, Any]] = []
    for (seed, variant), group in occupancy.groupby(
        ["source_training_seed", "source_variant"], sort=True
    ):
        row: dict[str, Any] = {
            "training_seed": int(seed),
            "variant": str(variant),
            "steps": int(len(group)),
        }
        for metric in metrics:
            row[f"mean_{metric}"] = float(
                pd.to_numeric(group[metric], errors="coerce").mean()
            )
        row["minimum_h_min"] = float(
            pd.to_numeric(group["h_min"], errors="coerce").min()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    array = np.sort(array[np.isfinite(array)])
    if not len(array):
        return np.asarray([]), np.asarray([])
    return array, np.arange(1, len(array) + 1, dtype=float) / len(array)


def make_plots(
    actions: pd.DataFrame,
    occupancy: pd.DataFrame,
    bank: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    variants = list(dict.fromkeys(actions["variant"].astype(str)))
    columns = (
        ("policy_action_ax", "Policy longitudinal action"),
        ("policy_action_ay", "Policy lateral action"),
        ("correction_box_norm", "External correction norm"),
        ("normal_correction_box_norm", "Unsafe-normal correction"),
        ("internal_mean_correction_norm", "Internal mean projection"),
        ("sample_intervention_probability", "Sample projection probability"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (column, title) in zip(axes.ravel(), columns):
        arrays = [
            pd.to_numeric(
                actions.loc[actions["variant"].eq(variant), column],
                errors="coerce",
            ).dropna()
            for variant in variants
        ]
        ax.boxplot(arrays, tick_labels=variants, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "fixed_state_action_distributions.png", dpi=180)
    plt.close(fig)

    occupancy_columns = (
        ("h_min", "Minimum barrier h"),
        ("h_dot", "Barrier derivative"),
        ("ttc_s", "TTC (s)"),
        ("vehicle_spacing_m", "Vehicle spacing (m)"),
        ("traffic_density_per_km", "Traffic density (/km)"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (column, title) in zip(axes.ravel(), occupancy_columns):
        for variant in variants:
            x, y = _ecdf(
                occupancy.loc[occupancy["source_variant"].eq(variant), column]
            )
            ax.plot(x, y, label=variant)
        ax.set_title(title)
        ax.set_ylabel("ECDF")
        ax.grid(alpha=0.2)
    axes.ravel()[-1].axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "on_policy_occupancy_distributions.png", dpi=180)
    plt.close(fig)

    selected_states = bank[: min(6, len(bank))]
    if selected_states:
        figure, axes = plt.subplots(
            2, 3, figsize=(15, 9), squeeze=False
        )
        colors = dict(zip(variants, plt.get_cmap("tab10").colors[: len(variants)]))
        for ax, state in zip(axes.ravel(), selected_states):
            _, padded_rows, padded_bounds, mask = split_cbf_context_numpy(
                state["observation"]
            )
            system_rows = padded_rows[mask > 0.5]
            system_bounds = padded_bounds[mask > 0.5]
            grid = np.linspace(-3.0, 3.0, 121)
            gx, gy = np.meshgrid(grid, grid)
            points = np.stack([gx.ravel(), gy.ravel()], axis=1)
            feasible = np.all(
                points @ system_rows.T <= system_bounds.reshape(1, -1) + 1e-6,
                axis=1,
            ).reshape(gx.shape)
            ax.contourf(
                gx,
                gy,
                feasible.astype(float),
                levels=[-0.1, 0.5, 1.1],
                colors=["#f4cccc", "#d9ead3"],
                alpha=0.65,
            )
            state_rows = actions[actions["bank_index"].eq(state["bank_index"])]
            for variant, group in state_rows.groupby("variant", sort=False):
                row = group.iloc[0]
                ax.scatter(
                    row["policy_action_ax"],
                    row["policy_action_ay"],
                    color=colors[str(variant)],
                    label=str(variant),
                    s=28,
                )
                ax.plot(
                    [row["policy_action_ax"], row["safe_ax"]],
                    [row["policy_action_ay"], row["safe_ay"]],
                    color=colors[str(variant)],
                    linewidth=1.2,
                )
            ax.set_title(f"{state['stratum']} | bank {state['bank_index']}")
            ax.set_xlim(-3.0, 3.0)
            ax.set_ylim(-3.0, 3.0)
            ax.set_xlabel("a_x")
            ax.set_ylabel("a_y")
            ax.grid(alpha=0.2)
        for ax in axes.ravel()[len(selected_states) :]:
            ax.axis("off")
        axes.ravel()[0].legend(fontsize=6)
        figure.tight_layout()
        figure.savefig(output_dir / "fixed_state_feasible_action_maps.png", dpi=180)
        plt.close(figure)


def run_counterfactual_analysis(
    namespace: Mapping[str, Any],
    model_paths: Mapping[tuple[int, str], Path],
    *,
    load_model: Callable[[str, Path, str], Any],
    make_env: Callable[..., Any],
    device: str,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    correction_epsilon: float,
    output_dir: Path,
    scenario_seeds: list[int],
    steps_per_scenario: int,
    states_per_stratum: int,
    stochastic_samples: int,
    neighbor_range: float,
    eps_side: float,
    k0: float,
    k1: float,
    ttc_cap: float,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = {
        key: load_model(key[1], path, device)
        for key, path in model_paths.items()
    }
    candidates, occupancy = collect_state_candidates(
        namespace,
        models,
        make_env=make_env,
        env_config=env_config,
        reward_config=reward_config,
        correction_epsilon=float(correction_epsilon),
        scenario_seeds=list(map(int, scenario_seeds)),
        steps_per_scenario=int(steps_per_scenario),
        neighbor_range=float(neighbor_range),
        eps_side=float(eps_side),
        k0=float(k0),
        k1=float(k1),
        ttc_cap=float(ttc_cap),
    )
    bank, coverage = stratify_state_bank(
        candidates,
        int(states_per_stratum),
        seed=int(seed),
        near_boundary_margin=0.5,
        dense_threshold=None,
    )
    if not bank:
        raise RuntimeError("Counterfactual state bank is empty")
    _write_state_bank(bank, output_dir)
    occupancy.to_csv(output_dir / "on_policy_occupancy.csv", index=False)
    actions = evaluate_fixed_state_bank(
        models,
        bank,
        correction_epsilon=float(correction_epsilon),
        stochastic_samples=int(stochastic_samples),
        seed=int(seed),
    )
    actions.to_csv(output_dir / "fixed_state_actions.csv", index=False)
    fixed_summary = summarize_fixed_actions(actions)
    fixed_summary.to_csv(output_dir / "fixed_state_summary.csv", index=False)
    occupancy_summary = summarize_occupancy(occupancy)
    occupancy_summary.to_csv(output_dir / "occupancy_summary.csv", index=False)
    active_distribution = pd.crosstab(
        actions["variant"],
        actions["active_constraint_type"],
        normalize="index",
    )
    active_distribution.to_csv(output_dir / "active_constraint_distribution.csv")
    make_plots(actions, occupancy, bank, output_dir)
    metadata = {
        "state_bank_size": int(len(bank)),
        "candidate_count": int(len(candidates)),
        "coverage": coverage,
        "stochastic_samples_per_state": int(stochastic_samples),
        "raw_semantics": (
            "policy_action is mu_safe for projected policies; mu_raw is logged "
            "separately to measure underlying-network internalization"
        ),
        "critic_visualization": (
            "not applicable: PPO uses V(s), not an action-value Q(s,a) critic"
        ),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# PPO CBF counterfactual analysis\n\n"
        "`on_policy_occupancy.csv` answers which states each policy visits. "
        "`fixed_state_actions.csv` passes the exact same bank through every "
        "actor and separates the unprojected network mean, projected policy "
        "mean, and final hard correction. Normal/tangent corrections and "
        "stochastic latent-sample projection rates are reported explicitly.\n\n"
        "PPO has a state-value critic V(s), so DDPG-style Q(s,a) contours are "
        "not mathematically available; feasible-region action maps are used "
        "instead.\n",
        encoding="utf-8",
    )
    print("\n[ppo-counterfactual] fixed-state summary", flush=True)
    print(
        fixed_summary.loc[fixed_summary["stratum"].eq("all")].to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        ),
        flush=True,
    )
    print("\n[ppo-counterfactual] occupancy summary", flush=True)
    print(
        occupancy_summary.to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        ),
        flush=True,
    )
    return metadata
