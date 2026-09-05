"""Stable mapping from legacy script filenames to organized module paths.

The canonical notebook invokes several helpers by filename. Keeping the map
in one small dependency-free module lets the notebook and external launchers
follow the organized package layout without duplicating path logic.
"""

from __future__ import annotations

from pathlib import Path


SCRIPT_MODULES: dict[str, str] = {
    "cbf_projection.py": "scripts.common.cbf_projection",
    "cbf_ray_mask.py": "scripts.common.cbf_ray_mask",
    "guided_cbf_minimal.py": "scripts.common.guided_cbf_minimal",
    "laneless_evaluation_registry.py": "scripts.common.laneless_evaluation_registry",
    "laneless_script_config.py": "scripts.common.laneless_script_config",
    "laneless_training_registry.py": "scripts.common.laneless_training_registry",
    "ppo_cbf_env.py": "scripts.common.ppo_cbf_env",
    "ppo_observation_variants.py": "scripts.common.ppo_observation_variants",
    "ppo_parallel_worker.py": "scripts.common.ppo_parallel_worker",
    "projected_ppo_cbf.py": "scripts.common.projected_ppo_cbf",
    "cbf_lambda_event_bc_pilot_sweep.py": "scripts.training.cbf_lambda_event_bc_pilot_sweep",
    "cbf_reward_term_ablation.py": "scripts.training.cbf_reward_term_ablation",
    "cbf_safety_obs_experiment.py": "scripts.training.cbf_safety_obs_experiment",
    "ddpg_cbf_gain_sweep_50k.py": "scripts.training.ddpg_cbf_gain_sweep_50k",
    "ddpg_cbf_intervention_sweep_50k.py": "scripts.training.ddpg_cbf_intervention_sweep_50k",
    "ddpg_cbf_lambda005_experiment.py": "scripts.training.ddpg_cbf_lambda005_experiment",
    "guided_cbf_lambda_ablation.py": "scripts.training.guided_cbf_lambda_ablation",
    "retrain_ddpg_stepwise_comparison.py": "scripts.training.retrain_ddpg_stepwise_comparison",
    "run_cbf_filter_ablation.py": "scripts.training.run_cbf_filter_ablation",
    "run_laneless_notebook_task.py": "scripts.training.run_laneless_notebook_task",
    "run_nominal_ddpg_parameter_pilot.py": "scripts.training.run_nominal_ddpg_parameter_pilot",
    "run_nominal_ppo_density_pilot.py": "scripts.training.run_nominal_ppo_density_pilot",
    "run_nominal_ppo_parameter_pilot.py": "scripts.training.run_nominal_ppo_parameter_pilot",
    "run_ppo_cbf_progression.py": "scripts.training.run_ppo_cbf_progression",
    "run_ppo_formulation_screen.py": "scripts.training.run_ppo_formulation_screen",
    "train_cbf_damped_gain.py": "scripts.training.train_cbf_damped_gain",
    "train_ddpg_cbf_500k.py": "scripts.training.train_ddpg_cbf_500k",
    "train_ddpg_cbf_ray_mask.py": "scripts.training.train_ddpg_cbf_ray_mask",
    "train_ddpg_ego_y_only_50k.py": "scripts.training.train_ddpg_ego_y_only_50k",
    "train_ddpg_y_target_50k.py": "scripts.training.train_ddpg_y_target_50k",
    "train_safety_potential_variants.py": "scripts.training.train_safety_potential_variants",
    "audit_mtm_density_ladder.py": "scripts.evaluation.audit_mtm_density_ladder",
    "audit_mtm_native_baseline.py": "scripts.evaluation.audit_mtm_native_baseline",
    "audit_nominal_mtm_collision_provenance.py": "scripts.evaluation.audit_nominal_mtm_collision_provenance",
    "audit_nominal_ppo_basics.py": "scripts.evaluation.audit_nominal_ppo_basics",
    "audit_nominal_reward.py": "scripts.evaluation.audit_nominal_reward",
    "cbf_damped_gain_eval_sweep.py": "scripts.evaluation.cbf_damped_gain_eval_sweep",
    "cbf_eps_side_eval_sweep.py": "scripts.evaluation.cbf_eps_side_eval_sweep",
    "cbf_lambda_gradient_calibration.py": "scripts.evaluation.cbf_lambda_gradient_calibration",
    "compare_nominal_ppo_ddpg.py": "scripts.evaluation.compare_nominal_ppo_ddpg",
    "diagnose_update_signal_pipeline.py": "scripts.evaluation.diagnose_update_signal_pipeline",
    "evaluate_cbf_counterfactuals.py": "scripts.evaluation.evaluate_cbf_counterfactuals",
    "evaluate_cbf_filter_ablation.py": "scripts.evaluation.evaluate_cbf_filter_ablation",
    "evaluate_filter_policy_contribution.py": "scripts.evaluation.evaluate_filter_policy_contribution",
    "evaluate_kpi_final_episodes.py": "scripts.evaluation.evaluate_kpi_final_episodes",
    "evaluate_laneless_karalakou.py": "scripts.evaluation.evaluate_laneless_karalakou",
    "evaluate_ppo_cbf_counterfactuals.py": "scripts.evaluation.evaluate_ppo_cbf_counterfactuals",
    "evaluate_ppo_cbf_deployment.py": "scripts.evaluation.evaluate_ppo_cbf_deployment",
    "evaluate_ppo_checkpoints.py": "scripts.evaluation.evaluate_ppo_checkpoints",
    "build_cbf_ablation_report.py": "scripts.reporting.build_cbf_ablation_report",
    "build_filter_contribution_dashboard.py": "scripts.reporting.build_filter_contribution_dashboard",
    "build_headline_kpi_dashboard.py": "scripts.reporting.build_headline_kpi_dashboard",
    "build_paper_results.py": "scripts.reporting.build_paper_results",
    "plot_laneless_sample_efficiency.py": "scripts.reporting.plot_laneless_sample_efficiency",
    "plot_nominal_ppo_results.py": "scripts.reporting.plot_nominal_ppo_results",
    "plot_ppo_500k_saved_results.py": "scripts.reporting.plot_ppo_500k_saved_results",
    "plot_ppo_progression_results.py": "scripts.reporting.plot_ppo_progression_results",
    "render_current_mtm.py": "scripts.rendering.render_current_mtm",
    "render_laneless_karalakou.py": "scripts.rendering.render_laneless_karalakou",
    "render_laneless_policy_scenario.py": "scripts.rendering.render_laneless_policy_scenario",
    "render_laneless_policy_videos.py": "scripts.rendering.render_laneless_policy_videos",
    "render_policy_scenarios.py": "scripts.rendering.render_policy_scenarios",
    "render_ppo_500k_nominal.py": "scripts.rendering.render_ppo_500k_nominal",
    "render_ppo_at1_cbf.py": "scripts.rendering.render_ppo_at1_cbf",
    "render_ppo_at1_raw.py": "scripts.rendering.render_ppo_at1_raw",
    "render_ppo_ego16_live.py": "scripts.rendering.render_ppo_ego16_live",
    "mtm_laneless_smoke.py": "scripts.ops.mtm_laneless_smoke",
    "run_current_mtm_live.py": "scripts.ops.run_current_mtm_live",
    "run_mtm_y_target_live.py": "scripts.ops.run_mtm_y_target_live",
}


def module_for_script(script_name: str) -> str:
    """Return the importable module for a filename or module-like name."""

    normalized = str(script_name).strip().replace("\\", "/")
    if normalized.startswith("scripts/"):
        return normalized[:-3].replace("/", ".") if normalized.endswith(".py") else normalized.replace("/", ".")
    if normalized.startswith("scripts."):
        return normalized
    if normalized in SCRIPT_MODULES:
        return SCRIPT_MODULES[normalized]
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    if "/" in normalized:
        candidate = normalized.replace("/", ".")
        if candidate.startswith("scripts."):
            return candidate
    raise KeyError(f"Unknown safeRL script: {script_name!r}")


def script_path(project_root: Path, script_name: str) -> Path:
    """Resolve an organized Python script path below ``project_root``."""

    module = module_for_script(script_name)
    parts = module.split(".")
    if parts[:1] != ["scripts"]:
        raise ValueError(f"Script module must live below scripts: {module}")
    return Path(project_root) / "scripts" / Path(*parts[1:]).with_suffix(".py")
