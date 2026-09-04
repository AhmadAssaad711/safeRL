# safeRL script and function reference

This document is the maintenance guide for the executable code that supports the
canonical laneless Karalakou experiment. The source-of-truth notebook is
[notebooks/lanelessKaralakou.ipynb](../notebooks/lanelessKaralakou.ipynb), and the
script inventory below describes how each module consumes or extends that
notebook.

The notebook owns the environment construction, reward definitions, CBF
geometry, baseline DDPG implementation, PPO training definitions, and the
canonical evaluation contract. Scripts are orchestration, experiment, audit,
plotting, or rendering layers around those definitions. When a script executes
notebook cells, that is deliberate: it keeps the experiment tied to the exact
notebook implementation and allows the notebook code hash to be recorded with
the outputs.

## How to read this reference

Every command-line script has a parse_args function for its CLI and a main
function for the executable entry point. The functions listed here are the
important public or experiment-facing functions. Private helpers with names
beginning with an underscore are included where they define a meaningful
protocol, data format, or safety calculation; small formatting and filesystem
helpers are grouped with their owning script.

The normal execution order is:

    notebook definitions -> environment or policy adapter -> training/evaluation
    -> registry and artifacts -> audit or plot/report

Run commands from the highway-rl-decision-making directory. Most scripts accept
--help before importing optional training dependencies.

    python scripts\mtm_laneless_smoke.py --help
    python scripts\evaluate_laneless_karalakou.py --help
    python scripts\run_ppo_cbf_progression.py --help

Do not launch the one-million-step PPO ladder merely to inspect a script. The
notebook's B.0 launch cell contains the long-run training guard; inspect the
configuration and use the smoke or evaluation entry points first.

## Repository layers

| Layer | Main modules | Responsibility |
| --- | --- | --- |
| Canonical experiment | notebooks/lanelessKaralakou.ipynb | Environment, reward, CBF, DDPG, PPO, and evaluation definitions |
| Notebook bridge | run_laneless_notebook_task.py, evaluate_laneless_karalakou.py, render_laneless_karalakou.py | Execute selected notebook cells with controlled overrides |
| Environment and observations | laneless_script_config.py, ppo_observation_variants.py, ppo_parallel_worker.py, ppo_cbf_env.py | Rebuild the canonical spaces and worker-safe observation/context wrappers |
| Safety projection | cbf_projection.py, cbf_ray_mask.py, guided_cbf_minimal.py | Convert CBF inequalities into safe physical actions and robust fallback actions |
| PPO methods | projected_ppo_cbf.py, run_ppo_cbf_progression.py, run_ppo_formulation_screen.py | Train and compare raw, detached, projected, and context-aware policies |
| DDPG methods | train_ddpg_*.py, ddpg_cbf_*.py, retrain_ddpg_stepwise_comparison.py | Train, sweep, and compare the legacy DDPG baselines and CBF variants |
| Protocol runners | run_nominal_ppo_parameter_pilot.py, run_nominal_ddpg_parameter_pilot.py, run_cbf_filter_ablation.py | Reproducible multi-seed training, checkpoints, paired evaluation, and factorial studies |
| Evaluation and audits | evaluate_*.py, audit_*.py, diagnose_update_signal_pipeline.py, compare_nominal_ppo_ddpg.py | Measure safety, task performance, interventions, geometry, and learning signals |
| Plotting and reporting | plot_*.py, build_*.py | Turn TensorBoard, CSV, JSON, and evaluation records into figures and reports |
| Live inspection | run_*_live.py, render_*.py | Short interactive rollouts and annotated videos; not the source of benchmark numbers |

## Canonical notebook contract

The notebook reference contains the exact values and cell order. Scripts should
not silently replace them with lane-indexed highway-v0 assumptions.

- Environment: lane-free-v0 with MTM traffic, a 380 by 10.2 m road, 55
  vehicles, five visible neighbors, and a 90 m sensing range.
- Timing: 100 Hz physics, 10 Hz policy decisions, and a 0.01 s simulation
  step.
- Base observation: five values per visible vehicle, with target-y and previous
  executed action wrappers producing the canonical PPO 32-dimensional input.
- Physical action: longitudinal acceleration and lateral target or control
  components bounded by the notebook configuration.
- Safety: HOCBF constraints are evaluated in physical action space and then
  mapped back to the policy action space.
- Evaluation: strict collision-free one-kilometre completion, 3,000 policy-step
  budget, and the ten canonical KPI fields. PPO reports distance completion in
  addition to the ten-field table.

See [lanelessKaralakou_reference.md](lanelessKaralakou_reference.md) for the
cell-by-cell notebook contract, formulas, variant names, and artifact rules.

## Notebook execution and live drivers

### run_laneless_notebook_task.py

This is the general-purpose notebook-cell runner. It is useful when a notebook
cell is the experiment unit but a repeatable command-line invocation is needed.

- find_project_root locates the repository from the script or current working
  directory.
- exec_notebook_cell, exec_notebook_cell_tail, and exec_notebook_cells load
  selected code cells into a shared namespace. The tail helper is useful for
  evaluating a final cell without replaying unrelated output cells.
- apply_overrides injects experiment-specific values such as model paths,
  seeds, traffic configuration, output stems, or training guards.
- snapshot_tensorboard_events captures the event-file state before and after a
  run so generated metrics can be associated with that run.
- normalize_artifact_suffix, apply_traffic_artifact_suffix, and
  _with_stem_suffix keep artifacts from different traffic or policy variants
  from overwriting one another.
- parse_args and main expose the selected cell range and overrides.

### evaluate_laneless_karalakou.py

This is the canonical notebook-backed evaluation entry point. It bootstraps the
notebook, applies stable native defaults and CBF overrides, evaluates one or
more models, and writes atomic JSON/CSV/TensorBoard outputs.

- set_stable_native_defaults makes evaluation deterministic and prevents
  accidental training-side defaults from changing the benchmark.
- exec_notebook_cell and exec_notebook_cells run the required notebook
  definitions in order.
- apply_cbf_overrides changes only explicitly requested CBF parameters.
- _initialize_laneless_eval_worker and
  _evaluate_laneless_episode_worker provide process-safe evaluation workers.
- _evaluate_laneless_with_workers coordinates parallel episodes while retaining
  deterministic seeds and stable output rows.
- ten_kpi_summary and print_ten_kpi_summary reduce episode records to the
  canonical KPI table.
- notebook_code_sha256 records the source identity, while artifact_path,
  with_stem_suffix, normalize_artifact_suffix, and atomic_write_metrics make
  result publication reproducible and interruption-safe.
- default_model_path resolves the expected saved model when the caller does not
  provide one.

### evaluate_kpi_final_episodes.py

This is a small evaluator for an already-created final-episode record. load_model
loads the requested policy, the shared metric helpers normalize numeric fields,
and write_tensorboard mirrors the final KPI rows into a compact event stream.
Use parse_args and main when a final checkpoint needs a narrow re-evaluation
without running the full notebook-backed report.

### render_laneless_karalakou.py

This renderer reuses the notebook environment and CBF definitions to produce a
single annotated rollout. set_stable_native_defaults and exec_notebook_cells
create the runtime; apply_cbf_overrides changes a requested safety setting;
artifact_path and suffix helpers select the output without changing the
benchmark naming convention.

### render_laneless_policy_scenario.py

This is the configurable single-scenario renderer used for targeted inspection.

- policy_for_variant selects a saved policy variant.
- apply_env_and_artifact_overrides changes scenario or output settings while
  preserving the canonical defaults unless explicitly overridden.
- make_live_env creates the environment for the chosen scenario.
- load_model loads the selected policy implementation.
- run_policy performs the rollout and writes frames or a summary.

### render_laneless_policy_videos.py

This renderer creates repeatable clips for a set of policy variants and traffic
scenarios.

- selected_scenarios defines the requested scenario subset.
- video_env_config and make_policy_env build per-scenario environments.
- load_models resolves all requested policy files once.
- rendered_frame and annotate_frame produce consistent visual frames.
- open_writer, write_repeated, and run_clip handle video encoding and repeated
  frame timing.
- slug, variant_file_name, and scenario_file_name create collision-resistant
  artifact names.

### run_current_mtm_live.py

This is a short live MTM environment runner. install_local_environment registers
the local environment, read_env_config loads the environment configuration,
apply_profile_mix selects traffic profiles, profile_counts reports the selected
mix, and main runs the interactive rollout. Its output is for sanity checking,
not for final KPI claims.

### run_mtm_y_target_live.py

This runner inspects target-y behavior. load_current_target_wrapper retrieves the
target-y observation wrapper, target_state extracts the relevant state, and
draw_frame renders the current geometry. It is useful for detecting an
observation or action-target mismatch before training.

### mtm_laneless_smoke.py

This is the smallest environment health check.

- find_project_root resolves the project location.
- make_env builds the canonical MTM environment.
- run_rollout performs a bounded rollout.
- summarize reports shape, bounds, collisions, and episode termination.
- plot optionally visualizes the short trajectory.

Use this before any long training or checkpoint evaluation.

### render_current_mtm.py

This renderer is the MTM counterpart to the smoke runner. install_local_environment
registers the local lane-free environment, read_env_config loads the selected
configuration, output_stem determines the artifact name, and main performs the
short rendered rollout. It is intended for visual environment validation rather
than benchmark evaluation.

## Configuration, observations, workers, and registries

### laneless_script_config.py

These functions provide one configuration path for scripts:

- load_json_object reads a JSON object from a path or inline value.
- deep_update applies nested, explicit overrides without discarding unrelated
  defaults.
- add_env_config_args adds shared environment arguments to a CLI parser.
- env_config_from_args merges CLI and JSON configuration.
- active_traffic_model reports the traffic model that is actually active after
  merging configuration.

### ppo_observation_variants.py

install_previous_action_observation installs the observation wrapper that
appends the previous executed normalized action. It is the adapter that turns
the target-y observation into the canonical PPO input and must be applied in
the same order during training and evaluation.

### ppo_parallel_worker.py

make_parallel_worker_env creates one worker environment with stable seeding and
canonical wrappers. make_parallel_subproc_training_env builds the vectorized
subprocess environment used by multi-worker PPO training. Worker construction
must remain import-safe because subprocesses do not share notebook state.

### ppo_cbf_env.py

This module carries CBF context through PPO observation and action boundaries.

- constraint_system_hash creates a stable identity for the active inequality
  system and is recorded with models or evaluation rows.
- CBFContextPhysicalActionWrapper exposes the physical CBF context while
  preserving the policy-facing action and observation contract.

### laneless_training_registry.py

The training registry turns a run into a self-describing artifact:

- make_run_tag creates the stable run identifier.
- _sha256_file and _copy_if_present capture source and configuration
  provenance.
- _copy_tensorboard_events and _tensorboard_archive_dir archive event files.
- _max_timestep and _load_latest identify the newest usable checkpoint.
- archive_training_outputs publishes the checkpoint, config, summaries, and
  TensorBoard outputs as one run record.

### laneless_evaluation_registry.py

The evaluation registry prevents stale or mismatched metrics from being treated
as current:

- LatestTrainingRun describes the newest completed training run.
- sha256_file and stable_json_digest identify source and configuration content.
- latest_completed_training selects a completed run.
- build_evaluation_request and evaluation_cache_paths define the request and
  cache identity.
- load_matching_evaluation accepts a cache only when its identity matches.
- write_evaluation_manifest records inputs and outputs.
- sync_metrics_to_requested_output copies verified metrics to the requested
  destination.

## CBF geometry and action projection

### cbf_projection.py

This is the reusable two-dimensional polytope projection implementation. It
operates in physical action coordinates and supports both NumPy evaluation and
Torch training paths.

- NumpyProjection2D and TorchProjection2D are small result containers for the
  projected action, feasibility, active constraints, and fallback status.
- CBFContextLayout describes how per-neighbor and global constraints are packed
  into an observation or wrapper context.
- _numpy_inputs, _numpy_action_bounds, _batched_torch_inputs, and
  _torch_action_bounds normalize bounds, constraint rows, and batch dimensions.
- max_constraint_violation_numpy computes the largest inequality violation.
- _enumerate_numpy_candidates checks box vertices and pairwise constraint
  intersections.
- _least_violating_grid_candidate and
  _torch_least_violating_grid_candidate provide bounded fallback searches when
  the feasible set is empty or a numerical solve fails.
- project_polytope_2d_numpy and project_polytope_2d_torch project the proposed
  action into the feasible polytope, preserving batch shape and diagnostics.
- projection_jacobian_from_active_rows returns the local active-set Jacobian
  used by differentiable policy variants.
- pad_cbf_context, append_cbf_context, split_cbf_context_numpy, and
  split_cbf_context_torch maintain the context packing contract.

### cbf_ray_mask.py

This module implements a ray-mask safety filter for comparison with the
polytope projection.

- _constraint_arrays, _append_box_constraints, _max_violation, and _is_feasible
  normalize and test the linear inequality system.
- build_cbf_action_constraints constructs the physical action constraints.
- choose_ray_center and _least_violating_center select a feasible or minimally
  violating center.
- ray_map_action maps a latent or normalized action along a safe ray.
- cbf_ray_mask_filter_2d is the functional filter entry point.
- RayMaskedSafetyFilteredAccelerationWrapper integrates the filter with the
  environment.
- install_ray_mask_cbf installs the wrapper into a notebook-created namespace.

### guided_cbf_minimal.py

This module contains the minimal guided-learning extension for DDPG.

- _projection_from_active_rows, _positive_kkt_support, and
  _score_constraint_violation interpret the projection geometry.
- _grid_least_violating_bounded_action,
  _continuous_least_violating_bounded_action,
  _linprog_minimax_bounded_action, and
  _soft_least_violating_bounded_action implement progressively more robust
  least-violating fallbacks.
- _local_projection_target and _actor_gradient_diagnostics create guidance
  targets and diagnostics.
- CBFGuidedReplayBufferSamples and CBFGuidedReplayBuffer retain the extra
  safety fields needed for guided updates.
- GuidedCBFDDPG adds the guided actor or critic update.
- _install_robust_cbf_fallback and _install_cbf_projection_reporting patch
  runtime behavior in the notebook namespace.
- install_minimal_guided_cbf is the public installation hook.

### CBF experiment scripts

| Script | Main functions and purpose |
| --- | --- |
| [train_cbf_damped_gain.py](../scripts/train_cbf_damped_gain.py) | set_cbf_gains and set_vec_cbf_gains apply gain overrides; make_eval_env builds the evaluation wrapper; evaluate_model and summarize measure the damped-gain candidate; DampedGainEvalCallback and plot_results support a sweep. |
| [cbf_damped_gain_eval_sweep.py](../scripts/cbf_damped_gain_eval_sweep.py) | coarse_damped_candidates and refined_damped_candidates generate the search grid; set_cbf_gains and evaluate_candidate run one point; summarize, plot_summary, and plot_summary_heatmap rank the sweep. |
| [cbf_eps_side_eval_sweep.py](../scripts/cbf_eps_side_eval_sweep.py) | parse_eps_values reads the epsilon grid; evaluate_eps runs one side-gain setting; summarize and plot_summary compare safety and intervention effects; write_outputs publishes rows. |
| [cbf_lambda_event_bc_pilot_sweep.py](../scripts/cbf_lambda_event_bc_pilot_sweep.py) | install_event_penalty_env installs the event penalty; evaluate_model measures one trial; PilotEvalCallback logs progress; trial_config_rows, summarize, score_summary, plot_trial_history, and plot_aggregate package the sweep. |
| [cbf_lambda_gradient_calibration.py](../scripts/cbf_lambda_gradient_calibration.py) | collect_diagnostic_batch and actor_grad_norm inspect gradient scale; bc_loss_from_batch and gradient_tables compare behavior-cloning and reward terms; reward_scale_tables and recommend_ranges turn measurements into candidate ranges; plot_outputs renders the calibration. |
| [cbf_reward_term_ablation.py](../scripts/cbf_reward_term_ablation.py) | make_reward_config isolates reward terms; make_single_env and make_training_env construct the trial; evaluate_model and summarize calculate behavior/safety metrics; RewardAblationEvalCallback and plot functions track the ablation. |
| [cbf_safety_obs_experiment.py](../scripts/cbf_safety_obs_experiment.py) | install_safety_observation_env adds the safety observation variant; evaluate_variant and SafetyObsEvalCallback measure it; plot_interpretability visualizes the added information. |
| [guided_cbf_lambda_ablation.py](../scripts/guided_cbf_lambda_ablation.py) | load_notebook_namespace and find_repo_root reconstruct the notebook runtime; evaluate_guided_policy measures one guidance weight; summarize_metrics reports the result; LambdaAblationCallback logs the sweep; trial_tag creates stable trial names. |
| [run_cbf_filter_ablation.py](../scripts/run_cbf_filter_ablation.py) | make_raw_env, make_cbf_env, and make_training_env construct the raw and filtered conditions; train_variant trains one condition; evaluate_scenario and evaluate_models run paired episodes; factorial_effects and paired_comparisons quantify the filter effect; checkpoint and wrapper-state functions make resume/evaluation reproducible. |
| [evaluate_cbf_filter_ablation.py](../scripts/evaluate_cbf_filter_ablation.py) | _make_ablation_env and _set_filter_info construct and instrument conditions; evaluate_condition runs episodes; part1_ten_kpi_table and part1_inline_kpi_table calculate the KPI table; make_figures creates the diagnostic figures. |
| [evaluate_filter_policy_contribution.py](../scripts/evaluate_filter_policy_contribution.py) | model_action_to_physical and physical_to_normalized make action conversions explicit; make_raw_eval_env and make_cbf_eval_env create paired environments; evaluate_one, aggregate_episode_rows, and compute_diagnostics measure raw, filtered, and rule baselines; write_tensorboard and plot_summary publish outputs. |

## PPO policy implementations and progression

### projected_ppo_cbf.py

This module contains the custom Stable-Baselines3-compatible PPO components.

- CBFBaseFeaturesExtractor handles the shared policy feature path.
- CBFSafetyRolloutBufferSamples and CBFSafetyRolloutBuffer store CBF context,
  action stages, and safety diagnostics alongside ordinary PPO rollout data.
- ProjectedPolicyEvaluation and DetachedPolicyEvaluation define the two
  policy-gradient treatments of the safety projection.
- DetachedCBFActorCriticPolicy and ProjectedCBFActorCriticPolicy are the
  actor-critic policy classes used by the corresponding variants.
- context_ignoring_policy_kwargs builds the raw or intentionally
  context-ignoring policy configuration.
- LatentActionPPO is the common PPO extension that carries latent, raw, and
  executed action stages.
- _gradient_pair_diagnostics measures the gradient relationship between the
  policy objective and the CBF guidance or projection path.
- DetachedCBFActorPPO and ProjectedCBFPPO are the concrete PPO trainers.

The projection is in physical action space. The policy still consumes and emits
the normalized action-space representation defined by the notebook and wrapper
configuration.

### run_ppo_cbf_progression.py

This is the main multi-variant PPO experiment runner.

- _base_observation_features and _base_observation_dim inspect the notebook
  observation contract.
- _ensure_ppo_observation_variant installs the selected previous-action or
  context variant.
- _deep_set_defaults, resolved_ppo_config, _effective_training_settings, and
  training_topology resolve the run without losing the canonical defaults.
- _cbf_training_snapshot and training_signature produce a configuration
  identity; _signature_path, _pending_signature_path, and _completion_path
  manage resumable run state.
- resolve_existing_variant_checkpoint and _latest_rollout_checkpoint locate a
  compatible checkpoint; _is_retryable_pending_run identifies incomplete runs.
- _base_environment, make_ppo_cbf_env, _make_ppo_worker_env, and
  make_training_vec_env create the training stack.
- build_model and load_model select or restore the PPO implementation.
- train_variant runs one named formulation and writes progress.
- make_evaluation_env, evaluate_scenario, evaluate_completed_episode, and
  evaluate_post_training_model implement post-training evaluation.
- _initialize_post_train_eval_worker, _evaluate_post_train_episode_worker,
  _evaluate_complete_episode_rows, and _ordered_episode_rows provide
  deterministic parallel post-training evaluation.
- _write_episode_progress_snapshot, _print_episode_progress,
  _record_post_training_evaluation, and _upsert_post_training_kpi_summary
  publish resumable evaluation progress.
- evaluate_raw_actor_ablation measures the actor without the training-time
  safety path when that comparison is requested.
- ten_kpi_table and factorial_effects summarize the final study.
- repair_post_training_summaries repairs an interrupted summary publication
  without retraining a model.

### run_ppo_formulation_screen.py

This runner screens alternative PPO action and observation formulations before
the expensive progression.

- boundary_state creates controlled boundary states.
- _semantic_observation and _formulation_components expose the meaning of each
  observation/action component.
- make_formulation_wrapper_class and make_formulation_namespace install a
  formulation in a notebook-backed environment.
- formulation_evaluation_action evaluates an action in physical coordinates.
- validate_formulation_action_space catches normalized/physical mismatches.
- formulation_config_payload records the formulation identity.
- rank_formulations and write_formulation_summaries compare candidates.

### PPO pilot scripts

| Script | Main functions and purpose |
| --- | --- |
| [run_nominal_ppo_parameter_pilot.py](../scripts/run_nominal_ppo_parameter_pilot.py) | build_ppo_model and effective_ppo_config define a pilot; train_one_run executes one seed/config; PPOEvaluationCallback, PPOActionClipCallback, and PPORolloutDiagnosticsCache collect progress; checkpoint helpers validate strict bundles; final_three_seed_averages, across_seed_final_three, rank_final_three, and write_summaries produce the selection report. |
| [run_nominal_ppo_density_pilot.py](../scripts/run_nominal_ppo_density_pilot.py) | ProgressCallback records training; _make_env and _make_vec_env create density conditions; evaluate_model and aggregate_evaluation compare episodes; main publishes the pilot table. |
| [evaluate_ppo_checkpoints.py](../scripts/evaluate_ppo_checkpoints.py) | _checkpoint_paths enumerates checkpoints; _load_run_config and _override_cbf_snapshot reconstruct the run; _summarize and _select_rows reduce checkpoint evaluations to comparable rows. |
| [evaluate_ppo_cbf_deployment.py](../scripts/evaluate_ppo_cbf_deployment.py) | _build_runtime reconstructs the deployment environment; _raw_final_rows selects terminal records; _summarize computes deployment metrics; _source_paths and _write_json preserve provenance. |
| [evaluate_ppo_cbf_counterfactuals.py](../scripts/evaluate_ppo_cbf_counterfactuals.py) | collect_state_candidates and _write_state_bank create fixed-state comparisons; evaluate_fixed_state_bank and summarize_fixed_actions compare raw, filtered, and executed actions; summarize_occupancy and make_plots report occupancy and action distributions. |

## DDPG baselines and legacy comparisons

The DDPG scripts remain because they are part of the canonical laneless
Karalakou comparison set. They use the notebook's DDPG implementation and
should not be confused with removed lane-indexed DQN material.

| Script | Main functions and purpose |
| --- | --- |
| [train_ddpg_y_target_50k.py](../scripts/train_ddpg_y_target_50k.py) | find_repo_root and load_notebook_namespace load the canonical implementation; main trains the target-y observation baseline; summarize reports its result. |
| [train_ddpg_ego_y_only_50k.py](../scripts/train_ddpg_ego_y_only_50k.py) | Loads the notebook DDPG stack and trains the ego-y-only observation variant; summarize creates the compact result. |
| [train_ddpg_cbf_500k.py](../scripts/train_ddpg_cbf_500k.py) | PersistentCBFEvalCallback evaluates during training; latest_checkpoint resolves recovery; summarize and plot_history publish progress; main runs the 500k CBF baseline. |
| [train_ddpg_cbf_ray_mask.py](../scripts/train_ddpg_cbf_ray_mask.py) | evaluate_ray_mask_policy measures the ray-mask filter; summarize and latest_checkpoint handle results and resume; main runs the comparison. |
| [ddpg_cbf_gain_sweep_50k.py](../scripts/ddpg_cbf_gain_sweep_50k.py) | set_cbf_gains applies each gain tuple; make_cbf_eval_env_for_gains and evaluate_cbf_policy_for_gains run it; action_trace and summarize_final expose action and KPI changes; GainSweepCallback records the sweep. |
| [ddpg_cbf_intervention_sweep_50k.py](../scripts/ddpg_cbf_intervention_sweep_50k.py) | set_cbf_params and set_vec_cbf_params apply scalar/vector settings; make_eval_env and evaluate_policy run episodes; action_trace and summarize report intervention behavior; SweepCallback records trials. |
| [ddpg_cbf_lambda005_experiment.py](../scripts/ddpg_cbf_lambda005_experiment.py) | _load_notebook_context and environment builders reconstruct the experiment; evaluate_policy_fixed_800_step_windows and evaluate_cbf_policy_with_paper_metrics implement the fixed evaluation protocol; CBFPaperMetricsCallback logs it; add_algorithm, rename_ddpg_history, evaluate_ddpg_final, and plot_paper_metrics build the comparison. |
| [retrain_ddpg_stepwise_comparison.py](../scripts/retrain_ddpg_stepwise_comparison.py) | make_baseline_env and make_cbf_env construct paired conditions; make_model and train_variant train them; StepwiseTrainingLogger records stepwise behavior; add_rolling_columns, plot_training_curves, plot_cbf_filter, and write_summary package the comparison. |
| [run_nominal_ddpg_parameter_pilot.py](../scripts/run_nominal_ddpg_parameter_pilot.py) | build_pilot_model and train_one_run run one controlled seed; OutputDirectoryRunLock prevents concurrent writers; PilotEvaluationCallback and RetainedStrictCheckpointCallback collect and retain evaluations; critic_diagnostics, build_critic_calibration_bins, and summarize_critic_calibration_samples inspect value learning; final_three_seed_averages, across_seed_final_three, rank_final_three, rank_confirmation_rollout, paired_seed_differences, and write_summaries select a configuration. |
| [compare_nominal_ppo_ddpg.py](../scripts/compare_nominal_ppo_ddpg.py) | build_common_formulation and validate_common_formulation ensure the methods share a fair contract; child_command and _run_child invoke method-specific pilots; verify_child_configs checks their payloads; aggregate_checkpoint, build_comparison_tables, and comparison_delta combine and compare the results. |

## Protocol runners and training internals

### run_nominal_ppo_parameter_pilot.py

The nominal PPO pilot is the controlled hyperparameter and seed runner. In
addition to the functions in the pilot table above:

- validate_training_device and configure_parallel_runtime make the worker and
  device choice explicit.
- default_tensorboard_root, _run_dir, and ppo_config_payload define stable
  output paths and metadata.
- validate_rollout_alignment checks that rollout, minibatch, and environment
  counts have the intended relationship.
- expected_checkpoint_steps, evaluation_steps, and
  checkpoint_evaluation_enabled define the checkpoint protocol.
- sb3_resume_learn_target_timesteps supports safe continuation to a target
  timestep without silently restarting.
- _latest_checkpoint_bundle and validate_ppo_checkpoint_bundle verify that a
  model and its metadata belong together.
- preflight_runs catches output collisions before training.

### run_nominal_ddpg_parameter_pilot.py

The DDPG pilot has the same reproducibility goals but also records critic
calibration. fixed_cbf_snapshot freezes the safety configuration,
SplitLearningRateDDPG carries the selected learning-rate split,
critic_diagnostics and aggregate_calibration_scenario_rows summarize value
estimates, and rank_confirmation_rollout checks the selected configuration on
an independent confirmation rollout.

### run_cbf_filter_ablation.py

This is the most complete protocol runner. Its state-capture and restore
functions snapshot wrapper state, monitor state, KPI state, environment RNG, and
space RNG. StrictCheckpointCallback writes a complete checkpoint bundle.
preflight_output checks that the requested output is safe to create.
policy_action_and_q_physical, policy_q_value, and discounted_return_to_go expose
action-value diagnostics. The final summarize_episodes,
summarize_paired_comparisons, and summarize_factorial_effects functions produce
within-seed, paired, and factorial summaries.

### train_safety_potential_variants.py

This runner tests reward and safety-potential variants around a shared baseline.
make_baseline_reward_config, make_cbf_reward_config, and
reward_config_for_variant create isolated reward configurations;
apply_guided_runtime_config adds optional guidance; train_variant runs one
variant; evaluate_paired_deployments compares variants at deployment;
evaluate_model, VariantEvalCallback, write_summarywriter_tensorboard_row, and
write_csv_resilient publish progress; export_videos creates optional
qualitative artifacts.

### run_ppo_cbf_progression.py and resume safety

The progression runner treats a run as valid only when its configuration
signature, checkpoint metadata, source identity, and post-training evaluation
agree. If a pending run has incompatible configuration, start a new output
directory or explicitly repair the summary; do not overwrite a completed run
to make it appear compatible.

## Audits, diagnostics, and comparisons

### evaluate_laneless_karalakou.py

Use this for the canonical ten-KPI report. It should be preferred over ad hoc
episode loops when a result is intended for comparison or publication.

### evaluate_cbf_counterfactuals.py

This evaluator compares actions at the same state across filter conditions.

- stable_state_hash gives a deterministic identity to a stored state.
- TypedConstraintSystem, make_typed_constraint_system, and
  typed_feasible_mask represent typed CBF inequalities.
- active_constraint_indices and normal_tangent_decomposition explain which
  constraints shape the action.
- time_to_contact and _candidate_memberships provide event and occupancy
  labels.
- stratify_state_bank and write_state_bank create the reusable fixed-state
  bank.
- evaluate_common_state_bank, summarize_fixed_state_actions,
  summarize_interventions_by_constraint, and
  compute_factorial_contrasts produce comparable counterfactual summaries.
- plot_fixed_state_distributions, plot_occupancy_distributions, and
  plot_critic_action_maps visualize the result.

### evaluate_cbf_filter_ablation and evaluate_filter_policy_contribution

These answer two related questions: what changes when a safety filter is
enabled, and how much of the deployed behavior is attributable to the learned
policy versus the filter. Keep raw, safe, and executed action stages separate
in the output; a filter contribution study is not reproducible if those stages
are overwritten.

### audit_mtm_native_baseline.py

This audit measures the native MTM controller without the learned policy.
_native_episode runs one deterministic native episode, _aggregate combines
episode rows, and _finite_mean makes missing or non-finite values explicit.

### audit_mtm_density_ladder.py

This audit evaluates the native controller across density settings.
_aggregate_density groups rows by density and _write_csv publishes the ladder
in a machine-readable form.

### audit_nominal_mtm_collision_provenance.py

This audit explains where a nominal MTM collision came from rather than only
counting it.

- _policy_dt and _bootstrap_nominal_namespace reconstruct the exact runtime.
- _snapshot captures the relevant pre-step state.
- _ego_index, _geometry, _initial_geometry, and _active_ego_pairs identify the
  ego and nearby collision geometry.
- _partner_index and _closing_rates identify the likely collision partner and
  approach rate.
- _action records the controller action.
- run_controller_provenance produces an event trace.
- _empty_collision_fields, _write_csv, _finite_mean, and _summary format the
  provenance report.

### audit_nominal_ppo_basics.py

This audit checks nominal PPO semantics before interpreting performance.
NominalPhysicalActionAdapter makes the physical action conversion explicit;
make_nominal_env constructs the test environment; _base, _state,
_distance_step, _collision_events, _ellipse_h, and _neighbor_visibility
calculate geometry and event facts; _observation_feature_names and
_controller_from_observation test semantic observation/controller alignment.
run_action_response, run_observation_audit, and run_controller are the three
audit sections.

### audit_nominal_reward.py

This audit decomposes the nominal reward and action behavior.
_load_config reads the run configuration, _override_cbf_snapshot applies a
controlled safety snapshot, _reward_terms extracts reward components,
_action_metrics measures action changes, _run_episode collects rows, and
_shapley_denominator_losses computes normalized contribution denominators.
_plot_trajectories provides the visual diagnostic.

### diagnose_update_signal_pipeline.py

This module follows the learning signal from raw policy action to filtered
action and critic value. model_action_from_physical, critic_q,
classify_projection, and evaluate_variant collect per-step diagnostics;
summarize_steps aggregates them; the plot_q_* functions,
plot_correction_distribution, plot_projection_buckets, and
plot_success_failure_split expose where the update signal may be weak or
contradictory.

### compare_nominal_ppo_ddpg.py

Use this only after both child experiments report compatible formulation
payloads. canonical_json and formulation_signature make configuration
comparison deterministic; build_common_formulation defines the shared
contract; verify_child_configs stops invalid comparisons; the table and delta
functions then report method-level differences.

## Plotting, dashboards, and reports

Plotting scripts read produced artifacts. They should not retrain a model or
silently regenerate missing evaluation data.

| Script | Function map |
| --- | --- |
| [plot_laneless_sample_efficiency.py](../scripts/plot_laneless_sample_efficiency.py) | RunSpec describes a run; load_run_scalars and load_training_logs read data; add_zero_start_anchor makes curves comparable; first_step_at_or_above and auc_over_logged_steps calculate sample-efficiency summaries; plot_episode_length, plot_training_return, plot_length_auc, and plot_steps_to_horizon render them. |
| [plot_ppo_progression_results.py](../scripts/plot_ppo_progression_results.py) | _concat_variant_files and _load_tb assemble variant inputs; plot_summary_bars and plot_episode_distributions show final results; plot_post_block_trends and plot_training_episodes show learning; plot_tb_scalars and plot_tb_cbf show logged diagnostics; plot_fixed_state_summary, plot_fixed_state_actions, plot_state_bank, plot_occupancy_summary, plot_occupancy_traces, and plot_active_constraints visualize counterfactual studies; inventory_page records inputs. |
| [plot_nominal_ppo_results.py](../scripts/plot_nominal_ppo_results.py) | _load_tensorboard loads scalars; _plot_kpi_table, _plot_kpi_bars, _plot_episode_distributions, _plot_block_trends, _plot_training_traces, and _plot_tensorboard create the nominal pilot pages; _inventory_page records file provenance. |
| [plot_ppo_500k_saved_results.py](../scripts/plot_ppo_500k_saved_results.py) | load_run reads a saved run; _read_events loads TensorBoard; plot_kpi_bars, plot_distributions, plot_block_trends, plot_training_traces, and plot_tb make the report; inventory_page and make_report package it. |
| [build_headline_kpi_dashboard.py](../scripts/build_headline_kpi_dashboard.py) | read_history, read_source_tensorboard_eval, and read_training_episode_tensorboard load compatible sources; load_train_eval and append_terminal_final_eval normalize final rows; write_tensorboard and write_clean_csvs publish clean data; plot_during_training, plot_training, and plot_final render the dashboard. |
| [build_filter_contribution_dashboard.py](../scripts/build_filter_contribution_dashboard.py) | read_training_episode_tensorboard and load_diagnostics load training and diagnostic streams; load_summary and row_value normalize summary fields; write_tensorboard, write_clean_csvs, plot_during_training, plot_headline, and plot_diagnostics publish the filter-contribution dashboard. |
| [build_cbf_ablation_report.py](../scripts/build_cbf_ablation_report.py) | StudyMetadata and _metadata capture study identity; load_inputs and _require_columns validate sources; _overview_metrics, _rollout_metric, _seed_means, _seed_sd, _normal_ci, and _factorial_ci calculate statistics; grouped_overview, factorial_figure, filter_dependence, paired_comparison_figure, table_page, _title_page, and build_report assemble the report; save writes it. |
| [build_paper_results.py](../scripts/build_paper_results.py) | discover_events and known_event_sources inventory TensorBoard; sha256, copy_file, and event_time preserve provenance; extract_scalars and summarize_final reduce events; write_filtered_trace and baseline_metadata create trace metadata; plot_bars and main package the laneless result bundle. |

The dashboard builders are downstream consumers. A missing metric should remain
missing or be reported as unavailable; it should not be silently replaced by a
different metric with a similar name.

## Rendering and scenario visualization

### render_policy_scenarios.py

This is the most detailed explanatory renderer for CBF behavior.

- VehicleSpec, ScenarioSpec, and PolicySpec are the typed descriptions of the
  scene, vehicle, and policy.
- make_scenarios defines the controlled scenarios.
- scenario_env_config and make_policy_env create the scenario runtime.
- apply_scenario installs initial vehicle states.
- policy_observation and normalized_to_physical expose the policy/action
  conversion.
- load_models loads the requested variants.
- initial_action_audit checks action semantics before a rollout.
- rollout_policy runs the scenario.
- apply_highway_vehicle_colors, draw_label, draw_arrow, draw_dashed_arrow,
  action_endpoint, cbf_effect_endpoint, draw_cbf_effect_legend, draw_scene,
  draw_action_arrows, and the plot functions create annotated frames that
  explain raw versus CBF-filtered action.

### Remaining renderers

- render_ppo_500k_nominal.py uses set_stable_native_defaults and
  make_render_env for a saved nominal PPO clip.
- render_ppo_at1_raw.py resolves a raw actor, builds the runtime, converts
  numbers with _as_number, and annotates the frame.
- render_ppo_at1_cbf.py builds the CBF render environment and writes a compact
  summary with _write_summary.
- render_ppo_ego16_live.py resolves a run, loads its configuration, builds the
  environment, and runs a live ego-target inspection.

## Function naming and data contracts

The following conventions are important when adding a new script:

1. Keep environment configuration in laneless_script_config or an explicit
   notebook override. Do not copy a second default configuration into a new
   script.
2. Keep physical and normalized actions named separately. Functions such as
   normalized_to_physical, physical_to_normalized, model_action_to_physical,
   and normalized_delta_norm exist to prevent accidental mixing.
3. Keep raw, safe, and executed action stages as separate fields in diagnostic
   output. A filter contribution study is not reproducible if those stages are
   overwritten.
4. Include seed, model variant, notebook source hash, CBF snapshot, environment
   configuration, and artifact paths in new result manifests.
5. Use atomic publication for final JSON/CSV summaries and retain partial
   progress files for long evaluations.
6. Reuse the canonical ten KPI names. Additional metrics, such as distance
   completion rate, should be additive rather than renaming an existing field.
7. Never use lane_index, highway-v0, or lane-indexed DQN helpers as a shortcut in
   a laneless experiment.

## Recommended workflows

### Environment smoke test

    python scripts\mtm_laneless_smoke.py --episodes 1

Then inspect the reported observation/action shapes, bounds, collision count,
and termination reason.

### Notebook-backed evaluation

    python scripts\evaluate_laneless_karalakou.py --help

Choose an existing model and an explicit output stem. The evaluator should
record the notebook hash and write a complete KPI summary before plotting.

### PPO progression

    python scripts\run_ppo_formulation_screen.py --help
    python scripts\run_ppo_cbf_progression.py --help

Screen semantics first, then run the named progression variant with a dedicated
output directory. Resume only when the stored training signature matches.

### DDPG comparison

    python scripts\run_nominal_ddpg_parameter_pilot.py --help
    python scripts\compare_nominal_ppo_ddpg.py --help

Run the DDPG pilot and its critic calibration before making a PPO/DDPG claim.
The comparison runner is the place to verify that action, observation, traffic,
and evaluation contracts are shared.

### Fixed-state safety analysis

    python scripts\evaluate_cbf_counterfactuals.py --help

Create or reuse a fixed state bank, compare raw and filtered actions at the same
states, and report typed active constraints and occupancy. Do not compare
independently sampled states as if they were counterfactual pairs.

### Reports

Run a plot or report builder only after its input manifest is complete. The
builder should be given explicit input paths when more than one experiment
directory exists.

## Tests and maintenance

The tests under [tests](../tests) are the first check for projection geometry,
observation dimensions, action conversion, deterministic state-bank behavior,
and protocol helpers. A normal change check is:

    python -m pytest -q tests

If the local Python environment cannot import the optional RL stack, still run
syntax and notebook-JSON checks and report the missing dependency rather than
claiming the behavioral suite passed.

When the notebook changes:

1. Update lanelessKaralakou_reference.md if the contract or cell order changed.
2. Re-run the smoke test and the focused projection/observation tests.
3. Re-run the relevant evaluator with a new artifact stem.
4. Check that the notebook source hash in the manifest changed as expected.
5. Update this script reference if a script's inputs, outputs, or function
   responsibilities changed.

The removed lane-indexed notebooks, DQN modules, duplicate machine-suffixed
files, scratch files, paper bundle, editor state, and tracked virtual
environment are intentionally outside this reference. Their historical commits
remain available through Git when provenance requires them.
