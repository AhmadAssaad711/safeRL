# Training and experiment workflows

This group contains reproducible training protocols and controlled safeRL
experiments. Every runner should record its configuration, seed, notebook
identity, and output location before writing results.

## Main entry points

- `run_laneless_notebook_task.py`: execute notebook-backed DDPG/PPO tasks in a
  separate process.
- `run_ppo_cbf_progression.py`: compare raw, guided, projected, and
  context-aware PPO variants.
- `run_ppo_formulation_screen.py`: screen the fixed PPO reward/observation
  formulations.
- `run_nominal_ppo_parameter_pilot.py` and `run_nominal_ppo_density_pilot.py`:
  controlled PPO parameter and density pilots.
- `run_cbf_filter_ablation.py`: paired raw-versus-filtered CBF protocol.
- `run_nominal_ddpg_parameter_pilot.py`: controlled legacy DDPG confirmation.

## Supporting experiments

- `train_safety_potential_variants.py`, `cbf_lambda_event_bc_pilot_sweep.py`,
  `cbf_reward_term_ablation.py`, and `cbf_safety_obs_experiment.py` study
  reward, guidance, and observation variants.
- `train_cbf_damped_gain.py`, `ddpg_cbf_gain_sweep_50k.py`,
  `ddpg_cbf_intervention_sweep_50k.py`, and
  `ddpg_cbf_lambda005_experiment.py` study CBF parameter effects.
- `train_ddpg_*.py` and `retrain_ddpg_stepwise_comparison.py` preserve the
  notebook-aligned DDPG baselines and comparisons.
- `guided_cbf_lambda_ablation.py` evaluates detached actor-guidance weights.

See the [script/function reference](../../docs/script_reference.md) for
function-level details and the [notebook contract](../../docs/lanelessKaralakou_reference.md)
for fixed experiment assumptions.
