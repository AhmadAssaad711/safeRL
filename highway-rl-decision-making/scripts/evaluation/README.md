# Evaluation and diagnostics

Evaluation modules consume saved models or compact run outputs and publish
canonical KPI rows, audits, counterfactual tables, or diagnostic summaries.
They should not silently retrain a policy.

- `evaluate_laneless_karalakou.py`: canonical ten-KPI, strict one-kilometre
  notebook-backed evaluation.
- `evaluate_ppo_checkpoints.py`, `evaluate_ppo_cbf_deployment.py`, and
  `evaluate_ppo_cbf_counterfactuals.py`: checkpoint, deployment, and fixed-state
  PPO analyses.
- `evaluate_cbf_filter_ablation.py`, `evaluate_cbf_counterfactuals.py`, and
  `evaluate_filter_policy_contribution.py`: safety-filter and action-contribution
  analyses.
- `evaluate_kpi_final_episodes.py`: final episode KPI extraction.
- `audit_nominal_*` and `audit_mtm_*`: observation, reward, MTM, and collision
  provenance audits.
- `cbf_*_eval_sweep.py` and `cbf_lambda_gradient_calibration.py`: CBF tuning
  and gradient-scale evaluations.
- `compare_nominal_ppo_ddpg.py` and `diagnose_update_signal_pipeline.py`:
  cross-algorithm and learning-signal diagnostics.

Begin with `--help`, then use the output and seed arguments documented by the
specific module. The canonical output rules are in the
[notebook reference](../../docs/lanelessKaralakou_reference.md).
