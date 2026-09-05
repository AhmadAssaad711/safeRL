# CBF reward-by-actor-loss ablation

This workflow separates two questions:

1. Does a treatment change which states the policy visits?
2. Does it change the raw action chosen at the same state?

The causal comparison is the filtered 2×2 study below. The nominal and random
policies are contextual controls, not factorial cells.

| Variant | Reward penalty | Actor CBF loss | Role |
| --- | --- | --- | --- |
| `b_filtered` | Off | Off | Shield-only reference |
| `c_reward` | On | Off | Reward effect |
| `d_loss` | Off | On | Actor-loss effect |
| `e_reward_actor` | On | On | Joint treatment / interaction |

All four factorial cells use the same guided learner and replay-buffer
plumbing. In loss-off cells, the effective actor-loss coefficient is zero.

The canonical fixed-scene and common-state comparison definitions are recorded
in [`diagnostic_scenarios.md`](diagnostic_scenarios.md). The six-scene core is
the recommended qualitative suite for comparing learned policies; the
normal, near-boundary, intervention, dense-traffic, and overtaking strata are
the recommended common state bank for same-state actor and critic analysis.

## 1. Train and evaluate raw and shielded deployment

From the repository root, run the pre-registered study with an explicit output
directory. Every variant is paired by training seed and evaluated on the same
scenario seeds in both raw and shielded modes.

```powershell
python -m scripts.training.run_cbf_filter_ablation `
  --project-root . `
  --output-dir artifacts/cbf_factorial_200k `
  --timesteps 200000 `
  --checkpoint-interval 5000 `
  --seeds 307 1307 2307 `
  --eval-scenarios 10 `
  --eval-timesteps 800
```

The runner records collisions/km, return, tracking errors, minimum safety
value, time near the boundary, state-occupancy statistics, intervention rate,
and action-correction magnitude. Linearized TTC is capped (30 s by default),
and an already violated barrier has TTC zero. In raw mode, `IR` and `mean_delta_a` describe
the correction actually applied and are therefore zero. The corresponding
`shadow_*` fields evaluate what the CBF would have done without changing the
rollout.

The environment can add a lateral road-boundary force after the policy/CBF
command. Accordingly, the step-level artifacts keep raw policy action, CBF-safe
command, and actually executed acceleration distinct.

## 2. Evaluate every actor on one common state bank

```powershell
python -m scripts.evaluation.evaluate_cbf_counterfactuals `
  --study-dir artifacts/cbf_factorial_200k `
  --collector-scenarios 2 `
  --collector-steps 400 `
  --states-per-category 80
```

The analyzer collects a deterministic common bank spanning normal,
near-boundary, intervention, dense-traffic, and overtaking states. It then
passes exactly the same observations and reconstructed simulator states through
every factorial actor.

Key outputs in `counterfactual_analysis/` are:

- `state_bank_strata_coverage.csv`: requested, available, and selected states,
  including explicit shortfalls or missing strata.
- `fixed_state_actions.csv`: raw/safe `a_x`, `a_y`, signed corrections,
  intervention indicators, active constraints, feasibility, and normal versus
  tangential correction components for every actor-state pair.
- `fixed_state_intervention_summary.csv` and
  `fixed_state_factorial_effects_summary.csv`: same-state summaries and causal
  contrasts by bank stratum.
- `occupancy_steps.csv` and `occupancy_factorial_effects_summary.csv`: on-policy
  distributions and contrasts for `h_min`, `h_dot`, TTC, spacing, and traffic
  density.
- `fixed_state_raw_action_distributions.png`,
  `fixed_state_correction_distributions.png`, and
  `active_constraint_types.png`: action and correction decompositions.
- `critic_action_map_*.png`: critic contours, CBF-feasible action region,
  actor actions from all four cells, and each selected actor's CBF projection.

Critic values should be interpreted within a trained model, not compared as an
absolute scale across reward variants. The CBF normal/tangential decomposition
uses tight constraints with a positive KKT contribution, excluding coincident
zero-multiplier rows; when no valid active set is reported, the analyzer flags
and documents its correction-direction fallback.

## 3. Build the deployment report

```powershell
python -m scripts.reporting.build_cbf_ablation_report `
  --study-dir artifacts/cbf_factorial_200k
```

The report presents the four factorial cells, reward and actor-loss main
effects, their interaction, raw-versus-shielded deployment, filter load, and
paired-seed effects.

## Interpretation guardrail

A correction reduction in `fixed_state_*` means the actor's action map changed
on common states. A correction reduction found only in `occupancy_*` or each
policy's own rollout means the policy may instead have learned to avoid
intervention-heavy states. Raw safety improvements indicate internalization;
shielded-only improvements can be produced by the runtime filter.

This distinction matters because replay stores the nominal RL action while the
transition is generated by the shielded command. The intervention reward helps
repair the critic's attribution, whereas the actor CBF loss directly changes
the action map. The loss evaluates a first-order local CBF projection at the
current actor action, cancels replayed tangential exploration through the saved
projection Jacobian, and becomes the identity after the actor moves to the
recorded feasible side. Training logs therefore include separate critic-driven
and CBF actor-gradient norms, their norm ratio, cosine similarity on valid
batches, and the cosine-validity rate.
