---
layout: page
title: Documentation
---

A short guide to the work and the code behind this page. The full,
maintained documentation lives in the
[repository](https://github.com/AhmadAssaad711/safeRL).

## The problem

Driving on a lane-free highway means choosing both a speed and a lateral
position at every moment, with no lanes to organize traffic. The ego vehicle
is a PPO policy. It outputs a longitudinal and a lateral acceleration 20 times
per second, while the simulator runs at 100 Hz. The surrounding 40 vehicles
follow MTM traffic dynamics, with personalities ranging from cautious to
aggressive. The road is a 380 m by 10.2 m loop.

## The safety shield

Safety is defined by a barrier function for every nearby vehicle pair and for
both road edges. Each vehicle is enclosed in an ellipse, and the barrier is
positive while the ego stays outside it. A high-order CBF (HOCBF) condition
turns each barrier into a linear constraint on the acceleration command. At
every policy step, the shield checks whether the policy's action satisfies
all of them. If not, it projects the action to the nearest one that does.

## The question

The shield guarantees nothing about what the policy itself learned. So every
policy is evaluated twice:

- **CBF-OFF**: the policy's own actions, with no shield.
- **CBF-ON**: the policy plus the deployed shield.

The policies differ only in how safety enters training, from not at all (A1),
through reward penalties (B1), to losses and projections that use the barrier
geometry directly (B2, C1, C2). A policy that has internalized safety should
stay safe with the shield off.

## How it is evaluated

- **Task:** drive 1,000 m without a collision within 3,000 policy steps. A
  collision ends the episode and counts as a failure.
- **Training:** seven seeds per policy (307 to 427 in steps of 20). The main
  five-policy study trains each policy for 1,000,000 steps.
- **Evaluation:** 200 episodes per seed in each mode. Tables report the mean
  and standard deviation across the seven seed means.
- **Safety KPIs:** collisions per km, minimum barrier value, QP failure rate,
  intervention rate and correction size.
- **Task KPIs:** return, episode length, completion rate, speed and lateral
  tracking error, and jerk.

## The code

| Part | Where |
| --- | --- |
| Environment | `safeRL_workspace/laneless highway env/lane_free_env.py` |
| Canonical experiment | `safeRL_workspace/notebooks/lanelessKaralakou.ipynb` |
| Configuration and new research code | `safeRL_workspace/src/saferl/` |
| Training runners | `safeRL_workspace/scripts/training/` |
| Evaluation and KPIs | `safeRL_workspace/scripts/evaluation/` |
| Tables and figures | `safeRL_workspace/scripts/reporting/` |
| Rollout videos | `safeRL_workspace/scripts/rendering/` |
| Tests | `safeRL_workspace/tests/` |

## Reproducing this page

Run from `safeRL_workspace`; each script documents its options under
`--help`.

| Output on this page | Command |
| --- | --- |
| Five-policy, seven-seed study | `python -m scripts.ops.run_safety_matrix_study` |
| Results tables | `python -m scripts.reporting.build_final_variant_comparison_tables` |
| Robustness plots | `python -m scripts.reporting.plot_hocbf_margin_aggregate` |
| Scenario videos | `python -m scripts.rendering.render_policy_scenarios` |

## Further reading

- [Experiment contract](https://github.com/AhmadAssaad711/safeRL/blob/main/safeRL_workspace/docs/lanelessKaralakou_reference.md):
  environment, reward, observation, CBF and evaluation in full.
- [Script and function reference](https://github.com/AhmadAssaad711/safeRL/blob/main/safeRL_workspace/docs/script_reference.md):
  which module to run for which task.
- [Diagnostic scenarios](https://github.com/AhmadAssaad711/safeRL/blob/main/safeRL_workspace/docs/diagnostic_scenarios.md):
  the fixed scenes used in the videos and plots.
- [Documentation index](https://github.com/AhmadAssaad711/safeRL/blob/main/safeRL_workspace/docs/README.md):
  every study and experiment log.
