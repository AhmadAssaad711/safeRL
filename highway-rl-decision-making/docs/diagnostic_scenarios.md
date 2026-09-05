# Laneless diagnostic scenario registry

The canonical scenario names come from the E.2 section of
lanelessKaralakou.ipynb. They are qualitative probes for the three retained
legacy DDPG policies: plain DDPG, DDPG-CBF reward, and guided DDPG-CBF. They
are not substitutes for the paired stochastic KPI evaluation.

## Fixed scenes

| Scenario | Initial condition | Expected behavior |
| --- | --- | --- |
| Open passing gap | Slow leader ahead and a clear side gap behind it | Commit smoothly when passing is safe |
| Fast closing side vehicle | A tempting side gap contains a rapidly closing vehicle | Reject the gap rather than cut across it |
| Boxed in | No clean nearby lateral gap | Avoid forcing an overtake |
| Rear pressure escape | A fast rear vehicle closes on the ego | Escape through the cleaner side only |
| Boundary recovery | Ego starts near a road boundary with a blocker ahead | Recover inward while preserving clearance |
| Sudden lead slowdown | The leader slows sharply while both side gaps are risky | Brake or wait without diving into traffic |

## Common state-bank strata

The factorial counterfactual workflow uses the following state categories:

- normal: ordinary traffic with a feasible raw action;
- near-boundary: ego close to a lateral boundary;
- intervention: a state where the shield changes the proposed action;
- dense-traffic: multiple nearby vehicles and reduced clearances;
- overtaking: a state with a plausible pass and a meaningful relative-speed
  change.

The state bank is reconstructed deterministically from saved simulator state.
Each selected record should retain the observation, ego state, neighbors,
target speed, safety geometry, and stable state hash. The same records are
then sent through every actor and deployment condition.

## Interpretation

The six scenes reveal behavioral intent and failure modes. The common bank
reveals whether an actor's action map changed at the same state. On-policy
occupancy alone cannot establish internalization because a policy may simply
avoid states where its own raw action would be filtered.

For a reproducible comparison, keep fixed:

- scenario ordering and seeds;
- road, vehicle dimensions, and MTM profile mix;
- CBF gains, ellipse inflation, and action bounds;
- observation layout and target speed;
- render horizon and output suffix.

The implementation entry points are
`scripts/rendering/render_policy_scenarios.py` for annotated comparisons,
`scripts/rendering/render_laneless_policy_scenario.py` for one live scene, and
`scripts/evaluation/evaluate_cbf_counterfactuals.py` for the common-state
quantitative analysis.
