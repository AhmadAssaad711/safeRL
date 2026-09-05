# Documentation index

This directory documents the safeRL lane-free experiment whose source of truth
is [`notebooks/lanelessKaralakou.ipynb`](../notebooks/lanelessKaralakou.ipynb).
The executable code is organized in [`scripts/README.md`](../scripts/README.md),
and the test layout is described in [`tests/README.md`](../tests/README.md).

## References

- [`lanelessKaralakou_reference.md`](lanelessKaralakou_reference.md): canonical
  environment, reward, observation, safety, training, and evaluation contract.
- [`script_reference.md`](script_reference.md): module-by-module function and
  workflow reference for the organized scripts.
- [`diagnostic_scenarios.md`](diagnostic_scenarios.md): fixed qualitative scenes
  and the common-state counterfactual protocol.
- [`cbf_factorial_ablation.md`](cbf_factorial_ablation.md): CBF filter ablation
  design, outputs, and interpretation.
- [`nominal_ppo_50k_pilot_handoff.md`](nominal_ppo_50k_pilot_handoff.md) and
  [`nominal_ddpg_confirmation_handoff.md`](nominal_ddpg_confirmation_handoff.md):
  reproducibility notes for the nominal pilot workflows.

## Command convention

Run commands from the `highway-rl-decision-making` directory. Python scripts
are package modules, for example:

```powershell
python -m scripts.ops.mtm_laneless_smoke --help
python -m scripts.evaluation.evaluate_laneless_karalakou --help
```

Experiment outputs belong under the repository-level
[`artifacts/`](../../artifacts/README.md) directory. Generated artifacts are
ignored by Git; only the directory guide is versioned.
