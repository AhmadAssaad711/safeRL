# safeRL repository guidance

This file is the persistent entry point for Codex when working anywhere in
this repository. Keep it concise and update it when a recurring wrong
assumption is discovered. The detailed, source-grounded references are under
`safeRL_workspace/docs/`.

## Purpose and source of truth

This repository contains the safe reinforcement-learning research code for a
lane-free highway environment inspired by the Karalakou experiment. The Git
root is the parent directory of `safeRL_workspace`; the implementation and
Python package live under `safeRL_workspace/`.

The canonical source of experiment behavior is:

1. `safeRL_workspace/notebooks/lanelessKaralakou.ipynb`
2. `safeRL_workspace/docs/lanelessKaralakou_reference.md` for the maintained
   notebook contract and cell-order explanation
3. `safeRL_workspace/docs/script_reference.md` for module/function duties and
   workflow details

When a script, README, or remembered behavior conflicts with the notebook,
verify the notebook and update the relevant reference. Scripts are generally
orchestration, adaptation, evaluation, reporting, or rendering around the
notebook definitions; do not silently create a second experiment contract.

## Repository map

- `safeRL_workspace/notebooks/`: canonical experiment notebook and notebook
  metadata.
- `safeRL_workspace/laneless highway env/`: custom `lane-free-v0` Gymnasium
  environment and demo; it has no lane indices, lane centers, target lanes,
  lane-change actions, or `highway-v0` assumptions.
- `safeRL_workspace/configs/`: reusable environment/traffic configuration.
- `safeRL_workspace/scripts/catalog.py`: stable mapping from legacy script
  filenames to organized importable modules.
- `safeRL_workspace/scripts/common/`: reusable configuration, registries,
  observation/worker adapters, CBF geometry/projection, and projected PPO.
- `safeRL_workspace/scripts/training/`: PPO/DDPG training and controlled
  pilots, ablations, and protocol runners.
- `safeRL_workspace/scripts/evaluation/`: KPI evaluation, audits,
  counterfactuals, diagnostics, and cross-algorithm comparisons. Evaluation
  scripts should consume saved models/results and must not silently retrain.
- `safeRL_workspace/scripts/reporting/`: plots, dashboards, and provenance-
  preserving report/paper builders; these consume upstream results.
- `safeRL_workspace/scripts/rendering/`: qualitative policy/scenario renders
  and videos; renders are not substitutes for canonical KPI evaluation.
- `safeRL_workspace/scripts/ops/`: bounded smoke checks, live inspection,
  simulator benchmark, PowerShell launchers, and monitors.
- `safeRL_workspace/tests/`: unit and protocol tests. Tests must use bounded,
  deterministic fixtures and temporary output locations.
- `artifacts/`: repository-level result manifests/summaries. Generated models,
  logs, plots, videos, and new run directories are normally ignored; preserve
  existing user artifacts and never overwrite a result without an explicit
  output stem.

For the complete script inventory and function-level descriptions, read
`safeRL_workspace/docs/script_reference.md`. The quick routing map is:

| Need | Start with |
| --- | --- |
| Check the environment | `scripts/ops/mtm_laneless_smoke.py` |
| Run notebook code out of process | `scripts/training/run_laneless_notebook_task.py` |
| Evaluate canonical KPIs | `scripts/evaluation/evaluate_laneless_karalakou.py` |
| Train/screen PPO variants | `scripts/training/run_ppo_formulation_screen.py`, then `run_ppo_cbf_progression.py` |
| Run controlled PPO/DDPG pilots | `scripts/training/run_nominal_ppo_parameter_pilot.py`, `run_nominal_ppo_density_pilot.py`, `run_nominal_ddpg_parameter_pilot.py` |
| Study CBF filtering | `scripts/training/run_cbf_filter_ablation.py`, `scripts/evaluation/evaluate_cbf_counterfactuals.py` |
| Compare PPO and DDPG | `scripts/evaluation/compare_nominal_ppo_ddpg.py` |
| Inspect learning signals | `scripts/evaluation/diagnose_update_signal_pipeline.py` |
| Produce figures/reports | `scripts/reporting/` after input manifests are complete |
| Inspect behavior visually | `scripts/rendering/` with bounded steps and explicit output paths |

## Canonical research invariants

Unless a task explicitly asks for a non-canonical demo or ablation, preserve
these contract values from the notebook/reference:

- environment `lane-free-v0` with MTM surrounding traffic;
- 380 m by 10.2 m periodic road, 55 vehicles, and five visible neighbors;
- 0.01 s physics step (100 Hz) and 10 Hz policy actions;
- canonical PPO input is 32D: 30D target-y vehicle features plus the two
  previous normalized executed-action values;
- retained legacy DDPG reference uses a separate 42D observation;
- common physical acceleration action box is `[-3, 3]` for both components;
- evaluation means strict collision-free 1,000 m completion within 3,000
  policy steps; collision prevents completion;
- CBF-OFF measures the policy action map, while CBF-ON measures policy plus
  deployed shielding.
- For `ppo_hocbf_reward_raw`, an omitted `s_psi` must be calibrated once from
  a deterministic, unshielded rollout of a fixed `ppo_nominal` checkpoint and
  then frozen for every HOCBF-reward seed; zero-action calibration is only
  historical provenance, not a valid learning-comparison protocol.

Keep normalized policy actions, physical actions, raw actions, safe actions,
and executed actions distinct. Diagnostic records should preserve the raw,
safe, and executed stages, correction, solver/fallback status, seed, variant,
notebook source hash, configuration, and artifact paths.

Never use `lane_index`, lane-indexed highway assumptions, `highway-v0`, or old
DQN helpers as a shortcut in a laneless experiment. Do not change defaults in
one script to make a run pass; use explicit configuration/overrides and record
them in the result manifest.

## Working and verification rules

- Run package commands from `safeRL_workspace` using `python -m`, for example:

  `python -m scripts.ops.mtm_laneless_smoke --help`

  `python -m scripts.evaluation.evaluate_laneless_karalakou --help`

  `python -m scripts.training.run_ppo_cbf_progression --help`

- Before a long run, inspect `--help`, the selected model/variant, seed,
  traffic model, evaluation budget, and explicit output directory. Never start
  the one-million-transition PPO ladder merely to inspect code.
- Run the smallest relevant smoke/focused test first. The normal suite is
  `python -m pytest -q tests` from `safeRL_workspace` (or
  `python -m pytest -q safeRL_workspace/tests` from the Git root).
- When the notebook contract changes, update
  `docs/lanelessKaralakou_reference.md` and `docs/script_reference.md`, run
  the smoke and focused projection/observation tests, and verify the notebook
  hash recorded in new manifests.
- Use atomic publication for final JSON/CSV summaries and retain partial
  progress for interruptible long evaluations.
- Do not modify generated artifacts, virtual environments, caches, backups, or
  unrelated worktree changes unless the user explicitly scopes them in.

## Documentation routing

For environment/reward/observation/CBF/evaluation questions, read
`safeRL_workspace/docs/lanelessKaralakou_reference.md`. For which script to
run or how a module works, read the matching section of
`safeRL_workspace/docs/script_reference.md` and the group README. For a new
experiment, preserve the notebook contract, use the existing registries and
configuration helpers, and add provenance rather than copying defaults.
