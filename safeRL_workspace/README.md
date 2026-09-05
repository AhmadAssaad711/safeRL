# safeRL laneless Karalakou workspace

This is the implementation workspace for safe reinforcement learning in the
lane-free highway environment. The canonical experiment specification is
[notebooks/lanelessKaralakou.ipynb](notebooks/lanelessKaralakou.ipynb).

## Scope

The retained code covers:

- the lane-free-v0 Gymnasium environment and MTM surrounding traffic;
- the Karalakou reward and target-y observation contract;
- hard HOCBF filtering and two-dimensional action projection;
- PPO progression, differentiable projection, detached actor guidance, and
  the legacy DDPG baselines;
- strict evaluation, KPI aggregation, common-state counterfactuals,
  parameter pilots, density audits, plots, and policy renders.

## Documentation

- [Documentation index](docs/README.md)
- [Canonical notebook reference](docs/lanelessKaralakou_reference.md)
- [Script and function reference](docs/script_reference.md)
- [Script package map](scripts/README.md)
- [Test map](tests/README.md)
- [Diagnostic scenario registry](docs/diagnostic_scenarios.md)
- [CBF factorial ablation](docs/cbf_factorial_ablation.md)
- [Nominal DDPG handoff](docs/nominal_ddpg_confirmation_handoff.md)
- [Nominal PPO handoff](docs/nominal_ppo_50k_pilot_handoff.md)

## Quick commands

From this directory:

    python -m scripts.ops.mtm_laneless_smoke --help
    python -m scripts.evaluation.evaluate_laneless_karalakou --help
    python -m scripts.training.run_ppo_cbf_progression --help
    python -m scripts.rendering.render_laneless_karalakou --help

The long-running commands require an explicit output directory and should be
launched only after confirming the selected model, seed, traffic model, and
evaluation budget.

## Layout

    configs/                 Reusable MTM live configuration
    ../artifacts/            Committed compact laneless result manifests
    docs/                    SafeRL experiment and API documentation
    laneless highway env/   lane-free-v0 implementation and demo
    notebooks/              Canonical lanelessKaralakou notebook
    scripts/                Organized safeRL Python package
      common/               Reusable notebook, CBF, and PPO components
      training/             Training and controlled experiment workflows
      evaluation/           KPI, audit, comparison, and diagnostics
      reporting/            Plots, dashboards, and report builders
      rendering/            Policy and scenario visualization
      ops/                  Smoke checks and launch monitors
    tests/                  Unit and protocol tests

Generated models, event logs, videos, plots, and new result folders are
ignored by Git. Existing ../artifacts/ppo_* directories are compact committed
safeRL result manifests retained for provenance.
