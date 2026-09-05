# safeRL scripts

The scripts package contains the executable and reusable code around the
canonical `notebooks/lanelessKaralakou.ipynb` experiment. The notebook remains
the source of truth for environment construction, reward definitions, CBF
geometry, and the baseline algorithms; these modules provide repeatable
training, evaluation, reporting, and rendering workflows.

## Run convention

Run commands from `highway-rl-decision-making` with Python module syntax so
package imports and repository paths are initialized consistently:

```powershell
python -m scripts.ops.mtm_laneless_smoke --help
python -m scripts.evaluation.evaluate_laneless_karalakou --help
python -m scripts.training.run_ppo_cbf_progression --help
python -m scripts.rendering.render_laneless_karalakou --help
```

The old flat filenames are retained as stable lookup names in
[`catalog.py`](catalog.py), which the notebook uses to resolve organized
modules. New documentation should link to each module's file in its group.

## Directory map

| Group | Contents | Start here |
| --- | --- | --- |
| [`common/`](common/README.md) | Notebook-compatible configuration, registries, observation/context wrappers, CBF projection, and PPO classes | `laneless_script_config.py`, `cbf_projection.py` |
| [`training/`](training/README.md) | PPO/DDPG training, protocol runners, ablations, and parameter experiments | `run_ppo_cbf_progression.py`, `run_cbf_filter_ablation.py` |
| [`evaluation/`](evaluation/README.md) | Canonical KPIs, checkpoint evaluation, audits, counterfactuals, and comparisons | `evaluate_laneless_karalakou.py` |
| [`reporting/`](reporting/README.md) | Plots, dashboards, paper-result packaging, and PDF reports | `plot_ppo_progression_results.py` |
| [`rendering/`](rendering/README.md) | Live policy scenes, annotated scenario comparisons, and videos | `render_laneless_karalakou.py` |
| [`ops/`](ops/README.md) | Smoke checks, live MTM inspection, launchers, and process monitors | `mtm_laneless_smoke.py` |

## Typical workflow

```text
notebook contract
    -> common adapters
    -> training protocol
    -> evaluation and audits
    -> artifacts/ manifests
    -> reporting or rendering
```

Use an explicit output directory for every long-running command. Generated
models, logs, plots, videos, and new run directories are ignored; deliberately
retained compact manifests belong under the repository-level
[`artifacts/`](../../artifacts/README.md) directory.

## Documentation

- [Notebook contract](../docs/lanelessKaralakou_reference.md)
- [Script and function reference](../docs/script_reference.md)
- [Diagnostic scenarios](../docs/diagnostic_scenarios.md)
- [CBF factorial ablation](../docs/cbf_factorial_ablation.md)
- [Tests](../tests/README.md)
