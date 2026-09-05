# Reporting and plots

Reporting modules turn evaluation tables, TensorBoard streams, and compact
manifests into inspectable figures and reports. They are downstream consumers;
they do not define the experiment contract.

- `plot_laneless_sample_efficiency.py`: sample-efficiency curves and summary.
- `plot_nominal_ppo_results.py`, `plot_ppo_progression_results.py`, and
  `plot_ppo_500k_saved_results.py`: PPO training/evaluation reports.
- `build_headline_kpi_dashboard.py` and `build_filter_contribution_dashboard.py`:
  KPI and filter-contribution dashboards.
- `build_cbf_ablation_report.py`: factorial CBF PDF report.
- `build_paper_results.py`: provenance-preserving paper-result bundle.

Prefer explicit `--input`, `--study-dir`, and `--output` arguments. Generated
figures and logs remain ignored; deliberate compact summaries belong under
[`artifacts/`](../../../artifacts/README.md).
