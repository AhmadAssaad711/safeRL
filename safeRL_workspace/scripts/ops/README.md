# Operations and live inspection

This group contains short smoke checks, live MTM helpers, and PowerShell
launch/monitor scripts for long-running pilots.

- `mtm_laneless_smoke.py`: inexpensive environment and traffic smoke test.
- `benchmark_simulator.py`: deterministic throughput benchmark for force/MTM
  traffic and the dynamics guard, including a final-state checksum.
- `run_current_mtm_live.py` and `run_mtm_y_target_live.py`: short live
  environment checks.
- `run_*.ps1`: background/foreground launchers with output manifests.
- `monitor_*.ps1`: progress, process, and log monitors for those launchers.

PowerShell launchers resolve the project root from their location and invoke
the organized Python modules with `python -m`. They remain machine-specific
helpers: review the Python path, output directory, seed, and budget before use.
