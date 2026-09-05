# safeRL tests

Tests are divided by the level of contract they exercise. They are collected
recursively by pytest from the `highway-rl-decision-making` directory.

```powershell
python -m pytest -q tests
```

## Layout

- [`unit/`](unit/README.md) checks the lane-free environment, CBF projection,
  observations, collision aggregation, and small reusable calculations.
- [`protocol/`](protocol/README.md) checks multi-stage runners, checkpoint and
  provenance rules, paired evaluation, pilot selection, and report pipelines.
- `conftest.py` provides one shared repository/import bootstrap so tests do not
  each need to guess the project root.

Tests should use deterministic fixtures and temporary output directories. They
must not launch long training runs, write into `artifacts/`, or depend on a
machine-local virtual environment.
