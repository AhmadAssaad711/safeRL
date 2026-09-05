"""Tests for the stable mapping from legacy filenames to script modules."""

from __future__ import annotations

from pathlib import Path

from scripts.catalog import SCRIPT_MODULES, module_for_script, script_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_catalog_entries_resolve_to_existing_python_files() -> None:
    """Every documented script module should point to a file in its group."""

    missing = [
        f"{filename} -> {script_path(PROJECT_ROOT, filename)}"
        for filename in SCRIPT_MODULES
        if not script_path(PROJECT_ROOT, filename).is_file()
    ]
    assert not missing, "Missing catalog targets: " + ", ".join(missing)


def test_catalog_accepts_legacy_and_organized_names() -> None:
    """Notebook and shell callers may use either supported spelling."""

    expected = "scripts.training.run_laneless_notebook_task"
    assert module_for_script("run_laneless_notebook_task.py") == expected
    assert module_for_script("scripts/training/run_laneless_notebook_task.py") == expected
    assert module_for_script("scripts.training.run_laneless_notebook_task") == expected
