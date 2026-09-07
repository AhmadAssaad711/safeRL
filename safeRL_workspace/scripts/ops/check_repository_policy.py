"""Mechanically check the repository rules documented in ``AGENTS.md``.

This is intentionally a small structural guard, not a replacement for code
review. It catches policy drift that is easy to detect automatically while
leaving semantic decisions—especially notebook algorithm changes—for review.
"""

from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT / "safeRL_workspace"
AGENTS_PATH = REPO_ROOT / "AGENTS.md"

REQUIRED_PATHS = (
    WORKSPACE_ROOT / "notebooks" / "lanelessKaralakou.ipynb",
    WORKSPACE_ROOT / "docs" / "lanelessKaralakou_reference.md",
    WORKSPACE_ROOT / "docs" / "script_reference.md",
    WORKSPACE_ROOT / "laneless highway env" / "lane_free_env.py",
    WORKSPACE_ROOT / "pyproject.toml",
    WORKSPACE_ROOT / "src" / "saferl" / "config.py",
    WORKSPACE_ROOT / "src" / "saferl" / "artifacts.py",
    WORKSPACE_ROOT / "src" / "saferl" / "safety",
    WORKSPACE_ROOT / "scripts" / "common",
    WORKSPACE_ROOT / "scripts" / "training",
    WORKSPACE_ROOT / "scripts" / "evaluation",
    WORKSPACE_ROOT / "scripts" / "reporting",
    WORKSPACE_ROOT / "tests",
    WORKSPACE_ROOT / "tests" / "contract",
)

REQUIRED_POLICY_TEXT = (
    "architecture and ownership",
    "canonical laneless experiment assumptions",
    "no new core algorithms inside notebooks",
    "new research logic goes in `src/saferl/`",
    "no new 2,000-line experiment scripts",
    "keep training, evaluation, reporting",
    "ddpg is legacy unless explicitly being studied",
    "canonical historical experiment and interactive analysis",
    "all new experiments",
    "every reportable training run, pilot, ablation, evaluation",
    "canonical environment factory",
    "make_laneless_env(config)",
    "new research code and centralized constants",
    "contract-first tests",
    "self-contained experiment artifacts",
    "experimental code versus core code",
    "100 hz physics, 20 hz policy",
    "methodical git and commit discipline",
    "commits small, atomic, and single-purpose",
    "git_commit",
)

# These are existing migration-debt files. New files over the limit are not
# allowed, and existing exceptions must not grow without a decomposition plan.
OVERSIZED_SCRIPT_ALLOWLIST = frozenset(
    {
        "safeRL_workspace/scripts/training/run_cbf_filter_ablation.py",
        "safeRL_workspace/scripts/training/run_nominal_ddpg_parameter_pilot.py",
        "safeRL_workspace/scripts/training/run_nominal_ppo_parameter_pilot.py",
        "safeRL_workspace/scripts/training/run_ppo_cbf_progression.py",
    }
)

# DDPG entry points are historical comparison surfaces. Adding a new one
# requires deliberate classification here and in AGENTS.md.
LEGACY_DDPG_SCRIPT_ALLOWLIST = frozenset(
    {
        "safeRL_workspace/scripts/evaluation/compare_nominal_ppo_ddpg.py",
        "safeRL_workspace/scripts/training/ddpg_cbf_gain_sweep_50k.py",
        "safeRL_workspace/scripts/training/ddpg_cbf_intervention_sweep_50k.py",
        "safeRL_workspace/scripts/training/ddpg_cbf_lambda005_experiment.py",
        "safeRL_workspace/scripts/training/retrain_ddpg_stepwise_comparison.py",
        "safeRL_workspace/scripts/training/run_nominal_ddpg_parameter_pilot.py",
        "safeRL_workspace/scripts/training/train_ddpg_cbf_500k.py",
        "safeRL_workspace/scripts/training/train_ddpg_cbf_ray_mask.py",
        "safeRL_workspace/scripts/training/train_ddpg_ego_y_only_50k.py",
        "safeRL_workspace/scripts/training/train_ddpg_y_target_50k.py",
    }
)

FORBIDDEN_NEW_CONSTANT_ASSIGNMENTS = re.compile(
    r"^\s*(?:EPS_SIDE|K0|K1|POLICY_HZ|PHYSICS_HZ|CBF_HZ)\s*=",
    re.MULTILINE,
)
STALE_REPOSITORY_LABEL = "highway-" "rl-decision-making"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def collect_violations() -> list[str]:
    """Return structural repository-policy violations."""

    violations: list[str] = []

    if not AGENTS_PATH.is_file():
        return [f"missing required repository guidance: {_relative(AGENTS_PATH)}"]

    agents_text = re.sub(
        r"\s+",
        " ",
        AGENTS_PATH.read_text(encoding="utf-8").lower(),
    )
    for required_text in REQUIRED_POLICY_TEXT:
        if required_text.lower() not in agents_text:
            violations.append(f"AGENTS.md is missing required policy: {required_text}")

    for required_path in REQUIRED_PATHS:
        if not required_path.exists():
            violations.append(f"missing required repository path: {_relative(required_path)}")

    new_source_root = WORKSPACE_ROOT / "src" / "saferl"
    if new_source_root.is_dir():
        for path in new_source_root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if FORBIDDEN_NEW_CONSTANT_ASSIGNMENTS.search(source):
                violations.append(
                    "new saferl code hardcodes a shared experiment constant; "
                    f"use the central config object: {_relative(path)}"
                )

    stale_path_roots = (
        REPO_ROOT / "README.md",
        WORKSPACE_ROOT / "docs",
        WORKSPACE_ROOT / "scripts",
        WORKSPACE_ROOT / "tests",
    )
    for root in stale_path_roots:
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in {".md", ".py", ".ps1", ".json"}:
                continue
            if STALE_REPOSITORY_LABEL in path.read_text(encoding="utf-8"):
                violations.append(
                    f"stale repository path reference in {_relative(path)}"
                )

    for root_name in ("training", "evaluation", "reporting"):
        root = WORKSPACE_ROOT / "scripts" / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            relative_path = _relative(path)
            if _line_count(path) > 2000 and relative_path not in OVERSIZED_SCRIPT_ALLOWLIST:
                violations.append(
                    f"unapproved oversized experiment script ({_line_count(path)} lines): "
                    f"{relative_path}"
                )

    scripts_root = WORKSPACE_ROOT / "scripts"
    if scripts_root.is_dir():
        for path in scripts_root.rglob("*.py"):
            if "ddpg" not in path.name.lower():
                continue
            relative_path = _relative(path)
            if relative_path not in LEGACY_DDPG_SCRIPT_ALLOWLIST:
                violations.append(
                    "unclassified DDPG entry point; explicitly classify legacy research "
                    f"before adding it: {relative_path}"
                )

    return violations


def main() -> int:
    violations = collect_violations()
    if violations:
        print("Repository policy: FAIL")
        for violation in violations:
            print(f"- {violation}")
        return 1

    print("Repository policy: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
