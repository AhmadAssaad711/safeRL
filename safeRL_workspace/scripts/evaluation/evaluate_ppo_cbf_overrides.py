"""Paired CBF-OFF/ON re-evaluation of a finished PPO run under explicit overrides.

This evaluation-only path reuses the complete-episode evaluator and pooled KPI
summary of ``run_ppo_cbf_progression`` with the run's saved environment,
reward, and CBF configuration.  Only the requested overrides change:

* ``--vehicles``: traffic density (``vehicles_count``);
* ``--hz``: a common physics = policy = CBF frequency (``dt = 1/hz``); the
  policy-step cap is rescaled so the task keeps its wall-clock horizon;
* ``--c1/--c2``: HOCBF class-K rates, ``psi1 = h_dot + c1*h`` and
  ``psi2 = psi1_dot + c2*psi1``, i.e. ``k1 = c1 + c2``, ``k0 = c1*c2``;
* ``--spawn-psi1-gain`` and ``--require-initial-safe-set``: reset contract.

The helpers here are shared by ``evaluate_hocbf_gain_grid`` and
``diagnose_cbf_qp_failures`` (2026-09-11 gain/density/frequency ablation).
"""

from __future__ import annotations

import os

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

import scripts.training.run_ppo_cbf_progression as progression
from scripts.evaluation.evaluate_cbf_random_actions import _git_provenance

DEFAULT_EPISODES = 200
DEFAULT_SEED_START = 1_100_000
_FREQUENCY_KEYS = ("simulation_frequency", "policy_frequency", "cbf_frequency")
_GEOMETRY_KEYS = ("CBF_RELATIVE_ELLIPSE_A", "CBF_RELATIVE_ELLIPSE_B")


def hocbf_gains(c1: float, c2: float) -> dict[str, float]:
    """Return namespace gains for the linear class-K HOCBF cascade.

    The QP row depends only on ``k1 = c1 + c2`` and ``k0 = c1*c2``, which is
    symmetric; the smaller rate is used as the first-level ``psi1`` gain.
    """

    low, high = sorted((float(c1), float(c2)))
    if not (math.isfinite(low) and math.isfinite(high) and low > 0.0):
        raise ValueError("HOCBF class-K rates must be finite and positive")
    return {"CBF_K0": low * high, "CBF_K1": low + high, "CBF_PSI1_GAIN": low}


def apply_env_overrides(
    env_config: dict[str, Any],
    *,
    max_policy_steps: int,
    vehicles: Optional[int] = None,
    hz: Optional[int] = None,
    spawn_psi1_gain: Optional[float] = None,
    require_initial_safe_set: Optional[bool] = None,
) -> tuple[dict[str, Any], int, dict[str, list[Any]]]:
    """Return ``(env_config, max_policy_steps, changes)`` without mutating the input."""

    config = copy.deepcopy(env_config)
    changes: dict[str, list[Any]] = {}

    def _set(key: str, value: Any) -> None:
        changes[key] = [config.get(key), value]
        config[key] = value

    if vehicles is not None:
        if int(vehicles) <= int(config.get("neighbors_count", 0)):
            raise ValueError("vehicles must exceed neighbors_count")
        _set("vehicles_count", int(vehicles))
    if hz is not None:
        if int(hz) <= 0:
            raise ValueError("hz must be positive")
        if bool(config.get("cbf_substep_filtering", False)):
            raise ValueError("--hz expects a policy-rate CBF (cbf_substep_filtering=false)")
        trained_policy_hz = float(config["policy_frequency"])
        for key in _FREQUENCY_KEYS:
            _set(key, int(hz))
        _set("dt", 1.0 / int(hz))
        scaled = int(round(int(max_policy_steps) * int(hz) / trained_policy_hz))
        changes["task_max_policy_steps"] = [int(max_policy_steps), scaled]
        max_policy_steps = scaled
    if spawn_psi1_gain is not None:
        gain = float(spawn_psi1_gain)
        if not (math.isfinite(gain) and gain > 0.0):
            raise ValueError("spawn psi1 gain must be finite and positive")
        traffic_safety = copy.deepcopy(config.get("traffic_safety", {}))
        changes["traffic_safety.spawn_cbf_psi1_gain"] = [traffic_safety.get("spawn_cbf_psi1_gain"), gain]
        traffic_safety["spawn_cbf_psi1_gain"] = gain
        traffic_safety["spawn_cbf_k1"] = gain  # legacy alias read by older configs
        config["traffic_safety"] = traffic_safety
    if require_initial_safe_set is not None:
        _set("cbf_require_initial_safe_set", bool(require_initial_safe_set))
    return config, int(max_policy_steps), changes


def load_source_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a ladder seed directory's run config and its study config."""

    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    study_config = json.loads((run_dir.parents[1] / "study_config.json").read_text(encoding="utf-8"))
    return run_config, study_config


def build_namespace(
    project_root: Path,
    run_config: dict[str, Any],
    gains: Optional[dict[str, float]] = None,
    max_neighbor_constraints: Optional[int] = None,
) -> dict[str, Any]:
    """Execute the notebook definitions and install the run's CBF settings.

    Spawned evaluation workers re-execute the notebook and receive only the
    non-geometry CBF keys, so the notebook ellipse must match the run's.
    """

    namespace = progression.protocol.bootstrap_notebook_namespace(project_root)
    progression.protocol.exec_required_notebook_cells(
        project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    namespace["DEVICE"] = "cpu"
    run_cbf = run_config["training_signature"]["cbf"]
    for key in _GEOMETRY_KEYS:
        if key in run_cbf and not math.isclose(float(namespace[key]), float(run_cbf[key]), abs_tol=1e-9):
            raise RuntimeError(f"{key}: notebook={namespace[key]} but the run used {run_cbf[key]}")
    namespace.update(copy.deepcopy(run_cbf))
    if gains:
        namespace.update(gains)
    if max_neighbor_constraints is not None:
        namespace["CBF_MAX_NEIGHBOR_CONSTRAINTS"] = int(max_neighbor_constraints)
    return namespace


def evaluation_args(
    run_config: dict[str, Any],
    study_config: dict[str, Any],
    *,
    max_policy_steps: int,
    workers: int,
    ttc_cap: float,
) -> argparse.Namespace:
    """The argument subset read by the progression's complete-episode evaluator."""

    return argparse.Namespace(
        device="cpu",
        correction_epsilon=float(run_config["training_signature"]["correction_epsilon"]),
        ttc_cap=float(ttc_cap),
        task_distance_m=float(study_config["evaluation_task_distance_m"]),
        task_max_policy_steps=int(max_policy_steps),
        post_train_eval_workers=int(workers),
    )


def add_override_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True, help="Ladder seed directory with run_config.json and model_final.zip.")
    parser.add_argument("--vehicles", type=int, default=None)
    parser.add_argument(
        "--max-neighbor-constraints",
        type=int,
        default=None,
        help="Limit the CBF to this many nearest in-range vehicle constraints.",
    )
    parser.add_argument("--hz", type=int, default=None, help="Set physics = policy = CBF frequency.")
    parser.add_argument("--spawn-psi1-gain", type=float, default=None)
    parser.add_argument(
        "--require-initial-safe-set", choices=("true", "false"), default=None,
        help="Override cbf_require_initial_safe_set (default: keep the run's value).",
    )
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--ttc-cap", type=float, default=30.0)
    parser.add_argument(
        "--allow-dirty-exploratory", action="store_true",
        help="Allow a dirty worktree and preserve its tracked patch in the result.",
    )


def resolve_run(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    project_root = progression.protocol.find_project_root(args.project_root or Path.cwd()).resolve()
    run_dir = args.run_dir if args.run_dir.is_absolute() else project_root / args.run_dir
    run_config, study_config = load_source_run(run_dir.resolve())
    return project_root, run_dir.resolve(), run_config, study_config


def prepare_output_dir(project_root: Path, output_dir: Path) -> Path:
    output = output_dir if output_dir.is_absolute() else project_root / output_dir
    if output.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output}")
    output.mkdir(parents=True)
    return output.resolve()


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_override_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--c1", type=float, default=None)
    parser.add_argument("--c2", type=float, default=None)
    parser.add_argument("--k0", type=float, default=None, help="Override the HOCBF h coefficient without changing psi1 gain.")
    parser.add_argument("--k1", type=float, default=None, help="Override the HOCBF h-dot coefficient without changing psi1 gain.")
    args = parser.parse_args()
    if (args.c1 is None) != (args.c2 is None):
        parser.error("--c1 and --c2 must be given together")
    if (args.k0 is None) != (args.k1 is None):
        parser.error("--k0 and --k1 must be given together")
    if args.c1 is not None and args.k0 is not None:
        parser.error("Use either --c1/--c2 or --k0/--k1, not both")
    if args.max_neighbor_constraints is not None and args.max_neighbor_constraints < 1:
        parser.error("--max-neighbor-constraints must be positive")

    project_root, run_dir, run_config, study_config = resolve_run(args)
    output_dir = prepare_output_dir(project_root, args.output_dir)
    source = _git_provenance(project_root, output_dir, allow_dirty=bool(args.allow_dirty_exploratory))
    if args.c1 is not None:
        gains = hocbf_gains(args.c1, args.c2)
    elif args.k0 is not None:
        if not (math.isfinite(args.k0) and math.isfinite(args.k1) and args.k0 > 0.0 and args.k1 > 0.0):
            parser.error("--k0 and --k1 must be finite and positive")
        gains = {"CBF_K0": float(args.k0), "CBF_K1": float(args.k1)}
    else:
        gains = None
    env_config, max_policy_steps, changes = apply_env_overrides(
        run_config["env_config"],
        max_policy_steps=int(study_config["evaluation_task_max_policy_steps"]),
        vehicles=args.vehicles,
        hz=args.hz,
        spawn_psi1_gain=args.spawn_psi1_gain,
        require_initial_safe_set=None if args.require_initial_safe_set is None else args.require_initial_safe_set == "true",
    )
    namespace = build_namespace(
        project_root,
        run_config,
        gains,
        max_neighbor_constraints=args.max_neighbor_constraints,
    )
    eval_args = evaluation_args(
        run_config, study_config, max_policy_steps=max_policy_steps, workers=args.workers, ttc_cap=args.ttc_cap
    )
    model_path = run_dir / "model_final.zip"
    manifest: dict[str, Any] = {
        "status": "running",
        "evaluation_kind": "post_training_complete_episodes_with_overrides",
        "entrypoint": "scripts.evaluation.evaluate_ppo_cbf_overrides",
        "source_run_dir": str(run_dir),
        "variant": run_config["variant"],
        "training_seed": int(run_config["training_seed"]),
        "model_path": str(model_path),
        "model_sha256": progression.protocol.file_sha256(model_path),
        "env_config_changes": changes,
        "hocbf_rates": (
            None if args.c1 is None else {"c1": gains["CBF_PSI1_GAIN"], "c2": gains["CBF_K1"] - gains["CBF_PSI1_GAIN"]}
        ),
        "hocbf_coefficient_override": None if args.k0 is None else {"k0": float(args.k0), "k1": float(args.k1)},
        "cbf": {key: namespace[key] for key in ("CBF_K0", "CBF_K1", "CBF_PSI1_GAIN", "CBF_EPS_SIDE", "CBF_MAX_NEIGHBOR_CONSTRAINTS")},
        "episodes_per_mode": int(args.episodes),
        "episode_seed_start": int(args.seed_start),
        "evaluation_workers": int(args.workers),
        "eval_args": vars(eval_args),
        "source": source,
    }
    manifest_path = output_dir / "manifest.json"
    write_manifest(manifest_path, manifest)
    print(f"[overrides] {run_config['variant']} changes={changes} gains={gains}", flush=True)

    started = time.perf_counter()
    rows = progression._evaluate_complete_episode_rows(
        namespace,
        model_path=model_path,
        variant=run_config["variant"],
        training_seed=int(run_config["training_seed"]),
        env_config=env_config,
        reward_config=copy.deepcopy(run_config["reward_config"]),
        args=eval_args,
        modes=tuple(progression.EVALUATION_MODES),
        episode_count=int(args.episodes),
        seed_start=int(args.seed_start),
        action_source="policy",
        progress_path=output_dir / "e.csv",
        status_path=output_dir / "progress.json",
        progress_started=started,
        progress_variant=run_config["variant"],
    )
    elapsed = time.perf_counter() - started
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "e.csv", index=False)
    blocks, table, geometry = progression.summarize_post_training_episodes(metrics, env_config=env_config)
    blocks.to_csv(output_dir / "b.csv", index=False)
    table.to_csv(output_dir / "kpi.csv", index=False)
    manifest.update(
        geometry,
        status="complete",
        elapsed_s=elapsed,
        policy_steps_per_s=float(metrics["timesteps"].sum()) / max(elapsed, 1e-9),
    )
    write_manifest(manifest_path, manifest)
    progression.print_post_training_results(table, variant=run_config["variant"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
