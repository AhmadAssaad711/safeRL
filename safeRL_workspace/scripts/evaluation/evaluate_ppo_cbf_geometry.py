"""Evaluate a saved PPO policy with CBF ON and current notebook geometry.

This evaluation-only counterfactual reuses the complete-episode and KPI path
from ``run_ppo_cbf_progression``.  It preserves the heading-aware,
inflated-vehicle geometry installed by the notebook, and applies explicit
HOCBF gain overrides only after the saved CBF snapshot is loaded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

try:
    from saferl.artifacts import validate_manifest, write_json_atomic
except ModuleNotFoundError:  # Support direct ``python -m`` execution from the workspace.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from saferl.artifacts import validate_manifest, write_json_atomic
import scripts.training.run_ppo_cbf_progression as progression


DEFAULT_EPISODES = 200
DEFAULT_SEED_START = 1_100_000
_LEGACY_GEOMETRY_KEYS = {"CBF_RELATIVE_ELLIPSE_A", "CBF_RELATIVE_ELLIPSE_B"}
_SNAPSHOT_KEY_MAP = {
    "k0": "CBF_K0", "k1": "CBF_K1", "eps_side": "CBF_EPS_SIDE",
    "ax_bounds": "CBF_AX_BOUNDS", "ay_bounds": "CBF_AY_BOUNDS",
    "max_neighbor_constraints": "CBF_MAX_NEIGHBOR_CONSTRAINTS",
    "qp_feasibility_tolerance": "CBF_QP_FEASIBILITY_TOL",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved PPO checkpoint with CBF ON and current heading-aware geometry."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(progression.VARIANT_SPECS), default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--geometry", choices=("current", "legacy_relative_ellipse"), default="current")
    # Old axes cannot affect the current footprint-support barrier.  Retain
    # their spelling solely to provide a clear, non-silent failure.
    parser.add_argument("--full-major-axis", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--full-minor-axis", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--k0", type=float, default=None, help="Override saved HOCBF h coefficient.")
    parser.add_argument("--k1", type=float, default=None, help="Override saved HOCBF h-dot coefficient.")
    parser.add_argument("--correction-epsilon", type=float, default=0.03)
    parser.add_argument("--task-distance-m", type=float, default=1_000.0)
    parser.add_argument("--task-max-policy-steps", type=int, default=3_000)
    parser.add_argument("--ttc-cap", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument(
        "--allow-dirty-exploratory", action="store_true",
        help="Allow a dirty worktree and preserve its tracked patch in the result.",
    )
    return parser.parse_args()


def _resolve_path(path: Path, project_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(project_root: Path, output_dir: Path, *, allow_dirty: bool) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(project_root), "status", "--porcelain=v1"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("A Git commit is required for this evaluation") from exc
    dirty = bool(status)
    if dirty and not allow_dirty:
        raise RuntimeError(
            "Refusing a dirty worktree. Commit inputs or use --allow-dirty-exploratory "
            "to preserve its patch."
        )
    result: dict[str, Any] = {
        "git_commit": commit, "git_dirty": dirty, "git_status": status.splitlines(),
        "dirty_patch_path": None, "dirty_patch_sha256": None,
        "untracked_paths": [line[3:] for line in status.splitlines() if line.startswith("?? ")],
    }
    if dirty:
        patch = subprocess.check_output(
            ["git", "-C", str(project_root), "diff", "--binary", "HEAD"], stderr=subprocess.DEVNULL
        )
        patch_path = output_dir / "source_worktree.patch"
        temporary = patch_path.with_suffix(".patch.tmp")
        temporary.write_bytes(patch)
        temporary.replace(patch_path)
        result["dirty_patch_path"] = str(patch_path)
        result["dirty_patch_sha256"] = hashlib.sha256(patch).hexdigest()
    return result


def _source_hashes(project_root: Path, run_config_path: Path) -> dict[str, str]:
    paths = {
        "notebook": project_root / "notebooks" / "lanelessKaralakou.ipynb",
        "evaluator_entrypoint": Path(__file__).resolve(),
        "ppo_progression": project_root / "scripts" / "training" / "run_ppo_cbf_progression.py",
        "ppo_cbf_wrapper": project_root / "scripts" / "common" / "ppo_cbf_env.py",
        "cbf_geometry": project_root / "scripts" / "common" / "cbf_geometry.py",
        "saved_run_config": run_config_path,
    }
    return {name: _sha256_file(path) for name, path in paths.items() if path.is_file()}


def _load_run_config(path: Path, *, explicit_variant: str | None) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Run configuration must be an object: {path}")
    source = raw.get("training_config", raw)
    if not isinstance(source, dict):
        raise TypeError("training_config must be an object when present")
    embedded_variant = raw.get("variant", source.get("variant"))
    variant = explicit_variant or embedded_variant
    if variant is None:
        raise KeyError("Run configuration has no variant; pass --variant")
    if embedded_variant is not None and str(embedded_variant) != str(variant):
        raise ValueError(f"--variant={variant!r} conflicts with saved variant={embedded_variant!r}")
    if str(variant) not in progression.VARIANT_SPECS:
        raise ValueError(f"Unsupported PPO progression variant: {variant}")
    for key in ("training_seed", "env_config", "reward_config"):
        if key not in source:
            raise KeyError(f"Run configuration is missing {key!r}: {path}")
    return {
        "raw": raw, "source": source, "variant": str(variant),
        "training_seed": int(source["training_seed"]),
        "env_config": copy.deepcopy(source["env_config"]),
        "reward_config": copy.deepcopy(source["reward_config"]),
        "training_steps": int(raw.get("completed_timesteps", source.get("target_timesteps", 0))),
    }


def _saved_cbf_snapshot(run_config: Mapping[str, Any]) -> dict[str, Any]:
    source = run_config["source"]
    assert isinstance(source, Mapping)
    snapshot: dict[str, Any] = {}
    signature = source.get("training_signature", {})
    if isinstance(signature, Mapping) and isinstance(signature.get("cbf"), Mapping):
        snapshot.update(copy.deepcopy(dict(signature["cbf"])))
    fixed = source.get("fixed_cbf_snapshot", {})
    if isinstance(fixed, Mapping):
        for saved_key, namespace_key in _SNAPSHOT_KEY_MAP.items():
            if saved_key in fixed:
                snapshot[namespace_key] = copy.deepcopy(fixed[saved_key])
    return snapshot


def _apply_saved_cbf_snapshot(
    namespace: dict[str, Any], run_config: Mapping[str, Any], *, k0: float | None, k1: float | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply saved non-geometry fields, followed by explicit gain overrides."""
    snapshot = _saved_cbf_snapshot(run_config)
    ignored_geometry = {key: copy.deepcopy(value) for key, value in snapshot.items() if key in _LEGACY_GEOMETRY_KEYS}
    for key, value in snapshot.items():
        if str(key).startswith("CBF_") and key not in _LEGACY_GEOMETRY_KEYS:
            namespace[str(key)] = copy.deepcopy(value)
    if k0 is not None:
        namespace["CBF_K0"] = float(k0)
    if k1 is not None:
        namespace["CBF_K1"] = float(k1)
    for name in ("CBF_K0", "CBF_K1"):
        value = float(namespace[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    keys = (
        "CBF_AX_BOUNDS", "CBF_AY_BOUNDS", "CBF_EPS_SIDE", "CBF_K0", "CBF_K1",
        "CBF_PSI1_GAIN", "CBF_NEIGHBOR_RANGE", "CBF_MAX_NEIGHBOR_CONSTRAINTS",
        "CBF_QP_FEASIBILITY_TOL", "CBF_TARGET_PAIR_DY",
    )
    return ({key: copy.deepcopy(namespace[key]) for key in keys if key in namespace}, ignored_geometry)


def _observation_definition(run_config: Mapping[str, Any]) -> dict[str, Any]:
    source, env_config = run_config["source"], run_config["env_config"]
    assert isinstance(source, Mapping) and isinstance(env_config, Mapping)
    return {
        "variant": source.get("observation_variant", "ppo_target_y_vehicle_features_plus_previous_action"),
        "append_previous_executed_action": bool(env_config.get("ppo_append_previous_action", True)),
        "expected_shape": [32],
    }


def _resolved_config(*, args: argparse.Namespace, run_config: Mapping[str, Any], model_path: Path,
                     run_config_path: Path, effective_cbf: Mapping[str, Any],
                     ignored_geometry: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    env_config = run_config["env_config"]
    assert isinstance(env_config, Mapping)
    return {
        "schema_version": 1, "evaluation_kind": "ppo_cbf_current_geometry_counterfactual",
        "variant": run_config["variant"], "external_cbf": "ON",
        "geometry": {
            "mode": "current",
            "implementation": "notebook_pairwise_cbf_geometry_heading_aware_inflated_vehicle_ellipses",
            "environment_metadata": copy.deepcopy(env_config.get("cbf_geometry", {})),
            "ignored_legacy_snapshot_axes": dict(ignored_geometry),
        },
        "requested_gains": {"k0": args.k0, "k1": args.k1},
        "effective_cbf": copy.deepcopy(dict(effective_cbf)),
        "training_seed": run_config["training_seed"],
        "evaluation_seeds": [int(args.seed_start) + i for i in range(int(args.episodes))],
        "episodes": int(args.episodes), "workers": int(args.workers), "device": str(args.device),
        "model_path": str(model_path), "model_sha256": _sha256_file(model_path),
        "run_config_path": str(run_config_path), "training_steps": run_config["training_steps"],
        "environment_config": copy.deepcopy(dict(env_config)),
        "reward_config": copy.deepcopy(run_config["reward_config"]),
        "observation_definition": _observation_definition(run_config),
        "frequencies": {
            "physics_hz": int(env_config["simulation_frequency"]),
            "policy_hz": int(env_config["policy_frequency"]), "cbf_hz": int(env_config["cbf_frequency"]),
        },
        "evaluation_protocol": {
            "strict_distance_m": float(args.task_distance_m), "max_policy_steps": int(args.task_max_policy_steps),
            "ttc_cap_s": float(args.ttc_cap), "correction_epsilon": float(args.correction_epsilon),
            "action_source": "policy",
        },
        "source": copy.deepcopy(dict(source)),
    }


def _running_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    assert isinstance(source, Mapping)
    manifest = {
        "schema_version": 1, "status": "running",
        "experiment": {"id": "ppo_cbf_current_geometry_counterfactual", "variant": config["variant"], "entrypoint": str(Path(__file__).resolve())},
        "algorithm": "PPO", "environment_config": config["environment_config"],
        "reward_config": config["reward_config"],
        "cbf_config": {"enabled": True, **config["effective_cbf"], "geometry": config["geometry"]},
        "observation_definition": config["observation_definition"], "seed": config["training_seed"],
        "evaluation_seeds": config["evaluation_seeds"], "frequencies": config["frequencies"],
        "training_steps": config["training_steps"], "git_commit": source["git_commit"],
        "git_dirty": source["git_dirty"], "model_path": config["model_path"],
        "model_sha256": config["model_sha256"], "evaluation_protocol": config["evaluation_protocol"],
        "source_hashes": source["source_hashes"],
    }
    validate_manifest(manifest)
    return manifest


def main() -> int:
    args = _parse_args()
    if args.episodes <= 0 or args.workers <= 0:
        raise ValueError("episodes and workers must be positive")
    if args.geometry != "current":
        raise ValueError("legacy_relative_ellipse is not effective with current notebook geometry; use --geometry current")
    if args.full_major_axis is not None or args.full_minor_axis is not None:
        raise ValueError("Current geometry does not use fixed relative axes; remove --full-*-axis")
    if args.correction_epsilon < 0.0:
        raise ValueError("correction-epsilon must be non-negative")
    if args.task_distance_m <= 0.0 or args.task_max_policy_steps <= 0:
        raise ValueError("task distance and policy-step cap must be positive")

    project_root = progression.protocol.find_project_root(args.project_root or Path.cwd()).resolve()
    model_path = _resolve_path(args.model_path, project_root)
    run_config_path = _resolve_path(args.run_config, project_root)
    output_dir = _resolve_path(args.output_dir, project_root)
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {model_path}")
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Run configuration does not exist: {run_config_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _git_provenance(project_root, output_dir, allow_dirty=bool(args.allow_dirty_exploratory))
    source["source_hashes"] = _source_hashes(project_root, run_config_path)

    run_config = _load_run_config(run_config_path, explicit_variant=args.variant)
    namespace = progression.protocol.bootstrap_notebook_namespace(project_root)
    progression.protocol.exec_required_notebook_cells(project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace)
    namespace["DEVICE"] = str(args.device)
    effective_cbf, ignored_geometry = _apply_saved_cbf_snapshot(namespace, run_config, k0=args.k0, k1=args.k1)
    config = _resolved_config(
        args=args, run_config=run_config, model_path=model_path, run_config_path=run_config_path,
        effective_cbf=effective_cbf, ignored_geometry=ignored_geometry, source=source,
    )
    write_json_atomic(output_dir / "config.json", config)
    write_json_atomic(output_dir / "manifest.json", _running_manifest(config))

    eval_args = argparse.Namespace(
        device=str(args.device), correction_epsilon=float(args.correction_epsilon),
        task_distance_m=float(args.task_distance_m), task_max_policy_steps=int(args.task_max_policy_steps),
        ttc_cap=float(args.ttc_cap), post_train_eval_workers=int(args.workers),
        training_seed=int(run_config["training_seed"]),
    )
    progress_path, status_path, started = output_dir / "episodes_progress.csv", output_dir / "status.json", time.perf_counter()
    write_json_atomic(status_path, {"state": "running", "external_cbf": "ON", "expected_episodes": int(args.episodes)})
    try:
        rows = progression._evaluate_complete_episode_rows(
            namespace, model_path=model_path, variant=str(run_config["variant"]), training_seed=int(run_config["training_seed"]),
            env_config=run_config["env_config"], reward_config=run_config["reward_config"], args=eval_args,
            modes=("cbf",), episode_count=int(args.episodes), seed_start=int(args.seed_start), action_source="policy",
            progress_path=progress_path, status_path=status_path, progress_started=started, progress_variant=str(run_config["variant"]),
        )
        metrics = pd.DataFrame(rows)
        metrics.to_csv(output_dir / "episodes.csv", index=False)
        blocks, kpis, summary_geometry = progression.summarize_post_training_episodes(metrics, env_config=run_config["env_config"])
        blocks.to_csv(output_dir / "pooled_blocks.csv", index=False)
        kpis.to_csv(output_dir / "kpi_summary.csv", index=False)
        completion = {
            "schema_version": 1, "status": "complete", "manifest_path": str(output_dir / "manifest.json"),
            "completed_episodes": len(metrics), "elapsed_sec": time.perf_counter() - started,
            "episode_metrics_path": str(output_dir / "episodes.csv"), "kpi_path": str(output_dir / "kpi_summary.csv"),
            "summary_geometry": summary_geometry,
        }
        write_json_atomic(output_dir / "completion.json", completion)
        write_json_atomic(status_path, completion)
        print(f"[ppo-cbf-current-geometry] complete: {output_dir}", flush=True)
        print(kpis[["KPI", "Mean", "SD", "N"]].to_string(index=False), flush=True)
        return 0
    except BaseException as exc:
        write_json_atomic(status_path, {"state": "failed", "external_cbf": "ON", "error": repr(exc), "elapsed_sec": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
