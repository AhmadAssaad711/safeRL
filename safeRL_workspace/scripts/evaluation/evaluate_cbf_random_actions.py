"""Evaluate the external CBF with seeded uniform physical random actions.

This is a CBF-only exploratory evaluator: no trained policy or saved run
configuration is loaded. It uses the notebook-compatible PPO environment,
the fixed relative-position CBF geometry, and the normal strict episode/KPI
path.
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
from typing import Any

import numpy as np
import pandas as pd

try:
    from saferl.artifacts import validate_manifest, write_json_atomic
except ModuleNotFoundError:  # Support ``python -m`` from safeRL_workspace.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from saferl.artifacts import validate_manifest, write_json_atomic

import scripts.training.run_cbf_filter_ablation as protocol
import scripts.training.run_ppo_cbf_progression as progression


DEFAULT_EPISODES = 200
DEFAULT_SEED_START = 1_100_000
DEFAULT_RESET_MAX_ATTEMPTS = 100
RANDOM_VARIANT = "random_uniform_physical_cbf"
RANDOM_ACTION_SOURCE = "uniform_physical_action_box"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the current external CBF with IID uniform physical "
            "actions; no model checkpoint is required."
        )
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--k0", type=float, default=2.5)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--correction-epsilon", type=float, default=0.03)
    parser.add_argument("--task-distance-m", type=float, default=1_000.0)
    parser.add_argument("--task-max-policy-steps", type=int, default=3_000)
    parser.add_argument("--ttc-cap", type=float, default=30.0)
    parser.add_argument(
        "--reset-max-attempts",
        type=int,
        default=DEFAULT_RESET_MAX_ATTEMPTS,
        help="Deterministic CBF-feasible reset candidates per episode.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Must be 1: each episode owns a deterministic random-action RNG.",
    )
    parser.add_argument(
        "--allow-dirty-exploratory",
        action="store_true",
        help="Allow a dirty worktree and preserve its tracked patch in the result.",
    )
    return parser.parse_args()


def uniform_physical_action(
    rng: np.random.Generator, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    """Return one IID physical action in the environment's two-dimensional Box."""

    low_array = np.asarray(low, dtype=np.float32).reshape(2)
    high_array = np.asarray(high, dtype=np.float32).reshape(2)
    if not np.all(np.isfinite(low_array)) or not np.all(np.isfinite(high_array)):
        raise ValueError("Random physical-action bounds must be finite")
    if np.any(low_array >= high_array):
        raise ValueError("Each random physical-action lower bound must be below its upper bound")
    return rng.uniform(low=low_array, high=high_array).astype(np.float32)


def _git_provenance(project_root: Path, output_dir: Path, *, allow_dirty: bool) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(project_root), "status", "--porcelain=v1"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("A Git commit is required for this evaluation") from exc
    dirty = bool(status)
    if dirty and not allow_dirty:
        raise RuntimeError(
            "Refusing a dirty worktree. Commit inputs or use "
            "--allow-dirty-exploratory to preserve its patch."
        )
    result: dict[str, Any] = {
        "git_commit": commit,
        "git_dirty": dirty,
        "git_status": status.splitlines(),
        "dirty_patch_path": None,
        "dirty_patch_sha256": None,
        "untracked_paths": [line[3:] for line in status.splitlines() if line.startswith("?? ")],
    }
    if dirty:
        patch = subprocess.check_output(
            ["git", "-C", str(project_root), "diff", "--binary", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        patch_path = output_dir / "source_worktree.patch"
        temporary = patch_path.with_suffix(".patch.tmp")
        temporary.write_bytes(patch)
        temporary.replace(patch_path)
        result["dirty_patch_path"] = str(patch_path)
        result["dirty_patch_sha256"] = hashlib.sha256(patch).hexdigest()
    return result


def _source_hashes(project_root: Path) -> dict[str, str]:
    paths = {
        "notebook": project_root / "notebooks" / "lanelessKaralakou.ipynb",
        "evaluator_entrypoint": Path(__file__).resolve(),
        "ppo_progression": project_root / "scripts" / "training" / "run_ppo_cbf_progression.py",
        "ppo_cbf_wrapper": project_root / "scripts" / "common" / "ppo_cbf_env.py",
        "cbf_geometry": project_root / "scripts" / "common" / "cbf_geometry.py",
    }
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _resolved_environment(namespace: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the same notebook-compatible MTM defaults as PPO evaluation."""

    env_config = copy.deepcopy(namespace["ENV_CONFIG"])
    if progression.active_traffic_model(env_config) == "mtm":
        progression._deep_set_defaults(
            env_config, copy.deepcopy(progression.MTM_CONGESTED_UNCERTAIN_UPDATES)
        )
    frequencies = progression._frequency_contract(env_config)
    expected = {"physics_hz": 100.0, "policy_hz": 20.0, "cbf_hz": 20.0}
    if any(not math.isclose(float(frequencies[key]), value) for key, value in expected.items()):
        raise ValueError(f"Random CBF evaluation requires canonical 100/20/20 timing, got {frequencies}")
    return env_config, frequencies


def _enable_cbf_feasible_resets(
    env_config: dict[str, Any], *, max_attempts: int
) -> dict[str, Any]:
    """Opt this evaluator into deterministic CBF-feasible reset sampling."""

    attempts = int(max_attempts)
    if attempts <= 0:
        raise ValueError("reset-max-attempts must be positive")
    resolved = copy.deepcopy(env_config)
    resolved["cbf_reset_feasibility"] = {
        "enabled": True,
        "max_attempts": attempts,
    }
    return resolved


def _effective_cbf(namespace: dict[str, Any], *, k0: float, k1: float) -> dict[str, Any]:
    for name, value in (("CBF_K0", k0), ("CBF_K1", k1)):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        namespace[name] = float(value)
    keys = (
        "CBF_AX_BOUNDS",
        "CBF_AY_BOUNDS",
        "CBF_EPS_SIDE",
        "CBF_K0",
        "CBF_K1",
        "CBF_PSI1_GAIN",
        "CBF_NEIGHBOR_RANGE",
        "CBF_MAX_NEIGHBOR_CONSTRAINTS",
        "CBF_QP_FEASIBILITY_TOL",
        "CBF_TARGET_PAIR_DY",
    )
    return {key: copy.deepcopy(namespace[key]) for key in keys if key in namespace}


def _install_random_variant() -> None:
    progression.VARIANT_SPECS.setdefault(
        RANDOM_VARIANT,
        {
            "label": "Uniform random physical action + external CBF",
            "execution_mode": "cbf",
        },
    )


def _evaluate_episode_with_random_actions(
    namespace: dict[str, Any],
    *,
    episode_index: int,
    episode_seed: int,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Reuse the canonical complete-episode metrics with an episode-local RNG."""

    low = np.asarray(
        [namespace["CBF_AX_BOUNDS"][0], namespace["CBF_AY_BOUNDS"][0]],
        dtype=np.float32,
    )
    high = np.asarray(
        [namespace["CBF_AX_BOUNDS"][1], namespace["CBF_AY_BOUNDS"][1]],
        dtype=np.float32,
    )
    rng = np.random.default_rng(int(episode_seed))
    original_predict = progression._predict_evaluation_action
    original_make_evaluation_env = progression.make_evaluation_env
    constructed_envs: list[Any] = []

    def random_predict(_model: Any, _observation: np.ndarray, *, action_source: str) -> np.ndarray:
        if action_source != RANDOM_ACTION_SOURCE:
            raise ValueError(f"Unexpected random action source: {action_source!r}")
        return uniform_physical_action(rng, low, high)

    def capture_evaluation_env(*inner_args: Any, **inner_kwargs: Any) -> Any:
        env = original_make_evaluation_env(*inner_args, **inner_kwargs)
        constructed_envs.append(env)
        return env

    progression._predict_evaluation_action = random_predict
    progression.make_evaluation_env = capture_evaluation_env
    try:
        row = progression.evaluate_completed_episode(
            namespace,
            model=None,
            variant=RANDOM_VARIANT,
            mode="cbf",
            training_seed=0,
            episode_index=int(episode_index),
            episode_seed=int(episode_seed),
            env_config=env_config,
            reward_config=reward_config,
            args=args,
            action_source=RANDOM_ACTION_SOURCE,
        )
        if constructed_envs:
            reset_info = constructed_envs[-1].get_wrapper_attr("last_reset_info")
            row = dict(row)
            row.update(
                {
                    "cbf_reset_feasibility_enabled": bool(
                        reset_info["cbf_reset_feasibility_enabled"]
                    ),
                    "cbf_reset_max_attempts": int(
                        reset_info["cbf_reset_max_attempts"]
                    ),
                    "cbf_reset_retry_count": int(reset_info["cbf_reset_retry_count"]),
                    "cbf_reset_candidate_seed": int(
                        reset_info["cbf_reset_candidate_seed"]
                    ),
                }
            )
        return row
    finally:
        progression._predict_evaluation_action = original_predict
        progression.make_evaluation_env = original_make_evaluation_env


def _manifest(
    config: dict[str, Any],
    *,
    status: str,
    artifacts: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "status": status,
        "experiment": {
            "id": "cbf_random_uniform_physical_actions",
            "variant": RANDOM_VARIANT,
            "entrypoint": str(Path(__file__).resolve()),
        },
        "algorithm": "random_uniform_physical_actions",
        "environment_config": config["environment_config"],
        "reward_config": config["reward_config"],
        "cbf_config": config["cbf"],
        "observation_definition": config["observation_definition"],
        "seed": config["seed_start"],
        "frequencies": config["frequencies"],
        "training_steps": 0,
        "git_commit": config["source"]["git_commit"],
        "git_dirty": config["source"]["git_dirty"],
        "model_path": None,
        "evaluation_protocol": config["evaluation_protocol"],
        "source_hashes": config["source"]["source_hashes"],
    }
    if artifacts is not None:
        manifest["artifacts"] = artifacts
    if error is not None:
        manifest["error"] = error
    validate_manifest(manifest)
    return manifest


def main() -> int:
    args = _parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.workers != 1:
        raise ValueError("Random-action evaluation is serial; use --workers 1")
    if args.correction_epsilon < 0.0:
        raise ValueError("correction-epsilon must be non-negative")
    if args.task_distance_m <= 0.0 or args.task_max_policy_steps <= 0:
        raise ValueError("task distance and policy-step cap must be positive")
    if args.reset_max_attempts <= 0:
        raise ValueError("reset-max-attempts must be positive")

    project_root = protocol.find_project_root(args.project_root or Path.cwd()).resolve()
    output_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else (project_root / args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    source = _git_provenance(project_root, output_dir, allow_dirty=bool(args.allow_dirty_exploratory))
    source["source_hashes"] = _source_hashes(project_root)

    protocol.set_stable_native_defaults()
    namespace = protocol.bootstrap_notebook_namespace(project_root)
    protocol.exec_required_notebook_cells(project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace)
    env_config, frequencies = _resolved_environment(namespace)
    env_config = _enable_cbf_feasible_resets(
        env_config, max_attempts=int(args.reset_max_attempts)
    )
    reward_config = protocol.make_base_reward_config(namespace)
    cbf = _effective_cbf(namespace, k0=float(args.k0), k1=float(args.k1))
    low = [float(cbf["CBF_AX_BOUNDS"][0]), float(cbf["CBF_AY_BOUNDS"][0])]
    high = [float(cbf["CBF_AX_BOUNDS"][1]), float(cbf["CBF_AY_BOUNDS"][1])]
    _install_random_variant()

    config: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_kind": "cbf_random_uniform_physical_actions",
        "external_cbf": "ON",
        "episodes": int(args.episodes),
        "seed_start": int(args.seed_start),
        "evaluation_seeds": [int(args.seed_start) + index for index in range(int(args.episodes))],
        "environment_config": env_config,
        "reward_config": reward_config,
        "frequencies": frequencies,
        "cbf": {
            "enabled": True,
            "geometry": {
                "mode": "fixed_axis_aligned_relative_ellipse",
                "implementation": "h_equals_dx_over_a_squared_plus_dy_over_b_squared_minus_one",
                "relative_ellipse_a_m": float(env_config["cbf_geometry"]["relative_ellipse_a_m"]),
                "relative_ellipse_b_m": float(env_config["cbf_geometry"]["relative_ellipse_b_m"]),
                "environment_metadata": copy.deepcopy(env_config.get("cbf_geometry", {})),
            },
            "reset_feasibility": copy.deepcopy(env_config["cbf_reset_feasibility"]),
            "effective": cbf,
        },
        "random_actions": {
            "distribution": "IID uniform physical action per policy step",
            "action_source": RANDOM_ACTION_SOURCE,
            "low": low,
            "high": high,
            "per_episode_rng_seed": "episode_seed",
        },
        "traffic_guard": copy.deepcopy(env_config.get("traffic_safety", {})),
        "observation_definition": {
            "variant": "ppo_target_y_vehicle_features_plus_previous_action",
            "expected_shape": [32],
        },
        "evaluation_protocol": {
            "strict_distance_m": float(args.task_distance_m),
            "max_policy_steps": int(args.task_max_policy_steps),
            "ttc_cap_s": float(args.ttc_cap),
            "correction_epsilon": float(args.correction_epsilon),
            "action_source": RANDOM_ACTION_SOURCE,
            "external_cbf": "ON",
        },
        "source": source,
    }
    write_json_atomic(output_dir / "config.json", config)
    write_json_atomic(output_dir / "manifest.json", _manifest(config, status="running"))

    eval_args = argparse.Namespace(
        correction_epsilon=float(args.correction_epsilon),
        task_distance_m=float(args.task_distance_m),
        task_max_policy_steps=int(args.task_max_policy_steps),
        ttc_cap=float(args.ttc_cap),
    )
    progress_path, status_path, started = output_dir / "episodes_progress.csv", output_dir / "status.json", time.perf_counter()
    rows: list[dict[str, Any]] = []
    progression._write_episode_progress_snapshot(
        progress_path=progress_path,
        status_path=status_path,
        rows=rows,
        variant=RANDOM_VARIANT,
        expected_episodes=int(args.episodes),
        started_at=started,
    )
    try:
        for episode_index in range(1, int(args.episodes) + 1):
            episode_seed = int(args.seed_start) + episode_index - 1
            row = _evaluate_episode_with_random_actions(
                namespace,
                episode_index=episode_index,
                episode_seed=episode_seed,
                env_config=env_config,
                reward_config=reward_config,
                args=eval_args,
            )
            rows.append(row)
            progression._write_episode_progress_snapshot(
                progress_path=progress_path,
                status_path=status_path,
                rows=rows,
                variant=RANDOM_VARIANT,
                expected_episodes=int(args.episodes),
                started_at=started,
            )
            progression._print_episode_progress(
                row=row,
                completed=len(rows),
                expected=int(args.episodes),
                variant=RANDOM_VARIANT,
            )

        metrics = pd.DataFrame(rows)
        metrics.to_csv(output_dir / "episodes.csv", index=False)
        blocks, kpis, summary_geometry = progression.summarize_post_training_episodes(
            metrics, env_config=env_config
        )
        blocks.to_csv(output_dir / "pooled_blocks.csv", index=False)
        kpis.to_csv(output_dir / "kpi_summary.csv", index=False)
        completion = {
            "schema_version": 1,
            "status": "complete",
            "completed_episodes": len(metrics),
            "elapsed_sec": time.perf_counter() - started,
            "episode_metrics_path": str((output_dir / "episodes.csv").resolve()),
            "kpi_path": str((output_dir / "kpi_summary.csv").resolve()),
            "summary_geometry": summary_geometry,
        }
        write_json_atomic(output_dir / "completion.json", completion)
        write_json_atomic(
            output_dir / "manifest.json",
            _manifest(
                config,
                status="complete",
                artifacts={
                    "episodes": completion["episode_metrics_path"],
                    "pooled_blocks": str((output_dir / "pooled_blocks.csv").resolve()),
                    "kpi_summary": completion["kpi_path"],
                    "completion": str((output_dir / "completion.json").resolve()),
                },
            ),
        )
        write_json_atomic(output_dir / "status.json", completion)
        print(f"[cbf-random-actions] complete: {output_dir}", flush=True)
        print(kpis[["KPI", "Mean", "SD", "N"]].to_string(index=False), flush=True)
        return 0
    except BaseException as exc:
        write_json_atomic(
            output_dir / "manifest.json",
            _manifest(config, status="failed", error=repr(exc)),
        )
        write_json_atomic(
            status_path,
            {"state": "failed", "external_cbf": "ON", "error": repr(exc), "elapsed_sec": time.perf_counter() - started},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
