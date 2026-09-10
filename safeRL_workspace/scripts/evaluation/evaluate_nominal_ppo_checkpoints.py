"""Select and evaluate nominal PPO checkpoints without deploying a CBF.

This entry point is deliberately limited to the raw nominal PPO policy.  It
reuses the canonical notebook-backed KPI evaluator, but executes only its base
cells and always passes ``variant='ppo'``/``needs_cbf=False``.  A small fixed
evaluation is run for every saved training snapshot; the selected snapshot for
each training seed is then evaluated on the requested final scenario set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.evaluation import evaluate_laneless_karalakou as canonical_eval


SCHEMA_VERSION = 1
DEFAULT_WORKERS = 20
DEFAULT_SMALL_EPISODES = 20
DEFAULT_FINAL_EPISODES = 200
DEFAULT_SMALL_SEED_START = 1_200_000
DEFAULT_FINAL_SEED_START = 1_300_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_row(row: pd.Series) -> dict[str, Any]:
    return {str(key): _json_value(value) for key, value in row.to_dict().items()}


def _completion_rate(metrics: pd.DataFrame) -> float:
    if "collision_free_completion" not in metrics:
        return float("nan")
    values = metrics["collision_free_completion"].map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
    )
    return float(values.mean())


def _kpi_summary_row(metrics: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {
        "collision_free_completion_rate": _completion_rate(metrics),
    }
    for _, row in canonical_eval.ten_kpi_summary(metrics).iterrows():
        column = next(
            column
            for label, column in canonical_eval.TEN_KPI_SPECS
            if label == row["KPI"]
        )
        result[f"{column}_mean"] = float(row["Mean"])
        result[f"{column}_sd"] = float(row["SD"])
    return result


def rank_checkpoint_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank snapshots within each training seed using only nominal CBF-OFF KPIs."""

    required = {
        "training_seed",
        "pilot_config",
        "model_timestep",
        "collision_free_completion_rate",
        "ego_collisions_per_km_mean",
        "episode_return_mean",
        "mean_abs_speed_deviation_mean",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Checkpoint selection summary is missing columns: {missing}")

    ranked_parts: list[pd.DataFrame] = []
    for _, group in summary.groupby(["pilot_config", "training_seed"], sort=True):
        ordered = group.sort_values(
            [
                "collision_free_completion_rate",
                "collision_episode_rate",
                "ego_collisions_per_km_mean",
                "episode_return_mean",
                "mean_abs_speed_deviation_mean",
                "model_timestep",
            ],
            ascending=[False, True, True, False, True, False],
            na_position="last",
        ).copy()
        ordered.insert(0, "selection_rank", np.arange(1, len(ordered) + 1))
        ranked_parts.append(ordered)
    if not ranked_parts:
        raise ValueError("No checkpoint candidates were available for selection")
    return pd.concat(ranked_parts, ignore_index=True)


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Nominal PPO run configuration is missing: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Nominal PPO run configuration is not an object: {path}")
    if bool(config.get("filtered_training", True)):
        raise ValueError("Checkpoint selector only accepts unfiltered nominal PPO training")
    return config


def _checkpoint_paths(
    run_dir: Path,
    *,
    target_timesteps: int,
    checkpoint_interval: int,
) -> list[tuple[int, Path]]:
    checkpoint_dir = run_dir / "model_checkpoints"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"PPO checkpoint directory is missing: {checkpoint_dir}")

    paths: dict[int, Path] = {}
    for path in checkpoint_dir.glob("*.zip"):
        if path.stem.isdigit():
            paths[int(path.stem)] = path
    expected = list(range(int(checkpoint_interval), int(target_timesteps) + 1, int(checkpoint_interval)))
    missing = [step for step in expected if step not in paths]
    if missing:
        raise RuntimeError(
            f"Nominal PPO checkpoints are incomplete in {checkpoint_dir}; missing {missing}"
        )

    selected: list[tuple[int, Path]] = []
    for step in expected:
        path = paths[step]
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            raise RuntimeError(f"Checkpoint provenance sidecar is missing: {sidecar}")
        sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
        expected_sha = str(sidecar_payload.get("model_sha256", ""))
        actual_sha = _sha256(path)
        if expected_sha != actual_sha:
            raise RuntimeError(f"Checkpoint checksum mismatch: {path}")
        selected.append((step, path))
    return selected


def _evaluation_namespace(project_root: Path, notebook_path: Path) -> dict[str, Any]:
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "PPO": canonical_eval.PPO,
        "DDPG": canonical_eval.DDPG,
    }
    canonical_eval.exec_notebook_cells(
        notebook_path,
        [2, 3, 5, 6, 8],
        namespace,
    )
    namespace["PROJECT_ROOT"] = project_root
    if not callable(namespace.get("_write_evaluation_episode_progress")):
        raise RuntimeError("Canonical notebook evaluation progress writer was not loaded")
    return namespace


def _worker_args(workers: int) -> argparse.Namespace:
    # Raw PPO evaluation does not execute the notebook CBF cells, but the
    # shared runner still reads these optional override attributes.
    return argparse.Namespace(
        workers=int(workers),
        lambda_filter=None,
        k0=None,
        k1=None,
        eps_side=None,
    )


def _run_parallel_eval(
    *,
    namespace: dict[str, Any],
    notebook_path: Path,
    model_path: Path,
    episodes: int,
    seed_start: int,
    workers: int,
    env_config: dict[str, Any],
    progress_path: Path,
    label: str,
) -> pd.DataFrame:
    return canonical_eval._evaluate_laneless_with_workers(
        namespace=namespace,
        notebook_path=notebook_path,
        model_path=model_path,
        variant="ppo",
        episodes=int(episodes),
        seed=int(seed_start),
        env_config=env_config,
        args=_worker_args(workers),
        needs_cbf=False,
        episode_log_path=progress_path,
        episode_log_label=label,
    )


def _evaluate_checkpoint_sweep(
    *,
    namespace: dict[str, Any],
    notebook_path: Path,
    run_dir: Path,
    config: dict[str, Any],
    training_seeds: list[int],
    pilot_configs: list[str],
    small_episodes: int,
    small_seed_start: int,
    workers: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[tuple[int, str, int], pd.DataFrame]]:
    env_config = json.loads(json.dumps(config["env_config"]))
    candidates: list[dict[str, Any]] = []
    raw_frames: dict[tuple[int, str, int], pd.DataFrame] = {}
    for training_seed in training_seeds:
        for pilot_config in pilot_configs:
            one_run_dir = run_dir / f"seed_{training_seed}" / pilot_config
            checkpoints = _checkpoint_paths(
                one_run_dir,
                target_timesteps=int(config["target_timesteps"]),
                checkpoint_interval=int(config["checkpoint_interval"]),
            )
            for model_timestep, model_path in checkpoints:
                key = (int(training_seed), str(pilot_config), int(model_timestep))
                stem = f"seed_{training_seed}_{model_timestep:09d}"
                metrics_path = output_dir / f"{stem}_episodes.csv"
                progress_path = output_dir / f"{stem}_progress.csv"
                if metrics_path.exists():
                    metrics = pd.read_csv(metrics_path)
                    if len(metrics) != int(small_episodes):
                        raise RuntimeError(
                            f"Existing small evaluation has the wrong episode count: {metrics_path}"
                        )
                    print(f"[nominal-eval] reusing small evaluation {metrics_path}", flush=True)
                else:
                    print(
                        f"[nominal-eval] small raw eval seed={training_seed} "
                        f"checkpoint={model_timestep:,} episodes={small_episodes}",
                        flush=True,
                    )
                    metrics = _run_parallel_eval(
                        namespace=namespace,
                        notebook_path=notebook_path,
                        model_path=model_path,
                        episodes=small_episodes,
                        seed_start=small_seed_start,
                        workers=workers,
                        env_config=env_config,
                        progress_path=progress_path,
                        label=f"small@seed{training_seed}@{model_timestep}",
                    )
                    _atomic_csv(metrics, metrics_path)
                kpis = _kpi_summary_row(metrics)
                collision_values = pd.to_numeric(
                    metrics["ego_collisions_per_km"], errors="coerce"
                )
                row: dict[str, Any] = {
                    "training_seed": int(training_seed),
                    "pilot_config": str(pilot_config),
                    "model_timestep": int(model_timestep),
                    "model_path": str(model_path.resolve()),
                    "model_sha256": _sha256(model_path),
                    "small_episodes": int(small_episodes),
                    "small_seed_start": int(small_seed_start),
                    "collision_episode_rate": float((collision_values > 0.0).mean()),
                    **kpis,
                }
                candidates.append(row)
                raw_frames[key] = metrics

    summary = pd.DataFrame(candidates)
    ranked = rank_checkpoint_candidates(summary)
    _atomic_csv(ranked, output_dir / "checkpoint_selection_summary.csv")
    return ranked, raw_frames


def _selected_records(
    ranked: pd.DataFrame,
    *,
    run_dir: Path,
) -> list[dict[str, Any]]:
    selections = ranked[ranked["selection_rank"] == 1].sort_values(
        ["pilot_config", "training_seed"]
    )
    records: list[dict[str, Any]] = []
    for _, row in selections.iterrows():
        model_path = Path(str(row["model_path"])).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Selected PPO checkpoint is missing: {model_path}")
        records.append(
            {
                "training_seed": int(row["training_seed"]),
                "pilot_config": str(row["pilot_config"]),
                "model_timestep": int(row["model_timestep"]),
                "model_path": str(model_path),
                "model_path_relative_to_run": str(
                    model_path.relative_to(run_dir)
                ),
                "model_sha256": str(row["model_sha256"]),
                "selection_metrics": _json_row(row),
            }
        )
    return records


def _final_evaluation(
    *,
    namespace: dict[str, Any],
    notebook_path: Path,
    config: dict[str, Any],
    selections: list[dict[str, Any]],
    final_episodes: int,
    final_seed_start: int,
    workers: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    env_config = json.loads(json.dumps(config["env_config"]))
    frames: list[pd.DataFrame] = []
    seed_kpis: list[pd.DataFrame] = []
    for selection in selections:
        training_seed = int(selection["training_seed"])
        model_timestep = int(selection["model_timestep"])
        model_path = Path(str(selection["model_path"])).resolve()
        stem = f"seed_{training_seed}_{model_timestep:09d}"
        metrics_path = output_dir / f"{stem}_episodes.csv"
        progress_path = output_dir / f"{stem}_progress.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            if len(metrics) != int(final_episodes):
                raise RuntimeError(
                    f"Existing final evaluation has the wrong episode count: {metrics_path}"
                )
            print(f"[nominal-eval] reusing final evaluation {metrics_path}", flush=True)
        else:
            print(
                f"[nominal-eval] final raw eval seed={training_seed} "
                f"checkpoint={model_timestep:,} episodes={final_episodes}",
                flush=True,
            )
            metrics = _run_parallel_eval(
                namespace=namespace,
                notebook_path=notebook_path,
                model_path=model_path,
                episodes=final_episodes,
                seed_start=final_seed_start,
                workers=workers,
                env_config=env_config,
                progress_path=progress_path,
                label=f"final@seed{training_seed}@{model_timestep}",
            )
            _atomic_csv(metrics, metrics_path)
        annotated = metrics.copy()
        annotated.insert(0, "training_seed", training_seed)
        annotated.insert(1, "pilot_config", str(selection["pilot_config"]))
        annotated.insert(2, "model_timestep", model_timestep)
        annotated.insert(3, "model_path", str(model_path))
        frames.append(annotated)

        kpi_rows = canonical_eval.ten_kpi_summary(metrics)
        kpi_rows.insert(0, "training_seed", training_seed)
        kpi_rows.insert(1, "pilot_config", str(selection["pilot_config"]))
        kpi_rows.insert(2, "model_timestep", model_timestep)
        completion_row = pd.DataFrame(
            [
                {
                    "training_seed": training_seed,
                    "pilot_config": str(selection["pilot_config"]),
                    "model_timestep": model_timestep,
                    "KPI": "Strict collision-free completion rate",
                    "Mean": _completion_rate(metrics),
                    "SD": 0.0,
                }
            ]
        )
        seed_kpis.append(pd.concat([completion_row, kpi_rows], ignore_index=True))

    all_metrics = pd.concat(frames, ignore_index=True)
    all_kpis = pd.concat(seed_kpis, ignore_index=True)
    _atomic_csv(all_metrics, output_dir / "final_episode_metrics.csv")
    _atomic_csv(all_kpis, output_dir / "final_kpi_by_seed.csv")

    pooled = canonical_eval.ten_kpi_summary(all_metrics)
    pooled_completion = pd.DataFrame(
        [
            {
                "KPI": "Strict collision-free completion rate",
                "Mean": _completion_rate(all_metrics),
                "SD": 0.0,
            }
        ]
    )
    pooled = pd.concat([pooled_completion, pooled], ignore_index=True)
    _atomic_csv(pooled, output_dir / "final_kpi_summary.csv")
    return all_metrics, all_kpis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and evaluate nominal PPO checkpoints with CBF-OFF only."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--small-episodes", type=int, default=DEFAULT_SMALL_EPISODES)
    parser.add_argument("--small-seed-start", type=int, default=DEFAULT_SMALL_SEED_START)
    parser.add_argument("--final-episodes", type=int, default=DEFAULT_FINAL_EPISODES)
    parser.add_argument("--final-seed-start", type=int, default=DEFAULT_FINAL_SEED_START)
    return parser.parse_args()


def main() -> int:
    canonical_eval.set_stable_native_defaults()
    args = parse_args()
    if int(args.workers) <= 0:
        raise ValueError("--workers must be positive")
    if int(args.small_episodes) <= 0 or int(args.final_episodes) <= 0:
        raise ValueError("Evaluation episode counts must be positive")

    run_dir = args.run_dir.resolve()
    config = _load_run_config(run_dir)
    training_seeds = [
        int(seed) for seed in (args.seeds if args.seeds is not None else config["training_seeds"])
    ]
    if not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("Training seeds must be non-empty and unique")
    pilot_configs = list(args.configs or config["selected_configs"])
    if not pilot_configs:
        raise ValueError("At least one nominal PPO configuration is required")

    env_config = config.get("env_config")
    if not isinstance(env_config, dict):
        raise ValueError("Training run configuration is missing the resolved env_config")
    guard_ego_interactions = bool(
        env_config.get("traffic_safety", {}).get("guard_ego_interactions", False)
    )
    if guard_ego_interactions:
        raise ValueError(
            "This nominal evaluation requires traffic_safety.guard_ego_interactions=false"
        )

    project_root = canonical_eval.find_project_root(args.project_root or Path.cwd())
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    selection_dir = run_dir / "evaluation" / "checkpoint_selection"
    final_dir = run_dir / "evaluation" / "final_selected"
    if selection_dir.exists() and any(selection_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite checkpoint-selection artifacts: {selection_dir}")
    if final_dir.exists() and any(final_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite final-evaluation artifacts: {final_dir}")
    selection_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "evaluation" / "nominal_ppo_evaluation_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "variant": "ppo_nominal",
        "cbf_enabled": False,
        "cbf_evaluation_requested": False,
        "traffic_guard": "social_only",
        "training_run": str(run_dir),
        "training_seeds": training_seeds,
        "pilot_configs": pilot_configs,
        "workers": int(args.workers),
        "selection_evaluation": {
            "episodes_per_checkpoint": int(args.small_episodes),
            "seed_start": int(args.small_seed_start),
            "protocol": "strict collision-free 1000 m within 3000 policy steps",
        },
        "final_evaluation": {
            "episodes_per_selected_checkpoint": int(args.final_episodes),
            "seed_start": int(args.final_seed_start),
            "protocol": "strict collision-free 1000 m within 3000 policy steps",
        },
    }
    _atomic_json(manifest_path, manifest)

    try:
        namespace = _evaluation_namespace(project_root, notebook_path)
        ranked, _ = _evaluate_checkpoint_sweep(
            namespace=namespace,
            notebook_path=notebook_path,
            run_dir=run_dir,
            config=config,
            training_seeds=training_seeds,
            pilot_configs=pilot_configs,
            small_episodes=int(args.small_episodes),
            small_seed_start=int(args.small_seed_start),
            workers=int(args.workers),
            output_dir=selection_dir,
        )
        selections = _selected_records(ranked, run_dir=run_dir)
        _atomic_json(
            run_dir / "evaluation" / "selected_checkpoints.json",
            {
                "schema_version": SCHEMA_VERSION,
                "selection_rule": (
                    "Within each training seed: maximize strict collision-free completion rate; "
                    "then minimize collision episode rate and ego collisions/km; then maximize "
                    "episode return; then minimize absolute speed error; ties prefer later checkpoint."
                ),
                "cbf_enabled": False,
                "selections": selections,
            },
        )
        all_metrics, _ = _final_evaluation(
            namespace=namespace,
            notebook_path=notebook_path,
            config=config,
            selections=selections,
            final_episodes=int(args.final_episodes),
            final_seed_start=int(args.final_seed_start),
            workers=int(args.workers),
            output_dir=final_dir,
        )
        manifest.update(
            {
                "status": "complete",
                "selected_checkpoints": selections,
                "final_episode_count": int(len(all_metrics)),
                "outputs": {
                    "selection_summary": str(
                        (selection_dir / "checkpoint_selection_summary.csv").resolve()
                    ),
                    "selected_checkpoints": str(
                        (run_dir / "evaluation" / "selected_checkpoints.json").resolve()
                    ),
                    "final_kpi_by_seed": str(
                        (final_dir / "final_kpi_by_seed.csv").resolve()
                    ),
                    "final_kpi_summary": str(
                        (final_dir / "final_kpi_summary.csv").resolve()
                    ),
                },
            }
        )
        _atomic_json(manifest_path, manifest)
        print(ranked[ranked["selection_rank"] == 1].to_string(index=False), flush=True)
        print(pd.read_csv(final_dir / "final_kpi_summary.csv").to_string(index=False), flush=True)
        print(f"[nominal-eval] complete: {run_dir / 'evaluation'}", flush=True)
        return 0
    except BaseException as exc:
        manifest.update({"status": "failed", "error": repr(exc)})
        _atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
