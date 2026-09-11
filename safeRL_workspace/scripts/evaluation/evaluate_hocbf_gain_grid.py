"""Evaluate a fixed PPO policy with the CBF ON over a grid of HOCBF rates.

Each cell uses the linear class-K cascade ``psi1 = h_dot + c1*h`` and
``psi2 = psi1_dot + c2*psi1`` (``k1 = c1 + c2``, ``k0 = c1*c2``), so every cell
with ``c1, c2 > 0`` is a valid HOCBF.  The QP depends only on ``(k0, k1)``,
which is symmetric in ``(c1, c2)``, so only ``c1 <= c2`` is evaluated.

Every cell runs the same episode seeds, the same policy, and the same spawn
sampler, so the comparison is paired.  By default the reset psi1 check is
disabled so cells whose own psi1 is negative at reset are still evaluated.
Optional reference cells: CBF OFF and the run's own saved gains.
"""

from __future__ import annotations

import os

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import argparse
import copy
import itertools
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

import scripts.training.run_ppo_cbf_progression as progression
from scripts.evaluation.evaluate_cbf_random_actions import _git_provenance
from scripts.evaluation.evaluate_ppo_cbf_overrides import (
    add_override_arguments,
    apply_env_overrides,
    build_namespace,
    evaluation_args,
    hocbf_gains,
    load_source_run,
    prepare_output_dir,
    resolve_run,
    write_manifest,
)

DEFAULT_C_VALUES = (0.5, 1.0, 1.5, 2.3, 3.5, 5.0, 8.0)
DEFAULT_EPISODES = 30
_WORKER: dict[str, Any] = {}


def gain_grid_cells(
    c_values: Iterable[float] = DEFAULT_C_VALUES,
    pairs: Optional[Iterable[tuple[float, float]]] = None,
) -> list[tuple[str, dict[str, float]]]:
    """Return ``(label, namespace gains)`` for every unique ``c1 <= c2`` cell."""

    if pairs is None:
        candidates = itertools.combinations_with_replacement(sorted(float(c) for c in c_values), 2)
    else:
        candidates = (tuple(sorted((float(a), float(b)))) for a, b in pairs)
    cells: dict[str, dict[str, float]] = {}
    for c1, c2 in candidates:
        cells.setdefault(f"c1={c1:g}_c2={c2:g}", hocbf_gains(c1, c2))
    return list(cells.items())


def _parse_pair(text: str) -> tuple[float, float]:
    first, second = text.split(",")
    return float(first), float(second)


def _init_worker(project_root: str, run_dir: str, env_config: dict[str, Any], eval_args: argparse.Namespace) -> None:
    import torch as th

    progression.protocol.set_stable_native_defaults()
    try:
        th.set_num_threads(1)
    except RuntimeError:
        pass
    run_config, _study = load_source_run(Path(run_dir))
    _WORKER.update(
        namespace=build_namespace(Path(project_root), run_config),
        run_config=run_config,
        env_config=env_config,
        eval_args=eval_args,
        model=progression.load_model(run_config["variant"], Path(run_dir) / "model_final.zip", "cpu"),
    )


def _evaluate_cell_episode(
    cell: str, mode: str, gains: dict[str, float], episode_index: int, seed: int
) -> dict[str, Any]:
    namespace = _WORKER["namespace"]
    namespace.update(gains)  # read when the episode's environment is built
    run_config = _WORKER["run_config"]
    row = progression.evaluate_completed_episode(
        namespace,
        model=_WORKER["model"],
        variant=run_config["variant"],
        mode=mode,
        training_seed=int(run_config["training_seed"]),
        episode_index=int(episode_index),
        episode_seed=int(seed),
        env_config=_WORKER["env_config"],
        reward_config=copy.deepcopy(run_config["reward_config"]),
        args=_WORKER["eval_args"],
        action_source="policy",
    )
    row = {key: value for key, value in row.items() if not str(key).startswith("_")}
    row.update(cell=cell, k0=gains["CBF_K0"], k1=gains["CBF_K1"], psi1_gain=gains["CBF_PSI1_GAIN"])
    return row


def summarize_cells(episodes: pd.DataFrame) -> pd.DataFrame:
    """Pool each cell's collisions over its distance; weight step rates by steps."""

    rows = []
    for cell, group in episodes.groupby("cell", sort=False):
        weights = group["timesteps"]
        distance_km = float(group["total_distance_m"].sum()) / 1000.0
        collisions = int(group["distinct_ego_collision_events"].sum())
        rows.append({
            "cell": cell,
            "k0": float(group["k0"].iloc[0]),
            "k1": float(group["k1"].iloc[0]),
            "c1": float(group["psi1_gain"].iloc[0]),
            "episodes": int(len(group)),
            "coll_per_km": collisions / distance_km if distance_km > 0 else np.nan,
            "completion": float(group["distance_completion_rate"].mean()),
            "mean_dist_m": float(group["total_distance_m"].mean()),
            "qp_fail_rate": float(np.average(group["qp_failure_rate"], weights=weights)),
            "intervention": float(np.average(group["event_intervention_rate"], weights=weights)),
            "speed_err": float(np.average(group["mean_abs_speed_deviation"], weights=weights)),
            "lat_err": float(np.average(group["mean_lat_y_error_m"], weights=weights)),
            "jerk": float(np.average(group["mean_jerk_norm"], weights=weights)),
            "collisions": collisions,
            "km": distance_km,
        })
    return pd.DataFrame(rows).sort_values("coll_per_km").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_override_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Episodes per cell.")
    parser.add_argument("--c", type=float, nargs="+", default=list(DEFAULT_C_VALUES))
    parser.add_argument("--pairs", type=_parse_pair, nargs="+", default=None, help="Explicit c1,c2 cells, e.g. 0.5,8 2.3,2.3.")
    parser.add_argument("--no-reference", action="store_true", help="Skip the CBF-OFF and saved-gain reference cells.")
    parser.set_defaults(require_initial_safe_set="false")
    args = parser.parse_args()

    project_root, run_dir, run_config, study_config = resolve_run(args)
    output_dir = prepare_output_dir(project_root, args.output_dir)
    source = _git_provenance(project_root, output_dir, allow_dirty=bool(args.allow_dirty_exploratory))
    env_config, max_policy_steps, changes = apply_env_overrides(
        run_config["env_config"],
        max_policy_steps=int(study_config["evaluation_task_max_policy_steps"]),
        vehicles=args.vehicles,
        hz=args.hz,
        spawn_psi1_gain=args.spawn_psi1_gain,
        require_initial_safe_set=args.require_initial_safe_set == "true",
    )
    eval_args = evaluation_args(
        run_config, study_config, max_policy_steps=max_policy_steps, workers=1, ttc_cap=args.ttc_cap
    )
    saved = run_config["training_signature"]["cbf"]
    saved_gains = {key: float(saved[key]) for key in ("CBF_K0", "CBF_K1", "CBF_PSI1_GAIN")}
    cells: list[tuple[str, str, dict[str, float]]] = []
    if not args.no_reference:
        cells += [("off", "raw", saved_gains),
                  (f"saved_k0={saved_gains['CBF_K0']:g}_k1={saved_gains['CBF_K1']:g}", "cbf", saved_gains)]
    cells += [(label, "cbf", gains) for label, gains in gain_grid_cells(args.c, args.pairs)]
    model_path = run_dir / "model_final.zip"
    manifest: dict[str, Any] = {
        "status": "running",
        "evaluation_kind": "hocbf_gain_grid",
        "entrypoint": "scripts.evaluation.evaluate_hocbf_gain_grid",
        "parameterization": "k1 = c1 + c2, k0 = c1*c2, psi1_gain = min(c1, c2); c1 <= c2 only",
        "source_run_dir": str(run_dir),
        "model_path": str(model_path),
        "model_sha256": progression.protocol.file_sha256(model_path),
        "env_config_changes": changes,
        "cells": {label: {"mode": mode, **gains} for label, mode, gains in cells},
        "episodes_per_cell": int(args.episodes),
        "episode_seed_start": int(args.seed_start),
        "evaluation_workers": int(args.workers),
        "eval_args": vars(eval_args),
        "source": source,
    }
    manifest_path = output_dir / "manifest.json"
    write_manifest(manifest_path, manifest)
    tasks = [
        (label, mode, gains, index + 1, int(args.seed_start) + index)
        for label, mode, gains in cells
        for index in range(int(args.episodes))
    ]
    print(f"[gain-grid] cells={len(cells)} episodes={len(tasks)} workers={args.workers}", flush=True)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        int(args.workers),
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(str(project_root), str(run_dir), env_config, eval_args),
    ) as executor:
        futures = [executor.submit(_evaluate_cell_episode, *task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if done % 20 == 0 or done == len(tasks):
                pd.DataFrame(rows).to_csv(output_dir / "episodes_progress.csv", index=False)
                elapsed = time.perf_counter() - started
                steps = sum(int(row["timesteps"]) for row in rows)
                print(f"[gain-grid] {done}/{len(tasks)} elapsed={elapsed:.0f}s policy_steps/s={steps / elapsed:.0f}", flush=True)
    episodes = pd.DataFrame(rows).sort_values(["cell", "episode_index"])
    episodes.to_csv(output_dir / "e.csv", index=False)
    summary = summarize_cells(episodes)
    summary.to_csv(output_dir / "summary.csv", index=False)
    manifest.update(status="complete", elapsed_s=time.perf_counter() - started)
    write_manifest(manifest_path, manifest)
    with pd.option_context("display.width", 250):
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
