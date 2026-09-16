"""Evaluate the last N checkpoints of a run instead of only its final step.

Training performs **no model selection**: ``run_ppo_cbf_progression`` saves the
policy as it stands after the last gradient step (``model.save`` immediately
after ``model.learn``) and the 200-episode KPI table is computed from that one
file. The archived ``rollout_*_steps.zip`` checkpoints exist only for crash
resumption, and even there the rule is "latest", not "best".

That is fine when a run has converged to a point and stays there. It is not
fine when the run is oscillating around a plateau, which is what the training
curves show: on one measured 1M control the mean episode length over the last
four deciles was 410, 361, 421, 445, a spread of about +/-12%. Taking the final
gradient step alone samples one draw from that oscillation, so part of the
seed-to-seed spread in any comparison is selection noise rather than a real
difference between policies.

This evaluates the last N distinct checkpoints under the identical paired
OFF/ON protocol and reports:

* each checkpoint's KPI table;
* the mean across checkpoints, which is the de-noised estimate;
* the **spread** across checkpoints, which measures how much the final-step
  convention was costing in the first place.

It changes no training behaviour and needs no retraining: every existing run
already has its checkpoints on disk, so this applies retroactively and, run
over every arm of a study, keeps the comparison uniform.

``model_final.zip`` is always included as the newest member. A rollout
checkpoint at the same step count is *not* the same policy -- the final save
happens after the last update, the callback fires before it -- so both are kept
when they exist and the older rollout files fill the remaining slots.
"""

from __future__ import annotations

import os

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import argparse
import copy
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.training.run_ppo_cbf_progression as progression
from scripts.evaluation.evaluate_ppo_cbf_overrides import (
    apply_env_overrides,
    build_namespace,
    load_source_run,
)

DEFAULT_CHECKPOINTS = 3
DEFAULT_EPISODES = 200
DEFAULT_SEED_START = 1_100_000


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"rollout_(\d+)_steps\.zip$", path.name)
    return int(match.group(1)) if match else -1


def select_checkpoints(run_dir: Path, count: int) -> list[tuple[str, Path]]:
    """Return the newest ``count`` distinct policies, newest first."""

    selected: list[tuple[str, Path]] = []
    final = run_dir / "model_final.zip"
    if final.is_file():
        selected.append(("final", final))
    rollouts = sorted(
        (p for p in (run_dir / "checkpoints").glob("rollout_*_steps.zip")),
        key=_checkpoint_step,
        reverse=True,
    )
    for path in rollouts:
        if len(selected) >= count:
            break
        selected.append((f"step_{_checkpoint_step(path)}", path))
    if not selected:
        raise FileNotFoundError(f"No checkpoints or final model under {run_dir}")
    return selected[:count]


def evaluate_one(
    namespace: dict[str, Any],
    *,
    label: str,
    model_path: Path,
    run_config: dict[str, Any],
    study_config: dict[str, Any],
    env_config: dict[str, Any],
    args: argparse.Namespace,
    scratch_dir: Path,
) -> pd.DataFrame:
    """Run the paired OFF/ON protocol for one checkpoint."""

    variant = str(run_config["variant"])
    training_seed = int(run_config["training_signature"]["training_seed"])

    # The evaluation helpers read only these fields from args, so build the
    # namespace explicitly rather than reconstructing the full training parser.
    evaluation_args = argparse.Namespace()
    for field, value in (
        ("post_train_eval_episodes", int(args.episodes)),
        ("post_train_eval_seed_start", int(args.seed_start)),
        ("post_train_eval_workers", int(args.workers)),
        ("device", str(args.device)),
        ("correction_epsilon", float(
            run_config["training_signature"]["correction_epsilon"]
        )),
        ("task_distance_m", float(study_config["evaluation_task_distance_m"])),
        ("task_max_policy_steps", int(
            study_config["evaluation_task_max_policy_steps"]
        )),
        ("ttc_cap", float(study_config.get("evaluation_ttc_cap", 30.0))),
    ):
        setattr(evaluation_args, field, value)

    progress_dir = scratch_dir / label
    progress_dir.mkdir(parents=True, exist_ok=True)
    rows = progression._evaluate_complete_episode_rows(
        namespace,
        model_path=model_path,
        variant=variant,
        training_seed=training_seed,
        env_config=copy.deepcopy(env_config),
        reward_config=copy.deepcopy(run_config["reward_config"]),
        args=evaluation_args,
        modes=tuple(progression.EVALUATION_MODES),
        episode_count=int(args.episodes),
        seed_start=int(args.seed_start),
        action_source="policy",
        progress_path=progress_dir / "episodes.csv",
        status_path=progress_dir / "progress.json",
        progress_started=time.perf_counter(),
        progress_variant=f"{variant}@{label}",
    )
    metrics = pd.DataFrame(rows)
    kpis, _blocks, _counts = progression.summarize_post_training_episodes(
        metrics, env_config=env_config
    )
    kpis = kpis.copy()
    kpis.insert(0, "checkpoint", label)
    kpis.insert(1, "model_sha256", progression.protocol.file_sha256(model_path))
    return kpis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=int, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if int(args.checkpoints) < 1:
        raise ValueError("--checkpoints must be at least 1")

    project_root = progression.protocol.find_project_root(
        args.project_root or Path.cwd()
    ).resolve()
    run_dir = (
        args.run_dir if args.run_dir.is_absolute() else project_root / args.run_dir
    ).resolve()
    run_config, study_config = load_source_run(run_dir)
    env_config, _max_steps, _changes = apply_env_overrides(
        run_config["env_config"],
        max_policy_steps=int(study_config["evaluation_task_max_policy_steps"]),
        vehicles=None,
    )
    namespace = build_namespace(project_root, run_config)

    selected = select_checkpoints(run_dir, int(args.checkpoints))
    print(
        f"[ensemble] {run_config['variant']} seed "
        f"{run_config['training_signature']['training_seed']}: "
        + ", ".join(label for label, _path in selected),
        flush=True,
    )

    # --out names a directory; create it rather than silently writing beside it.
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = out_dir / "_scratch"

    tables = [
        evaluate_one(
            namespace,
            label=label,
            model_path=path,
            run_config=run_config,
            study_config=study_config,
            env_config=env_config,
            args=args,
            scratch_dir=scratch_dir,
        )
        for label, path in selected
    ]
    combined = pd.concat(tables, ignore_index=True)

    value_columns = [
        column
        for column in combined.columns
        if column not in {"checkpoint", "model_sha256", "mode", "variant",
                          "variant_label", "kpi", "KPI"}
        and pd.api.types.is_numeric_dtype(combined[column])
    ]
    group_columns = [c for c in ("mode", "kpi", "KPI") if c in combined.columns]
    ensemble = (
        combined.groupby(group_columns, sort=False)[value_columns]
        .agg(["mean", "std"])
        if group_columns
        else None
    )

    combined.to_csv(out_dir / "checkpoint_ensemble_per_checkpoint.csv", index=False)
    if ensemble is not None:
        ensemble.to_csv(out_dir / "checkpoint_ensemble_mean.csv")
    (out_dir / "checkpoint_ensemble_manifest.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "variant": run_config["variant"],
                "training_seed": run_config["training_signature"]["training_seed"],
                "checkpoints": [
                    {"label": label, "path": str(path),
                     "sha256": progression.protocol.file_sha256(path)}
                    for label, path in selected
                ],
                "episodes_per_mode": int(args.episodes),
                "seed_start": int(args.seed_start),
                "modes": list(progression.EVALUATION_MODES),
                "selection_policy": (
                    "last N distinct checkpoints, newest first, model_final included"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[ensemble] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
