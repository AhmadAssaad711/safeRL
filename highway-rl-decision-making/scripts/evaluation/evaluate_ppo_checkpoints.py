"""Evaluate saved nominal PPO checkpoints on a fixed CBF-OFF scenario set.

This is deliberately a thin wrapper around the canonical evaluation helpers in
``run_ppo_cbf_progression.py``.  It does not train, alter, or replace any
checkpoint.  The default ranking is safety-first and is reported alongside a
return-first candidate so checkpoint selection remains inspectable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

import scripts.training.run_ppo_cbf_progression as progression


CHECKPOINT_PATTERN = re.compile(r"rollout_(?P<steps>\d+)_steps\.zip$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved nominal PPO checkpoints with external CBF OFF."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed ppo_nominal/seed_<seed> directory containing checkpoints.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Fixed CBF-OFF episodes per checkpoint; default=%(default)s.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1_300_000,
        help="First paired scenario seed reused for every checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-stride",
        type=int,
        default=10_000,
        help="Evaluate every saved checkpoint at this step interval; ignored when --checkpoint-steps is provided.",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate exactly these saved checkpoint step counts.",
    )
    parser.add_argument(
        "--ttc-cap",
        type=float,
        default=30.0,
        help="TTC cap used by the existing evaluation diagnostics.",
    )
    return parser.parse_args()


def _checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Unrecognized PPO checkpoint filename: {path}")
    return int(match.group("steps"))


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint run configuration: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("variant") != "ppo_nominal":
        raise ValueError(
            "Checkpoint sweep is intentionally limited to ppo_nominal; "
            f"observed variant={config.get('variant')!r}."
        )
    required = ("training_seed", "env_config", "reward_config", "training_signature")
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError("Run configuration is missing: " + ", ".join(missing))
    return config


def _override_cbf_snapshot(namespace: dict[str, Any], config: dict[str, Any]) -> None:
    snapshot = config.get("training_signature", {}).get("cbf", {})
    if not isinstance(snapshot, dict):
        raise TypeError("training_signature.cbf must be a JSON object")
    for key, value in snapshot.items():
        if key.startswith("CBF_"):
            namespace[key] = value


def _evaluation_args(config: dict[str, Any], parsed: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        device=str(parsed.device),
        correction_epsilon=float(config.get("correction_epsilon", 0.03)),
        ttc_cap=float(parsed.ttc_cap),
    )


def _checkpoint_paths(
    run_dir: Path,
    stride: int,
    requested_steps: list[int] | None,
) -> list[tuple[int, Path]]:
    if requested_steps:
        selected: list[tuple[int, Path]] = []
        for steps in sorted(set(int(value) for value in requested_steps)):
            path = run_dir / "checkpoints" / f"rollout_{steps}_steps.zip"
            if not path.is_file():
                raise FileNotFoundError(f"Requested checkpoint does not exist: {path}")
            selected.append((steps, path))
        return selected
    if stride <= 0:
        raise ValueError("checkpoint-stride must be positive")
    candidates: list[tuple[int, Path]] = []
    for path in (run_dir / "checkpoints").glob("rollout_*_steps.zip"):
        steps = _checkpoint_step(path)
        if steps % stride == 0:
            candidates.append((steps, path))
    candidates.sort(key=lambda item: item[0])
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints divisible by stride={stride} found in {run_dir / 'checkpoints'}"
        )
    return candidates


def _summarize(rows: list[dict[str, Any]], checkpoint_steps: int) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No evaluation rows for checkpoint {checkpoint_steps}")
    distance_m = float(pd.to_numeric(frame["total_distance_m"], errors="coerce").sum())
    collisions = int(
        pd.to_numeric(frame["distinct_ego_collision_events"], errors="coerce")
        .fillna(0.0)
        .sum()
    )
    return {
        "checkpoint_steps": int(checkpoint_steps),
        "episodes": int(len(frame)),
        "mean_return": float(frame["episode_return"].mean()),
        "sd_return": float(frame["episode_return"].std(ddof=1)),
        "mean_episode_length": float(frame["episode_length_steps"].mean()),
        "collision_episode_rate": float(
            (frame["distinct_ego_collision_events"] > 0).mean()
        ),
        "collision_episodes": int(
            (frame["distinct_ego_collision_events"] > 0).sum()
        ),
        "collision_events": collisions,
        "distance_km": float(distance_m / 1000.0),
        "pooled_collisions_per_km": (
            float(collisions / (distance_m / 1000.0))
            if distance_m > 1e-9
            else float("nan")
        ),
        "mean_h_min": float(frame["h_min"].mean()),
        "mean_jerk_norm": float(frame["mean_jerk_norm"].mean()),
    }


def _select_rows(summary: pd.DataFrame) -> dict[str, dict[str, Any]]:
    safety = summary.sort_values(
        [
            "collision_episode_rate",
            "pooled_collisions_per_km",
            "mean_return",
        ],
        ascending=[True, True, False],
        kind="stable",
    ).iloc[0]
    distance_safety = summary.sort_values(
        ["pooled_collisions_per_km", "collision_episode_rate", "mean_return"],
        ascending=[True, True, False],
        kind="stable",
    ).iloc[0]
    return_first = summary.sort_values(
        ["mean_return", "collision_episode_rate", "pooled_collisions_per_km"],
        ascending=[False, True, True],
        kind="stable",
    ).iloc[0]
    return {
        "safety_first_lexicographic": safety.to_dict(),
        "distance_safety_first": distance_safety.to_dict(),
        "return_first": return_first.to_dict(),
    }


def main() -> int:
    progression.protocol.set_stable_native_defaults()
    os.environ.setdefault("MPLBACKEND", "Agg")
    parsed = parse_args()
    project_root = progression.protocol.find_project_root(
        parsed.project_root or Path.cwd()
    )
    run_dir = parsed.run_dir.resolve()
    config = _load_run_config(run_dir)
    episodes = int(parsed.episodes)
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    scenario_seeds = [int(parsed.seed_start) + index for index in range(episodes)]
    checkpoints = _checkpoint_paths(
        run_dir,
        int(parsed.checkpoint_stride),
        parsed.checkpoint_steps,
    )

    namespace = progression.protocol.bootstrap_notebook_namespace(project_root)
    progression.protocol.exec_required_notebook_cells(
        project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    _override_cbf_snapshot(namespace, config)
    evaluation_args = _evaluation_args(config, parsed)
    env_config = config["env_config"]
    reward_config = config["reward_config"]
    training_seed = int(config["training_seed"])

    print(
        "[ppo-checkpoint-sweep] starting",
        {
            "run_dir": str(run_dir),
            "checkpoints": len(checkpoints),
            "checkpoint_steps": [step for step, _path in checkpoints],
            "episodes_per_checkpoint": episodes,
            "mode": "CBF OFF",
            "scenario_seeds": [scenario_seeds[0], scenario_seeds[-1]],
        },
        flush=True,
    )

    summaries: list[dict[str, Any]] = []
    for index, (checkpoint_steps, checkpoint_path) in enumerate(checkpoints, start=1):
        model = progression.load_model(
            "ppo_nominal", checkpoint_path, str(parsed.device)
        )
        rows: list[dict[str, Any]] = []
        for episode_index, episode_seed in enumerate(scenario_seeds, start=1):
            rows.append(
                progression.evaluate_completed_episode(
                    namespace,
                    model=model,
                    variant="ppo_nominal",
                    mode="raw",
                    training_seed=training_seed,
                    episode_index=episode_index,
                    episode_seed=episode_seed,
                    env_config=env_config,
                    reward_config=reward_config,
                    args=evaluation_args,
                )
            )
        summary = _summarize(rows, checkpoint_steps)
        summary["checkpoint_path"] = str(checkpoint_path.resolve())
        summaries.append(summary)
        print(
            "[ppo-checkpoint-sweep] checkpoint",
            f"{index}/{len(checkpoints)}",
            {
                "steps": checkpoint_steps,
                "collision_episode_rate": round(summary["collision_episode_rate"], 4),
                "pooled_collisions_per_km": round(
                    summary["pooled_collisions_per_km"], 4
                ),
                "mean_return": round(summary["mean_return"], 4),
                "mean_length": round(summary["mean_episode_length"], 2),
            },
            flush=True,
        )

    summary_frame = pd.DataFrame(summaries).sort_values("checkpoint_steps")
    selections = _select_rows(summary_frame)
    output_dir = (
        parsed.output_dir
        or run_dir / "checkpoint_sweep_raw"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "checkpoint_sweep_summary.csv"
    manifest_path = output_dir / "checkpoint_sweep_manifest.json"
    summary_frame.to_csv(summary_path, index=False)
    manifest = {
        "schema_version": 1,
        "evaluation_kind": "nominal_ppo_checkpoint_sweep",
        "mode": "CBF OFF",
        "run_dir": str(run_dir),
        "training_seed": training_seed,
        "episodes_per_checkpoint": episodes,
        "scenario_seed_start": int(parsed.seed_start),
        "scenario_seeds_reused_for_every_checkpoint": True,
        "checkpoint_stride": int(parsed.checkpoint_stride),
        "summary_path": str(summary_path),
        "selections": selections,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print("[ppo-checkpoint-sweep] summary", summary_path, flush=True)
    print(json.dumps(selections, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
