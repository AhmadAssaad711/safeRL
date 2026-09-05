"""Evaluate the two final 50k PPO pilots with the fixed CBF enabled.

This is deliberately an evaluation-only runner: it loads the saved final PPO
snapshots, recreates the exact saved MTM/reward configuration, and applies the
same fixed CBF snapshot used by the nominal-PPO protocol.  The existing raw
rows are copied into the output so the resulting report is a directly paired
raw-versus-CBF comparison over the same ten reset seeds.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import PPO

import scripts.training.run_cbf_filter_ablation as pipeline
from scripts.common.ppo_observation_variants import install_previous_action_observation


FINAL_MODEL_SPECS = (
    ("target_y", "artifacts/ppo_y_desired_50k_cuda8_v2"),
    ("target_y_plus_at1", "artifacts/ppo_y_desired_at1_50k_cuda8_v2"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _number(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _finite_mean(frame: pd.DataFrame, column: str) -> float:
    values = _number(frame, column).to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _summarize(model_id: str, mode: str, frame: pd.DataFrame) -> dict[str, Any]:
    timesteps = float(_number(frame, "timesteps").sum())
    total_return = float(_number(frame, "total_return").sum())
    distance = float(_number(frame, "total_distance_m").sum())
    collisions = float(_number(frame, "distinct_ego_collision_events").sum())
    row: dict[str, Any] = {
        "model_id": model_id,
        "mode": mode,
        "scenarios": int(len(frame)),
        "timesteps": timesteps,
        "total_return": total_return,
        "return_per_timestep": total_return / timesteps if timesteps else float("nan"),
        "total_distance_m": distance,
        "distinct_ego_collision_events": collisions,
        "distance_per_collision_m": distance / collisions if collisions else float("inf"),
        "ego_collisions_per_km": 1000.0 * collisions / distance if distance else float("nan"),
        "collision_free_scenarios": int((_number(frame, "distinct_ego_collision_events") == 0).sum()),
    }
    for metric in (
        "mean_abs_speed_error",
        "mean_jerk_norm",
        "IR",
        "mean_delta_a",
        "shadow_IR",
        "shadow_mean_delta_a",
        "qp_failure_rate",
        "qp_fallback_rate",
    ):
        row[metric] = _finite_mean(frame, metric)
    return row


def _build_runtime(
    project_root: Path,
    run_config: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    namespace = pipeline.bootstrap_notebook_namespace(project_root)
    pipeline.exec_required_notebook_cells(
        project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    if str(run_config.get("observation_variant")) == "target_y_plus_previous_action":
        install_previous_action_observation(namespace)
    snapshot = dict(run_config["fixed_cbf_snapshot"])
    namespace["DEVICE"] = str(device)
    namespace["CBF_K0"] = float(snapshot["k0"])
    namespace["CBF_K1"] = float(snapshot["k1"])
    namespace["CBF_EPS_SIDE"] = float(snapshot["eps_side"])
    namespace["CBF_FILTER_REWARD_LAMBDA"] = 0.0
    namespace["GUIDED_CBF_ENABLE_PROJECTION_REPORTING"] = True
    pipeline.install_minimal_guided_cbf(namespace)
    install_reporting = namespace.get("install_cbf_projection_reporting")
    if callable(install_reporting):
        install_reporting()
    pipeline.install_correction_reward_env(namespace)
    return namespace


def _evaluation_args(run_config: dict[str, Any]) -> argparse.Namespace:
    snapshot = dict(run_config["fixed_cbf_snapshot"])
    seeds = [int(value) for value in run_config["eval_seeds"]]
    timesteps = int(run_config["eval_timesteps"])
    return argparse.Namespace(
        correction_epsilon=0.03,
        k0=float(snapshot["k0"]),
        k1=float(snapshot["k1"]),
        eps_side=float(snapshot["eps_side"]),
        ttc_cap=30.0,
        eval_seeds=seeds,
        eval_seed_start=int(seeds[0]),
        eval_scenarios=len(seeds),
        eval_episodes=len(seeds),
        eval_timesteps=timesteps,
        eval_horizon=timesteps,
    )


def _source_paths(project_root: Path, relative_dir: str) -> tuple[Path, dict[str, Any], Path]:
    source_dir = (project_root / relative_dir).resolve()
    config_path = source_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing PPO run configuration: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["training_seeds"][0])
    variant = str(config["selected_configs"][0])
    target_step = int(config["target_timesteps"])
    model_path = source_dir / f"seed_{seed}" / variant / "model_checkpoints" / f"{target_step:09d}.zip"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing final PPO snapshot: {model_path}")
    return source_dir, config, model_path


def _raw_final_rows(source_dir: Path, config: dict[str, Any], model_id: str) -> pd.DataFrame:
    raw_path = source_dir / "evaluation_scenarios.csv"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing raw PPO evaluation rows: {raw_path}")
    seeds = {int(value) for value in config["eval_seeds"]}
    target_step = int(config["target_timesteps"])
    raw = pd.read_csv(raw_path)
    raw = raw[
        _number(raw, "model_timestep").eq(target_step)
        & _number(raw, "scenario_seed").isin(seeds)
    ].copy()
    if len(raw) != len(seeds):
        raise RuntimeError(
            f"Expected {len(seeds)} raw final rows for {model_id}, found {len(raw)} in {raw_path}"
        )
    raw["model_id"] = model_id
    raw["mode"] = "raw"
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate final PPO pilots with CBF enabled.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/ppo_cbf_on_final_50k_cuda8"),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(args.device).lower().startswith("cuda") and not th.cuda.is_available():
        raise RuntimeError("CBF deployment evaluation requested CUDA, but CUDA is unavailable")
    project_root = pipeline.find_project_root(
        args.project_root or Path(__file__).resolve().parents[2]
    ).resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (project_root / args.output_dir).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    rows_path = output_dir / "cbf_on_scenarios.csv"
    started = time.perf_counter()
    _write_json(
        status_path,
        {
            "state": "running",
            "started_at_utc": _utc_now(),
            "output_dir": str(output_dir),
            "models": [name for name, _ in FINAL_MODEL_SPECS],
            "completed_scenarios": 0,
            "expected_scenarios": 20,
        },
    )
    try:
        rows = pd.read_csv(rows_path).to_dict("records") if rows_path.is_file() else []
        completed = {
            (str(row.get("model_id")), int(row.get("scenario_seed")))
            for row in rows
            if pd.notna(row.get("scenario_seed"))
        }
        raw_frames: list[pd.DataFrame] = []
        source_metadata: list[dict[str, Any]] = []
        for model_id, relative_dir in FINAL_MODEL_SPECS:
            source_dir, run_config, model_path = _source_paths(project_root, relative_dir)
            eval_args = _evaluation_args(run_config)
            env_config = dict(run_config["env_config"])
            reward_config = dict(run_config["reward_config"])
            raw_frames.append(_raw_final_rows(source_dir, run_config, model_id))
            namespace = _build_runtime(project_root, run_config, device=str(args.device))
            model = PPO.load(str(model_path), device=str(args.device))
            try:
                expected_dimension = int(np.prod(model.observation_space.shape))
                for scenario_seed in eval_args.eval_seeds:
                    key = (model_id, int(scenario_seed))
                    if key in completed:
                        continue
                    row, _ = pipeline.evaluate_scenario(
                        namespace,
                        model=model,
                        variant="Q1_stable",
                        mode="cbf",
                        scenario_seed=int(scenario_seed),
                        training_seed=int(run_config["training_seeds"][0]),
                        env_config=env_config,
                        reward_config=reward_config,
                        args=eval_args,
                        critic_calibration_samples=None,
                    )
                    row.update(
                        {
                            "model_id": model_id,
                            "source_dir": str(source_dir),
                            "source_model_path": str(model_path),
                            "observation_variant": str(run_config["observation_variant"]),
                            "model_observation_dimension": expected_dimension,
                            "model_timestep": int(run_config["target_timesteps"]),
                        }
                    )
                    rows.append(row)
                    completed.add(key)
                    pd.DataFrame(rows).to_csv(rows_path, index=False)
                    _write_json(
                        status_path,
                        {
                            "state": "running",
                            "started_at_utc": _utc_now(),
                            "current_model": model_id,
                            "completed_scenarios": len(completed),
                            "expected_scenarios": 20,
                            "last_completed_seed": int(scenario_seed),
                            "elapsed_sec": time.perf_counter() - started,
                        },
                    )
                    print(
                        f"[ppo-cbf-on] model={model_id} seed={int(scenario_seed)} "
                        f"completed={len(completed)}/20",
                        flush=True,
                    )
            finally:
                del model
                if th.cuda.is_available():
                    th.cuda.empty_cache()
            source_metadata.append(
                {
                    "model_id": model_id,
                    "source_dir": str(source_dir),
                    "model_path": str(model_path),
                    "eval_seeds": eval_args.eval_seeds,
                    "eval_timesteps": int(eval_args.eval_timesteps),
                    "cbf_snapshot": run_config["fixed_cbf_snapshot"],
                    "observation_variant": run_config["observation_variant"],
                }
            )

        cbf = pd.DataFrame(rows)
        if len(cbf) != 20:
            raise RuntimeError(f"Expected 20 CBF rows after evaluation, found {len(cbf)}")
        raw = pd.concat(raw_frames, ignore_index=True)
        comparison = pd.concat((raw, cbf), ignore_index=True, sort=False)
        summary = pd.DataFrame(
            [
                _summarize(model_id, mode, group)
                for (model_id, mode), group in comparison.groupby(["model_id", "mode"], sort=True)
            ]
        )
        raw_hashes = raw.set_index(["model_id", "scenario_seed"])["initial_state_hash"]
        cbf_hashes = cbf.set_index(["model_id", "scenario_seed"])["initial_state_hash"]
        hash_matches = raw_hashes.eq(cbf_hashes.reindex(raw_hashes.index))
        summary["initial_state_hash_matches_raw"] = summary["model_id"].map(
            hash_matches.groupby(level=0).all().to_dict()
        )
        comparison.to_csv(output_dir / "raw_and_cbf_scenarios.csv", index=False)
        summary.to_csv(output_dir / "deployment_summary.csv", index=False)
        _write_json(
            output_dir / "evaluation_config.json",
            {
                "source_models": source_metadata,
                "deployment_mode": "cbf",
                "correction_reward": 0.0,
                "protocol": "same final snapshots and fixed evaluation reset seeds as the raw 50k pilots",
            },
        )
        _write_json(
            status_path,
            {
                "state": "complete",
                "completed_at_utc": _utc_now(),
                "completed_scenarios": len(cbf),
                "expected_scenarios": 20,
                "elapsed_sec": time.perf_counter() - started,
                "summary_path": str(output_dir / "deployment_summary.csv"),
                "initial_state_hash_matches_raw": {
                    key: bool(value) for key, value in hash_matches.groupby(level=0).all().items()
                },
            },
        )
        print(f"[ppo-cbf-on] complete: {output_dir}", flush=True)
        return 0
    except Exception:
        _write_json(
            status_path,
            {
                "state": "failed",
                "failed_at_utc": _utc_now(),
                "elapsed_sec": time.perf_counter() - started,
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
