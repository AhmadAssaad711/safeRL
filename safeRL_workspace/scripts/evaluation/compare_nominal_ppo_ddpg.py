"""Compare PPO and DDPG on one frozen nominal lane-free formulation.

This runner is deliberately an orchestration layer around the two existing
nominal pilots.  It does not tune either algorithm and it does not introduce a
new reward.  Both child runs receive the same environment configuration and
the same fixed evaluation seeds; the only intended difference is the learning
algorithm and its algorithm-specific baseline hyperparameters.

The compared formulation is the current nominal formulation (P0/Q0):

* the MTM congested/uncertain lane-free environment;
* the 42-dimensional nearest-neighbor observation from ``KaralakouRewardWrapper``;
* normalized ``[-1, 1]^2`` acceleration commands, mapped by the environment;
* the reciprocal Karalakou reward with the current potential enabled;
* no CBF execution, CBF reward, or actor-side safety loss.

The child pilots retain their strict checkpointing and evaluation logic.  This
file only makes the shared formulation explicit and produces a directly
comparable report.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import scripts.training.run_cbf_filter_ablation as pipeline
from scripts.common.laneless_script_config import active_traffic_model, env_config_from_args
from scripts.training.train_safety_potential_variants import (
    MTM_CONGESTED_UNCERTAIN_UPDATES,
    deep_update,
)


SCHEMA_VERSION = 1
FORMULATION_ID = "P0_current_Q0_aligned"
PPO_PILOT_CONFIG = "Q0_current_aligned"
DDPG_PILOT_CONFIG = "P0_current"
DEFAULT_TIMESTEPS = 50_000
DEFAULT_CHECKPOINT_INTERVAL = 10_000
DEFAULT_TRAINING_SEED = 307
DEFAULT_EVAL_SEEDS = tuple(range(900_000, 900_010))
DEFAULT_EVAL_TIMESTEPS = 800

COMMON_METRICS = (
    "return_per_timestep",
    "total_distance_m",
    "distinct_ego_collision_events",
    "ego_collisions_per_km",
    "distance_per_collision_m",
    "distance_per_collision_exposure_bound_m",
    "mean_abs_speed_error",
    "episode_length_mean",
    "nominal_action_saturation_rate",
)


def canonical_json(value: Any) -> str:
    """Serialize nested config data deterministically for equality checks."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def formulation_signature(
    env_config: Mapping[str, Any], reward_config: Mapping[str, Any]
) -> str:
    payload = {
        "formulation_id": FORMULATION_ID,
        "env_config": dict(env_config),
        "reward_config": dict(reward_config),
        "action_space": {
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
            "semantics": "normalized longitudinal/lateral acceleration",
        },
        "cbf": {"training": False, "evaluation_mode": "raw"},
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_common_formulation(
    namespace: Mapping[str, Any], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, float]]:
    """Resolve the one environment/reward pair used by both child pilots."""

    env_config = env_config_from_args(args, namespace["ENV_CONFIG"])
    if active_traffic_model(env_config) == "mtm":
        deep_update(env_config, copy.deepcopy(MTM_CONGESTED_UNCERTAIN_UPDATES))
    # Make the implicit native default explicit in the shared file.  P0 and Q0
    # must both use the same boundary assistance semantics.
    env_config["ego_boundary_force"] = True
    if not bool(env_config.get("terminate_on_collision", False)):
        raise ValueError("The exact nominal comparison requires terminate_on_collision=True")
    reward_config = pipeline.make_base_reward_config(dict(namespace))
    return env_config, reward_config


def validate_common_formulation(
    env_config: Mapping[str, Any], reward_config: Mapping[str, Any]
) -> None:
    """Reject accidental changes to the formulation before launching training."""

    bounds = env_config.get("bounds", {})
    if tuple(float(value) for value in (bounds.get("ax_min"), bounds.get("ax_max"))) != (-3.0, 3.0):
        raise ValueError("The nominal comparison expects longitudinal bounds [-3, 3]")
    if tuple(float(value) for value in (bounds.get("ay_min"), bounds.get("ay_max"))) != (-3.0, 3.0):
        raise ValueError("The nominal comparison expects lateral bounds [-3, 3]")
    if int(env_config.get("neighbors_count", -1)) != 5:
        raise ValueError("The nominal comparison expects five nearest-neighbor slots")
    if not bool(env_config.get("ego_controlled", False)):
        raise ValueError("The nominal comparison requires ego_controlled=True")
    if not bool(env_config.get("ego_boundary_force", False)):
        raise ValueError("The nominal comparison requires ego_boundary_force=True")
    if str(reward_config.get("reward_mode", "")).lower() != "reciprocal":
        raise ValueError("The nominal comparison requires the reciprocal reward")
    if float(reward_config.get("use_current_potential", 0.0)) != 1.0:
        raise ValueError("The nominal comparison requires the current potential")
    if float(reward_config.get("use_safety_potential", 0.0)) != 0.0:
        raise ValueError("The nominal comparison disables the safety potential")
    if float(reward_config.get("w_safe", 0.0)) != 0.0:
        raise ValueError("The nominal comparison requires w_safe=0")
    if float(reward_config.get("progress_reward_weight", 0.0)) != 0.0:
        raise ValueError("The nominal comparison freezes progress shaping at zero")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def child_command(
    *,
    script_module: str,
    output_dir: Path,
    project_root: Path,
    args: argparse.Namespace,
    config_name: str,
    env_config_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        script_module,
        "--stage",
        "screen",
        "--project-root",
        str(project_root),
        "--output-dir",
        str(output_dir),
        "--device",
        str(args.device),
        "--timesteps",
        str(int(args.timesteps)),
        "--checkpoint-interval",
        str(int(args.checkpoint_interval)),
        "--seeds",
        str(int(args.training_seed)),
        "--configs",
        str(config_name),
        "--eval-seeds",
        *(str(int(seed)) for seed in args.eval_seeds),
        "--eval-timesteps",
        str(int(args.eval_timesteps)),
        "--env-config-file",
        str(env_config_path),
        "--traffic-model",
        str(args.traffic_model),
    ]
    if bool(args.resume):
        command.append("--resume")
    return command


def _run_child(
    label: str,
    command: Sequence[str],
    log_path: Path,
    cwd: Path,
    environment: Optional[Mapping[str, str]] = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[exact-compare] launching {label}: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=None if environment is None else dict(environment),
        )
        if process.stdout is None:
            raise RuntimeError(f"{label} did not expose stdout")
        for line in process.stdout:
            print(f"[{label}] {line}", end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{label} failed with exit code {return_code}; see {log_path}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read JSON configuration: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object at {path}")
    return value


def verify_child_configs(
    ppo_config_path: Path,
    ddpg_config_path: Path,
    env_config: Mapping[str, Any],
    reward_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the saved child manifests after training."""

    ppo_config = _load_json(ppo_config_path)
    ddpg_config = _load_json(ddpg_config_path)
    for key in ("env_config", "reward_config"):
        if canonical_json(ppo_config.get(key, {})) != canonical_json(ddpg_config.get(key, {})):
            raise RuntimeError(f"PPO and DDPG saved {key} values differ")
    if canonical_json(ppo_config.get("env_config", {})) != canonical_json(env_config):
        raise RuntimeError("PPO did not save the requested common environment configuration")
    if canonical_json(ddpg_config.get("env_config", {})) != canonical_json(env_config):
        raise RuntimeError("DDPG did not save the requested common environment configuration")
    if canonical_json(ppo_config.get("reward_config", {})) != canonical_json(reward_config):
        raise RuntimeError("PPO did not save the requested common reward configuration")
    if canonical_json(ddpg_config.get("reward_config", {})) != canonical_json(reward_config):
        raise RuntimeError("DDPG did not save the requested common reward configuration")
    if bool(ppo_config.get("filtered_training", True)) or bool(ddpg_config.get("filtered_training", True)):
        raise RuntimeError("The exact nominal comparison must have filtered_training=False")
    return {
        "ppo": ppo_config,
        "ddpg": ddpg_config,
        "formulation_signature": formulation_signature(env_config, reward_config),
    }


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column).to_numpy(dtype=float)
    weights = _numeric(frame, "timesteps").to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def aggregate_checkpoint(frame: pd.DataFrame) -> dict[str, float | int]:
    """Aggregate fixed-seed scenario rows using exposure-weighted metrics."""

    if frame.empty:
        raise ValueError("Cannot aggregate an empty checkpoint")
    timesteps = float(_numeric(frame, "timesteps").fillna(0.0).sum())
    distance = float(_numeric(frame, "total_distance_m").fillna(0.0).sum())
    collisions = int(_numeric(frame, "distinct_ego_collision_events").fillna(0.0).sum())
    total_return = float(_numeric(frame, "total_return").fillna(0.0).sum())
    segments = float(_numeric(frame, "episode_segments").fillna(0.0).sum())
    episode_length_sum = float(_numeric(frame, "episode_length_sum").fillna(0.0).sum())
    result: dict[str, float | int] = {
        "evaluation_scenarios": int(len(frame)),
        "timesteps": int(timesteps),
        "total_return": total_return,
        "return_per_timestep": total_return / max(timesteps, 1.0),
        "total_distance_m": distance,
        "distinct_ego_collision_events": collisions,
        "ego_collisions_per_km": 1_000.0 * collisions / distance if distance > 1e-12 else float("nan"),
        "distance_per_collision_m": distance / collisions if collisions > 0 else float("inf"),
        "distance_per_collision_exposure_bound_m": distance,
        "collision_free_scenarios": int(
            (_numeric(frame, "distinct_ego_collision_events").fillna(0.0) == 0.0).sum()
        ),
        "mean_abs_speed_error": _weighted_mean(frame, "mean_abs_speed_error"),
        "episode_length_mean": (
            episode_length_sum / segments if segments > 0.0 else _weighted_mean(frame, "episode_length_mean")
        ),
        "nominal_action_saturation_rate": _weighted_mean(
            frame, "nominal_action_saturation_rate"
        ),
    }
    return result


def _read_checkpoint_rows(
    path: Path, expected_steps: Iterable[int], eval_seeds: Sequence[int]
) -> dict[int, pd.DataFrame]:
    if not path.is_file():
        raise RuntimeError(f"Missing evaluation scenarios: {path}")
    frame = pd.read_csv(path)
    required = {"model_timestep", "scenario_seed", "initial_state_hash"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path} is missing evaluation columns: {missing}")
    frame["model_timestep"] = pd.to_numeric(frame["model_timestep"], errors="raise").astype(int)
    frame["scenario_seed"] = pd.to_numeric(frame["scenario_seed"], errors="raise").astype(int)
    expected_seed_set = {int(seed) for seed in eval_seeds}
    result: dict[int, pd.DataFrame] = {}
    for step in expected_steps:
        checkpoint = frame.loc[frame["model_timestep"] == int(step)].copy()
        if set(checkpoint["scenario_seed"]) != expected_seed_set:
            raise RuntimeError(
                f"{path} checkpoint {step} does not cover exactly the requested evaluation seeds"
            )
        if checkpoint["scenario_seed"].duplicated().any():
            raise RuntimeError(f"{path} checkpoint {step} contains duplicate scenario seeds")
        result[int(step)] = checkpoint.sort_values("scenario_seed").reset_index(drop=True)
    return result


def build_comparison_tables(
    *,
    ppo_scenarios: Path,
    ddpg_scenarios: Path,
    timesteps: int,
    checkpoint_interval: int,
    eval_seeds: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    steps = list(range(int(checkpoint_interval), int(timesteps) + 1, int(checkpoint_interval)))
    if not steps or steps[-1] != int(timesteps):
        raise ValueError("timesteps must be a positive multiple of checkpoint_interval")
    ppo_rows = _read_checkpoint_rows(ppo_scenarios, steps, eval_seeds)
    ddpg_rows = _read_checkpoint_rows(ddpg_scenarios, steps, eval_seeds)

    curve_rows: list[dict[str, Any]] = []
    for step in steps:
        ppo_checkpoint = ppo_rows[step]
        ddpg_checkpoint = ddpg_rows[step]
        ppo_hashes = dict(zip(ppo_checkpoint["scenario_seed"], ppo_checkpoint["initial_state_hash"]))
        ddpg_hashes = dict(zip(ddpg_checkpoint["scenario_seed"], ddpg_checkpoint["initial_state_hash"]))
        if ppo_hashes != ddpg_hashes:
            raise RuntimeError(f"PPO/DDPG reset states differ at checkpoint {step}")
        for algorithm, checkpoint in (("PPO", ppo_checkpoint), ("DDPG", ddpg_checkpoint)):
            row = aggregate_checkpoint(checkpoint)
            row.update({"algorithm": algorithm, "model_timestep": int(step)})
            curve_rows.append(row)

    curve = pd.DataFrame(curve_rows)
    final = curve.loc[curve["model_timestep"] == int(timesteps)].copy()
    final = final[["algorithm", "model_timestep", *[column for column in COMMON_METRICS if column in final.columns]]]
    return curve, final


def comparison_delta(final: pd.DataFrame) -> pd.DataFrame:
    if set(final["algorithm"]) != {"PPO", "DDPG"}:
        raise ValueError("Final comparison must contain exactly PPO and DDPG")
    ppo = final.loc[final["algorithm"] == "PPO"].iloc[0]
    ddpg = final.loc[final["algorithm"] == "DDPG"].iloc[0]
    row: dict[str, Any] = {"comparison": "DDPG_minus_PPO"}
    for column in COMMON_METRICS:
        if column in final.columns:
            row[column] = float(ddpg[column]) - float(ppo[column])
    return pd.DataFrame([row])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare nominal PPO and DDPG on the exact same P0/Q0 formulation."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "ppo_ddpg_exact_p0",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--training-seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--eval-timesteps", type=int, default=DEFAULT_EVAL_TIMESTEPS)
    parser.add_argument("--traffic-model", choices=("mtm", "force"), default="mtm")
    parser.add_argument("--env-config-json", default=None)
    parser.add_argument("--env-config-file", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only validate existing ppo/ and ddpg/ child outputs and rebuild reports.",
    )
    return parser.parse_args()


def main() -> int:
    pipeline.set_stable_native_defaults()
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()
    project_root = (
        Path(args.project_root).resolve()
        if args.project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    if int(args.timesteps) <= 0 or int(args.checkpoint_interval) <= 0:
        raise ValueError("timesteps and checkpoint interval must be positive")
    if int(args.timesteps) % int(args.checkpoint_interval) != 0:
        raise ValueError("timesteps must be divisible by checkpoint interval")
    if int(args.timesteps) < 3 * int(args.checkpoint_interval):
        raise ValueError("at least three checkpoints are required for a comparison")
    if not args.eval_seeds or len(set(int(seed) for seed in args.eval_seeds)) != len(args.eval_seeds):
        raise ValueError("eval-seeds must be non-empty and unique")
    if int(args.eval_timesteps) <= 0:
        raise ValueError("eval-timesteps must be positive")

    namespace = pipeline.bootstrap_notebook_namespace(project_root)
    pipeline.exec_required_notebook_cells(
        project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    env_config, reward_config = build_common_formulation(namespace, args)
    validate_common_formulation(env_config, reward_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_config_path = output_dir / "common_env_config.json"
    reward_config_path = output_dir / "common_reward_config.json"
    _write_json(env_config_path, env_config)
    _write_json(reward_config_path, reward_config)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study": "exact_nominal_ppo_ddpg_comparison",
        "formulation_id": FORMULATION_ID,
        "ppo_pilot_config": PPO_PILOT_CONFIG,
        "ddpg_pilot_config": DDPG_PILOT_CONFIG,
        "training_seed": int(args.training_seed),
        "target_timesteps": int(args.timesteps),
        "checkpoint_interval": int(args.checkpoint_interval),
        "evaluation_seeds": [int(seed) for seed in args.eval_seeds],
        "evaluation_timesteps": int(args.eval_timesteps),
        "environment_config_path": str(env_config_path),
        "reward_config_path": str(reward_config_path),
        "environment_config": env_config,
        "reward_config": reward_config,
        "shared_invariants": {
            "observation": "42-dimensional Karalakou nearest-neighbor observation",
            "action_space": {"low": [-1.0, -1.0], "high": [1.0, 1.0]},
            "action_semantics": "normalized longitudinal/lateral acceleration",
            "reward": "reciprocal Karalakou reward; current potential on",
            "cbf_training": False,
            "evaluation_mode": "raw",
            "collision_protocol": "terminate on collision, reset immediately, fixed timestep budget",
        },
        "formulation_signature": formulation_signature(env_config, reward_config),
    }
    _write_json(output_dir / "comparison_manifest.json", manifest)

    ppo_dir = output_dir / "ppo"
    ddpg_dir = output_dir / "ddpg"
    if not bool(args.skip_training):
        # TensorBoard appends its run name and event filename to this path.
        # Keep that disposable path short on Windows; the durable models and
        # CSVs remain in the requested output directory.
        tensorboard_root = Path(tempfile.gettempdir()) / "highway_rl_exact_compare" / output_dir.name
        tensorboard_root.mkdir(parents=True, exist_ok=True)
        child_environment = os.environ.copy()
        child_environment["NOMINAL_PPO_PILOT_TENSORBOARD_ROOT"] = str(
            tensorboard_root / "ppo"
        )
        child_environment["NOMINAL_DDPG_PILOT_TENSORBOARD_ROOT"] = str(
            tensorboard_root / "ddpg"
        )
        _run_child(
            "PPO",
            child_command(
                script_module="scripts.training.run_nominal_ppo_parameter_pilot",
                output_dir=ppo_dir,
                project_root=project_root,
                args=args,
                config_name=PPO_PILOT_CONFIG,
                env_config_path=env_config_path,
            ),
            output_dir / "ppo_training.log",
            project_root,
            child_environment,
        )
        _run_child(
            "DDPG",
            child_command(
                script_module="scripts.training.run_nominal_ddpg_parameter_pilot",
                output_dir=ddpg_dir,
                project_root=project_root,
                args=args,
                config_name=DDPG_PILOT_CONFIG,
                env_config_path=env_config_path,
            ),
            output_dir / "ddpg_training.log",
            project_root,
            child_environment,
        )

    saved_configs = verify_child_configs(
        ppo_dir / "run_config.json",
        ddpg_dir / "run_config.json",
        env_config,
        reward_config,
    )
    manifest["saved_child_formulation_signature"] = saved_configs["formulation_signature"]
    _write_json(output_dir / "comparison_manifest.json", manifest)

    curve, final = build_comparison_tables(
        ppo_scenarios=ppo_dir / "evaluation_scenarios.csv",
        ddpg_scenarios=ddpg_dir / "evaluation_scenarios.csv",
        timesteps=int(args.timesteps),
        checkpoint_interval=int(args.checkpoint_interval),
        eval_seeds=[int(seed) for seed in args.eval_seeds],
    )
    curve.to_csv(output_dir / "checkpoint_comparison.csv", index=False)
    final.to_csv(output_dir / "final_comparison.csv", index=False)
    comparison_delta(final).to_csv(output_dir / "final_comparison_delta_ddpg_minus_ppo.csv", index=False)

    display_columns = [
        "algorithm",
        "return_per_timestep",
        "total_distance_m",
        "distinct_ego_collision_events",
        "ego_collisions_per_km",
        "distance_per_collision_m",
        "mean_abs_speed_error",
        "episode_length_mean",
        "nominal_action_saturation_rate",
    ]
    print("\n[exact-compare] final raw-deployment comparison", flush=True)
    print(final[[column for column in display_columns if column in final.columns]].to_string(index=False), flush=True)
    print(
        f"[exact-compare] wrote {output_dir / 'final_comparison.csv'}"
        f" and {output_dir / 'checkpoint_comparison.csv'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
