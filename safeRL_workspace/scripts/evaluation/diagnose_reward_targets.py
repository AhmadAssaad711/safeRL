"""Characterise the reward's lateral and speed targets before training on them.

Read-only.  A saved policy is rolled out through the evaluation environment
under a *given* reward configuration, and the two guidance signals the reward
tracks -- ``target_y`` and ``target_speed`` -- are logged at every policy step
together with the ego state.  Nothing is trained.

The questions this answers for each target are the ones that sank the earlier
designs:

* **Is it informative?**  A target that never moves carries no gradient and
  wastes its observation slot; the notebook speed target is the constant
  ``ego_desired_speed``.
* **Is it smooth?**  The original blocker-derived speed target was replaced by
  a fixed one because it jumped when the selected blocker changed.  The
  per-step change distribution, and its worst case, say whether a new rule
  reintroduces that.
* **Is it reachable?**  A target the ego is always far from is a constant
  penalty, not guidance.  The notebook lateral target sits on the ego's
  opposite road half about half the time at the canonical density.
* **How often does it bind?**  A cap that never activates changes nothing.

Compare two configurations by running it twice with different flags; the
per-step CSV is written for exactly that.
"""

from __future__ import annotations

import os

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scripts.training.run_ppo_cbf_progression as progression
from saferl.rewards import (
    add_reward_variant_arguments,
    apply_reward_variant_arguments,
    reward_variant_summary,
    speed_target_parameters,
)
from scripts.evaluation.evaluate_cbf_random_actions import _git_provenance
from scripts.evaluation.evaluate_ppo_cbf_overrides import (
    apply_env_overrides,
    build_namespace,
    load_source_run,
    prepare_output_dir,
    write_manifest,
)

DEFAULT_EPISODES = 30
DEFAULT_SEED_START = 1_100_000


def rollout(
    namespace: dict[str, Any],
    *,
    model: Any,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    mode: str,
    scenario_seed: int,
    correction_epsilon: float,
    task_distance_m: float,
    task_max_policy_steps: int,
) -> list[dict[str, Any]]:
    env = progression.make_evaluation_env(
        namespace,
        mode=mode,
        env_config=env_config,
        reward_config=reward_config,
        correction_epsilon=float(correction_epsilon),
        task_distance_m=float(task_distance_m),
        task_max_policy_steps=int(task_max_policy_steps),
    )
    rows: list[dict[str, Any]] = []
    try:
        observation, _ = env.reset(seed=int(scenario_seed))
        for step in range(int(task_max_policy_steps)):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            info = dict(info)
            rows.append(
                {
                    "seed": int(scenario_seed),
                    "step": int(step),
                    "reward": float(reward),
                    "ego_y": float(info.get("karalakou_ego_y", np.nan)),
                    "ego_speed": float(info.get("karalakou_ego_speed", np.nan)),
                    "target_y": float(info.get("karalakou_target_y", np.nan)),
                    "target_speed": float(info.get("karalakou_target_speed", np.nan)),
                    "zone_found": float(info.get("karalakou_zone_found", np.nan)),
                    "cx": float(info.get("karalakou_cx", np.nan)),
                    "cy": float(info.get("karalakou_cy", np.nan)),
                    "cf": float(info.get("karalakou_cf", np.nan)),
                    "lat_y_error_m": float(info.get("karalakou_lat_y_error_m", np.nan)),
                    "collision": float(info.get("karalakou_ego_collision", 0.0)),
                }
            )
            if bool(terminated) or bool(truncated):
                break
    finally:
        env.close()
    return rows


def _jump_statistics(steps: pd.DataFrame, column: str) -> dict[str, float]:
    """Per-step change of a target, pooled within episodes."""

    deltas = (
        steps.groupby("seed", sort=False)[column].diff().abs().dropna().to_numpy(dtype=float)
    )
    if deltas.size == 0:
        return {}
    return {
        f"{column}_mean_abs_step_change": float(deltas.mean()),
        f"{column}_p95_abs_step_change": float(np.quantile(deltas, 0.95)),
        f"{column}_max_abs_step_change": float(deltas.max()),
        f"{column}_frozen_rate": float((deltas < 1e-9).mean()),
    }


def summarize(
    steps: pd.DataFrame, *, reward_config: dict[str, Any], road_width: float
) -> dict[str, Any]:
    ego_half = 0.5 * float(road_width)
    opposite_half = (steps["ego_y"] - ego_half) * (steps["target_y"] - ego_half) < 0.0
    v_nominal = speed_target_parameters(reward_config)["v_nominal"]
    summary: dict[str, Any] = {
        "episodes": int(steps["seed"].nunique()),
        "steps": int(len(steps)),
        "collisions": int(steps["collision"].sum()),
        # --- lateral target ---
        "zone_found_rate": float(steps["zone_found"].mean()),
        "target_y_mean": float(steps["target_y"].mean()),
        "target_y_sd": float(steps["target_y"].std()),
        "target_y_at_center_rate": float((steps["target_y"] - ego_half).abs().lt(1e-6).mean()),
        "target_on_opposite_road_half_rate": float(opposite_half.mean()),
        "lateral_error_mean_m": float(steps["lat_y_error_m"].mean()),
        "mean_cy": float(steps["cy"].mean()),
        # --- speed target ---
        "target_speed_mean": float(steps["target_speed"].mean()),
        "target_speed_sd": float(steps["target_speed"].std()),
        "target_speed_p10": float(steps["target_speed"].quantile(0.10)),
        "target_speed_p50": float(steps["target_speed"].quantile(0.50)),
        "target_speed_binding_rate": float(steps["target_speed"].lt(v_nominal - 0.25).mean()),
        "ego_speed_mean": float(steps["ego_speed"].mean()),
        "ego_below_target_rate": float((steps["ego_speed"] < steps["target_speed"]).mean()),
        "mean_cx": float(steps["cx"].mean()),
        "mean_cf": float(steps["cf"].mean()),
        "mean_reward": float(steps["reward"].mean()),
    }
    summary.update(_jump_statistics(steps, "target_y"))
    summary.update(_jump_statistics(steps, "target_speed"))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--mode", choices=tuple(progression.EVALUATION_MODES), default="raw")
    parser.add_argument("--vehicles", type=int, default=None)
    parser.add_argument("--label", default=None, help="Name for this configuration in the summary.")
    parser.add_argument("--allow-dirty-exploratory", action="store_true")
    add_reward_variant_arguments(parser)
    args = parser.parse_args()

    project_root = progression.protocol.find_project_root(args.project_root or Path.cwd()).resolve()
    run_dir = (args.run_dir if args.run_dir.is_absolute() else project_root / args.run_dir).resolve()
    run_config, study_config = load_source_run(run_dir)
    output_dir = prepare_output_dir(project_root, args.output_dir)
    source = _git_provenance(project_root, output_dir, allow_dirty=bool(args.allow_dirty_exploratory))

    env_config, max_policy_steps, changes = apply_env_overrides(
        run_config["env_config"],
        max_policy_steps=int(study_config["evaluation_task_max_policy_steps"]),
        vehicles=args.vehicles,
    )
    reward_config = copy.deepcopy(run_config["reward_config"])
    apply_reward_variant_arguments(args, reward_config)
    namespace = build_namespace(project_root, run_config)
    model_path = run_dir / "model_final.zip"
    model = progression.load_model(run_config["variant"], model_path, device="cpu")

    variants = reward_variant_summary(reward_config)
    manifest: dict[str, Any] = {
        "status": "running",
        "evaluation_kind": "reward_target_behaviour",
        "entrypoint": "scripts.evaluation.diagnose_reward_targets",
        "label": args.label or "unlabelled",
        "source_run_dir": str(run_dir),
        "variant": run_config["variant"],
        "model_path": str(model_path),
        "model_sha256": progression.protocol.file_sha256(model_path),
        "mode": str(args.mode),
        "env_config_changes": changes,
        "reward_variants": variants,
        "reward_weights": {key: reward_config.get(key) for key in ("epsilon_r", "wx", "wy", "wf", "way")},
        "episodes": int(args.episodes),
        "episode_seed_start": int(args.seed_start),
        "source": source,
    }
    manifest_path = output_dir / "manifest.json"
    write_manifest(manifest_path, manifest)
    print(f"[targets] {args.label or run_config['variant']} variants={variants}", flush=True)

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for index in range(int(args.episodes)):
        seed = int(args.seed_start) + index
        rows.extend(
            rollout(
                namespace,
                model=model,
                env_config=env_config,
                reward_config=reward_config,
                mode=str(args.mode),
                scenario_seed=seed,
                correction_epsilon=float(run_config["training_signature"]["correction_epsilon"]),
                task_distance_m=float(study_config["evaluation_task_distance_m"]),
                task_max_policy_steps=int(max_policy_steps),
            )
        )
    elapsed = time.perf_counter() - started

    steps = pd.DataFrame(rows)
    steps.to_csv(output_dir / "steps.csv", index=False)
    summary = summarize(
        steps, reward_config=reward_config, road_width=float(env_config["road_width"])
    )
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    manifest.update(summary, status="complete", elapsed_s=elapsed)
    write_manifest(manifest_path, manifest)

    print(f"\n[targets] {args.label or run_config['variant']}: pooled over {summary['steps']} steps")
    for key, value in summary.items():
        print(f"  {key:42s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
