"""Multi-fidelity Bayesian-optimization sweep over PPO's own hyperparameters
(learning rate, rollout/batch sizing, epochs, gamma, entropy coef, clip
range, GAE lambda, value-loss coef) for the A1 (nominal) arm, using Ax.

Deliberately excludes every problem-specific parameter (CBF gains, the
lambda_detached_actor/lambda_mean actor-loss weights, reward shaping) --
this searches PPO mechanics only, on the arm that has none of those extra
terms, so the result is meaningful to warm-start every other arm's shared
infra hyperparameters too.

Two phases:
  1. Search: Sobol init + GP-BO trials at a reduced timestep budget
     (--search-timesteps, default 250k) with a smaller post-train eval, to
     keep each trial cheap.
  2. Confirm: the top --n-confirm candidates by predicted objective are
     re-run at full budget (--confirm-timesteps, default 1M) with
     --confirm-seeds-per-candidate seeds each, and the winner is the one
     that actually holds up out-of-sample -- never trust the search-phase
     ranking alone.

Objective: maximize CBF-ON distance-based completion rate, subject to
CBF-ON collisions/km staying at or below the current Q1_stable baseline
(0.303, A1's own already-measured number) plus a small slack -- a
constrained objective so the search can't just find a faster-but-more-
dangerous policy.

Every trial launch is wrapped by a watchdog: if no new checkpoint appears
for --stall-minutes, the subprocess is killed and the trial is logged to
Ax as failed (not as a fabricated bad score) rather than hanging the whole
sweep the way the density=60 run hung earlier tonight.

The whole state is checkpointed to --state-file after every trial via
AxClient.save_to_json_file, so the sweep survives an interrupted
orchestrator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models

HOME = Path.home()
DEFAULT_OUT_ROOT = HOME / "AppData" / "Local" / "Temp" / "saferl_runs" / "ax_ppo_sweep_nominal"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_COLLISIONS_PER_KM = 0.303  # A1's own current Q1_stable CBF-ON number
COLLISION_CONSTRAINT_SLACK = 0.05   # allow a little room, don't force strictly <= baseline

N_ENVS = 20
COMPLETED_MARKER = "training_complete.json"


def n_steps_choices() -> list[int]:
    return [1000, 1500, 2000]


def n_minibatch_choices() -> list[int]:
    return [5, 10, 20]


def build_search_space() -> list[dict[str, Any]]:
    return [
        {"name": "learning_rate", "type": "range", "bounds": [2e-5, 5e-4], "log_scale": True, "value_type": "float"},
        {"name": "n_steps", "type": "choice", "values": n_steps_choices(), "value_type": "int", "is_ordered": True},
        {"name": "n_minibatches", "type": "choice", "values": n_minibatch_choices(), "value_type": "int", "is_ordered": True},
        {"name": "n_epochs", "type": "range", "bounds": [4, 15], "value_type": "int"},
        {"name": "gamma", "type": "range", "bounds": [0.97, 0.999], "value_type": "float"},
        {"name": "ent_coef", "type": "range", "bounds": [1e-4, 2e-2], "log_scale": True, "value_type": "float"},
        {"name": "clip_range", "type": "range", "bounds": [0.1, 0.3], "value_type": "float"},
        {"name": "gae_lambda", "type": "range", "bounds": [0.90, 0.99], "value_type": "float"},
        {"name": "vf_coef", "type": "range", "bounds": [0.25, 1.0], "value_type": "float"},
    ]


def resolved_ppo_cli_args(params: dict[str, Any]) -> list[str]:
    n_steps = int(params["n_steps"])
    n_minibatches = int(params["n_minibatches"])
    batch_size = n_steps // n_minibatches
    return [
        "--learning-rate", str(float(params["learning_rate"])),
        "--n-steps", str(n_steps),
        "--batch-size", str(batch_size),
        "--n-epochs", str(int(params["n_epochs"])),
        "--gamma", str(float(params["gamma"])),
        "--ent-coef", str(float(params["ent_coef"])),
        "--clip-range", str(float(params["clip_range"])),
        "--gae-lambda", str(float(params["gae_lambda"])),
        "--vf-coef", str(float(params["vf_coef"])),
    ]


def build_train_cmd(
    params: dict[str, Any], *, seed: int, timesteps: int, eval_episodes: int,
    output_dir: Path, run_label: str,
) -> list[str]:
    cmd = [
        "python", "-m", "scripts.training.run_ppo_cbf_progression",
        "--traffic-model", "mtm", "--device", "cpu",
        "--timesteps", str(timesteps),
        "--ppo-config", "Q1_stable",
        "--n-envs", str(N_ENVS),
        "--checkpoint-freq", "25000",
        "--task-distance-m", "1000", "--task-max-policy-steps", "3000",
        "--post-train-eval-episodes", str(eval_episodes),
        "--post-train-eval-workers", str(min(N_ENVS, eval_episodes)),
        "--post-train-eval-seed-start", "1100000",
        "--post-train-evaluate-reused",
        "--skip-evaluation", "--skip-counterfactual",
        "--env-config-json", '{"neighbors_count": 7, "vehicles_count": 40}',
        "--variants", "ppo_nominal",
        "--seeds", str(seed),
        "--output-dir", str(output_dir),
        "--tensorboard-run-label", run_label,
        "--force-retrain",
    ]
    cmd.extend(resolved_ppo_cli_args(params))
    return cmd


def kill_process_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)


def count_checkpoints(output_dir: Path) -> int:
    matches = list(output_dir.rglob("checkpoints/rollout_*_steps.zip"))
    return len(matches)


@dataclass
class TrialResult:
    completion_rate: float | None
    collisions_per_km: float | None
    output_dir: Path
    ok: bool
    reason: str = ""


def run_watchdog_trial(
    cmd: list[str], *, output_dir: Path, log_path: Path,
    stall_seconds: int, hard_timeout_seconds: int,
) -> bool:
    """Launch cmd, polling checkpoint progress. Return True iff it completed cleanly."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=log_f, stderr=subprocess.STDOUT, env=env)
    start = time.time()
    last_ckpt_count = 0
    last_progress = start
    try:
        while proc.poll() is None:
            time.sleep(30)
            n_ckpt = count_checkpoints(output_dir)
            if n_ckpt > last_ckpt_count:
                last_ckpt_count = n_ckpt
                last_progress = time.time()
            now = time.time()
            if now - last_progress > stall_seconds:
                print(f"  [watchdog] STALLED: no new checkpoint for {stall_seconds}s -- killing pid {proc.pid}")
                kill_process_tree(proc.pid)
                return False
            if now - start > hard_timeout_seconds:
                print(f"  [watchdog] HARD TIMEOUT after {hard_timeout_seconds}s -- killing pid {proc.pid}")
                kill_process_tree(proc.pid)
                return False
    finally:
        if proc.poll() is None:
            kill_process_tree(proc.pid)
    return proc.returncode == 0


def parse_kpi(output_dir: Path, variant: str, seed: int) -> TrialResult:
    # post_train_200ep_kpis.csv is written directly under the --output-dir
    # passed to the training script (which already IS the per-seed
    # directory here) -- NOT under an extra <variant>/seed_<seed> nesting.
    # That nesting only applies to checkpoints/tensorboard, not this file.
    del variant, seed
    kpi_path = output_dir / "post_train_200ep_kpis.csv"
    if not kpi_path.is_file():
        return TrialResult(None, None, output_dir, ok=False, reason=f"missing {kpi_path}")
    df = pd.read_csv(kpi_path)
    cbf = df[df["mode"] == "cbf"]
    completion = cbf[cbf["KPI"] == "Distance-based completion rate"]
    collisions = cbf[cbf["KPI"] == "Ego collisions / km"]
    if completion.empty or collisions.empty:
        return TrialResult(None, None, output_dir, ok=False, reason="KPI rows missing")
    return TrialResult(
        float(completion.iloc[0]["Mean"]), float(collisions.iloc[0]["Mean"]),
        output_dir, ok=True,
    )


def resolve_trial_timesteps(params: dict[str, Any], requested_timesteps: int) -> int:
    """Round the requested fidelity budget to the nearest multiple of this
    trial's own sampled global rollout size (n_steps).

    run_ppo_cbf_progression.py requires --timesteps % --n-steps == 0, and
    n_steps is itself a search-space choice (1000/1500/2000). A fixed
    --search-timesteps/--confirm-timesteps is not divisible by every choice
    (e.g. 250_000 and 1_000_000 both fail for n_steps=1500), which crashed
    ~1/3 of trials instantly regardless of the sampled hyperparameters. Round
    per-trial instead of picking a magic constant tied to today's choice list.
    """
    n_steps = int(params["n_steps"])
    multiples = max(1, round(requested_timesteps / n_steps))
    return multiples * n_steps


def evaluate_trial(
    params: dict[str, Any], *, seeds: list[int], timesteps: int, eval_episodes: int,
    out_root: Path, tag: str, stall_seconds: int, hard_timeout_seconds: int,
) -> list[TrialResult]:
    results = []
    resolved_timesteps = resolve_trial_timesteps(params, timesteps)
    for seed in seeds:
        run_label = f"{tag}_s{seed}"
        output_dir = out_root / tag / f"seed_{seed}"
        log_path = out_root / "_logs" / f"{run_label}.log"
        ok = run_watchdog_trial(
            build_train_cmd(params, seed=seed, timesteps=resolved_timesteps, eval_episodes=eval_episodes,
                             output_dir=output_dir, run_label=run_label),
            output_dir=output_dir, log_path=log_path,
            stall_seconds=stall_seconds, hard_timeout_seconds=hard_timeout_seconds,
        )
        if not ok:
            results.append(TrialResult(None, None, output_dir, ok=False, reason="watchdog kill or nonzero exit"))
            continue
        results.append(parse_kpi(output_dir, "ppo_nominal", seed))
    return results


def aggregate(results: list[TrialResult]) -> tuple[float, float, float, float] | None:
    ok_results = [r for r in results if r.ok and r.completion_rate is not None]
    if not ok_results:
        return None
    comp = [r.completion_rate for r in ok_results]
    coll = [r.collisions_per_km for r in ok_results]
    comp_mean = sum(comp) / len(comp)
    coll_mean = sum(coll) / len(coll)
    comp_sem = (pd.Series(comp).std(ddof=1) / math.sqrt(len(comp))) if len(comp) > 1 else 0.0
    coll_sem = (pd.Series(coll).std(ddof=1) / math.sqrt(len(coll))) if len(coll) > 1 else 0.0
    return comp_mean, float(comp_sem), coll_mean, float(coll_sem)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--state-file", type=Path, default=None)
    p.add_argument("--n-sobol", type=int, default=8)
    p.add_argument("--n-bo", type=int, default=12)
    p.add_argument("--search-timesteps", type=int, default=250_000)
    p.add_argument("--search-eval-episodes", type=int, default=50)
    p.add_argument("--search-seed", type=int, default=307)
    p.add_argument("--n-confirm", type=int, default=3)
    p.add_argument("--confirm-timesteps", type=int, default=1_000_000)
    p.add_argument("--confirm-eval-episodes", type=int, default=200)
    p.add_argument("--confirm-seeds", type=int, nargs="+", default=[307, 327, 347])
    p.add_argument("--stall-minutes", type=float, default=20.0)
    p.add_argument("--search-hard-timeout-minutes", type=float, default=90.0)
    p.add_argument("--confirm-hard-timeout-minutes", type=float, default=240.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    state_file = args.state_file or (args.out_root / "ax_state.json")
    stall_s = int(args.stall_minutes * 60)
    search_timeout_s = int(args.search_hard_timeout_minutes * 60)
    confirm_timeout_s = int(args.confirm_hard_timeout_minutes * 60)

    if state_file.is_file():
        print(f"Resuming Ax search from {state_file}")
        ax_client = AxClient.load_from_json_file(str(state_file))
    else:
        # Explicit Sobol(n_sobol) -> BoTorch(remaining) schedule -- Ax's own
        # default scales Sobol-phase length with dimensionality (18 trials
        # for this 9-parameter space), which would swallow most or all of a
        # modest total budget before BO ever kicks in.
        generation_strategy = GenerationStrategy(steps=[
            GenerationStep(model=Models.SOBOL, num_trials=args.n_sobol, min_trials_observed=args.n_sobol),
            GenerationStep(model=Models.BOTORCH_MODULAR, num_trials=-1),
        ])
        ax_client = AxClient(random_seed=0, generation_strategy=generation_strategy)
        ax_client.create_experiment(
            name="ppo_nominal_hparam_search",
            parameters=build_search_space(),
            objectives={"completion_rate": ObjectiveProperties(minimize=False)},
            outcome_constraints=[
                f"collisions_per_km <= {BASELINE_COLLISIONS_PER_KM + COLLISION_CONSTRAINT_SLACK}"
            ],
        )
        # Anchor the search with the currently-trusted Q1_stable point.
        ax_client.attach_trial(parameters={
            "learning_rate": 1e-4, "n_steps": 1000, "n_minibatches": 10,
            "n_epochs": 10, "gamma": 0.99, "ent_coef": 0.005,
            "clip_range": 0.2, "gae_lambda": 0.95, "vf_coef": 0.5,
        })

    n_total = args.n_sobol + args.n_bo + 1  # +1 for the anchored Q1_stable point
    trial_count = len(ax_client.experiment.trials)
    print(f"Search phase: {trial_count}/{n_total} trials already recorded")

    while trial_count < n_total:
        params, trial_index = ax_client.get_next_trial()
        tag = f"search_trial{trial_index:03d}"
        print(f"\n[trial {trial_index}] params={params}")
        results = evaluate_trial(
            params, seeds=[args.search_seed], timesteps=args.search_timesteps,
            eval_episodes=args.search_eval_episodes, out_root=args.out_root, tag=tag,
            stall_seconds=stall_s, hard_timeout_seconds=search_timeout_s,
        )
        agg = aggregate(results)
        if agg is None:
            print(f"  [trial {trial_index}] FAILED ({results[0].reason if results else 'no result'})")
            ax_client.log_trial_failure(trial_index=trial_index)
        else:
            comp_mean, comp_sem, coll_mean, coll_sem = agg
            print(f"  [trial {trial_index}] completion={comp_mean:.3f} collisions_per_km={coll_mean:.3f}")
            ax_client.complete_trial(trial_index=trial_index, raw_data={
                "completion_rate": (comp_mean, comp_sem),
                "collisions_per_km": (coll_mean, coll_sem),
            })
        ax_client.save_to_json_file(str(state_file))
        trial_count = len(ax_client.experiment.trials)

    print("\n=== Search phase complete; selecting confirmation candidates ===")
    df = ax_client.get_trials_data_frame()
    completed = df[df["trial_status"] == "COMPLETED"].copy()
    feasible = completed[completed["collisions_per_km"] <= BASELINE_COLLISIONS_PER_KM + COLLISION_CONSTRAINT_SLACK]
    ranked = feasible.sort_values("completion_rate", ascending=False)
    if ranked.empty:
        print("No feasible trials satisfied the collision constraint; falling back to all completed trials.")
        ranked = completed.sort_values("completion_rate", ascending=False)
    top_trial_indices = ranked["trial_index"].head(args.n_confirm).tolist()
    print(f"Top {len(top_trial_indices)} candidates (trial indices): {top_trial_indices}")

    confirm_results: dict[int, Any] = {}
    for trial_index in top_trial_indices:
        params = ax_client.get_trial_parameters(trial_index)
        tag = f"confirm_trial{trial_index:03d}"
        print(f"\n[confirm {trial_index}] params={params}")
        results = evaluate_trial(
            params, seeds=args.confirm_seeds, timesteps=args.confirm_timesteps,
            eval_episodes=args.confirm_eval_episodes, out_root=args.out_root, tag=tag,
            stall_seconds=stall_s, hard_timeout_seconds=confirm_timeout_s,
        )
        agg = aggregate(results)
        confirm_results[trial_index] = {
            "params": params,
            "n_ok_seeds": sum(1 for r in results if r.ok),
            "n_seeds": len(results),
            "completion_mean": agg[0] if agg else None,
            "completion_sem": agg[1] if agg else None,
            "collisions_per_km_mean": agg[2] if agg else None,
            "collisions_per_km_sem": agg[3] if agg else None,
        }
        (args.out_root / "confirm_results.json").write_text(
            json.dumps(confirm_results, indent=2, default=str), encoding="utf-8"
        )
        print(f"  [confirm {trial_index}] {confirm_results[trial_index]}")

    print("\n=== FINAL RESULTS ===")
    print(json.dumps(confirm_results, indent=2, default=str))
    best = max(
        (k for k in confirm_results if confirm_results[k]["completion_mean"] is not None),
        key=lambda k: confirm_results[k]["completion_mean"],
        default=None,
    )
    if best is not None:
        print(f"\nBest confirmed candidate: trial {best}")
        print(json.dumps(confirm_results[best], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
