"""Bayesian-optimization sweep over C2 (``ppo_cbf_projected``) at full 1M fidelity.

Differs from ``ax_ppo_sweep.py`` (the A1 sweep) in four ways, each forced by a
measurement rather than a preference.

**1. No low-fidelity search phase.** The A1 sweep searches at 250k and confirms
at 1M. That assumes rank is preserved across budget, and this repository holds a
direct counterexample: in ``hocbf_state_cost_20260915.md`` the state-cost term
measures 32% and 20% *better* than control at 250k and 13% *worse* at 1M -- the
sign flips. A checkpoint probe on the nominal arm shows the same scale of
movement (8.715 -> 2.911 collisions/km from 250k to 1M on one seed). Every trial
here therefore runs the full 1M.

**2. The objective is the raw actor, not the deployed policy.** On C2 the
projection masks almost the entire parameter effect: across the whole
jacobian_alpha range, CBF-ON collisions/km moved 0.211 -> 0.230 while the
raw-actor-mean moved 8.517 -> 3.300. Optimizing the CBF-ON number would spend
the budget on a flat surface. The objective is raw-actor collisions/km, with
CBF-ON collisions/km kept as an outcome constraint so the search cannot buy a
better unshielded policy by degrading deployed safety.

**3. Observation noise is reported honestly.** The A1 sweep passes SEM = 0.0
whenever a trial has a single seed, which tells the GP its observations are
exact; it will then interpolate noise. Measured here: evaluation-only SE of a
run mean 0.331, genuine seed-to-seed SD 0.522, snapshot SD up to 0.379 on
collisions/km. Each trial reports sqrt(eval_sem^2 + SEED_SD^2), where eval_sem
comes from the per-block SD already present in every KPI file.

**4. The confirm phase is generous with seeds.** A six-cell sweep this week
produced a p = 0.031 result that then failed to replicate; with ~100 trials,
spurious winners are expected. Confirmation re-runs the top candidates on 7
spaced seeds, which is near the n = 9 that the variance decomposition says the
effect sizes in play actually require.

Search space is 11-dimensional: the nine PPO mechanics plus ``target_kl`` and
``max_grad_norm`` (both previously unreachable from the CLI), and the two
C2-specific terms ``lambda_mean`` and ``jacobian_alpha`` where the entire
measured C2 effect lives.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties

HOME = Path.home()
DEFAULT_OUT_ROOT = HOME / "AppData" / "Local" / "Temp" / "saferl_runs" / "ax_c2_sweep"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# C2 incumbents at 1M / 7 seeds, from lambda1p0_evaluation_tables.md.
BASELINE_RAW_ACTOR_COLLISIONS = 2.860
BASELINE_CBF_ON_COLLISIONS = 0.255
CBF_ON_CONSTRAINT = 0.30  # baseline + ~1 SD; forbids trading deployed safety away

# Genuine seed-to-seed SD of collisions/km, from the four-seed variance
# decomposition. A single-seed trial cannot observe this, so it is added to the
# reported SEM rather than left at zero.
SEED_SD_COLLISIONS = 0.522

N_ENVS = 20
VARIANT = "ppo_cbf_projected"
SEARCH_SEEDS = [307, 327, 347]
CONFIRM_SEEDS = [307, 327, 347, 367, 387, 407, 427]


def build_search_space() -> list[dict[str, Any]]:
    return [
        # --- PPO mechanics ---
        {"name": "learning_rate", "type": "range", "bounds": [2e-5, 5e-4],
         "log_scale": True, "value_type": "float"},
        {"name": "n_steps", "type": "choice", "values": [500, 1000, 2000, 4000],
         "value_type": "int", "is_ordered": True},
        {"name": "n_minibatches", "type": "choice", "values": [4, 5, 8, 10, 20],
         "value_type": "int", "is_ordered": True},
        {"name": "n_epochs", "type": "range", "bounds": [3, 15], "value_type": "int"},
        {"name": "ent_coef", "type": "range", "bounds": [1e-4, 2e-2],
         "log_scale": True, "value_type": "float"},
        {"name": "clip_range", "type": "range", "bounds": [0.1, 0.3], "value_type": "float"},
        {"name": "vf_coef", "type": "range", "bounds": [0.25, 1.0], "value_type": "float"},
        # --- previously unreachable from the CLI ---
        {"name": "max_grad_norm", "type": "range", "bounds": [0.2, 5.0],
         "log_scale": True, "value_type": "float"},
        # 1.0 is far above any observed approx_kl (~0.003-0.008) and so acts as
        # "uncapped", letting the search decide whether capping helps at all.
        {"name": "target_kl", "type": "range", "bounds": [0.003, 1.0],
         "log_scale": True, "value_type": "float"},
        # --- C2-specific ---
        {"name": "lambda_mean", "type": "range", "bounds": [0.0, 3.0], "value_type": "float"},
        {"name": "jacobian_alpha", "type": "range", "bounds": [0.0, 1.0], "value_type": "float"},
    ]


def resolved_cli_args(params: dict[str, Any]) -> list[str]:
    n_steps = int(params["n_steps"])
    batch_size = max(n_steps // int(params["n_minibatches"]), 1)
    args = [
        "--learning-rate", repr(float(params["learning_rate"])),
        "--n-steps", str(n_steps),
        "--batch-size", str(batch_size),
        "--n-epochs", str(int(params["n_epochs"])),
        "--ent-coef", repr(float(params["ent_coef"])),
        "--clip-range", repr(float(params["clip_range"])),
        "--vf-coef", repr(float(params["vf_coef"])),
        "--max-grad-norm", repr(float(params["max_grad_norm"])),
        "--lambda-mean", repr(float(params["lambda_mean"])),
        "--jacobian-alpha", repr(float(params["jacobian_alpha"])),
    ]
    # Above ~0.1 the cap can never bind, so pass nothing and keep the stock path.
    if float(params["target_kl"]) < 0.1:
        args += ["--target-kl", repr(float(params["target_kl"]))]
    return args


def build_train_cmd(params: dict[str, Any], *, seed: int, timesteps: int,
                    eval_episodes: int, output_dir: Path, run_label: str) -> list[str]:
    """The canonical safety_matrix_1M contract, which is where C2's baseline lives."""
    cmd = [
        "python", "-m", "scripts.training.run_ppo_cbf_progression",
        "--traffic-model", "mtm", "--device", "cpu",
        "--timesteps", str(timesteps),
        "--ppo-config", "Q1_stable",
        "--n-envs", str(N_ENVS),
        "--checkpoint-freq", "25000",
        "--remove-vehicle-dimensions", "--expose-target-y", "--drop-ego-speed-slot",
        "--lateral-target", "field",
        "--lateral-field-wall-weight", "1.0", "--lateral-y-weight", "0.96",
        "--potential-field-weight", "0",
        "--env-config-json", json.dumps({"neighbors_count": 7, "vehicles_count": 40}),
        "--max-neighbor-constraints", "7",
        "--task-distance-m", "1000", "--task-max-policy-steps", "3000",
        "--raw-actor-eval",
        "--raw-actor-eval-episodes", str(eval_episodes),
        "--raw-actor-eval-seed-start", "1100000",
        "--post-train-eval-episodes", str(eval_episodes),
        "--post-train-eval-workers", str(min(N_ENVS, eval_episodes)),
        "--post-train-eval-seed-start", "1100000",
        "--post-train-evaluate-reused",
        "--skip-evaluation", "--skip-counterfactual",
        "--variants", VARIANT,
        "--seeds", str(seed),
        "--output-dir", str(output_dir),
        "--tensorboard-run-label", run_label,
        "--force-retrain",
    ]
    return cmd + resolved_cli_args(params)


def kill_process_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)


def count_checkpoints(output_dir: Path) -> int:
    return len(list(output_dir.rglob("checkpoints/rollout_*_steps.zip")))


@dataclass
class TrialResult:
    raw_collisions: float | None
    raw_collisions_sem: float | None
    cbf_collisions: float | None
    output_dir: Path
    ok: bool
    reason: str = ""


def run_watchdog_trial(cmd: list[str], *, output_dir: Path, log_path: Path,
                       stall_seconds: int, hard_timeout_seconds: int) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=log_f,
                                stderr=subprocess.STDOUT, env=env)
    start = time.time()
    last_count, last_progress = 0, start
    try:
        while proc.poll() is None:
            time.sleep(30)
            n = count_checkpoints(output_dir)
            if n > last_count:
                last_count, last_progress = n, time.time()
            now = time.time()
            if now - last_progress > stall_seconds:
                print(f"  [watchdog] STALLED -- killing {proc.pid}", flush=True)
                kill_process_tree(proc.pid)
                return False
            if now - start > hard_timeout_seconds:
                print(f"  [watchdog] HARD TIMEOUT -- killing {proc.pid}", flush=True)
                kill_process_tree(proc.pid)
                return False
    finally:
        if proc.poll() is None:
            kill_process_tree(proc.pid)
    return proc.returncode == 0


def parse_kpi(output_dir: Path, seed: int) -> TrialResult:
    """Raw-actor collisions is the objective; CBF-ON collisions is the constraint."""
    raw_path = (output_dir / "raw_actor_ablation" / VARIANT / f"seed_{seed}" / "kpi.csv")
    std_path = output_dir / "post_train_200ep_kpis.csv"
    if not raw_path.is_file():
        return TrialResult(None, None, None, output_dir, False, f"missing {raw_path}")
    if not std_path.is_file():
        return TrialResult(None, None, None, output_dir, False, f"missing {std_path}")

    raw = pd.read_csv(raw_path)
    row = raw[raw["KPI"] == "Ego collisions / km"]
    if row.empty:
        return TrialResult(None, None, None, output_dir, False, "raw-actor KPI row missing")
    mean = float(row.iloc[0]["Mean"])
    # The KPI file reports SD across N equal episode blocks, so the evaluation
    # SEM of this run's mean is SD/sqrt(N) -- a real per-trial noise estimate.
    sd = float(row.iloc[0].get("SD", float("nan")))
    n_blocks = float(row.iloc[0].get("N", float("nan")))
    eval_sem = sd / math.sqrt(n_blocks) if n_blocks and n_blocks > 1 and sd == sd else 0.0

    std = pd.read_csv(std_path)
    cbf = std[(std["mode"] == "cbf") & (std["KPI"] == "Ego collisions / km")]
    if cbf.empty:
        return TrialResult(None, None, None, output_dir, False, "CBF-ON KPI row missing")
    return TrialResult(mean, eval_sem, float(cbf.iloc[0]["Mean"]), output_dir, True)


def aggregate(results: list[TrialResult]) -> dict[str, tuple[float, float]] | None:
    ok = [r for r in results if r.ok and r.raw_collisions is not None]
    if not ok:
        return None
    raw = pd.Series([r.raw_collisions for r in ok])
    cbf = pd.Series([r.cbf_collisions for r in ok])
    if len(ok) > 1:
        raw_sem = float(raw.std(ddof=1) / math.sqrt(len(ok)))
        cbf_sem = float(cbf.std(ddof=1) / math.sqrt(len(ok)))
    else:
        # Single seed: the within-run evaluation SEM is observable, the
        # seed-to-seed component is not. Reporting only the former (or zero, as
        # the A1 sweep does) would tell the GP this point is far more certain
        # than it is.
        eval_sem = ok[0].raw_collisions_sem or 0.0
        raw_sem = math.sqrt(eval_sem**2 + SEED_SD_COLLISIONS**2)
        cbf_sem = SEED_SD_COLLISIONS * (BASELINE_CBF_ON_COLLISIONS / BASELINE_RAW_ACTOR_COLLISIONS)
    return {"raw_actor_collisions": (float(raw.mean()), raw_sem),
            "cbf_on_collisions": (float(cbf.mean()), cbf_sem)}


def evaluate_trial(params: dict[str, Any], *, seeds: list[int], timesteps: int,
                   eval_episodes: int, out_root: Path, label: str,
                   stall_minutes: int, hard_timeout_minutes: int) -> list[TrialResult]:
    results = []
    for seed in seeds:
        out = out_root / label / f"seed_{seed}"
        cmd = build_train_cmd(params, seed=seed, timesteps=timesteps,
                              eval_episodes=eval_episodes, output_dir=out,
                              run_label=f"{label}_s{seed}")
        ok = run_watchdog_trial(cmd, output_dir=out, log_path=out / "train.log",
                                stall_seconds=stall_minutes * 60,
                                hard_timeout_seconds=hard_timeout_minutes * 60)
        results.append(parse_kpi(out, seed) if ok
                       else TrialResult(None, None, None, out, False, "watchdog failure"))
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--state-file", type=Path, default=None)
    p.add_argument("--n-sobol", type=int, default=30)
    p.add_argument("--n-bo", type=int, default=70)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--n-confirm", type=int, default=3)
    p.add_argument("--confirm-seeds", type=int, nargs="+", default=CONFIRM_SEEDS)
    p.add_argument("--concurrency", type=int, default=2,
                   help="Runs in parallel. 2 matches this machine: 32 cores, 20 workers each.")
    p.add_argument("--stall-minutes", type=int, default=30)
    p.add_argument("--hard-timeout-minutes", type=int, default=180)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    state_file = args.state_file or (args.out_root / "ax_state.json")

    client = AxClient(random_seed=0)
    client.create_experiment(
        name="c2_full_fidelity",
        parameters=build_search_space(),
        objectives={"raw_actor_collisions": ObjectiveProperties(minimize=True)},
        outcome_constraints=[f"cbf_on_collisions <= {CBF_ON_CONSTRAINT}"],
    )

    n_trials = args.n_sobol + args.n_bo
    print(f"[c2-sweep] {n_trials} trials at {args.timesteps:,} steps, "
          f"objective=raw-actor collisions/km (incumbent {BASELINE_RAW_ACTOR_COLLISIONS}), "
          f"constraint cbf_on <= {CBF_ON_CONSTRAINT}", flush=True)

    # Two concurrent runs at n_envs=20 is this machine's efficient mode (32 cores,
    # 40 workers); one run alone would leave a third of the box idle and put the
    # sweep outside the budget. Ax is asked for both trials before either
    # completes, so the second is generated against the first as pending.
    def run_one(job: tuple[dict[str, Any], int, int]) -> None:
        params, index, seed = job
        label = f"t{index:03d}"
        print(f"\n[c2-sweep] trial {index} seed {seed}: "
              + json.dumps({k: round(v, 5) if isinstance(v, float) else v
                            for k, v in params.items()}), flush=True)
        results = evaluate_trial(params, seeds=[seed], timesteps=args.timesteps,
                                 eval_episodes=args.eval_episodes, out_root=args.out_root,
                                 label=label, stall_minutes=args.stall_minutes,
                                 hard_timeout_minutes=args.hard_timeout_minutes)
        data = aggregate(results)
        with completion_lock:
            if data is None:
                client.log_trial_failure(trial_index=index)
                print(f"[c2-sweep] trial {index} FAILED: {results[0].reason}", flush=True)
            else:
                client.complete_trial(trial_index=index, raw_data=data)
                print(f"[c2-sweep] trial {index}: "
                      f"raw={data['raw_actor_collisions'][0]:.3f}"
                      f" +/-{data['raw_actor_collisions'][1]:.3f}"
                      f"  cbf_on={data['cbf_on_collisions'][0]:.3f}", flush=True)
            client.save_to_json_file(str(state_file))

    completion_lock = threading.Lock()
    issued = 0
    while issued < n_trials:
        batch = []
        for _ in range(min(args.concurrency, n_trials - issued)):
            params, index = client.get_next_trial()
            batch.append((params, index, SEARCH_SEEDS[issued % len(SEARCH_SEEDS)]))
            issued += 1
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            list(pool.map(run_one, batch))

    frame = client.get_trials_data_frame()
    frame.to_csv(args.out_root / "search_trials.csv", index=False)

    ok = frame[frame["trial_status"] == "COMPLETED"].copy()
    ok = ok[ok["cbf_on_collisions"] <= CBF_ON_CONSTRAINT]
    top = ok.nsmallest(args.n_confirm, "raw_actor_collisions")
    print(f"\n[c2-sweep] confirming {len(top)} candidates on "
          f"{len(args.confirm_seeds)} seeds each", flush=True)

    rows = []
    for rank, (_, row) in enumerate(top.iterrows()):
        params = {p["name"]: row[p["name"]] for p in build_search_space()}
        label = f"confirm{rank}"
        results = evaluate_trial(params, seeds=args.confirm_seeds,
                                 timesteps=args.timesteps, eval_episodes=args.eval_episodes,
                                 out_root=args.out_root, label=label,
                                 stall_minutes=args.stall_minutes,
                                 hard_timeout_minutes=args.hard_timeout_minutes)
        data = aggregate(results)
        rows.append({"label": label, "search_trial": int(row["trial_index"]),
                     "search_raw": float(row["raw_actor_collisions"]),
                     "confirm_raw": data["raw_actor_collisions"][0] if data else None,
                     "confirm_raw_sem": data["raw_actor_collisions"][1] if data else None,
                     "confirm_cbf_on": data["cbf_on_collisions"][0] if data else None,
                     "n_ok": sum(1 for r in results if r.ok), **params})
        pd.DataFrame(rows).to_csv(args.out_root / "confirm_results.csv", index=False)
        print(f"[c2-sweep] {label}: search {row['raw_actor_collisions']:.3f} -> "
              f"confirm {rows[-1]['confirm_raw']}", flush=True)

    print(f"[c2-sweep] done. incumbent raw-actor {BASELINE_RAW_ACTOR_COLLISIONS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
