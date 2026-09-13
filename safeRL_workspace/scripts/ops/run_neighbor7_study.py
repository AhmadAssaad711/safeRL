"""Launcher for the 7-variant x 5-seed x 1M-step nearest-7-neighbor study.

This is a sequencing/supervision launcher, not a new experiment protocol --
the protocol itself lives entirely in the committed
``scripts.training.run_ppo_cbf_progression`` entrypoint. Its role mirrors
``scripts/ops/run_lateral_target_ladder.ps1`` (same hardening: checkpoint
resume, stall detection, native-worker-fault retry, one lock per output
base), just in Python and for a different variant/seed/observation set.

Study definition (as confirmed with the user). Every arm maps onto an
existing VARIANT_SPECS entry in run_ppo_cbf_progression.py -- none of the
7 arms required new training-loop code:

  A1  ppo_nominal                     field lateral target, no HOCBF term
  B1  ppo_hocbf_reward_raw            HOCBF reward term, CBF NOT in training
  B2  ppo_cbf_reward                  CBF IN training + correction reward
  B3  ppo_cbf_nd_actor_only           CBF IN training + detached actor loss, no reward term
  C1  ppo_cbf_diff_reward_only        differentiable CBF projection + reward only
  C2  ppo_cbf_projected_reward_off    differentiable correction in actor only
  C3  ppo_cbf_projected               differentiable correction in actor + reward

Shared, ASSUMED contract (matches the completed 250k lateral-target ladder;
confirm before running if anything here should differ):
  mtm traffic, canonical 40-vehicle / 100-20-20 timing, Q1_stable PPO config,
  20 envs x 1000 steps, field lateral target, potential-field-weight 0,
  strict 1000 m / 3000-step completion, 200+200 post-train evaluation from
  seed 1,100,000.

New for this study (per explicit user request):
  --env-config-json '{"neighbors_count": 7}'   (observation: nearest 7)
  --max-neighbor-constraints 7                  (CBF: nearest 7)
  1,000,000 timesteps per run (vs. 250k in the lateral-target ladder)
  5 seeds per variant: 307, 308, 309, 310, 311 (ASSUMED -- matches this
  repo's existing sequential-seed convention seen under artifacts/; change
  SEEDS below if a different 5 seeds were intended)

B1 (ppo_hocbf_reward_raw) needs a psi-scale. Per AGENTS.md, the psi-scale
must be calibrated once from a fixed ppo_nominal policy and frozen across
all HOCBF-reward seeds -- so B1's first seed co-trains ppo_nominal (for
calibration only) alongside ppo_hocbf_reward_raw, and every later B1 seed
reuses that one calibrated scale via --hocbf-psi-scale. This mirrors what
run_lateral_target_ladder.ps1 already does for its runs 3/4.

Results are generated output and never belong in the repository, so
OUT_BASE defaults to a directory under %LOCALAPPDATA%\\Temp.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TIMESTEPS = 1_000_000
SEEDS = [307, 308, 309, 310, 311]
MAX_ATTEMPTS_PER_RUN = 100
STALL_MINUTES = 25
DEVICE = "cuda"
PYTHON = r"C:\Program Files\Python39\python.exe"

# scripts/ops -> scripts -> safeRL_workspace
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_ARGS = [
    "-u", "-m", "scripts.training.run_ppo_cbf_progression",
    "--project-root", str(PROJECT_ROOT),
    "--traffic-model", "mtm", "--device", DEVICE, "--timesteps", str(TIMESTEPS),
    "--ppo-config", "Q1_stable", "--n-envs", "20", "--n-steps", "1000",
    "--batch-size", "100", "--n-epochs", "10", "--checkpoint-freq", "25000",
    "--remove-vehicle-dimensions", "--expose-target-y",
    "--lateral-target", "field",
    "--task-distance-m", "1000", "--task-max-policy-steps", "3000",
    "--post-train-eval-episodes", "200", "--post-train-eval-workers", "20",
    "--post-train-eval-seed-start", "1100000", "--post-train-evaluate-reused",
    "--skip-evaluation", "--skip-counterfactual",
    "--potential-field-weight", "0",
    "--env-config-json", json.dumps({"neighbors_count": 7}),
    "--max-neighbor-constraints", "7",
]

# label -> (variants passed to the training CLI, KPI rows required to count as done)
ARMS: dict[str, dict[str, Any]] = {
    "A1_nominal": {"variants": ["ppo_nominal"], "need": ["ppo_nominal"]},
    "B1_hocbf_reward_raw": {"variants": ["ppo_hocbf_reward_raw"], "need": ["ppo_hocbf_reward_raw"]},
    "B2_cbf_reward": {"variants": ["ppo_cbf_reward"], "need": ["ppo_cbf_reward"]},
    "B3_cbf_nd_actor_only": {"variants": ["ppo_cbf_nd_actor_only"], "need": ["ppo_cbf_nd_actor_only"]},
    "C1_cbf_diff_reward_only": {"variants": ["ppo_cbf_diff_reward_only"], "need": ["ppo_cbf_diff_reward_only"]},
    "C2_cbf_projected_reward_off": {"variants": ["ppo_cbf_projected_reward_off"], "need": ["ppo_cbf_projected_reward_off"]},
    "C3_cbf_projected": {"variants": ["ppo_cbf_projected"], "need": ["ppo_cbf_projected"]},
}


def log(message: str, log_path: Path) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_complete(out_dir: Path, need: list[str]) -> bool:
    kpi = out_dir / "post_train_200ep_kpis.csv"
    if not kpi.is_file():
        return False
    text = kpi.read_text(encoding="utf-8", errors="ignore")
    return all(variant in text for variant in need)


def has_checkpoint(out_dir: Path) -> bool:
    return any(out_dir.rglob("*.zip"))


def newest_write(out_dir: Path) -> float:
    times = [p.stat().st_mtime for p in out_dir.rglob("*") if p.is_file()]
    return max(times) if times else time.time()


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        import psutil  # already a project dependency (stable-baselines3[extra])

        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except Exception:
        proc.kill()
    time.sleep(5)


def run_one(
    label: str,
    out_dir: Path,
    variants: list[str],
    need: list[str],
    seed: int,
    extra: list[str],
    log_path: Path,
) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    if run_complete(out_dir, need):
        log(f"{label} seed {seed} already complete, skipping", log_path)
        return True

    for attempt in range(1, MAX_ATTEMPTS_PER_RUN + 1):
        argv = (
            [PYTHON] + BASE_ARGS
            + ["--variants"] + variants
            + ["--seeds", str(seed)]
            + extra
            + ["--output-dir", str(out_dir), "--tensorboard-run-label", f"{label}_s{seed}"]
        )
        if not has_checkpoint(out_dir):
            argv.append("--force-retrain")

        stdout_path = out_dir / f"attempt{attempt}.out"
        stderr_path = out_dir / f"attempt{attempt}.err"
        log(f"{label} seed {seed} attempt {attempt} starting", log_path)
        started = time.time()
        with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
            proc = subprocess.Popen(
                argv, cwd=str(PROJECT_ROOT), stdout=out_f, stderr=err_f
            )
            while True:
                time.sleep(30)
                if run_complete(out_dir, need):
                    proc.wait(timeout=5) if proc.poll() is not None else None
                    return True

                text = ""
                for path in (stdout_path, stderr_path):
                    try:
                        text += path.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        pass

                if "access violation" in text or "error return without exception" in text:
                    log(f"{label} seed {seed} attempt {attempt} FAULT: native worker fault, killing pool", log_path)
                    kill_tree(proc)
                    break

                elapsed = time.time() - started
                exit_code = proc.poll()
                if exit_code is not None and exit_code != 0 and elapsed < 120:
                    log(
                        f"{label} seed {seed} attempt {attempt} CONFIG ERROR, "
                        f"exit {exit_code} after {int(elapsed)}s. Not retrying.",
                        log_path,
                    )
                    tail = "\n".join(line for line in text.splitlines() if line.strip())[-600:]
                    log(f"    {tail}", log_path)
                    return False

                idle_minutes = (time.time() - newest_write(out_dir)) / 60.0
                if idle_minutes > STALL_MINUTES:
                    log(f"{label} seed {seed} attempt {attempt} STALLED: no writes for {int(idle_minutes)} min, killing pool", log_path)
                    kill_tree(proc)
                    break

                if exit_code is not None:
                    log(f"{label} seed {seed} attempt {attempt} exited with {exit_code}", log_path)
                    break

        if run_complete(out_dir, need):
            return True

    log(f"{label} seed {seed} GAVE UP after {MAX_ATTEMPTS_PER_RUN} attempts", log_path)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", default=None, help="Subset of arm labels to run.")
    parser.add_argument(
        "--out-base",
        type=Path,
        default=Path(os.environ.get("LOCALAPPDATA", ".")) / "Temp" / "saferl_runs" / "neighbor7_study_1m",
    )
    args = parser.parse_args()

    out_base: Path = args.out_base
    out_base.mkdir(parents=True, exist_ok=True)
    log_path = out_base / "STUDY.log"
    lock_path = out_base / "STUDY.lock"

    if lock_path.exists():
        try:
            old_pid = int(lock_path.read_text().strip())
            os.kill(old_pid, 0)  # raises if not alive (best-effort on Windows)
            log(f"ABORT: another study run appears alive at PID {old_pid}", log_path)
            return 1
        except (ValueError, OSError, ProcessLookupError):
            log(f"reclaiming stale lock", log_path)
    lock_path.write_text(str(os.getpid()))

    arms = ARMS if args.only is None else {k: v for k, v in ARMS.items() if k in args.only}
    log(
        f"=== study starting: {len(arms)} arm(s) x {len(SEEDS)} seed(s), "
        f"{TIMESTEPS} steps each, PID {os.getpid()}, out {out_base} ===",
        log_path,
    )

    psi_scale: float | None = None

    for label, spec in arms.items():
        variants = list(spec["variants"])
        need = list(spec["need"])
        arm_dir = out_base / label

        for index, seed in enumerate(SEEDS):
            seed_dir = arm_dir / f"seed_{seed}"
            extra: list[str] = []
            run_variants = variants
            run_need = need

            if label == "B1_hocbf_reward_raw":
                if index == 0:
                    # First seed co-trains ppo_nominal to calibrate the psi
                    # scale once; every later seed reuses that one value.
                    run_variants = ["ppo_nominal", "ppo_hocbf_reward_raw"]
                    run_need = ["ppo_nominal", "ppo_hocbf_reward_raw"]
                    extra += ["--hocbf-calibration-steps", "200"]
                else:
                    if psi_scale is None:
                        calib_path = arm_dir / f"seed_{SEEDS[0]}" / "hocbf_scale_calibration.json"
                        if calib_path.is_file():
                            psi_scale = json.loads(calib_path.read_text())["selected_psi_scale"]
                    if psi_scale is None:
                        log(f"{label} seed {seed} SKIPPED: no psi scale from seed {SEEDS[0]}", log_path)
                        continue
                    extra += ["--hocbf-psi-scale", repr(float(psi_scale))]
                    log(f"{label} seed {seed} using shared psi scale {psi_scale}", log_path)

            done = run_one(label, seed_dir, run_variants, run_need, seed, extra, log_path)
            if done:
                log(f"--- {label} seed {seed} COMPLETE ---", log_path)
                if label == "B1_hocbf_reward_raw" and index == 0:
                    calib_path = seed_dir / "hocbf_scale_calibration.json"
                    if calib_path.is_file():
                        psi_scale = json.loads(calib_path.read_text())["selected_psi_scale"]
                        log(f"B1 calibrated psi scale = {psi_scale}", log_path)
                    else:
                        log("WARNING: B1 seed 307 left no hocbf_scale_calibration.json; later B1 seeds cannot run", log_path)
            else:
                log(f"--- {label} seed {seed} GAVE UP, continuing ---", log_path)

    log("=== study finished ===", log_path)
    lock_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
