"""Run the five-arm CBF safety matrix at 1M timesteps across seven seeds.

The arms separate three things that the earlier ladders confounded:

* whether the CBF is in the training loop at all (A1 versus everything else);
* whether feedback reaches the policy through the reward, the actor loss, or
  both (B2/B3);
* whether the projection is differentiable, so gradients flow through it
  (C1/C3), or not (B2/B3).

Two choices in here are deliberate and worth reading before changing them.

**Seeds are spaced by twenty.** ``VecEnv.seed(s)`` assigns worker seeds
``s .. s+19`` and the scenario stream is a pure function of the worker seed --
``np_random`` is consumed only in the reset/spawn path, never in ``step``. So
consecutive training seeds share 18-19 of 20 scenario streams and are not
independent replicates. Spacing by at least ``n_envs`` makes them disjoint.

**The projected arms are also evaluated without their projection.** C1 and C3
carry ``projected_mean``, so their deployed action passes through the CBF
projection inside the policy. ``--raw-actor-eval`` adds a second evaluation of
the raw network mean with that projection bypassed, which is what shows whether
the *policy* learned anything or whether the projection is carrying it.

Resumable: a run whose ``post_train_200ep_kpis.csv`` already names its variant
is skipped, so the study can be interrupted and restarted. One arm/seed at a
time by default; ``--concurrency 2`` matches this machine's efficient mode
(two 20-worker runs on 32 cores).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
DEVICE = "cpu"
DEFAULT_TIMESTEPS = 1_000_000
# Spaced by 20 = n_envs, so no two runs share a scenario stream.
DEFAULT_SEEDS = [307, 327, 347, 367, 387, 407, 427]
MAX_ATTEMPTS_PER_RUN = 100
STALL_MINUTES = 25
POLL_SECONDS = 30


def base_args(timesteps: int, lateral_fix: bool,
              ego_desired_speed: float | None) -> list[str]:
    args = [
        "-m", "scripts.training.run_ppo_cbf_progression",
        "--traffic-model", "mtm", "--device", DEVICE, "--timesteps", str(timesteps),
        "--ppo-config", "Q1_stable", "--n-envs", "20", "--n-steps", "1000",
        "--batch-size", "100", "--n-epochs", "10", "--checkpoint-freq", "25000",
        "--remove-vehicle-dimensions", "--expose-target-y",
        # 41D at neighbors_count 7, matching every run in the lateral-fix and
        # state-cost lines. The ego row's desired-speed feature is a constant
        # under a fixed speed target, and checkpoints are not portable across
        # this choice, so it has to agree with the runs being compared against.
        "--drop-ego-speed-slot",
        "--lateral-target", "field",
        "--task-distance-m", "1000", "--task-max-policy-steps", "3000",
        "--post-train-eval-episodes", "200", "--post-train-eval-workers", "20",
        "--post-train-eval-seed-start", "1100000", "--post-train-evaluate-reused",
        "--skip-evaluation", "--skip-counterfactual",
        "--potential-field-weight", "0",
        "--env-config-json", json.dumps({"neighbors_count": 7}),
        "--max-neighbor-constraints", "7",
    ]
    if lateral_fix:
        # Wall repulsion on the lateral target plus the rebalanced reciprocal.
        # Measured to remove wall-hugging at seed 308 (77.3% -> 0.2%) but NOT at
        # seed 327 (90.2%), so it is opt-in and unproven across seeds.
        args += [
            "--lateral-field-wall-weight", "1.0",
            "--lateral-y-weight", "0.96",
        ]
    if ego_desired_speed is not None:
        # v_desired is the reward's cx normalizer as well as the speed target,
        # and wx is NOT retuned with it, so a run at another speed is not a
        # pure speed change. Recorded in every run's manifest either way.
        args += ["--ego-desired-speed", repr(float(ego_desired_speed))]
    return args


# label -> variants, KPI rows required to count as done, extra CLI.
# ``state_cost`` marks the arm whose extra flags are filled in from the
# --state-cost-* options, because its term is the one still under development.
# ``actor_lambda_flag`` names the arm's actor-loss safety coefficient, filled in
# from --actor-lambda. B2 and C2 are the only arms carrying one, and they carry
# different ones: B2's loss is the detached hard-CBF mean target, C2's is the
# differentiable projected mean. The arms without an actor loss ignore it.
ARMS: dict[str, dict[str, Any]] = {
    "A1_nominal": {
        "variants": ["ppo_nominal"], "need": ["ppo_nominal"], "extra": [],
        "what": "PPO alone: no CBF in the loop, no state cost",
    },
    "B1_state_cost_raw": {
        "variants": ["ppo_hocbf_state_cost_raw"], "need": ["ppo_hocbf_state_cost_raw"],
        "extra": [], "state_cost": True,
        "what": "box execution + the C(s) state cost only, CBF NOT in training",
    },
    "B2_cbf_nd_reward_actor": {
        "variants": ["ppo_cbf_nd_reward_actor"], "need": ["ppo_cbf_nd_reward_actor"],
        "extra": [], "actor_lambda_flag": "--lambda-detached-actor",
        "what": "CBF in training + reward penalty + detached (non-differentiable) actor loss",
    },
    "C1_cbf_diff_reward_only": {
        "variants": ["ppo_cbf_diff_reward_only"], "need": ["ppo_cbf_diff_reward_only"],
        "extra": ["--raw-actor-eval"],
        "what": "differentiable projection + reward penalty, no actor loss",
    },
    "C2_cbf_projected": {
        "variants": ["ppo_cbf_projected"], "need": ["ppo_cbf_projected"],
        "extra": ["--raw-actor-eval"], "actor_lambda_flag": "--lambda-mean",
        "what": "differentiable projection + reward penalty + actor loss",
    },
    # Constrained-MDP arms. Under the 2026-10-12 lettering these are the C
    # family and the two projection arms above are D1/D2; the labels here are
    # kept as-is because they are output *directory* names and renaming them
    # would orphan every existing run directory. See
    # docs/cmdp_implementation_notes.md.
    #
    # They carry no actor-loss coefficient, so --actor-lambda does not apply.
    # They need --cost-limit, which has no default and must come from measured
    # cost distributions; pass it through --extra-arg, e.g.
    #   --extra-arg=--cost-limit --extra-arg 0.35
    "C1_cmdp_state": {
        "variants": ["ppo_cmdp_state"], "need": ["ppo_cmdp_state"],
        "extra": ["--state-cost-barrier-form", "distance"],
        "what": "CMDP: HOCBF state cost as a constraint, filter OFF in training",
    },
    "C2_cmdp_action": {
        "variants": ["ppo_cmdp_action"], "need": ["ppo_cmdp_action"],
        "extra": [],
        "what": "CMDP: HOCBF action-distance cost as a constraint, filter OFF",
    },
}

# Arms that require --cost-limit to be supplied through --extra-arg.
CMDP_ARMS = ("C1_cmdp_state", "C2_cmdp_action")


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


def state_cost_args(barrier_form: str, lambda_s: float) -> list[str]:
    """Flags for the arm carrying the C(s) state cost.

    Margins, scales and the psi1 gain are deliberately left to fall back to the
    chosen form's reference calibration; they are in different units per form,
    so mixing an explicit margin with the other form's scale is a silent error.
    """

    return [
        "--state-cost-barrier-form", str(barrier_form),
        "--state-cost-lambda", repr(float(lambda_s)),
    ]


def actor_lambda_args(spec: dict[str, Any], actor_lambda: float) -> list[str]:
    """The arm's actor-loss safety coefficient, if it has one."""

    flag = spec.get("actor_lambda_flag")
    return [] if flag is None else [str(flag), repr(float(actor_lambda))]


def run_one(label: str, seed: int, out_base: Path, log_path: Path,
            timesteps: int, lateral_fix: bool,
            barrier_form: str, lambda_s: float,
            ego_desired_speed: float | None, actor_lambda: float,
            extra_args: Sequence[str] = ()) -> bool:
    spec = ARMS[label]
    out_dir = out_base / label / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if run_complete(out_dir, spec["need"]):
        log(f"{label} seed {seed} already complete, skipping", log_path)
        return True

    for attempt in range(1, MAX_ATTEMPTS_PER_RUN + 1):
        argv = (
            [PYTHON] + base_args(timesteps, lateral_fix, ego_desired_speed)
            + ["--variants"] + spec["variants"]
            + ["--seeds", str(seed)]
            + list(spec["extra"])
            + actor_lambda_args(spec, actor_lambda)
            + (state_cost_args(barrier_form, lambda_s) if spec.get("state_cost") else [])
            + list(extra_args)
            + ["--output-dir", str(out_dir),
               "--tensorboard-run-label", f"{label}_s{seed}"]
        )
        if not has_checkpoint(out_dir):
            argv.append("--force-retrain")

        stdout_path = out_dir / f"attempt{attempt}.out"
        stderr_path = out_dir / f"attempt{attempt}.err"
        log(f"{label} seed {seed} attempt {attempt} starting", log_path)
        started = time.time()
        # ``scripts.common.ppo_cbf_env`` imports ``saferl`` at module scope,
        # before the training entry point's own src fallback runs, so a run from
        # a shell without PYTHONPATH set dies on import. Same fix as
        # scripts/tuning/ax_ppo_sweep.py and cbf_congested_tuning_sweep.py.
        env = dict(os.environ)
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
            proc = subprocess.Popen(argv, cwd=str(PROJECT_ROOT), stdout=out_f,
                                    stderr=err_f, env=env)
            while True:
                time.sleep(POLL_SECONDS)
                if run_complete(out_dir, spec["need"]):
                    return True

                text = ""
                for path in (stdout_path, stderr_path):
                    try:
                        text += path.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        pass

                if "access violation" in text or "error return without exception" in text:
                    log(f"{label} seed {seed} attempt {attempt} FAULT: native worker fault", log_path)
                    kill_tree(proc)
                    break

                elapsed = time.time() - started
                exit_code = proc.poll()
                if exit_code is not None and exit_code != 0 and elapsed < 120:
                    log(f"{label} seed {seed} attempt {attempt} CONFIG ERROR, exit "
                        f"{exit_code} after {int(elapsed)}s. Not retrying.", log_path)
                    tail = "\n".join(l for l in text.splitlines() if l.strip())[-600:]
                    log(f"    {tail}", log_path)
                    return False

                idle = (time.time() - newest_write(out_dir)) / 60.0
                if idle > STALL_MINUTES:
                    log(f"{label} seed {seed} attempt {attempt} STALLED: no writes for "
                        f"{int(idle)} min", log_path)
                    kill_tree(proc)
                    break

                if exit_code is not None:
                    log(f"{label} seed {seed} attempt {attempt} exited with {exit_code}", log_path)
                    break

        if run_complete(out_dir, spec["need"]):
            return True

    log(f"{label} seed {seed} GAVE UP after {MAX_ATTEMPTS_PER_RUN} attempts", log_path)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", default=None, help="Subset of arm labels.")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Runs in parallel. 2 matches this machine (32 cores, 20 workers each).")
    parser.add_argument("--no-lateral-fix", dest="lateral_fix", action="store_false",
                        help="Drop the wall repulsion + rebalanced reciprocal. On by "
                             "default: it removed wall-hugging at seed 308 (77.3%% -> "
                             "0.2%%) though not at seed 327, and without it most "
                             "policies sit against a boundary for 72-85%% of every "
                             "episode, which is a large between-seed variance source.")
    parser.set_defaults(lateral_fix=True)
    parser.add_argument("--state-cost-barrier-form", choices=("ellipse", "distance"),
                        default="distance",
                        help="Barrier form for the B1 state-cost arm. 'distance' (TTC) "
                             "fires on ~100%% of steps with a non-zero gradient from "
                             "~25 m; 'ellipse' is silent until 7.2 m with exactly zero "
                             "gradient over ~83%% of the approach.")
    parser.add_argument("--state-cost-lambda", type=float, default=0.1619,
                        help="lambda_s for the B1 arm. 0.1619 is the calibrated value "
                             "for a 15%% share of base reward under the distance form; "
                             "0.0563 is what the earlier ellipse runs used, which "
                             "measured at only ~1.8%%. The calibration was done at "
                             "v_desired 16, so the share moves with --ego-desired-speed.")
    parser.add_argument("--actor-lambda", type=float, default=0.10,
                        help="Actor-loss safety coefficient for the two arms that "
                             "carry one: --lambda-detached-actor on B2, --lambda-mean "
                             "on C2. 0.10 is what the 1M safety matrix ran; the "
                             "published evaluation tables report those two arms at 1.0 "
                             "from the separate lambda studies.")
    parser.add_argument("--ego-desired-speed", type=float, default=None,
                        help="v_desired in m/s. Unset keeps the contract's 16.0. It is "
                             "also the reward's cx normalizer and wx is not retuned "
                             "with it, so changing it is not a pure speed change.")
    parser.add_argument("--extra-arg", dest="extra_args", action="append", default=[],
                        metavar="ARG",
                        help="Verbatim argument appended to every run's command line, "
                             "repeatable, for knobs this launcher does not name. Flag "
                             "and value are separate occurrences, and anything starting "
                             "with '-' needs the '=' form or argparse reads it as an "
                             "option here: --extra-arg=--collision-penalty --extra-arg "
                             "-25. Give each configuration its own --out-base: a run is "
                             "resumed or skipped on its output path alone, which does "
                             "not encode these.")
    parser.add_argument(
        "--out-base", type=Path,
        default=Path(os.environ.get("LOCALAPPDATA", "."))
        / "Temp" / "saferl_runs" / "safety_matrix_1M",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    args = parser.parse_args()

    # The CMDP arms are opt-in: they require a --cost-limit that has no
    # defensible default, so including them in the bare invocation would turn
    # a previously working command into an error.
    labels = (
        [l for l in ARMS if l not in CMDP_ARMS]
        if args.only is None
        else [l for l in args.only if l in ARMS]
    )
    unknown = [] if args.only is None else [l for l in args.only if l not in ARMS]
    if unknown:
        parser.error(f"unknown arm(s): {unknown}. Known: {list(ARMS)}")

    # Fail before any training rather than letting every run die seconds in.
    requested_cmdp = [l for l in labels if l in CMDP_ARMS]
    if requested_cmdp and not any(
        str(a).startswith("--cost-limit") for a in args.extra_args
    ):
        parser.error(
            f"{requested_cmdp} need a constraint budget: pass it with "
            "--extra-arg=--cost-limit --extra-arg <d>. It has no default and "
            "must come from measured cost distributions; see "
            "docs/cmdp_implementation_notes.md."
        )

    out_base: Path = args.out_base
    out_base.mkdir(parents=True, exist_ok=True)
    log_path = out_base / "STUDY.log"
    lock = out_base / "STUDY.lock"

    jobs = [(label, seed) for label in labels for seed in args.seeds]
    print()
    print(f"arms        : {len(labels)}  {labels}")
    print(f"seeds       : {args.seeds}  (spaced by {args.seeds[1] - args.seeds[0] if len(args.seeds) > 1 else '-'})")
    print(f"timesteps   : {args.timesteps:,}")
    print(f"runs        : {len(jobs)}")
    print(f"concurrency : {args.concurrency}")
    print(f"lateral fix : {args.lateral_fix}")
    print(f"state cost  : form={args.state_cost_barrier_form} lambda={args.state_cost_lambda}")
    print(f"actor lambda: {args.actor_lambda}")
    print(f"v_desired   : {16.0 if args.ego_desired_speed is None else args.ego_desired_speed}"
          f"{'  (contract default)' if args.ego_desired_speed is None else ''}")
    print(f"extra args  : {' '.join(args.extra_args) or '-'}")
    print(f"out         : {out_base}")
    for label in labels:
        extra = list(ARMS[label]["extra"]) + actor_lambda_args(ARMS[label], args.actor_lambda)
        if ARMS[label].get("state_cost"):
            extra += state_cost_args(args.state_cost_barrier_form, args.state_cost_lambda)
        print(f"   {label:26s} {ARMS[label]['what']}")
        print(f"   {'':26s}    variant={ARMS[label]['variants'][0]}  extra={' '.join(extra) or '-'}")
    if args.dry_run:
        return 0

    if lock.exists():
        try:
            pid = int(lock.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
        except (ValueError, OSError):
            pass
        else:
            print(f"another study is running (pid {pid}); remove {lock} if that is wrong")
            return 2
    lock.write_text(str(os.getpid()), encoding="utf-8")

    failures: list[str] = []
    try:
        log(f"STUDY START {len(jobs)} runs, {args.timesteps:,} steps, "
            f"concurrency {args.concurrency}, lateral_fix={args.lateral_fix}, "
            f"v_desired={16.0 if args.ego_desired_speed is None else args.ego_desired_speed}, "
            f"actor_lambda={args.actor_lambda}, "
            f"extra=[{' '.join(args.extra_args)}]", log_path)
        started = time.time()

        def work(job: tuple[str, int]) -> None:
            label, seed = job
            if not run_one(label, seed, out_base, log_path, args.timesteps,
                           args.lateral_fix, args.state_cost_barrier_form,
                           args.state_cost_lambda, args.ego_desired_speed,
                           args.actor_lambda, args.extra_args):
                failures.append(f"{label}/seed_{seed}")

        if args.concurrency <= 1:
            for job in jobs:
                work(job)
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                list(pool.map(work, jobs))

        hours = (time.time() - started) / 3600.0
        log(f"STUDY COMPLETE in {hours:.1f} h, {len(failures)} failure(s) {failures}", log_path)
    finally:
        lock.unlink(missing_ok=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
