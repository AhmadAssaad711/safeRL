"""Raw-actor-mean ablation for the projected (C) arms of the neighbor-7 study.

The ordinary post-training evaluation reports two columns, external CBF OFF
and ON.  For a ``projected_mean`` variant the OFF column is *not* an
unfiltered policy: the architectural ``mu_raw -> mu_safe`` CBF projection
lives inside the actor and still runs.  Those arms were therefore never
evaluated without a CBF anywhere in the loop.

This launcher closes that gap.  For every projected arm it replays the
trained checkpoint with ``--raw-actor-eval``, which executes ``mu_raw`` and
keeps only the physical action-box clipping, with the external filter OFF.
The difference against the study's OFF column isolates how much of the
measured safety came from the network and how much from the projection
layer sitting on top of it.

Evaluation settings mirror the study's post-training evaluation exactly --
same episode count, same worker count and, critically, the same episode
seed start -- so the ablation is paired episode-for-episode with the
numbers it is being compared against.

``--model-path`` is used rather than in-place checkpoint reuse, so nothing
is written into the completed study tree; results land under a separate
``--out-base``.  Results are generated output and never belong in the
repository, so that defaults to a directory under %LOCALAPPDATA%\\Temp.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEVICE = "cuda"
PYTHON = r"C:\Program Files\Python39\python.exe"
DEFAULT_SEEDS = [307, 308, 309]
DEFAULT_EPISODES = 200
# Must match the study's --post-train-eval-seed-start for a paired comparison.
DEFAULT_SEED_START = 1_100_000

# scripts/ops -> scripts -> safeRL_workspace
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# study arm directory -> variant name
ARMS: dict[str, str] = {
    "C1_cbf_diff_reward_only": "ppo_cbf_diff_reward_only",
    "C2_cbf_projected_reward_off": "ppo_cbf_projected_reward_off",
    "C3_cbf_projected": "ppo_cbf_projected",
}


def base_args() -> list[str]:
    """The study's environment/reward contract, verbatim.

    These still matter in evaluation-only mode because the evaluation
    environment is rebuilt from them; only the training-side flags are inert.
    """

    return [
        "-u", "-m", "scripts.training.run_ppo_cbf_progression",
        "--project-root", str(PROJECT_ROOT),
        "--traffic-model", "mtm", "--device", DEVICE, "--timesteps", "250000",
        "--ppo-config", "Q1_stable", "--n-envs", "20", "--n-steps", "1000",
        "--batch-size", "100", "--n-epochs", "10", "--checkpoint-freq", "25000",
        "--remove-vehicle-dimensions", "--expose-target-y",
        "--lateral-target", "field",
        "--task-distance-m", "1000", "--task-max-policy-steps", "3000",
        "--potential-field-weight", "0",
        "--env-config-json", json.dumps({"neighbors_count": 7}),
        "--max-neighbor-constraints", "7",
        # Evaluation-only: no training, no OFF/ON rerun, no counterfactuals.
        "--skip-training", "--skip-evaluation", "--skip-counterfactual",
        "--skip-post-train-evaluation",
    ]


def log(message: str, log_path: Path) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--study-base",
        type=Path,
        default=Path(os.environ.get("LOCALAPPDATA", "."))
        / "Temp" / "saferl_runs" / "neighbor7_study_250k",
        help="Completed study whose checkpoints are replayed.",
    )
    parser.add_argument("--out-base", type=Path, default=None)
    parser.add_argument("--only", nargs="+", default=None, help="Subset of arm labels.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    out_base: Path = args.out_base or (
        Path(os.environ.get("LOCALAPPDATA", "."))
        / "Temp" / "saferl_runs" / "neighbor7_250k_raw_actor"
    )
    out_base.mkdir(parents=True, exist_ok=True)
    log_path = out_base / "ABLATION.log"

    arms = ARMS if args.only is None else {k: v for k, v in ARMS.items() if k in args.only}
    log(
        f"=== raw-actor ablation: {len(arms)} arm(s) x {len(args.seeds)} seed(s), "
        f"{args.episodes} episodes from seed {args.seed_start}, out {out_base} ===",
        log_path,
    )

    failures: list[str] = []
    for label, variant in arms.items():
        for seed in args.seeds:
            model_path = (
                args.study_base / label / f"seed_{seed}" / variant / f"seed_{seed}"
                / "model_final.zip"
            )
            if not model_path.is_file():
                log(f"{label} seed {seed} SKIPPED: no checkpoint at {model_path}", log_path)
                failures.append(f"{label}/{seed} (missing checkpoint)")
                continue

            run_dir = out_base / label / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            kpi_path = run_dir / "raw_actor_ablation" / variant / f"seed_{seed}" / "kpi.csv"
            if kpi_path.is_file():
                log(f"{label} seed {seed} already complete, skipping", log_path)
                continue

            argv = (
                [PYTHON] + base_args()
                + ["--variants", variant, "--seeds", str(seed)]
                + ["--model-path", str(model_path)]
                + ["--raw-actor-eval",
                   "--raw-actor-eval-episodes", str(args.episodes),
                   "--raw-actor-eval-seed-start", str(args.seed_start),
                   "--post-train-eval-workers", str(args.workers)]
                + ["--output-dir", str(run_dir)]
            )

            log(f"{label} seed {seed} starting", log_path)
            started = time.time()
            stdout_path = run_dir / "ablation.out"
            stderr_path = run_dir / "ablation.err"
            with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
                code = subprocess.call(
                    argv, cwd=str(PROJECT_ROOT), stdout=out_f, stderr=err_f
                )
            elapsed = int(time.time() - started)

            if code == 0 and kpi_path.is_file():
                log(f"--- {label} seed {seed} COMPLETE in {elapsed}s ---", log_path)
            else:
                tail = ""
                try:
                    tail = stderr_path.read_text(encoding="utf-8", errors="ignore")[-800:]
                except OSError:
                    pass
                log(f"!!! {label} seed {seed} FAILED exit={code} after {elapsed}s", log_path)
                log(f"    {tail}", log_path)
                failures.append(f"{label}/{seed} (exit {code})")

    if failures:
        log(f"=== finished WITH FAILURES: {failures} ===", log_path)
        return 1
    log("=== ablation finished ===", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
