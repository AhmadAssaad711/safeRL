"""CBF parameter tuning on the congested (vehicles_count=60) environment,
parameterised as a second-order damped system.

The HOCBF row is ``A @ a <= b`` with ``b = hddot_exo + k1*h_dot + k0*h``, so a
binding constraint puts the executed action exactly on

    psi_2 = b - A @ u = 0   <=>   hddot + k1*h_dot + k0*h = 0

which is the homogeneous second-order ODE. Writing ``k0 = omega_n^2`` and
``k1 = 2*zeta*omega_n`` makes the pole pair explicit:

    s = -zeta*omega_n +/- omega_n*sqrt(zeta^2 - 1)

The current operating point k0=4.0, k1=8.5 is omega_n=2.0, zeta=2.125 -- poles
at -0.5 and -8.0, so a dominant time constant of 2.0 s, or 40 policy steps at
20 Hz.

What is and is not being tested
-------------------------------
That psi_2 = 0 holds at binding steps is true *by construction*; plotting only
that would be circular. The testable content is that the realised h(t) is a
switched, externally forced, discretised version of the ODE -- the binding
neighbour changes between steps, hddot_exo depends on other vehicles, the
actuator saturates at +/-3 m/s^2, and the filter runs at 20 Hz rather than
continuously. So this sweep records, per grid point:

  P1  whether h(t) in binding windows decays at zeta*omega_n and rings at
      omega_d = omega_n*sqrt(1-zeta^2), i.e. whether the measured poles match
      the placed ones;
  P2  whether the KPIs move the way second-order theory says (overshoot, and
      therefore collisions, rising below zeta=1; effort minimised near
      zeta=1; intervention growing with omega_n);
  P3  what fraction of steps the model can govern at all -- feasible AND
      binding AND unsaturated. At d60 the QP already fails ~27% of the time,
      so this is bounded well below 1 before the sweep even starts;
  P4  whether the match degrades as omega_n*dt grows (dt=0.05 s, so the
      continuous-time description is only expected to hold for omega_n well
      under ~4 rad/s; the larger grid values are deliberate probes of the
      breakdown, not candidate settings).

Two stages per grid point
-------------------------
KPI stage runs the ordinary post-training evaluation as a subprocess with
``--k0/--k1`` overridden, so the KPI table is produced by the same accounting
as every other table in this project (distinct collision events, distance
completion, pooled blocks) rather than a reimplementation here.

Trace stage then runs a smaller number of episodes in-process to record
per-step barrier state, which the evaluation subprocess does not emit. Both
stages share the same episode seed start, so the traced episodes are a subset
of the evaluated distribution.

Results are appended after every grid point, so a partial sweep is usable and
an interrupted one can be resumed with --skip-existing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Reward keys that run_ppo_cbf_progression's --reward-config-file rejects as
# unknown; they are not scalar overrides of the notebook reward.
REWARD_KEYS_NOT_OVERRIDABLE = ("lateral_target", "lateral_field_wall_weight")

TRACE_FIELDS = (
    "cbf_min_h",
    "cbf_min_h_dot",
    "cbf_min_row_x",
    "cbf_min_row_y",
    "cbf_min_bound",
    "cbf_min_h_index",
    "cbf_a_safe_x",
    "cbf_a_safe_y",
    "cbf_hocbf_min_margin",
    "actuator_clip_norm",
    "cbf_constraint_count",
    "cbf_num_neighbor_constraints",
)
TRACE_FLAGS = ("cbf_qp_success", "cbf_fallback_used", "cbf_intervened")


def gains_from_damping(omega_n: float, zeta: float) -> tuple[float, float]:
    return float(omega_n) ** 2, 2.0 * float(zeta) * float(omega_n)


def predicted_poles(omega_n: float, zeta: float) -> tuple[complex, complex]:
    disc = complex(float(zeta) ** 2 - 1.0, 0.0) ** 0.5
    return (
        -zeta * omega_n + omega_n * disc,
        -zeta * omega_n - omega_n * disc,
    )


def write_config_files(policy: dict[str, Any], out_root: Path) -> tuple[Path, Path]:
    """Materialise the checkpoint's own env and reward config for the subprocess."""

    env_path = out_root / "_env_config.json"
    reward_path = out_root / "_reward_config.json"
    env_path.write_text(json.dumps(policy["env_config"], indent=2), encoding="utf-8")
    reward = {
        key: value
        for key, value in policy["reward_config"].items()
        if key not in REWARD_KEYS_NOT_OVERRIDABLE
    }
    reward_path.write_text(json.dumps(reward, indent=2), encoding="utf-8")
    return env_path, reward_path


def run_kpi_stage(
    *,
    policy: dict[str, Any],
    k0: float,
    k1: float,
    point_dir: Path,
    env_path: Path,
    reward_path: Path,
    episodes: int,
    workers: int,
    seed_start: int,
    log_path: Path,
) -> pd.DataFrame | None:
    point_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "scripts.training.run_ppo_cbf_progression",
        "--traffic-model", "mtm",
        "--device", "cpu",
        "--skip-training",
        "--model-path", str(policy["model_path"]),
        "--variants", str(policy["variant"]),
        "--seeds", str(policy["training_seed"]),
        "--env-config-file", str(env_path),
        "--reward-config-file", str(reward_path),
        "--k0", str(k0),
        "--k1", str(k1),
        # The checkpoint's own CBF settings have to be restated, or the
        # subprocess falls back to the notebook defaults (which differ from
        # this study: max_neighbor_constraints 12 vs the recorded 7) and every
        # grid point is measured against a different filter than k0/k1 alone.
        "--psi1-gain", str(float(policy["cbf"]["CBF_PSI1_GAIN"])),
        "--max-neighbor-constraints",
        str(int(policy["cbf"]["CBF_MAX_NEIGHBOR_CONSTRAINTS"])),
        "--post-train-eval-episodes", str(episodes),
        "--post-train-eval-workers", str(workers),
        "--post-train-eval-seed-start", str(seed_start),
        "--skip-evaluation",
        "--skip-counterfactual",
        "--output-dir", str(point_dir),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    with open(log_path, "w", encoding="utf-8") as handle:
        completed = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), stdout=handle, stderr=subprocess.STDOUT, env=env
        )
    kpi_path = point_dir / "post_train_200ep_kpis.csv"
    if completed.returncode != 0 or not kpi_path.is_file():
        return None
    return pd.read_csv(kpi_path)


def kpi_row(table: pd.DataFrame) -> dict[str, float]:
    cbf = table[table["mode"] == "cbf"]
    out: dict[str, float] = {}
    for kpi, column in (
        ("Ego collisions / km", "collisions_per_km"),
        ("QP failure rate", "qp_failure_rate"),
        ("Intervention rate", "intervention_rate"),
        ("Correction norm", "correction_norm"),
        ("Minimum h", "min_h"),
        ("Episode length (steps)", "episode_length"),
        ("Episode return", "episode_return"),
        ("Distance-based completion rate", "completion_rate"),
        ("Abs speed error (m/s)", "abs_speed_error"),
        ("Mean lateral tracking error (m)", "lateral_error"),
        ("Mean jerk norm", "jerk_norm"),
    ):
        match = cbf[cbf["KPI"] == kpi]
        out[column] = float(match.iloc[0]["Mean"]) if not match.empty else float("nan")
        out[f"{column}_sd"] = float(match.iloc[0]["SD"]) if not match.empty else float("nan")
    return out


def run_trace_stage(
    *,
    namespace: dict[str, Any],
    policy: dict[str, Any],
    model: Any,
    episodes: int,
    seed_start: int,
    max_steps: int,
) -> pd.DataFrame:
    """Per-step barrier state for a handful of episodes at the current gains."""

    from scripts.training.run_ppo_cbf_progression import (
        _predict_evaluation_action,
        make_evaluation_env,
    )

    rows: list[dict[str, float]] = []
    for episode_index in range(int(episodes)):
        env = make_evaluation_env(
            namespace,
            mode="cbf",
            env_config=policy["env_config"],
            reward_config=policy["reward_config"],
            correction_epsilon=policy["correction_epsilon"],
        )
        episode_seed = int(seed_start) + episode_index
        try:
            obs, _ = env.reset(seed=episode_seed)
            for step_index in range(int(max_steps)):
                action = _predict_evaluation_action(model, obs, action_source="policy")
                obs, _reward, terminated, truncated, info = env.step(action)
                row = {
                    "episode_seed": episode_seed,
                    "step": step_index,
                    "ego_collision_events": int(info.get("ego_collision_events", 0)),
                }
                for field in TRACE_FIELDS:
                    row[field] = float(info.get(field, np.nan))
                for flag in TRACE_FLAGS:
                    row[flag] = bool(info.get(flag, False))
                # Whether the QP is actually sitting on the min-h constraint.
                # Neighbour barriers occupy the first rows of the constraint
                # system in the same order, so the neighbour index and the
                # active-set index refer to the same row. Without this the
                # only available test is |psi_2| ~ 0 on a constraint that may
                # simply be slack while the QP binds on another one.
                active = np.asarray(
                    info.get("cbf_active_constraint_indices", ()), dtype=np.int64
                ).reshape(-1)
                min_index = int(row["cbf_min_h_index"])
                row["cbf_min_is_active"] = bool(
                    min_index >= 0 and min_index in set(active.tolist())
                )
                row["cbf_active_count"] = int(active.size)
                rows.append(row)
                if terminated or truncated:
                    break
        finally:
            env.close()
    return pd.DataFrame(rows)


def coverage_from_trace(trace: pd.DataFrame, k0: float, k1: float) -> dict[str, float]:
    """P3/P4 coverage: how much of the episode the second-order model can govern.

    psi_2 is recomputed here from the logged barrier state rather than read from
    cbf_hocbf_min_margin, because that margin is the minimum over every
    constraint (road boundaries included) while the second-order reconstruction
    has to follow one neighbour pair.
    """

    if trace.empty:
        return {
            "steps_traced": 0.0,
            "feasible_fraction": float("nan"),
            "binding_fraction": float("nan"),
            "active_fraction": float("nan"),
            "saturated_fraction": float("nan"),
            "governed_fraction": float("nan"),
            "pair_switch_rate": float("nan"),
        }
    psi2 = trace["cbf_min_bound"] - (
        trace["cbf_min_row_x"] * trace["cbf_a_safe_x"]
        + trace["cbf_min_row_y"] * trace["cbf_a_safe_y"]
    )
    feasible = trace["cbf_qp_success"].to_numpy(dtype=bool)
    saturated = trace["actuator_clip_norm"].to_numpy(dtype=float) > 1e-9
    # "Binding" means the QP is sitting on this constraint, so psi_2 is driven
    # to zero; a slack constraint leaves psi_2 strictly positive and the ODE
    # says nothing about that step.
    binding = np.abs(psi2.to_numpy(dtype=float)) <= 1e-3
    index = trace["cbf_min_h_index"].to_numpy(dtype=float)
    switched = np.zeros_like(index, dtype=bool)
    switched[1:] = index[1:] != index[:-1]
    if "cbf_min_is_active" in trace.columns:
        active = trace["cbf_min_is_active"].to_numpy(dtype=bool)
    else:
        active = binding
    governed = feasible & active & (~saturated)
    return {
        "steps_traced": float(len(trace)),
        "feasible_fraction": float(np.mean(feasible)),
        "binding_fraction": float(np.mean(binding)),
        "active_fraction": float(np.mean(active)),
        "saturated_fraction": float(np.mean(saturated)),
        "governed_fraction": float(np.mean(governed)),
        "pair_switch_rate": float(np.mean(switched)),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--omega-n", type=float, nargs="+",
        default=[1.0, 1.4, 2.0, 2.8, 4.0],
    )
    parser.add_argument(
        "--zeta", type=float, nargs="+",
        default=[0.5, 0.7, 1.0, 1.4, 2.125, 3.0],
    )
    parser.add_argument(
        "--probe-omega-n", type=float, nargs="*", default=[8.0],
        help="Deliberately-too-fast omega_n values, run only at zeta=1.0, to "
             "show the discretisation breakdown predicted by omega_n*dt.",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eval-workers", type=int, default=10)
    parser.add_argument("--trace-episodes", type=int, default=8)
    parser.add_argument("--trace-max-steps", type=int, default=3000)
    parser.add_argument("--episode-seed-start", type=int, default=1_100_000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-traces", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")

    from scripts.rendering.render_ppo_scenario_gifs import build_namespace, load_policy
    from scripts.training.run_ppo_cbf_progression import load_model

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    trace_dir = out_root / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.csv"

    policy = load_policy(args.run_config)
    if not Path(policy["model_path"]).is_file():
        raise FileNotFoundError(f"checkpoint missing: {policy['model_path']}")
    env_path, reward_path = write_config_files(policy, out_root)
    vehicles = int(policy["env_config"].get("vehicles_count", -1))
    print(
        f"[cbf-congested] variant={policy['variant']} seed={policy['training_seed']} "
        f"vehicles_count={vehicles} episodes={args.episodes} "
        f"workers={args.eval_workers}",
        flush=True,
    )

    grid: list[tuple[float, float, bool]] = [
        (omega, zeta, False) for omega in args.omega_n for zeta in args.zeta
    ]
    grid.extend((omega, 1.0, True) for omega in (args.probe_omega_n or ()))

    namespace = None
    model = None
    if not args.skip_traces:
        namespace = build_namespace(PROJECT_ROOT, policy["cbf"])
        model = load_model(policy["variant"], policy["model_path"], "cpu")

    existing: set[tuple[float, float]] = set()
    if args.skip_existing and summary_path.is_file():
        done = pd.read_csv(summary_path)
        existing = {
            (round(float(a), 6), round(float(b), 6))
            for a, b in zip(done["omega_n"], done["zeta"])
        }

    for point_index, (omega_n, zeta, is_probe) in enumerate(grid):
        if (round(omega_n, 6), round(zeta, 6)) in existing:
            print(f"[cbf-congested] skip existing wn={omega_n} zeta={zeta}", flush=True)
            continue
        k0, k1 = gains_from_damping(omega_n, zeta)
        tag = f"wn{omega_n:g}_z{zeta:g}".replace(".", "p")
        started = time.time()
        print(
            f"[cbf-congested] {point_index + 1}/{len(grid)} wn={omega_n:g} "
            f"zeta={zeta:g} -> k0={k0:.4f} k1={k1:.4f}"
            + (" (probe)" if is_probe else ""),
            flush=True,
        )

        table = run_kpi_stage(
            policy=policy,
            k0=k0,
            k1=k1,
            point_dir=out_root / tag,
            env_path=env_path,
            reward_path=reward_path,
            episodes=int(args.episodes),
            workers=int(args.eval_workers),
            seed_start=int(args.episode_seed_start),
            log_path=log_dir / f"{tag}_kpi.log",
        )
        record: dict[str, Any] = {
            "omega_n": float(omega_n),
            "zeta": float(zeta),
            "k0": k0,
            "k1": k1,
            "is_probe": bool(is_probe),
            "omega_n_dt": float(omega_n) * 0.05,
            "kpi_ok": table is not None,
        }
        pole_a, pole_b = predicted_poles(omega_n, zeta)
        record["pole_real_dominant"] = float(max(pole_a.real, pole_b.real))
        record["pole_imag_abs"] = float(abs(pole_a.imag))
        if table is not None:
            record.update(kpi_row(table))

        if not args.skip_traces:
            namespace["CBF_K0"] = float(k0)
            namespace["CBF_K1"] = float(k1)
            trace = run_trace_stage(
                namespace=namespace,
                policy=policy,
                model=model,
                episodes=int(args.trace_episodes),
                seed_start=int(args.episode_seed_start),
                max_steps=int(args.trace_max_steps),
            )
            if not trace.empty:
                trace.insert(0, "zeta", float(zeta))
                trace.insert(0, "omega_n", float(omega_n))
                trace.to_csv(trace_dir / f"{tag}.csv", index=False)
            record.update(coverage_from_trace(trace, k0, k1))

        record["elapsed_sec"] = float(time.time() - started)
        frame = pd.DataFrame([record])
        frame.to_csv(
            summary_path,
            mode="a" if summary_path.is_file() else "w",
            header=not summary_path.is_file(),
            index=False,
        )
        print(
            f"[cbf-congested] {tag}: "
            f"collisions/km={record.get('collisions_per_km', float('nan')):.3f} "
            f"qp_fail={record.get('qp_failure_rate', float('nan')):.3f} "
            f"governed={record.get('governed_fraction', float('nan')):.3f} "
            f"({record['elapsed_sec'] / 60.0:.1f} min)",
            flush=True,
        )

    print(f"[cbf-congested] complete: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
