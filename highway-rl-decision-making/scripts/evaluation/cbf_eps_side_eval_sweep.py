from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


def set_stable_native_defaults() -> None:
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    ]:
        os.environ.setdefault(key, "1")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return candidate
        nested = candidate / "highway-rl-decision-making"
        if (nested / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return nested
    raise RuntimeError("Could not find project root containing notebooks/lanelessKaralakou.ipynb")


def exec_notebook_cells(notebook_path: Path, cell_indices: list[int], namespace: dict[str, Any]) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_index in cell_indices:
        cell = notebook["cells"][cell_index]
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        print(f"[cbf_eps_side_eval_sweep] executing notebook cell {cell_index}", flush=True)
        exec(compile(source, f"{notebook_path}:cell-{cell_index}", "exec"), namespace)


def set_cbf_gains(env: Any, k0: float, k1: float) -> None:
    current = env
    while current is not None:
        if hasattr(current, "k0") and hasattr(current, "k1"):
            current.k0 = float(k0)
            current.k1 = float(k1)
            return
        current = getattr(current, "env", None)
    raise RuntimeError("Could not find SafetyFilteredAccelerationWrapper to set k0/k1.")


def default_damped_model_path(namespace: dict[str, Any], k0: float, k1: float, lambda_filter: float) -> Path:
    tag = f"k0_{k0:.2f}_k1_{k1:.2f}_lambda_{lambda_filter:.3f}".replace(".", "p")
    candidate = Path(namespace["ARTIFACT_DIR"]) / "cbf_damped_retrain" / tag / "model.zip"
    if candidate.exists():
        return candidate
    legacy_tag = f"k0_{k0:.2f}_k1_{k1:.2f}".replace(".", "p")
    legacy_candidate = Path(namespace["ARTIFACT_DIR"]) / "cbf_damped_retrain" / legacy_tag / "model.zip"
    if legacy_candidate.exists():
        return legacy_candidate
    return Path(namespace["DDPG_CBF_MODEL_PATH"])


def evaluate_eps(
    namespace: dict[str, Any],
    model: Any,
    eps_side: float,
    k0: float,
    k1: float,
    lambda_filter: float,
    episodes: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for episode in range(episodes):
        env = namespace["make_cbf_single_env"](
            seed=seed + episode,
            lambda_filter=lambda_filter,
            eps_side=eps_side,
        )
        set_cbf_gains(env, k0, k1)
        namespace["configure_paper_evaluation_env"](env, steps=namespace["PAPER_EVAL_STEPS"])
        obs, _ = env.reset(seed=seed + episode)

        done = False
        step_count = 0
        rewards: list[float] = []
        signed_speed_errors: list[float] = []
        abs_speed_errors: list[float] = []
        lat_y_errors: list[float] = []
        corrections: list[float] = []
        interventions: list[float] = []
        qp_successes: list[float] = []
        min_h_values: list[float] = []
        ego_collisions = 0
        ego_collision_steps = 0
        all_collision_events = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            base = env.unwrapped
            speed_error = float(base.vehicle.vx) - float(base.vehicle.desired_speed)
            lat_y_error = float(info.get("karalakou_lat_y_error_m", np.nan))

            rewards.append(float(reward))
            signed_speed_errors.append(speed_error)
            abs_speed_errors.append(abs(speed_error))
            if np.isfinite(lat_y_error):
                lat_y_errors.append(lat_y_error)
            corrections.append(float(info.get("cbf_correction_norm", 0.0)))
            interventions.append(float(info.get("cbf_intervened", False)))
            qp_successes.append(float(info.get("cbf_qp_success", True)))
            min_h_values.append(float(info.get("cbf_min_h", np.nan)))
            all_collision_events += int(info.get("collisions", 0))
            ego_collisions += int(info.get("ego_collision_events", 0))
            if bool(info.get("ego_collision", False)):
                ego_collision_steps += 1

            step_count += 1
            done = bool(terminated or truncated)

        rows.append(
            {
                "eps_side": float(eps_side),
                "episode": float(episode),
                "steps": float(step_count),
                "return": float(np.sum(rewards)),
                "mean_signed_speed_error": float(np.mean(signed_speed_errors)) if signed_speed_errors else 0.0,
                "mean_abs_speed_error": float(np.mean(abs_speed_errors)) if abs_speed_errors else 0.0,
                "mean_lat_y_error_m": float(np.mean(lat_y_errors)) if lat_y_errors else np.nan,
                "mean_correction_norm": float(np.mean(corrections)) if corrections else 0.0,
                "max_correction_norm": float(np.max(corrections)) if corrections else 0.0,
                "intervention_rate": float(np.mean(interventions)) if interventions else 0.0,
                "qp_failure_rate": float(1.0 - np.mean(qp_successes)) if qp_successes else 0.0,
                "min_h": float(np.nanmin(min_h_values)) if min_h_values and not np.all(np.isnan(min_h_values)) else np.nan,
                "ego_collisions": float(ego_collisions),
                "ego_collision_steps": float(ego_collision_steps),
                "total_collision_events": float(all_collision_events),
            }
        )
        env.close()
    return pd.DataFrame(rows)


def summarize(episodes: pd.DataFrame) -> pd.DataFrame:
    summary = (
        episodes.groupby("eps_side", as_index=False)
        .agg(
            episodes=("episode", "count"),
            return_mean=("return", "mean"),
            return_std=("return", "std"),
            abs_speed_error_mean=("mean_abs_speed_error", "mean"),
            lat_y_error_mean=("mean_lat_y_error_m", "mean"),
            correction_norm_mean=("mean_correction_norm", "mean"),
            max_correction_norm_max=("max_correction_norm", "max"),
            intervention_rate_mean=("intervention_rate", "mean"),
            qp_failure_rate_mean=("qp_failure_rate", "mean"),
            min_h_min=("min_h", "min"),
            ego_collisions_mean=("ego_collisions", "mean"),
            ego_collision_steps_mean=("ego_collision_steps", "mean"),
            total_collision_events_mean=("total_collision_events", "mean"),
        )
        .sort_values("eps_side")
        .reset_index(drop=True)
    )
    feasible = (
        (summary["ego_collisions_mean"] <= 0.0)
        & (summary["qp_failure_rate_mean"] <= 0.01)
        & (summary["return_mean"] >= summary["return_mean"].max() - 35.0)
    )
    score = (
        summary["return_mean"]
        - 80.0 * summary["intervention_rate_mean"]
        - 55.0 * summary["correction_norm_mean"]
        - 25.0 * summary["abs_speed_error_mean"]
        - 250.0 * summary["ego_collisions_mean"]
        - 900.0 * summary["qp_failure_rate_mean"]
    )
    summary["selection_feasible"] = feasible
    summary["selection_score"] = score
    return summary


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    axes = axes.ravel()
    panels = [
        ("return_mean", "Mean Return", None),
        ("intervention_rate_mean", "Intervention Rate", "percent"),
        ("correction_norm_mean", "Mean Correction Norm", None),
        ("abs_speed_error_mean", "Abs Speed Error (m/s)", None),
        ("lat_y_error_mean", "Lateral Y Error (m)", None),
        ("qp_failure_rate_mean", "QP Failure Rate", "percent"),
    ]
    x = summary["eps_side"].to_numpy(dtype=float)
    for axis, (column, title, scale) in zip(axes, panels):
        y = summary[column].to_numpy(dtype=float)
        axis.plot(x, y, marker="o", linewidth=2.0)
        axis.set_title(title)
        axis.set_xlabel("CBF eps_side (m)")
        axis.grid(True, alpha=0.28)
        if scale == "percent":
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    fig.suptitle("Evaluation-Only Sweep: CBF Safe-Set Buffer eps_side", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_eps_values(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation-only sweep over CBF safe-set eps_side.")
    parser.add_argument("--eps-values", default="0.00,0.05,0.10,0.149,0.20,0.25")
    parser.add_argument("--k0", type=float, default=5.29)
    parser.add_argument("--k1", type=float, default=3.68)
    parser.add_argument("--lambda-filter", type=float, default=0.05)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=290_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def write_outputs(
    episodes: pd.DataFrame,
    output_dir: Path,
    model_path: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    summary = summarize(episodes)
    episodes_path = output_dir / "episodes.csv"
    summary_path = output_dir / "summary.csv"
    plot_path = output_dir / "summary.png"
    config_path = output_dir / "run_config.json"
    episodes.to_csv(episodes_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, plot_path)
    config_path.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "k0": args.k0,
                "k1": args.k1,
                "lambda_filter": args.lambda_filter,
                "episodes": args.episodes,
                "seed": args.seed,
                "eps_values": parse_eps_values(args.eps_values),
                "resume": not args.no_resume,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return episodes_path, summary_path, plot_path, config_path


def main() -> int:
    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    args = parse_args()

    project_root = find_project_root(args.project_root or Path.cwd())
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    namespace: dict[str, Any] = {"__name__": "__main__"}
    exec_notebook_cells(notebook_path, [2, 3, 5, 6, 8, 33, 35, 37, 39, 41, 43], namespace)
    namespace["DEVICE"] = args.device

    artifact_dir: Path = namespace["ARTIFACT_DIR"]
    output_dir = args.output_dir or (artifact_dir / "cbf_eps_side_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_path or default_damped_model_path(namespace, args.k0, args.k1, args.lambda_filter)
    episodes_path = output_dir / "episodes.csv"

    print(
        "[eps-sweep] starting",
        {
            "model_path": str(model_path),
            "eps_values": parse_eps_values(args.eps_values),
            "k0": args.k0,
            "k1": args.k1,
            "lambda_filter": args.lambda_filter,
            "episodes": args.episodes,
            "seed": args.seed,
            "output_dir": str(output_dir),
        },
        flush=True,
    )
    model = namespace["DDPG"].load(str(model_path), device=args.device)

    episode_frames: list[pd.DataFrame] = []
    completed_eps: set[float] = set()
    if episodes_path.exists() and not args.no_resume:
        existing = pd.read_csv(episodes_path)
        for eps_side, group in existing.groupby("eps_side"):
            if len(group) >= args.episodes:
                completed_eps.add(float(eps_side))
        if not existing.empty:
            episode_frames.append(existing)
            print(
                "[eps-sweep] resume loaded"
                f" rows={len(existing):,}"
                f" completed_eps={sorted(completed_eps)}",
                flush=True,
            )

    for eps_side in parse_eps_values(args.eps_values):
        if float(eps_side) in completed_eps:
            print(f"[eps-sweep] skipping complete eps_side={eps_side:.3f}", flush=True)
            continue
        print(f"[eps-sweep] eps_side={eps_side:.3f}", flush=True)
        metrics = evaluate_eps(
            namespace,
            model,
            eps_side=eps_side,
            k0=args.k0,
            k1=args.k1,
            lambda_filter=args.lambda_filter,
            episodes=args.episodes,
            seed=args.seed + int(round(eps_side * 10_000)),
        )
        episode_frames.append(metrics)
        episodes_so_far = pd.concat(episode_frames, ignore_index=True)
        write_outputs(episodes_so_far, output_dir, model_path, args)
        partial = summarize(metrics).iloc[0]
        print(
            "[eps-sweep-result]"
            f" eps={eps_side:.3f}"
            f" return={partial['return_mean']:.2f}"
            f" intervention={partial['intervention_rate_mean']:.2%}"
            f" correction={partial['correction_norm_mean']:.3f}"
            f" qp_fail={partial['qp_failure_rate_mean']:.2%}"
            f" ego_coll={partial['ego_collisions_mean']:.2f}",
            flush=True,
        )
        print(f"[eps-sweep] checkpointed {episodes_path}", flush=True)

    episodes = pd.concat(episode_frames, ignore_index=True)
    summary = summarize(episodes)

    episodes_path, summary_path, plot_path, _ = write_outputs(episodes, output_dir, model_path, args)

    print(f"[eps-sweep] wrote {episodes_path}", flush=True)
    print(f"[eps-sweep] wrote {summary_path}", flush=True)
    print(f"[eps-sweep] wrote {plot_path}", flush=True)
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
