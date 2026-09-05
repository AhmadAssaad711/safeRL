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
        print(f"[cbf_damped_gain_eval_sweep] executing notebook cell {cell_index}", flush=True)
        exec(compile(source, f"{notebook_path}:cell-{cell_index}", "exec"), namespace)


def coarse_damped_candidates() -> list[dict[str, float | str]]:
    base_omega = float(np.sqrt(2.0))
    current_zeta = float(4.0 / (2.0 * base_omega))
    raw = [
        ("low-frequency critical", 1.0, 1.0),
        ("base underdamped", base_omega, 0.7),
        ("base critical", base_omega, 1.0),
        ("current overdamped", base_omega, current_zeta),
        ("high-frequency critical", 2.0, 1.0),
    ]
    return [
        {
            "label": label,
            "omega_n": omega,
            "zeta": zeta,
            "k0": omega**2,
            "k1": 2.0 * zeta * omega,
        }
        for label, omega, zeta in raw
    ]


def refined_damped_candidates() -> list[dict[str, float | str]]:
    raw = []
    for omega in [1.7, 2.0, 2.3]:
        for zeta in [0.8, 1.0, 1.2]:
            raw.append((f"wn={omega:.1f}, z={zeta:.1f}", omega, zeta))
    return [
        {
            "label": label,
            "omega_n": omega,
            "zeta": zeta,
            "k0": omega**2,
            "k1": 2.0 * zeta * omega,
        }
        for label, omega, zeta in raw
    ]


def damped_candidates(mode: str) -> list[dict[str, float | str]]:
    if mode == "refined":
        return refined_damped_candidates()
    return coarse_damped_candidates()


def set_cbf_gains(env: Any, k0: float, k1: float) -> None:
    current = env
    while current is not None:
        if hasattr(current, "k0") and hasattr(current, "k1"):
            current.k0 = float(k0)
            current.k1 = float(k1)
            return
        current = getattr(current, "env", None)
    raise RuntimeError("Could not find SafetyFilteredAccelerationWrapper to set k0/k1.")


def evaluate_candidate(
    namespace: dict[str, Any],
    model: Any,
    candidate: dict[str, float | str],
    episodes: int,
    seed: int,
    deterministic: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    k0 = float(candidate["k0"])
    k1 = float(candidate["k1"])
    for episode in range(episodes):
        env = namespace["make_cbf_single_env"](
            seed=seed + episode,
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
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
            action, _ = model.predict(obs, deterministic=deterministic)
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
                "label": str(candidate["label"]),
                "omega_n": float(candidate["omega_n"]),
                "zeta": float(candidate["zeta"]),
                "k0": k0,
                "k1": k1,
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


def summarize(episodes: pd.DataFrame, candidates: list[dict[str, float | str]]) -> pd.DataFrame:
    grouped = episodes.groupby(["label", "omega_n", "zeta", "k0", "k1"], as_index=False)
    summary = grouped.agg(
        return_mean=("return", "mean"),
        return_std=("return", "std"),
        abs_speed_error_mean=("mean_abs_speed_error", "mean"),
        lat_y_error_mean=("mean_lat_y_error_m", "mean"),
        intervention_rate_mean=("intervention_rate", "mean"),
        correction_norm_mean=("mean_correction_norm", "mean"),
        max_correction_norm_max=("max_correction_norm", "max"),
        min_h_min=("min_h", "min"),
        ego_collisions_mean=("ego_collisions", "mean"),
        total_collision_events_mean=("total_collision_events", "mean"),
        qp_failure_rate_mean=("qp_failure_rate", "mean"),
    )
    order = {str(candidate["label"]): index for index, candidate in enumerate(candidates)}
    summary["_order"] = summary["label"].map(lambda label: order.get(str(label), 999))
    return summary.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary["omega_n"].nunique() > 1 and summary["zeta"].nunique() > 1 and len(summary) > 5:
        plot_summary_heatmap(summary, output_path)
        return

    x = np.arange(len(summary))
    labels = [
        f"{row.label}\nwn={row.omega_n:.2f}, z={row.zeta:.2f}"
        for row in summary.itertuples(index=False)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    panels = [
        (axes[0, 0], "return_mean", "Mean Return", None),
        (axes[0, 1], "abs_speed_error_mean", "Abs Speed Error (m/s)", None),
        (axes[1, 0], "lat_y_error_mean", "Lateral Y Error (m)", None),
        (axes[1, 1], "intervention_rate_mean", "CBF Intervention Rate", "percent"),
    ]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    for axis, column, title, scale in panels:
        values = summary[column].to_numpy(dtype=float)
        axis.bar(x, values, color=colors[: len(summary)])
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if scale == "percent":
            axis.set_ylim(0.0, max(1.0, float(np.nanmax(values)) * 1.15))
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))

    fig.suptitle("Damped-System CBF Gain Sweep: evaluation only", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_summary_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("return_mean", "Mean Return", "{:.1f}"),
        ("abs_speed_error_mean", "Abs Speed Error (m/s)", "{:.3f}"),
        ("lat_y_error_mean", "Lateral Y Error (m)", "{:.3f}"),
        ("intervention_rate_mean", "CBF Intervention Rate", "{:.1%}"),
    ]
    zetas = sorted(summary["zeta"].unique())
    omegas = sorted(summary["omega_n"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    for axis, (column, title, formatter) in zip(axes.flat, metrics):
        grid = (
            summary.pivot(index="zeta", columns="omega_n", values=column)
            .reindex(index=zetas, columns=omegas)
            .to_numpy(dtype=float)
        )
        image = axis.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("Natural frequency omega_n")
        axis.set_ylabel("Damping ratio zeta")
        axis.set_xticks(np.arange(len(omegas)))
        axis.set_xticklabels([f"{omega:.1f}" for omega in omegas])
        axis.set_yticks(np.arange(len(zetas)))
        axis.set_yticklabels([f"{zeta:.1f}" for zeta in zetas])
        for row_index, zeta in enumerate(zetas):
            for col_index, omega in enumerate(omegas):
                value = grid[row_index, col_index]
                text = formatter.format(value)
                axis.text(col_index, row_index, text, ha="center", va="center", color="white", fontsize=9)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle("Refined Damped-System CBF Gain Sweep: evaluation only", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small evaluation-only CBF damped-gain sweep.")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=190_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--mode", choices=["coarse", "refined"], default="coarse")
    return parser.parse_args()


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
    output_dir = artifact_dir / ("cbf_damped_gain_sweep_refined" if args.mode == "refined" else "cbf_damped_gain_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = namespace["DDPG_CBF_MODEL_PATH"]
    print(f"[sweep] loading {model_path}", flush=True)
    model = namespace["DDPG"].load(str(model_path), device=args.device)

    episode_frames = []
    candidates = damped_candidates(args.mode)
    for index, candidate in enumerate(candidates):
        print(
            "[sweep]"
            f" {candidate['label']}: omega_n={float(candidate['omega_n']):.3f}"
            f" zeta={float(candidate['zeta']):.3f}"
            f" k0={float(candidate['k0']):.3f}"
            f" k1={float(candidate['k1']):.3f}"
            f" episodes={args.episodes}",
            flush=True,
        )
        episode_frames.append(evaluate_candidate(namespace, model, candidate, args.episodes, int(args.seed)))

    episodes = pd.concat(episode_frames, ignore_index=True)
    summary = summarize(episodes, candidates)

    episodes_path = output_dir / "episodes.csv"
    summary_path = output_dir / "summary.csv"
    plot_path = output_dir / "summary.png"
    episodes.to_csv(episodes_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, plot_path)

    print(f"[sweep] wrote {episodes_path}", flush=True)
    print(f"[sweep] wrote {summary_path}", flush=True)
    print(f"[sweep] wrote {plot_path}", flush=True)
    print(
        summary[
            [
                "label",
                "omega_n",
                "zeta",
                "k0",
                "k1",
                "return_mean",
                "abs_speed_error_mean",
                "lat_y_error_mean",
                "intervention_rate_mean",
                "correction_norm_mean",
                "min_h_min",
                "ego_collisions_mean",
                "qp_failure_rate_mean",
            ]
        ].to_string(index=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
