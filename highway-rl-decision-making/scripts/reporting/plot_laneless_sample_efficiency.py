from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    color: str
    tensorboard_run: str


RUNS = [
    RunSpec(
        key="baseline",
        label="Baseline DDPG",
        color="#2563eb",
        tensorboard_run="DDPG_62",
    ),
    RunSpec(
        key="cbf_reward",
        label="DDPG-CBF reward",
        color="#059669",
        tensorboard_run="DDPG_63",
    ),
    RunSpec(
        key="cbf_reward_actor_loss",
        label="DDPG-CBF reward + actor loss",
        color="#c2410c",
        tensorboard_run="DDPG_64",
    ),
]

SCALAR_TAGS = {
    "rollout/ep_len_mean": "ep_len_mean",
    "rollout/ep_rew_mean": "ep_rew_mean",
    "time/fps": "fps",
}

EPISODE_HORIZON = 800.0
HORIZON_THRESHOLD = 0.95 * EPISODE_HORIZON


def find_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    for candidate in [script_path.parent, *script_path.parents]:
        if (candidate / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return candidate
    raise RuntimeError("Could not find repo root containing notebooks/lanelessKaralakou.ipynb")


def load_run_scalars(tensorboard_dir: Path, run: RunSpec) -> pd.DataFrame:
    run_dir = tensorboard_dir / run.tensorboard_run
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing TensorBoard run for {run.label}: {run_dir}")

    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    available_tags = set(accumulator.Tags().get("scalars", []))

    frames: list[pd.DataFrame] = []
    for tag, column in SCALAR_TAGS.items():
        if tag not in available_tags:
            continue
        events = accumulator.Scalars(tag)
        frames.append(
            pd.DataFrame(
                {
                    "timesteps": [float(event.step) for event in events],
                    column: [float(event.value) for event in events],
                }
            )
        )

    if not frames:
        raise RuntimeError(f"No rollout scalars found in {run_dir}")

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="timesteps", how="outer")
    merged = merged.sort_values("timesteps").reset_index(drop=True)
    merged["timesteps_k"] = merged["timesteps"] / 1000.0
    merged["run_key"] = run.key
    merged["run_label"] = run.label
    merged["tensorboard_run"] = run.tensorboard_run
    return merged


def load_training_logs(artifact_dir: Path) -> pd.DataFrame:
    tensorboard_dir = artifact_dir / "tensorboard"
    frames = [load_run_scalars(tensorboard_dir, run) for run in RUNS]
    return pd.concat(frames, ignore_index=True)


def add_zero_start_anchor(data: pd.DataFrame) -> pd.DataFrame:
    anchors: list[dict[str, float | str]] = []
    for run in RUNS:
        anchors.append(
            {
                "timesteps": 0.0,
                "ep_len_mean": 0.0,
                "ep_rew_mean": 0.0,
                "fps": np.nan,
                "timesteps_k": 0.0,
                "run_key": run.key,
                "run_label": run.label,
                "tensorboard_run": run.tensorboard_run,
                "point_source": "zero_start_anchor",
            }
        )

    plot_data = data.copy()
    plot_data["point_source"] = "tensorboard"
    return pd.concat([pd.DataFrame(anchors), plot_data], ignore_index=True).sort_values(
        ["run_key", "timesteps"]
    )


def first_step_at_or_above(frame: pd.DataFrame, column: str, threshold: float) -> float:
    hits = frame.loc[frame[column] >= threshold, "timesteps"]
    return float(hits.iloc[0]) if len(hits) else np.nan


def auc_over_logged_steps(frame: pd.DataFrame, column: str) -> float:
    clean = frame[["timesteps", column]].dropna().sort_values("timesteps")
    if len(clean) == 0:
        return np.nan
    x = clean["timesteps"].to_numpy(dtype=float)
    y = clean[column].to_numpy(dtype=float)
    if len(clean) == 1 or x[-1] == x[0]:
        return float(y[-1])
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for run in RUNS:
        frame = data.loc[data["run_key"] == run.key].sort_values("timesteps")
        if "point_source" in frame.columns:
            logged_frame = frame.loc[frame["point_source"] != "zero_start_anchor"].copy()
        else:
            logged_frame = frame
        rows.append(
            {
                "run_key": run.key,
                "run_label": run.label,
                "tensorboard_run": run.tensorboard_run,
                "zero_start_anchor_used": "yes" if len(logged_frame) != len(frame) else "no",
                "num_tensorboard_points": float(len(logged_frame)),
                "first_tensorboard_timestep": float(logged_frame["timesteps"].min()),
                "last_tensorboard_timestep": float(logged_frame["timesteps"].max()),
                "ep_len_mean_first_tensorboard": float(logged_frame["ep_len_mean"].dropna().iloc[0]),
                "ep_len_mean_last_tensorboard": float(logged_frame["ep_len_mean"].dropna().iloc[-1]),
                "ep_len_mean_auc": auc_over_logged_steps(frame, "ep_len_mean"),
                "first_timestep_ep_len_ge_760": first_step_at_or_above(frame, "ep_len_mean", HORIZON_THRESHOLD),
                "ep_rew_mean_first_tensorboard": float(logged_frame["ep_rew_mean"].dropna().iloc[0]),
                "ep_rew_mean_last_tensorboard": float(logged_frame["ep_rew_mean"].dropna().iloc[-1]),
                "ep_rew_mean_auc": auc_over_logged_steps(frame, "ep_rew_mean"),
            }
        )
    return pd.DataFrame(rows)


def style_axes(ax) -> None:
    ax.grid(True, color="#d1d5db", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=12)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "legend.fontsize": 12,
        }
    )


def plot_episode_length(data: pd.DataFrame, output_path: Path, *, early_zoom: bool) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    for run in RUNS:
        frame = data.loc[data["run_key"] == run.key].sort_values("timesteps")
        ax.plot(
            frame["timesteps_k"],
            frame["ep_len_mean"],
            marker="o",
            markersize=6.0 if early_zoom else 5.0,
            linewidth=3.0,
            color=run.color,
            label=run.label,
        )

        if early_zoom and run.key != "baseline":
            first = frame.loc[frame["point_source"] == "tensorboard"].dropna(subset=["ep_len_mean"]).iloc[0]
            x = float(first["timesteps_k"])
            y = float(first["ep_len_mean"])
            ax.annotate(
                f"first log\n{x:.1f}k, {y:.0f}",
                xy=(x, y),
                xytext=(x + 0.8, y - (90 if run.key == "cbf_reward" else 160)),
                arrowprops={"arrowstyle": "->", "color": run.color, "lw": 1.8},
                color=run.color,
                fontsize=12,
                fontweight="bold",
            )

    ax.axhline(EPISODE_HORIZON, color="#111827", linewidth=1.8, alpha=0.7)
    ax.axhline(HORIZON_THRESHOLD, color="#111827", linewidth=1.4, linestyle="--", alpha=0.55)

    right_x = 12.2 if early_zoom else 50.5
    ax.text(right_x, EPISODE_HORIZON, "episode horizon 800", va="center", ha="left", fontsize=11)
    ax.text(right_x, HORIZON_THRESHOLD, "95% horizon", va="center", ha="left", fontsize=11)
    title = "Early Training Zoom: Episode Length Jumps To Horizon" if early_zoom else "Training Log: Rolling Mean Episode Length"
    ax.set_title(title)
    ax.set_xlabel("environment steps (thousands)")
    ax.set_ylabel("SB3 rollout/ep_len_mean")
    ax.set_xlim(0, 12.5 if early_zoom else 53)
    ax.set_ylim(0, 850)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    style_axes(ax)

    if early_zoom:
        fig.text(
            0.5,
            0.025,
            "Curves include a zero-start anchor at training start; all later points are TensorBoard rollout logs.",
            ha="center",
            fontsize=11,
            color="#374151",
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    else:
        fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_return(data: pd.DataFrame, output_path: Path) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    for run in RUNS:
        frame = data.loc[data["run_key"] == run.key].sort_values("timesteps")
        ax.plot(
            frame["timesteps_k"],
            frame["ep_rew_mean"],
            marker="o",
            markersize=5.0,
            linewidth=2.7,
            color=run.color,
            label=run.label,
        )

    ax.set_title("Training Log: Rolling Mean Return")
    ax.set_xlabel("environment steps (thousands)")
    ax.set_ylabel("SB3 rollout/ep_rew_mean")
    ax.set_xlim(0, 53)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    style_axes(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_length_auc(summary: pd.DataFrame, output_path: Path) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=180)
    ordered = summary.set_index("run_key").loc[[run.key for run in RUNS]].reset_index()
    colors = [run.color for run in RUNS]
    labels = ["Baseline", "CBF\nreward", "CBF +\nactor loss"]

    bars = ax.bar(labels, ordered["ep_len_mean_auc"], color=colors, width=0.66)
    ax.set_title("Length Sample Efficiency")
    ax.set_ylabel("episode-length AUC")
    ax.set_ylim(0, 850)
    style_axes(ax)
    ax.tick_params(axis="x", length=0)
    for bar, value in zip(bars, ordered["ep_len_mean_auc"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 18,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_steps_to_horizon(summary: pd.DataFrame, output_path: Path) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=180)
    ordered = summary.set_index("run_key").loc[[run.key for run in RUNS]].reset_index()
    colors = [run.color for run in RUNS]
    labels = ["Baseline", "CBF\nreward", "CBF +\nactor loss"]
    threshold_steps = ordered["first_timestep_ep_len_ge_760"].copy()
    reached = threshold_steps.notna()
    threshold_steps = threshold_steps.fillna(50_000.0) / 1000.0
    bars = ax.bar(labels, threshold_steps, color=colors, width=0.66)
    ax.set_title("Steps To 95% Horizon")
    ax.set_ylabel("thousand steps")
    ax.set_ylim(0, 55)
    style_axes(ax)
    ax.tick_params(axis="x", length=0)
    for bar, value, did_reach in zip(bars, threshold_steps, reached):
        text = f"{value:.1f}k" if did_reach else ">50k"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            text,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def run() -> tuple[list[Path], Path, Path]:
    repo_root = find_repo_root()
    artifact_dir = repo_root / "artifacts" / "lanelessKaralakou"
    output_dir = artifact_dir / "training_data"
    summary_path = output_dir / "sample_efficiency_summary.csv"
    raw_path = output_dir / "training_log_rollout_scalars.csv"

    data = load_training_logs(artifact_dir)
    plot_data = add_zero_start_anchor(data)
    summary = summarize(plot_data)

    legacy_combined = output_dir / "sample_efficiency_training_curves.png"
    if legacy_combined.exists():
        legacy_combined.unlink()

    output_paths = [
        output_dir / "episode_length_training_early_zoom.png",
        output_dir / "episode_length_training_full.png",
        output_dir / "training_return.png",
        output_dir / "episode_length_auc.png",
        output_dir / "steps_to_95_horizon.png",
    ]
    plot_episode_length(plot_data, output_paths[0], early_zoom=True)
    plot_episode_length(plot_data, output_paths[1], early_zoom=False)
    plot_training_return(plot_data, output_paths[2])
    plot_length_auc(summary, output_paths[3])
    plot_steps_to_horizon(summary, output_paths[4])

    data.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    return output_paths, summary_path, raw_path


def main() -> None:
    output_paths, summary_path, raw_path = run()
    print("Saved training-log sample-efficiency figures:")
    for path in output_paths:
        print(f"  {path}")
    print(f"Saved training-log summary: {summary_path}")
    print(f"Saved extracted rollout scalars: {raw_path}")


if __name__ == "__main__":
    main()
