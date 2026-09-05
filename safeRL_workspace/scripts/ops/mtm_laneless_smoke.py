from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common.laneless_script_config import deep_update, env_config_from_args


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        env_file = candidate / "laneless highway env" / "lane_free_env.py"
        if env_file.exists():
            return candidate
        nested = candidate / "safeRL_workspace"
        if (nested / "laneless highway env" / "lane_free_env.py").exists():
            return nested
    raise RuntimeError("Could not find project root containing laneless highway env/lane_free_env.py")


def make_env(project_root: Path, traffic_model: str, args: argparse.Namespace) -> gym.Env:
    sys.path.insert(0, str(project_root / "laneless highway env"))
    import lane_free_env  # noqa: F401

    base_config: dict[str, Any] = {
        "traffic_model": traffic_model,
        "vehicles_count": args.vehicles,
        "episode_steps": args.steps,
        "duration": args.steps,
        "terminate_on_collision": False,
        "gamma_nudge": args.gamma_nudge,
    }
    config = env_config_from_args(args, base_config)
    config["traffic_model"] = traffic_model
    config["episode_steps"] = args.steps
    config["duration"] = args.steps
    if args.road_length is not None:
        config["road_length"] = args.road_length
    if args.road_width is not None:
        config["road_width"] = args.road_width
    return gym.make("lane-free-v0", config=config)


def run_rollout(project_root: Path, traffic_model: str, args: argparse.Namespace) -> pd.DataFrame:
    env = make_env(project_root, traffic_model, args)
    obs, info = env.reset(seed=args.seed)
    previous_snapshot = env.unwrapped.snapshot()
    rows: list[dict[str, float | str]] = []
    action = np.zeros(2, dtype=np.float32)

    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(action)
        snapshot = env.unwrapped.snapshot()
        non_ego = snapshot[snapshot[:, 7] < 0.5]
        previous_non_ego = previous_snapshot[previous_snapshot[:, 7] < 0.5]
        if len(non_ego) and len(previous_non_ego) == len(non_ego):
            mean_abs_y_step = float(np.mean(np.abs(non_ego[:, 1] - previous_non_ego[:, 1])))
        else:
            mean_abs_y_step = 0.0
        rows.append(
            {
                "traffic_model": traffic_model,
                "step": float(step + 1),
                "reward": float(reward),
                "mean_speed": float(info.get("mean_speed", 0.0)),
                "collisions": float(info.get("collisions", 0)),
                "active_collisions": float(info.get("active_collisions", 0)),
                "flow_count": float(info.get("flow_count", 0)),
                "mtm_active_leader_rate": float(info.get("mtm_active_leader_rate", 0.0)),
                "mtm_mean_abs_vy": float(info.get("mtm_mean_abs_vy", 0.0)),
                "mtm_mean_abs_desired_vy": float(info.get("mtm_mean_abs_desired_vy", 0.0)),
                "mean_abs_y_step": mean_abs_y_step,
                "lateral_spread": float(np.std(non_ego[:, 1])) if len(non_ego) else 0.0,
                "mean_abs_non_ego_vy": float(np.mean(np.abs(non_ego[:, 3]))) if len(non_ego) else 0.0,
            }
        )
        previous_snapshot = snapshot
        if terminated or truncated:
            break

    env.close()
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("traffic_model", as_index=False)
        .agg(
            steps=("step", "max"),
            mean_speed=("mean_speed", "mean"),
            mean_abs_y_step=("mean_abs_y_step", "mean"),
            mean_abs_non_ego_vy=("mean_abs_non_ego_vy", "mean"),
            lateral_spread=("lateral_spread", "mean"),
            mtm_active_leader_rate=("mtm_active_leader_rate", "mean"),
            total_collision_events=("collisions", "sum"),
            max_active_collisions=("active_collisions", "max"),
            final_flow_count=("flow_count", "max"),
        )
        .sort_values("traffic_model")
    )


def plot(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    axes = axes.ravel()
    panels = [
        ("mean_speed", "Mean Speed"),
        ("mean_abs_non_ego_vy", "Mean |Non-Ego vy|"),
        ("mean_abs_y_step", "Mean |Delta y| Per Step"),
        ("active_collisions", "Active Collisions"),
    ]
    for axis, (column, title) in zip(axes, panels):
        for traffic_model, group in frame.groupby("traffic_model"):
            axis.plot(group["step"], group[column], label=traffic_model)
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle("Lane-Free Traffic Model Smoke Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test force vs MTM traffic in the existing lane-free env.")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--vehicles", type=int, default=35)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gamma-nudge", type=float, default=0.0)
    parser.add_argument("--road-length", type=float, default=None)
    parser.add_argument("--road-width", type=float, default=None)
    parser.add_argument("--env-config-json", default=None, help="JSON object merged into both smoke-test env configs.")
    parser.add_argument("--env-config-file", type=Path, default=None, help="JSON file merged into both smoke-test env configs.")
    parser.set_defaults(traffic_model=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = find_project_root(args.project_root or Path.cwd())
    output_dir = args.output_dir or (project_root / "artifacts" / "lanelessKaralakou" / "mtm_laneless_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        run_rollout(project_root, "force", args),
        run_rollout(project_root, "mtm", args),
    ]
    frame = pd.concat(frames, ignore_index=True)
    summary = summarize(frame)
    trace_path = output_dir / "traffic_model_smoke_trace.csv"
    summary_path = output_dir / "traffic_model_smoke_summary.csv"
    plot_path = output_dir / "traffic_model_smoke_comparison.png"
    frame.to_csv(trace_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot(frame, plot_path)

    print(f"[mtm-smoke] wrote {trace_path}", flush=True)
    print(f"[mtm-smoke] wrote {summary_path}", flush=True)
    print(f"[mtm-smoke] wrote {plot_path}", flush=True)
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
