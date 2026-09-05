from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import lane_free_env  # noqa: F401 - registers lane-free-v0


def run_case(
    gamma_nudge: float,
    *,
    steps: int = 2000,
    seed: int = 7,
    render_mode: str | None = None,
    step_delay: float = 0.0,
) -> dict:
    env = gym.make(
        "lane-free-v0",
        render_mode=render_mode,
        config={
            "gamma_nudge": gamma_nudge,
            "ego_controlled": False,
            "episode_steps": steps,
            "vehicles_count": 30,
        },
    )
    obs, info = env.reset(seed=seed)
    mean_speed = []
    collisions = []
    flow = []

    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        mean_speed.append(info["mean_speed"])
        collisions.append(info["cumulative_collisions"])
        flow.append(info["flow_per_hour"])
        if render_mode == "human" and step_delay > 0.0:
            time.sleep(step_delay)
        if terminated or truncated:
            break

    snapshot = env.unwrapped.snapshot()
    env.close()
    return {
        "gamma_nudge": gamma_nudge,
        "mean_speed": np.asarray(mean_speed),
        "collisions": np.asarray(collisions),
        "flow": np.asarray(flow),
        "snapshot": snapshot,
    }


def plot_results(results: list[dict]) -> Path:
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lane_free_demo.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=140)
    ax_snapshot, ax_speed, ax_collision, ax_flow = axes.ravel()

    for result in results:
        label = f"gamma_nudge={result['gamma_nudge']}"
        snapshot = result["snapshot"]
        ego = snapshot[:, 7] > 0.5
        ax_snapshot.scatter(snapshot[~ego, 0], snapshot[~ego, 1], s=22, alpha=0.65, label=label)
        ax_snapshot.scatter(snapshot[ego, 0], snapshot[ego, 1], s=70, marker="*", edgecolor="black")
        ax_speed.plot(result["mean_speed"], label=label)
        ax_collision.plot(result["collisions"], label=label)
        ax_flow.plot(result["flow"], label=label)

    ax_snapshot.set_title("Final x-y Snapshot")
    ax_snapshot.set_xlabel("x [m]")
    ax_snapshot.set_ylabel("y [m]")
    ax_snapshot.set_ylim(0, 10.2)
    ax_snapshot.legend()

    ax_speed.set_title("Mean Speed")
    ax_speed.set_xlabel("step")
    ax_speed.set_ylabel("m/s")
    ax_speed.legend()

    ax_collision.set_title("Cumulative Collisions")
    ax_collision.set_xlabel("step")
    ax_collision.set_ylabel("count")
    ax_collision.legend()

    ax_flow.set_title("Approximate Flow at x=0")
    ax_flow.set_xlabel("step")
    ax_flow.set_ylabel("veh/hour")
    ax_flow.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lane-free TrafficFluid demo.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of simulation steps per run.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--render",
        choices=["none", "human"],
        default="none",
        help="Use --render human to open a live pygame window.",
    )
    parser.add_argument(
        "--gamma-nudge",
        type=float,
        default=0.5,
        help="Nudging value used for the live human render run.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Optional wall-clock delay between human-rendered steps.",
    )
    args = parser.parse_args()

    if args.render == "human":
        result = run_case(
            args.gamma_nudge,
            steps=args.steps,
            seed=args.seed,
            render_mode="human",
            step_delay=args.step_delay,
        )
        print(
            f"gamma_nudge={result['gamma_nudge']}: "
            f"final mean speed={result['mean_speed'][-1]:.2f} m/s, "
            f"collisions={int(result['collisions'][-1])}, "
            f"flow={result['flow'][-1]:.1f} veh/hour"
        )
        return

    results = [
        run_case(0.0, steps=args.steps, seed=args.seed),
        run_case(0.5, steps=args.steps, seed=args.seed),
    ]
    output_path = plot_results(results)
    print(f"Saved demo plot to {output_path}")
    for result in results:
        print(
            f"gamma_nudge={result['gamma_nudge']}: "
            f"final mean speed={result['mean_speed'][-1]:.2f} m/s, "
            f"collisions={int(result['collisions'][-1])}, "
            f"flow={result['flow'][-1]:.1f} veh/hour"
        )


if __name__ == "__main__":
    main()
