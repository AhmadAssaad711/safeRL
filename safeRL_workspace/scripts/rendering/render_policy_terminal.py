"""Render a policy in the terminal as a top-down ASCII road, with the lateral target drawn as a line.

The episode runs through the *same* stack the post-training evaluation uses --
``run_ppo_cbf_progression.make_evaluation_env`` with the run's saved env,
reward, and CBF configuration -- so a rendered episode reproduces the measured
one step for step on the same seed.  ``--mode raw`` renders with the external
CBF disabled and ``--mode cbf`` with the filter in the loop.

Every frame is one policy step (20 Hz).  The view is a window of road around
the ego: columns are metres of longitudinal distance, rows are metres across
the 10.2 m road, top of the frame is the high-y edge.  The reward's lateral
target is drawn as a full-width line of ``=`` through the frame, which is the
point of this renderer: it shows where the reward is currently telling the
policy to be, against the traffic that produced that target.

    E   ego footprint            =   lateral target the reward tracks
    #   surrounding vehicle      >   the target line, in the left margin

Reward-variant flags are accepted, so the same policy can be rendered against
the target a *different* reward definition would have produced.  Note what that
means: the target also occupies an observation slot, so a variant changes what
the policy reads and the episode then diverges from the measured one.  Render
without variant flags to reproduce the evaluated episode exactly; render with
them to see what a candidate target would point at.
"""

from __future__ import annotations

import os

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import scripts.training.run_ppo_cbf_progression as progression
from saferl.rewards import (
    add_reward_variant_arguments,
    apply_reward_variant_arguments,
    reward_variant_summary,
)
from scripts.evaluation.evaluate_ppo_cbf_overrides import (
    apply_env_overrides,
    build_namespace,
    load_source_run,
)

DEFAULT_SEED = 1_100_000
DEFAULT_AHEAD_M = 70.0
DEFAULT_BEHIND_M = 25.0
DEFAULT_COLUMNS = 95
DEFAULT_ROWS = 21
CLEAR_SCREEN = "\033[H\033[J"

EMPTY = " "
TARGET_MARK = "="
EGO_MARK = "E"
VEHICLE_MARK = "#"


def _base_env(env: Any) -> Any:
    """The lane-free environment under the wrapper chain."""

    return env.unwrapped


class RoadFrame:
    """A character grid of the road window around the ego."""

    def __init__(self, *, road_width: float, ahead_m: float, behind_m: float, columns: int, rows: int) -> None:
        self.road_width = float(road_width)
        self.ahead_m = float(ahead_m)
        self.behind_m = float(behind_m)
        self.columns = int(columns)
        self.rows = int(rows)
        self.metres_per_column = (self.ahead_m + self.behind_m) / self.columns
        self.metres_per_row = self.road_width / self.rows
        self.grid = [[EMPTY] * self.columns for _ in range(self.rows)]

    def column(self, dx: float) -> int:
        return int(round((float(dx) + self.behind_m) / self.metres_per_column))

    def row(self, y: float) -> int:
        # Row 0 is the top of the frame, which is the high-y road edge.
        index = int((self.road_width - float(y)) / self.metres_per_row)
        return min(max(index, 0), self.rows - 1)

    def draw_line(self, y: float, mark: str = TARGET_MARK) -> int:
        row = self.row(y)
        self.grid[row] = [mark if cell == EMPTY else cell for cell in self.grid[row]]
        return row

    def draw_box(self, *, dx: float, y: float, length: float, width: float, mark: str) -> None:
        first = self.column(dx - 0.5 * float(length))
        last = self.column(dx + 0.5 * float(length))
        top = self.row(y + 0.5 * float(width))
        bottom = self.row(y - 0.5 * float(width))
        for row in range(min(top, bottom), max(top, bottom) + 1):
            for column in range(first, last + 1):
                if 0 <= row < self.rows and 0 <= column < self.columns:
                    self.grid[row][column] = mark

    def render(self, *, target_row: int, target_y: float, ego_y: float) -> list[str]:
        lines = ["      +" + "-" * self.columns + "+"]
        for index, row in enumerate(self.grid):
            y_high = self.road_width - index * self.metres_per_row
            margin = ">>>>> " if index == target_row else f"{y_high:5.1f} "
            lines.append(margin + "|" + "".join(row) + "|")
        lines.append("      +" + "-" * self.columns + "+")
        ticks = [EMPTY] * self.columns
        labels = [EMPTY] * self.columns
        for metres in range(int(-self.behind_m), int(self.ahead_m) + 1, 10):
            column = self.column(metres)
            if not 0 <= column < self.columns:
                continue
            ticks[column] = "'"
            text = f"{metres}"
            start = max(0, column - len(text) // 2)
            for offset, character in enumerate(text):
                if start + offset < self.columns:
                    labels[start + offset] = character
        lines.append("       " + "".join(ticks))
        lines.append("       " + "".join(labels) + "  m from ego")
        lines.append(
            f"      target y = {target_y:5.2f} m (the '=' line), ego y = {ego_y:5.2f} m, "
            f"error = {abs(target_y - ego_y):4.2f} m"
        )
        return lines


def build_frame(env: Any, *, info: dict[str, Any], arguments: argparse.Namespace) -> list[str]:
    base = _base_env(env)
    ego = base.vehicle
    road_width = float(base.config["road_width"])
    frame = RoadFrame(
        road_width=road_width,
        ahead_m=float(arguments.ahead_m),
        behind_m=float(arguments.behind_m),
        columns=int(arguments.columns),
        rows=int(arguments.rows),
    )
    target_y = float(info.get("karalakou_target_y", 0.5 * road_width))
    # The target line goes down first so vehicles and the ego draw over it.
    target_row = frame.draw_line(target_y)
    ego_x = float(ego.position[0])
    for vehicle in base.road.vehicles:
        if vehicle is ego:
            continue
        dx = float(base._signed_distance(ego_x, float(vehicle.position[0])))
        if not -frame.behind_m - 5.0 <= dx <= frame.ahead_m + 5.0:
            continue
        frame.draw_box(
            dx=dx,
            y=float(vehicle.position[1]),
            length=float(vehicle.length),
            width=float(vehicle.width),
            mark=VEHICLE_MARK,
        )
    frame.draw_box(
        dx=0.0,
        y=float(ego.position[1]),
        length=float(ego.length),
        width=float(ego.width),
        mark=EGO_MARK,
    )
    return frame.render(target_row=target_row, target_y=target_y, ego_y=float(ego.position[1]))


def status_lines(
    *,
    step: int,
    reward: float,
    episode_return: float,
    info: dict[str, Any],
    mode: str,
    variants: dict[str, str],
) -> list[str]:
    zone = "GAP" if float(info.get("karalakou_zone_found", 0.0)) > 0.5 else "FALLBACK"
    lines = [
        f"step {step:4d}  return {episode_return:8.2f}  reward {reward:6.3f}   "
        f"v {float(info.get('karalakou_ego_speed', float('nan'))):5.2f} -> "
        f"{float(info.get('karalakou_target_speed', float('nan'))):5.2f} m/s   "
        f"cx {float(info.get('karalakou_cx', float('nan'))):5.3f}  "
        f"cy {float(info.get('karalakou_cy', float('nan'))):5.3f}  "
        f"cf {float(info.get('karalakou_cf', float('nan'))):5.3f}",
        f"target {zone:8s}  blockers={variants['lateral_blockers']}  "
        f"fallback={variants['lateral_target_fallback']}  wf={variants['potential_field_weight']}  "
        f"cbf={mode}",
    ]
    safety = []
    for key, label in (
        ("cbf_min_h", "min h"),
        ("cbf_qp_failed", "qp failed"),
        ("cbf_intervened", "intervened"),
        ("cbf_correction_norm", "correction"),
    ):
        if key in info:
            safety.append(f"{label} {float(info[key]):.3f}")
    if safety:
        lines.append("  ".join(safety))
    if float(info.get("karalakou_ego_collision", 0.0)) > 0.5:
        lines.append("*** EGO COLLISION ***")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Ladder seed directory holding run_config.json and model_final.zip.",
    )
    parser.add_argument("--mode", choices=tuple(progression.EVALUATION_MODES), default="raw")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=200, help="Maximum policy steps to render.")
    parser.add_argument("--fps", type=float, default=6.0, help="Frames per second; 0 renders as fast as it can.")
    parser.add_argument("--vehicles", type=int, default=None)
    parser.add_argument("--ahead-m", type=float, default=DEFAULT_AHEAD_M)
    parser.add_argument("--behind-m", type=float, default=DEFAULT_BEHIND_M)
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument(
        "--scroll",
        action="store_true",
        help="Print frames one after another instead of redrawing in place.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Render every Nth policy step; the steps in between still execute.",
    )
    parser.add_argument("--stop-on-collision", action="store_true")
    add_reward_variant_arguments(parser)
    arguments = parser.parse_args()

    project_root = progression.protocol.find_project_root(
        arguments.project_root or Path.cwd()
    ).resolve()
    run_dir = (
        arguments.run_dir if arguments.run_dir.is_absolute() else project_root / arguments.run_dir
    ).resolve()
    run_config, study_config = load_source_run(run_dir)
    env_config, max_policy_steps, changes = apply_env_overrides(
        run_config["env_config"],
        max_policy_steps=int(study_config["evaluation_task_max_policy_steps"]),
        vehicles=arguments.vehicles,
    )
    reward_config = copy.deepcopy(run_config["reward_config"])
    apply_reward_variant_arguments(arguments, reward_config)
    variants = reward_variant_summary(reward_config)
    namespace = build_namespace(project_root, run_config)
    model_path = run_dir / "model_final.zip"
    model = progression.load_model(run_config["variant"], model_path, device="cpu")

    env = progression.make_evaluation_env(
        namespace,
        mode=str(arguments.mode),
        env_config=env_config,
        reward_config=reward_config,
        correction_epsilon=float(run_config["training_signature"]["correction_epsilon"]),
        task_distance_m=float(study_config["evaluation_task_distance_m"]),
        task_max_policy_steps=int(max_policy_steps),
    )
    header = (
        f"{run_config['variant']}  seed {arguments.seed}  mode {arguments.mode}  "
        f"vehicles {env_config['vehicles_count']}  env changes {changes or 'none'}"
    )
    delay = 0.0 if float(arguments.fps) <= 0.0 else 1.0 / float(arguments.fps)
    episode_return = 0.0
    try:
        observation, _ = env.reset(seed=int(arguments.seed))
        for step in range(min(int(arguments.steps), int(max_policy_steps))):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            info = dict(info)
            episode_return += float(reward)
            if step % max(1, int(arguments.every)) == 0 or terminated or truncated:
                lines = [header]
                lines.extend(
                    status_lines(
                        step=step,
                        reward=float(reward),
                        episode_return=episode_return,
                        info=info,
                        mode=str(arguments.mode),
                        variants=variants,
                    )
                )
                lines.extend(build_frame(env, info=info, arguments=arguments))
                print(("" if arguments.scroll else CLEAR_SCREEN) + "\n".join(lines), flush=True)
                if delay:
                    time.sleep(delay)
            collided = float(info.get("karalakou_ego_collision", 0.0)) > 0.5
            if terminated or truncated or (collided and arguments.stop_on_collision):
                reason = "collision" if collided else ("terminated" if terminated else "truncated")
                print(f"\nepisode ended after {step + 1} steps: {reason}", flush=True)
                break
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
