"""Open a live MTM simulator window with the current ``y_desired`` overlay.

The renderer intentionally does not create a video or import the training/CBF
pipeline.  It executes only the current reward-wrapper definition from the
notebook, then displays the simulator's RGB frame in a local pygame window.

Overlay legend:
    magenta line: current ``y_desired`` target used by the reward/observation
    cyan line: ego vehicle's current lateral position
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pygame


DEFAULT_CONFIG = Path("configs") / "current_mtm_live.json"
DEFAULT_NOTEBOOK = Path("notebooks") / "lanelessKaralakou.ipynb"


def find_project_root(start: Path) -> Path:
    """Find the inner project directory regardless of the launch directory."""

    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return candidate
        nested = candidate / "highway-rl-decision-making"
        if (nested / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return nested
    raise RuntimeError("Could not find the project root containing notebooks/lanelessKaralakou.ipynb")


def install_local_environment(project_root: Path) -> None:
    environment_root = project_root / "laneless highway env"
    for path in (environment_root / "HighwayEnv", environment_root):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    # Registration happens on import.
    import lane_free_env  # noqa: F401


def load_current_target_wrapper(notebook_path: Path) -> tuple[type, dict[str, Any]]:
    """Load only the notebook's current reward wrapper and reward defaults.

    This avoids importing torch/stable-baselines (and therefore keeps the live
    renderer independent from training processes) while preserving the exact
    target calculation used by the current notebook.
    """

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "Any": Any,
        "Optional": Optional,
        "gym": gym,
        "np": np,
    }
    wrapper_source: str | None = None
    reward_source: str | None = None
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "class KaralakouRewardWrapper" in source:
            wrapper_source = source
            wrapper_index = index
        if "REWARD_CONFIG = {" in source:
            reward_source = source
            reward_index = index
    if wrapper_source is None or reward_source is None:
        raise RuntimeError("Notebook is missing the current Karalakou wrapper or REWARD_CONFIG")
    exec(compile(wrapper_source, f"{notebook_path}:cell-{wrapper_index}", "exec"), namespace)
    # The configuration cell also contains later experiment setup that imports
    # training-only modules.  Execute only the literal REWARD_CONFIG assignment
    # so this viewer remains independent of torch/stable-baselines.
    reward_tree = ast.parse(reward_source, filename=f"{notebook_path}:cell-{reward_index}")
    reward_nodes: list[ast.stmt] = []
    for node in reward_tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(target, ast.Name) and target.id == "REWARD_CONFIG" for target in targets):
            reward_nodes.append(node)
            break
    if not reward_nodes:
        raise RuntimeError("Notebook configuration cell has no REWARD_CONFIG assignment")
    reward_module = ast.Module(body=reward_nodes, type_ignores=[])
    exec(compile(reward_module, f"{notebook_path}:cell-{reward_index}", "exec"), namespace)
    return namespace["KaralakouRewardWrapper"], copy.deepcopy(namespace["REWARD_CONFIG"])


def read_env_config(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    config = document.get("env_config")
    if not isinstance(config, dict):
        raise RuntimeError(f"No env_config object found in {path}")
    return copy.deepcopy(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=1_100_000)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Maximum policy steps; 0 runs until the window is closed or q/Esc is pressed.",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Display refresh cap (not physics dt).")
    parser.add_argument("--vehicles", type=int, default=None)
    return parser.parse_args()


def target_state(wrapper: Any) -> tuple[float, float, bool]:
    target_y, target_speed, zone_found = wrapper._lateral_target_and_speed()
    return float(target_y), float(target_speed), bool(zone_found)


def draw_frame(
    display: pygame.Surface,
    frame: np.ndarray,
    wrapper: Any,
    target_y: float,
    target_speed: float,
    zone_found: bool,
    font: pygame.font.Font,
) -> None:
    """Draw one native RGB frame and map world-y coordinates exactly."""

    frame = np.asarray(frame, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"Expected an HxWx3 RGB frame, got {frame.shape}")
    image = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    display.blit(image, (0, 0))

    base = wrapper.base_env
    viewer = getattr(base, "viewer", None)
    sim_surface = getattr(viewer, "sim_surface", None)
    if sim_surface is None:
        raise RuntimeError("Simulator did not create its native rendering surface")
    width, height = display.get_size()

    # WorldSurface.pos2pix is the renderer's own transform.  Using it avoids
    # guessing the camera origin or whether the y-axis is inverted.
    target_px = int(sim_surface.pos2pix(0.0, target_y)[1])
    ego_y = float(base.vehicle.position[1])
    ego_px = int(sim_surface.pos2pix(0.0, ego_y)[1])
    road_top = int(sim_surface.pos2pix(0.0, 0.0)[1])
    road_bottom = int(sim_surface.pos2pix(0.0, float(base.config["road_width"]))[1])

    # Keep the lines visible even at the edge of the camera crop.
    if 0 <= target_px < height:
        pygame.draw.line(display, (255, 0, 255), (0, target_px), (width - 1, target_px), 3)
    if 0 <= ego_px < height:
        pygame.draw.line(display, (0, 255, 255), (0, ego_px), (width - 1, ego_px), 2)
    if 0 <= road_top < height:
        pygame.draw.line(display, (255, 255, 255), (0, road_top), (width - 1, road_top), 1)
    if 0 <= road_bottom < height:
        pygame.draw.line(display, (255, 255, 255), (0, road_bottom), (width - 1, road_bottom), 1)

    lines = [
        f"MAGENTA y_target = {target_y:5.2f} m   ({'free gap' if zone_found else 'no free gap / follow'})",
        f"CYAN ego y = {ego_y:5.2f} m    target speed = {target_speed:4.1f} m/s",
        "q / Esc / close window: stop    |    target is recomputed every policy step",
    ]
    text_surfaces = [font.render(line, True, color) for line, color in zip(lines, ((255, 0, 255), (0, 255, 255), (245, 245, 245)))]
    box_height = sum(surface.get_height() for surface in text_surfaces) + 10
    box_width = min(width - 10, max(surface.get_width() for surface in text_surfaces) + 12)
    pygame.draw.rect(display, (20, 20, 20), pygame.Rect(5, 5, box_width, box_height))
    y_cursor = 9
    for surface in text_surfaces:
        display.blit(surface, (10, y_cursor))
        y_cursor += surface.get_height()


def main() -> int:
    args = parse_args()
    if args.steps < 0:
        raise ValueError("--steps must be nonnegative")
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    project_root = find_project_root(args.project_root or Path.cwd())
    os.chdir(project_root)
    install_local_environment(project_root)
    notebook_path = project_root / DEFAULT_NOTEBOOK
    wrapper_class, reward_config = load_current_target_wrapper(notebook_path)

    config_path = args.config if args.config.is_absolute() else project_root / args.config
    env_config = read_env_config(config_path.resolve())
    if str(env_config.get("traffic_model", "")).strip().lower() != "mtm":
        raise RuntimeError("The y_target visualizer requires traffic_model='mtm'.")
    if args.vehicles is not None:
        if args.vehicles <= 0:
            raise ValueError("--vehicles must be positive")
        env_config["vehicles_count"] = int(args.vehicles)
    env_config["ego_controlled"] = False
    env_config["terminate_on_collision"] = False
    env_config["offscreen_rendering"] = True
    env_config["real_time_rendering"] = False

    raw_env = gym.make("lane-free-v0", render_mode="rgb_array", config=env_config)
    env = wrapper_class(raw_env, reward_config=reward_config)
    action = np.zeros(2, dtype=np.float32)

    pygame.init()
    display: pygame.Surface | None = None
    font = pygame.font.Font(None, 18)
    clock = pygame.time.Clock()
    total_steps = 0
    episode = 0
    closed = False
    try:
        env.reset(seed=int(args.seed))
        target_y, target_speed, zone_found = target_state(env)
        frame = env.render()
        if frame is None:
            raise RuntimeError("Simulator returned no RGB frame")
        first = np.asarray(frame)
        display = pygame.display.set_mode((int(first.shape[1]), int(first.shape[0])))
        pygame.display.set_caption("MTM live simulator — y_target overlay")
        print(
            "[mtm-y-target-live] local window active; no video writer is enabled. "
            "Magenta=target y, cyan=ego y; press q/Esc to stop.",
            flush=True,
        )
        print(f"[mtm-y-target-live] config: {config_path.resolve()}", flush=True)

        while not closed and (args.steps == 0 or total_steps < args.steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE)
                ):
                    closed = True
                    break
            if closed:
                break

            _, _, terminated, truncated, info = env.step(action)
            total_steps += 1
            target_y = float(info.get("karalakou_target_y", target_y))
            target_speed = float(info.get("karalakou_target_speed", target_speed))
            zone_found = bool(float(info.get("karalakou_zone_found", float(zone_found))))
            frame = env.render()
            if frame is None:
                raise RuntimeError("Simulator returned no RGB frame")
            draw_frame(display, frame, env, target_y, target_speed, zone_found, font)
            pygame.display.flip()
            clock.tick(float(args.fps))

            if total_steps % 100 == 0:
                print(
                    f"[mtm-y-target-live] step={total_steps:,} ego_y={env.base_env.vehicle.position[1]:.2f} "
                    f"target_y={target_y:.2f} target_speed={target_speed:.2f} "
                    f"zone_found={zone_found}",
                    flush=True,
                )
            if terminated or truncated:
                episode += 1
                env.reset(seed=int(args.seed) + episode)
                target_y, target_speed, zone_found = target_state(env)
    except KeyboardInterrupt:
        print("[mtm-y-target-live] stopped by Ctrl+C.", flush=True)
    finally:
        env.close()
        pygame.quit()
    print(f"[mtm-y-target-live] closed after {total_steps:,} policy steps; no video was written.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
