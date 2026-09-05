from __future__ import annotations

import argparse
import copy
import io
import json
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import DDPG

from scripts.common.guided_cbf_minimal import install_minimal_guided_cbf

warnings.filterwarnings("ignore", message="OSQP exited.*")


POLICY_COLORS = {
    "baseline": "#2563eb",
    "ddpg_cbf_reward": "#059669",
    "guided_ddpg_cbf": "#c2410c",
}

CBF_EFFECT_COLORS = {
    "baseline_rl": "#2563eb",
    "cbf_actor_raw": "#7c3aed",
    "rl_plus_cbf": "#dc4a0a",
}

HIGHWAY_FRAME_CONFIG = {
    "screen_width": 1200,
    "screen_height": 320,
    "centering_position": [0.5, 0.5],
    "scaling": 9.8,
    "offscreen_rendering": True,
    "real_time_rendering": False,
}

HIGHWAY_RGB_COLORS = {
    "ego": (50, 200, 0),
    "traffic": (100, 200, 255),
}


@dataclass(frozen=True)
class VehicleSpec:
    label: str
    dx: float
    y: float
    vx: float
    vy: float = 0.0
    length: float = 3.5
    width: float = 1.8
    role: str = "traffic"
    desired_speed: float | None = None


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    title: str
    expected: str
    vehicles: list[VehicleSpec]
    bands: list[tuple[float, float, str, str]] = field(default_factory=list)
    xlim: tuple[float, float] = (-28.0, 72.0)


@dataclass(frozen=True)
class PolicySpec:
    key: str
    label: str
    model_key: str
    env_kind: str


POLICIES = [
    PolicySpec("baseline", "Baseline DDPG", "DDPG_MODEL_PATH", "baseline"),
    PolicySpec("ddpg_cbf_reward", "DDPG-CBF reward", "DDPG_CBF_MODEL_PATH", "cbf"),
    PolicySpec("guided_ddpg_cbf", "DDPG-CBF reward + actor loss", "GUIDED_DDPG_CBF_MODEL_PATH", "guided"),
]


def find_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    for candidate in [script_path.parent, *script_path.parents]:
        if (candidate / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return candidate
    raise RuntimeError("Could not find repo root containing notebooks/lanelessKaralakou.ipynb")


def exec_notebook_cell(notebook: dict[str, Any], notebook_path: Path, prefix: str, namespace: dict[str, Any]) -> None:
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.startswith(prefix):
            with redirect_stdout(io.StringIO()):
                exec(compile(source, f"{notebook_path}:cell_{index}", "exec"), namespace)
            return
    raise RuntimeError(f"Could not find notebook cell starting with {prefix!r}")


def load_notebook_namespace(repo_root: Path) -> dict[str, Any]:
    notebook_path = repo_root / "notebooks" / "lanelessKaralakou.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {}

    for prefix in [
        "from __future__ import annotations",
        "class KaralakouRewardWrapper",
        "ENV_CONFIG = {",
        "class LaneFreeObservationNormalizationWrapper",
        "try:\n    from qpsolvers import solve_qp",
        "CBF_AX_BOUNDS =",
        "def _lane_free_base",
        "class SafetyFilteredAccelerationWrapper",
        "# Tuned DDPG-CBF shield overrides",
    ]:
        exec_notebook_cell(notebook, notebook_path, prefix, namespace)

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.startswith("# Guided DDPG-CBF actor update"):
            definitions = source.split("\nguided_ddpg_cbf_train_env = ", 1)[0]
            with redirect_stdout(io.StringIO()):
                exec(compile(definitions, f"{notebook_path}:cell_{index}_defs_only", "exec"), namespace)
            break
    else:
        raise RuntimeError("Could not find guided DDPG-CBF actor update cell")

    install_minimal_guided_cbf(namespace)
    return namespace


def make_scenarios(road_width: float) -> list[ScenarioSpec]:
    center = 0.5 * road_width
    ego = VehicleSpec("ego", 0.0, center, 20.0, role="ego", length=3.5, width=1.8, desired_speed=20.0)
    return [
        ScenarioSpec(
            name="open_road_no_neighbors",
            title="Open Road: No Nearby Traffic",
            expected="With no nearby vehicles, a reasonable policy should keep steady progress with little lateral drift and near-zero CBF correction.",
            bands=[(center - 0.8, center + 0.8, "#16a34a", "nominal center band")],
            vehicles=[ego],
            xlim=(-24.0, 76.0),
        ),
        ScenarioSpec(
            name="safe_overtake_open_upper_gap",
            title="Open Passing Gap: Overtake Diagnostic",
            expected="A passing gap is available; compare which policies actually commit to it.",
            bands=[(7.1, 9.5, "#16a34a", "open upper pass"), (2.0, 4.0, "#dc2626", "blocked lower slot")],
            vehicles=[
                ego,
                VehicleSpec("slow blocker", 24.0, center, 13.5, role="blocker"),
                VehicleSpec("lower car", 13.0, 2.9, 18.5, role="hazard"),
                VehicleSpec("upper leader", 56.0, 8.3, 21.0, role="traffic"),
                VehicleSpec("upper rear", -28.0, 8.2, 18.0, role="traffic"),
                VehicleSpec("far lead", 58.0, 5.5, 20.5, role="traffic"),
            ],
        ),
        ScenarioSpec(
            name="unsafe_overtake_fast_closing_upper",
            title="Reject Overtake: Fast Closing Vehicle",
            expected="Do not enter the closing gap; move away while keeping separation.",
            bands=[(7.0, 9.4, "#dc2626", "closing upper gap"), (4.2, 6.0, "#f97316", "slow blocker band")],
            vehicles=[
                ego,
                VehicleSpec("slow blocker", 22.0, center, 13.0, role="blocker"),
                VehicleSpec("fast closer", -14.0, 8.0, 30.0, role="hazard"),
                VehicleSpec("upper front", 31.0, 8.2, 22.0, role="hazard"),
                VehicleSpec("lower car", 12.0, 2.8, 19.0, role="hazard"),
                VehicleSpec("far lead", 56.0, 5.5, 20.0, role="traffic"),
            ],
        ),
        ScenarioSpec(
            name="narrow_gap_wait_or_upper_escape",
            title="Tight-Slot Trap: Shield Correction",
            expected="Avoid diving into the tight slot; the shield should redirect the action.",
            bands=[(7.3, 9.6, "#16a34a", "usable upper corridor"), (2.1, 4.3, "#dc2626", "lower slot trap")],
            vehicles=[
                ego,
                VehicleSpec("slow blocker", 20.0, center, 12.5, role="blocker"),
                VehicleSpec("lower front", 18.0, 3.1, 18.0, role="hazard"),
                VehicleSpec("lower rear", -8.0, 3.0, 23.0, role="hazard"),
                VehicleSpec("upper front", 48.0, 8.2, 20.5, role="traffic"),
                VehicleSpec("far center", 58.0, 5.2, 19.0, role="traffic"),
            ],
        ),
        ScenarioSpec(
            name="boundary_recovery_no_upper_squeeze",
            title="Boundary Recovery: Road-Edge Squeeze",
            expected="Recover inward from the road edge and avoid the slow blocker.",
            bands=[(8.2, 10.2, "#dc2626", "upper boundary squeeze"), (4.6, 6.8, "#16a34a", "inward recovery band")],
            vehicles=[
                VehicleSpec("ego", 0.0, 8.7, 20.0, role="ego", length=3.5, width=1.8, desired_speed=20.0),
                VehicleSpec("upper blocker", 18.0, 8.8, 13.0, role="blocker"),
                VehicleSpec("center car", 12.0, 6.2, 18.0, role="hazard"),
                VehicleSpec("center rear", -16.0, 6.0, 20.0, role="traffic"),
                VehicleSpec("lower open lead", 48.0, 3.0, 21.0, role="traffic"),
                VehicleSpec("far lead", 56.0, 8.4, 20.0, role="traffic"),
            ],
            xlim=(-24.0, 66.0),
        ),
        ScenarioSpec(
            name="opposite_edge_recovery",
            title="Boundary Recovery: Opposite Edge",
            expected="Recover inward from the opposite road edge and avoid the slow blocker.",
            bands=[(0.0, 2.0, "#dc2626", "road-edge squeeze"), (4.2, 6.6, "#16a34a", "inward recovery band")],
            vehicles=[
                VehicleSpec("ego", 0.0, 1.45, 20.0, role="ego", length=3.5, width=1.8, desired_speed=20.0),
                VehicleSpec("edge blocker", 17.0, 1.5, 13.0, role="blocker"),
                VehicleSpec("center car", 12.0, 4.2, 18.0, role="hazard"),
                VehicleSpec("center rear", -15.0, 4.0, 20.0, role="traffic"),
                VehicleSpec("open lead", 46.0, 7.0, 21.0, role="traffic"),
                VehicleSpec("far lead", 56.0, 1.8, 20.0, role="traffic"),
            ],
        ),
        ScenarioSpec(
            name="boxed_in_hold_position",
            title="Boxed In: No Clean Gap",
            expected="Avoid forcing a pass when both nearby side gaps are occupied.",
            bands=[(1.9, 4.1, "#dc2626", "occupied side gap"), (6.2, 8.5, "#dc2626", "occupied side gap")],
            vehicles=[
                ego,
                VehicleSpec("front blocker", 15.0, center, 13.0, role="blocker"),
                VehicleSpec("rear pressure", -11.0, center, 24.0, role="hazard"),
                VehicleSpec("side front A", 10.0, 3.0, 18.0, role="hazard"),
                VehicleSpec("side rear A", -8.0, 3.1, 23.0, role="hazard"),
                VehicleSpec("side front B", 11.0, 8.0, 18.0, role="hazard"),
                VehicleSpec("side rear B", -9.0, 8.1, 23.0, role="hazard"),
            ],
        ),
        ScenarioSpec(
            name="rear_pressure_escape",
            title="Rear Pressure: Escape Closing Car",
            expected="A fast rear vehicle closes on ego; move toward the cleaner side gap.",
            bands=[(6.8, 9.4, "#16a34a", "clean escape gap"), (2.0, 4.2, "#dc2626", "blocked side gap")],
            vehicles=[
                ego,
                VehicleSpec("fast rear", -12.0, center, 31.0, role="hazard"),
                VehicleSpec("slow lead", 23.0, center, 14.0, role="blocker"),
                VehicleSpec("blocked side", 10.0, 3.0, 19.0, role="hazard"),
                VehicleSpec("open lead", 46.0, 8.2, 21.5, role="traffic"),
                VehicleSpec("open rear", -30.0, 8.2, 17.0, role="traffic"),
            ],
        ),
        ScenarioSpec(
            name="sudden_lead_slowdown",
            title="Sudden Lead Slowdown: Reactive Safety",
            expected="React to a sharply slower leader without diving into occupied side gaps.",
            bands=[(1.9, 4.0, "#dc2626", "occupied lower gap"), (6.7, 9.2, "#f97316", "partly closing upper gap")],
            vehicles=[
                ego,
                VehicleSpec("hard-braking lead", 15.0, center, 8.0, role="blocker", desired_speed=8.0),
                VehicleSpec("upper closer", -10.0, 8.1, 27.0, role="hazard"),
                VehicleSpec("upper front", 25.0, 8.0, 18.0, role="hazard"),
                VehicleSpec("lower side", 8.0, 3.0, 18.5, role="hazard"),
                VehicleSpec("lower rear", -14.0, 3.1, 23.0, role="hazard"),
                VehicleSpec("far center", 48.0, center, 18.0, role="traffic"),
            ],
        ),
        ScenarioSpec(
            name="staggered_gap_selection",
            title="Staggered Traffic: Pick The Clean Gap",
            expected="Thread through staggered traffic by avoiding the closer blocked side.",
            bands=[(1.8, 4.1, "#dc2626", "early blocked side"), (6.7, 9.3, "#16a34a", "later clean gap")],
            vehicles=[
                ego,
                VehicleSpec("near side", 9.0, 3.0, 18.5, role="hazard"),
                VehicleSpec("center blocker", 22.0, center, 13.0, role="blocker"),
                VehicleSpec("far side", 34.0, 8.1, 20.0, role="traffic"),
                VehicleSpec("rear side", -18.0, 2.7, 22.0, role="hazard"),
                VehicleSpec("open lead", 52.0, 5.4, 21.0, role="traffic"),
            ],
        ),
    ]


def scenario_env_config(namespace: dict[str, Any], scenario: ScenarioSpec) -> dict[str, Any]:
    config = copy.deepcopy(namespace["ENV_CONFIG"])
    config["vehicles_count"] = len(scenario.vehicles)
    config["neighbors_count"] = 5
    config["episode_steps"] = 120
    config["duration"] = 120
    config["terminate_on_collision"] = False
    config["real_time_rendering"] = False
    config.update(HIGHWAY_FRAME_CONFIG)
    return config


def make_policy_env(namespace: dict[str, Any], scenario: ScenarioSpec, policy: PolicySpec, seed: int):
    env_config = scenario_env_config(namespace, scenario)
    if policy.env_kind == "baseline":
        return namespace["make_single_env"](seed=seed, render_mode=None, env_config=env_config)
    if policy.env_kind == "cbf":
        return namespace["make_cbf_single_env"](
            seed=seed,
            render_mode=None,
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
            env_config=env_config,
        )
    if policy.env_kind == "guided":
        return namespace["make_guided_cbf_single_env"](
            seed=seed,
            render_mode=None,
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
            env_config=env_config,
        )
    raise ValueError(f"Unknown policy env kind: {policy.env_kind}")


def apply_scenario(env, scenario: ScenarioSpec, ego_x: float = 100.0) -> np.ndarray:
    base = env.unwrapped
    road_length = float(base.config["road_length"])
    vehicles = list(base.road.vehicles)
    if len(vehicles) < len(scenario.vehicles):
        raise RuntimeError(f"Environment has {len(vehicles)} vehicles but scenario needs {len(scenario.vehicles)}")

    for index, (vehicle, spec) in enumerate(zip(vehicles, scenario.vehicles)):
        vehicle.position[0] = float((ego_x + spec.dx) % road_length)
        vehicle.position[1] = float(spec.y)
        vehicle.vx = float(spec.vx)
        vehicle.vy = float(spec.vy)
        vehicle.length = float(spec.length)
        vehicle.width = float(spec.width)
        vehicle.LENGTH = vehicle.length
        vehicle.WIDTH = vehicle.width
        vehicle.desired_speed = float(spec.desired_speed if spec.desired_speed is not None else max(spec.vx, 1.0))
        vehicle.is_ego = index == 0
        vehicle.crashed = False
        vehicle.hit = False
        vehicle._sync_graphics_fields()

    base.vehicle = vehicles[0]
    base.controlled_vehicles = [vehicles[0]]
    base._last_action = np.zeros(2, dtype=np.float32)
    base._last_accelerations = np.zeros((len(vehicles), 2), dtype=float)
    base._last_boundary_violations = 0
    base._last_collision_count = 0
    base._last_active_collision_count = 0
    base._last_ego_collision_count = 0
    base._last_ego_collision = False
    base._cumulative_collision_count = 0
    base._active_collision_pairs = set()
    base._flow_count = 0
    return policy_observation(env)


def policy_observation(env) -> np.ndarray:
    base = env.unwrapped
    obs = base._observe()
    current = env
    augmenters = []
    while current is not None:
        if hasattr(current, "_augment_observation"):
            augmenters.append(current)
        current = getattr(current, "env", None)
    for wrapper in reversed(augmenters):
        obs = wrapper._augment_observation(obs)
    return np.asarray(obs, dtype=np.float32)


def normalized_to_physical(env, action: np.ndarray) -> np.ndarray:
    base = env.unwrapped
    action = np.asarray(action, dtype=float).reshape(-1)[:2]
    return np.asarray(
        [
            base._map_action(float(action[0]), "longitudinal"),
            base._map_action(float(action[1]), "lateral"),
        ],
        dtype=np.float32,
    )


def load_models(namespace: dict[str, Any]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for policy in POLICIES:
        model_path = Path(namespace[policy.model_key])
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model for {policy.label}: {model_path}")
        model_cls = namespace["GuidedCBFDDPG"] if policy.key == "guided_ddpg_cbf" else DDPG
        models[policy.key] = model_cls.load(str(model_path), device=namespace["DEVICE"])
    return models


def initial_action_audit(namespace: dict[str, Any], model: Any, policy: PolicySpec, scenario: ScenarioSpec, seed: int):
    env = make_policy_env(namespace, scenario, policy, seed)
    env.reset(seed=seed)
    obs = apply_scenario(env, scenario)
    action, _ = model.predict(obs, deterministic=True)
    action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]

    if policy.env_kind == "baseline":
        raw_phys = normalized_to_physical(env, action)
        safe_phys = raw_phys.copy()
        filter_info = {
            "correction_norm": 0.0,
            "qp_success": True,
            "fallback_used": False,
            "min_h": np.nan,
            "max_constraint_violation_safe": np.nan,
        }
    else:
        raw_phys = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
        base = env.unwrapped
        safe_phys, filter_info = namespace["cbf_filter_2d"](
            raw_phys,
            namespace["get_ego_state"](env),
            namespace["get_neighbor_states"](env),
            float(base.config["road_width"]),
            ax_bounds=namespace["CBF_AX_BOUNDS"],
            ay_bounds=namespace["CBF_AY_BOUNDS"],
            eps_side=namespace["CBF_EPS_SIDE"],
            k0=namespace["CBF_K0"],
            k1=namespace["CBF_K1"],
        )
        safe_phys = np.asarray(safe_phys, dtype=np.float32).reshape(-1)[:2]

    env.close()
    return {
        "raw_action": raw_phys,
        "safe_action": safe_phys,
        "correction_norm": float(filter_info.get("correction_norm", np.linalg.norm(safe_phys - raw_phys))),
        "qp_success": bool(filter_info.get("qp_success", True)),
        "fallback_used": bool(filter_info.get("fallback_used", not bool(filter_info.get("qp_success", True)))),
        "min_h": float(filter_info.get("min_h", np.nan)),
        "max_constraint_violation_safe": float(filter_info.get("max_constraint_violation_safe", np.nan)),
    }


def rollout_policy(namespace: dict[str, Any], model: Any, policy: PolicySpec, scenario: ScenarioSpec, seed: int, steps: int):
    env = make_policy_env(namespace, scenario, policy, seed)
    env.reset(seed=seed)
    obs = apply_scenario(env, scenario)
    base = env.unwrapped
    ego_x0 = float(base.vehicle.position[0])
    path = [(0.0, float(base.vehicle.position[1]))]
    ego_collisions = 0
    min_h_values: list[float] = []
    total_corrections: list[float] = []

    for _ in range(int(steps)):
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        obs, _, terminated, truncated, info = env.step(action)
        base = env.unwrapped
        dx = float(base._signed_distance(ego_x0, base.vehicle.position[0]))
        path.append((dx, float(base.vehicle.position[1])))
        ego_collisions += int(info.get("ego_collision_events", 0))
        if "cbf_min_h" in info:
            min_h_values.append(float(info.get("cbf_min_h", np.nan)))
        if "cbf_correction_norm" in info:
            total_corrections.append(float(info.get("cbf_correction_norm", 0.0)))
        if terminated or truncated:
            break

    env.close()
    return {
        "path": np.asarray(path, dtype=float),
        "ego_collisions": float(ego_collisions),
        "rollout_min_h": float(np.nanmin(min_h_values)) if min_h_values and not np.all(np.isnan(min_h_values)) else np.nan,
        "rollout_mean_correction": float(np.mean(total_corrections)) if total_corrections else 0.0,
    }


def rgb_to_hex(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ["arialbd.ttf", "segoeuib.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def apply_highway_vehicle_colors(env, scenario: ScenarioSpec) -> None:
    for vehicle, spec in zip(env.unwrapped.road.vehicles, scenario.vehicles):
        vehicle.color = HIGHWAY_RGB_COLORS["ego"] if spec.role == "ego" else HIGHWAY_RGB_COLORS["traffic"]


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    x, y = xy
    pad_x, pad_y = 10, 5
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rounded_rectangle(
        (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y),
        radius=10,
        fill=(20, 20, 20),
        outline=fill,
        width=2,
    )
    draw.text((x, y), text, font=font, fill=(255, 255, 255))


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    *,
    width: int = 10,
) -> None:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    norm = float(np.hypot(dx, dy))
    if norm < 1.0:
        return

    ux, uy = dx / norm, dy / norm
    head_len = 22.0
    head_w = 14.0
    base_x = ex - head_len * ux
    base_y = ey - head_len * uy
    perp_x, perp_y = -uy, ux
    head = [
        (ex, ey),
        (base_x + 0.5 * head_w * perp_x, base_y + 0.5 * head_w * perp_y),
        (base_x - 0.5 * head_w * perp_x, base_y - 0.5 * head_w * perp_y),
    ]

    outline = (18, 18, 18)
    draw.line((sx, sy, base_x, base_y), fill=outline, width=width + 6)
    draw.polygon(head, fill=outline)
    draw.line((sx, sy, base_x, base_y), fill=color, width=width)
    draw.polygon(head, fill=color)


def action_endpoint(start: tuple[int, int], action: np.ndarray, offset: tuple[float, float]) -> tuple[float, float]:
    scale = 17.0
    return (
        float(start[0] + offset[0] + scale * float(action[0])),
        float(start[1] + offset[1] + scale * float(action[1])),
    )


def draw_cbf_effect_legend(draw: ImageDraw.ImageDraw, image: Image.Image, footer_h: int) -> None:
    legend_font = load_font(22, bold=True)
    legend_items = [
        ("Baseline RL", "baseline_rl", "solid"),
        ("CBF RL only", "cbf_actor_raw", "dashed"),
        ("RL + CBF", "rl_plus_cbf", "solid"),
    ]
    x = 24
    y = image.height - footer_h + 16
    for text, key, line_style in legend_items:
        color = hex_to_rgb(CBF_EFFECT_COLORS[key])
        if line_style == "dashed":
            draw.line((x, y + 10, x + 28, y + 10), fill=color, width=7)
            draw.line((x + 36, y + 10, x + 64, y + 10), fill=color, width=7)
            text_x = x + 78
            step = 270
        else:
            draw.rounded_rectangle((x, y - 2, x + 26, y + 22), radius=5, fill=color)
            text_x = x + 38
            step = 245 if key == "baseline_rl" else 220
        draw.text((text_x, y - 5), text, font=legend_font, fill=(245, 245, 245))
        x += step


def draw_dashed_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    *,
    width: int = 8,
) -> None:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    norm = float(np.hypot(dx, dy))
    if norm < 1.0:
        return

    ux, uy = dx / norm, dy / norm
    dash_len = 13.0
    gap_len = 8.0
    cursor = 0.0
    outline = (18, 18, 18)
    while cursor < max(norm - 20.0, 0.0):
        seg_start = cursor
        seg_end = min(cursor + dash_len, norm - 20.0)
        p0 = (sx + ux * seg_start, sy + uy * seg_start)
        p1 = (sx + ux * seg_end, sy + uy * seg_end)
        draw.line((*p0, *p1), fill=outline, width=width + 5)
        draw.line((*p0, *p1), fill=color, width=width)
        cursor += dash_len + gap_len

    head_len = 22.0
    head_w = 14.0
    base_x = ex - head_len * ux
    base_y = ey - head_len * uy
    perp_x, perp_y = -uy, ux
    head = [
        (ex, ey),
        (base_x + 0.5 * head_w * perp_x, base_y + 0.5 * head_w * perp_y),
        (base_x - 0.5 * head_w * perp_x, base_y - 0.5 * head_w * perp_y),
    ]
    draw.polygon(head, fill=outline)
    draw.polygon(head, fill=color)


def cbf_effect_endpoint(
    start: tuple[int, int],
    action: np.ndarray,
    offset: tuple[float, float],
) -> tuple[float, float]:
    scale = 16.0
    return (
        float(start[0] + offset[0] + scale * float(action[0])),
        float(start[1] + offset[1] + scale * float(action[1])),
    )


def plot_cbf_effect_frame(
    output_path: Path,
    namespace: dict[str, Any],
    scenario: ScenarioSpec,
    audits: dict[str, dict[str, Any]],
    seed: int,
) -> None:
    env = make_policy_env(namespace, scenario, POLICIES[0], seed)
    env.unwrapped.render_mode = "rgb_array"
    env.unwrapped.config["offscreen_rendering"] = True
    env.reset(seed=seed)
    apply_scenario(env, scenario)
    apply_highway_vehicle_colors(env, scenario)

    frame = env.render()
    if frame is None:
        env.close()
        raise RuntimeError("highway-env render returned None; expected an rgb_array frame")

    surface = env.unwrapped.viewer.sim_surface
    ego_px = surface.vec2pix(env.unwrapped.vehicle.position)
    env.close()

    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    title_font = load_font(33, bold=True)

    header_h = 76
    draw.rectangle((0, 0, image.width, header_h), fill=(35, 35, 35))
    draw.text((24, 13), f"{scenario.title}: CBF Effect", font=title_font, fill=(245, 245, 245))

    footer_h = 58
    draw.rectangle((0, image.height - footer_h, image.width, image.height), fill=(35, 35, 35))
    draw_cbf_effect_legend(draw, image, footer_h)

    baseline_action = np.asarray(audits["baseline"]["safe_action"], dtype=float)
    cbf_raw_action = np.asarray(audits["ddpg_cbf_reward"]["raw_action"], dtype=float)
    cbf_safe_action = np.asarray(audits["ddpg_cbf_reward"]["safe_action"], dtype=float)

    arrow_specs = [
        ("baseline_rl", baseline_action, (-9.0, -8.0), "solid"),
        ("cbf_actor_raw", cbf_raw_action, (1.0, 5.0), "dashed"),
        ("rl_plus_cbf", cbf_safe_action, (10.0, 18.0), "solid"),
    ]
    for key, action, offset, style in arrow_specs:
        color = hex_to_rgb(CBF_EFFECT_COLORS[key])
        start = (ego_px[0] + offset[0], ego_px[1] + offset[1])
        end = cbf_effect_endpoint(ego_px, action, offset)
        if style == "dashed":
            draw_dashed_arrow(draw, start, end, color, width=8)
        else:
            draw_arrow(draw, start, end, color, width=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def plot_highway_comparison_frame(
    output_path: Path,
    namespace: dict[str, Any],
    scenario: ScenarioSpec,
    audits: dict[str, dict[str, Any]],
    seed: int,
) -> None:
    env = make_policy_env(namespace, scenario, POLICIES[0], seed)
    env.unwrapped.render_mode = "rgb_array"
    env.unwrapped.config["offscreen_rendering"] = True
    env.reset(seed=seed)
    apply_scenario(env, scenario)
    apply_highway_vehicle_colors(env, scenario)

    frame = env.render()
    if frame is None:
        env.close()
        raise RuntimeError("highway-env render returned None; expected an rgb_array frame")

    surface = env.unwrapped.viewer.sim_surface
    ego_px = surface.vec2pix(env.unwrapped.vehicle.position)
    env.close()

    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    title_font = load_font(34, bold=True)
    legend_font = load_font(23, bold=True)

    header_h = 74
    draw.rectangle((0, 0, image.width, header_h), fill=(35, 35, 35))
    title = scenario.title
    draw.text((24, 14), title, font=title_font, fill=(245, 245, 245))

    footer_h = 54
    draw.rectangle((0, image.height - footer_h, image.width, image.height), fill=(35, 35, 35))
    legend_x = 24
    legend_items = [
        ("Baseline", "baseline"),
        ("CBF reward", "ddpg_cbf_reward"),
        ("CBF + actor loss", "guided_ddpg_cbf"),
    ]
    x = legend_x
    for text, key in legend_items:
        color = hex_to_rgb(POLICY_COLORS[key])
        y = image.height - footer_h + 16
        draw.rounded_rectangle((x, y - 2, x + 24, y + 22), radius=5, fill=color)
        draw.text((x + 36, y - 5), text, font=legend_font, fill=(245, 245, 245))
        x += 225 if key != "guided_ddpg_cbf" else 300

    arrow_offsets = {
        "baseline": (-10.0, -6.0),
        "ddpg_cbf_reward": (0.0, 6.0),
        "guided_ddpg_cbf": (10.0, 18.0),
    }

    for policy in POLICIES:
        action = np.asarray(audits[policy.key]["safe_action"], dtype=float)
        color = hex_to_rgb(POLICY_COLORS[policy.key])
        offset = arrow_offsets[policy.key]
        start = (ego_px[0] + offset[0], ego_px[1] + offset[1])
        end = action_endpoint(ego_px, action, offset)
        draw_arrow(draw, start, end, color, width=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def vehicle_color(spec: VehicleSpec) -> str:
    return {
        "ego": "#2563eb",
        "blocker": "#f59e0b",
        "hazard": "#dc2626",
        "traffic": "#64748b",
    }.get(spec.role, "#64748b")


def draw_scene(ax, scenario: ScenarioSpec, road_width: float, *, labels: bool = True) -> None:
    ax.set_facecolor("#f8fafc")
    ax.axhspan(0.0, road_width, color="#e5e7eb", alpha=0.85, zorder=0)
    ax.axhline(0.0, color="#111827", lw=1.5)
    ax.axhline(road_width, color="#111827", lw=1.5)
    for lo, hi, color, label in scenario.bands:
        ax.axhspan(lo, hi, color=color, alpha=0.11, zorder=0)
        if labels:
            ax.text(
                scenario.xlim[0] + 1.0,
                0.5 * (lo + hi),
                label,
                va="center",
                ha="left",
                fontsize=7,
                color=color,
                alpha=0.9,
            )

    for spec in scenario.vehicles:
        rect = Rectangle(
            (spec.dx - 0.5 * spec.length, spec.y - 0.5 * spec.width),
            spec.length,
            spec.width,
            facecolor=vehicle_color(spec),
            edgecolor="#111827",
            linewidth=0.8,
            alpha=0.96,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.arrow(
            spec.dx - 0.2 * spec.length,
            spec.y,
            min(max(spec.vx / 10.0, 1.0), 3.2),
            0.0,
            width=0.025,
            head_width=0.24,
            head_length=0.7,
            color="white",
            alpha=0.85,
            length_includes_head=True,
            zorder=4,
        )
        if labels:
            ax.text(spec.dx, spec.y + 0.5 * spec.width + 0.22, spec.label, ha="center", va="bottom", fontsize=7)

    ax.set_xlim(*scenario.xlim)
    ax.set_ylim(-0.25, road_width + 0.25)
    ax.set_xlabel("x relative to ego [m]", fontsize=8)
    ax.set_ylabel("lateral y [m]", fontsize=8)
    ax.grid(True, color="#cbd5e1", linewidth=0.5, alpha=0.5)
    ax.tick_params(labelsize=7)


def draw_action_arrows(ax, origin_y: float, audit: dict[str, Any], color: str, policy: PolicySpec) -> None:
    raw = np.asarray(audit["raw_action"], dtype=float)
    safe = np.asarray(audit["safe_action"], dtype=float)

    def endpoint(action: np.ndarray) -> tuple[float, float]:
        return float(2.2 * action[0]), float(origin_y + 0.85 * action[1])

    raw_end = endpoint(raw)
    safe_end = endpoint(safe)
    if policy.env_kind != "baseline":
        ax.annotate(
            "",
            xy=raw_end,
            xytext=(0.0, origin_y),
            arrowprops={"arrowstyle": "->", "lw": 1.9, "color": "#a855f7", "linestyle": "--"},
            zorder=6,
        )
    ax.annotate(
        "",
        xy=safe_end,
        xytext=(0.0, origin_y),
        arrowprops={"arrowstyle": "->", "lw": 2.4, "color": color},
        zorder=7,
    )


def plot_scenario(
    output_path: Path,
    scenario: ScenarioSpec,
    road_width: float,
    audits: dict[str, dict[str, Any]],
    rollouts: dict[str, dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, len(POLICIES), figsize=(16.5, 4.8), dpi=170, sharex=True, sharey=True)
    ego_y = scenario.vehicles[0].y

    for ax, policy in zip(axes, POLICIES):
        color = POLICY_COLORS[policy.key]
        draw_scene(ax, scenario, road_width, labels=True)
        path = rollouts[policy.key]["path"]
        ax.plot(path[:, 0], path[:, 1], color=color, lw=2.4, marker="o", markersize=2.5, zorder=5)
        draw_action_arrows(ax, ego_y, audits[policy.key], color, policy)

        audit = audits[policy.key]
        raw = audit["raw_action"]
        safe = audit["safe_action"]
        text = (
            f"raw a=({raw[0]:+.2f}, {raw[1]:+.2f}) m/s^2\n"
            f"exec a=({safe[0]:+.2f}, {safe[1]:+.2f}) m/s^2\n"
            f"correction={audit['correction_norm']:.2f}"
        )
        if policy.env_kind != "baseline":
            text += f" | QP={'ok' if audit['qp_success'] else 'fallback'}"
        if rollouts[policy.key]["ego_collisions"] > 0:
            text += f"\nego collisions={rollouts[policy.key]['ego_collisions']:.0f}"
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1", "alpha": 0.92},
        )
        ax.set_title(policy.label, fontsize=10, color=color, pad=8)

    fig.suptitle(f"{scenario.title}\nExpected: {scenario.expected}", fontsize=13, y=1.04)
    fig.text(
        0.5,
        0.02,
        "Solid arrow = executed physical acceleration. Dashed purple arrow = raw CBF-policy proposal before hard no-slack shield.",
        ha="center",
        fontsize=8,
        color="#334155",
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run(output_dir: Path, selected: set[str] | None, rollout_steps: int) -> tuple[list[Path], list[Path], Path, Path]:
    repo_root = find_repo_root()
    namespace = load_notebook_namespace(repo_root)
    models = load_models(namespace)
    road_width = float(namespace["ENV_CONFIG"]["road_width"])
    scenarios = make_scenarios(road_width)
    if selected:
        scenarios = [scenario for scenario in scenarios if scenario.name in selected]
    if not scenarios:
        raise RuntimeError("No scenarios selected")

    rows: list[dict[str, Any]] = []
    cbf_effect_rows: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    cbf_effect_paths: list[Path] = []
    for scenario_index, scenario in enumerate(scenarios):
        audits: dict[str, dict[str, Any]] = {}
        rollouts: dict[str, dict[str, Any]] = {}
        for policy_index, policy in enumerate(POLICIES):
            seed = int(namespace["SEED"]) + 10_000 * scenario_index + 100 * policy_index
            audits[policy.key] = initial_action_audit(namespace, models[policy.key], policy, scenario, seed)
            rollouts[policy.key] = rollout_policy(
                namespace,
                models[policy.key],
                policy,
                scenario,
                seed,
                steps=rollout_steps,
            )
            raw = audits[policy.key]["raw_action"]
            safe = audits[policy.key]["safe_action"]
            rows.append(
                {
                    "scenario": scenario.name,
                    "expected": scenario.expected,
                    "policy": policy.key,
                    "policy_label": policy.label,
                    "raw_ax": float(raw[0]),
                    "raw_ay": float(raw[1]),
                    "exec_ax": float(safe[0]),
                    "exec_ay": float(safe[1]),
                    "correction_norm": float(audits[policy.key]["correction_norm"]),
                    "qp_success": bool(audits[policy.key]["qp_success"]),
                    "fallback_used": bool(audits[policy.key]["fallback_used"]),
                    "initial_min_h": float(audits[policy.key]["min_h"]),
                    "rollout_min_h": float(rollouts[policy.key]["rollout_min_h"]),
                    "rollout_mean_correction": float(rollouts[policy.key]["rollout_mean_correction"]),
                    "rollout_ego_collisions": float(rollouts[policy.key]["ego_collisions"]),
                    "rollout_final_dx": float(rollouts[policy.key]["path"][-1, 0]),
                    "rollout_final_y": float(rollouts[policy.key]["path"][-1, 1]),
                }
            )

        baseline_action = np.asarray(audits["baseline"]["safe_action"], dtype=float)
        cbf_raw_action = np.asarray(audits["ddpg_cbf_reward"]["raw_action"], dtype=float)
        cbf_safe_action = np.asarray(audits["ddpg_cbf_reward"]["safe_action"], dtype=float)
        cbf_effect_rows.append(
            {
                "scenario": scenario.name,
                "expected": scenario.expected,
                "baseline_rl_ax": float(baseline_action[0]),
                "baseline_rl_ay": float(baseline_action[1]),
                "cbf_trained_rl_only_ax": float(cbf_raw_action[0]),
                "cbf_trained_rl_only_ay": float(cbf_raw_action[1]),
                "rl_plus_cbf_executed_ax": float(cbf_safe_action[0]),
                "rl_plus_cbf_executed_ay": float(cbf_safe_action[1]),
                "cbf_correction_norm": float(audits["ddpg_cbf_reward"]["correction_norm"]),
                "cbf_qp_success": bool(audits["ddpg_cbf_reward"]["qp_success"]),
                "cbf_fallback_used": bool(audits["ddpg_cbf_reward"]["fallback_used"]),
                "cbf_initial_min_h": float(audits["ddpg_cbf_reward"]["min_h"]),
                "cbf_max_constraint_violation_safe": float(
                    audits["ddpg_cbf_reward"]["max_constraint_violation_safe"]
                ),
            }
        )

        image_path = output_dir / f"{scenario.name}.png"
        plot_highway_comparison_frame(
            image_path,
            namespace,
            scenario,
            audits,
            seed=int(namespace["SEED"]) + 10_000 * scenario_index,
        )
        image_paths.append(image_path)

        cbf_effect_path = output_dir / "cbf_effect" / f"{scenario.name}.png"
        plot_cbf_effect_frame(
            cbf_effect_path,
            namespace,
            scenario,
            audits,
            seed=int(namespace["SEED"]) + 10_000 * scenario_index,
        )
        cbf_effect_paths.append(cbf_effect_path)

    summary_path = output_dir / "policy_scenario_action_summary.csv"
    cbf_effect_summary_path = output_dir / "cbf_effect_action_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    pd.DataFrame(cbf_effect_rows).to_csv(cbf_effect_summary_path, index=False)
    return image_paths, cbf_effect_paths, summary_path, cbf_effect_summary_path


def main() -> None:
    repo_root = find_repo_root()
    default_output = repo_root / "artifacts" / "lanelessKaralakou" / "policy_scenarios"
    parser = argparse.ArgumentParser(description="Render fixed policy-comparison scenarios for laneless DDPG variants.")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--rollout-steps", type=int, default=12)
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Scenario name to render. Repeat to select multiple. Default renders all.",
    )
    args = parser.parse_args()

    selected = set(args.scenario) if args.scenario else None
    image_paths, cbf_effect_paths, summary_path, cbf_effect_summary_path = run(
        args.output_dir,
        selected,
        args.rollout_steps,
    )
    print("Saved policy scenario screenshots:")
    for path in image_paths:
        print(f"  {path}")
    print("Saved CBF-effect screenshots:")
    for path in cbf_effect_paths:
        print(f"  {path}")
    print(f"Saved action summary: {summary_path}")
    print(f"Saved CBF-effect action summary: {cbf_effect_summary_path}")


if __name__ == "__main__":
    main()
