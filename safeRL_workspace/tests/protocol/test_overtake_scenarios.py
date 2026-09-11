"""Scripted overtaking scenarios in the real lane-free env and reward wrapper.

Each scenario places the ego and one or two traffic vehicles at chosen gaps,
lateral offsets, and speeds close to the ego's real operating speed (the
nominal policy drives about 11-12.5 m/s against a 16 m/s target; nearby
traffic averages about 12.7 m/s). The ego follows a scripted speed schedule
and holds its lateral position; traffic vehicles keep their MTM behavior with
``desired_speed`` set to the scenario speed. Every scenario runs with the
notebook's one-step detector and with the latched detector, and the paid
bonus is read from the wrapper's own ``karalakou_overtakes`` and reward.

Run as a script to print the per-scenario table:
    python -m tests.protocol.test_overtake_scenarios
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from scripts.training import run_ppo_cbf_progression as progression
from scripts.training.run_cbf_filter_ablation import (
    bootstrap_notebook_namespace,
    exec_required_notebook_cells,
)

POLICY_HZ = 20


@dataclass(frozen=True)
class Car:
    dx: float  # initial gap to the ego, m (positive = ahead)
    dy: float  # lateral offset from the ego, m
    speed: float  # m/s, initial and desired


@dataclass(frozen=True)
class Scenario:
    name: str
    ego_y: float
    ego_schedule: tuple[tuple[float, float], ...]  # (duration s, ego speed m/s)
    cars: tuple[Car, ...]
    physical_passes: int  # passes the scenario is built to produce
    expected_latched: int  # bonuses the latched detector should pay
    final_dx_window: tuple[float, float] | None = None  # precondition on car 0


SCENARIOS = (
    Scenario("slow pass, car on the left", 3.0, ((20.0, 13.0),), (Car(8.0, 3.5, 12.0),), 1, 1),
    Scenario("typical pass, car on the right", 6.5, ((20.0, 14.0),), (Car(12.0, -3.5, 12.5),), 1, 1),
    Scenario("brisk pass of a slow car", 3.0, ((15.0, 16.0),), (Car(20.0, 3.5, 12.0),), 1, 1),
    # Both cars in the left half: a car placed near an edge drifts toward the
    # centre under the boundary force and would squeeze a centred ego.
    Scenario(
        "two cars passed in sequence", 3.0, ((20.0, 15.0),),
        (Car(10.0, 3.5, 12.5), Car(20.0, 4.0, 13.5)), 2, 2,
    ),
    Scenario("ego overtaken by a faster car", 3.0, ((20.0, 12.5),), (Car(-15.0, 3.5, 14.5),), 0, 0),
    Scenario("leader pulls away", 3.0, ((20.0, 13.0),), (Car(10.0, 3.5, 15.0),), 0, 0),
    Scenario(
        "alongside, pass not completed", 3.0, ((6.0, 13.5),), (Car(5.0, 3.5, 12.5),), 0, 0,
        final_dx_window=(-1.75, 0.0),
    ),
    Scenario(
        "pass, fall back, re-pass same car", 3.0, ((10.0, 15.0), (15.0, 11.0), (15.0, 15.0)),
        (Car(5.0, 3.5, 13.0),), 2, 1,
    ),
)


def _namespace() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    namespace = bootstrap_notebook_namespace(root)
    exec_required_notebook_cells(root / "notebooks" / "lanelessKaralakou.ipynb", namespace)
    return namespace


def _normalized(value: float, low: float, high: float) -> float:
    value = float(np.clip(value, low, high))
    return value / high if value >= 0.0 else value / abs(low)


def run_scenario(namespace: dict[str, object], scenario: Scenario, detection: str) -> dict[str, object]:
    env_config = copy.deepcopy(namespace["ENV_CONFIG"])
    neighbors = int(env_config["neighbors_count"])
    env_config["vehicles_count"] = max(neighbors + 1, 1 + len(scenario.cars)) + 1
    reward_config = {**namespace["REWARD_CONFIG"], "overtake_detection": detection}
    env = progression._base_environment(namespace, env_config=env_config, reward_config=reward_config)
    env.reset(seed=7)
    base = env.unwrapped
    ego = base.vehicle
    others = [vehicle for vehicle in base.road.vehicles if vehicle is not ego]
    road_length = float(base.config["road_length"])
    x0 = float(ego.position[0])
    ego_speed0 = scenario.ego_schedule[0][1]
    mean_ego_speed = float(np.mean([speed for _duration, speed in scenario.ego_schedule]))

    def place(vehicle, dx: float, y: float, speed: float) -> None:
        vehicle.position[:] = ((x0 + dx) % road_length, y)
        vehicle.vx, vehicle.vy, vehicle.ax, vehicle.ay = float(speed), 0.0, 0.0, 0.0
        vehicle.desired_speed = float(speed)
        vehicle._sync_graphics_fields()

    place(ego, 0.0, scenario.ego_y, ego_speed0)
    for vehicle, car in zip(others, scenario.cars):
        place(vehicle, car.dx, scenario.ego_y + car.dy, car.speed)
    # Park the remaining vehicles near the ring's antipode at the ego's mean
    # speed so they stay far outside the 90 m sensing range.
    for index, vehicle in enumerate(others[len(scenario.cars):]):
        place(vehicle, (175.0 + 7.0 * index) * (-1) ** index, 1.5 + 3.5 * (index % 3), mean_ego_speed)

    bounds = base.config["bounds"]
    cfg = env.reward_config
    schedule = [speed for duration, speed in scenario.ego_schedule for _ in range(int(round(duration * POLICY_HZ)))]
    tracked = others[: len(scenario.cars)]
    paid_steps, bonus_dx, bonus_reward_residuals = [], [], []
    min_abs_dy = np.inf
    car_speeds: list[float] = []
    ego_speeds: list[float] = []
    collided = False
    for step, target_speed in enumerate(schedule):
        ax = 1.5 * (target_speed - ego.vx)
        ay = 1.0 * (scenario.ego_y - ego.position[1]) - 1.5 * ego.vy
        action = np.array(
            [
                _normalized(ax, float(bounds["ax_min"]), float(bounds["ax_max"])),
                _normalized(ay, float(bounds["ay_min"]), float(bounds["ay_max"])),
            ],
            dtype=np.float32,
        )
        _obs, reward, terminated, truncated, info = env.step(action)
        collided |= bool(info["karalakou_ego_collision"] > 0.5)
        dxs = [base._signed_distance(ego.position[0], v.position[0]) for v in tracked]
        min_abs_dy = min([min_abs_dy] + [abs(v.position[1] - ego.position[1]) for v in tracked])
        car_speeds.extend(float(v.vx) for v in tracked)
        ego_speeds.append(float(ego.vx))
        if info["karalakou_overtakes"] > 0:
            paid_steps.append(step)
            bonus_dx.append(list(dxs))
            tracking = cfg["epsilon_r"] / (
                cfg["epsilon_r"] + cfg["wx"] * info["karalakou_cx"] + cfg["wy"] * info["karalakou_cy"]
                + cfg["wf"] * info["karalakou_cf"] + cfg.get("way", 0.0) * info["karalakou_cay"]
            )
            bonus_reward_residuals.append(
                float(reward) - (tracking + info["karalakou_progress_reward"] - info["karalakou_jerk_penalty"])
            )
        if terminated or truncated:
            break
    env.close()
    return {
        "bonuses": len(paid_steps),
        "paid_at_s": [round(step / POLICY_HZ, 2) for step in paid_steps],
        "dx_at_bonus": bonus_dx,
        "bonus_in_reward": bonus_reward_residuals,
        "final_dx": list(dxs),
        "min_abs_dy": float(min_abs_dy),
        "ego_speed": (min(ego_speeds), max(ego_speeds)),
        "car_speed": (min(car_speeds), max(car_speeds)),
        "collided": collided,
        "steps": step + 1,
    }


@pytest.fixture(scope="module")
def namespace():
    return _namespace()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_latched_detector_pays_each_real_pass_once(namespace, scenario):
    step = run_scenario(namespace, scenario, "step")
    latched = run_scenario(namespace, scenario, "latched")
    for result in (step, latched):
        assert not result["collided"], "scenario must be collision-free"
        assert result["min_abs_dy"] > 1.8, "vehicles must never overlap laterally"
    if scenario.final_dx_window is not None:
        low, high = scenario.final_dx_window
        assert low < latched["final_dx"][0] < high
    assert step["bonuses"] == 0  # the notebook detector never fires at 20 Hz
    assert latched["bonuses"] == scenario.expected_latched
    assert latched["bonus_in_reward"] == pytest.approx(
        [float(namespace["REWARD_CONFIG"]["overtake_bonus"])] * latched["bonuses"]
    )
    for dxs in latched["dx_at_bonus"]:
        # Paid on the first step a tracked car is more than half an ego length behind.
        assert any(-1.75 - 0.2 < dx < -1.75 for dx in dxs)


if __name__ == "__main__":
    ns = _namespace()
    print(f"{'scenario':36s} {'passes':>6s} {'step':>5s} {'latched':>7s}  paid at (s)      dx at bonus (m)        final dx (m)     ego m/s      car m/s      min|dy|")
    for scenario in SCENARIOS:
        s = run_scenario(ns, scenario, "step")
        r = run_scenario(ns, scenario, "latched")
        print(
            f"{scenario.name:36s} {scenario.physical_passes:6d} {s['bonuses']:5d} {r['bonuses']:7d}  "
            f"{str(r['paid_at_s']):16s} {str([[round(dx, 2) for dx in row] for row in r['dx_at_bonus']]):22s} "
            f"{str([round(dx, 1) for dx in r['final_dx']]):16s} "
            f"{r['ego_speed'][0]:4.1f}-{r['ego_speed'][1]:4.1f}  {r['car_speed'][0]:4.1f}-{r['car_speed'][1]:4.1f}  "
            f"{r['min_abs_dy']:5.2f}{'  COLLISION' if r['collided'] or s['collided'] else ''}"
            f"  bonus in reward: {[round(x, 3) for x in r['bonus_in_reward']]}"
        )
