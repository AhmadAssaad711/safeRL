# Lane-Free Highway Env

Custom highway-env extension for lane-free traffic inspired by the TrafficFluid artificial-fluid concept.

The environment is implemented in `lane_free_env.py` and registers as `lane-free-v0`. It uses highway-env's existing `AbstractEnv`, `Road`, `RoadNetwork`, vehicle graphics, and native `EnvViewer`, but does not edit highway-env source files.

It intentionally has no lane indices, lane centers, target lanes, lane-change actions, or `DiscreteMetaAction`.

The benchmark configuration used by the safeRL work is defined by
[notebooks/lanelessKaralakou.ipynb](../notebooks/lanelessKaralakou.ipynb). For the
cell order, PPO/DDPG variants, CBF contract, KPI definitions, and script
responsibilities, see [docs/lanelessKaralakou_reference.md](../docs/lanelessKaralakou_reference.md)
and [docs/script_reference.md](../docs/script_reference.md). The defaults below
describe the reusable environment itself and are not a substitute for the
notebook benchmark configuration.

## Files

- `lane_free_env.py` defines `LaneFreeTrafficEnv`, `LaneFreeVehicle`, Gymnasium registration, and the small runtime renderer extension needed for a lane-free road surface
- `demo_lane_free.py` runs nudging/no-nudging demos
- `outputs/` is created by the demo for plots

## Usage

Install the runtime dependencies:

```powershell
python -m pip install numpy gymnasium highway-env matplotlib pygame
```

```python
import gymnasium as gym
import lane_free_env  # registers lane-free-v0

env = gym.make(
    "lane-free-v0",
    config={
        "road_length": 1000.0,
        "road_width": 10.2,
        "vehicles_count": 30,
        "gamma_nudge": 0.5,
        "ego_controlled": True,
    },
)

obs, info = env.reset(seed=7)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Config

Key defaults:

```python
{
    "road_length": 1000.0,
    "road_width": 10.2,
    "dt": 1 / 15,
    "vehicles_count": 30,
    "sensing_range": 80.0,
    "episode_steps": 2000,
    "gamma_nudge": 0.0,
    "ego_controlled": True,
    "neighbors_count": 8,
    "desired_speed_range": [25.0, 35.0],
    "force": {
        "k_target_x": 2.5,
        "k_target_y": 2.0,
        "k_rep": 8.0,
        "k_nudge": 0.18,
        "k_boundary": 0.18,
        "sigma_x": 5.0,
        "sigma_y": 1.0,
        "T_x": 1.2,
        "T_y": 0.8,
    },
    "bounds": {
        "ax_min": -6.0,
        "ax_max": 3.0,
        "ay_min": -4.0,
        "ay_max": 4.0,
        "max_lateral_speed_ratio": 0.3,
        "max_speed": 45.0,
    },
}
```

The observation is a `(1 + neighbors_count, 7)` normalized array with rows:

```text
[dx, dy, vx, vy, length, width, desired_speed]
```

The first row is ego. Neighbor `dx` is the signed wrapped distance relative to ego.

## MTM Surrounding Traffic

The same `lane-free-v0` environment can keep the current force-based traffic model
or switch non-ego vehicles to a Kanagaraj-Treiber-style MTM model:

```python
env = gym.make(
    "lane-free-v0",
    config={
        "traffic_model": "mtm",
        "vehicles_count": 35,
    },
)
```

The ego vehicle is still controlled by the RL action. Existing CBF wrappers still
filter only ego acceleration. MTM is used only for surrounding vehicles, with the
ego included as a neighbor so traffic can react to it.

Available built-in driver profiles are `normal`, `aggressive`, and `cautious`.
Their probabilities and parameters live under the `mtm` config block.

Quick force-vs-MTM smoke comparison:

```powershell
# If your shell is currently in this environment directory:
Set-Location ..
# The module command must run from highway-rl-decision-making.
python -m scripts.ops.mtm_laneless_smoke
```

## Demo

Saved plot comparison:

```powershell
python "demo_lane_free.py"
```

Live human render:

```powershell
python "demo_lane_free.py" --render human --steps 2000 --gamma-nudge 0.5
```

This uses highway-env's native pygame `EnvViewer`, not a separate custom renderer.

The demo compares:

- `gamma_nudge = 0.0`
- `gamma_nudge = 0.5`

It plots final x-y positions, mean speed, cumulative collisions, and approximate flow at `x=0`.
