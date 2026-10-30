# safeRL

Safe reinforcement learning on a lane-free highway. A PPO agent controls
continuous longitudinal and lateral acceleration among 40 vehicles in MTM
traffic. A high-order control barrier function (HOCBF) shield can correct its
commands before they reach the simulator. The study compares ways of bringing
safety into training, and evaluates every policy both with the shield off
(CBF-OFF) and on (CBF-ON).

**Project page:** <https://ahmadassaad711.github.io/safeRL/>.
It has rollout videos in full traffic and in diagnostic scenarios,
robustness plots and the results.

## Policies compared

| Label | Variant | How safety enters training |
| --- | --- | --- |
| A1 | `ppo_nominal` | None: the nominal reward only |
| B1 | `ppo_hocbf_state_cost_raw` | Fixed state-cost penalty in the reward |
| B2 | `ppo_cbf_nd_reward_actor` | Direct geometric actor loss, HOCBF-filtered training |
| C1 | `ppo_cbf_diff_reward_only` | Differentiable projection plus reward |
| C2 | `ppo_cbf_projected` | Differentiable projection plus actor loss |

## Experiment contract

| Item | Value |
| --- | --- |
| Environment | `lane-free-v0`, periodic 380 m by 10.2 m road, no lanes |
| Traffic | 40 vehicles, MTM surrounding traffic |
| Timing | 100 Hz physics, 20 Hz policy, 20 Hz CBF |
| Action | Physical acceleration, both components in [-3, 3] m/s² |
| PPO observation | 32D at five neighbor rows: target-y vehicle features plus the previous action |
| Evaluation | Collision-free 1,000 m within 3,000 policy steps |

The source of truth is the notebook
[lanelessKaralakou.ipynb](safeRL_workspace/notebooks/lanelessKaralakou.ipynb),
explained in the
[lanelessKaralakou reference](safeRL_workspace/docs/lanelessKaralakou_reference.md).

## Repository map

| Path | Role |
| --- | --- |
| [safeRL_workspace/](safeRL_workspace/README.md) | Implementation workspace; run commands from here |
| safeRL_workspace/notebooks/ | Canonical notebook and experiment specification |
| safeRL_workspace/laneless highway env/ | `lane-free-v0` environment and renderer |
| safeRL_workspace/src/saferl/ | New research code: configuration, safety algorithms, artifacts |
| [safeRL_workspace/scripts/](safeRL_workspace/scripts/README.md) | Training, evaluation, reporting, rendering and ops modules |
| [safeRL_workspace/tests/](safeRL_workspace/tests/README.md) | Unit, contract and protocol tests |
| [safeRL_workspace/docs/](safeRL_workspace/docs/README.md) | Experiment, scenario and script documentation |
| [artifacts/](artifacts/README.md) | Committed result manifests and compact summaries |
| docs/ | Source of the GitHub Pages project page |
| [AGENTS.md](AGENTS.md) | Architecture and change rules for contributors |

## Setup

Python 3.10 or newer. On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r safeRL_workspace/requirements.txt
```

On Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r safeRL_workspace\requirements.txt
```

## Quick start

Run every command from `safeRL_workspace`:

```bash
cd safeRL_workspace
python -m scripts.ops.mtm_laneless_smoke --help        # environment smoke check
python -m pytest -q tests                              # unit and contract tests
python -m scripts.evaluation.evaluate_laneless_karalakou --help
```

Inspect `--help` before starting any training run. The full training ladder
is long, so check the variant, seed, budget and output directory first.

## Documentation

- [Experiment contract](safeRL_workspace/docs/lanelessKaralakou_reference.md):
  environment, reward, observation, CBF and evaluation.
- [Script and function reference](safeRL_workspace/docs/script_reference.md):
  which module to run for which task.
- [Diagnostic scenarios](safeRL_workspace/docs/diagnostic_scenarios.md):
  the fixed scenes behind the scenario videos.
- [Documentation index](safeRL_workspace/docs/README.md): every study and
  experiment log.
