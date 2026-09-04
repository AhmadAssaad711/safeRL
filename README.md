# safeRL

This repository contains the safe reinforcement-learning work for the
laneless Karalakou lane-free highway environment.

The canonical source of truth is
[lanelessKaralakou.ipynb](highway-rl-decision-making/notebooks/lanelessKaralakou.ipynb).
It defines the environment contract, Karalakou reward, CBF geometry, PPO
progression, legacy DDPG comparisons, evaluation protocol, and artifact
layout. The supporting Python runners in the scripts directory execute the
same notebook definitions out of process when a long run should not occupy
the notebook kernel.

## Start here

1. Install the dependencies from
   [requirements.txt](highway-rl-decision-making/requirements.txt).
2. Read the
   [lanelessKaralakou reference](highway-rl-decision-making/docs/lanelessKaralakou_reference.md).
3. Use the
   [script and function reference](highway-rl-decision-making/docs/script_reference.md)
   to choose a runner.
4. Open the canonical notebook for the seven-policy PPO/CBF ladder and the
   retained DDPG reference experiments.

The current research contract uses MTM surrounding traffic, 100 Hz physics,
10 Hz policy actions, a 32D target-y plus previous-action PPO observation, and
strict collision-free 1 km completion for evaluation.

## Repository map

| Path | Role |
| --- | --- |
| highway-rl-decision-making/notebooks/lanelessKaralakou.ipynb | Canonical notebook and experiment specification |
| highway-rl-decision-making/laneless highway env/ | lane-free-v0 environment and renderer |
| highway-rl-decision-making/scripts/ | Training, CBF, evaluation, audit, plotting, and rendering runners |
| highway-rl-decision-making/tests/ | Unit and protocol tests |
| highway-rl-decision-making/docs/ | Experiment, scenario, and API documentation |
| Root ppo_* directories | Committed safeRL result manifests and compact summaries |

Lane-indexed highway notebooks, DQN implementations, unrelated planning
experiments, machine-specific duplicate snapshots, scratch backups, and the
old paper bundle were removed from this repository. The Git history still
contains the prior state if an old reference is ever needed.

## Setup

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r highway-rl-decision-making\requirements.txt

For environment-only smoke testing:

    Set-Location highway-rl-decision-making
    python scripts\mtm_laneless_smoke.py --help

Do not run the full 1M-transition ladder accidentally. The notebook's shared
launcher and the script CLI should be checked before enabling training.
