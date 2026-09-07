# HOCBF reward speed pilot — 10k transitions

Date: 2026-09-07

## Hypothesis and motivation

The raw-execution nominal PPO condition and the raw-execution PPO condition
with a direct second-order HOCBF reward should both initialize and train at a
practical speed under the requested 100 Hz physics, 20 Hz policy, and 20 Hz
HOCBF-reward rates. This run is a runtime check only; it is not a policy
comparison.

## Baseline and change

- Baseline: `ppo_nominal`, canonical task/collision reward, physical action-box
  execution, no CBF-specific reward.
- Treatment: `ppo_hocbf_reward_raw`, identical policy input and task reward,
  physical action-box execution, plus the complete second-order HOCBF residual
  penalty over active neighbor and road-boundary rows.
- Both conditions used MTM traffic, 55 vehicles, the canonical 32D PPO input,
  and seed `307`.
- Training used five 100 Hz physics steps per 20 Hz policy action. Hard CBF
  filtering was disabled during training.

## Command and configuration

Run from `safeRL_workspace`:

```powershell
python -m scripts.training.run_ppo_cbf_progression --project-root . --output-dir ..\artifacts\hocbf_speed_pilot_10k_20260907_v2 --device cpu --timesteps 10000 --n-envs 20 --seeds 307 --variants ppo_nominal ppo_hocbf_reward_raw --traffic-model mtm --env-config-file configs\hocbf_reward_speed_pilot.json --skip-evaluation --skip-post-train-evaluation --skip-counterfactual --tensorboard-run-label hocbf_speed_10k_20260907_v2
```

- Budget: 10,000 transitions per variant.
- Rollout topology: 20 spawned environments.
- Timing config: `simulation_frequency=100`, `policy_frequency=20`,
  `dt=0.01`, `cbf_substep_filtering=false`.
- Output: `artifacts/hocbf_speed_pilot_10k_20260907_v2/`.
- Historical note: this speed-only run used a zero-action calibration rollout
  with seed `307` and 200 policy steps, selecting
  `s_psi=585.2092561280535`. Its scale is retained only as provenance for the
  completed speed test; it is not valid for the learning comparison. Current
  runs calibrate from one fixed deterministic nominal PPO policy rollout.

## Source hashes

- Notebook `notebooks/lanelessKaralakou.ipynb`: `38DE05E87A27008BC1E7CF143534862D8EE6BC327C36B8A72DB98BF3212DF686`
- Runner `scripts/training/run_ppo_cbf_progression.py`: `38A1507365768E536DF9B3C75CD8CDA6F19BA722FBCCDB6F581EEAC3725FFA88`
- Wrapper `scripts/common/ppo_cbf_env.py`: `D451E3B731CC7862EC230F083267DC5E8A2683DCEB9D4E780904933CC02A2750`
- Timing config `configs/hocbf_reward_speed_pilot.json`: `2259FE155B66A3A733EE12D02343626EE415632B925EE67119EA2B647FCCBD85`

## Results

Both checkpoints completed successfully. The full cold two-variant invocation
took 58.12 wall-clock seconds. Recorded training times were:

| Variant | Training time | Throughput |
| --- | ---: | ---: |
| `ppo_nominal` | 19.06 s | 524.69 transitions/s |
| `ppo_hocbf_reward_raw` | 23.32 s | 428.88 transitions/s |

The run passed the focused validation set: 28 tests passed, with two unrelated
pre-existing protocol tests deselected. KPI and counterfactual evaluation were
intentionally skipped.

## Interpretation and next steps

The simulator and training path are fast enough for the planned 50k pilot.
The HOCBF reward adds modest runtime overhead at this scale. The speed result
does not establish safety or learning performance. Before the 50k run, confirm
the frozen `s_psi` using a fixed nominal-policy raw rollout, then run seeds
`307`, `308`, and `309` with both raw CBF-OFF and runtime-shield CBF-ON
evaluation.
