# Nominal PPO 50k pilot handoff

The pilot is prepared but has not been started. The authoritative entry point
is `scripts/training/run_nominal_ppo_parameter_pilot.py`. The notebook PPO training cell
now delegates to that runner, and its result cells read the runner's corrected
fixed-timestep outputs; the legacy in-notebook PPO implementation is removed.

## Fixed protocol

- Nominal PPO only; the CBF is inactive and its tuned values are recorded but
  not changed.
- Environment, MTM traffic setup, reward, action bounds, and collision handling
  match the corrected nominal-DDPG protocol.
- Training seed: `307` for every configuration. The environment is seeded once
  when the run starts; normal episode resets are not reseeded.
- Budget: exactly 50,000 timesteps per configuration.
- PPO rollout: 1,000 global transitions, batch size 100. The CUDA pilot uses
  eight workers with 125 steps each; it still has exact post-update boundaries
  at 10k, 20k, 30k, 40k, and 50k.
- Evaluation: deterministic, fixed seeds `900000` through `900009`, 800
  timesteps per seed, once after the final 50k post-update policy. Lightweight
  model snapshots remain at every 10k boundary. Use `--evaluate-checkpoints`
  only when a full learning curve is specifically needed.
- Collision protocol: `terminate_on_collision=True`, reset immediately, and
  continue until the fixed training/evaluation timestep budget is consumed.
- Collisions are distinct events; collision-active timesteps are separate.

## Configurations

| Config | LR | Epochs | Gamma | Clip | Entropy | Initial log std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Q0_current_aligned` | 3e-4 | 10 | .98 | .20 | .010 | 0.0 |
| `Q1_stable` | 1e-4 | 10 | .99 | .20 | .005 | -0.5 |
| `Q2_exploratory` | 1e-4 | 10 | .99 | .20 | .020 | -0.25 |
| `Q3_conservative_update` | 2e-4 | 5 | .99 | .15 | .010 | -0.5 |

All use GAE .95, value coefficient .5, maximum gradient norm .5, separate
`[256,128]` policy/value networks with Tanh, orthogonal initialization, no gSDE,
and no value-function clipping.

`Q2_exploratory` increases exploration relative to the stable `Q1` baseline;
`Q0` still has the largest initial standard deviation.

## Start when ready

From any PowerShell working directory:

```powershell
& "<repo>\scripts\ops\run_nominal_ppo_pilot.ps1"
```

Default results go to `C:\agv_ppo_pilot_50k`. Standard output/error are kept
under `artifacts\pilot_run_logs`, outside the guarded result directory.

Monitor once or continuously:

```powershell
& "<repo>\scripts\ops\monitor_nominal_ppo_pilot.ps1"
& "<repo>\scripts\ops\monitor_nominal_ppo_pilot.ps1" -Watch
```

Strict resume uses only a validated callback checkpoint:

```powershell
& "<repo>\scripts\ops\run_nominal_ppo_pilot.ps1" -Resume
```

`model_final.zip` is never silently selected as a resume source. Do not edit
hashed source/configuration files between interruption and strict resume.

## Outputs and selection

The primary outputs are `evaluation_scenarios.csv`,
`checkpoint_diagnostics.csv`, `training_episodes.csv`, TensorBoard event files,
`final_evaluation_seed_averages.csv`, `final_evaluation_across_seeds.csv`, and
`ranking_final_evaluation.csv`.

Ranking uses the final 50k behavior: distance per distinct collision,
return/timestep, speed error, episode length, and deterministic actor action
saturation. PPO value loss, value-target error/magnitude, KL, entropy, clip
fraction, policy standard deviation, and raw-action clipping are reported as
diagnostics. With one screening seed, across-seed variance is intentionally
undefined; finalists should later be confirmed with independent seeds.
