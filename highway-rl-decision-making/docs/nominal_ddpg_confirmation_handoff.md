# Nominal DDPG P0/P2 confirmation handoff

This stage compares the two screened nominal-DDPG finalists. It does not activate
the CBF during training or evaluation, and it does not change the environment,
reward, collision protocol, or recorded CBF snapshot.

## Fixed protocol

- Configurations: `P0_current`, `P2_more_exploration`
- Training seeds: `307`, `1307`, `2307` (paired across configurations)
- Training budget: 150,000 timesteps per configuration and seed
- Checkpoint/evaluation cadence: every 10,000 timesteps
- Evaluation seeds: `900000` through `900009` at every checkpoint
- Evaluation budget: 800 timesteps per evaluation seed
- Collision protocol: terminate, reset immediately, and continue to the fixed budget
- Critic calibration anchors: segment step 0 and every 20 steps
- Primary calibration target: discounted Monte Carlo return only when a true
  terminal state is observed
- Truncation/evaluation-budget tails: right-censored for primary calibration;
  a critic-bootstrapped sensitivity target is recorded separately

The confirmation ranking uses rollout seed means, then worst-seed rollout values
and seed variances. Raw critic MSE, TD error, Q magnitude, and critic-calibration
errors are reported but do not contribute to the winner rank.

## Start

From any PowerShell working directory:

```powershell
& "<repo>\scripts\ops\run_nominal_ddpg_confirmation.ps1"
```

The default output is `C:\agv_pilot_confirm_final`. Standard output and error
logs are kept outside the result directory under `artifacts\pilot_run_logs`, so
the fresh-run guard remains valid.

## Monitor

One status snapshot:

```powershell
& "<repo>\scripts\ops\monitor_nominal_ddpg_confirmation.ps1"
```

Continuous monitoring:

```powershell
& "<repo>\scripts\ops\monitor_nominal_ddpg_confirmation.ps1" -Watch
```

One Windows virtual-environment invocation normally appears as two `python.exe`
PIDs. The monitor groups the launcher and child interpreter and reports one
`InvocationCount`.

## Strict resume

Use exactly the same launcher parameters:

```powershell
& "<repo>\scripts\ops\run_nominal_ddpg_confirmation.ps1" -Resume
```

Do not edit the pilot runner, shared evaluation pipeline, environment, notebook,
or configuration sources after a checkpoint is created. Their hashes are part
of strict-resume validation.

## Main outputs

- `ranking_final_three.csv`: rollout-first confirmation ordering
- `final_three_seed_averages.csv`: final three checkpoints within each seed
- `final_three_across_seeds.csv`: equal-seed means, sample variances, minima,
  and maxima
- `paired_seed_differences.csv`: per-seed P2-minus-P0 differences
- `paired_difference_summary.csv`: mean and sample variance of paired differences
- `critic_calibration_samples.csv`: anchor-level Q and return records
- `critic_calibration_bins.csv`: five-bin Q calibration curves
- `evaluation_scenarios.csv`: fixed-seed rollout KPIs
- `checkpoint_diagnostics.csv`: critic, calibration, and training diagnostics

Positive `critic_calibration_bias_mean` means Q overestimation. Interpret that
metric together with `critic_calibration_exact_coverage`; low coverage means too
few truly terminated segments were observed for a strong empirical conclusion.
