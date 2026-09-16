# HOCBF state cost, 2026-09-15 to 2026-09-16

Status: **the state cost does not improve unshielded safety.** Three paired 1M
seeds, every seed against the term. Results below; raw run data is under
`artifacts/runs/` (gitignored, OneDrive-synced) mirrored from
`%LOCALAPPDATA%\Temp\saferl_runs\`.

> **Correction, 2026-09-16 — read this before quoting any number below.**
> The headline "+13%, three of three seeds" was never statistically
> established, a fourth seed changes the estimate, and the seeds are not
> independent of each other. See "Statistical correction" at the end of this
> document. The measurements are unchanged; their interpretation is not.

## What was tested

    r_t = r_task,t - lambda_s * C(s_{t+1})

    C(s) = max_i { [(delta_h,c - h_i)/sigma_h,c]_+^2
                 + [(delta_psi,c - psi_1,i)/sigma_psi,c]_+^2 }

`psi_1,i = h_dot_i + gain * h_i`, evaluated on the post-step constraint system,
vehicle and boundary barriers normalized separately. Margins fix when feedback
starts, scales fix what counts as substantial risk, `lambda_s` fixes how
strongly learning responds. Implemented as variant `ppo_hocbf_state_cost_raw`
(`execution_mode="box"`, no CBF in the training loop).
`ppo_hocbf_reward_raw` and `ppo_hocbf_dense_raw` are untouched.

Two barrier forms, `--state-cost-barrier-form`:

- **ellipse** (default): `h` as built, the squared relative-ellipse value.
- **distance**: `rho = sqrt(h+1)` is separation in ellipse radii, rate
  `h_dot/(2 rho)`, so `psi_1 >= 0 <=> clearance/closing_rate >= 1/gain` and the
  gain is a time-to-collision threshold. Boundary rows are never transformed.

`tests/unit/test_state_cost_behaviour.py` is the acceptance gate (far/receding
costs nothing; closing fast with room warns and rises; close and closing is
large; a braking recovery lowers the cost while the gap still shrinks). The
distance form passes all four. **The ellipse form carries two strict xfails**
for early warning and braking: `h` is quadratic in distance, so at 30 m closing
at 6 m/s `h = 33.7` and `psi_1 = 63.7`, far outside any margin that still
discriminates near contact, and the cost cannot fire until about 7.2 m
longitudinally. That is a property of the barrier's units, unreachable by any
delta, sigma or lambda.

## Headline result: 1M, v_desired 16, 41D, three paired seeds

lambda_s = 0.0563 (calibrated to 15% of base reward at v_d 16). Contract: mtm,
40 vehicles, 100/20/20, Q1_stable, 20 envs x 1000, field lateral target,
`--potential-field-weight 0`, `neighbors_count 7`,
`--max-neighbor-constraints 7`, `--drop-ego-speed-slot`, 200+200 evaluation
from seed 1,100,000.

**CBF-OFF (the primary target), mean of seeds 307/308/309:**

| KPI | control | ellipse | delta |
| --- | ---: | ---: | ---: |
| Ego collisions / km | 3.712 | 4.196 | **+13.0%** |
| Episode return | 252.5 | 206.0 | -18.4% |
| Episode length (steps) | 370.7 | 319.2 | -13.9% |
| Abs speed error (m/s) | 1.945 | 2.173 | +11.8% |
| Mean lateral tracking err (m) | 1.574 | 1.966 | +24.9% |
| Mean jerk norm | 5.510 | 2.842 | **-48.4%** |
| Distance completion rate | 0.033 | 0.025 | -25.0% |
| Minimum h | -0.507 | -0.509 | +0.4% |

**CBF-ON:**

| KPI | control | ellipse | delta |
| --- | ---: | ---: | ---: |
| Ego collisions / km | 0.268 | 0.320 | +19.4% |
| Distance completion rate | 0.773 | 0.740 | -4.3% |
| Episode return | 805.5 | 751.0 | -6.8% |
| Intervention rate | 0.744 | 0.906 | +21.8% |
| Correction norm | 0.446 | 0.547 | +22.7% |
| Mean lateral tracking err (m) | 1.588 | 2.044 | +28.8% |
| Abs speed error (m/s) | 1.858 | 2.109 | +13.5% |
| QP failure rate | 0.071 | 0.078 | +10.1% |
| Mean jerk norm | 6.441 | 5.263 | -18.3% |

Per-seed raw collisions/km: 3.683/4.049 (+9.9%), 4.244/4.345 (+2.4%),
3.209/4.193 (+30.7%). Three of three against the term.

**Jerk is not a reliable benefit either.** It reads -48% on the mean but the
per-seed control values are 5.268, 8.455 and 2.807: on seed 309 the term is
*worse* (2.962 vs 2.807). The mean is driven by the control's variance, not by
the term.

## The budget reverses the sign, and nothing has converged

| budget | raw collisions/km, control -> ellipse |
| --- | --- |
| 250k, v_d 20 | 10.375 -> 7.060 (term 32% better) |
| 250k, v_d 16 | 8.684 -> 6.915 (term 20% better) |
| 1M, v_d 16, 3 seeds | 3.712 -> 4.196 (term 13% worse) |

A tail-trend check over the last 50% of training, all six 1M runs, slope of
episode length per 100k steps with t:

| run | slope | t | run | slope | t |
| --- | ---: | ---: | --- | ---: | ---: |
| control 307 | +12.3 | 1.22 | ellipse 307 | +23.3 | 4.36 |
| control 308 | +25.9 | 4.23 | ellipse 308 | +19.6 | 3.22 |
| control 309 | +51.3 | 5.38 | ellipse 309 | +24.0 | 2.85 |

**Five of six are still significantly improving at 1M**, and collisions per km
has not flattened anywhere (all |t| <= 1.72). So every comparison to date
measures learning speed at an arbitrary cutoff, not converged quality. An
earlier claim that the control had converged while the treatment had not came
from seed 307 alone and does not hold: on 308 and 309 the control climbs faster.

## Supporting measurements

- **Model selection does not exist.** `model.save` runs immediately after
  `model.learn`; the 25k checkpoints are for crash resumption only, selected by
  "latest", not "best". No EvalCallback, no early stopping. Every KPI above is
  one draw from an oscillating policy.
  `scripts/evaluation/evaluate_checkpoint_ensemble.py` addresses this by
  averaging the newest N checkpoints, and `--checkpoint-steps` samples across
  training for a deterministic learning curve. **Neither has been run yet on
  any arm.** Best-of-N selection was deliberately *not* implemented: choosing
  on the same 200 episodes that are then reported is test-set selection and
  would bias the noisier arm more.
- **Jerk is unregulated by design.** The notebook penalty is
  `0.02 * tanh(jerk/10)`, bounded by the weight: measured at 0.6-1.1% of the
  reward, and already at tanh 0.51 by jerk 5.6, so it has no gradient left
  where control is most violent. Action saturation reaches 0.107.
- **Jerk is measured on the actuator request, not realised motion.**
  `_last_accelerations` is stored after the action-box clip
  (`lane_free_env.py:1161`) but before `_integrate`, which clamps `position[1]`
  and zeroes `vy` at the wall (`:2210`) without touching the acceleration. Jerk
  therefore counts commands the wall discarded. Measured, this is **not** the
  main contributor: jerk is *lower* at the wall than off it in all four runs
  checked. What predicts jerk is lateral effort - mean |a_y| 0.41 -> 1.22 from
  250k to 1M policies.
- **The wall channel is not dominant in these policies.** Boundary argmax share
  is ~0.000 at almost every checkpoint of the 250k ellipse run; the 52-67%
  measured during calibration came from the older 42D v_d 16 wall-rider.
- **A lateral-suppression explanation was tested and rejected.** At 1M the
  cost-term policy uses *more* lateral authority than the control (mean |a_y|
  0.98 vs 0.73, decisive maneuvers 0.213 vs 0.124).

## Configuration findings

- **v_desired 16 beats 20.** At 250k: 8.684 vs 10.375 raw collisions/km, speed
  error 4.718 vs 10.218. `wx` was never retuned when the `cx` normalizer moved
  with v_desired, which is the likely cause.
- **41D observation.** `--drop-ego-speed-slot` removes the ego row's
  desired-speed feature, constant whenever the speed target is fixed, narrowing
  42D to 41D. Checkpoints are not portable across that change.
- **"42D" is ambiguous** and must never appear unqualified: neighbor-7 PPO is
  8 rows x 5 + 2, legacy DDPG is 6 rows x 7.

## New control-effort term (implemented, not yet trained)

`--control-effort-weight` adds `w * mean_i (a_i/a_i_max)^2` on the applied
acceleration. Quadratic rather than tanh-bounded so it keeps producing
gradient, and on magnitude rather than rate so it pushes against action
saturation as well as jerk. Default 0.0 keeps the notebook reward.

Calibrated on the trained 1M nominal: effort mean 0.223, p90 0.675, action
saturation 0.107, base reward 0.765, so **w = 0.343 places it at 10% of the
reward** against the jerk term's ineffective ~1%.

A 1M nominal run with this term was dispatched to a second machine
(Remote Control session "Pull and update repo"). **That session went offline
without replying; the run has not started anywhere.** Its matched control
already exists (`nom_control_v16_1M`, seed 307), so it is a single run.

## Run inventory

Mirrored to `artifacts/runs/` (439 MB: all provenance, KPI/episode CSVs, final
models, TensorBoard, plus the 1M study's intermediate checkpoints; the older
studies' 567 MB of checkpoints remain only in Temp).

| tree | seeds | contents |
| --- | --- | --- |
| `state_cost_1M` | 307-310 | the headline study; **seed 310 trained but has no KPIs** - stopped mid-evaluation, `model_final.zip` saved, evaluable later without retraining |
| `state_cost_250k` | 307 | 250k arms: v_d 20 control/ellipse/TTC, v_d 16 control/ellipse |
| `neighbor7_study_250k` | 307-309 | the completed 21-run study (7 variants), 42D |
| `hocbf_fix` | 307-311 | the earlier dense-HOCBF lambda/tau investigation |
| `lateral_ladder_250k` | 307 | lateral-target ladder |

## Open items

1. **Nothing has converged.** The approved plan is: deterministic KPI curve via
   `--checkpoint-steps 250000 500000 750000 1000000` on seeds 307-309, pick a
   budget where both arms flatten, then run the converged set. Plan file:
   `~/.claude/plans/are-the-policies-converging-tender-cookie.md`.
2. **Checkpoint-ensemble re-evaluation** of all arms, uniformly, ~27 min per run.
3. **Control-effort 1M run** still unstarted.
4. **TTC barrier form has one data point** (250k, v_d 20) and has never run at
   1M, despite being the only form that passes the acceptance gate.
5. Seed 310 evaluation, if a fourth seed is wanted. **Done, see below.**

## Statistical correction, 2026-09-16

Everything measured above stands. What was claimed from it does not.
Reproducible via `scripts/evaluation/summarize_training_stability.py`.

### The effect was never significant, and unanimity could not have made it so

| | n = 3 (307-309) | n = 4 (+ seed 310) |
| --- | ---: | ---: |
| effect, raw collisions/km | +0.484 (+13.0%) | **+0.563 (+16.1%)** |
| paired t | 1.85 | 2.80 |
| p | 0.205 | **0.068** |
| exact sign-test p | 0.250 | 0.125 |
| paired SD | 0.453 | 0.402 |
| detectable effect at 80% power | +37.8% | +24.0% |

Seed 310 was already trained and only needed evaluating: control 2.809, ellipse
3.609, difference +0.800. So the effect is now **4 of 4 seeds and stronger**,
which points at something real — but it is still not significant at alpha 0.05.

Two things to retire permanently:

- **"Three of three seeds against the term" is not evidence.** With `n` seeds
  agreeing, the smallest reachable two-sided sign-test p is `2/2^n` = 0.250 at
  n = 3 and 0.125 at n = 4. A unanimous 3-seed result cannot reach 0.05 under
  any test that respects the pairing.
- **The earlier MDE figures were 50%-power numbers.** `t_crit * sd / sqrt(n)` is
  the effect detected half the time. At 80% power the observed effect needs
  **n = 9**, not the n = 4 or 6 quoted before. A pre-specified 20% relative
  regression needs n = 6 at the measured SD, or n = 10 at a conservative one.

The paired SD of 0.453 carries two degrees of freedom and its 95% CI is
**[0.236, 2.844]** — a factor of twelve. Any seed budget planned from that point
estimate is planned from noise; plan from the variance decomposition instead.

### Variance decomposition (control arm, 4 seeds)

| component | value |
| --- | ---: |
| within-run block SD | 1.048 |
| evaluation-only SE of a run mean | 0.331 |
| observed between-seed SD | 0.619 |
| **genuine seed-to-seed SD** | **0.522** (71% of between-seed variance) |

So roughly a third of the apparent seed spread is evaluation noise that more
episodes would remove; the rest needs more seeds.

### The training seeds are not independent

`VecEnv.seed(s)` assigns worker seeds `s .. s+19`, and the scenario stream is a
pure function of the worker seed: `np_random` is consumed **only** in the reset
and spawn path (`_reset`, `_create_vehicles`, `_sample_legacy_initial_states`,
`_sample_safe_initial_states`, `_sample_mtm_profile`,
`_sample_mtm_aggressiveness`) and nowhere in `step`. Verified: two envs reset
with the same seed produce byte-identical scenarios, and a whole stream replays
identically.

| shared worker seeds | 307 | 308 | 309 | 310 |
| --- | ---: | ---: | ---: | ---: |
| **307** | 20 | 19 | 18 | 17 |
| **310** | 17 | 18 | 19 | 20 |

The runs therefore share 17-19 of 20 scenario streams. The within-seed pairing
is unaffected, but the paired differences are correlated through shared training
data, so the across-seed p-value is optimistic. **Fix: space training seeds by
at least `n_envs`** (307, 327, 347), which also means 311-313 are *not* clean
additions to this set.

### It is not an RL optimisation problem

All six 1M runs, first-decile to last-decile: `approx_kl` 0.005 -> 0.009 (an
order of magnitude under 0.02), `clip_fraction` 0.03 -> 0.09, `explained_variance`
0.24-0.37 -> 0.66-0.79, `train/std` flat at ~0.6 against `exp(-0.5) = 0.6065`,
i.e. within 6% of initialisation. Healthy, and **indistinguishable between the
two arms** on every one of them. Hyperparameter tuning is not indicated.

### Final-step selection is mostly, but not entirely, harmless

Four late checkpoints (925k-1M), 200 episodes each, control arm; the spread
across checkpoints corrected for within-checkpoint evaluation noise:

| seed | 925k | 950k | 975k | 1M | corrected snapshot SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 307 | 3.811 | 4.318 | 4.507 | 4.089 | 0.000 |
| 308 | 4.236 | 4.057 | 4.142 | 4.032 | 0.000 |
| 309 | 2.067 | 2.377 | 2.735 | 3.071 | **0.379** |

Seeds 307 and 308 show no drift beyond evaluation noise. Seed 309 drifts
monotonically worse and its 0.379 is **78% of the effect being interpreted** —
and 309 also carries the largest ellipse-vs-control gap (+0.984), so part of
that gap may be drift rather than the term. Pooled snapshot SD 0.219.

### The whole study ran on wall-hugging policies

Six of seven arms spend 72-85% of every step within half a metre of a road
boundary, because the reward's own lateral target sits within 1.5 m of a wall
46-47% of the time. **Seed 307, the reference for every single-seed comparison
here, is the sole exception at 21.6%** — which is why treatments appeared to
degrade lateral tracking and increase CBF workload. See
`lateral_emphasis_20260916.md`; that defect is now fixed and the fix removes
wall-hugging entirely.
