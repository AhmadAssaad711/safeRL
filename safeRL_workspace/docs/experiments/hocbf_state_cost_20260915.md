# HOCBF state cost, 2026-09-15

Handoff note. Written so a cold session can pick this up without re-deriving
it. Results live outside the repository under
`%LOCALAPPDATA%\Temp\saferl_runs\state_cost_250k\`.

## What is being tested

Whether a margin-hinged HOCBF **state cost** added to the nominal reward makes
the **unshielded** policy safer:

    r_t = r_task,t - lambda_s * C(s_{t+1})

    C(s) = max_i { [(delta_h,c - h_i)/sigma_h,c]_+^2
                 + [(delta_psi,c - psi_1,i)/sigma_psi,c]_+^2 }

with `psi_1,i = h_dot_i + gain * h_i`, evaluated on the post-step constraint
system, and vehicle/boundary barriers normalized separately. Margins fix when
feedback starts, scales fix what counts as substantial risk, `lambda_s` fixes
how strongly learning responds.

Implemented as `ppo_hocbf_state_cost_raw` (`execution_mode="box"`, no CBF in
the training loop). `ppo_hocbf_reward_raw` and `ppo_hocbf_dense_raw` are
untouched, so the completed 21-run study and the five-seed dense result stay
comparable.

## Two barrier forms

`--state-cost-barrier-form`:

- **ellipse** (default): `h` as built, the squared relative-ellipse value.
  Quadratic in distance, so it cannot warn early; measured, it first becomes
  non-zero near 7.2 m longitudinally.
- **distance**: the same geometry read as a clearance. `rho = sqrt(h+1)` is the
  separation in ellipse radii, rate `h_dot/(2 rho)`, so
  `psi_1 >= 0 <=> clearance/closing_rate >= 1/gain` and the gain is a
  time-to-collision threshold. `gain = 0.3` warns below 3.33 s. Boundary rows
  are never transformed; their `h` is already metres.

`tests/unit/test_state_cost_behaviour.py` is the acceptance gate: far/receding
costs nothing, closing fast with room warns and rises, close and closing is
large, a braking recovery lowers the cost while the gap still shrinks. The
distance form passes all four; the ellipse form carries two **strict xfails**
for the early-warning and braking checks, which are properties of the barrier's
units and unreachable by any margin, scale or lambda.

## Configuration shared by every run

mtm traffic, 40 vehicles, 100/20/20, Q1_stable, 20 envs x 1000, seed 307, 250k
steps, field lateral target, `--potential-field-weight 0`,
`neighbors_count = 7`, `--max-neighbor-constraints 7`, `--drop-ego-speed-slot`
(41D learned observation), 200+200 post-training evaluation from seed
1,100,000.

`--drop-ego-speed-slot` removes the ego row's desired-speed feature, which is a
constant whenever the speed target is fixed, narrowing 42D to 41D. Checkpoints
are **not** portable across that change.

## Completed, v_desired = 20

| arm | directory | lambda_s |
| --- | --- | ---: |
| control | `nom_control_s307` | 0 |
| ellipse | `s307` | 0.0754 |
| TTC | `ttc_s307` | 0.0509 |

Raw (CBF-OFF), ellipse against its matched control: ego collisions/km
10.375 -> 7.060 (-32%), abs speed error 10.218 -> 5.606, mean jerk
1.711 -> 1.192, lateral tracking 1.905 -> 1.397. CBF-ON: collisions/km
0.390 -> 0.259, completion 0.680 -> 0.780, intervention 0.792 -> 0.526.

**One seed.** The dense-term investigation gave -18.7% on seed 307 and -7.2%
over five seeds, which is exactly why this needs 308/309 before it is an
effect rather than an anecdote.

## Queued, v_desired = 16

`lambda_s` recalibrated at v_desired = 16 to the same 15% share of base reward
(ellipse `C_mean` 1.180, distance 1.445, base reward 0.443):

| arm | lambda_s | directory |
| --- | ---: | --- |
| C1 control | 0 | `nom_control_v16_s307` |
| ellipse 1x | 0.0563 | `ell_1x_v16` |
| ttc 1x | 0.0459 | `ttc_1x_v16` |
| ellipse 3x | 0.169 | `ell_3x_v16` |
| ttc 3x | 0.138 | `ttc_3x_v16` |
| ellipse 10x | 0.563 | `ell_10x_v16` |
| ttc 10x | 0.459 | `ttc_10x_v16` |

Scales are frozen across the sweep; only `lambda_s` moves.

C1 also separates the two changes made at once. Against the 42D v_desired = 16
A1 study arm (raw 7.679 collisions/km) it measures the observation drop;
against the 41D v_desired = 20 control (10.375) it measures the speed target.

## Guardrails

The post-training KPI table cannot show either degenerate behaviour this term
has historically produced. `scripts/evaluation/probe_behaviour_guardrails.py`
covers the gap: ego y distribution and time spent near a road edge (sideline
hugging), mean speed and the **signed** deficit against the run's own target
(lagging, kept signed so crawling and speeding do not cancel), and the
vehicle/boundary split of the state cost's `max`.

First measurement, v_desired = 20 control, 3 episodes: `y_median` 4.365,
0% of steps within 0.5 m of an edge, mean edge distance 2.98 m, speed
19.43 against a 20.0 target. That policy does **not** hug the wall, unlike the
older 42D v_desired = 16 checkpoint the calibration rollouts used, where the
boundary channel won the `max` on 52-67% of steps.

## Open questions

1. Seeds. Everything is seed 307. Nothing here is an effect until 308/309.
2. `wx` was not retuned when the `cx` normalizer changed with v_desired, which
   is the likely reason the v_desired = 20 control sits 10.2 m/s off its target.
3. The distance form over-warns on crossing and passing geometry, because it
   extrapolates the instantaneous radial rate. `h(t) = 0` is a quadratic under
   constant relative velocity, so exact time-to-contact and closest approach are
   both closed-form if that matters later.
