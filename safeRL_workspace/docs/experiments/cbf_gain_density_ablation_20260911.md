# HOCBF gain, density, and frequency ablation

Date: 2026-09-11

## Purpose

Choose the HOCBF gains and traffic density used for the final results. The
former gains `k0 = 5.29`, `k1 = 3.68` have complex characteristic roots
(`k1^2 = 13.5 < 4*k0 = 21.2`), so `h_ddot + k1*h_dot + k0*h >= 0` cannot be
factored into a cascade with real class-K rates. They were not a valid HOCBF.
With the CBF ON, 94% of 200 episodes at 40 vehicles still ended in a collision.

## Setup

- Policy: B.1 `ppo_nominal`, seed 307, model sha256 `7fc8d0f1...`, from
  `artifacts/1MRun/nom_orig_0911_backup/ppo_nominal/seed_307/`. It was trained
  without a CBF, so every cell filters the same policy.
- Environment: the run's saved configuration (MTM, 100/20/20 Hz, 1,000 m task,
  3,000-step cap) with `vehicles_count = 40` instead of 55.
- Gains: cascade `psi1 = h_dot + c1*h`, `psi2 = psi1_dot + c2*psi1`, so
  `k1 = c1 + c2` and `k0 = c1*c2`. The QP is symmetric in `(c1, c2)`, so only
  `c1 <= c2` was run, with `c in {0.5, 1, 1.5, 2.3, 3.5, 5, 8}`: 28 cells
  plus CBF OFF and the former gains.
- Paired design: every cell used the same seeds from 1100000, and the same
  spawn guard `h_dot + 2.3*h >= 0`, so initial states match across cells.
- The runs were exploratory (dirty tree, scratchpad drivers). The committed
  entrypoints `scripts.evaluation.evaluate_hocbf_gain_grid`,
  `evaluate_ppo_cbf_overrides`, and `diagnose_cbf_qp_failures` reproduce
  them: a 3-seed rerun of the CBF-OFF and (0.5, 8) cells matched the grid's
  steps, distance, collisions, and QP failure rate exactly.

## Density and frequency (former gains)

200 + 200 episodes per setting. Collisions/km, CBF OFF → ON:

| Setting | OFF | ON | QP failure rate |
| --- | ---: | ---: | ---: |
| 55 vehicles, 100/20/20 Hz | 10.32 | 4.83 | 11.4% |
| 40 vehicles, 100/20/20 Hz | 5.34 | 2.83 | 8.5% |
| 40 vehicles, 20/20/20 Hz | 6.66 | 2.80 | 9.2% |
| 40 vehicles, 100/100/100 Hz | 6.05 | 2.88 | 9.2% |

The CBF sampling rate is not the cause of the remaining collisions.

## Gain grid (30 episodes per cell)

Output: `artifacts/1MRun/v40_gain_grid/`. Selected cells, pooled over
distance:

| (c1, c2) | k0 | k1 | Collisions/km | Completion | QP failure rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| (0.5, 5) | 2.5 | 5.5 | 0.51 | 18/30 | 7.5% |
| (0.5, 8) | 4.0 | 8.5 | 0.66 | 15/30 | 7.3% |
| (0.5, 3.5) | 1.75 | 4.0 | 0.67 | 15/30 | 7.9% |
| (1, 8) | 8.0 | 9.0 | 0.70 | 14/30 | 7.0% |
| (0.5, 0.5) | 0.25 | 1.0 | 1.41 | 6/30 | 34.1% |
| (2.3, 2.3), critical | 5.29 | 4.6 | 1.17 | 9/30 | 8.2% |
| (8, 8) | 64 | 16 | 4.41 | 1/30 | 10.0% |
| former 5.29 / 3.68 | 5.29 | 3.68 | 2.78 | 1/30 | 8.6% |
| CBF OFF | – | – | 7.95 | 0/30 | – |

The smaller rate controls the outcome: every cell with `min(c1, c2) = 0.5`
gives 0.51 to 1.01 collisions/km (except (0.5, 0.5), which is so conservative
that 34% of QPs fail). Cells with both rates at least 2.3 give 1.2 or more.

## Confirmation (200 episodes per cell)

Output: `artifacts/1MRun/v40_gain_confirm200/`, seeds 1100000–1100199, paired
with the 200-episode CBF OFF and former-gain runs in `artifacts/1MRun/v40/`.
The SD is across 10 blocks of 20 episodes.

| Cell | Collisions/km (SD) | Completion | Mean distance | QP failure rate | Speed error | Lateral error | Jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **(0.5, 8)** | **0.50** (0.16) | 121/200 | 789 m | 7.3% | 2.69 | 4.77 | 5.8 |
| (0.5, 5) | 0.51 (0.19) | 121/200 | 779 m | 7.8% | 2.88 | 4.61 | 6.0 |
| (0.5, 3.5) | 0.63 (0.18) | 106/200 | 743 m | 7.9% | 2.72 | 4.60 | 6.1 |
| (2.3, 2.3) | 1.71 (0.37) | 35/200 | 483 m | 9.2% | 3.77 | 4.17 | 4.1 |
| former 5.29 / 3.68 | 2.71 (0.70) | 12/200 | 347 m | 8.3% | 4.47 | 4.11 | 4.2 |
| CBF OFF | 4.90 (1.72) | 4/200 | 200 m | – | 4.88 | 4.07 | 3.2 |

## Decision

The final-results configuration uses `(c1, c2) = (0.5, 8)`, i.e.
`k1 = 8.5`, `k0 = 4.0`, at 40 vehicles. (0.5, 8) and (0.5, 5) are tied on
collisions and completion; (0.5, 8) has the lower QP failure rate, speed
error, and collision rate, and travels farther. Against the former gains it
cuts collisions/km 5.4× (2.71 → 0.50) and raises completion from 6% to 60%.
The cost is about 0.7 m more lateral error and higher jerk (4.2 → 5.8).

The spawn and reset guard keeps `psi1_gain = 2.3`. It does not enter the QP.
A 5-seed replay of the (0.5, 8) cell with `psi1_gain = 2.3` and the reset
guard enforced matched the grid (which used `psi1_gain = 0.5` and no reset
guard) exactly in steps, distance, collisions, and QP failure rate; only the
`psi1_min` diagnostic changes. Setting the spawn guard itself to 0.5 makes
40-vehicle scene construction fail.

## Limits

- One policy and one training seed, trained without a CBF. The CBF-in-the-loop
  ladder variants must be retrained with the new gains before their results
  are comparable.
- 79 of 200 episodes still collide with (0.5, 8). The QP failure rate barely
  changes across good cells; the better gains keep the ego out of sandwich
  states rather than making those states feasible.
- A 2.3-guarded scene does not guarantee the cascade's own first-level
  condition `h_dot + 0.5*h >= 0` at reset, so forward invariance is not
  formally guaranteed from the first step.
