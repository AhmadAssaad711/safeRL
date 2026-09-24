# Trial log

Running ledger of every trial run **after the B and C results landed**
(`artifacts/Results/Evaluation tables/lambda1p0_evaluation_tables.md`, the five-arm
1M/7-seed safety matrix). Newest section last. One row per trial: what changed,
what it measured, and the verdict.

This is a ledger, not an explanation. The source-grounded write-ups live in:

- `docs/congested_d60_experiment_log.md` — CBF gain tuning, constraint cap, action space
- `docs/experiments/hocbf_state_cost_20260915.md` — the C(s) state cost
- `docs/experiments/lateral_emphasis_20260916.md` — wall-hugging, `wx`/`wy` rebalance

Raw run data is under `%LOCALAPPDATA%\Temp\saferl_runs\<out-base>` (gitignored).

**Shared protocol** unless a row says otherwise: `lane-free-v0`, MTM traffic,
380 x 10.2 m periodic road, `vehicles_count 40`, `neighbors_count 7` (41D with
`--drop-ego-speed-slot`), 100 Hz physics / 20 Hz policy / 20 Hz CBF, action box
[-3, 3]^2, `Q1_stable`, 20 envs x 1000, checkpoints every 25k, 200+200 episode
evaluation from seed 1,100,000, strict completion = collision-free 1,000 m
within 3,000 policy steps.

---

## 1. CBF tuning (2026-09-19, d60)

Deployment-time tuning of the hard CBF on the d60 environment, seed 307, B2
policy. Full detail in `congested_d60_experiment_log.md`.

| trial | swept | result |
|---|---|---|
| gain sweep | 31 points, `k0 = wn^2`, `k1 = 2*zeta*wn` | **best (wn 2.0, zeta 1.0) = k0 4.0, k1 4.0 -> 2.482 coll/km, 0.129 QP fail**, against production 2.859 / 0.272. Nine settings beat baseline |
| collision attribution | per-episode QP records | **81-97% of collisions occur on a QP-infeasible step.** Production gains refuse at h = 0.48, well clear of contact; retuning moves refusal to h = 0.056 |
| `max_neighbor_constraints` | n in {3,4,5,6,7,8,10,12} | **Refuted.** Flat from n=4 to n=12 (4% spread). Only n=3 is clearly worse. Lever is dead |
| action space 2x2 | `ax_min` {-3,-6} x `ay` {+/-3,+/-5} | **Axes move in opposite directions.** Lateral widening alone 2.150 coll/km (best to date); longitudinal widening 4.320 (+74% regression). OOD-confounded |
| lateral extension | `ay` in {4,5} | Monotone improvement through +/-5 (2.482 -> 2.296 -> 2.150) |
| retrained at widened box | `ax [-6,3]`, `ay [-4,4]`, 3 seeds, 1M | Launched 09-19; superseded by the nominal line below |

---

## 2. Reward terms

| trial | change | seeds | verdict |
|---|---|---|---|
| state cost `C(s)`, 1M | `ppo_hocbf_state_cost_raw`, distance form, lambda_s 0.1619 | 307-310 | **Does not improve unshielded safety.** +16.1% collisions at n=4, p = 0.068. Never significant |
| lateral emphasis arm A | wall weight 1.0, `wx` 1.0, `wy` 0.65 | 308 | failed — hugging climbed to 88% |
| lateral emphasis arm B | `wy` 2.0 | 308 | failed — past the gradient peak, *worse* lateral error |
| **lateral emphasis arm C** | **`wx` 0.51, `wy` 0.96** | 308 | wall time 77.3% -> 0.2%. Collisions -27.7% but z = -0.68, flagged not reportable |

**Correction found 2026-09-23:** every study after arm C ran `wx = 1.0, wy = 0.96`
— the `wy` half of arm C without the `wx` half. `safety_matrix_1M`, the lambda
studies and the 3M v20 run are all the half-fix, not arm C.

---

## 3. 3M / v_desired 20 (2026-09-22)

Five-arm safety matrix at 3,000,000 steps, 5 seeds, `ego_desired_speed` 20.

| | |
|---|---|
| out-base | `safety_matrix_3M_v20` |
| completed | A1 all 5 seeds; B1 seed 307. Stopped by request mid-B1 |
| A1 CBF-OFF | 4.476 +/- 1.370 coll/km, completion 0.028, jerk 10.07, speed err 4.334 |
| A1 CBF-ON | 0.300 +/- 0.053 coll/km, completion 0.755 |
| **B1 seed 307 jerk** | **70.7 CBF-OFF / 58.4 CBF-ON** against A1's 10.1 / 11.9 |

Jerk finding: not outliers — the whole per-episode distribution shifts (p5 21.5,
median 59.5, p95 96.2). Median command swing 2.98 m/s^2 per 50 ms step against a
+/-3 box, i.e. near bang-bang alternation. Not saturation-driven: two A1 seeds sit
at the bound *more* often with 4-5x less jerk. The `0.02*tanh(jerk/10)` penalty is
flat to 5 decimals at that magnitude.

Also recorded: absolute speed error is 4.3-4.8 m/s across every arm at v_d 20
against 1.9-2.1 at v_d 16. `wx` is not retuned when the `cx` normalizer moves
with `v_desired`.

---

## 4. Nominal line, 1M, 3 seeds (307/327/347), arm C (2026-09-22 to 23)

All `ppo_nominal` only, 1M, seeds 307/327/347, `wx 0.51`, `wy 0.96`.
Paired control `safety_matrix_1M/A1_nominal` at the same seeds.

### 4a. Arm C replication — does not replicate

| KPI | control (wx 1.0) | arm C (wx 0.51) | delta | p |
|---|---:|---:|---:|---:|
| Ego collisions / km | 3.822 +/- 0.717 | 3.501 +/- 0.777 | -8.4% | 0.745 |
| Episode return | 226.1 | 265.8 | +17.5% | 0.183 |
| Lateral tracking err | 1.637 | 1.495 | -8.7% | 0.908 |

Per-seed: 307 -17.1%, 327 -36.1%, 347 **+43.6%**. The n=1 result was -27.7%;
replicated it is -8.4% at p = 0.75. Consistent with the original having been
noise, which `lateral_emphasis_20260916.md` predicted at z = -0.68.

### 4b. Collision-penalty ladder — monotonically worse

Out-bases `nom_1M_armC`, `_cp25`, `_cp60`. CBF-OFF:

| KPI | P=-2.5 | P=-25 | P=-60 |
|---|---:|---:|---:|
| **Ego collisions / km** | **3.501** | 4.915 | **5.413** |
| Completion rate | 0.057 | 0.028 | 0.005 |
| Episode length | 388 | 292 | 286 |
| Lateral tracking err (m) | 1.495 | 1.935 | 3.035 |

**-60 is unanimous**: all three seeds worse than their own arm C run. CBF-ON is
flat (0.277 / 0.263 / 0.324) — the shield masks it entirely.

Theory: avoidance incentive is `gamma * dp * (V - P)` with
`V - P = [r_bar + |P|(1-gamma)] / [(1-gamma) + gamma*p] = 55 + 0.80|P|`. A 720x
larger penalty buys only 26x more incentive because `V` falls as `P` falls. The
GAE half-life is fixed at 11 steps (0.57 s) by `gamma*lambda`; raising `P` only
lifts the tail, buying 1.9 s of reach per decade.

### 4c. Gradient probe — the largest finding of this line

200k, seed 307. `train/grad_*` scalars added this session.

| | P=-2.5 shared | P=-2.5 split | P=-60 shared | P=-60 split |
|---|---:|---:|---:|---:|
| `grad_norm_pi` | 0.913 | 1.064 | 0.823 | 0.826 |
| `grad_norm_vf` | 31.96 | 13.90 | 65.37 | 86.37 |
| **`grad_scale_pi`** | **0.0185** | **0.530** | **0.0095** | **0.659** |
| `grad_clip_active` | 1.0 | 1.0 | 1.0 | 1.0 |

Stock PPO applies one `clip_grad_norm_` over all parameters. With
`net_arch={"pi":[256,128], "vf":[256,128]}` the networks are disjoint, so the
shared budget is the only coupling — and **the clip binds on every update, cutting
the policy gradient to ~1-2% of its magnitude.** This is not a penalty artefact:
at the default `P = -2.5` the scale is 0.0185. **Every result in this repo was
produced with a ~98% attenuated policy gradient.**

### 4d. 2x2 fix factorial at P=-250

| CBF-OFF | MSE+shared | MSE+split | Huber+shared | Huber+split | armC P=-2.5 |
|---|---:|---:|---:|---:|---:|
| Ego collisions / km | 8.399 | 6.104 | 7.322 | 5.199 | **3.501** |
| `grad_scale_pi` | 0.0016 | 0.547 | 0.036 | 0.838 | — |
| `train/value_loss` | 842 | 783 | 107 | 103 | 35.9 |
| `explained_variance` | 0.581 | 0.639 | 0.205 | 0.251 | 0.691 |
| `train/std` | 0.738 | 1.40 | 1.12 | **2.93** | 0.647 |

Both fixes work on the mechanism (8.40 -> 5.20, -38%) but **do not rescue the
outcome** — still 49% worse than P = -2.5. Two costs: Huber wrecks value accuracy
(EV 0.58 -> 0.21), and restoring the gradient inflates exploration noise
(`std` 0.65 -> 2.93 against `exp(-0.5) = 0.607` at init).

One result the other way, CBF-ON, **Huber+split**: 0.176 coll/km (per-seed 0.172 /
0.173 / 0.184) vs 0.277, p = 0.066; completion 0.847 vs 0.775, p = 0.075. Best
shielded result in the study, 3/3 consistent, variance collapsed. No mechanism
offered — A1 never sees the CBF in training. Needs more seeds.

### 4e. Completion by distance — recomputed, no new compute

From existing per-episode records; reproduces the recorded 1,000 m rate exactly.

| CBF-OFF | 250 m | 500 m | 750 m | 1000 m |
|---|---:|---:|---:|---:|
| armC P=-2.5 | 0.393 | 0.205 | 0.108 | 0.057 |
| spread across arms | 0.138 | 0.137 | 0.098 | 0.054 |

| CBF-ON | 250 m | 500 m | 750 m | 1000 m |
|---|---:|---:|---:|---:|
| spread across arms | 0.026 | 0.042 | 0.076 | **0.112** |

**The two modes want opposite distances.** 500 m gives CBF-OFF 2.5x the
discriminating range (1,000 m is a floor effect) but halves CBF-ON's, because the
shielded policy travels 850-895 m and 1,000 m is exactly where it starts failing.
Report both; 1,000 m stays canonical per `AGENTS.md`.

Related finding: **the task distance is an evaluation construct only.** Training
episodes run to `episode_steps = 30000` *physics frames* (6,000 policy steps) and
end on collision. Training never sees the task. But completion is not a speed
failure: 1,000 m in 3,000 steps needs 6.67 m/s, the policy runs 13.7 m/s, and
episodes end at 286 m. **Episodes end because the car crashes, not because it is
slow.**

### 4f. Six-idea sweep (running, 2026-09-23 07:14, ~16:50)

All on arm C + `--separate-grad-clip`, `P = -2.5`. `id1_split` is the control;
each other trial changes one thing on top of it.

| trial | change | question |
|---|---|---|
| `id1_split` | separate grad clip only | Is every baseline undertrained? |
| `id2_gamma999` | `--gamma 0.999` | 5 s horizon against a 67 s task; triples avoidance incentive at no penalty |
| `id3_ent001` | `--ent-coef 0.001` | Companion to the `std` blow-up |
| `id4_field05` | `--potential-field-weight 0.5` | The nominal reward has **zero** per-step neighbour coupling (`wf = 0`, `way = 0`) |
| `id5_effort0343` | `--control-effort-weight 0.343` | Implemented, calibrated, never trained |
| `id6_huber` | `--value-loss huber` | Isolates 4d's CBF-ON result from the penalty |

### 4g. Progress-weight dose curve (queued behind 4f)

`s4_prog050` and `s5_prog150`: `--progress-reward-weight` 0.05 -> 0.5 -> 1.5.
Tests whether rewarding distance rather than time moves completion. An episode
cap matched to the evaluation budget was considered and dropped: episodes end on
collision at 388 policy steps, so a cap at 3,000 never binds.

---

## Code added this session

| change | file | default |
|---|---|---|
| `--separate-grad-clip` | `projected_ppo_cbf.py`, `run_ppo_cbf_progression.py` | off (stock) |
| `train/grad_norm_{pi,vf,total}`, `grad_clip_active`, `grad_scale_pi` | `projected_ppo_cbf.py` | always logged |
| `--value-loss {mse,huber}`, `--huber-delta` | both | `mse`, 10.0 |
| `--ego-desired-speed`, `--actor-lambda`, `--extra-arg` pass-throughs | `run_safety_matrix_study.py` | contract defaults |
| `PYTHONPATH` set for child processes | `run_safety_matrix_study.py` | — |
| `configs/reward_wx051.json` | new | `{"wx": 0.51}` |
