# Congested-traffic (vehicles_count=60) experiment log

Running record of every test on the d60 environment since B2 and C2 were
retrained there. Policy under test unless stated otherwise: **B2**
(`ppo_cbf_nd_reward_actor`), seed 307, 1M timesteps,
`density60_1M/B2_lambda1p0_d60/seed_307`.

All evaluations are 200 complete episodes, CBF-ON (`mode=cbf`), episode seeds
from 1_100_000, unless noted.

## Reference points

| quantity | value | source |
|---|---|---|
| B2 d60 baseline, CBF-ON | 2.859 coll/km, 0.272 QP fail | `density60_1M` study |
| B2 d60 baseline, RAW | 7.473 coll/km | same |
| C2 d60 baseline, CBF-ON | 2.865 coll/km, 0.270 QP fail | same |
| C2 d60, internal projection bypassed | 8.105 coll/km | raw-actor-mean ablation |
| Harness re-measurement of baseline | 2.916 / 0.269 | gain sweep, mid-run check |

Baseline gains: `k0=4.0, k1=8.5` (omega_n=2.0, zeta=2.125), `psi1_gain=2.3`,
`max_neighbor_constraints=7`, action box `ax,ay in [-3,3]`.

---

## 1. C2 raw-actor-mean ablation (complete)

Question: does "raw" mode bypass C2's internal `mu_raw -> mu_safe` projection?
It does not — `mode=raw` only disables the external filter, so the architectural
projection still runs on every `model.predict()`.

| condition | coll/km | completion |
|---|---|---|
| fully unprotected (`raw_actor_mean`) | 8.105 +/- 0.639 | 0.000 |
| standard raw (internal projection on) | 2.865 +/- 0.016 | 0.005 |
| CBF-ON (internal + external) | 2.865 +/- 0.033 | 0.0025 |

Almost all of C2's safety comes from the internal projection; the external
filter adds essentially nothing on top (intervention rate ~3e-6).

## 2. HOCBF gain sweep, second-order parameterisation (complete)

31 points, `k0 = omega_n^2`, `k1 = 2*zeta*omega_n`.
Output: `%LOCALAPPDATA%\Temp\saferl_runs\cbf_congested_tuning`.

**Best: (omega_n=2.0, zeta=1.0) = k0=4.0, k1=4.0 -> 2.482 coll/km, 0.129 QP
fail, 995-step episodes.** Same k0 as production; only k1 changes 8.5 -> 4.0.
14.9% fewer collisions at 52% less QP failure. Nine settings beat baseline.

Findings:
- Collisions U-shaped in zeta in all five columns; episode length inverted-U.
- Minimum drifts overdamped as omega_n rises: zeta=1.0 for omega_n in
  {1.0, 1.4, 2.0}, zeta=1.4 at 2.8, zeta=2.125 at 4.0. "zeta=1 always" is NOT
  supported.
- omega_n optimum is interior (~2.0-2.8), not at a grid edge.
- Pole identification (fit never uses k0/k1) holds to ~10% for
  `omega_n*dt <= 0.14` and breaks at `>= 0.20`, as pre-registered. 20/31 points
  within 25%. The omega_n=8 probe returns a0=4.06 against a placed k0=64.
- REFUTED: effort is monotone in zeta, not U-shaped; intervention rate falls
  with omega_n rather than rising.
- Limitation: the governed regime (feasible AND binding AND unsaturated) is
  only 2-14% of steps; four points rest on under 50 fit samples.

## 3. Collision attribution (complete)

Per-episode records, `collision_events_direct_qp_failure_same_step`.

| setting | coll/km | % of collisions on a QP-infeasible step |
|---|---|---|
| (2.0, 1.0) best | 2.449 | 81.5% |
| (2.0, 2.125) production | 2.855 | 92.5% |
| (4.0, 2.125) | 2.560 | 96.9% |
| (2.0, 0.5) | 3.982 | 47.0% |

**81-97% of collisions occur where the QP had no feasible solution.** Gain
tuning only helps via its effect on feasibility. If feasibility were solved,
the arithmetic implies ~0.45 coll/km.

Median barrier value when the QP goes infeasible:

| setting | QP fail | h at failure | h at success |
|---|---|---|---|
| (2.0, 2.125) production | 0.348 | 0.480 | 2.559 |
| (2.0, 1.0) best | 0.109 | 0.056 | 0.910 |
| (1.0, 3.0) | 0.754 | 1.736 | 1.564 |

Production gains refuse at h=0.48 — well clear of contact, i.e. self-inflicted
conservatism. Retuning moves refusal to h=0.056, essentially at the boundary.
Residual: 57.8% of failures still have h>0 and 11.4% have h>1.0, which no
(k0,k1) pair removed — attributable to the linear class-K structure.

NOTE / correction: `actuator_clip_norm` is 0 at failure in every setting, but
this does NOT rule out the action box. It measures whether the *executed*
action was clipped, and the infeasible fallback returns something inside the
box by construction. The box remains a live suspect.

## 4. max_neighbor_constraints sweep (COMPLETE — hypothesis refuted)

Gains fixed at k0=4.0, k1=4.0. n in {3,4,5,6,7,8,10,12}. n=7 is the control and
must reproduce 2.482 / 0.129. Constraint system is n+6 rows against a padded
capacity of 18, so n<=12.
Output: `%LOCALAPPDATA%\Temp\saferl_runs\cbf_congested_mnc`.

| n | coll/km | QP fail | ep. len |
|---|---|---|---|
| 3 | 2.724 | 0.115 | 895 |
| 4 | 2.544 | 0.122 | 969 |
| 5 | 2.529 | 0.128 | 985 |
| 6 | 2.446 | 0.130 | 1013 |
| 7 (control) | 2.482 | 0.129 | 995 |
| 8 | 2.488 | 0.133 | 992 |
| 10 | 2.515 | 0.131 | 984 |
| 12 | 2.511 | 0.131 | 988 |

Control reproduced 2.482 / 0.129 exactly.

**RESULT: the cap is not the binding limit.** Flat from n=4 to n=12 — a 4%
spread on collisions and 1.8 points of QP failure across a 3x change in
constraint count. n=6's 2.446 is 1.4% better than n=7, inside the ~2%
reproducibility of the harness. Only n=3 is clearly worse, from missing the
vehicle that actually hits. This lever is dead.

## 5. Action-space sweep (2x2 COMPLETE — lateral is the binding constraint)

| ax_min | ay | corner | coll/km | QP fail | interv | ep. len |
|---|---|---|---|---|---|---|
| -3 | +/-3 (control) | 4.24 | 2.482 | 0.129 | 0.332 | 995 |
| **-3** | **+/-5** | 5.83 | **2.150** | 0.140 | 0.309 | **1022** |
| -6 | +/-3 | 6.71 | 4.320 | 0.163 | 0.406 | 762 |
| -6 | +/-5 | 7.81 | 3.698 | 0.169 | 0.392 | 850 |

Control reproduced 2.482 / 0.129 exactly.

**The two axes move in OPPOSITE directions.** Lateral widening alone gives
2.150 coll/km — the best result to date, 13.4% under the retuned gains and
24.8% under the production baseline of 2.859. Longitudinal widening is
harmful: 4.320, a 74% regression, with episodes collapsing 995 -> 762.

The longitudinal damage is the OOD confound behaving as predicted (braking
commands double). Crucially, the confound is a headwind for BOTH axes — ay=5
also inflates lateral commands by 67% — and lateral improved anyway, so the
true lateral benefit is likely LARGER than 13.4%, not smaller.

Consistent with the geometry: the safe-set ellipse is 5.09 m longitudinal
against 2.55 m lateral, so sideways displacement clears a conflict cheaply,
and at ay=3 arresting a fast lateral closing needs more room than the road has.

Caveat: QP failure rose slightly in every widened case (0.129 -> 0.140), so the
box widening did not reduce infeasibility as measured; the OOD state shift
masks any geometric gain. Collisions improved regardless.

### 5b. Lateral extension (running)

ay in {4, 6, 7} at fixed ax_min=-3, to find whether +/-5 is the optimum.
Appends to the same summary.csv; log `sweep_lateral.log`.

## 5-old. Action-space sweep setup notes

2x2 factorial at k0=4.0, k1=4.0: `ax_min in {-3, -6}` x `ay in {+/-3, +/-5}`,
with (-3, 3) as the control. `ax_max` stays +3 (asymmetric: cars brake harder
than they accelerate). Corners exceeding an 8.0 m/s^2 friction budget are
rejected — (-6, +/-6) is 8.49 and was skipped.
Output: `%LOCALAPPDATA%\Temp\saferl_runs\cbf_congested_bounds`.

**Known confound, must be stated with any result.**
`validate_matching_physical_action_bounds` requires the CBF box and the
simulator box to be identical (a filter certifying actions the actuator cannot
execute would be unsound), so the box cannot be widened filter-side only.
Widening it also widens what the policy can command, because the normalized
action maps onto the box — so at ax_min=-6 the policy drives roughly twice as
hard and is out of distribution. A 6-episode smoke at (-6, +/-5) gave 10.36
coll/km and QP failure *rising* to 0.189.

Therefore: collisions from this sweep are confounded. The decisive clean test
needs a `--policy-action-scale` control that holds the policy's physical
command fixed while the QP box widens. Not yet built.

### 5c. Lateral extension results

| ax_min | ay | coll/km | QP fail | ep. len |
|---|---|---|---|---|
| -3 | +/-3 | 2.482 | 0.129 | 995 |
| -3 | +/-4 | 2.296 | 0.122 | 1021 |
| -3 | +/-5 | 2.150 | 0.140 | 1022 |
| -3 | +/-6, +/-7 | (pending) | | |

Monotone improvement through +/-5; had not turned over when the training
study was launched.

---

## 6. Retrained study at the widened action space (RUNNING, overnight)

Motivation: every action-space number above comes from evaluating a policy
trained at +/-3 under a wider box, so the policy is out of distribution. The
hypothesis under test is that an agent TRAINED inside the wider box learns to
use the extra lateral authority, which the frozen-policy probe cannot show.

Output: `%LOCALAPPDATA%\Temp\saferl_runs\d60_ax6_ay4_1M\{A1_nominal,
B2_nd_reward_actor, C2_projected}`. Launched 2026-09-19 ~21:05, ~5 h.

An earlier launch at `ax [-3,+3] / ay [-5,+5]` was killed after ~10 minutes
and its output deleted; superseded by the bounds below.

Configuration:
- **Action space, BOTH axes widened from the density60_1M study:**
  `ax in [-6, +3]` (was [-3,+3]) and `ay in [-4, +4]` (was [-3,+3]).
  Corner sqrt(6^2+4^2) = 7.21 m/s^2 ~ 0.74g, inside the dry-asphalt limit
  but beyond wet grip (~5-6).
- Physical justification for ay=+/-4 specifically: arresting the maximum
  representable lateral speed (observation_vymax = 7.2 m/s) needs
  v^2/(2a) of road -- 8.64 m at a=3, which EXCEEDS the 8.4 m of usable
  road width, versus 6.48 m at a=4. So +/-4 is the smallest bound at which
  a worst-case lateral closing can be arrested inside the road at all.
  It is also the bottom of the real-world "evasive" band (4-6 m/s^2) and
  keeps the steering-lag overclaim smaller than +/-5 would.
- ax=-6 ~ 0.61g is firm emergency braking (dry limit is 8-9); +3 is
  engine-limited and realistic. The asymmetry matches real vehicles.
- **CONSEQUENCE: two variables moved at once.** If this study improves on
  the baseline, it will not isolate whether the lateral or the longitudinal
  widening did it. The frozen-policy probe hinted lateral helps and
  longitudinal hurts, but that probe is OOD and is not valid evidence for a
  retrained agent.
- CBF gains left at the STUDY values k0=4.0, k1=8.5, psi1_gain=2.3, n=7 --
  deliberately NOT the retuned k1=4.0, so the action space is a single
  variable against B2=2.859 / C2=2.865.
- 3 seeds each: 307, 327, 347 (spaced by 20; VecEnv assigns worker seeds
  s..s+19, so closer spacing would share scenario streams).
- 1M timesteps, Q1_stable, n_envs=10 per arm, 3 arms concurrent.
- B2: `--lambda-detached-actor 1.0`; C2: `--lambda-mean 1.0` plus
  `--raw-actor-eval` (200 ep) for the true-unprotected number.
- 200-episode post-train eval, seeds from 1_100_000.

Baselines to compare against (same env, ay=+/-3, same gains):
  A1 d40 raw 4.036 +/- 0.760 (no d60 A1 baseline exists yet)
  B2 d60: 2.859 coll/km CBF-ON, 7.473 raw
  C2 d60: 2.865 coll/km CBF-ON, 8.105 with internal projection bypassed

Watch item: `log_std_init` stays at -0.5, so lateral exploration noise is now
+/-3 m/s^2 rather than +/-1.8. If an arm trains unstably that is the first
suspect, not the action space itself.

Decision recorded: widening `ax` too (asymmetric braking, e.g. [-6, +3]) was
considered and deferred. The 2x2 result against it (4.320) is OOD-confounded
and therefore NOT valid evidence for a retrained agent; the only sound reason
for deferring is to keep lateral as a single variable.

---

## Open items

- Confirm the best gains (k1=4.0) on seed 327 and on C2/A1; currently seed 307,
  B2 only, deployment-time only (no retraining under the new gain).
- Four pole fits rest on under 50 binding samples; a trace-only rerun with more
  episodes is cheap (6 trace episodes vs the KPI stage's 200).
- Nonlinear class-K alpha(h) is the only candidate identified for the residual
  conservatism at h>1.0; not tested.
- Discrete-time pole parameterisation: partially wired into the sweep script
  (`gains_from_discrete_poles`, `--z-slow`, `--z-fast-ratio`) but NOT connected
  to grid construction, so those flags are currently inert.
