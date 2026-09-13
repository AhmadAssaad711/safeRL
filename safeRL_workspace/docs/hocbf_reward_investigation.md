# HOCBF reward term investigation

Goal: make the direct second-order HOCBF reward term (`ppo_hocbf_reward_raw`)
produce a real safety effect on the **unshielded** policy, without degrading
driving quality.

All runs here use the same contract as the completed 250k neighbour-7 study:
mtm traffic, 40 vehicles, 100/20/20, Q1_stable, 20 envs x 1000 steps, field
lateral target, `neighbors_count=7`, `--max-neighbor-constraints 7`, 250k
timesteps, 200+200 post-train evaluation from seed 1,100,000.

## Baselines (from the completed study, 3 seeds each)

| KPI (mean over 3 seeds) | A1 `ppo_nominal` | B1 `ppo_hocbf_reward_raw` |
| --- | --- | --- |
| Raw collisions / km | 7.813 | 7.430 |
| Raw return | 118.069 | 118.380 |
| Raw completion | 0.002 | 0.003 |
| Raw lateral err (m) | 1.978 | 2.218 |
| CBF-ON collisions / km | 0.300 | 0.312 |
| CBF-ON return | 766.689 | 751.526 |
| CBF-ON completion | 0.748 | 0.743 |
| CBF-ON lateral err (m) | 2.005 | 2.103 |
| CBF-ON intervention rate | 0.873 | 0.896 |

B1 is the treatment; it is indistinguishable from the control on safety and
slightly worse on driving quality. That is the defect under investigation.

## Success criteria

Primary (the term's actual purpose):
- raw (CBF-OFF) collisions/km materially below nominal's 7.813; treat >= 20%
  reduction as a candidate signal, confirm on >= 2 further seeds before
  believing it.

Guardrails - a candidate is void if it degrades these vs nominal:
- CBF-ON completion and return;
- abs speed error, mean jerk;
- **sideline hugging**: mean/min ego distance to road boundary, fraction of
  steps within 1.0 m of a boundary;
- **crawling to let traffic pass**: mean ego speed, speed deficit vs target,
  policy steps required per metre of progress.

## Method

Phase 0: quantify why the term is inert (no training).
Phase 1: sweep existing CLI knobs only - `--hocbf-reward-lambda`,
         `--hocbf-reward-margin`, `--hocbf-psi-scale`. No code change.
Phase 2 (only if needed): change the functional form as a **new** variant;
         do not mutate `ppo_hocbf_reward_raw`, so the completed 21-run study
         stays comparable.

---

## Log

### 2026-09-13 ~23:15 - Phase 0 opened

Static reading of `scripts/common/ppo_cbf_env.py:772-784` gives the term as

```
penalty = hocbf_reward_lambda * (max(0, hocbf_reward_margin - psi2) / hocbf_reward_scale) ** 2
```

with the values B1 actually trained under:

- `hocbf_reward_lambda` = 1.0 (argparse default, `run_ppo_cbf_progression.py:4033`)
- `hocbf_reward_margin` = 0.0 (argparse default, `:4052`)
- `hocbf_reward_scale`  = 314.8163127575424 (calibrated p95 residual, frozen
  across B1 seeds per AGENTS.md)

Two suspected compounding defects:

1. `margin = 0` means the term only activates once psi2 is already negative,
   i.e. after the second-order barrier condition is already violated. There is
   no gradient pressure while merely *approaching* danger.
2. `scale = 314.82` inside a squared denominator crushes the magnitude. A psi2
   violation of 30 physical units costs `(30/314.82)^2 = 0.0091` per step,
   against a typical per-step environment reward near 0.8 (raw return ~118 over
   ~150 steps). The safety term would be ~1% of the reward signal.

Next: measure the real per-step distribution of psi2, the hinge violation, and
the resulting penalty on a trained B1 checkpoint, to confirm (1) and (2)
quantitatively rather than by inspection.

### 2026-09-13 ~23:30 - Phase 0 measurement (10 deterministic episodes each, seed 1,100,000+)

Probe rolls each checkpoint through the training wrapper with identical HOCBF
accounting (lambda=1.0, scale=314.816, margin=0.0) and records the per-step
term alongside guardrail signals.

| | A1 `ppo_nominal` | B1 `ppo_hocbf_reward_raw` |
| --- | --- | --- |
| mean episode steps | 175.0 | **345.9** |
| psi2 fraction negative | 0.895 | 0.950 |
| psi2 median | -4.405 | **-1.712** |
| hinge violation median (when active) | 5.778 | **1.896** |
| `min_h` p05 | -0.283 | **-0.147** |
| penalty mean | 0.00720 | 0.00446 |
| base reward mean abs | 0.526 | 0.577 |
| **penalty share of reward** | **1.37%** | **0.77%** |
| mean ego y (road centre = 5.1) | 5.266 | **8.182** |
| mean boundary distance | 0.839 | **0.440** |
| steps within 1 m of an edge | 73.0% | **84.2%** |
| mean speed (target 16.0) | 12.158 | 13.909 |

Three conclusions, two of them not what I expected:

1. **Magnitude defect confirmed.** The term is 0.77-1.37% of the reward signal.
   It cannot compete with progress/tracking terms. This part of the static
   diagnosis holds.

2. **The term is NOT sparse - it is nearly always on, and it does move its own
   objective.** psi2 is negative 89-95% of the time for *both* policies, so the
   `margin=0` hinge fires almost every step rather than rarely. And B1 really
   did improve the quantity it optimises: median psi2 -4.41 -> -1.71, median
   violation 5.78 -> 1.90, `min_h` p05 -0.283 -> -0.147, and it survives twice
   as long before crashing (175 -> 346 steps). So the term is not ignored; it
   is *optimised and still does not buy collision safety*. That reframes the
   problem: the defect is not only magnitude, it is that psi2-violation is a
   poor proxy for collision risk as currently reduced.

   Corollary: because psi2 < 0 ~90% of the time, the barrier condition is
   effectively unsatisfiable under the action bounds, so the term is a
   permanent tax with a gradient, never an achievable target. It can only say
   "violate less", never "you are safe now".

3. **The term induces exactly the degenerate behaviour we were told to watch
   for.** B1 sits at mean y = 8.18 on a 10.2 m road (centre 5.1) versus the
   nominal's 5.27, with mean boundary distance 0.44 m vs 0.84 m and 84% of
   steps within 1 m of an edge. B1 learned to **hug the sideline**, which is a
   cheap way to reduce psi2 violations - fewer neighbours to violate
   constraints against - without driving any more safely. This also explains
   B1's worse lateral tracking error in the study (2.10 vs 2.01 m).

So the fix must satisfy three things at once, not just one: raise the
magnitude, make the signal *informative about collision risk* rather than about
neighbour count, and make edge-hugging a non-solution.

Caveat on the guardrail numbers: `frac_steps_within_1m_of_edge` is high (0.73)
even for the nominal, and mean-of-min is not min-of-mean, so the absolute level
needs the y *distribution* rather than its mean. Adding percentiles next. The
A1-vs-B1 *comparison* is what carries the finding, and it is unambiguous.

### Next: functional analysis before any 250k trial

Per instruction, no training run until a candidate term is characterised
offline. Building a harness that scores candidate penalty functions on a fixed
empirical rollout trace:

- **magnitude**: mean penalty as a share of base reward (target ~10-30%, not 1%);
- **dynamic range**: p95/median ratio - does it discriminate states at all;
- **informativeness**: does the term's value predict a collision in the next
  K steps (AUC), which is the property that actually matters and the one the
  current term appears to lack;
- **degeneracy**: does the term's value fall when the ego moves toward a road
  edge - if it does, edge-hugging is a way to farm it, and the candidate is
  rejected.

### 2026-09-13 ~23:35 - functional analysis result

Trace: 30 deterministic episodes of the A1 nominal checkpoint, 4070 policy
steps, every episode ending in a collision. A step is labelled positive if it
is within 15 steps of the terminal collision (11.8% of steps). AUC is
`P(term ranks a pre-collision step above a non-pre-collision step)`.

**Stage 1 - which quantity is informative.** AUC is invariant under monotone
transforms, so this stage answers "what should the term be built from", and
tuning lambda/scale/margin cannot change any of it:

| quantity | AUC | edge corr |
| --- | --- | --- |
| dense sum `exp(-(h + 0.5*h_dot)/0.25)` | **0.950** | -0.042 |
| `-min(h + 0.5*h_dot)` | 0.948 | -0.726 |
| `-min(h + 1.0*h_dot)` | 0.900 | -0.472 |
| dense sum `exp(-h/1.0)` (no rate) | 0.747 | -0.677 |
| **`-min psi2 slack` (what the term uses today)** | **0.722** | -0.050 |
| `-min_h` (no rate) | 0.676 | -0.841 |

The decisive variable is the **rate term**. A half-second lookahead
`h + 0.5*h_dot` lifts AUC from 0.68 (position only) to 0.95. The psi2 residual
the current term optimises sits at 0.72, below even a crude predicted barrier.

This also explains the Phase 0 paradox - B1 measurably improved psi2 yet gained
no collision safety. It was optimising a mediocre proxy. **No setting of
`--hocbf-reward-lambda`, `--hocbf-psi-scale` or `--hocbf-reward-margin` can fix
this**, because all three are monotone rescalings that leave the ranking, and
therefore the AUC, exactly unchanged. Phase 1 as originally planned is
therefore dead on arrival for informativeness; it can only fix magnitude. Going
to Phase 2.

**Stage 2 - shape.** An unbounded `exp` aggregation reaches AUC 0.949 but is
spiky (p95/median = 20) and weakly edge-protective. A bounded sigmoid
aggregation is the better training signal:

| shape | AUC | edge corr | p95/med | max penalty |
| --- | --- | --- | --- | --- |
| `exp(cap 20), tau=0.5, w=0.25` | 0.949 | -0.171 | 20.0 | 1.302 |
| **`sigmoid, tau=0.5, w=0.25`** | **0.925** | **-0.547** | **2.94** | **0.377** |

Chosen candidate:

```
penalty = lambda * sum_i sigmoid( -(h_i + tau*h_dot_i) / w )
          over vehicle barriers AND the two road-boundary barriers
          tau = 0.5 s, w = 0.25, lambda = 0.154  (-> ~15% of base reward)
```

Each barrier contributes at most 1.0, so the term is bounded by the barrier
count regardless of geometry - it cannot produce the spike that would
destabilise a PPO update. The negative edge correlation (-0.547) means it
*discourages* rather than rewards sideline hugging, because the road-boundary
barriers are inside the sum.

### 2026-09-13 ~23:40 - behavioural test cases (before any training)

Hand-built geometries, evaluated on both terms.

| scenario | candidate | current term |
| --- | --- | --- |
| empty road, centred | 0.0000 | 0.000000 |
| vehicle 8 m ahead, same speed | 0.0004 | 0.000000 |
| vehicle 8 m ahead, closing 6 m/s | 0.1266 | 0.005258 |
| vehicle 8 m ahead, receding 6 m/s | 0.0000 | 0.000000 |
| vehicle 4 m ahead, closing 6 m/s | 0.1532 | **0.002120** |
| side vehicle 2.2 m, closing | 0.1459 | 0.000785 |
| side vehicle 2.2 m, separating | 0.0362 | 0.000000 |
| near left edge, drifting in | 0.1063 | 0.000538 |
| near left edge, drifting out | 0.0060 | 0.000000 |
| edge + traffic, closing | 0.2329 | 0.005116 |

Candidate passes all seven ordering checks. The current term fails two of four.

*Correction to a claim I made an hour ago in this log*: I first wrote that the
current term is blind to the road boundary. That was an artefact of my test
harness calling `batch_pairwise_hocbf_constraints` directly, which returns
vehicle rows only. The environment builds its rows through
`build_cbf_action_constraints`, which **does** append the two boundary rows
(`cbf_ray_mask.py:156-159`). Re-run against the faithful row set, the boundary
case scores 0.000538 rather than 0. The term sees the wall - just ~10x more
weakly than a closing vehicle (0.000538 vs 0.005258), which is a weighting
problem rather than blindness. The edge-hugging mechanism is therefore the
*neighbour-count* effect (fewer vehicle rows near the edge) only weakly opposed
by the boundary row, not an absent boundary term.

The surviving failures are still diagnostic:

1. **It is non-monotone in distance** - a vehicle 4 m ahead closing at 6 m/s
   scores 0.00212, *less than half* the same vehicle at 8 m (0.00526). Closing
   the gap lowers the penalty, which is backwards for a safety term.
2. **It is exactly zero for a vehicle 8 m ahead at matched speed**, so it
   cannot distinguish steady following from an opening gap, and cannot
   distinguish either from an empty road.

Next: implement the candidate as a **new** variant (leaving
`ppo_hocbf_reward_raw` untouched so the completed 21-run study stays valid),
then the first paired 250k trial against the existing A1 nominal control.

### 2026-09-13 23:40 - implementation

New variant `ppo_hocbf_dense_raw` (commit 0e02b5e), `execution_mode="box"` like
B1 so it remains a pure reward-shaping treatment with no CBF in training:

- `cbf_ray_mask.build_cbf_action_constraints` now also returns `barrier_h` and
  `barrier_h_dot` (vehicles plus both road boundaries). These were already
  computed and discarded; only the minimum was kept.
- `CBFContextPhysicalActionWrapper._dense_hocbf_penalty` implements
  `lambda * sum_i sigmoid(-(h_i + tau*h_dot_i)/w)`.
- The penalty is evaluated on the constraint system **after** the step, so it
  credits the action that produced the state rather than the state the action
  was taken from. That system was already being built for the next
  observation, so this costs no extra geometry.
- Evaluation environments leave `lambda=0`, so post-training KPIs measure
  driving and safety only and stay directly comparable with every other
  variant's numbers, including the completed study's.
- New CLI: `--dense-hocbf-lambda` (default 0.154), `--dense-hocbf-tau` (0.5),
  `--dense-hocbf-w` (0.25). Unavailable `h_dot` raises rather than silently
  zeroing the safety term.

Smoke test in the real environment on the A1 checkpoint, 400 steps:

| lambda | penalty mean | p95 | max | share of reward |
| --- | --- | --- | --- | --- |
| 0.0 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| 0.154 | 0.0729 | 0.2310 | 0.2955 | 0.225 |

Clean off-switch at lambda=0, and at 0.154 the term is 22.5% of the reward with
a bounded maximum of 0.30 - the intended regime. A 4,000-step end-to-end
training run completed and produced a valid manifest and KPI table, confirming
the variant works through the worker/vectorised path, not just in-process.

Launcher bugs hit and fixed on the way (all mine, none in the repo): PowerShell
`-File` does not parse array parameters, native-argument quote stripping
mangled `--env-config-json`, and `Out-File -Encoding utf8` on PowerShell 5.1
writes a BOM that `json.loads` rejects. Switched to a `--env-config-file`
written BOM-free.

### 2026-09-13 23:45 - trials T1-T3 launched

Seed 307, 250k, everything else identical to the study contract. Control is the
existing A1 nominal seed 307. Sweeping only magnitude, since the shape is what
the offline analysis already selected:

| trial | lambda | expected share |
| --- | --- | --- |
| T1 | 0.154 | ~22% |
| T2 | 0.30 | ~44% |
| T3 | 0.08 | ~12% |

### 2026-09-14 00:10 - T1 result (lambda=0.154, seed 307)

All three columns are seed 307 only, 200+200 post-training episodes, plus a
12-episode behaviour probe. Bold marks an improvement over the A1 control.

**CBF-OFF (raw) - the primary target:**

| KPI | A1 nominal | B1 hocbf | T1 dense |
| --- | --- | --- | --- |
| Ego collisions / km | 7.679 | 7.050 | **6.247** |
| Episode return | 121.806 | 117.049 | **141.761** |
| Mean lateral tracking err (m) | 2.102 | 2.167 | **1.621** |
| Abs speed error (m/s) | 5.105 | 3.132 | **3.087** |
| Mean jerk norm | 1.112 | 1.306 | 2.005 |

**CBF-ON:**

| KPI | A1 nominal | B1 hocbf | T1 dense |
| --- | --- | --- | --- |
| Completion | 0.785 | 0.795 | **0.805** |
| Ego collisions / km | 0.251 | 0.242 | **0.233** |
| Episode return | 779.439 | 778.501 | **813.790** |
| Mean lateral tracking err (m) | 2.083 | 2.046 | **1.494** |
| Intervention rate | 0.957 | 0.895 | **0.757** |
| Minimum h | -0.427 | -0.444 | **-0.360** |
| Mean jerk norm | 4.173 | 4.323 | 4.612 |

**Behaviour guardrails (12-episode probe):**

| signal | A1 nominal | B1 hocbf | T1 dense |
| --- | --- | --- | --- |
| ego y median (centre = 5.1) | 6.140 | **9.300** | **5.732** |
| steps within 0.5 m of an edge | 70.5% | 83.4% | **59.7%** |
| mean speed (target 16.0) | 12.575 | 14.146 | **15.273** |
| speed deficit | 3.425 | 1.854 | **0.727** |

Reading:

- **Raw collisions/km fall 18.6%** (7.679 -> 6.247), just under the 20% bar I
  set, and in the right direction for the first configuration tried. B1 managed
  8.2% on the same seed.
- **The shield intervenes far less** (0.957 -> 0.757) and `min_h` improves
  (-0.427 -> -0.360), i.e. the policy is making choices that need less
  correcting, which is exactly the effect the term was supposed to have and
  never did.
- **Both named degeneracies moved the right way, not the wrong way.** B1's
  median y is pinned at **9.300**, the road edge itself - the edge-hugging is
  even starker per-seed than the means suggested. T1 sits at 5.732 against a
  centre of 5.1 and spends *less* time near an edge than the nominal (59.7% vs
  70.5%). And it does not crawl: mean speed 15.27 of a 16.0 target, the best of
  the three, with the speed deficit cut from 3.43 to 0.73.
- Driving quality improves rather than degrades: lateral tracking error 2.08 ->
  1.49 and return 779 -> 814 with the shield on.

**The one regression is jerk**: 4.173 -> 4.612 with the shield on (+11%) and
1.112 -> 2.005 raw (+80%). The term buys its safety partly with more reactive
steering. Worth watching in the lambda sweep - if jerk scales with lambda while
the safety gain saturates, a lower lambda is the better operating point.

Single seed, so this is a candidate signal, not a confirmed result. Seeds 308
and 309 required before believing it.

### 2026-09-14 00:40 - lambda sweep complete (all seed 307)

| lambda | raw coll/km | CBF-ON completion | CBF-ON coll/km | CBF-ON return | intervention | CBF-ON jerk | ego y median | mean speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (A1 control) | 7.679 | 0.785 | 0.251 | 779.4 | 0.957 | 4.173 | 6.14 | 12.58 |
| 0.08 | 6.336 | 0.765 | 0.282 | 781.6 | 0.807 | 4.289 | **1.74** | 14.96 |
| **0.154** | **6.247** | **0.805** | **0.233** | **813.8** | **0.757** | 4.612 | **5.73** | 15.27 |
| 0.30 | 7.663 | 0.740 | 0.310 | 748.7 | 0.895 | 4.820 | **9.30** | 16.96 |

The response is an inverted U, not monotone, and 0.154 is the optimum on
essentially every axis. That shape is itself the interesting finding:

- **At lambda=0.30 the safety benefit disappears entirely** (7.663 raw
  collisions/km, statistically the control's 7.679) *and* the degeneracies come
  back hard: median y pinned at **9.30**, the road edge, and mean speed 16.96
  against a 16.0 target, i.e. it now speeds. Over-weighting the term makes the
  policy game it instead of drive well.
- **Why it can still be gamed at high lambda**: the aggregation is a *sum* over
  barriers, and each barrier contributes at most 1.0. Moving to the edge sheds
  several vehicle barriers while adding at most one boundary barrier, so in
  dense traffic the trade is net-favourable once lambda is large enough to
  dominate the task reward. The boundary term cannot outweigh several shed
  neighbours. A max/softmax aggregation, or boundary barriers weighted above
  vehicle ones, would close this; worth testing if time allows.
- lambda=0.08 gets most of the raw-collision benefit (6.336) with less jerk
  (4.289) but is worse with the shield on across completion, collisions,
  tracking and intervention. It also drifts to the *other* edge (median y 1.74).

Operating point selected: **lambda=0.154, tau=0.5, w=0.25**. Confirmation runs
on seeds 308 and 309 launched at 00:38.

### 2026-09-14 01:20 - THREE-SEED RESULT (lambda=0.154, tau=0.5, w=0.25)

Mean +/- sd over seeds 307/308/309, 200+200 evaluation episodes each.

**CBF-OFF (raw) - the primary target:**

| KPI | A1 nominal | B1 hocbf (old term) | dense term |
| --- | --- | --- | --- |
| **Ego collisions / km** | 7.813 +- 0.700 | 7.430 +- 0.689 | **6.686 +- 0.342** |
| Episode return | 118.07 +- 11.94 | 118.38 +- 4.94 | **134.77 +- 9.62** |
| Mean lateral tracking err (m) | 1.978 +- 0.133 | 2.218 +- 0.095 | **1.617 +- 0.047** |
| Abs speed error (m/s) | 4.699 +- 0.450 | 4.703 +- 2.261 | **3.620 +- 1.099** |
| Mean jerk norm | 1.887 +- 0.788 | **1.361 +- 0.447** | 1.911 +- 0.429 |
| Episode length (steps) | 217.2 +- 18.9 | 222.5 +- 37.8 | **227.4 +- 29.8** |

**CBF-ON:**

| KPI | A1 nominal | B1 hocbf (old term) | dense term |
| --- | --- | --- | --- |
| Ego collisions / km | 0.300 +- 0.042 | 0.312 +- 0.066 | **0.271 +- 0.047** |
| Completion | 0.748 +- 0.033 | 0.743 +- 0.050 | **0.775 +- 0.039** |
| Episode return | 766.7 +- 9.1 | 751.5 +- 19.2 | **800.7 +- 17.5** |
| Mean lateral tracking err (m) | 2.005 +- 0.134 | 2.103 +- 0.049 | **1.526 +- 0.025** |
| **Intervention rate** | 0.873 +- 0.090 | 0.896 +- 0.018 | **0.722 +- 0.040** |
| Minimum h | -0.431 +- 0.019 | -0.443 +- 0.004 | **-0.413 +- 0.038** |
| Mean jerk norm | 4.704 +- 0.528 | 4.520 +- 0.255 | 4.720 +- 0.104 |

Per-seed on the primary metric:

| seed | A1 | dense | change |
| --- | --- | --- | --- |
| 307 | 7.679 | 6.247 | **-18.7%** |
| 308 | 7.030 | 7.083 | +0.7% |
| 309 | 8.729 | 6.729 | **-22.9%** |
| mean | 7.813 | 6.686 | **-14.4%** |

**Assessment - a real effect, short of the bar I set.**

- Raw collisions/km fall **14.4%** where the old term managed 4.9%. Two seeds
  improve ~20%, one is flat. That is below the >=20% I pre-registered, so by my
  own criterion this is a *candidate*, not a confirmed fix. It is, however, the
  first version of this term with any measurable safety effect at all, and its
  seed spread is tighter than the control's (+-0.342 vs +-0.700).
- **Shield intervention drops 17%** (0.873 -> 0.722) and `min_h` improves. The
  policy demonstrably needs less correcting - the property the term was always
  supposed to produce.
- **Driving quality improves rather than degrades**, which was the explicit
  constraint: completion +2.7 pp, return +34, and lateral tracking error 24%
  better (2.005 -> 1.526 m), the largest single improvement in the table.
- **The seed-307 jerk regression did not survive replication**: 4.704 -> 4.720
  with the shield on, i.e. flat. I called that out as the main worry after T1;
  on three seeds it is noise.

**Guardrails, and one genuine failure.** Averaged over the probe episodes,
time within 0.5 m of an edge falls 65.4% -> 48.5% and mean speed is 13.74 ->
14.01, so neither degeneracy appears on average - and the old term's
edge-hugging (median y pinned at 9.30) is gone. **But seed 309 crawls**: mean
speed 13.07 -> 11.02 and the speed deficit nearly doubles to 4.98. That is
exactly the "slowing down to let traffic past" failure to watch for, and it
appears on the same seed that gave the *best* collision improvement (-22.9%).
Some of the safety gain on that seed is bought by driving slower, which is not
an acceptable trade and is not visible in the collisions/km number alone.

So: the term now works, the mechanism is right, and it does not induce
edge-hugging - but on one seed of three it buys safety with speed, and the mean
effect is 14% rather than the 20% I set as the bar.

### 2026-09-14 01:40 - tau sweep (lambda=0.154, seed 307)

| tau | raw coll/km | CBF-ON completion | CBF-ON coll/km | CBF-ON return | intervention | steps near edge | mean speed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.35 | **6.226** | 0.770 | 0.279 | 784.3 | 0.779 | 61.7% | **15.97** |
| **0.50** | 6.247 | **0.805** | **0.233** | 813.8 | 0.757 | 59.7% | 15.27 |
| 0.75 | 8.596 | 0.805 | 0.232 | **818.3** | **0.605** | **35.6%** | 12.28 |

A clean conservatism gradient. Longer lookahead makes the policy progressively
more cautious: intervention rate falls monotonically (0.779 -> 0.757 -> 0.605)
and so does speed (15.97 -> 15.27 -> 12.28). But **tau=0.75 is over-cautious in
a way that costs real safety** - raw collisions jump to 8.596, worse than the
7.679 control, while the ego crawls at 12.28 against a 16.0 target. Caution
that slows the car does not prevent unshielded collisions here.

tau=0.35 keeps the best speed and ties on raw collisions but gives up the
shield-side gains (completion 0.770 vs 0.805, collisions 0.279 vs 0.233,
return 784 vs 814).

**tau=0.5 retained.** Worth noting the offline AUC analysis picked tau=0.5
before any training ran, and the trained outcomes agree - the cheap offline
ranking predicted the right hyperparameter, which is some evidence the
analyse-then-train loop is worth keeping for the next iteration.

### 2026-09-14 01:36 - extending to five seeds

The headline (-14.4%, 2 of 3 seeds) rests on three seeds with a control spread
of +-0.700, so the estimate is the weakest part of the result. Rather than
sweep further hyperparameters, remaining time goes to seeds 310 and 311 for
both the dense variant **and** matched `ppo_nominal` controls (the completed
study only has controls at 307-309), giving a five-seed paired comparison.
