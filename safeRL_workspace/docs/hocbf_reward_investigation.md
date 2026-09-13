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
