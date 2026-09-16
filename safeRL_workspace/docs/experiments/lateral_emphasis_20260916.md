# Wall-hugging and lateral emphasis, 2026-09-16

Status: **solved.** Rebalancing the reciprocal tracking term to `wx = 0.51`,
`wy = 0.96` with wall repulsion on the lateral target takes the unshielded
policy from spending 77.3% of its steps within half a metre of a road boundary
to **0.2%**, and improves every other KPI at the same time. Two earlier attempts
at "more lateral emphasis" failed, and why they failed is the useful part.

One seed (308). Raw run data under `artifacts/runs/lateral_emphasis/`.

## The behaviour nobody was measuring

`scripts/evaluation/probe_behaviour_guardrails.py` exists precisely to catch a
policy that buys safety by hugging the sideline or lagging the traffic. Run
across the arms of the 1M state-cost study it found both, everywhere:

| arm | median ego y | % steps within 0.5 m of a wall | speed deficit |
| --- | ---: | ---: | ---: |
| control 307 | 4.79 | 21.6% | +0.67 |
| control 308 | 5.37 | 77.9% | +1.12 |
| control 309 | 0.91 | 78.6% | +0.96 |
| control 310 | 2.17 | 82.0% | +0.78 |
| ellipse 307 | 0.90 | 85.2% | +0.30 |
| ellipse 309 | 8.54 | 72.0% | +0.41 |
| control-effort 307 | 9.30 | 84.7% | +1.44 |

The road is 10.2 m and the ego centre is confined to `[0.90, 9.30]`, so dead
centre is 4.20 m from the nearer edge. Six of seven policies sit within half a
metre of a boundary for 72-85% of every episode. Seed 309's median y is 0.91
against a wall at 0.90. Every arm also crawls: 2-9% under the 16 m/s target,
none ever speeds.

Hugging does **not** buy safety. Across the four control seeds the correlation
between wall time and collisions/km is **r = -0.26**, and the most wall-bound
seed (310, 82%) has the fewest collisions (2.809).

### Seed 307 is not a good policy, it is a different one

Seed 307 ranks **third of four on collisions/km** (3.683; 310 is best at 2.809)
with the second-lowest return. It is simply the only centre-driver, and every
KPI on which it looked exceptional is a mechanical consequence of not being at a
wall: lateral error 0.976 vs ~1.9, correction norm 0.254 vs 0.473, intervention
0.488 vs 0.745. Pressed against a boundary the CBF's boundary barrier is
permanently active and the shield corrects constantly; in the middle it idles.

The mode is fragile. Same seed, same initialisation: adding *either* the state
cost or the control-effort term flips 307 to the wall (85.2% / 84.7%). This
matters because **307 was the reference seed for every single-seed comparison in
the state-cost study**, so treatments appeared to "degrade lateral tracking and
increase CBF workload" when they had merely moved the policy into the mode the
other three controls were already in.

## Root cause: the lateral target was pointing at the wall

`_lateral_occupancy_field` (`src/saferl/rewards.py:534-576`) sums Gaussian
occupancy from *vehicles only*. A road boundary contributes nothing, and the
strip beside a wall has no vehicles on its far side by construction, so it
scores as the freest space on the road. The code comment recorded ~23%; measured
on this configuration it is worse:

| `lateral_field_wall_weight` | target within 1.5 m of a boundary | mean target distance to edge |
| --- | ---: | ---: |
| **0.0 (every run to date)** | **45.7 - 47.2%** | 1.82 - 1.85 |
| 1.0 | 5.4 - 6.0% | 2.90 - 3.03 |
| 2.0 | 1.0 - 3.9% | 2.69 - 3.15 |

So for roughly half of every episode the reward was *asking* the policy to go to
the wall. `wall_weight > 0` gives each boundary an occupancy contribution of the
same functional form and units as a vehicle, measured edge-to-edge, and fixes
the target immediately.

## Three arms, and why the first two failed

All at seed 308, 1M, paired against `nom_control_v16_1M_s308`, single-variable
verified against the matched control's resolved signature before launch.

| arm | wall weight | `wx` | `wy` | result |
| --- | ---: | ---: | ---: | --- |
| A | 1.0 | 1.0 | 0.65 | **failed** — hugging climbed 36 -> 52 -> 78 -> 88% |
| B | 1.0 | 1.0 | 2.0 | **failed** — hugged from the start, 70-80%, *worse* lateral error |
| **C** | **1.0** | **0.51** | **0.96** | **0.2% wall time** |

**Arm A** proves the target was not the cause. The target moved off the wall
(57% -> 6.5%) and the policy went to the wall anyway, monotonically, as training
progressed. A trained hugger has decoupled from the target entirely: applying
wall repulsion to an already-trained hugger post hoc moves its wall time from
91.1% to 90.4%.

**Arm B** proves that "more emphasis" is not a matter of raising `wy`. The
restoring force is

    dr/dcy = -eps * wy / (eps + wx*cx + wy*cy)^2

which is maximised over `wy` at `wy*cy = eps + wx*cx`, i.e.

    wy* = (eps + wx*cx) / cy

With `eps = 0.4`, `wx = 1.0`, `cx ~ 0.119` that constant is 0.519, so at the
wall (`cy ~ 0.5`) the optimum is `wy ~ 1.04`. **Arm B's `wy = 2.0` was past the
peak**, where the expression collapses to `-eps/(wy*cy^2)` — inversely
proportional to `wy`. Raising the weight *reduced* the force pulling the policy
back, which is exactly what was observed: B's lateral error (2.63-2.72) is
higher than A's (2.19-2.50) despite three times the lateral weight. Raising `wy`
also inflates the denominator, shrinking the whole tracking reward by 28% and
making it cheaper to abandon in favour of everything outside the reciprocal.

**Arm C** does the opposite of inflating: it holds the denominator constant and
shifts the balance. At the measured operating point (`cx = 0.119`, `cy = 0.19`)
the lateral:speed ratio was already only 1.04:1; `wx = 0.51`, `wy = 0.96` makes
it 3:1 with the tracking reward magnitude unchanged at 0.622, and puts `wy` on
the peak rather than past it.

## Result, arm C vs its matched control

CBF-OFF, 200 episodes, seed 308. `z` is against the spread of the four control
seeds, which is the only benchmark that accounts for run-to-run variation at
n = 1.

| KPI | control 308 | rebalanced | delta | z |
| --- | ---: | ---: | ---: | ---: |
| **% steps within 0.5 m of a wall** | **77.3%** | **0.2%** | — | off-scale |
| Mean distance to nearer wall (m) | 0.49 | 3.00 | — | — |
| Lateral tracking error (m) | 1.848 | 0.633 | -65.7% | **-2.25** |
| Episode return | 209.2 | 346.3 | +65.5% | +1.98 |
| Episode length (steps) | 325.0 | 440.2 | +35.4% | +0.84 |
| Ego collisions / km | 4.244 | 3.066 | -27.7% | -0.68 |
| Mean jerk norm | 8.455 | 5.856 | -30.7% | +0.23 |
| Distance completion rate | 0.010 | 0.095 | +850% | +1.45 |
| Abs speed error (m/s) | 2.219 | 2.062 | -7.1% | +0.75 |

CBF-ON: collisions -18.5%, return +18.3%, lateral error -56.8%, completion
0.735 -> 0.790.

The policy now occupies the middle of the road (y p05 2.63, p95 7.51) and speed
is unchanged (14.98 vs 14.96 m/s), so the centre is not being bought with
progress.

**What is and is not established at n = 1.** The behavioural change is beyond
argument: control seeds span 21.6-82% wall time and this is 0.2%. Lateral error
clears the four-seed spread at z = -2.25. **The -27.7% collisions does not**
(z = -0.68) and must not be reported as a safety result until replicated; return
at z = +1.98 is borderline. Replication on 2-3 spaced seeds is the next step.

## Linear learning-rate decay: negative

`--lr-schedule linear` (new, see `config(ppo)` commit) annealed 1e-4 -> 1e-7 over
1M steps on a nominal control at seed 307, single-variable against
`nom_control_v16_1M`.

| | control 307 | LR linear |
| --- | ---: | ---: |
| Ego collisions / km | 3.683 | 4.239 (+15.1%) |
| Episode return | 270.9 | 222.6 (-17.8%) |
| `rollout/ep_len_mean`, last decile | 458 | 348 |
| Mean jerk norm | 5.268 | 1.603 (-69.6%) |
| % steps near a wall | 25.8% | 72.9% |
| `train/clip_fraction`, last decile | 0.084 | 0.0033 |
| `train/approx_kl`, last decile | 0.0089 | 0.0016 |

The schedule froze the policy hard — almost no clipping and a tenth the KL by
the end — but the **tail slope of episode length is +17.3 per 100k, t = 3.12**,
statistically identical to the control's +17.3, t = 2.87. It annealed a policy
that had not converged, ending 24% lower on episode length. Decay is the right
tool only at a budget long enough for the curve to flatten on its own; at 1M it
is premature. It did deliver a genuinely stable endpoint, of a worse policy.

## Reproduction

```powershell
# arm C
python -m scripts.training.run_ppo_cbf_progression `
  --output-dir <run dir> --variants ppo_nominal --seeds 308 `
  --timesteps 1000000 --ppo-config Q1_stable --n-envs 20 --checkpoint-freq 25000 `
  --ego-desired-speed 16 --drop-ego-speed-slot `
  --lateral-target field --potential-field-weight 0.0 `
  --lateral-field-wall-weight 1.0 --lateral-y-weight 0.96 `
  --reward-config-file <file containing {"wx": 0.51}> `
  --max-neighbor-constraints 7 --env-config-file <{"neighbors_count": 7}> `
  --post-train-eval-episodes 200 --post-train-eval-seed-start 1100000 `
  --post-train-eval-workers 20 --skip-counterfactual --force-retrain
```

`--reward-config-file` accepts any key already present in the resolved reward
config and records the file path, its sha256 and the applied overrides in the
manifest, so `wx` and `epsilon_r` are reachable without touching the notebook.

## Open items

1. **Replicate arm C on 2-3 spaced seeds** (327, 347 — not 309/310, see below).
   Only then is the collisions result reportable.
2. The `wall_weight` value is unswept: 1.0 was chosen because it moved the
   target from 47% to 6%. 2.0 reaches 1%. Whether over-repelling suppresses
   side-gap overtaking is untested.
3. Arms A and B stopped early (825k, 725k) with checkpoints retained; neither
   has a KPI table.
4. The `way * cay` lateral-acceleration term in the denominator is still 0.0 and
   has never been tried.
