# CMDP arms: implementation notes and handoff

Status as of 2026-10-12. Covers the two new constrained-RL arms, what is
built, what is deliberately not built yet, and the things most likely to
confuse a later debugging session.

## 1. What the arms are

The hierarchy after the rename (see §6):

| Letter | Variant | Safety information | Training execution |
|---|---|---|---|
| A1 | `ppo_nominal` | none | raw |
| B1 | `ppo_hocbf_state_cost_raw` | fixed state-cost reward penalty | raw |
| **C1** | **CMDP-State** (new) | state cost critic + dual | **raw** |
| **C2** | **CMDP-Action** (new) | action-distance cost critic + dual | **raw** |
| B2 | `ppo_cbf_nd_reward_actor` | direct geometric actor loss | HOCBF-filtered |
| D1 | `ppo_cbf_diff_reward_only` (was C1) | differentiable projection + reward | HOCBF-filtered |
| D2 | `ppo_cbf_projected` (was C2) | differentiable projection + actor loss | HOCBF-filtered |
| D3 | `ppo_cbf_projected_reward_off` | reward-off architecture control | HOCBF-filtered |

`ppo_cbf_integrated_actor_only` (`iao`) is a **legacy duplicate** of
`ppo_cbf_projected` — identical on every spec field — and gets no paper letter.

The CMDP arms solve

```
max J_R(theta)   s.t.   J_C(theta) <= d
```

with `J_C` the **average per-step cost under the policy's occupancy measure**,
not an episodic sum. Episodes here have very different lengths and a collision
terminates one, so an unnormalized sum would reward a policy for crashing
early and accumulating less cost.

The scientific contrast against B2 is the information pathway, not the
quantity:

```
C2 : c_a -> cost critic -> A_C -> policy gradient
B2 : c_a -> grad_a c_a -> grad_theta a,  with Pi(a) a detached target
D2 : c_a -> differentiable projection, Jacobian in the backward pass
```

All three use the same `c_a`. That is why it must stay the raw geometric
quantity — see §5.3.

## 2. What is built

| Path | Contents | Commit |
|---|---|---|
| `src/saferl/safety/action_cost.py` | `c_a = 1/2 \|\|a - Pi(a)\|\|^2`, infeasible-state rules | `373a759` |
| `src/saferl/algorithms/cmdp.py` | `CMDPConfig`, cost GAE, constraint estimate, `DualVariable` | `373a759` |
| `src/saferl/algorithms/ppo_lagrangian.py` | `CostRolloutBuffer`, `CMDPActorCriticPolicy`, `PPOLagrangian` | `c707ed2` |
| `tests/unit/test_action_cost.py` | 20 contract tests | `373a759` |
| `tests/unit/test_cmdp.py` | 24 contract tests | `373a759` |
| `tests/unit/test_ppo_lagrangian.py` | 18 contract tests, full loop on a synthetic env | `c707ed2` |

`src/saferl/safety/state_cost.py` already implemented `c_s` before this work
and is reused unchanged.

Run them with:

```powershell
cd "safeRL_workspace"
$env:PYTHONPATH = "$PWD\src"
python -m pytest tests/unit/test_cmdp.py tests/unit/test_action_cost.py tests/unit/test_ppo_lagrangian.py -q
```

## 3. Locked design decisions

| Decision | Value | Where |
|---|---|---|
| Training execution | filter **OFF**, `execution_mode="box"` | enforced in `PPOLagrangian.__init__` |
| `J_C` | average per-step cost | `cost_estimator="batch_mean"` |
| `gamma_c` | `= gamma` initially | `CMDPConfig.gamma_c` |
| Dual update | plain projected ascent, once per rollout | `DualVariable.update` |
| `A_C` normalization | **off** | `normalize_cost_advantage=False` |
| `V_C` head | unconstrained, no softplus | `CMDPActorCriticPolicy._build` |
| `V_C` parameters | **independent network**, no shared trunk | same |
| `1/(1+lambda)` scaling | **off** initially | `use_penalty_scaling=False` |
| Infeasible `c_a` | `box_bound` constant from the action box | `action_cost.BOX_BOUND_RULE` |
| `d_s`, `d_a` | **deliberately unset** | `cost_limit` has no default |

`cost_limit` raising on omission is intentional, and
`PPOLagrangian.__init__` refuses a missing `cmdp_config` on every path except
SB3's `load()`.

## 4. Why the integration looks the way it does

These are the non-obvious choices. Each one was forced by something in the
existing code.

### 4.1 The barrier cost is computed in the trainer, not read from `info`

`info["cbf_state_cost"]` looks like the obvious source for `c_s`. It is not
usable: `ppo_cbf_env.py:1203-1210` only populates it when
`state_cost_diagnostics` is non-empty, which requires
`StateCostParameters.enabled`, i.e. `lambda_s > 0` — and the same block does
`reward = float(reward) - state_cost_penalty`. **Reading the cost out of
`info` would force the reward shaping the CMDP must not have.**

Instead `PPOLagrangian._state_costs` calls the public
`current_constraint_system()` through `env.env_method`, takes `barrier_h` and
`barrier_h_dot`, and calls `barrier_state_cost` itself. The environment's own
`state_cost` therefore stays at `lambda_s = 0` and the reward is untouched.
The trainer's own `state_cost_params` needs `lambda_s > 0` only to enable the
cost *definition*; that value never reaches a reward.

**Debugging implication:** if C1 ever shows a reward penalty, check that the
runner is not also passing `--state-cost-lambda` for that variant.

### 4.2 Cost timing differs between the two arms, on purpose

* `c_a` is scored **before** `env.step()`, against the pre-step context parsed
  from `self._last_obs`. It is a property of `(s_t, a_t)`.
* `c_s` is scored **after** `env.step()`, from the post-step system. It is
  `c_s(s_{t+1})`, so it credits the action that produced the state. This
  matches the existing B1 semantics, which evaluates on `next_system`
  (`ppo_cbf_env.py:1200-1205`).

Both land at buffer index `t`.

### 4.3 One broadcast `env_method` per step, not one per env

`_state_costs` calls `env.env_method("current_constraint_system")` once and
receives a list for all envs. The existing rollout loop calls
`set_projection_record` once **per env** (20 IPC calls per step); the cost adds
one broadcast instead.

Measured, C2 is the slower arm, not C1: 20k steps took **108 s for
CMDP-Action** against **91 s for CMDP-State** at `n_envs = 20`. C2 runs 20
`project_polytope_2d_numpy` calls per step in a Python loop, which costs more
than one broadcast IPC. An earlier version of this note predicted the opposite.

### 4.4 `collect_rollouts` is duplicated rather than hooked

`LatentActionPPO.collect_rollouts` hardcodes its cost extras behind
`isinstance(rollout_buffer, CBFSafetyRolloutBuffer)`
(`projected_ppo_cbf.py:1021-1052`, `:1062-1073`). There is no generic hook.
Adding one would have meant editing `scripts/common` — see §5.1 — so the
method is reimplemented in `PPOLagrangian`. If `LatentActionPPO`'s rollout
logic changes, **this copy must be updated too**; that is the main maintenance
cost of this design.

### 4.5 `cbf_base_observation_dim` is needed in two places

The algorithm keeps its own `cbf_layout` (from the *model* kwargs
`cbf_base_observation_dim` / `cbf_max_constraints`) for the rollout-time
projection, and the policy keeps a separate one for slicing the cost critic's
input. Passing it only in `policy_kwargs` leaves the model's layout at the
42/18 default and fails with
`expected augmented observation width 114, got <n>`. This cost a debugging
cycle during development; the test helper passes both.

### 4.6 The cost surrogate takes `max`

```python
cost_surrogate = th.max(ratio * cost_advantages, clipped_ratio * cost_advantages).mean()
```

Cost is driven down, so the pessimistic bound is the **upper** one. Using
`min` here mirrors the reward term and silently inverts the constraint — the
policy would be rewarded for increasing cost. There is no test that can catch
this from the outside cheaply, so it is called out here.

## 4a. Smoke results, 2026-10-12

Both arms run end-to-end at 20k steps, `n_envs = 20`, `Q1_stable`,
`neighbors_count = 7` (41D observation with `--drop-ego-speed-slot`).

| | CMDP-Action | CMDP-State |
|---|---|---|
| wall clock, 20k steps | 108 s | 91 s |
| `cost_limit` used | 0.05 (placeholder) | 0.5 (placeholder) |
| `J_C` measured | 13 - 21 | 0.90 |
| `lambda` after 20 updates | 0 -> 35.0 | 0 -> 0.61 |
| `cost_explained_variance` | ~1e-5 | ~4e-5 |

Three things to read from this, none of which are bugs but all of which matter
for choosing `d`:

1. **`c_a` is dominated by the infeasibility constant.** `box_bound` on the
   [-3,3]^2 box is `0.5 * (6^2 + 6^2) = 36.0`, and an untrained policy is
   infeasible often, so a mean of 13-21 implies roughly 40-55% of steps have an
   empty safe set. In that regime `c_a` behaves closer to
   `36 * P(infeasible)` than to a graded distance. The choice of `box_bound` is
   still right -- a solver-fallback distance would score those states as nearly
   free -- but `d_a` has to be set against this distribution, not against the
   feasible-state distances.
2. **lambda diverges monotonically** when `d` is far below the achievable cost.
   With `d = 0.05` against `J_C ~ 17` and `dual_lr = 0.1`, lambda climbs ~1.7
   per update and never turns around. That is correct dual-ascent behaviour on
   an infeasible constraint, and it is exactly why `cost_limit` has no default.
3. **The cost critic does not fit at this scale.** `cost_explained_variance`
   is ~1e-5 for both arms with `cost_value_loss` in the 5e4-1e5 range for C2.
   At 20k steps that is uninformative, but it is the first number to check on a
   real run: if it stays near zero, `A_C` is noise and the constraint is being
   enforced through lambda alone.

Diagnostics are written to `cmdp_dual.csv` in the run directory, and the final
dual state is printed as `[ppo-progression] CMDP dual state`.

## 4b. Diagnosing and fixing both arms, 2026-10-12 evening

The first 1M x 7-seed attempt **failed on both arms**: the constraint was never
satisfied and `J_C` *rose*. What follows is the chain of measurements that
fixed it, because each step rules something out.

### Attempt 1 -- projection cost, no context, dual_lr 0.01

| | C2 action (d=9.3) | C1 state (d=0.5) |
|---|---|---|
| seeds | 6 | 7 |
| lambda final | 121 - 127 | 3.5 - 4.2 |
| `J_C` first -> last | 18-21 -> 21-25 | 0.78-0.96 -> 0.80-1.03 |
| `cost_EV` | **0.0000 every seed** | 0.17 - 0.28 |

### Attempt 2 -- C1 with dual_lr 0.3

lambda reached **46** (so the dial was strong), but `J_C` still rose
0.788 -> 0.846 and `cost_EV` *collapsed* 0.17-0.28 -> **0.00005**, with
`active_fraction` going 0.987 -> **1.000**. Raising the dial is not the fix.

### What each measurement ruled out

1. **Row scale.** The first violation-form attempt was unnormalized and gave a
   mean cost of ~354 in the rows' own units, `cost_value_loss` 2.3e8 and
   `cost_EV` 0.0000. `(A_i, b_i)` and `(k A_i, k b_i)` are the same halfspace,
   so dividing by `||A_i||` is required, not cosmetic. Fixed; cost fell to
   ~145 and `max_row_violation` 12.6 -> 9.4.
2. **Critic blindness.** The cost critic could not see the CBF rows that
   *define* the cost. Turning on `--cost-critic-sees-context` took C1's
   `cost_EV` from 0.17-0.28 to **0.72**. It did nothing for C2 (see 3).
3. **`cost_EV` is the wrong diagnostic for an action cost.** `c_a(s,a)` depends
   on the action, and `V_C(s)` can only predict its state-average; the
   within-state spread from action sampling is irreducible, so `cost_EV` is
   near zero *by construction*. C2 later drove its cost 172.8 -> 90.7 while
   `cost_EV` stayed at 0. Judge an action cost by whether `J_C` falls, never by
   `cost_EV`.
4. **A near-constant cost has no gradient.** C1's `distance` barrier form fires
   on ~100% of steps in a narrow 0.66-0.97 band -- a per-step tax, which the
   margins in `state_cost.py` exist precisely to prevent. The `ellipse` form
   gives `active_fraction` 0.39-0.78 with `cost_EV` 0.68.
5. **The shared gradient clip throttles the cost critic once lambda grows.**
   With lambda pinned at 0 a profile reached `cost_EV` 0.72; with lambda live
   and otherwise identical settings it was 0.019. `--separate-grad-clip` is
   therefore not optional for the CMDP arms -- and note the cost critic only
   reaches the value group because `PPOLagrangian` overrides
   `_gradient_parameter_groups` (commit `24cd7b7`).

### Configurations that work

**C1 CMDP-State** -- `ellipse` barrier form, `--cost-critic-sees-context`,
`--separate-grad-clip`, `d_s = 0.5`, `dual_lr = 0.05`. At 175k:
`J_C` **0.441 -> 0.400**, constraint **satisfied** (violation negative, -0.10),
`cost_EV` 0.007 -> 0.33, `active_fraction` steady ~0.39. lambda stays small
(0.0997) because it does not need to grow -- the correct dual signature.

**C2 CMDP-Action** -- normalized `violation` form,
`--cost-critic-sees-context`, `--separate-grad-clip`, `d_a = 60`,
`dual_lr = 0.0007`. Drove `J_C` **172.8 -> 90.7** by 200k, which the projection
form never did. **But it then reversed**, climbing to 215 by 550k with
lambda 47.5 -- plain dual ascent oscillating. Relaunched with
`--cmdp-penalty-scaling` (the `1/(1+lambda)` stabilizer), which at 175k shows
the same early descent, 179.0 -> 92.6. **Whether the stabilizer prevents the
reversal past 200k is unresolved as of this writing.** If it does not, the next
levers are a `--dual-max` cap or a PID controller on the same error.

### Open discrepancy to reconcile before publication

C1 and C2 now differ in more than their cost source: C2 additionally carries
`--cmdp-penalty-scaling`. Either apply it to both arms or neither before the
two are compared, and record the choice.

## 5. What is NOT built, and why

### 5.1 The C -> D rename of existing artifacts

Registration **is** applied (the B2 sweep was stopped by request on
2026-10-12 at 10:42, after lambda=0 seeds 307 and 327 completed; seeds 347 and
367 hold 825k checkpoints and resume rather than restart). What remains is the
artifact-side rename.

`VARIANT_LETTERS` now carries the paper-facing letters and
`VARIANT_LETTER_ALIASES` maps the pre-rename `C1`/`C2` to
`ppo_cbf_diff_reward_only` / `ppo_cbf_projected`, with
`variant_for_letter(letter, legacy=True)` for reading old labels. The
*variant keys* were not renamed, so **no existing checkpoint or signature is
affected** — the rename is presentational.

Still to do on the artifact side:

1. `ARMS` rows in `run_safety_matrix_study.py` for `C1_cmdp_state` and
   `C2_cmdp_action`, and relabelling its existing `C1_*`/`C2_*` rows to
   `D1_*`/`D2_*`. Note those labels are **directory names** under the
   out-base, so renaming them orphans existing run directories unless the
   alias is used when reading them.
2. Report builders and figure filenames under `artifacts/reports/` that embed
   `C1`/`C2` with the old meaning.
3. `docs/` and `trial_log.md` references to C1/C2 predating 2026-10-12.

### 5.2 The shared gradient-clip fix is not done

`trial_log.md` §4c measured that stock PPO's single `clip_grad_norm_` across
all parameters cuts the policy gradient to ~1-2% of its magnitude, on every
update. `LatentActionPPO` already has `separate_grad_clip` as an opt-in and
`PPOLagrangian` inherits it, but it defaults off.

The CMDP arms make this worse: a **third** network (the cost critic) now
contributes to the same shared norm. This was locked as a prerequisite before
interpreting CMDP results, and it is not applied.

### 5.3 `d_s` and `d_a` are unmeasured

Both cost channels must be profiled under the same rollout protocol before a
limit can be defended. Note that the per-seed data and the checkpoints for the
published A1/B1/B2 table were **purged from `%LOCALAPPDATA%\Temp`** (the
directories survive with zero files), so the B2 sweep's fresh checkpoints are
what is available to profile against.

`action_cost_scale_from_reference` converts a physically meaningful correction
norm into the cost it scores, so `d_a` can be quoted as "a sustained
correction of this many m/s^2" rather than as a bare number.

## 6. Open questions

1. **Does `V_C` see the CBF context?** Implemented as
   `cost_critic_sees_context`, defaulting to **False** so the cost critic sees
   the same base observation as the actor and the other variants' critics.
   "Give the cost critic the same HOCBF context available to the other
   HOCBF-aware variants" is ambiguous between that and the full augmented
   observation; confirm before the first real run.
2. **Letter for `ppo_cbf_projected_reward_off`** — D3 is assumed above.
3. Whether the legacy `iao` duplicate should be deleted or retained under the
   legacy policy.

## 7. Known-failing tests, unrelated to this work

Verified by running the suite with the new files moved aside: baseline is
**232 passed / 2 failed / 1 collection error**, and with the new files
**294 passed / same 2 failed / same 1 collection error**.

| Test | Failure |
|---|---|
| `test_cbf_critical_c_sweep.py` | collection: `cannot import name 'critical_damping_candidates' from 'scripts.evaluation.cbf_damped_gain_eval_sweep'` |
| `test_hocbf_reward.py::test_direct_hocbf_reward_uses_second_order_action_residual` | assertion |
| `test_simulator_benchmark.py::test_benchmark_is_seed_deterministic_and_reports_physics_rate` | `dt and simulation_frequency disagree: dt=0.01 implies 100 Hz, not 4 Hz` |

Note that `git status` showing no tracked modification is *not* sufficient
evidence of unrelatedness — new untracked modules can affect imports and
collection. The baseline comparison above is the evidence.

## 8. Environment notes

* The `saferl` package is not installed; runs need `PYTHONPATH=<workspace>\src`.
  `run_ppo_cbf_progression.py` has its own src fallback, but direct `python -m`
  invocations of other modules do not.
* `git` is not on `PATH` by default in this environment; it lives at
  `%LOCALAPPDATA%\Programs\Git\cmd`.
* `run_ppo_cbf_progression.py` records **no git provenance** — it imports only
  `saferl.rewards`, never `saferl.artifacts`. This is the origin of
  `Git commit: unknown` in the published safety-matrix report, and it means
  CMDP runs will also lack a recorded SHA until that is addressed.

## 9. How to run, and what is left

Both arms are runnable now. A working invocation, matching the canonical
safety-matrix contract:

```powershell
cd "safeRL_workspace"
$env:PYTHONPATH = "$PWD\src"
python -m scripts.training.run_ppo_cbf_progression `
  --variants ppo_cmdp_action --seeds 307 --timesteps 20000 `
  --n-envs 20 --n-steps 1000 --batch-size 100 `
  --traffic-model mtm --device cpu --ppo-config Q1_stable `
  --remove-vehicle-dimensions --expose-target-y --drop-ego-speed-slot `
  --lateral-target field --lateral-field-wall-weight 1.0 --lateral-y-weight 0.96 `
  --potential-field-weight 0 `
  --env-config-json '{\"neighbors_count\": 7}' --max-neighbor-constraints 7 `
  --cost-limit 0.05 --dual-lr 0.1 `
  --output-dir <explicit path>
```

Use `--variants ppo_cmdp_state` plus
`--state-cost-barrier-form distance` for C1. `--cost-limit` is required and
has no default.

### The end-goal sweep, prepared

Both CMDP arms are registered in `scripts/ops/run_safety_matrix_study.py` as
`C1_cmdp_state` and `C2_cmdp_action`. They are **opt-in**: a bare invocation
still runs the original five arms, because the CMDP arms require a
`--cost-limit` that has no defensible default, and the launcher refuses them
without one rather than letting fourteen runs die seconds in.

`d_s` and `d_a` are different numbers, so the two arms need **separate
invocations** — a single `--cost-limit` cannot serve both:

```powershell
cd "safeRL_workspace"
$env:PYTHONPATH = "$PWD\src"

# CMDP-Action, 1M x 7 canonical seeds
python -m scripts.ops.run_safety_matrix_study `
  --only C2_cmdp_action --timesteps 1000000 --concurrency 2 `
  --extra-arg=--cost-limit --extra-arg <d_a> `
  --out-base "$env:LOCALAPPDATA\Temp\saferl_runs\cmdp_1M_action"

# CMDP-State, 1M x 7 canonical seeds
python -m scripts.ops.run_safety_matrix_study `
  --only C1_cmdp_state --timesteps 1000000 --concurrency 2 `
  --extra-arg=--cost-limit --extra-arg <d_s> `
  --out-base "$env:LOCALAPPDATA\Temp\saferl_runs\cmdp_1M_state"
```

Each is 7 runs at ~1.3 h, so roughly 5 h per arm at concurrency 2. Give each
its own `--out-base`: a run is resumed or skipped on its output path alone, and
the path does not encode the cost limit. Add
`--extra-arg=--separate-grad-clip` if the gradient-clip fix is being applied.

### The cost profile, for choosing d

`--cost-limit 1e9 --cost-vf-coef 0` makes the constraint unreachable and the
cost critic inert, so `lambda` stays pinned at 0 and the actor trains as plain
filter-off PPO while the cost is still scored and logged every rollout through
the same code path the constrained runs use. Read the converged
`cost_mean_raw` from `cmdp_dual.csv`, then set `d` as a fraction of it.

A profile run at 1M, seed 307, for both arms was launched 2026-10-12 12:12
into `%LOCALAPPDATA%\Temp\saferl_runs\cmdp_cost_profile\{action,state}\`.

**Launcher gotcha, cost two attempts:** `Start-Process -ArgumentList` first
splits an inline `--env-config-json '{"neighbors_count": 7}'` on the internal
space, then strips the double quotes, so Python receives `{neighbors_count:7}`
and raises `JSONDecodeError`. Use `--env-config-file` with a JSON file instead.
`run_safety_matrix_study.py` is unaffected: it builds an argv list for
`subprocess.run`, which does no shell re-parsing.

Remaining before a *reportable* result:

1. **Choose `d_s` and `d_a`** from measured distributions (§5.3). The smoke
   values are placeholders and lambda diverges under them.
2. **Apply the gradient-clip fix** (§5.2) — locked as a prerequisite, and the
   cost critic adds a third network to the same shared budget.
3. **Confirm the §6 open questions**, principally whether `V_C` should see the
   CBF context.
4. **Watch `cost_explained_variance`.** At 20k steps it is ~1e-5; if it stays
   there on a real run, `A_C` is noise and the constraint is being carried by
   lambda alone.
5. Finish the artifact-side rename (§5.1).
