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
one broadcast instead. C1 will still be somewhat slower than C2 because C2
needs no extra IPC at all.

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

## 5. What is NOT built, and why

### 5.1 Variant registration and the C -> D rename are blocked on the live sweep

The B2 lambda sweep launched 2026-10-12 08:07 runs `python -m
scripts.training.run_ppo_cbf_progression` as a **fresh subprocess per (lambda,
seed)**, 42 of them over roughly 40 hours. Every one imports
`scripts/training/run_ppo_cbf_progression.py`, `scripts/ops/run_safety_matrix_study.py`
and `scripts/common/*` from disk at launch.

Editing any of those files while the sweep is in flight would mean **later
arms of the sweep run different code from earlier arms**. Even a strictly
additive change is not worth that: the sweep is the data behind a paper table.

So the following are written up but not applied:

1. `VARIANT_SPECS` entries for `ppo_cmdp_state` and `ppo_cmdp_action`
   (`execution_mode: "box"`, `projected_mean: False`,
   `detached_actor_loss: False`, plus a new `cmdp: True` flag).
2. `TENSORBOARD_VARIANT_IDS` entries (suggest `cmdps`, `cmdpa`).
3. A dispatch branch in `build_model` / `load_model` next to the existing
   `projected_mean` and `detached_actor_loss` branches
   (`run_ppo_cbf_progression.py:2003-2071`).
4. CLI args: `--cost-limit`, `--gamma-c`, `--gae-lambda-c`, `--dual-lr`,
   `--dual-init`, `--cost-vf-coef`, `--cmdp-infeasible-rule`.
5. `_variant_relevant_settings` entries so the CMDP settings enter the
   training signature **only** for the CMDP variants — this is what keeps B2's
   signature unchanged and its checkpoints reusable.
6. `ARMS` rows in `run_safety_matrix_study.py` for `C1_cmdp_state` and
   `C2_cmdp_action`.
7. The rename: `ppo_cbf_diff_reward_only` -> D1, `ppo_cbf_projected` -> D2,
   `ppo_cbf_projected_reward_off` -> D3 in the paper-facing labels, with a
   **backward-compatible signature alias** so existing C1/C2 checkpoints still
   resolve. `training_signature` includes the variant name, so without an
   alias every existing projected checkpoint becomes unreusable.

**Check the sweep is finished before applying any of this:**

```powershell
Get-Content "$env:LOCALAPPDATA\Temp\saferl_runs\b2_lambda_sweep\SWEEP.log" -Tail 5
Get-Process python -ErrorAction SilentlyContinue | Measure-Object
```

The log ends with `=== B2 lambda sweep complete ===` when done.

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

## 9. Order of work to a first CMDP run

1. Wait for `=== B2 lambda sweep complete ===`.
2. Apply §5.1 items 1-6, then the §5.1 item 7 rename with aliases. Run
   `python -m scripts.ops.check_repository_policy` and the focused tests.
3. Confirm the §6 open questions.
4. Smoke: 50k steps, one seed, each CMDP variant, with an explicit
   `--cost-limit`, checking `cmdp/lambda` moves and
   `cost_explained_variance` is not pinned at zero.
5. Profile `c_s` and `c_a` to choose `d_s` and `d_a` (§5.3).
6. Apply the gradient-clip fix (§5.2).
7. Real runs.
