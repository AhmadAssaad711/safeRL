---
layout: page
title: Documentation
---

<p class="code-link"><a class="button" href="https://github.com/AhmadAssaad711/safeRL">View the code on GitHub</a> <span>github.com/AhmadAssaad711/safeRL</span></p>

A guide to the work and the code behind this site. The full, maintained
documentation lives in the repository.

- [The problem](#the-problem)
- [Environment](#environment)
- [Observation and action](#observation-and-action)
- [The safety shield](#the-safety-shield)
- [Reward](#reward)
- [The policies](#the-policies)
- [Training](#training)
- [Evaluation](#evaluation)
- [Diagnostic scenarios](#diagnostic-scenarios)
- [Reading the robustness plots](#reading-the-robustness-plots)
- [Reproducing this site](#reproducing-this-site)
- [The code](#the-code)
- [Further reading](#further-reading)

## The problem

On a lane-free highway, there are no lanes to organize traffic, so every
vehicle chooses both its speed and its lateral position at every moment. The
ego vehicle is a PPO policy. A control barrier function (CBF) shield can
correct its commands at runtime, but a shield keeps a policy safe whether or
not the policy learned anything about safety.

So every policy is evaluated twice:

- **CBF-OFF**: the policy's own actions, with no shield.
- **CBF-ON**: the policy plus the deployed shield.

The policies differ only in how safety information reaches training. A policy
that has internalized safety should stay safe with the shield off, not only
with it on.

## Environment

The custom Gymnasium environment `lane-free-v0`:

| Item | Value |
| --- | --- |
| Road | Periodic 380 m by 10.2 m ring with no lanes |
| Traffic | 40 vehicles; surrounding traffic follows the MTM model, with personalities from cautious to aggressive |
| Surrounding speeds | Desired speeds sampled from 15 to 25 m/s |
| Ego speed target | 16 m/s |
| Timing | 100 Hz physics, 20 Hz policy, 20 Hz CBF; each action is held for five physics frames |
| Collisions | A collision ends the episode |

The simulator keeps surrounding vehicles from colliding with each other, but
by default that guard does not react to the ego. Avoiding the ego's collisions
is left to its policy and shield.

## Observation and action

**Observation.** One row for the ego and one per nearest neighbor, each
`[dx, dy, vx, vy, v_desired]`, normalized. The ego row carries the ego's
lateral position and the target lateral position. The previous executed
action is appended. The width is `(1 + neighbors) × 5 + 2`. The policies on
this site see seven neighbors and drop the constant ego desired-speed slot,
for 41 features.

**Action.** Physical longitudinal and lateral acceleration, each in
[-3, 3] m/s². The action passes through separately logged stages:

```text
actor mean → sampled action → box-clipped action → CBF-safe action
           → normalized simulator command → executed acceleration
```

The previous *executed* action is what enters the next observation, so the
policy sees what actually happened, not what it asked for.

## The safety shield

Each nearby vehicle is enclosed in a fixed ellipse sized for a 3.6 m by 1.8 m
reference vehicle. The barrier

```text
h(dx, dy) = dx²/A² + dy²/B² − 1,    A = 3.6/√2 m,  B = 1.8/√2 m
```

is non-negative while the ego is outside the ellipse. The two road edges add
boundary barriers. Acceleration appears only in the second derivative of h,
so the shield uses a second-order (high-order) CBF condition:

```text
ψ1 = ḣ + c1·h
ψ2 = ψ̇1 + c2·ψ1 = ḧ + (c1 + c2)·ḣ + c1·c2·h ≥ 0
```

with (c1, c2) = (0.5, 8). Each condition is linear in the acceleration
command, so the safe commands form a polygon in the 2D action plane.

At every policy step, the shield:

1. builds one row per nearest neighbor (up to seven here) and per road edge;
2. keeps the policy's action if it satisfies every row;
3. otherwise projects it onto the safe polygon, using an exact 2D solver
   with OSQP as a fallback;
4. if no action is safe, uses the least-violating action, then an evasive
   emergency action.

## Reward

The reward combines tracking, progress and comfort terms:

```text
r = ε / (ε + wx·cx + wy·cy + wf·cf + way·cay)
    + progress − jerk_penalty + collision_or_overtake_adjustment
```

- `cx`: speed error from the 16 m/s target;
- `cy`: lateral error from a target lateral position, chosen from a smooth
  map of free space ahead;
- `cf`, `cay`: a potential-field cost from nearby vehicles and a lateral
  comfort cost. The potential-field weight is zero in these runs.
- progress: forward metres, normalized by 2 m;
- a bounded jerk penalty, a −2.5 collision penalty and a +0.5 overtake bonus.

## The policies

| Label | Variant | CBF in training | How safety reaches the policy |
| --- | --- | --- | --- |
| A1 | `ppo_nominal` | No | None: the reward above only |
| B1 | `ppo_hocbf_state_cost_raw` | No | A fixed state-cost penalty, built from the barrier, added to the reward |
| B2 | `ppo_cbf_nd_reward_actor` | Yes, hard shield | A reward penalty, plus an actor loss pulling the policy toward the shield's safe action |

The policies form a ladder: A1 never sees the barrier, B1 sees it only as a
cost in its reward, and B2 learns directly from the shield's corrections.

## Training

| Setting | Value |
| --- | --- |
| Algorithm | PPO (Stable-Baselines3) |
| Hyperparameters | learning rate 1e-4, γ 0.99, GAE λ 0.95, clip 0.2, entropy 0.005, 10 epochs, batch 100 |
| Networks | Separate policy and value networks, [256, 128] each |
| Rollouts | 20 parallel environments sharing a 1,000-step rollout (50 steps each) |
| Budget | 1,000,000 timesteps per policy and seed |
| Seeds | 307, 327, 347, 367, 387, 407, 427 |
| Actor-loss weight | λ = 1.0 for B2 |

The seeds are spaced by 20 because each run's 20 environments use seeds `s`
to `s + 19`. Closer seeds would share traffic scenarios and would not be
independent repeats.

## Evaluation

Each trained policy drives 200 episodes with the shield off, then the same 200
scenarios with it on. An episode **completes** only if the ego covers 1,000 m
with no collision within 3,000 policy steps (150 s).

| KPI | Meaning |
| --- | --- |
| Ego collisions / km | Collisions per kilometre driven |
| Minimum h | Lowest barrier value reached; below zero means inside an ellipse |
| QP failure rate | Share of steps where the shield found no safe action |
| Intervention rate | Share of steps where the shield changed the action |
| Correction norm | Size of the shield's correction |
| Episode return | Total reward per episode |
| Episode length | Policy steps per episode |
| Completion rate | Share of episodes that complete the 1,000 m task |
| Speed error | Mean absolute deviation from 16 m/s |
| Lateral error | Mean distance from the lateral target |
| Jerk | Mean change in acceleration |

Each seed gives one value per KPI. The results tables report the mean over
seeds with a 95% confidence interval, and test each policy against A1 with a
Holm correction for multiple comparisons.

The scenario videos differ in one respect: a collision is counted but does
not end the rendered episode, so the vehicles keep driving and every clip runs
its full length. The highway videos follow the evaluation rule instead and stop
at the ego's first collision.

## Diagnostic scenarios

Fixed, scripted scenes that each test one driving decision. Every policy
starts from the same scene.

| Scenario | What a good policy does |
| --- | --- |
| Open road | Keeps steady progress with little drift; the shield barely acts |
| Open passing gap | Commits to an available passing gap |
| Reject overtake | Refuses a tempting gap that holds a fast closing vehicle |
| Tight-slot trap | Avoids diving into a slot that is too tight |
| Boundary recovery, both edges | Moves inward from the road edge past a slow blocker |
| Boxed in | Does not force a pass when both side gaps are occupied |
| Rear pressure | Escapes a fast car closing from behind through the cleaner side |
| Sudden lead slowdown | Reacts to a sharply braking leader without diving into traffic |
| Staggered traffic | Threads through staggered traffic via the clean side |

## Reading the robustness plots

The plots on the home page show the HOCBF margin of the policy's own action:

```text
S(t) = min over constraints i of [ b_i(t) − A_i(t) · u_raw(t) ]
```

Above zero, the raw action satisfies every safety constraint. Below zero, at
least one is violated and a shield would have to step in. The curves are
means over seven rollouts per policy. A
rollout stops contributing after it crashes, so a curve that ends early means
no rollout of that policy was still running.

## Reproducing this site

Run from `safeRL_workspace`. Each script documents its options under
`--help`.

| Output | Command |
| --- | --- |
| A1, B1 and B2 × seven seeds | `python -m scripts.ops.run_safety_matrix_study --only A1_nominal B1_state_cost_raw B2_cbf_nd_reward_actor` |
| B2 at λ = 1.0 | `... run_safety_matrix_study --only B2_cbf_nd_reward_actor --actor-lambda 1.0` |
| Robustness plots | `python -m scripts.reporting.plot_hocbf_margin_aggregate` |
| Highway GIFs | `python -m scripts.rendering.render_policy_traffic_gifs` |
| Scenario GIFs | `python -m scripts.rendering.render_scenario_policy_gifs` |

The three policies are 21 training runs; add `--dry-run` to print the plan first.

## The code

All the code is in the **[safeRL repository on GitHub](https://github.com/AhmadAssaad711/safeRL)**. Paths below are
relative to its root.

| Part | Where |
| --- | --- |
| Environment | `safeRL_workspace/laneless highway env/lane_free_env.py` |
| Canonical experiment | `safeRL_workspace/notebooks/lanelessKaralakou.ipynb` |
| Configuration, rewards, safety | `safeRL_workspace/src/saferl/` |
| CBF projection | `safeRL_workspace/scripts/common/` |
| Training runners | `safeRL_workspace/scripts/training/` |
| Evaluation and KPIs | `safeRL_workspace/scripts/evaluation/` |
| Tables and figures | `safeRL_workspace/scripts/reporting/` |
| Rollout GIFs | `safeRL_workspace/scripts/rendering/` |
| Study launchers, checks | `safeRL_workspace/scripts/ops/` |
| Tests | `safeRL_workspace/tests/` |

## Further reading

- [Experiment contract](https://github.com/AhmadAssaad711/safeRL/blob/main/safeRL_workspace/docs/lanelessKaralakou_reference.md):
  environment, observation, reward, CBF and evaluation in full.
- [Script and function reference](https://github.com/AhmadAssaad711/safeRL/blob/main/safeRL_workspace/docs/script_reference.md):
  which module to run for which task.
- [Diagnostic scenarios](https://github.com/AhmadAssaad711/safeRL/blob/main/safeRL_workspace/docs/diagnostic_scenarios.md):
  the fixed scenes and the counterfactual protocol.
- [Trial log](https://github.com/AhmadAssaad711/safeRL/blob/main/safeRL_workspace/docs/experiments/trial_log.md):
  every trial after the main study, with verdicts.
- [Documentation index](https://github.com/AhmadAssaad711/safeRL/blob/main/safeRL_workspace/docs/README.md):
  all other studies and notes.
