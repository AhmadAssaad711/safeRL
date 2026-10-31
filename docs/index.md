---
layout: default
title: Home
---

# safeRL: safe reinforcement learning on a lane-free highway

<!-- TODO: paper title, authors, venue, and PDF link. -->

## Overview

A PPO agent drives on a lane-free highway: a periodic 380 m by 10.2 m road
with no lanes, shared with 40 vehicles following MTM traffic dynamics. The
policy outputs continuous longitudinal and lateral acceleration. A high-order
control barrier function (HOCBF) filter can correct that command before it
reaches the simulator.

The study separates what the policy learns from what the shield supplies:

- **CBF-OFF** evaluates the policy's own actions, with no shield.
- **CBF-ON** evaluates the policy together with the deployed HOCBF shield.

## Policies compared

| Label | Policy | Variant | How safety enters training |
| --- | --- | --- | --- |
| A1 | Nominal PPO | `ppo_nominal` | None: the nominal reward only |
| B1 | HOCBF State-Cost | `ppo_hocbf_state_cost_raw` | Fixed HOCBF state-cost penalty in the reward; trained without the shield |
| B2 | HOCBF Actor-Loss | `ppo_cbf_nd_reward_actor` | Actor loss toward the shield's safe action, with the shield in training |
| C0 | HOCBF CMDP | `ppo_cmdp_collision` | Constrained PPO: a Lagrange multiplier keeps an episodic collision cost under a budget; trained without the shield |
| C1 | Differentiable projection | `ppo_cbf_diff_reward_only` | Differentiable projection plus reward (videos and robustness plots only) |
| C2 | Differentiable projection + loss | `ppo_cbf_projected` | Differentiable projection plus actor loss (robustness plots only) |

## On the highway

Each policy driving in the full 40-vehicle traffic. The ego vehicle is
green. The left clip has the shield off (CBF-OFF) and the right clip has it on
(CBF-ON). In these videos a collision does not end the episode: it is counted,
and the vehicles stay on the road and keep driving.

<div class="clip-grid">
  <div></div><div class="head">CBF-OFF</div><div class="head">CBF-ON</div>
  <div class="arm">A1</div>
  <img src="media/full_road/a1_cbf_off.gif" alt="A1 rollout, full highway traffic, CBF-OFF">
  <img src="media/full_road/a1_cbf_on.gif" alt="A1 rollout, full highway traffic, CBF-ON">
  <div class="arm">B1</div>
  <img src="media/full_road/b1_cbf_off.gif" alt="B1 rollout, full highway traffic, CBF-OFF">
  <img src="media/full_road/b1_cbf_on.gif" alt="B1 rollout, full highway traffic, CBF-ON">
  <div class="arm">C1</div>
  <img src="media/full_road/c1_cbf_off.gif" alt="C1 rollout, full highway traffic, CBF-OFF">
  <img src="media/full_road/c1_cbf_on.gif" alt="C1 rollout, full highway traffic, CBF-ON">
</div>

## Scenarios

Fixed diagnostic scenes that each test one driving decision. Every policy
starts from the same scene, with the shield off and on. The camera follows the
green ego vehicle. As on the highway, a collision is counted but does not end
the episode, so every clip runs the same length.

### Open passing gap

A passing gap is available; compare which policies actually commit to it.

<div class="clip-grid">
  <div></div><div class="head">CBF-OFF</div><div class="head">CBF-ON</div>
  <div class="arm">A1</div>
  <img src="media/scenarios/open_passing_gap/a1_cbf_off.gif" alt="A1 rollout, open passing gap, CBF-OFF">
  <img src="media/scenarios/open_passing_gap/a1_cbf_on.gif" alt="A1 rollout, open passing gap, CBF-ON">
  <div class="arm">B1</div>
  <img src="media/scenarios/open_passing_gap/b1_cbf_off.gif" alt="B1 rollout, open passing gap, CBF-OFF">
  <img src="media/scenarios/open_passing_gap/b1_cbf_on.gif" alt="B1 rollout, open passing gap, CBF-ON">
  <div class="arm">B2</div>
  <img src="media/scenarios/open_passing_gap/b2_cbf_off.gif" alt="B2 rollout, open passing gap, CBF-OFF">
  <img src="media/scenarios/open_passing_gap/b2_cbf_on.gif" alt="B2 rollout, open passing gap, CBF-ON">
</div>

### Boxed in

Both nearby side gaps are occupied; the policy should not force a pass.

<div class="clip-grid">
  <div></div><div class="head">CBF-OFF</div><div class="head">CBF-ON</div>
  <div class="arm">A1</div>
  <img src="media/scenarios/boxed_in/a1_cbf_off.gif" alt="A1 rollout, boxed in, CBF-OFF">
  <img src="media/scenarios/boxed_in/a1_cbf_on.gif" alt="A1 rollout, boxed in, CBF-ON">
  <div class="arm">B1</div>
  <img src="media/scenarios/boxed_in/b1_cbf_off.gif" alt="B1 rollout, boxed in, CBF-OFF">
  <div class="missing">clip not available</div>
  <div class="arm">B2</div>
  <img src="media/scenarios/boxed_in/b2_cbf_off.gif" alt="B2 rollout, boxed in, CBF-OFF">
  <img src="media/scenarios/boxed_in/b2_cbf_on.gif" alt="B2 rollout, boxed in, CBF-ON">
</div>

### Rear pressure

A fast rear vehicle closes on the ego, which should move toward the cleaner
side gap.

<div class="clip-grid">
  <div></div><div class="head">CBF-OFF</div><div class="head">CBF-ON</div>
  <div class="arm">A1</div>
  <img src="media/scenarios/rear_pressure/a1_cbf_off.gif" alt="A1 rollout, rear pressure, CBF-OFF">
  <img src="media/scenarios/rear_pressure/a1_cbf_on.gif" alt="A1 rollout, rear pressure, CBF-ON">
  <div class="arm">B1</div>
  <img src="media/scenarios/rear_pressure/b1_cbf_off.gif" alt="B1 rollout, rear pressure, CBF-OFF">
  <img src="media/scenarios/rear_pressure/b1_cbf_on.gif" alt="B1 rollout, rear pressure, CBF-ON">
  <div class="arm">B2</div>
  <img src="media/scenarios/rear_pressure/b2_cbf_off.gif" alt="B2 rollout, rear pressure, CBF-OFF">
  <img src="media/scenarios/rear_pressure/b2_cbf_on.gif" alt="B2 rollout, rear pressure, CBF-ON">
</div>

### Boundary recovery

The ego starts near the road edge behind a slow blocker and should recover
inward.

<div class="clip-grid">
  <div></div><div class="head">CBF-OFF</div><div class="head">CBF-ON</div>
  <div class="arm">A1</div>
  <img src="media/scenarios/boundary_recovery_road_edge/a1_cbf_off.gif" alt="A1 rollout, boundary recovery road edge, CBF-OFF">
  <img src="media/scenarios/boundary_recovery_road_edge/a1_cbf_on.gif" alt="A1 rollout, boundary recovery road edge, CBF-ON">
  <div class="arm">B1</div>
  <img src="media/scenarios/boundary_recovery_road_edge/b1_cbf_off.gif" alt="B1 rollout, boundary recovery road edge, CBF-OFF">
  <img src="media/scenarios/boundary_recovery_road_edge/b1_cbf_on.gif" alt="B1 rollout, boundary recovery road edge, CBF-ON">
  <div class="arm">B2</div>
  <img src="media/scenarios/boundary_recovery_road_edge/b2_cbf_off.gif" alt="B2 rollout, boundary recovery road edge, CBF-OFF">
  <img src="media/scenarios/boundary_recovery_road_edge/b2_cbf_on.gif" alt="B2 rollout, boundary recovery road edge, CBF-ON">
</div>

### Sudden lead slowdown

The leader slows sharply; the ego should react without diving into occupied
side gaps.

<div class="clip-grid">
  <div></div><div class="head">CBF-OFF</div><div class="head">CBF-ON</div>
  <div class="arm">A1</div>
  <img src="media/scenarios/sudden_lead_slowdown/a1_cbf_off.gif" alt="A1 rollout, sudden lead slowdown, CBF-OFF">
  <img src="media/scenarios/sudden_lead_slowdown/a1_cbf_on.gif" alt="A1 rollout, sudden lead slowdown, CBF-ON">
  <div class="arm">B1</div>
  <img src="media/scenarios/sudden_lead_slowdown/b1_cbf_off.gif" alt="B1 rollout, sudden lead slowdown, CBF-OFF">
  <img src="media/scenarios/sudden_lead_slowdown/b1_cbf_on.gif" alt="B1 rollout, sudden lead slowdown, CBF-ON">
  <div class="arm">B2</div>
  <img src="media/scenarios/sudden_lead_slowdown/b2_cbf_off.gif" alt="B2 rollout, sudden lead slowdown, CBF-OFF">
  <img src="media/scenarios/sudden_lead_slowdown/b2_cbf_on.gif" alt="B2 rollout, sudden lead slowdown, CBF-ON">
</div>

## Robustness

How close each policy's own action comes to breaking the safety constraints,
over ten diagnostic scenarios. The plotted quantity is the HOCBF margin of the
raw policy action,

S<sub>HOCBF</sub>(t) = min<sub>i</sub> [ b<sub>i</sub>(t) − A<sub>i</sub>(t) u<sub>raw</sub>(t) ],

the smallest slack across every vehicle-pair and road-boundary constraint.
Above the dashed zero line, the raw action satisfies every constraint; below
it, at least one is violated and a shield would have to correct it. Curves are
means over seven rollouts per policy, with the standard error shaded. A
rollout contributes no data after it crashes, so a curve that stops early
means no rollout of that policy was still running.

<div class="figure-grid">
  <figure>
    <a href="media/margins/open_road_no_nearby_traffic.png"><img src="media/margins/open_road_no_nearby_traffic.png" alt="HOCBF margin over time, open road scenario" loading="lazy"></a>
    <figcaption>No nearby vehicles: steady progress with little drift and near-zero correction is expected.</figcaption>
  </figure>
  <figure>
    <a href="media/margins/open_passing_gap_overtake_diagnostic.png"><img src="media/margins/open_passing_gap_overtake_diagnostic.png" alt="HOCBF margin over time, open passing gap scenario" loading="lazy"></a>
    <figcaption>A passing gap is available.</figcaption>
  </figure>
  <figure>
    <a href="media/margins/reject_overtake_fast_closing_vehicle.png"><img src="media/margins/reject_overtake_fast_closing_vehicle.png" alt="HOCBF margin over time, reject overtake scenario" loading="lazy"></a>
    <figcaption>The tempting gap holds a fast closing vehicle and should be rejected.</figcaption>
  </figure>
  <figure>
    <a href="media/margins/tight_slot_trap_shield_correction.png"><img src="media/margins/tight_slot_trap_shield_correction.png" alt="HOCBF margin over time, tight-slot trap scenario" loading="lazy"></a>
    <figcaption>A tight slot the policy should not dive into.</figcaption>
  </figure>
  <figure>
    <a href="media/margins/boundary_recovery_road_edge_squeeze.png"><img src="media/margins/boundary_recovery_road_edge_squeeze.png" alt="HOCBF margin over time, road-edge boundary recovery scenario" loading="lazy"></a>
    <figcaption>Recover inward from the road edge past a slow blocker.</figcaption>
  </figure>
  <figure>
    <a href="media/margins/boundary_recovery_opposite_edge.png"><img src="media/margins/boundary_recovery_opposite_edge.png" alt="HOCBF margin over time, opposite-edge boundary recovery scenario" loading="lazy"></a>
    <figcaption>The same recovery from the opposite road edge.</figcaption>
  </figure>
  <figure>
    <a href="media/margins/boxed_in_no_clean_gap.png"><img src="media/margins/boxed_in_no_clean_gap.png" alt="HOCBF margin over time, boxed-in scenario" loading="lazy"></a>
    <figcaption>Both side gaps are occupied; no pass should be forced.</figcaption>
  </figure>
  <figure>
    <a href="media/margins/rear_pressure_escape_closing_car.png"><img src="media/margins/rear_pressure_escape_closing_car.png" alt="HOCBF margin over time, rear pressure scenario" loading="lazy"></a>
    <figcaption>A fast rear vehicle closes on the ego.</figcaption>
  </figure>
  <figure>
    <a href="media/margins/sudden_lead_slowdown_reactive_safety.png"><img src="media/margins/sudden_lead_slowdown_reactive_safety.png" alt="HOCBF margin over time, sudden lead slowdown scenario" loading="lazy"></a>
    <figcaption>The leader slows sharply.</figcaption>
  </figure>
  <figure>
    <a href="media/margins/staggered_traffic_pick_clean_gap.png"><img src="media/margins/staggered_traffic_pick_clean_gap.png" alt="HOCBF margin over time, staggered traffic scenario" loading="lazy"></a>
    <figcaption>Thread through staggered traffic via the clean side.</figcaption>
  </figure>
</div>

## Results

Every policy is evaluated twice: with the shield off (CBF-OFF, the policy
alone) and on (CBF-ON, the policy plus the deployed HOCBF shield). An episode
completes only if the ego covers 1,000 m without a collision within 3,000
policy steps; in these evaluations a collision ends the episode. Entries are
means with 95% confidence intervals.

**Without the shield, only B2 is significantly safer than nominal PPO**: 2.55
against 4.04 collisions per km (p < 0.01), with higher return and nearly three
times the completion rate. **With the shield on, every policy lands at
0.24–0.30 collisions per km**, but B2 leans on it least: the shield intervenes
on 23% of its steps against 62% for A1 (p < 0.05), with the smallest
corrections.

### Table II: CBF-OFF evaluation

<div class="table-wrap">
  <table class="results">
    <thead>
      <tr><th>Policy</th><th>Coll./km ↓</th><th>Return ↑</th><th>Completion ↑</th><th>Speed err. ↓</th><th>Lat. err. ↓</th><th>Jerk ↓</th></tr>
    </thead>
    <tbody>
      <tr><th scope="row">Nominal PPO (A1)</th><td>4.04<span class="ci">[3.33, 4.74]</span></td><td>208.5<span class="ci">[159.1, 258.0]</span></td><td>0.036<span class="ci">[0.019, 0.053]</span></td><td>2.07<span class="ci">[1.82, 2.32]</span></td><td>1.99<span class="ci">[1.09, 2.90]</span></td><td>3.44<span class="ci">[2.43, 4.46]</span></td></tr>
      <tr><th scope="row">HOCBF State-Cost (B1)</th><td>3.74<span class="ci">[3.27, 4.22]</span></td><td>259.2<span class="ci">[232.4, 286.0]</span></td><td>0.056<span class="ci">[0.027, 0.085]</span></td><td>1.76<span class="ci">[1.45, 2.07]</span></td><td><strong>0.84</strong><span class="ci">[0.78, 0.90]</span></td><td>4.45<span class="ci">[2.40, 6.50]</span></td></tr>
      <tr><th scope="row">HOCBF Actor-Loss (B2)</th><td class="key">2.55<sup>**</sup><span class="ci">[2.15, 2.95]</span></td><td class="key">354.2<sup>***</sup><span class="ci">[314.5, 393.8]</span></td><td class="key">0.103<sup>**</sup><span class="ci">[0.070, 0.136]</span></td><td><strong>1.41</strong><sup>**</sup><span class="ci">[1.33, 1.49]</span></td><td>1.09<span class="ci">[0.88, 1.30]</span></td><td><strong>1.83</strong><sup>*</sup><span class="ci">[1.62, 2.04]</span></td></tr>
      <tr><th scope="row">HOCBF CMDP (C0)</th><td>3.72<span class="ci">[2.21, 5.24]</span></td><td>222.6<span class="ci">[57.2, 387.9]</span></td><td>0.038<span class="ci">[−0.012, 0.089]</span></td><td>2.20<span class="ci">[1.61, 2.78]</span></td><td>1.89<span class="ci">[−0.73, 4.52]</span></td><td>5.34<span class="ci">[0.86, 9.83]</span></td></tr>
    </tbody>
  </table>
</div>

### Table III: CBF-ON evaluation

<div class="table-wrap">
  <table class="results">
    <thead>
      <tr><th>Policy</th><th>Coll./km ↓</th><th>Interv. ↓</th><th>Corr. ↓</th><th>Return ↑</th><th>Completion ↑</th><th>Speed err. ↓</th><th>Lat. err. ↓</th><th>Jerk ↓</th></tr>
    </thead>
    <tbody>
      <tr><th scope="row">Nominal PPO (A1)</th><td>0.30<span class="ci">[0.22, 0.38]</span></td><td>0.62<span class="ci">[0.40, 0.84]</span></td><td>0.308<span class="ci">[0.108, 0.508]</span></td><td>699.9<span class="ci">[600.9, 799.0]</span></td><td>0.754<span class="ci">[0.697, 0.810]</span></td><td>1.94<span class="ci">[1.59, 2.29]</span></td><td>2.13<span class="ci">[1.23, 3.03]</span></td><td>6.17<span class="ci">[5.09, 7.25]</span></td></tr>
      <tr><th scope="row">HOCBF State-Cost (B1)</th><td>0.28<span class="ci">[0.26, 0.31]</span></td><td>0.37<span class="ci">[0.34, 0.40]</span></td><td>0.210<span class="ci">[0.166, 0.253]</span></td><td><strong>814.4</strong><span class="ci">[800.7, 828.0]</span></td><td>0.766<span class="ci">[0.745, 0.786]</span></td><td>1.67<span class="ci">[1.56, 1.78]</span></td><td><strong>1.00</strong><span class="ci">[0.93, 1.07]</span></td><td>7.85<span class="ci">[5.60, 10.10]</span></td></tr>
      <tr><th scope="row">HOCBF Actor-Loss (B2)</th><td>0.24<span class="ci">[0.21, 0.27]</span></td><td class="key">0.23<sup>*</sup><span class="ci">[0.21, 0.24]</span></td><td class="key">0.063<span class="ci">[0.059, 0.066]</span></td><td>810.0<span class="ci">[788.7, 831.3]</span></td><td>0.797<span class="ci">[0.773, 0.821]</span></td><td><strong>1.50</strong><span class="ci">[1.44, 1.56]</span></td><td>1.29<span class="ci">[1.13, 1.44]</span></td><td><strong>4.34</strong><sup>*</sup><span class="ci">[4.09, 4.59]</span></td></tr>
      <tr><th scope="row">HOCBF CMDP (C0)</th><td>0.24<span class="ci">[0.20, 0.28]</span></td><td>0.64<span class="ci">[0.04, 1.23]</span></td><td>0.347<span class="ci">[−0.126, 0.820]</span></td><td>732.0<span class="ci">[472.1, 992.0]</span></td><td><strong>0.800</strong><span class="ci">[0.767, 0.833]</span></td><td>1.91<span class="ci">[1.50, 2.31]</span></td><td>1.85<span class="ci">[−0.48, 4.19]</span></td><td>7.60<span class="ci">[3.19, 12.01]</span></td></tr>
    </tbody>
  </table>
</div>

<p class="caption">Mean [95% CI]. <sup>*</sup>p &lt; 0.05, <sup>**</sup>p &lt; 0.01,
<sup>***</sup>p &lt; 0.001, Holm-corrected against A1. Boxed values are the
primary outcomes in Table II and reduced filter dependence in Table III; bold
marks the best value in a column. Interv.: share of steps where the shield
changed the action. Corr.: size of its correction. Speed and lateral errors are
in m/s and m.</p>

## Documentation

A [short guide to the work and the code](documentation.html): the problem,
the shield, the evaluation, and the commands that reproduce this page. The
full documentation is in the
[repository](https://github.com/AhmadAssaad711/safeRL), starting with the
[experiment contract](https://github.com/AhmadAssaad711/safeRL/blob/main/safeRL_workspace/docs/lanelessKaralakou_reference.md).
