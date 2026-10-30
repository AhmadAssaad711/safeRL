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

| Label | Variant | How safety enters training |
| --- | --- | --- |
| A1 | `ppo_nominal` | None: the nominal reward only |
| B1 | `ppo_hocbf_state_cost_raw` | Fixed state-cost penalty in the reward |
| B2 | `ppo_cbf_nd_reward_actor` | Direct geometric actor loss, HOCBF-filtered training |
| C1 | `ppo_cbf_diff_reward_only` | Differentiable projection plus reward |
| C2 | `ppo_cbf_projected` | Differentiable projection plus actor loss |

## On the highway

Each policy driving in the full 40-vehicle traffic. The ego vehicle is
green. The left clip has the shield off (CBF-OFF) and the right clip has it on
(CBF-ON). A clip stops when its episode ends, so a shorter clip is an episode
that ended early.

<div class="clip-grid">
  <div></div><div class="head">CBF-OFF</div><div class="head">CBF-ON</div>
  <div class="arm">A1</div>
  <video src="media/full_road/a1_cbf_off.mp4" autoplay loop muted playsinline></video>
  <video src="media/full_road/a1_cbf_on.mp4" autoplay loop muted playsinline></video>
  <div class="arm">B1</div>
  <video src="media/full_road/b1_cbf_off.mp4" autoplay loop muted playsinline></video>
  <video src="media/full_road/b1_cbf_on.mp4" autoplay loop muted playsinline></video>
  <div class="arm">C1</div>
  <video src="media/full_road/c1_cbf_off.mp4" autoplay loop muted playsinline></video>
  <video src="media/full_road/c1_cbf_on.mp4" autoplay loop muted playsinline></video>
</div>

## Scenarios

Fixed diagnostic scenes that each test one driving decision. Every policy
starts from the same scene, with the shield off and on. The camera follows the
green ego vehicle.

### Open passing gap

A passing gap is available; compare which policies actually commit to it.

<div class="clip-grid">
  <div></div><div class="head">CBF-OFF</div><div class="head">CBF-ON</div>
  <div class="arm">A1</div>
  <video src="media/scenarios/open_passing_gap/a1_cbf_off.mp4" autoplay loop muted playsinline></video>
  <video src="media/scenarios/open_passing_gap/a1_cbf_on.mp4" autoplay loop muted playsinline></video>
  <div class="arm">B1</div>
  <video src="media/scenarios/open_passing_gap/b1_cbf_off.mp4" autoplay loop muted playsinline></video>
  <video src="media/scenarios/open_passing_gap/b1_cbf_on.mp4" autoplay loop muted playsinline></video>
  <div class="arm">B2</div>
  <video src="media/scenarios/open_passing_gap/b2_cbf_off.mp4" autoplay loop muted playsinline></video>
  <video src="media/scenarios/open_passing_gap/b2_cbf_on.mp4" autoplay loop muted playsinline></video>
</div>

### Boxed in

Both nearby side gaps are occupied; the policy should not force a pass.

<div class="clip-grid">
  <div></div><div class="head">CBF-OFF</div><div class="head">CBF-ON</div>
  <div class="arm">A1</div>
  <video src="media/scenarios/boxed_in/a1_cbf_off.mp4" autoplay loop muted playsinline></video>
  <video src="media/scenarios/boxed_in/a1_cbf_on.mp4" autoplay loop muted playsinline></video>
  <div class="arm">B1</div>
  <video src="media/scenarios/boxed_in/b1_cbf_off.mp4" autoplay loop muted playsinline></video>
  <div class="missing">clip not available</div>
  <div class="arm">B2</div>
  <video src="media/scenarios/boxed_in/b2_cbf_off.mp4" autoplay loop muted playsinline></video>
  <video src="media/scenarios/boxed_in/b2_cbf_on.mp4" autoplay loop muted playsinline></video>
</div>

### Rear pressure

A fast rear vehicle closes on the ego, which should move toward the cleaner
side gap.

<div class="clip-grid">
  <div></div><div class="head">CBF-OFF</div><div class="head">CBF-ON</div>
  <div class="arm">A1</div>
  <video src="media/scenarios/rear_pressure/a1_cbf_off.mp4" autoplay loop muted playsinline></video>
  <video src="media/scenarios/rear_pressure/a1_cbf_on.mp4" autoplay loop muted playsinline></video>
  <div class="arm">B1</div>
  <video src="media/scenarios/rear_pressure/b1_cbf_off.mp4" autoplay loop muted playsinline></video>
  <video src="media/scenarios/rear_pressure/b1_cbf_on.mp4" autoplay loop muted playsinline></video>
  <div class="arm">B2</div>
  <video src="media/scenarios/rear_pressure/b2_cbf_off.mp4" autoplay loop muted playsinline></video>
  <video src="media/scenarios/rear_pressure/b2_cbf_on.mp4" autoplay loop muted playsinline></video>
</div>

### Boundary recovery

The ego starts near the road edge behind a slow blocker and should recover
inward.

<div class="clip-grid">
  <div></div><div class="head">CBF-OFF</div><div class="head">CBF-ON</div>
  <div class="arm">A1</div>
  <video src="media/scenarios/boundary_recovery_road_edge/a1_cbf_off.mp4" autoplay loop muted playsinline></video>
  <video src="media/scenarios/boundary_recovery_road_edge/a1_cbf_on.mp4" autoplay loop muted playsinline></video>
  <div class="arm">B1</div>
  <video src="media/scenarios/boundary_recovery_road_edge/b1_cbf_off.mp4" autoplay loop muted playsinline></video>
  <video src="media/scenarios/boundary_recovery_road_edge/b1_cbf_on.mp4" autoplay loop muted playsinline></video>
  <div class="arm">B2</div>
  <video src="media/scenarios/boundary_recovery_road_edge/b2_cbf_off.mp4" autoplay loop muted playsinline></video>
  <video src="media/scenarios/boundary_recovery_road_edge/b2_cbf_on.mp4" autoplay loop muted playsinline></video>
</div>

### Sudden lead slowdown

The leader slows sharply; the ego should react without diving into occupied
side gaps.

<div class="clip-grid">
  <div></div><div class="head">CBF-OFF</div><div class="head">CBF-ON</div>
  <div class="arm">A1</div>
  <video src="media/scenarios/sudden_lead_slowdown/a1_cbf_off.mp4" autoplay loop muted playsinline></video>
  <video src="media/scenarios/sudden_lead_slowdown/a1_cbf_on.mp4" autoplay loop muted playsinline></video>
  <div class="arm">B1</div>
  <video src="media/scenarios/sudden_lead_slowdown/b1_cbf_off.mp4" autoplay loop muted playsinline></video>
  <video src="media/scenarios/sudden_lead_slowdown/b1_cbf_on.mp4" autoplay loop muted playsinline></video>
  <div class="arm">B2</div>
  <video src="media/scenarios/sudden_lead_slowdown/b2_cbf_off.mp4" autoplay loop muted playsinline></video>
  <video src="media/scenarios/sudden_lead_slowdown/b2_cbf_on.mp4" autoplay loop muted playsinline></video>
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

*Coming next: the CBF-OFF and CBF-ON results tables.*

## Documentation

*Coming next: links to the experiment contract and code documentation.*
