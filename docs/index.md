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

## Videos

*Coming next: rollout clips of each policy with the shield off and on.*

## Results

*Coming next: the CBF-OFF and CBF-ON results tables.*

## Documentation

*Coming next: links to the experiment contract and code documentation.*
