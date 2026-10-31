# safeRL

Safe reinforcement learning on a lane-free highway. A PPO agent controls
continuous longitudinal and lateral acceleration among 40 vehicles in MTM
traffic. A high-order control barrier function (HOCBF) shield can correct its
commands before they reach the simulator. The study compares ways of bringing
safety into training, and evaluates every policy both with the shield off
(CBF-OFF) and on (CBF-ON).

**Project page:** <https://ahmadassaad711.github.io/safeRL/>.
It has rollout videos in full traffic and in diagnostic scenarios,
robustness plots, the results and a short guide to the work.

## Contents

- [Research question](#research-question)
- [Environment](#environment)
- [Safety shield](#safety-shield)
- [Reward](#reward)
- [Policies compared](#policies-compared)
- [Training and evaluation](#training-and-evaluation)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Running the code](#running-the-code)
- [Reproducing the study](#reproducing-the-study)
- [Tests and repository checks](#tests-and-repository-checks)
- [Documentation](#documentation)
- [Contributing](#contributing)

## Research question

A runtime shield can keep a policy safe without the policy learning anything
about safety. This study separates the two:

- **CBF-OFF** measures the policy's own action map, with no shield at
  deployment;
- **CBF-ON** measures the policy plus the deployed HOCBF shield.

The policies differ only in how safety information reaches training: not at
all, through a reward penalty built from the barrier, or through an actor
loss built from the shield's output. A policy
that has internalized safety should stay safe with the shield off, not only
with it on.

## Environment

The custom Gymnasium environment `lane-free-v0` is a lane-free highway: the
road has no lane indices, lane centres or lane-change actions, and vehicles
choose any lateral position.

| Item | Value |
| --- | --- |
| Road | Periodic 380 m by 10.2 m ring; longitudinal distance wraps around |
| Traffic | 40 vehicles; surrounding traffic follows the MTM model with cautious-to-aggressive personalities |
| Surrounding speeds | Desired speeds sampled from 15 to 25 m/s |
| Ego speed target | 16 m/s |
| Timing | 100 Hz physics (dt = 0.01 s), 20 Hz policy, 20 Hz CBF; each action is held for five physics frames |
| Action | Physical acceleration (a_x, a_y), both components in [-3, 3] m/s² |
| Observation | Normalized ego row plus nearest-neighbor rows of [dx, dy, vx, vy, v_desired], plus the previous executed action |

**Observation width.** The width is derived, not fixed:
`(1 + neighbors_count) * 5 + 2`. The canonical notebook contract uses five
neighbor rows (32D). The study reported on the project page uses
seven neighbor rows and drops the constant ego desired-speed slot (41D).
Always state `neighbors_count` with an observation width.

**Action stages.** The action passes through distinct, separately logged
stages: actor mean, sampled action, box-clipped action, CBF-safe action, the
normalized simulator command, and the acceleration actually executed. The
previous *executed* action is what enters the next observation.

The simulator also guards social-social vehicle pairs. By default this guard
does not react to the ego, so avoiding collisions is left to the ego policy
and its shield.

## Safety shield

Each nearby vehicle is enclosed in a fixed axis-aligned ellipse sized for a
3.6 m by 1.8 m reference vehicle, and the barrier

```text
h(dx, dy) = dx² / A² + dy² / B² − 1,   A = 3.6/√2 m,  B = 1.8/√2 m
```

is non-negative on the collision-free side. The two road edges add boundary
barriers. A second-order HOCBF cascade with class-K rates c1 and c2,

```text
ψ1 = ḣ + c1·h
ψ2 = ψ̇1 + c2·ψ1 = ḧ + (c1 + c2)·ḣ + c1·c2·h ≥ 0
```

makes each barrier a linear constraint on the acceleration command. The
canonical gains are (c1, c2) = (0.5, 8), i.e. k1 = 8.5 and k0 = 4.0.

At every policy step the shield:

1. builds rows for the nearest neighbors (up to a configured maximum) and both
   road edges;
2. keeps the policy's action if it satisfies every row;
3. otherwise projects it onto the feasible set with an exact 2D active-set
   solver, with OSQP as a numerical fallback;
4. if nothing is feasible, falls back to the least-violating bounded action
   and then an evasive emergency action.

Every step records the minimum h, the HOCBF margin, the correction size, the
solver status and any fallback.

## Reward

The Karalakou reward combines a tracking term with progress and event terms:

```text
r = ε / (ε + wx·cx + wy·cy + wf·cf + way·cay)
    + progress − jerk_penalty + collision_or_overtake_adjustment
```

- `cx`: normalized ego-speed error from the target speed;
- `cy`: normalized lateral error from a target lateral position;
- `cf`: potential-field cost from nearby vehicles;
- `cay`: lateral-acceleration comfort cost;
- progress: forward metres, normalized by 2.0 m;
- a bounded jerk penalty, a collision penalty of −2.5, and an overtake bonus
  of +0.5.

The study runs set the potential-field weight to zero and use a smooth
occupancy field to choose the lateral target. Reward variants are documented
in the [experiment contract](safeRL_workspace/docs/lanelessKaralakou_reference.md#karalakou-reward).

## Policies compared

| Label | Variant | CBF in training | How safety reaches the policy |
| --- | --- | --- | --- |
| A1 | `ppo_nominal` | No | None: the nominal reward only |
| B1 | `ppo_hocbf_state_cost_raw` | No | Fixed barrier-based state-cost penalty in the reward |
| B2 | `ppo_cbf_nd_reward_actor` | Yes, hard shield | Reward penalty plus a detached actor loss toward the shield's safe action |

The policies form a ladder: A1 never sees the barrier, B1 sees it only as a
cost in its reward, and B2 learns directly from the shield's corrections.

## Training and evaluation

The study (`scripts/ops/run_safety_matrix_study.py`) trains every policy the
same way:

| Setting | Value |
| --- | --- |
| Algorithm | Stable-Baselines3 PPO, `Q1_stable` configuration |
| PPO hyperparameters | lr 1e-4, γ 0.99, GAE λ 0.95, clip 0.2, entropy 0.005, 10 epochs, batch 100 |
| Networks | Separate policy and value MLPs, [256, 128] each |
| Rollouts | 20 parallel environments sharing a 1,000-step rollout (50 steps each) |
| Budget | 1,000,000 timesteps per policy and seed |
| Seeds | 307, 327, 347, 367, 387, 407, 427 |
| Observation | 41D: seven neighbor rows, ego desired-speed slot dropped |
| CBF rows | At most seven neighbor constraints plus the road edges |
| Actor-loss weight | λ = 1.0 for B2 in the reported tables (a follow-up to the base study's 0.1) |

The seeds are spaced by 20, the number of parallel environments. Worker seeds
are `s` to `s + 19`, so seeds closer than 20 apart would share scenario
streams and would not be independent replicates.

**Evaluation.** After training, each policy runs 200 complete episodes with
the shield off and the same 200 scenario seeds with it on, starting from
seed 1,100,000. An episode **completes** only if the ego covers 1,000 m
without a collision within 3,000 policy steps; a collision ends the episode.

| KPI | Meaning |
| --- | --- |
| Ego collisions / km | Ego collisions per kilometre driven |
| Minimum h | Lowest barrier value reached; negative means inside an ellipse |
| QP failure rate | Share of steps where the shield found no feasible action |
| Intervention rate | Share of steps where the shield changed the action |
| Correction norm | Size of the shield's correction, normalized, after box clipping |
| Episode return | Total reward per episode |
| Episode length | Policy steps per episode |
| Completion rate | Share of episodes that complete the 1,000 m task |
| Abs speed error | Mean absolute deviation from the target speed (m/s) |
| Lateral tracking error | Mean distance from the lateral target (m) |
| Mean jerk norm | Mean size of the change in applied acceleration |

Each seed's 200 episodes give one value per KPI. Tables report the mean over
seeds with a 95% confidence interval, and test each policy against A1 with a
Holm correction for multiple comparisons.

## Results

Headline results, mean with 95% confidence interval. Significance is
Holm-corrected against A1: \* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001.

| Policy | CBF-OFF coll./km ↓ | CBF-OFF completion ↑ | CBF-OFF return ↑ | CBF-ON coll./km ↓ | CBF-ON intervention ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1, nominal PPO | 4.04 [3.33, 4.74] | 0.036 [0.019, 0.053] | 208.5 [159.1, 258.0] | 0.30 [0.22, 0.38] | 0.62 [0.40, 0.84] |
| B1, HOCBF state cost | 3.74 [3.27, 4.22] | 0.056 [0.027, 0.085] | 259.2 [232.4, 286.0] | 0.28 [0.26, 0.31] | 0.37 [0.34, 0.40] |
| B2, HOCBF actor loss | **2.55\*\*** [2.15, 2.95] | **0.103\*\*** [0.070, 0.136] | **354.2\*\*\*** [314.5, 393.8] | 0.24 [0.21, 0.27] | **0.23\*** [0.21, 0.24] |

Without the shield, only B2 is significantly safer than nominal PPO. With the
shield on, all three reach 0.24–0.30 collisions per km, but the shield
intervenes on far fewer of B2's steps. The full tables (every KPI), rollout
videos and robustness plots are on the
[project page](https://ahmadassaad711.github.io/safeRL/).

Each run writes its per-seed KPIs to `post_train_200ep_kpis.csv`. Run outputs
are not committed (see [artifacts/](artifacts/README.md)), so the tables are
built on the machine that holds the runs. The confidence intervals and
Holm-corrected tests in the paper are not yet produced by a committed script;
`scripts.reporting.build_final_variant_comparison_tables` gives the per-seed
mean ± SD summary of the same files.

## Repository layout

```text
safeRL/
├── README.md
├── AGENTS.md                     architecture and change rules
├── artifacts/                    committed manifests and compact summaries
├── docs/                         GitHub Pages project page (Jekyll)
└── safeRL_workspace/             implementation; run commands from here
    ├── notebooks/                canonical experiment notebook
    ├── laneless highway env/     lane-free-v0 environment and renderer
    ├── configs/                  environment and traffic configuration
    ├── src/saferl/               new research code: config, rewards, safety, artifacts
    ├── scripts/
    │   ├── common/               notebook-compatible adapters, CBF projection
    │   ├── training/             PPO training, pilots, ablations
    │   ├── evaluation/           KPIs, audits, counterfactuals, comparisons
    │   ├── reporting/            tables, figures, dashboards
    │   ├── rendering/            rollout GIFs and scenario renders
    │   └── ops/                  smoke checks, study launchers, policy checks
    ├── tests/                    unit, contract and protocol tests
    └── docs/                     experiment and code documentation
```

The notebook
[lanelessKaralakou.ipynb](safeRL_workspace/notebooks/lanelessKaralakou.ipynb)
is the canonical experiment contract and is frozen; new research code goes in
`src/saferl/` and new experiments in `scripts/`.

## Setup

Python 3.10 or newer. On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r safeRL_workspace/requirements.txt
```

On Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r safeRL_workspace\requirements.txt
```

The main dependencies are Gymnasium, highway-env, Stable-Baselines3, PyTorch,
and the QP solvers qpsolvers, OSQP and Clarabel.

## Running the code

Run every command from `safeRL_workspace` as a module. Each script documents
its options under `--help`; read it before launching anything long.

```bash
cd safeRL_workspace
python -m scripts.ops.mtm_laneless_smoke --help
```

| Task | Module |
| --- | --- |
| Environment smoke check | `scripts.ops.mtm_laneless_smoke` |
| Train PPO variants | `scripts.training.run_ppo_cbf_progression` |
| Evaluate canonical KPIs | `scripts.evaluation.evaluate_laneless_karalakou` |
| Study CBF filtering | `scripts.training.run_cbf_filter_ablation`, `scripts.evaluation.evaluate_cbf_counterfactuals` |
| Inspect learning signals | `scripts.evaluation.diagnose_update_signal_pipeline` |
| Render a policy | `scripts.rendering.render_laneless_policy_scenario` |

The [script and function reference](safeRL_workspace/docs/script_reference.md)
covers every module.

## Reproducing the study

The three policies are 21 training runs of 1,000,000 timesteps each. Use
`--dry-run` first to print the commands without running them.

```bash
cd safeRL_workspace
python -m scripts.ops.run_safety_matrix_study --dry-run --out-base <output-dir> \
    --only A1_nominal B1_state_cost_raw
python -m scripts.ops.run_safety_matrix_study --dry-run --out-base <output-dir> \
    --only B2_cbf_nd_reward_actor --actor-lambda 1.0
```

The launcher's default actor-loss weight is λ = 0.1; the reported B2 results
use λ = 1.0, hence the second command.

| Output | Module |
| --- | --- |
| A1, B1 and B2 × seven seeds | `scripts.ops.run_safety_matrix_study` |
| Per-seed KPI summary (mean ± SD) | `scripts.reporting.build_final_variant_comparison_tables` |
| Robustness (HOCBF margin) plots | `scripts.reporting.plot_hocbf_margin_aggregate` |
| Highway rollout GIFs | `scripts.rendering.render_policy_traffic_gifs` |
| Scenario rollout GIFs | `scripts.rendering.render_scenario_policy_gifs` |

Every serious run writes a manifest with its resolved configuration, seeds,
frequencies, Git commit and model checksum; see
[AGENTS.md](AGENTS.md#mandatory-experiment-configuration-and-manifests).

## Tests and repository checks

```bash
cd safeRL_workspace
python -m pytest -q tests
python -m scripts.ops.check_repository_policy
```

The tests are deterministic and protect scientific contracts: the canonical
configuration, observation rows and spaces, seeded resets, self-contained run
directories, the simulation timing, action units, CBF geometry, projection and
reset feasibility, and the reward terms. The policy check enforces the
repository rules in `AGENTS.md`.

## Documentation

| Document | Covers |
| --- | --- |
| [Experiment contract](safeRL_workspace/docs/lanelessKaralakou_reference.md) | Environment, observation, action path, reward, CBF geometry, evaluation |
| [Script and function reference](safeRL_workspace/docs/script_reference.md) | Every module and which one to run for which task |
| [Diagnostic scenarios](safeRL_workspace/docs/diagnostic_scenarios.md) | The fixed scenes behind the scenario videos and plots |
| [CBF factorial ablation](safeRL_workspace/docs/cbf_factorial_ablation.md) | Separating policy occupancy from same-state action effects |
| [HOCBF reward investigation](safeRL_workspace/docs/hocbf_reward_investigation.md) | HOCBF-based reward terms |
| [Trial log](safeRL_workspace/docs/experiments/trial_log.md) | Every trial after the main study, with verdicts |
| [Documentation index](safeRL_workspace/docs/README.md) | All other studies and handoff notes |

## Contributing

Read [AGENTS.md](AGENTS.md) before changing code. In short:

- keep the notebook frozen; new algorithms go in `src/saferl/`, new
  experiments in committed entrypoints under `scripts/`;
- import shared constants (timing, action bounds, CBF gains) from the central
  configuration rather than redeclaring them;
- give every reportable run a manifest, a clean worktree and its own output
  directory;
- make small, single-purpose commits, and run the tests and policy check
  before committing.
