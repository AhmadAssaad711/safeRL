# safeRL repository guidance

This file is the persistent architecture and change-policy entry point for
agents and contributors working anywhere in this repository. Read it before
inspecting or changing code. Keep it concise; put detailed, source-grounded
explanations in `safeRL_workspace/docs/`.

## Repository purpose and source of truth

This repository contains safe reinforcement-learning research code for a
laneless highway environment inspired by the Karalakou experiment. The Git
root is the parent directory of `safeRL_workspace`; the implementation and
Python package live under `safeRL_workspace/`.

The laneless source hierarchy is:

1. `safeRL_workspace/notebooks/lanelessKaralakou.ipynb` is the executable
   experiment contract.
2. `safeRL_workspace/docs/lanelessKaralakou_reference.md` is the maintained,
   source-grounded explanation of the notebook contract.
3. `safeRL_workspace/docs/script_reference.md` explains module responsibilities
   and workflow routing.
4. `safeRL_workspace/laneless highway env/lane_free_env.py` owns the base
   Gymnasium environment mechanics: vehicles, traffic, integration,
   collisions, observations, and environment diagnostics.

When code, a README, or memory conflicts with the notebook and reference,
inspect the executable source and update the relevant documentation. Do not
silently create a second experiment contract.

The notebook is currently a compatibility and research composition layer. It
is not the place for new reusable algorithms. New research logic belongs in
importable modules under `src/saferl/`; existing notebook-compatible logic
remains under `scripts/common/` until a scientific migration is justified.

## Architecture and ownership

| Layer | Owns | Must not own |
| --- | --- | --- |
| `notebooks/` | Canonical configuration, experiment composition, research narrative, small calls into reusable code | New reusable algorithms, hidden defaults, large production workflows |
| `laneless highway env/` | `lane-free-v0` dynamics, vehicles, MTM/force traffic, integration, collision and base observation mechanics | Paper-specific training loops or report generation |
| `scripts/common/` | Existing notebook-compatible configuration, registries, observation/worker adapters, CBF geometry/projection, and legacy algorithm components | New research algorithms or one-off experiment conclusions |
| `src/saferl/` | New research-only configuration, safety algorithms, contract utilities, and artifact helpers | Silent migration of historical behavior or compatibility hacks |
| `scripts/training/` | Training entry points, bounded pilots, ablations, protocol orchestration | Canonical evaluator implementations or report builders |
| `scripts/evaluation/` | Saved-model evaluation, KPIs, audits, counterfactuals, diagnostics, comparisons | Silent retraining or publication-only formatting |
| `scripts/reporting/` | Figures, dashboards, paper tables, provenance-preserving summaries | Training, evaluation side effects, or hidden data regeneration |
| `scripts/rendering/` | Bounded qualitative rollouts and videos | Replacing canonical KPI evaluation |
| `scripts/ops/` | Smoke checks, benchmarks, launchers, monitors, repository checks | New research algorithms |
| `tests/` | Deterministic unit and protocol tests | Long training runs, persistent artifacts, network dependence |

The normal dependency direction is:

```text
environment/common -> training -> evaluation -> reporting
          \-> rendering and ops
```

Training may call evaluation as an explicit post-training step, but reusable
evaluation logic belongs in `scripts/evaluation/`. Reporting consumes saved
results and must not retrain or silently change the experiment.

## Repository map and routing

- `safeRL_workspace/notebooks/`: canonical notebook and notebook metadata.
- `safeRL_workspace/laneless highway env/`: custom `lane-free-v0` environment;
  it has no lane indices, lane centers, target lanes, lane-change actions, or
  `highway-v0` assumptions.
- `safeRL_workspace/configs/`: reusable environment and traffic configuration.
- `safeRL_workspace/src/saferl/`: new research-only package. New algorithms,
  configuration objects, contract utilities, and self-contained artifact
  helpers begin here; do not migrate old code into it opportunistically.
- `safeRL_workspace/scripts/catalog.py`: compatibility mapping from legacy
  script filenames to organized importable modules.
- `safeRL_workspace/scripts/common/`: existing reusable compatibility logic.
  Do not add new research algorithms here by habit.
- `safeRL_workspace/scripts/training/`: PPO and explicitly requested legacy
  DDPG training, pilots, ablations, and protocol runners.
- `safeRL_workspace/scripts/evaluation/`: KPI evaluation, audits,
  counterfactuals, diagnostics, and cross-algorithm comparisons.
- `safeRL_workspace/scripts/reporting/`: plots, dashboards, and paper/report
  builders that consume upstream manifests and metrics.
- `safeRL_workspace/scripts/rendering/`: qualitative policy and scenario
  renders with bounded steps and explicit output paths.
- `safeRL_workspace/scripts/ops/`: bounded smoke checks, live inspection,
  simulator benchmarks, launchers, monitors, and repository checks.
- `safeRL_workspace/tests/`: deterministic unit and protocol tests.
- `safeRL_workspace/artifacts/`: manifests and generated research outputs;
  preserve existing user artifacts and never overwrite a result without an
  explicit output stem.
- `safeRL_workspace/pyproject.toml`: source-package metadata for the new
  `saferl` package. The historical `scripts` package remains runnable from
  `safeRL_workspace`.

For the complete function-level inventory, read
`safeRL_workspace/docs/script_reference.md`. Start here for common tasks:

| Need | Start with |
| --- | --- |
| Check the environment | `scripts/ops/mtm_laneless_smoke.py` |
| Check repository policy | `scripts/ops/check_repository_policy.py` |
| Run notebook code out of process | `scripts/training/run_laneless_notebook_task.py` |
| Evaluate canonical KPIs | `scripts/evaluation/evaluate_laneless_karalakou.py` |
| Screen/train PPO variants | `scripts/training/run_ppo_formulation_screen.py`, then `run_ppo_cbf_progression.py` |
| Study CBF filtering | `scripts/training/run_cbf_filter_ablation.py`, `scripts/evaluation/evaluate_cbf_counterfactuals.py` |
| Compare PPO and DDPG | `scripts/evaluation/compare_nominal_ppo_ddpg.py` |
| Inspect learning signals | `scripts/evaluation/diagnose_update_signal_pipeline.py` |
| Produce figures/reports | `scripts/reporting/` after manifests are complete |
| Inspect behavior visually | `scripts/rendering/` with bounded steps |

## Canonical laneless experiment assumptions

Unless a task explicitly requests a non-canonical demo or ablation, preserve
these assumptions from the current notebook and reference:

- environment ID: `lane-free-v0`;
- MTM surrounding traffic;
- periodic road: 380 m by 10.2 m;
- 55 vehicles and five visible neighbor rows;
- physics: 0.01 s / 100 Hz;
- policy and hard-CBF rate: 20 Hz, with five physics frames per policy action;
- PPO observation: 30D target-y vehicle features plus two previous normalized
  executed-action features;
- retained legacy DDPG reference: separate 42D observation;
- common physical acceleration action box: `[-3, 3]` for both components;
- strict evaluation: collision-free 1,000 m completion within 3,000 policy
  steps; a collision prevents completion;
- CBF-OFF measures the policy action map; CBF-ON measures policy plus deployed
  shielding;
- the simulator traffic dynamics guard remains enabled for social-social pairs,
  while `traffic_safety.guard_ego_interactions` defaults to `false`; social
  traffic reacts to the controlled ego only when that option is explicitly set
  to `true`;
- for `ppo_hocbf_reward_raw`, an omitted `hocbf-psi-scale` is calibrated once
  from a deterministic unshielded rollout of a fixed `ppo_nominal` checkpoint
  and frozen across all HOCBF-reward seeds. Zero-action calibration is only
  historical provenance, not a valid learning-comparison protocol.

The canonical timing is confirmed as 100 Hz physics, 20 Hz policy, and 20 Hz
CBF. Any 10 Hz policy statement is stale and must not be copied into new
experiments. If the rate is intentionally changed in the future, update the
notebook, reference, tests, configuration object, manifests, and this file
together.

Keep normalized policy actions, physical actions, raw actions, safe actions,
and executed actions distinct. Diagnostic records should preserve the raw,
safe, and executed stages, correction, solver/fallback status, seed, variant,
notebook source hash, configuration, and artifact paths.

Never use `lane_index`, lane-indexed highway assumptions, `highway-v0`, or old
DQN helpers as a shortcut in a laneless experiment. Do not change a default in
one script merely to make a run pass; use explicit configuration/overrides and
record them in the result manifest.

## New research code and centralized constants

New research code starts in `safeRL_workspace/src/saferl/`. The first intended
branch is:

```text
src/saferl/
└── safety/
    ├── learnable_cbf.py
    ├── hocbf.py
    └── parameterization.py
```

This is a home for new work, not a migration mandate. Existing
`scripts/common/*` remains the compatibility surface for old experiments and
stays untouched unless there is a specific scientific reason to change it.

Do not independently hardcode shared experiment constants in new scripts.
Values such as CBF gains, side margins, action bounds, road dimensions,
observation layout, and frequencies must come from one imported Python
configuration object. The current canonical timing is 100 Hz physics, 20 Hz
policy, and 20 Hz CBF. A new branch must import those values rather than
redeclare `POLICY_HZ`, `K0`, `K1`, or `EPS_SIDE` locally.

The configuration object must expose both the resolved values and derived
contracts—physics step, frames per policy action, expected observation shape,
physical action bounds, and evaluation limits. Experiment-specific changes
should create an explicit derived configuration and record it in the manifest;
they must not mutate process-global defaults.

Do not introduce YAML merely to satisfy this rule. A typed Python configuration
object is sufficient until the number of independent configurations justifies
another format.

## Contract-first tests

Tests should protect scientific contracts, not maximize implementation
coverage. Prefer small deterministic tests that would catch silent scientific
invalidation:

- observation shape and feature definition equal the declared contract;
- normalized and physical action bounds remain unchanged;
- physics/policy/CBF frequencies are 100/20/20 for the canonical experiment;
- the same seed produces the same initial scenario;
- CBF projections satisfy active constraints or explicitly report fallback;
- CBF-OFF bypasses CBF projection rather than merely reporting zero correction;
- declared reward components sum to the returned reward;
- collision and strict completion definitions remain unchanged;
- manifest/config fingerprints change when a scientific input changes.

Use focused unit/contract tests for these invariants and bounded protocol tests
for workflow behavior. Do not add broad tests that freeze incidental class
layout, private helper names, or historical cell numbers without scientific
value.

## Self-contained experiment artifacts

Every new serious run must produce a self-contained directory that can be
understood without opening the training script:

```text
run_<timestamp>_<slug>/
├── model.zip
├── config.json
├── manifest.json
├── training_metrics.csv
├── evaluation/
└── notes.md
```

The exact files may be extended, but the run directory must contain the
resolved configuration, manifest, model/checkpoint references and checksums,
training metrics, evaluation outputs, and human-readable notes. A manifest in
some parent directory is not enough. Use atomic writes, explicit run IDs, and
never reuse an output directory silently.

Keep run directories compact. Store configuration, manifest, notes, summaries,
and selected checkpoints; do not copy temporary PowerShell launchers or other
one-off glue into them. Large logs, videos, caches, and generated model sets
remain uncommitted outputs unless a result-selection policy explicitly says
otherwise.

The manifest must include at least: algorithm and variant, environment
configuration, reward configuration, CBF/safety configuration, observation
definition, seed(s), policy/physics/CBF frequencies, training steps, Git
commit and dirty-state information, model path/checksum, source hashes, and
the evaluation protocol. Training and evaluation registries should enforce
this schema for new runs; legacy artifacts require an explicit compatibility
path.

## Naming and path discipline

The actual repository is `safeRL`, and commands run from the
`safeRL_workspace` directory. Do not introduce the stale
`highway-rl-decision-making` path into new documentation, manifests, scripts,
or commands. Use `safeRL_workspace` or a repository-relative path.

Use these terms consistently:

- `laneless` describes the research/task family;
- `lane-free-v0` is the Gymnasium environment ID;
- `lane_free_env.py` is the historical implementation module;
- `src/saferl/` is the new research package.

The directory named `laneless highway env/` is retained temporarily because
the frozen notebook and legacy scripts reference it. Do not rename it in a
behavioral refactor. New code should reach the environment through the
canonical factory and should not copy that path into new APIs. Fix stale text
paths now; defer the physical directory rename until a factory migration can
prove legacy parity.

## Experimental code versus core code

Experimental code may die. Core code may not.

It is acceptable to create a focused script such as
`run_learnable_cbf_parameter_pilot.py`, use it to test a hypothesis, document
the run, and delete it later. Keep such code small, explicit, and isolated.
Do not prematurely generalize every experiment into a framework. Promote an
experimental component into core/shared code only when it has a stable
interface, a scientific reason for reuse, contract tests, and provenance
requirements.

Classify new work before deciding what enters Git:

```text
CORE        reusable implementation              -> commit
PROTOCOL    scientifically meaningful experiment -> commit
GLUE        one-off launch/monitor/debug code    -> delete after use
RESULT      large/generated output               -> do not commit
PROVENANCE  config/manifest/summary/notes        -> commit selectively
```

Temporary launchers are not a substitute for a reproducible experiment
entrypoint. Do not normally copy them into run directories. If reproducing a
launcher is scientifically important, that is evidence that its protocol or
implementation belongs in a committed entrypoint or an explicitly archived
Git commit/tag.

The objective is experimental integrity, not cosmetic repository cleanliness.
Avoid broad architecture rewrites when a config, manifest, parity test, or
small adapter prevents scientific drift.

## Methodical Git and commit discipline

This repository is now directed methodically. Git history is part of the
scientific record, especially for experiment configuration and results.

Use branches for actual lines of development, not for every experiment,
seed, or pilot. Typical branches include `feature/learnable-cbf`,
`perf/simulator-optimizations`, and `fix/evaluation-protocol`. Individual
experiments are identified by the combination of Git SHA, resolved
configuration, seed, and run manifest. Do not create branch clutter merely to
name a run.

Before changing code:

- inspect `git status --short` and the relevant diff;
- preserve unrelated user changes;
- identify the intended files and the scientific purpose of the change;
- do not start a reportable experiment from an unexplained dirty state.

Commit rules:

1. Make commits small, atomic, and single-purpose. Separate policy/config,
   implementation, tests, documentation, and experiment-result changes when
   they can be reviewed independently.
2. Do not mix formatting sweeps, path renames, refactors, and algorithmic
   changes in one commit.
3. Use an imperative, scoped subject such as:

   ```text
   policy(agents): require self-contained experiment manifests
   config(laneless): centralize canonical 100/20/20 timing
   test(contract): protect deterministic laneless reset
   experiment(cbf): add learnable-CBF parameter pilot
   ```

4. Stage explicit paths with `git add -- <paths>`. Never use a broad staging
   command that could capture models, logs, caches, temporary files, or
   unrelated user work.
5. Before committing, review the staged diff, run `git diff --check`, run the
   repository policy checker, and run the smallest relevant focused tests.
6. A serious or paper-facing training/evaluation run must start from a named
   commit with a clean worktree, an immutable resolved configuration, and a
   manifest. A SHA with `dirty: true` is not sufficient reproducibility. New
   serious-run entrypoints should refuse a dirty worktree by default; an
   explicit exploratory override may allow it, must warn, and must record the
   dirty state plus a diff hash or preserved patch.
7. Do not rewrite history, reset user changes, force-push, or discard work
   without explicit authorization.

The preferred sequence for a new research idea is:

```text
policy/config commit
  -> contract-test commit
  -> implementation commit
  -> committed entrypoint/config and clean-tree bounded pilot
  -> evaluation/report commit
```

An experimental branch or pilot script may be deleted later, but its useful
result must remain reconstructable through its manifest, configuration, Git
commit, and notes. Commit history should make it possible to answer what
changed, why it changed, and which results depended on it.

## Mandatory change rules

These are repository rules, not suggestions:

1. **No new core algorithms inside notebooks.** A notebook may explain,
   compose, visualize, or call an algorithm. New reusable policies, filters,
   geometry, rewards, wrappers, training logic, or diagnostics belong in
   importable Python modules first.
2. **New research logic goes in `src/saferl/`.** Add a focused module with
   tests and a documented interface. Existing compatibility logic stays in
   `scripts/common/`; do not copy it into `src/saferl/` without a scientific
   migration plan, and do not add new algorithms to either location without a
   clear ownership decision.
3. **No new 2,000-line experiment scripts.** Existing oversized runners are
   migration debt and are explicitly allowlisted by the repository checker.
   Do not enlarge them with another subsystem; extract reusable or protocol
   logic before adding substantial functionality.
4. **Keep training, evaluation, reporting, rendering, and operations separate.**
   A runner may orchestrate these stages, but reusable responsibilities stay
   in their owning directory.
5. **DDPG is legacy unless explicitly being studied.** New algorithmic work
   uses PPO and the current laneless interfaces by default. Changes to DDPG
   must identify the historical comparison or requested study and preserve its
   separate observation/action contract.
6. **Preserve the notebook contract.** New experiments should use existing
   configuration helpers, registries, observation adapters, and provenance
   fields rather than copying notebook defaults.
7. **Add bounded verification with every behavior change.** Prefer a focused
   unit/protocol test and a smoke or `--help` check before any long run.
8. **Never hide expensive or stateful work.** Evaluation must not silently
   retrain; reporting must not silently regenerate upstream data; commands
   must use explicit seeds, budgets, and output paths.
9. **Do not overwrite results.** Use a new run directory or explicit output
   stem, publish final summaries atomically, and retain resumable progress for
   long evaluations.
10. **Update the documentation contract when behavior changes.** Changes to
    environment, reward, observations, CBF semantics, frequencies, or KPI
    definitions require updates to the reference docs and relevant tests.
11. **Every canonical experiment must have a committed entrypoint.** New
    experiments must be runnable through a committed Python entrypoint under
    `scripts/training/`, `scripts/evaluation/`, `scripts/rendering/`, or the
    appropriate new package surface. A temporary PowerShell launcher may wrap
    that entrypoint, but it must not be the only place where the experiment
    protocol exists. One-off launchers may be deleted after use.

Notebook cell execution and the notebook bridge are existing compatibility
mechanisms. Do not add new cell-index coupling when an importable module can
be used. If a temporary bridge is unavoidable, document the dependency and
the migration path.

## Scientific lifecycle: freeze the notebook

Treat the notebook as a **canonical historical experiment and interactive
analysis surface**. Treat the scripts as the home for **all new experiments**.

The notebook is frozen architecturally. Do not refactor it, move its cells, or
expand it with new algorithm families as part of ordinary development. A
small notebook edit is allowed only when it is necessary to:

- correct a demonstrable bug in the historical contract;
- make provenance or documentation more explicit without changing behavior;
- preserve a reproducibility fix that has a focused parity test.

Any such edit must update the maintained reference, record the notebook hash
in affected manifests, and state whether historical results remain comparable.
New learnable-CBF work, projection methods, reward formulations, observation
variants, training loops, diagnostics, and evaluation protocols must be
implemented outside the notebook. The notebook may call those modules for
interactive analysis, but it must not become their implementation site.

This policy intentionally reverses the dependency direction gradually:

```text
frozen notebook -> historical reference and analysis
scripts/common  -> reusable implementation
scripts/training/evaluation/reporting -> all new research workflows
```

Do not attempt to make old notebook experiments use the new factory or new
modules in the same change as this policy. Historical behavior must remain
available while parity is being established.

## Mandatory experiment configuration and manifests

Every reportable training run, pilot, ablation, evaluation, comparison, or
paper-facing rendering run must have both:

1. a resolved experiment configuration; and
2. an immutable, versioned run manifest.

The existing training and evaluation registries are the starting point. Extend
them instead of creating per-script manifest formats. A serious run is not
complete or publishable merely because a model or CSV exists; its manifest
must validate successfully.

There are two execution classes:

- **Serious/paper run:** committed Git SHA, clean worktree, immutable resolved
  configuration, validated manifest, and an explicit committed entrypoint.
  The runner must warn or refuse before expensive work when these conditions
  are not met; refusal is the default for new canonical runners.
- **Exploratory pilot:** may use a dirty worktree, but only through an explicit
  opt-in override. The runner must warn and record the dirty state and diff
  provenance. A pilot may later be discarded, but a reportable result must be
  rerun from the serious-run class.

At minimum, the manifest must record:

```json
{
  "schema_version": 1,
  "experiment": {"id": "...", "variant": "...", "entrypoint": "..."},
  "algorithm": {"name": "PPO", "implementation": "..."},
  "environment": {"id": "lane-free-v0", "config": {}},
  "reward": {"name": "Karalakou", "config": {}},
  "observation": {"definition": {}, "shape": []},
  "cbf": {"enabled": false, "config": {}},
  "randomness": {"seed": 0, "worker_seeds": [], "evaluation_seeds": []},
  "frequencies": {"physics_hz": 100, "policy_hz": 20, "cbf_hz": 20},
  "training": {"timesteps": 0, "n_envs": 1, "device": "..."},
  "evaluation_protocol": {},
  "source": {
    "git_commit": "...",
    "git_dirty": false,
    "notebook_hash": "...",
    "module_hashes": {}
  },
  "artifacts": {"model_path": "...", "model_sha256": "..."}
}
```

The actual schema may grow, but these facts must not disappear. In particular:

- store the fully resolved environment, reward, observation, and CBF configs,
  not only CLI overrides;
- record the Git commit and whether the worktree was dirty. For a dirty run,
  record a diff hash or preserved patch as well;
- record the notebook hash whenever notebook-defined compatibility code is
  used, plus hashes or versions for reusable Python modules that affect the
  run;
- record policy/physics/CBF frequencies, training steps, worker count, model
  path and checksum, and the complete evaluation protocol;
- use canonical JSON ordering for configuration fingerprints;
- write `status: running` before expensive work and atomically publish
  `status: complete` only after the model and required metrics exist;
- preserve partial progress and mark interrupted or failed runs explicitly;
- make evaluation and reporting reject missing or invalid manifests by
  default. Historical artifacts may use an explicit, visible
  `--allow-legacy-artifact` escape hatch.

The intended implementation is the new `src/saferl/artifacts.py` surface
(split into a manifest module only when justified). It owns typed manifest
construction, required-field validation, canonical hashing, Git/source
provenance, and atomic writes. New runners should call it at run start and
completion. Do not let each runner invent another `run_config.json` shape.

## Canonical environment factory

The largest current reproducibility risk is not directory organization; it is
constructing subtly different environments in different scripts. Establish
one canonical factory for new code:

```python
env = make_laneless_env(config)
```

The planned home is `src/saferl/env_factory.py`. The factory must
be the only new-code boundary that:

- imports/registers `lane-free-v0`;
- validates and deep-copies the resolved environment configuration;
- constructs the base environment;
- applies the canonical reward and observation assembly in a documented order;
- applies explicitly requested instrumentation and seed handling; and
- exposes the resolved configuration needed by the run manifest.

Do not turn the factory into an opaque algorithm switchboard. Keep algorithm-
specific components composable and explicit:

```text
make_laneless_env(config)
  -> canonical environment/reward/observation assembly
  -> explicit CBF or projection adapter when requested
  -> explicit KPI/Monitor/vectorization adapter for the caller
```

The configuration should have named sections for `environment`, `reward`,
`observation`, `cbf` or `safety`, and `instrumentation`. It should be resolved
once, validated once, and passed by value/deep copy. Hidden defaults and
script-local mutations are prohibited. Legacy DDPG may use an explicit legacy
profile, but that profile must not become the default for new work.

### Factory implementation sequence

Implement the factory incrementally without refactoring the notebook:

1. Define the configuration/assembly contract and a canonical wrapper order.
2. Implement the factory as a behavior-preserving wrapper around the current
   `gym.make("lane-free-v0", ...)` construction.
3. Add parity tests against the current notebook factory for a fixed seed:
   configuration, action/observation spaces, reset observation, one fixed
   action step, timing plan, and selected diagnostics.
4. Migrate the smoke check and canonical evaluator first.
5. Migrate one PPO training path, then pilots and renderers.
6. Keep notebook factories and explicitly classified legacy paths unchanged
   until parity evidence exists.
7. After migration, make new direct `gym.make("lane-free-v0", ...)` calls fail
   repository policy checks; retain only the environment implementation,
   frozen notebook bridge, tests, and documented legacy exceptions.

The factory should not silently normalize, change CBF gains, change traffic,
or choose a different reward. Every behavior-changing option must appear in
the resolved config and manifest. A factory parity test is more important than
a broad refactor.

### Environment-factory acceptance criteria

A factory migration is complete only when:

- training, evaluation, rendering, and pilots can all construct the same
  canonical assembly through the factory;
- the same resolved config produces the same spaces and timing plan;
- fixed-seed reset/step parity is demonstrated within documented tolerances;
- wrapper order and observation feature order are tested;
- raw, safe, and executed action semantics remain distinct;
- the manifest records the factory/profile version and resolved config; and
- no old experiment is silently changed merely because a new factory exists.

## Legacy and exception policy

The following are retained for historical comparison, not as templates for
new work:

- legacy DDPG training and guided-CBF scripts;
- notebook-defined legacy DDPG sections;
- the current oversized protocol runners;
- old flat script names handled by `scripts/catalog.py`;
- historical DQN or lane-based helpers, which must not be reused by laneless
  code.

Current oversized runner exceptions are:

- `scripts/training/run_cbf_filter_ablation.py`;
- `scripts/training/run_nominal_ddpg_parameter_pilot.py`;
- `scripts/training/run_nominal_ppo_parameter_pilot.py`;
- `scripts/training/run_ppo_cbf_progression.py`.

An exception is not permission to add unrelated logic. A new exception
requires an explicit design note and a decomposition plan.

## Enforcement and verification

Run the repository policy check after structural changes:

```powershell
cd "safeRL_workspace"
python -m scripts.ops.check_repository_policy
```

The checker verifies that this file exists and contains the required policy,
that the documented source/layout files exist, that no unapproved training,
evaluation, or reporting script exceeds 2,000 lines, and that new DDPG-named
entry points are explicitly classified. Semantic rules—especially keeping
core algorithms out of notebooks—also require review against this file.

For normal code verification:

```powershell
cd "safeRL_workspace"
python -m pytest -q tests
```

Use the smallest relevant smoke or focused test first. Before a long run,
inspect `--help`, the selected model/variant, seed, traffic model, evaluation
budget, worker count, and explicit output directory. Never start a million-
transition ladder merely to inspect code.

When the notebook contract changes, update
`docs/lanelessKaralakou_reference.md` and `docs/script_reference.md`, run the
smoke and focused projection/observation tests, and verify that new manifests
record the notebook and reusable-module provenance.

## Command and documentation conventions

- Run package commands from `safeRL_workspace` using `python -m`.
- Use explicit flags for model paths, seeds, episode counts, worker counts,
  frequencies, and output directories.
- Quote Windows paths containing spaces.
- Keep generated models, logs, caches, virtual environments, backups, and
  unrelated worktree changes out of documentation or source changes.
- For environment/reward/observation/CBF/evaluation questions, read
  `docs/lanelessKaralakou_reference.md`.
- For script selection or module responsibilities, read the matching section
  of `docs/script_reference.md` and the relevant group README.
