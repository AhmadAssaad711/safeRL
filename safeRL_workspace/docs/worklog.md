# Work log

A running record of changes and checks, newest entry first. Read it before
starting a session. Each entry says what changed, why, which files, and how it
was verified, with the measured numbers.

---

## 2026-09-11: HOCBF gain grid over (c1, c2), status running

- Why: the current k0=5.29, k1=3.68 have complex roots, so they do not form a valid HOCBF. The grid is parameterized by the cascade ψ1 = ḣ + c1·h and ψ2 = ψ̇1 + c2·ψ1, so k1 = c1+c2 and k0 = c1·c2. Every cell with c1, c2 > 0 is therefore a valid HOCBF. The QP row depends only on (k0, k1), which is symmetric in c1 and c2, so only c1 ≤ c2 is run. c1 is used as the reset ψ1 gain (a diagnostic only).
- Setup: scratchpad `eval_gain_grid.py`, c ∈ {0.5, 1, 1.5, 2.3, 3.5, 5, 8}, which gives 28 cells, plus a CBF-OFF cell and a cell with the current gains. Each cell runs 30 episodes on the same seeds (1100000–1100029), 40 vehicles, 100/20/20, with the same B.1 nominal policy (sha 7fc8d0f1). Spawning keeps the canonical ψ1 gain of 2.3, so initial states are identical in every cell. `cbf_require_initial_safe_set` is set to false so that cells where ψ1(c1) < 0 at reset still run. One pool of 20 workers serves all cells.
- Check: the current-gains cell reproduced the 200-episode v40 run exactly on seeds 1100000/1 (608 / 1377 steps).
- Launch: at 13:51 to `artifacts/1MRun/v40_gain_grid/`, 900 episodes. At the user's choice it runs **concurrently** with the other session's `nom_lin` 1M training, which was at 350k steps then, with 42 python processes. `nom_lin` throughput after 13:51 is therefore contended and should not be used as a speed baseline.
- Plan: rank the cells at 30 episodes, then confirm the best 2–3 at 200 episodes.

---

## 2026-09-11: 40 vehicles with equal physics/policy/CBF frequencies, and an audit of the CBF definition

### Equal frequencies (same policy, 40 vehicles, 200+200 episodes each)

- Harness: scratchpad `eval_density.py --hz N` sets `simulation_frequency = policy_frequency = cbf_frequency = N` and `dt = 1/N`. It rescales `task_max_policy_steps` to keep the same 150 s horizon (3000 steps at 20 Hz, 15000 at 100 Hz), and `cbf_substep_filtering` stays off. The frame cap `episode_steps=30000` counts physics frames, so it never binds first.
- Outputs are in `artifacts/1MRun/v40_hz20/` and `v40_hz100/`. Speed with 20 workers confirmed: 20 Hz ran 1469 policy steps/s (114 s total), 100 Hz ran 1384 policy steps/s (614 s total).
- Collisions/km, CBF OFF → ON: 100/20/20 gives 5.34 → 2.83; 20/20/20 gives 6.66 → 2.80; 100/100/100 gives 6.05 → 2.88. QP failure rate: 8.5%, 9.2%, 9.2%. Collisions on a QP-fail/fallback step: 78%, 82%, 77%. **CBF sampling rate is not the cause.** Jerk at 100 Hz (9.9) is not comparable, because jerk is Δa/dt over a 0.01 s step.

### CBF definition audit (notebook cells 44 and 50, `scripts/common/cbf_geometry.py`, `cbf_ray_mask.py`)

- The barrier h = dx²/A² + dy²/B² − 1, with A=5.09 and B=2.55. The collision box corner (3.5, 1.8) gives h = −0.03, so the ellipse just encloses the axis-aligned collision test. Contact gives h ≈ −0.5, which matches the KPI `Minimum h` ≈ −0.5. The dynamics are a double integrator (`Vehicle.step`), and the derivatives, the neighbor ax/ay terms, the ring-road dx, and the A·a ≤ b sign are all consistent. The 2D projection is exact.
- **Gains are not a valid HOCBF.** k1² = 13.5 < 4·k0 = 21.2, so the roots are −1.84 ± 1.38i (ζ = 0.8). The condition ḧ + k1ḣ + k0h ≥ 0 therefore cannot be factored into ψ1 = ḣ + α1h and ψ2 = ψ̇1 + α2ψ1 with real α > 0, and the forward-invariance guarantee is lost. `psi1_gain = 2.3` implies k1 = 4.6 (critical damping), so it is inconsistent with k1 = 3.68.
- The CBF has no awareness of the input bounds: nothing guarantees a feasible action with |a| ≤ 3.
- Replay: scratchpad `diag_cbf_failures.py`, 40 CBF-ON episodes, 40 vehicles, 100/20/20, 23.9k steps.
  - 7.9% of steps had an infeasible QP. Of the 191 onsets of infeasibility, 83% were joint conflicts (every row satisfiable on its own), and 17% had a single row impossible within the box. In 29 of those 32 single-row cases, the neighbor was BEHIND.
  - At onset, the worst row had h < 0 in 27% of cases and ψ1 < 0 in 63%.
  - Barrier crossings from h ≥ 0 to h < 0: 109 in total. 37% happened with the QP feasible, all tiny (median h after crossing −0.007). The invalid gains have a real but small effect.
  - Collisions: 37 total, **29 rear-end (78%)**, 5 side, 3 ahead. In the rear-end cases, the ego was already at ax = +3.0 (full throttle) at vx 12.5 m/s. The follower closed at 3.5 m/s and was NOT braking (median follower ax +0.37). About half of all collisions happened with the ego pinned at a road edge (y = 0.9 or 9.3). The ego's lateral speed was never at the |vy| = 0.3·vx clamp.
- Conclusion: the main failures come from the scenario, not the CBF parameters. The constraints hold the ego responsible for followers it cannot influence. Followers ignore the ego (`guard_ego_interactions=false`, and MTM discounts leaders that overlap laterally). The ego drives slower than traffic and hugs the road edge. Together these produce sandwich states for which no admissible action exists.

---

## 2026-09-11: linear reward mode, and a 1M nominal retrain with it

- Why: the wy=3.0 run showed the reciprocal term's lateral pull, ε·wy/denom², peaks at about wy≈1.8 and then falls. A linear tracking term gives each cost a constant pull.
- Change (uncommitted): new `src/saferl/rewards.py` with `r_track = Σ wᵢ(1−clip(cᵢ,0,1)) / Σ wᵢ`, which uses the same wx/wy/wf/way (0.377/0.245/0.377/0) and stays in [0,1]. Progress, jerk, collision, and overtake terms are unchanged. `run_ppo_cbf_progression --reward-mode {reciprocal,linear}` selects the mode in `_base_environment`, the single path used by training workers and evaluation. Omitting the flag gives the notebook reward, unchanged. The notebook is not edited.
- Found: `--reward-mode additive` used to be accepted but did nothing, because the reverted notebook wrapper never reads `reward_mode`. It is now rejected. These flags still have no effect: `--speed/lateral-reward-weight`, `--risk-penalty-weight`, `--safety-*`, the reward sigmas, and `--collision-reward-override`.
- Tests: new `tests/unit/test_linear_reward.py`, 7 passed, including a real notebook env check that the components sum to the reward. Full suite: 177 passed, 3 failed, none in touched code (the notebook launch switch set to `True`, the HOCBF-reward pairwise-geometry test, the simulator-benchmark dt check). `test_cbf_critical_c_sweep.py` fails at collection because it imports a missing name; the files involved are unchanged from HEAD. Repository policy: PASS.
- Smoke: 4 envs, 2k steps, in `%TEMP%\saferl_ss\linear_smoke`. The signature has `reward_mode: linear`. Non-collision reward per step was 0.50, against 0.185 early in the reciprocal run, which confirms the spawned workers use the linear wrapper.
- Launch: 13:32:09, cell 11's nominal flags plus `--reward-mode linear --tensorboard-run-label n1_lin`, collision still −2.5, to `artifacts/1MRun/nom_lin/`. It was held until another session's 20-worker density eval exited. Measured 20 workers under one learner, **412 t/s** over 60 s at 10k→36k steps, CPU 67%, uncontended.
- Next, if wanted: a linear-mode collision-penalty run at −10 to −20. At −50, the value targets span about −50 to +60, and the joint `max_grad_norm=0.5` clip would let critic gradients crowd out actor updates.

---

## 2026-09-11: CBF effect on B.1 nominal at 40 vehicles (instead of 55)

- Why: the user asked to test the CBF effect on a policy at `vehicles_count = 40` instead of 55.
- Policy: the original `ppo_nominal` (wy=0.65, sha 7fc8d0f1…, trained at 55 vehicles), from `artifacts/1MRun/nom_orig_0911_backup/`. At 40 vehicles this is an out-of-distribution, lower-density test.
- Harness: scratchpad `eval_density.py` calls the runner's own `_evaluate_complete_episode_rows` and `summarize_post_training_episodes`, using the run's saved env/reward/CBF config. The only change is `env_config["vehicles_count"]` = 40. Settings: 1000 m task, 3000-step cap, corr_eps 0.03, ttc_cap 30, k0/k1/psi1 = 5.29/3.68/2.3, doubled ellipse (the script asserts the notebook ellipse matches the run's). Seeds 1100000+, 200+200 episodes.
- Harness check: with 55 vehicles and 2+2 episodes, it reproduced the original `pe/e.csv` exactly (OFF 79/69 steps, ON 662/68 steps, same distances).
- Speed: 21 python processes (1 parent + 20 workers), CPU 73%. 400 episodes took 226 s, 809 policy steps/s.
- Output: `artifacts/1MRun/v40/ppo_nominal/seed_307/pe/{e,b,kpi}.csv` and `m.json`, which records the git commit, the dirty flag, and the vehicles 55→40 change.
- Result (Mean ± SD over 10 blocks of 20 episodes), 55v OFF / 55v ON → 40v OFF / 40v ON:
  - collisions/km: 10.32 / 4.83 → **5.34 / 2.83**. The CBF cuts the rate by 47% at 40 vehicles and 53% at 55.
  - QP failure rate with CBF ON: 11.4% → 8.5%. Intervention rate: 78% → 80%.
  - Completion: 0 / 0.5% → 2% / 6%. With CBF ON, 188 of 200 episodes still end in a collision.
  - Collisions on a QP-fail/fallback step, CBF ON: 79% → 78%. The failure mechanism does not change with density.
  - Speed error: at 55 vehicles the CBF worsens it (5.46 → 6.32); at 40 it improves it (4.88 → 4.46). Lateral error is about 4.1 m in every case.

---

## 2026-09-11: retraining nominal PPO with a stronger lateral weight (wy 0.65 → 3.0)

- Why: the nominal policy has a mean lateral tracking error of 4.2 m. With a road width of 10.2 m, that puts the lateral cost `wy·cy` (≈0.27) at about the same level as the speed cost (≈0.34). With wy=3.0 the lateral cost becomes the largest term (≈1.2).
- The ladder was stopped at the user's request. B.2 `ppo_cbf_reward` was killed at 370k/1M; its directory `artifacts/1MRun/rwd/` is partial.
- Launch: at 10:06, `run_ppo_cbf_progression` with cell 11's nominal flags plus `--lateral-y-weight 3.0 --tensorboard-run-label n1_wy3`, output to `artifacts/1MRun/nom_wy3/`. `training_signature.pending.json` confirms wy=3.0 and 20 envs. Speed running alone was 373 t/s.
- Incident: a notebook Run All from a new kernel started at 10:05:53. It force-retrained `ppo_nominal` (wy=0.65) into `artifacts/1MRun/nom/` and was stopped at 10:10 (PID 19832). The original `nom/` `training_episodes.csv` and monitor files were overwritten by about 80k steps of that partial run and are lost. The original model (sha 7fc8d0f1…), configs, checkpoints, and `pe/` evaluation were backed up to `artifacts/1MRun/nom_orig_0911_backup/`. They are also still in place in `nom/`, next to a stale `training_signature.pending.json`.
- Result: training took 73 min (it ran alongside the notebook run for the first 4 min) and the evaluation is complete, 200+200 episodes. Values are wy=0.65 → wy=3.0.
  - CBF OFF: lateral error 4.18 → **4.54 m (worse)**, speed error 5.46 → 4.38, collisions/km 10.3 → 9.6.
  - CBF ON: lateral error 4.33 → 4.33, collisions/km 4.83 → 5.01, QP failure rate 11.4% → 12.5%. 175 of 201 collisions happened on a step where the QP failed.
- Why wy did not help: the reward is reciprocal, r = ε/(ε + wx·cx + wy·cy + …). The pull toward the lateral target, |∂r/∂cy| = ε·wy/denom², rises with wy only until wy·cy ≈ ε + wx·cx + … (wy ≈ 1.8 here), then falls. At cy≈0.41 it is 0.26 for wy=0.65 and 0.31 for wy=3.0, only about 20% stronger. Meanwhile the base reward roughly halves relative to the collision penalty and the progress reward. Raising wy further would weaken the lateral pull.
- Render helper: scratchpad `render_ladder_raw.py` live-renders a ladder checkpoint through `make_evaluation_env`, and reproduced the evaluation's raw episodes exactly (seeds 1100000 and 1100001: 79 and 69 steps).

---

## 2026-09-11: CBF check on B.1 nominal, and do collisions come from QP failure?

Read-only analysis of existing output; nothing was launched while the ladder ran. The source is `artifacts/1MRun/nom/ppo_nominal/seed_307/pe/{kpi,e}.csv`, which used the doubled ellipse (5.09 × 2.55 m). There were 200 episodes per mode, on the same seeds.

- Collisions/km went from 10.32 ± 1.84 with CBF OFF to 4.83 ± 0.91 with CBF ON. The QP failure rate with CBF ON was 11.4% ± 1.9%, and the intervention rate was 78%. The minimum h was still −0.49.
- With CBF ON, 199 of 200 episodes ended in a collision. For each collision, the per-step records (`_collision_event_records`) show:
  - 158 (79%) happened on a step where the QP failed and the fallback ran. In 97% of these, the QP had also failed on the step before, so the infeasibility lasted several steps.
  - 41 (21%) happened on a step where the QP succeeded. In 40 of these, the QP had failed earlier in the episode, and in all 41, h was already below 0 before the step.
  - About 99% of collisions follow a QP failure, either on the same step or earlier. So the cause is QP infeasibility, not a feasible QP that picked an unsafe action.
- Anomaly: `cbf_hocbf_condition_satisfied` is True on every fallback step, even though `cbf_max_constraint_violation_safe` has a median of 1.8. The flag may not mean "the condition is satisfied" on fallback steps.

---

## 2026-09-11: baseline and 20-worker speed check (PPO track)

### Baseline state (nothing committed yet)

- HEAD is `bd4b048` ("experiment(laneless): add reproducible PPO and CBF evaluations").
- `notebooks/lanelessKaralakou.ipynb`: the cell sources equal `bd4b048` plus three edits.
  1. The B.0 launch switches are on (`PPO_1M_RUN_TRAINING = True`, `PPO_1M_FORCE_RETRAIN = True`), so Run All launches and retrains the full seven-policy 7M-transition ladder. The intro markdown and `docs/lanelessKaralakou_reference.md` ("Launch guard") were updated to say so.
  2. `ENV_CONFIG["cbf_geometry"]` uses the doubled (Minkowski-sum) relative ellipse, 2·3.6/√2 × 2·1.8/√2.
- `scripts/common/cbf_geometry.py`: uncommitted doubled ellipse, matching the notebook. `src/saferl/config.py` still uses the single-vehicle 3.6/√2 × 1.8/√2.
- Other uncommitted changes: `--ent-coef` and `--gamma` overrides in `run_ppo_cbf_progression.py`, and the untracked `configs/ppo_nominal_guard_ego_on.json`.
- The notebook CBF gains in force at runtime are the tuned no-slack override: k0=5.29, k1=3.68, psi1_gain=2.3, max 12 neighbour constraints.

### Worker path, traced statically

- Notebook cell 5 sets `TRAINING_WORKERS = 20` and `EVALUATION_WORKERS = 20`.
- Cell 11 (B.0) sets `PPO_1M_NUM_ENVS = TRAINING_WORKERS` and launches `scripts.training.run_ppo_cbf_progression` with `--n-envs 20 --n-steps 1000` (50 steps per env per rollout) and `--post-train-eval-workers 20`.
- `training_topology()` creates a `SubprocVecEnv` with `spawn` whenever n_envs > 1, and raises if `num_envs != n_envs`. Post-training evaluation uses `ProcessPoolExecutor(max_workers=20, spawn)`.
- OMP/MKL/OpenBLAS threads are set to 1 before the numpy and torch imports. The learner calls `th.set_num_threads(1)`.
- The script executes only the notebook setup cells, chosen by marker (3, 5, 6, 42, 44, 46, 48, 50), so cell 11's launch switch cannot trigger from a script.

### Runtime check

- Harness: scratchpad `speed_smoke.ps1`, which uses cell 11's exact flags with a shorter horizon. Output goes to `%TEMP%\saferl_ss\` because the scratchpad path exceeds Windows MAX_PATH.
- Machine: 32 logical CPUs and an RTX 4000 Ada. Python 3.9.13, torch 2.8.0+cu128, SB3 2.7.1.
- Machine detail: the i9-14900 has 8 P-cores and 16 E-cores. The lockstep `SubprocVecEnv` waits for the slowest worker, and at least 12 of the 20 workers must run on E-cores.

Results. Speed is transitions/s measured by SB3 `time/fps` or by the change in `global_timestep`.

| Run | Variant | Envs | Learner | Steps | Speed | Clean? |
|---|---|---|---|---|---|---|
| A | `ppo_nominal` | 20 | cuda | 20k | **297** (first rollout alone: 414) | yes |
| B | `ppo_nominal` | 20 | cpu | 20k | 274 (first rollout: 490) | partly overlapped the user's 1M run |
| C | `ppo_nominal` | 1 | cuda | 5k | 45 | no, contended |
| D | `ppo_cbf_reward` | 20 | cuda | 10k | 180 | no, contended (lower bound) |
| E | `ppo_cbf_projected_reward_off` | 20 | cuda | 10k | 88, and falling as updates dominate | no, contended (lower bound) |
| live | B.1 `ppo_nominal` 1M from the notebook | 20 | cuda | 139k to 163k | **401** (60 s sample, CPU load 70%) | yes |

- **20 workers confirmed live.** Smoke A had 21 python processes, as did its 20-worker evaluation pool (40 episodes in 62 s). The user's notebook 1M run had 23 python processes: 2 kernels, 1 learner, and 20 workers.
- Nominal speed rises from about 300 to about 400/s as episodes get longer, because there are fewer resets. At about 400/s, one 1M nominal policy trains in roughly 42 min.
- About 30% of wall time goes to the PPO update: 10 epochs × 10 minibatches = 100 gradient steps per 1000 transitions. The CUDA and CPU learners run at about the same speed.
- Scaling is about 10× over a single env, not 20×. CPU load of 70% points to lockstep waiting (the P/E core mix) plus idle workers during updates, not to a CPU ceiling.
- Incident: the user launched B.1 from the notebook at 08:16 while smokes B to E were running. Those smokes and the first ~100k steps of B.1 competed for the CPU. No results were overwritten, since `artifacts/1MRun/` held no earlier run.
- The TensorBoard events for `artifacts/1MRun` runs are not written inside the repo; they were not found in `%TEMP%` either. For live speed, use the `global_timestep` column of `training_episodes.csv`.
- Smoke outputs are in `%TEMP%\saferl_ss\` (5.8 MB, outside the repo, safe to delete).

Progress check at 08:28 (B.1 `ppo_nominal`, `artifacts/1MRun/nom`):
- Reached 212k/1M. Speed over the clean 100k to 200k interval was 360 t/s. Training ETA is about 09:05, followed by the 200+200 episode evaluation.
- All 2,000 training episodes so far ended in a collision. Mean length rose from 96 to 130 steps, distance from 73 to 93 m, and time to first collision from 5.1 to 6.5 s. Collisions/km fell from 23.9 to 18.6, and return rose from 22.6 to 31.1. The improvement starts at about 150k. Action saturation is 0.

Script for this check: scratchpad `progress.py <seed_dir>`, which reads `training_episodes.csv` and the checkpoint mtimes.

Open items:
- Measure clean CBF-variant speeds from the live ladder as it reaches B.2 and B.3 (no extra launches while it runs).
- Possible speedups, not applied: pin workers to P-cores or cut per-step lockstep waiting; reduce update cost for the differentiable-CBF variants.
