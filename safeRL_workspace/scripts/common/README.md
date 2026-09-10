# Common components

These modules are reusable building blocks rather than long-running experiment
entry points. They reconstruct the notebook contract and should be imported by
training/evaluation workflows instead of copied.

- `laneless_script_config.py`: shared CLI arguments, traffic/environment
  configuration normalization, and the canonical 20-worker default used by
  PPO rollout and laneless evaluation entry points. Legacy DDPG controls remain
  explicit in their historical runners.
- `laneless_training_registry.py` and `laneless_evaluation_registry.py`:
  provenance, run identity, manifest, and atomic-output helpers.
- `cbf_geometry.py`, `cbf_projection.py`, and `cbf_ray_mask.py`: batched
  fixed-relative-ellipse HOCBF geometry, physical-action safety constraints,
  projection, and ray-mask filtering.
- `guided_cbf_minimal.py`: detached actor guidance and diagnostic gradients.
- `ppo_cbf_env.py`, `ppo_observation_variants.py`, and
  `ppo_parallel_worker.py`: PPO context, observation, and worker adapters.
- `projected_ppo_cbf.py`: differentiable and detached projected PPO policies.

These files normally do not create experiment results by themselves. Start a
workflow from [`../training/`](../training/README.md) or
[`../evaluation/`](../evaluation/README.md).
