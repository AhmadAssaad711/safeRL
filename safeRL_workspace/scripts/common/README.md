# Common components

These modules are reusable building blocks rather than long-running experiment
entry points. They reconstruct the notebook contract and should be imported by
training/evaluation workflows instead of copied.

- `laneless_script_config.py`: shared CLI arguments and traffic/environment
  configuration normalization.
- `laneless_training_registry.py` and `laneless_evaluation_registry.py`:
  provenance, run identity, manifest, and atomic-output helpers.
- `cbf_geometry.py`, `cbf_projection.py`, and `cbf_ray_mask.py`: batched
  finite-difference HOCBF geometry, physical-action safety constraints,
  projection, and ray-mask filtering.
- `guided_cbf_minimal.py`: detached actor guidance and diagnostic gradients.
- `ppo_cbf_env.py`, `ppo_observation_variants.py`, and
  `ppo_parallel_worker.py`: PPO context, observation, and worker adapters.
- `projected_ppo_cbf.py`: differentiable and detached projected PPO policies.

These files normally do not create experiment results by themselves. Start a
workflow from [`../training/`](../training/README.md) or
[`../evaluation/`](../evaluation/README.md).
