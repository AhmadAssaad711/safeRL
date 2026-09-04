# Canonical notebook

This directory intentionally contains one research notebook:
[lanelessKaralakou.ipynb](lanelessKaralakou.ipynb).

It is the source of truth for the lane-free environment, Karalakou reward,
observation layout, CBF geometry, PPO/CBF progression, legacy DDPG variants,
evaluation metrics, and fixed diagnostic scenes. The external scripts load
selected notebook cells so training and evaluation use the same definitions.

## Notebook sections

| Section | Purpose |
| --- | --- |
| A | Imports, reward, environment configuration, observation/KPI wrappers, callbacks, and TensorBoard bridges |
| B | Seven-policy 1M PPO/CBF ladder and paired post-training evaluation |
| C | Legacy DDPG baseline, CBF shield, reward-feedback, and guided actor-loss reference |
| D | Saved PPO results, contract checks, interpretation, and read-only status |
| E | Optional legacy policy renders and six fixed diagnostic scenarios |
| Appendix | Opt-in historical safety-potential reward variant |

The notebook does not replace the scripts. Use the notebook to understand and
specify an experiment; use a script for long-running, resumable, or
out-of-process work.
