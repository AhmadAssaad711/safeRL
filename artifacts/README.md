# safeRL artifacts

This directory is the single home for retained experiment results. The
directories were moved here from the repository root without changing their
contents or names.

## Contents

- ppo_clean_* contains the clean PPO progression manifests.
- ppo_du* and ppo_m10_* contain CBF or projection parameter studies.
- ppo_mtm_* contains native MTM density and baseline audits.
- ppo_nominal_* contains nominal PPO smoke tests and observation/collision
  audits.
- ppo_reward_* contains reward, reset-potential, TTC, and CBF-reward studies.

The retained files are compact JSON manifests, summaries, and selected
TensorBoard records. Full checkpoints, videos, plots, logs, and newly generated
run directories remain ignored by the repository policy.

Source code stays under
[highway-rl-decision-making/scripts](../safeRL_workspace/scripts/).
The canonical experiment contract is documented in
[lanelessKaralakou_reference.md](../safeRL_workspace/docs/lanelessKaralakou_reference.md).
The script and function map is in
[script_reference.md](../safeRL_workspace/docs/script_reference.md).

New generated files under artifacts are ignored by default. Add a new result
only when it is a deliberate compact artifact with its configuration and
provenance manifest.
