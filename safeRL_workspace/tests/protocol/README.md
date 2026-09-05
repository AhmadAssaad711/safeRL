# Protocol tests

Protocol tests verify the experiment-facing workflows that connect notebook
definitions, training/evaluation configuration, output manifests, and reports.
They cover PPO progression/formulation screens, nominal PPO/DDPG pilots, CBF
filter/counterfactual pipelines, and report construction.

They use temporary directories and mocked or minimal episodes. A protocol test
must never launch a full training budget as part of normal collection.
