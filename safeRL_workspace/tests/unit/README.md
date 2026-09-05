# Unit tests

Unit tests exercise small, deterministic contracts in the lane-free environment
and reusable safety/PPO components:

- lane-free traffic, personality, and collision aggregation;
- CBF projection, ray/context behavior, and guided-gradient diagnostics;
- projected PPO component behavior;
- critical-C sweep helper calculations;
- the organized script catalog and its legacy-name compatibility layer.

These tests should remain fast and should not require saved models or a live
TensorBoard run.
