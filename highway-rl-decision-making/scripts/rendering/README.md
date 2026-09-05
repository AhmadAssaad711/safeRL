# Rendering and scenarios

Rendering modules provide visual inspection of policies and safety filters.
They are useful for debugging and qualitative evidence, but their videos are
not substitutes for the strict canonical KPI evaluation.

- `render_laneless_karalakou.py`: notebook-backed single-policy rollout.
- `render_laneless_policy_scenario.py` and `render_laneless_policy_videos.py`:
  one or many named diagnostic scenes.
- `render_policy_scenarios.py`: shared scenario registry and plotting helpers.
- `render_ppo_500k_nominal.py`, `render_ppo_at1_raw.py`,
  `render_ppo_at1_cbf.py`, and `render_ppo_ego16_live.py`: PPO deployment views.
- `render_current_mtm.py`: current MTM environment inspection.

Use bounded step/episode counts and explicit output directories. See
[`docs/diagnostic_scenarios.md`](../../docs/diagnostic_scenarios.md) for the
scenario names and interpretation.
