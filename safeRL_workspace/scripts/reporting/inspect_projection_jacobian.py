"""Directly measure the differentiable CBF projection's Jacobian, d mu_safe /
d mu_raw, on real states from a trained projected-mean policy (C2).

The projection (scripts/common/cbf_projection.py:project_polytope_2d_torch)
is documented to be, exactly: identity in the interior (raw action already
feasible), the tangent projector onto one active face (rank-1: one singular
value 1, one 0), or the zero map at an independent two-face vertex
(rank-0), and undefined (gradient detached, zero Jacobian) when no feasible
point exists at all. This script does not re-derive that analytically; it
evaluates the actual autograd graph batch-by-batch on states from a real
rollout and reports what regime the policy actually spends its time in,
via the Jacobian's singular values -- which is what "differentiability" is
mechanically contributing (or not) to the gradient the actor receives.

sigma_min close to 1  -> interior, projection is a no-op (identity Jacobian).
sigma_min close to 0, sigma_max close to 1 -> on one active face (rank-1).
both singular values close to 0 -> at a two-face vertex, or infeasible
    (detached fallback) -- the projection contributes no gradient at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th

from scripts.rendering.render_policy_scenarios import find_repo_root
from scripts.rendering.render_ppo_scenario_gifs import build_namespace, load_policy
from scripts.training.run_ppo_cbf_progression import load_model, make_evaluation_env


def collect_observations(
    namespace: dict[str, Any], policy: dict[str, Any], *, episodes: int, max_steps: int, seed: int
) -> np.ndarray:
    """Raw observation arrays actually fed to the network on a real rollout."""

    env = make_evaluation_env(
        namespace,
        mode="raw",
        env_config=policy["env_config"],
        reward_config=policy["reward_config"],
        correction_epsilon=policy["correction_epsilon"],
    )
    observations: list[np.ndarray] = []
    try:
        for episode in range(int(episodes)):
            obs, _info = env.reset(seed=int(seed) + episode)
            for _ in range(int(max_steps)):
                observations.append(np.asarray(obs, dtype=np.float32))
                action, _ = policy["model"].predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    break
    finally:
        env.close()
    return np.stack(observations, axis=0)


def jacobian_at(model, obs_row: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (jacobian 2x2, mu_raw, feasible) at one observation.

    mu_raw is read off the network's own forward pass at this state (not an
    arbitrary probe point), then only the projection step is re-run with
    mu_raw as an independent leaf, so the Jacobian is exactly d mu_safe /
    d mu_raw with the observation (and therefore the constraint system) held
    fixed -- not conflated with the actor network's own smoothness.
    """

    net_policy = model.policy
    obs_tensor = th.as_tensor(obs_row, dtype=th.float32, device=device).unsqueeze(0)
    with th.no_grad():
        _dist, _values, _safety, mu_raw0, _mu_box0, _mu_safe0, _proj0 = (
            net_policy._distribution_and_stages(obs_tensor)
        )
    mu_raw_leaf = mu_raw0.detach().clone().requires_grad_(True)

    def mu_safe_of(mu_raw: th.Tensor) -> th.Tensor:
        projection = net_policy.project_actions(obs_tensor, mu_raw)
        return th.where(projection.feasible.unsqueeze(1), projection.action, projection.action.detach())

    with th.no_grad():
        feasible = bool(net_policy.project_actions(obs_tensor, mu_raw_leaf).feasible.item())

    if not feasible:
        # mu_safe = projection.action.detach() entirely in this branch (no
        # feasible point exists for the no-slack QP), so there is no graph to
        # backprop through -- the Jacobian is exactly zero by construction,
        # not merely small.
        jac = np.zeros((2, 2), dtype=np.float64)
        return jac, mu_raw0.detach().cpu().numpy().reshape(-1), feasible

    rows = []
    for out_index in range(2):
        grad = th.autograd.grad(
            mu_safe_of(mu_raw_leaf)[0, out_index], mu_raw_leaf, retain_graph=(out_index == 0)
        )[0]
        rows.append(grad.detach().cpu().numpy().reshape(-1))
    jac = np.stack(rows, axis=0)
    return jac, mu_raw0.detach().cpu().numpy().reshape(-1), feasible


def summarize(label: str, jacobians: np.ndarray, feasible_flags: np.ndarray, correction_norms: np.ndarray) -> dict[str, Any]:
    singular_values = np.linalg.svd(jacobians, compute_uv=False)
    sigma_max = singular_values[:, 0]
    sigma_min = singular_values[:, 1]
    n = len(jacobians)
    interior = sigma_min > 0.9
    one_face = (sigma_min <= 0.9) & (sigma_max > 0.1)
    vertex_or_infeasible = sigma_max <= 0.1
    print(f"\n=== {label} (n={n} states) ===")
    print(f"  infeasible (detached, zero Jacobian by construction): {np.mean(~feasible_flags):.1%}")
    print(f"  interior / identity  (sigma_min > 0.9):        {np.mean(interior):.1%}")
    print(f"  one active face      (rank-1, 0.1<sigma_min<=0.9 or sigma_max>0.1): {np.mean(one_face):.1%}")
    print(f"  vertex / near-zero Jacobian (sigma_max <= 0.1): {np.mean(vertex_or_infeasible):.1%}")
    print(f"  mean sigma_max={sigma_max.mean():.3f}  mean sigma_min={sigma_min.mean():.3f}")
    print(f"  mean correction norm ||mu_safe-mu_raw||: {correction_norms.mean():.4f}")
    return {
        "label": label,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "feasible": feasible_flags,
        "correction_norms": correction_norms,
    }


def plot_summary(results: list[dict[str, Any]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = ["#997300", "#B8860B", "#4472C4"][: len(results)]

    ax = axes[0]
    for res, color in zip(results, colors):
        ax.hist(res["sigma_min"], bins=30, range=(0, 1.05), alpha=0.55, label=res["label"], color=color)
    ax.set_title("Smaller singular value of d(mu_safe)/d(mu_raw)")
    ax.set_xlabel("sigma_min (0=vertex/no gradient, 1=identity)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)

    ax = axes[1]
    for res, color in zip(results, colors):
        ax.scatter(res["correction_norms"], res["sigma_min"], s=8, alpha=0.35, label=res["label"], color=color)
    ax.set_title("Projection Jacobian vs. correction magnitude")
    ax.set_xlabel(r"$||\mu_{safe}-\mu_{raw}||$")
    ax.set_ylabel("sigma_min")
    ax.legend(fontsize=8)

    ax = axes[2]
    labels = [res["label"] for res in results]
    regimes = ["infeasible", "vertex", "one face", "interior"]
    bottoms = np.zeros(len(results))
    regime_colors = ["#7f1d1d", "#dc2626", "#f59e0b", "#16a34a"]
    for regime, rcolor in zip(regimes, regime_colors):
        fracs = []
        for res in results:
            feasible = res["feasible"]
            sigma_min = res["sigma_min"]
            sigma_max = res["sigma_max"]
            if regime == "infeasible":
                mask = ~feasible
            elif regime == "vertex":
                mask = feasible & (sigma_max <= 0.1)
            elif regime == "one face":
                mask = feasible & (sigma_max > 0.1) & (sigma_min <= 0.9)
            else:
                mask = feasible & (sigma_min > 0.9)
            fracs.append(float(np.mean(mask)))
        ax.bar(labels, fracs, bottom=bottoms, label=regime, color=rcolor)
        bottoms += np.asarray(fracs)
    ax.set_title("Regime breakdown")
    ax.set_ylabel("fraction of states")
    ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(axis="x", rotation=15)

    fig.suptitle(
        "Differentiable projection Jacobian on real rollout states (C2)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"\nwrote {out_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", required=True, metavar="LABEL=RUN_CONFIG_JSON")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1100000)
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "reports" / "safety_matrix_1M" / "figures")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = find_repo_root()
    results = []
    for spec in args.policy:
        label, config_path_str = spec.split("=", 1)
        policy = load_policy(Path(config_path_str).resolve())
        if not bool(policy.get("cbf", {})) or "ppo_cbf_projected" not in policy["variant"] and "diff_reward_only" not in policy["variant"]:
            print(f"  NOTE: {label} variant={policy['variant']} may not carry a differentiable projection")
        namespace = build_namespace(project_root, policy["cbf"])
        policy["model"] = load_model(policy["variant"], policy["model_path"], "cpu")
        print(f"[jacobian] {label}: variant={policy['variant']} seed={policy['training_seed']}")

        observations = collect_observations(
            namespace, policy, episodes=int(args.episodes), max_steps=int(args.max_steps), seed=int(args.seed)
        )
        print(f"  collected {len(observations)} observations from real rollout")

        jacobians = np.zeros((len(observations), 2, 2), dtype=np.float64)
        feasible_flags = np.zeros(len(observations), dtype=bool)
        correction_norms = np.zeros(len(observations), dtype=np.float64)
        for i, obs_row in enumerate(observations):
            jac, mu_raw, feasible = jacobian_at(policy["model"], obs_row, "cpu")
            jacobians[i] = jac
            feasible_flags[i] = feasible
            with th.no_grad():
                obs_tensor = th.as_tensor(obs_row, dtype=th.float32).unsqueeze(0)
                mu_raw_t = th.as_tensor(mu_raw, dtype=th.float32).unsqueeze(0)
                projection = policy["model"].policy.project_actions(obs_tensor, mu_raw_t)
                mu_safe = np.where(feasible, projection.action.numpy(), projection.action.numpy())
            correction_norms[i] = float(np.linalg.norm(mu_safe.reshape(-1) - mu_raw))

        results.append(summarize(label, jacobians, feasible_flags, correction_norms))

    plot_summary(results, args.out / "projection_jacobian_inspection.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
