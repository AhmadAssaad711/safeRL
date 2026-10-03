"""How much of PPO's desired actor update does the differentiable CBF projection remove?

For one active linear constraint the projection Jacobian is the tangent
projector ``J = I - a a^T / ||a||^2``. The component of the PPO gradient that
the projection suppresses is ``(I - J) g``, so the scale-free quantity is

    r_perp = || (I - J) g_ppo || / (|| g_ppo || + eps)

which for a single active constraint equals ``|cos phi|``, phi being the angle
between the PPO gradient and the constraint normal. It answers a different
question from ``cos(g_ppo, g_cbf)`` (already logged as ``g_ppo_g_cbf_cosine``):
that one asks whether two *objectives* agree, this one asks how much of PPO's
update is *geometrically* removed, and it does not depend on how the auxiliary
CBF loss was designed.

Post-hoc: this runs on checkpoints that already exist. Nothing is retrained.

Three things this implementation is careful about.

**1. Where the tap goes.** ``J`` is a symmetric orthogonal projector, so
``g_in = J^T g_out = J g_out`` and ``r_perp = ||g_out - g_in|| / ||g_out||``
needs no explicit Jacobian -- but *only* where the loss reaches the tapped
tensor exclusively through the projection. In this policy the chain is

    mu_raw --box clip--> mu_box --P(.)--> mu_safe --> N(mu_safe, sigma) --> L

(``_distribution_and_stages``). The box clip's Jacobian is a diagonal 0/1 mask,
and ``J_P J_clip`` is **not** symmetric, so the identity is invalid at
``mu_raw``. The tap is therefore at **mu_box**, the projection's own input,
where it is exact. Clip suppression is a separate channel and is reported
separately as ``r_clip`` rather than folded in.

**2. Infeasible is not rank-0.** ``mu_safe = where(feasible, safe, proj.detach())``,
so on an infeasible sample the gradient to ``mu_box`` is zero *by construction*
and ``r_perp = 1`` mechanically. That is "the projection passed no gradient at
all", which is a different object from a rank-0 vertex where two independent
constraints pin the action. They are reported as separate regimes and never
merged; whether their rates coincide is left as a measurement.

**3. PPO surrogate only.** The auxiliary CBF actor loss (``lambda_mean``,
``lambda_sample``) has its own direct dependence on the projection stages and
would contaminate the decomposition. ``ProjectedCBFPPO.train`` already keeps
``actor_primary_loss`` and ``actor_auxiliary_loss`` separate; only the primary
surrogate is differentiated here.

The surrogate is evaluated at ratio = 1 (the first PPO epoch on a fresh
rollout), where the clipped objective is inactive, so ``dL/dmu`` is the pure
advantage-weighted score-function gradient rather than an epoch-dependent
clipped one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch as th

from scripts.rendering.render_policy_scenarios import find_repo_root
from scripts.rendering.render_ppo_scenario_gifs import build_namespace, load_policy
from scripts.training.run_ppo_cbf_progression import load_model, make_evaluation_env

EPS = 1e-12
SINGULAR_TOL = 1e-6


def collect_rollout(
    namespace: dict[str, Any],
    policy: dict[str, Any],
    *,
    episodes: int,
    max_steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Stochastic rollout, recording what GAE and the surrogate need."""

    env = make_evaluation_env(
        namespace,
        mode="raw",
        env_config=policy["env_config"],
        reward_config=policy["reward_config"],
        correction_epsilon=policy["correction_epsilon"],
    )
    model = policy["model"]
    obs_list, act_list, rew_list, done_list = [], [], [], []
    hmin_list, interv_list = [], []
    try:
        for episode in range(int(episodes)):
            obs, _info = env.reset(seed=int(seed) + episode)
            for _ in range(int(max_steps)):
                action, _ = model.predict(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = env.step(action)
                obs_list.append(np.asarray(obs, dtype=np.float32))
                act_list.append(np.asarray(action, dtype=np.float32).reshape(-1))
                rew_list.append(float(reward))
                done_list.append(bool(terminated))
                hmin_list.append(float(info.get("kpi_h_min", info.get("h_min", np.nan))))
                interv_list.append(
                    float(info.get("cbf_intervention", info.get("intervention", np.nan)))
                )
                obs = next_obs
                if terminated or truncated:
                    break
    finally:
        env.close()
    return {
        "obs": np.stack(obs_list, axis=0),
        "actions": np.stack(act_list, axis=0),
        "rewards": np.asarray(rew_list, dtype=np.float64),
        "dones": np.asarray(done_list, dtype=bool),
        "h_min": np.asarray(hmin_list, dtype=np.float64),
        "intervention": np.asarray(interv_list, dtype=np.float64),
    }


def compute_advantages(
    model, obs: np.ndarray, rewards: np.ndarray, dones: np.ndarray
) -> np.ndarray:
    """GAE(gamma, lambda) using the policy's own value head."""

    device = model.device
    with th.no_grad():
        values = (
            model.policy.predict_values(th.as_tensor(obs, dtype=th.float32, device=device))
            .cpu()
            .numpy()
            .reshape(-1)
        )
    gamma = float(model.gamma)
    lam = float(model.gae_lambda)
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float64)
    last = 0.0
    for t in reversed(range(n)):
        non_terminal = 0.0 if dones[t] else 1.0
        next_value = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last = delta + gamma * lam * non_terminal * last
        advantages[t] = last
    return advantages


def suppression_batch(
    model, obs: np.ndarray, actions: np.ndarray, advantages: np.ndarray
) -> pd.DataFrame:
    """Per-sample r_perp, clip suppression, regime, and correction magnitude."""

    device = model.device
    net = model.policy
    obs_t = th.as_tensor(obs, dtype=th.float32, device=device)
    act_t = th.as_tensor(actions, dtype=th.float32, device=device)
    adv_t = th.as_tensor(advantages, dtype=th.float32, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    evaluation = net.evaluate_actions_with_projection(obs_t, act_t)
    mu_raw, mu_box, mu_safe = evaluation.mu_raw, evaluation.mu_box, evaluation.mu_safe
    log_prob = evaluation.log_prob

    # ratio == 1 in value, but carries d log_prob / d mu: the first-epoch,
    # un-clipped PPO surrogate.
    ratio = th.exp(log_prob - log_prob.detach())
    policy_loss = -(adv_t * ratio).mean()

    g_box, g_safe = th.autograd.grad(
        policy_loss, [mu_box, mu_safe], retain_graph=True, allow_unused=True
    )
    if g_box is None:
        g_box = th.zeros_like(mu_box)
    if g_safe is None:
        g_safe = th.zeros_like(mu_safe)

    # Clip channel, reported separately and never folded into r_perp.
    g_raw = th.autograd.grad(policy_loss, mu_raw, retain_graph=True, allow_unused=True)[0]
    if g_raw is None:
        g_raw = th.zeros_like(mu_raw)

    # Explicit 2x2 Jacobian d mu_safe / d mu_box, batched: one vjp per output
    # component gives that row for every sample at once.
    jac_rows = []
    for k in range(2):
        row = th.autograd.grad(
            mu_safe[:, k].sum(), mu_box, retain_graph=(k == 0), allow_unused=True
        )[0]
        jac_rows.append(th.zeros_like(mu_box) if row is None else row)
    jac = th.stack(jac_rows, dim=1).detach().cpu().numpy()  # [N, 2, 2]

    g_safe_np = g_safe.detach().cpu().numpy()
    g_box_np = g_box.detach().cpu().numpy()
    g_raw_np = g_raw.detach().cpu().numpy()

    norm_safe = np.linalg.norm(g_safe_np, axis=1)
    r_perp = np.linalg.norm(g_safe_np - g_box_np, axis=1) / (norm_safe + EPS)
    r_clip = np.linalg.norm(g_box_np - g_raw_np, axis=1) / (
        np.linalg.norm(g_box_np, axis=1) + EPS
    )

    feasible = evaluation.projection.feasible.detach().cpu().numpy().reshape(-1).astype(bool)
    singular = np.linalg.svd(jac, compute_uv=False)  # [N, 2]
    rank = (singular > SINGULAR_TOL).sum(axis=1)

    regime = np.where(
        ~feasible,
        "detached_fallback",
        np.where(rank == 2, "interior", np.where(rank == 1, "rank1_face", "rank0_vertex")),
    )

    correction_vec = (mu_box - mu_safe).detach().cpu().numpy()
    correction = np.linalg.norm(correction_vec, axis=1)

    # --- tangent SPACE (what J does) versus tangent CONE (what the constrained
    # --- step should do).
    #
    # J g removes the whole normal component of g. The solution of
    #     max_d  g^T d - ||d||^2 / 2eta   s.t.  n^T d <= 0
    # is the projection onto the tangent *cone*,
    #     g* = g - max(0, n^T g) / ||n||^2 * n,
    # which removes that component only when it points out of the safe set. So
    #     g* - J g = min(0, n^T g) / ||n||^2 * n,
    # i.e. the two agree iff n^T g >= 0, and where n^T g < 0 the projection
    # Jacobian has destroyed a descent direction that was already feasible.
    #
    # Projection onto a halfspace moves along its outward normal, so for a
    # single active face the correction direction recovers n with its
    # orientation: n_hat = (mu_box - mu_safe) / ||mu_box - mu_safe||. No
    # constraint-row bookkeeping needed, and it is exactly the quantity already
    # logged as correction_norm.
    with np.errstate(invalid="ignore", divide="ignore"):
        n_hat = correction_vec / correction[:, None]
    n_hat = np.nan_to_num(n_hat)
    n_dot_g = np.einsum("ij,ij->i", n_hat, g_safe_np)

    # g* = g - max(0, n^T g) n_hat  (n_hat is already unit)
    g_star = g_safe_np - np.maximum(0.0, n_dot_g)[:, None] * n_hat
    r_star = np.linalg.norm(g_safe_np - g_star, axis=1) / (norm_safe + EPS)
    # How far the projection Jacobian sits from the constrained-optimal step.
    jacobian_vs_optimal = np.linalg.norm(g_box_np - g_star, axis=1) / (norm_safe + EPS)

    return pd.DataFrame(
        {
            "r_perp": r_perp,
            "r_star": r_star,
            "jacobian_vs_optimal": jacobian_vs_optimal,
            "n_dot_g": n_dot_g,
            "kills_feasible_direction": (n_dot_g < 0) & (correction > 1e-9),
            "r_clip": r_clip,
            "regime": regime,
            "feasible": feasible,
            "jac_rank": rank,
            "sigma_max": singular[:, 0],
            "sigma_min": singular[:, 1],
            "correction_norm": correction,
            "g_safe_norm": norm_safe,
            "advantage": advantages,
        }
    )


def build_reference_states(
    namespace: dict[str, Any],
    policy: dict[str, Any],
    *,
    checkpoint_paths: list[Path],
    variant: str,
    device: str,
    episodes: int,
    max_steps: int,
    seed: int,
    limit: int,
) -> np.ndarray:
    """One fixed state set D_ref, pooled over several checkpoints.

    Every checkpoint is later evaluated on exactly these states, so a change in
    r_perp cannot be produced by the policy merely visiting easier states. The
    pool spans early/mid/late policies deliberately: a D_ref drawn from the final
    policy alone would under-represent exactly the constrained regions the
    question is about.

    The CBF context travels inside the observation, so the constraint system at
    a stored state is the one that state really had -- replaying it under a
    different checkpoint is sound.
    """

    pooled: list[np.ndarray] = []
    for path in checkpoint_paths:
        policy["model"] = load_model(variant, Path(path), device)
        data = collect_rollout(
            namespace, policy, episodes=episodes, max_steps=max_steps, seed=seed
        )
        pooled.append(data["obs"])
        print(f"[r_perp] D_ref += {len(data['obs'])} states from {Path(path).name}",
              flush=True)
    states = np.concatenate(pooled, axis=0)
    if len(states) > limit:
        rng = np.random.default_rng(0)
        states = states[rng.choice(len(states), size=limit, replace=False)]
    return states


def suppression_on_states(model, obs: np.ndarray) -> pd.DataFrame:
    """Same decomposition on a fixed state set, with no advantage estimate.

    ``g_i = -(A_i/N) * (z_i - mu_i) / sigma^2``, and r_perp is invariant to both
    the scale and the sign of g_i, so the per-sample value depends only on the
    direction of ``z_i - mu_i`` relative to J. The advantage therefore cancels,
    which is what makes this measurable off-policy at all: no importance
    weighting, no value function from the wrong policy. Actions are sampled from
    the evaluated checkpoint's own distribution at each reference state.
    """

    device = model.device
    obs_t = th.as_tensor(obs, dtype=th.float32, device=device)
    with th.no_grad():
        actions = model.policy.get_distribution(obs_t).get_actions(deterministic=False)
    advantages = np.ones(len(obs), dtype=np.float64)
    return suppression_batch(model, obs, actions.cpu().numpy(), advantages)


def summarize(frame: pd.DataFrame, *, grad_floor: float = 1e-3) -> dict[str, Any]:
    """Distributional summary over samples where r_perp is actually defined.

    r_perp is a ratio of gradient norms, so it carries no information where
    ``||g_ppo|| -> 0`` (a sample whose normalised advantage is ~0 asks for no
    update, and the epsilon in the denominator would silently score it 0). Those
    samples are excluded and their share is reported, rather than averaged in.
    """

    total = int(len(frame))
    scale = float(np.median(frame["g_safe_norm"].to_numpy())) or 1.0
    defined = frame[frame["g_safe_norm"] > grad_floor * scale]
    frame = defined
    r = frame["r_perp"].to_numpy()
    if len(r) == 0:
        return {"samples": 0, "samples_total": total, "P(undefined)": 1.0}
    out = {
        "samples": int(len(r)),
        "samples_total": total,
        "P(undefined)": float(1.0 - len(r) / total),
        "E[r_perp]": float(np.mean(r)),
        "E[r_perp^2]": float(np.mean(r**2)),
        "P(r_perp>0.25)": float(np.mean(r > 0.25)),
        "P(r_perp>0.5)": float(np.mean(r > 0.5)),
        "P(r_perp>0.9)": float(np.mean(r > 0.9)),
        "E[r_clip]": float(np.mean(frame["r_clip"].to_numpy())),
        "mean_correction_norm": float(np.mean(frame["correction_norm"].to_numpy())),
    }
    for regime, group in frame.groupby("regime"):
        out[f"P({regime})"] = float(len(group) / len(frame))
        out[f"E[r_perp|{regime}]"] = float(group["r_perp"].mean())

    # Tangent-space (J) versus tangent-cone (constrained-optimal) step, on the
    # rank-1 samples where a single active face makes the comparison exact.
    r1 = frame[frame["regime"] == "rank1_face"]
    if len(r1):
        out["E[r_perp|rank1]"] = float(r1["r_perp"].mean())
        out["E[r_star|rank1]"] = float(r1["r_star"].mean())
        out["P(J kills feasible dir|rank1)"] = float(r1["kills_feasible_direction"].mean())
        out["E[||Jg-g*||/||g|| |rank1]"] = float(r1["jacobian_vs_optimal"].mean())
        killed = r1[r1["kills_feasible_direction"]]
        if len(killed):
            out["E[r_perp|rank1,killed]"] = float(killed["r_perp"].mean())
            out["E[r_star|rank1,killed]"] = float(killed["r_star"].mean())
    finite = frame[frame["g_safe_norm"] > 1e-9]
    if len(finite) > 2:
        out["corr(r_perp, correction_norm)"] = float(
            np.corrcoef(finite["r_perp"], finite["correction_norm"])[0, 1]
        )
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", action="append", required=True,
                        metavar="LABEL=RUN_CONFIG_JSON")
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=None,
                        help="Evaluate these training steps instead of the final "
                             "model, giving the r_perp-vs-budget curve.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1100000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = find_repo_root()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, per_sample = [], []
    for spec in args.policy:
        label, _, config_path = spec.partition("=")
        policy = load_policy(Path(config_path).resolve())
        variant = str(policy["variant"])
        if "projected" not in variant and "diff_reward_only" not in variant:
            print(f"[r_perp] SKIP {label}: variant={variant} carries no differentiable "
                  f"projection, so r_perp is N/A (not 0) for it.", flush=True)
            rows.append({"label": label, "variant": variant, "checkpoint": "n/a",
                         "r_perp_status": "N/A: no differentiable projection"})
            continue

        namespace = build_namespace(project_root, policy["cbf"])
        model_path = Path(policy["model_path"])
        targets = [("final", model_path)]
        if args.checkpoint_steps:
            root = model_path.parent / "checkpoints"
            targets = [(f"step_{s}", root / f"rollout_{s}_steps.zip")
                       for s in args.checkpoint_steps]

        for ckpt_label, path in targets:
            if not Path(path).is_file():
                print(f"[r_perp] {label} {ckpt_label}: missing {path}", flush=True)
                continue
            policy["model"] = load_model(variant, Path(path), args.device)
            jac_alpha = float(getattr(policy["model"].policy, "jacobian_alpha", 1.0))
            print(f"[r_perp] {label} {ckpt_label} jacobian_alpha={jac_alpha}", flush=True)

            data = collect_rollout(namespace, policy, episodes=args.episodes,
                                   max_steps=args.max_steps, seed=args.seed)
            advantages = compute_advantages(
                policy["model"], data["obs"], data["rewards"], data["dones"]
            )
            frame = suppression_batch(
                policy["model"], data["obs"], data["actions"], advantages
            )
            frame["label"] = label
            frame["checkpoint"] = ckpt_label
            frame["h_min"] = data["h_min"]
            frame["intervention"] = data["intervention"]
            per_sample.append(frame)

            summary = summarize(frame)
            summary.update({"label": label, "variant": variant,
                            "checkpoint": ckpt_label, "jacobian_alpha": jac_alpha})
            rows.append(summary)
            print("    " + json.dumps(
                {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in summary.items()}), flush=True)

    if per_sample:
        pd.concat(per_sample, ignore_index=True).to_csv(
            args.out / "r_perp_per_sample.csv", index=False
        )
    pd.DataFrame(rows).to_csv(args.out / "r_perp_summary.csv", index=False)
    print(f"[r_perp] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
