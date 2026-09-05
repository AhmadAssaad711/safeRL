from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import DDPG

from scripts.common.laneless_script_config import active_traffic_model, env_config_from_args
from scripts.rendering.render_policy_scenarios import POLICIES, ScenarioSpec, apply_scenario, load_notebook_namespace, make_scenarios


VARIANT_TO_POLICY_KEY = {
    "ddpg": "baseline",
    "ddpg-cbf": "ddpg_cbf_reward",
    "guided-ddpg-cbf": "guided_ddpg_cbf",
}


def set_stable_native_defaults() -> None:
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    ]:
        os.environ.setdefault(key, "1")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")


def normalize_artifact_suffix(value: str | None) -> str | None:
    if value is None:
        return None
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    suffix = suffix.strip("._-")
    if len(suffix) > 16:
        digest = hashlib.sha1(suffix.encode("utf-8")).hexdigest()[:8]
        suffix = f"{suffix[:7].strip('._-')}_{digest}"
    return suffix or None


def with_stem_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def apply_cbf_overrides(namespace: dict[str, Any], args: argparse.Namespace) -> None:
    namespace["DEVICE"] = args.device
    if args.lambda_filter is not None:
        namespace["CBF_FILTER_REWARD_LAMBDA"] = float(args.lambda_filter)
    if args.k0 is not None:
        namespace["CBF_K0"] = float(args.k0)
    if args.k1 is not None:
        namespace["CBF_K1"] = float(args.k1)
    if args.eps_side is not None:
        namespace["CBF_EPS_SIDE"] = float(args.eps_side)


def apply_env_and_artifact_overrides(namespace: dict[str, Any], args: argparse.Namespace) -> None:
    namespace["ENV_CONFIG"] = env_config_from_args(args, namespace["ENV_CONFIG"])
    traffic_model = active_traffic_model(namespace["ENV_CONFIG"])
    suffix = normalize_artifact_suffix(args.artifact_suffix)
    if suffix is None and traffic_model != "force":
        suffix = f"traffic_{traffic_model}"
    if suffix is None:
        return
    for key in ["DDPG_MODEL_PATH", "DDPG_CBF_MODEL_PATH", "GUIDED_DDPG_CBF_MODEL_PATH"]:
        namespace[key] = with_stem_suffix(Path(namespace[key]), suffix)


def find_project_root(start: Path | None) -> Path:
    if start is not None:
        for candidate in [start.resolve(), *start.resolve().parents]:
            if (candidate / "notebooks" / "lanelessKaralakou.ipynb").exists():
                return candidate
            nested = candidate / "safeRL_workspace"
            if (nested / "notebooks" / "lanelessKaralakou.ipynb").exists():
                return nested
    script_path = Path(__file__).resolve()
    for candidate in [script_path.parent, *script_path.parents]:
        if (candidate / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return candidate
    raise RuntimeError("Could not find project root containing notebooks/lanelessKaralakou.ipynb")


def policy_for_variant(variant: str):
    policy_key = VARIANT_TO_POLICY_KEY[variant]
    for policy in POLICIES:
        if policy.key == policy_key:
            return policy
    raise RuntimeError(f"Missing policy spec for variant {variant!r}")


def make_live_env(namespace: dict[str, Any], scenario: ScenarioSpec, policy, seed: int, steps: int):
    config = {
        **namespace["ENV_CONFIG"],
        "vehicles_count": len(scenario.vehicles),
        "neighbors_count": 5,
        "episode_steps": int(steps),
        "duration": int(steps),
        "terminate_on_collision": False,
        "show_trajectories": False,
        "screen_width": 1200,
        "screen_height": 320,
        "centering_position": [0.5, 0.5],
        "scaling": 9.8,
        "offscreen_rendering": False,
        "real_time_rendering": True,
    }
    if policy.env_kind == "baseline":
        return namespace["make_single_env"](seed=seed, render_mode="human", env_config=config)
    if policy.env_kind == "cbf":
        return namespace["make_cbf_single_env"](
            seed=seed,
            render_mode="human",
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
            eps_side=namespace["CBF_EPS_SIDE"],
            env_config=config,
        )
    if policy.env_kind == "guided":
        return namespace["make_guided_cbf_single_env"](
            seed=seed,
            render_mode="human",
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
            eps_side=namespace["CBF_EPS_SIDE"],
            env_config=config,
        )
    raise ValueError(f"Unknown policy env kind: {policy.env_kind}")


def load_model(namespace: dict[str, Any], policy, model_path: Path | None):
    path = Path(model_path) if model_path is not None else Path(namespace[policy.model_key])
    if not path.exists():
        raise FileNotFoundError(f"Missing model for {policy.label}: {path}")
    model_cls = namespace["GuidedCBFDDPG"] if policy.key == "guided_ddpg_cbf" else DDPG
    return model_cls.load(str(path), device=namespace["DEVICE"]), path


def run_policy(
    namespace: dict[str, Any],
    scenario: ScenarioSpec,
    variant: str,
    *,
    model_path: Path | None,
    seed: int,
    steps: int,
    episodes: int,
    pause: float,
) -> None:
    policy = policy_for_variant(variant)
    model, loaded_path = load_model(namespace, policy, model_path)
    print(
        f"[scenario-render] {scenario.name} | {policy.label} | loaded {loaded_path}",
        flush=True,
    )

    for episode in range(int(episodes)):
        episode_seed = int(seed) + 10_000 * episode
        env = make_live_env(namespace, scenario, policy, episode_seed, steps)
        try:
            env.reset(seed=episode_seed)
            obs = apply_scenario(env, scenario)
            env.render()
            total_reward = 0.0
            ego_collision_events = 0
            ego_collision_steps = 0
            correction_norms: list[float] = []
            min_h_values: list[float] = []
            for step in range(int(steps)):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32).reshape(-1)[:2])
                total_reward += float(reward)
                ego_collision_events += int(info.get("ego_collision_events", 0))
                if bool(info.get("ego_collision", False)):
                    ego_collision_steps += 1
                if "cbf_correction_norm" in info:
                    correction_norms.append(float(info.get("cbf_correction_norm", 0.0)))
                if "cbf_min_h" in info:
                    min_h_values.append(float(info.get("cbf_min_h", np.nan)))
                if terminated or truncated:
                    break
            mean_correction = float(np.mean(correction_norms)) if correction_norms else 0.0
            min_h = float(np.nanmin(min_h_values)) if min_h_values and not np.all(np.isnan(min_h_values)) else np.nan
            print(
                "[scenario-render] episode",
                {
                    "episode": episode + 1,
                    "steps": step + 1,
                    "return": round(total_reward, 3),
                    "ego_collision_events": ego_collision_events,
                    "ego_collision_steps": ego_collision_steps,
                    "mean_correction_norm": round(mean_correction, 4),
                    "min_h": round(min_h, 4) if np.isfinite(min_h) else None,
                },
                flush=True,
            )
        finally:
            env.close()
        if pause > 0.0:
            time.sleep(float(pause))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live-render fixed laneless policy diagnostic scenarios.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(VARIANT_TO_POLICY_KEY),
        default=None,
        help="Policy to render. Repeat for multiple. Default renders all three sequentially.",
    )
    parser.add_argument("--scenario", default=None, help="Scenario name to render.")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--model-path", type=Path, default=None, help="Only valid when rendering one variant.")
    parser.add_argument("--lambda-filter", type=float, default=None)
    parser.add_argument("--k0", type=float, default=None)
    parser.add_argument("--k1", type=float, default=None)
    parser.add_argument("--eps-side", type=float, default=None)
    parser.add_argument("--traffic-model", choices=["force", "mtm"], default=None)
    parser.add_argument("--env-config-json", default=None)
    parser.add_argument("--env-config-file", type=Path, default=None)
    parser.add_argument("--pause-between-episodes", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    set_stable_native_defaults()
    args = parse_args()
    if args.model_path is not None and args.variant is not None and len(args.variant) > 1:
        raise ValueError("--model-path can only be used with a single --variant")

    project_root = find_project_root(args.project_root)
    namespace = load_notebook_namespace(project_root)
    apply_cbf_overrides(namespace, args)
    apply_env_and_artifact_overrides(namespace, args)

    scenarios = {scenario.name: scenario for scenario in make_scenarios(float(namespace["ENV_CONFIG"]["road_width"]))}
    if args.list_scenarios:
        print("Available scenarios:")
        for scenario in scenarios.values():
            print(f"  {scenario.name}: {scenario.title}")
        return 0
    if not args.scenario:
        raise ValueError("Pass --scenario or --list-scenarios")
    if args.scenario not in scenarios:
        raise ValueError(f"Unknown scenario {args.scenario!r}. Use --list-scenarios.")

    variants = args.variant or ["ddpg", "ddpg-cbf", "guided-ddpg-cbf"]
    print(
        "[scenario-render] starting",
        {
            "scenario": args.scenario,
            "variants": variants,
            "traffic_model": active_traffic_model(namespace["ENV_CONFIG"]),
            "artifact_suffix": normalize_artifact_suffix(args.artifact_suffix),
            "steps": int(args.steps),
            "episodes": int(args.episodes),
        },
        flush=True,
    )
    for variant_index, variant in enumerate(variants):
        run_policy(
            namespace,
            scenarios[args.scenario],
            variant,
            model_path=args.model_path,
            seed=int(args.seed) + 100_000 * variant_index,
            steps=int(args.steps),
            episodes=int(args.episodes),
            pause=float(args.pause_between_episodes),
        )
    print("[scenario-render] finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
