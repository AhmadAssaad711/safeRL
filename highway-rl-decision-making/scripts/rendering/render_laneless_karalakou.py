from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from scripts.common.guided_cbf_minimal import install_minimal_guided_cbf
from scripts.common.laneless_script_config import active_traffic_model, env_config_from_args


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


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return candidate
        nested = candidate / "highway-rl-decision-making"
        if (nested / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return nested
    raise RuntimeError("Could not find project root containing notebooks/lanelessKaralakou.ipynb")


def exec_notebook_cells(notebook_path: Path, cell_indices: list[int], namespace: dict[str, Any]) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_index in cell_indices:
        cell = notebook["cells"][cell_index]
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        print(f"[render_laneless_karalakou] executing notebook cell {cell_index}", flush=True)
        exec(compile(source, f"{notebook_path}:cell-{cell_index}", "exec"), namespace)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render laneless Karalakou policies out of the notebook kernel.")
    parser.add_argument("--variant", choices=["ppo", "ddpg", "ddpg-cbf", "guided-ddpg-cbf"], required=True)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--lambda-filter", type=float, default=None)
    parser.add_argument("--k0", type=float, default=None)
    parser.add_argument("--k1", type=float, default=None)
    parser.add_argument("--eps-side", type=float, default=None)
    parser.add_argument("--traffic-model", choices=["force", "mtm"], default=None)
    parser.add_argument("--env-config-json", default=None)
    parser.add_argument("--env-config-file", type=Path, default=None)
    return parser.parse_args()


def with_stem_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def normalize_artifact_suffix(value: str | None) -> str | None:
    if value is None:
        return None
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    suffix = suffix.strip("._-")
    if len(suffix) > 16:
        digest = hashlib.sha1(suffix.encode("utf-8")).hexdigest()[:8]
        suffix = f"{suffix[:7].strip('._-')}_{digest}"
    return suffix or None


def artifact_path(path: Path, traffic_model: str, artifact_suffix: str | None) -> Path:
    suffix = normalize_artifact_suffix(artifact_suffix)
    if suffix is None and traffic_model != "force":
        suffix = f"traffic_{traffic_model}"
    return with_stem_suffix(Path(path), suffix) if suffix else Path(path)


def apply_cbf_overrides(namespace: dict[str, Any], args: argparse.Namespace) -> None:
    if args.lambda_filter is not None:
        namespace["CBF_FILTER_REWARD_LAMBDA"] = float(args.lambda_filter)
    if args.k0 is not None:
        namespace["CBF_K0"] = float(args.k0)
    if args.k1 is not None:
        namespace["CBF_K1"] = float(args.k1)
    if args.eps_side is not None:
        namespace["CBF_EPS_SIDE"] = float(args.eps_side)


def main() -> int:
    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    args = parse_args()

    project_root = find_project_root(args.project_root or Path.cwd())
    # The notebook bootstrap derives PROJECT_ROOT from the process working
    # directory.  Run it from the resolved inner project so that it imports
    # and registers the local ``lane-free-v0`` environment.
    os.chdir(project_root)
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    namespace: dict[str, Any] = {"__name__": "__main__"}
    base_cells = [2, 3, 5, 6, 8]
    cbf_cells = [33, 35, 37, 39, 41]
    needs_cbf = args.variant in {"ddpg-cbf", "guided-ddpg-cbf"}
    exec_notebook_cells(
        notebook_path,
        base_cells + (cbf_cells if needs_cbf else []),
        namespace,
    )
    namespace["DEVICE"] = args.device
    namespace["ENV_CONFIG"] = env_config_from_args(args, namespace["ENV_CONFIG"])
    apply_cbf_overrides(namespace, args)
    traffic_model = active_traffic_model(namespace["ENV_CONFIG"])

    if args.variant == "ppo":
        model_path = args.model_path or artifact_path(namespace["MODEL_PATH"], traffic_model, args.artifact_suffix)
        model = namespace["PPO"].load(str(model_path), device=args.device)
        env = namespace["make_single_env"](seed=args.seed, render_mode="human", env_config=namespace["ENV_CONFIG"])
    elif args.variant == "ddpg":
        model_path = args.model_path or artifact_path(namespace["DDPG_MODEL_PATH"], traffic_model, args.artifact_suffix)
        model = namespace["DDPG"].load(str(model_path), device=args.device)
        env = namespace["make_single_env"](seed=args.seed, render_mode="human", env_config=namespace["ENV_CONFIG"])
    elif args.variant == "ddpg-cbf":
        model_path = args.model_path or artifact_path(namespace["DDPG_CBF_MODEL_PATH"], traffic_model, args.artifact_suffix)
        model = namespace["DDPG"].load(str(model_path), device=args.device)
        env = namespace["make_cbf_single_env"](
            seed=args.seed,
            render_mode="human",
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
            eps_side=namespace["CBF_EPS_SIDE"],
            env_config=namespace["ENV_CONFIG"],
        )
    else:
        install_minimal_guided_cbf(namespace)
        model_path = args.model_path or artifact_path(namespace["GUIDED_DDPG_CBF_MODEL_PATH"], traffic_model, args.artifact_suffix)
        model = namespace["GuidedCBFDDPG"].load(str(model_path), device=args.device)
        env = namespace["make_guided_cbf_single_env"](
            seed=args.seed,
            render_mode="human",
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
            eps_side=namespace["CBF_EPS_SIDE"],
            env_config=namespace["ENV_CONFIG"],
        )

    print(f"[render-runner] loaded {model_path}", flush=True)
    print(f"[render-runner] rendering {args.variant} for {args.episodes:,} episode(s) x {args.steps:,} steps", flush=True)
    try:
        for episode in range(int(args.episodes)):
            obs, _ = env.reset(seed=int(args.seed) + episode)
            for _ in range(args.steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
    finally:
        env.close()
    print("[render-runner] finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
