from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from scripts.common.guided_cbf_minimal import install_minimal_guided_cbf
from scripts.common.laneless_script_config import active_traffic_model, env_config_from_args
from scripts.common.laneless_training_registry import archive_training_outputs, make_run_tag


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
        nested = candidate / "safeRL_workspace"
        if (nested / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return nested
    raise RuntimeError("Could not find project root containing notebooks/lanelessKaralakou.ipynb")


def snapshot_tensorboard_events(roots: dict[str, Path]) -> dict[str, dict[str, str]]:
    """Return the event files below each root, keyed by resolved source path."""
    found: dict[str, dict[str, str]] = {}
    for kind, root in roots.items():
        if not root.exists():
            continue
        for path in root.rglob("events.out.tfevents.*"):
            if path.is_file():
                resolved = str(path.resolve())
                # Later roots intentionally win when the custom root happens
                # to be nested below the standard TensorBoard directory.
                found[resolved] = {"kind": kind, "path": resolved}
    return found


def exec_notebook_cell(notebook: dict[str, Any], notebook_path: Path, cell_index: int, namespace: dict[str, Any]) -> None:
    cell = notebook["cells"][cell_index]
    if cell.get("cell_type") != "code":
        return
    source = "".join(cell.get("source", []))
    print(f"[notebook-task] executing notebook cell {cell_index}", flush=True)
    exec(compile(source, f"{notebook_path}:cell-{cell_index}", "exec"), namespace)


def exec_notebook_cell_tail(
    notebook: dict[str, Any],
    notebook_path: Path,
    cell_index: int,
    namespace: dict[str, Any],
    marker: str,
) -> None:
    source = "".join(notebook["cells"][cell_index].get("source", []))
    if marker not in source:
        raise RuntimeError(f"Could not find marker {marker!r} in notebook cell {cell_index}")
    tail = source[source.index(marker) :]
    print(f"[notebook-task] executing notebook cell {cell_index} from {marker!r}", flush=True)
    exec(compile(tail, f"{notebook_path}:cell-{cell_index}-tail", "exec"), namespace)


def exec_notebook_cells(notebook_path: Path, cell_indices: list[int], namespace: dict[str, Any]) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_index in cell_indices:
        cell = notebook["cells"][cell_index]
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        print(f"[run_laneless_notebook_task] executing notebook cell {cell_index}", flush=True)
        exec(compile(source, f"{notebook_path}:cell-{cell_index}", "exec"), namespace)


def apply_overrides(namespace: dict[str, Any], args: argparse.Namespace, task: dict[str, Any]) -> None:
    namespace["DEVICE"] = args.device
    namespace[task["flag"]] = True
    if "ENV_CONFIG" in namespace:
        namespace["ENV_CONFIG"] = env_config_from_args(args, namespace["ENV_CONFIG"])

    if args.timesteps is not None:
        timesteps = int(args.timesteps)
        namespace[str(task["timesteps_key"])] = timesteps
        if args.task == "guided-ddpg-cbf-train":
            namespace["DDPG_CBF_TOTAL_TIMESTEPS"] = timesteps
    if args.n_envs is not None:
        namespace["DDPG_NUM_ENVS"] = int(args.n_envs)
        namespace["DDPG_CBF_NUM_ENVS"] = int(args.n_envs)
        ppo_n_envs_key = task.get("n_envs_key")
        if ppo_n_envs_key:
            namespace[str(ppo_n_envs_key)] = int(args.n_envs)
    if args.lambda_filter is not None:
        namespace["CBF_FILTER_REWARD_LAMBDA"] = float(args.lambda_filter)
    if args.k0 is not None:
        namespace["CBF_K0"] = float(args.k0)
    if args.k1 is not None:
        namespace["CBF_K1"] = float(args.k1)
    if args.eps_side is not None:
        namespace["CBF_EPS_SIDE"] = float(args.eps_side)


def _with_stem_suffix(path: Path, suffix: str) -> Path:
    suffixed = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    if os.name != "nt" or len(str(suffixed)) < 260:
        return suffixed

    digest = hashlib.sha1(suffixed.stem.encode("utf-8")).hexdigest()[:8]
    overflow = len(str(suffixed)) - 248
    keep = max(24, len(suffixed.stem) - overflow - len(digest) - 1)
    short_stem = suffixed.stem[:keep].rstrip("._-")
    return suffixed.with_name(f"{short_stem}_{digest}{suffixed.suffix}")


def normalize_artifact_suffix(value: str | None) -> str | None:
    if value is None:
        return None
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    suffix = suffix.strip("._-")
    if len(suffix) > 16:
        digest = hashlib.sha1(suffix.encode("utf-8")).hexdigest()[:8]
        suffix = f"{suffix[:7].strip('._-')}_{digest}"
    return suffix or None


def apply_traffic_artifact_suffix(namespace: dict[str, Any], artifact_suffix: str | None = None) -> None:
    traffic_model = active_traffic_model(namespace.get("ENV_CONFIG", {}))
    suffix = normalize_artifact_suffix(artifact_suffix)
    if suffix is None and traffic_model != "force":
        suffix = f"traffic_{traffic_model}"
    if suffix is None:
        return
    for key in [
        "MODEL_PATH",
        "HISTORY_PATH",
        "PLOT_PATH",
        "DDPG_MODEL_PATH",
        "DDPG_HISTORY_PATH",
        "DDPG_PLOT_PATH",
        "DDPG_CBF_MODEL_PATH",
        "DDPG_CBF_HISTORY_PATH",
        "DDPG_CBF_PLOT_PATH",
        "GUIDED_DDPG_CBF_MODEL_PATH",
        "GUIDED_DDPG_CBF_HISTORY_PATH",
    ]:
        if key in namespace and namespace[key] is not None:
            namespace[key] = _with_stem_suffix(Path(namespace[key]), suffix)


TASKS = {
    "ppo-train": {
        "deps": [2, 3, 5, 6, 8],
        "cell": 11,
        # The canonical notebook setup cell only defines the shared launcher.
        # These seven cells invoke its sequential 1M-per-policy runners.
        "train_cells": [13, 15, 17, 19, 21, 23, 25],
        "post_cells": [27, 28],
        "flag": "PPO_1M_RUN_TRAINING",
        "timesteps_key": "PPO_1M_TIMESTEPS_PER_POLICY",
        "n_envs_key": "PPO_1M_NUM_ENVS",
    },
    "ddpg-train": {
        "deps": [2, 3, 5, 6, 8, 9],
        "cell": 33,
        "flag": "RUN_DDPG_TRAIN",
        "timesteps_key": "DDPG_TOTAL_TIMESTEPS",
    },
    "ddpg-cbf-train": {
        "deps": [2, 3, 5, 6, 8, 9, 42, 44, 46, 48, 50, 52],
        "cell": 54,
        "flag": "RUN_DDPG_CBF_TRAIN",
        "timesteps_key": "DDPG_CBF_TOTAL_TIMESTEPS",
    },
    "guided-ddpg-cbf-train": {
        "deps": [2, 3, 5, 6, 8, 9, 42, 44, 46, 48, 50, 52],
        "cell": 64,
        "flag": "RUN_GUIDED_DDPG_CBF_TRAIN",
        "timesteps_key": "GUIDED_DDPG_CBF_TOTAL_TIMESTEPS",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run crash-prone laneless notebook training cells out of process.")
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--lambda-filter", type=float, default=None)
    parser.add_argument("--k0", type=float, default=None)
    parser.add_argument("--k1", type=float, default=None)
    parser.add_argument("--eps-side", type=float, default=None)
    parser.add_argument("--traffic-model", choices=["force", "mtm"], default=None)
    parser.add_argument("--env-config-json", default=None)
    parser.add_argument("--env-config-file", type=Path, default=None)
    parser.add_argument("--artifact-suffix", default=None, help="Suffix model/history artifacts for this environment.")
    parser.add_argument("--run-tag", default=None)
    return parser.parse_args()


def main() -> int:
    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    args = parse_args()
    if args.run_tag is None:
        args.run_tag = make_run_tag()

    project_root = find_project_root(args.project_root or Path.cwd())
    # Notebook bootstrap derives PROJECT_ROOT from the process working
    # directory.  Use the resolved inner project so the local lane-free
    # environment is imported and registered before task cells run.
    os.chdir(project_root)
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    task = TASKS[args.task]
    # Notebook cells use IPython's display() for their final summaries.  A
    # plain subprocess has no injected display helper, so print the same value
    # instead of failing after a successful training/save.
    namespace: dict[str, Any] = {"__name__": "__main__", "display": print}

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_index in task["deps"]:
        exec_notebook_cell(notebook, notebook_path, cell_index, namespace)
        if cell_index in {6, 44}:
            apply_overrides(namespace, args, task)
    apply_overrides(namespace, args, task)
    apply_traffic_artifact_suffix(namespace, args.artifact_suffix)

    # TensorBoard appends a long generated event filename.  On deeply nested
    # Windows workspaces, the notebook's custom_metrics/run-name hierarchy can
    # exceed MAX_PATH even though the directory itself can be created.
    custom_tb_root = namespace.get("TENSORBOARD_CUSTOM_ROOT")
    if os.name == "nt" and custom_tb_root is not None:
        representative_event = Path(custom_tb_root) / ("x" * 32) / ("events.out.tfevents." + "x" * 48)
        if len(str(representative_event)) >= 260:
            # The OneDrive workspace itself is already deep enough that even
            # the shortened artifact path can exceed Windows MAX_PATH once
            # TensorBoard appends its generated event filename.  Keep the
            # custom bridge logs local and short; SB3's normal TensorBoard
            # logs and all training artifacts remain under ARTIFACT_DIR.
            custom_tb_root = Path(tempfile.gettempdir()) / "laneless_tb"
            custom_tb_root.mkdir(parents=True, exist_ok=True)
            namespace["TENSORBOARD_CUSTOM_ROOT"] = custom_tb_root
            print(f"[notebook-task] shortened custom TensorBoard root to {custom_tb_root}", flush=True)

    tensorboard_roots = {
        "standard": Path(namespace["ARTIFACT_DIR"]) / "tensorboard",
        "custom": Path(namespace.get("TENSORBOARD_CUSTOM_ROOT", Path(namespace["ARTIFACT_DIR"]) / "tensorboard_custom")),
    }
    tensorboard_before = snapshot_tensorboard_events(tensorboard_roots)

    print(
        "[notebook-task] starting",
        {
            "task": args.task,
            "device": namespace["DEVICE"],
            "timesteps": namespace.get(str(task["timesteps_key"])),
            "n_envs": namespace.get("DDPG_NUM_ENVS"),
            "lambda_filter": namespace.get("CBF_FILTER_REWARD_LAMBDA"),
            "k0": namespace.get("CBF_K0"),
            "k1": namespace.get("CBF_K1"),
            "eps_side": namespace.get("CBF_EPS_SIDE"),
            "traffic_model": active_traffic_model(namespace.get("ENV_CONFIG", {})),
            "artifact_suffix": normalize_artifact_suffix(args.artifact_suffix),
        },
        flush=True,
    )
    if args.task == "ddpg-cbf-train":
        # This process is already the isolated child requested by notebook
        # cell 54.  Re-enter the cell's in-process branch instead of spawning
        # another copy of this runner recursively.
        namespace["RUN_DDPG_CBF_TRAIN_SUBPROCESS"] = False
    if args.task == "guided-ddpg-cbf-train":
        # This runner is already the isolated training subprocess requested by
        # notebook cell 64.  Disable that cell's optional subprocess delegate
        # here; otherwise each child re-enters the same cell and recursively
        # launches more guided-training children instead of training.
        namespace["RUN_GUIDED_DDPG_CBF_TRAIN_SUBPROCESS"] = False
        install_minimal_guided_cbf(namespace)
        apply_traffic_artifact_suffix(namespace, args.artifact_suffix)
        exec_notebook_cell_tail(
            notebook,
            notebook_path,
            int(task["cell"]),
            namespace,
            "RUN_GUIDED_DDPG_CBF_TRAIN =",
        )
    else:
        exec_notebook_cell(notebook, notebook_path, int(task["cell"]), namespace)
        for followup_cell in task.get("train_cells", []):
            exec_notebook_cell(notebook, notebook_path, int(followup_cell), namespace)
        for post_cell in task.get("post_cells", []):
            exec_notebook_cell(notebook, notebook_path, int(post_cell), namespace)
    if args.task == "ppo-train":
        # Each of the seven notebook runners owns its model manifest and paired
        # 200-OFF/200-ON evaluation artifacts, so the legacy single-model
        # archive path below does not apply.
        print("[notebook-task] completed canonical seven-policy PPO-CBF ladder", flush=True)
        return 0

    tensorboard_after = snapshot_tensorboard_events(tensorboard_roots)
    namespace["PAPER_TENSORBOARD_EVENT_FILES"] = [
        entry for path, entry in tensorboard_after.items() if path not in tensorboard_before
    ]
    print(
        f"[notebook-task] captured {len(namespace['PAPER_TENSORBOARD_EVENT_FILES'])} new TensorBoard event file(s)",
        flush=True,
    )

    archived = archive_training_outputs(
        namespace=namespace,
        task_name=args.task,
        run_tag=args.run_tag,
        command=sys.argv,
    )
    if archived is not None:
        if archived["complete"]:
            print(
                "[notebook-task] linked latest run",
                {
                    "variant": archived["label"],
                    "run_dir": archived["run_dir"],
                    "max_timestep": archived["max_timestep"],
                },
                flush=True,
            )
        else:
            print(
                "[notebook-task] archived partial run without updating latest",
                {
                    "variant": archived["label"],
                    "run_dir": archived["run_dir"],
                    "max_timestep": archived["max_timestep"],
                    "expected_timesteps": archived["expected_timesteps"],
                },
                flush=True,
            )
    print(f"[notebook-task] completed {args.task}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
