from __future__ import annotations

import argparse
import copy
import faulthandler
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from stable_baselines3 import DDPG, PPO

from scripts.common.guided_cbf_minimal import install_minimal_guided_cbf
from scripts.common.laneless_evaluation_registry import (
    build_evaluation_request,
    evaluation_cache_paths,
    latest_completed_training,
    load_matching_evaluation,
    sha256_file,
    stable_json_digest,
    sync_metrics_to_requested_output,
    write_evaluation_manifest,
)
from scripts.common.laneless_script_config import active_traffic_model, env_config_from_args


TEN_KPI_SPECS: tuple[tuple[str, str], ...] = (
    ("Episode return", "episode_return"),
    ("Episode length (steps)", "episode_length_steps"),
    ("Ego collisions / km", "ego_collisions_per_km"),
    ("Minimum h", "h_min"),
    ("QP failure rate", "qp_failure_rate"),
    ("Abs speed error (m/s)", "mean_abs_speed_deviation"),
    ("Mean lateral tracking error (m)", "mean_lat_y_error_m"),
    ("Intervention rate", "event_intervention_rate"),
    ("Correction norm", "mean_correction_norm"),
    ("Mean jerk norm", "mean_jerk_norm"),
)
DEFAULT_EVALUATION_WORKERS = 20


def ten_kpi_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarise the agreed final-evaluation KPIs as mean and sample SD."""

    rows: list[dict[str, float | str]] = []
    for label, column in TEN_KPI_SPECS:
        if column not in metrics:
            raise KeyError(f"Evaluation metrics are missing required KPI column {column!r}")
        values = pd.to_numeric(metrics[column], errors="coerce").dropna()
        rows.append(
            {
                "KPI": label,
                "Mean": float(values.mean()) if len(values) else float("nan"),
                "SD": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def print_ten_kpi_summary(metrics: pd.DataFrame, *, label: str) -> None:
    """Emit the standard KPI table directly in every preset final evaluation."""

    table = ten_kpi_summary(metrics)
    print(f"[eval-runner] 10-KPI final evaluation: {label} (mean +/- sample SD)", flush=True)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"), flush=True)


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


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return candidate
        nested = candidate / "highway-rl-decision-making"
        if (nested / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return nested
    raise RuntimeError("Could not find project root containing notebooks/lanelessKaralakou.ipynb")


def exec_notebook_cell(notebook: dict[str, Any], notebook_path: Path, cell_index: int, namespace: dict[str, Any]) -> None:
    source = "".join(notebook["cells"][cell_index].get("source", []))
    print(f"[eval-runner] executing notebook cell {cell_index}", flush=True)
    exec(compile(source, f"{notebook_path}:cell-{cell_index}", "exec"), namespace)


def apply_cbf_overrides(namespace: dict[str, Any], args: argparse.Namespace) -> None:
    if args.lambda_filter is not None:
        namespace["CBF_FILTER_REWARD_LAMBDA"] = float(args.lambda_filter)
    if args.k0 is not None:
        namespace["CBF_K0"] = float(args.k0)
    if args.k1 is not None:
        namespace["CBF_K1"] = float(args.k1)
    if args.eps_side is not None:
        namespace["CBF_EPS_SIDE"] = float(args.eps_side)


def exec_notebook_cells(
    notebook_path: Path,
    cell_indices: list[int],
    namespace: dict[str, Any],
    args: argparse.Namespace | None = None,
) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_index in cell_indices:
        cell = notebook["cells"][cell_index]
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        print(f"[evaluate_laneless_karalakou] executing notebook cell {cell_index}", flush=True)
        exec(compile(source, f"{notebook_path}:cell-{cell_index}", "exec"), namespace)
        if args is not None and cell_index == 33:
            apply_cbf_overrides(namespace, args)


_LANELESS_EVAL_WORKER_STATE: dict[str, Any] | None = None


def _initialize_laneless_eval_worker(
    project_root: str,
    notebook_path: str,
    variant: str,
    model_path: str,
    device: str,
    args: argparse.Namespace,
    env_config: dict[str, Any],
    needs_cbf: bool,
) -> None:
    """Load one notebook-backed model and evaluator in an isolated CPU worker."""

    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    root = Path(project_root).resolve()
    notebook = Path(notebook_path).resolve()
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "PPO": PPO,
        "DDPG": DDPG,
    }
    base_cells = [2, 3, 5, 6, 8]
    cbf_cells = [42, 44, 46, 48, 50, 52]
    if variant == "guided-ddpg-cbf":
        cbf_cells.append(64)
    exec_notebook_cells(
        notebook,
        base_cells + (cbf_cells if needs_cbf else []),
        namespace,
        args if needs_cbf else None,
    )
    apply_cbf_overrides(namespace, args)
    namespace["DEVICE"] = str(device)
    namespace["ENV_CONFIG"] = copy.deepcopy(env_config)
    if variant == "guided-ddpg-cbf":
        install_minimal_guided_cbf(namespace)

    if variant == "ppo":
        model = PPO.load(str(model_path), device=str(device))
    elif variant == "ddpg":
        model = DDPG.load(str(model_path), device=str(device))
    else:
        model = namespace["GuidedCBFDDPG"].load(
            str(model_path), device=str(device)
        )

    global _LANELESS_EVAL_WORKER_STATE
    _LANELESS_EVAL_WORKER_STATE = {
        "namespace": namespace,
        "model": model,
        "variant": variant,
        "args": args,
        "env_config": env_config,
    }


def _evaluate_laneless_episode_worker(
    task: tuple[int, int]
) -> tuple[int, dict[str, Any]]:
    state = _LANELESS_EVAL_WORKER_STATE
    if state is None:
        raise RuntimeError("laneless evaluation worker was not initialized")
    episode_index, episode_seed = task
    namespace = state["namespace"]
    model = state["model"]
    variant = state["variant"]
    args = state["args"]
    if variant in {"ppo", "ddpg"}:
        frame = namespace["evaluate_policy_with_metrics"](
            model,
            episodes=1,
            seed=int(episode_seed),
            deterministic=True,
            env_config=state["env_config"],
            episode_log_path=None,
            episode_log_label="parallel-worker",
        )
    else:
        frame = namespace["evaluate_cbf_policy_with_metrics"](
            model,
            episodes=1,
            seed=int(episode_seed),
            deterministic=True,
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
            eps_side=namespace["CBF_EPS_SIDE"],
            env_config=state["env_config"],
            episode_log_path=None,
            episode_log_label="parallel-worker",
        )
    if len(frame) != 1:
        raise RuntimeError(
            f"worker returned {len(frame)} rows for episode {episode_index}"
        )
    row = frame.iloc[0].to_dict()
    # Match the existing serial evaluator's zero-based episode numbering.
    row["episode"] = int(episode_index) - 1
    row["seed"] = int(episode_seed)
    return int(episode_index), row


def _evaluate_laneless_with_workers(
    *,
    namespace: dict[str, Any],
    notebook_path: Path,
    model_path: Path,
    variant: str,
    episodes: int,
    seed: int,
    env_config: dict[str, Any],
    args: argparse.Namespace,
    needs_cbf: bool,
    episode_log_path: Path,
    episode_log_label: str,
) -> pd.DataFrame:
    workers = int(args.workers)
    tasks = [
        (episode_index, int(seed) + episode_index)
        for episode_index in range(int(episodes))
    ]
    print(
        f"[eval-runner] workers={workers} evaluation_device=cpu "
        f"pending={len(tasks)}",
        flush=True,
    )
    rows_by_index: dict[int, dict[str, Any]] = {}
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_initialize_laneless_eval_worker,
        initargs=(
            str(Path(namespace["PROJECT_ROOT"]).resolve()),
            str(notebook_path),
            variant,
            str(model_path),
            "cpu",
            args,
            env_config,
            bool(needs_cbf),
        ),
    )
    try:
        futures = [
            executor.submit(_evaluate_laneless_episode_worker, task)
            for task in tasks
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            episode_index, row = future.result()
            rows_by_index[int(episode_index)] = row
            ordered_rows = [
                rows_by_index[index]
                for index in sorted(rows_by_index)
            ]
            namespace["_write_evaluation_episode_progress"](
                ordered_rows,
                episode_log_path,
                label=episode_log_label,
                expected_episodes=int(episodes),
            )
            print(
                f"[eval-runner] completed={completed}/{len(tasks)} "
                f"episode={episode_index + 1}/{len(tasks)} seed={row['seed']}",
                flush=True,
            )
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    namespace["_write_evaluation_episode_progress"](
        rows,
        episode_log_path,
        label=episode_log_label,
        expected_episodes=int(episodes),
        state="complete",
    )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run laneless Karalakou final evaluations out of process.")
    parser.add_argument("--variant", choices=["ppo", "ddpg", "ddpg-cbf", "guided-ddpg-cbf"], required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_EVALUATION_WORKERS,
        help="independent single-threaded CPU evaluation workers",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--traffic-model", choices=["force", "mtm"], default=None)
    parser.add_argument("--env-config-json", default=None)
    parser.add_argument("--env-config-file", type=Path, default=None)
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--lambda-filter", type=float, default=None)
    parser.add_argument("--k0", type=float, default=None)
    parser.add_argument("--k1", type=float, default=None)
    parser.add_argument("--eps-side", type=float, default=None)
    parser.add_argument(
        "--use-latest-training",
        action="store_true",
        help="Evaluate the latest completed immutable training run and reuse only its matching cached evaluation.",
    )
    parser.add_argument(
        "--force-reevaluate",
        action="store_true",
        help="Ignore a matching cached evaluation and run the requested evaluation again.",
    )
    return parser.parse_args()


def with_stem_suffix(path: Path, suffix: str) -> Path:
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


def artifact_path(path: Path, traffic_model: str, artifact_suffix: str | None) -> Path:
    suffix = normalize_artifact_suffix(artifact_suffix)
    if suffix is None and traffic_model != "force":
        suffix = f"traffic_{traffic_model}"
    return with_stem_suffix(Path(path), suffix) if suffix else Path(path)


def notebook_code_sha256(notebook_path: Path) -> str:
    """Fingerprint notebook code only, so display-output changes do not invalidate an evaluation."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    return stable_json_digest(sources)


def default_model_path(
    namespace: dict[str, Any],
    args: argparse.Namespace,
    traffic_model: str,
) -> Path:
    model_key_by_variant = {
        "ppo": "MODEL_PATH",
        "ddpg": "DDPG_MODEL_PATH",
        "ddpg-cbf": "DDPG_CBF_MODEL_PATH",
        "guided-ddpg-cbf": "GUIDED_DDPG_CBF_MODEL_PATH",
    }
    return artifact_path(
        Path(namespace[model_key_by_variant[args.variant]]),
        traffic_model,
        args.artifact_suffix,
    )


def atomic_write_metrics(metrics: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    metrics.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def main() -> int:
    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    args = parse_args()
    if int(args.workers) <= 0:
        raise ValueError("--workers must be positive")

    project_root = find_project_root(args.project_root or Path.cwd())
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "PPO": PPO,
        "DDPG": DDPG,
    }

    base_cells = [2, 3, 5, 6, 8]
    # Cell 43 is the DDPG-CBF training cell.  Evaluation only needs the CBF
    # definitions and metric helper cells; executing 43 would honor its
    # notebook default and launch a fresh training subprocess before loading
    # the saved model.
    # The notebook's CBF definitions are split across the current C-section:
    # QP imports/geometries, wrapper, tuned override, evaluator, and (for the
    # guided variant) its actor class.  Keep this explicit so final DDPG/CBF
    # evaluations load the same helpers as an interactive notebook run rather
    # than relying on stale pre-ladder cell numbers.
    cbf_cells = [42, 44, 46, 48, 50, 52]
    needs_cbf = args.variant in {"ddpg-cbf", "guided-ddpg-cbf"}
    if args.variant == "guided-ddpg-cbf":
        cbf_cells.append(64)
    exec_notebook_cells(
        notebook_path,
        base_cells + (cbf_cells if needs_cbf else []),
        namespace,
        args,
    )
    apply_cbf_overrides(namespace, args)

    namespace["DEVICE"] = args.device
    namespace["ENV_CONFIG"] = env_config_from_args(args, namespace["ENV_CONFIG"])
    if args.variant == "guided-ddpg-cbf":
        install_minimal_guided_cbf(namespace)
    traffic_model = active_traffic_model(namespace["ENV_CONFIG"])
    output_path = args.output
    training_run = None
    model_path = Path(args.model_path) if args.model_path is not None else default_model_path(namespace, args, traffic_model)

    if args.use_latest_training:
        if args.model_path is not None:
            raise ValueError("--model-path cannot be combined with --use-latest-training")
        training_run = latest_completed_training(Path(namespace["ARTIFACT_DIR"]), args.variant)
        if training_run is None:
            raise RuntimeError(
                f"No completed archived training run is registered for {args.variant!r}. "
                "Train through the notebook task runner first, or evaluate an explicit --model-path without cache mode."
            )
        model_path = training_run.model_path
        print(
            f"[eval-runner] latest saved training: {training_run.label} | "
            f"run={training_run.run_dir.name} | model={model_path}",
            flush=True,
        )

    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model does not exist: {model_path}")

    cache_metrics_path: Path | None = None
    cache_manifest_path: Path | None = None
    evaluation_request: dict[str, Any] | None = None
    evaluation_fingerprint: str | None = None
    if args.use_latest_training:
        cbf_config = (
            {
                "lambda_filter": float(namespace["CBF_FILTER_REWARD_LAMBDA"]),
                "k0": float(namespace["CBF_K0"]),
                "k1": float(namespace["CBF_K1"]),
                "eps_side": float(namespace["CBF_EPS_SIDE"]),
            }
            if needs_cbf
            else {}
        )
        evaluation_request = build_evaluation_request(
            variant=args.variant,
            model_path=model_path,
            training_run=training_run,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
            traffic_model=traffic_model,
            env_config=namespace["ENV_CONFIG"],
            cbf_config=cbf_config,
            evaluator_code_sha256=sha256_file(Path(__file__).resolve()),
            notebook_code_sha256=notebook_code_sha256(notebook_path),
        )
        evaluation_fingerprint = stable_json_digest(evaluation_request)
        cache_metrics_path, cache_manifest_path = evaluation_cache_paths(
            artifact_dir=Path(namespace["ARTIFACT_DIR"]),
            model_path=model_path,
            training_run=training_run,
            evaluation_fingerprint=evaluation_fingerprint,
        )
        cached = None if args.force_reevaluate else load_matching_evaluation(
            metrics_path=cache_metrics_path,
            manifest_path=cache_manifest_path,
            evaluation_fingerprint=evaluation_fingerprint,
            model_sha256=str(evaluation_request["model_sha256"]),
        )
        if cached is not None:
            sync_metrics_to_requested_output(cache_metrics_path, output_path)
            metrics = pd.read_csv(cache_metrics_path)
            print(
                f"[eval-runner] matching evaluation cache hit for model {evaluation_request['model_sha256'][:12]}; "
                f"reused {cache_metrics_path}",
                flush=True,
            )
            print(f"[eval-runner] refreshed {output_path}", flush=True)
            print_ten_kpi_summary(metrics, label=f"{args.variant} (cached)")
            return 0
        print(
            "[eval-runner] no matching evaluation manifest; evaluating the latest saved training now",
            flush=True,
        )

    episode_log_path = output_path.with_name(output_path.stem + "_episodes_progress.csv")
    if int(args.workers) > 1:
        print(
            f"[eval-runner] evaluating {args.variant} with {int(args.workers)} workers",
            flush=True,
        )
        metrics = _evaluate_laneless_with_workers(
            namespace=namespace,
            notebook_path=notebook_path,
            model_path=model_path,
            variant=args.variant,
            episodes=int(args.episodes),
            seed=int(args.seed),
            env_config=namespace["ENV_CONFIG"],
            args=args,
            needs_cbf=needs_cbf,
            episode_log_path=episode_log_path,
            episode_log_label=f"{args.variant}@{output_path.stem}",
        )
    elif args.variant == "ppo":
        print(f"[eval-runner] loading {model_path}", flush=True)
        model = namespace["PPO"].load(str(model_path), device=args.device)
        print("[eval-runner] evaluating PPO", flush=True)
        episode_log_path = output_path.with_name(output_path.stem + "_episodes_progress.csv")
        metrics = namespace["evaluate_policy_with_metrics"](
            model,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
            env_config=namespace["ENV_CONFIG"],
            episode_log_path=episode_log_path,
            episode_log_label=f"{args.variant}@{output_path.stem}",
        )
    elif args.variant == "ddpg":
        print(f"[eval-runner] loading {model_path}", flush=True)
        model = namespace["DDPG"].load(str(model_path), device=args.device)
        print("[eval-runner] evaluating DDPG", flush=True)
        episode_log_path = output_path.with_name(output_path.stem + "_episodes_progress.csv")
        metrics = namespace["evaluate_policy_with_metrics"](
            model,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
            env_config=namespace["ENV_CONFIG"],
            episode_log_path=episode_log_path,
            episode_log_label=f"{args.variant}@{output_path.stem}",
        )
    elif args.variant == "ddpg-cbf":
        print(f"[eval-runner] loading {model_path}", flush=True)
        model = namespace["DDPG"].load(str(model_path), device=args.device)
        print("[eval-runner] evaluating DDPG-CBF", flush=True)
        episode_log_path = output_path.with_name(output_path.stem + "_episodes_progress.csv")
        metrics = namespace["evaluate_cbf_policy_with_metrics"](
            model,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
            eps_side=namespace["CBF_EPS_SIDE"],
            env_config=namespace["ENV_CONFIG"],
            episode_log_path=episode_log_path,
            episode_log_label=f"{args.variant}@{output_path.stem}",
        )
    else:
        print(f"[eval-runner] loading {model_path}", flush=True)
        model = namespace["GuidedCBFDDPG"].load(str(model_path), device=args.device)
        print("[eval-runner] evaluating guided DDPG-CBF", flush=True)
        episode_log_path = output_path.with_name(output_path.stem + "_episodes_progress.csv")
        metrics = namespace["evaluate_cbf_policy_with_metrics"](
            model,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
            eps_side=namespace["CBF_EPS_SIDE"],
            env_config=namespace["ENV_CONFIG"],
            episode_log_path=episode_log_path,
            episode_log_label=f"{args.variant}@{output_path.stem}",
        )

    if cache_metrics_path is not None and cache_manifest_path is not None:
        atomic_write_metrics(metrics, cache_metrics_path)
        write_evaluation_manifest(
            manifest_path=cache_manifest_path,
            metrics_path=cache_metrics_path,
            request=evaluation_request or {},
            evaluation_fingerprint=evaluation_fingerprint or "",
        )
        sync_metrics_to_requested_output(cache_metrics_path, output_path)
        print(f"[eval-runner] cached evaluation at {cache_metrics_path}", flush=True)
    else:
        atomic_write_metrics(metrics, output_path)
    print(f"[eval-runner] wrote {output_path}", flush=True)
    print_ten_kpi_summary(metrics, label=args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
