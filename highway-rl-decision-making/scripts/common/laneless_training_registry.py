from __future__ import annotations

import json
import shutil
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Any

import pandas as pd


VARIANT_BY_TASK: dict[str, dict[str, str]] = {
    "ddpg-train": {
        "label": "DDPG without CBF",
        "slug": "ddpg_without_cbf",
        "model_key": "DDPG_MODEL_PATH",
        "history_key": "DDPG_HISTORY_PATH",
        "step_trace": "ddpg_without_cbf_training_step_trace.csv",
        "episode_trace": "ddpg_without_cbf_training_episode_trace.csv",
        "timesteps_key": "DDPG_TOTAL_TIMESTEPS",
    },
    "ddpg-cbf-train": {
        "label": "DDPG-CBF reward",
        "slug": "ddpg_cbf_reward",
        "model_key": "DDPG_CBF_MODEL_PATH",
        "history_key": "DDPG_CBF_HISTORY_PATH",
        "step_trace": "ddpg_cbf_reward_training_step_trace.csv",
        "episode_trace": "ddpg_cbf_reward_training_episode_trace.csv",
        "timesteps_key": "DDPG_CBF_TOTAL_TIMESTEPS",
    },
    "guided-ddpg-cbf-train": {
        "label": "DDPG-CBF reward + loss",
        "slug": "ddpg_cbf_reward_loss",
        "model_key": "GUIDED_DDPG_CBF_MODEL_PATH",
        "history_key": "GUIDED_DDPG_CBF_HISTORY_PATH",
        "step_trace": "ddpg_cbf_reward_loss_training_step_trace.csv",
        "episode_trace": "ddpg_cbf_reward_loss_training_episode_trace.csv",
        "timesteps_key": "GUIDED_DDPG_CBF_TOTAL_TIMESTEPS",
    },
}


def make_run_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _copy_if_present(source: Path, destination: Path) -> str | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensorboard_archive_dir(namespace: dict[str, Any], run_dir: Path) -> Path:
    """Return a short, stable archive root for a run's TensorBoard files.

    The main training-run directory can already be close to Windows' legacy
    260-character path limit when the project is inside a long OneDrive path.
    Keeping event files in ``artifacts/tb/<run-hash>`` avoids adding another
    deep directory hierarchy while ``run_config.json`` retains the mapping.
    """
    artifact_root = Path(namespace["ARTIFACT_DIR"]).parent
    run_hash = sha1(str(run_dir.resolve()).encode("utf-8")).hexdigest()[:10]
    return artifact_root / "tb" / run_hash


def _copy_tensorboard_events(namespace: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    archive_root = _tensorboard_archive_dir(namespace, run_dir)
    for entry in namespace.get("PAPER_TENSORBOARD_EVENT_FILES", []):
        if isinstance(entry, dict):
            source = Path(entry.get("path", ""))
            kind = str(entry.get("kind", "unknown"))
        else:
            source = Path(entry)
            kind = "unknown"
        if not source.is_file():
            continue
        # Keep only a one-character category directory.  TensorBoard event
        # names are unique for a run, and their archived locations are stored
        # in the run metadata, so source-parent nesting is unnecessary.
        kind_code = {"standard": "s", "custom": "c"}.get(kind.lower(), "x")
        destination = archive_root / kind_code / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            {
                "kind": kind,
                "source": str(source),
                "archived_path": str(destination),
                "size_bytes": int(destination.stat().st_size),
            }
        )
    return copied


def _max_timestep(path: Path) -> float:
    if not path.exists():
        return 0.0
    frame = pd.read_csv(path, usecols=lambda column: column in {"global_timestep", "timestep", "sb3_num_timesteps"})
    for column in ["global_timestep", "timestep", "sb3_num_timesteps"]:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            return float(values.max()) if not values.empty else 0.0
    return 0.0


def _load_latest(latest_path: Path) -> dict[str, Any]:
    if not latest_path.exists():
        return {}
    try:
        return json.loads(latest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def archive_training_outputs(
    *,
    namespace: dict[str, Any],
    task_name: str,
    run_tag: str | None = None,
    command: list[str] | None = None,
    complete_threshold: float = 0.99,
) -> dict[str, Any] | None:
    if task_name not in VARIANT_BY_TASK:
        return None

    spec = VARIANT_BY_TASK[task_name]
    artifact_dir = Path(namespace["ARTIFACT_DIR"])
    run_tag = run_tag or make_run_tag()
    run_dir = artifact_dir / "training_runs" / spec["slug"] / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    model_source = Path(namespace[spec["model_key"]])
    history_source = Path(namespace[spec["history_key"]])
    step_source = artifact_dir / spec["step_trace"]
    episode_source = artifact_dir / spec["episode_trace"]

    copied = {
        "model_path": _copy_if_present(model_source, run_dir / "model.zip"),
        "history_path": _copy_if_present(history_source, run_dir / "eval_history.csv"),
        "step_trace": _copy_if_present(step_source, run_dir / "step_trace.csv"),
        "episode_trace": _copy_if_present(episode_source, run_dir / "episode_trace.csv"),
    }
    tensorboard_events = _copy_tensorboard_events(namespace, run_dir)

    expected_timesteps = float(namespace.get(spec["timesteps_key"], 0.0))
    max_timestep = _max_timestep(run_dir / "step_trace.csv")
    complete = bool(expected_timesteps > 0.0 and max_timestep >= complete_threshold * expected_timesteps)

    metadata: dict[str, Any] = {
        "label": spec["label"],
        "slug": spec["slug"],
        "task": task_name,
        "run_tag": run_tag,
        "run_dir": str(run_dir),
        "expected_timesteps": expected_timesteps,
        "max_timestep": max_timestep,
        "complete_threshold": complete_threshold,
        "complete": complete,
        "archived_at": datetime.now().isoformat(timespec="seconds"),
        "command": command or [],
        "tensorboard_events": tensorboard_events,
        "model_sha256": _sha256_file(run_dir / "model.zip"),
        **copied,
    }
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    if complete:
        (run_dir / "completed.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        latest_path = artifact_dir / "latest_training_runs.json"
        latest = _load_latest(latest_path)
        latest[spec["label"]] = metadata
        latest_path.write_text(json.dumps(latest, indent=2, default=str), encoding="utf-8")
    return metadata
