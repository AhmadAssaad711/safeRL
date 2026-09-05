from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


TRAINING_LABEL_BY_VARIANT: dict[str, str] = {
    "ddpg": "DDPG without CBF",
    "ddpg-cbf": "DDPG-CBF reward",
    "guided-ddpg-cbf": "DDPG-CBF reward + loss",
}


@dataclass(frozen=True)
class LatestTrainingRun:
    """A completed immutable model selected from the training registry."""

    variant: str
    label: str
    run_dir: Path
    model_path: Path
    metadata: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def latest_completed_training(artifact_dir: Path, variant: str) -> LatestTrainingRun | None:
    """Resolve the latest completed archived training run for an RL variant."""
    label = TRAINING_LABEL_BY_VARIANT.get(variant)
    if label is None:
        return None

    latest = _load_json(Path(artifact_dir) / "latest_training_runs.json")
    metadata = latest.get(label)
    if not isinstance(metadata, dict) or not bool(metadata.get("complete", False)):
        return None

    run_dir_value = metadata.get("run_dir")
    model_path_value = metadata.get("model_path")
    if not run_dir_value or not model_path_value:
        return None

    run_dir = Path(str(run_dir_value))
    model_path = Path(str(model_path_value))
    if not run_dir.is_dir() or not model_path.is_file():
        return None

    return LatestTrainingRun(
        variant=variant,
        label=label,
        run_dir=run_dir,
        model_path=model_path,
        metadata=metadata,
    )


def build_evaluation_request(
    *,
    variant: str,
    model_path: Path,
    training_run: LatestTrainingRun | None,
    episodes: int,
    seed: int,
    device: str,
    traffic_model: str,
    env_config: dict[str, Any],
    cbf_config: dict[str, float] | None,
    evaluator_code_sha256: str,
    notebook_code_sha256: str,
) -> dict[str, Any]:
    """Describe every input that can change a deterministic final evaluation."""
    return {
        "schema_version": 1,
        "variant": str(variant),
        "model_path": str(Path(model_path).resolve()),
        "model_sha256": sha256_file(model_path),
        "training_run_dir": str(training_run.run_dir.resolve()) if training_run is not None else None,
        "episodes": int(episodes),
        "seed": int(seed),
        "device": str(device),
        "traffic_model": str(traffic_model),
        "env_config": env_config,
        "cbf_config": cbf_config or {},
        "evaluator_code_sha256": str(evaluator_code_sha256),
        "notebook_code_sha256": str(notebook_code_sha256),
    }


def evaluation_cache_paths(
    *,
    artifact_dir: Path,
    model_path: Path,
    training_run: LatestTrainingRun | None,
    evaluation_fingerprint: str,
) -> tuple[Path, Path]:
    """Return short Windows-safe paths for a cached evaluation and manifest."""
    identity = (
        str(training_run.run_dir.resolve())
        if training_run is not None
        else str(Path(model_path).resolve())
    )
    run_id = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]
    cache_dir = Path(artifact_dir).parent / "eval" / run_id / evaluation_fingerprint[:20]
    return cache_dir / "metrics.csv", cache_dir / "manifest.json"


def load_matching_evaluation(
    *,
    metrics_path: Path,
    manifest_path: Path,
    evaluation_fingerprint: str,
    model_sha256: str,
) -> dict[str, Any] | None:
    """Return a manifest only when its metrics correspond to the same model and protocol."""
    if not metrics_path.is_file() or not manifest_path.is_file():
        return None
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("evaluation_fingerprint") != evaluation_fingerprint
        or manifest.get("model_sha256") != model_sha256
        or manifest.get("metrics_path") != str(metrics_path)
        or not bool(manifest.get("complete", False))
    ):
        return None
    return manifest


def write_evaluation_manifest(
    *,
    manifest_path: Path,
    metrics_path: Path,
    request: dict[str, Any],
    evaluation_fingerprint: str,
) -> dict[str, Any]:
    """Atomically record a completed evaluation after its metrics have been saved."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_fingerprint": evaluation_fingerprint,
        "model_sha256": request["model_sha256"],
        "model_path": request.get("model_path"),
        "training_run_dir": request.get("training_run_dir"),
        "metrics_path": str(metrics_path),
        "request": request,
    }
    temporary = manifest_path.with_name(f"{manifest_path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest


def sync_metrics_to_requested_output(source: Path, output: Path) -> None:
    """Refresh the notebook's conventional CSV path from the immutable cache."""
    source = Path(source)
    output = Path(output)
    if source.resolve() == output.resolve():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, output)
