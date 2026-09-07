"""Self-contained artifact layout for new research runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


MANIFEST_SCHEMA_VERSION = 1
REQUIRED_MANIFEST_KEYS = (
    "algorithm",
    "environment_config",
    "reward_config",
    "cbf_config",
    "observation_definition",
    "seed",
    "frequencies",
    "training_steps",
    "git_commit",
    "model_path",
    "evaluation_protocol",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(manifest)).encode("utf-8")).hexdigest()


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ValueError(f"manifest is missing required fields: {', '.join(missing)}")
    schema_version = int(manifest.get("schema_version", MANIFEST_SCHEMA_VERSION))
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema_version={schema_version}")


def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON beside the final path and publish it atomically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


@dataclass(frozen=True)
class RunDirectory:
    """Paths for one self-contained new-research run."""

    root: Path

    @classmethod
    def create(cls, output_root: Path, run_name: str) -> "RunDirectory":
        root = Path(output_root) / run_name
        if root.exists():
            raise FileExistsError(f"refusing to reuse existing run directory: {root}")
        root.mkdir(parents=True, exist_ok=False)
        (root / "evaluation").mkdir()
        return cls(root=root)

    @property
    def model_path(self) -> Path:
        return self.root / "model.zip"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def training_metrics_path(self) -> Path:
        return self.root / "training_metrics.csv"

    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation"

    @property
    def notes_path(self) -> Path:
        return self.root / "notes.md"

    def initialize(
        self,
        *,
        config: Mapping[str, Any],
        manifest: Mapping[str, Any],
        notes: str = "",
    ) -> None:
        validate_manifest(manifest)
        write_json_atomic(self.config_path, dict(config))
        write_json_atomic(self.manifest_path, dict(manifest))
        self.notes_path.write_text(notes.rstrip() + "\n", encoding="utf-8")
