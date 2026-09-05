from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def add_env_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--traffic-model", choices=["force", "mtm"], default=None)
    parser.add_argument("--env-config-json", default=None, help="JSON object merged into ENV_CONFIG.")
    parser.add_argument("--env-config-file", type=Path, default=None, help="JSON file merged into ENV_CONFIG.")


def load_json_object(value: str | None, *, label: str) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must decode to a JSON object.")
    return loaded


def env_config_from_args(args: argparse.Namespace, base_config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    if getattr(args, "env_config_file", None):
        file_updates = json.loads(Path(args.env_config_file).read_text(encoding="utf-8"))
        if not isinstance(file_updates, dict):
            raise ValueError("--env-config-file must contain a JSON object.")
        deep_update(config, file_updates)
    deep_update(config, load_json_object(getattr(args, "env_config_json", None), label="--env-config-json"))
    traffic_model = getattr(args, "traffic_model", None)
    if traffic_model is not None:
        config["traffic_model"] = traffic_model
    return config


def active_traffic_model(config: dict[str, Any]) -> str:
    return str(config.get("traffic_model", "force")).strip().lower()
