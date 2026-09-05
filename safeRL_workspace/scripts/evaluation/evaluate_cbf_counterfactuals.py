"""Post-training fixed-state and occupancy analysis for the 2x2 CBF study.

The training runner deliberately stays focused on training and protocol-level
rollouts.  This module consumes its ``models.csv`` and ``run_config.json`` and
answers two separate questions:

1. Did a treatment change the states visited by the policy?
2. Did it change the raw action selected at exactly the same state?

Importing this file has no environment, notebook, model-loading, or plotting
side effects.  The small numerical helpers are intentionally usable in fast
unit tests with synthetic data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import scripts.training.run_cbf_filter_ablation as pipeline  # noqa: E402


ANALYSIS_SCHEMA_VERSION = 1
EXPECTED_PIPELINE_SCHEMA_VERSION = 4
EXPECTED_FACTORIAL_VARIANTS: dict[tuple[bool, bool], str] = {
    (False, False): "b_filtered",
    (True, False): "c_reward",
    (False, True): "d_loss",
    (True, True): "e_reward_actor",
}
STRATA = ("normal", "near_boundary", "intervention", "dense", "overtaking")
# Allocate uncommon states before overlapping, common states.  Output ordering
# still follows STRATA.
STRATIFICATION_PRIORITY = ("overtaking", "intervention", "near_boundary", "dense", "normal")


def _finite_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _round_finite(value: Any, digits: int = 9) -> Any:
    """Canonicalize nested numeric state data for deterministic hashing."""

    if isinstance(value, np.ndarray):
        return _round_finite(value.tolist(), digits)
    if isinstance(value, np.generic):
        return _round_finite(value.item(), digits)
    if isinstance(value, Mapping):
        return {
            str(key): _round_finite(item, digits)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_round_finite(item, digits) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        return round(float(value), digits)
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return str(value)
    return round(scalar, digits) if np.isfinite(scalar) else str(scalar)


def stable_state_hash(
    observation: Any,
    ego: Mapping[str, Any] | None = None,
    neighbors: Sequence[Mapping[str, Any]] | None = None,
    road_width: float | None = None,
    *,
    digits: int = 9,
) -> str:
    """Hash the exact actor input plus physical state used by the CBF.

    Dictionary ordering and numpy scalar dtypes do not affect the result.  The
    physical state is included because it is required to replay the filter,
    while the observation is included because it is the exact actor input.
    """

    payload = {
        "observation": np.asarray(observation).tolist(),
        "ego": dict(ego or {}),
        "neighbors": [dict(item) for item in (neighbors or [])],
        "road_width": road_width,
    }
    canonical = json.dumps(
        _round_finite(payload, digits),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TypedConstraintSystem:
    """Linear action set ``rows @ action <= bounds`` with semantic row types."""

    rows: np.ndarray
    bounds: np.ndarray
    constraint_types: tuple[str, ...]

    def __post_init__(self) -> None:
        rows = np.asarray(self.rows, dtype=float)
        bounds = np.asarray(self.bounds, dtype=float).reshape(-1)
        if rows.size == 0:
            rows = np.zeros((0, 2), dtype=float)
        rows = rows.reshape(-1, 2)
        if rows.shape[0] != bounds.size:
            raise ValueError("rows and bounds must contain the same number of constraints")
        if len(self.constraint_types) != bounds.size:
            raise ValueError("constraint_types must contain one label per row")
        if not np.all(np.isfinite(rows)) or not np.all(np.isfinite(bounds)):
            raise ValueError("constraint rows and bounds must be finite")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "constraint_types", tuple(map(str, self.constraint_types)))


def make_typed_constraint_system(
    rows: Any,
    bounds: Any,
    constraint_types: Sequence[str] | None = None,
) -> TypedConstraintSystem:
    rows_array = np.asarray(rows, dtype=float)
    if rows_array.size == 0:
        rows_array = np.zeros((0, 2), dtype=float)
    rows_array = rows_array.reshape(-1, 2)
    if constraint_types is None:
        constraint_types = tuple(f"constraint_{index}" for index in range(rows_array.shape[0]))
    return TypedConstraintSystem(rows_array, np.asarray(bounds, dtype=float), tuple(constraint_types))


def typed_feasible_mask(
    ax: Any,
    ay: Any,
    system: TypedConstraintSystem,
    *,
    tolerance: float = 1e-7,
) -> dict[str, np.ndarray]:
    """Return overall and per-type feasibility masks on a broadcast action grid."""

    ax_array, ay_array = np.broadcast_arrays(np.asarray(ax, dtype=float), np.asarray(ay, dtype=float))
    actions = np.stack([ax_array.ravel(), ay_array.ravel()], axis=1)
    shape = ax_array.shape
    if system.rows.shape[0] == 0:
        return {"all": np.ones(shape, dtype=bool)}
    satisfied = actions @ system.rows.T <= system.bounds.reshape(1, -1) + float(tolerance)
    result: dict[str, np.ndarray] = {"all": np.all(satisfied, axis=1).reshape(shape)}
    for constraint_type in dict.fromkeys(system.constraint_types):
        indices = [index for index, label in enumerate(system.constraint_types) if label == constraint_type]
        result[str(constraint_type)] = np.all(satisfied[:, indices], axis=1).reshape(shape)
    return result


def active_constraint_indices(
    action: Any,
    system: TypedConstraintSystem,
    *,
    active_tolerance: float = 1e-3,
) -> np.ndarray:
    """Indices of constraints tight at ``action`` (not merely violated by raw action)."""

    if system.rows.shape[0] == 0:
        return np.zeros(0, dtype=int)
    values = system.rows @ np.asarray(action, dtype=float).reshape(2) - system.bounds
    return np.flatnonzero(values >= -float(active_tolerance)).astype(int)


def normal_tangent_decomposition(
    delta: Any,
    active_rows_scaled: Any,
    *,
    rcond: float = 1e-10,
) -> dict[str, np.ndarray | float | int]:
    """Orthogonally split a scaled correction into active-normal/tangent parts.

    Multiple or rank-deficient active constraints are handled with the Moore-
    Penrose pseudoinverse.  Scaling must be applied to both ``delta`` and the
    action-space constraint rows before calling this function.
    """

    vector = np.asarray(delta, dtype=float).reshape(2)
    rows = np.asarray(active_rows_scaled, dtype=float)
    if rows.size == 0:
        rows = np.zeros((0, 2), dtype=float)
    rows = rows.reshape(-1, 2)
    if rows.shape[0] == 0:
        normal_projector = np.zeros((2, 2), dtype=float)
        rank = 0
    else:
        normal_projector = rows.T @ np.linalg.pinv(rows @ rows.T, rcond=rcond) @ rows
        # Symmetrize to suppress insignificant numerical asymmetry.
        normal_projector = 0.5 * (normal_projector + normal_projector.T)
        rank = int(np.linalg.matrix_rank(rows, tol=rcond))
    tangent_projector = np.eye(2, dtype=float) - normal_projector
    normal = normal_projector @ vector
    tangent = tangent_projector @ vector
    return {
        "normal": normal,
        "tangent": tangent,
        "normal_norm": float(np.linalg.norm(normal)),
        "tangent_norm": float(np.linalg.norm(tangent)),
        "normal_projector": normal_projector,
        "tangent_projector": tangent_projector,
        "rank": rank,
        "reconstruction_error": float(np.linalg.norm(normal + tangent - vector)),
        "orthogonality_error": float(abs(normal @ tangent)),
    }


def time_to_contact(relative_position: Any, relative_velocity: Any, radius: float) -> float:
    """First positive time at which a constant-velocity disc pair touches."""

    position = np.asarray(relative_position, dtype=float).reshape(2)
    velocity = np.asarray(relative_velocity, dtype=float).reshape(2)
    radius = max(float(radius), 0.0)
    c = float(position @ position - radius * radius)
    if c <= 0.0:
        return 0.0
    a = float(velocity @ velocity)
    if a <= 1e-12:
        return np.inf
    b = float(2.0 * position @ velocity)
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return np.inf
    root = math.sqrt(max(discriminant, 0.0))
    candidates = [value for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)) if value >= 0.0]
    return float(min(candidates)) if candidates else np.inf


def _candidate_memberships(
    candidate: Mapping[str, Any],
    *,
    near_boundary_margin: float,
    dense_threshold: float,
) -> tuple[str, ...]:
    h_min = _finite_float(candidate.get("h_min"), default=np.inf)
    intervention = bool(candidate.get("intervention", False))
    overtaking = bool(candidate.get("overtaking", False))
    density = _finite_float(candidate.get("traffic_density_per_km"), default=-np.inf)
    memberships: list[str] = []
    if h_min <= float(near_boundary_margin):
        memberships.append("near_boundary")
    if intervention:
        memberships.append("intervention")
    if density >= float(dense_threshold):
        memberships.append("dense")
    if overtaking:
        memberships.append("overtaking")
    if h_min > float(near_boundary_margin) and not intervention and not overtaking:
        memberships.append("normal")
    return tuple(memberships)


def stratify_state_bank(
    candidates: Sequence[Mapping[str, Any]],
    states_per_category: int,
    *,
    seed: int,
    near_boundary_margin: float,
    dense_threshold: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic, unique, category-balanced common state bank."""

    if int(states_per_category) <= 0:
        raise ValueError("states_per_category must be positive")
    prepared: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    densities: list[float] = []
    for raw in candidates:
        item = dict(raw)
        state_hash = str(item.get("state_hash", ""))
        if not state_hash:
            state_hash = stable_state_hash(
                item.get("observation", []), item.get("ego", {}), item.get("neighbors", []), item.get("road_width")
            )
        # Identical physical/observation states can arise after deterministic
        # reset; retain the first in stable collector order.
        if state_hash in seen_hashes:
            continue
        seen_hashes.add(state_hash)
        item["state_hash"] = state_hash
        density = _finite_float(item.get("traffic_density_per_km"))
        if np.isfinite(density):
            densities.append(density)
        prepared.append(item)
    if not prepared:
        raise ValueError("cannot stratify an empty candidate pool")

    if dense_threshold is None:
        dense_threshold = float(np.quantile(densities, 0.75)) if densities else np.inf
    for item in prepared:
        item["categories"] = _candidate_memberships(
            item,
            near_boundary_margin=float(near_boundary_margin),
            dense_threshold=float(dense_threshold),
        )

    selected: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    availability: dict[str, int] = {}
    selection_counts: dict[str, int] = {}
    for priority_index, category in enumerate(STRATIFICATION_PRIORITY):
        eligible = [item for item in prepared if category in item["categories"]]
        availability[category] = len(eligible)
        eligible.sort(key=lambda item: item["state_hash"])
        rng = np.random.default_rng(int(seed) + 104729 * priority_index)
        order = rng.permutation(len(eligible)) if eligible else np.zeros(0, dtype=int)
        count = 0
        for index in order:
            item = eligible[int(index)]
            if item["state_hash"] in selected_hashes:
                continue
            chosen = dict(item)
            chosen["stratum"] = category
            chosen["categories"] = tuple(item["categories"])
            selected.append(chosen)
            selected_hashes.add(item["state_hash"])
            count += 1
            if count >= int(states_per_category):
                break
        selection_counts[category] = count

    order_index = {name: index for index, name in enumerate(STRATA)}
    selected.sort(key=lambda item: (order_index[item["stratum"]], item["state_hash"]))
    for bank_index, item in enumerate(selected):
        item["bank_index"] = int(bank_index)
    metadata = {
        "candidate_count": len(candidates),
        "unique_candidate_count": len(prepared),
        "bank_count": len(selected),
        "states_per_category_requested": int(states_per_category),
        "near_boundary_margin": float(near_boundary_margin),
        "dense_threshold_per_km": float(dense_threshold),
        "availability": {name: int(availability.get(name, 0)) for name in STRATA},
        "selection_counts": {name: int(selection_counts.get(name, 0)) for name in STRATA},
        "selection_shortfall": {
            name: int(max(int(states_per_category) - selection_counts.get(name, 0), 0)) for name in STRATA
        },
        "missing_strata": [name for name in STRATA if selection_counts.get(name, 0) == 0],
        "complete_requested_coverage": bool(
            all(selection_counts.get(name, 0) >= int(states_per_category) for name in STRATA)
        ),
        "bank_hash": hashlib.sha256(
            "\n".join(item["state_hash"] for item in selected).encode("utf-8")
        ).hexdigest(),
    }
    return selected, metadata


DEFAULT_FACTORIAL_CONTRASTS: dict[str, dict[tuple[bool, bool], float]] = {
    "reward_main_effect": {(False, False): -0.5, (True, False): 0.5, (False, True): -0.5, (True, True): 0.5},
    "actor_loss_main_effect": {(False, False): -0.5, (True, False): -0.5, (False, True): 0.5, (True, True): 0.5},
    "reward_actor_interaction": {(False, False): 1.0, (True, False): -1.0, (False, True): -1.0, (True, True): 1.0},
}


def compute_factorial_contrasts(
    summary: pd.DataFrame,
    metric_columns: Sequence[str],
    *,
    variant_map: Mapping[tuple[bool, bool], str] | None = None,
    group_columns: Sequence[str] = ("training_seed",),
) -> pd.DataFrame:
    """Compute paired 2x2 contrasts, returning one long-form row per metric."""

    variant_map = dict(variant_map or EXPECTED_FACTORIAL_VARIANTS)
    required = set(group_columns) | {"variant"} | set(metric_columns)
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"summary is missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = list(group_columns)
    if len(group_columns) == 1:
        grouper = str(group_columns[0])
    for group_key, group in summary.groupby(grouper, dropna=False, sort=True):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        group_values = dict(zip(group_columns, keys))
        by_variant = {
            str(row["variant"]): row
            for _, row in group.drop_duplicates(subset=["variant"], keep="last").iterrows()
        }
        missing_variants = [variant for variant in variant_map.values() if variant not in by_variant]
        if missing_variants:
            raise ValueError(f"incomplete factorial cell for {group_values}: {missing_variants}")
        for effect, coefficients in DEFAULT_FACTORIAL_CONTRASTS.items():
            formula = " ".join(
                f"{coefficient:+g}*{variant_map[cell]}" for cell, coefficient in coefficients.items()
            )
            for metric in metric_columns:
                values = [
                    float(coefficient) * _finite_float(by_variant[variant_map[cell]][metric])
                    for cell, coefficient in coefficients.items()
                ]
                estimate = float(sum(values)) if all(np.isfinite(values)) else np.nan
                rows.append(
                    {
                        **group_values,
                        "effect": effect,
                        "metric": str(metric),
                        "estimate": estimate,
                        "formula": formula,
                    }
                )
    return pd.DataFrame(rows)


def summarize_factorial_contrasts(effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reserved = {"training_seed", "effect", "metric", "estimate", "formula"}
    context_columns = [column for column in effects.columns if column not in reserved]
    grouping = context_columns + ["effect", "metric"]
    for group_key, group in effects.groupby(grouping, sort=True, dropna=False):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        context = dict(zip(grouping, keys))
        values = pd.to_numeric(group["estimate"], errors="coerce").dropna()
        rows.append(
            {
                **context,
                "training_seeds": int(values.size),
                "seed_mean": float(values.mean()) if values.size else np.nan,
                "seed_sd": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                "seed_se": float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else np.nan,
                "formula": str(group["formula"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True, help="Directory containing models.csv and run_config.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--collector-scenarios", type=int, default=2)
    parser.add_argument("--collector-steps", type=int, default=400)
    parser.add_argument("--states-per-category", type=int, default=80)
    parser.add_argument("--bank-seed-start", type=int, default=None)
    parser.add_argument("--near-boundary-margin", type=float, default=1.0)
    parser.add_argument("--dense-quantile", type=float, default=0.75)
    parser.add_argument("--neighbor-range", type=float, default=None)
    parser.add_argument("--ttc-cap", type=float, default=30.0)
    parser.add_argument("--active-tolerance", type=float, default=1e-3)
    parser.add_argument("--selected-states", type=int, default=5)
    parser.add_argument("--contour-grid", type=int, default=61)
    parser.add_argument("--skip-contours", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args(argv)


def _validate_study(study_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    study_dir = study_dir.resolve()
    config_path = study_dir / "run_config.json"
    models_path = study_dir / "models.csv"
    if not config_path.is_file() or not models_path.is_file():
        raise FileNotFoundError(f"{study_dir} must contain run_config.json and models.csv")
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    models = pd.read_csv(models_path)
    required_columns = {"training_seed", "variant", "model_path"}
    if not required_columns.issubset(models.columns):
        raise ValueError(f"models.csv is missing columns: {sorted(required_columns - set(models.columns))}")
    schema = int(run_config.get("schema_version", -1))
    if schema != EXPECTED_PIPELINE_SCHEMA_VERSION:
        raise ValueError(
            f"expected pipeline schema {EXPECTED_PIPELINE_SCHEMA_VERSION}, found {schema}; "
            "regenerate the study or update this analyzer deliberately"
        )
    runtime_map = dict(getattr(pipeline, "FACTORIAL_VARIANTS", {}))
    if runtime_map != EXPECTED_FACTORIAL_VARIANTS:
        raise RuntimeError(f"runner factorial registry changed: {runtime_map!r}")
    config_map = {
        (bool(reward_on), bool(loss_on)): str(variant)
        for reward_on in (False, True)
        for loss_on in (False, True)
        for key, variant in run_config.get("factorial_variants", {}).items()
        if key == f"reward_{int(reward_on)}_actor_loss_{int(loss_on)}"
    }
    if config_map and config_map != EXPECTED_FACTORIAL_VARIANTS:
        raise ValueError(f"run_config factorial registry differs from schema 4: {config_map!r}")

    learned = models[models["variant"].isin(EXPECTED_FACTORIAL_VARIANTS.values())].copy()
    learned["training_seed"] = learned["training_seed"].astype(int)
    learned["variant"] = learned["variant"].astype(str)
    duplicates = learned.duplicated(["training_seed", "variant"], keep=False)
    if duplicates.any():
        raise ValueError(
            "models.csv contains duplicate factorial seed/variant rows: "
            f"{learned.loc[duplicates, ['training_seed', 'variant']].to_dict('records')}"
        )
    expected_variants = set(EXPECTED_FACTORIAL_VARIANTS.values())
    for seed, group in learned.groupby("training_seed", sort=True):
        missing = sorted(expected_variants - set(group["variant"]))
        if missing:
            raise ValueError(f"training seed {seed} lacks factorial models: {missing}")
    if learned.empty:
        raise ValueError("models.csv contains no schema-4 factorial variants")
    return run_config, learned.sort_values(["training_seed", "variant"]).reset_index(drop=True)


def _resolve_model_path(raw_path: Any, study_dir: Path, project_root: Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    candidates = [path] if path.is_absolute() else [study_dir / path, project_root / path, path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
        if candidate.suffix != ".zip" and candidate.with_suffix(".zip").is_file():
            return candidate.with_suffix(".zip").resolve()
    raise FileNotFoundError(f"model checkpoint not found for models.csv path {raw_path!r}")


def _bootstrap_runtime(
    project_root: Path,
    run_config: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    namespace = pipeline.bootstrap_notebook_namespace(project_root)
    pipeline.exec_required_notebook_cells(notebook_path, namespace)
    namespace["DEVICE"] = str(device)
    namespace["CBF_K0"] = float(run_config["k0"])
    namespace["CBF_K1"] = float(run_config["k1"])
    namespace["CBF_EPS_SIDE"] = float(run_config["eps_side"])
    namespace["CBF_FILTER_REWARD_LAMBDA"] = 0.0
    namespace["GUIDED_CBF_ENABLE_PROJECTION_REPORTING"] = True
    pipeline.install_minimal_guided_cbf(namespace)
    installer = namespace.get("install_cbf_projection_reporting")
    if callable(installer):
        installer()
    pipeline.install_correction_reward_env(namespace)
    return namespace


def _evaluation_args(run_config: Mapping[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        correction_epsilon=float(run_config["correction_epsilon_normalized"]),
        k0=float(run_config["k0"]),
        k1=float(run_config["k1"]),
        eps_side=float(run_config["eps_side"]),
    )


def filter_physical_bounds(
    namespace: Mapping[str, Any], run_config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the CBF filter box, falling back to the registered environment box."""

    env_low, env_high = pipeline.physical_bounds(dict(run_config["env_config"]))
    ax_bounds = namespace.get("CBF_AX_BOUNDS", (float(env_low[0]), float(env_high[0])))
    ay_bounds = namespace.get("CBF_AY_BOUNDS", (float(env_low[1]), float(env_high[1])))
    low = np.asarray([float(ax_bounds[0]), float(ay_bounds[0])], dtype=np.float32)
    high = np.asarray([float(ax_bounds[1]), float(ay_bounds[1])], dtype=np.float32)
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(high <= low):
        raise ValueError(f"invalid configured CBF action bounds: low={low}, high={high}")
    return low, high


def _neighbor_constraint_type(ego: Mapping[str, Any], neighbor: Mapping[str, Any]) -> str:
    signed_dx = float(neighbor.get("signed_dx", float(neighbor.get("x", 0.0)) - float(ego.get("x", 0.0))))
    lateral_offset = abs(float(neighbor.get("y", 0.0)) - float(ego.get("y", 0.0)))
    alongside = 0.5 * (float(ego.get("length", 0.0)) + float(neighbor.get("length", 0.0)))
    if abs(signed_dx) <= max(alongside, 1e-6) and lateral_offset > 0.5:
        return "neighbor_side"
    return "neighbor_front" if signed_dx >= 0.0 else "neighbor_rear"


def _typed_system_from_filter_info(info: Mapping[str, Any]) -> TypedConstraintSystem:
    rows = np.asarray(info.get("constraint_rows_physical", np.zeros((0, 2))), dtype=float).reshape(-1, 2)
    bounds = np.asarray(info.get("constraint_bounds_physical", np.zeros(rows.shape[0])), dtype=float).reshape(-1)
    labels = tuple(map(str, info.get("constraint_types", ())))
    if len(labels) != rows.shape[0]:
        labels = tuple(f"constraint_{index}" for index in range(rows.shape[0]))
    return make_typed_constraint_system(rows, bounds, labels)


def _filter_action(
    namespace: Mapping[str, Any],
    raw_action: Any,
    state: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    low, high = filter_physical_bounds(namespace, run_config)
    safe, info = namespace["cbf_filter_2d"](
        np.asarray(raw_action, dtype=np.float32).reshape(2),
        dict(state["ego"]),
        [dict(item) for item in state["neighbors"]],
        float(state["road_width"]),
        ax_bounds=(float(low[0]), float(high[0])),
        ay_bounds=(float(low[1]), float(high[1])),
        eps_side=float(run_config["eps_side"]),
        k0=float(run_config["k0"]),
        k1=float(run_config["k1"]),
        max_neighbor_constraints=namespace.get("CBF_MAX_NEIGHBOR_CONSTRAINTS"),
    )
    return np.asarray(safe, dtype=np.float32).reshape(2), dict(info)


def occupancy_metrics(
    namespace: Mapping[str, Any],
    ego: Mapping[str, Any],
    neighbors: Sequence[Mapping[str, Any]],
    road_width: float,
    *,
    neighbor_range: float,
    eps_side: float,
    k0: float,
    k1: float,
    ttc_cap: float,
) -> dict[str, float | str]:
    """Compute state occupancy measures before executing an action."""

    barrier_candidates: list[tuple[float, float, str]] = []
    spacings: list[float] = []
    ttcs: list[float] = []
    for neighbor in neighbors:
        other_acc = np.asarray(
            [float(neighbor.get("ax", 0.0)), float(neighbor.get("ay", 0.0))], dtype=float
        )
        row, _, h_value, center_distance, required_distance = namespace["pairwise_hocbf_constraint"](
            ego,
            neighbor,
            eps_side=float(eps_side),
            k0=float(k0),
            k1=float(k1),
            other_acc=other_acc,
        )
        dx, dy, dvx, dvy = namespace["pairwise_relative_state"](ego, neighbor)
        h_dot = float(np.asarray(row, dtype=float).reshape(2) @ np.asarray([dvx, dvy], dtype=float))
        barrier_candidates.append((float(h_value), h_dot, _neighbor_constraint_type(ego, neighbor)))
        spacings.append(float(center_distance) - float(required_distance))
        ttcs.append(time_to_contact([dx, dy], [dvx, dvy], float(required_distance)))

    ego_y = float(ego["y"])
    ego_vy = float(ego["vy"])
    half_width = 0.5 * float(ego["width"])
    barrier_candidates.extend(
        [
            (ego_y - half_width, ego_vy, "road_left"),
            (float(road_width) - half_width - ego_y, -ego_vy, "road_right"),
        ]
    )
    h_min, h_dot, h_type = min(barrier_candidates, key=lambda value: value[0])
    finite_ttcs = [value for value in ttcs if np.isfinite(value)]
    min_ttc = min(finite_ttcs) if finite_ttcs else float(ttc_cap)
    min_ttc = min(max(float(min_ttc), 0.0), float(ttc_cap))
    spacing = min(spacings) if spacings else np.nan
    sensed_km = max(2.0 * float(neighbor_range) / 1000.0, 1e-9)
    return {
        "h_min": float(h_min),
        "h_dot": float(h_dot),
        "h_min_constraint_type": str(h_type),
        "ttc_s": float(min_ttc),
        "ttc_right_censored": int(not finite_ttcs or min(finite_ttcs) > float(ttc_cap)),
        "vehicle_spacing_m": float(spacing),
        "traffic_density_per_km": float(len(neighbors) / sensed_km),
        "neighbor_count": int(len(neighbors)),
    }


def _is_overtaking_state(ego: Mapping[str, Any], neighbors: Sequence[Mapping[str, Any]], step_info: Mapping[str, Any]) -> bool:
    event = max(
        _finite_float(step_info.get("kpi_overtakes_step"), 0.0),
        _finite_float(step_info.get("karalakou_overtakes"), 0.0),
        _finite_float(step_info.get("overtakes"), 0.0),
    )
    if event > 0.0:
        return True
    # Also retain the maneuver immediately around a pass, not only the single
    # transition on which a signed longitudinal ordering flips.
    if abs(float(ego.get("vy", 0.0))) < 0.20:
        return False
    for neighbor in neighbors:
        signed_dx = float(neighbor.get("signed_dx", float(neighbor.get("x", 0.0)) - float(ego.get("x", 0.0))))
        lateral = abs(float(neighbor.get("y", 0.0)) - float(ego.get("y", 0.0)))
        if abs(signed_dx) <= 20.0 and lateral <= 8.0:
            return True
    return False


def collect_shielded_state_candidates(
    namespace: Mapping[str, Any],
    models: Mapping[tuple[int, str], Any],
    run_config: Mapping[str, Any],
    *,
    scenario_count: int,
    steps_per_scenario: int,
    scenario_seed_start: int,
    neighbor_range: float,
    ttc_cap: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Pool deterministic shielded rollouts from every factorial actor/seed."""

    env_config = dict(run_config["env_config"])
    reward_config = dict(run_config["base_reward_config"])
    eval_args = _evaluation_args(run_config)
    correction_epsilon = float(run_config["correction_epsilon_normalized"])
    candidates: list[dict[str, Any]] = []
    occupancy_rows: list[dict[str, Any]] = []
    for (training_seed, variant), model in sorted(models.items()):
        for scenario_index in range(int(scenario_count)):
            scenario_seed = int(scenario_seed_start) + int(scenario_index)
            env = pipeline.make_evaluation_env(
                namespace,
                mode="cbf",
                scenario_seed=scenario_seed,
                env_config=env_config,
                reward_config=reward_config,
                args=eval_args,
            )
            try:
                obs, _ = env.reset(seed=scenario_seed)
                episode_index = 0
                rng = np.random.default_rng(
                    int(scenario_seed) + 1009 * int(training_seed) + 9176 * list(EXPECTED_FACTORIAL_VARIANTS.values()).index(variant)
                )
                for scenario_step in range(int(steps_per_scenario)):
                    ego = dict(namespace["get_ego_state"](env))
                    neighbors = [
                        dict(item)
                        for item in namespace["get_neighbor_states"](env, neighbor_range=float(neighbor_range))
                    ]
                    road_width = float(env.unwrapped.config["road_width"])
                    raw_action = pipeline.policy_action_physical(
                        model=model,
                        obs=np.asarray(obs),
                        env_config=env_config,
                        rng=rng,
                    )
                    state = {
                        "observation": np.asarray(obs, dtype=np.float32).copy(),
                        "ego": ego,
                        "neighbors": neighbors,
                        "road_width": road_width,
                    }
                    safe_action, filter_info = _filter_action(namespace, raw_action, state, run_config)
                    low, high = filter_physical_bounds(namespace, run_config)
                    half_range = np.maximum(0.5 * (high - low), 1e-6)
                    delta_box = (safe_action - raw_action) / half_range
                    correction_box_norm = float(np.linalg.norm(delta_box))
                    metrics = occupancy_metrics(
                        namespace,
                        ego,
                        neighbors,
                        road_width,
                        neighbor_range=float(neighbor_range),
                        eps_side=float(run_config["eps_side"]),
                        k0=float(run_config["k0"]),
                        k1=float(run_config["k1"]),
                        ttc_cap=float(ttc_cap),
                    )
                    next_obs, _, terminated, truncated, step_info = env.step(raw_action)
                    overtaking = _is_overtaking_state(ego, neighbors, step_info)
                    state_hash = stable_state_hash(obs, ego, neighbors, road_width)
                    common = {
                        "source_training_seed": int(training_seed),
                        "source_variant": str(variant),
                        "scenario_index": int(scenario_index),
                        "scenario_seed": int(scenario_seed),
                        "episode_index": int(episode_index),
                        "scenario_step": int(scenario_step),
                        "state_hash": state_hash,
                        **metrics,
                        "overtaking": bool(overtaking),
                        "raw_action_ax": float(raw_action[0]),
                        "raw_action_ay": float(raw_action[1]),
                        "safe_action_ax": float(safe_action[0]),
                        "safe_action_ay": float(safe_action[1]),
                        "correction_box_norm": correction_box_norm,
                        "intervention": bool(correction_box_norm > correction_epsilon),
                        "qp_success": bool(filter_info.get("qp_success", False)),
                        "fallback_used": bool(filter_info.get("fallback_used", False)),
                    }
                    candidate = {**common, **state}
                    candidates.append(candidate)
                    occupancy_rows.append(common)
                    obs = next_obs
                    if terminated or truncated:
                        episode_index += 1
                        reset_seed = int(scenario_seed) + 100_003 * episode_index
                        obs, _ = env.reset(seed=reset_seed)
            finally:
                env.close()
    return candidates, pd.DataFrame(occupancy_rows)


def _state_bank_records(bank: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in bank:
        records.append(
            {
                "bank_index": int(item["bank_index"]),
                "state_hash": str(item["state_hash"]),
                "stratum": str(item["stratum"]),
                "categories": "|".join(map(str, item["categories"])),
                "source_training_seed": int(item["source_training_seed"]),
                "source_variant": str(item["source_variant"]),
                "scenario_seed": int(item["scenario_seed"]),
                "scenario_index": int(item["scenario_index"]),
                "scenario_step": int(item["scenario_step"]),
                "h_min": float(item["h_min"]),
                "h_dot": float(item["h_dot"]),
                "ttc_s": float(item["ttc_s"]),
                "vehicle_spacing_m": _finite_float(item["vehicle_spacing_m"]),
                "traffic_density_per_km": float(item["traffic_density_per_km"]),
                "source_intervention": bool(item["intervention"]),
                "overtaking": bool(item["overtaking"]),
                "neighbor_count": int(item["neighbor_count"]),
            }
        )
    return records


def write_state_bank(bank: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    pd.DataFrame(_state_bank_records(bank)).to_csv(output_dir / "state_bank.csv", index=False)
    observations = np.stack([np.asarray(item["observation"], dtype=np.float32) for item in bank])
    np.savez_compressed(
        output_dir / "state_bank_observations.npz",
        observations=observations,
        state_hashes=np.asarray([item["state_hash"] for item in bank]),
    )
    with (output_dir / "state_bank.jsonl").open("w", encoding="utf-8") as handle:
        for item in bank:
            payload = {
                "bank_index": int(item["bank_index"]),
                "state_hash": str(item["state_hash"]),
                "stratum": str(item["stratum"]),
                "categories": list(item["categories"]),
                "observation": np.asarray(item["observation"], dtype=float).tolist(),
                "ego": _round_finite(item["ego"]),
                "neighbors": _round_finite(item["neighbors"]),
                "road_width": float(item["road_width"]),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _active_description(
    filter_info: Mapping[str, Any],
    delta_phys: np.ndarray,
    delta_box: np.ndarray,
    *,
    correction_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], str]:
    active_physical = np.asarray(
        filter_info.get("active_constraint_rows_physical", np.zeros((0, 2))), dtype=float
    ).reshape(-1, 2)
    active_scaled = np.asarray(
        filter_info.get("active_constraint_rows_scaled", np.zeros((0, 2))), dtype=float
    ).reshape(-1, 2)
    active_types = tuple(map(str, filter_info.get("active_constraint_types", ())))
    if len(active_types) != active_physical.shape[0]:
        active_types = tuple(f"active_{index}" for index in range(active_physical.shape[0]))
    basis_source = "reported_positive_kkt_constraints"
    if active_physical.shape[0] == 0 and float(np.linalg.norm(delta_box)) > float(correction_epsilon):
        # Infeasible/soft fallback projections do not have a valid KKT active
        # set.  Using the actual correction direction provides a conservative,
        # explicit normal basis instead of silently calling it tangential.
        active_physical = np.asarray(delta_phys, dtype=float).reshape(1, 2)
        active_scaled = np.asarray(delta_box, dtype=float).reshape(1, 2)
        active_types = ("fallback_projection_direction",)
        basis_source = "correction_direction_fallback"
    return active_physical, active_scaled, active_types, basis_source


def evaluate_common_state_bank(
    namespace: Mapping[str, Any],
    models: Mapping[tuple[int, str], Any],
    bank: Sequence[Mapping[str, Any]],
    run_config: Mapping[str, Any],
    *,
    active_tolerance: float,
) -> pd.DataFrame:
    """Pass exactly the same stored observations/physical states through all actors."""

    env_config = dict(run_config["env_config"])
    low, high = filter_physical_bounds(namespace, run_config)
    half_range = np.maximum(0.5 * (high - low), 1e-6).astype(float)
    correction_epsilon = float(run_config["correction_epsilon_normalized"])
    rows: list[dict[str, Any]] = []
    for (training_seed, variant), model in sorted(models.items()):
        rng = np.random.default_rng(int(training_seed))
        for state in bank:
            observation = np.asarray(state["observation"], dtype=np.float32)
            raw_action, q_at_actor = pipeline.policy_action_and_q_physical(
                model=model,
                obs=observation,
                env_config=env_config,
                rng=rng,
                compute_q=True,
            )
            raw_action = np.asarray(raw_action, dtype=float).reshape(2)
            safe_action, filter_info = _filter_action(namespace, raw_action, state, run_config)
            safe_action = np.asarray(safe_action, dtype=float).reshape(2)
            delta_phys = safe_action - raw_action
            delta_box = delta_phys / half_range
            correction_phys = float(np.linalg.norm(delta_phys))
            correction_box = float(np.linalg.norm(delta_box))
            active_phys, active_scaled, active_types, basis_source = _active_description(
                filter_info,
                delta_phys,
                delta_box,
                correction_epsilon=max(float(active_tolerance), 1e-10),
            )
            physical_decomposition = normal_tangent_decomposition(delta_phys, active_phys)
            scaled_decomposition = normal_tangent_decomposition(delta_box, active_scaled)
            normal_phys = np.asarray(physical_decomposition["normal"], dtype=float)
            tangent_phys = np.asarray(physical_decomposition["tangent"], dtype=float)
            normal_box = np.asarray(scaled_decomposition["normal"], dtype=float)
            tangent_box = np.asarray(scaled_decomposition["tangent"], dtype=float)
            normal_projector_phys = np.asarray(physical_decomposition["normal_projector"], dtype=float)
            tangent_projector_phys = np.asarray(physical_decomposition["tangent_projector"], dtype=float)
            normal_projector_box = np.asarray(scaled_decomposition["normal_projector"], dtype=float)
            tangent_projector_box = np.asarray(scaled_decomposition["tangent_projector"], dtype=float)
            unique_types = tuple(dict.fromkeys(active_types))
            active_type = "+".join(unique_types) if unique_types else "none"
            if len(unique_types) > 1:
                active_type = f"multi:{active_type}"
            primary_type = unique_types[0] if unique_types else "none"
            signed_normal_box = np.nan
            signed_normal_phys = np.nan
            if active_scaled.shape[0]:
                scaled_unit = active_scaled[0] / max(float(np.linalg.norm(active_scaled[0])), 1e-12)
                signed_normal_box = float(delta_box @ scaled_unit)
            if active_phys.shape[0]:
                physical_unit = active_phys[0] / max(float(np.linalg.norm(active_phys[0])), 1e-12)
                signed_normal_phys = float(delta_phys @ physical_unit)
            rows.append(
                {
                    "training_seed": int(training_seed),
                    "variant": str(variant),
                    "reward_penalty": bool(variant in {"c_reward", "e_reward_actor"}),
                    "actor_cbf_loss": bool(variant in {"d_loss", "e_reward_actor"}),
                    "bank_index": int(state["bank_index"]),
                    "state_hash": str(state["state_hash"]),
                    "stratum": str(state["stratum"]),
                    "categories": "|".join(map(str, state["categories"])),
                    "source_training_seed": int(state["source_training_seed"]),
                    "source_variant": str(state["source_variant"]),
                    "raw_ax": float(raw_action[0]),
                    "raw_ay": float(raw_action[1]),
                    "safe_ax": float(safe_action[0]),
                    "safe_ay": float(safe_action[1]),
                    "delta_ax": float(delta_phys[0]),
                    "delta_ay": float(delta_phys[1]),
                    "correction_physical_norm": correction_phys,
                    "delta_box_ax": float(delta_box[0]),
                    "delta_box_ay": float(delta_box[1]),
                    "correction_box_norm": correction_box,
                    "intervention": bool(correction_box > correction_epsilon),
                    "q_at_raw_actor_action": float(q_at_actor),
                    "raw_feasible": bool(filter_info.get("raw_feasible", correction_box <= 1e-10)),
                    "qp_success": bool(filter_info.get("qp_success", False)),
                    "fallback_used": bool(filter_info.get("fallback_used", False)),
                    "projection_solver": str(filter_info.get("projection_solver", "unknown")),
                    "active_constraint_type": active_type,
                    "primary_active_constraint_type": primary_type,
                    "active_constraint_count": int(active_phys.shape[0]),
                    "active_constraint_rank_physical": int(physical_decomposition["rank"]),
                    "active_constraint_rank_box": int(scaled_decomposition["rank"]),
                    "decomposition_basis_source": basis_source,
                    "normal_projector_physical_00": float(normal_projector_phys[0, 0]),
                    "normal_projector_physical_01": float(normal_projector_phys[0, 1]),
                    "normal_projector_physical_10": float(normal_projector_phys[1, 0]),
                    "normal_projector_physical_11": float(normal_projector_phys[1, 1]),
                    "tangent_projector_physical_00": float(tangent_projector_phys[0, 0]),
                    "tangent_projector_physical_01": float(tangent_projector_phys[0, 1]),
                    "tangent_projector_physical_10": float(tangent_projector_phys[1, 0]),
                    "tangent_projector_physical_11": float(tangent_projector_phys[1, 1]),
                    "normal_projector_box_00": float(normal_projector_box[0, 0]),
                    "normal_projector_box_01": float(normal_projector_box[0, 1]),
                    "normal_projector_box_10": float(normal_projector_box[1, 0]),
                    "normal_projector_box_11": float(normal_projector_box[1, 1]),
                    "tangent_projector_box_00": float(tangent_projector_box[0, 0]),
                    "tangent_projector_box_01": float(tangent_projector_box[0, 1]),
                    "tangent_projector_box_10": float(tangent_projector_box[1, 0]),
                    "tangent_projector_box_11": float(tangent_projector_box[1, 1]),
                    "normal_delta_ax": float(normal_phys[0]),
                    "normal_delta_ay": float(normal_phys[1]),
                    "normal_correction_physical_norm": float(physical_decomposition["normal_norm"]),
                    "tangent_delta_ax": float(tangent_phys[0]),
                    "tangent_delta_ay": float(tangent_phys[1]),
                    "tangent_correction_physical_norm": float(physical_decomposition["tangent_norm"]),
                    "normal_delta_box_ax": float(normal_box[0]),
                    "normal_delta_box_ay": float(normal_box[1]),
                    "normal_correction_box_norm": float(scaled_decomposition["normal_norm"]),
                    "tangent_delta_box_ax": float(tangent_box[0]),
                    "tangent_delta_box_ay": float(tangent_box[1]),
                    "tangent_correction_box_norm": float(scaled_decomposition["tangent_norm"]),
                    "signed_primary_normal_correction_physical": signed_normal_phys,
                    "signed_primary_normal_correction_box": signed_normal_box,
                    "decomposition_reconstruction_error_physical": float(
                        physical_decomposition["reconstruction_error"]
                    ),
                    "decomposition_reconstruction_error_box": float(
                        scaled_decomposition["reconstruction_error"]
                    ),
                    "decomposition_orthogonality_error_physical": float(
                        physical_decomposition["orthogonality_error"]
                    ),
                    "decomposition_orthogonality_error_box": float(
                        scaled_decomposition["orthogonality_error"]
                    ),
                    "state_h_min": float(state["h_min"]),
                    "state_h_dot": float(state["h_dot"]),
                    "state_ttc_s": float(state["ttc_s"]),
                    "state_vehicle_spacing_m": _finite_float(state["vehicle_spacing_m"]),
                    "state_traffic_density_per_km": float(state["traffic_density_per_km"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["training_seed", "variant", "bank_index"]).reset_index(drop=True)


FIXED_SUMMARY_METRICS = (
    "mean_raw_ax",
    "mean_raw_ay",
    "intervention_probability",
    "mean_correction_physical_norm",
    "p95_correction_physical_norm",
    "mean_correction_box_norm",
    "p95_correction_box_norm",
    "mean_abs_delta_ax",
    "mean_abs_delta_ay",
    "mean_delta_ax",
    "mean_delta_ay",
    "mean_normal_correction_physical_norm",
    "mean_tangent_correction_physical_norm",
    "mean_normal_correction_box_norm",
    "mean_tangent_correction_box_norm",
    "raw_feasible_probability",
    "fallback_probability",
)


def summarize_fixed_state_actions(actions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped_frames: list[tuple[str, pd.DataFrame]] = [("all", actions)]
    grouped_frames.extend((str(stratum), group) for stratum, group in actions.groupby("stratum", sort=True))
    for stratum, frame in grouped_frames:
        for (seed, variant), group in frame.groupby(["training_seed", "variant"], sort=True):
            rows.append(
                {
                    "training_seed": int(seed),
                    "variant": str(variant),
                    "stratum": stratum,
                    "states": int(len(group)),
                    "mean_raw_ax": float(group["raw_ax"].mean()),
                    "sd_raw_ax": float(group["raw_ax"].std(ddof=1)) if len(group) > 1 else np.nan,
                    "p05_raw_ax": float(group["raw_ax"].quantile(0.05)),
                    "p95_raw_ax": float(group["raw_ax"].quantile(0.95)),
                    "mean_raw_ay": float(group["raw_ay"].mean()),
                    "sd_raw_ay": float(group["raw_ay"].std(ddof=1)) if len(group) > 1 else np.nan,
                    "p05_raw_ay": float(group["raw_ay"].quantile(0.05)),
                    "p95_raw_ay": float(group["raw_ay"].quantile(0.95)),
                    "intervention_probability": float(group["intervention"].mean()),
                    "mean_correction_physical_norm": float(group["correction_physical_norm"].mean()),
                    "p95_correction_physical_norm": float(group["correction_physical_norm"].quantile(0.95)),
                    "mean_correction_box_norm": float(group["correction_box_norm"].mean()),
                    "p95_correction_box_norm": float(group["correction_box_norm"].quantile(0.95)),
                    "mean_abs_delta_ax": float(group["delta_ax"].abs().mean()),
                    "mean_abs_delta_ay": float(group["delta_ay"].abs().mean()),
                    "mean_delta_ax": float(group["delta_ax"].mean()),
                    "mean_delta_ay": float(group["delta_ay"].mean()),
                    "mean_normal_correction_physical_norm": float(
                        group["normal_correction_physical_norm"].mean()
                    ),
                    "mean_tangent_correction_physical_norm": float(
                        group["tangent_correction_physical_norm"].mean()
                    ),
                    "mean_normal_correction_box_norm": float(group["normal_correction_box_norm"].mean()),
                    "mean_tangent_correction_box_norm": float(group["tangent_correction_box_norm"].mean()),
                    "raw_feasible_probability": float(group["raw_feasible"].mean()),
                    "fallback_probability": float(group["fallback_used"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["stratum", "training_seed", "variant"]).reset_index(drop=True)


def summarize_interventions_by_constraint(actions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (seed, variant, constraint_type), group in actions.groupby(
        ["training_seed", "variant", "active_constraint_type"], sort=True
    ):
        rows.append(
            {
                "training_seed": int(seed),
                "variant": str(variant),
                "active_constraint_type": str(constraint_type),
                "states": int(len(group)),
                "fraction_of_states": float(len(group) / max(len(actions[(actions.training_seed == seed) & (actions.variant == variant)]), 1)),
                "intervention_probability": float(group["intervention"].mean()),
                "mean_correction_box_norm": float(group["correction_box_norm"].mean()),
                "mean_normal_correction_box_norm": float(group["normal_correction_box_norm"].mean()),
                "mean_tangent_correction_box_norm": float(group["tangent_correction_box_norm"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _boxplot_by_variant(actions: pd.DataFrame, column: str, ax: plt.Axes, title: str) -> None:
    variants = list(EXPECTED_FACTORIAL_VARIANTS.values())
    arrays = [pd.to_numeric(actions.loc[actions["variant"] == variant, column], errors="coerce").dropna() for variant in variants]
    ax.boxplot(arrays, tick_labels=variants, showfliers=False)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)


def plot_fixed_state_distributions(actions: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    _boxplot_by_variant(actions, "raw_ax", axes[0], "fixed-state raw longitudinal action ax")
    _boxplot_by_variant(actions, "raw_ay", axes[1], "fixed-state raw lateral action ay")
    axes[0].set_ylabel("physical acceleration")
    axes[1].set_ylabel("physical acceleration")
    fig.savefig(output_dir / "fixed_state_raw_action_distributions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    specifications = [
        ("correction_box_norm", "|Delta a| (box scale)"),
        ("delta_ax", "longitudinal Delta ax"),
        ("delta_ay", "lateral Delta ay"),
        ("normal_correction_box_norm", "normal correction (box scale)"),
        ("tangent_correction_box_norm", "tangent correction (box scale)"),
        ("signed_primary_normal_correction_box", "signed primary-normal correction"),
    ]
    for ax, (column, title) in zip(axes.ravel(), specifications):
        _boxplot_by_variant(actions, column, ax, title)
    fig.savefig(output_dir / "fixed_state_correction_distributions.png", dpi=180)
    plt.close(fig)

    active = pd.crosstab(actions["variant"], actions["active_constraint_type"], normalize="index")
    active = active.reindex(list(EXPECTED_FACTORIAL_VARIANTS.values())).fillna(0.0)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    active.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("fraction of fixed states")
    ax.set_title("Active CBF constraint type")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.savefig(output_dir / "active_constraint_types.png", dpi=180)
    plt.close(fig)


def _ecdf(values: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    array.sort()
    return array, np.arange(1, len(array) + 1, dtype=float) / max(len(array), 1)


def plot_occupancy_distributions(occupancy: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("h_min", "minimum h"),
        ("h_dot", "h dot at minimum-h constraint"),
        ("ttc_s", "TTC (s; capped)"),
        ("vehicle_spacing_m", "minimum vehicle clearance (m)"),
        ("traffic_density_per_km", "traffic density (/km)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for variant in EXPECTED_FACTORIAL_VARIANTS.values():
            x, y = _ecdf(occupancy.loc[occupancy["source_variant"] == variant, metric])
            ax.plot(x, y, label=variant)
        ax.set_title(title)
        ax.set_ylabel("empirical CDF")
        ax.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.04))
    fig.savefig(output_dir / "on_policy_occupancy_distributions.png", dpi=180)
    plt.close(fig)


def _critic_values_on_physical_grid(
    model: Any,
    observation: np.ndarray,
    physical_actions: np.ndarray,
    env_config: Mapping[str, Any],
) -> np.ndarray:
    model_observation = pipeline._model_observation(model, observation)
    actions = np.asarray(physical_actions, dtype=np.float32).reshape(-1, 2)
    model_low = np.asarray(model.action_space.low, dtype=np.float32).reshape(-1)[:2]
    model_high = np.asarray(model.action_space.high, dtype=np.float32).reshape(-1)[:2]
    if np.allclose(model_low, -1.0, atol=1e-5) and np.allclose(model_high, 1.0, atol=1e-5):
        model_actions = np.stack(
            [pipeline.physical_to_normalized(action, dict(env_config)) for action in actions]
        ).astype(np.float32)
    else:
        model_actions = np.clip(actions, model_low, model_high).astype(np.float32)
    buffer_actions = model.policy.scale_action(model_actions)
    observation_batch = np.repeat(np.asarray(model_observation)[None, ...], len(actions), axis=0)
    observation_tensor, _ = model.policy.obs_to_tensor(observation_batch)
    action_tensor = th.as_tensor(buffer_actions, device=model.device, dtype=th.float32)
    with th.no_grad():
        q_values = model.critic(observation_tensor, action_tensor)
        minimum_q = th.min(th.cat(q_values, dim=1), dim=1).values
    return minimum_q.detach().cpu().numpy().reshape(-1)


def _selected_bank_states(bank: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    selected_hashes: set[str] = set()
    for stratum in STRATA:
        matches = [item for item in bank if item["stratum"] == stratum]
        if matches:
            selected.append(matches[0])
            selected_hashes.add(str(matches[0]["state_hash"]))
        if len(selected) >= int(count):
            return selected
    for item in bank:
        if str(item["state_hash"]) not in selected_hashes:
            selected.append(item)
            selected_hashes.add(str(item["state_hash"]))
        if len(selected) >= int(count):
            break
    return selected


def plot_critic_action_maps(
    namespace: Mapping[str, Any],
    models: Mapping[tuple[int, str], Any],
    bank: Sequence[Mapping[str, Any]],
    fixed_actions: pd.DataFrame,
    run_config: Mapping[str, Any],
    output_dir: Path,
    *,
    selected_count: int,
    grid_size: int,
) -> list[str]:
    """Plot Q contours, feasible sets, and all four actor/filter actions."""

    warnings: list[str] = []
    first_seed = min(seed for seed, _ in models)
    variants = list(EXPECTED_FACTORIAL_VARIANTS.values())
    env_config = dict(run_config["env_config"])
    low, high = filter_physical_bounds(namespace, run_config)
    ax_values = np.linspace(float(low[0]), float(high[0]), int(grid_size))
    ay_values = np.linspace(float(low[1]), float(high[1]), int(grid_size))
    ax_grid, ay_grid = np.meshgrid(ax_values, ay_values)
    physical_grid = np.stack([ax_grid.ravel(), ay_grid.ravel()], axis=1)
    contour_dir = output_dir / "critic_action_maps"
    contour_dir.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(variants, plt.get_cmap("tab10").colors[: len(variants)]))

    for state in _selected_bank_states(bank, selected_count):
        try:
            reference_raw = fixed_actions[
                (fixed_actions["training_seed"] == first_seed)
                & (fixed_actions["variant"] == variants[0])
                & (fixed_actions["state_hash"] == state["state_hash"])
            ].iloc[0]
            _, filter_info = _filter_action(
                namespace,
                np.asarray([reference_raw["raw_ax"], reference_raw["raw_ay"]]),
                state,
                run_config,
            )
            system = _typed_system_from_filter_info(filter_info)
            feasible = typed_feasible_mask(ax_grid, ay_grid, system)["all"]
            state_rows = fixed_actions[
                (fixed_actions["training_seed"] == first_seed)
                & (fixed_actions["state_hash"] == state["state_hash"])
            ].set_index("variant")

            fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True, constrained_layout=True)
            for ax, variant in zip(axes.ravel(), variants):
                model = models[(first_seed, variant)]
                q_values = _critic_values_on_physical_grid(
                    model,
                    np.asarray(state["observation"], dtype=np.float32),
                    physical_grid,
                    env_config,
                ).reshape(ax_grid.shape)
                contour = ax.contourf(ax_grid, ay_grid, q_values, levels=22, cmap="viridis")
                if np.any(feasible) and np.any(~feasible):
                    ax.contour(ax_grid, ay_grid, feasible.astype(float), levels=[0.5], colors="white", linewidths=2.0)
                    ax.contourf(
                        ax_grid,
                        ay_grid,
                        (~feasible).astype(float),
                        levels=[0.5, 1.5],
                        colors="none",
                        hatches=["////"],
                        alpha=0.0,
                    )
                for other_variant in variants:
                    other = state_rows.loc[other_variant]
                    ax.scatter(
                        float(other["raw_ax"]),
                        float(other["raw_ay"]),
                        color=colors[other_variant],
                        s=30,
                        alpha=0.65,
                        marker="o",
                    )
                own = state_rows.loc[variant]
                ax.annotate(
                    "",
                    xy=(float(own["safe_ax"]), float(own["safe_ay"])),
                    xytext=(float(own["raw_ax"]), float(own["raw_ay"])),
                    arrowprops={"arrowstyle": "->", "color": "red", "lw": 2.0},
                )
                ax.scatter(float(own["raw_ax"]), float(own["raw_ay"]), color="red", s=65, marker="o", label="own raw")
                ax.scatter(float(own["safe_ax"]), float(own["safe_ay"]), color="white", edgecolor="red", s=65, marker="X", label="own safe")
                ax.set_title(variant)
                ax.set_xlabel("longitudinal acceleration ax")
                ax.set_ylabel("lateral acceleration ay")
                fig.colorbar(contour, ax=ax, shrink=0.75, label="Q(s,a)")
            fig.suptitle(
                f"seed {first_seed} | {state['stratum']} | state {str(state['state_hash'])[:10]}\n"
                "white boundary = CBF-feasible region; hatch = infeasible"
            )
            target = contour_dir / f"state_{int(state['bank_index']):04d}_{state['stratum']}.png"
            fig.savefig(target, dpi=180)
            plt.close(fig)
        except Exception as exc:  # Plotting is diagnostic; tabular analysis remains authoritative.
            warnings.append(f"state {state.get('state_hash')}: {type(exc).__name__}: {exc}")
            plt.close("all")
    return warnings


def _occupancy_summary(occupancy: pd.DataFrame) -> pd.DataFrame:
    metrics = ["h_min", "h_dot", "ttc_s", "vehicle_spacing_m", "traffic_density_per_km"]
    rows: list[dict[str, Any]] = []
    for (seed, variant), group in occupancy.groupby(["source_training_seed", "source_variant"], sort=True):
        row: dict[str, Any] = {
            "training_seed": int(seed),
            "variant": str(variant),
            "steps": int(len(group)),
            "intervention_probability": float(group["intervention"].mean()),
            "overtaking_state_probability": float(group["overtaking"].mean()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"mean_{metric}"] = float(values.mean())
            row[f"p05_{metric}"] = float(values.quantile(0.05))
            row[f"p50_{metric}"] = float(values.quantile(0.50))
            row[f"p95_{metric}"] = float(values.quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.collector_scenarios <= 0 or args.collector_steps <= 0 or args.states_per_category <= 0:
        raise ValueError("collector scenarios, collector steps, and states per category must be positive")
    if not 0.0 <= float(args.dense_quantile) <= 1.0:
        raise ValueError("--dense-quantile must lie in [0, 1]")
    if args.contour_grid < 5:
        raise ValueError("--contour-grid must be at least 5")

    pipeline.set_stable_native_defaults()
    study_dir = args.study_dir.resolve()
    run_config, model_index = _validate_study(study_dir)
    project_root = pipeline.find_project_root((args.project_root or SCRIPT_DIR.parent).resolve())
    output_dir = (args.output_dir or (study_dir / "counterfactual_analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    namespace = _bootstrap_runtime(project_root, run_config, device=args.device)
    namespace["CBF_QP_ACTIVE_TOL"] = float(args.active_tolerance)

    models: dict[tuple[int, str], Any] = {}
    resolved_models: list[dict[str, Any]] = []
    for _, row in model_index.iterrows():
        key = (int(row["training_seed"]), str(row["variant"]))
        path = _resolve_model_path(row["model_path"], study_dir, project_root)
        models[key] = pipeline.load_model(key[1], path, args.device)
        resolved_models.append({"training_seed": key[0], "variant": key[1], "model_path": str(path)})

    evaluation_protocol = dict(run_config.get("evaluation_protocol", {}))
    scenario_seed_start = (
        int(args.bank_seed_start)
        if args.bank_seed_start is not None
        else int(evaluation_protocol.get("eval_seed_start", evaluation_protocol.get("scenario_seed_start", 90_000))) + 50_000
    )
    neighbor_range = (
        float(args.neighbor_range)
        if args.neighbor_range is not None
        else float(namespace.get("CBF_NEIGHBOR_RANGE", 60.0))
    )
    candidates, occupancy = collect_shielded_state_candidates(
        namespace,
        models,
        run_config,
        scenario_count=int(args.collector_scenarios),
        steps_per_scenario=int(args.collector_steps),
        scenario_seed_start=scenario_seed_start,
        neighbor_range=neighbor_range,
        ttc_cap=float(args.ttc_cap),
    )
    occupancy.to_csv(output_dir / "occupancy_steps.csv", index=False)
    occupancy_summary = _occupancy_summary(occupancy)
    occupancy_summary.to_csv(output_dir / "occupancy_summary_by_seed.csv", index=False)
    occupancy_metrics_for_effects = [
        column
        for column in occupancy_summary.columns
        if column not in {"training_seed", "variant", "steps"}
    ]
    occupancy_effects = compute_factorial_contrasts(
        occupancy_summary,
        occupancy_metrics_for_effects,
        group_columns=("training_seed",),
    )
    occupancy_effects.to_csv(output_dir / "occupancy_factorial_effects_by_seed.csv", index=False)
    summarize_factorial_contrasts(occupancy_effects).to_csv(
        output_dir / "occupancy_factorial_effects_summary.csv", index=False
    )

    density_values = pd.to_numeric(occupancy["traffic_density_per_km"], errors="coerce").dropna()
    dense_threshold = float(density_values.quantile(float(args.dense_quantile)))
    bank, bank_metadata = stratify_state_bank(
        candidates,
        int(args.states_per_category),
        seed=scenario_seed_start,
        near_boundary_margin=float(args.near_boundary_margin),
        dense_threshold=dense_threshold,
    )
    pd.DataFrame(
        [
            {
                "stratum": stratum,
                "available_unique_states": int(bank_metadata["availability"][stratum]),
                "requested_states": int(args.states_per_category),
                "selected_states": int(bank_metadata["selection_counts"][stratum]),
                "shortfall": int(bank_metadata["selection_shortfall"][stratum]),
                "missing": bool(stratum in bank_metadata["missing_strata"]),
            }
            for stratum in STRATA
        ]
    ).to_csv(output_dir / "state_bank_strata_coverage.csv", index=False)
    if bank_metadata["missing_strata"]:
        print(
            "[counterfactual] warning: collector produced no states for strata "
            + ", ".join(bank_metadata["missing_strata"]),
            flush=True,
        )
    write_state_bank(bank, output_dir)
    fixed_actions = evaluate_common_state_bank(
        namespace,
        models,
        bank,
        run_config,
        active_tolerance=float(args.active_tolerance),
    )
    fixed_actions.to_csv(output_dir / "fixed_state_actions.csv", index=False)
    fixed_summary = summarize_fixed_state_actions(fixed_actions)
    fixed_summary.to_csv(output_dir / "fixed_state_intervention_summary.csv", index=False)
    summarize_interventions_by_constraint(fixed_actions).to_csv(
        output_dir / "fixed_state_constraint_summary.csv", index=False
    )
    fixed_effects = compute_factorial_contrasts(
        fixed_summary,
        FIXED_SUMMARY_METRICS,
        group_columns=("training_seed", "stratum"),
    )
    fixed_effects.to_csv(output_dir / "fixed_state_factorial_effects_by_seed.csv", index=False)
    summarize_factorial_contrasts(fixed_effects).to_csv(
        output_dir / "fixed_state_factorial_effects_summary.csv", index=False
    )

    plot_warnings: list[str] = []
    if not args.skip_plots:
        plot_fixed_state_distributions(fixed_actions, output_dir)
        plot_occupancy_distributions(occupancy, output_dir)
        if not args.skip_contours:
            plot_warnings.extend(
                plot_critic_action_maps(
                    namespace,
                    models,
                    bank,
                    fixed_actions,
                    run_config,
                    output_dir,
                    selected_count=int(args.selected_states),
                    grid_size=int(args.contour_grid),
                )
            )

    filter_low, filter_high = filter_physical_bounds(namespace, run_config)
    manifest = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "pipeline_schema_version": int(run_config["schema_version"]),
        "study_dir": str(study_dir),
        "output_dir": str(output_dir),
        "factorial_variants": {
            f"reward_{int(reward)}_actor_loss_{int(loss)}": variant
            for (reward, loss), variant in EXPECTED_FACTORIAL_VARIANTS.items()
        },
        "models": resolved_models,
        "collector": {
            "mode": "shielded",
            "all_factorial_actors_and_training_seeds": True,
            "scenario_seed_start": scenario_seed_start,
            "scenario_count": int(args.collector_scenarios),
            "steps_per_scenario_per_actor": int(args.collector_steps),
            "neighbor_range_m": neighbor_range,
            "ttc_definition": "constant-velocity first disc-contact time, capped",
            "ttc_cap_s": float(args.ttc_cap),
        },
        "state_bank": bank_metadata,
        "correction_conventions": {
            "delta": "a_safe - a_raw",
            "physical": "environment acceleration units",
            "box_scaled": "per-axis physical delta divided by half action-box range",
            "intervention_threshold_box_norm": float(run_config["correction_epsilon_normalized"]),
            "filter_action_low": filter_low.astype(float).tolist(),
            "filter_action_high": filter_high.astype(float).tolist(),
            "filter_bounds_source": "namespace CBF_AX_BOUNDS / CBF_AY_BOUNDS with env-config fallback",
            "normal_tangent": (
                "orthogonal projection onto the row span / null space of tight constraints "
                "with positive KKT contribution"
            ),
            "fallback_basis": "actual correction direction when no valid active set is reported",
        },
        "interpretation_guardrail": (
            "fixed_state files isolate the actor action map; occupancy files measure visited-state shifts. "
            "Do not infer policy internalization solely from on-policy correction reductions."
        ),
        "plot_warnings": plot_warnings,
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[counterfactual] wrote {len(bank)} common states, {len(fixed_actions)} actor-state rows, "
        f"and {len(occupancy)} occupancy steps to {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
