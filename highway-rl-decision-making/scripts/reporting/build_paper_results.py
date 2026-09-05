from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


VARIANTS = {
    "DDPG without CBF": {
        "folder": "DDPG",
        "final": "ddpg_flat42_vmax24_ego_y_only_laneless_karalakou_final_metrics.csv",
        "color": "#4472C4",
    },
    "DDPG-CBF reward": {
        "folder": "DDPG_CBF_Reward",
        "final": "ddpg_cbf_flat42_vmax24_noslack_tuned_laneless_karalakou_final_metrics.csv",
        "color": "#ED7D31",
    },
    "DDPG-CBF reward + loss": {
        "folder": "DDPG_CBF_Reward_Loss",
        "final": "guided_ddpg_cbf_flat42_vmax24_noslack_tuned_laneless_karalakou_final_metrics.csv",
        "color": "#70AD47",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "source": str(source.resolve()),
        "archived": str(destination.resolve()),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def event_time(path: Path) -> datetime:
    try:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 1})
        accumulator.Reload()
        times = [event.wall_time for tag in accumulator.Tags().get("scalars", []) for event in accumulator.Scalars(tag)[:1]]
        if times:
            return datetime.fromtimestamp(min(times))
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def discover_events(artifact_dir: Path, metadata: dict[str, object]) -> list[dict[str, str]]:
    recorded = metadata.get("tensorboard_events", [])
    if recorded:
        return [
            {
                "kind": str(item.get("kind", "unknown")),
                "path": str(item.get("archived_path") or item.get("source") or item.get("path")),
            }
            for item in recorded
            if Path(str(item.get("archived_path") or item.get("source") or item.get("path"))).is_file()
        ]

    start = datetime.strptime(str(metadata["run_tag"]), "%Y%m%d_%H%M%S")
    end = datetime.fromisoformat(str(metadata["archived_at"]))
    roots = {
        "standard": artifact_dir / "tensorboard",
        "custom": Path(os.environ.get("TEMP") or os.environ.get("TMP") or r"C:\Windows\Temp") / "laneless_tb",
    }
    selected: list[dict[str, str]] = []
    for kind, root in roots.items():
        if not root.exists():
            continue
        for path in root.rglob("events.out.tfevents.*"):
            timestamp = event_time(path)
            if start <= timestamp <= end:
                selected.append({"kind": kind, "path": str(path)})
    return selected


def write_filtered_trace(
    source: Path,
    destination: Path,
    *,
    variant: str,
    timestep_column: str,
) -> dict[str, object]:
    """Archive one variant from the existing combined trace without retraining."""
    frame = pd.read_csv(source)
    if "variant" not in frame.columns:
        raise RuntimeError(f"Combined trace has no variant column: {source}")
    filtered = frame.loc[frame["variant"].astype(str) == variant].copy()
    if filtered.empty:
        raise RuntimeError(f"No rows for {variant} in {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(destination, index=False)
    numeric = pd.to_numeric(filtered[timestep_column], errors="coerce") if timestep_column in filtered else pd.Series(dtype=float)
    return {
        "source": str(source.resolve()),
        "filter": {"column": "variant", "equals": variant},
        "archived": str(destination.resolve()),
        "rows": int(len(filtered)),
        "max_timestep": float(numeric.max()) if not numeric.empty else None,
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def baseline_metadata(artifact_dir: Path) -> dict[str, object]:
    """Describe the existing canonical baseline; never launch a replacement run."""
    event_files = sorted((artifact_dir / "tensorboard" / "DDPG_136").glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError("Existing baseline TensorBoard event file DDPG_136 was not found")
    return {
        "label": "DDPG without CBF",
        "slug": "ddpg_without_cbf",
        "task": "canonical-existing-model",
        "run_tag": "canonical_20260714",
        "run_dir": str(artifact_dir),
        "expected_timesteps": 50000.0,
        "max_timestep": 50000.0,
        "complete": True,
        "archived_at": "2026-07-14T12:04:53",
        "source_policy": "existing canonical model and evaluation; no retraining",
        "tensorboard_events": [{"kind": "standard", "path": str(event_files[0])}],
        "notes": [
            "The canonical baseline model and 50-episode evaluation already existed.",
            "The existing combined step trace supplies 50,000 baseline timesteps.",
            "The existing combined episode trace stops at timestep 19,844; this limitation is recorded rather than filled by retraining.",
        ],
    }


def known_event_sources(artifact_dir: Path, variant: str) -> list[dict[str, str]]:
    """Use the event files tied to the two completed July 15 archives."""
    temp_root = Path(os.environ.get("TEMP") or os.environ.get("TMP") or r"C:\Windows\Temp") / "laneless_tb"
    if variant == "DDPG-CBF reward":
        standard_dirs = [artifact_dir / "tensorboard" / "DDPG_145"]
        custom_dirs = [temp_root / "ddpg-cbf_reward_20260715_085621"]
    elif variant == "DDPG-CBF reward + loss":
        standard_dirs = [artifact_dir / "tensorboard" / "DDPG_147"]
        custom_dirs = [temp_root / "ddpg-cbf_reward_plus_loss_20260715_092430"]
    else:
        return []
    events: list[dict[str, str]] = []
    for kind, directories in (("standard", standard_dirs), ("custom", custom_dirs)):
        for directory in directories:
            events.extend({"kind": kind, "path": str(path)} for path in sorted(directory.glob("events.out.tfevents.*")))
    return [event for event in events if Path(event["path"]).is_file()]


def extract_scalars(event_path: Path, variant: str, kind: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    rows: list[dict[str, object]] = []
    tags: list[dict[str, object]] = []
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        tags.append({
            "variant": variant, "source_kind": kind, "event_file": event_path.name,
            "tag": tag, "count": len(events),
            "min_step": min((e.step for e in events), default=None),
            "max_step": max((e.step for e in events), default=None),
        })
        rows.extend({
            "variant": variant, "source_kind": kind, "event_file": event_path.name,
            "tag": tag, "step": e.step, "wall_time": e.wall_time, "value": e.value,
        } for e in events)
    return rows, tags


def summarize_final(frame: pd.DataFrame, variant: str) -> dict[str, float | str]:
    def mean(column: str) -> float:
        return float(pd.to_numeric(frame[column], errors="coerce").mean()) if column in frame else float("nan")
    return {
        "variant": variant,
        "episodes": int(len(frame)),
        "mean_return": mean("return"),
        "completion_rate_pct": 100.0 * mean("task_completed"),
        "mean_abs_speed_deviation": mean("mean_abs_speed_deviation"),
        "ego_collisions_per_km": mean("ego_collisions_per_km"),
        "total_collisions_per_km": mean("total_collision_events_per_km"),
        "mean_jerk_norm": mean("mean_jerk_norm"),
        "action_saturation_rate": mean("action_saturation_rate"),
        "intervention_rate": mean("event_intervention_rate"),
        "qp_failure_rate": mean("qp_failure_rate"),
    }


def plot_bars(summary: pd.DataFrame, colors: list[str], destination: Path) -> None:
    panels = [
        ("mean_return", "Mean episode return", "Return"),
        ("mean_abs_speed_deviation", "Mean speed error", "m/s"),
        ("ego_collisions_per_km", "Ego collisions", "collisions/km"),
        ("mean_jerk_norm", "Mean jerk", "m/s³"),
    ]
    labels = ["DDPG", "CBF reward", "CBF reward + loss"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    fig.suptitle("Evaluated 50k DDPG Models", fontsize=18, fontweight="bold")
    for ax, (column, title, ylabel) in zip(axes.flat, panels):
        values = summary[column].to_numpy(dtype=float)
        bars = ax.bar(labels, values, color=colors, width=0.68, edgecolor="white", linewidth=1.2)
        ax.set_title(title, fontweight="semibold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.23, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=8)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=9)
        top = np.nanmax(values) if np.isfinite(values).any() else 1.0
        ax.set_ylim(0, max(top * 1.18, 1e-6))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reproducible 50k paper-result archives.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    artifact_dir = root / "artifacts" / "lanelessKaralakou"
    output = artifact_dir / "PaperResults"
    comparison = output / "Comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    registry = json.loads((artifact_dir / "latest_training_runs.json").read_text(encoding="utf-8"))

    all_scalars: list[dict[str, object]] = []
    all_tags: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    histories: list[pd.DataFrame] = []
    manifest: dict[str, object] = {"created_at": datetime.now().isoformat(timespec="seconds"), "variants": {}}

    for variant, spec in VARIANTS.items():
        is_baseline = variant == "DDPG without CBF"
        metadata = baseline_metadata(artifact_dir) if is_baseline else registry.get(variant)
        if not metadata or not metadata.get("complete") or float(metadata.get("max_timestep", 0)) < 50000:
            raise RuntimeError(f"{variant} is not available as a complete 50k source")
        run_dir = Path(str(metadata["run_dir"]))
        variant_dir = output / str(spec["folder"])
        files: dict[str, object] = {}
        if is_baseline:
            files["model.zip"] = copy_file(
                artifact_dir / "ddpg_flat42_vmax24_ego_y_only_laneless_karalakou.zip",
                variant_dir / "model.zip",
            )
            files["eval_history.csv"] = copy_file(
                artifact_dir / "ddpg_flat42_vmax24_ego_y_only_laneless_karalakou_eval_history.csv",
                variant_dir / "eval_history.csv",
            )
            files["training_step_trace.csv"] = write_filtered_trace(
                artifact_dir / "ddpg_direct_training_step_trace_combined.csv",
                variant_dir / "training_step_trace.csv",
                variant=variant,
                timestep_column="global_timestep",
            )
            files["training_episode_trace.csv"] = write_filtered_trace(
                artifact_dir / "ddpg_direct_training_episode_trace_combined.csv",
                variant_dir / "training_episode_trace.csv",
                variant=variant,
                timestep_column="end_global_timestep",
            )
        else:
            for source_name, destination_name in [
                ("model.zip", "model.zip"), ("eval_history.csv", "eval_history.csv"),
                ("step_trace.csv", "training_step_trace.csv"), ("episode_trace.csv", "training_episode_trace.csv"),
            ]:
                files[destination_name] = copy_file(run_dir / source_name, variant_dir / destination_name)
        final_path = artifact_dir / str(spec["final"])
        files["final_evaluation_50_episodes.csv"] = copy_file(final_path, variant_dir / "final_evaluation_50_episodes.csv")
        final = pd.read_csv(final_path)
        if len(final) != 50:
            raise RuntimeError(f"{variant} final evaluation has {len(final)} rows, expected 50")
        summary = summarize_final(final, variant)
        summaries.append(summary)
        pd.DataFrame([summary]).to_csv(variant_dir / "final_summary.csv", index=False)

        history_source = artifact_dir / "ddpg_flat42_vmax24_ego_y_only_laneless_karalakou_eval_history.csv" if is_baseline else run_dir / "eval_history.csv"
        history = pd.read_csv(history_source)
        history.insert(0, "variant", variant)
        histories.append(history)
        events = discover_events(artifact_dir, metadata) if is_baseline else known_event_sources(artifact_dir, variant)
        if not events:
            raise RuntimeError(f"No TensorBoard event files found for {variant}")
        event_manifest = []
        for event_index, event in enumerate(events):
            source = Path(event["path"])
            # Keep the archive path short enough for deeply nested Windows
            # workspaces; the manifest retains the complete original path.
            destination = variant_dir / "tensorboard" / "events" / event["kind"] / f"event_{event_index:02d}.tfevents"
            event_manifest.append(
                copy_file(source, destination)
                | {
                    "kind": event["kind"],
                    "original_name": source.name,
                    "original_parent": str(source.parent),
                }
            )
            rows, tags = extract_scalars(destination, variant, event["kind"])
            all_scalars.extend(rows)
            all_tags.extend(tags)
        variant_scalars = pd.DataFrame([row for row in all_scalars if row["variant"] == variant])
        variant_tags = [row for row in all_tags if row["variant"] == variant]
        variant_scalars.to_csv(variant_dir / "tensorboard" / "scalars_long.csv", index=False)
        wide = variant_scalars.assign(series=lambda x: x["source_kind"] + "::" + x["tag"]).pivot_table(
            index="step", columns="series", values="value", aggfunc="last"
        ).reset_index()
        wide.to_csv(variant_dir / "tensorboard" / "scalars_wide.csv", index=False)
        (variant_dir / "tensorboard" / "tags.json").write_text(json.dumps(variant_tags, indent=2), encoding="utf-8")
        metadata = {
            **metadata,
            "paper_results_files": files,
            "final_evaluation_source": str(final_path.resolve()),
            "tensorboard_event_count": len(event_manifest),
            "tensorboard_events": event_manifest,
        }
        (variant_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        manifest["variants"][variant] = {"run": metadata, "files": files, "tensorboard_events": event_manifest}

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(comparison / "final_evaluation_summary.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(comparison / "training_eval_history.csv", index=False)
    pd.DataFrame(all_scalars).to_csv(comparison / "tensorboard_scalars_long.csv", index=False)
    pd.DataFrame(all_tags).to_csv(comparison / "tensorboard_tags.csv", index=False)
    colors = [str(VARIANTS[name]["color"]) for name in VARIANTS]
    figure = comparison / "figures" / "evaluated_50k_model_comparison.png"
    plot_bars(summary_frame, colors, figure)
    copy_file(figure, artifact_dir / "ddpg_evaluated_final_comparison.png")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (output / "README.md").write_text(
        "# Paper Results\n\nExisting 50k training/evaluation artifacts for DDPG, DDPG-CBF reward, "
        "and DDPG-CBF reward + loss. The two CBF variants are complete archived 50k runs. "
        "The DDPG baseline uses the existing canonical model and 50-episode evaluation; its "
        "step trace reaches 50,000, while the available episode trace reaches 19,844. That "
        "coverage is recorded in `DDPG/run_metadata.json`; no retraining was performed. Each "
        "model folder contains raw traces, evaluation data, original TensorBoard event files, "
        "exported scalar CSVs, hashes, and run provenance. `Comparison/` contains combined tables "
        "and publication-ready bar charts.\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
