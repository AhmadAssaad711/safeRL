"""Decide whether a finished paired study's conclusion can be believed.

This answers one question -- "is the difference between these two arms an
artefact of how reinforcement learning was run rather than of the thing being
tested?" -- from data that already exists. It trains nothing and evaluates
nothing; it reads TensorBoard event files, the per-block KPI tables, and the run
manifests that every completed run already wrote.

Three checks, in the order that can stop the investigation soonest:

1. **Arm symmetry.** Diff the resolved training signatures across all runs. If
   the arms differ in anything beyond the variant and the seed -- a different Git
   commit, a different PPO config, a different evaluation protocol -- the
   comparison is confounded and no amount of seeds or statistics repairs it.
   This is the cheapest check and the only one that can invalidate everything.

2. **Optimizer health.** ``approx_kl``, ``clip_fraction``,
   ``explained_variance`` and ``train/std`` over each run's first and last
   decile of updates, plus the tail trend of episode length. These diagnose the
   failures hyperparameter tuning can fix -- steps too large, exploration
   collapsing, critic not fitting -- and, just as importantly, show whether the
   two arms differ on them at all. They are *per-update* quantities: healthy
   values bound how far one update moves, not how far the policy drifts across
   the tens of updates separating the checkpoints actually being compared.

3. **Whether the effect is resolvable.** The paired test, the variance split
   into evaluation noise versus genuine seed effect, the exact sign-test floor,
   and the detectable-effect table. A three-seed paired design resolves about a
   38% relative effect at 80% power; quoting a 13% effect from it is quoting
   noise, however many seeds happen to agree in sign.

Read the emitted ``training_stability_summary.md`` first; the CSVs carry the
same numbers for plotting and for the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from saferl.stability import (
    DEFAULT_ALPHA,
    DEFAULT_POWER,
    DEFAULT_TAIL_FRACTION,
    decile_summary,
    diagnostic_verdict,
    minimum_detectable_effect,
    paired_comparison,
    required_pairs,
    tail_slope,
    variance_decomposition,
)

# The per-update optimization diagnostics, matching the tag list already used by
# scripts/reporting/plot_ppo_progression_results.py so both read the same names.
DIAGNOSTIC_TAGS = (
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
    "train/std",
    "train/entropy_loss",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/learning_rate",
)
DEFAULT_KPI_COLUMN = "ego_collisions_per_km"
DEFAULT_MODE = "raw"
BLOCK_TABLE = Path("pe") / "b.csv"
RUN_CONFIG = "run_config.json"
EPISODE_TABLE = "training_episodes.csv"

# Trend is fitted on raw per-episode lengths from ``training_episodes.csv``, not
# on the ``rollout/ep_len_mean`` TensorBoard tag. That tag is Stable-Baselines3's
# rolling mean over the last 100 episodes, so consecutive points share up to 99
# episodes and are heavily autocorrelated; an OLS t statistic on it violates the
# independence assumption and overstates significance. The raw episode rows do
# not have that defect.
EPISODE_STEP_COLUMN = "global_timestep"
EPISODE_LENGTH_COLUMN = "episode_length"

# Signature keys that are *expected* to differ between a control and its
# treatment arm, or between seeds. Anything else differing is a confound.
# ``training_signature_hash`` is a pure function of the signature, so it differs
# whenever the term does -- that is the mechanism that stops checkpoint reuse,
# not an independent discrepancy.
EXPECTED_SIGNATURE_DIFFERENCES = (
    "variant",
    "variant_spec",
    "training_seed",
    "tensorboard_run_label",
    "reward_config",
    "effective_training_settings",
    "state_cost",
    "training_signature_hash",
)

# Provenance the manifest is required to carry. Absent, the audit cannot rule out
# that the arms were trained from different code, which would explain an arm
# difference entirely. Missing provenance is reported, never treated as a pass.
REQUIRED_PROVENANCE_KEYS = ("git_commit", "git_dirty")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("ARM", "SEED", "DIR"),
        required=True,
        help=(
            "One run: its arm label, its training seed, and its run directory "
            "(the seed_<n> directory holding run_config.json). Repeat per run."
        ),
    )
    parser.add_argument(
        "--control-arm",
        default=None,
        help="Arm label treated as the control when pairing; default is the first given.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--kpi-column", default=DEFAULT_KPI_COLUMN)
    parser.add_argument("--mode", default=DEFAULT_MODE, help="Evaluation mode: raw (CBF-OFF) or cbf.")
    parser.add_argument("--tail-fraction", type=float, default=DEFAULT_TAIL_FRACTION)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--power", type=float, default=DEFAULT_POWER)
    parser.add_argument(
        "--target-effect",
        type=float,
        default=None,
        help=(
            "Effect worth detecting, in KPI units, chosen on scientific grounds. "
            "Required for a non-circular seed-budget recommendation; powering for "
            "the effect already observed flatters the design and is reported as such."
        ),
    )
    return parser.parse_args(argv)


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / RUN_CONFIG
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; this expects a completed run directory, not a study root"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_event_files(run_dir: Path, config: dict[str, Any]) -> list[Path]:
    """Find a run's TensorBoard event files.

    The training runner does not always write them under the run directory: long
    OneDrive paths make it fall back to an ``artifacts/tb/ppo`` tree or a
    ``%LOCALAPPDATA%`` location, and it records where they actually went in
    ``run_config.json``. Trust that record first and only then guess, so a
    relocated log reads as data rather than as a run with no diagnostics.
    """

    recorded = config.get("tensorboard") or {}
    files: list[Path] = []
    for key in ("all_event_files", "new_event_files"):
        for entry in recorded.get(key, ()) or ():
            candidate = Path(str(entry))
            if candidate.exists():
                files.append(candidate)
    if not files:
        log_dir = recorded.get("log_dir")
        roots = [Path(str(log_dir))] if log_dir else []
        roots.append(run_dir / "tensorboard")
        for root in roots:
            if root.exists():
                files.extend(sorted(root.rglob("events.out.tfevents.*")))
    # Deduplicate while keeping order; SB3 writes one file per train_N subdir.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def load_scalars(event_files: Iterable[Path], tags: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Read the requested scalar tags, concatenated across event files in order."""

    collected: dict[str, list[pd.DataFrame]] = {tag: [] for tag in tags}
    for path in event_files:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", ()))
        for tag in tags:
            if tag not in available:
                continue
            scalars = accumulator.Scalars(tag)
            collected[tag].append(
                pd.DataFrame(
                    {
                        "step": [s.step for s in scalars],
                        "value": [s.value for s in scalars],
                    }
                )
            )
    out: dict[str, pd.DataFrame] = {}
    for tag, frames in collected.items():
        if frames:
            out[tag] = pd.concat(frames, ignore_index=True).sort_values("step")
    return out


def audit_arm_symmetry(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Report every training-signature key that differs across the given runs.

    A comparison is only about the term under test if nothing else moved. Runs
    launched at different times can silently differ in Git commit, PPO config or
    evaluation protocol, and that would explain an arm difference entirely.
    """

    def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
        flat: dict[str, Any] = {}
        if isinstance(value, dict):
            for key, item in value.items():
                flat.update(flatten(item, f"{prefix}/{key}"))
        elif isinstance(value, list):
            flat[prefix] = json.dumps(value, sort_keys=True)
        else:
            flat[prefix] = value
        return flat

    flats: dict[str, dict[str, Any]] = {}
    for run in runs:
        config = run["config"]
        signature = config.get("training_signature", config)
        flat = flatten(signature)
        # Provenance lives outside the signature but is just as confounding.
        for key in ("git_commit", "git_dirty", "training_signature_hash"):
            if key in config:
                flat[f"/{key}"] = config[key]
        flats[run["label"]] = flat

    all_keys = sorted({key for flat in flats.values() for key in flat})
    rows: list[dict[str, Any]] = []
    for key in all_keys:
        values = {label: flat.get(key) for label, flat in flats.items()}
        if len({json.dumps(v, sort_keys=True, default=str) for v in values.values()}) <= 1:
            continue
        expected = any(token in key for token in EXPECTED_SIGNATURE_DIFFERENCES)
        row: dict[str, Any] = {"key": key, "expected_to_differ": expected}
        row.update({f"value[{label}]": values[label] for label in flats})
        rows.append(row)
    return pd.DataFrame(rows)


def load_training_episodes(run_dir: Path) -> pd.DataFrame | None:
    """Raw per-episode training rows, the un-smoothed source for the tail trend."""

    path = run_dir / EPISODE_TABLE
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    needed = {EPISODE_STEP_COLUMN, EPISODE_LENGTH_COLUMN}
    if not needed.issubset(frame.columns):
        return None
    return frame.sort_values(EPISODE_STEP_COLUMN)


def summarize_runs(runs: list[dict[str, Any]], tail_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-run optimizer diagnostics and tail trends."""

    diag_rows: list[dict[str, Any]] = []
    trend_rows: list[dict[str, Any]] = []
    for run in runs:
        scalars = load_scalars(run["event_files"], DIAGNOSTIC_TAGS)
        tail: dict[str, float] = {}
        for tag in DIAGNOSTIC_TAGS:
            frame = scalars.get(tag)
            if frame is None or frame.empty:
                continue
            summary = decile_summary(frame["value"].to_numpy())
            diag_rows.append({"arm": run["arm"], "seed": run["seed"], "tag": tag, **summary})
            tail[tag] = summary["last_decile_mean"]
            if tag == "train/std":
                tail["std_first"] = summary["first_decile_mean"]
        healthy, reasons = diagnostic_verdict(
            tail.get("train/approx_kl", float("nan")),
            tail.get("train/clip_fraction", float("nan")),
            tail.get("std_first", float("nan")),
            tail.get("train/std", float("nan")),
        )
        run["healthy"] = healthy
        run["health_reasons"] = reasons
        episodes = load_training_episodes(run["dir"])
        if episodes is None:
            run["trend_source"] = "missing"
            continue
        run["trend_source"] = EPISODE_TABLE
        slope = tail_slope(
            episodes[EPISODE_STEP_COLUMN].to_numpy(),
            episodes[EPISODE_LENGTH_COLUMN].to_numpy(),
            tail_fraction,
        )
        trend_rows.append(
            {
                "arm": run["arm"],
                "seed": run["seed"],
                "metric": EPISODE_LENGTH_COLUMN,
                "source": EPISODE_TABLE,
                "slope_per_100k": slope.slope,
                "t_statistic": slope.t_statistic,
                "p_value": slope.p_value,
                "n_episodes": slope.n_points,
                "flat": slope.is_flat(),
            }
        )
    return pd.DataFrame(diag_rows), pd.DataFrame(trend_rows)


def load_blocks(run_dir: Path, kpi_column: str, mode: str) -> np.ndarray:
    """Per-block KPI values for one run under one evaluation mode."""

    path = run_dir / BLOCK_TABLE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; this run has no KPI evaluation yet, so it cannot "
            "enter the paired comparison"
        )
    frame = pd.read_csv(path)
    if kpi_column not in frame.columns:
        raise KeyError(f"{path} has no column {kpi_column!r}; found {sorted(frame.columns)}")
    selected = frame[frame["mode"] == mode]
    if selected.empty:
        raise ValueError(f"{path} has no rows for mode {mode!r}")
    return selected[kpi_column].to_numpy(dtype=float)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs: list[dict[str, Any]] = []
    for arm, seed, directory in args.run:
        run_dir = Path(directory)
        config = _load_run_config(run_dir)
        runs.append(
            {
                "arm": arm,
                "seed": int(seed),
                "dir": run_dir,
                "label": f"{arm}:{seed}",
                "config": config,
                "event_files": resolve_event_files(run_dir, config),
            }
        )
    args.out.mkdir(parents=True, exist_ok=True)

    symmetry = audit_arm_symmetry(runs)
    diagnostics, trends = summarize_runs(runs, args.tail_fraction)

    arms = sorted({run["arm"] for run in runs})
    control_arm = args.control_arm or runs[0]["arm"]
    if control_arm not in arms:
        raise SystemExit(f"--control-arm {control_arm!r} is not among {arms}")
    treatment_arms = [arm for arm in arms if arm != control_arm]

    blocks = {
        (run["arm"], run["seed"]): load_blocks(run["dir"], args.kpi_column, args.mode)
        for run in runs
    }
    control_seeds = sorted(seed for arm, seed in blocks if arm == control_arm)
    decomposition = (
        variance_decomposition([blocks[(control_arm, s)] for s in control_seeds])
        if len(control_seeds) >= 2
        else {}
    )

    comparison_rows: list[dict[str, Any]] = []
    mde_rows: list[dict[str, Any]] = []
    for arm in treatment_arms:
        paired_seeds = sorted(
            s for s in control_seeds if (arm, s) in blocks
        )
        if len(paired_seeds) < 2:
            continue
        control = np.array([blocks[(control_arm, s)].mean() for s in paired_seeds])
        treatment = np.array([blocks[(arm, s)].mean() for s in paired_seeds])
        pc = paired_comparison(control, treatment, args.alpha)
        sd_lo, sd_hi = pc.sd_confidence_interval
        comparison_rows.append(
            {
                "kpi": args.kpi_column,
                "mode": args.mode,
                "control_arm": control_arm,
                "treatment_arm": arm,
                "seeds": ",".join(str(s) for s in paired_seeds),
                "n_pairs": pc.n_pairs,
                "control_mean": pc.control_mean,
                "treatment_mean": pc.treatment_mean,
                "mean_difference": pc.mean_difference,
                "relative_difference": pc.relative_difference,
                "sd_difference": pc.sd_difference,
                "sd_ci_low": sd_lo,
                "sd_ci_high": sd_hi,
                "standard_error": pc.standard_error,
                "t_statistic": pc.t_statistic,
                "p_value": pc.p_value,
                "significant": pc.significant,
                "sign_test_p_value": pc.sign_test_p_value,
                "mde_at_power": pc.minimum_detectable_effect(args.power),
                "power": args.power,
            }
        )
        for n in (3, 4, 5, 6, 8, 9, 10, 12, 16, 20):
            mde_rows.append(
                {
                    "treatment_arm": arm,
                    "n_pairs": n,
                    "mde_50pct_power": minimum_detectable_effect(
                        pc.sd_difference, n, args.alpha, 0.5
                    ),
                    "mde_at_power": minimum_detectable_effect(
                        pc.sd_difference, n, args.alpha, args.power
                    ),
                    "mde_at_power_relative": minimum_detectable_effect(
                        pc.sd_difference, n, args.alpha, args.power
                    )
                    / pc.control_mean
                    if pc.control_mean
                    else float("nan"),
                    "power": args.power,
                }
            )

    comparisons = pd.DataFrame(comparison_rows)
    mde = pd.DataFrame(mde_rows)
    decomposition_frame = (
        pd.DataFrame([{"arm": control_arm, **decomposition}]) if decomposition else pd.DataFrame()
    )

    outputs = {
        "arm_symmetry_audit.csv": symmetry,
        "training_optimizer_diagnostics.csv": diagnostics,
        "training_tail_trends.csv": trends,
        "kpi_paired_comparison.csv": comparisons,
        "kpi_variance_decomposition.csv": decomposition_frame,
        "kpi_detectable_effect.csv": mde,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.out / name, index=False)

    summary = _render_summary(
        runs, symmetry, diagnostics, trends, comparisons, decomposition, mde, args
    )
    (args.out / "training_stability_summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


def _render_summary(
    runs: list[dict[str, Any]],
    symmetry: pd.DataFrame,
    diagnostics: pd.DataFrame,
    trends: pd.DataFrame,
    comparisons: pd.DataFrame,
    decomposition: dict[str, float],
    mde: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    lines: list[str] = ["# Training stability and effect resolvability", ""]

    lines.append("## 1. Arm symmetry")
    lines.append("")
    missing_provenance = {
        run["label"]: [
            key for key in REQUIRED_PROVENANCE_KEYS if key not in run["config"]
        ]
        for run in runs
    }
    incomplete = {label: keys for label, keys in missing_provenance.items() if keys}
    if incomplete:
        lines.append(
            f"**Provenance incomplete: {len(incomplete)} of {len(runs)} runs record "
            f"none of {list(REQUIRED_PROVENANCE_KEYS)}.** The audit below compares "
            "resolved configuration only, so it *cannot* rule out that the arms were "
            "trained from different code. Treat the symmetry result as necessary but "
            "not sufficient."
        )
        lines.append("")
    unexpected = symmetry[~symmetry["expected_to_differ"]] if not symmetry.empty else symmetry
    if symmetry.empty:
        lines.append("Every run shares an identical training signature. No confound.")
    elif unexpected.empty:
        lines.append(
            f"{len(symmetry)} signature keys differ, all of them expected "
            "(variant, seed, reward term, label, derived hash). No configuration "
            "confound."
        )
    else:
        lines.append(
            f"**{len(unexpected)} UNEXPECTED signature differences.** The arms differ "
            "in more than the term under test, so the comparison is confounded and "
            "no number below is interpretable until this is resolved:"
        )
        lines.append("")
        for _, row in unexpected.iterrows():
            lines.append(f"- `{row['key']}`")
    lines.append("")

    lines.append("## 2. Optimizer health")
    lines.append("")
    for run in runs:
        verdict = "healthy" if run.get("healthy") else "**SUSPECT**"
        detail = "" if run.get("healthy") else " -- " + "; ".join(run.get("health_reasons", []))
        lines.append(f"- `{run['label']}`: {verdict}{detail}")
    lines.append("")
    if not diagnostics.empty:
        pivot = diagnostics.pivot_table(
            index=["arm", "seed"], columns="tag", values="last_decile_mean"
        )
        lines.append("Last-decile means:")
        lines.append("")
        lines.append("```")
        lines.append(pivot.to_string(float_format=lambda v: f"{v:.4g}"))
        lines.append("```")
        lines.append("")
    if not trends.empty:
        still_learning = trends[~trends["flat"]]
        lines.append(
            f"Tail trend over the last {args.tail_fraction:.0%} of training, fitted on "
            f"raw per-episode lengths from `{EPISODE_TABLE}` (not the rolling-mean "
            "TensorBoard tag, whose autocorrelation would overstate significance): "
            f"**{len(still_learning)} of {len(trends)} runs are still significantly "
            "improving.**"
        )
        lines.append("")
        lines.append("```")
        lines.append(
            trends.to_string(index=False, float_format=lambda v: f"{v:.4g}")
        )
        lines.append("```")
        if len(still_learning):
            lines.append("")
            lines.append(
                "A comparison drawn at a budget where the curves have not flattened "
                "measures learning speed at an arbitrary cutoff, not converged "
                "quality, and its sign can move with the budget."
            )
        lines.append("")

    lines.append("## 3. Is the effect resolvable?")
    lines.append("")
    if comparisons.empty:
        lines.append("No paired comparison available (need >= 2 seeds in both arms).")
    for _, row in comparisons.iterrows():
        lines.append(
            f"### {row['treatment_arm']} vs {row['control_arm']} "
            f"-- {row['kpi']}, mode {row['mode']}"
        )
        lines.append("")
        lines.append(
            f"- effect **{row['mean_difference']:+.3f} "
            f"({row['relative_difference']:+.1%})**, n={row['n_pairs']} paired seeds"
        )
        lines.append(
            f"- paired t = {row['t_statistic']:.2f}, p = {row['p_value']:.3f} -> "
            f"**{'significant' if row['significant'] else 'NOT significant'}** "
            f"at alpha={args.alpha}"
        )
        lines.append(
            f"- exact sign test p = {row['sign_test_p_value']:.3f} "
            f"(the floor at n={row['n_pairs']} is {2 / 2 ** row['n_pairs']:.3f}; "
            "'every seed agreed' cannot beat it)"
        )
        lines.append(
            f"- paired SD {row['sd_difference']:.3f}, 95% CI "
            f"[{row['sd_ci_low']:.3f}, {row['sd_ci_high']:.3f}] on "
            f"{row['n_pairs'] - 1} df -- a seed budget planned from this point "
            "estimate is planned from noise"
        )
        lines.append(
            f"- this design resolves **{row['mde_at_power']:+.3f}** at "
            f"{args.power:.0%} power; the measured effect is "
            f"{'inside' if abs(row['mean_difference']) < row['mde_at_power'] else 'outside'} "
            "that limit"
        )
        lines.append("")
    if decomposition:
        lines.append("### Variance decomposition (control arm)")
        lines.append("")
        lines.append(
            f"- within-run block SD {decomposition['within_run_block_sd']:.3f}"
        )
        lines.append(
            f"- evaluation-only SE of a run mean "
            f"{decomposition['evaluation_se_of_run_mean']:.3f}"
        )
        lines.append(f"- observed between-seed SD {decomposition['between_run_sd']:.3f}")
        lines.append(
            f"- **genuine seed-to-seed SD {decomposition['seed_sd']:.3f}** "
            f"({decomposition['seed_fraction_of_between_variance']:.0%} of the "
            "between-seed variance; the rest is evaluation noise that more "
            "episodes would remove)"
        )
        lines.append("")
    if not mde.empty:
        lines.append("### Detectable effect by seed budget")
        lines.append("")
        lines.append("```")
        lines.append(
            mde.to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )
        lines.append("```")
        lines.append("")
        if args.target_effect:
            for arm in mde["treatment_arm"].unique():
                sd = float(
                    comparisons.loc[
                        comparisons["treatment_arm"] == arm, "sd_difference"
                    ].iloc[0]
                )
                n = required_pairs(sd, args.target_effect, args.alpha, args.power)
                lines.append(
                    f"To detect a pre-specified effect of {args.target_effect:.3f} "
                    f"at {args.power:.0%} power, arm `{arm}` needs **n = {n}** paired seeds."
                )
            lines.append("")
        else:
            lines.append(
                "No `--target-effect` given, so no seed recommendation is made. "
                "Powering for the effect already observed is circular and would "
                "flatter the design; choose the effect worth detecting first."
            )
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
