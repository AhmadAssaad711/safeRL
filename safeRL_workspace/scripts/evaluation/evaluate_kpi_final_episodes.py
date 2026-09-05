from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from stable_baselines3 import DDPG
from torch.utils.tensorboard import SummaryWriter

from scripts.training.cbf_lambda_event_bc_pilot_sweep import (
    exec_notebook_cells,
    find_project_root,
    install_event_penalty_env,
    set_stable_native_defaults,
)
from scripts.training.cbf_reward_term_ablation import NOTEBOOK_DEPS, behavior_score, install_safety_set_reward_wrapper, summarize
from scripts.common.guided_cbf_minimal import install_minimal_guided_cbf
from scripts.training.train_safety_potential_variants import (
    INITIAL_VARIANT_NAMES,
    TB_VARIANT_RUN_NAMES,
    VARIANTS,
    evaluate_model,
)


DISPLAY_COLS = [
    "variant",
    "episodes",
    "return_mean",
    "completion_rate",
    "episode_length_steps_mean",
    "distance_traveled_m_mean",
    "episode_time_s_mean",
    "mean_abs_speed_error",
    "mean_lat_y_error_m",
    "ego_collisions_per_km_mean",
    "ego_collisions_mean",
    "h_min",
    "boundary_h_min",
    "qp_failure_rate",
    "event_intervention_rate",
    "mean_correction_norm",
    "mean_meaningful_correction_norm",
    "progress_rate_mps_mean",
    "mean_neighbor_density_per_km",
]


HEADLINE_KPIS = [
    ("01_performance/episode_return", "return_mean"),
    ("01_performance/episode_length_steps", "episode_length_steps_mean"),
    ("02_safety/ego_collisions_per_km", "ego_collisions_per_km_mean"),
    ("02_safety/h_min", "h_min"),
    ("02_safety/qp_failure_rate", "qp_failure_rate"),
    ("03_efficiency/abs_speed_error_mps", "mean_abs_speed_error"),
    ("03_efficiency/progress_rate_mps", "progress_rate_mps_mean"),
    ("04_control_filter/intervention_rate", "event_intervention_rate"),
    ("04_control_filter/correction_norm", "mean_correction_norm"),
    ("05_traffic/neighbor_density_per_km", "mean_neighbor_density_per_km"),
]


def _as_float(value: Any) -> float | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if pd.notna(scalar) else None


def load_model(namespace: dict[str, Any], variant_cfg: dict[str, Any], model_path: Path, device: str) -> Any:
    model_cls = DDPG
    if variant_cfg.get("model_class_name") == "GuidedCBFDDPG":
        model_cls = namespace["GuidedCBFDDPG"]
    return model_cls.load(str(model_path), device=device)


def write_tensorboard(summary: pd.DataFrame, output_dir: Path, step: int) -> None:
    tb_dir = output_dir.resolve() / "tb"
    for _, row in summary.iterrows():
        variant = str(row["variant"])
        run_name = TB_VARIANT_RUN_NAMES.get(variant, variant)
        run_dir = tb_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_dir))
        for tag, column in HEADLINE_KPIS:
            value = _as_float(row.get(column))
            if value is not None:
                writer.add_scalar(f"final_eval_200/{tag}", value, int(step))
        writer.add_text(
            "00_index",
            "\n".join(
                [
                    "# 200-Episode Final Evaluation",
                    "",
                    "Same task protocol as the KPI experiment final evaluation, rerun with 200 episodes.",
                    "Collision termination is disabled; episodes end on 1000 m task completion or 1200-step timeout.",
                ]
            ),
            0,
        )
        writer.flush()
        writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun final KPI evaluation from existing saved models.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/sp3_mtm_e0p1_p0p0_kpi_full"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--seed", type=int, default=407_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-tensorboard", action="store_true")
    return parser.parse_args()


def main() -> int:
    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()
    project_root = find_project_root(args.project_root or Path.cwd())
    source_dir = (project_root / args.source_dir).resolve() if not args.source_dir.is_absolute() else args.source_dir.resolve()
    output_dir = args.output_dir or source_dir / f"final_eval_{int(args.episodes)}eps"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = source_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(run_config_path)
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))

    namespace: dict[str, Any] = {"__name__": "__main__"}
    exec_notebook_cells(project_root / "notebooks" / "lanelessKaralakou.ipynb", NOTEBOOK_DEPS, namespace)
    namespace["DEVICE"] = args.device
    namespace["CBF_K0"] = float(run_config["k0"])
    namespace["CBF_K1"] = float(run_config["k1"])
    namespace["CBF_EPS_SIDE"] = float(run_config["eps_side"])
    namespace["CBF_FILTER_REWARD_LAMBDA"] = float(run_config["lambda_norm"])
    install_minimal_guided_cbf(namespace)
    install_safety_set_reward_wrapper(namespace)
    install_event_penalty_env(namespace)

    env_config = dict(run_config["env_config"])
    reward_config = dict(run_config["reward_config"])
    baseline_reward_config = dict(run_config.get("baseline_reward_config", reward_config))
    cbf_reward_config = dict(run_config.get("cbf_reward_config", reward_config))
    use_distance_task = bool(run_config.get("use_distance_task_eval", True))
    task_distance_m = float(run_config.get("task_distance_m", 1000.0))
    task_max_steps = int(run_config.get("task_max_steps", 1200))
    event_threshold = float(run_config.get("event_threshold", 0.03))
    k0 = float(run_config["k0"])
    k1 = float(run_config["k1"])
    eps_side = float(run_config["eps_side"])
    timesteps = int(run_config.get("timesteps", 50_000))

    print(
        "[kpi-final-eval]"
        f" source={source_dir}"
        f" output={output_dir}"
        f" episodes={args.episodes}"
        f" task={task_distance_m:g}m/{task_max_steps}steps",
        flush=True,
    )

    summaries: list[dict[str, Any]] = []
    if args.variants is None:
        configured_variants = [
            str(item.get("variant"))
            for item in run_config.get("variants", [])
            if isinstance(item, dict) and item.get("variant")
        ]
        variant_filter = set(configured_variants or INITIAL_VARIANT_NAMES)
    else:
        variant_filter = set(args.variants)
    selected_variants = [cfg for cfg in VARIANTS if str(cfg["variant"]) in variant_filter]
    if not selected_variants:
        raise ValueError(f"No variants selected from {args.variants!r}")

    for variant_cfg in selected_variants:
        index = VARIANTS.index(variant_cfg)
        variant = str(variant_cfg["variant"])
        model_path = source_dir / variant / "model.zip"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = load_model(namespace, variant_cfg, model_path, args.device)
        variant_reward_config = baseline_reward_config if str(variant_cfg["env_kind"]) == "baseline" else cbf_reward_config
        print(f"[kpi-final-eval] evaluating {variant} from {model_path}", flush=True)
        episodes = evaluate_model(
            namespace,
            model,
            env_kind=str(variant_cfg["env_kind"]),
            episodes=int(args.episodes),
            seed=int(args.seed) + 100_000 * (index + 1) + int(args.episode_start),
            reward_config=variant_reward_config,
            env_config=env_config,
            event_threshold=event_threshold,
            k0=k0,
            k1=k1,
            eps_side=eps_side,
            use_distance_task=use_distance_task,
            task_distance_m=task_distance_m,
            task_max_steps=task_max_steps,
        )
        episodes["episode"] = pd.to_numeric(episodes["episode"], errors="coerce") + int(args.episode_start)
        episodes_path = output_dir / f"{variant}_episodes.csv"
        if args.append and episodes_path.exists():
            existing = pd.read_csv(episodes_path)
            episodes = (
                pd.concat([existing, episodes], ignore_index=True)
                .drop_duplicates(subset=["episode"], keep="last")
                .sort_values("episode")
                .reset_index(drop=True)
            )
        episodes.to_csv(episodes_path, index=False)
        row: dict[str, Any] = {
            "variant": variant,
            "label": str(variant_cfg["label"]),
            "model_path": str(model_path),
            "episodes": float(len(episodes)),
            "timesteps": float(timesteps),
            "use_distance_task_eval": use_distance_task,
            "task_distance_m": task_distance_m,
            "task_max_steps": float(task_max_steps),
            **summarize(episodes),
        }
        row["behavior_score"] = behavior_score(row)
        summaries.append(row)
        print(
            "[kpi-final-eval-result]"
            f" {variant}"
            f" return={row['return_mean']:.2f}"
            f" complete={row.get('completion_rate', 0.0):.2%}"
            f" C/km={row.get('ego_collisions_per_km_mean', 0.0):.2f}"
            f" h_min={row.get('h_min', float('nan')):.3f}"
            f" qp={row.get('qp_failure_rate', 0.0):.2%}"
            f" int={row.get('event_intervention_rate', 0.0):.2%}"
            f" corr={row.get('mean_correction_norm', 0.0):.3f}",
            flush=True,
        )

    summary = pd.DataFrame(summaries)
    summary_path = output_dir / "summary.csv"
    if args.append and summary_path.exists():
        existing_summary = pd.read_csv(summary_path)
        summary = (
            pd.concat([existing_summary, summary], ignore_index=True)
            .drop_duplicates(subset=["variant"], keep="last")
            .sort_values("variant")
            .reset_index(drop=True)
        )
    summary.to_csv(summary_path, index=False)
    config_path = output_dir / "eval_config.json"
    config_path.write_text(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "episodes": int(args.episodes),
                "seed": int(args.seed),
                "task_distance_m": task_distance_m,
                "task_max_steps": task_max_steps,
                "use_distance_task_eval": use_distance_task,
                "event_threshold": event_threshold,
                "k0": k0,
                "k1": k1,
                "eps_side": eps_side,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.skip_tensorboard:
        write_tensorboard(summary, output_dir, timesteps)

    print(f"[kpi-final-eval] wrote {summary_path}", flush=True)
    display_cols = [column for column in DISPLAY_COLS if column in summary.columns]
    print(summary[display_cols].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
