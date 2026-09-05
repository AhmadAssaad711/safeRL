from __future__ import annotations

import json
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from scripts.common.guided_cbf_minimal import install_minimal_guided_cbf

warnings.filterwarnings("ignore", message="OSQP exited.*")


TRAINING_TIMESTEPS = 50_000
TRAIN_EVAL_FREQ = 10_000
TRAIN_EVAL_EPISODES = 2
FINAL_EVAL_EPISODES = 50
FINAL_EVAL_LAMBDA_REWARD = 0.0

TRIALS = [
    ("shield_only", 0.0, 0.0),
    ("reward_only_005", 0.05, 0.0),
    ("bc_only_001", 0.0, 0.01),
    ("bc_only_003", 0.0, 0.03),
    ("bc_only_010", 0.0, 0.10),
    ("bc_only_030", 0.0, 0.30),
    ("bc_only_100", 0.0, 1.00),
    ("reward0025_bc003", 0.025, 0.03),
    ("reward0025_bc010", 0.025, 0.10),
    ("reward0025_bc030", 0.025, 0.30),
    ("reward005_bc003", 0.05, 0.03),
    ("reward005_bc010", 0.05, 0.10),
    ("reward005_bc030", 0.05, 0.30),
]


def find_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    for candidate in [script_path.parent, *script_path.parents]:
        notebook = candidate / "notebooks" / "lanelessKaralakou.ipynb"
        env_file = candidate / "laneless highway env" / "lane_free_env.py"
        if notebook.exists() and env_file.exists():
            return candidate
    raise RuntimeError("Could not find repo root containing notebooks/lanelessKaralakou.ipynb")


def load_notebook_namespace(repo_root: Path) -> dict[str, Any]:
    notebook_path = repo_root / "notebooks" / "lanelessKaralakou.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {}

    required_prefixes = [
        "from __future__ import annotations",
        "class KaralakouRewardWrapper",
        "ENV_CONFIG = {",
        "class LaneFreeObservationNormalizationWrapper",
        "try:\n    from qpsolvers import solve_qp",
        "CBF_AX_BOUNDS =",
        "def _lane_free_base",
        "class SafetyFilteredAccelerationWrapper",
        "# Tuned DDPG-CBF shield overrides",
        "def evaluate_cbf_policy_with_metrics",
    ]

    for prefix in required_prefixes:
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if source.startswith(prefix):
                exec(compile(source, f"{notebook_path}:cell_{index}", "exec"), namespace)
                break
        else:
            raise RuntimeError(f"Could not find notebook cell starting with {prefix!r}")

    guided_source = None
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.startswith("# Guided DDPG-CBF actor update"):
            guided_source = source.split("\nguided_ddpg_cbf_train_env = ", 1)[0]
            exec(compile(guided_source, f"{notebook_path}:cell_{index}_defs_only", "exec"), namespace)
            break
    if guided_source is None:
        raise RuntimeError("Could not find guided DDPG-CBF actor update cell")

    install_minimal_guided_cbf(namespace)
    return namespace


def trial_tag(trial_name: str, lambda_reward: float, lambda_bc: float, seed: int) -> str:
    def fmt(value: float) -> str:
        return str(float(value)).replace(".", "p")

    return f"{trial_name}_lr_{fmt(lambda_reward)}_lbc_{fmt(lambda_bc)}_seed_{seed}"


def evaluate_guided_policy(
    namespace: dict[str, Any],
    model: Any,
    episodes: int,
    seed: int,
    lambda_reward_eval: float = FINAL_EVAL_LAMBDA_REWARD,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for episode in range(episodes):
        env = namespace["make_guided_cbf_single_env"](
            seed=seed + episode,
            lambda_filter=lambda_reward_eval,
        )
        namespace["configure_paper_evaluation_env"](env, steps=namespace["PAPER_EVAL_STEPS"])
        obs, _ = env.reset(seed=seed + episode)
        done = False
        step_count = 0

        rewards: list[float] = []
        signed_deviations: list[float] = []
        abs_deviations: list[float] = []
        speeds: list[float] = []
        corrections: list[float] = []
        interventions: list[float] = []
        qp_successes: list[float] = []
        fallbacks: list[float] = []
        min_h_values: list[float] = []
        a_rl_x: list[float] = []
        a_rl_y: list[float] = []
        a_safe_x: list[float] = []
        a_safe_y: list[float] = []
        ego_collisions = 0
        ego_collision_steps = 0
        all_collision_events = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            base = env.unwrapped
            desired = float(base.vehicle.desired_speed)
            speed = float(base.vehicle.vx)
            deviation = speed - desired

            rewards.append(float(reward))
            signed_deviations.append(deviation)
            abs_deviations.append(abs(deviation))
            speeds.append(speed)
            corrections.append(float(info.get("correction_norm", info.get("cbf_correction_norm", 0.0))))
            interventions.append(float(info.get("intervention", info.get("cbf_intervened", False))))
            qp_successes.append(float(info.get("qp_success", info.get("cbf_qp_success", True))))
            fallbacks.append(float(info.get("fallback_used", False)))
            min_h_values.append(float(info.get("cbf_min_h", np.nan)))
            a_rl_x.append(float(info.get("cbf_a_rl_x", np.nan)))
            a_rl_y.append(float(info.get("cbf_a_rl_y", np.nan)))
            a_safe_x.append(float(info.get("cbf_a_safe_x", np.nan)))
            a_safe_y.append(float(info.get("cbf_a_safe_y", np.nan)))
            all_collision_events += int(info.get("collisions", 0))
            ego_collisions += int(info.get("ego_collision_events", 0))
            if bool(info.get("ego_collision", False)):
                ego_collision_steps += 1

            step_count += 1
            done = bool(terminated or truncated)

        rows.append(
            {
                "episode": float(episode),
                "steps": float(step_count),
                "return": float(np.sum(rewards)),
                "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
                "mean_signed_speed_deviation": float(np.mean(signed_deviations)) if signed_deviations else 0.0,
                "mean_abs_speed_deviation": float(np.mean(abs_deviations)) if abs_deviations else 0.0,
                "ego_collisions": float(ego_collisions),
                "ego_collision_steps": float(ego_collision_steps),
                "total_collision_events": float(all_collision_events),
                "mean_correction_norm": float(np.mean(corrections)) if corrections else 0.0,
                "max_correction_norm": float(np.max(corrections)) if corrections else 0.0,
                "intervention_rate": float(np.mean(interventions)) if interventions else 0.0,
                "qp_failure_rate": float(1.0 - np.mean(qp_successes)) if qp_successes else 0.0,
                "fallback_rate": float(np.mean(fallbacks)) if fallbacks else 0.0,
                "min_h": float(np.nanmin(min_h_values)) if min_h_values and not np.all(np.isnan(min_h_values)) else np.nan,
                "mean_a_rl_x": float(np.nanmean(a_rl_x)) if a_rl_x else np.nan,
                "mean_a_rl_y": float(np.nanmean(a_rl_y)) if a_rl_y else np.nan,
                "mean_a_safe_x": float(np.nanmean(a_safe_x)) if a_safe_x else np.nan,
                "mean_a_safe_y": float(np.nanmean(a_safe_y)) if a_safe_y else np.nan,
            }
        )
        env.close()
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "steps": float(metrics["steps"].mean()),
        "return_common_eval": float(metrics["return"].mean()),
        "mean_speed": float(metrics["mean_speed"].mean()),
        "mean_signed_speed_deviation": float(metrics["mean_signed_speed_deviation"].mean()),
        "mean_abs_speed_deviation": float(metrics["mean_abs_speed_deviation"].mean()),
        "ego_collisions": float(metrics["ego_collisions"].mean()),
        "ego_collision_steps": float(metrics["ego_collision_steps"].mean()),
        "total_collision_events": float(metrics["total_collision_events"].mean()),
        "mean_correction_norm": float(metrics["mean_correction_norm"].mean()),
        "max_correction_norm": float(metrics["max_correction_norm"].max()),
        "intervention_rate": float(metrics["intervention_rate"].mean()),
        "qp_failure_rate": float(metrics["qp_failure_rate"].mean()),
        "fallback_rate": float(metrics["fallback_rate"].mean()),
        "min_h": float(metrics["min_h"].min()),
        "mean_a_rl_x": float(metrics["mean_a_rl_x"].mean()),
        "mean_a_rl_y": float(metrics["mean_a_rl_y"].mean()),
        "mean_a_safe_x": float(metrics["mean_a_safe_x"].mean()),
        "mean_a_safe_y": float(metrics["mean_a_safe_y"].mean()),
    }


class LambdaAblationCallback(BaseCallback):
    def __init__(
        self,
        namespace: dict[str, Any],
        trial_name: str,
        lambda_reward: float,
        lambda_bc: float,
        seed: int,
        eval_freq: int = TRAIN_EVAL_FREQ,
        n_eval_episodes: int = TRAIN_EVAL_EPISODES,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.namespace = namespace
        self.trial_name = trial_name
        self.lambda_reward = float(lambda_reward)
        self.lambda_bc = float(lambda_bc)
        self.seed = int(seed)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.records: list[dict[str, float | str]] = []
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        metrics = evaluate_guided_policy(
            self.namespace,
            self.model,
            episodes=self.n_eval_episodes,
            seed=self.seed + 20_000 + self.num_timesteps,
            lambda_reward_eval=FINAL_EVAL_LAMBDA_REWARD,
        )
        summary = summarize_metrics(metrics)
        row: dict[str, float | str] = {
            "trial_name": self.trial_name,
            "lambda_reward": self.lambda_reward,
            "lambda_bc": self.lambda_bc,
            "timesteps": float(self.num_timesteps),
            **summary,
        }
        self.records.append(row)
        if self.verbose:
            print(
                "[eval]"
                f" {self.trial_name}"
                f" steps={self.num_timesteps:,}"
                f" return={summary['return_common_eval']:.2f}"
                f" ego_col={summary['ego_collisions']:.2f}"
                f" abs_dev={summary['mean_abs_speed_deviation']:.3f}"
                f" intervention={summary['intervention_rate']:.2%}"
                f" corr={summary['mean_correction_norm']:.3f}"
                f" qp_fail={summary['qp_failure_rate']:.3%}",
                flush=True,
            )
        return True


def main() -> None:
    repo_root = find_repo_root()
    namespace = load_notebook_namespace(repo_root)

    artifact_dir: Path = namespace["ARTIFACT_DIR"]
    legacy_output_dir = artifact_dir / "guided_cbf_lambda_ablation_bounds3"
    output_dir = artifact_dir / "gcbf_lam_b3"
    model_dir = output_dir / "models"
    history_dir = output_dir / "train_eval_history"
    final_dir = output_dir / "final_eval"
    for directory in [model_dir, history_dir, final_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    seed = int(namespace["SEED"])
    partial_path = output_dir / "summary_partial.csv"
    if partial_path.exists():
        summary_rows: list[dict[str, float | str]] = pd.read_csv(partial_path).to_dict("records")
        completed = {str(row["trial_name"]) for row in summary_rows}
    else:
        summary_rows = []
        completed = set()

    print(
        "Starting guided DDPG-CBF lambda ablation",
        {
            "trials": len(TRIALS),
            "timesteps_per_trial": TRAINING_TIMESTEPS,
            "final_eval_episodes": FINAL_EVAL_EPISODES,
            "final_eval_lambda_reward": FINAL_EVAL_LAMBDA_REWARD,
            "cbf_bounds": [namespace["CBF_AX_BOUNDS"], namespace["CBF_AY_BOUNDS"]],
            "base_bounds": namespace["ENV_CONFIG"]["bounds"],
            "k0": namespace["CBF_K0"],
            "k1": namespace["CBF_K1"],
            "output_dir": str(output_dir),
        },
        flush=True,
    )

    for index, (name, lambda_reward, lambda_bc) in enumerate(TRIALS, start=1):
        tag = trial_tag(name, lambda_reward, lambda_bc, seed)
        short_tag = f"t{index:02d}_{name}"
        model_path = model_dir / f"{short_tag}.zip"
        history_path = history_dir / f"{short_tag}.csv"
        final_path = final_dir / f"{short_tag}.csv"
        legacy_model_path = legacy_output_dir / "models" / f"{tag}.zip"
        if not model_path.exists() and legacy_model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_model_path, model_path)
            print(f"Migrated existing model from legacy long path: {legacy_model_path}", flush=True)

        if name in completed and model_path.exists() and final_path.exists():
            print(f"\n=== [{index}/{len(TRIALS)}] {name} already complete; skipping ===", flush=True)
            continue

        print(
            f"\n=== [{index}/{len(TRIALS)}] {name}: lambda_reward={lambda_reward:g}, lambda_bc={lambda_bc:g} ===",
            flush=True,
        )

        for path in [model_path, history_path, final_path]:
            path.parent.mkdir(parents=True, exist_ok=True)

        if model_path.exists():
            print(f"Found existing model for {name}; evaluating without retraining.", flush=True)
            model = namespace["GuidedCBFDDPG"].load(str(model_path), device=namespace["DEVICE"])
            elapsed = float("nan")
        else:
            start = time.time()
            train_env = namespace["make_guided_cbf_training_env"](
                seed=seed,
                lambda_filter=float(lambda_reward),
            )
            n_actions = train_env.action_space.shape[-1]
            noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions, dtype=np.float32),
                sigma=namespace["DDPG_OU_SIGMA"] * np.ones(n_actions, dtype=np.float32),
            )
            callback = LambdaAblationCallback(
                namespace=namespace,
                trial_name=name,
                lambda_reward=lambda_reward,
                lambda_bc=lambda_bc,
                seed=seed + index * 1_000,
                verbose=1,
            )

            model = namespace["GuidedCBFDDPG"](
                "MlpPolicy",
                train_env,
                learning_rate=namespace["DDPG_LEARNING_RATE"],
                buffer_size=namespace["DDPG_REPLAY_MEMORY"],
                learning_starts=namespace["DDPG_LEARNING_STARTS"],
                batch_size=namespace["DDPG_BATCH_SIZE"],
                tau=namespace["DDPG_TAU"],
                gamma=namespace["DDPG_GAMMA"],
                train_freq=(1, "step"),
                gradient_steps=1,
                action_noise=noise,
                policy_kwargs={"net_arch": [256, 128]},
                tensorboard_log=str(artifact_dir / "tensorboard"),
                verbose=0,
                seed=seed,
                device=namespace["DEVICE"],
                lambda_bc=float(lambda_bc),
                bc_delta=namespace["GUIDED_CBF_BC_DELTA"],
                bc_action_scale=namespace["GUIDED_CBF_ACTION_SCALE"],
                bc_weight_max=namespace["GUIDED_CBF_WEIGHT_MAX"],
            )

            model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=callback, progress_bar=False)
            elapsed = time.time() - start
            model.save(str(model_path))
            train_env.close()
            history_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(callback.records).to_csv(history_path, index=False)

        final_metrics = evaluate_guided_policy(
            namespace,
            model,
            episodes=FINAL_EVAL_EPISODES,
            seed=seed + 100_000 + 10_000 * index,
            lambda_reward_eval=FINAL_EVAL_LAMBDA_REWARD,
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_metrics.to_csv(final_path, index=False)
        final_summary = summarize_metrics(final_metrics)

        row: dict[str, float | str] = {
            "trial_name": name,
            "lambda_reward": float(lambda_reward),
            "lambda_bc": float(lambda_bc),
            "bc_delta": float(namespace["GUIDED_CBF_BC_DELTA"]),
            "seed": float(seed),
            "timesteps": float(TRAINING_TIMESTEPS),
            "elapsed_min": float(elapsed / 60.0),
            "model_path": str(model_path),
            "history_path": str(history_path),
            "final_eval_path": str(final_path),
            **final_summary,
        }
        summary_rows = [existing for existing in summary_rows if existing.get("trial_name") != name]
        summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(partial_path, index=False)

        print(
            "[final]"
            f" {name}"
            f" return={final_summary['return_common_eval']:.2f}"
            f" ego_col={final_summary['ego_collisions']:.3f}"
            f" abs_dev={final_summary['mean_abs_speed_deviation']:.3f}"
            f" intervention={final_summary['intervention_rate']:.2%}"
            f" corr={final_summary['mean_correction_norm']:.3f}"
            f" qp_fail={final_summary['qp_failure_rate']:.3%}"
            f" fallback={final_summary['fallback_rate']:.3%}"
            f" elapsed={elapsed / 60.0:.1f}m",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    ranked = summary.sort_values(
        by=[
            "ego_collisions",
            "ego_collision_steps",
            "mean_correction_norm",
            "intervention_rate",
            "mean_abs_speed_deviation",
            "return_common_eval",
        ],
        ascending=[True, True, True, True, True, False],
    )
    ranked_path = output_dir / "summary_ranked.csv"
    ranked.to_csv(ranked_path, index=False)

    print("\n=== GUIDED CBF LAMBDA ABLATION COMPLETE ===", flush=True)
    print(
        ranked[
            [
                "trial_name",
                "lambda_reward",
                "lambda_bc",
                "return_common_eval",
                "mean_abs_speed_deviation",
                "ego_collisions",
                "ego_collision_steps",
                "total_collision_events",
                "intervention_rate",
                "mean_correction_norm",
                "qp_failure_rate",
                "fallback_rate",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("Saved summary:", summary_path, flush=True)
    print("Saved ranked summary:", ranked_path, flush=True)


if __name__ == "__main__":
    main()
