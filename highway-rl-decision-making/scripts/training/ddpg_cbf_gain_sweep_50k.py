from __future__ import annotations

import json
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise


warnings.filterwarnings("ignore", message="OSQP exited.*")


GAIN_CANDIDATES = [
    (8.0, 0.5),
    (4.0, 4.0),
    (2.0, 4.0),
    (2.0, 6.0),
    (1.0, 4.0),
    (1.0, 6.0),
]

TRAINING_TIMESTEPS = 50_000
TRAIN_EVAL_FREQ = 10_000
TRAIN_EVAL_EPISODES = 2
FINAL_EVAL_EPISODES = 50
ACTION_TRACE_EPISODES = 5


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

    executed: set[str] = set()
    for prefix in required_prefixes:
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if source.startswith(prefix):
                exec(compile(source, f"{notebook_path}:cell_{index}", "exec"), namespace)
                executed.add(prefix)
                break
        else:
            raise RuntimeError(f"Could not find notebook cell starting with {prefix!r}")

    missing = set(required_prefixes) - executed
    if missing:
        raise RuntimeError(f"Missing notebook cells: {sorted(missing)}")
    return namespace


def gain_tag(k0: float, k1: float, seed: int) -> str:
    def fmt(value: float) -> str:
        return str(float(value)).replace(".", "p")

    return f"k0_{fmt(k0)}_k1_{fmt(k1)}_seed_{seed}"


class GainSweepCallback(BaseCallback):
    def __init__(
        self,
        namespace: dict[str, Any],
        k0: float,
        k1: float,
        seed: int,
        eval_freq: int = TRAIN_EVAL_FREQ,
        n_eval_episodes: int = TRAIN_EVAL_EPISODES,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.namespace = namespace
        self.k0 = float(k0)
        self.k1 = float(k1)
        self.seed = int(seed)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.records: list[dict[str, float]] = []
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        metrics = evaluate_cbf_policy_for_gains(
            self.namespace,
            self.model,
            self.k0,
            self.k1,
            episodes=self.n_eval_episodes,
            seed=self.seed + 20_000 + self.num_timesteps,
            deterministic=True,
        )
        row = {
            "k0": self.k0,
            "k1": self.k1,
            "seed": float(self.seed),
            "timesteps": float(self.num_timesteps),
            "return": float(metrics["return"].mean()),
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
            "min_h": float(metrics["min_h"].min()),
        }
        self.records.append(row)
        if self.verbose:
            print(
                "[eval]"
                f" k0={self.k0:g} k1={self.k1:g}"
                f" steps={self.num_timesteps:,}"
                f" return={row['return']:.2f}"
                f" ego_col={row['ego_collisions']:.2f}"
                f" speed_dev={row['mean_signed_speed_deviation']:.3f}"
                f" abs_dev={row['mean_abs_speed_deviation']:.3f}"
                f" intervention={row['intervention_rate']:.2%}"
                f" qp_fail={row['qp_failure_rate']:.3%}",
                flush=True,
            )
        return True


def set_cbf_gains(env: Any, k0: float, k1: float) -> None:
    """Set gains on the SafetyFilteredAccelerationWrapper, possibly wrapped by Monitor."""
    current = env
    while current is not None:
        if hasattr(current, "k0") and hasattr(current, "k1"):
            current.k0 = float(k0)
            current.k1 = float(k1)
            return
        current = getattr(current, "env", None)
    raise RuntimeError("Could not find SafetyFilteredAccelerationWrapper to set k0/k1.")


def make_cbf_eval_env_for_gains(namespace: dict[str, Any], seed: int, k0: float, k1: float) -> Any:
    env = namespace["make_cbf_single_env"](
        seed=seed,
        lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
    )
    set_cbf_gains(env, k0, k1)
    namespace["configure_paper_evaluation_env"](env, steps=namespace["PAPER_EVAL_STEPS"])
    return env


def evaluate_cbf_policy_for_gains(
    namespace: dict[str, Any],
    model: DDPG,
    k0: float,
    k1: float,
    episodes: int,
    seed: int,
    deterministic: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for episode in range(episodes):
        env = make_cbf_eval_env_for_gains(namespace, seed + episode, k0, k1)
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
        min_h_values: list[float] = []
        ego_collisions = 0
        ego_collision_steps = 0
        all_collision_events = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            base = env.unwrapped
            desired = float(base.vehicle.desired_speed)
            speed = float(base.vehicle.vx)
            deviation = speed - desired

            rewards.append(float(reward))
            signed_deviations.append(deviation)
            abs_deviations.append(abs(deviation))
            speeds.append(speed)
            corrections.append(float(info.get("cbf_correction_norm", 0.0)))
            interventions.append(float(info.get("cbf_intervened", False)))
            qp_successes.append(float(info.get("cbf_qp_success", True)))
            min_h_values.append(float(info.get("cbf_min_h", np.nan)))
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
                "min_h": float(np.nanmin(min_h_values)) if min_h_values and not np.all(np.isnan(min_h_values)) else np.nan,
            }
        )
        env.close()
    return pd.DataFrame(rows)


def action_trace(namespace: dict[str, Any], model: DDPG, k0: float, k1: float, seed: int, episodes: int) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for episode in range(episodes):
        env = make_cbf_eval_env_for_gains(namespace, seed + episode, k0, k1)
        obs, _ = env.reset(seed=seed + episode)
        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            base = env.unwrapped
            rows.append(
                {
                    "episode": float(episode),
                    "step": float(step),
                    "speed": float(base.vehicle.vx),
                    "desired_speed": float(base.vehicle.desired_speed),
                    "a_rl_x": float(info.get("cbf_a_rl_x", np.nan)),
                    "a_rl_y": float(info.get("cbf_a_rl_y", np.nan)),
                    "a_safe_x": float(info.get("cbf_a_safe_x", np.nan)),
                    "a_safe_y": float(info.get("cbf_a_safe_y", np.nan)),
                    "correction_norm": float(info.get("cbf_correction_norm", np.nan)),
                    "intervened": float(info.get("cbf_intervened", False)),
                    "qp_success": float(info.get("cbf_qp_success", True)),
                    "ego_collision": float(info.get("ego_collision", False)),
                }
            )
            done = bool(terminated or truncated)
            step += 1
        env.close()
    return pd.DataFrame(rows)


def summarize_final(metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "steps": float(metrics["steps"].mean()),
        "return": float(metrics["return"].mean()),
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
        "min_h": float(metrics["min_h"].min()),
    }


def main() -> None:
    repo_root = find_repo_root()
    namespace = load_notebook_namespace(repo_root)

    artifact_dir: Path = namespace["ARTIFACT_DIR"]
    sweep_dir = artifact_dir / "cbf_gain_tuning_training_50k_noslack"
    model_dir = sweep_dir / "models"
    history_dir = sweep_dir / "train_eval_history"
    final_dir = sweep_dir / "final_eval"
    action_dir = sweep_dir / "action_trace"
    for directory in [model_dir, history_dir, final_dir, action_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    seed = int(namespace["SEED"])
    partial_path = sweep_dir / "summary_partial.csv"
    if partial_path.exists():
        existing_summary = pd.read_csv(partial_path)
        summary_rows: list[dict[str, float | str]] = existing_summary.to_dict("records")
        completed_gains = {
            (round(float(row["k0"]), 8), round(float(row["k1"]), 8))
            for row in summary_rows
        }
    else:
        summary_rows = []
        completed_gains = set()
    print(
        "Starting DDPG-CBF gain sweep",
        {
            "timesteps_per_config": TRAINING_TIMESTEPS,
            "final_eval_episodes": FINAL_EVAL_EPISODES,
            "gain_candidates": GAIN_CANDIDATES,
            "output_dir": str(sweep_dir),
            "obs_vmax": namespace["ENV_CONFIG"]["observation_vmax"],
            "reward_wx": namespace["REWARD_CONFIG"]["wx"],
            "overtake_bonus": namespace["REWARD_CONFIG"]["overtake_bonus"],
            "lambda_filter": namespace["CBF_FILTER_REWARD_LAMBDA"],
        },
        flush=True,
    )

    for index, (k0, k1) in enumerate(GAIN_CANDIDATES, start=1):
        tag = gain_tag(k0, k1, seed)
        model_path = model_dir / f"ddpg_cbf_{tag}.zip"
        history_path = history_dir / f"history_{tag}.csv"
        final_path = final_dir / f"final_eval_{tag}.csv"
        action_path = action_dir / f"action_trace_{tag}.csv"

        gain_key = (round(float(k0), 8), round(float(k1), 8))
        if gain_key in completed_gains:
            print(f"\n=== [{index}/{len(GAIN_CANDIDATES)}] k0={k0:g}, k1={k1:g} already complete; skipping ===", flush=True)
            continue

        print(f"\n=== [{index}/{len(GAIN_CANDIDATES)}] k0={k0:g}, k1={k1:g} ===", flush=True)
        if model_path.exists():
            print(f"Found existing model for {tag}; loading and evaluating without retraining.", flush=True)
            model = DDPG.load(str(model_path), device=namespace["DEVICE"])
            elapsed = float("nan")
        else:
            start = time.time()
            train_env = namespace["make_cbf_training_env"](
                seed=seed,
                lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
            )
            # The wrapper is constructed with default gains from the override cell; set them explicitly per run.
            set_cbf_gains(train_env.envs[0], k0, k1)

            n_actions = train_env.action_space.shape[-1]
            noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions, dtype=np.float32),
                sigma=namespace["DDPG_OU_SIGMA"] * np.ones(n_actions, dtype=np.float32),
            )
            callback = GainSweepCallback(namespace, k0, k1, seed, verbose=1)
            model = DDPG(
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
            )
            model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=callback, progress_bar=False)
            elapsed = time.time() - start
            model.save(str(model_path))
            train_env.close()

            pd.DataFrame(callback.records).to_csv(history_path, index=False)
            print(f"Training complete for {tag} in {elapsed / 60:.1f} min", flush=True)

        final_metrics = evaluate_cbf_policy_for_gains(
            namespace,
            model,
            k0,
            k1,
            episodes=FINAL_EVAL_EPISODES,
            seed=seed + 100_000 + 10_000 * index,
            deterministic=True,
        )
        final_metrics.to_csv(final_path, index=False)
        trace = action_trace(namespace, model, k0, k1, seed + 500_000 + 10_000 * index, ACTION_TRACE_EPISODES)
        trace.to_csv(action_path, index=False)

        final_summary = summarize_final(final_metrics)
        action_summary = {
            "mean_a_rl_x": float(trace["a_rl_x"].mean()),
            "mean_a_rl_y": float(trace["a_rl_y"].mean()),
            "mean_a_safe_x": float(trace["a_safe_x"].mean()),
            "mean_a_safe_y": float(trace["a_safe_y"].mean()),
            "action_qp_failure_rate": float(1.0 - trace["qp_success"].mean()),
            "action_ego_collision_rate": float(trace["ego_collision"].mean()),
        }
        row: dict[str, float | str] = {
            "k0": float(k0),
            "k1": float(k1),
            "seed": float(seed),
            "timesteps": float(TRAINING_TIMESTEPS),
            "elapsed_min": float(elapsed / 60.0),
            "model_path": str(model_path),
            "history_path": str(history_path),
            "final_eval_path": str(final_path),
            "action_trace_path": str(action_path),
            **final_summary,
            **action_summary,
        }
        summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(partial_path, index=False)
        print(
            "[final]"
            f" k0={k0:g} k1={k1:g}"
            f" return={final_summary['return']:.2f}"
            f" ego_col={final_summary['ego_collisions']:.3f}"
            f" speed_dev={final_summary['mean_signed_speed_deviation']:.3f}"
            f" abs_dev={final_summary['mean_abs_speed_deviation']:.3f}"
            f" intervention={final_summary['intervention_rate']:.2%}"
            f" qp_fail={final_summary['qp_failure_rate']:.3%}"
            f" mean_a_rl_x={action_summary['mean_a_rl_x']:.3f}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = sweep_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    ranked = summary.sort_values(
        by=["ego_collisions", "mean_abs_speed_deviation", "qp_failure_rate", "return"],
        ascending=[True, True, True, False],
    )
    ranked_path = sweep_dir / "summary_ranked.csv"
    ranked.to_csv(ranked_path, index=False)
    print("\n=== SWEEP COMPLETE ===", flush=True)
    print(ranked.to_string(index=False), flush=True)
    print("Saved summary:", summary_path, flush=True)
    print("Saved ranked summary:", ranked_path, flush=True)

    best = ranked.iloc[0]
    best_model = Path(str(best["model_path"]))
    promoted = artifact_dir / "ddpg_cbf_flat42_vmax24_noslack_tuned_laneless_karalakou_best_gain_sweep.zip"
    shutil.copy2(best_model, promoted)
    print("Promoted best model copy:", promoted, flush=True)


if __name__ == "__main__":
    main()
