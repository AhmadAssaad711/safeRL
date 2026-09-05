from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise


TRAIN_EVAL_FREQ = 10_000
TRAIN_EVAL_EPISODES = 5


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
        "import os",
        "class KaralakouRewardWrapper",
        "ENV_CONFIG = {",
        "class LaneFreeObservationNormalizationWrapper",
        "def evaluate_policy_with_metrics",
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

    return namespace


def summarize(metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "steps": float(metrics["steps"].mean()),
        "return": float(metrics["return"].mean()),
        "mean_speed": float(metrics["mean_speed"].mean()),
        "mean_signed_speed_deviation": float(metrics["mean_signed_speed_deviation"].mean()),
        "mean_abs_speed_deviation": float(metrics["mean_abs_speed_deviation"].mean()),
        "ego_collisions": float(metrics["ego_collisions"].mean()),
        "ego_collision_steps": float(metrics["ego_collision_steps"].mean()),
        "total_collision_events": float(metrics["total_collision_events"].mean()),
    }


def main() -> None:
    repo_root = find_repo_root()
    ns = load_notebook_namespace(repo_root)

    seed = int(ns["SEED"])
    artifact_dir: Path = ns["ARTIFACT_DIR"]
    model_path: Path = ns["DDPG_MODEL_PATH"]
    history_path: Path = ns["DDPG_HISTORY_PATH"]
    final_eval_path = artifact_dir / f"{model_path.stem}_final_eval.csv"
    summary_path = artifact_dir / f"{model_path.stem}_summary.csv"

    preflight_env = ns["make_training_env"](seed=seed)
    preflight_obs = preflight_env.reset()
    print("Project root:", repo_root, flush=True)
    print("Device:", ns["DEVICE"], flush=True)
    print("NORMALIZE_RL_OBSERVATIONS:", ns["NORMALIZE_RL_OBSERVATIONS"], flush=True)
    print("Model path:", model_path, flush=True)
    print("Observation space:", preflight_env.observation_space, flush=True)
    print("Reset obs shape:", preflight_obs.shape, flush=True)
    print("First two 7-feature rows:", preflight_obs[0, :14].reshape(2, 7), flush=True)
    preflight_env.close()

    train_env = ns["make_training_env"](seed=seed)
    n_actions = train_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions, dtype=np.float32),
        sigma=ns["DDPG_OU_SIGMA"] * np.ones(n_actions, dtype=np.float32),
    )
    callback = ns["EvalMetricsCallback"](
        eval_freq=TRAIN_EVAL_FREQ,
        n_eval_episodes=TRAIN_EVAL_EPISODES,
        seed=seed + 20_000,
        verbose=1,
    )

    model = DDPG(
        "MlpPolicy",
        train_env,
        learning_rate=ns["DDPG_LEARNING_RATE"],
        buffer_size=ns["DDPG_REPLAY_MEMORY"],
        learning_starts=ns["DDPG_LEARNING_STARTS"],
        batch_size=ns["DDPG_BATCH_SIZE"],
        tau=ns["DDPG_TAU"],
        gamma=ns["DDPG_GAMMA"],
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
        policy_kwargs={"net_arch": [256, 128]},
        tensorboard_log=str(artifact_dir / "tensorboard"),
        verbose=1,
        seed=seed,
        device=ns["DEVICE"],
    )

    start = time.time()
    model.learn(total_timesteps=int(ns["DDPG_TOTAL_TIMESTEPS"]), callback=callback, progress_bar=False)
    elapsed = time.time() - start
    model.save(str(model_path))
    train_env.close()

    history = pd.DataFrame(callback.records)
    history.to_csv(history_path, index=False)

    final_metrics = ns["evaluate_policy_with_metrics"](
        model,
        episodes=int(ns["FINAL_EVAL_EPISODES"]),
        seed=seed + 300_000,
        deterministic=True,
    )
    final_metrics.to_csv(final_eval_path, index=False)

    summary = {
        "timesteps": float(ns["DDPG_TOTAL_TIMESTEPS"]),
        "elapsed_min": float(elapsed / 60.0),
        "model_path": str(model_path),
        "history_path": str(history_path),
        "final_eval_path": str(final_eval_path),
        **summarize(final_metrics),
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("Training complete in %.1f min" % (elapsed / 60.0), flush=True)
    print("Saved model:", model_path, flush=True)
    print("Saved history:", history_path, flush=True)
    print("Saved final eval:", final_eval_path, flush=True)
    print("Saved summary:", summary_path, flush=True)
    print(pd.DataFrame([summary]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
