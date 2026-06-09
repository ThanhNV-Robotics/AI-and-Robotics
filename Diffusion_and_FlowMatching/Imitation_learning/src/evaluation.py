"""Evaluation utilities for Push-T policies."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from tensorboardX import SummaryWriter

from data import Normalizer
from model import BasePolicy

ENV_ID = "gym_pusht/PushT-v0"
NUM_EVAL_EPISODES = 100


class Logger:
    """Logger for logging metrics."""

    def __init__(self, path: Path):
        if path.exists():
            raise FileExistsError(f"Log directory {path} already exists.")
        path.mkdir(parents=True)
        self.path = path
        self.csv_path = path / "log.csv"
        self.header = None
        self.rows = []
        self.writer = SummaryWriter(log_dir=str(path), flush_secs=10)

    def log(self, row: dict[str, Any], step: int) -> None:
        row["step"] = step
        if self.header is None:
            self.header = list(row.keys())
            with self.csv_path.open("w") as f:
                f.write(",".join(self.header) + "\n")
        with self.csv_path.open("a") as f:
            f.write(",".join([str(row.get(k, "")) for k in self.header]) + "\n")
        for k, v in row.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, global_step=step)
        self.rows.append(copy.deepcopy(row))

    def dump_for_grading(self) -> None:
        self.writer.close()


def evaluate_policy(
    model: BasePolicy,
    normalizer: Normalizer,
    device: torch.device,
    chunk_size: int,
    flow_num_steps: int,
    step: int,
    logger: Logger,
) -> None:
    """Evaluate a policy in the Push-T environment and log mean reward to log.csv."""
    # switches the model from training mode to evaluation/inference mode.
    model.eval()

    rewards: list[float] = []

    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low # lower bound of the action space
    action_high = env.action_space.high # upper bound of the action space

    for ep_idx in range(NUM_EVAL_EPISODES):
        obs, _ = env.reset(seed=ep_idx)
        done = False
        chunk_index = chunk_size
        action_chunk: np.ndarray | None = None
        max_reward = 0.0

        while not done:
            if action_chunk is None or chunk_index >= chunk_size:
                state = (
                    torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
                )
                with torch.no_grad():
                    pred_chunk = (
                        model.sample_actions(
                            state.unsqueeze(0), num_steps=flow_num_steps
                        )
                        .cpu()
                        .numpy()[0]
                    )
                action_chunk = normalizer.denormalize_action(pred_chunk)
                action_chunk = np.clip(action_chunk, action_low, action_high)
                chunk_index = 0

            action = action_chunk[chunk_index]
            obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
            max_reward = max(max_reward, float(reward))
            done = terminated or truncated
            chunk_index += 1

        rewards.append(max_reward)

    env.close()
    mean_reward = float(np.mean(rewards))
    print(f"[Eval @ step {step}] mean_reward={mean_reward:.4f}")
    logger.log({"eval/mean_reward": mean_reward}, step=step)
