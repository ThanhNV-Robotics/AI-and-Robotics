"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from torch.utils.data import DataLoader

from data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from model import build_policy, PolicyType
from evaluation import Logger, evaluate_policy

LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "flow"
    # policy_type: PolicyType = "mse"
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    # The action chunk size.
    chunk_size: int = 10

    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (256, 256, 256) # 3 hidden layers with 256 units each
    # hidden_dims: tuple[int, ...] = (256, 256) # 2 hidden layers with 256 units each, test changing the network's parameters
    # The number of epochs to train for.
    num_epochs: int = 400
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "hw1-imitation"
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    ) # load the dataset of (state, action_chunk) pairs using a sliding window approach, where each action_chunk has length equal to config.chunk_size

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    ) # create a DataLoader to iterate through the dataset in batches for training

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device) # build the policy model based on the specified policy type and move it to the appropriate device (CPU or GPU)
    
    # Compile model for faster training (PyTorch 2.0+)
    print("Compiling model with torch.compile...")
    model = torch.compile(model)
    print("Model compiled successfully!")

    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    logger = Logger(log_dir)
    
    ### TODO: PUT YOUR MAIN TRAINING LOOP HERE ###
    # The main training loop should iterate through the DataLoader, compute the loss using model.compute_loss(), and update the model parameters using an optimizer. You should also log training metrics to WandB and run evaluation using the logger at the specified intervals.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    model.train() # set model to training mode
    global_step = 0
    
    # Initialize CSV header with all columns we want to track
    # This ensures both eval/mean_reward and train_loss are in the CSV from the start
    logger.log({"train_loss": 0.0, "eval/mean_reward": 0.0}, step=-1)
    
    for epoch in range(config.num_epochs):
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch in loader:
            state, action_chunk = batch
            state = state.to(device)
            action_chunk = action_chunk.to(device)

            loss = model.compute_loss(state, action_chunk) # compute the loss for the current batch using the model's compute_loss method

            optimizer.zero_grad() # zero the gradients before backpropagation
            loss.backward() # backpropagate the loss to compute gradients
            optimizer.step() # update the model parameters using the optimizer

            # Run evaluation after a specific number of steps
            if global_step % config.eval_interval == 0:
                evaluate_policy(
                    model=model,
                    normalizer=normalizer,
                    device=device,
                    chunk_size=config.chunk_size,
                    flow_num_steps=config.flow_num_steps,
                    step=global_step,
                    logger=logger,
                ) # run evaluation at the specified intervals
                model.train() # set model back to training mode after evaluation

            if global_step % config.log_interval == 0:
                logger.log({"train_loss": loss.item()}, step=global_step) # log the training loss using the logger

            global_step += 1
    logger.dump_for_grading()

    # save the final trained model
    save_dir = Path("trained_model")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"{exp_name}.pt"
    # torch.compile wraps the model; _orig_mod holds the original module
    orig_model = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "model_state_dict": orig_model.state_dict(),
            "normalizer": normalizer,
            "config": config_to_dict(config),
        },
        save_path,
    )
    print(f"Model saved to {save_path}")



def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
