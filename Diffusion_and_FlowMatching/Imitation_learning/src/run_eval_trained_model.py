"""Load a trained Push-T policy, run one episode, and save a video."""

from __future__ import annotations

import argparse
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from model import build_policy
from evaluation import ENV_ID

# Push-T has a fixed state/action space
STATE_DIM = 5
ACTION_DIM = 2


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    normalizer = checkpoint["normalizer"]
    model = build_policy(
        config["policy_type"],
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        chunk_size=config["chunk_size"],
        hidden_dims=tuple(config["hidden_dims"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, normalizer, config


def resize_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).resize(size, resample=Image.BILINEAR))


def collect_episode_frames(
    model,
    normalizer,
    config: dict,
    device: torch.device,
    seed: int = 0,
    video_size: tuple[int, int] = (256, 256),
) -> tuple[list[np.ndarray], float]:
    """Run one episode and return (frames, max_reward)."""
    chunk_size = config["chunk_size"]
    flow_num_steps = config["flow_num_steps"]

    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low
    action_high = env.action_space.high

    obs, _ = env.reset(seed=seed)
    done = False
    chunk_index = chunk_size
    action_chunk: np.ndarray | None = None
    frames: list[np.ndarray] = []
    max_reward = 0.0

    while not done:
        if action_chunk is None or chunk_index >= chunk_size:
            state = torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
            with torch.no_grad():
                pred_chunk = (
                    model.sample_actions(state.unsqueeze(0), num_steps=flow_num_steps)
                    .cpu()
                    .numpy()[0]
                )
            action_chunk = normalizer.denormalize_action(pred_chunk)
            action_chunk = np.clip(action_chunk, action_low, action_high)
            chunk_index = 0

        action = action_chunk[chunk_index]
        obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
        frames.append(resize_frame(env.render(), video_size))
        max_reward = max(max_reward, float(reward))
        done = terminated or truncated
        chunk_index += 1

    env.close()
    return frames, max_reward


def run_episode(
    checkpoint_path: Path,
    output_path: Path,
    seed: int = 0,
    video_size: tuple[int, int] = (256, 256),
    num_episodes: int = 1,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, normalizer, config = load_checkpoint(checkpoint_path, device)
    print(f"Loaded {config['policy_type']} policy (chunk_size={config['chunk_size']})")

    all_frames: list[list[np.ndarray]] = []
    for i in range(num_episodes):
        frames, max_reward = collect_episode_frames(
            model, normalizer, config, device, seed=seed, video_size=video_size
        )
        print(f"Episode {i}: steps={len(frames)}, max_reward={max_reward:.4f}")
        all_frames.append(frames)

    # Pad shorter episodes with their last frame
    max_len = max(len(f) for f in all_frames)
    for frames in all_frames:
        while len(frames) < max_len:
            frames.append(frames[-1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(output_path), fps=20, codec="libx264", macro_block_size=1
    ) as writer:
        for t in range(max_len):
            tiled = np.concatenate([f[t] for f in all_frames], axis=1)
            writer.append_data(tiled)
    print(f"Video saved to {output_path}")

def flow_integration_trajectory(
    model,
    normalizer,
    obs: np.ndarray,
    device: torch.device,
    num_steps: int = 10,
) -> list[np.ndarray]:
    """Trace the flow matching integration from noise to final action chunk.

    Runs the Euler integration step by step, capturing the action chunk after
    each step. Only meaningful for FlowMatchingPolicy.

    Args:
        model: Loaded FlowMatchingPolicy (uncompiled).
        normalizer: Normalizer from the checkpoint.
        obs: Initial environment observation (state).
        device: Torch device.
        num_steps: Number of Euler integration steps.

    Returns:
        List of (chunk_size, action_dim) arrays in the original action space,
        length = num_steps + 1:
            [initial_noise, after_step_1, ..., after_step_num_steps]
    """
    state = (
        torch.from_numpy(normalizer.normalize_state(obs))
        .float()
        .to(device)
        .unsqueeze(0)
    )
    chunk_size = model.chunk_size
    action_dim = model.action_dim
    dt = 1.0 / num_steps

    # Sample initial noise A_0 ~ N(0, I)
    action = torch.randn(1, chunk_size, action_dim, device=device)

    trajectory: list[np.ndarray] = [
        normalizer.denormalize_action(action.cpu().numpy()[0])
    ]

    with torch.no_grad():
        for step in range(num_steps):
            tau = torch.full((1, 1), step * dt, device=device)
            network_input = torch.cat([state, action.view(1, -1), tau], dim=1)
            velocity = model.mlp(network_input).view(1, chunk_size, action_dim)
            action = action + dt * velocity
            trajectory.append(normalizer.denormalize_action(action.cpu().numpy()[0]))

    return trajectory



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Push-T policy.")
    parser.add_argument("checkpoint", type=Path, help="Path to the .pt checkpoint file.")
    parser.add_argument("--output", type=Path, default=Path("eval_video.mp4"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--video-size", type=int, nargs=2, default=[256, 256], metavar=("W", "H")

    )
    parser.add_argument(
        "--num_tests", type=int, default=5, metavar=("N")
    )
    parser.add_argument(
        "--visualize-flow", action="store_true",
        help="Visualize the flow integration trajectory instead of running a full episode.",
    )
    parser.add_argument("--flow-steps", type=int, default=5, help="Number of Euler integration steps.")
    args = parser.parse_args()

    if args.visualize_flow:
        import matplotlib.pyplot as plt

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, normalizer, _ = load_checkpoint(args.checkpoint, device)

        env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
        num_tests = args.num_tests

        _, axes = plt.subplots(1, num_tests, figsize=(5 * num_tests, 5))
        if num_tests == 1:
            axes = [axes]

        for i in range(num_tests):
            obs, _ = env.reset(seed=args.seed)
            trajectory = flow_integration_trajectory(
                model, normalizer, obs, device, num_steps=args.flow_steps
            )
            ax = axes[i]
            for chunk in trajectory[:-1]:
                ax.plot(chunk[:, 0], chunk[:, 1], color="steelblue", linewidth=0.8, alpha=0.3)
            final = trajectory[-1]
            ax.plot(final[:, 0], final[:, 1], color="red", linewidth=2.5,
                    marker="o", markersize=4, label="final")
            ax.set_title(f"Run {i} ")
            ax.set_xlabel("action x")
            ax.set_ylabel("action y")
            ax.legend()

        env.close()
        plt.tight_layout()
        plot_path = args.output.with_suffix(".png")
        plt.savefig(plot_path)
        print(f"Flow trajectory plot saved to {plot_path}")
        plt.show()
    else:
        run_episode(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            seed=args.seed,
            video_size=tuple(args.video_size),
            num_episodes=args.num_tests,
        )


if __name__ == "__main__":
    main()
