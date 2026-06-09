"""Plot training curves from log.csv file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(log_path: Path, save_path: Path | None = None) -> None:
    """Plot training loss and evaluation reward curves.
    
    Args:
        log_path: Path to the log.csv file
        save_path: Optional path to save the figure
    """
    # Load data
    df = pd.read_csv(log_path)
    print(f"Loaded {len(df)} log entries from {log_path}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Training Loss
    if 'train_loss' in df.columns:
        train_data = df[df['train_loss'].notna()]
        ax1.plot(train_data['step'], train_data['train_loss'], 
                     linewidth=1.5, alpha=0.7, color='blue')
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Training Loss (MSE)', fontsize=12)
        ax1.set_title('Training Loss over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
    
    # Plot 2: Evaluation Reward
    if 'eval/mean_reward' in df.columns:
        eval_data = df[df['eval/mean_reward'].notna()]
        ax2.plot(eval_data['step'], eval_data['eval/mean_reward'], 
                marker='o', markersize=6, linewidth=2, color='green')
        ax2.set_xlabel('Training Steps', fontsize=12)
        ax2.set_ylabel('Mean Evaluation Reward', fontsize=12)
        ax2.set_title('Evaluation Performance over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)
        
        # Add horizontal line for final performance
        final_reward = eval_data['eval/mean_reward'].iloc[-1]
        ax2.axhline(y=final_reward, color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.5, 
                   label=f'Final: {final_reward:.4f}')
        ax2.legend()
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    # Print summary statistics
    print("\n=== Training Summary ===")
    if 'train_loss' in df.columns:
        train_data = df[df['train_loss'].notna()]
        print(f"Training Loss:")
        print(f"  Initial: {train_data['train_loss'].iloc[0]:.6f}")
        print(f"  Final:   {train_data['train_loss'].iloc[-1]:.6f}")
        print(f"  Min:     {train_data['train_loss'].min():.6f}")
        print(f"  Mean:    {train_data['train_loss'].mean():.6f}")
    
    if 'eval/mean_reward' in df.columns:
        eval_data = df[df['eval/mean_reward'].notna()]
        print(f"\nEvaluation Reward:")
        print(f"  Initial: {eval_data['eval/mean_reward'].iloc[0]:.6f}")
        print(f"  Final:   {eval_data['eval/mean_reward'].iloc[-1]:.6f}")
        print(f"  Max:     {eval_data['eval/mean_reward'].max():.6f}")
        print(f"  Mean:    {eval_data['eval/mean_reward'].mean():.6f}")
        print(f"\nTotal training steps: {df['step'].max()}")
        print(f"Number of evaluations: {len(eval_data)}")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from log.csv")
    parser.add_argument(
        "--log-path",
        type=Path,
        required=True,
        help="Path to log.csv file",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the figure (e.g., training_curves.png)",
    )
    
    args = parser.parse_args()
    plot_training_curves(args.log_path, args.save)


if __name__ == "__main__":
    main()
