import os

import matplotlib.pyplot as plt
import numpy as np


def get_next_filename(base_filename="reinforce_training.png"):
    """Generate the next available filename with auto-increment."""
    if not os.path.exists(base_filename):
        return base_filename

    # Split filename and extension
    name, ext = os.path.splitext(base_filename)

    # Find the next available number
    counter = 1
    while True:
        new_filename = f"{name}_{counter:03d}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1


def plot_training_results(
    episodes,
    episode_rewards,
    episode_losses,
    training_params=None,
    filename=None,
    window_size=50,
    success_threshold=475,
):
    """Plot training results showing rewards and loss over episodes."""
    filename = get_next_filename(filename)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Calculate running average
    def running_average(data, window):
        return np.convolve(data, np.ones(window) / window, mode="valid")

    running_avg = running_average(episode_rewards, window_size)
    running_avg_episodes = episodes[
        window_size - 1 :
    ]  # Adjust episode indices for running average

    # Plot rewards
    ax1.plot(episodes, episode_rewards, "b-", alpha=0.3, label="Episode Reward")
    ax1.plot(
        running_avg_episodes,
        running_avg,
        "b-",
        linewidth=2,
        label=f"Running Average ({window_size} episodes)",
    )
    ax1.axhline(
        y=success_threshold, color="r", linestyle="--", label="Solved Threshold"
    )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("{} Training: Episode Rewards".format(training_params['algorithm']))
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot loss
    ax2.plot(episodes, episode_losses, "g-", alpha=0.7, label="Loss")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.set_title("{} Training: Loss".format(training_params['algorithm']))
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Add training parameters text box if provided
    if training_params:
        param_text = "Training Parameters:\n"
        for key, value in training_params.items():
            if isinstance(value, float):
                param_text += f"{key}: {value:.2e}\n"
            else:
                param_text += f"{key}: {value}\n"

        # Add text box to the lower left of the rewards plot
        ax1.text(
            0.02,
            0.02,
            param_text.strip(),
            transform=ax1.transAxes,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=9,
            fontfamily="monospace",
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary statistics
    print("\nTraining completed!")
    print(
        f"Average reward over last 100 episodes: {np.mean(episode_rewards[-100:]):.2f}"
    )
    print(f"Plot saved as '{filename}'")
