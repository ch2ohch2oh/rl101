import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt

gamma = 0.99


class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.Softmax(dim=-1),
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        return self.model(x)

    def act(self, state):
        x = torch.from_numpy(state)
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()


def train(pi, optimizer):
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32)
    future_ret = 0.0

    for i in reversed(range(T)):
        future_ret = pi.rewards[i] + gamma * future_ret
        rets[i] = future_ret

    rets = torch.tensor(rets)
    rets = (rets - rets.mean()) / (rets.std() + 1e-8)

    log_probs = torch.stack(pi.log_probs)

    loss = -(log_probs * rets).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def plot_training_results(
    episodes,
    episode_rewards,
    episode_losses,
    filename="reinforce_training.png",
    window_size=50,
):
    """Plot training results showing rewards and loss over episodes."""
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
    ax1.axhline(y=195, color="r", linestyle="--", label="Solved Threshold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("REINFORCE Training: Episode Rewards")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot loss
    ax2.plot(episodes, episode_losses, "g-", alpha=0.7, label="Loss")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.set_title("REINFORCE Training: Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary statistics
    print("\nTraining completed!")
    print(
        f"Average reward over last 100 episodes: {np.mean(episode_rewards[-100:]):.2f}"
    )
    print(f"Plot saved as '{filename}'")


def main():
    env = gym.make("CartPole-v1")
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n
    pi = Pi(in_dim, out_dim)
    optimizer = optim.Adam(pi.parameters(), lr=1e-3)

    # Lists to store metrics for plotting
    episode_losses = []
    episode_rewards = []
    episodes = []

    for epi in range(1000):
        state, _ = env.reset()
        for t in range(100000):
            action = pi.act(state)
            state, reward, done, truncated, _ = env.step(action)
            pi.rewards.append(reward)

            if done or truncated:
                break

        loss = train(pi, optimizer)
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset()

        # Store metrics
        episodes.append(epi)
        episode_losses.append(loss)
        episode_rewards.append(total_reward)

        print(
            f"Episode {epi}, loss: {loss:.2f}, total_reward: {total_reward}, solved: {solved}"
        )

    # Plot training results
    plot_training_results(episodes, episode_rewards, episode_losses)


if __name__ == "__main__":
    main()
