from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from plot_utils import plot_training_results


env = gym.make("CartPole-v1")


class Policy(nn.Module):
    def __init__(self, state_size=4, action_size=2, hidden_size=32):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        model = Categorical(probs)
        action = model.sample()
        return action.item(), model.log_prob(action)


def reinforce(
    policy,
    optimizer,
    n_episodes=1000,
    max_t=1000,
    gamma=1.0,
    print_every=100,
    success_threshold=475,
):
    rolling_scores = deque(maxlen=200)  # Discount factor not applied
    episodes = []
    episode_rewards = []
    episode_losses = []
    for e in range(1, n_episodes):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()
        # Collect trajectory
        for t in range(max_t):
            # Sample the action from current policy
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)
            if done or truncated:
                break
        # Calculate total expected reward
        rolling_scores.append(sum(rewards))

        # Recalculate the total reward applying discounted factor
        discounts = [gamma**i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        # Calculate the loss
        policy_loss = []
        for log_prob in saved_log_probs:
            # Note that we are using Gradient Ascent, not Descent. So we need to calculate it with negative rewards.
            policy_loss.append(-log_prob * R)
        # After that, we concatenate whole policy loss in 0th dimension
        policy_loss = torch.cat(policy_loss).sum()

        episodes.append(e)
        episode_rewards.append(sum(rewards))
        episode_losses.append(policy_loss.item())

        # Backpropagation
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if e % print_every == 0:
            print(
                "Episode {}\tAverage Score: {:.2f}".format(e, np.mean(rolling_scores))
            )
        if np.mean(rolling_scores) >= success_threshold:
            print(
                "Environment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    e - 100, np.mean(rolling_scores)
                )
            )
            break
    return episodes, episode_rewards, episode_losses


if __name__ == "__main__":
    # Training hyperparameters
    lr = 1e-3
    gamma = 1
    hidden_size = 16
    
    policy = Policy(hidden_size=hidden_size)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    episodes, episode_rewards, episode_losses = reinforce(
        policy, optimizer, n_episodes=10000, gamma=gamma
    )

    # Prepare training parameters for display
    training_params = {
        'lr': lr,
        'gamma': gamma,
        'hidden_size': hidden_size,
    }

    plot_training_results(episodes, episode_rewards, episode_losses, training_params)
