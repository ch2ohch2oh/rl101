import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from epsilon_scheduler import EpsilonScheduler, LinearEpsilonScheduler
from plot_utils import plot_training_results


env = gym.make("CartPole-v1")


class QNet(nn.Module):
    def __init__(self, state_size=4, action_size=2, hidden_size=32):
        super(QNet, self).__init__()
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x  # Return Q-values, not softmax for Q-learning


def select_action(qnet, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(qnet.action_size)
    with torch.no_grad():
        q_values = qnet.forward(torch.tensor(state, dtype=torch.float32))
        return int(torch.argmax(q_values, dim=-1).item())


def sarsa_episodic(
    qnet,
    optimizer,
    epsilon_scheduler: EpsilonScheduler,
    n_episodes=10000,
    max_t=1000,
    gamma=0.99,
    print_every=100,
    success_threshold=475,
):
    """Episodic SARSA - accumulates experience over full episodes before updating."""
    rolling_rewards = deque(maxlen=200)
    episodes = []
    episode_rewards = []
    episode_losses = []

    for e in range(1, n_episodes + 1):
        epsilon = epsilon_scheduler.get_epsilon(e)
        state, _ = env.reset()
        action = select_action(qnet, state, epsilon)

        # Store episode experience
        states = []
        actions = []
        rewards = []
        next_states = []
        next_actions = []
        dones = []

        for t in range(max_t):
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = (
                select_action(qnet, next_state, epsilon)
                if not (done or truncated)
                else None
            )

            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            next_actions.append(next_action)
            dones.append(done or truncated)

            if done or truncated:
                break

            state = next_state
            action = next_action

        # Episodic update - process all experience at once
        total_loss = 0
        for i in range(len(states)):
            state_tensor = torch.tensor(states[i], dtype=torch.float32)
            current_q = qnet.forward(state_tensor)[actions[i]]

            if dones[i]:
                target_q = rewards[i]
            else:
                with torch.no_grad():
                    next_state_tensor = torch.tensor(
                        next_states[i], dtype=torch.float32
                    )
                    next_q = qnet.forward(next_state_tensor)[next_actions[i]]
                    target_q = rewards[i] + gamma * next_q

            loss = (current_q - target_q) ** 2
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_reward = sum(rewards)
        rolling_rewards.append(total_reward)

        episodes.append(e)
        episode_rewards.append(total_reward)
        episode_losses.append(total_loss / len(states) if states else 0)

        if e % print_every == 0:
            print(
                "Episode {}\tEpsilon={:.4f}\tAverage Reward: {:.2f}\tEpisode Loss: {:.4f}".format(
                    e, epsilon, np.mean(rolling_rewards), episode_losses[-1]
                )
            )
        if (
            len(rolling_rewards) >= 100
            and np.mean(rolling_rewards) >= success_threshold
        ):
            print("Environment solved!")
            break

    return episodes, episode_rewards, episode_losses


if __name__ == "__main__":
    # Training hyperparameters
    lr = 1e-3
    gamma = 1
    hidden_size = 8
    action_size = 2

    # Create epsilon scheduler
    epsilon_scheduler = LinearEpsilonScheduler(
        start_epsilon=1.0, end_epsilon=0.05, decay_episodes=2000
    )

    qnet = QNet(action_size=action_size, hidden_size=hidden_size)
    optimizer = optim.Adam(qnet.parameters(), lr=lr)
    episodes, episode_rewards, episode_losses = sarsa_episodic(
        qnet, optimizer, epsilon_scheduler, n_episodes=10000, gamma=1
    )

    # Prepare training parameters for display
    training_params = {
        "lr": lr,
        "gamma": gamma,
        "hidden_size": hidden_size,
        "algorithm": "SARSA Episodic",
        "epsilon_scheduler": str(epsilon_scheduler),
    }

    plot_training_results(
        episodes,
        episode_rewards,
        episode_losses,
        training_params,
        filename="cartpole_sarsa_episodic.png",
    )
