import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from networks import PPOAgent
import numpy as np

class PPO:
    """
    Proximal Policy Optimization (PPO) implementation.
    """
    def __init__(self, obs_shape, action_space, lr=3e-4, gamma=0.99, epsilon=0.2, gae_lambda=0.95):
        self.policy = PPOAgent(obs_shape, action_space)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.gae_lambda = gae_lambda

    @staticmethod
    def compute_returns_and_advantages(rewards, values, dones, next_value, gamma, gae_lambda):
        """
        Computes returns and advantages using Generalized Advantage Estimation (GAE).

        Parameters:
        rewards (list): List of rewards for each step in the trajectory.
        values (list): List of value estimates for each step.
        dones (list): List of done flags for each step.
        next_value (float): Value estimate for the next state.
        gamma (float): Discount factor for rewards.
        gae_lambda (float): Lambda parameter for GAE.

        Returns:
        tuple: A tuple of (returns, advantages).
        """
        returns = []
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]
            returns.insert(0, gae + values[step])
        return returns, advantages


    def update(self, buffer):
        """
        Updates the policy and value networks using PPO loss.

        Parameters:
        buffer (RolloutBuffer): The buffer containing rollout data.
        """
        obs_array = np.array(buffer.observations, dtype=np.float32)  # Combine into a single numpy array
        obs = torch.tensor(obs_array).permute(0, 3, 1, 2) / 255.0  # Normalize and permute

        actions = torch.tensor(buffer.actions, dtype=torch.int64)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)
        returns = torch.tensor(buffer.returns, dtype=torch.float32)
        advantages = torch.tensor(buffer.advantages, dtype=torch.float32)

        for _ in range(10):  # Number of PPO epochs
            # Get new log probabilities and values
            new_probs = self.policy.forward_policy(obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            new_log_probs = torch.log(new_probs + 1e-8)
            values = self.policy.forward_value(obs).squeeze(-1)

            # Compute the ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Compute the clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(values, returns)

            # Total loss
            loss = policy_loss + 0.5 * value_loss

            # Gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        