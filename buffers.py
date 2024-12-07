# Rollout Buffer

import torch
from ppo import PPO
"""
This file contains the implementation of the Rollout Buffer.
The Rollout Buffer is used to store the experiences of the agent during the training phase.
"""
class RolloutBuffer:
    """
    Buffer to store rollout data for PPO.
    """
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.returns = []
        self.advantages = []

    def store(self, obs, action, log_prob, reward, done):
        """
        Stores a single step of rollout data.
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns_and_advantages(self, policy, gamma, gae_lambda):
        """
        Computes returns and advantages after a rollout.
        """
        # Compute values using the policy's value network
        values = [
            policy.forward_value(torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0).item()
            for obs in self.observations
        ]
        next_value = 0 if self.dones[-1] else values[-1]
        
        # Use PPO's compute_returns_and_advantages with explicit gamma and gae_lambda
        self.returns, self.advantages = PPO.compute_returns_and_advantages(
            rewards=self.rewards,
            values=values,
            dones=self.dones,
            next_value=next_value,
            gamma=gamma,
            gae_lambda=gae_lambda
        )

    def clear(self):
        """
        Clears the buffer.
        """
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.returns = []
        self.advantages = []
