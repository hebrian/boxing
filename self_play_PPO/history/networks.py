import torch
import torch.nn as nn
import numpy as np

class PPOAgent(nn.Module):
    """
    Implements a PPO agent with Value and Policy networks.
    """
    def __init__(self, obs_shape, action_space):
        """
        Initializes the agent (value and policy networks).
        
        Parameters:
        obs_shape (tuple): The shape of the observation space (C, H, W).
        action_space (gym.Space): The action space of the environment.
        """
        super(PPOAgent, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        
        # Dynamically calculate the feature map size
        feature_map_size = self.calculate_feature_map_size(obs_shape)
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_map_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_space.n),
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_map_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
    def forward_policy(self, x):
        """
        Forward pass of the policy network.
        Converts the observation to a probability distribution over actions.
        
        Parameters:
        x (torch.Tensor): The input observation.
        
        Returns:
        torch.Tensor: Probability distribution over actions.
        """
        x = self.conv_layers(x)
        return torch.softmax(self.policy_net(x), dim=-1)
    
    def forward_value(self, x):
        """
        Forward pass of the value network.
        Converts the observation to an estimated value.
        
        Parameters:
        x (torch.Tensor): The input observation.
        
        Returns:
        torch.Tensor: Estimated value of the input state.
        """
        x = self.conv_layers(x)
        return self.value_net(x)

    def calculate_feature_map_size(self, input_shape):
        """
        Calculates the feature map size after the convolutional layers.
        
        Parameters:
        input_shape (tuple): The shape of the input observation (C, H, W).
        
        Returns:
        int: Flattened size of the feature map.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # Batch size of 1
            output = self.conv_layers(dummy_input)
            print("Shape after conv layers:", output.size())  # Debugging output shape
        return int(np.prod(output.size()[1:]))  # Flatten the output shape
