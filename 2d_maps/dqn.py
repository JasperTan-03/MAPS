import random
from collections import deque, namedtuple
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the transition tuple
Transition = namedtuple(
    "Transition", ("state", "action", "classification", "next_state", "reward")
)


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> Tuple:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQN(nn.Module):
    def __init__(
        self,
        num_observations: int,
        num_actions: int,
        gcn_latent_dim: int,
        hidden_dim: int,
    ):
        super(DQN, self).__init__()

        #  Combined input size
        combined_input_size = num_observations + gcn_latent_dim

        # DQN layers
        self.layer1 = nn.Linear(combined_input_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x
