import math
import random
from collections import deque, namedtuple
from itertools import count
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_environment import SegmentationEnv

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

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

    def forward(
        self, state_input: torch.Tensor, gcn_features: torch.Tensor
    ) -> torch.Tensor:

        # Combine inputs
        x = torch.cat([state_input, gcn_features], dim=1)

        # Forward pass
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x
