import random
from collections import deque, namedtuple
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import k_hop_subgraph, to_dense_adj

# Define experience tuple structure
Experience = namedtuple(
    "Experience",
    (
        "state",
        "action_nav",
        "action_cls",
        "next_state",
        "reward_nav",
        "reward_cls",
        "done",
    ),
)


class ReplayBuffer:
    """Experience replay buffer with uniform sampling"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """Save an experience"""
        self.buffer.append(Experience(*args))

    def sample(self, batch_size: int) -> Tuple:
        experiences = random.sample(self.buffer, batch_size)
        return Experience(*zip(*experiences))

    def __len__(self) -> int:
        return len(self.buffer)


class DuelingGNN(nn.Module):
    """Graph Neural Network with dueling architecture"""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_hops: int = 2
    ):
        super(DuelingGNN, self).__init__()

        self.num_hops = num_hops
        self.convs = nn.ModuleList()

        # GCN layers
        self.convs = nn.ModuleList(
            [
                GCNConv(input_dim, hidden_dim),
                GCNConv(hidden_dim, hidden_dim),
                GCNConv(hidden_dim, output_dim),
            ]
        )

        self.layer_norms = nn.ModuleList(
            [
                nn.LayerNorm(hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LayerNorm(output_dim),
            ]
        )

    def get_subgraph(
        self, x: torch.Tensor, edge_index: torch.Tensor, current_node: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Get k-hop subgraph
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            current_node, self.num_hops, edge_index, relabel_nodes=True
        )

        # Get subgraph features
        sub_x = x[subset]

        return sub_x, sub_edge_index, mapping

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, current_node: int
    ) -> torch.Tensor:
        # Get k-hop subgraph
        sub_x, sub_edge_index, mapping = self.get_subgraph(x, edge_index, current_node)

        # Apply GCN layers
        for i, conv in enumerate(self.convs):
            sub_x = conv(sub_x, sub_edge_index)
            sub_x = F.relu(sub_x)
            sub_x = F.dropout(sub_x, p=0.1, training=self.training)
            sub_x = self.layer_norms[i](sub_x)

        batch = torch.zeros(sub_x.shape[0], dtype=torch.long)
        # put batch to same device as sub_x
        if sub_x.is_cuda:
            batch = batch.cuda(sub_x.device)

        # Pooling
        x = global_mean_pool(sub_x, batch=batch)

        return x


class DuelingGraphDQN(nn.Module):
    """Dueling DQN architecture with separate value and advantage streams"""

    def __init__(
        self,
        node_feature_dim: int,
        gnn_hidden_dim: int,
        gnn_output_dim: int,
        dqn_hidden_dim: int,
        num_classes: int,
    ):
        super(DuelingGraphDQN, self).__init__()

        # Feature extractor
        self.gnn = DuelingGNN(
            input_dim=node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
        )

        # Classification streams
        self.cls_value = nn.Sequential(
            nn.Linear(gnn_output_dim, dqn_hidden_dim),
            nn.ReLU(),
            nn.Linear(dqn_hidden_dim, 1),
        )

        self.cls_advantage = nn.Sequential(
            nn.Linear(gnn_output_dim, dqn_hidden_dim),
            nn.ReLU(),
            nn.Linear(dqn_hidden_dim, num_classes),
        )

        # Navigation streams
        self.nav_value = nn.Sequential(
            nn.Linear(gnn_output_dim, dqn_hidden_dim),
            nn.ReLU(),
            nn.Linear(dqn_hidden_dim, 1),
        )

        self.nav_advantage = nn.Sequential(
            nn.Linear(gnn_output_dim, dqn_hidden_dim),
            nn.ReLU(),
            nn.Linear(dqn_hidden_dim, 4),  # for up, down, left, right
        )

    def forward(
        self, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using only local neighborhood

        Args:
            state: Dictionary containing:
                - x: Node features tensor
                - edge_index: Graph connectivity
                - current_node: Index of current node
                - valid_actions_mask: Binary mask for valid navigation actions
        """

        # Extract node features
        node_features = self.gnn(state["x"], state["edge_index"], state["current_node"])

        # Get features for current node (assuming it's the first node in batch)
        current_node_features = node_features[0]

        # Classification stream
        cls_value = self.cls_value(current_node_features)
        cls_advantage = self.cls_advantage(current_node_features)
        cls_logits = cls_value + (
            cls_advantage - cls_advantage.mean(dim=-1, keepdim=True)
        )

        # Navigation stream
        nav_value = self.nav_value(current_node_features)
        nav_advantage = self.nav_advantage(current_node_features)
        nav_logits = nav_value + (
            nav_advantage - nav_advantage.mean(dim=-1, keepdim=True)
        )
        # Apply action mask
        if "valid_actions_mask" in state:
            nav_logits[~state["valid_actions_mask"]] = float("-inf")

        return cls_logits, nav_logits
