import random
from collections import deque, namedtuple
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import k_hop_subgraph, to_dense_adj


class DuelingGNN(nn.Module):
    """Graph Neural Network with dueling architecture"""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_hops: int = 4
    ):
        super(DuelingGNN, self).__init__()

        self.num_hops = num_hops
        self.output_dim = output_dim

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
        self, x: torch.Tensor, edge_index: torch.Tensor, current_node: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Get k-hop subgraph for one or more nodes.

        Args:
            x: Node feature tensor [num_nodes, feature_dim]
            edge_index: Edge index tensor [2, num_edges]
            current_nodes: Index or indices of current nodes (int or [batch_size])

        Returns:
            sub_x: Subgraph node features
            sub_edge_index: Subgraph edge indices
            mapping: Mapping from original nodes to subgraph nodes
        """

        # Get k-hop subgraph
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            current_node, self.num_hops, edge_index, relabel_nodes=True
        )

        # Get subgraph features
        sub_x = x[subset]

        return sub_x, sub_edge_index, mapping

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, current_node: torch.Tensor
    ) -> torch.Tensor:

        if current_node.dim() == 0:  # If it's a scalar tensor (0D)
            current_node = current_node.unsqueeze(0)  # Convert to [1] tensor

        # Get k-hop subgraph
        sub_x, sub_edge_index, mapping = self.get_subgraph(x, edge_index, current_node)

        # Apply GCN layers
        for i, conv in enumerate(self.convs):
            sub_x = conv(sub_x, sub_edge_index)
            sub_x = F.relu(sub_x)
            sub_x = F.dropout(sub_x, p=0.1, training=self.training)
            sub_x = self.layer_norms[i](sub_x)

        # Pooling (to handle batch dimensions)
        batch = torch.zeros(sub_x.size(0), dtype=torch.long, device=sub_x.device)
        batch[mapping] = torch.arange(len(current_node), device=sub_x.device)
        x_pooled = global_mean_pool(sub_x, batch=batch, size=current_node.size(0))
        
        return x_pooled


class DuelingGraphDQN(nn.Module):
    """Dueling DQN architecture with separate value and advantage streams"""

    def __init__(
        self,
        node_feature_dim: int,
        gnn_hidden_dim: int,
        gnn_output_dim: int,
        dqn_hidden_dim: int,
        num_classes: int,
        k_hops: int = 4,
    ):
        super(DuelingGraphDQN, self).__init__()

        # Feature extractor
        self.gnn = DuelingGNN(
            input_dim=node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_hops=k_hops,
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

        # # Navigation streams
        # self.nav_value = nn.Sequential(
        #     nn.Linear(gnn_output_dim, dqn_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(dqn_hidden_dim, 1),
        # )

        # self.nav_advantage = nn.Sequential(
        #     nn.Linear(gnn_output_dim, dqn_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(dqn_hidden_dim, 4),  # for up, down, left, right
        # )

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

        if state["current_node"].dim() == 0:  # If it's a scalar tensor (0D)
            state["current_node"] = state["current_node"].unsqueeze(0)  # Convert to [1] tensor

        # Extract node features
        node_features = self.gnn(state["x"], state["edge_index"], state["current_node"])
        # Classification stream
        cls_value = self.cls_value(node_features)
        cls_advantage = self.cls_advantage(node_features)
        cls_logits = cls_value + (
            cls_advantage - cls_advantage.mean(dim=-1, keepdim=True)
        )

        # # Navigation stream
        # nav_value = self.nav_value(current_node_features)
        # nav_advantage = self.nav_advantage(current_node_features)
        # nav_logits = nav_value + (
        #     nav_advantage - nav_advantage.mean(dim=-1, keepdim=True)
        # )
        # # Apply action mask
        # if "valid_actions_mask" in state:
        #     nav_logits[~state["valid_actions_mask"]] = float("-inf")

        return cls_logits   # , nav_logits
