import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import Tuple, Dict, Optional
import gymnasium as gym
import numpy as np
from collections import deque, namedtuple
import random

# Define experience tuple structure
Experience = namedtuple('Experience', 
    ('state', 'action_nav', 'action_cls', 'next_state', 'reward', 'done'))

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
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super(DuelingGNN, self).__init__()
        self.convs = nn.ModuleList()
        
        # GCN layers
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
        ])
        self.layer_norms.append(nn.LayerNorm(output_dim))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch=None) -> torch.Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs, self.layer_norms)):
            # Residual connection if dimensions match
            identity = x if x.size(-1) == conv.out_channels else None
            
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
            
            if identity is not None:
                x = x + identity
                
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
        max_num_edges: int
    ):
        super(DuelingGraphDQN, self).__init__()
        
        # Feature extractor
        self.gnn = DuelingGNN(
            input_dim=node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim
        )
        
        # Classification streams
        self.cls_value = nn.Sequential(
            nn.Linear(gnn_output_dim, dqn_hidden_dim),
            nn.ReLU(),
            nn.Linear(dqn_hidden_dim, 1)
        )
        
        self.cls_advantage = nn.Sequential(
            nn.Linear(gnn_output_dim, dqn_hidden_dim),
            nn.ReLU(),
            nn.Linear(dqn_hidden_dim, num_classes)
        )
        
        # Navigation streams
        self.nav_value = nn.Sequential(
            nn.Linear(gnn_output_dim, dqn_hidden_dim),
            nn.ReLU(),
            nn.Linear(dqn_hidden_dim, 1)
        )
        
        self.nav_advantage = nn.Sequential(
            nn.Linear(gnn_output_dim, dqn_hidden_dim),
            nn.ReLU(),
            nn.Linear(dqn_hidden_dim, max_num_edges)
        )
        
    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract features
        node_features = self.gnn(state['x'], state['edge_index'], state.get('batch', None))
        current_node_features = node_features[0].unsqueeze(0)
        
        # Classification Q-values
        cls_value = self.cls_value(current_node_features)
        cls_advantage = self.cls_advantage(current_node_features)
        cls_q_values = cls_value + (cls_advantage - cls_advantage.mean(dim=1, keepdim=True))
        
        # Navigation Q-values
        nav_value = self.nav_value(current_node_features)
        nav_advantage = self.nav_advantage(current_node_features)
        nav_q_values = nav_value + (nav_advantage - nav_advantage.mean(dim=1, keepdim=True))
        
        # Mask invalid actions
        if 'valid_actions_mask' in state:
            nav_q_values[~state['valid_actions_mask']] = float('-inf')
            
        return cls_q_values, nav_q_values
    


### THIS MIGHT BE SIMPLER BELOW ###

# class GraphGNN(nn.Module):
#     """Graph Neural Network for node feature extraction"""
#     def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
#         super(GraphGNN, self).__init__()
#         self.convs = nn.ModuleList()
        
#         # First layer
#         self.convs.append(GCNConv(input_dim, hidden_dim))
        
#         # Hidden layers
#         for _ in range(num_layers - 2):
#             self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
#         # Output layer
#         self.convs.append(GCNConv(hidden_dim, output_dim))
        
#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch=None) -> torch.Tensor:
#         # Apply GCN layers with residual connections
#         for i, conv in enumerate(self.convs[:-1]):
#             x = conv(x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, p=0.1, training=self.training)
            
#         x = self.convs[-1](x, edge_index)
#         return x

# class GraphDQN(nn.Module):
#     def __init__(
#         self,
#         node_feature_dim: int,
#         gnn_hidden_dim: int,
#         gnn_output_dim: int,
#         dqn_hidden_dim: int,
#         num_classes: int,
#         max_num_edges: int
#     ):
#         super(GraphDQN, self).__init__()
        
#         # GNN for feature extraction
#         self.gnn = GraphGNN(
#             input_dim=node_feature_dim,
#             hidden_dim=gnn_hidden_dim,
#             output_dim=gnn_output_dim
#         )
        
#         # Classification head
#         self.classification_head = nn.Sequential(
#             nn.Linear(gnn_output_dim, dqn_hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(dqn_hidden_dim, num_classes)
#         )
        
#         # Navigation head
#         self.navigation_head = nn.Sequential(
#             nn.Linear(gnn_output_dim, dqn_hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(dqn_hidden_dim, max_num_edges)
#         )
        
#     def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Forward pass of the network
        
#         Args:
#             state: Dictionary containing:
#                 - x: Node features
#                 - edge_index: Graph connectivity
#                 - batch: Batch indices for multiple graphs (optional)
#                 - valid_actions_mask: Binary mask for valid navigation actions
        
#         Returns:
#             classification_logits: Logits for node classification
#             navigation_logits: Logits for navigation (masked for valid actions only)
#         """
#         # Extract node features using GNN
#         node_features = self.gnn(state['x'], state['edge_index'], state.get('batch', None))
        
#         # Get features for current node (assuming it's the first node in batch)
#         current_node_features = node_features[0].unsqueeze(0)
        
#         # Get classification logits
#         classification_logits = self.classification_head(current_node_features)
        
#         # Get navigation logits and apply action mask
#         navigation_logits = self.navigation_head(current_node_features)
#         if 'valid_actions_mask' in state:
#             navigation_logits[~state['valid_actions_mask']] = float('-inf')
            
#         return classification_logits, navigation_logits