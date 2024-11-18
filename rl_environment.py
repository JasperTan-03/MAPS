import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

class GraphSegmentationEnv(gym.Env):
    """Custom Environment for graph-based image/point cloud segmentation that follows gym interface."""

    def __init__(
        self, 
        graph_data: Data,
        num_classes: int,
        max_edges: int,
        reward_correct: float = 1.0,
        reward_incorrect: float = -1.0,
        reward_step: float = -0.01,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        # Store graph data
        self.graph_data = graph_data
        self.num_nodes = graph_data.x.size(0)
        self.num_classes = num_classes
        self.max_edges = max_edges
        
        # Reward configuration
        self.reward_correct = reward_correct
        self.reward_incorrect = reward_incorrect
        self.reward_step = reward_step
        
        self.render_mode = render_mode
        
        # Define action spaces:
        # - Navigation: Which edge to traverse (max_edges possible edges)
        # - Classification: Which class to assign to current node (num_classes)
        self.action_space = spaces.Dict({
            'navigation': spaces.Discrete(max_edges),
            'classification': spaces.Discrete(num_classes)
        })
        
        # Define observation space
        node_features = self.graph_data.x.size(1)
        self.observation_space = spaces.Dict({
            'x': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.num_nodes, node_features), 
                dtype=np.float32
            ),
            'edge_index': spaces.Box(
                low=0,
                high=self.num_nodes,
                shape=(2, self.graph_data.edge_index.size(1)),
                dtype=np.int64
            ),
            'current_node': spaces.Discrete(self.num_nodes),
            'valid_actions_mask': spaces.Box(
                low=0,
                high=1,
                shape=(max_edges,),
                dtype=np.int8
            ),
            'segmentation_mask': spaces.Box(
                low=0,
                high=num_classes,
                shape=(self.num_nodes,),
                dtype=np.int64
            )
        })
        
        # Initialize state
        self.current_node = None
        self.segmentation_mask = None
        self.steps = 0
        self.max_steps = self.num_nodes * 2  # Adjust based on needs
        
    def _get_valid_actions_mask(self) -> np.ndarray:
        """Create a mask for valid navigation actions at current state."""
        # Get neighboring nodes
        neighbors = self.graph_data.edge_index[1][
            self.graph_data.edge_index[0] == self.current_node
        ]
        
        # Create mask
        mask = np.zeros(10, dtype=np.int8) # UPDATE THIS BASED ON HOW MANY EDGES WE CAN NAVIGATE TO
        mask[:len(neighbors)] = 1
        
        return mask
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation state."""
        return {
            'x': self.graph_data.x.numpy(),
            'edge_index': self.graph_data.edge_index.numpy(),
            'current_node': self.current_node,
            'valid_actions_mask': self._get_valid_actions_mask(),
            'segmentation_mask': self.segmentation_mask
        }
    
    def _get_reward(self, action_cls: int) -> float:
        """Calculate reward based on classification action."""
        true_label = self.graph_data.y[self.current_node].item()
        
        reward = self.reward_step  # Base reward/penalty for each step
        
        if action_cls == true_label:
            reward += self.reward_correct
        else:
            reward += self.reward_incorrect
            
        return reward
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        
        # Reset environment state
        self.steps = 0
        self.current_node = self.np_random.integers(0, self.num_nodes)
        self.segmentation_mask = np.zeros(self.num_nodes, dtype=np.int64)
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action: Dict[str, int]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Dictionary containing:
                - navigation: Index of edge to traverse
                - classification: Class to assign to current node
        """
        self.steps += 1
        
        # Get valid neighbors
        neighbors = self.graph_data.edge_index[1][
            self.graph_data.edge_index[0] == self.current_node
        ]
        
        # Execute navigation action if valid
        if action['navigation'] < len(neighbors):
            self.current_node = neighbors[action['navigation']].item()
            
        # Execute classification action
        self.segmentation_mask[self.current_node] = action['classification']
        
        # Calculate reward
        reward = self._get_reward(action['classification'])
        
        # Get new observation
        observation = self._get_observation()
        
        # Check termination conditions
        terminated = False
        if self.steps >= self.max_steps:
            terminated = True
            
        # Check truncation conditions
        truncated = False
        
        # Additional info
        info = {
            'steps': self.steps,
            'segmentation_accuracy': np.mean(
                self.segmentation_mask == self.graph_data.y.numpy().flatten()
            )
        }
        
        return observation, reward, terminated, truncated, info

    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        super().reset(seed=seed)
        return [seed]