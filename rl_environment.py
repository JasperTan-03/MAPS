from collections import deque
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


class GraphSegmentationEnv(gym.Env):
    """Custom Environment for graph-based image/point cloud segmentation that follows gym interface."""

    def __init__(
        self, 
        graph_data: Data,
        num_classes: int,
    ):
        super().__init__()
        
        # Store graph data
        self.graph = graph_data
        self.num_nodes = graph_data.x.size(0)
        self.num_classes = num_classes

        # Define action spaces:
        # - Navigation: Which edge to traverse (max_edges possible edges)
        # - Classification: Which class to assign to current node (num_classes)
        self.action_space = spaces.Dict({
            'navigation': spaces.Discrete(4), # UPDATE THIS BASED ON HOW MANY EDGES WE CAN NAVIGATE TO
            'classification': spaces.Discrete(num_classes)
        })
        
        # Define observation space
        node_features = self.graph.x.size(1)
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
                shape=(2, self.graph.edge_index.size(1)),
                dtype=np.int64
            ),
            'current_node': spaces.Discrete(self.num_nodes),
            'valid_actions_mask': spaces.Box(
                low=0,
                high=1,
                shape=(4,),
                dtype=np.int8
            ),
        })
        
        # Initialize state
        self.steps = 0
        self.max_steps = self.num_nodes * 2  # Adjust based on needs

        self.bfs_queue = deque()
        self.visited = set()
        self._initialize_bfs()

    def _initialize_bfs(self):
        """Initialize BFS traversal."""
        self.bfs_queue.clear()
        self.visited.clear()
        self.bfs_queue.append(0)
        self.visited.add(0)
        self.current_node = torch.tensor(self.num_nodes // 2)

    def reset(
        self, 
        new_graph: Optional[Data] = None,
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        
        if new_graph is not None:
            self.graph = new_graph
            self.num_nodes = new_graph.x.size(0)

        super().reset(seed=seed)
        
        # Reset environment state
        self.steps = 0
        self._initialize_bfs()

        observation = self._get_observation()
        
        return observation
        
    def _get_valid_actions_mask(self) -> np.ndarray:
        """Create a mask for valid navigation actions at current state."""
        # Get neighboring nodes
        cur_loc = self.graph.x[self.current_node][:2]
        cur_x, cur_y = cur_loc[0].item(), cur_loc[1].item()
        max_loc = self.graph.x[-1][:2]
        max_x, max_y = max_loc[0].item(), max_loc[1].item()

        mask = torch.zeros(4, dtype=torch.bool)

        # Check if we can move left
        if cur_y > 0:
            mask[0] = 1

        # Check if we can move right
        if cur_y < max_y:
            mask[1] = 1

        # Check if we can move up
        if cur_x > 0:
            mask[2] = 1

        # Check if we can move down
        if cur_x < max_x:
            mask[3] = 1
        
        return mask
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation state."""
        return {
            'x': self.graph.x,
            'edge_index': self.graph.edge_index,
            'current_node': self.current_node,
            'valid_actions_mask': self._get_valid_actions_mask(),
        }
    
    def _get_reward(self, action_cls: int, action_nav: int) -> float:
        """Calculate reward based on classification action."""
        true_label = self.graph.y[self.current_node].item()
        
        classification_reward = torch.full((self.num_classes,), -0.1) # small negative baseline

        if action_cls == true_label:
            classification_reward[action_cls] = 1.0
        else:
            classification_reward[action_cls] = -1.0
            classification_reward[true_label] = 1.0 # still give true label a reward

        navigation_reward = torch.zeros(4)

        return {'cls': classification_reward, 'nav': navigation_reward }
    
    def step(self, action: Dict[str, int]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Dictionary containing:
                - navigation: Index of edge to traverse
                - classification: Class to assign to current node
        """
        self.steps += 1
        
        # Calculate reward
        reward = self._get_reward(action['cls'], action['nav'])
        
        # Check termination conditions
        done = False
        if self.steps >= self.max_steps or not self.bfs_queue:
            done = True

        if not done:
            neighbors = self.graph.edge_index[1][self.graph.edge_index[0] == self.current_node]
            for neighbor in neighbors:
                if neighbor.item() not in self.visited:
                    self.bfs_queue.append(neighbor.item())
                    self.visited.add(neighbor.item())
            self.current_node = torch.tensor(self.bfs_queue.popleft())
        else:
            self.current_node = None

        # Get new observation
        observation = self._get_observation()

        return observation, reward, done

    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        super().reset(seed=seed)
        return [seed]
