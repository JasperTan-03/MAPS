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
    ):
        super().__init__()
        
        # Store graph data
        self.graph = graph_data
        self.num_nodes = graph_data.x.size(0)
        self.num_classes = num_classes

        self.visited_nodes = set()
        
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
                shape=(4),
                dtype=np.int8
            ),
            # 'segmentation_mask': spaces.Box(
            #     low=0,
            #     high=num_classes,
            #     shape=(self.num_nodes,),
            #     dtype=np.int64
            # )
        })
        
        # Initialize state
        self.current_node = None
        # self.segmentation_mask = None
        self.steps = 0
        self.max_steps = self.num_nodes * 2  # Adjust based on needs

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        
        # Reset environment state
        self.steps = 0
        self.current_node = self.np_random.integers(0, self.num_nodes)
        # self.segmentation_mask = np.zeros(self.num_nodes, dtype=np.int64)
        
        observation = self._get_observation()
        self.visited_nodes = set(self.current_node)
        
        return observation
        
    def _get_valid_actions_mask(self) -> np.ndarray:
        """Create a mask for valid navigation actions at current state."""
        # Get neighboring nodes
        cur_x, cur_y = self.graph.x[self.current_node][:2]
        max_x, max_y = self.graph.x[-1][:2]

        mask = torch.zeros(4, dtype=np.int8)

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
            # 'segmentation_mask': self.segmentation_mask
        }
    
    def _get_reward(self, action_cls: int, action_nav: int) -> float:
        """Calculate reward based on classification action."""
        true_label = self.graph.y[self.current_node].item()
        
        # reward = self.reward_step  # Base reward/penalty for each step
        
        classification_reward = torch.zeros(self.num_classes)

        # set index of correct class to 1 and all others to -1
        classification_reward[true_label] = 1

        # set all other indices other than true label to -1
        classification_reward[true_label] = 1
        for i in range(self.num_classes):
            if i != true_label:
                classification_reward[i] = -1

                
        # Navigation reward
        navigation_reward = torch.zeros(4)

        neighbors = self.graph.edge_index[1][
            self.graph.edge_index[0] == self.current_node
        ]

        cur_x, cur_y = self.graph.x[self.current_node][:2]


        # find which neighbor we are moving to based on action_nav and set current node to that neighbor
        for neighbor in neighbors:
            neighbor_x, neighbor_y = self.graph.x[neighbor][:2]
            if action_nav == 0 and neighbor_x == cur_x - 1:
                if neighbor in self.visited_nodes:
                    navigation_reward[0] = -1
                else:
                    navigation_reward[0] = 1

                self.current_node = neighbor

            elif action_nav == 1 and neighbor_x == cur_x + 1:
                if neighbor in self.visited_nodes:
                    navigation_reward[1] = -1
                else:
                    navigation_reward[1] = 1

                self.current_node = neighbor

            elif action_nav == 2 and neighbor_y == cur_y - 1:
                if neighbor in self.visited_nodes:
                    navigation_reward[2] = -1
                else:
                    navigation_reward[2] = 1

                self.current_node = neighbor

            elif action_nav == 3 and neighbor_y == cur_y + 1:
                if neighbor in self.visited_nodes:
                    navigation_reward[3] = -1
                else:
                    navigation_reward[3] = 1

                self.current_node = neighbor

        # Add current node to visited nodes
        self.visited_nodes.add(self.current_node)

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
        
        # Get new observation
        observation = self._get_observation()
        
        # Check termination conditions
        done = False
        if self.steps >= self.max_steps or len(self.visited_nodes) == self.num_nodes:
            done = True

        return observation, reward, done

    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        super().reset(seed=seed)
        return [seed]