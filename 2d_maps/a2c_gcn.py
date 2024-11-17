# agent.py
import random
from enum import Enum
from typing import Dict, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent import AgentAction, SegmentationAgent
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class AgentAction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class GCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return torch.sigmoid(self.classifier(x))


class ActorCritic(nn.Module):
    def __init__(self, input_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )

        self.actor = nn.Linear(64, len(AgentAction))
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        shared_features = self.shared(x)
        action_probs = F.softmax(self.actor(shared_features), dim=-1)
        value = self.critic(shared_features)
        return action_probs, value


class SegmentationAgent:
    def __init__(self, height: int, width: int, labels: np.ndarray):
        self.height = height
        self.width = width
        self.labels = labels

        # State variables
        self.state = None
        self.path = None
        self.current_position = None
        self.visited_positions = set()

        # Neural networks
        self.feature_dim = 7  # position (2) + pixel value (1) + path value (1) + visited (1) + neighborhood stats (2)
        self.actor_critic = ActorCritic(self.feature_dim)
        self.gcn_classifier = GCNClassifier(self.feature_dim)

        # Optimizer setup
        self.optimizer = torch.optim.Adam(
            list(self.actor_critic.parameters())
            + list(self.gcn_classifier.parameters()),
            lr=0.001,
        )

        self.reset()

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Reset the environment to initial state."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Reset state matrices
        self.state = np.zeros((self.height, self.width), dtype=np.float32)
        self.path = np.zeros((self.height, self.width), dtype=np.float32)
        self.visited_positions = set()

        # Start from a random unvisited position
        self.current_position = (
            random.randint(0, self.height - 1),
            random.randint(0, self.width - 1),
        )
        self.visited_positions.add(self.current_position)
        self.path[self.current_position] = 1

        return self.get_observation()

    def get_observation(self) -> Dict[str, np.ndarray]:
        """Return the current observation."""
        return {
            "state": self.state.copy(),
            "path": self.path.copy(),
            "position": np.array(self.current_position),
        }

    def perform_action(self, action: AgentAction) -> float:
        """Execute the given action and return the reward."""
        prev_position = self.current_position

        # Update position based on action
        if action == AgentAction.LEFT:
            new_pos = (prev_position[0], max(0, prev_position[1] - 1))
        elif action == AgentAction.RIGHT:
            new_pos = (prev_position[0], min(self.width - 1, prev_position[1] + 1))
        elif action == AgentAction.UP:
            new_pos = (max(0, prev_position[0] - 1), prev_position[1])
        elif action == AgentAction.DOWN:
            new_pos = (min(self.height - 1, prev_position[0] + 1), prev_position[1])

        self.current_position = new_pos
        self.visited_positions.add(self.current_position)

        # Classify the current position
        prediction = self.classify()
        self.state[self.current_position] = prediction
        self.path[self.current_position] = 1

        # Calculate reward
        reward = self.calculate_reward(prediction)
        return reward

    def calculate_reward(self, prediction: float) -> float:
        """Calculate the reward for the current action."""
        true_label = self.labels[self.current_position]

        # Base reward for classification accuracy
        reward = 1.0 if abs(prediction - true_label) < 0.5 else -0.5

        # Penalty for revisiting
        if (
            len(self.visited_positions) > 1
            and self.current_position in self.visited_positions
        ):
            reward -= 0.25

        # Additional reward for exploring new areas
        if len(self.visited_positions) == 1:  # First visit to this position
            reward += 0.1

        return reward

    def get_node_features(self, position):
        row, col = position
        features = []

        # Position features
        features.extend([row / self.height, col / self.width])

        # Pixel value
        features.append(float(self.labels[row, col]))

        # Path value
        features.append(float(self.path[row, col]))

        # Visited status
        features.append(float(position in self.visited_positions))

        # Neighborhood statistics
        neighborhood = self.get_neighborhood_features(position)
        features.extend(neighborhood)

        return torch.tensor(features, dtype=torch.float32)

    def get_neighborhood_features(self, position):
        row, col = position
        neighborhood = []
        visited_count = 0
        labeled_count = 0

        for i in range(max(0, row - 1), min(self.height, row + 2)):
            for j in range(max(0, col - 1), min(self.width, col + 2)):
                if (i, j) in self.visited_positions:
                    visited_count += 1
                if self.state[i, j] == 1:
                    labeled_count += 1

        return [visited_count / 9, labeled_count / 9]  # Normalize by neighborhood size

    def build_graph(self, position):
        # Create graph structure for GCN
        nodes = []
        edges = []
        center_idx = 0

        # Add center node
        nodes.append(self.get_node_features(position))

        # Add neighboring nodes
        row, col = position
        idx = 1
        for i in range(max(0, row - 1), min(self.height, row + 2)):
            for j in range(max(0, col - 1), min(self.width, col + 2)):
                if (i, j) != position:
                    nodes.append(self.get_node_features((i, j)))
                    edges.append([center_idx, idx])
                    edges.append([idx, center_idx])
                    idx += 1

        nodes = torch.stack(nodes)
        edge_index = torch.tensor(edges, dtype=torch.long).t()

        return Data(x=nodes, edge_index=edge_index)

    def classify(self):
        graph = self.build_graph(self.current_position)
        with torch.no_grad():
            prediction = self.gcn_classifier(graph.x, graph.edge_index)
        return float(prediction[0].item() > 0.5)

    def select_action(self):
        features = self.get_node_features(self.current_position)
        action_probs, value = self.actor_critic(features.unsqueeze(0))
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()

        return action.item(), action_probs, value

    def update_networks(self, transitions):
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Convert to tensors
        states = torch.stack([self.get_node_features(s) for s in states])
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.stack([self.get_node_features(s) for s in next_states])
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute returns
        _, next_values = self.actor_critic(next_states)
        returns = rewards + (1 - dones) * 0.99 * next_values.squeeze()

        # Get current predictions
        action_probs, values = self.actor_critic(states)

        # Compute losses
        advantage = returns - values.squeeze()
        actor_loss = -torch.mean(
            torch.log(action_probs.gather(1, actions.unsqueeze(1))) * advantage.detach()
        )
        critic_loss = F.mse_loss(values.squeeze(), returns)

        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss

        # Update networks
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()


# rl_environment.py
class SegmentationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        step_limit: int = 100,
        render_mode: str = "human",
    ):
        super().__init__()
        self.image = image
        self.labels = labels
        self.height, self.width = image.shape[:2]
        self.step_limit = step_limit
        self.num_steps = 0
        self.render_mode = render_mode

        # Initialize agent
        self.agent = SegmentationAgent(self.height, self.width, labels)

        # Define spaces
        self.action_space = spaces.Discrete(len(AgentAction))
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=0, high=1, shape=(self.height, self.width), dtype=np.float32
                ),
                "path": spaces.Box(
                    low=0, high=1, shape=(self.height, self.width), dtype=np.float32
                ),
                "position": spaces.Box(
                    low=0, high=max(self.height, self.width), shape=(2,), dtype=np.int32
                ),
            }
        )

        # Training memory
        self.transitions = []

        # Visualization setup
        self.fig = None
        self.ax = None

    def render(self, mode="human"):
        """
        Render the environment.

        Args:
            mode (str): "human" or "rgb_array"

        Returns:
            numpy array if mode is "rgb_array", None otherwise
        """
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(2, 2, figsize=(10, 10))
            self.fig.suptitle("Segmentation Progress")

        # Clear previous plots
        for ax_row in self.ax:
            for ax in ax_row:
                ax.clear()

        # Plot original image
        self.ax[0, 0].imshow(self.image, cmap="gray")
        self.ax[0, 0].set_title("Original Image")
        self.ax[0, 0].axis("off")

        # Plot ground truth labels
        self.ax[0, 1].imshow(self.labels, cmap="coolwarm")
        self.ax[0, 1].set_title("Ground Truth")
        self.ax[0, 1].axis("off")

        # Plot current segmentation state
        self.ax[1, 0].imshow(self.agent.state, cmap="coolwarm")
        self.ax[1, 0].set_title("Current Segmentation")
        self.ax[1, 0].axis("off")

        # Plot agent path
        path_plot = self.ax[1, 1].imshow(self.agent.path, cmap="hot")
        self.ax[1, 1].set_title("Agent Path")
        self.ax[1, 1].axis("off")

        # Mark current position on all plots
        current_pos = self.agent.current_position
        for ax_row in self.ax:
            for ax in ax_row:
                ax.plot(current_pos[1], current_pos[0], "go", markersize=10)

        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.0001)

        if mode == "rgb_array":
            # Convert plot to RGB array
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data

    def close(self):
        """Clean up resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            plt.ioff()

    def step(self, action):
        self.num_steps += 1
        prev_position = self.agent.current_position

        # Perform action and get reward
        reward = self.agent.perform_action(AgentAction(action))

        # Store transition
        self.transitions.append(
            (
                prev_position,
                action,
                reward,
                self.agent.current_position,
                self.num_steps >= self.step_limit,
            )
        )

        # Update networks every N steps
        if len(self.transitions) >= 32:
            loss = self.agent.update_networks(self.transitions)
            self.transitions = []

        # Get new observation
        observation = self.agent.get_observation()

        # Check if episode is done
        done = (self.num_steps >= self.step_limit) or (
            np.sum(self.agent.path) >= self.height * self.width
        )

        # Additional Info
        info = {
            "steps": self.num_steps,
            "coverage": np.sum(self.agent.path) / (self.height * self.width),
            "accuracy": np.mean(self.agent.state == self.labels),
        }

        return observation, reward, done, info

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        # Reset base environment
        super().reset(seed=seed)

        # Reset environment variables
        self.num_steps = 0
        self.transitions = []

        # Reset agent
        observation = self.agent.reset(seed=seed)

        # Reset done flag
        self.done = False


if __name__ == "__main__":
    # Create sample data
    height, width = 100, 100
    random_image = torch.randint(0, 2, (height, width)).numpy()
    random_labels = torch.randint(0, 2, (height, width)).numpy()

    # Create environment
    env = SegmentationEnv(image=random_image, labels=random_labels)

    # Training loop
    num_episodes = 1
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        print(f"Episode {episode}")
        while True:
            # Select action using actor network
            action, _, _ = env.agent.select_action()

            # Take step in environment
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if episode % 10 == 0:
                env.render()

            if done:
                print(f"Episode {episode}")
                print(f"Total reward: {total_reward:.2f}")
                print(f"Coverage: {info['coverage']*100:.1f}%")
                print(f"Accuracy: {info['accuracy']*100:.1f}%")
                break
