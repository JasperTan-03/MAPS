import math
import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from neural_net import DuelingGraphDQN, Experience, ReplayBuffer
from rl_environment import SegmentationEnv

# Hyperparameters
UPDATE_EVERY = 4


class GraphDQNAgent:
    def __init__(
        self,
        node_feature_dim: int,
        gnn_hidden_dim: int,
        gnn_output_dim: int,
        dqn_hidden_dim: int,
        num_classes: int,
        max_num_edges: int,
        seed: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 32,
        tau: float = 1e-3,
        target_update_freq: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.seed = random.seed(seed)
        self.device = device

        # Q-Networks
        self.policy_net = DuelingGraphDQN(
            node_feature_dim=node_feature_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            dqn_hidden_dim=dqn_hidden_dim,
            num_classes=num_classes,
            max_num_edges=max_num_edges,
        ).to(self.device)

        self.target_net = DuelingGraphDQN(
            node_feature_dim=node_feature_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            dqn_hidden_dim=dqn_hidden_dim,
            num_classes=num_classes,
            max_num_edges=max_num_edges,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, amsgrad=True)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size)

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.target_update_freq = target_update_freq

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.losses = []

    def select_action(self, state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
        """Select an action using epsilon-greedy policy.

        Args:
            state (dict): Current state

        Returns:
            Tuple[int, int]: Chosen action
        """
        # Move state tensors to device and add batch dimension if needed
        state = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in state.items()
        }

        self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * math.exp(
            -1.0 * self.t_step / self.epsilon_decay
        )
        self.t_step += 1

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            with torch.no_grad():
                cls_q_values, nav_q_values = self.policy_net(state)
                cls_action = cls_q_values.argmax(dim=1).item()
                nav_action = nav_q_values.argmax(dim=1).item()
        else:
            # Random classification action
            cls_action = random.randrange(self.policy_net.cls_advantage[2].out_features)
            # Random navigation action from valid actions
            valid_actions = torch.where(state["valid_actions_mask"][0])[0]
            nav_action = valid_actions[random.randrange(len(valid_actions))].item()

        return cls_action, nav_action

    def optimize_model(self) -> Optional[torch.Tensor]:
        """Perform a single step of optimization on the policy network.

        Returns:
            Optional[torch.Tensor]: Loss value
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample a batch of experiences
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        # Separate experiences into batches
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_nav_batch = torch.cat(batch.action_nav)
        action_cls_batch = torch.cat(batch.action_cls)
        reward_batch = torch.cat(batch.reward)

        # Current Q values
        policy_cls_q, policy_nav_q = self.policy_net(state_batch)
        policy_cls_q = policy_cls_q.gather(1, action_cls_batch)
        policy_nav_q = policy_nav_q.gather(1, action_nav_batch)

        # Compute target Q values
        next_cls_q = torch.zeros(self.batch_size, device=self.device)
        next_nav_q = torch.zeros(self.batch_size, device=self.device)

        with torch.no_grad():
            next_cls_q_values, next_nav_q_values = self.target_net(
                non_final_next_states
            )
            next_cls_q[non_final_mask] = next_cls_q_values.max(1).values
            next_nav_q[non_final_mask] = next_nav_q_values.max(1).values

        # Compute target Q values
        expected_cls_q = (next_cls_q * self.gamma) + reward_batch
        expected_nav_q = (next_nav_q * self.gamma) + reward_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        cls_loss = criterion(policy_cls_q, expected_cls_q.unsqueeze(1))
        nav_loss = criterion(policy_nav_q, expected_nav_q.unsqueeze(1))
        total_loss = cls_loss + nav_loss

        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return total_loss.item()

    def train(
        self,
        env,
        num_episodes: int,
        max_steps: int = 1000,
    ) -> List[float]:
        episode_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                # Select action
                cls_action, nav_action = self.select_action(state)

                # Take action
                next_state, reward, done, _ = env.step((cls_action, nav_action))
                episode_reward += reward

                # Store experience
                self.memory.push(
                    state, nav_action, cls_action, next_state, reward, done
                )
                state = next_state

                # Optimize model
                loss = self.optimize_model()
                if loss is not None:
                    self.losses.append(loss)

                # Soft update of the target network's weight
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[
                        key
                    ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    break

            episode_rewards.append(episode_reward)

            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                print(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Epsilon: {self.epsilon:.2f}"
                )

        return episode_rewards

    def plot_training_results(self, episode_rewards: List[float]):
        """
        Plot training rewards and losses
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot rewards
        ax1.plot(episode_rewards)
        ax1.set_title("Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")

        # Plot losses
        if self.losses:
            ax2.plot(self.losses)
            ax2.set_title("Training Loss")
            ax2.set_xlabel("Optimization Step")
            ax2.set_ylabel("Loss")

        plt.tight_layout()
        plt.show()

    def save_model(self, path: str):
        """Save model weights.

        Args:
            path (str): path to save the model weights
        """
        os.makedirs("weights", exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"weights/{path}.pth")

    def load_model(self, path: str):
        """Load model weights.

        Args:
            path (str): path to load the model weights
        """
        self.policy_net.load_state_dict(
            torch.load(f"weights/{path}.pth", map_location=self.device)
        )
        self.policy_net.eval()


if __name__ == "__main__":
    # Get graph
    graph = torch.load("data/image_graph.pt")

    # Create sample data
    node_feature_dim = graph.x.size(-1)
    gnn_hidden_dim = 64
    gnn_output_dim = 64
    dqn_hidden_dim = 64
    num_classes = 2
    max_num_edges = graph.edge_index.size(1)
    seed = 0

    # Create environment
    env = SegmentationEnv(graph, seed=seed)

    # Create agent
    agent = GraphDQNAgent(
        node_feature_dim=node_feature_dim,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_output_dim=gnn_output_dim,
        dqn_hidden_dim=dqn_hidden_dim,
        num_classes=num_classes,
        max_num_edges=max_num_edges,
        seed=seed,
    )

    # Train agent
    num_episodes = 100
    episode_rewards = agent.train(env, num_episodes)

    # Plot training results
    agent.plot_training_results(episode_rewards)

    # Save model weights
    agent.save_model("graph_dqn_agent")
