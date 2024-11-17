import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from neural_net import DuelingGraphDQN, ReplayBuffer

# Hyperparameters
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
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

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR, amsgrad=True)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(
        self,
        state: Dict[str, torch.Tensor],
        action: Tuple[int, int],
        reward: float,
        next_state: Dict[str, torch.Tensor],
        done: bool,
    ):
        # Save experience in replay memory
        self.memory.push(state, action[0], action[1], next_state, reward, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)

    def act(self, state: Dict[str, torch.Tensor], eps: float = 0.0) -> Tuple[int, int]:
        """Returns actions for given state as per current policy.

        Args:
            state (dict): Current state
            eps (float): Epsilon, for epsilon-greedy action selection
        """
        # Move state tensors to device and add batch dimension if needed
        state = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in state.items()
        }

        self.policy_net.eval()
        with torch.no_grad():
            cls_q_values, nav_q_values = self.policy_net(state)
        self.policy_net.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            cls_action = cls_q_values.argmax(dim=1).item()
            # Only select from valid navigation actions
            valid_nav_q_values = nav_q_values.clone()
            valid_nav_q_values[~state["valid_actions_mask"]] = float("-inf")
            nav_action = valid_nav_q_values.argmax(dim=1).item()
        else:
            cls_action = random.randrange(self.policy_net.cls_advantage[2].out_features)
            # Random selection from valid actions
            valid_actions = torch.where(state["valid_actions_mask"][0])[0]
            nav_action = valid_actions[random.randrange(len(valid_actions))].item()

        return cls_action, nav_action

    def learn(self, experiences: Tuple, gamma: float):
        """Update value parameters using given batch of experience tuples.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a_nav, a_cls, s', r, done)
            gamma (float): discount factor
        """
        states, nav_actions, cls_actions, next_states, rewards, dones = experiences

        # Get Q values for next states
        with torch.no_grad():
            next_cls_q_values, next_nav_q_values = self.target_net(next_states)
            cls_q_targets_next = next_cls_q_values.max(1)[0].unsqueeze(1)
            nav_q_targets_next = next_nav_q_values.max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        cls_q_targets = rewards + (gamma * cls_q_targets_next * (1 - dones))
        nav_q_targets = rewards + (gamma * nav_q_targets_next * (1 - dones))

        # Get current Q values
        cls_q_values, nav_q_values = self.policy_net(states)
        cls_q_expected = cls_q_values.gather(1, cls_actions.unsqueeze(1))
        nav_q_expected = nav_q_values.gather(1, nav_actions.unsqueeze(1))

        # Compute loss
        cls_loss = F.mse_loss(cls_q_expected, cls_q_targets)
        nav_loss = F.mse_loss(nav_q_expected, nav_q_targets)
        total_loss = cls_loss + nav_loss

        # Minimize the loss
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        # Update target network
        if self.t_step == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

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


# Training loop
def train_dqn(
    agent: GraphDQNAgent,
    env,
    n_episodes: int = 2000,
    max_t: int = 1000,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: float = 0.995,
):
    """Deep Q-Learning.

    Args:
        agent (GraphDQNAgent): The agent to train
        env: The environment
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []
    eps = eps_start

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0

        for t in range(max_t):
            # Get action
            action = agent.act(state, eps)

            # Take action
            next_state, reward, done, _ = env.step(action)

            # Update agent
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward

            if done:
                break

        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        # Print progress
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores[-100:]):.2f}")

    return scores
