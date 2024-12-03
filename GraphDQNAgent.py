import math
import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from PIL import Image
from tqdm import tqdm

import wandb
from neural_net import DuelingGraphDQN, ReplayBuffer
from render import SegmentationRenderer
from rl_environment import GraphSegmentationEnv


class GraphDQNAgent:
    def __init__(
        self,
        num_classes: int,
        node_feature_dim: int,
        gnn_hidden_dim: int = 64,
        gnn_output_dim: int = 64,
        dqn_hidden_dim: int = 64,
        k_hops: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        args = yaml.safe_load(open("configs/agent.yaml", "r"))
        args = args["dqn-agent"]

        # Initialize wandb
        wandb.init(
            project="graph-dqn-segmentation",
            name=args["name"],
            config=args,
        )

        self.seed = random.seed(int(args["random_seed"]))
        self.device = device

        # Q-Networks
        self.policy_net = DuelingGraphDQN(
            node_feature_dim=node_feature_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            dqn_hidden_dim=dqn_hidden_dim,
            num_classes=num_classes,
            k_hops=k_hops,
        ).to(self.device)

        self.target_net = DuelingGraphDQN(
            node_feature_dim=node_feature_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            dqn_hidden_dim=dqn_hidden_dim,
            num_classes=num_classes,
            k_hops=k_hops,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=float(args["lr"]), amsgrad=True
        )

        # Replay memory
        self.memory = ReplayBuffer(int(args["replay_buffer_size"]))

        self.batch_size = int(args["batch_size"])
        self.gamma = float(args["gamma"])
        self.epsilon = float(args["epsilon_start"])
        self.epsilon_end = float(args["epsilon_end"])
        self.epsilon_decay = float(args["epsilon_decay"])
        self.tau = float(args["tau"])
        self.target_update_freq = int(args["target_update_freq"])
        self.num_classes = num_classes

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.losses = []
        self.update_every = args["update_every"]

    def select_action(
        self, expected_cls_q, expected_nav_q, valid_actions_mask
    ) -> Tuple[int, int]:
        """Select an action using epsilon-greedy policy.

        Args:
            state (dict): Current state

        Returns:
            Tuple[int, int]: Chosen action
        """
        # # Move state tensors to device and add batch dimension if needed
        # state = {
        #     k: v.to(self.device) if isinstance(v, torch.Tensor) else v
        #     for k, v in state.items()
        # }

        self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * math.exp(
            -1.0 * self.t_step / self.epsilon_decay
        )

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            with torch.no_grad():
                cls_action = expected_cls_q.argmax().item()
                # nav_action = expected_nav_q.argmax().item()
        else:
            # Random classification action
            cls_action = random.randrange(self.policy_net.cls_advantage[2].out_features)
            # Random navigation action from valid actions
            # valid_actions = torch.where(valid_actions_mask)[0]
            # nav_action = valid_actions[random.randrange(len(valid_actions))].item()

        # with torch.no_grad():
        #     cls_action = expected_cls_q.argmax().item()

        return cls_action   # , nav_action

    def optimize_model(self) -> Optional[torch.Tensor]:
        """Perform a single step of optimization on the policy network.

        Returns:
            Optional[torch.Tensor]: Loss value
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample a batch of experiences
        batch = self.memory.sample(self.batch_size)

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
        # criterion = nn.SmoothL1Loss()
        criterion = nn.MSELoss()
        cls_loss = criterion(policy_cls_q, expected_cls_q.unsqueeze(1))
        nav_loss = criterion(policy_nav_q, expected_nav_q.unsqueeze(1))
        total_loss = cls_loss + nav_loss

        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), max_norm=0.5)
        self.optimizer.step()

        return total_loss.item()

    def train(
        self,
        env,
        num_episodes: int,
        max_steps: int = 1000,
        train_dir: str = "data/aachen_graphs",
        image_dir: str = "data/aachen",
        render: bool = False,
    ) -> List[float]:
        episode_rewards_cls = []
        # episode_rewards_nav = []

        files = os.listdir(train_dir)

        for i, episode in enumerate(range(num_episodes)):
            # Load graph
            graph = torch.load(f"{train_dir}/{files[i]}")
            state = env.reset(new_graph=graph)
            episode_reward_cls = 0
            # episode_reward_nav = 0
            if render:
                # image_files = os.listdir(f"{image_dir}/images")
                # label_files = os.listdir(f"{image_dir}/labels")
                raw_image_path = "data/aachen_raw/aachen_000000_000019_leftImg8bit.png" #f"{image_dir}/images/{image_files[i]}"
                label_image_path = "data/aachen_labeled/aachen_000000_000019_gtFine_labelIds.png" #f"{image_dir}/labels/{label_files[i]}"
                raw_image = Image.open(raw_image_path)
                label_image = Image.open(label_image_path)
                raw_image_array = np.array(raw_image)
                label_image_array = np.array(label_image)
                image_size = raw_image_array.shape[:2]
                renderer = SegmentationRenderer(
                    original_image=raw_image_array,
                    ground_truth=label_image_array,
                    num_classes=self.num_classes,
                    image_size=image_size,
                )

            for step in range(max_steps):
                # Select action
                # expected_cls_q, expected_nav_q = self.policy_net(state)
                expected_cls_q = self.policy_net(state) # only classification for now 

                cls_action = self.select_action(
                    expected_cls_q=expected_cls_q,
                    expected_nav_q=0,   # random value for now since we are not using nav
                    valid_actions_mask=state["valid_actions_mask"],
                )

                # if selected action is the correct label print it
                if cls_action == graph.y[state["current_node"]]:
                    print(f"Correct label selected: {cls_action}")

                next_step, rewards, done = env.step(
                    {"cls": cls_action, "nav": 0} # random value for now since we are not using nav
                )

                if render:
                    y, x = graph.x[state["current_node"], :2].cpu().numpy().astype(int)
                    renderer.update_position((x, y))
                    renderer.update_segmentation((x, y), cls_action)
                    renderer.render((x, y))

                # Unpack rewards
                cls_rewards = rewards["cls"].to(self.device)
                # nav_rewards = rewards["nav"].to(self.device)
                print(cls_rewards)

                # Add rewards
                episode_reward_cls += cls_rewards[cls_action]
                # episode_reward_nav += nav_rewards[nav_action]

                # Compute Loss
                criterion = nn.SmoothL1Loss()
                # before calculted loss set the masked expected q values from nav from -inf to some finite
                # negative value
                # expected_nav_q[~state["valid_actions_mask"]] = -1

                cls_loss = criterion(expected_cls_q, cls_rewards)
                # nav_loss = criterion(expected_nav_q, nav_rewards)
                # total_loss = cls_loss + nav_loss

                # Log metrics to wandb
                wandb.log(
                    {
                        "step": self.t_step,
                        "classification_reward": cls_rewards[cls_action],
                        # "navigation_reward": nav_rewards[nav_action],
                        # "total_reward": cls_rewards[cls_action]
                        # + nav_rewards[nav_action],
                        "classification_loss": cls_loss.item(),
                        # "navigation_loss": nav_loss.item(),
                        # "total_loss": total_loss.item(),
                        "epsilon": self.epsilon,
                    }
                )

                # Optimize the model
                self.optimizer.zero_grad()
                cls_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1)

                self.optimizer.step()
                state = next_step
                self.t_step += 1

                if done:
                    print(f"Episode {episode} finished after {step} steps")
                    break

            episode_rewards_cls.append(episode_reward_cls)
            # episode_rewards_nav.append(episode_reward_nav)

            # Log episode metrics
            wandb.log(
                {
                    "episode": episode,
                    "episode_classification_reward": episode_reward_cls,
                    # "episode_navigation_reward": episode_reward_nav,
                    # "episode_total_reward": episode_reward_cls + episode_reward_nav,
                }
            )
            if render:
                renderer.close()

        wandb.finish()
        return episode_rewards_cls

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
    seed = 0

    # Create environment
    env = GraphSegmentationEnv(graph, seed=seed)

    # Create agent
    agent = GraphDQNAgent(
        node_feature_dim=node_feature_dim,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_output_dim=gnn_output_dim,
        dqn_hidden_dim=dqn_hidden_dim,
        num_classes=num_classes,
        seed=seed,
    )

    # Train agent
    num_episodes = 100
    episode_rewards = agent.train(env, num_episodes)

    # Plot training results
    agent.plot_training_results(episode_rewards)

    # Save model weights
    agent.save_model("graph_dqn_agent")
