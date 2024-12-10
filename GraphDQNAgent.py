import math
import os
import random
from collections import deque, namedtuple
from typing import Dict, List, Optional, Tuple

import matplotlib as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from PIL import Image
from torch_geometric.data import Batch, Data
from tqdm import tqdm

import wandb
from neural_net import DuelingGraphDQN
from render import SegmentationRenderer
from rl_environment import GraphSegmentationEnv

Experience = namedtuple(
    "Experience",
    (
        "state",       # Current state
        "action_cls",  # Action taken (classification)
        "next_state",  # Next state
        "reward_cls",  # Reward for classification action
        "done",        # Whether the episode is done
    ),
)


class ReplayBuffer:
    """Experience replay buffer with uniform sampling"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_cls, next_state, reward_cls, done):
        """Save an experience"""
        state_data = Data(x=state["x"], edge_index=state["edge_index"], current_node=state["current_node"], valid_actions_mask=state["valid_actions_mask"])
        next_state_data = Data(x=next_state["x"], edge_index=next_state["edge_index"], current_node=next_state["current_node"], valid_actions_mask=next_state["valid_actions_mask"])
        
        self.buffer.append(Experience(state_data, action_cls, next_state_data, reward_cls, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)
        return Experience(*zip(*experiences))

    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()



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

    # def select_action(
    #     self, expected_cls_q, expected_nav_q, valid_actions_mask
    # ) -> Tuple[int, int]:
    #     """Select an action using epsilon-greedy policy.

    #     Args:
    #         state (dict): Current state

    #     Returns:
    #         Tuple[int, int]: Chosen action
    #     """
    #     # # Move state tensors to device and add batch dimension if needed
    #     # state = {
    #     #     k: v.to(self.device) if isinstance(v, torch.Tensor) else v
    #     #     for k, v in state.items()
    #     # }

    #     self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * math.exp(
    #         -1.0 * self.t_step / self.epsilon_decay
    #     )

    #     # Epsilon-greedy action selection
    #     if random.random() > self.epsilon:
    #         with torch.no_grad():
    #             cls_action = expected_cls_q.argmax().item()
    #             # nav_action = expected_nav_q.argmax().item()
    #     else:
    #         # Random classification action
    #         cls_action = random.randrange(self.policy_net.cls_advantage[2].out_features)
    #         # Random navigation action from valid actions
    #         # valid_actions = torch.where(valid_actions_mask)[0]
    #         # nav_action = valid_actions[random.randrange(len(valid_actions))].item()

    #     # with torch.no_grad():
    #     #     cls_action = expected_cls_q.argmax().item()

    #     return cls_action   # , nav_action

    def optimize_model(self) -> Optional[torch.Tensor]:
        """Perform a single step of optimization on the policy network."""
        if len(self.memory) < self.batch_size:
            return None

        # Sample a batch of experiences
        batch = self.memory.sample(self.batch_size)

        # Create mask for non-final states
        non_final_mask = torch.tensor([s["current_node"] is not None for s in batch.next_state], device=self.device)
        # Filter batch components based on the mask
        state_batch = [s for i, s in enumerate(batch.state) if non_final_mask[i]]
        next_state_batch = [s for i, s in enumerate(batch.next_state) if non_final_mask[i]]
        action_cls_batch = torch.cat([a.view(-1) for i, a in enumerate(batch.action_cls) if non_final_mask[i]]).unsqueeze(1)

        reward_batch = torch.cat([r.view(-1, self.num_classes) for i, r in enumerate(batch.reward_cls) if non_final_mask[i]], dim=0)
        # Convert states to lists of Data objects
        def create_data(state):
            return Data(
                x=state["x"],
                edge_index=state["edge_index"],
                current_node=state["current_node"].view(-1),
                valid_actions_mask=state["valid_actions_mask"].view(-1),
            )

        state_batch = Batch.from_data_list([create_data(s) for s in state_batch])
        if next_state_batch:
            next_state_batch = Batch.from_data_list([create_data(s) for s in next_state_batch])

        # Current Q values (prediction from policy network)
        policy_cls_q = torch.zeros((len(non_final_mask), self.num_classes), device=self.device)
        policy_cls_q_values = self.policy_net(state_batch)
        policy_cls_q[non_final_mask] = policy_cls_q_values

        # Compute target Q values
        next_cls_q = torch.zeros((len(non_final_mask), self.num_classes), device=self.device)
        with torch.no_grad():
            if next_state_batch:
                next_cls_q_values = self.target_net(next_state_batch)
                next_cls_q[non_final_mask] = next_cls_q_values

        # Compute expected Q values
        expected_cls_q = (next_cls_q * self.gamma) + reward_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        cls_loss = criterion(policy_cls_q, expected_cls_q)

        # Optimize the model
        self.optimizer.zero_grad()
        cls_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return cls_loss.item()

    
    def select_action(self, state):

        
        
        self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * math.exp(
            -1.0 * self.t_step / self.epsilon_decay
        )

        self.t_step += 1

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                cls_action = q_values.max(1)[1].view(1, 1)
        else:
            cls_action = torch.tensor([[random.randrange(self.num_classes)]], device=self.device, dtype=torch.long)

        return cls_action
        

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
            
            if i == 0: 
                continue

            # Load graph
            graph = torch.load(f"{train_dir}/{files[i]}")
            state = env.reset(new_graph=graph)
            episode_reward_cls = 0
            self.memory.clear()

            if render:
                # image_files = os.listdir(f"{image_dir}/images")
                # label_files = os.listdir(f"{image_dir}/labels")
                # raw_image_path = "data/aachen_raw_downsampled/aachen_000000_000019_leftImg8bit.png" #f"{image_dir}/images/{image_files[i]}"
                # label_image_path = "data/aachen_labeled_downsampled/aachen_000000_000019_gtFine_labelIds.png" #f"{image_dir}/labels/{label_files[i]}"
                # raw_image = Image.open(raw_image_path)
                # label_image = Image.open(label_image_path)
                # raw_image_array = np.array(raw_image)
                # label_image_array = np.array(label_image)
                # image_size = raw_image_array.shape[:2]
                # renderer = SegmentationRenderer(
                #     original_image=raw_image_array,
                #     ground_truth=label_image_array,
                #     num_classes=self.num_classes,
                #     image_size=image_size,
                # )
                raw_image_path = f"data/aachen_raw_downsampled/{files[i].replace('.pt', '_leftImg8bit.png')}"
                raw_image = Image.open(raw_image_path)
                raw_image_array = np.array(raw_image)
                predictions = np.zeros_like(raw_image_array)


            for step in range(max_steps):
                
                cls_action = self.select_action(state)

                next_state, rewards, done = env.step(
                    {"cls": cls_action, "nav": 0} # random value for now since we are not using nav
                )

                if render:
                    y, x = graph.x[state["current_node"].squeeze(), :2].cpu().numpy().astype(int)
                    # renderer.update_position((x, y))
                    # renderer.update_segmentation((x, y), cls_action)
                    # renderer.render((x, y))
                    predictions[x, y] = cls_action.item()

                # Unpack rewards 
                cls_rewards = rewards["cls"].to(self.device)

                # Add rewards
                episode_reward_cls += cls_rewards[cls_action]

                self.memory.push(state, cls_action, next_state, cls_rewards, done)

                state = next_state

                loss = self.optimize_model()

                wandb.log(
                    {
                        "step": self.t_step,
                        "classification_reward": cls_rewards[cls_action],
                        "classification_loss": loss,
                        "epsilon": self.epsilon,
                        "total_reward": episode_reward_cls
                    }
                )

                # soft update of target network
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = self.tau*policy_net_state_dict[key] + (1-self.tau)*target_net_state_dict[key]
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    print(f"Episode {episode} finished after {step} steps")
                    # save the predictions
                    predictions = Image.fromarray(predictions.astype(np.uint8))
                    os.makedirs("predictions", exist_ok=True)
                    predictions.save(f"predictions/{files[i].replace('.pt', '.png')}")


                    self.save_model("maps")
                    break




        #         expected_cls_q = self.policy_net(state) # only classification for now 

        #         cls_action = self.select_action(
        #             expected_cls_q=expected_cls_q,
        #             expected_nav_q=0,   # random value for now since we are not using nav
        #             valid_actions_mask=state["valid_actions_mask"],
        #         )

        #         # if selected action is the correct label print it
        #         if cls_action == graph.y[state["current_node"]]:
        #             print(f"Correct label selected: {cls_action}")

        #         next_step, rewards, done = env.step(
        #             {"cls": cls_action, "nav": 0} # random value for now since we are not using nav
        #         )

        #         if render:
        #             y, x = graph.x[state["current_node"], :2].cpu().numpy().astype(int)
        #             renderer.update_position((x, y))
        #             renderer.update_segmentation((x, y), cls_action)
        #             renderer.render((x, y))

        #         # Unpack rewards
        #         cls_rewards = rewards["cls"].to(self.device)
        #         print(cls_rewards)

        #         # Add rewards
        #         episode_reward_cls += cls_rewards[cls_action]

        #         # Compute Loss
        #         criterion = nn.SmoothL1Loss()
        #         cls_loss = criterion(expected_cls_q, cls_rewards)

        #         # Log metrics to wandb
        #         wandb.log(
        #             {
        #                 "step": self.t_step,
        #                 "classification_reward": cls_rewards[cls_action],
        #                 # "navigation_reward": nav_rewards[nav_action],
        #                 # "total_reward": cls_rewards[cls_action]
        #                 # + nav_rewards[nav_action],
        #                 "classification_loss": cls_loss.item(),
        #                 # "navigation_loss": nav_loss.item(),
        #                 # "total_loss": total_loss.item(),
        #                 "epsilon": self.epsilon,
        #             }
        #         )

        #         # Optimize the model
        #         self.optimizer.zero_grad()
        #         cls_loss.backward()

        #         # Gradient clipping
        #         torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1)

        #         self.optimizer.step()
        #         state = next_step
        #         self.t_step += 1

        #         if done:
        #             print(f"Episode {episode} finished after {step} steps")
        #             break

        #     episode_rewards_cls.append(episode_reward_cls)
        #     # episode_rewards_nav.append(episode_reward_nav)

        #     # Log episode metrics
        #     wandb.log(
        #         {
        #             "episode": episode,
        #             "episode_classification_reward": episode_reward_cls,
        #             # "episode_navigation_reward": episode_reward_nav,
        #             # "episode_total_reward": episode_reward_cls + episode_reward_nav,
        #         }
        #     )
        #     if render:
        #         renderer.close()

        # wandb.finish()

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
        torch.save(self.policy_net.state_dict(), f"weights/{path}_policy.pt")
        torch.save(self.target_net.state_dict(), f"weights/{path}_target.pt")

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
