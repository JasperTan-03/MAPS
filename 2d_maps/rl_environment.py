<<<<<<< HEAD
from typing import Dict, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from agent import AgentAction, SegmentationAgent
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

# Register this module as a custom environment
register(
    id="Segmentation-v0",
    entry_point="rl_environment:SegmentationEnv",
    # max_episode_steps=100,
)


class SegmentationEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, image: np.ndarray, labels: np.ndarray, step_limit: int = 100):
        self.image = image  # Input image to be segmented
        self.labels = labels  # Ground truth labels for segmentation
        self.height, self.width = image.shape[:2]
        self.step_limit = step_limit
        self.num_steps = 0

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
                    low=0,
                    high=max(self.height, self.width),
                    shape=(2,),
                    dtype=np.int32,
                ),
            }
        )

        # Visualization setup
        self.fig = None
        self.ax = None

    def reset(self, seed=None) -> Dict[str, np.ndarray]:
        super().reset(seed=seed)
        self.num_steps = 0
        return self.agent.reset(seed=seed)

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        self.num_steps += 1

        # Perform action and get reward
        reward = self.agent.perform_action(AgentAction(action))

        # Get new observation
        observation = self.agent.get_observation()

        # Check if the episode is done
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

    def render(self, mode="human"):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 5))
            self.fig.suptitle("Segmentation Progress")

        # Clear previous plots
        for ax in self.ax:
            ax.clear()

        # Plot original image
        self.ax[0].imshow(self.image, cmap="gray")
        self.ax[0].set_title("Original Image")

        # Plot current segmentation state
        self.ax[1].imshow(self.agent.state, cmap="coolwarm")
        self.ax[1].set_title("Current Segmentation")

        # Plot agent path
        self.ax[2].imshow(self.agent.path, cmap="hot")
        self.ax[2].set_title("Agent Path")

        # Mark current position on all plots
        for ax in self.ax:
            ax.plot(
                self.agent.current_position[1],
                self.agent.current_position[0],
                "go",
                markersize=10,
            )

        plt.draw()
        plt.pause(0.1)

        if mode == "rgb_array":
            # Convert plot to RGB array
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img


if __name__ == "__main__":
    # Create sample data
    height, width = 100, 100
    random_image = torch.randint(0, 2, (height, width)).numpy()
    random_labels = torch.randint(0, 2, (height, width)).numpy()

    # Create environment
    print("Creating Environment...")
    env = SegmentationEnv(image=random_image, labels=random_labels, step_limit=10)

    # Run Episode
    print("Running Episode...")
    obs = env.reset()
    total_reward = 0

    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward

        env.render()

        if done:
            print(f"Episode finished!")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Coverage: {info['coverage']*100:.1f}%")
            print(f"Accuracy: {info['accuracy']*100:.1f}%")
            break

    plt.ioff()
    plt.show()
=======
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete


class SegmentationEnv(gym.Env):
    def __init__(self, image, labels):
        super(SegmentationEnv, self).__init__()
        self.image = image  # Input image to be segmented
        self.labels = labels  # Ground truth labels for segmentation
        self.height, self.width = image.shape[:2]
        self.state = None

        # Action space: 5 actions (4 directions + 1 classification)
        self.action_space = Discrete(5)

        # State space: position and pixel features
        self.observation_space = Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

    def reset(self):
        # Initialize agent positions and labeled image
        self.state = np.zeros((self.height, self.width))  # Agent state
        self.current_position = (
            np.random.randint(self.height),
            np.random.randint(self.width),
        )
        return self.state

    def step(self, action):
        reward = 0
        done = False

        # Parse action
        if action == 0:  # Classify pixel
            label = self.classify(self.current_position)
            if label == self.labels[self.current_position]:  # Correct classification
                reward += 1
            else:
                reward -= 1
        else:  # Move agent
            self.move(action)

        # Check if episode is done
        if self.is_done():
            done = True

        # Next observation
        observation = self.get_observation()
        return observation, reward, done, {}

    def classify(self, position):
        # Placeholder for classification model
        return np.random.choice([0, 1])  # Classify as object (1) or background (0)

    def move(self, direction):
        # Move agent in the given direction
        x, y = self.current_position
        if direction == 1:  # Up
            self.current_position = (max(0, x - 1), y)
        elif direction == 2:  # Down
            self.current_position = (min(self.height - 1, x + 1), y)
        elif direction == 3:  # Left
            self.current_position = (x, max(0, y - 1))
        elif direction == 4:  # Right
            self.current_position = (x, min(self.width - 1, y + 1))

    def get_observation(self):
        # Return current image with agent's position highlighted
        observation = self.state.copy()
        x, y = self.current_position
        observation[x, y] = 255  # Mark agent position
        return observation

    def is_done(self):
        # Check termination conditions
        return np.all(self.state > 0)
>>>>>>> ef32f6c (WIP environment and PPO agent)
