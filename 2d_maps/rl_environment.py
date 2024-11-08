import agent
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.envs.registration import register
from gymnasium.spaces import Box, Discrete
from gymnasium.utils.env_checker import check_env

# Register this module as a custom environment
register(
    id="Segmentation-v0",
    entry_point="rl_environment:SegmentationEnv",
    # max_episode_steps=100,
)


class SegmentationEnv(gym.Env):
    def __init__(self, image, labels, step_limit=100):
        self.image = image  # Input image to be segmented
        self.labels = labels  # Ground truth labels for segmentation
        self.height, self.width = image.shape[:2]
        self.step_limit = step_limit
        self.num_steps = 0
        self.obs = None

        # Initialize agent
        self.agent = agent.SegmentationAgent(self.height, self.width, labels)

        # Action space: 5 actions (4 directions)
        self.action_space = Discrete(4)

        # Observation space: 2D image with agent's position highlighted
        self.observation_space = Box(
            low=0, high=1, shape=(self.height, self.width), dtype=np.uint8
        )

    def reset(self, seed=None):
        super().reset(seed=seed)

        # Reset agent
        self.agent.reset(seed=seed)

        # Reset obs
        state, path = self.agent.get_observation()
        self.obs = {"state": state, "path": path}

        return self.obs

    def step(self, action):
        done = False if self.num_steps < self.step_limit else True

        # Perform action
        reward = self.agent.perform_action(agent.AgentAction(action))
        self.num_steps += 1

        # Update observation
        state, path = self.agent.get_observation()
        self.obs = {"state": state, "path": path}

        # Additional Info
        info = {}

        return self.obs, reward, done, info

    def render(self):
        self.agent.render()


if __name__ == "__main__":
    random_image = torch.randint(0, 2, (100, 100))
    random_labels = torch.randint(0, 2, (100, 100))
    print("start")
    env = gym.make(
        "Segmentation-v0", image=random_image, labels=random_labels, step_limit=1000
    )
    # print("initialize")

    obs = env.reset()
    # print("starting state:", obs["state"])

    while True:
        action = env.action_space.sample()
        # print(action)
        obs, reward, done, info = env.step(action)
        # env.render()

        if done:
            break

    # Display the final state
    plt.imshow(obs["state"])
    plt.show()

    plt.imshow(obs["path"])
    plt.show()
