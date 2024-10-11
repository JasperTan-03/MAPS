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
            low=0, high=255, shape=(self.height, self.width, 4), dtype=np.uint8
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
