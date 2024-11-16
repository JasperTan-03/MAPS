import random
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import pygame
import torch


class AgentAction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class SegmentationAgent:
    def __init__(self, height: int, width: int, labels: np.ndarray):
        self.height = height
        self.width = width

        self.labels = labels
        self.state = None
        self.path = None
        self.current_position = None
        self.visited_positions = set()

        self.reset()

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.state = np.zeros((self.height, self.width))
        self.path = np.zeros((self.height, self.width))
        self.visited_positions = set()

        # Start from a random unvisited position
        self.current_position = (
            random.randint(0, self.height - 1),
            random.randint(0, self.width - 1),
        )
        self.visited_positions.add(self.current_position)
        self.path[self.current_position] = 1

        return self.get_observation()

    def perform_action(self, action: AgentAction) -> float:
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

    def classify(self):
        # Get the true label for the current position
        true_label = self.labels[self.current_position]

        # Add some noise to make it more realistic
        noise = np.random.random() < 0.5
        return 1 - true_label if noise else true_label

    def calculate_reward(self, prediction: int) -> float:
        true_label = self.labels[self.current_position]

        # Higher reward for correct classification
        if prediction == true_label:
            reward = 1.0
        else:
            reward = -0.5

        if (
            len(self.visited_positions) > 1
            and self.current_position in self.visited_positions
        ):
            reward -= 0.25

        return reward

    def get_observation(self) -> Dict[str, np.ndarray]:
        return {
            "state": self.state.copy(),
            "path": self.path.copy(),
            "position": np.array(self.current_position),
        }

    def render(self):
        pygame.init()
        screen = pygame.display.set_mode((self.width * 10, self.height * 10))
        pygame.display.set_caption("Segmentation Agent")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((0, 0, 0))

            for y in range(self.height):
                for x in range(self.width):
                    color = (255, 255, 255) if self.state[y, x] == 255 else (0, 0, 0)
                    pygame.draw.rect(
                        screen,
                        color,
                        pygame.Rect(x * 10, y * 10, 10, 10),
                    )

            pygame.display.flip()

        pygame.quit()

    def get_position(self):
        return self.current_position
