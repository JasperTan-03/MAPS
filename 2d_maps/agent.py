import random
from enum import Enum

import numpy as np
import pygame
import torch


class AgentAction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    CLASSIFY = 4


class SegmentationAgent:
    def __init__(self, height, width, labels):
        self.height = height
        self.width = width

        self.state = None
        self.path = None
        self.current_position = None
        self.labels = labels

        self.reset()

    def reset(self, seed=None):
        self.state = np.zeros((self.height, self.width))
        self.path = np.zeros((self.height, self.width))

        random.seed(seed)
        self.current_position = (
            random.randint(0, self.height - 1),
            random.randint(0, self.width - 1),
        )
        self.path[self.current_position] = 1

    def perform_action(self, action: AgentAction) -> bool:
        if action == AgentAction.LEFT:
            self.current_position = (
                self.current_position[0],
                max(0, self.current_position[1] - 1),
            )
        elif action == AgentAction.RIGHT:
            self.current_position = (
                self.current_position[0],
                min(self.width - 1, self.current_position[1] + 1),
            )
        elif action == AgentAction.UP:
            self.current_position = (
                max(0, self.current_position[0] - 1),
                self.current_position[1],
            )
        elif action == AgentAction.DOWN:
            self.current_position = (
                min(self.height - 1, self.current_position[0] + 1),
                self.current_position[1],
            )
        self.state[self.current_position] = self.classify()

        self.path[self.current_position] = 1
        return self.is_done()

    def classify(self):
        return random.choice([1, 2])

    def is_done(self):
        return self.labels[self.current_position] == self.state[self.current_position]

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

    def get_observation(self):
        return self.state.copy(), self.path.copy()

    def get_position(self):
        return self.current_position
