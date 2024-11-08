import random
from enum import Enum

import pygame
import torch


class AgentAction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class PicturePixel(Enum):
    OBJECT = 1
    BACKGROUND = 0


class SegmentationAgent:
    def __init__(self, height, width, labels):
        self.height = height
        self.width = width

        self.state = torch.zeros((height, width))
        self.current_position = None
        self.labels = labels

        self.reset()

    def reset(self, seed=None):
        self.state = torch.zeros((self.height, self.width))

        random.seed(seed)
        self.current_position = (
            random.randint(0, self.height - 1),
            random.randint(0, self.width - 1),
        )
        self.state[self.current_position] = 255

    def perform_action(self, action) -> bool:
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

        return self.is_done()

    def is_done(self):
        return self.labels[self.current_position] == PicturePixel.OBJECT.value

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
        observation = self.state.clone()
        x, y = self.current_position
        observation[x, y] = 255
        return observation
