from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class SegmentationRenderer:
    def __init__(
        self,
        original_image: np.ndarray,
        ground_truth: np.ndarray,
        num_classes: int,
        image_size: Tuple[int, int],
    ):
        """
        Initialize the renderer for segmentation visualization.

        Args:
            original_image: Original input image (H x W)
            ground_truth: Ground truth segmentation (H x W)
            num_classes: Number of possible segmentation classes
            image_size: Tuple of (height, width) for the image
        """
        self.original_image = original_image
        self.ground_truth = ground_truth
        self.num_classes = num_classes
        self.image_size = image_size

        # Initialize visualization attributes
        self.fig = None
        self.ax = None

        # Initialize tracking maps
        self.agent_segmentation = np.zeros(image_size, dtype=np.int32)
        self.visit_map = np.zeros(image_size, dtype=np.int32)

    def update_position(self, position: Tuple[int, int]):
        """
        Update the agent's current position and visit map.

        Args:
            position: (x, y) coordinates of current position
        """
        x, y = position
        self.visit_map[x, y] += 1

    def update_segmentation(self, position: Tuple[int, int], class_label: int):
        """
        Update the segmentation map at the given position.

        Args:
            position: (x, y) coordinates to update
            class_label: Class label to assign
        """
        x, y = position
        self.agent_segmentation[x, y] = class_label

    def render(
        self, current_position: Tuple[int, int], mode: str = "human"
    ) -> Optional[np.ndarray]:
        """
        Render the current state of the segmentation process.

        Args:
            current_position: Current (x, y) position of the agent
            mode: Rendering mode ('human' or 'rgb_array')

        Returns:
            np.ndarray if mode is 'rgb_array', None otherwise
        """
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 10))
            self.fig.suptitle("Segmentation Progress")

        # Clear previous plots
        for ax_row in self.ax:
            for ax in ax_row:
                ax.clear()

        # Plot 1: Original Image
        self.ax[0, 0].imshow(self.original_image, cmap="gray")
        self.ax[0, 0].plot(
            current_position[1], current_position[0], "go", markersize=10
        )
        self.ax[0, 0].set_title("Original Image")

        # Plot 2: Ground Truth Segmentation
        gt_plot = self.ax[0, 1].imshow(
            self.ground_truth, cmap="tab10", vmin=0, vmax=self.num_classes - 1
        )
        self.ax[0, 1].plot(
            current_position[1], current_position[0], "go", markersize=10
        )
        self.ax[0, 1].set_title("Ground Truth Segmentation")

        # Plot 3: Agent's Segmentation
        seg_plot = self.ax[1, 0].imshow(
            self.agent_segmentation, cmap="tab10", vmin=0, vmax=self.num_classes - 1
        )
        self.ax[1, 0].plot(
            current_position[1], current_position[0], "go", markersize=10
        )
        self.ax[1, 0].set_title("Agent Segmentation")

        # Plot 4: Visit Heatmap
        visit_plot = self.ax[1, 1].imshow(self.visit_map, cmap="hot", norm=None)
        self.ax[1, 1].plot(
            current_position[1], current_position[0], "go", markersize=10
        )
        self.ax[1, 1].set_title("Visit Heatmap")

        # Adjust layout and display
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

        if mode == "rgb_array":
            # Convert plot to RGB array
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img

    def close(self):
        """Close the renderer."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


if __name__ == "__main__":
    # Example usage
    image = np.random.randint(0, 255, (100, 100))
    gt = np.random.randint(0, 5, (100, 100))
    renderer = SegmentationRenderer(image, gt, 5, (100, 100))

    for step in range(100):
        position = np.random.randint(0, 100, 2)
        class_label = np.random.randint(0, 5)
        renderer.update_position(position)
        renderer.update_segmentation(position, class_label)
        renderer.render(position)
    renderer.close()
