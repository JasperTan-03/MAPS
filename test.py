from GraphDQNAgent import GraphDQNAgent
from rl_environment import GraphSegmentationEnv
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from render import SegmentationRenderer
from scipy.ndimage import gaussian_filter

graph = torch.load("downsampled_graph.pt")
num_classes = 34
folder = "./data/aachen_graphs"

# # Load image with PIL
# raw_image_path = f"{folder}/aachen_000000_000019_leftImg8bit.png"
# label_image_path = f"{folder}/aachen_000000_000019_gtFine_labelIds.png"
# raw_image = Image.open(raw_image_path)
# label_image = Image.open(label_image_path)

# # Display the image
# plt.imshow(raw_image)
# plt.imshow(label_image)

# # Convert to numpy array
# raw_image_array = np.array(raw_image)
# label_image_array = np.array(label_image)

# # Downsample the image
# def downsample_image(image_array, factor, sigma=2):
#     blurred_image = gaussian_filter(image_array, sigma=sigma)
#     downsampled_image = blurred_image[::factor, ::factor]
#     return downsampled_image

# downsampled_raw_image = downsample_image(raw_image_array, 4, 5)
# downsampled_label_image = downsample_image(label_image_array, 4, 5)
# downsampled_raw_image.shape, downsampled_raw_image.shape

# image_size = downsampled_raw_image.shape[:2]

# create an agent
"""        self,
        node_feature_dim: int,
        gnn_hidden_dim: int,
        gnn_output_dim: int,
        dqn_hidden_dim: int,
        num_classes: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        """

agent = GraphDQNAgent(graph.x.size(-1), 64, 64, 64, num_classes)
env = GraphSegmentationEnv(graph, num_classes)

agent.train(env, num_episodes=20, max_steps=10000, render=True)
