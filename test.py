import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from GraphDQNAgent import GraphDQNAgent
from render import SegmentationRenderer
from rl_environment import GraphSegmentationEnv

graph = torch.load("data/aachen_graphs_downsampled/aachen_000000_000019.pt")
num_classes = 34
folder = "./data/aachen_graphs"


agent = GraphDQNAgent(node_feature_dim=graph.x.size(-1), gnn_hidden_dim=128,
                       gnn_output_dim=128, dqn_hidden_dim=128, num_classes=num_classes, k_hops=8)
env = GraphSegmentationEnv(graph, num_classes)

agent.train(env, train_dir= "data/aachen_graphs_downsampled",num_episodes=100, max_steps=5000000, render=True)