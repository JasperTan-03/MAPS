
import yaml
import torch
from GraphDQNAgent import GraphDQNAgent
from rl_environment import GraphSegmentationEnv

# Load configuration from YAML file
with open('configs/training_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

# Load graph
graph = torch.load(config['graph_path'])

# Initialize agent and environment with parameters from config
agent = GraphDQNAgent(
        node_feature_dim=graph.x.size(-1),
        gnn_hidden_dim=config['gnn_hidden_dim'],
        gnn_output_dim=config['gnn_output_dim'],
        dqn_hidden_dim=config['dqn_hidden_dim'],
        num_classes=config['num_classes'],
        k_hops=config['k_hops']
)
env = GraphSegmentationEnv(graph, config['num_classes'])

# Train the agent
agent.train(
        env,
        train_dir=config['train_dir'],
        num_episodes=config['num_episodes'],
        max_steps=config['max_steps'],
        render=config['render']
)
