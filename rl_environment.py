import numpy as np
import torch
import dgl
from gym import spaces
from gym import Env

class MultiAgentTaskSchedulingEnv(Env):
    def __init__(self, num_nodes, task_arrival_rate):
        self.num_nodes = num_nodes
        self.task_arrival_rate = task_arrival_rate

        # Observation space for each node: [local_state, global_state (snapshot from messages)]
        self.observation_space = spaces.Dict({
            'local_state': spaces.Dict({
                'resources': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'timestamp': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                'queue_length': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32)
            }),
            'global_snapshot': spaces.Box(low=0, high=1, shape=(num_nodes, 3), dtype=np.float32)
        })

        # Action space: Scheduling a task on the current node or sending messages to other nodes
        self.action_space = spaces.Discrete(2)  # 0: schedule a task, 1: send a message

        # Initial state for each node
        self.node_states = {
            node_id: {
                'resources': np.random.uniform(0.1, 1),
                'timestamp': 0.0,
                'queue_length': 0,
                'global_snapshot': np.zeros((num_nodes, 3))  # Simulates vector clock and other nodes' states
            }
            for node_id in range(num_nodes)
        }

        # Create graph structure
        self.graph = dgl.DGLGraph()
        self.graph.add_nodes(num_nodes)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                self.graph.add_edge(i, j)
                self.graph.add_edge(j, i)

    def reset(self):
        self.node_states = {
            node_id: {
                'resources': np.random.uniform(0.1, 1),
                'timestamp': 0.0,
                'queue_length': 0,
                'global_snapshot': np.zeros((self.num_nodes, 3))
            }
            for node_id in range(self.num_nodes)
        }
        return self._get_observations()

    def _get_observations(self):
        # Returns the observation for all nodes (each node sees its local state + global snapshot)
        return {
            node_id: {
                'local_state': {
                    'resources': self.node_states[node_id]['resources'],
                    'timestamp': self.node_states[node_id]['timestamp'],
                    'queue_length': self.node_states[node_id]['queue_length'],
                },
                'global_snapshot': self.node_states[node_id]['global_snapshot']
            }
            for node_id in range(self.num_nodes)
        }

    def step(self, actions):
        # Perform actions for each node (actions will be a dictionary of node actions)
        rewards = {}
        for node_id, action in actions.items():
            if action == 0:  # Schedule task
                task_size = np.random.uniform(0.1, 0.5)
                self.node_states[node_id]['resources'] -= task_size
                self.node_states[node_id]['queue_length'] += 1
            elif action == 1:  # Send message to update other nodes
                for other_node_id in range(self.num_nodes):
                    if other_node_id != node_id:
                        self._send_message(node_id, other_node_id)
        
        # Simulate some processing and resource updates
        for node_id in range(self.num_nodes):
            self.node_states[node_id]['resources'] = np.clip(self.node_states[node_id]['resources'] + np.random.uniform(0.05, 0.1), 0, 1)
            rewards[node_id] = -np.var([self.node_states[n]['resources'] for n in range(self.num_nodes)])
        
        done = False  # Continue indefinitely
        return self._get_observations(), rewards, done, {}

    def _send_message(self, sender, receiver):
        # Update global snapshot for the receiver based on the sender's current state
        self.node_states[receiver]['global_snapshot'][sender] = np.array([
            self.node_states[sender]['resources'],
            self.node_states[sender]['timestamp'],
            self.node_states[sender]['queue_length']
        ])

    def render(self):
        for node_id in range(self.num_nodes):
            print(f"Node {node_id} State: {self.node_states[node_id]}")
