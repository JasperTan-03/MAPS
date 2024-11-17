import argparse
import os

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.neighbors import KDTree
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import (
    is_undirected,
    structured_negative_sampling_feasible,
    subgraph,
    to_undirected,
)
from tqdm import tqdm
from PIL import Image


def get_memory_usage():
    """Get current memory usage statistics."""
    # System memory
    ram = psutil.Process().memory_info()
    ram_usage = ram.rss / (1024 * 1024 * 1024)  # Convert to GB
    ram_total = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # Convert to GB

    # GPU memory if available
    gpu_stats = {}
    if torch.cuda.is_available():
        gpu_stats = {
            "allocated": torch.cuda.memory_allocated()
            / (1024 * 1024 * 1024),  # Convert to GB
            "reserved": torch.cuda.memory_reserved()
            / (1024 * 1024 * 1024),  # Convert to GB
            "max_allocated": torch.cuda.max_memory_allocated()
            / (1024 * 1024 * 1024),  # Convert to GB
        }

    return {
        "ram_usage_gb": ram_usage,
        "ram_total_gb": ram_total,
        "ram_percentage": (ram_usage / ram_total) * 100,
        "gpu_stats": gpu_stats,
    }


def print_memory_stats(memory_stats, step=""):
    """Print memory statistics in a formatted way."""
    print(f"\n--- Memory Usage {step} ---")
    print(
        f"RAM Usage: {memory_stats['ram_usage_gb']:.2f}GB / {memory_stats['ram_total_gb']:.2f}GB "
        f"({memory_stats['ram_percentage']:.1f}%)"
    )

    if memory_stats["gpu_stats"]:
        print(f"GPU Memory:")
        print(f"  Allocated: {memory_stats['gpu_stats']['allocated']:.2f}GB")
        print(f"  Reserved:  {memory_stats['gpu_stats']['reserved']:.2f}GB")
        print(f"  Max Allocated: {memory_stats['gpu_stats']['max_allocated']:.2f}GB")


def load_and_process_point_cloud(filename, chunk_size=1000000, k=5):
    edge_index_list = []
    node_attr_list = []
    label_list = []
    total_points = 0
    
    # Initial memory state
    print_memory_stats(get_memory_usage(), "Initial State")
    
    print("Loading and processing point cloud in chunks...")
    for i, chunk in enumerate(tqdm(pd.read_csv(filename, sep=" ", header=None, chunksize=chunk_size))):
        # Memory before processing chunk
        if i % 5 == 0:  # Print every 5 chunks to avoid too much output
            print_memory_stats(get_memory_usage(), f"Before Processing Chunk {i}")
            
        points = chunk.iloc[:, :3].values  # Extract x, y, z coordinates
        colors = chunk.iloc[:, 3:].values  # Extract intensity, R, G, B values
        # temporary value for label of node classifation label is 0 for now
        labels = np.zeros((points.shape[0], 1))
        
        
        # Combine point and color information as node attributes
        node_attrs = np.hstack((points, colors))
        node_attrs = torch.tensor(node_attrs, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            node_attrs = node_attrs.cuda()
        
        # Create edges with KNN and add to list
        edge_index = knn_graph(node_attrs[:, :3], k=k, loop=False)
        
        # Append current chunk results
        edge_index_list.append(edge_index)
        node_attr_list.append(node_attrs)
        label_list.append(labels)
        
        total_points += node_attrs.shape[0]
        
        # Memory after processing chunk
        if i % 5 == 0:  # Print every 5 chunks
            print_memory_stats(get_memory_usage(), f"After Processing Chunk {i}")
    
    print("\nConcatenating results...")
    # Memory before concatenation
    print_memory_stats(get_memory_usage(), "Before Concatenation")
    
    # Concatenate edges and node attributes from all chunks
    edge_index = torch.cat(edge_index_list, dim=1)
    node_attrs = torch.cat(node_attr_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    
    # Memory after concatenation
    print_memory_stats(get_memory_usage(), "After Concatenation")
    
    # Make sure the edge index is undirected for connectivity check
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)

    # Create PyG Data object
    graph_data = Data(x=node_attrs, edge_index=edge_index, y=labels)

    # Check if the graph is connected using structured_negative_sampling_feasible
    is_connected = structured_negative_sampling_feasible(edge_index)

    # Print final statistics
    print("\nProcessing complete.")
    print(f"Total points (nodes): {total_points}")
    print(f"Total edges: {edge_index.size(1)}")
    print(f"Graph is {'connected' if is_connected else 'disconnected'}.")
    
    # Final memory state
    print_memory_stats(get_memory_usage(), "Final State")

    return graph_data


def load_and_process_image(filename, k=5):
    image = Image.open(filename)
    image = image.convert("RGB")
    width, height = image.size

    node_attr_list = []
    edge_index_list = []
    label_list = []

    print_memory_stats(get_memory_usage(), "Initial State")

    print("Loading and processing image...")
    for y in range(height):
        for x in range(width):
            r, g, b = image.getpixel((x, y))
            node_attrs = [x, y, 0, r, g, b]
            node_attr_list.append(node_attrs)
            label_list.append([0])  # Use y as the label for the node

            # Create edges to neighboring pixels
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    if dx != 0 and dy != 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        edge_index_list.append([y * width + x, ny * width + nx])

    node_attrs = torch.tensor(node_attr_list, dtype=torch.float)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    labels = torch.tensor(label_list, dtype=torch.long)

    if torch.cuda.is_available():
        node_attrs = node_attrs.cuda()
        edge_index = edge_index.cuda()
        labels = labels.cuda()

    print_memory_stats(get_memory_usage(), "After Processing Image")

    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)

    graph_data = Data(x=node_attrs, edge_index=edge_index, y=labels)

    is_connected = structured_negative_sampling_feasible(edge_index)

    print("\nProcessing complete.")
    print(f"Total points (nodes): {len(node_attr_list)}")
    print(f"Total edges: {edge_index.size(1)}")
    print(f"Graph is {'connected' if is_connected else 'disconnected'}.")

    print_memory_stats(get_memory_usage(), "Final State")

    return graph_data


def save_graph(graph_data, filename):
    """Save PyG Data object to a file."""
    torch.save(graph_data, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--point_cloud",
        type=str,
        default="data/bildstein_station1_xyz_intensity_rgb.txt",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000000,
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    data = args.point_cloud
    chunk_size = args.chunk_size
    output_dir = args.output_dir

    if args.image:
        graph = load_and_process_image(args.image, k=5)
    else:
        graph = load_and_process_point_cloud(data, chunk_size=chunk_size, k=5)

    if args.save:
        input_filename = os.path.basename(data)
        save_filename = os.path.join(
            output_dir, f"{os.path.splitext(input_filename)[0]}.pt"
        )
        save_graph(graph, save_filename)
        print(f"Graph saved to {save_filename}")
