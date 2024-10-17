import argparse

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import KDTree
from tqdm import tqdm

def load_point_cloud_chunks(filename, chunk_size=1000000):
    chunks = []
    for chunk in pd.read_csv(filename, sep=" ", header=None, chunksize=chunk_size):
        chunks.append(chunk)

    data = pd.concat(chunks)
    points = data.iloc[:, :3].values 
    colors = data.iloc[:, 3:].values
    return points, colors


def load_point_cloud(filename):
    data = np.loadtxt(filename)
    print(data.shape)
    points = data[:, :3]  # get x, y, z
    colors = data[:, 3:]  # get r, g, b
    return points, colors


def build_knn_network(points: np.ndarray, colors: np.ndarray, k: int = 5):

    print("Building KDTree")
    tree = KDTree(points)
    distances, indices = tree.query(points, k=k + 1)

    # Create a graph
    G = nx.Graph()

    print("Creating Nodes")
    for i in tqdm(range(points.shape[0])):
        intensity, r, g, b = colors[i]
        x, y, z = points[i]
        G.add_node(i, x=x, y=y, z=z, intensity=intensity, r=r, g=g, b=b)

    print("Creating Edges")
    for i in tqdm(range(points.shape[0])):
        for j in range(1, k + 1):
            G.add_edge(i, indices[i, j], weight=distances[i, j])

    return G

def save_graph(graph, filename):
    nx.write_gexf(graph, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--point_cloud",
        type=str,
        default="data/bildstein_station1_xyz_intensity_rgb.txt",
    )
    args = parser.parse_args()

    data = args.point_cloud

    points, colors = load_point_cloud_chunks(data, chunk_size=10000)

    graph = build_knn_network(points, colors, k=5)

    save_graph(graph, "data/bildstein_station1.gexf")
