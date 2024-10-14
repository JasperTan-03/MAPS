import argparse

import numpy as np
import networkx as nx
from sklearn.neighbors import KDTree


def load_point_cloud(filename):
    data = np.loadtxt(filename)
    print(data.shape)
    points = data[:, :3]  # get x, y, z
    colors = data[:, 3:]  # get r, g, b
    return points, colors


def build_knn_network(points: np.ndarray, colors: np.ndarray, k: int = 5):

    tree = KDTree(points)
    distances, indices = tree.query(points, k=k + 1)

    # Create a graph
    G = nx.Graph()

    for i in range(points.shape[0]):
        intensity, r, g, b = colors[i]
        x, y, z = points[i]
        G.add_node(i, x=x, y=y, z=z, intensity=intensity, r=r, g=g, b=b)

    for i in range(points.shape[0]):
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

    points, colors = load_point_cloud(data)

    graph = build_knn_network(points, colors, k=5)

    save_graph(graph, "data/bildstein_station1.gexf")
