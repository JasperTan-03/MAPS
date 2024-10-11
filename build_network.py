import argparse

import numpy as np
import networkx as nx
from sklearn.neighbors import KDTree


def load_point_cloud(filename):
    data = np.loadtxt(filename, max_rows=1000)
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
        G.add_node(i, pos=points[i], intensity=intensity, r=r, g=g, b=b)

    for i in range(points.shape[0]):
        for j in range(1, k + 1):
            G.add_edge(i, indices[i, j], weight=distances[i, j])

    return G


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

    print("Number of nodes:", graph.number_of_nodes())

    print("Number of edges:", graph.number_of_edges())

    print("Average degree:", np.mean(list(dict(graph.degree()).values())))

    print("Average shortest path length:", nx.average_shortest_path_length(graph))

    print("Average clustering coefficient:", nx.average_clustering(graph))
