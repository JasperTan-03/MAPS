import argparse

import numpy as np
import networkx as nx
from sklearn.neighbors import KDTree


def load_point_cloud(filename):
    points = np.loadtxt(filename, max_rows=1000)
    print("Loaded", points.shape[0], "points")
    print(points[0])
    points = points[:, :3]  # get x, y, z
    print(points[0])
    return points


def build_knn_network(points: np.ndarray, k: int = 5):

    tree = KDTree(points)
    distances, indices = tree.query(points, k=k + 1)

    # Create a graph
    G = nx.Graph()
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

    points = load_point_cloud(data)

    graph = build_knn_network(points, k=5)

    print("Number of nodes:", graph.number_of_nodes())

    print("Number of edges:", graph.number_of_edges())

    print("Average degree:", np.mean(list(dict(graph.degree()).values())))

    print("Average shortest path length:", nx.average_shortest_path_length(graph))

    print("Average clustering coefficient:", nx.average_clustering(graph))
