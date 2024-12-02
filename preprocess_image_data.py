import os
import glob
from build_network import build_image_network, save_graph  # Assuming build_point_cloud_network is in build_network.py

from tqdm import tqdm

def create_graphs_for_dataset(raw_images_path, labeled_images_path, output_graphs_path):
    # Ensure the output directory exists
    if not os.path.exists(output_graphs_path):
        os.makedirs(output_graphs_path)

    # Iterate over each image in the raw images path
    raw_image_files = glob.glob(os.path.join(raw_images_path, "*.png"))
    for raw_image_path in tqdm(raw_image_files):
        # Extract the base name and construct the corresponding labeled image path
        raw_filename = os.path.basename(raw_image_path)
        labeled_filename = raw_filename.replace("leftImg8bit", "gtFine_labelIds")
        labeled_image_path = os.path.join(labeled_images_path, labeled_filename)

        # Check if the corresponding labeled image exists
        if os.path.exists(labeled_image_path):
            # Construct the output graph file path
            base_filename = raw_filename.rsplit('_', 1)[0]  # Split at the last underscore
            output_graph_path = os.path.join(output_graphs_path, f"{base_filename}.pt")

            # Build the graph from the raw and labeled images
            graph = build_image_network(raw_image_path, labeled_image_path)

            # Save the graph
            save_graph(graph, output_graph_path)
            print(f"Graph saved to {output_graph_path}")

if __name__ == "__main__":
    raw_images_path = "data/aachen_raw"
    labeled_images_path = "data/aachen_labeled"
    output_graphs_path = "data/aachen_graphs"
    
    create_graphs_for_dataset(raw_images_path, labeled_images_path, output_graphs_path)
