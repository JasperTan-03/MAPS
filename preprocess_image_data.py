import glob
import os

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter
from scipy.stats import mode
from tqdm import tqdm

from build_network import build_image_network, save_graph


def safe_read_image(image_path):
    """
    Safely read an image and convert it to a standard numpy array.

    Parameters:
    - image_path (str): Path to the image file

    Returns:
    numpy.ndarray: Image as a 2D or 3D numpy array
    """
    with Image.open(image_path) as img:
        # Convert to RGB if it's not already
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Convert to numpy array
        img_array = np.array(img)

        # Handle single channel images by repeating the channel
        if len(img_array.shape) == 2:
            img_array = np.repeat(img_array[:, :, np.newaxis], 3, axis=2)

    return img_array


def max_pool_image(image_array, pool_size=2):
    """
    Efficiently downsample an image using maximum filter.

    Parameters:
    - image_array (numpy.ndarray): Input image array
    - pool_size (int): Size of the pooling window

    Returns:
    numpy.ndarray: Downsampled image
    """
    # If it's a color image, max pool each channel
    if len(image_array.shape) == 3:
        pooled_channels = []
        for channel in range(image_array.shape[2]):
            channel_pooled = maximum_filter(
                image_array[:, :, channel], size=pool_size, mode="constant"
            )[::pool_size, ::pool_size]
            pooled_channels.append(channel_pooled)
        return np.stack(pooled_channels, axis=2)

    # For single channel images
    return maximum_filter(image_array, size=pool_size, mode="constant")[
        ::pool_size, ::pool_size
    ]


def downsample_labeled_image(label_array, pool_size=2):
    """
    Efficiently downsample a label image by finding the most frequent label.

    Parameters:
    - label_array (numpy.ndarray): Input label array
    - pool_size (int): Size of the pooling window

    Returns:
    numpy.ndarray: Downsampled label array
    """
    # Ensure the input is a 2D array
    if len(label_array.shape) > 2:
        label_array = label_array[:, :, 0]  # Take first channel if multi-channel

    # Reshape the array to make it easy to find mode
    h, w = label_array.shape
    new_h, new_w = h // pool_size, w // pool_size

    # Reshape and use mode to find most frequent label in each pool
    reshaped = label_array[: new_h * pool_size, : new_w * pool_size].reshape(
        new_h, pool_size, new_w, pool_size
    )

    # Find the most common label in each pool
    downsampled_label = mode(
        reshaped.transpose(0, 2, 1, 3).reshape(new_h, new_w, -1), axis=2
    ).mode.reshape(new_h, new_w)

    return downsampled_label


def save_max_pooled_images(
    raw_images_path,
    labeled_images_path,
    downsampled_raw_path,
    downsampled_labeled_path,
    pool_size=2,
):
    """
    Saves max-pooled raw and labeled images to specified directories.

    Parameters:
    - raw_images_path (str): Directory containing raw images.
    - labeled_images_path (str): Directory containing labeled images.
    - downsampled_raw_path (str): Directory to save downsampled raw images.
    - downsampled_labeled_path (str): Directory to save downsampled labeled images.
    - pool_size (int): Size of the pooling window (default is 2).
    """
    # Ensure output directories exist
    os.makedirs(downsampled_raw_path, exist_ok=True)
    os.makedirs(downsampled_labeled_path, exist_ok=True)

    # Use glob to find all raw image files
    raw_image_files = sorted(glob.glob(os.path.join(raw_images_path, "*.png")))

    # Use tqdm for progress tracking
    for raw_image_path in tqdm(raw_image_files, desc="Downsampling images"):
        try:
            # Construct corresponding labeled image path
            raw_filename = os.path.basename(raw_image_path)
            labeled_filename = raw_filename.replace("leftImg8bit", "gtFine_labelIds")
            labeled_image_path = os.path.join(labeled_images_path, labeled_filename)

            # Check if the corresponding labeled image exists
            if not os.path.exists(labeled_image_path):
                print(
                    f"Skipping {raw_image_path}: No corresponding labeled image found"
                )
                continue

            # Load images as numpy arrays using safe method
            raw_image = safe_read_image(raw_image_path)
            labeled_image = safe_read_image(labeled_image_path)

            # Downsample images
            raw_image_downsampled = max_pool_image(raw_image, pool_size)
            labeled_image_downsampled = downsample_labeled_image(
                labeled_image, pool_size
            )

            # Save downsampled images
            raw_output_path = os.path.join(downsampled_raw_path, raw_filename)
            labeled_output_path = os.path.join(
                downsampled_labeled_path, labeled_filename
            )

            Image.fromarray(raw_image_downsampled.astype(np.uint8)).save(
                raw_output_path
            )
            Image.fromarray(labeled_image_downsampled.astype(np.uint8)).save(
                labeled_output_path
            )

        except Exception as e:
            print(f"Error processing {raw_image_path}: {e}")
            # Optionally, print the full traceback for more detailed debugging
            # import traceback
            # traceback.print_exc()


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
            base_filename = raw_filename.rsplit("_", 1)[
                0
            ]  # Split at the last underscore
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

    # create_graphs_for_dataset(raw_images_path, labeled_images_path, output_graphs_path)

    downsampled_raw_path = "data/aachen_raw_downsampled"
    downsampled_labeled_path = "data/aachen_labeled_downsampled"
    pool_size = 8

    save_max_pooled_images(
        raw_images_path,
        labeled_images_path,
        downsampled_raw_path,
        downsampled_labeled_path,
        pool_size,
    )
