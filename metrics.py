from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def calculate_iou(
    prediction: np.ndarray, ground_truth: np.ndarray, num_classes: int
) -> Dict[int, float]:
    """
    Calculate Intersection over Union (IoU) for each class.

    Returns:
        Dict[int, float]: Dictionary of IoU scores for each class
    """
    # Validate input shapes
    if prediction.shape != ground_truth.shape:
        raise ValueError("Prediction and ground truth must have the same shape")

    # Initialize IoU dictionary
    iou_scores = {}

    # Compute IoU for each class
    for class_id in range(num_classes):
        # Create binary masks for the current class
        pred_class_mask = prediction == class_id
        gt_class_mask = ground_truth == class_id

        # Calculate intersection and union
        intersection = np.logical_and(pred_class_mask, gt_class_mask)
        union = np.logical_or(pred_class_mask, gt_class_mask)

        # Compute IoU, handling division by zero
        intersection_area = np.sum(intersection)
        union_area = np.sum(union)

        iou = intersection_area / union_area if union_area > 0 else 0.0
        iou_scores[class_id] = iou

    return iou_scores


def calculate_average_precision(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    num_classes: int,
    overlap_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
) -> Dict[int, float]:
    """
    Calculate Average Precision (AP) on the region level for each class.

    Returns:
        Dict[int, float]: Dictionary of Average Precision scores for each class
    """
    # Validate input shapes
    if prediction.shape != ground_truth.shape:
        raise ValueError("Prediction and ground truth must have the same shape")

    # Initialize average precision dictionary
    average_precisions = {}

    # Compute precision for each class
    for class_id in tqdm(range(num_classes)):
        # Create binary masks for the current class
        pred_class_mask = prediction == class_id
        gt_class_mask = ground_truth == class_id

        # Find connected regions (instances) for prediction and ground truth
        pred_regions = _find_connected_regions(pred_class_mask)
        gt_regions = _find_connected_regions(gt_class_mask)

        # Compute precision at different IoU thresholds
        class_precisions = []
        for threshold in overlap_thresholds:
            precision = _compute_precision_at_threshold(
                pred_regions, gt_regions, threshold
            )
            class_precisions.append(precision)

        # Average precision across thresholds
        average_precisions[class_id] = np.mean(class_precisions)

    return average_precisions


def _find_connected_regions(binary_mask: np.ndarray) -> List[np.ndarray]:
    """
    Find connected regions in a binary mask using scipy's label function.

    Returns:
        List[np.ndarray]: List of binary masks for each connected region
    """
    from scipy import ndimage

    # Label connected regions
    labeled_array, num_features = ndimage.label(binary_mask)

    # Extract individual regions
    regions = []
    for i in range(1, num_features + 1):
        region_mask = labeled_array == i
        regions.append(region_mask)

    return regions


def _compute_precision_at_threshold(
    pred_regions: List[np.ndarray], gt_regions: List[np.ndarray], threshold: float
) -> float:
    """
    Compute precision at a specific IoU threshold.

    Returns:
        float: Precision at the given threshold
    """
    # If no predicted regions, precision is 0
    if not pred_regions:
        return 0.0

    # If no ground truth regions, precision is 0
    if not gt_regions:
        return 0.0

    # Track matched ground truth regions
    matched_gt_regions = set()
    true_positives = 0

    # Check each predicted region against ground truth regions
    for pred_region in pred_regions:
        for i, gt_region in enumerate(gt_regions):
            # Compute IoU between predicted and ground truth region
            intersection = np.logical_and(pred_region, gt_region)
            union = np.logical_or(pred_region, gt_region)

            iou = np.sum(intersection) / np.sum(union)

            # Check if IoU is above threshold and ground truth region is not already matched
            if iou >= threshold and i not in matched_gt_regions:
                true_positives += 1
                matched_gt_regions.add(i)
                break

    # Compute precision
    precision = true_positives / len(pred_regions)
    return precision


# Example usage
def main():
    # Paths of the images
    prediction_path = (
        "data/aachen_labeled_downsampled/aachen_000000_000019_gtFine_labelIds.png"
    )
    ground_truth_path = (
        "data/aachen_labeled_downsampled/aachen_000001_000019_gtFine_labelIds.png"
    )

    Image.open(prediction_path)
    Image.open(ground_truth_path)

    # Load images
    prediction = np.array(Image.open(prediction_path))
    ground_truth = np.array(Image.open(ground_truth_path))

    # Set parameters
    num_classes = 34
    height, width = prediction.shape

    # Calculate IoU
    iou_scores = calculate_iou(prediction, ground_truth, num_classes)
    print("IoU Scores:", iou_scores)

    # Calculate Average Precision
    average_precisions = calculate_average_precision(
        prediction, ground_truth, num_classes
    )
    print("Average Precision Scores:", average_precisions)


if __name__ == "__main__":
    main()
