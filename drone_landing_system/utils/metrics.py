"""
utils/metrics.py
Evaluation metrics: IoU, pixel accuracy, risk scoring accuracy.
"""

import numpy as np


def pixel_accuracy(y_true, y_pred):
    """Global pixel accuracy."""
    correct = np.sum(y_true == y_pred)
    total   = y_true.size
    return correct / total


def mean_iou(y_true, y_pred, n_classes):
    """Mean Intersection over Union across all classes."""
    iou_list = []
    for cls in range(n_classes):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        denom = tp + fp + fn
        if denom == 0:
            continue
        iou_list.append(tp / denom)
    return np.mean(iou_list) if iou_list else 0.0


def class_iou(y_true, y_pred, n_classes):
    """Per-class IoU dictionary."""
    result = {}
    for cls in range(n_classes):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        denom = tp + fp + fn
        result[cls] = tp / denom if denom > 0 else None
    return result


def dice_score(y_true, y_pred, n_classes):
    """Mean Dice coefficient."""
    dice_list = []
    for cls in range(n_classes):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        pred_pos  = np.sum(y_pred == cls)
        true_pos  = np.sum(y_true == cls)
        denom = pred_pos + true_pos
        if denom == 0:
            continue
        dice_list.append(2 * tp / denom)
    return np.mean(dice_list) if dice_list else 0.0


def landing_zone_accuracy(predicted_zones, ground_truth_zones, iou_threshold=0.5):
    """
    Compute landing zone detection accuracy.
    Matches predicted zones to GT zones using centroid distance.

    Args:
        predicted_zones: list of {'centroid': (x,y), 'final_score': float}
        ground_truth_zones: list of {'centroid': (x,y)}
        iou_threshold: matching distance threshold (pixels)

    Returns:
        precision, recall, f1
    """
    if not predicted_zones or not ground_truth_zones:
        return 0.0, 0.0, 0.0

    matched_pred = set()
    matched_gt   = set()

    for i, pred in enumerate(predicted_zones):
        px, py = pred["centroid"]
        for j, gt in enumerate(ground_truth_zones):
            if j in matched_gt:
                continue
            gx, gy = gt["centroid"]
            dist = np.sqrt((px - gx)**2 + (py - gy)**2)
            if dist <= iou_threshold:
                matched_pred.add(i)
                matched_gt.add(j)
                break

    tp = len(matched_pred)
    fp = len(predicted_zones) - tp
    fn = len(ground_truth_zones) - len(matched_gt)

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1


def print_metrics(metrics_dict):
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 50)
    print("  EVALUATION METRICS")
    print("=" * 50)
    for key, val in metrics_dict.items():
        if isinstance(val, float):
            print(f"  {key:<30} {val:.4f}")
        else:
            print(f"  {key:<30} {val}")
    print("=" * 50 + "\n")
