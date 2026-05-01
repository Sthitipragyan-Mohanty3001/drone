"""
modules/terrain_analyzer.py
Converts segmentation output into a terrain safety score map.
"""

import yaml
import numpy as np
import cv2
from scipy import ndimage

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

CLASS_SAFETY = {int(k): float(v) for k, v in cfg["class_safety"].items()}
DE_CFG = cfg["decision_engine"]


def build_terrain_safety_map(class_mask, prob_map=None):
    """
    Convert class segmentation mask → per-pixel terrain safety map.

    Args:
        class_mask: np.ndarray (H, W) with class indices
        prob_map:   np.ndarray (H, W, C) with class probabilities (optional)

    Returns:
        safety_map: np.ndarray (H, W) float32 in [0, 1]
    """
    h, w = class_mask.shape
    safety_map = np.zeros((h, w), dtype=np.float32)

    for cls_idx, safety_val in CLASS_SAFETY.items():
        mask = class_mask == cls_idx
        safety_map[mask] = safety_val

    # If probabilities available, weight by confidence
    if prob_map is not None:
        confidence = np.max(prob_map, axis=-1)         # (H, W)
        # Blend: high-confidence predictions stay as-is, low confidence → penalized
        safety_map = safety_map * (0.5 + 0.5 * confidence)

    return safety_map.astype(np.float32)


def find_landing_zone_candidates(safety_map, min_area=500, smooth_sigma=10):
    """
    Identify candidate landing zones from terrain safety map.

    Args:
        safety_map: np.ndarray (H, W) float32
        min_area:   minimum pixel area for a valid landing zone
        smooth_sigma: Gaussian smoothing sigma before zone detection

    Returns:
        list of dicts: [{'centroid': (cx, cy), 'score': float, 'area': int, 'bbox': (x,y,w,h)}]
    """
    # Smooth safety map (creates blobs of safe regions)
    smooth = ndimage.gaussian_filter(safety_map, sigma=smooth_sigma)

    # Threshold at 60% safety
    threshold = DE_CFG.get("min_safe_score", 0.60)
    binary    = (smooth >= threshold).astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    candidates = []
    for lbl in range(1, num_labels):  # skip background (0)
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        cx, cy = centroids[lbl]
        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]

        # Mean safety score in this region
        region_scores = safety_map[labels == lbl]
        mean_score    = float(np.mean(region_scores))

        candidates.append({
            "centroid": (int(cx), int(cy)),
            "score":    mean_score,
            "area":     int(area),
            "bbox":     (int(x), int(y), int(bw), int(bh)),
            "label":    int(lbl),
        })

    # Sort by score descending
    candidates.sort(key=lambda z: z["score"], reverse=True)
    return candidates


def non_maximum_suppression(candidates, nms_radius=50):
    """
    Suppress overlapping candidates (keep best score within radius).
    """
    if not candidates:
        return []

    kept = []
    suppressed = set()

    for i, c1 in enumerate(candidates):
        if i in suppressed:
            continue
        kept.append(c1)
        x1, y1 = c1["centroid"]
        for j, c2 in enumerate(candidates[i+1:], start=i+1):
            if j in suppressed:
                continue
            x2, y2 = c2["centroid"]
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < nms_radius:
                suppressed.add(j)

    return kept


def compute_terrain_score(safety_map, centroid, radius=30):
    """
    Compute terrain safety score for a given centroid location.
    Average safety within a circle of given radius.
    """
    h, w = safety_map.shape
    cx, cy = centroid
    y_grid, x_grid = np.ogrid[:h, :w]
    mask = (x_grid - cx)**2 + (y_grid - cy)**2 <= radius**2
    return float(np.mean(safety_map[mask]))
