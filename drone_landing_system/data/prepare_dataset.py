"""
data/prepare_dataset.py
Prepares the Aerial Semantic Segmentation Drone Dataset for training.
Downloads, validates, splits, and preprocesses images + masks.
"""

import os
import sys
import shutil
import random
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm

# ── Load config ─────────────────────────────────────────────────────────────
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

RAW_DIR    = Path(cfg["paths"]["raw_data"])
PROC_DIR   = Path(cfg["paths"]["processed_data"])
CLASS_CSV  = Path(cfg["paths"]["class_dict"])
N_CLASSES  = cfg["dataset"]["n_classes"]
IMG_H      = cfg["dataset"]["image_height"]
IMG_W      = cfg["dataset"]["image_width"]
VAL_SPLIT  = cfg["dataset"]["val_split"]
TEST_SPLIT = cfg["dataset"]["test_split"]
SEED       = cfg["dataset"]["seed"]
MAX_SAMPLES = cfg["dataset"].get("max_samples", None)  # None = use all


# ── Helper: RGB mask → class index mask ─────────────────────────────────────
def build_color_map(class_csv_path):
    """Build mapping from RGB tuple → class index."""
    # skipinitialspace=True handles column names like " r" instead of "r"
    df = pd.read_csv(class_csv_path, skipinitialspace=True)
    
    # Strip any remaining whitespace from column names just to be safe
    df.columns = df.columns.str.strip()
    
    color_map = {}
    for idx, row in df.iterrows():
        rgb = (int(row["r"]), int(row["g"]), int(row["b"]))
        color_map[rgb] = idx
    return color_map


def rgb_mask_to_class_mask(rgb_mask_np, color_map):
    """Convert RGB segmentation mask → single-channel class-index mask."""
    h, w, _ = rgb_mask_np.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for rgb, cls_idx in color_map.items():
        match = np.all(rgb_mask_np == rgb, axis=-1)
        class_mask[match] = cls_idx
    return class_mask


# ── Validate raw dataset ─────────────────────────────────────────────────────
def validate_dataset():
    orig_dir   = RAW_DIR / "original_images"
    label_dir  = RAW_DIR / "label_images_semantic"

    if not orig_dir.exists():
        print(f"[ERROR] original_images not found at: {orig_dir}")
        print("  → Download from: https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset")
        sys.exit(1)

    images = sorted(orig_dir.glob("*.jpg"))
    masks  = sorted(label_dir.glob("*.png"))

    print(f"[INFO] Found {len(images)} images and {len(masks)} masks.")
    assert len(images) > 0, "No images found!"
    assert len(images) == len(masks), "Mismatch between images and masks!"

    # Limit to first N samples if configured
    if MAX_SAMPLES and MAX_SAMPLES < len(images):
        images = images[:MAX_SAMPLES]
        masks  = masks[:MAX_SAMPLES]
        print(f"[INFO] Limited to first {MAX_SAMPLES} samples (max_samples={MAX_SAMPLES})")

    return images, masks


# ── Split into train / val / test ────────────────────────────────────────────
def split_dataset(images, masks):
    combined = list(zip(images, masks))
    random.seed(SEED)
    random.shuffle(combined)

    n     = len(combined)
    n_val  = int(n * VAL_SPLIT)
    n_test = int(n * TEST_SPLIT)
    n_train = n - n_val - n_test

    train = combined[:n_train]
    val   = combined[n_train:n_train + n_val]
    test  = combined[n_train + n_val:]

    print(f"[INFO] Split → Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


# ── Process and save a split ─────────────────────────────────────────────────
def process_split(split_data, split_name, color_map):
    img_out_dir  = PROC_DIR / split_name / "images"
    mask_out_dir = PROC_DIR / split_name / "masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    for img_path, mask_path in tqdm(split_data, desc=f"Processing {split_name}"):
        stem = img_path.stem

        # -- Image: resize + save
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (IMG_W, IMG_H))
        cv2.imwrite(str(img_out_dir / f"{stem}.jpg"), img)

        # -- Mask: RGB → class index → resize → save as PNG
        rgb_mask = np.array(Image.open(mask_path).convert("RGB"))
        class_mask = rgb_mask_to_class_mask(rgb_mask, color_map)
        class_mask_resized = cv2.resize(
            class_mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST
        )
        cv2.imwrite(str(mask_out_dir / f"{stem}.png"), class_mask_resized)

    print(f"[OK] {split_name} saved → {img_out_dir}")


# ── Generate synthetic weather data for training Module 2 ────────────────────
def generate_weather_dataset(n_samples=5000):
    """
    Generates synthetic weather data + landing risk label.
    In a real system, this is replaced with actual weather sensor logs.
    """
    np.random.seed(SEED)
    data = {
        "visibility":      np.random.uniform(0, 10, n_samples),
        "wind_speed":       np.random.uniform(0, 30, n_samples),
        "precipitation":    np.random.uniform(0, 50, n_samples),
        "humidity":         np.random.uniform(20, 100, n_samples),
        "temperature":      np.random.uniform(-10, 45, n_samples),
        "fog_index":        np.random.uniform(0, 1, n_samples),
        "smoke_density":    np.random.uniform(0, 1, n_samples),
    }
    df = pd.DataFrame(data)

    # Weather risk score (rule-based ground truth for synthetic data)
    df["risk_score"] = (
        (1 - df["visibility"] / 10) * 0.30 +
        (df["wind_speed"]    / 30)  * 0.25 +
        (df["precipitation"] / 50)  * 0.20 +
        (df["fog_index"])           * 0.15 +
        (df["smoke_density"])       * 0.10
    ).clip(0, 1)

    # Safety score = 1 - risk
    df["weather_safety"] = 1 - df["risk_score"]

    out_path = PROC_DIR / "weather_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Weather dataset saved → {out_path}  ({n_samples} samples)")
    return df


# ── Generate synthetic threat data for training Module 3 ─────────────────────
def generate_threat_dataset(n_samples=2000):
    """
    Generates synthetic threat map data + threat risk label.
    In a real system, military intelligence data populates this.
    """
    np.random.seed(SEED + 1)
    data = {
        "hostile_area":        np.random.uniform(0, 1, n_samples),
        "prohibited_zone":     np.random.uniform(0, 1, n_samples),
        "mine_region":         np.random.uniform(0, 1, n_samples),
        "gunfire_probability": np.random.uniform(0, 1, n_samples),
        "blast_radius":        np.random.uniform(0, 1, n_samples),
    }
    df = pd.DataFrame(data)

    # Threat score (weighted combination)
    df["threat_score"] = (
        df["hostile_area"]        * 0.35 +
        df["prohibited_zone"]     * 0.20 +
        df["mine_region"]         * 0.20 +
        df["gunfire_probability"] * 0.15 +
        df["blast_radius"]        * 0.10
    ).clip(0, 1)

    df["threat_safety"] = 1 - df["threat_score"]

    out_path = PROC_DIR / "threat_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Threat dataset saved → {out_path}  ({n_samples} samples)")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Drone Landing System — Dataset Preparation")
    print("=" * 60)

    # 1. Validate
    images, masks = validate_dataset()

    # 2. Build color map
    color_map = build_color_map(CLASS_CSV)
    print(f"[INFO] Color map loaded: {len(color_map)} classes")

    # 3. Split
    train, val, test = split_dataset(images, masks)

    # 4. Process each split
    process_split(train, "train", color_map)
    process_split(val,   "val",   color_map)
    process_split(test,  "test",  color_map)

    # 5. Generate synthetic auxiliary datasets
    generate_weather_dataset(n_samples=5000)
    generate_threat_dataset(n_samples=2000)

    print("\n[✓] Dataset preparation complete!")
    print(f"    Processed data at: {PROC_DIR}")
