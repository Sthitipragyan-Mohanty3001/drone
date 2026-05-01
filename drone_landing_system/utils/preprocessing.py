"""
utils/preprocessing.py
Image preprocessing, augmentation, and data generators.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tensorflow as tf

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

IMG_H      = cfg["dataset"]["image_height"]
IMG_W      = cfg["dataset"]["image_width"]
N_CLASSES  = cfg["dataset"]["n_classes"]


# ── Augmentation pipelines ────────────────────────────────────────────────────
def get_train_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        # Fog / weather simulation
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.15),
        A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, p=0.1),
        A.Resize(IMG_H, IMG_W),
    ])


def get_val_augmentation():
    return A.Compose([
        A.Resize(IMG_H, IMG_W),
    ])


# ── TensorFlow data generator ─────────────────────────────────────────────────
class DroneDataGenerator(tf.keras.utils.Sequence):
    """Keras Sequence generator for aerial imagery + segmentation masks."""

    def __init__(self, image_dir, mask_dir, batch_size=4, augment=True, shuffle=True):
        self.image_dir  = Path(image_dir)
        self.mask_dir   = Path(mask_dir)
        self.batch_size = batch_size
        self.augment    = augment
        self.shuffle    = shuffle
        self.aug_fn     = get_train_augmentation() if augment else get_val_augmentation()

        self.image_paths = sorted(self.image_dir.glob("*.jpg"))
        self.mask_paths  = sorted(self.mask_dir.glob("*.png"))
        assert len(self.image_paths) == len(self.mask_paths), "Image/mask count mismatch"

        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, masks = [], []

        for i in batch_indices:
            img  = cv2.imread(str(self.image_paths[i]))
            img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(self.mask_paths[i]), cv2.IMREAD_GRAYSCALE)

            # Apply augmentation
            augmented = self.aug_fn(image=img, mask=mask)
            img  = augmented["image"]
            mask = augmented["mask"]

            # Normalize image
            img = img.astype(np.float32) / 255.0

            # One-hot encode mask
            mask_one_hot = tf.keras.utils.to_categorical(mask, num_classes=N_CLASSES)

            images.append(img)
            masks.append(mask_one_hot)

        return np.array(images), np.array(masks)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ── Preprocess single image for inference ─────────────────────────────────────
def preprocess_image(image_path_or_array, target_size=(IMG_H, IMG_W)):
    """Load and preprocess a single image for model inference."""
    if isinstance(image_path_or_array, (str, Path)):
        img = cv2.imread(str(image_path_or_array))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image_path_or_array.copy()

    original_shape = img.shape[:2]  # (H, W)
    img_resized = cv2.resize(img, (target_size[1], target_size[0]))
    img_norm    = img_resized.astype(np.float32) / 255.0
    img_batch   = np.expand_dims(img_norm, axis=0)  # (1, H, W, 3)

    return img_batch, original_shape, img_resized


# ── Class mask → color visualization ─────────────────────────────────────────
CLASS_COLORS = [
    (0,   0,   0),    # unlabeled
    (128, 64,  128),  # paved-area
    (130, 76,  0),    # dirt
    (0,   102, 0),    # grass
    (112, 103, 87),   # gravel
    (28,  42,  168),  # water
    (48,  41,  30),   # rocks
    (0,   50,  89),   # pool
    (107, 142, 35),   # vegetation
    (70,  70,  70),   # roof
    (102, 102, 156),  # wall
    (254, 228, 12),   # window
    (254, 148, 12),   # door
    (190, 153, 153),  # fence
    (153, 153, 153),  # fence-pole
    (255, 22,  96),   # person
    (102, 51,  0),    # dog
    (9,   143, 150),  # car
    (119, 11,  32),   # bicycle
    (51,  51,  0),    # roof2
    (190, 250, 190),  # obstacle
    (112, 150, 146),  # reserved
    (2,   135, 115),  # reserved
    (255, 0,   0),    # reserved
]


def class_mask_to_color(class_mask):
    """Convert H×W class mask → H×W×3 RGB visualization."""
    h, w = class_mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(CLASS_COLORS):
        color_img[class_mask == cls_idx] = color
    return color_img


# ── Weather input normalization ───────────────────────────────────────────────
WEATHER_RANGES = {
    "visibility":      (0.0, 10.0),
    "wind_speed":      (0.0, 30.0),
    "precipitation":   (0.0, 50.0),
    "humidity":        (0.0, 100.0),
    "temperature":     (-10.0, 45.0),
    "fog_index":       (0.0, 1.0),
    "smoke_density":   (0.0, 1.0),
}


def normalize_weather(weather_dict):
    """Normalize raw weather values to [0, 1] range."""
    features = list(WEATHER_RANGES.keys())
    normalized = []
    for feat in features:
        val    = weather_dict.get(feat, 0.0)
        lo, hi = WEATHER_RANGES[feat]
        norm   = np.clip((val - lo) / (hi - lo + 1e-8), 0, 1)
        normalized.append(norm)
    return np.array(normalized, dtype=np.float32)


# ── Threat input normalization ────────────────────────────────────────────────
THREAT_KEYS = ["hostile_area", "prohibited_zone", "mine_region",
               "gunfire_probability", "blast_radius"]


def normalize_threat(threat_dict):
    """Extract and normalize threat values to [0, 1] range."""
    return np.array(
        [float(threat_dict.get(k, 0.0)) for k in THREAT_KEYS],
        dtype=np.float32
    )
