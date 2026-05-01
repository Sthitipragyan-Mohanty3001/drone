"""
models/threat_model.py
Module 3 — Threat Zone Risk Scoring
Geospatial CNN + rule-based threat analysis for war zones.
Input: threat feature vector + optional spatial threat map
Output: threat safety score [0, 1]
"""

import os
import yaml
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

TH_CFG    = cfg["threat"]
THREAT_KEYS = ["hostile_area", "prohibited_zone", "mine_region",
               "gunfire_probability", "blast_radius"]


# ── Threat MLP (feature-based) ────────────────────────────────────────────────
def build_threat_mlp(n_features=5):
    """
    Input:  5 threat scalar features → Output: threat safety score [0, 1]
    """
    inputs = keras.Input(shape=(n_features,), name="threat_features")
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation="sigmoid", name="threat_safety")(x)

    model = Model(inputs, output, name="threat_mlp")
    return model


# ── Geospatial CNN (spatial threat map → zone risk) ───────────────────────────
def build_geospatial_cnn(map_height=64, map_width=64, n_threat_channels=5):
    """
    Input:  threat map image (H × W × n_channels)
            Each channel = one threat type's spatial heatmap
    Output: per-pixel safety score map (H, W, 1)
    """
    inputs = keras.Input(shape=(map_height, map_width, n_threat_channels), name="threat_map")

    # Encoder
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)

    # Decoder
    x = layers.UpSampling2D(2, interpolation="bilinear")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D(2, interpolation="bilinear")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)

    # Safety score per pixel
    output = layers.Conv2D(1, 1, activation="sigmoid", name="threat_safety_map")(x)

    model = Model(inputs, output, name="geospatial_cnn")
    return model


# ── Factory ───────────────────────────────────────────────────────────────────
def get_threat_model(model_type=None):
    model_type = model_type or TH_CFG["model"]
    if model_type == "threat_mlp":
        model = build_threat_mlp()
    elif model_type == "geospatial_cnn":
        model = build_geospatial_cnn()
    else:
        # Default to MLP for simplicity
        model = build_threat_mlp()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=TH_CFG["learning_rate"]),
        loss="mse",
        metrics=["mae"]
    )
    print(f"[INFO] Threat model ({model_type}) | Params: {model.count_params():,}")
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train_threat_model(csv_path="data/processed/threat_dataset.csv"):
    df = pd.read_csv(csv_path)

    X = df[THREAT_KEYS].values.astype(np.float32)
    y = df["threat_safety"].values.astype(np.float32)

    n_val = int(len(X) * 0.2)
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]

    model = build_threat_mlp()
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="mse", metrics=["mae"]
    )

    save_path = TH_CFG["checkpoint_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    callbacks = [
        ModelCheckpoint(save_path, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=TH_CFG["epochs"],
        batch_size=TH_CFG["batch_size"],
        callbacks=callbacks,
        verbose=1
    )

    print(f"[OK] Threat model saved → {save_path}")
    return model, history


# ── Threat map → spatial safety grid ─────────────────────────────────────────
def threat_map_to_safety_grid(threat_json, grid_h=416, grid_w=608):
    """
    Convert threat zone JSON (with bounding box zones) into
    a per-pixel safety grid matching the aerial image dimensions.

    Args:
        threat_json: dict with 'zones' list, each zone having:
                     x_min, y_min, x_max, y_max (in [0,1] normalized coords)
                     threat_level (0–1)
        grid_h, grid_w: dimensions of the output grid

    Returns:
        safety_grid: np.ndarray (H, W) with safety scores [0, 1]
    """
    # Start fully safe
    safety_grid = np.ones((grid_h, grid_w), dtype=np.float32)

    zones = threat_json.get("zones", [])
    for zone in zones:
        # Convert normalized [0,1] coords → pixel coords
        x0 = int(zone["x_min"] * grid_w)
        y0 = int(zone["y_min"] * grid_h)
        x1 = int(zone["x_max"] * grid_w)
        y1 = int(zone["y_max"] * grid_h)
        threat_level = float(zone.get("threat_level", 0.5))

        # Set safety = 1 - threat_level in the zone rectangle
        safety_grid[y0:y1, x0:x1] = np.minimum(
            safety_grid[y0:y1, x0:x1],
            1.0 - threat_level
        )

    return safety_grid


# ── Rule-based scalar threat safety ──────────────────────────────────────────
def threat_safety_rule_based(threat_dict):
    """
    Compute a scalar threat safety score from threat feature dict.
    Returns safety score in [0, 1]: higher = safer
    """
    weights = {
        "hostile_area":        0.35,
        "prohibited_zone":     0.20,
        "mine_region":         0.20,
        "gunfire_probability": 0.15,
        "blast_radius":        0.10,
    }
    threat_score = sum(
        threat_dict.get(k, 0.0) * w
        for k, w in weights.items()
    )
    return float(1.0 - np.clip(threat_score, 0, 1))


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_threat_safety(model, threat_features):
    """
    Args:
        threat_features: np.array (5,) normalized threat values
    Returns:
        safety_score: float [0, 1]
    """
    if threat_features.ndim == 1:
        threat_features = threat_features[np.newaxis, :]
    score = model.predict(threat_features, verbose=0)[0, 0]
    return float(score)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_threat_mlp()
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    dummy = np.random.rand(1, 5).astype(np.float32)
    score = model.predict(dummy, verbose=0)
    print(f"[TEST] Threat safety score: {score[0, 0]:.4f}")

    sample_threat = {
        "hostile_area": 0.7, "prohibited_zone": 0.4,
        "mine_region": 0.2, "gunfire_probability": 0.5, "blast_radius": 0.3
    }
    rule_score = threat_safety_rule_based(sample_threat)
    print(f"[TEST] Rule-based threat safety: {rule_score:.4f}")
