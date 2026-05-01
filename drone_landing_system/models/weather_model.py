"""
models/weather_model.py
Module 2 — Weather Risk Scoring
MLP and LSTM-based weather safety score predictor.
Input: 7 weather features → Output: weather safety score [0, 1]
"""

import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

WX_CFG = cfg["weather"]
N_FEATURES = len(WX_CFG["features"])  # 7


# ── MLP Weather Model ─────────────────────────────────────────────────────────
def build_weather_mlp(n_features=N_FEATURES, hidden_layers=None, dropout=0.3):
    """
    Simple feedforward MLP.
    Input:  weather feature vector (7,)
    Output: safety score scalar [0, 1]
    """
    hidden_layers = hidden_layers or WX_CFG["hidden_layers"]

    inputs = keras.Input(shape=(n_features,), name="weather_input")
    x = inputs

    for i, units in enumerate(hidden_layers):
        x = layers.Dense(units, name=f"dense_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        x = layers.Dropout(dropout, name=f"dropout_{i}")(x)

    output = layers.Dense(1, activation="sigmoid", name="weather_safety_score")(x)

    model = Model(inputs, output, name="weather_mlp")
    return model


# ── LSTM Weather Model (for sequential/time-series weather) ───────────────────
def build_weather_lstm(n_features=N_FEATURES, seq_len=10, hidden_units=64, dropout=0.3):
    """
    LSTM model for sequential weather readings.
    Input:  (seq_len, n_features) time series
    Output: safety score scalar [0, 1]
    """
    inputs = keras.Input(shape=(seq_len, n_features), name="weather_sequence")

    x = layers.LSTM(hidden_units, return_sequences=True, name="lstm_1")(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(hidden_units // 2, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu", name="dense_1")(x)
    output = layers.Dense(1, activation="sigmoid", name="weather_safety_score")(x)

    model = Model(inputs, output, name="weather_lstm")
    return model


# ── Factory ───────────────────────────────────────────────────────────────────
def get_weather_model(model_type=None):
    model_type = model_type or WX_CFG["model"]
    if model_type == "mlp":
        model = build_weather_mlp()
    elif model_type == "lstm":
        model = build_weather_lstm()
    else:
        raise ValueError(f"Unknown weather model: {model_type}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=WX_CFG["learning_rate"]),
        loss="mse",
        metrics=["mae"]
    )
    print(f"[INFO] Weather model ({model_type}) | Params: {model.count_params():,}")
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train_weather_model(csv_path="data/processed/weather_dataset.csv"):
    df = pd.read_csv(csv_path)

    feature_cols = WX_CFG["features"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["weather_safety"].values.astype(np.float32)

    # Simple train/val split
    n_val = int(len(X) * 0.2)
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]

    model = get_weather_model("mlp")

    save_path = WX_CFG["checkpoint_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    callbacks = [
        ModelCheckpoint(save_path, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=WX_CFG["epochs"],
        batch_size=WX_CFG["batch_size"],
        callbacks=callbacks,
        verbose=1
    )

    print(f"[OK] Weather model saved → {save_path}")
    return model, history


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_weather_safety(model, weather_features_normalized):
    """
    Args:
        weather_features_normalized: numpy array (7,) or (1, 7)
    Returns:
        safety_score: float in [0, 1]
    """
    if weather_features_normalized.ndim == 1:
        weather_features_normalized = weather_features_normalized[np.newaxis, :]
    score = model.predict(weather_features_normalized, verbose=0)[0, 0]
    return float(score)


# ── Rule-based fallback (no trained model needed) ─────────────────────────────
def weather_safety_rule_based(weather_dict):
    """
    Deterministic weather safety score using domain rules.
    Use this when no trained model is available.

    Returns safety score in [0, 1]: higher = safer
    """
    vis    = weather_dict.get("visibility",    10.0)   # km
    wind   = weather_dict.get("wind_speed",     0.0)   # m/s
    precip = weather_dict.get("precipitation",  0.0)   # mm/hr
    fog    = weather_dict.get("fog_index",      0.0)   # 0–1
    smoke  = weather_dict.get("smoke_density",  0.0)   # 0–1

    # Risk contributions (0 = safe, 1 = dangerous)
    vis_risk    = max(0, 1 - vis / 10.0)          # low vis → high risk
    wind_risk   = min(1, wind / 15.0)             # >15 m/s → max risk
    precip_risk = min(1, precip / 20.0)           # >20 mm/hr → max risk
    fog_risk    = fog
    smoke_risk  = smoke

    total_risk = (
        vis_risk    * 0.30 +
        wind_risk   * 0.25 +
        precip_risk * 0.20 +
        fog_risk    * 0.15 +
        smoke_risk  * 0.10
    )

    safety_score = 1.0 - np.clip(total_risk, 0, 1)
    return float(safety_score)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = get_weather_model("mlp")
    model.summary()

    # Test inference
    dummy_input = np.random.rand(1, N_FEATURES).astype(np.float32)
    score = model.predict(dummy_input, verbose=0)
    print(f"[TEST] Weather safety score: {score[0, 0]:.4f}")

    # Test rule-based
    sample_weather = {
        "visibility": 3.0, "wind_speed": 12.0, "precipitation": 5.0,
        "humidity": 85.0, "temperature": 18.0, "fog_index": 0.4, "smoke_density": 0.2
    }
    rule_score = weather_safety_rule_based(sample_weather)
    print(f"[TEST] Rule-based safety score: {rule_score:.4f}")
