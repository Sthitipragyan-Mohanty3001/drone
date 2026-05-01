"""
train.py
Master training script for all three modules.

Usage:
    python train.py --module all
    python train.py --module segmentation
    python train.py --module weather
    python train.py --module threat
"""

import argparse
import os
import yaml
import numpy as np
from pathlib import Path

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)


def train_segmentation():
    print("\n" + "=" * 60)
    print("  MODULE 1: Training Segmentation Model")
    print("=" * 60)

    from models.segmentation_model import get_segmentation_model, compile_model, get_callbacks
    from utils.preprocessing import DroneDataGenerator

    seg_cfg = cfg["segmentation"]
    paths   = cfg["paths"]

    # Check processed data exists
    train_img_dir = paths["train_images"]
    train_msk_dir = paths["train_masks"]
    val_img_dir   = paths["val_images"]
    val_msk_dir   = paths["val_masks"]

    if not Path(train_img_dir).exists():
        print("[ERROR] Processed data not found. Run: python data/prepare_dataset.py first.")
        return

    # Build data generators
    train_gen = DroneDataGenerator(
        train_img_dir, train_msk_dir,
        batch_size=seg_cfg["batch_size"],
        augment=True
    )
    val_gen = DroneDataGenerator(
        val_img_dir, val_msk_dir,
        batch_size=seg_cfg["batch_size"],
        augment=False
    )

    print(f"[INFO] Train batches: {len(train_gen)} | Val batches: {len(val_gen)}")

    # Build and compile model
    model = get_segmentation_model(seg_cfg["model"], pretrained=seg_cfg["pretrained"])
    model = compile_model(model)

    # Train
    callbacks = get_callbacks(seg_cfg["checkpoint_path"])
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=seg_cfg["epochs"],
        callbacks=callbacks,
        verbose=1
    )

    # Save history
    import json
    os.makedirs("logs", exist_ok=True)
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open("logs/segmentation_history.json", "w") as f:
        json.dump(hist_dict, f, indent=2)

    print("\n[✓] Segmentation training complete!")
    print(f"    Best model: {seg_cfg['checkpoint_path']}/best_model.h5")
    return history


def train_weather():
    print("\n" + "=" * 60)
    print("  MODULE 2: Training Weather Model")
    print("=" * 60)

    from models.weather_model import train_weather_model

    csv_path = "data/processed/weather_dataset.csv"
    if not Path(csv_path).exists():
        print("[ERROR] Weather dataset not found. Run: python data/prepare_dataset.py first.")
        return

    model, history = train_weather_model(csv_path)
    print("[✓] Weather training complete!")
    return history


def train_threat():
    print("\n" + "=" * 60)
    print("  MODULE 3: Training Threat Model")
    print("=" * 60)

    from models.threat_model import train_threat_model

    csv_path = "data/processed/threat_dataset.csv"
    if not Path(csv_path).exists():
        print("[ERROR] Threat dataset not found. Run: python data/prepare_dataset.py first.")
        return

    model, history = train_threat_model(csv_path)
    print("[✓] Threat training complete!")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train drone landing system models.")
    parser.add_argument(
        "--module",
        choices=["all", "segmentation", "weather", "threat"],
        default="all",
        help="Which module to train"
    )
    args = parser.parse_args()

    print("\n" + "█" * 60)
    print("  DRONE EMERGENCY LANDING SYSTEM — Training")
    print("█" * 60)

    if args.module in ("all", "segmentation"):
        train_segmentation()

    if args.module in ("all", "weather"):
        train_weather()

    if args.module in ("all", "threat"):
        train_threat()

    print("\n[✓] All training complete! Run predict.py to test the system.")


if __name__ == "__main__":
    main()
