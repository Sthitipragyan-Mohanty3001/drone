"""
evaluate.py
Evaluate all models on the test set.

Usage:
    python evaluate.py
    python evaluate.py --module segmentation
"""

import argparse
import yaml
import numpy as np
import json
from pathlib import Path
import cv2
from tqdm import tqdm
import tensorflow as tf

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)


def evaluate_segmentation():
    print("\n" + "=" * 60)
    print("  Evaluating Segmentation Model")
    print("=" * 60)

    from utils.metrics import pixel_accuracy, mean_iou, dice_score
    from utils.preprocessing import preprocess_image

    seg_path = Path(cfg["segmentation"]["checkpoint_path"]) / "best_model.h5"
    if not seg_path.exists():
        print(f"[ERROR] Model not found: {seg_path}")
        return

    model      = tf.keras.models.load_model(str(seg_path), compile=False)
    test_imgs  = sorted(Path(cfg["paths"]["test_images"]).glob("*.jpg"))
    test_masks = sorted(Path(cfg["paths"]["test_masks"]).glob("*.png"))
    n_classes  = cfg["dataset"]["n_classes"]

    if not test_imgs:
        print("[ERROR] No test images found. Run prepare_dataset.py first.")
        return

    all_pa, all_miou, all_dice = [], [], []

    for img_path, mask_path in tqdm(zip(test_imgs, test_masks), total=len(test_imgs)):
        # Load and preprocess
        img_batch, _, _ = preprocess_image(img_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Predict
        prob_map   = model.predict(img_batch, verbose=0)[0]
        pred_mask  = np.argmax(prob_map, axis=-1)

        # Resize mask to match prediction if needed
        h, w = pred_mask.shape
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Metrics
        all_pa.append(pixel_accuracy(mask, pred_mask))
        all_miou.append(mean_iou(mask, pred_mask, n_classes))
        all_dice.append(dice_score(mask, pred_mask, n_classes))

    metrics = {
        "Pixel Accuracy":  float(np.mean(all_pa)),
        "Mean IoU":        float(np.mean(all_miou)),
        "Dice Score":      float(np.mean(all_dice)),
        "N test images":   len(test_imgs),
    }

    from utils.metrics import print_metrics
    print_metrics(metrics)

    # Save
    Path("logs").mkdir(exist_ok=True)
    with open("logs/segmentation_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[OK] Eval saved → logs/segmentation_eval.json")
    return metrics


def evaluate_weather():
    print("\n" + "=" * 60)
    print("  Evaluating Weather Model")
    print("=" * 60)

    import pandas as pd
    from sklearn.metrics import mean_absolute_error, r2_score

    wx_path = Path(cfg["weather"]["checkpoint_path"])
    if not wx_path.exists():
        print(f"[ERROR] Model not found: {wx_path}")
        return

    model    = tf.keras.models.load_model(str(wx_path), compile=False)
    csv_path = "data/processed/weather_dataset.csv"
    df = pd.read_csv(csv_path)

    features = cfg["weather"]["features"]
    X = df[features].values.astype(np.float32)
    y = df["weather_safety"].values.astype(np.float32)

    # Use last 20% as test
    n_test = int(len(X) * 0.2)
    X_test = X[-n_test:]
    y_test = y[-n_test:]

    y_pred = model.predict(X_test, verbose=0).flatten()
    metrics = {
        "MAE":  float(mean_absolute_error(y_test, y_pred)),
        "R²":   float(r2_score(y_test, y_pred)),
        "N":    n_test,
    }

    from utils.metrics import print_metrics
    print_metrics(metrics)
    return metrics


def evaluate_threat():
    print("\n" + "=" * 60)
    print("  Evaluating Threat Model")
    print("=" * 60)

    import pandas as pd
    from sklearn.metrics import mean_absolute_error, r2_score

    th_path = Path(cfg["threat"]["checkpoint_path"])
    if not th_path.exists():
        print(f"[ERROR] Model not found: {th_path}")
        return

    model    = tf.keras.models.load_model(str(th_path), compile=False)
    csv_path = "data/processed/threat_dataset.csv"
    df = pd.read_csv(csv_path)

    keys = ["hostile_area", "prohibited_zone", "mine_region",
            "gunfire_probability", "blast_radius"]
    X = df[keys].values.astype(np.float32)
    y = df["threat_safety"].values.astype(np.float32)

    n_test = int(len(X) * 0.2)
    X_test = X[-n_test:]
    y_test = y[-n_test:]

    y_pred = model.predict(X_test, verbose=0).flatten()
    metrics = {
        "MAE":  float(mean_absolute_error(y_test, y_pred)),
        "R²":   float(r2_score(y_test, y_pred)),
        "N":    n_test,
    }

    from utils.metrics import print_metrics
    print_metrics(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate drone landing system models.")
    parser.add_argument("--module", choices=["all", "segmentation", "weather", "threat"],
                        default="all")
    args = parser.parse_args()

    if args.module in ("all", "segmentation"):
        evaluate_segmentation()
    if args.module in ("all", "weather"):
        evaluate_weather()
    if args.module in ("all", "threat"):
        evaluate_threat()


if __name__ == "__main__":
    main()
