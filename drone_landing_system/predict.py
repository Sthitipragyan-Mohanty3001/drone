"""
predict.py
Full end-to-end inference pipeline.

Usage:
    # Single image (with trained models)
    python predict.py --image path/to/image.jpg \
                      --weather_input configs/sample_weather.json \
                      --threat_map configs/sample_threat_map.json

    # Single image (rule-based, no models needed)
    python predict.py --image path/to/image.jpg \
                      --weather_input configs/sample_weather.json \
                      --threat_map configs/sample_threat_map.json \
                      --rule_based

    # Batch prediction on a folder
    python predict.py --image_dir data/processed/test/images/ \
                      --output_dir outputs/
"""

import argparse
import json
import os
import sys
import time
import yaml
import numpy as np
import cv2
from pathlib import Path

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)


def load_models(rule_based=False):
    """Load all three trained models (or return None for rule-based fallback)."""
    models = {"segmentation": None, "weather": None, "threat": None}

    if rule_based:
        print("[INFO] Running in rule-based mode (no ML models needed)")
        return models

    # -- Segmentation model
    seg_path = Path(cfg["segmentation"]["checkpoint_path"]) / "best_model.h5"
    if seg_path.exists():
        import tensorflow as tf
        models["segmentation"] = tf.keras.models.load_model(str(seg_path), compile=False)
        print(f"[OK] Loaded segmentation model from {seg_path}")
    else:
        print(f"[WARN] Segmentation model not found at {seg_path}")
        print("       Using rule-based terrain scoring (less accurate)")

    # -- Weather model
    wx_path = Path(cfg["weather"]["checkpoint_path"])
    if wx_path.exists():
        import tensorflow as tf
        models["weather"] = tf.keras.models.load_model(str(wx_path), compile=False)
        print(f"[OK] Loaded weather model from {wx_path}")
    else:
        print(f"[WARN] Weather model not found — using rule-based scoring")

    # -- Threat model
    th_path = Path(cfg["threat"]["checkpoint_path"])
    if th_path.exists():
        import tensorflow as tf
        models["threat"] = tf.keras.models.load_model(str(th_path), compile=False)
        print(f"[OK] Loaded threat model from {th_path}")
    else:
        print(f"[WARN] Threat model not found — using rule-based scoring")

    return models


def predict_single(image_path, weather_dict, threat_dict, models, output_dir="outputs"):
    """
    Run full prediction pipeline on a single image.

    Returns:
        results dict with ranked zones and scores
    """
    from utils.preprocessing import preprocess_image, normalize_weather, normalize_threat
    from models.segmentation_model import predict_segmentation
    from models.weather_model import predict_weather_safety, weather_safety_rule_based
    from models.threat_model import (predict_threat_safety, threat_safety_rule_based,
                                      threat_map_to_safety_grid)
    from modules.terrain_analyzer import (build_terrain_safety_map,
                                           find_landing_zone_candidates,
                                           non_maximum_suppression)
    from modules.decision_engine import rank_landing_zones, format_results, save_results
    from utils.visualization import (overlay_segmentation, draw_safety_heatmap,
                                      draw_landing_zones, plot_full_analysis)
    from utils.preprocessing import class_mask_to_color

    image_path = Path(image_path)
    print(f"\n[>>] Processing: {image_path.name}")
    t0 = time.time()

    # ── Step 1: Preprocess image ──────────────────────────────────────────────
    img_batch, orig_shape, img_resized = preprocess_image(image_path)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) \
        if img_resized.shape[-1] == 3 else img_resized

    # ── Step 2: Terrain segmentation ─────────────────────────────────────────
    if models["segmentation"] is not None:
        class_mask, prob_map = predict_segmentation(models["segmentation"], img_batch)
    else:
        # Rule-based: assign safety based on simple color heuristics
        h, w = img_resized.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)
        # Very basic: bright areas → paved, green areas → grass
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        green_mask = (hsv[:,:,0] > 35) & (hsv[:,:,0] < 85) & (hsv[:,:,1] > 40)
        class_mask[green_mask] = 3   # grass
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        class_mask[(gray > 150) & ~green_mask] = 1  # paved
        prob_map = None

    print(f"   Segmentation: {time.time()-t0:.2f}s")

    # ── Step 3: Terrain safety map ────────────────────────────────────────────
    terrain_map = build_terrain_safety_map(class_mask, prob_map)

    # ── Step 4: Find candidate zones ──────────────────────────────────────────
    candidates = find_landing_zone_candidates(
        terrain_map,
        min_area=cfg["decision_engine"]["min_zone_area"],
        smooth_sigma=8
    )
    candidates = non_maximum_suppression(
        candidates, nms_radius=cfg["decision_engine"]["nms_radius"]
    )
    print(f"   Found {len(candidates)} candidate landing zones")

    # ── Step 5: Weather safety score ──────────────────────────────────────────
    if models["weather"] is not None:
        wx_features = normalize_weather(weather_dict)
        weather_safety = predict_weather_safety(models["weather"], wx_features)
    else:
        weather_safety = weather_safety_rule_based(weather_dict)
    print(f"   Weather safety score: {weather_safety:.3f}")

    # ── Step 6: Threat safety ─────────────────────────────────────────────────
    threat_grid = threat_map_to_safety_grid(
        threat_dict,
        grid_h=terrain_map.shape[0],
        grid_w=terrain_map.shape[1]
    )
    if models["threat"] is not None:
        th_features = normalize_threat(threat_dict)
        threat_safety_scalar = predict_threat_safety(models["threat"], th_features)
    else:
        threat_safety_scalar = threat_safety_rule_based(threat_dict)
    print(f"   Threat safety score:  {threat_safety_scalar:.3f}")

    # ── Step 7: Decision engine ───────────────────────────────────────────────
    ranked_zones, combined_heatmap = rank_landing_zones(
        candidates, terrain_map, threat_grid,
        weather_safety, top_n=cfg["decision_engine"]["top_n"]
    )

    results = format_results(ranked_zones, image_shape=terrain_map.shape)
    print(f"   Top zone score: {ranked_zones[0]['final_score']:.3f}" if ranked_zones else "   No safe zone found!")

    # ── Step 8: Visualize & save ──────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    stem = image_path.stem

    seg_overlay   = overlay_segmentation(img_rgb, class_mask, alpha=0.5)
    terrain_rgb   = draw_safety_heatmap(terrain_map)
    annotated_img = draw_landing_zones(img_rgb.copy(), ranked_zones, terrain_map.shape)

    # Save individual outputs
    cv2.imwrite(f"{output_dir}/{stem}_segmentation.png",
                cv2.cvtColor(seg_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{output_dir}/{stem}_terrain_safety.png",
                cv2.cvtColor(terrain_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{output_dir}/{stem}_landing_zones.png",
                cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

    # Save full analysis figure
    try:
        fig = plot_full_analysis(
            img_rgb, seg_overlay, terrain_map, combined_heatmap,
            annotated_img, results["zones"],
            save_path=f"{output_dir}/{stem}_analysis.png"
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as e:
        print(f"   [WARN] Could not save analysis figure: {e}")

    # Save JSON results
    save_results(results, f"{output_dir}/{stem}_results.json")

    elapsed = time.time() - t0
    print(f"   Total time: {elapsed:.2f}s")

    return results


def predict_batch(image_dir, weather_dict, threat_dict, models, output_dir="outputs"):
    """Run prediction on all images in a directory."""
    image_dir = Path(image_dir)
    images    = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

    print(f"\n[INFO] Batch prediction: {len(images)} images in {image_dir}")

    all_results = []
    for img_path in images:
        try:
            result = predict_single(img_path, weather_dict, threat_dict, models, output_dir)
            all_results.append({"image": str(img_path), "result": result})
        except Exception as e:
            print(f"[ERROR] Failed on {img_path.name}: {e}")

    # Save batch summary
    summary_path = f"{output_dir}/batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Batch results saved → {summary_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Drone Emergency Landing Zone Predictor")
    parser.add_argument("--image",         type=str, help="Single image path")
    parser.add_argument("--image_dir",     type=str, help="Folder of images for batch prediction")
    parser.add_argument("--weather_input", type=str, default="configs/sample_weather.json",
                        help="JSON file with weather parameters")
    parser.add_argument("--threat_map",    type=str, default="configs/sample_threat_map.json",
                        help="JSON file with threat zone data")
    parser.add_argument("--output_dir",    type=str, default="outputs",
                        help="Output directory for results")
    parser.add_argument("--rule_based",    action="store_true",
                        help="Use rule-based scoring (no trained models needed)")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.print_help()
        sys.exit(1)

    # Load inputs
    with open(args.weather_input) as f:
        weather_dict = json.load(f)
    with open(args.threat_map) as f:
        threat_dict = json.load(f)

    # Load models
    models = load_models(rule_based=args.rule_based)

    if args.image:
        results = predict_single(args.image, weather_dict, threat_dict, models, args.output_dir)
        print("\n── RESULTS ──────────────────────────")
        print(json.dumps(results, indent=2))

    elif args.image_dir:
        predict_batch(args.image_dir, weather_dict, threat_dict, models, args.output_dir)


if __name__ == "__main__":
    main()
