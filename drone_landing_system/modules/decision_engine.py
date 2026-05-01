"""
modules/decision_engine.py
Final Decision Engine — Combines terrain + weather + threat scores
to produce a ranked list of safe landing zones.
"""

import yaml
import numpy as np
import json
from pathlib import Path

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

DE_CFG = cfg["decision_engine"]
WEIGHTS = DE_CFG["weights"]


def compute_final_score(terrain_score, weather_score, threat_score, weights=None):
    """
    Weighted combination of the three module scores.

    Formula:
        Final = w_terrain * terrain + w_weather * weather + w_threat * threat

    All scores in [0, 1].  1 = safest, 0 = most dangerous.
    """
    weights = weights or WEIGHTS
    w_t = weights["terrain"]
    w_w = weights["weather"]
    w_th = weights["threat"]

    total = w_t + w_w + w_th
    score = (
        w_t  * terrain_score +
        w_w  * weather_score +
        w_th * threat_score
    ) / total

    return float(np.clip(score, 0.0, 1.0))


def rank_landing_zones(
    terrain_candidates,
    terrain_safety_map,
    threat_safety_grid,
    weather_safety_score,
    top_n=3,
    weights=None
):
    """
    Rank all terrain candidates using combined score.

    Args:
        terrain_candidates: list of candidate zones from terrain_analyzer
        terrain_safety_map: (H, W) terrain safety array
        threat_safety_grid: (H, W) threat safety array (or None → scalar used)
        weather_safety_score: float scalar weather safety score
        top_n: number of top zones to return
        weights: dict with keys terrain, weather, threat

    Returns:
        ranked_zones: list of dicts sorted by final_score desc
        combined_heatmap: np.ndarray (H, W) showing overall safety
    """
    weights = weights or WEIGHTS

    # Build combined heatmap
    if threat_safety_grid is not None:
        h, w = terrain_safety_map.shape
        # Resize threat grid if needed
        import cv2
        tg = cv2.resize(threat_safety_grid, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        tg = np.ones_like(terrain_safety_map)

    # Pixel-wise heatmap
    weather_map   = np.full_like(terrain_safety_map, weather_safety_score)
    combined_heatmap = (
        weights["terrain"] * terrain_safety_map +
        weights["weather"] * weather_map +
        weights["threat"]  * tg
    ) / sum(weights.values())

    # Score each candidate
    ranked = []
    for zone in terrain_candidates:
        cx, cy = zone["centroid"]
        terrain_s  = float(terrain_safety_map[cy, cx])
        threat_s   = float(tg[cy, cx])

        final_score = compute_final_score(terrain_s, weather_safety_score, threat_s, weights)

        ranked.append({
            **zone,
            "terrain_score":  round(terrain_s, 4),
            "weather_score":  round(weather_safety_score, 4),
            "threat_score":   round(threat_s, 4),
            "final_score":    round(final_score, 4),
            "safe":           final_score >= DE_CFG["min_safe_score"],
        })

    # Sort by final_score desc
    ranked.sort(key=lambda z: z["final_score"], reverse=True)

    # Return top-N safe zones
    top_zones = [z for z in ranked if z["safe"]][:top_n]

    # Emergency fallback: if no safe zones, return best available
    if not top_zones and ranked:
        top_zones = ranked[:1]
        top_zones[0]["emergency_fallback"] = True

    return top_zones, combined_heatmap


def format_results(ranked_zones, image_shape=None):
    """
    Format ranked zones for display/output.
    Returns a human-readable summary dict.
    """
    h, w = image_shape if image_shape else (416, 608)

    output = {
        "status": "OK" if ranked_zones else "NO_SAFE_ZONE",
        "n_zones_found": len(ranked_zones),
        "zones": []
    }

    for i, zone in enumerate(ranked_zones):
        cx, cy = zone["centroid"]
        # Normalize coordinates to [0, 1] for display
        zone_info = {
            "rank":            i + 1,
            "centroid_px":     (cx, cy),
            "centroid_norm":   (round(cx / w, 3), round(cy / h, 3)),
            "bbox":            zone.get("bbox"),
            "area_px":         zone.get("area"),
            "terrain_score":   zone.get("terrain_score"),
            "weather_score":   zone.get("weather_score"),
            "threat_score":    zone.get("threat_score"),
            "final_score":     zone.get("final_score"),
            "safe":            zone.get("safe"),
            "emergency":       zone.get("emergency_fallback", False),
        }
        output["zones"].append(zone_info)

    # Summary for top zone
    if ranked_zones:
        best = ranked_zones[0]
        output["recommendation"] = {
            "action":      "LAND" if best["final_score"] >= 0.7 else "APPROACH_CAREFULLY",
            "location":    best["centroid"],
            "confidence":  best["final_score"],
        }
    else:
        output["recommendation"] = {
            "action":  "ABORT_LANDING",
            "reason":  "No safe landing zones found in current area.",
        }

    return output


def save_results(results, path="outputs/landing_results.json"):
    """Save results as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Results saved → {path}")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate inputs
    terrain_map = np.random.rand(416, 608).astype(np.float32)
    threat_grid = np.random.rand(416, 608).astype(np.float32)
    weather_s   = 0.72

    # Fake candidates
    candidates = [
        {"centroid": (200, 150), "score": 0.85, "area": 1200, "bbox": (180, 130, 40, 40)},
        {"centroid": (400, 300), "score": 0.70, "area": 800,  "bbox": (380, 280, 40, 40)},
        {"centroid": (100, 80),  "score": 0.65, "area": 600,  "bbox": (80,  60,  40, 40)},
    ]

    ranked, heatmap = rank_landing_zones(
        candidates, terrain_map, threat_grid, weather_s, top_n=3
    )

    results = format_results(ranked, image_shape=(416, 608))
    print(json.dumps(results, indent=2))
