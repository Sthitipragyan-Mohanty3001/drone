"""
utils/visualization.py
Heatmaps, overlays, annotation rendering, and result visualization.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


# ── Custom colormap (green=safe, red=danger) ──────────────────────────────────
SAFETY_CMAP = LinearSegmentedColormap.from_list(
    "safety", [(0, "#FF2200"), (0.4, "#FF8800"), (0.6, "#FFEE00"), (1, "#00CC44")]
)


def overlay_segmentation(image_rgb, class_mask, alpha=0.55):
    """
    Overlay colored segmentation mask on original image.
    """
    from utils.preprocessing import CLASS_COLORS, class_mask_to_color

    color_mask = class_mask_to_color(class_mask)
    overlay    = cv2.addWeighted(image_rgb, 1 - alpha, color_mask, alpha, 0)
    return overlay


def draw_safety_heatmap(safety_map, colormap=SAFETY_CMAP):
    """
    Convert a safety float map → RGB visualization.
    Green = safe, Red = dangerous.
    """
    normed = np.clip(safety_map, 0, 1)
    rgba   = colormap(normed)               # (H, W, 4)
    rgb    = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb


def draw_landing_zones(image_rgb, ranked_zones, image_shape=None):
    """
    Draw top-3 ranked landing zones on the image with color coding.

    Zone colors:
        Rank 1 → bright green
        Rank 2 → yellow
        Rank 3 → orange
        Fallback → red dashed
    """
    output = image_rgb.copy()
    colors = [(0, 220, 60), (255, 220, 0), (255, 120, 0), (220, 0, 0)]
    labels = ["BEST", "2nd", "3rd", "FALLBACK"]

    for i, zone in enumerate(ranked_zones[:4]):
        cx, cy   = zone["centroid"]
        score    = zone["final_score"]
        color    = colors[min(i, len(colors)-1)]
        label    = labels[min(i, len(labels)-1)]
        bbox     = zone.get("bbox")
        is_emrg  = zone.get("emergency", False)

        # Draw bounding box
        if bbox:
            x, y, bw, bh = bbox
            thickness = 1 if is_emrg else 2
            cv2.rectangle(output, (x, y), (x + bw, y + bh), color, thickness)

        # Draw centroid circle
        cv2.circle(output, (cx, cy), 18, color, -1)
        cv2.circle(output, (cx, cy), 20, (255, 255, 255), 2)

        # Draw rank label
        text      = f"#{i+1} {score:.2f}"
        font      = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        thickness  = 1

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        tx = cx - tw // 2
        ty = cy - 26

        # Background pill for readability
        cv2.rectangle(output, (tx - 4, ty - th - 2), (tx + tw + 4, ty + 2),
                      (20, 20, 20), -1)
        cv2.putText(output, text, (tx, ty), font, font_scale, color, thickness)

    return output


def plot_full_analysis(
    original_image,
    seg_overlay,
    terrain_heatmap,
    combined_heatmap,
    annotated_image,
    ranked_zones,
    save_path=None
):
    """
    Create a comprehensive 2×3 analysis figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor("#0A0E1A")

    titles = [
        "Original Aerial Image",
        "Terrain Segmentation",
        "Terrain Safety Map",
        "Threat + Weather Overlay",
        "Combined Risk Heatmap",
        "Ranked Landing Zones",
    ]
    images_to_show = [
        original_image,
        seg_overlay,
        draw_safety_heatmap(terrain_heatmap),
        draw_safety_heatmap(combined_heatmap),
        draw_safety_heatmap(combined_heatmap),
        annotated_image,
    ]

    for ax, img, title in zip(axes.flat, images_to_show, titles):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    # Add legend for safety colors
    legend_elements = [
        mpatches.Patch(facecolor="#00CC44", label="Safe (≥ 0.7)"),
        mpatches.Patch(facecolor="#FFEE00", label="Marginal (0.5–0.7)"),
        mpatches.Patch(facecolor="#FF8800", label="Risky (0.3–0.5)"),
        mpatches.Patch(facecolor="#FF2200", label="Dangerous (< 0.3)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               facecolor="#0A0E1A", labelcolor="white", fontsize=10,
               framealpha=0.8, bbox_to_anchor=(0.5, 0.01))

    # Score table for ranked zones
    if ranked_zones:
        textstr = "Ranked Landing Zones\n" + "─" * 48 + "\n"
        textstr += f"{'Rank':<6} {'Loc (x,y)':<16} {'Terrain':<10} {'Weather':<10} {'Threat':<10} {'Final':<8}\n"
        textstr += "─" * 48 + "\n"
        for zone in ranked_zones:
            cx, cy = zone["centroid"]
            textstr += (
                f"#{zone.get('rank', '?'):<5} ({cx:>4},{cy:>4})     "
                f"{zone['terrain_score']:.2f}      "
                f"{zone['weather_score']:.2f}      "
                f"{zone['threat_score']:.2f}      "
                f"{zone['final_score']:.2f}\n"
            )
        fig.text(0.01, 0.01, textstr, color="#AAFFAA", fontfamily="monospace",
                 fontsize=8.5, verticalalignment="bottom",
                 bbox=dict(facecolor="#050A14", alpha=0.85, edgecolor="#334455"))

    plt.tight_layout(rect=[0, 0.10, 1, 0.98])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[OK] Analysis figure saved → {save_path}")

    return fig


def save_heatmap(safety_map, path):
    """Save safety heatmap as image file."""
    rgb = draw_safety_heatmap(safety_map)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
