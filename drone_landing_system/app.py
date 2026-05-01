"""
app.py
Streamlit interactive dashboard for the drone landing system.

Launch:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
import json
import yaml
import time
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drone Emergency Landing System",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0A0E1A; color: #E0E8FF; }
    .stApp { background-color: #0A0E1A; }
    .metric-card {
        background: linear-gradient(135deg, #111827, #1E2A3A);
        border: 1px solid #2A3F5F;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .zone-safe { border-left: 4px solid #00CC44; padding-left: 12px; }
    .zone-warn { border-left: 4px solid #FF8800; padding-left: 12px; }
    .zone-danger { border-left: 4px solid #FF2200; padding-left: 12px; }
    h1, h2, h3 { color: #7EB8F7; }
</style>
""", unsafe_allow_html=True)

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)


# ── Sidebar — Inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/280x80/0A0E1A/7EB8F7?text=🛩️+Landing+AI",
             use_column_width=True)
    st.title("⚙️ System Controls")

    st.subheader("🌩️ Weather Conditions")
    visibility    = st.slider("Visibility (km)",      0.0, 10.0, 5.0, 0.1)
    wind_speed    = st.slider("Wind Speed (m/s)",     0.0, 30.0, 8.0, 0.5)
    precipitation = st.slider("Precipitation (mm/hr)", 0.0, 50.0, 2.0, 0.5)
    humidity      = st.slider("Humidity (%)",         0.0, 100.0, 65.0, 1.0)
    fog_index     = st.slider("Fog Index",            0.0, 1.0,  0.2,  0.01)
    smoke_density = st.slider("Smoke Density",        0.0, 1.0,  0.1,  0.01)

    st.subheader("⚠️ Threat Assessment")
    hostile_area   = st.slider("Hostile Area Level",   0.0, 1.0, 0.2, 0.01)
    prohibited     = st.slider("Prohibited Zone",      0.0, 1.0, 0.1, 0.01)
    mine_region    = st.slider("Mine Region",          0.0, 1.0, 0.05, 0.01)
    gunfire_prob   = st.slider("Gunfire Probability",  0.0, 1.0, 0.15, 0.01)
    blast_radius   = st.slider("Blast Radius Risk",    0.0, 1.0, 0.1,  0.01)

    st.subheader("⚖️ Score Weights")
    w_terrain = st.slider("Terrain weight", 0.1, 0.8, 0.5, 0.05)
    w_weather = st.slider("Weather weight", 0.1, 0.6, 0.3, 0.05)
    w_threat  = 1.0 - w_terrain - w_weather
    st.info(f"Threat weight: {w_threat:.2f}")

    rule_based = st.checkbox("Rule-based mode (no ML model)", value=True)
    analyze_btn = st.button("🚀 Analyze Landing Zones", use_container_width=True)


# ── Main content ──────────────────────────────────────────────────────────────
st.title("🛩️ Drone Emergency Landing Zone Prediction")
st.caption("Real-time safe landing zone prediction in bad weather + war-prone zones")

# Upload image
col_upload, col_info = st.columns([2, 1])
with col_upload:
    uploaded = st.file_uploader("Upload Aerial Image", type=["jpg", "jpeg", "png"],
                                 help="Upload a drone/aerial view image")
with col_info:
    st.info("""
    **Dataset Source:**
    Aerial Semantic Segmentation Drone Dataset (Kaggle)

    **Supported:**
    - .jpg / .jpeg / .png
    - Any aerial/drone imagery
    """)

if uploaded is None:
    st.markdown("---")
    st.markdown("### 📡 Waiting for aerial image...")
    st.markdown("""
    Upload an aerial image to begin analysis. The system will:
    1. **Segment** the terrain into 24 semantic classes
    2. **Score** each region for landing safety
    3. **Factor in** weather and threat conditions
    4. **Rank** the top 3 safest landing zones
    """)
    st.stop()


# ── Run analysis when button clicked ─────────────────────────────────────────
def run_analysis(uploaded_file, weather_dict, threat_dict, weights, rule_based):
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    H, W = cfg["dataset"]["image_height"], cfg["dataset"]["image_width"]
    img_resized = cv2.resize(img_rgb, (W, H))

    # -- Terrain segmentation (rule-based or ML)
    if rule_based or True:  # Always use rule-based in demo mode
        from models.weather_model import weather_safety_rule_based
        from models.threat_model import threat_safety_rule_based, threat_map_to_safety_grid

        # Simple heuristic segmentation
        class_mask = np.zeros((H, W), dtype=np.uint8)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
        green = (hsv[:,:,0] > 35) & (hsv[:,:,0] < 85) & (hsv[:,:,1] > 50)
        gray  = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        class_mask[green] = 3          # grass
        class_mask[(gray > 160) & ~green] = 1  # paved
        class_mask[(gray > 100) & (gray <= 160) & ~green] = 4  # gravel
        class_mask[gray <= 60] = 5     # water/dark

        weather_safety = weather_safety_rule_based(weather_dict)
        threat_safety  = threat_safety_rule_based(threat_dict)
        threat_grid    = threat_map_to_safety_grid(threat_dict, H, W)
    else:
        # Load ML models (if available)
        pass

    # Terrain safety map
    from modules.terrain_analyzer import (build_terrain_safety_map,
                                           find_landing_zone_candidates,
                                           non_maximum_suppression)
    from modules.decision_engine import rank_landing_zones, format_results
    from utils.visualization import (overlay_segmentation, draw_safety_heatmap,
                                      draw_landing_zones)
    from utils.preprocessing import class_mask_to_color

    terrain_map = build_terrain_safety_map(class_mask)
    candidates  = find_landing_zone_candidates(terrain_map, min_area=400, smooth_sigma=8)
    candidates  = non_maximum_suppression(candidates, nms_radius=40)

    ranked_zones, combined_heatmap = rank_landing_zones(
        candidates, terrain_map, threat_grid, weather_safety,
        top_n=3, weights=weights
    )

    results = format_results(ranked_zones, image_shape=(H, W))

    # Visualization
    seg_overlay   = overlay_segmentation(img_resized, class_mask, alpha=0.45)
    terrain_rgb   = draw_safety_heatmap(terrain_map)
    combined_rgb  = draw_safety_heatmap(combined_heatmap)
    annotated     = draw_landing_zones(img_resized.copy(), ranked_zones, (H, W))

    return {
        "original":     img_resized,
        "seg_overlay":  seg_overlay,
        "terrain_rgb":  terrain_rgb,
        "combined_rgb": combined_rgb,
        "annotated":    annotated,
        "results":      results,
        "weather_s":    weather_safety,
        "threat_s":     threat_safety,
        "terrain_map":  terrain_map,
        "ranked_zones": ranked_zones,
    }


# Build input dicts
weather_dict = {
    "visibility": visibility, "wind_speed": wind_speed,
    "precipitation": precipitation, "humidity": humidity,
    "fog_index": fog_index, "smoke_density": smoke_density,
    "temperature": 22.0
}
threat_dict = {
    "hostile_area": hostile_area, "prohibited_zone": prohibited,
    "mine_region": mine_region, "gunfire_probability": gunfire_prob,
    "blast_radius": blast_radius, "zones": []
}
weights = {"terrain": w_terrain, "weather": w_weather, "threat": max(0.05, w_threat)}

if analyze_btn:
    with st.spinner("🔍 Analyzing aerial image..."):
        uploaded.seek(0)
        t0 = time.time()
        data = run_analysis(uploaded, weather_dict, threat_dict, weights, rule_based)
        elapsed = time.time() - t0

    st.session_state["analysis"] = data
    st.session_state["elapsed"]  = elapsed

if "analysis" in st.session_state:
    data    = st.session_state["analysis"]
    results = data["results"]
    elapsed = st.session_state.get("elapsed", 0)

    # ── Score cards ───────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("⏱️ Analysis Time", f"{elapsed:.2f}s")
    with c2:
        st.metric("🌿 Weather Safety", f"{data['weather_s']:.1%}")
    with c3:
        st.metric("🎯 Threat Safety", f"{data['threat_s']:.1%}")
    with c4:
        n_zones = results["n_zones_found"]
        st.metric("📍 Safe Zones Found", str(n_zones))

    # ── Image grid ────────────────────────────────────────────────────────────
    st.markdown("### 📊 Analysis Results")
    col1, col2 = st.columns(2)
    with col1:
        st.image(data["original"],    caption="Original Aerial Image",    use_column_width=True)
        st.image(data["terrain_rgb"], caption="Terrain Safety Map (Green=Safe, Red=Danger)", use_column_width=True)
    with col2:
        st.image(data["seg_overlay"],  caption="Semantic Segmentation",   use_column_width=True)
        st.image(data["annotated"],    caption="🎯 Ranked Landing Zones", use_column_width=True)

    # Combined heatmap
    st.image(data["combined_rgb"], caption="Combined Risk Heatmap (Terrain + Weather + Threat)",
             use_column_width=True)

    # ── Ranked zones table ────────────────────────────────────────────────────
    st.markdown("### 📍 Ranked Landing Zones")

    if not results["zones"]:
        st.error("❌ No safe landing zones found! Consider aborting or changing approach vector.")
    else:
        for zone in results["zones"]:
            score = zone["final_score"]
            cls   = "zone-safe" if score >= 0.70 else ("zone-warn" if score >= 0.50 else "zone-danger")
            icon  = "🟢" if score >= 0.70 else ("🟡" if score >= 0.50 else "🔴")
            action = results.get("recommendation", {}).get("action", "")

            with st.expander(f"{icon} Zone #{zone['rank']} — Score: {score:.3f}  {'⚡ RECOMMENDED' if zone['rank'] == 1 else ''}",
                             expanded=zone["rank"] == 1):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Terrain",  f"{zone['terrain_score']:.3f}")
                c2.metric("Weather",  f"{zone['weather_score']:.3f}")
                c3.metric("Threat",   f"{zone['threat_score']:.3f}")
                c4.metric("FINAL",    f"{zone['final_score']:.3f}")
                cx, cy = zone["centroid_px"]
                st.caption(f"📍 Centroid: ({cx}, {cy}) px  |  Area: {zone['area_px']} px²")

    # ── Recommendation ────────────────────────────────────────────────────────
    rec = results.get("recommendation", {})
    if rec.get("action") == "LAND":
        st.success(f"✅ **RECOMMENDATION: LAND** — Confidence: {rec['confidence']:.1%}")
    elif rec.get("action") == "APPROACH_CAREFULLY":
        st.warning(f"⚠️ **RECOMMENDATION: APPROACH CAREFULLY** — Confidence: {rec['confidence']:.1%}")
    elif rec.get("action") == "ABORT_LANDING":
        st.error(f"🚫 **RECOMMENDATION: ABORT LANDING** — {rec.get('reason', '')}")

    # ── JSON output ───────────────────────────────────────────────────────────
    with st.expander("📄 Full JSON Results"):
        st.json(results)
