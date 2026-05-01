# 🛩️ Drone Emergency Landing Zone Prediction System
### ML/DL-based UAV Safe Landing in Bad Weather & War-Prone Zones

---

## 📁 Project Structure

```
drone_landing_system/
│
├── configs/
│   └── config.yaml              # All hyperparameters and paths
│
├── data/
│   └── prepare_dataset.py       # Download & prepare dataset
│
├── models/
│   ├── segmentation_model.py    # U-Net / ResNet50-UNet (Module 1)
│   ├── weather_model.py         # Weather MLP/LSTM (Module 2)
│   └── threat_model.py          # Threat zone CNN (Module 3)
│
├── modules/
│   ├── terrain_analyzer.py      # Terrain safety scoring
│   ├── weather_analyzer.py      # Weather risk scoring
│   ├── threat_analyzer.py       # Threat zone analysis
│   └── decision_engine.py       # Final scoring + ranking
│
├── utils/
│   ├── visualization.py         # Heatmaps, overlays, plots
│   ├── preprocessing.py         # Image & data preprocessing
│   └── metrics.py               # IoU, accuracy, risk metrics
│
├── notebooks/
│   └── full_pipeline_demo.ipynb # End-to-end Jupyter demo
│
├── train.py                     # Train all models
├── predict.py                   # Run full prediction pipeline
├── evaluate.py                  # Evaluate model performance
├── app.py                       # Streamlit dashboard UI
└── requirements.txt             # All dependencies
```

---

## 🚀 Quick Start

### Step 1 — Clone & Install

```bash
git clone <your-repo>
cd drone_landing_system
pip install -r requirements.txt
```

### Step 2 — Download the Dataset

Download the **Aerial Semantic Segmentation Drone Dataset** from Kaggle:
- URL: https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset

```bash
# Option A: Using Kaggle CLI
pip install kaggle
kaggle datasets download -d bulentsiyah/semantic-drone-dataset
unzip semantic-drone-dataset.zip -d data/raw/

# Option B: Manual download
# Place the dataset at: data/raw/semantic_drone_dataset/
```

Expected folder structure after extraction:
```
data/raw/semantic_drone_dataset/
├── original_images/       (400 images, .jpg)
├── label_images_semantic/ (400 labels, .png)
└── class_dict_seg.csv     (24 class color mappings)
```

### Step 3 — Prepare Dataset

```bash
python data/prepare_dataset.py
```

### Step 4 — Train Models

```bash
# Train all three modules
python train.py --module all

# Or train individually
python train.py --module segmentation
python train.py --module weather
python train.py --module threat
```

### Step 5 — Run Prediction

```bash
# Predict on a single image
python predict.py --image data/raw/semantic_drone_dataset/original_images/001.jpg \
                  --weather_input configs/sample_weather.json \
                  --threat_map configs/sample_threat_map.json

# Predict on a folder of images
python predict.py --image_dir data/raw/semantic_drone_dataset/original_images/ \
                  --output_dir outputs/
```

### Step 6 — Launch Dashboard (Optional)

```bash
streamlit run app.py
```

---

## ⚙️ Configuration

Edit `configs/config.yaml` to change:
- Model architecture (unet / deeplabv3)
- Number of segmentation classes
- Training epochs, batch size, learning rate
- Risk weight coefficients
- Weather and threat thresholds

---

## 📊 Dataset Classes (24 Classes)

| Class | Label |
|---|---|
| 0 | unlabeled |
| 1 | paved-area ✅ Safe |
| 2 | dirt ✅ Safe |
| 3 | grass ✅ Safe |
| 4 | gravel ⚠️ Marginal |
| 5 | water ❌ Unsafe |
| 6 | rocks ❌ Unsafe |
| 7 | pool ❌ Unsafe |
| 8 | vegetation ⚠️ Marginal |
| 9 | roof ⚠️ Marginal |
| 10 | wall ❌ Unsafe |
| 11 | window ❌ Unsafe |
| 12 | door ❌ Unsafe |
| 13 | fence ❌ Unsafe |
| 14 | fence-pole ❌ Unsafe |
| 15 | person ❌ Unsafe |
| 16 | dog ❌ Unsafe |
| 17 | car ❌ Unsafe |
| 18 | bicycle ❌ Unsafe |
| 19 | roof ⚠️ Marginal |
| 20 | obstacle ❌ Unsafe |

---

## 🧮 Risk Scoring Formula

```
Final_Score = (w1 × Terrain_Safety) + (w2 × Weather_Safety) + (w3 × Threat_Safety)

Default weights: w1=0.5, w2=0.3, w3=0.2
Score range: 0.0 (most dangerous) → 1.0 (fully safe)
```

---

## 📦 Requirements

- Python 3.8+
- TensorFlow 2.x or PyTorch
- OpenCV, NumPy, Matplotlib
- Streamlit (for dashboard)
- See `requirements.txt` for full list

---

## 🏋️ Training Tips

- Use GPU for segmentation training (ResNet50-UNet is heavy)
- Kaggle free tier (P100) works fine — use the provided notebook
- Recommended: 20–30 epochs for good convergence
- Use pretrained ResNet50 weights (auto-downloaded)

---

## 📈 Expected Results

| Module | Metric | Expected |
|---|---|---|
| Segmentation | Pixel Accuracy | ~92–95% |
| Segmentation | Mean IoU | ~65–75% |
| Weather Model | MAE | < 0.05 |
| Overall System | Top-3 Zone Accuracy | ~85%+ |
