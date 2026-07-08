# 🥗 FoodWasteAI

**AI-Powered Canteen Food Waste Quantification System using Plate-Level Image Segmentation and Machine Learning**

A production-styled web application that estimates leftover food waste from plate photographs using classical Computer Vision (OpenCV) and a scikit-learn `RandomForestClassifier`, wrapped in a premium Streamlit dashboard.

> **Academic Mini Project** — B.Tech Data Science & Artificial Intelligence
> **Author:** Rayi Venkata Raghu Veer

---

## 📌 Project Description

Canteens generate substantial food waste every day, but it is rarely measured objectively. FoodWasteAI automates this measurement: a user uploads a photo of a plate after a meal, and the system runs it through a full computer-vision pipeline to segment leftover food from the plate, extracts quantitative features, and classifies the waste level (**Low / Moderate / High**) using a trained machine learning model. The app then generates dynamic recommendations, tracks a sustainability score, stores every analysis in a searchable history, and can export a professional PDF report.

---

## ✨ Features

- 🔐 **Login system** with Admin / Student demo roles
- 🏠 **Landing dashboard** with live statistics and AI workflow overview
- 📤 **Drag & drop image upload** (JPG / JPEG / PNG) with instant preview
- 🔬 **Full CV pipeline visualization** — resize → denoise → blur → HSV → segmentation → morphology → contours
- 🧬 **Feature extraction** — food area, plate area, waste %, contour count, average RGB/HSV, coverage ratio
- 🌲 **RandomForestClassifier** (n_estimators=100) waste-level prediction with confidence score
- ♻️ **Sustainability score** (0–100) with a gauge/progress-ring visualization
- 💡 **Dynamic recommendation engine** — suggestions vary by predicted category
- 📊 **Analytics dashboard** — pie chart, bar chart, trend line, daily & monthly statistics
- 🗂️ **History module** — search, filter, sort, delete, and download predictions as CSV
- 📄 **One-click PDF report export** with embedded image, metrics, and recommendations
- 🎨 **Premium SaaS-style UI** (Apple / Stripe / Notion inspired) — green/white/dark-gray theme, cards, hover effects, responsive layout

---

## ⚙️ Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| Computer Vision | OpenCV |
| Machine Learning | scikit-learn (RandomForestClassifier) |
| Visualization | Plotly, Matplotlib |
| Data Handling | NumPy, Pandas |
| Report Export | ReportLab (PDF) |
| Storage | CSV |

No TensorFlow, PyTorch, YOLO, U-Net, Flask, Django, React, Node.js, Firebase, MongoDB, or paid APIs are used — everything runs locally with classical CV + traditional ML.

---

## 🧠 AI Architecture / Workflow

```
Plate Image Upload
        │
        ▼
 ┌─────────────────────┐
 │  Preprocessing       │  Resize → Noise Removal → Gaussian Blur → HSV Conversion
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │  Segmentation         │  Color Thresholding → Morphological Ops → Contour Detection
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │  Feature Extraction  │  Food/Plate Area, Waste %, Contour Count, Avg RGB/HSV, Coverage Ratio
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │  ML Classification   │  RandomForestClassifier → Low / Moderate / High Waste + Confidence
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │  Post-processing      │  Sustainability Score + Dynamic Recommendations
 └─────────┬───────────┘
           ▼
   History (CSV) + Analytics + PDF Report
```

The ML model is trained on a **realistic synthetic dataset** that is auto-generated (`models/dataset_generator.py`) whenever a labelled real-world dataset is unavailable — this keeps the project fully runnable out-of-the-box while remaining easy to swap for real data later.

---

## 📁 Folder Structure

```
FoodWasteAI/
├── app.py                     # Streamlit application entry point
├── requirements.txt
├── README.md
├── dataset/
│   ├── sample_images/         # Sample plate images for demo/testing
│   └── training_data.csv      # Auto-generated synthetic training dataset
├── outputs/
│   ├── reports/                # Generated PDF reports
│   ├── processed_images/       # Segmented/processed image outputs
│   └── predictions.csv         # Prediction history (created on first run)
├── models/
│   ├── train_model.py          # Model training script
│   ├── dataset_generator.py    # Synthetic dataset generator
│   └── random_forest.pkl       # Trained model (created on first run)
├── utils/
│   ├── preprocessing.py        # Resize, denoise, blur, HSV conversion
│   ├── segmentation.py         # Color segmentation, morphology, contours
│   ├── feature_extraction.py   # Feature engineering
│   ├── predictor.py            # Model loading + inference
│   ├── recommendation.py       # Dynamic recommendation engine
│   ├── charts.py                # Plotly chart builders
│   ├── pdf_export.py           # PDF report generation
│   └── helpers.py              # Shared constants & utilities
└── assets/
    ├── css/style.css            # Premium SaaS theme
    └── icons/
```

---

## 🚀 Installation & Execution

```bash
# 1. Navigate into the project folder
cd FoodWasteAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`. On first run, if no trained model exists, the app automatically generates a synthetic dataset and trains the RandomForestClassifier — no manual setup step is required.

### 🔑 Demo Credentials

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `admin123` |
| Student | `student` | `student123` |

### 🖼️ Try it out

Sample plate images are provided in `dataset/sample_images/` for quick demonstration if you don't have your own plate photos handy.

---

## 📊 Feature Set Used by the ML Model

| Feature | Description |
|---|---|
| `food_area` | Pixel area of detected leftover food |
| `plate_area` | Pixel area of the detected plate |
| `waste_percentage` | food_area / (food_area + plate_area) × 100 |
| `contour_count` | Number of distinct food blobs detected |
| `coverage_ratio` | food_area / plate_area |
| `avg_h`, `avg_s`, `avg_v` | Mean Hue/Saturation/Value of food pixels |

---

## 🔮 Future Enhancements

- Deep-learning based segmentation (U-Net / Mask R-CNN) for finer-grained masks
- Food-type classification alongside waste quantification
- Multi-camera live canteen deployment with real-time dashboards
- Integration with kitchen inventory systems for closed-loop waste reduction
- Migrate history storage from CSV to a lightweight relational database

---

## 📝 Notes for Evaluators

- All computer vision is classical/rule-based (OpenCV color segmentation + morphology + contours) — no deep learning is used for segmentation, per project constraints.
- Waste classification is performed by a genuinely trained `RandomForestClassifier`, not hardcoded thresholds.
- The synthetic training dataset generator (`models/dataset_generator.py`) mirrors the statistical distribution the live feature extractor produces, so the trained model generalizes sensibly to real feature vectors from uploaded images.
- Segmentation accuracy depends on plate/background contrast in the uploaded photo, as is expected for a classical CV approach without deep learning.
