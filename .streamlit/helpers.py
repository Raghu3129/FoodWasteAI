"""
helpers.py
----------
Shared constants, path management, and small utility functions used
across the FoodWasteAI application. Centralising these values keeps the
rest of the codebase free of "magic numbers" and duplicated file paths.
"""

import os
import time
import uuid
from datetime import datetime

# --------------------------------------------------------------------
# Base directory (root of the FoodWasteAI project)
# --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SAMPLE_IMAGES_DIR = os.path.join(DATASET_DIR, "sample_images")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PROCESSED_IMAGES_DIR = os.path.join(OUTPUTS_DIR, "processed_images")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
PREDICTIONS_CSV = os.path.join(OUTPUTS_DIR, "predictions.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")

ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CSS_PATH = os.path.join(ASSETS_DIR, "css", "style.css")

# --------------------------------------------------------------------
# Application constants
# --------------------------------------------------------------------
INSTITUTION_NAME = "Department of Data Science & Artificial Intelligence"
PROJECT_TITLE = "AI-Powered Canteen Food Waste Quantification System"
PROJECT_SUBTITLE = "Plate-Level Image Segmentation & Machine Learning"

WASTE_CATEGORIES = ["Low Waste", "Moderate Waste", "High Waste"]

CATEGORY_COLORS = {
    "Low Waste": "#1DB954",       # green
    "Moderate Waste": "#F5A623",  # amber
    "High Waste": "#E5484D",      # red
}

DEMO_CREDENTIALS = {
    "admin": {"password": "admin123", "role": "Admin", "name": "Administrator"},
    "student": {"password": "student123", "role": "Student", "name": "Student User"},
}

CSV_COLUMNS = [
    "id",
    "timestamp",
    "image_name",
    "food_area",
    "plate_area",
    "waste_percentage",
    "contour_count",
    "coverage_ratio",
    "avg_r",
    "avg_g",
    "avg_b",
    "avg_h",
    "avg_s",
    "avg_v",
    "predicted_category",
    "confidence",
    "sustainability_score",
    "processing_time_ms",
]


def ensure_directories():
    """Create every directory the app depends on if it doesn't exist yet."""
    for path in [
        DATASET_DIR,
        SAMPLE_IMAGES_DIR,
        OUTPUTS_DIR,
        PROCESSED_IMAGES_DIR,
        REPORTS_DIR,
        MODELS_DIR,
        ASSETS_DIR,
    ]:
        os.makedirs(path, exist_ok=True)


def new_record_id():
    """Generate a short, unique, human-friendly record ID."""
    return f"FW-{uuid.uuid4().hex[:8].upper()}"


def now_timestamp():
    """Return the current timestamp as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_filename_safe():
    """Return a filesystem-safe timestamp, useful for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Timer:
    """Small context manager used to measure processing time in ms."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000

    def __call__(self):
        return getattr(self, "elapsed_ms", 0.0)


def classify_sustainability_color(score):
    """Return a hex color for a given sustainability score (0-100)."""
    if score >= 75:
        return CATEGORY_COLORS["Low Waste"]
    elif score >= 45:
        return CATEGORY_COLORS["Moderate Waste"]
    return CATEGORY_COLORS["High Waste"]
