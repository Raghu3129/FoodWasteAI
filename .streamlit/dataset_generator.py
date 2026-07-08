"""
dataset_generator.py
---------------------
Generates a realistic SYNTHETIC training dataset for the waste
classification RandomForestClassifier when a real labelled dataset is
not available (which is the common case for a college mini project).

The synthetic generator is built to mirror the statistical behaviour of
the real feature-extraction pipeline (utils/feature_extraction.py) so
that a model trained on it generalises reasonably well to real plate
photos:

    - Low waste plates   -> small food_area, low waste_percentage,
                              few contours, food is mostly sauce/scraps
    - Moderate waste      -> mid-range values
    - High waste plates   -> large leftover area, high waste_percentage,
                              many contours (rice grains, veggie pieces)

Run directly to regenerate dataset/training_data.csv:
    python models/dataset_generator.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import DATASET_DIR, ensure_directories
from utils.feature_extraction import FEATURE_COLUMNS

RANDOM_STATE = 42
PLATE_AREA_RANGE = (90000, 180000)  # plausible plate pixel-area at 512x512


def _generate_class_samples(rng, n_samples, category):
    """Generate `n_samples` synthetic rows for a single waste category."""
    plate_area = rng.uniform(*PLATE_AREA_RANGE, size=n_samples)

    if category == "Low Waste":
        waste_pct = rng.normal(loc=8, scale=4, size=n_samples)
        contour_count = rng.integers(1, 6, size=n_samples)
        hue = rng.normal(loc=25, scale=8, size=n_samples)      # sauces/gravy tones
        sat = rng.normal(loc=140, scale=20, size=n_samples)
        val = rng.normal(loc=90, scale=25, size=n_samples)
    elif category == "Moderate Waste":
        waste_pct = rng.normal(loc=35, scale=8, size=n_samples)
        contour_count = rng.integers(5, 14, size=n_samples)
        hue = rng.normal(loc=32, scale=10, size=n_samples)
        sat = rng.normal(loc=120, scale=25, size=n_samples)
        val = rng.normal(loc=120, scale=25, size=n_samples)
    else:  # High Waste
        waste_pct = rng.normal(loc=65, scale=10, size=n_samples)
        contour_count = rng.integers(12, 30, size=n_samples)
        hue = rng.normal(loc=40, scale=12, size=n_samples)      # rice/veg/varied tones
        sat = rng.normal(loc=100, scale=30, size=n_samples)
        val = rng.normal(loc=150, scale=30, size=n_samples)

    waste_pct = np.clip(waste_pct, 0, 100)
    food_area = plate_area * (waste_pct / 100.0)
    coverage_ratio = np.clip(food_area / plate_area, 0, 5)
    hue = np.clip(hue, 0, 180)
    sat = np.clip(sat, 0, 255)
    val = np.clip(val, 0, 255)
    contour_count = np.clip(contour_count, 1, None)

    df = pd.DataFrame(
        {
            "food_area": food_area.astype(int),
            "plate_area": plate_area.astype(int),
            "waste_percentage": np.round(waste_pct, 2),
            "contour_count": contour_count.astype(int),
            "coverage_ratio": np.round(coverage_ratio, 4),
            "avg_h": np.round(hue, 2),
            "avg_s": np.round(sat, 2),
            "avg_v": np.round(val, 2),
            "label": category,
        }
    )
    return df


def generate_synthetic_dataset(n_per_class=400, save_path=None):
    """
    Build a balanced synthetic dataset across the three waste categories
    and optionally persist it to CSV.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    frames = [
        _generate_class_samples(rng, n_per_class, "Low Waste"),
        _generate_class_samples(rng, n_per_class, "Moderate Waste"),
        _generate_class_samples(rng, n_per_class, "High Waste"),
    ]
    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # sanity check: columns must line up with the live feature extractor
    assert list(dataset.columns[:-1]) == FEATURE_COLUMNS, "Feature column mismatch!"

    if save_path:
        ensure_directories()
        dataset.to_csv(save_path, index=False)

    return dataset


if __name__ == "__main__":
    ensure_directories()
    out_path = os.path.join(DATASET_DIR, "training_data.csv")
    data = generate_synthetic_dataset(n_per_class=400, save_path=out_path)
    print(f"Synthetic dataset generated: {out_path}")
    print(f"Shape: {data.shape}")
    print(data["label"].value_counts())
