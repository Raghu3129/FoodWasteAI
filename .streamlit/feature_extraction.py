"""
feature_extraction.py
----------------------
Stage 3 of the Computer Vision pipeline.

Converts segmentation masks and contours into the numeric feature
vector that is fed to the RandomForestClassifier:

    Food Area, Plate Area, Waste %, Contour Count,
    Avg RGB, Avg HSV, Coverage Ratio
"""

import cv2
import numpy as np


def extract_features(bgr_image, hsv_image, plate_mask, food_mask, contours):
    """
    Compute the full feature set used both for model training and for
    live prediction on an uploaded image.

    Returns
    -------
    dict of scalar features
    """
    plate_area = int(np.count_nonzero(plate_mask))
    food_area = int(np.count_nonzero(food_mask))

    # Total plate+food area is used as the denominator for waste % so the
    # measure is robust to how much of the frame the plate occupies.
    total_area = plate_area + food_area
    waste_percentage = (food_area / total_area * 100) if total_area > 0 else 0.0
    waste_percentage = float(np.clip(waste_percentage, 0, 100))

    coverage_ratio = float(food_area / plate_area) if plate_area > 0 else 0.0
    coverage_ratio = float(np.clip(coverage_ratio, 0, 5))

    contour_count = len(contours)

    # Average RGB / HSV are computed only over the "food" pixels, since
    # that's the region whose colour profile is diagnostic of leftover
    # food type/amount (empty plate vs. full plate look very different).
    if food_area > 0:
        food_pixels_bgr = bgr_image[food_mask > 0]
        food_pixels_hsv = hsv_image[food_mask > 0]
        avg_b, avg_g, avg_r = food_pixels_bgr.mean(axis=0)
        avg_h, avg_s, avg_v = food_pixels_hsv.mean(axis=0)
    else:
        avg_b = avg_g = avg_r = 0.0
        avg_h = avg_s = avg_v = 0.0

    features = {
        "food_area": food_area,
        "plate_area": plate_area,
        "waste_percentage": round(waste_percentage, 2),
        "contour_count": contour_count,
        "coverage_ratio": round(coverage_ratio, 4),
        "avg_r": round(float(avg_r), 2),
        "avg_g": round(float(avg_g), 2),
        "avg_b": round(float(avg_b), 2),
        "avg_h": round(float(avg_h), 2),
        "avg_s": round(float(avg_s), 2),
        "avg_v": round(float(avg_v), 2),
    }
    return features


def feature_vector_for_model(features):
    """
    Convert the feature dict into the ordered numeric vector expected by
    the trained RandomForestClassifier. Order MUST match
    models/train_model.py and models/dataset_generator.py.
    """
    return [
        features["food_area"],
        features["plate_area"],
        features["waste_percentage"],
        features["contour_count"],
        features["coverage_ratio"],
        features["avg_h"],
        features["avg_s"],
        features["avg_v"],
    ]


FEATURE_COLUMNS = [
    "food_area",
    "plate_area",
    "waste_percentage",
    "contour_count",
    "coverage_ratio",
    "avg_h",
    "avg_s",
    "avg_v",
]
