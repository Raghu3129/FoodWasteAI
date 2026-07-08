"""
segmentation.py
----------------
Stage 2 of the Computer Vision pipeline.

Responsible for separating the "plate" region from the "food" region
using classical (non deep-learning) computer vision:

    HSV Image -> Color Thresholding -> Morphological Ops -> Contours

This is intentionally rule-based (no YOLO / U-Net) as required by the
project's technology constraints, while still being robust enough to
give meaningful, explainable results for a variety of plate photos.
"""

import cv2
import numpy as np


def segment_plate_and_food(hsv_image, bgr_image):
    """
    Segment the plate and the leftover food from the HSV image.

    Strategy
    --------
    1. The plate is assumed to be a large, relatively low-saturation
       region (white / light ceramic plates are the common case).
    2. Everything with high enough saturation/value sitting inside the
       plate region that is *not* plate-colored is treated as food.
    3. A generic "not background" mask is also computed so the approach
       degrades gracefully for colored plates.

    Returns
    -------
    dict with keys: plate_mask, food_mask, segmented_display
    """
    h, s, v = cv2.split(hsv_image)

    # --- Plate mask: low saturation, high brightness (ceramic/white plate) ---
    plate_mask = cv2.inRange(
        hsv_image,
        np.array([0, 0, 120], dtype=np.uint8),
        np.array([180, 60, 255], dtype=np.uint8),
    )

    # --- Background mask: very dark regions (table/background clutter) ---
    background_mask = cv2.inRange(
        hsv_image,
        np.array([0, 0, 0], dtype=np.uint8),
        np.array([180, 255, 40], dtype=np.uint8),
    )

    # Everything that is neither plate nor background is candidate "food"
    not_plate = cv2.bitwise_not(plate_mask)
    not_background = cv2.bitwise_not(background_mask)
    food_mask = cv2.bitwise_and(not_plate, not_background)

    # Restrict food detection to the plate's convex region so stray table
    # objects outside the plate are not counted as leftover food.
    plate_region_mask = get_largest_contour_mask(cv2.bitwise_or(plate_mask, food_mask))
    food_mask = cv2.bitwise_and(food_mask, plate_region_mask)

    # --- Morphological clean-up ---
    plate_mask = apply_morphology(plate_mask)
    food_mask = apply_morphology(food_mask)

    # Build a visual overlay: plate in translucent green, food in translucent red
    segmented_display = bgr_image.copy()
    overlay = bgr_image.copy()
    overlay[plate_mask > 0] = (60, 200, 60)
    overlay[food_mask > 0] = (50, 50, 220)
    segmented_display = cv2.addWeighted(overlay, 0.45, segmented_display, 0.55, 0)

    return {
        "plate_mask": plate_mask,
        "food_mask": food_mask,
        "segmented_display": segmented_display,
    }


def get_largest_contour_mask(mask):
    """Return a filled mask of the single largest contour in `mask`."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask)
    if not contours:
        return np.ones_like(mask) * 255  # fallback: use whole frame
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(result, [largest], -1, 255, thickness=cv2.FILLED)
    return result


def apply_morphology(mask, kernel_size=5):
    """
    Clean a binary mask using morphological opening and closing.

    Opening removes small noisy specks; closing fills small holes inside
    larger detected regions so contour detection produces solid blobs.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


def detect_contours(mask):
    """
    Detect external contours in a binary mask.

    Returns
    -------
    contours : list of contour arrays
    contour_image : mask converted to a 3-channel image with contours drawn
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter tiny noise contours (< 15 px area)
    contours = [c for c in contours if cv2.contourArea(c) > 15]

    contour_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return contours, contour_image
