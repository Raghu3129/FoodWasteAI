"""
preprocessing.py
-----------------
Stage 1 of the Computer Vision pipeline.

Responsible for taking a raw uploaded plate image and preparing it for
segmentation:

    Raw Image -> Resize -> Noise Removal -> Gaussian Blur -> HSV Conversion

Every function returns a new NumPy array (OpenCV images are just NumPy
arrays under the hood) so intermediate stages can be displayed in the
Streamlit UI for transparency.
"""

import cv2
import numpy as np

TARGET_SIZE = (512, 512)  # width, height used for all downstream processing


def resize_image(image, target_size=TARGET_SIZE):
    """
    Resize the input image to a fixed target size.

    A fixed size keeps every downstream measurement (areas, pixel counts)
    comparable across different uploaded photos, regardless of the
    original camera resolution.
    """
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized


def remove_noise(image):
    """
    Remove sensor / compression noise using a fast Non-Local Means filter.

    This step cleans up JPEG artefacts and small speckles before blurring,
    which prevents noise from being mistaken for food texture.
    """
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21
    )
    return denoised


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """
    Smooth the image with a Gaussian blur.

    Blurring reduces high-frequency detail so that colour-based
    segmentation (next stage) produces cleaner, more contiguous regions
    instead of being thrown off by fine texture/noise.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigmaX=0)
    return blurred


def convert_to_hsv(image):
    """
    Convert a BGR image to the HSV color space.

    HSV separates colour (Hue) from brightness (Value), which makes it
    far more robust than RGB/BGR for segmenting food and plate regions
    under uneven lighting conditions.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv


def run_preprocessing_pipeline(image):
    """
    Run the full preprocessing pipeline and return every intermediate
    stage so the UI can visualise each step.

    Returns
    -------
    dict with keys: original, resized, denoised, blurred, hsv
    """
    stages = {}
    stages["original"] = image
    stages["resized"] = resize_image(image)
    stages["denoised"] = remove_noise(stages["resized"])
    stages["blurred"] = apply_gaussian_blur(stages["denoised"])
    stages["hsv"] = convert_to_hsv(stages["blurred"])
    return stages
