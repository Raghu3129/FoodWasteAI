"""
predictor.py
------------
Loads the trained RandomForestClassifier and exposes a simple
`predict()` function used by the Streamlit app. If no trained model is
found on disk, it transparently triggers training (which will in turn
auto-generate a synthetic dataset if needed).
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import MODEL_PATH
from utils.feature_extraction import feature_vector_for_model, FEATURE_COLUMNS

_MODEL_CACHE = {"model": None}


def load_model():
    """Load the trained model from disk, training one first if missing."""
    if _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"]

    if not os.path.exists(MODEL_PATH):
        # Lazy import to avoid a circular import at module load time
        from models.train_model import train_and_save_model

        train_and_save_model()

    model = joblib.load(MODEL_PATH)
    _MODEL_CACHE["model"] = model
    return model


def predict_waste_category(features: dict):
    """
    Run inference on an extracted feature dict.

    Returns
    -------
    dict with keys: category, confidence, probabilities
    """
    model = load_model()
    vector = feature_vector_for_model(features)
    X = pd.DataFrame([vector], columns=FEATURE_COLUMNS)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    class_labels = model.classes_

    prob_dict = {label: float(round(p * 100, 2)) for label, p in zip(class_labels, probabilities)}
    confidence = float(round(max(probabilities) * 100, 2))

    return {
        "category": prediction,
        "confidence": confidence,
        "probabilities": prob_dict,
    }


def compute_sustainability_score(waste_percentage: float) -> int:
    """
    Convert a waste percentage into an intuitive 0-100 sustainability
    score, where 100 = zero waste and 0 = entirely wasted.

    A mild non-linear curve is used so that small amounts of waste don't
    disproportionately punish the score (matches example: 8% -> ~92,
    35% -> ~65, 70% -> ~30).
    """
    score = 100 - (waste_percentage * 1.0)
    score = int(np.clip(round(score), 0, 100))
    return score
