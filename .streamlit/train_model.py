"""
train_model.py
---------------
Trains the RandomForestClassifier used to classify plate waste into
Low / Moderate / High categories, and persists it with joblib.

Usage:
    python models/train_model.py
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import DATASET_DIR, MODEL_PATH, MODELS_DIR, ensure_directories
from utils.feature_extraction import FEATURE_COLUMNS
from models.dataset_generator import generate_synthetic_dataset

RANDOM_STATE = 42


def load_or_create_dataset():
    """
    Load dataset/training_data.csv if it exists; otherwise generate a
    fresh synthetic dataset automatically (satisfies the requirement to
    auto-generate data when a real dataset is unavailable).
    """
    csv_path = os.path.join(DATASET_DIR, "training_data.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return generate_synthetic_dataset(n_per_class=400, save_path=csv_path)


def train_and_save_model():
    ensure_directories()
    dataset = load_or_create_dataset()

    X = dataset[FEATURE_COLUMNS]
    y = dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model trained. Test accuracy: {accuracy:.4f}")
    print(report)
    print(f"Model saved to: {MODEL_PATH}")

    return model, accuracy


if __name__ == "__main__":
    train_and_save_model()
