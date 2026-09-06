"""
train_model.py
Trains a supervised classifier (Random Forest) for attack detection and an
unsupervised Isolation Forest for anomaly detection. Also computes SHAP
values for explainability.
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def prepare_features(df, target="Label_binary", drop_cols=None):
    drop_cols = drop_cols or []
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in [target] + drop_cols
    ]
    X = df[feature_cols]
    y = df[target]
    return X, y, feature_cols


def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))

    joblib.dump(clf, os.path.join(MODEL_DIR, "rf_classifier.pkl"))
    return clf, X_test, y_test


def train_anomaly_detector(X):
    iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(X)
    joblib.dump(iso, os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    return iso


def explain_model(clf, X_sample):
    """Compute SHAP values for a sample of the data."""
    import shap

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


if __name__ == "__main__":
    from data_loader import load_data, clean_data, add_binary_label

    df = load_data()
    df = clean_data(df)
    df = add_binary_label(df)

    X, y, feature_cols = prepare_features(df)
    clf, X_test, y_test = train_classifier(X, y)
    iso = train_anomaly_detector(X)

    print("Models saved to /models")
