"""
CardioSense - Model Training Pipeline
======================================
Input: cardio_train.csv (Kaggle "Cardiovascular Disease" dataset)
Columns: id, age(days), gender(1/2), height(cm), weight(kg),
         ap_hi, ap_lo, cholesterol(1-3), gluc(1-3),
         smoke(0/1), alco(0/1), active(0/1), cardio(0/1 target)

Output (saved into ml/models/):
    model.pkl        -> trained VotingClassifier
    scaler.pkl        -> fitted StandardScaler
    feature_names.pkl -> ordered list of feature columns (needed at inference time)
    shap_explainer.pkl-> SHAP explainer for the ensemble
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

DATA_PATH = "data/cardio_train.csv"     # adjust path as needed
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
def load_data(path):
    # Kaggle version of this file is semicolon-separated
    df = pd.read_csv(path, sep=";")
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 2. CLEAN
# ---------------------------------------------------------------------------
def clean_data(df):
    df = df.drop(columns=["id"], errors="ignore")

    # age is stored in days -> convert to years
    df["age"] = (df["age"] / 365).astype(int)

    # Drop physiologically impossible blood pressure values
    df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 220)]
    df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 160)]
    # Diastolic should never exceed systolic
    df = df[df["ap_hi"] > df["ap_lo"]]

    # Drop unrealistic height/weight (data entry errors)
    df = df[(df["height"] >= 130) & (df["height"] <= 210)]
    df = df[(df["weight"] >= 35) & (df["weight"] <= 200)]

    df = df.reset_index(drop=True)
    print(f"After cleaning: {df.shape[0]} rows")
    return df


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(df):
    # BMI (explicit step in your architecture diagram)
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

    # Pulse pressure - clinically meaningful derived vital
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

    # Mean arterial pressure
    df["map"] = df["ap_lo"] + (df["pulse_pressure"] / 3)

    # BP category (encoded numerically so it's model-ready)
    def bp_category(row):
        if row["ap_hi"] < 120 and row["ap_lo"] < 80:
            return 0  # normal
        elif row["ap_hi"] < 130 and row["ap_lo"] < 80:
            return 1  # elevated
        elif row["ap_hi"] < 140 or row["ap_lo"] < 90:
            return 2  # hypertension stage 1
        else:
            return 3  # hypertension stage 2
    df["bp_category"] = df.apply(bp_category, axis=1)

    # gender in this dataset: 1=female, 2=male -> normalize to 0/1
    df["gender"] = df["gender"].map({1: 0, 2: 1})

    return df


# ---------------------------------------------------------------------------
# 4. TRAIN
# ---------------------------------------------------------------------------
def train(df):
    X = df.drop(columns=["cardio"])
    y = df["cardio"]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42, n_jobs=-1
    )
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        eval_metric="logloss", random_state=42, n_jobs=-1
    )
    lgbm = LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=42, n_jobs=-1, verbose=-1
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        voting="soft"  # averages predicted probabilities -> needed for risk %
    )

    print("Training ensemble...")
    ensemble.fit(X_train_scaled, y_train)

    # ---- Evaluate ----
    y_pred = ensemble.predict(X_test_scaled)
    y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Evaluation ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1       : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")

    return ensemble, scaler, feature_names, X_train_scaled


# ---------------------------------------------------------------------------
# 5. SHAP EXPLAINER
# ---------------------------------------------------------------------------
def build_shap_explainer(ensemble, X_train_scaled, feature_names):
    # KernelExplainer works for any model incl. VotingClassifier,
    # but is slow -> explain against a small background sample
    background = shap.sample(X_train_scaled, 100)
    explainer = shap.KernelExplainer(ensemble.predict_proba, background)
    return explainer


# ---------------------------------------------------------------------------
# 6. SAVE ARTIFACTS
# ---------------------------------------------------------------------------
def save_artifacts(ensemble, scaler, feature_names, explainer):
    joblib.dump(ensemble, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))
    joblib.dump(explainer, os.path.join(MODEL_DIR, "shap_explainer.pkl"))
    print(f"\nArtifacts saved to ./{MODEL_DIR}/")


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)

    ensemble, scaler, feature_names, X_train_scaled = train(df)
    explainer = build_shap_explainer(ensemble, X_train_scaled, feature_names)
    save_artifacts(ensemble, scaler, feature_names, explainer)

    print("\nFinal feature order (IMPORTANT - backend must send data in this order):")
    print(feature_names)