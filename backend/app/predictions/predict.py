"""
CardioSense - Prediction Service
==================================
Loads the trained ensemble + scaler + feature_names + SHAP explainer once
at import time, then exposes predict_risk() for routes.py to call per-request.
"""

import os
import joblib

from .preprocess import FEATURE_ORDER, engineer_features, to_feature_vector
from .shap_analysis import explain_prediction

# Adjust ML_ARTIFACT_DIR (env var) or the default below to match wherever
# train_model.py's MODEL_DIR ("models/") actually lives relative to your
# deployed backend. This is the "artifact path" item from your checklist —
# double check this points at the real folder before testing.
ARTIFACT_DIR = os.getenv(
    "ML_ARTIFACT_DIR",
    os.path.join(os.path.dirname(__file__), "..", "ml", "models"),
)

_model = joblib.load(os.path.join(ARTIFACT_DIR, "model.pkl"))
_scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
_feature_names = joblib.load(os.path.join(ARTIFACT_DIR, "feature_names.pkl"))
_explainer = joblib.load(os.path.join(ARTIFACT_DIR, "shap_explainer.pkl"))

# Sanity check — catches silent drift between train_model.py and
# preprocess.py before it ever reaches a patient's prediction.
assert list(_feature_names) == FEATURE_ORDER, (
    "Feature order mismatch between saved model and preprocess.py!\n"
    f"  model expects   : {list(_feature_names)}\n"
    f"  preprocess builds: {FEATURE_ORDER}\n"
    "Update FEATURE_ORDER in preprocess.py to match feature_names.pkl."
)

# --- Risk thresholds — placeholder defaults, confirm with your clinical
# reference / adjust as needed (this is checklist item "set final risk
# thresholds") ---
RISK_LOW_MAX = 0.30        # < 30%        -> low
RISK_MODERATE_MAX = 0.70   # 30% - 70%    -> moderate
                            # > 70%        -> high


def classify_risk(probability: float) -> str:
    if probability < RISK_LOW_MAX:
        return "low"
    elif probability < RISK_MODERATE_MAX:
        return "moderate"
    return "high"


def predict_risk(**patient_inputs) -> dict:
    """
    patient_inputs: date_of_birth, gender, height, weight, ap_hi, ap_lo,
                     cholesterol, gluc, smoke, alco, active

    Returns derived vitals, risk_probability, risk_level, confidence_score,
    and a formatted SHAP explanation — everything routes.py needs to persist
    across the 5 related tables in one go.
    """
    features = engineer_features(**patient_inputs)
    X = to_feature_vector(features)
    X_scaled = _scaler.transform(X)

    proba = _model.predict_proba(X_scaled)[0]
    risk_probability = float(proba[1])  # P(cardio=1)
    risk_level = classify_risk(risk_probability)

    # Confidence: distance from the 50/50 line, rescaled to 0-1
    # (0.5 probability -> 0 confidence, 0.0 or 1.0 -> full confidence)
    confidence_score = float(abs(risk_probability - 0.5) * 2)

    shap_result = explain_prediction(_explainer, X_scaled, FEATURE_ORDER)

    return {
        "derived_vitals": {
            "bmi": features["bmi"],
            "pulse_pressure": features["pulse_pressure"],
            "map": features["map"],
            "bp_category": features["bp_category"],
        },
        "risk_probability": risk_probability,
        "risk_level": risk_level,
        "confidence_score": confidence_score,
        "shap_values": shap_result,
    }