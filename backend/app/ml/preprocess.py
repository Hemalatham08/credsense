"""
CardioSense - Inference Preprocessing
======================================
Mirrors train_model.py EXACTLY: same feature order, same derived-feature
formulas, same categorical encoding. If this drifts from train_model.py,
predictions will be silently wrong (garbage in, garbage out) — no error
will be thrown, the model will just score confidently on nonsense.

Feature order below comes straight from train_model.py: after
clean_data() + engineer_features(), X = df.drop(columns=["cardio"])
keeps columns in this order:
    age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc,
    smoke, alco, active, bmi, pulse_pressure, map, bp_category
"""

from datetime import date
import numpy as np

FEATURE_ORDER = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active",
    "bmi", "pulse_pressure", "map", "bp_category",
]


def calculate_age(date_of_birth: date, on_date: date = None) -> int:
    """
    Whole years, matching train_model.py's `(age_days / 365).astype(int)`
    (a truncation, not a calendar-exact birthday calc — this replicates
    that same truncation behavior so ages line up with training data).
    """
    on_date = on_date or date.today()
    days = (on_date - date_of_birth).days
    return int(days / 365)


def bp_category(ap_hi: float, ap_lo: float) -> int:
    """Identical branching to train_model.py's bp_category()."""
    if ap_hi < 120 and ap_lo < 80:
        return 0  # normal
    elif ap_hi < 130 and ap_lo < 80:
        return 1  # elevated
    elif ap_hi < 140 or ap_lo < 90:
        return 2  # hypertension stage 1
    else:
        return 3  # hypertension stage 2


def engineer_features(
    *,
    date_of_birth: date,
    gender: int,       # 0=female, 1=male — matches Patient.gender AND
                        # train_model's remapped gender (1->0, 2->1), so
                        # NO conversion needed between DB and model input
    height: float,      # cm
    weight: float,       # kg
    ap_hi: int,
    ap_lo: int,
    cholesterol: int,     # 1/2/3
    gluc: int,              # 1/2/3
    smoke: int,              # 0/1
    alco: int,                # 0/1
    active: int,               # 0/1
    assessment_date: date = None,
) -> dict:
    """
    Builds the exact feature dict train_model.py would have produced for
    one row, PLUS the DB-storable derived vitals (bmi, pulse_pressure,
    map, bp_category) so routes.py can write them straight into
    the `assessments` table.
    """
    age = calculate_age(date_of_birth, assessment_date)

    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo
    map_value = ap_lo + (pulse_pressure / 3)
    category = bp_category(ap_hi, ap_lo)

    return {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "bmi": round(bmi, 2),
        "pulse_pressure": pulse_pressure,
        "map": round(map_value, 2),
        "bp_category": category,
    }


def to_feature_vector(features: dict) -> np.ndarray:
    """Order-locked 2D array, ready for scaler.transform()."""
    return np.array([[features[name] for name in FEATURE_ORDER]])