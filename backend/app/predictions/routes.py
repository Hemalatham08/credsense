"""
CardioSense - Prediction API Routes
=====================================
POST /predictions/ - runs one assessment through the ML pipeline and
persists it across all 5 related tables in a single transaction:
    assessments -> lab_results -> lifestyle_factors -> predictions -> recommendations

NOTE: adjust the relative imports below (`..database`, `..models`,
`..schemas`) to match wherever this file actually lands in your project
(you said it's already placed and registered — just confirm the import
paths line up).
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Patient, Assessment, LabResult, LifestyleFactor, Prediction, Recommendation
from ..schemas import AssessmentInput, PredictionResult, RecommendationOut
from .predict import predict_risk

router = APIRouter(prefix="/predictions", tags=["predictions"])


# Rule-based recommendation text, keyed to the features most likely to show
# up as top SHAP contributors. This is a starting point — refine wording
# and priority logic once you see real predictions come through.
RECOMMENDATION_MAP = {
    "bmi": "Weight management may reduce cardiovascular risk — consider a referral to nutrition counseling.",
    "ap_hi": "Elevated systolic blood pressure detected — recommend follow-up BP monitoring.",
    "ap_lo": "Elevated diastolic blood pressure detected — recommend follow-up BP monitoring.",
    "bp_category": "Blood pressure category indicates hypertension risk — clinical follow-up advised.",
    "cholesterol": "Cholesterol level is a significant risk factor — consider a lipid panel review.",
    "gluc": "Glucose level is contributing to risk — consider screening for prediabetes/diabetes.",
    "smoke": "Smoking cessation counseling is strongly recommended given its contribution to risk.",
    "active": "Increasing physical activity may help lower cardiovascular risk.",
    "alco": "Reducing alcohol consumption may help lower cardiovascular risk.",
    "age": "Age is a non-modifiable risk factor — reinforce importance of routine monitoring.",
    "pulse_pressure": "Elevated pulse pressure detected — recommend cardiovascular follow-up.",
    "map": "Mean arterial pressure is elevated — recommend follow-up.",
}


def generate_recommendations(shap_result: dict, risk_level: str, top_n: int = 3) -> list[dict]:
    top_features = shap_result["ranked"][:top_n]
    recs = []
    for i, item in enumerate(top_features):
        text = RECOMMENDATION_MAP.get(item["feature"])
        if not text:
            continue
        if risk_level == "high" and i == 0:
            priority = "high"
        elif item["contribution"] > 0:
            priority = "medium"
        else:
            priority = "low"
        recs.append({"recommendation_text": text, "priority": priority})
    return recs


@router.post("/", response_model=PredictionResult)
def create_prediction(payload: AssessmentInput, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == payload.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    result = predict_risk(
        date_of_birth=patient.date_of_birth,
        gender=patient.gender,
        height=payload.height,
        weight=payload.weight,
        ap_hi=payload.ap_hi,
        ap_lo=payload.ap_lo,
        cholesterol=payload.lab_result.cholesterol,
        gluc=payload.lab_result.gluc,
        smoke=payload.lifestyle.smoke,
        alco=payload.lifestyle.alco,
        active=payload.lifestyle.active,
    )

    # 1. assessments
    assessment = Assessment(
        patient_id=payload.patient_id,
        height=payload.height,
        weight=payload.weight,
        ap_hi=payload.ap_hi,
        ap_lo=payload.ap_lo,
        bmi=result["derived_vitals"]["bmi"],
        pulse_pressure=result["derived_vitals"]["pulse_pressure"],
        map=result["derived_vitals"]["map"],
        bp_category=result["derived_vitals"]["bp_category"],
    )
    db.add(assessment)
    db.flush()  # assigns assessment.id without committing yet

    # 2. lab_results
    db.add(LabResult(
        assessment_id=assessment.id,
        cholesterol=payload.lab_result.cholesterol,
        gluc=payload.lab_result.gluc,
    ))

    # 3. lifestyle_factors
    db.add(LifestyleFactor(
        assessment_id=assessment.id,
        smoke=payload.lifestyle.smoke,
        alco=payload.lifestyle.alco,
        active=payload.lifestyle.active,
    ))

    # 4. predictions
    prediction = Prediction(
        assessment_id=assessment.id,
        model_used="voting_ensemble",
        risk_probability=result["risk_probability"],
        risk_level=result["risk_level"],
        confidence_score=result["confidence_score"],
        shap_values=result["shap_values"],
    )
    db.add(prediction)
    db.flush()

    # 5. recommendations
    for rec in generate_recommendations(result["shap_values"], result["risk_level"]):
        db.add(Recommendation(prediction_id=prediction.id, **rec))

    db.commit()
    db.refresh(prediction)

    return PredictionResult(
        assessment_id=assessment.id,
        model_used=prediction.model_used,
        risk_probability=float(prediction.risk_probability),
        risk_level=prediction.risk_level,
        confidence_score=float(prediction.confidence_score) if prediction.confidence_score else None,
        shap_values=prediction.shap_values,
        recommendations=[
            RecommendationOut.model_validate(r) for r in prediction.recommendations
        ],
    )