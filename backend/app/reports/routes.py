"""
CardioSense - Report API Routes
==================================
GET /reports/{assessment_id} - fetches everything tied to one assessment
(patient, vitals, lab results, lifestyle, prediction, recommendations) and
streams back a generated PDF report.

Read-only against existing tables — no new columns, no schema changes.
"""

import io

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, joinedload

from ..database import get_db
from ..models import Assessment,Prediction
from .pdf_builder import build_report_pdf

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/{assessment_id}")
def get_report(assessment_id: int, db: Session = Depends(get_db)):
    assessment = (
        db.query(Assessment)
        .options(
            joinedload(Assessment.patient),
            joinedload(Assessment.lab_result),
            joinedload(Assessment.lifestyle),
            joinedload(Assessment.prediction).joinedload(Prediction.recommendations),
        )
        .filter(Assessment.id == assessment_id)
        .first()
    )

    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    if not assessment.prediction:
        raise HTTPException(
            status_code=400,
            detail="No prediction exists for this assessment yet — run /predictions/ first",
        )

    patient = assessment.patient
    prediction = assessment.prediction

    pdf_buf = build_report_pdf(
        patient={
            "name": patient.name,
            "patient_code": patient.patient_code,
            "gender": patient.gender,
            "date_of_birth": patient.date_of_birth,
        },
        assessment={
            "id": assessment.id,
            "assessment_date": assessment.assessment_date,
            "height": assessment.height,
            "weight": assessment.weight,
            "ap_hi": assessment.ap_hi,
            "ap_lo": assessment.ap_lo,
            "bmi": assessment.bmi,
            "pulse_pressure": assessment.pulse_pressure,
            "map": assessment.map,
            "bp_category": assessment.bp_category,
        },
        lab_result={
            "cholesterol": assessment.lab_result.cholesterol if assessment.lab_result else None,
            "gluc": assessment.lab_result.gluc if assessment.lab_result else None,
        },
        lifestyle={
            "smoke": assessment.lifestyle.smoke if assessment.lifestyle else None,
            "alco": assessment.lifestyle.alco if assessment.lifestyle else None,
            "active": assessment.lifestyle.active if assessment.lifestyle else None,
        },
        prediction={
            "model_used": prediction.model_used,
            "risk_probability": float(prediction.risk_probability),
            "risk_level": prediction.risk_level,
            "confidence_score": float(prediction.confidence_score) if prediction.confidence_score else None,
            "shap_values": prediction.shap_values,
        },
        recommendations=[
            {"recommendation_text": r.recommendation_text, "priority": r.priority}
            for r in prediction.recommendations
        ],
    )

    filename = f"cardiosense_report_{patient.patient_code}_{assessment.id}.pdf"
    return StreamingResponse(
        pdf_buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )