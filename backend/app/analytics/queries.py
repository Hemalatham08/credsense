"""
CardioSense - Analytics Queries
==================================
Pure aggregate/read queries against existing tables. No writes, no schema
changes — safe to build in parallel with any other phase.
"""

from datetime import datetime, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models import Patient, Assessment, Prediction


def get_summary(db: Session) -> dict:
    total_patients = db.query(func.count(Patient.id)).scalar() or 0
    total_assessments = db.query(func.count(Assessment.id)).scalar() or 0
    total_predictions = db.query(func.count(Prediction.id)).scalar() or 0
    avg_risk = db.query(func.avg(Prediction.risk_probability)).scalar()

    return {
        "total_patients": total_patients,
        "total_assessments": total_assessments,
        "total_predictions": total_predictions,
        "average_risk_probability": round(float(avg_risk), 4) if avg_risk is not None else None,
    }


def get_risk_distribution(db: Session) -> list[dict]:
    rows = (
        db.query(Prediction.risk_level, func.count(Prediction.id))
        .group_by(Prediction.risk_level)
        .all()
    )
    total = sum(count for _, count in rows) or 1
    return [
        {
            "risk_level": level,
            "count": count,
            "percentage": round(count / total * 100, 1),
        }
        for level, count in rows
    ]


def get_gender_distribution(db: Session) -> list[dict]:
    rows = (
        db.query(Patient.gender, func.count(Patient.id))
        .group_by(Patient.gender)
        .all()
    )
    labels = {0: "female", 1: "male"}
    total = sum(count for _, count in rows) or 1
    return [
        {
            "gender": labels.get(gender, str(gender)),
            "count": count,
            "percentage": round(count / total * 100, 1),
        }
        for gender, count in rows
    ]


def get_assessment_trend(db: Session, days: int = 30) -> list[dict]:
    """
    Daily assessment counts + average risk probability for the last N days.
    Uses func.date() which works on both Postgres and SQLite for grouping
    a TIMESTAMP column by calendar day.
    """
    since = datetime.utcnow() - timedelta(days=days)

    rows = (
        db.query(
            func.date(Assessment.assessment_date).label("day"),
            func.count(Assessment.id).label("assessment_count"),
            func.avg(Prediction.risk_probability).label("avg_risk"),
        )
        .outerjoin(Prediction, Prediction.assessment_id == Assessment.id)
        .filter(Assessment.assessment_date >= since)
        .group_by(func.date(Assessment.assessment_date))
        .order_by(func.date(Assessment.assessment_date))
        .all()
    )

    return [
        {
            "date": str(row.day),
            "assessment_count": row.assessment_count,
            "average_risk_probability": round(float(row.avg_risk), 4) if row.avg_risk is not None else None,
        }
        for row in rows
    ]


def get_bp_category_distribution(db: Session) -> list[dict]:
    labels = {0: "normal", 1: "elevated", 2: "hypertension_stage_1", 3: "hypertension_stage_2"}
    rows = (
        db.query(Assessment.bp_category, func.count(Assessment.id))
        .filter(Assessment.bp_category.isnot(None))
        .group_by(Assessment.bp_category)
        .all()
    )
    total = sum(count for _, count in rows) or 1
    return [
        {
            "bp_category": labels.get(cat, str(cat)),
            "count": count,
            "percentage": round(count / total * 100, 1),
        }
        for cat, count in rows
    ]