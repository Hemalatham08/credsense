"""
CardioSense - Analytics API Routes
=====================================
Read-only aggregate endpoints for the dashboard: totals, risk distribution,
demographics, blood-pressure category breakdown, and assessment trends
over time. No writes, no schema changes.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas import (
    AnalyticsSummary, RiskDistributionItem, GenderDistributionItem,
    BpCategoryDistributionItem, TrendPoint,
)
from . import queries

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummary)
def summary(db: Session = Depends(get_db)):
    return queries.get_summary(db)


@router.get("/risk-distribution", response_model=list[RiskDistributionItem])
def risk_distribution(db: Session = Depends(get_db)):
    return queries.get_risk_distribution(db)


@router.get("/gender-distribution", response_model=list[GenderDistributionItem])
def gender_distribution(db: Session = Depends(get_db)):
    return queries.get_gender_distribution(db)


@router.get("/bp-category-distribution", response_model=list[BpCategoryDistributionItem])
def bp_category_distribution(db: Session = Depends(get_db)):
    return queries.get_bp_category_distribution(db)


@router.get("/trends", response_model=list[TrendPoint])
def trends(
    days: int = Query(30, ge=1, le=365, description="Number of past days to include"),
    db: Session = Depends(get_db),
):
    return queries.get_assessment_trend(db, days=days)