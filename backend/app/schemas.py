from pydantic import BaseModel, EmailStr, Field
from datetime import date, datetime
from typing import Optional


class PatientCreate(BaseModel):
    name: str
    gender: int = Field(..., ge=0, le=1)  # 0=female, 1=male
    date_of_birth: date
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    address: Optional[str] = None


class PatientResponse(BaseModel):
    id: int
    patient_code: str
    name: str
    gender: int
    date_of_birth: date
    phone: Optional[str]
    email: Optional[str]
    address: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True  # lets Pydantic read straight from the SQLAlchemy model


# --- Phase 4: Prediction API ---

class LabResultInput(BaseModel):
    cholesterol: int = Field(..., ge=1, le=3)
    gluc: int = Field(..., ge=1, le=3)


class LifestyleInput(BaseModel):
    smoke: int = Field(..., ge=0, le=1)
    alco: int = Field(..., ge=0, le=1)
    active: int = Field(..., ge=0, le=1)


class AssessmentInput(BaseModel):
    patient_id: int
    height: float = Field(..., gt=0)   # cm
    weight: float = Field(..., gt=0)   # kg
    ap_hi: int = Field(..., gt=0)
    ap_lo: int = Field(..., gt=0)
    lab_result: LabResultInput
    lifestyle: LifestyleInput


class RecommendationOut(BaseModel):
    recommendation_text: str
    priority: str

    class Config:
        from_attributes = True


class PredictionResult(BaseModel):
    assessment_id: int
    model_used: str
    risk_probability: float
    risk_level: str
    confidence_score: Optional[float] = None
    shap_values: Optional[dict] = None
    recommendations: list[RecommendationOut] = []

    class Config:
        from_attributes = True