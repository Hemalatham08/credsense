from sqlalchemy import (
    Column, Integer, String, SmallInteger, Numeric, Text, Date,
    TIMESTAMP, ForeignKey
)
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from .database import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_code = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    gender = Column(SmallInteger, nullable=False)  # 0=female, 1=male
    date_of_birth = Column(Date, nullable=False)
    phone = Column(String(20))
    email = Column(String(100))
    address = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())

    assessments = relationship("Assessment", back_populates="patient", cascade="all, delete")


class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    assessment_date = Column(TIMESTAMP, server_default=func.now())

    height = Column(Numeric(5, 2), nullable=False)
    weight = Column(Numeric(5, 2), nullable=False)
    ap_hi = Column(SmallInteger, nullable=False)
    ap_lo = Column(SmallInteger, nullable=False)

    bmi = Column(Numeric(5, 2))
    pulse_pressure = Column(SmallInteger)
    map = Column(Numeric(5, 2))
    bp_category = Column(SmallInteger)

    created_at = Column(TIMESTAMP, server_default=func.now())

    patient = relationship("Patient", back_populates="assessments")
    lab_result = relationship("LabResult", back_populates="assessment", uselist=False, cascade="all, delete")
    lifestyle = relationship("LifestyleFactor", back_populates="assessment", uselist=False, cascade="all, delete")
    prediction = relationship("Prediction", back_populates="assessment", uselist=False, cascade="all, delete")


class LabResult(Base):
    __tablename__ = "lab_results"

    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("assessments.id", ondelete="CASCADE"), nullable=False)
    cholesterol = Column(SmallInteger, nullable=False)  # 1/2/3
    gluc = Column(SmallInteger, nullable=False)          # 1/2/3
    created_at = Column(TIMESTAMP, server_default=func.now())

    assessment = relationship("Assessment", back_populates="lab_result")


class LifestyleFactor(Base):
    __tablename__ = "lifestyle_factors"

    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("assessments.id", ondelete="CASCADE"), nullable=False)
    smoke = Column(SmallInteger, nullable=False)
    alco = Column(SmallInteger, nullable=False)
    active = Column(SmallInteger, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())

    assessment = relationship("Assessment", back_populates="lifestyle")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("assessments.id", ondelete="CASCADE"), nullable=False)
    model_used = Column(String(50), default="voting_ensemble")
    risk_probability = Column(Numeric(5, 4), nullable=False)
    risk_level = Column(String(20), nullable=False)
    confidence_score = Column(Numeric(5, 4))
    shap_values = Column(JSONB)
    created_at = Column(TIMESTAMP, server_default=func.now())

    assessment = relationship("Assessment", back_populates="prediction")
    recommendations = relationship("Recommendation", back_populates="prediction", cascade="all, delete")


class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False)
    recommendation_text = Column(Text, nullable=False)
    priority = Column(String(10), default="medium")
    created_at = Column(TIMESTAMP, server_default=func.now())

    prediction = relationship("Prediction", back_populates="recommendations")