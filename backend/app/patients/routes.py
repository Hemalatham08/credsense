from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Optional

from ..database import get_db
from .. import models, schemas

router = APIRouter(prefix="/api/patients", tags=["Patients"])


def generate_patient_code(db: Session) -> str:
    """Generates PT-0001, PT-0002, ... based on current row count."""
    count = db.query(models.Patient).count()
    return f"PT-{count + 1:04d}"


@router.post("/", response_model=schemas.PatientResponse)
def register_patient(patient: schemas.PatientCreate, db: Session = Depends(get_db)):
    new_patient = models.Patient(
        patient_code=generate_patient_code(db),
        name=patient.name,
        gender=patient.gender,
        date_of_birth=patient.date_of_birth,
        phone=patient.phone,
        email=patient.email,
        address=patient.address,
    )
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    return new_patient


@router.get("/", response_model=List[schemas.PatientResponse])
def list_patients(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    return db.query(models.Patient).offset(skip).limit(limit).all()


@router.get("/search", response_model=List[schemas.PatientResponse])
def search_patients(q: str, db: Session = Depends(get_db)):
    """Search by name or patient_code (partial match, case-insensitive)."""
    results = db.query(models.Patient).filter(
        or_(
            models.Patient.name.ilike(f"%{q}%"),
            models.Patient.patient_code.ilike(f"%{q}%"),
        )
    ).all()
    return results


@router.get("/{patient_id}", response_model=schemas.PatientResponse)
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient