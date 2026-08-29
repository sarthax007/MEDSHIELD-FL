from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db import models
from app.api.deps import get_current_active_user, get_current_admin_user
from app.schemas.hospitals import HospitalCreate, HospitalResponse

router = APIRouter(
    prefix="/hospitals",
    tags=["Hospitals"],
)

@router.get("/", response_model=List[HospitalResponse])
def get_hospitals(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    db_hospitals = db.query(models.Hospital).all()
    results = []
    for h in db_hospitals:
        # Determine participation status (e.g. if any users are registered for it)
        status = "Active" if h.users else "Pending"
        results.append({
            "id": h.id,
            "name": h.name,
            "location": h.location,
            "created_at": h.created_at,
            "participation_status": status
        })
    return results

@router.post("/", response_model=HospitalResponse, status_code=status.HTTP_201_CREATED)
def register_hospital(
    hospital: HospitalCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_admin_user)
):
    db_hospital = models.Hospital(name=hospital.name, location=hospital.location)
    db.add(db_hospital)
    db.commit()
    db.refresh(db_hospital)
    
    return {
        "id": db_hospital.id,
        "name": db_hospital.name,
        "location": db_hospital.location,
        "created_at": db_hospital.created_at,
        "participation_status": "Pending"
    }
