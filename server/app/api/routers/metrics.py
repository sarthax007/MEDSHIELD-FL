from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from app.db.session import get_db
from app.db import models
from app.api.deps import get_current_active_user
from app.schemas.metrics import (
    AccuracyTrendItem,
    HospitalParticipationItem,
    CurrentRoundInfo,
)

router = APIRouter(
    prefix="/metrics",
    tags=["Metrics"],
)


@router.get("/accuracy-trend", response_model=List[AccuracyTrendItem])
def get_accuracy_trend(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    rounds = (
        db.query(
            models.TrainingRound.round_number, models.TrainingRound.global_accuracy
        )
        .filter(models.TrainingRound.global_accuracy.isnot(None))
        .order_by(models.TrainingRound.round_number)
        .all()
    )
    return [
        {"round_number": r.round_number, "global_accuracy": r.global_accuracy}
        for r in rounds
    ]


@router.get("/current-round", response_model=CurrentRoundInfo)
def get_current_round(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    latest_round = (
        db.query(models.TrainingRound)
        .order_by(models.TrainingRound.round_number.desc())
        .first()
    )
    if not latest_round:
        return {"round_number": 0, "status": "No rounds yet"}
    return {"round_number": latest_round.round_number, "status": latest_round.status}


@router.get("/hospital-participation", response_model=List[HospitalParticipationItem])
def get_hospital_participation(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    # Group by hospital and count images
    results = (
        db.query(
            models.Hospital.name,
            func.count(models.ImageMetadata.id).label("contribution_count"),
        )
        .outerjoin(
            models.ImageMetadata, models.Hospital.id == models.ImageMetadata.hospital_id
        )
        .group_by(models.Hospital.id, models.Hospital.name)
        .all()
    )

    return [
        {"hospital_name": r.name, "contribution_count": r.contribution_count}
        for r in results
    ]
