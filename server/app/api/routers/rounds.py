from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import uuid

from app.db.session import get_db
from app.db import models
from app.api.deps import get_current_active_user, get_current_admin_user
from app.schemas.rounds import TrainingRoundCreate, TrainingRoundResponse

router = APIRouter(
    prefix="/rounds",
    tags=["Training Rounds"],
)


@router.get("/", response_model=List[TrainingRoundResponse])
def get_rounds(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    db_rounds = db.query(models.TrainingRound).all()
    results = []
    for r in db_rounds:
        # Determine participants (e.g. hospitals that submitted models/images for this round)
        # For now, we can extract unique hospital names from model_versions or predictions
        # Or simply mock it if not fully implemented in db logic
        participants = []
        # If there's logic to fetch participants:
        # participants = list(set([mv.training_round.hospital.name for mv in r.model_versions]))

        results.append(
            {
                "id": r.id,
                "round_number": r.round_number,
                "status": r.status,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "global_accuracy": r.global_accuracy,
                "participants": participants,
            }
        )
    return results


@router.post(
    "/", response_model=TrainingRoundResponse, status_code=status.HTTP_201_CREATED
)
def trigger_round(
    round_in: TrainingRoundCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_admin_user),
):
    # Check if round number already exists
    existing = (
        db.query(models.TrainingRound)
        .filter(models.TrainingRound.round_number == round_in.round_number)
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Round number already exists")

    db_round = models.TrainingRound(
        round_number=round_in.round_number, status=round_in.status
    )
    db.add(db_round)
    db.flush()

    audit = models.AuditLog(
        user_id=current_user.id,
        action="trigger_round",
        resource_type="TrainingRound",
        resource_id=uuid.UUID(int=db_round.id),
        details={"round_number": db_round.round_number},
    )
    db.add(audit)

    db.commit()
    db.refresh(db_round)

    return {
        "id": db_round.id,
        "round_number": db_round.round_number,
        "status": db_round.status,
        "start_time": db_round.start_time,
        "end_time": db_round.end_time,
        "global_accuracy": db_round.global_accuracy,
        "participants": [],
    }
