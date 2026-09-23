import os
import sys
from typing import Optional

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "server"))
)
from app.db.session import SessionLocal
from app.db import models


def save_metrics(round_number: int, accuracy: float, loss: float, clients: list):
    session = SessionLocal()
    try:
        r = (
            session.query(models.TrainingRound)
            .filter_by(round_number=round_number)
            .first()
        )
        if r:
            r.global_accuracy = accuracy
            r.status = "Completed"
        else:
            r = models.TrainingRound(
                round_number=round_number, global_accuracy=accuracy, status="Completed"
            )
            session.add(r)
        session.commit()
    finally:
        session.close()


def save_global_model(round_number: int, model_weights: bytes):
    session = SessionLocal()
    try:
        r = (
            session.query(models.TrainingRound)
            .filter_by(round_number=round_number)
            .first()
        )
        if not r:
            r = models.TrainingRound(round_number=round_number, status="Completed")
            session.add(r)
            session.commit()
            session.refresh(r)

        # We save weights to disk and path to DB because ModelVersion expects s3_path
        os.makedirs("data/models", exist_ok=True)
        path = f"data/models/round_{round_number}.bin"
        with open(path, "wb") as f:
            f.write(model_weights)

        mv = models.ModelVersion(
            version_tag=f"v{round_number}.0", training_round_id=r.id, s3_path=path
        )
        session.add(mv)
        session.commit()
    finally:
        session.close()


def load_global_model(round_number: int) -> Optional[bytes]:
    session = SessionLocal()
    try:
        r = (
            session.query(models.TrainingRound)
            .filter_by(round_number=round_number)
            .first()
        )
        if not r:
            return None
        mv = (
            session.query(models.ModelVersion).filter_by(training_round_id=r.id).first()
        )
        if not mv:
            return None
        if os.path.exists(mv.s3_path):
            with open(mv.s3_path, "rb") as f:
                return f.read()
        return None
    finally:
        session.close()
