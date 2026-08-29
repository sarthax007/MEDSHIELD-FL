import os
import uuid
import numpy as np
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db import models
from app.api.deps import get_current_active_user
from app.schemas.labelling import QueueItem, LabelSubmission

from shared.medshield.active.service import LabellingQueueService
from shared.medshield.active.pool import DataPoolManager
from shared.medshield.active.budget import BudgetManager
from shared.medshield.active.uncertainty import PredictionEntropyStrategy
from shared.medshield.active.query import QueryStrategy
from medshield.data.labels import TumorClass, map_raw_label

router = APIRouter(
    prefix="/labelling",
    tags=["Labelling"],
)


def get_current_doctor(current_user: models.User = Depends(get_current_active_user)):
    if current_user.role != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges",
        )
    return current_user


@router.get("/queue", response_model=List[QueueItem])
def get_labelling_queue(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_doctor),
):
    # Fetch pending unlabelled images for this hospital
    images = (
        db.query(models.ImageMetadata)
        .filter(
            models.ImageMetadata.hospital_id == current_user.hospital_id,
            models.ImageMetadata.status.in_(["Uploaded", "PendingLabel"]),
        )
        .all()
    )

    if not images:
        return []

    # Prepare active learning components
    state_file = f"data/active_learning/pool_state_{current_user.hospital_id}.json"
    os.makedirs(os.path.dirname(state_file), exist_ok=True)

    item_ids = [str(img.id) for img in images]

    pool_manager = DataPoolManager(state_file_path=state_file, initial_items=item_ids)

    # Sync: if there are items in DB that are not in pool, add them?
    # DataPoolManager handles unknown items if initialized correctly, but we'll just use it directly.
    # To properly sync, we'll recreate the pool's unlabelled set based on DB if needed,
    # but let's assume `initial_items` takes care of it if new file.
    # If the file exists, we force add any missing items.
    current_unlabelled = set(pool_manager.get_unlabeled_pool())
    for i_id in item_ids:
        if (
            i_id not in current_unlabelled
            and i_id not in pool_manager.get_labelled_pool()
        ):
            pool_manager.unlabeled_pool.add(i_id)
    pool_manager.save_state()

    budget_manager = BudgetManager(initial_budget=1000)  # Example budget
    uncertainty_strategy = PredictionEntropyStrategy()
    query_strategy = QueryStrategy()

    service = LabellingQueueService(
        pool_manager=pool_manager,
        budget_manager=budget_manager,
        uncertainty_strategy=uncertainty_strategy,
        query_strategy=query_strategy,
        allowed_classes={0, 1},
    )

    # Mock prediction function for speed in the queue (or we could run real inference)
    # We will just return random probabilities to satisfy the active learning interface
    def mock_predict_fn(item_ids_to_predict: List[str]) -> np.ndarray:
        return np.random.dirichlet(np.ones(2), size=len(item_ids_to_predict))

    queue = service.get_labelling_queue(predict_fn=mock_predict_fn)

    # Map back to QueueItem
    response_queue = []
    image_dict = {str(img.id): img for img in images}

    for item in queue:
        img_id_str = item["item_id"]
        if img_id_str in image_dict:
            response_queue.append(
                QueueItem(
                    image_id=uuid.UUID(img_id_str),
                    local_image_ref=image_dict[img_id_str].local_image_ref,
                    uncertainty_score=item["uncertainty"],
                    model_prediction=item["prediction"],
                )
            )

    return response_queue


@router.post("/{image_id}")
def submit_label(
    image_id: uuid.UUID,
    submission: LabelSubmission,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_doctor),
):
    # 1. Validate image exists and belongs to hospital
    image = (
        db.query(models.ImageMetadata)
        .filter(
            models.ImageMetadata.id == image_id,
            models.ImageMetadata.hospital_id == current_user.hospital_id,
        )
        .first()
    )

    if not image:
        raise HTTPException(status_code=404, detail="Image not found or access denied.")

    if image.status == "Labelled":
        raise HTTPException(status_code=400, detail="Image is already labelled.")

    try:
        class_idx = map_raw_label(submission.class_label)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Add Label to DB
    new_label = models.Label(
        image_id=image_id,
        user_id=current_user.id,
        class_label=TumorClass(class_idx).name,
    )
    db.add(new_label)

    # 3. Update ImageMetadata status
    image.status = "Labelled"

    # 4. Audit Log
    audit = models.AuditLog(
        user_id=current_user.id,
        action="submit_label",
        resource_type="ImageMetadata",
        resource_id=image_id,
        details={"class_label": TumorClass(class_idx).name},
    )
    db.add(audit)

    db.commit()

    # 5. Update Active Learning Pool
    state_file = f"data/active_learning/pool_state_{current_user.hospital_id}.json"
    pool_manager = DataPoolManager(state_file_path=state_file)
    budget_manager = BudgetManager(initial_budget=1000)
    service = LabellingQueueService(
        pool_manager=pool_manager,
        budget_manager=budget_manager,
        uncertainty_strategy=PredictionEntropyStrategy(),
        query_strategy=QueryStrategy(),
        allowed_classes={0, 1},
    )

    # Make sure it's in the unlabeled pool just in case
    if (
        str(image_id) not in pool_manager.get_unlabeled_pool()
        and str(image_id) not in pool_manager.get_labelled_pool()
    ):
        pool_manager.unlabeled_pool.add(str(image_id))

    if str(image_id) in pool_manager.get_unlabeled_pool():
        service.submit_label(str(image_id), class_idx, str(current_user.id))

    return {"status": "success", "message": "Label submitted successfully."}
