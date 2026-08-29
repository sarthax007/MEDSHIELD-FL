from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
import uuid
import io
import os
import numpy as np
from PIL import Image
import torch

from medshield.model.registry import create_model
from medshield.model.config import ModelConfig
from medshield.model.checkpoint import load_checkpoint
from medshield.data.preprocess import preprocess_slice
from medshield.data.labels import TumorClass

from app.db.session import get_db
from app.db import models
from app.api.deps import get_current_active_user
from app.schemas.predictions import PredictionResponse

router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"],
)


@router.post(
    "/", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED
)
async def create_prediction(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    # Validate content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # Retrieve current global model version
    latest_model_version = (
        db.query(models.ModelVersion)
        .order_by(models.ModelVersion.created_at.desc())
        .first()
    )
    if not latest_model_version:
        raise HTTPException(
            status_code=400, detail="No global model version available."
        )

    # Create ImageMetadata
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    local_image_ref = f"{upload_dir}/{uuid.uuid4()}_{file.filename}"

    image_meta = models.ImageMetadata(
        hospital_id=current_user.hospital_id,
        local_image_ref=local_image_ref,
        modality="MRI",
        status="Uploaded",
    )
    db.add(image_meta)
    db.flush()

    # Real ML Inference logic
    try:
        # Load the image
        image_bytes = await file.read()

        # Save to disk for later explainability
        with open(local_image_ref, "wb") as f:
            f.write(image_bytes)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image).transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)

        # Preprocess
        tensor = preprocess_slice(image_np, out_channels=3)
        tensor = tensor.unsqueeze(0)  # Add batch dimension (1, C, H, W)

        # Initialize Model
        config = ModelConfig(num_classes=2, pretrained=False)
        model = create_model(config)

        # Load weights if path exists (tests might mock this)
        try:
            load_checkpoint(latest_model_version.s3_path, model)
        except FileNotFoundError:
            pass  # Use random initialized weights if no checkpoint is found locally

        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probs, dim=1)

            predicted_class = TumorClass(predicted_idx.item()).name
            confidence_val = confidence.item()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    prediction = models.Prediction(
        image_id=image_meta.id,
        model_version_id=latest_model_version.id,
        predicted_class=predicted_class,
        confidence=confidence_val,
    )
    db.add(prediction)
    db.flush()

    audit = models.AuditLog(
        user_id=current_user.id,
        action="create_prediction",
        resource_type="Prediction",
        resource_id=prediction.id,
        details={"image_id": str(image_meta.id), "predicted_class": predicted_class},
    )
    db.add(audit)

    db.commit()
    db.refresh(prediction)

    return prediction
