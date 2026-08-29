import base64
import json
import os
import uuid

import cv2
import nibabel as nib
import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from medshield.explain.explanation_text import (
    determine_heatmap_region,
    generate_clinical_explanation,
)
from medshield.explain.vit_cam import (
    ViTGradCAM,
    reshape_transform_vit_timm,
    show_cam_on_image,
)
from medshield.model.registry import create_model
from medshield.model.config import ModelConfig
from medshield.model.checkpoint import load_checkpoint

from app.db.session import get_db
from app.db import models
from app.api.deps import get_current_active_user

router = APIRouter(
    prefix="/explain",
    tags=["explainability"],
)

# Storage location for explanation artifacts.
# Retention policy: Indefinitely retained until database storage is implemented in Task 77.
CACHE_DIR = "data/explanations"
os.makedirs(CACHE_DIR, exist_ok=True)


@router.get("/{prediction_id}")
async def get_explanation(
    prediction_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user),
):
    """
    Generate or retrieve an explanation for a stored prediction.
    """
    # 1. Look up the prediction
    prediction = (
        db.query(models.Prediction)
        .filter(models.Prediction.id == prediction_id)
        .first()
    )
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    # 2. Look up the image metadata
    image_meta = prediction.image
    if not image_meta:
        raise HTTPException(status_code=404, detail="Image metadata not found.")

    # Check cache first
    cache_file = os.path.join(CACHE_DIR, f"{prediction_id}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached_data = json.load(f)
        return JSONResponse(content=cached_data)

    # 3. Load the image from disk
    img_path = image_meta.local_image_ref
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image file not found on disk.")

    try:
        if img_path.endswith(".nii.gz") or img_path.endswith(".nii"):
            nii_img = nib.load(img_path)
            if hasattr(nii_img, "get_fdata"):
                img_data = nii_img.get_fdata()
            elif hasattr(nii_img, "dataobj"):
                img_data = np.asanyarray(nii_img.dataobj)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is not a valid spatial image.",
                )
            slice_idx = img_data.shape[2] // 2
            img_2d = img_data[:, :, slice_idx]
        else:
            img_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_cv is None:
                raise HTTPException(
                    status_code=400, detail="Could not read the image file."
                )
            img_2d = img_cv

        # Preprocess for CAM (which needs the image to be 224x224 and normalized in 3 channels)
        img_2d = cv2.resize(img_2d, (224, 224))
        img_2d_norm = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8)
        img_3d = np.stack([img_2d_norm] * 3, axis=0)
        input_tensor = torch.tensor(img_3d, dtype=torch.float32).unsqueeze(0)

        # 4. Initialize model
        config = ModelConfig(num_classes=2, pretrained=False)
        clf_model = create_model(config)

        # Load weights if available
        model_version = prediction.model_version
        if model_version:
            try:
                load_checkpoint(model_version.s3_path, clf_model)
            except FileNotFoundError:
                pass  # Use initialized weights

        clf_model.eval()

        # 5. Extract CAM
        import typing

        blocks = typing.cast(torch.nn.Sequential, clf_model.backbone.blocks)
        target_layer = blocks[-1].norm1

        cam_extractor = ViTGradCAM(
            model=clf_model,
            target_layer=target_layer,
            reshape_transform=reshape_transform_vit_timm,
        )

        cam = cam_extractor(input_tensor)

        # 6. Generate explanation
        predicted_class = prediction.predicted_class
        confidence_val = prediction.confidence

        region = determine_heatmap_region(cam)
        explanation = generate_clinical_explanation(
            predicted_class, confidence_val, region
        )

        # 7. Generate Heatmap image base64
        vis_img = np.stack([img_2d_norm] * 3, axis=-1)
        cam_vis = show_cam_on_image(vis_img, cam, use_rgb=True)
        cam_vis_bgr = cv2.cvtColor(cam_vis, cv2.COLOR_RGB2BGR)

        _, buffer = cv2.imencode(".jpg", cam_vis_bgr)
        heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

        response_data = {
            "prediction_id": str(prediction_id),
            "prediction": predicted_class,
            "confidence": confidence_val,
            "explanation": explanation,
            "heatmap_base64": heatmap_base64,
        }

        # Save to cache
        with open(cache_file, "w") as f:
            json.dump(response_data, f)

        return JSONResponse(content=response_data)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Explanation generation failed: {str(e)}"
        )
