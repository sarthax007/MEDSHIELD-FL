import base64
import hashlib
import json
import os
import tempfile

import cv2
import nibabel as nib
import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from shared.medshield.explain.explanation_text import (
    determine_heatmap_region,
    generate_clinical_explanation,
)
from shared.medshield.explain.vit_cam import (
    ViTGradCAM,
    reshape_transform_vit_timm,
    show_cam_on_image,
)
from shared.medshield.model.vit import TumorClassifier

router = APIRouter(
    prefix="/explain",
    tags=["explainability"],
)

# Storage location for explanation artifacts.
# Retention policy: Indefinitely retained until database storage is implemented in Task 77.
CACHE_DIR = "data/explanations"
os.makedirs(CACHE_DIR, exist_ok=True)

# Global model instance for the endpoint
model = None


def get_model():
    global model
    if model is None:
        model = TumorClassifier(pretrained=False)
        model.eval()
    return model


@router.post("")
async def explain_prediction(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    ext = file.filename.lower()
    allowed_exts = (".nii.gz", ".nii", ".jpg", ".jpeg", ".png")
    if not any(ext.endswith(e) for e in allowed_exts):
        raise HTTPException(
            status_code=400, detail=f"Unsupported file type. Allowed: {allowed_exts}"
        )

    suffix = (
        ".nii.gz"
        if file.filename.lower().endswith(".nii.gz")
        else os.path.splitext(file.filename)[1]
    )

    content = await file.read()
    prediction_id = hashlib.sha256(content).hexdigest()

    cache_file = os.path.join(CACHE_DIR, f"{prediction_id}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached_data = json.load(f)
            # Ensure prediction_id is in the response just in case the cached version didn't have it
            cached_data["prediction_id"] = prediction_id
            return JSONResponse(content=cached_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if tmp_path.endswith(".nii.gz") or tmp_path.endswith(".nii"):
            nii_img = nib.load(tmp_path)
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
            img_cv = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
            if img_cv is None:
                raise HTTPException(
                    status_code=400, detail="Could not read the image file."
                )
            img_2d = img_cv

        img_2d = cv2.resize(img_2d, (224, 224))
        img_2d_norm = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8)
        img_3d = np.stack([img_2d_norm] * 3, axis=0)
        input_tensor = torch.tensor(img_3d, dtype=torch.float32).unsqueeze(0)

        clf_model = get_model()
        import typing

        blocks = typing.cast(torch.nn.Sequential, clf_model.backbone.blocks)
        target_layer = blocks[-1].norm1

        cam_extractor = ViTGradCAM(
            model=clf_model,
            target_layer=target_layer,
            reshape_transform=reshape_transform_vit_timm,
        )

        cam = cam_extractor(input_tensor)

        with torch.no_grad():
            logits = clf_model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)

        class_names = ["Healthy", "Tumor"]
        predicted_class = class_names[int(pred_idx.item())]
        confidence_val = conf.item()

        region = determine_heatmap_region(cam)
        explanation = generate_clinical_explanation(
            predicted_class, confidence_val, region
        )

        vis_img = np.stack([img_2d_norm] * 3, axis=-1)
        cam_vis = show_cam_on_image(vis_img, cam, use_rgb=True)
        cam_vis_bgr = cv2.cvtColor(cam_vis, cv2.COLOR_RGB2BGR)

        _, buffer = cv2.imencode(".jpg", cam_vis_bgr)
        heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

        response_data = {
            "prediction_id": prediction_id,
            "prediction": predicted_class,
            "confidence": confidence_val,
            "explanation": explanation,
            "heatmap_base64": heatmap_base64,
        }

        # Save to cache
        with open(cache_file, "w") as f:
            json.dump(response_data, f)

        return JSONResponse(content=response_data)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.get("/{prediction_id}")
async def get_explanation(prediction_id: str):
    """
    Retrieve a cached explanation artifact by its prediction ID.
    """
    cache_file = os.path.join(CACHE_DIR, f"{prediction_id}.json")
    if not os.path.exists(cache_file):
        raise HTTPException(status_code=404, detail="Explanation artifact not found.")

    with open(cache_file, "r") as f:
        cached_data = json.load(f)

    return JSONResponse(content=cached_data)
