import pytest
from fastapi.testclient import TestClient
import uuid
import os
import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.db.session import get_db
from app.db import models
from app.core.security import create_access_token

# Use the same setup as other tests
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


client = TestClient(app)


@pytest.fixture(scope="module")
def setup_db():
    from app.db.base import Base

    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    db = TestingSessionLocal()
    # Create Hospital
    hospital = models.Hospital(name="Explain Hospital", location="City X")
    db.add(hospital)
    db.commit()

    # Create User
    user = models.User(
        hospital_id=hospital.id,
        username="doctor_explain",
        hashed_password="hashedpassword",
        role="doctor",
    )
    db.add(user)
    db.commit()

    # Create Image file on disk
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    local_image_ref = f"{upload_dir}/test_explain_image.png"

    # Create a simple valid image (grayscale or RGB, cv2 reads as grayscale anyway)
    img = np.zeros((100, 100), dtype=np.uint8)
    img[25:75, 25:75] = 255
    cv2.imwrite(local_image_ref, img)

    # Create ImageMeta
    image_meta = models.ImageMetadata(
        hospital_id=hospital.id,
        local_image_ref=local_image_ref,
        modality="MRI",
        status="Uploaded",
    )
    db.add(image_meta)
    db.commit()

    # Create Model Version
    mv = models.ModelVersion(
        version_tag="v1.0", training_round_id=1, s3_path="dummy_path"
    )
    db.add(mv)
    db.commit()

    # Create Prediction
    pred = models.Prediction(
        image_id=image_meta.id,
        model_version_id=mv.id,
        predicted_class="TUMOR",
        confidence=0.95,
    )
    db.add(pred)
    db.commit()

    yield {"user_id": user.id, "prediction_id": pred.id, "image_path": local_image_ref}

    # Cleanup
    db.close()
    if os.path.exists(local_image_ref):
        os.remove(local_image_ref)


def get_token(username: str = "doctor_explain", role: str = "doctor"):
    return create_access_token(subject=username)


def test_explain_valid_prediction(setup_db):
    token = get_token()
    pred_id = setup_db["prediction_id"]

    response = client.get(
        f"/explain/{pred_id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_id"] == str(pred_id)
    assert data["prediction"] == "TUMOR"
    assert "confidence" in data
    assert "explanation" in data
    assert "heatmap_base64" in data


def test_explain_invalid_prediction(setup_db):
    token = get_token()
    invalid_id = str(uuid.uuid4())

    response = client.get(
        f"/explain/{invalid_id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 404
    assert "Prediction not found" in response.json()["error"]
