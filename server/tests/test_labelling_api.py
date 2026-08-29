import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.db.session import get_db
from app.db import models
from app.core.security import create_access_token

# Setup test DB
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
    hospital = models.Hospital(name="Labelling Hospital", location="City X")
    db.add(hospital)
    db.commit()

    # Create Doctor
    doctor = models.User(
        hospital_id=hospital.id,
        username="doctor_label",
        hashed_password="hashedpassword",
        role="doctor",
    )
    db.add(doctor)

    # Create Operator
    operator = models.User(
        hospital_id=hospital.id,
        username="operator_label",
        hashed_password="hashedpassword",
        role="operator",
    )
    db.add(operator)
    db.commit()

    # Create Unlabelled Images
    img1 = models.ImageMetadata(
        hospital_id=hospital.id,
        local_image_ref="data/uploads/1.png",
        modality="MRI",
        status="Uploaded",
    )
    img2 = models.ImageMetadata(
        hospital_id=hospital.id,
        local_image_ref="data/uploads/2.png",
        modality="MRI",
        status="Uploaded",
    )
    db.add(img1)
    db.add(img2)
    db.commit()

    yield {
        "doctor_id": doctor.id,
        "operator_id": operator.id,
        "hospital_id": hospital.id,
        "img1_id": img1.id,
        "img2_id": img2.id,
    }

    db.close()


def get_token(username: str, role: str):
    return create_access_token(subject=username)


def test_labelling_queue_unauthorized(setup_db):
    token = get_token("operator_label", "operator")
    response = client.get(
        "/labelling/queue", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403


def test_labelling_queue_success(setup_db):
    token = get_token("doctor_label", "doctor")
    response = client.get(
        "/labelling/queue", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert "uncertainty_score" in data[0]


def test_submit_label_success(setup_db):
    token = get_token("doctor_label", "doctor")
    img_id = setup_db["img1_id"]

    response = client.post(
        f"/labelling/{img_id}",
        json={"class_label": "TUMOR"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # Verify DB updates
    db = TestingSessionLocal()
    img = (
        db.query(models.ImageMetadata).filter(models.ImageMetadata.id == img_id).first()
    )
    assert img.status == "Labelled"

    label = db.query(models.Label).filter(models.Label.image_id == img_id).first()
    assert label is not None
    assert label.class_label == "TUMOR"

    audit = (
        db.query(models.AuditLog).filter(models.AuditLog.resource_id == img_id).first()
    )
    assert audit is not None
    assert audit.action == "submit_label"
    db.close()


def test_submit_label_invalid_class(setup_db):
    token = get_token("doctor_label", "doctor")
    img_id = setup_db["img2_id"]

    response = client.post(
        f"/labelling/{img_id}",
        json={"class_label": "UNKNOWN_CLASS"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400
    assert "Unmapped or null label encountered" in response.json()["error"]
