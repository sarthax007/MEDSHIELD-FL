import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import uuid

from app.main import app
from app.db.base import Base
from app.db.session import get_db
from app.db.models import User, Hospital, TrainingRound, ModelVersion
from app.core.security import get_password_hash
from sqlalchemy.pool import StaticPool

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

@pytest.fixture(scope="module")
def setup_db():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    
    hospital_id = uuid.uuid4()
    hospital = Hospital(id=hospital_id, name="Test Hospital", location="Test City")
    db.add(hospital)
    
    doctor_user = User(
        username="doctor_predict",
        hashed_password=get_password_hash("testpass"),
        role="doctor",
        hospital_id=hospital_id
    )
    db.add(doctor_user)

    t_round = TrainingRound(round_number=1, status="completed")
    db.add(t_round)
    db.commit()

    model_v = ModelVersion(
        version_tag="v1.0",
        training_round_id=t_round.id,
        s3_path="s3://test/model.pth"
    )
    db.add(model_v)
    
    db.commit()
    
    yield
    
    Base.metadata.drop_all(bind=engine)

def get_token(username):
    response = client.post("/auth/login", data={"username": username, "password": "testpass"})
    return response.json()["access_token"]

def test_prediction_valid_image(setup_db):
    token = get_token("doctor_predict")
    
    # Create a real, minimal valid PNG image in memory for PIL to open
    from PIL import Image
    import io
    image = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    file_content = img_byte_arr.getvalue()
    
    files = {"file": ("test.png", file_content, "image/png")}
    
    response = client.post(
        "/predictions/", 
        headers={"Authorization": f"Bearer {token}"},
        files=files
    )
    assert response.status_code == 201
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "model_version_id" in data
    assert data["predicted_class"] in ["NO_TUMOR", "TUMOR"]

def test_prediction_invalid_file(setup_db):
    token = get_token("doctor_predict")
    
    file_content = b"fake text content"
    files = {"file": ("test.txt", file_content, "text/plain")}
    
    response = client.post(
        "/predictions/", 
        headers={"Authorization": f"Bearer {token}"},
        files=files
    )
    assert response.status_code == 400
    assert response.json()["error"] == "Uploaded file must be an image."
