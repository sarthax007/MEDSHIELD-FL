import pytest
import uuid
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.base import Base
from app.db.session import get_db
from app.db.models import User, Hospital, TrainingRound, ImageMetadata
from app.core.security import get_password_hash

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
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()

    # Create test hospital
    hospital_id1 = uuid.uuid4()
    hospital1 = Hospital(id=hospital_id1, name="Hospital A", location="City A")

    hospital_id2 = uuid.uuid4()
    hospital2 = Hospital(id=hospital_id2, name="Hospital B", location="City B")

    db.add_all([hospital1, hospital2])

    # Create test user
    admin_user = User(
        username="admin_user_metrics",
        hashed_password=get_password_hash("testpass"),
        role="admin",
        hospital_id=hospital_id1,
    )
    db.add(admin_user)

    # Create training rounds
    round1 = TrainingRound(round_number=1, status="Completed", global_accuracy=0.85)
    round2 = TrainingRound(round_number=2, status="Completed", global_accuracy=0.88)
    round3 = TrainingRound(round_number=3, status="Training", global_accuracy=None)
    db.add_all([round1, round2, round3])

    # Create images
    img1 = ImageMetadata(
        hospital_id=hospital_id1,
        local_image_ref="a.png",
        modality="CT",
        status="Uploaded",
    )
    img2 = ImageMetadata(
        hospital_id=hospital_id1,
        local_image_ref="b.png",
        modality="CT",
        status="Uploaded",
    )
    img3 = ImageMetadata(
        hospital_id=hospital_id2,
        local_image_ref="c.png",
        modality="CT",
        status="Uploaded",
    )
    db.add_all([img1, img2, img3])

    db.commit()

    yield

    Base.metadata.drop_all(bind=engine)


def get_auth_token():
    response = client.post(
        "/auth/login",
        data={"username": "admin_user_metrics", "password": "testpass"},
    )
    return response.json()["access_token"]


def test_get_accuracy_trend(setup_db):
    token = get_auth_token()
    response = client.get(
        "/metrics/accuracy-trend",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["round_number"] == 1
    assert data[0]["global_accuracy"] == 0.85
    assert data[1]["round_number"] == 2
    assert data[1]["global_accuracy"] == 0.88


def test_get_current_round(setup_db):
    token = get_auth_token()
    response = client.get(
        "/metrics/current-round",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["round_number"] == 3
    assert data["status"] == "Training"


def test_get_hospital_participation(setup_db):
    token = get_auth_token()
    response = client.get(
        "/metrics/hospital-participation",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2

    participation_dict = {
        item["hospital_name"]: item["contribution_count"] for item in data
    }
    assert participation_dict["Hospital A"] == 2
    assert participation_dict["Hospital B"] == 1
