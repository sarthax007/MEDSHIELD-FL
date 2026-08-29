import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import uuid
from datetime import datetime, timezone

from app.main import app
from app.db.base import Base
from app.db.session import get_db
from app.db.models import User, Hospital, TrainingRound
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
    
    admin_user = User(
        username="admin_round",
        hashed_password=get_password_hash("testpass"),
        role="admin",
        hospital_id=hospital_id
    )
    doctor_user = User(
        username="doctor_round",
        hashed_password=get_password_hash("testpass"),
        role="doctor",
        hospital_id=hospital_id
    )
    operator_user = User(
        username="operator_round",
        hashed_password=get_password_hash("testpass"),
        role="operator",
        hospital_id=hospital_id
    )
    db.add_all([admin_user, doctor_user, operator_user])
    
    t_round = TrainingRound(round_number=1, status="completed", global_accuracy=0.95, start_time=datetime.now(timezone.utc))
    db.add(t_round)
    
    db.commit()
    
    yield
    
    Base.metadata.drop_all(bind=engine)

def get_token(username):
    response = client.post("/auth/login", data={"username": username, "password": "testpass"})
    return response.json()["access_token"]

def test_get_rounds(setup_db):
    token = get_token("doctor_round")
    response = client.get("/rounds/", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["global_accuracy"] == 0.95
    assert "participants" in data[0]

def test_trigger_round_admin(setup_db):
    token = get_token("admin_round")
    response = client.post(
        "/rounds/", 
        json={"round_number": 2, "status": "started"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["round_number"] == 2
    assert "participants" in data

def test_trigger_round_operator_forbidden(setup_db):
    token = get_token("operator_round")
    response = client.post(
        "/rounds/", 
        json={"round_number": 3, "status": "started"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403
