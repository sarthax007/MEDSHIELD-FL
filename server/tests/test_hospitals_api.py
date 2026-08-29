import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import uuid

from app.main import app
from app.db.base import Base
from app.db.session import get_db
from app.db.models import User, Hospital
from app.core.security import get_password_hash
from sqlalchemy.pool import StaticPool

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

    hospital_id = uuid.uuid4()
    hospital = Hospital(id=hospital_id, name="Test Hospital", location="Test City")
    db.add(hospital)

    admin_user = User(
        username="admin_hosp",
        hashed_password=get_password_hash("testpass"),
        role="admin",
        hospital_id=hospital_id,
    )
    doctor_user = User(
        username="doctor_hosp",
        hashed_password=get_password_hash("testpass"),
        role="doctor",
        hospital_id=hospital_id,
    )
    db.add_all([admin_user, doctor_user])
    db.commit()

    yield

    Base.metadata.drop_all(bind=engine)


def get_token(username):
    response = client.post(
        "/auth/login", data={"username": username, "password": "testpass"}
    )
    return response.json()["access_token"]


def test_get_hospitals(setup_db):
    token = get_token("doctor_hosp")
    response = client.get("/hospitals/", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert "participation_status" in data[0]


def test_register_hospital_admin(setup_db):
    token = get_token("admin_hosp")
    response = client.post(
        "/hospitals/",
        json={"name": "New Hospital", "location": "New City"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "New Hospital"
    assert "participation_status" in data


def test_register_hospital_doctor_forbidden(setup_db):
    token = get_token("doctor_hosp")
    response = client.post(
        "/hospitals/",
        json={"name": "Another Hospital", "location": "Another City"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403
