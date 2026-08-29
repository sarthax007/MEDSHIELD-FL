import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.db.base import Base
from app.db.session import get_db
from app.db.models import User, Hospital
from app.core.security import get_password_hash
import uuid

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
    
    # Create test hospital
    hospital_id = uuid.uuid4()
    hospital = Hospital(id=hospital_id, name="Test Hospital", location="Test City")
    db.add(hospital)
    
    # Create test users
    admin_user = User(
        username="admin_user",
        hashed_password=get_password_hash("testpass"),
        role="admin",
        hospital_id=hospital_id
    )
    doctor_user = User(
        username="doctor_user",
        hashed_password=get_password_hash("testpass"),
        role="doctor",
        hospital_id=hospital_id
    )
    db.add_all([admin_user, doctor_user])
    db.commit()
    
    yield
    
    Base.metadata.drop_all(bind=engine)


def test_login_success(setup_db):
    response = client.post(
        "/auth/login",
        data={"username": "admin_user", "password": "testpass"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


def test_login_failure(setup_db):
    response = client.post(
        "/auth/login",
        data={"username": "admin_user", "password": "wrongpass"},
    )
    assert response.status_code == 401


def test_access_me(setup_db):
    login_response = client.post(
        "/auth/login",
        data={"username": "admin_user", "password": "testpass"},
    )
    token = login_response.json()["access_token"]
    
    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "admin_user"
    assert data["role"] == "admin"
    

def test_refresh_token(setup_db):
    login_response = client.post(
        "/auth/login",
        data={"username": "doctor_user", "password": "testpass"},
    )
    refresh_token = login_response.json()["refresh_token"]
    
    response = client.post(
        f"/auth/refresh?refresh_token={refresh_token}"
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


def test_refresh_token_invalid(setup_db):
    response = client.post(
        "/auth/refresh?refresh_token=invalidtoken"
    )
    assert response.status_code == 403
