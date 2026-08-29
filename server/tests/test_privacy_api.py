import pytest
import uuid
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.base import Base
from app.db.session import get_db
from app.db.models import User, Hospital, ImageMetadata, AuditLog
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
    hospital1 = Hospital(id=hospital_id1, name="Privacy Hospital", location="City")
    db.add(hospital1)

    # Create test user
    admin_user = User(
        id=uuid.uuid4(),
        username="admin_privacy",
        hashed_password=get_password_hash("testpass"),
        role="admin",
        hospital_id=hospital_id1,
    )
    db.add(admin_user)

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
    db.add_all([img1, img2])

    # Create audit logs
    log1 = AuditLog(
        user_id=admin_user.id,
        action="login",
        resource_type="User",
        resource_id=admin_user.id,
        timestamp=datetime.now(timezone.utc),
    )
    db.add(log1)

    db.commit()

    yield

    Base.metadata.drop_all(bind=engine)


def get_auth_token():
    response = client.post(
        "/auth/login",
        data={"username": "admin_privacy", "password": "testpass"},
    )
    return response.json()["access_token"]


def test_get_privacy_status(setup_db):
    token = get_auth_token()
    response = client.get(
        "/privacy/status",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "verifiable_local"
    assert data["total_local_images_processed"] == 2
    assert data["raw_images_shared"] == 0
    assert "message" in data

    assert "recent_audit_logs" in data
    assert len(data["recent_audit_logs"]) >= 1
    # The first one should be our setup log or the login log we just made
    assert data["recent_audit_logs"][0]["action"] in ["login"]
