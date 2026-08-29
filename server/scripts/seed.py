import os
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import SessionLocal
from app.db.models import Hospital, User


def seed():
    db = SessionLocal()
    try:
        # Check if already seeded
        if db.query(Hospital).first():
            print("Database is already seeded.")
            return

        print("Seeding hospitals...")
        h1 = Hospital(name="General Hospital", location="New York")
        h2 = Hospital(name="City Medical Center", location="Los Angeles")
        h3 = Hospital(name="Lakeside Clinic", location="Chicago")
        db.add_all([h1, h2, h3])
        db.commit()
        db.refresh(h1)

        from app.core.security import get_password_hash
        print("Seeding users...")
        hashed_pw = get_password_hash("password123")
        u1 = User(
            hospital_id=h1.id,
            username="admin1",
            hashed_password=hashed_pw,
            role="admin",
        )
        u2 = User(
            hospital_id=h1.id,
            username="doctor_alice",
            hashed_password=hashed_pw,
            role="doctor",
        )
        u3 = User(
            hospital_id=h2.id,
            username="operator_bob",
            hashed_password=hashed_pw,
            role="operator",
        )
        db.add_all([u1, u2, u3])
        db.commit()

        print("Seeding complete.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
