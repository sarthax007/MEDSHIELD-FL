from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

# Use a default SQLite DB for development if no POSTGRES_URL is provided
SQLALCHEMY_DATABASE_URL = os.getenv("POSTGRES_URL", "sqlite:///./sql_app.db")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    # check_same_thread=False is needed only for SQLite
    connect_args={"check_same_thread": False}
    if SQLALCHEMY_DATABASE_URL.startswith("sqlite")
    else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
