from datetime import datetime, timezone
import uuid
from typing import Optional, List, Any

from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, JSON, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Hospital(Base):
    __tablename__ = "hospitals"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), index=True)
    location: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    users: Mapped[List["User"]] = relationship(
        back_populates="hospital", cascade="all, delete-orphan"
    )
    images: Mapped[List["ImageMetadata"]] = relationship(
        back_populates="hospital", cascade="all, delete-orphan"
    )


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    hospital_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("hospitals.id"))
    username: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    hospital: Mapped["Hospital"] = relationship(back_populates="users")
    labels: Mapped[List["Label"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    audit_logs: Mapped[List["AuditLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class TrainingRound(Base):
    __tablename__ = "training_rounds"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    round_number: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    status: Mapped[str] = mapped_column(String(50))
    start_time: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    global_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    model_versions: Mapped[List["ModelVersion"]] = relationship(
        back_populates="training_round", cascade="all, delete-orphan"
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    version_tag: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    training_round_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("training_rounds.id")
    )
    s3_path: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    training_round: Mapped["TrainingRound"] = relationship(
        back_populates="model_versions"
    )
    predictions: Mapped[List["Prediction"]] = relationship(
        back_populates="model_version", cascade="all, delete-orphan"
    )


class ImageMetadata(Base):
    __tablename__ = "image_metadata"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    hospital_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("hospitals.id"))
    local_image_ref: Mapped[str] = mapped_column(String(512), index=True)
    modality: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    hospital: Mapped["Hospital"] = relationship(back_populates="images")
    labels: Mapped[List["Label"]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )
    predictions: Mapped[List["Prediction"]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )


class Label(Base):
    __tablename__ = "labels"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    image_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("image_metadata.id"))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"))
    class_label: Mapped[str] = mapped_column(String(100))
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    image: Mapped["ImageMetadata"] = relationship(back_populates="labels")
    user: Mapped["User"] = relationship(back_populates="labels")


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    image_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("image_metadata.id"))
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("model_versions.id")
    )
    predicted_class: Mapped[str] = mapped_column(String(100))
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    image: Mapped["ImageMetadata"] = relationship(back_populates="predictions")
    model_version: Mapped["ModelVersion"] = relationship(back_populates="predictions")


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(255))
    resource_type: Mapped[str] = mapped_column(String(255))
    resource_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    details: Mapped[Any] = mapped_column(JSON, nullable=True)

    user: Mapped["User"] = relationship(back_populates="audit_logs")
