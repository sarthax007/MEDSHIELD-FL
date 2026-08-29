import uuid
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class UserBase(BaseModel):
    username: str
    role: str
    hospital_id: uuid.UUID


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None
    password: Optional[str] = None


class User(UserBase):
    id: uuid.UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
