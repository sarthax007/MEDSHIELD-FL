from pydantic import BaseModel, ConfigDict
from datetime import datetime
from uuid import UUID
from typing import Optional

class HospitalBase(BaseModel):
    name: str
    location: str

class HospitalCreate(HospitalBase):
    pass

class HospitalResponse(HospitalBase):
    id: UUID
    created_at: datetime
    participation_status: str

    model_config = ConfigDict(from_attributes=True)
