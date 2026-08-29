from pydantic import BaseModel, ConfigDict
from datetime import datetime
from uuid import UUID

class PredictionResponse(BaseModel):
    id: UUID
    image_id: UUID
    model_version_id: UUID
    predicted_class: str
    confidence: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
