from pydantic import BaseModel, ConfigDict
from uuid import UUID


class QueueItem(BaseModel):
    image_id: UUID
    local_image_ref: str
    uncertainty_score: float
    model_prediction: int

    model_config = ConfigDict(from_attributes=True)


class LabelSubmission(BaseModel):
    class_label: str  # e.g., "TUMOR" or "NO_TUMOR"
