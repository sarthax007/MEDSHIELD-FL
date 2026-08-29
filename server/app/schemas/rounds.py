from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional, List

class TrainingRoundBase(BaseModel):
    round_number: int
    status: str

class TrainingRoundCreate(TrainingRoundBase):
    pass

class TrainingRoundResponse(TrainingRoundBase):
    id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    global_accuracy: Optional[float] = None
    participants: List[str] = []

    model_config = ConfigDict(from_attributes=True)
