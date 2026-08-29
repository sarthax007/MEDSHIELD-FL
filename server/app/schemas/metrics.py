from pydantic import BaseModel
from typing import Optional


class AccuracyTrendItem(BaseModel):
    round_number: int
    global_accuracy: Optional[float]


class HospitalParticipationItem(BaseModel):
    hospital_name: str
    contribution_count: int


class CurrentRoundInfo(BaseModel):
    round_number: int
    status: str
