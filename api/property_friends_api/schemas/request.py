from pydantic import BaseModel
from typing import Optional


class PredictionRequest(BaseModel):
    property_type: str
    location: str
    size: float
    bedrooms: int
    bathrooms: int
    age: Optional[int] = None
