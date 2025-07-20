from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_value: float
