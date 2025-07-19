from fastapi import APIRouter, Depends
from ..schemas.request import PredictionRequest
from ..schemas.response import PredictionResponse
from ..middleware.auth import verify_api_key
from property_friends.models.prediction import predict_from_files

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/", response_model=PredictionResponse)
async def predict_property_value(
    request: PredictionRequest, api_key: str = Depends(verify_api_key)
):
    return {"predicted_value": 0.0, "confidence": 0.0}
