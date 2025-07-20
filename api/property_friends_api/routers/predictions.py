import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends
from ..schemas.request import PredictionRequest
from ..schemas.response import PredictionResponse
from ..middleware.auth import verify_api_key
from property_friends.models.prediction import predict_from_files

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/", response_model=PredictionResponse)
async def predict_property_value(
    request: PredictionRequest, api_key: str = Depends(verify_api_key)
):
    logger.info(f"Prediction request received: {request.dict()}")

    model_path = Path("/data/models/model.joblib")
    preprocessor_path = Path("/data/models/preprocessor.joblib")
    dataset = pd.DataFrame([request.dict()])

    predicted_value = predict_from_files(preprocessor_path, model_path, dataset)
    result = predicted_value[0]

    logger.info(f"Prediction completed: {result}")
    return {"predicted_value": result}
