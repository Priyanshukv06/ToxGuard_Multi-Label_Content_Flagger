"""
Prediction API router.
"""

from fastapi import APIRouter, HTTPException
from app.models import PredictRequest, PredictionResult, BatchPredictRequest, BatchPredictResponse
from app.model_loader import get_models
from app.inference import predict_pipeline

router = APIRouter(prefix="/api/v1", tags=["Prediction"])

@router.post("/predict", response_model=PredictionResult)
async def predict(request: PredictRequest):
    """
    Predict toxicity for a single text.
    """
    try:
        models = get_models()
        results = predict_pipeline([request.text], request.model_choice, request.threshold_preset, models)
        return results[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """
    Predict toxicity for multiple texts.
    """
    try:
        models = get_models()
        results = predict_pipeline(request.texts, request.model_choice, request.threshold_preset, models)
        return BatchPredictResponse(predictions=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
