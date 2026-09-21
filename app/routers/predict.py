"""
Prediction API router.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from app.models import PredictRequest, PredictionResult, BatchPredictRequest, BatchPredictResponse
from app.model_loader import get_models
from app.inference import predict_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Prediction"])


async def _run_pipeline(texts: list[str], model_choice: str, threshold_preset: str, with_explanation: bool = False):
    """Run the blocking, CPU-bound inference in a worker thread so it doesn't
    block the event loop (and freeze other requests) on a single worker.

    Model loading for 'multilabel'/'two_stage' is required (get_models() raises
    if unready); the 'transformer' model is optional and self-reports via
    ValueError inside predict_pipeline if it isn't deployed, so we don't gate
    on get_models() here for that case.
    """
    try:
        models = get_models()
    except RuntimeError:
        if model_choice != "transformer":
            raise HTTPException(status_code=503, detail="Models are not loaded yet. Please retry shortly.")
        models = {}

    try:
        return await asyncio.to_thread(
            predict_pipeline, texts, model_choice, threshold_preset, models, with_explanation
        )
    except ValueError as e:
        # e.g. invalid model_choice / preset, or transformer not deployed — client-facing
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal error during prediction.")


@router.post("/predict", response_model=PredictionResult)
async def predict(request: PredictRequest):
    """Predict toxicity for a single text."""
    results = await _run_pipeline(
        [request.text], request.model_choice, request.threshold_preset, with_explanation=request.explain
    )
    return results[0]


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Predict toxicity for multiple texts. Explanations are not computed for
    batch requests (would add per-row latency); use /predict for that."""
    results = await _run_pipeline(request.texts, request.model_choice, request.threshold_preset)
    return BatchPredictResponse(predictions=results)
