"""
Pydantic request/response schemas for ToxGuard Multi-Label Content Flagger.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict


# Threshold presets based on optimal validation F1 scores
# Format: (toxicity, obscene, sexual_explicit, identity_attack, insult, threat)
# Stage 1 Toxicity Threshold must be passed before Stage 2 evaluates.

MULTILABEL_PRESETS = {
    "lenient":    {"toxicity": 0.65, "obscene": 0.50, "sexual_explicit": 0.45, "identity_attack": 0.55, "insult": 0.60, "threat": 0.45},
    "moderate":   {"toxicity": 0.60, "obscene": 0.45, "sexual_explicit": 0.40, "identity_attack": 0.50, "insult": 0.55, "threat": 0.40},
    "balanced":   {"toxicity": 0.55, "obscene": 0.40, "sexual_explicit": 0.35, "identity_attack": 0.45, "insult": 0.50, "threat": 0.35},
    "cautious":   {"toxicity": 0.50, "obscene": 0.35, "sexual_explicit": 0.30, "identity_attack": 0.40, "insult": 0.45, "threat": 0.30},
    "aggressive": {"toxicity": 0.45, "obscene": 0.30, "sexual_explicit": 0.25, "identity_attack": 0.35, "insult": 0.40, "threat": 0.25},
}

TWO_STAGE_PRESETS = {
    "lenient":    {"toxicity": 0.65, "obscene": 0.50, "sexual_explicit": 0.45, "identity_attack": 0.55, "insult": 0.60, "threat": 0.40},
    "moderate":   {"toxicity": 0.60, "obscene": 0.45, "sexual_explicit": 0.40, "identity_attack": 0.50, "insult": 0.55, "threat": 0.35},
    "balanced":   {"toxicity": 0.55, "obscene": 0.40, "sexual_explicit": 0.35, "identity_attack": 0.45, "insult": 0.50, "threat": 0.30},
    "cautious":   {"toxicity": 0.50, "obscene": 0.35, "sexual_explicit": 0.30, "identity_attack": 0.40, "insult": 0.45, "threat": 0.25},
    "aggressive": {"toxicity": 0.45, "obscene": 0.30, "sexual_explicit": 0.25, "identity_attack": 0.35, "insult": 0.40, "threat": 0.20},
}

class PredictRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    model_choice: str = Field(
        "both", 
        description="Model to use: 'multilabel', 'two_stage', or 'both'"
    )
    threshold_preset: str = Field(
        "balanced",
        description="Threshold preset: 'lenient', 'moderate', 'balanced', 'cautious', 'aggressive'"
    )

class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., description="List of texts to analyze")
    model_choice: str = Field("both")
    threshold_preset: str = Field("balanced")

class PredictionResult(BaseModel):
    is_toxic: bool = Field(..., description="Overall toxicity flag")
    probabilities: Dict[str, float] = Field(..., description="Raw probabilities for each class")
    flags: Dict[str, bool] = Field(..., description="Boolean flags for each class based on thresholds")
    confidence: str = Field(..., description="HIGH, MEDIUM, or LOW based on margin from threshold")
    model_used: str = Field(..., description="The model used for this prediction")
    thresholds_applied: Dict[str, float] = Field(..., description="The thresholds used for evaluation")
    
    # If using 'both' mode, this field holds the individual model results for detailed comparison
    detailed_results: Optional[Dict[str, 'PredictionResult']] = None

class BatchPredictResponse(BaseModel):
    predictions: list[PredictionResult]

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    models_count: int

class RandomPatientResponse(BaseModel):
    patient: dict
