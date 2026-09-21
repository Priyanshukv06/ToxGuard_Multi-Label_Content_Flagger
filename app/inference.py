"""
Core inference logic for ToxGuard Content Flagger.
"""

import os
import json
import numpy as np
import tensorflow as tf
from app.models import PredictionResult, MULTILABEL_PRESETS, TWO_STAGE_PRESETS, TRANSFORMER_PRESETS
from app.text_cleaning import clean_text
from app.model_loader import get_transformer, transformer_available

CLASS_NAMES = ['toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat']

# Stage-1 gating thresholds loaded from the training artifact (with a safe fallback
# to the known-good values if the file is missing).
_ARTIFACTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "two_tier_inference_artifacts.json"
)


def _load_v1_thresholds():
    try:
        with open(_ARTIFACTS_PATH) as f:
            art = json.load(f)
        return float(art["model_1_recall_95_thresh"]), float(art["model_1_optimal_f1_thresh"])
    except Exception:
        return 0.3136042356491089, 0.5281431078910828


V1_RECALL_95_THRESH, V1_OPTIMAL_F1_THRESH = _load_v1_thresholds()


def _get_confidence(prob: float, threshold: float) -> str:
    """Returns confidence level based on absolute margin from threshold."""
    margin = abs(prob - threshold)
    if margin >= 0.2:
        return "HIGH"
    elif margin >= 0.1:
        return "MEDIUM"
    return "LOW"


def _get_overall_confidence(probs: dict, thresholds: dict) -> str:
    """Gets worst-case confidence among all evaluated classes."""
    confidences = [_get_confidence(probs[c], thresholds[c]) for c in CLASS_NAMES]
    if "LOW" in confidences:
        return "LOW"
    elif "MEDIUM" in confidences:
        return "MEDIUM"
    return "HIGH"


def predict_multilabel(texts: list[str], models: dict, preset_name: str) -> list[PredictionResult]:
    """Inference using the single Multi-Label BiLSTM."""
    model = models["multilabel"]
    thresholds = MULTILABEL_PRESETS[preset_name]

    # Keras expects numpy array of strings
    X = np.array(texts, dtype=object)

    # Predict (shape: [N, 6])
    preds = model.predict(X, batch_size=len(texts), verbose=0)

    results = []
    for i in range(len(texts)):
        probs = {c: float(preds[i, j]) for j, c in enumerate(CLASS_NAMES)}
        flags = {c: probs[c] >= thresholds[c] for c in CLASS_NAMES}

        results.append(PredictionResult(
            is_toxic=flags['toxicity'],
            probabilities=probs,
            flags=flags,
            confidence=_get_overall_confidence(probs, thresholds),
            model_used="multilabel",
            thresholds_applied=thresholds
        ))
    return results


def predict_two_stage(texts: list[str], models: dict, preset_name: str) -> list[PredictionResult]:
    """Inference using the Two-Stage BiLSTM pipeline."""
    model_v1 = models["two_stage_v1"]
    model_v2 = models["two_stage_v2"]
    thresholds = TWO_STAGE_PRESETS[preset_name]

    X = np.array(texts, dtype=object)

    # Stage 1: Predict Toxicity
    preds_v1 = model_v1.predict(X, batch_size=len(texts), verbose=0).flatten()

    # Stage 2: Sublabels (only for texts passing recall threshold)
    results = []

    # Prepare meta features for all, even if we zero them out later, for easier batching
    meta_features = np.column_stack((
        preds_v1,
        (preds_v1 >= V1_OPTIMAL_F1_THRESH).astype(float)
    ))

    # Run Stage 2 on everything for simplicity in batch processing
    preds_v2 = model_v2.predict([X, meta_features], batch_size=len(texts), verbose=0)

    for i in range(len(texts)):
        prob_tox = float(preds_v1[i])

        # If toxicity doesn't pass the validation 95% recall threshold, sublabels are strictly 0
        if prob_tox < V1_RECALL_95_THRESH:
            probs = {c: 0.0 for c in CLASS_NAMES[1:]}
        else:
            probs = {c: float(preds_v2[i, j]) for j, c in enumerate(CLASS_NAMES[1:])}

        probs['toxicity'] = prob_tox

        flags = {c: probs[c] >= thresholds[c] for c in CLASS_NAMES}

        results.append(PredictionResult(
            is_toxic=flags['toxicity'],
            probabilities=probs,
            flags=flags,
            confidence=_get_overall_confidence(probs, thresholds),
            model_used="two_stage",
            thresholds_applied=thresholds
        ))

    return results


def predict_transformer(texts: list[str], preset_name: str, with_explanation: bool = False) -> list[PredictionResult]:
    """
    Inference using the optional DistilBERT transformer model.

    Lazily loaded via app.model_loader.get_transformer(); raises ValueError
    (-> 422/handled by the caller) if the model isn't available, rather than a
    generic 500, so the client gets a clear "this model_choice isn't deployed
    here" signal instead of an opaque failure.
    """
    bundle = get_transformer()
    if bundle is None:
        raise ValueError(
            "The 'transformer' model is not available on this deployment. "
            "Train it with scripts/train_transformer.py and ensure torch + "
            "transformers are installed, then restart the API."
        )

    import torch
    from app.explain import explain_text

    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    max_len = bundle["max_len"]
    target_cols = bundle["target_cols"]

    # Presets center on the model's own trained optimal thresholds where available,
    # falling back to the generic TRANSFORMER_PRESETS table otherwise (see app/models.py).
    trained_thresholds = bundle.get("thresholds") or {}
    thresholds = {c: trained_thresholds.get(c, TRANSFORMER_PRESETS[preset_name][c]) for c in CLASS_NAMES}

    results = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])
        probs_vec = torch.sigmoid(logits)[0].numpy()

        probs = {c: float(probs_vec[j]) for j, c in enumerate(target_cols)}
        flags = {c: probs[c] >= thresholds[c] for c in CLASS_NAMES}

        explanation = None
        if with_explanation:
            words = explain_text(model, tokenizer, text, max_len=max_len)
            explanation = [{"token": w["token"], "weight": w["weight"]} for w in words]

        results.append(PredictionResult(
            is_toxic=flags['toxicity'],
            probabilities=probs,
            flags=flags,
            confidence=_get_overall_confidence(probs, thresholds),
            model_used="transformer",
            thresholds_applied=thresholds,
            explanation=explanation,
        ))
    return results


def predict_both(texts: list[str], models: dict, preset_name: str) -> list[PredictionResult]:
    """Inference using BOTH models, OR-ing the final flags."""
    res_multi = predict_multilabel(texts, models, preset_name)
    res_two_stage = predict_two_stage(texts, models, preset_name)

    results = []
    for m_res, t_res in zip(res_multi, res_two_stage):
        # Average probabilities just for display
        avg_probs = {c: (m_res.probabilities[c] + t_res.probabilities[c]) / 2 for c in CLASS_NAMES}

        # OR flags
        final_flags = {c: m_res.flags[c] or t_res.flags[c] for c in CLASS_NAMES}

        # Combined confidence
        conf_m = m_res.confidence
        conf_t = t_res.confidence
        if "LOW" in [conf_m, conf_t]:
            final_conf = "LOW"
        elif "MEDIUM" in [conf_m, conf_t]:
            final_conf = "MEDIUM"
        else:
            final_conf = "HIGH"

        final_res = PredictionResult(
            is_toxic=final_flags['toxicity'],
            probabilities=avg_probs,
            flags=final_flags,
            confidence=final_conf,
            model_used="both (OR ensemble)",
            thresholds_applied=MULTILABEL_PRESETS[preset_name],  # arbitrary, since we show detailed
            detailed_results={
                "multilabel": m_res,
                "two_stage": t_res
            }
        )
        results.append(final_res)

    return results


def predict_pipeline(
    texts: list[str], model_choice: str, preset_name: str, models: dict, with_explanation: bool = False
) -> list[PredictionResult]:
    """Router for the inference pipelines."""
    if not texts:
        return []

    # Apply the SAME Python-side cleaning used during training (train/serve parity).
    texts = [clean_text(t) for t in texts]

    if model_choice == "multilabel":
        return predict_multilabel(texts, models, preset_name)
    elif model_choice == "two_stage":
        return predict_two_stage(texts, models, preset_name)
    elif model_choice == "transformer":
        return predict_transformer(texts, preset_name, with_explanation=with_explanation)
    elif model_choice == "both":
        return predict_both(texts, models, preset_name)
    else:
        raise ValueError(f"Invalid model_choice: {model_choice}")
