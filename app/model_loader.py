"""
Model loader — loads the three TF SavedModel directories at startup, plus an
OPTIONAL DistilBERT transformer model (lazy-loaded on first use, not required
for the API to run — if torch/transformers aren't installed or the model
directory doesn't exist, the transformer is simply unavailable and everything
else keeps working).
"""

import os
import json
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Global model storage
_models: dict = {}
_load_error: "str | None" = None
EXPECTED_MODEL_COUNT = 3

# --- Optional transformer (lazy) ---
_transformer_bundle: "dict | None" = None   # {"model", "tokenizer", "max_len", "thresholds"}
_transformer_load_error: "str | None" = None
_transformer_load_attempted = False


@tf.keras.utils.register_keras_serializable()
def tf_standardize(input_data):
    """
    Custom standardization run INSIDE the TextVectorization layer.
    """
    lowercase = tf.strings.lower(input_data)
    ascii_only = tf.strings.regex_replace(lowercase, r"[^\x00-\x7F]+", " ")
    no_punct = tf.strings.regex_replace(ascii_only, r"[^a-z0-9\s]", " ")
    return tf.strings.regex_replace(no_punct, r"\s+", " ")


def models_ready() -> bool:
    """True only when all expected models are loaded and no load error occurred."""
    return len(_models) == EXPECTED_MODEL_COUNT and _load_error is None


def get_models() -> dict:
    """Returns the loaded models dict. Raises if not fully loaded."""
    if not models_ready():
        raise RuntimeError("Models not loaded.")
    return _models


def load_all_models():
    """
    Loads all 3 TF SavedModel directories into memory.
    Called once at application startup. Records _load_error on failure so the
    background loader task never dies silently.
    """
    global _models, _load_error
    _load_error = None

    model_dirs = {
        "multilabel": "multilabel_model",
        "two_stage_v1": "two_tier_model_v1_label",
        "two_stage_v2": "two_tier_model_v2_sublabels"
    }

    logger.info(f"Loading models from: {MODELS_DIR}")

    try:
        for key, dirname in model_dirs.items():
            dirpath = os.path.join(MODELS_DIR, dirname)
            if not os.path.exists(dirpath):
                raise FileNotFoundError(f"Model directory not found: {dirpath}")

            logger.info(f"  Loading {key} from {dirname}...")
            _models[key] = tf.keras.models.load_model(
                dirpath, custom_objects={'tf_standardize': tf_standardize}
            )

        logger.info(f"✅ All {len(_models)} TF models loaded successfully.")
        return _models
    except Exception as e:
        _load_error = str(e)
        raise


def transformer_available() -> bool:
    """Cheap check: does the transformer model directory exist on disk?"""
    return os.path.exists(os.path.join(MODELS_DIR, "transformer_model"))


def get_transformer():
    """
    Lazily loads the DistilBERT transformer bundle on first call and caches it.
    Returns None (never raises) if torch/transformers aren't installed, the
    model directory is missing, or loading otherwise fails — callers should
    treat None as "transformer model_choice unavailable" rather than an error
    in the two existing TF models.
    """
    global _transformer_bundle, _transformer_load_error, _transformer_load_attempted

    if _transformer_bundle is not None:
        return _transformer_bundle
    if _transformer_load_attempted:
        return None  # already tried and failed this process lifetime
    _transformer_load_attempted = True

    model_dir = os.path.join(MODELS_DIR, "transformer_model")
    artifacts_path = os.path.join(MODELS_DIR, "transformer_inference_artifacts.json")

    if not os.path.exists(model_dir):
        logger.info("ℹ️ Transformer model directory not found — 'transformer' model_choice disabled.")
        return None

    try:
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel

        with open(artifacts_path) as f:
            artifacts = json.load(f)

        class _DistilBertMultiLabel(nn.Module):
            def __init__(self, backbone, num_classes, dropout=0.3):
                super().__init__()
                self.backbone = backbone
                hidden_size = backbone.config.hidden_size
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(hidden_size, num_classes)

            def forward(self, input_ids, attention_mask, output_attentions=False):
                out = self.backbone(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )
                cls_hidden = out.last_hidden_state[:, 0, :]
                logits = self.classifier(self.dropout(cls_hidden))
                if output_attentions:
                    return logits, out.attentions
                return logits

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        backbone = AutoModel.from_pretrained(model_dir)
        num_classes = len(artifacts["target_cols"])
        model = _DistilBertMultiLabel(backbone, num_classes)
        head_path = os.path.join(model_dir, "classifier_head.pt")
        model.classifier.load_state_dict(torch.load(head_path, map_location="cpu"))
        model.eval()

        _transformer_bundle = {
            "model": model,
            "tokenizer": tokenizer,
            "max_len": int(artifacts.get("max_len", 128)),
            "thresholds": artifacts["optimal_thresholds"],
            "target_cols": artifacts["target_cols"],
        }
        logger.info("✅ Transformer model loaded (lazy).")
        return _transformer_bundle
    except Exception as e:
        _transformer_load_error = str(e)
        logger.warning(f"⚠️ Transformer model failed to load, 'transformer' model_choice disabled: {e}")
        return None
