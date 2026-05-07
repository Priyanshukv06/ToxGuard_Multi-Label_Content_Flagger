"""
Model loader — loads the three TF SavedModel directories at startup.
"""

import os
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Global model storage
_models: dict = {}

@tf.keras.utils.register_keras_serializable()
def tf_standardize(input_data):
    """
    Custom standardization run INSIDE the TextVectorization layer.
    """
    lowercase = tf.strings.lower(input_data)
    ascii_only = tf.strings.regex_replace(lowercase, r"[^\x00-\x7F]+", " ")
    no_punct = tf.strings.regex_replace(ascii_only, r"[^a-z0-9\s]", " ")
    return tf.strings.regex_replace(no_punct, r"\s+", " ")

def get_models() -> dict:
    """Returns the loaded models dict. Raises if not loaded."""
    if not _models:
        raise RuntimeError("Models not loaded. Call load_all_models() first.")
    return _models

def load_all_models():
    """
    Loads all 3 TF SavedModel directories into memory.
    Called once at application startup.
    """
    global _models

    model_dirs = {
        "multilabel": "multilabel_model",
        "two_stage_v1": "two_tier_model_v1_label",
        "two_stage_v2": "two_tier_model_v2_sublabels"
    }

    logger.info(f"Loading models from: {MODELS_DIR}")

    for key, dirname in model_dirs.items():
        dirpath = os.path.join(MODELS_DIR, dirname)
        if not os.path.exists(dirpath):
            raise FileNotFoundError(f"Model directory not found: {dirpath}")

        logger.info(f"  Loading {key} from {dirname}...")
        _models[key] = tf.keras.models.load_model(dirpath, custom_objects={'tf_standardize': tf_standardize})

    logger.info(f"✅ All {len(_models)} TF models loaded successfully.")
    return _models
