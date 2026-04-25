import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve
from pathlib import Path

# Fix for model loading error
@tf.keras.utils.register_keras_serializable()
def tf_standardize(input_data):
    lowercase = tf.strings.lower(input_data)
    ascii_only = tf.strings.regex_replace(lowercase, r"[^\x00-\x7F]+", " ")
    no_punct = tf.strings.regex_replace(ascii_only, r"[^a-z0-9\s]", " ")
    return tf.strings.regex_replace(no_punct, r"\s+", " ")

def downsample_curve(precision, recall, thresholds, num_points=50):
    # Select roughly num_points evenly spaced indices
    indices = np.linspace(0, len(thresholds)-1, num_points, dtype=int)
    return {
        "precision": precision[indices].tolist(),
        "recall": recall[indices].tolist(),
        "thresholds": thresholds[indices].tolist()
    }

def calculate_f1(precision, recall):
    p = np.array(precision)
    r = np.array(recall)
    # Avoid division by zero
    f1 = np.divide(2 * (p * r), (p + r), out=np.zeros_like(p), where=(p + r) != 0)
    return f1.tolist()

def main():
    import gc
    
    print("Loading test data...")
    with open("data_sample/test_sample.json", "r") as f:
        data = json.load(f)
    
    texts = [d["comment_text"] for d in data]
    texts_tf = tf.constant(texts)
    
    y_true_tox = np.array([d["toxicity"] for d in data])
    sub_cols = ["obscene", "sexual_explicit", "identity_attack", "insult", "threat"]
    y_true_sub = np.array([[d[col] for col in sub_cols] for d in data])
    y_true_multi = np.array([[d["toxicity"]] + [d[col] for col in sub_cols] for d in data])
    
    multi_cols = ["toxicity"] + sub_cols
    
    print("Loading Multi-Label model...")
    multi_model = tf.keras.models.load_model("models/multilabel_model", custom_objects={'tf_standardize': tf_standardize})
    print("Running Multi-Label inference...")
    preds_multi = multi_model.predict(texts_tf, batch_size=256)
    
    del multi_model
    tf.keras.backend.clear_session()
    gc.collect()
    
    print("Loading Stage 1 model...")
    stage1 = tf.keras.models.load_model("models/two_tier_model_v1_label", custom_objects={'tf_standardize': tf_standardize})
    print("Running Two-Stage inference (Stage 1)...")
    preds_s1 = stage1.predict(texts_tf, batch_size=256).flatten()
    
    del stage1
    meta_features = np.column_stack((
        preds_s1,
        (preds_s1 >= 0.5281431078910828).astype(np.float32)
    ))
    
    print("Loading Stage 2 model...")
    stage2 = tf.keras.models.load_model("models/two_tier_model_v2_sublabels", custom_objects={'tf_standardize': tf_standardize})
    print("Running Two-Stage inference (Stage 2)...")
    preds_s2 = stage2.predict([texts_tf, meta_features], batch_size=256)
    
    del stage2
    tf.keras.backend.clear_session()
    gc.collect()


    
    curves = {
        "multilabel": {},
        "two_stage": {}
    }
    
    print("Calculating metrics...")
    # Multi-label curves
    for i, col in enumerate(multi_cols):
        p, r, t = precision_recall_curve(y_true_multi[:, i], preds_multi[:, i])
        curve = downsample_curve(p, r, t)
        curve["f1"] = calculate_f1(curve["precision"], curve["recall"])
        curves["multilabel"][col] = curve
        
    # Two-stage curves
    p, r, t = precision_recall_curve(y_true_tox, preds_s1)
    curve = downsample_curve(p, r, t)
    curve["f1"] = calculate_f1(curve["precision"], curve["recall"])
    curves["two_stage"]["toxicity"] = curve
    
    for i, col in enumerate(sub_cols):
        p, r, t = precision_recall_curve(y_true_sub[:, i], preds_s2[:, i])
        curve = downsample_curve(p, r, t)
        curve["f1"] = calculate_f1(curve["precision"], curve["recall"])
        curves["two_stage"][col] = curve
        
    print("Saving to pr_curves.json...")
    with open("models/pr_curves.json", "w") as f:
        json.dump(curves, f)
        
    print("Done!")

if __name__ == "__main__":
    main()
