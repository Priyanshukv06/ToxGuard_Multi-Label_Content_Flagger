"""
Offline evaluation — computes REAL held-out metrics for both models and writes
them to models/metrics.json (consumed by the Streamlit "Model Performance" page,
replacing the previously hard-coded numbers).

Run in the training env (has pandas + scikit-learn + tensorflow):

    conda activate toxguard-train
    python scripts/evaluate.py

Outputs per model (multilabel, two_stage):
  - threshold-free per-class ROC-AUC and PR-AUC (average precision)
  - per-preset (lenient..aggressive) per-class precision/recall/F1 + macro/micro F1
    and the real toxic flag-rate ("Filtered %")
  - metrics at the F1-optimal thresholds from the training artifacts
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, average_precision_score, f1_score
)

# Make the project root importable so we can reuse the SERVING cleaning + standardizer
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from app.text_cleaning import clean_text            # noqa: E402
from app.model_loader import tf_standardize         # noqa: E402  (load-critical custom object)

# ---- Config -----------------------------------------------------------------
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TEST_CSV = os.path.join(PROJECT_ROOT, "data", "input", "test_split.csv")
OUT_PATH = os.path.join(MODELS_DIR, "metrics.json")
BATCH_SIZE = 512

CLASS_NAMES = ['toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat']
SUB_COLS = CLASS_NAMES[1:]

# NOTE: these mirror app/models.py (kept in sync manually — offline script).
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
TRANSFORMER_PRESETS = {
    "lenient":    {"toxicity": 0.65, "obscene": 0.50, "sexual_explicit": 0.45, "identity_attack": 0.55, "insult": 0.60, "threat": 0.45},
    "moderate":   {"toxicity": 0.60, "obscene": 0.45, "sexual_explicit": 0.40, "identity_attack": 0.50, "insult": 0.55, "threat": 0.40},
    "balanced":   {"toxicity": 0.55, "obscene": 0.40, "sexual_explicit": 0.35, "identity_attack": 0.45, "insult": 0.50, "threat": 0.35},
    "cautious":   {"toxicity": 0.50, "obscene": 0.35, "sexual_explicit": 0.30, "identity_attack": 0.40, "insult": 0.45, "threat": 0.30},
    "aggressive": {"toxicity": 0.45, "obscene": 0.30, "sexual_explicit": 0.25, "identity_attack": 0.35, "insult": 0.40, "threat": 0.25},
}


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _threshold_free(Y, P):
    """Per-class ROC-AUC and PR-AUC (average precision) — independent of thresholds."""
    auc, pr_auc = {}, {}
    for j, c in enumerate(CLASS_NAMES):
        y, p = Y[:, j], P[:, j]
        if len(np.unique(y)) < 2:      # AUC undefined if only one class present
            auc[c], pr_auc[c] = None, None
            continue
        auc[c] = float(roc_auc_score(y, p))
        pr_auc[c] = float(average_precision_score(y, p))
    return auc, pr_auc


def _metrics_at(Y, P, thresh_vec):
    """Per-class precision/recall/F1 + macro/micro F1 + toxic flag-rate at given thresholds."""
    preds = (P >= np.array(thresh_vec)).astype(int)
    per_class = {}
    f1s = []
    for j, c in enumerate(CLASS_NAMES):
        p, r, f1, _ = precision_recall_fscore_support(
            Y[:, j], preds[:, j], average="binary", pos_label=1, zero_division=0
        )
        per_class[c] = {"precision": float(p), "recall": float(r), "f1": float(f1),
                        "support": int(Y[:, j].sum())}
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    micro_f1 = float(f1_score(Y.ravel(), preds.ravel(), average="binary", pos_label=1, zero_division=0))
    return {
        "per_class": per_class,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "toxic_flag_rate": float(preds[:, 0].mean()),
    }


def _preset_block(Y, P, presets):
    return {name: _metrics_at(Y, P, [th[c] for c in CLASS_NAMES]) for name, th in presets.items()}


def _evaluate_transformer(texts, Y):
    """
    Optional: evaluates models/transformer_model/ if (a) torch+transformers are
    importable in whatever env this script is run from, and (b) the model
    directory exists. Fully isolated with try/except so this never breaks
    evaluation of the two TF models above, regardless of which env you run in.
    """
    model_dir = os.path.join(MODELS_DIR, "transformer_model")
    artifacts_path = os.path.join(MODELS_DIR, "transformer_inference_artifacts.json")
    if not os.path.exists(model_dir):
        print("\n[transformer] models/transformer_model/ not found — skipping "
              "(run scripts/train_transformer.py first if you want this).")
        return None

    try:
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("\n[transformer] torch/transformers not installed in this env — skipping. "
              "Re-run this script from an env that has them (e.g. toxguard-transformer) "
              "if you want transformer metrics included.")
        return None

    try:
        with open(artifacts_path) as f:
            art = json.load(f)

        class _DistilBertMultiLabel(nn.Module):
            def __init__(self, backbone, num_classes, dropout=0.3):
                super().__init__()
                self.backbone = backbone
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(backbone.config.hidden_size, num_classes)

            def forward(self, input_ids, attention_mask):
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
                return self.classifier(self.dropout(out.last_hidden_state[:, 0, :]))

        print("\n[transformer] Loading models/transformer_model ...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        backbone = AutoModel.from_pretrained(model_dir)
        target_cols = art["target_cols"]
        max_len = int(art.get("max_len", 128))
        model = _DistilBertMultiLabel(backbone, len(target_cols))
        model.classifier.load_state_dict(torch.load(os.path.join(model_dir, "classifier_head.pt"), map_location="cpu"))
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print(f"[transformer] Predicting on {len(texts):,} test rows ({device})...")
        probs = np.zeros((len(texts), len(target_cols)), dtype=float)
        bs = 64
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                chunk = [str(t) for t in texts[i:i + bs]]
                enc = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(enc["input_ids"], enc["attention_mask"])
                probs[i:i + bs] = torch.sigmoid(logits).cpu().numpy()

        opt_vec = [art["optimal_thresholds"][c] for c in CLASS_NAMES]
        auc, prauc = _threshold_free(Y, probs)
        return {
            "roc_auc": auc,
            "pr_auc": prauc,
            "optimal": _metrics_at(Y, probs, opt_vec),
            "optimal_thresholds": {c: float(art["optimal_thresholds"][c]) for c in CLASS_NAMES},
            "presets": _preset_block(Y, probs, TRANSFORMER_PRESETS),
        }
    except Exception as e:
        print(f"\n[transformer] Evaluation failed, skipping: {e}")
        return None


def main():
    t0 = time.time()
    print(f"Loading test data: {TEST_CSV}")
    df = pd.read_csv(TEST_CSV).fillna({"comment_text": "missing_text"})
    df["comment_text"] = df["comment_text"].map(clean_text)
    X = df["comment_text"].to_numpy(dtype=object)
    Y = df[CLASS_NAMES].to_numpy().astype(int)
    print(f"  {len(df):,} test rows")

    art_multi = _load_json(os.path.join(MODELS_DIR, "multilabel_inference_artifacts.json"))
    art_two = _load_json(os.path.join(MODELS_DIR, "two_tier_inference_artifacts.json"))
    v1_recall95 = float(art_two["model_1_recall_95_thresh"])
    v1_opt_f1 = float(art_two["model_1_optimal_f1_thresh"])

    custom = {"tf_standardize": tf_standardize}

    # ---- Multi-Label model ----
    print("\nLoading multilabel_model ...")
    m_multi = tf.keras.models.load_model(os.path.join(MODELS_DIR, "multilabel_model"), custom_objects=custom)
    print("Predicting (multilabel) ...")
    P_multi = m_multi.predict(X, batch_size=BATCH_SIZE, verbose=1)

    # ---- Two-Stage model ----
    print("\nLoading two_tier_model_v1_label ...")
    m_v1 = tf.keras.models.load_model(os.path.join(MODELS_DIR, "two_tier_model_v1_label"), custom_objects=custom)
    print("Loading two_tier_model_v2_sublabels ...")
    m_v2 = tf.keras.models.load_model(os.path.join(MODELS_DIR, "two_tier_model_v2_sublabels"), custom_objects=custom)

    print("Predicting (two-stage) ...")
    p_v1 = m_v1.predict(X, batch_size=BATCH_SIZE, verbose=1).flatten()
    meta = np.column_stack((p_v1, (p_v1 >= v1_opt_f1).astype(float)))
    p_v2 = m_v2.predict([X, meta], batch_size=BATCH_SIZE, verbose=1)

    # Assemble two-stage [N,6] prob matrix with the recall-95 gate (mirrors serving)
    P_two = np.zeros((len(X), len(CLASS_NAMES)), dtype=float)
    P_two[:, 0] = p_v1
    gate = p_v1 >= v1_recall95
    for j in range(len(SUB_COLS)):
        P_two[gate, j + 1] = p_v2[gate, j]

    # ---- Assemble metrics ----
    multi_auc, multi_prauc = _threshold_free(Y, P_multi)
    two_auc, two_prauc = _threshold_free(Y, P_two)

    multi_opt_vec = [art_multi["optimal_thresholds"][c] for c in CLASS_NAMES]
    two_opt_vec = [v1_opt_f1] + [art_two["model_2_optimal_thresholds"][c] for c in SUB_COLS]

    metrics = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_size": int(len(df)),
        "class_names": CLASS_NAMES,
        "multilabel": {
            "roc_auc": multi_auc,
            "pr_auc": multi_prauc,
            "optimal": _metrics_at(Y, P_multi, multi_opt_vec),
            "optimal_thresholds": {c: float(art_multi["optimal_thresholds"][c]) for c in CLASS_NAMES},
            "presets": _preset_block(Y, P_multi, MULTILABEL_PRESETS),
        },
        "two_stage": {
            "roc_auc": two_auc,
            "pr_auc": two_prauc,
            "optimal": _metrics_at(Y, P_two, two_opt_vec),
            "optimal_thresholds": {c: float(v) for c, v in zip(CLASS_NAMES, two_opt_vec)},
            "presets": _preset_block(Y, P_two, TWO_STAGE_PRESETS),
        },
    }

    # ---- Optional third model: transformer (skips gracefully if unavailable) ----
    transformer_block = _evaluate_transformer(df["comment_text"].tolist(), Y)
    if transformer_block is not None:
        metrics["transformer"] = transformer_block

    with open(OUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Console summary ----
    print("\n" + "=" * 62)
    print(f"Saved: {OUT_PATH}   (test_size={len(df):,}, {time.time()-t0:.0f}s)")
    for model_key in ("multilabel", "two_stage", "transformer"):
        if model_key not in metrics:
            continue
        b = metrics[model_key]
        print(f"\n[{model_key}] per-class ROC-AUC / PR-AUC:")
        for c in CLASS_NAMES:
            a = b["roc_auc"][c]
            pa = b["pr_auc"][c]
            print(f"   {c:<18} AUC={a if a is None else round(a,4)}   PR-AUC={pa if pa is None else round(pa,4)}")
        print(f"   balanced-preset macro-F1 = {b['presets']['balanced']['macro_f1']:.4f}"
              f" | optimal macro-F1 = {b['optimal']['macro_f1']:.4f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
