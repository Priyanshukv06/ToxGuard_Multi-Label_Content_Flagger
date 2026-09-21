"""
Fairness / unintended-bias evaluation.

The train/val/test splits only kept comment_text + the 6 labels, but the original
data/input/all_data.csv still has the identity annotations. This script rejoins the
test split to all_data.csv on comment_text to recover those identity columns, then
computes the standard Jigsaw bias metrics per identity subgroup:

  - Subgroup AUC : toxicity-model AUC restricted to comments mentioning the group.
  - BPSN AUC     : Background Positive, Subgroup Negative — low value => the model
                   over-flags NON-toxic comments that mention the group (false positives).
  - BNSP AUC     : Background Negative, Subgroup Positive — low value => the model
                   MISSES toxic comments mentioning the group (false negatives).

Run in the training env (pandas + scikit-learn + tensorflow):

    conda activate toxguard-train
    python scripts/fairness_eval.py

Writes models/fairness.json (consumed by the Streamlit "Model Performance" page).
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from app.text_cleaning import clean_text        # noqa: E402
from app.model_loader import tf_standardize      # noqa: E402

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TEST_CSV = os.path.join(PROJECT_ROOT, "data", "input", "test_split.csv")
ALL_CSV = os.path.join(PROJECT_ROOT, "data", "input", "all_data.csv")
OUT_PATH = os.path.join(MODELS_DIR, "fairness.json")
BATCH_SIZE = 512
MIN_SUBGROUP_SIZE = 100          # skip subgroups with too few examples to be meaningful
IDENTITY_THRESHOLD = 0.5         # a comment "mentions" a group if annotation >= 0.5

IDENTITY_COLS = [
    "male", "female", "transgender", "other_gender", "heterosexual",
    "homosexual_gay_or_lesbian", "bisexual", "other_sexual_orientation",
    "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",
    "black", "white", "asian", "latino", "other_race_or_ethnicity",
    "physical_disability", "intellectual_or_learning_disability",
    "psychiatric_or_mental_illness", "other_disability",
]


def _safe_auc(y, s):
    y = np.asarray(y)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, np.asarray(s)))


def _subgroup_metrics(y, score, member):
    """Compute subgroup / BPSN / BNSP AUCs for one identity subgroup."""
    bg = ~member
    pos = y == 1
    neg = y == 0
    return {
        "n": int(member.sum()),
        "subgroup_pos_rate": float(y[member].mean()) if member.sum() else None,
        "subgroup_auc": _safe_auc(y[member], score[member]),
        "bpsn_auc": _safe_auc(y[(bg & pos) | (member & neg)], score[(bg & pos) | (member & neg)]),
        "bnsp_auc": _safe_auc(y[(bg & neg) | (member & pos)], score[(bg & neg) | (member & pos)]),
    }


def _analyze(model_key, score, y, ident_bin):
    subgroups = {}
    for col in IDENTITY_COLS:
        member = ident_bin[col].to_numpy()
        if member.sum() < MIN_SUBGROUP_SIZE:
            continue
        subgroups[col] = _subgroup_metrics(y, score, member)

    # Summaries: mean subgroup AUC, and the worst (lowest) BPSN/BNSP groups
    def _mean(metric):
        vals = [m[metric] for m in subgroups.values() if m[metric] is not None]
        return float(np.mean(vals)) if vals else None

    worst_bpsn = sorted(
        [(c, m["bpsn_auc"]) for c, m in subgroups.items() if m["bpsn_auc"] is not None],
        key=lambda kv: kv[1],
    )[:5]
    worst_bnsp = sorted(
        [(c, m["bnsp_auc"]) for c, m in subgroups.items() if m["bnsp_auc"] is not None],
        key=lambda kv: kv[1],
    )[:5]

    return {
        "overall_toxicity_auc": _safe_auc(y, score),
        "n_subgroups": len(subgroups),
        "mean_subgroup_auc": _mean("subgroup_auc"),
        "mean_bpsn_auc": _mean("bpsn_auc"),
        "mean_bnsp_auc": _mean("bnsp_auc"),
        "worst_bpsn": [{"group": c, "bpsn_auc": v} for c, v in worst_bpsn],
        "worst_bnsp": [{"group": c, "bnsp_auc": v} for c, v in worst_bnsp],
        "subgroups": subgroups,
    }


def main():
    t0 = time.time()
    if not os.path.exists(ALL_CSV):
        raise SystemExit(f"Missing {ALL_CSV} — needed for identity columns. Aborting.")

    print(f"Loading test split: {TEST_CSV}")
    test_df = pd.read_csv(TEST_CSV).fillna({"comment_text": "missing_text"})

    print(f"Loading identity columns from: {ALL_CSV} (this is large, ~3.4M rows)...")
    usecols = ["comment_text"] + IDENTITY_COLS
    all_df = pd.read_csv(ALL_CSV, usecols=usecols)
    all_df = all_df.dropna(subset=["comment_text"]).drop_duplicates(subset=["comment_text"])
    print(f"  all_data unique comments: {len(all_df):,}")

    # Attach identity annotations to the test rows via comment_text
    merged = test_df.merge(all_df, on="comment_text", how="left")
    matched = merged[IDENTITY_COLS].notna().any(axis=1).sum()
    print(f"  test rows: {len(test_df):,} | rows with identity annotations: {matched:,}")

    ident_bin = (merged[IDENTITY_COLS] >= IDENTITY_THRESHOLD).fillna(False)
    y = merged["toxicity"].to_numpy().astype(int)

    # Toxicity scores from both deployed models
    X = merged["comment_text"].map(clean_text).to_numpy(dtype=object)
    custom = {"tf_standardize": tf_standardize}

    print("\nLoading multilabel_model ...")
    m_multi = tf.keras.models.load_model(os.path.join(MODELS_DIR, "multilabel_model"), custom_objects=custom)
    print("Predicting toxicity (multilabel) ...")
    score_multi = m_multi.predict(X, batch_size=BATCH_SIZE, verbose=1)[:, 0]

    print("Loading two_tier_model_v1_label ...")
    m_v1 = tf.keras.models.load_model(os.path.join(MODELS_DIR, "two_tier_model_v1_label"), custom_objects=custom)
    print("Predicting toxicity (two-stage stage-1) ...")
    score_v1 = m_v1.predict(X, batch_size=BATCH_SIZE, verbose=1).flatten()

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_size": int(len(test_df)),
        "rows_with_identity": int(matched),
        "min_subgroup_size": MIN_SUBGROUP_SIZE,
        "identity_threshold": IDENTITY_THRESHOLD,
        "multilabel": _analyze("multilabel", score_multi, y, ident_bin),
        "two_stage": _analyze("two_stage", score_v1, y, ident_bin),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 62)
    print(f"Saved: {OUT_PATH}   ({time.time()-t0:.0f}s)")
    for mk in ("multilabel", "two_stage"):
        b = result[mk]
        print(f"\n[{mk}] overall AUC={round(b['overall_toxicity_auc'],4) if b['overall_toxicity_auc'] else None} "
              f"| mean subgroup AUC={round(b['mean_subgroup_auc'],4) if b['mean_subgroup_auc'] else None} "
              f"| mean BPSN={round(b['mean_bpsn_auc'],4) if b['mean_bpsn_auc'] else None} "
              f"| mean BNSP={round(b['mean_bnsp_auc'],4) if b['mean_bnsp_auc'] else None}")
        if b["worst_bpsn"]:
            worst = ", ".join(f"{d['group']}={round(d['bpsn_auc'],3)}" for d in b["worst_bpsn"][:3])
            print(f"   worst BPSN (most false-positive bias): {worst}")
    print("=" * 62)


if __name__ == "__main__":
    main()
