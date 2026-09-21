---
title: ToxGuard API
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
---

# 🛡️ ToxGuard Multi-Label Content Flagger

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://toxguard-06.streamlit.app/)
[![API Backend](https://img.shields.io/badge/API_Backend-Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://gencoder-toxguard-api.hf.space)
[![GitHub](https://img.shields.io/badge/GitHub-Source_Code-181717?logo=github&style=for-the-badge)](https://github.com/Priyanshukv06/ToxGuard_Multi-Label_Content_Flagger.git)

A production-ready content moderation system that detects overall toxicity and categorizes it into 5 specific sublabels using dual BiLSTM model pipelines, plus an optional fine-tuned DistilBERT transformer with attention-rollout explainability.

---

## 🔗 Live Deployments

* **Frontend Dashboard (Streamlit):** [https://toxguard-06.streamlit.app/](https://toxguard-06.streamlit.app/)
* **Backend API (Hugging Face Spaces):** [https://gencoder-toxguard-api.hf.space](https://gencoder-toxguard-api.hf.space)
* **API Documentation:** [https://gencoder-toxguard-api.hf.space/docs](https://gencoder-toxguard-api.hf.space/docs)

---

## 🚀 Features

- **Multiple Model Backends**:
  - *Multi-Label BiLSTM*: Single-pass prediction for all 6 labels.
  - *Two-Stage BiLSTM*: High-recall toxicity filter followed by a specialist sublabel classifier.
  - *Transformer (DistilBERT)*: Fine-tuned transformer baseline with per-word explainability.
  - *Both (OR Ensemble)*: Runs the two BiLSTM models and flags the content if *either* detects toxicity.
- **Explainability**: Attention-rollout word highlighting shows which words drove the transformer's decision.
- **Configurable Risk Thresholds**: 5 adjustable presets (Lenient to Aggressive) for fine-tuning the balance between precision and recall.
- **Batch Processing**: Upload a CSV to analyze thousands of comments at once.
- **Detailed Insights**: View raw probabilities, confidence levels, side-by-side model comparisons, real held-out metrics, and per-identity fairness/bias analysis.

## 🧠 Model Architecture

This system compares three model families trained on the same data/splits:

1. **Multi-Label BiLSTM**
   - A standard approach with a shared BiLSTM encoder and 6 independent sigmoid outputs.
2. **Two-Stage BiLSTM**
   - **Stage 1 (Filter)**: A binary classifier trained solely on overall toxicity.
   - **Stage 2 (Specialist)**: A 5-class sublabel classifier that takes the raw text *plus* the probability and discrete prediction from Stage 1 as meta-features. Stage 2 only evaluates if Stage 1 passes a strict 95% validation recall threshold.
3. **Transformer (DistilBERT)** — *optional, lazy-loaded*
   - Fine-tuned `distilbert-base-uncased` with a linear classification head on top of the `[CLS]` token.
   - Explainability via **attention rollout** (Abnar & Zuidema, 2020): attention matrices are averaged across heads and multiplied across all 6 layers (with residual-connection correction) to get a faithful per-token importance score, then merged from WordPiece subwords back into whole words.
   - Not required for the API to run — if the model isn't present or `torch`/`transformers` aren't installed, `model_choice="transformer"` simply returns a clear error instead of breaking anything else.

## 🛠️ Tech Stack

- **Backend API**: FastAPI, Uvicorn, TensorFlow 2.10 (CPU-only), PyTorch (CPU-only, optional transformer)
- **Frontend UI**: Streamlit (with custom dark theme CSS)
- **Data Processing**: Pandas, NumPy
- **Deployment**: Docker, Hugging Face Spaces (16GB RAM Tier), Streamlit Community Cloud

## 📂 Repository Structure

```text
.
├── app/                        # FastAPI Backend
│   ├── main.py                 # API Entrypoint
│   ├── inference.py            # Core ML prediction logic
│   ├── model_loader.py         # Loads TF SavedModels into memory
│   └── routers/                # API Endpoints (/predict, /data)
├── frontend/                   # Streamlit Frontend
│   └── streamlit_app.py        # UI code
├── models/                     # SavedModel + transformer artifacts (see below)
├── scripts/                    # Utilities (evaluate, fairness_eval, train_transformer, sample data)
├── Dockerfile                  # Lean deployment image
└── requirements_backend.txt    # Backend dependencies
```

## 💻 Local Development

### 1. Start the Backend (FastAPI)

```bash
pip install -r requirements_backend.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the Frontend (Streamlit)

```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 🌐 API Endpoints

- `POST /api/v1/predict`: Single text analysis. `model_choice` ∈ `multilabel` | `two_stage` | `transformer` | `both`; set `explain: true` with `transformer` to get word-level attention-rollout scores.
- `POST /api/v1/predict/batch`: Bulk analysis for multiple texts (no explanations, to keep latency low).
- `GET /api/v1/data/random`: Get a random comment from the test dataset.

## 📊 Sample Data

The `data_sample/test_sample.json` file holds a stratified draw from the test split
(5,000 non-toxic plus up to 5,000 toxic, sampled to cover each subtype) that powers the
"Randomize Comment" feature. Regenerate it with `python scripts/prepare_sample_data.py`.

## 📈 Model Metrics

The dashboard's performance numbers are computed from the held-out test set, not hard-coded:

```bash
python scripts/evaluate.py        # -> models/metrics.json  (ROC-AUC, PR-AUC, per-preset F1 for all 3 models)
python scripts/fairness_eval.py   # -> models/fairness.json (per-identity Subgroup/BPSN/BNSP AUC)
```

## 🤖 Training the Transformer

```bash
python scripts/train_transformer.py   # fine-tunes DistilBERT, saves models/transformer_model/
```

Needs `torch` + `transformers` (already in `requirements_backend.txt`). GPU recommended for
training (CPU works but is slow); serving runs on CPU either way. See
`notebooks/distilbert_transformer.ipynb` for the exploratory version with the training rationale.
