# ToxGuard Multi-Label Content Flagger

A production-ready content moderation system that detects overall toxicity and categorizes it into 5 specific sublabels using dual BiLSTM model pipelines.

[![GitHub](https://img.shields.io/badge/GitHub-Source_Code-181717?logo=github&style=for-the-badge)](https://github.com/Priyanshukv06/ToxGuard_Multi-Label_Content_Flagger.git)

---

## 🚀 Features

- **Dual Model Inference**: 
  - *Multi-Label BiLSTM*: Single-pass prediction for all 6 labels.
  - *Two-Stage BiLSTM*: High-recall toxicity filter followed by a specialist sublabel classifier.
  - *Both (OR Ensemble)*: Runs both models and flags the content if *either* model detects toxicity.
- **Configurable Risk Thresholds**: 5 adjustable presets (Lenient to Aggressive) for fine-tuning the balance between precision and recall.
- **Batch Processing**: Upload a CSV to analyze thousands of comments at once.
- **Detailed Insights**: View raw probabilities, confidence levels, and side-by-side model comparisons.

## 🧠 Model Architecture

This system uses two deep learning architectures trained on scraped comment data:

1. **Multi-Label BiLSTM** 
   - A standard approach with a shared BiLSTM encoder and 6 independent sigmoid outputs.
2. **Two-Stage BiLSTM**
   - **Stage 1 (Filter)**: A binary classifier trained solely on overall toxicity.
   - **Stage 2 (Specialist)**: A 5-class sublabel classifier that takes the raw text *plus* the probability and discrete prediction from Stage 1 as meta-features. Stage 2 only evaluates if Stage 1 passes a strict 95% validation recall threshold.

## 🛠️ Tech Stack

- **Backend API**: FastAPI, Uvicorn, TensorFlow 2.10 (CPU-only for efficient deployment)
- **Frontend UI**: Streamlit (with custom dark theme CSS)
- **Data Processing**: Pandas, NumPy
- **Deployment**: Docker, Hugging Face Spaces (16GB RAM Tier)

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
├── models/                     # SavedModel artifacts (Multi-Label & Two-Stage)
├── scripts/                    # Utilities (e.g., sample data extractor)
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

- `POST /api/v1/predict`: Single text analysis.
- `POST /api/v1/predict/batch`: Bulk analysis for multiple texts.
- `GET /api/v1/data/random`: Get a random comment from the test dataset.

## 📊 Sample Data

The `data_sample/test_sample.json` file contains 10,000 stratified samples (5,000 toxic, 5,000 non-toxic) extracted from the test split to demonstrate the application's functionality.
