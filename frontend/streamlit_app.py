"""
ToxGuard Content Flagger — Streamlit Frontend
"""

import streamlit as st
import httpx
import pandas as pd
import os
import json
import time

API_BASE = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 60
MAX_BATCH_ROWS = 1000  # cap client-side to keep one request responsive / avoid backend OOM

FILTER_LEVELS = {
    "🟢 Lenient": "lenient",
    "🔵 Moderate": "moderate",
    "⚖️ Balanced": "balanced",
    "🟠 Cautious": "cautious",
    "🔴 Aggressive": "aggressive",
}

MODELS = {
    "Multi-Label BiLSTM": "multilabel",
    "Two-Stage BiLSTM": "two_stage",
    "Both (OR Ensemble)": "both",
    "Transformer (DistilBERT)": "transformer",
}

CLASS_NAMES = ['toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat']
CLASS_LABELS = {
    'toxicity': 'Overall Toxicity',
    'obscene': 'Obscene',
    'sexual_explicit': 'Sexual Explicit',
    'identity_attack': 'Identity Attack',
    'insult': 'Insult',
    'threat': 'Threat'
}


def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .main-header {
            background: linear-gradient(135deg, #6C5CE7 0%, #a29bfe 50%, #74b9ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 0;
            letter-spacing: -0.02em;
        }
        .sub-header { color: #a0a0b0; font-size: 1.05rem; margin-top: -8px; margin-bottom: 24px; font-weight: 300; }
        .metric-card {
            background: linear-gradient(145deg, #1e2130 0%, #252839 100%);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(108, 92, 231, 0.15);
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
            margin-bottom: 16px;
        }
        .badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
        .badge-safe { background: rgba(0, 184, 148, 0.15); color: #00b894; border: 1px solid rgba(0, 184, 148, 0.3); }
        .badge-risky { background: rgba(255, 118, 117, 0.15); color: #ff7675; border: 1px solid rgba(255, 118, 117, 0.3); }
        .risk-bar-container { background: #1a1d26; border-radius: 8px; height: 8px; overflow: hidden; margin: 6px 0; }
        .risk-bar { height: 100%; border-radius: 8px; transition: width 0.5s ease; }
    </style>
    """, unsafe_allow_html=True)


def render_risk_bar(probability: float, label: str, threshold: float, flagged: bool):
    color = "#ff7675" if flagged else "#00b894"
    width_pct = min(probability * 100, 100)
    st.markdown(f"""
    <div style="margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #8a8a9a; font-size: 0.8rem;">{label} (Thresh: {threshold:.2f})</span>
            <span style="color: {color}; font-weight: 600; font-size: 0.85rem;">{probability:.1%}</span>
        </div>
        <div class="risk-bar-container">
            <div class="risk-bar" style="width: {width_pct}%; background: {color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---- Backend calls ----------------------------------------------------------
def fetch_random_comment():
    try:
        r = httpx.get(f"{API_BASE}/api/v1/data/random", timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json().get("comment", {})
    except httpx.HTTPStatusError as e:
        st.error(f"Could not fetch a sample comment (HTTP {e.response.status_code}).")
        return {}
    except Exception:
        st.error("Backend unreachable while fetching a sample comment.")
        return {}


def predict(text: str, model_choice: str, preset: str, explain: bool = False):
    """Returns (result_dict, latency_seconds) or (None, None) on failure."""
    payload = {"text": text, "model_choice": model_choice, "threshold_preset": preset, "explain": explain}
    try:
        t0 = time.perf_counter()
        r = httpx.post(f"{API_BASE}/api/v1/predict", json=payload, timeout=REQUEST_TIMEOUT)
        latency = time.perf_counter() - t0
        r.raise_for_status()
        return r.json(), latency
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            st.error("Models are still loading on the backend — please retry in a few seconds.")
        elif e.response.status_code == 422:
            st.error(e.response.json().get("detail", "This model isn't available on this deployment."))
        else:
            st.error(f"Prediction failed (HTTP {e.response.status_code}).")
        return None, None
    except Exception:
        st.error("Could not reach the backend. Is it running?")
        return None, None


def render_explanation(explanation: list):
    """Renders word-level attention-rollout importance as a highlighted-text heatmap."""
    if not explanation:
        return
    st.markdown("### 🔬 Why the transformer flagged this (attention rollout)")
    st.caption(
        "Darker highlight = the [CLS] token (the one the classifier reads) attended to that "
        "word more, after rolling attention up through all 6 DistilBERT layers."
    )
    spans = []
    for w in explanation:
        weight = w["weight"]
        # Purple highlight scaling with importance; low-importance words stay unstyled.
        alpha = 0.08 + 0.55 * weight
        token_display = w["token"].replace("##", "")
        spans.append(
            f'<span style="background: rgba(108, 92, 231, {alpha:.2f}); '
            f'padding: 2px 4px; border-radius: 4px; margin: 0 1px;">{token_display}</span>'
        )
    st.markdown(
        f'<div class="metric-card" style="line-height: 2.2;">{" ".join(spans)}</div>',
        unsafe_allow_html=True,
    )


def predict_batch(texts: list, model_choice: str, preset: str):
    payload = {"texts": texts, "model_choice": model_choice, "threshold_preset": preset}
    try:
        r = httpx.post(f"{API_BASE}/api/v1/predict/batch", json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            st.error("Models are still loading on the backend — please retry in a few seconds.")
        else:
            st.error(f"Batch prediction failed (HTTP {e.response.status_code}).")
        return None
    except Exception:
        st.error("Could not reach the backend. Is it running?")
        return None


# ---- Local metric files (produced by scripts/evaluate.py & fairness_eval.py) ----
def _models_path(filename: str) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", filename)


@st.cache_data
def load_json_file(filename: str):
    try:
        with open(_models_path(filename), "r") as f:
            return json.load(f)
    except Exception:
        return None


def fmt(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def main():
    st.set_page_config(page_title="ToxGuard Content Flagger", page_icon="🛡️", layout="wide")
    inject_css()

    st.markdown('<h1 class="main-header">ToxGuard Content Flagger</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Label Toxicity Detection powered by Dual BiLSTM pipelines</p>', unsafe_allow_html=True)

    app_mode = st.radio("Navigation", ["Content Evaluation", "Model Performance"], horizontal=True, label_visibility="collapsed")
    st.markdown("---")

    if app_mode == "Content Evaluation":
        with st.sidebar:
            st.markdown("### ⚙️ Settings")
            model_label = st.radio("Model Selection", list(MODELS.keys()), index=2)
            model_choice = MODELS[model_label]

            filter_label = st.select_slider("Threshold Preset", options=list(FILTER_LEVELS.keys()), value="⚖️ Balanced")
            preset = FILTER_LEVELS[filter_label]

            explain = False
            if model_choice == "transformer":
                explain = st.checkbox(
                    "🔬 Explain prediction (attention rollout)", value=True,
                    help="Highlights which words most influenced the transformer's decision."
                )

            st.markdown("---")
            st.markdown("### 🎲 Sample Data")
            if st.button("🔀 Randomize Comment", type="primary", use_container_width=True):
                random_comment = fetch_random_comment()
                if random_comment:
                    st.session_state["text_input"] = random_comment.get("comment_text", "")
                    st.session_state["auto_predict"] = True
                    st.rerun()

            st.markdown("---")
            st.markdown("### 📡 Backend Status")
            try:
                health = httpx.get(f"{API_BASE}/health", timeout=5).json()
                if health.get("models_loaded"):
                    st.success(f"Connected — {health.get('models_count', 0)} models loaded")
                else:
                    st.warning(f"Backend up, models {health.get('status', 'loading')}…")
            except Exception:
                st.error("Backend unreachable")

            st.markdown("---")
            st.markdown("### 📂 Project")
            st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Source_Code-181717?logo=github&style=for-the-badge)](https://github.com/Priyanshukv06/ToxGuard_Multi-Label_Content_Flagger.git)")

        tab1, tab2 = st.tabs(["Single Comment Analysis", "Batch Analysis"])

        with tab1:
            text_input = st.text_area("Enter comment text:", value=st.session_state.get("text_input", ""), height=150)

            auto_predict = st.session_state.pop("auto_predict", False)

            if st.button("🚀 Analyze Text", type="primary", use_container_width=True) or auto_predict:
                if text_input.strip():
                    with st.spinner("Analyzing..."):
                        result, latency = predict(text_input, model_choice, preset, explain=explain)

                    if result:
                        st.markdown("---")

                        is_toxic = result["is_toxic"]
                        badge_class = "badge-risky" if is_toxic else "badge-safe"
                        status_text = "TOXIC CONTENT DETECTED" if is_toxic else "CONTENT SAFE"
                        latency_caption = f" · {latency * 1000:.0f} ms" if latency is not None else ""

                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center;">
                            <h2>Overall Verdict</h2>
                            <span class="badge {badge_class}" style="font-size: 1.2rem; padding: 10px 20px;">{status_text}</span>
                            <p style="margin-top: 10px; color: #a0a0b0;">Confidence: {result['confidence']}{latency_caption}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        if model_choice == "transformer" and result.get("explanation"):
                            render_explanation(result["explanation"])

                        st.markdown("### Subclass Breakdown")
                        cols = st.columns(3)

                        for i, cls in enumerate(CLASS_NAMES[1:]):
                            col = cols[i % 3]
                            with col:
                                st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                                render_risk_bar(
                                    result["probabilities"][cls],
                                    CLASS_LABELS[cls],
                                    result["thresholds_applied"][cls],
                                    result["flags"][cls]
                                )
                                st.markdown('</div>', unsafe_allow_html=True)

                        if model_choice == "both" and "detailed_results" in result and result["detailed_results"]:
                            st.markdown("---")
                            st.markdown("### Detailed Model Comparison")

                            m_res = result["detailed_results"]["multilabel"]
                            t_res = result["detailed_results"]["two_stage"]

                            comp_data = []
                            for cls in CLASS_NAMES:
                                comp_data.append({
                                    "Class": CLASS_LABELS[cls],
                                    "Multilabel Prob": f"{m_res['probabilities'][cls]:.2%}",
                                    "Multilabel Flag": "🔴" if m_res['flags'][cls] else "🟢",
                                    "Two-Stage Prob": f"{t_res['probabilities'][cls]:.2%}",
                                    "Two-Stage Flag": "🔴" if t_res['flags'][cls] else "🟢",
                                    "Final Flag (OR)": "🔴" if result['flags'][cls] else "🟢"
                                })

                            st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

        with tab2:
            st.markdown("### Batch Analysis")
            st.markdown("Upload a CSV file containing a `comment_text` column.")

            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                if "comment_text" not in df.columns:
                    st.error("CSV must contain a `comment_text` column.")
                else:
                    total = len(df)
                    if total > MAX_BATCH_ROWS:
                        st.warning(f"File has {total:,} rows — analyzing the first {MAX_BATCH_ROWS:,} to keep the request responsive.")
                        df = df.head(MAX_BATCH_ROWS).copy()
                    st.success(f"Loaded {len(df)} rows.")
                    if st.button("Run Batch Analysis"):
                        texts = df["comment_text"].fillna("").astype(str).tolist()
                        with st.spinner(f"Analyzing {len(texts)} comments..."):
                            batch_res = predict_batch(texts, model_choice, preset)

                        if batch_res:
                            preds = batch_res["predictions"]

                            out_df = df.copy()
                            out_df["is_toxic"] = [p["is_toxic"] for p in preds]
                            for cls in CLASS_NAMES:
                                out_df[f"{cls}_prob"] = [p["probabilities"][cls] for p in preds]
                                out_df[f"{cls}_flag"] = [p["flags"][cls] for p in preds]

                            st.markdown("### Results")
                            st.dataframe(out_df.head(50))

                            csv = out_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Full Results as CSV",
                                data=csv,
                                file_name='toxicity_analysis_results.csv',
                                mime='text/csv',
                            )

    elif app_mode == "Model Performance":
        st.markdown(
            '<style>[data-testid="stSidebar"] {display: none;}</style>',
            unsafe_allow_html=True,
        )
        st.markdown("## 🔍 Model Performance & Test Statistics")

        model_toggle = st.radio(
            "Select Model Metrics", ["Multi-Label BiLSTM", "Two-Stage BiLSTM", "Transformer (DistilBERT)"], horizontal=True
        )
        model_key = {"Multi-Label BiLSTM": "multilabel", "Two-Stage BiLSTM": "two_stage",
                     "Transformer (DistilBERT)": "transformer"}[model_toggle]

        # ---- Real held-out metrics (from scripts/evaluate.py) ----
        metrics = load_json_file("metrics.json")
        if not metrics or model_key not in metrics:
            st.info("Real metrics not found. Run `python scripts/evaluate.py` (env: toxguard-train) to generate `models/metrics.json`.")
        else:
            b = metrics[model_key]
            st.caption(f"Held-out test set: {metrics.get('test_size', 0):,} comments · generated {metrics.get('generated_at', 'n/a')}")

            st.markdown("### 📊 Threshold-free performance (ROC-AUC / PR-AUC)")
            auc_rows = [{
                "Category": CLASS_LABELS[c],
                "ROC-AUC": fmt(b["roc_auc"].get(c)),
                "PR-AUC": fmt(b["pr_auc"].get(c)),
            } for c in CLASS_NAMES]
            st.dataframe(pd.DataFrame(auc_rows), use_container_width=True, hide_index=True)

            st.markdown("### 🎚️ Per-preset performance")
            st.caption("Real macro/micro F1 and toxic flag-rate at each threshold preset.")
            preset_rows = []
            for name, m in b["presets"].items():
                preset_rows.append({
                    "Preset": name.capitalize(),
                    "Macro F1": f"{m['macro_f1']:.3f}",
                    "Micro F1": f"{m['micro_f1']:.3f}",
                    "Flagged Toxic %": f"{m['toxic_flag_rate'] * 100:.1f}%",
                })
            st.dataframe(pd.DataFrame(preset_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 📈 Precision, Recall, and F1 Score Curves")
        st.markdown("Explore the performance tradeoffs across all classification thresholds for each category.")

        pr_curves = load_json_file("pr_curves.json")
        if not pr_curves:
            st.info("PR curves not found. Run `python scripts/generate_pr_curves.py` to generate `models/pr_curves.json`.")
        else:
            selected_class = st.selectbox("Select Category", CLASS_NAMES, format_func=lambda x: CLASS_LABELS[x])
            curve_data = pr_curves.get(model_key, {}).get(selected_class)

            if curve_data:
                min_len = min(len(curve_data["thresholds"]), len(curve_data["precision"]))
                df_chart = pd.DataFrame({
                    "Precision": curve_data["precision"][:min_len],
                    "Recall": curve_data["recall"][:min_len],
                    "F1 Score": curve_data["f1"][:min_len]
                }, index=curve_data["thresholds"][:min_len])
                st.line_chart(df_chart, height=400)

        # ---- Fairness / unintended bias (from scripts/fairness_eval.py) ----
        st.markdown("---")
        st.markdown("### ⚖️ Fairness / Unintended Bias")
        fairness = load_json_file("fairness.json")
        if not fairness or model_key not in fairness:
            st.info("Fairness metrics not found. Run `python scripts/fairness_eval.py` (env: toxguard-train) to generate `models/fairness.json`.")
        else:
            fb = fairness[model_key]
            st.caption(
                f"{fairness.get('rows_with_identity', 0):,} test comments carry identity annotations · "
                f"subgroups with ≥ {fairness.get('min_subgroup_size', 0)} examples"
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall AUC", fmt(fb.get("overall_toxicity_auc")))
            c2.metric("Mean Subgroup AUC", fmt(fb.get("mean_subgroup_auc")))
            c3.metric("Mean BPSN AUC", fmt(fb.get("mean_bpsn_auc")))
            c4.metric("Mean BNSP AUC", fmt(fb.get("mean_bnsp_auc")))
            st.caption("Low BPSN ⇒ over-flags non-toxic comments mentioning a group (false positives). "
                       "Low BNSP ⇒ misses toxic comments mentioning a group (false negatives).")

            subgroups = fb.get("subgroups", {})
            if subgroups:
                sg_rows = [{
                    "Identity Subgroup": g.replace("_", " ").title(),
                    "n": m["n"],
                    "Subgroup AUC": fmt(m["subgroup_auc"]),
                    "BPSN AUC": fmt(m["bpsn_auc"]),
                    "BNSP AUC": fmt(m["bnsp_auc"]),
                } for g, m in subgroups.items()]
                st.dataframe(pd.DataFrame(sg_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
