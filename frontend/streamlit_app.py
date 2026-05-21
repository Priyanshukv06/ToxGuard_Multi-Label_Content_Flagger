"""
ToxGuard Content Flagger — Streamlit Frontend
"""

import streamlit as st
import httpx
import pandas as pd
import io
import os
import json

API_BASE = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 60

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
    "Both (OR Ensemble)": "both"
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

def fetch_random_patient():
    try:
        r = httpx.get(f"{API_BASE}/api/v1/data/random", timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json().get("patient", {})
    except Exception as e:
        st.error(f"Failed to fetch random patient: {e}")
        return {}

def predict(text: str, model_choice: str, preset: str):
    payload = {"text": text, "model_choice": model_choice, "threshold_preset": preset}
    try:
        r = httpx.post(f"{API_BASE}/api/v1/predict", json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

def predict_batch(texts: list, model_choice: str, preset: str):
    payload = {"texts": texts, "model_choice": model_choice, "threshold_preset": preset}
    try:
        r = httpx.post(f"{API_BASE}/api/v1/predict/batch", json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Batch Prediction failed: {e}")
        return None

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

            st.markdown("---")
            st.markdown("### 🎲 Sample Data")
            if st.button("🔀 Randomize Comment", type="primary", use_container_width=True):
                random_patient = fetch_random_patient()
                if random_patient:
                    st.session_state["text_input"] = random_patient.get("comment_text", "")
                    st.session_state["auto_predict"] = True
                    st.rerun()

            st.markdown("---")
            st.markdown("### 📡 Backend Status")
            try:
                health = httpx.get(f"{API_BASE}/health", timeout=5).json()
                st.success(f"Connected — {health.get('models_count', 0)} models loaded")
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
                        result = predict(text_input, model_choice, preset)
                
                    if result:
                        st.markdown("---")
                    
                        is_toxic = result["is_toxic"]
                        badge_class = "badge-risky" if is_toxic else "badge-safe"
                        status_text = "TOXIC CONTENT DETECTED" if is_toxic else "CONTENT SAFE"
                    
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center;">
                            <h2>Overall Verdict</h2>
                            <span class="badge {badge_class}" style="font-size: 1.2rem; padding: 10px 20px;">{status_text}</span>
                            <p style="margin-top: 10px; color: #a0a0b0;">Confidence: {result['confidence']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
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
                            
                        if model_choice == "both" and "detailed_results" in result:
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
        st.markdown('<div class="section-title">🔍 Model Performance & Test Statistics</div>', unsafe_allow_html=True)
        st.caption("Aggregated performance of the ToxGuard models on the external test dataset.")

        model_toggle = st.radio("Select Model Metrics", ["Multi-Label BiLSTM", "Two-Stage BiLSTM"], horizontal=True)
        
        st.markdown("### 📊 Dual Risk Impact Analysis")
        st.markdown("The table below illustrates the impact of different threshold presets on the model's flagging behavior.")
        
        # Hardcoding the threshold data from our artifacts for display purposes
        if "Multi" in model_toggle:
            df_thresh = pd.DataFrame([
                {"Preset": "Lenient", "Toxicity Thresh": "0.65", "Sublabel Avg Thresh": "0.51", "Filtered %": "8.2%", "F1-Score": "0.78"},
                {"Preset": "Moderate", "Toxicity Thresh": "0.60", "Sublabel Avg Thresh": "0.46", "Filtered %": "10.1%", "F1-Score": "0.80"},
                {"Preset": "Balanced", "Toxicity Thresh": "0.55", "Sublabel Avg Thresh": "0.41", "Filtered %": "12.4%", "F1-Score": "0.82 (Optimal)"},
                {"Preset": "Cautious", "Toxicity Thresh": "0.50", "Sublabel Avg Thresh": "0.36", "Filtered %": "15.0%", "F1-Score": "0.79"},
                {"Preset": "Aggressive", "Toxicity Thresh": "0.45", "Sublabel Avg Thresh": "0.31", "Filtered %": "18.3%", "F1-Score": "0.75"}
            ])
        else:
            df_thresh = pd.DataFrame([
                {"Preset": "Lenient", "Toxicity Thresh": "0.65", "Sublabel Avg Thresh": "0.50", "Filtered %": "8.0%", "F1-Score": "0.79"},
                {"Preset": "Moderate", "Toxicity Thresh": "0.60", "Sublabel Avg Thresh": "0.45", "Filtered %": "9.8%", "F1-Score": "0.81"},
                {"Preset": "Balanced", "Toxicity Thresh": "0.55", "Sublabel Avg Thresh": "0.40", "Filtered %": "12.1%", "F1-Score": "0.83 (Optimal)"},
                {"Preset": "Cautious", "Toxicity Thresh": "0.50", "Sublabel Avg Thresh": "0.35", "Filtered %": "14.8%", "F1-Score": "0.80"},
                {"Preset": "Aggressive", "Toxicity Thresh": "0.45", "Sublabel Avg Thresh": "0.30", "Filtered %": "18.1%", "F1-Score": "0.76"}
            ])
            
        st.dataframe(df_thresh, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📈 Precision, Recall, and F1 Score Curves")
        st.markdown("Explore the performance tradeoffs across all possible classification thresholds for each individual category.")
        
        @st.cache_data
        def load_pr_curves():
            try:
                path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "pr_curves.json")
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Failed to load PR curves: {e}")
                return {}
                
        pr_curves = load_pr_curves()
        
        if pr_curves:
            model_key = "multilabel" if "Multi" in model_toggle else "two_stage"
            selected_class = st.selectbox("Select Category", CLASS_NAMES, format_func=lambda x: CLASS_LABELS[x])
            
            curve_data = pr_curves.get(model_key, {}).get(selected_class)
            
            if curve_data:
                # Truncate to match lengths (sometimes thresholds is length N-1 from sklearn)
                min_len = min(len(curve_data["thresholds"]), len(curve_data["precision"]))
                df_chart = pd.DataFrame({
                    "Precision": curve_data["precision"][:min_len],
                    "Recall": curve_data["recall"][:min_len],
                    "F1 Score": curve_data["f1"][:min_len]
                }, index=curve_data["thresholds"][:min_len])
                
                st.line_chart(df_chart, height=400)

if __name__ == "__main__":
    main()
