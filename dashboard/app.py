"""
app.py
Streamlit dashboard for the AI-Powered Network Intrusion Detection,
Threat Intelligence (RAG) & Automated Response System.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

st.set_page_config(page_title="AI-NIDS + RAG SOC Assistant", layout="wide")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

st.title("🛡️ AI-Powered NIDS + Threat Intel (RAG) + Automation")

tab1, tab2, tab3 = st.tabs(["📊 Live Detection", "🤖 SOC AI Assistant", "⚡ Automation Log"])

# ---------------- TAB 1: Live Detection ----------------
with tab1:
    st.sidebar.header("Upload Traffic Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV of network flow features", type=["csv"])

    col1, col2, col3 = st.columns(3)

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        try:
            clf = joblib.load(os.path.join(MODEL_DIR, "rf_classifier.pkl"))
            iso = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))

            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
            preds = clf.predict(df[feature_cols])
            anomaly_scores = iso.predict(df[feature_cols])

            df["Prediction"] = np.where(preds == 1, "Attack", "Benign")
            df["Anomaly"] = np.where(anomaly_scores == -1, "Yes", "No")

            col1.metric("Total Records Scanned", len(df))
            col2.metric("Threats Detected", int((df["Prediction"] == "Attack").sum()))
            col3.metric("Anomalies Flagged", int((df["Anomaly"] == "Yes").sum()))

            st.subheader("Prediction Breakdown")
            fig, ax = plt.subplots()
            df["Prediction"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
            st.pyplot(fig)

            st.subheader("Detection Log")
            st.dataframe(df[["Prediction", "Anomaly"]].join(df[feature_cols].head()))

        except FileNotFoundError:
            st.warning("No trained model found yet. Run src/train_model.py first.")
    else:
        st.info("Upload a CSV file from the sidebar to see live predictions.")

# ---------------- TAB 2: SOC AI Assistant (RAG) ----------------
with tab2:
    st.subheader("Ask the Threat Intelligence Assistant")
    st.caption("Grounded in a MITRE ATT&CK-style knowledge base via RAG (ChromaDB + Gemini)")

    api_key = st.text_input("Gemini API Key", type="password", help="Or set GEMINI_API_KEY as an env var")
    query = st.text_input("Ask about a detected attack type (e.g. 'What is a port scan and how do I mitigate it?')")

    if st.button("Ask") and query:
        try:
            from rag_engine import ask_threat_intel
            key = api_key or os.environ.get("GEMINI_API_KEY", "")
            answer, context = ask_threat_intel(query, key)
            st.markdown("**Answer:**")
            st.write(answer)
            with st.expander("Retrieved context"):
                for c in context:
                    st.write("-", c)
        except Exception as e:
            st.error(f"Could not complete RAG lookup: {e}")

# ---------------- TAB 3: Automation Log ----------------
with tab3:
    st.subheader("Automated Response / Incident Log")
    try:
        from automation import get_recent_incidents
        rows = get_recent_incidents()
        if rows:
            log_df = pd.DataFrame(
                rows,
                columns=["ID", "Timestamp", "Source IP", "Attack Type", "Severity", "Action Taken", "AI Summary"],
            )
            st.dataframe(log_df)
        else:
            st.info("No incidents logged yet. Run the agent pipeline to generate automated responses.")
    except Exception as e:
        st.error(f"Could not load incident log: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("Detection: RF + Isolation Forest · Explainability: SHAP · Intel: RAG (ChromaDB + Gemini) · Automation: SQLite incident log")
