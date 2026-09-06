"""
eda.py
Feature analysis helpers: correlation, low-variance feature detection,
and an optional AI-generated natural-language threat summary via Gemini.
"""

import pandas as pd
import numpy as np


def top_correlated_features(df, target="Label_binary", n=15):
    corr = df.select_dtypes(include=[np.number]).corr()[target].sort_values(ascending=False)
    return corr.head(n), corr.tail(10)


def low_variance_features(df, threshold=0.01):
    variances = df.select_dtypes(include=[np.number]).var()
    return variances[variances < threshold].index.tolist()


def generate_ai_summary(df, corr, api_key, model_name="gemini-3.6-flash"):
    """
    Sends dataset statistics to Gemini and returns a short natural-language
    threat-landscape summary. Requires `google-generativeai` and a valid API key.
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    summary_stats = f"""
    Total records: {df.shape[0]}
    Attack distribution: {df['Label'].value_counts().to_dict()}
    Top correlated features with attacks: {corr.head(10).to_dict()}
    """

    prompt = f"""You are a cybersecurity analyst. Given this network traffic dataset
    summary, write a short professional threat-landscape report (4-5 bullet points)
    explaining what patterns are visible and which features look most suspicious.

    Data:
    {summary_stats}
    """

    response = model.generate_content(prompt)
    return response.text
