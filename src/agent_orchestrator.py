"""
agent_orchestrator.py
Orchestrates the full multi-agent pipeline:

  Detection Agent   -> classifies traffic (RF + Isolation Forest)
  Threat Intel Agent -> RAG lookup for context on the detected attack type
  Response Agent    -> scores severity and takes an automated action

This turns the project from "just a classifier" into an agentic,
automated SOC-style pipeline.
"""

import os
import joblib
import numpy as np

from rag_engine import ask_threat_intel
from automation import auto_respond

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class DetectionAgent:
    def __init__(self):
        self.clf = joblib.load(os.path.join(MODEL_DIR, "rf_classifier.pkl"))
        self.iso = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))

    def analyze(self, features_row):
        X = features_row.values.reshape(1, -1)
        pred = self.clf.predict(X)[0]
        confidence = self.clf.predict_proba(X)[0][1]
        is_anomaly = self.iso.predict(X)[0] == -1
        return {
            "prediction": "Attack" if pred == 1 else "Benign",
            "confidence": float(confidence),
            "is_anomaly": bool(is_anomaly),
        }


class ThreatIntelAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def lookup(self, attack_type_query):
        summary, context = ask_threat_intel(attack_type_query, self.api_key)
        return summary, context


class ResponseAgent:
    def act(self, source_ip, attack_type, confidence, is_anomaly, ai_summary):
        return auto_respond(source_ip, attack_type, confidence, is_anomaly, ai_summary)


def run_pipeline(features_row, source_ip, gemini_api_key, attack_query="suspicious network traffic"):
    """
    Runs the full Detection -> Threat Intel -> Response pipeline for one
    traffic record and returns a combined result dict.
    """
    detector = DetectionAgent()
    result = detector.analyze(features_row)

    ai_summary = ""
    context = []
    if result["prediction"] == "Attack":
        intel_agent = ThreatIntelAgent(gemini_api_key)
        ai_summary, context = intel_agent.lookup(attack_query)

    response_agent = ResponseAgent()
    action = response_agent.act(
        source_ip=source_ip,
        attack_type=attack_query,
        confidence=result["confidence"],
        is_anomaly=result["is_anomaly"],
        ai_summary=ai_summary,
    )

    return {
        "detection": result,
        "threat_intel_summary": ai_summary,
        "threat_intel_context": context,
        "response": action,
    }
