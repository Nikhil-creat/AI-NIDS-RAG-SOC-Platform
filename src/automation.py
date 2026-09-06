"""
automation.py
Automated response layer. Simulates SOC-style actions when a threat is
detected: severity scoring, "blocking" flagged IPs, and logging incidents
to a local SQLite database (acts as an auto-generated incident ticket log).
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "incidents.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source_ip TEXT,
            attack_type TEXT,
            severity TEXT,
            action_taken TEXT,
            ai_summary TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def score_severity(confidence, is_anomaly):
    """Simple rule-based severity scoring — swap for a trained model later."""
    if confidence > 0.9 and is_anomaly:
        return "Critical"
    elif confidence > 0.7:
        return "High"
    elif confidence > 0.5:
        return "Medium"
    return "Low"


def auto_respond(source_ip, attack_type, confidence, is_anomaly, ai_summary=""):
    """
    Decides and logs an automated action based on severity.
    In a real deployment this would call a firewall/SIEM API — here it's
    simulated and logged, which is enough to demonstrate the automation
    pipeline end-to-end.
    """
    severity = score_severity(confidence, is_anomaly)

    if severity == "Critical":
        action = f"Auto-blocked IP {source_ip}"
    elif severity == "High":
        action = f"Flagged IP {source_ip} for analyst review"
    else:
        action = "Logged for monitoring"

    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO incidents (timestamp, source_ip, attack_type, severity, action_taken, ai_summary)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (datetime.now().isoformat(), source_ip, attack_type, severity, action, ai_summary),
    )
    conn.commit()
    conn.close()

    return {
        "source_ip": source_ip,
        "attack_type": attack_type,
        "severity": severity,
        "action_taken": action,
    }


def get_recent_incidents(limit=20):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return rows
