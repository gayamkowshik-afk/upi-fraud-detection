"""
predict.py
----------
Loads the saved best model and scores new UPI transactions.
Accepts a JSON payload (single transaction or batch) and returns
fraud probability + risk label + explanation.

Usage:
    python src/predict.py --input data/upi_transactions_sample.json --output reports/predictions.json
    python src/predict.py --single '{"amount":85000,"hour_of_day":2,"velocity_1hr":18,...}'
"""

import argparse
import json
import pickle
import re
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# ── Regex for UPI validation ──────────────────────────────────────────────────
UPI_REGEX = re.compile(r"^[a-zA-Z0-9._-]+@[a-zA-Z]{3,}$")
SUSPICIOUS_UPI_RE = re.compile(r"^(usr|anon|tmp|test|fake)[0-9]+[xzqXZQ]?@", re.IGNORECASE)
NUMERIC_HEAVY_RE = re.compile(r"^\d{6,}@")

RISK_THRESHOLD_HIGH   = 0.70
RISK_THRESHOLD_MEDIUM = 0.40

def load_model(path: str = "models/best_model.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

def build_single_feature_vector(txn: dict, feature_cols: list) -> np.ndarray:
    """
    Derive the same feature set used during training from a raw transaction dict.
    Missing fields default to safe values.
    """
    amount = float(txn.get("amount", 0))
    hour   = int(txn.get("hour_of_day", 12))
    sender = str(txn.get("sender_upi", ""))

    features = {
        "amount":               amount,
        "hour_of_day":          hour,
        "is_weekend":           int(txn.get("is_weekend", 0)),
        "new_device":           int(txn.get("new_device", 0)),
        "location_mismatch":    int(txn.get("location_mismatch", 0)),
        "velocity_1hr":         int(txn.get("velocity_1hr", 1)),
        "prior_fraud_flag":     int(txn.get("prior_fraud_flag", 0)),
        "failure_count_24hr":   int(txn.get("failure_count_24hr", 0)),
        "sender_upi_valid":     int(bool(UPI_REGEX.match(sender))),
        "receiver_upi_valid":   int(bool(UPI_REGEX.match(str(txn.get("receiver_upi", ""))))),
        "log_amount":           np.log1p(amount),
        "is_micro_txn":         int(amount < 10),
        "is_high_value":        int(amount > 40000),
        "is_late_night":        int(hour < 4 or hour >= 23),
        "suspicious_upi_pattern": int(bool(SUSPICIOUS_UPI_RE.match(sender))),
        "numeric_heavy_upi":    int(bool(NUMERIC_HEAVY_RE.match(sender))),
        "risk_score": (
            int(txn.get("new_device", 0)) * 2
            + int(txn.get("location_mismatch", 0)) * 2
            + int(txn.get("prior_fraud_flag", 0)) * 3
            + int(txn.get("failure_count_24hr", 0)) * 0.5
            + int(bool(SUSPICIOUS_UPI_RE.match(sender))) * 1.5
            + int(hour < 4 or hour >= 23) * 1
            + int(txn.get("velocity_1hr", 1)) * 0.3
        ),
        "rfm_total":        float(txn.get("rfm_total", 6)),
        "risk_segment_enc": int(txn.get("risk_segment_enc", 1)),
        "bank_enc":         int(txn.get("bank_enc", 0)),
        "category_enc":     int(txn.get("category_enc", 0)),
    }

    return np.array([features.get(col, 0) for col in feature_cols], dtype=float)

def explain(txn: dict, prob: float) -> list[str]:
    """Rule-based explanation of top fraud signals."""
    reasons = []
    if float(txn.get("amount", 0)) > 40000:
        reasons.append("High-value transaction (>₹40,000)")
    if int(txn.get("hour_of_day", 12)) < 4:
        reasons.append("Initiated during late-night hours (00:00–04:00)")
    if int(txn.get("new_device", 0)):
        reasons.append("New/unrecognised device used")
    if int(txn.get("location_mismatch", 0)):
        reasons.append("Location mismatch detected")
    if int(txn.get("prior_fraud_flag", 0)):
        reasons.append("Sender account has prior fraud flag")
    if int(txn.get("velocity_1hr", 1)) >= 10:
        reasons.append(f"High transaction velocity ({txn['velocity_1hr']} txns/hr)")
    if int(txn.get("failure_count_24hr", 0)) >= 3:
        reasons.append(f"Multiple recent failures ({txn['failure_count_24hr']} in 24hr)")
    sender = str(txn.get("sender_upi", ""))
    if SUSPICIOUS_UPI_RE.match(sender):
        reasons.append("Suspicious UPI ID pattern detected (regex match)")
    if float(txn.get("amount", 0)) < 10:
        reasons.append("Micro-transaction — possible card/account testing")
    return reasons or ["No dominant fraud signals; model flagged via combined risk score"]

def score_transactions(transactions: list[dict], bundle: dict) -> list[dict]:
    model        = bundle["model"]
    scaler       = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    model_name   = bundle["model_name"]

    results = []
    for txn in transactions:
        vec = build_single_feature_vector(txn, feature_cols).reshape(1, -1)
        vec_scaled = scaler.transform(vec)
        prob = model.predict_proba(vec_scaled)[0][1]

        if prob >= RISK_THRESHOLD_HIGH:
            risk_label = "HIGH"
        elif prob >= RISK_THRESHOLD_MEDIUM:
            risk_label = "MEDIUM"
        else:
            risk_label = "LOW"

        results.append({
            "transaction_id": txn.get("transaction_id", "N/A"),
            "fraud_probability": round(float(prob), 4),
            "risk_label": risk_label,
            "predicted_fraud": int(prob >= RISK_THRESHOLD_HIGH),
            "model_used": model_name,
            "scored_at": datetime.utcnow().isoformat() + "Z",
            "explanation": explain(txn, prob),
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="UPI Fraud Scorer")
    parser.add_argument("--input", help="Path to JSON file (with 'transactions' key)")
    parser.add_argument("--single", help="Single transaction as JSON string")
    parser.add_argument("--output", default="reports/predictions.json")
    parser.add_argument("--model", default="models/best_model.pkl")
    args = parser.parse_args()

    bundle = load_model(args.model)
    print(f"Loaded: {bundle['model_name']}")

    if args.single:
        txns = [json.loads(args.single)]
    elif args.input:
        with open(args.input) as f:
            data = json.load(f)
        txns = data.get("transactions", data) if isinstance(data, dict) else data
    else:
        print("Provide --input or --single"); sys.exit(1)

    print(f"Scoring {len(txns):,} transaction(s)...")
    results = score_transactions(txns, bundle)

    high_risk = sum(1 for r in results if r["risk_label"] == "HIGH")
    print(f"  HIGH risk: {high_risk} | MEDIUM: {sum(1 for r in results if r['risk_label']=='MEDIUM')} | LOW: {sum(1 for r in results if r['risk_label']=='LOW')}")

    output = {
        "scored_at": datetime.utcnow().isoformat() + "Z",
        "total": len(results),
        "high_risk_count": high_risk,
        "results": results
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Predictions saved → {args.output}")

if __name__ == "__main__":
    main()
