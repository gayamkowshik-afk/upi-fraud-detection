"""
preprocessing.py
----------------
Loads raw UPI transaction data, engineers features, handles class imbalance
via SMOTE, and returns train/test splits ready for model training.
"""

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# ── Regex patterns for UPI anomaly detection ─────────────────────────────────
SUSPICIOUS_UPI_RE = re.compile(
    r"^(usr|anon|tmp|test|fake)[0-9]+[xzqXZQ]?@",  # suspicious prefixes
    re.IGNORECASE
)
NUMERIC_HEAVY_RE = re.compile(r"^\d{6,}@")         # purely numeric usernames

NUMERIC_FEATURES = [
    "amount", "hour_of_day", "is_weekend", "new_device",
    "location_mismatch", "velocity_1hr", "prior_fraud_flag",
    "failure_count_24hr", "sender_upi_valid", "receiver_upi_valid"
]
CATEGORICAL_FEATURES = ["bank", "category"]
TARGET = "is_fraud"

# ── RFM Segmentation ──────────────────────────────────────────────────────────
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Recency-Frequency-Monetary (RFM) scores per sender.
    Assigns risk_segment: High / Medium / Low.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    snapshot = df["timestamp"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("sender_upi")
        .agg(
            recency=("timestamp", lambda x: (snapshot - x.max()).days),
            frequency=("transaction_id", "count"),
            monetary=("amount", "sum")
        )
        .reset_index()
    )

    for col, label in [("recency", "R"), ("frequency", "F"), ("monetary", "M")]:
        try:
            rfm[f"{label}_score"] = pd.qcut(rfm[col], q=4, labels=[4, 3, 2, 1]
                                            if label == "R" else [1, 2, 3, 4],
                                            duplicates="drop").astype(int)
        except ValueError:
            rfm[f"{label}_score"] = 2  # fallback if not enough distinct values

    rfm["rfm_total"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
    rfm["risk_segment"] = pd.cut(
        rfm["rfm_total"], bins=[0, 5, 9, 12],
        labels=["Low", "Medium", "High"], right=True
    )
    return rfm[["sender_upi", "recency", "frequency", "monetary", "rfm_total", "risk_segment"]]


# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    """Add derived features and merge RFM data onto transaction frame."""
    df = df.copy()

    # Amount-based features
    df["log_amount"] = np.log1p(df["amount"])
    df["is_micro_txn"] = (df["amount"] < 10).astype(int)
    df["is_high_value"] = (df["amount"] > 40000).astype(int)

    # Time-based features
    df["is_late_night"] = df["hour_of_day"].apply(lambda h: int(h < 4 or h >= 23))

    # UPI anomaly flags via regex
    df["suspicious_upi_pattern"] = df["sender_upi"].apply(
        lambda x: int(bool(SUSPICIOUS_UPI_RE.match(str(x))))
    )
    df["numeric_heavy_upi"] = df["sender_upi"].apply(
        lambda x: int(bool(NUMERIC_HEAVY_RE.match(str(x))))
    )

    # Risk score composite
    df["risk_score"] = (
        df["new_device"] * 2
        + df["location_mismatch"] * 2
        + df["prior_fraud_flag"] * 3
        + df["failure_count_24hr"] * 0.5
        + df["suspicious_upi_pattern"] * 1.5
        + df["is_late_night"] * 1
        + df["velocity_1hr"] * 0.3
    )

    # Merge RFM
    df = df.merge(rfm[["sender_upi", "rfm_total", "risk_segment"]], on="sender_upi", how="left")
    df["rfm_total"] = df["rfm_total"].fillna(rfm["rfm_total"].median())

    # Encode risk_segment
    seg_map = {"Low": 0, "Medium": 1, "High": 2}
    df["risk_segment_enc"] = df["risk_segment"].map(seg_map).fillna(1).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns."""
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    return df


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> tuple:
    """
    Full preprocessing pipeline.
    Returns: X_train_res, X_test, y_train_res, y_test, scaler, feature_cols
    """
    rfm = compute_rfm(df)
    df = engineer_features(df, rfm)
    df = encode_categoricals(df)

    feature_cols = (
        NUMERIC_FEATURES
        + ["log_amount", "is_micro_txn", "is_high_value", "is_late_night",
           "suspicious_upi_pattern", "numeric_heavy_upi", "risk_score",
           "rfm_total", "risk_segment_enc"]
        + [c + "_enc" for c in CATEGORICAL_FEATURES]
    )

    X = df[feature_cols].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Before SMOTE → Fraud: {y_train.sum():,} | Legit: {(y_train==0).sum():,}")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"  After  SMOTE → Fraud: {y_train_res.sum():,} | Legit: {(y_train_res==0).sum():,}")

    return X_train_res, X_test, y_train_res, y_test, scaler, feature_cols
