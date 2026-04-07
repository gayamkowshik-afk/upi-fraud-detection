"""
tests/test_pipeline.py
----------------------
Unit tests for data generation, preprocessing, and inference pipeline.
Run with: pytest tests/ -v
"""

import sys
import os
import re
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from data_generator import generate_transactions, UPI_REGEX
from preprocessing import compute_rfm, engineer_features, encode_categoricals
from predict import build_single_feature_vector, explain, SUSPICIOUS_UPI_RE


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def small_df():
    """Generate a small transaction dataset for fast testing."""
    import numpy as np
    np.random.seed(42)
    # Patch N_TRANSACTIONS temporarily
    import data_generator as dg
    original = dg.N_TRANSACTIONS
    dg.N_TRANSACTIONS = 2000
    df = dg.generate_transactions()
    dg.N_TRANSACTIONS = original
    return df


@pytest.fixture(scope="module")
def feature_cols():
    from preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES
    return (
        NUMERIC_FEATURES
        + ["log_amount", "is_micro_txn", "is_high_value", "is_late_night",
           "suspicious_upi_pattern", "numeric_heavy_upi", "risk_score",
           "rfm_total", "risk_segment_enc"]
        + [c + "_enc" for c in CATEGORICAL_FEATURES]
    )


# ── Data Generation Tests ─────────────────────────────────────────────────────
class TestDataGeneration:
    def test_shape(self, small_df):
        assert len(small_df) == 2000

    def test_fraud_rate_reasonable(self, small_df):
        rate = small_df.is_fraud.mean()
        assert 0.02 <= rate <= 0.07, f"Fraud rate {rate:.3f} out of expected range"

    def test_required_columns(self, small_df):
        required = ["transaction_id", "amount", "is_fraud", "sender_upi",
                    "receiver_upi", "timestamp", "hour_of_day"]
        for col in required:
            assert col in small_df.columns, f"Missing column: {col}"

    def test_amounts_positive(self, small_df):
        assert (small_df.amount > 0).all()

    def test_hour_range(self, small_df):
        assert small_df.hour_of_day.between(0, 23).all()

    def test_upi_format(self, small_df):
        # Majority of generated UPI IDs should pass basic regex
        valid = small_df.sender_upi.apply(lambda x: bool(UPI_REGEX.match(str(x))))
        assert valid.mean() > 0.60

    def test_transaction_id_unique(self, small_df):
        assert small_df.transaction_id.is_unique

    def test_labels_binary(self, small_df):
        assert set(small_df.is_fraud.unique()).issubset({0, 1})


# ── Preprocessing Tests ───────────────────────────────────────────────────────
class TestPreprocessing:
    def test_rfm_output_columns(self, small_df):
        rfm = compute_rfm(small_df)
        for col in ["sender_upi", "recency", "frequency", "monetary", "rfm_total"]:
            assert col in rfm.columns

    def test_rfm_frequency_positive(self, small_df):
        rfm = compute_rfm(small_df)
        assert (rfm.frequency > 0).all()

    def test_engineer_features_adds_cols(self, small_df):
        rfm = compute_rfm(small_df)
        df_eng = engineer_features(small_df, rfm)
        for col in ["log_amount", "is_micro_txn", "is_high_value",
                    "is_late_night", "risk_score", "suspicious_upi_pattern"]:
            assert col in df_eng.columns, f"Missing engineered feature: {col}"

    def test_log_amount_non_negative(self, small_df):
        rfm = compute_rfm(small_df)
        df_eng = engineer_features(small_df, rfm)
        assert (df_eng.log_amount >= 0).all()

    def test_risk_score_non_negative(self, small_df):
        rfm = compute_rfm(small_df)
        df_eng = engineer_features(small_df, rfm)
        assert (df_eng.risk_score >= 0).all()

    def test_encode_categoricals(self, small_df):
        rfm = compute_rfm(small_df)
        df_eng = engineer_features(small_df, rfm)
        df_enc = encode_categoricals(df_eng)
        assert "bank_enc" in df_enc.columns
        assert "category_enc" in df_enc.columns
        assert df_enc.bank_enc.dtype in [np.int32, np.int64, int]


# ── Regex Pattern Tests ───────────────────────────────────────────────────────
class TestRegexPatterns:
    @pytest.mark.parametrize("upi,expected", [
        ("usr12345x@okaxis",  True),
        ("anon99999@ybl",     True),
        ("rahul123@oksbi",    False),
        ("priya@paytm",       False),
        ("fake001q@okhdfcbank", True),
    ])
    def test_suspicious_upi_regex(self, upi, expected):
        result = bool(SUSPICIOUS_UPI_RE.match(upi))
        assert result == expected, f"SUSPICIOUS_UPI_RE mismatch for '{upi}'"

    def test_upi_regex_valid(self):
        valid_ids = ["rahul123@oksbi", "priya.sharma@ybl", "user-1@paytm"]
        for uid in valid_ids:
            assert bool(UPI_REGEX.match(uid)), f"Expected valid: {uid}"

    def test_upi_regex_invalid(self):
        invalid_ids = ["notaupi", "@only", "no_at_sign"]
        for uid in invalid_ids:
            assert not bool(UPI_REGEX.match(uid)), f"Expected invalid: {uid}"


# ── Predict Module Tests ──────────────────────────────────────────────────────
class TestPredict:
    def test_feature_vector_length(self, feature_cols):
        txn = {
            "amount": 5000, "hour_of_day": 14, "is_weekend": 0,
            "new_device": 0, "location_mismatch": 0, "velocity_1hr": 3,
            "prior_fraud_flag": 0, "failure_count_24hr": 0,
            "sender_upi": "rahul42@oksbi", "receiver_upi": "shop@paytm",
            "sender_upi_valid": 1, "receiver_upi_valid": 1
        }
        vec = build_single_feature_vector(txn, feature_cols)
        assert vec.shape == (len(feature_cols),)

    def test_explain_high_value(self):
        txn = {"amount": 85000, "hour_of_day": 14}
        reasons = explain(txn, 0.8)
        assert any("High-value" in r for r in reasons)

    def test_explain_late_night(self):
        txn = {"amount": 1000, "hour_of_day": 2}
        reasons = explain(txn, 0.75)
        assert any("late-night" in r for r in reasons)

    def test_explain_micro_txn(self):
        txn = {"amount": 1, "hour_of_day": 10}
        reasons = explain(txn, 0.6)
        assert any("Micro" in r or "testing" in r for r in reasons)

    def test_explain_new_device(self):
        txn = {"amount": 500, "hour_of_day": 12, "new_device": 1}
        reasons = explain(txn, 0.5)
        assert any("device" in r.lower() for r in reasons)
