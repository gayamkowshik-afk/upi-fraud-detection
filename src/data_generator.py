"""
data_generator.py
-----------------
Generates a synthetic UPI transaction dataset (100K+ records) with realistic
fraud patterns, saving to data/upi_transactions.csv and data/upi_transactions.json.
"""

import numpy as np
import pandas as pd
import json
import re
import os
from datetime import datetime, timedelta

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_TRANSACTIONS = 100_000
FRAUD_RATE = 0.035  # 3.5% fraud rate (realistic for UPI ecosystem)

BANKS = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "BOB", "Canara", "Union", "Yes"]
CATEGORIES = ["Retail", "Food", "Travel", "Utilities", "Entertainment",
               "Healthcare", "Education", "Recharge", "Insurance", "Investment"]
UPI_REGEX = re.compile(r"^[a-zA-Z0-9._-]+@[a-zA-Z]{3,}$")

def generate_upi_id(is_fraudulent: bool = False) -> str:
    """Generate realistic UPI IDs; fraudulent ones use suspicious patterns."""
    if is_fraudulent and np.random.rand() < 0.3:
        # Suspicious: random numeric heavy or newly registered patterns
        username = f"usr{np.random.randint(10000,99999)}{np.random.choice(['x','z','q'])}"
    else:
        names = ["rahul", "priya", "amit", "sneha", "vikram", "kavya",
                 "suresh", "deepa", "ravi", "anita", "mohan", "pooja"]
        username = np.random.choice(names) + str(np.random.randint(1, 9999))
    handles = ["okaxis", "oksbi", "okhdfcbank", "okicici", "ybl", "paytm", "gpay"]
    return f"{username}@{np.random.choice(handles)}"

def generate_transactions() -> pd.DataFrame:
    """Build the full synthetic transaction DataFrame."""
    n_fraud = int(N_TRANSACTIONS * FRAUD_RATE)
    n_legit = N_TRANSACTIONS - n_fraud
    labels = [0] * n_legit + [1] * n_fraud
    np.random.shuffle(labels)

    base_time = datetime(2024, 1, 1)
    records = []

    for i, label in enumerate(labels):
        is_fraud = label == 1

        # Timestamp: fraud skews toward late night (00:00–04:00)
        if is_fraud and np.random.rand() < 0.55:
            hour = np.random.randint(0, 4)
        else:
            hour = np.random.choice(range(24), p=_hour_distribution())
        ts = base_time + timedelta(
            days=np.random.randint(0, 365),
            hours=hour,
            minutes=np.random.randint(0, 60),
            seconds=np.random.randint(0, 60)
        )

        # Amount: fraud tends to be high-value or micro-test transactions
        if is_fraud:
            amount = (
                np.random.choice([
                    np.random.uniform(1, 10),        # micro-test
                    np.random.uniform(45000, 99999)  # high-value sweep
                ], p=[0.25, 0.75])
            )
        else:
            amount = np.abs(np.random.lognormal(mean=6.5, sigma=1.2))
            amount = min(amount, 100000)

        sender = generate_upi_id(is_fraud)
        receiver = generate_upi_id(False)

        # Device & location features
        new_device = 1 if (is_fraud and np.random.rand() < 0.65) else int(np.random.rand() < 0.05)
        location_mismatch = 1 if (is_fraud and np.random.rand() < 0.6) else int(np.random.rand() < 0.04)

        # Velocity: number of transactions by same sender in last hour (simulated)
        velocity = np.random.randint(5, 30) if is_fraud else np.random.randint(1, 8)

        # Previous fraud flag on sender account
        prior_fraud = 1 if (is_fraud and np.random.rand() < 0.4) else int(np.random.rand() < 0.01)

        bank = np.random.choice(BANKS)
        category = np.random.choice(CATEGORIES)
        failure_count = np.random.randint(2, 8) if is_fraud else np.random.randint(0, 2)

        records.append({
            "transaction_id": f"TXN{i:08d}",
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "sender_upi": sender,
            "receiver_upi": receiver,
            "amount": round(amount, 2),
            "bank": bank,
            "category": category,
            "hour_of_day": hour,
            "is_weekend": int(ts.weekday() >= 5),
            "new_device": new_device,
            "location_mismatch": location_mismatch,
            "velocity_1hr": velocity,
            "prior_fraud_flag": prior_fraud,
            "failure_count_24hr": failure_count,
            "is_fraud": label
        })

    df = pd.DataFrame(records)
    # Validate UPI format with regex
    df["sender_upi_valid"] = df["sender_upi"].apply(lambda x: int(bool(UPI_REGEX.match(x))))
    df["receiver_upi_valid"] = df["receiver_upi"].apply(lambda x: int(bool(UPI_REGEX.match(x))))
    return df

def _hour_distribution() -> list:
    """Realistic hour-of-day probability for legitimate transactions."""
    weights = np.array([
        0.5, 0.3, 0.2, 0.2, 0.3, 0.8,   # 00-05
        2.0, 4.0, 5.0, 5.5, 5.5, 5.5,   # 06-11
        6.0, 5.5, 5.0, 5.0, 5.5, 6.5,   # 12-17
        7.0, 7.5, 6.5, 5.0, 3.0, 1.5    # 18-23
    ])
    return (weights / weights.sum()).tolist()

def save_json_sample(df: pd.DataFrame, path: str, n: int = 1000) -> None:
    """Save a stratified JSON sample (for API / downstream integration demo)."""
    sample = pd.concat([
        df[df.is_fraud == 1].sample(min(n // 2, df.is_fraud.sum()), random_state=RANDOM_SEED),
        df[df.is_fraud == 0].sample(n // 2, random_state=RANDOM_SEED)
    ]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "record_count": len(sample),
        "transactions": json.loads(sample.to_json(orient="records"))
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  JSON sample saved → {path} ({len(sample)} records)")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print("Generating 100K UPI transactions...")
    df = generate_transactions()
    csv_path = "data/upi_transactions.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved → {csv_path} ({len(df):,} rows, {df.is_fraud.sum():,} fraud)")
    save_json_sample(df, "data/upi_transactions_sample.json")
    print("Done.")
