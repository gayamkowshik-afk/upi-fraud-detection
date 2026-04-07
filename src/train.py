"""
train.py
--------
Trains Logistic Regression, Random Forest, and SVM classifiers on the
preprocessed UPI fraud dataset. Evaluates each model and saves the best
one to models/best_model.pkl.

Usage:
    python src/train.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    classification_report
)

from preprocessing import build_features

# ── Model Definitions ─────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "SVM": SVC(
        C=1.5, kernel="rbf", gamma="scale",
        class_weight="balanced", probability=True, random_state=42
    ),
}

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": name,
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred),    4),
        "f1_score":  round(f1_score(y_test, y_pred),        4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_negative_rate": round(fn / (fn + tp), 4),
        "false_positive_rate": round(fp / (fp + tn), 4),
    }

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1-Score   : {metrics['f1_score']:.4f}")
    if metrics["roc_auc"]:
        print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"  False Neg↓ : {metrics['false_negative_rate']:.4f}  (fraud missed)")
    print(f"\n  Confusion Matrix:")
    print(f"               Predicted 0   Predicted 1")
    print(f"  Actual 0  :     {tn:>7,}       {fp:>7,}")
    print(f"  Actual 1  :     {fn:>7,}       {tp:>7,}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legit','Fraud'])}")

    return metrics


# ── Feature Importance (Random Forest) ───────────────────────────────────────
def log_feature_importance(model, feature_cols: list, top_n: int = 10) -> list:
    if not hasattr(model, "feature_importances_"):
        return []
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    top = imp.nlargest(top_n)
    print(f"\n  Top {top_n} Feature Importances (Random Forest):")
    for feat, val in top.items():
        bar = "█" * int(val * 200)
        print(f"    {feat:<32} {val:.4f}  {bar}")
    return [{"feature": k, "importance": round(v, 5)} for k, v in top.items()]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("Loading data...")
    df = pd.read_csv("data/upi_transactions.csv")
    print(f"  {len(df):,} transactions | {df.is_fraud.sum():,} fraud ({df.is_fraud.mean()*100:.2f}%)")

    print("\nPreprocessing + SMOTE...")
    X_train, X_test, y_train, y_test, scaler, feature_cols = build_features(df)

    all_metrics = []
    trained_models = {}

    for name, model in MODELS.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate(name, model, X_test, y_test)
        all_metrics.append(metrics)
        trained_models[name] = model

    # Feature importance for RF
    feat_imp = log_feature_importance(
        trained_models["Random Forest"], feature_cols
    )

    # Pick best model by F1 (minimises false negatives for fraud)
    best = max(all_metrics, key=lambda m: m["f1_score"])
    best_model = trained_models[best["model"]]
    print(f"\n★  Best Model: {best['model']}  (F1={best['f1_score']:.4f})")

    # Save model + scaler
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler,
                     "feature_cols": feature_cols, "model_name": best["model"]}, f)
    print("  Saved → models/best_model.pkl")

    # Save report
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset_size": len(df),
        "fraud_rate": round(df.is_fraud.mean(), 4),
        "best_model": best["model"],
        "models": all_metrics,
        "top_features": feat_imp
    }
    with open("reports/model_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("  Report → reports/model_report.json")

if __name__ == "__main__":
    main()
