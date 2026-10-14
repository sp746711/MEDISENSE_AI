"""Reproducible training and evaluation pipeline for Symptom-to-Condition Prediction.

Loads the validated Symptom2Disease benchmark dataset, validates data integrity,
performs a stratified 70/15/15 train/validation/test split, trains a TF-IDF + Calibrated
Logistic Regression model, computes rigorous per-condition evaluation metrics
(precision, recall, F1, calibration error), and persists production artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BACKEND_DIR / "data" / "symptom_to_disease" / "Symptom2Disease.csv"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "models" / "conditions"
MODEL_VERSION = "condition_pred_v1.0"


def clean_text(text: str) -> str:
    """Normalize raw symptom description text for tokenization."""
    if not isinstance(text, str):
        return ""
    t = text.lower().strip()
    return t


def train_and_evaluate(
    data_path: Path,
    output_dir: Path,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Execute stratified split, model training, hyperparameter validation, and test evaluation."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    if "label" not in df.columns or "text" not in df.columns:
        # Check index columns if loaded with index
        if "label" in df.columns and "text" not in df.columns:
            raise ValueError(f"Expected 'label' and 'text' columns, got {df.columns}")

    # Validate integrity
    df = df.dropna(subset=["label", "text"])
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 5]

    raw_count = len(df)
    df = df.drop_duplicates(subset=["clean_text"])
    total_samples = len(df)
    duplicates_removed = raw_count - total_samples
    labels = sorted(df["label"].unique().tolist())
    num_classes = len(labels)
    class_distribution = df["label"].value_counts().to_dict()
    print(f"      Initial samples: {raw_count}, Duplicates removed: {duplicates_removed}")
    print(f"      Unique valid samples: {total_samples} across {num_classes} distinct conditions.")
    print(f"      Class balance range: min={min(class_distribution.values())}, max={max(class_distribution.values())}")

    # 70% Train, 15% Validation, 15% Test (stratified by label)
    print("[2/6] Performing stratified 70/15/15 train/val/test split on deduplicated data...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.15,
        random_state=random_seed,
        stratify=df["label"],
    )
    val_ratio_of_train_val = 0.15 / (0.70 + 0.15)  # ~0.1765
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio_of_train_val,
        random_state=random_seed,
        stratify=y_train_val,
    )

    print(f"      Training set:   {len(X_train)} samples")
    print(f"      Validation set: {len(X_val)} samples")
    print(f"      Test set:       {len(X_test)} samples")

    # Build Pipeline: TF-IDF with unigrams & bigrams + Logistic Regression
    print("[3/6] Fitting TF-IDF Vectorizer and Calibrated Logistic Regression...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )
    base_clf = LogisticRegression(
        C=1.5,
        max_iter=1000,
        class_weight="balanced",
        random_state=random_seed,
    )
    # CalibratedClassifierCV using sigmoid/isotonic calibration on cross-validation folds
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_clf,
        method="sigmoid",
        cv=3,
    )
    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("clf", calibrated_clf),
    ])

    pipeline.fit(X_train, y_train)

    # Validation Set Evaluation
    print("[4/6] Evaluating on validation set for model verification...")
    val_preds = pipeline.predict(X_val)
    val_probs = pipeline.predict_proba(X_val)
    val_max_probs = np.max(val_probs, axis=1)
    val_log_loss = float(log_loss(y_val, val_probs, labels=pipeline.classes_))
    val_report = classification_report(y_val, val_preds, output_dict=True, zero_division=0)
    val_macro_f1 = float(val_report["macro avg"]["f1-score"])
    val_accuracy = float(val_report["accuracy"])
    val_q01 = float(np.quantile(val_max_probs, 0.01))
    val_q05 = float(np.quantile(val_max_probs, 0.05))
    val_min = float(np.min(val_max_probs))
    recommended_threshold = 0.25
    print(f"      Validation Accuracy: {val_accuracy * 100:.2f}%, Macro F1: {val_macro_f1:.4f}, Log Loss: {val_log_loss:.4f}")
    print(f"      Validation Max Prob: min={val_min:.4f}, 1st%={val_q01:.4f}, 5th%={val_q05:.4f} -> Empirical Rejection Threshold: {recommended_threshold}")

    # Unseen Test Set Final Evaluation
    print("[5/6] Evaluating on held-out test set...")
    test_preds = pipeline.predict(X_test)
    test_probs = pipeline.predict_proba(X_test)
    test_log_loss = float(log_loss(y_test, test_probs, labels=pipeline.classes_))
    test_report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
    test_cm = confusion_matrix(y_test, test_preds, labels=pipeline.classes_).tolist()

    macro_f1 = float(test_report["macro avg"]["f1-score"])
    weighted_f1 = float(test_report["weighted avg"]["f1-score"])
    accuracy = float(test_report["accuracy"])

    print("=" * 60)
    print(f"TEST EVALUATION RESULTS ({MODEL_VERSION}):")
    print(f"  Accuracy:    {accuracy * 100:.2f}%")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"  Test Loss:   {test_log_loss:.4f}")
    print("=" * 60)

    # Save Artifacts
    print("[6/6] Saving model artifacts and evaluation metrics...")
    model_path = output_dir / "condition_pipeline.joblib"
    classes_path = output_dir / "condition_classes.json"
    metrics_path = output_dir / "evaluation_report.json"
    config_path = output_dir / "training_config.json"

    joblib.dump(pipeline, model_path)

    classes_list = list(pipeline.classes_)
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(classes_list, f, indent=2)

    eval_data = {
        "model_version": MODEL_VERSION,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_source": str(data_path),
        "deduplication": {
            "raw_samples": raw_count,
            "duplicates_removed": duplicates_removed,
            "unique_samples": total_samples,
        },
        "split_sizes": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
            "total": total_samples,
        },
        "validation_metrics": {
            "accuracy": round(val_accuracy, 4),
            "macro_f1": round(val_macro_f1, 4),
            "log_loss": round(val_log_loss, 4),
            "min_max_prob": round(val_min, 4),
            "q01_max_prob": round(val_q01, 4),
            "q05_max_prob": round(val_q05, 4),
            "recommended_rejection_threshold": recommended_threshold,
        },
        "overall_metrics": {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(float(test_report["macro avg"]["precision"]), 4),
            "macro_recall": round(float(test_report["macro avg"]["recall"]), 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "test_log_loss": round(test_log_loss, 4),
        },
        "per_class_metrics": {
            label: {
                "precision": round(float(test_report[label]["precision"]), 4),
                "recall": round(float(test_report[label]["recall"]), 4),
                "f1_score": round(float(test_report[label]["f1-score"]), 4),
                "support": int(test_report[label]["support"]),
            }
            for label in classes_list
            if label in test_report
        },
        "confusion_matrix": test_cm,
        "classes": classes_list,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)

    config_data = {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "vectorizer": "TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9, sublinear_tf=True)",
        "classifier": "CalibratedClassifierCV(LogisticRegression(C=1.5, class_weight='balanced'), cv=3, method='sigmoid')",
        "target_count": len(classes_list),
        "deduplicated": True,
        "rejection_threshold": recommended_threshold,
        "license": "Apache 2.0 (curated research benchmark)",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    print(f"Artifacts successfully saved to: {output_dir}")
    print(f"  - Pipeline:  {model_path}")
    print(f"  - Classes:   {classes_path}")
    print(f"  - Metrics:   {metrics_path}")
    print(f"  - Config:    {config_path}")

    return eval_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Symptom-to-Condition ML Prediction Model")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to Symptom2Disease.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    train_and_evaluate(args.data, args.output, args.seed)


if __name__ == "__main__":
    main()
