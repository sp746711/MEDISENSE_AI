
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = (
    BASE_DIR / "data" / "symptom_to_disease" / "Symptom2Disease.csv"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "models" / "conditions_v2_audit"
MODEL_VERSION = "condition_pred_v2.0-audit"
SIMILARITY_THRESHOLD = 0.97
REJECTION_THRESHOLD = 0.25


def clean_text(text: Any) -> str:
    """Normalize text consistently without changing its clinical meaning."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().strip().split())


def build_similarity_groups(
    texts: list[str],
    labels: list[str],
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[list[int], dict[str, Any]]:
    """
    Group near-duplicate descriptions using character TF-IDF similarity.

    Groups are connected components of pairs whose cosine similarity
    meets the threshold. If a component contains different labels, fail
    safely rather than merging conflicting disease labels.
    """
    if not 0 < threshold <= 1:
        raise ValueError("Similarity threshold must be in (0, 1].")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts).tocsr()
    n = len(texts)

    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        if rank[root_a] == rank[root_b]:
            rank[root_a] += 1

    # Sparse multiplication avoids constructing a dense N x N matrix.
    block_size = 128
    pair_count = 0
    cross_label_pairs: list[dict[str, Any]] = []

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        similarities = (matrix[start:end] @ matrix.T).tocoo()

        for local_i, j, score in zip(
            similarities.row, similarities.col, similarities.data
        ):
            i = start + int(local_i)
            j = int(j)

            if i >= j or float(score) < threshold:
                continue

            pair_count += 1

            if labels[i] != labels[j]:
                cross_label_pairs.append(
                    {
                        "left_index": i,
                        "right_index": j,
                        "similarity": round(float(score), 4),
                        "left_label": labels[i],
                        "right_label": labels[j],
                    }
                )
            else:
                union(i, j)

    if cross_label_pairs:
        raise ValueError(
            "High-similarity descriptions with conflicting labels were found. "
            "Review these records before training; no model was saved.\n"
            + json.dumps(cross_label_pairs[:20], indent=2)
        )

    root_to_group: dict[int, int] = {}
    groups: list[int] = []

    for i in range(n):
        root = find(i)
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group)
        groups.append(root_to_group[root])

    sizes = pd.Series(groups).value_counts()
    group_info = {
        "method": "character TF-IDF cosine similarity connected components",
        "similarity_threshold": threshold,
        "near_duplicate_pairs_found": pair_count,
        "group_count": int(len(sizes)),
        "largest_group_size": int(sizes.max()),
        "groups_larger_than_one": int((sizes > 1).sum()),
        "cross_label_high_similarity_pairs": 0,
    }
    return groups, group_info


def group_aware_split(
    labels: list[str],
    groups: list[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create stratified 70/15/15 splits without separating similarity groups.

    Splitting is performed at the group level, then converted back to row
    indices. This avoids the old greedy algorithm's severe split imbalance.
    Row percentages can differ slightly from 70/15/15 when groups have
    different sizes, because every group must remain intact.
    """
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have the same length.")
    if not labels:
        raise ValueError("Cannot split an empty dataset.")

    group_rows: dict[int, list[int]] = {}
    for row_index, group_id in enumerate(groups):
        group_rows.setdefault(group_id, []).append(row_index)

    group_ids = list(group_rows)
    group_labels: dict[int, str] = {}

    # Every similarity group must represent exactly one class.
    for group_id, rows in group_rows.items():
        unique_labels = {labels[i] for i in rows}
        if len(unique_labels) != 1:
            raise ValueError(
                f"Similarity group {group_id} contains conflicting labels: "
                f"{sorted(unique_labels)}. No model was saved."
            )
        group_labels[group_id] = next(iter(unique_labels))

    group_targets = [group_labels[group_id] for group_id in group_ids]
    if len(set(group_targets)) < 2:
        raise ValueError("Stratified splitting requires at least two classes.")

    # First hold out 15% of groups for the final test set.
    try:
        train_val_groups, test_groups = train_test_split(
            group_ids,
            test_size=0.15,
            random_state=seed,
            stratify=group_targets,
        )

        # Remaining 85%: reserve 15% of the whole dataset for validation.
        train_val_targets = [group_labels[group_id] for group_id in train_val_groups]
        train_groups, val_groups = train_test_split(
            train_val_groups,
            test_size=(0.15 / 0.85),
            random_state=seed,
            stratify=train_val_targets,
        )
    except ValueError as exc:
        raise ValueError(
            "Could not create stratified group-aware 70/15/15 splits. "
            "Check that every class has enough independent similarity groups. "
            f"Original error: {exc}"
        ) from exc

    def rows_for(selected_groups: list[int]) -> np.ndarray:
        return np.asarray(
            sorted(
                row_index
                for group_id in selected_groups
                for row_index in group_rows[group_id]
            ),
            dtype=int,
        )

    train_indices = rows_for(train_groups)
    val_indices = rows_for(val_groups)
    test_indices = rows_for(test_groups)

    all_labels = set(labels)
    for split_name, indices in (
        ("train", train_indices),
        ("validation", val_indices),
        ("test", test_indices),
    ):
        present_labels = {labels[i] for i in indices}
        missing = sorted(all_labels - present_labels)
        if missing:
            raise ValueError(
                f"Group-aware {split_name} split is missing classes: {missing}. "
                "No model was saved."
            )

    # Verify that no group appears in more than one split.
    train_group_set = {groups[i] for i in train_indices}
    val_group_set = {groups[i] for i in val_indices}
    test_group_set = {groups[i] for i in test_indices}
    if (
        train_group_set & val_group_set
        or train_group_set & test_group_set
        or val_group_set & test_group_set
    ):
        raise RuntimeError("Internal split error: a similarity group crosses splits.")

    return train_indices, val_indices, test_indices

def train_and_evaluate(
    data_path: Path,
    output_dir: Path,
    random_seed: int = 42,
) -> dict[str, Any]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    df = pd.read_csv(data_path)

    required = {"label", "text"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset missing required columns: {sorted(missing_columns)}. "
            f"Found columns: {df.columns.tolist()}"
        )

    raw_rows = len(df)
    df = df.dropna(subset=["label", "text"]).copy()
    df["label"] = df["label"].astype(str).str.strip()
    df["clean_text"] = df["text"].map(clean_text)
    df = df[
        (df["label"].str.len() > 0)
        & (df["clean_text"].str.len() > 5)
    ].copy()

    valid_rows_before_dedup = len(df)
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    duplicate_count = valid_rows_before_dedup - len(df)

    if df.empty:
        raise ValueError("No valid dataset rows remain after cleaning.")

    label_counts = df["label"].value_counts()
    if label_counts.min() < 3:
        raise ValueError(
            "Every class needs at least 3 unique descriptions for this audit."
        )

    texts = df["clean_text"].tolist()
    labels = df["label"].tolist()
    classes = sorted(set(labels))

    print(f"Dataset: {data_path}")
    print(f"Raw rows: {raw_rows}")
    print(f"Valid rows before exact deduplication: {valid_rows_before_dedup}")
    print(f"Exact duplicates removed: {duplicate_count}")
    print(f"Unique samples: {len(df)}")
    print(f"Disease classes: {len(classes)}")

    groups, group_info = build_similarity_groups(texts, labels)
    train_idx, val_idx, test_idx = group_aware_split(
        labels, groups, random_seed
    )

    X_train = [texts[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_val = [texts[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]
    X_test = [texts[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    print("Group-aware split sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    print("Near-duplicate grouping:", json.dumps(group_info, indent=2))

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                CalibratedClassifierCV(
                    estimator=LogisticRegression(
                        C=1.5,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_seed,
                    ),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    val_probs = pipeline.predict_proba(X_val)
    val_preds = pipeline.predict(X_val)
    val_report = classification_report(
        y_val, val_preds, output_dict=True, zero_division=0
    )
    val_max_probs = np.max(val_probs, axis=1)
    val_metrics = {
        "accuracy": round(float(accuracy_score(y_val, val_preds)), 4),
        "macro_f1": round(float(val_report["macro avg"]["f1-score"]), 4),
        "log_loss": round(
            float(log_loss(y_val, val_probs, labels=pipeline.classes_)), 4
        ),
        "min_max_prob": round(float(np.min(val_max_probs)), 4),
        "q01_max_prob": round(float(np.quantile(val_max_probs, 0.01)), 4),
        "q05_max_prob": round(float(np.quantile(val_max_probs, 0.05)), 4),
        "recommended_rejection_threshold": REJECTION_THRESHOLD,
    }

    test_probs = pipeline.predict_proba(X_test)
    test_preds = pipeline.predict(X_test)
    test_report = classification_report(
        y_test, test_preds, output_dict=True, zero_division=0
    )
    test_metrics = {
        "accuracy": round(float(accuracy_score(y_test, test_preds)), 4),
        "macro_precision": round(
            float(test_report["macro avg"]["precision"]), 4
        ),
        "macro_recall": round(float(test_report["macro avg"]["recall"]), 4),
        "macro_f1": round(float(test_report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(test_report["weighted avg"]["f1-score"]), 4),
        "test_log_loss": round(
            float(log_loss(y_test, test_probs, labels=pipeline.classes_)), 4
        ),
    }

    print("\nExperimental test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(
        "\nWARNING: These are benchmark text-classification metrics, "
        "not clinical diagnostic accuracy or clinically validated probabilities."
    )

    # Do not create output files until all splitting and evaluation checks pass.
    output_dir.mkdir(parents=True, exist_ok=True)
    classes_list = list(pipeline.classes_)

    eval_data = {
        "model_version": MODEL_VERSION,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_source": str(data_path),
        "evaluation_note": (
            "Experimental group-aware evaluation; not clinically validated."
        ),
        "deduplication": {
            "raw_samples": raw_rows,
            "duplicates_removed": duplicate_count,
            "unique_samples": len(df),
        },
        "similarity_grouping": group_info,
        "split_sizes": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
            "total": len(df),
        },
        "validation_metrics": val_metrics,
        "overall_metrics": test_metrics,
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
        "confusion_matrix": confusion_matrix(
            y_test, test_preds, labels=classes_list
        ).tolist(),
        "classes": classes_list,
    }

    joblib.dump(pipeline, output_dir / "condition_pipeline.joblib")
    (output_dir / "condition_classes.json").write_text(
        json.dumps(classes_list, indent=2), encoding="utf-8"
    )
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(eval_data, indent=2), encoding="utf-8"
    )
    (output_dir / "training_config.json").write_text(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "random_seed": random_seed,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "target_count": len(classes_list),
                "rejection_threshold": REJECTION_THRESHOLD,
                "model_type": "TF-IDF + calibrated Logistic Regression",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nExperimental artifacts saved to: {output_dir.resolve()}")
    return eval_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimental group-aware symptom-condition model audit"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_and_evaluate(args.data, args.output, args.seed)


if __name__ == "__main__":
    main()
