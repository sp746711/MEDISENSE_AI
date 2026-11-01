"""Multimodal Condition Prediction Training & Evaluation Pipeline for MediSense AI.

This module implements the end-to-end training, patient-level validation, and evaluation pipeline
for the MultimodalConditionPredictor architecture.

CLINICAL SAFETY & ETHICAL DATA RULE (Part 9 & 10):
1. Multimodal training strictly requires authentic, patient-level linked clinical EHR records
   (e.g., MIMIC-IV v2.2 linked to MIMIC-CXR JPG via subject_id and study_id).
2. NEVER join unrelated datasets by index, disease name, or assumptions.
3. NEVER fabricate synthetic patient associations or claim clinical validation without authentic data.
4. If authentic linked datasets are not present on disk, this script safely halts and
   outputs a structured blocked-audit manifest detailing the prerequisite data-use agreements
   (PhysioNet CITI training + DUA) required.
5. When authentic data is present, training strictly uses patient-level splitting (GroupKFold by subject_id)
   to eliminate patient-leakage across train/val/test partitions.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multimodal_training")

from app.ai.multimodal_condition_model import (
    CANONICAL_LAB_TESTS,
    CANONICAL_RAD_FINDINGS,
    CANONICAL_XRAY_PATHOLOGIES,
    CONDITION_CLASSES,
    MultimodalConditionPredictor,
)
DEFAULT_DATA_DIR = BASE_DIR / "data" / "multimodal_linked"
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "conditions_multimodal"
BASELINE_MODEL_PATH = BASE_DIR / "models" / "conditions" / "condition_pipeline.joblib"
MODEL_VERSION = "multimodal_gmu_v2.0"


class PatientLinkedMultimodalDataset(Dataset):
    """
    Dataset wrapper for genuine patient-level linked EHR records.
    Each sample represents an authentic patient encounter linked by subject_id.
    """

    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        return {
            "symptom_tokens": torch.tensor(rec["symptom_tokens"], dtype=torch.long),
            "symptom_features": torch.tensor(rec["symptom_features"], dtype=torch.float32),
            "lab_features": torch.tensor(rec["lab_features"], dtype=torch.float32),
            "rad_features": torch.tensor(rec["rad_features"], dtype=torch.float32),
            "xray_pathology_features": torch.tensor(rec["xray_pathology_features"], dtype=torch.float32),
            "xray_anatomy_probs": torch.tensor(rec.get("xray_anatomy_probs", [0.0] * 33), dtype=torch.float32),
            "xray_pathology_eligible": torch.tensor([rec["xray_pathology_eligible"]], dtype=torch.float32),
            "modality_mask": torch.tensor(rec["modality_mask"], dtype=torch.float32),
            "target": torch.tensor(rec["target_class_idx"], dtype=torch.long),
            "subject_id": str(rec.get("subject_id", idx)),
        }


def check_authentic_data_availability(data_dir: Path) -> dict[str, Any]:
    """Audit local filesystem for genuine patient-level linked multimodal cohorts."""
    status = {
        "is_available": False,
        "cohort_name": "None",
        "sample_count": 0,
        "patient_count": 0,
        "missing_prerequisites": [],
        "roadmap_action": "",
        "authenticity_rules": [
            "Patient-level linkage strictly enforced via authentic subject_id.",
            "No cross-joining unrelated datasets by row index, label, or synthetic association.",
            "PhysioNet credentialed Data Use Agreement (DUA) required for MIMIC-IV and MIMIC-CXR.",
        ],
    }

    linked_manifest = data_dir / "linked_patient_cohort.json"
    if not linked_manifest.exists():
        status["missing_prerequisites"] = [
            "Authentic EHR database link (e.g. MIMIC-IV v2.2 subject_id mapping).",
            "Paired thoracic/musculoskeletal radiographs (e.g. MIMIC-CXR JPG / DICOM linked by study_id).",
            "Extracted and de-identified radiology reports linked by study_id.",
            "Complete blood counts and inflammatory panels linked by hadm_id / charttime.",
            "PhysioNet Credentialed Data Use Agreement (DUA) approval.",
            "CITI Program 'Data or Specimens Only Research' certification.",
        ]
        status["roadmap_action"] = (
            "Complete CITI Data or Specimens Only Research training, obtain PhysioNet credentialed "
            "status for MIMIC-IV / MIMIC-CXR, sign DUA, download linked records to data/multimodal_linked/, "
            "and execute patient-partitioned training."
        )
        return status

    try:
        with open(linked_manifest, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("records", [])
        subjects = set(r.get("subject_id") for r in records if r.get("subject_id"))
        status["is_available"] = bool(records and len(subjects) > 0)
        status["cohort_name"] = data.get("cohort_name", "Verified Linked EHR")
        status["sample_count"] = len(records)
        status["patient_count"] = len(subjects)
    except Exception as e:
        logger.error(f"Failed to read linked cohort manifest: {e}")

    return status


def split_patient_cohort(
    records: list[dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Partition records strictly by patient (subject_id) to eliminate cross-split patient data leakage.
    """
    patient_records: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        sid = str(r.get("subject_id", id(r)))
        patient_records.setdefault(sid, []).append(r)

    unique_patients = list(patient_records.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_patients)

    n_patients = len(unique_patients)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)

    train_pats = set(unique_patients[:n_train])
    val_pats = set(unique_patients[n_train : n_train + n_val])
    test_pats = set(unique_patients[n_train + n_val :])

    train_records = [r for sid in train_pats for r in patient_records[sid]]
    val_records = [r for sid in val_pats for r in patient_records[sid]]
    test_records = [r for sid in test_pats for r in patient_records[sid]]

    # Sanity verify zero overlap in patients
    assert len(train_pats & val_pats) == 0, "Patient leakage between train and val!"
    assert len(train_pats & test_pats) == 0, "Patient leakage between train and test!"
    assert len(val_pats & test_pats) == 0, "Patient leakage between val and test!"

    return train_records, val_records, test_records


def evaluate_dataset_across_modality_subsets(
    model: MultimodalConditionPredictor,
    records: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """
    Evaluate the model on held-out test patients under different modality configurations:
    1. Symptoms only: [1, 0, 0, 0]
    2. Symptoms + labs: [1, 1, 0, 0]
    3. Symptoms + rad: [1, 0, 1, 0]
    4. Symptoms + xray: [1, 0, 0, 1]
    5. Symptoms + labs + rad: [1, 1, 1, 0]
    6. All available modalities (natural mask)
    """
    model.eval()
    dataset = PatientLinkedMultimodalDataset(records)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    eval_configs = {
        "symptoms_only": [1.0, 0.0, 0.0, 0.0],
        "symptoms_plus_labs": [1.0, 1.0, 0.0, 0.0],
        "symptoms_plus_radiology": [1.0, 0.0, 1.0, 0.0],
        "symptoms_plus_xray": [1.0, 0.0, 0.0, 1.0],
        "symptoms_labs_radiology": [1.0, 1.0, 1.0, 0.0],
        "all_available_modalities": None,  # natural mask from record
    }

    results: dict[str, Any] = {}

    for config_name, mask_override in eval_configs.items():
        all_preds: list[int] = []
        all_targets: list[int] = []

        with torch.no_grad():
            for batch in loader:
                sym_tok = batch["symptom_tokens"].to(device)
                sym_feat = batch["symptom_features"].to(device)
                lab_feat = batch["lab_features"].to(device)
                rad_feat = batch["rad_features"].to(device)
                xray_path = batch["xray_pathology_features"].to(device)
                xray_anat = batch["xray_anatomy_probs"].to(device)
                xray_elig = batch["xray_pathology_eligible"].to(device)

                if mask_override is not None:
                    b_size = sym_tok.size(0)
                    mask = torch.tensor([mask_override] * b_size, dtype=torch.float32, device=device)
                else:
                    mask = batch["modality_mask"].to(device)

                targets = batch["target"].to(device)

                logits, is_ev = model(
                    sym_tok,
                    sym_feat,
                    lab_feat,
                    rad_feat,
                    xray_path,
                    xray_anat,
                    xray_elig,
                    mask,
                )

                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().tolist())

        correct = sum(p == t for p, t in zip(all_preds, all_targets))
        total = len(all_targets)
        acc = (correct / total) if total > 0 else 0.0

        # Calculate macro F1
        per_class_f1: dict[str, float] = {}
        for c_idx, c_name in enumerate(CONDITION_CLASSES):
            tp = sum(1 for p, t in zip(all_preds, all_targets) if p == c_idx and t == c_idx)
            fp = sum(1 for p, t in zip(all_preds, all_targets) if p == c_idx and t != c_idx)
            fn = sum(1 for p, t in zip(all_preds, all_targets) if p != c_idx and t == c_idx)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            per_class_f1[c_name] = f1

        macro_f1 = sum(per_class_f1.values()) / len(per_class_f1) if per_class_f1 else 0.0

        results[config_name] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "sample_count": total,
            "per_class_f1": {k: round(v, 4) for k, v in per_class_f1.items()},
        }

    return results


def train_multimodal_model(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """Train MultimodalConditionPredictor on patient-partitioned cohort with GMU fusion."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    model = MultimodalConditionPredictor(
        num_classes=len(CONDITION_CLASSES),
        hidden_dim=64,
        dropout_rate=0.2,
    ).to(device)

    train_ds = PatientLinkedMultimodalDataset(train_records)
    val_ds = PatientLinkedMultimodalDataset(val_records)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_model_state: Optional[dict[str, Any]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            sym_tok = batch["symptom_tokens"].to(device)
            sym_feat = batch["symptom_features"].to(device)
            lab_feat = batch["lab_features"].to(device)
            rad_feat = batch["rad_features"].to(device)
            xray_path = batch["xray_pathology_features"].to(device)
            xray_anat = batch["xray_anatomy_probs"].to(device)
            xray_elig = batch["xray_pathology_eligible"].to(device)
            mask = batch["modality_mask"].to(device)
            targets = batch["target"].to(device)

            logits, is_ev = model(
                sym_tok,
                sym_feat,
                lab_feat,
                rad_feat,
                xray_path,
                xray_anat,
                xray_elig,
                mask,
            )

            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                sym_tok = batch["symptom_tokens"].to(device)
                sym_feat = batch["symptom_features"].to(device)
                lab_feat = batch["lab_features"].to(device)
                rad_feat = batch["rad_features"].to(device)
                xray_path = batch["xray_pathology_features"].to(device)
                xray_anat = batch["xray_anatomy_probs"].to(device)
                xray_elig = batch["xray_pathology_eligible"].to(device)
                mask = batch["modality_mask"].to(device)
                targets = batch["target"].to(device)

                logits, is_ev = model(
                    sym_tok,
                    sym_feat,
                    lab_feat,
                    rad_feat,
                    xray_path,
                    xray_anat,
                    xray_elig,
                    mask,
                )

                loss = criterion(logits, targets)
                val_loss += loss.item() * targets.size(0)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        t_acc = train_correct / train_total if train_total else 0.0
        v_acc = val_correct / val_total if val_total else 0.0
        v_loss = val_loss / val_total if val_total else 0.0

        logger.info(f"Epoch {epoch:02d}/{epochs:02d} - Train Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_model_state = model.state_dict()

    if best_model_state:
        model.load_state_dict(best_model_state)

    # Held-out patient evaluation across modality configurations
    logger.info("Evaluating on held-out test patients across modality configurations...")
    eval_report = evaluate_dataset_across_modality_subsets(model, test_records, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = output_dir / "multimodal_model.pth"
    torch.save(
        {
            "version": MODEL_VERSION,
            "model_state": model.state_dict(),
            "condition_classes": CONDITION_CLASSES,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "best_val_loss": best_val_loss,
            "canonical_labs": CANONICAL_LAB_TESTS,
            "canonical_radiology": CANONICAL_RAD_FINDINGS,
            "canonical_xray": CANONICAL_XRAY_PATHOLOGIES,
        },
        model_save_path,
    )

    report_save_path = output_dir / "evaluation_report.json"
    with open(report_save_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    logger.info(f"Multimodal model saved to: {model_save_path}")
    logger.info(f"Evaluation report saved to: {report_save_path}")
    return eval_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Multimodal Condition Predictor")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    logger.info("=== AUDITING MULTIMODAL TRAINING PREREQUISITES ===")
    data_audit = check_authentic_data_availability(args.data_dir)

    if not data_audit["is_available"]:
        logger.warning(
            "CRITICAL TRAINING DATA RULE ACTIVATED: Authentic patient-linked multimodal dataset is UNAVAILABLE on disk."
        )
        logger.warning(
            "Per clinical safety and AI engineering constraints, training on fabricated or unlinked data is STRICTLY BLOCKED."
        )
        logger.info("\nMissing Prerequisites:")
        for prereq in data_audit["missing_prerequisites"]:
            logger.info(f"  * {prereq}")
        logger.info(f"\nRequired Next Action to Unblock:\n  {data_audit['roadmap_action']}")
        logger.info("\nActive Baseline System:")
        logger.info("  * Unimodal Symptom-Based Classifier: ACTIVE (backend/models/conditions/condition_pipeline.joblib)")
        logger.info("  * Downstream Evidence Reconciliation & Safety Triage: ACTIVE")
        logger.info("  * Multimodal ML Training Status: BLOCKED (Awaiting authentic linked EHR data access)")
        return 2  # Exit code 2 indicates prerequisite data blocked per protocol

    logger.info(f"Authentic dataset verified: {data_audit['cohort_name']} ({data_audit['patient_count']} patients)")

    # Read linked patient cohort records
    with open(args.data_dir / "linked_patient_cohort.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", [])

    train_rec, val_rec, test_rec = split_patient_cohort(records, train_ratio=0.7, val_ratio=0.15)
    logger.info(f"Patient-level split: {len(train_rec)} train, {len(val_rec)} val, {len(test_rec)} test")

    train_multimodal_model(
        train_rec,
        val_rec,
        test_rec,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
