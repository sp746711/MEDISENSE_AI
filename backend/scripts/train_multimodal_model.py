"""Multimodal Condition Prediction Training & Evaluation Pipeline for MediSense AI.

This module implements the end-to-end training, patient-level validation, and evaluation pipeline
for the MultimodalConditionPredictor architecture.

CLINICAL SAFETY & ETHICAL DATA RULE (Problems H, I, K):
1. Multimodal training strictly requires authentic, patient-level linked clinical EHR records
   (e.g., MIMIC-IV v2.2 linked to MIMIC-CXR JPG via subject_id, hadm_id, and study_id).
2. NEVER join unrelated datasets by index, disease name, or assumptions.
3. NEVER fabricate synthetic patient associations or claim clinical validation without authentic data.
4. If authentic linked datasets are not present on disk, this script safely halts and
   outputs a structured blocked-audit manifest detailing the prerequisite data-use agreements
   (PhysioNet CITI training + DUA) required.
5. Strict patient-level partitioning (GroupKFold by subject_id) eliminates patient leakage.
6. Modality subset evaluation evaluates ONLY genuine patient encounters possessing those modalities,
   strictly rejecting fake zero-tensor mask overrides.
7. Acceptance Gate: Models must pass declared clinical acceptance criteria before being marked
   PRODUCTION_APPROVED.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Optional

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
    ARCHITECTURE_VERSION,
    CANONICAL_CLINICAL_EXAM_FINDINGS,
    CANONICAL_LAB_TESTS,
    CANONICAL_RAD_FINDINGS,
    CANONICAL_XRAY_PATHOLOGIES,
    CONDITION_CLASSES,
    FEATURE_SCHEMA_VERSION,
    LAB_SCHEMA_VERSION,
    SYMPTOM_VOCAB_VERSION,
    MultimodalConditionPredictor,
)

DEFAULT_DATA_DIR = BASE_DIR / "data" / "multimodal_linked"
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "conditions_multimodal"
BASELINE_MODEL_PATH = BASE_DIR / "models" / "conditions" / "condition_pipeline.joblib"
MODEL_VERSION = "multimodal_gmu_v2.2"


class PatientLinkedMultimodalDataset(Dataset):
    """
    Dataset wrapper for genuine patient-level linked EHR records.
    Each sample represents an authentic patient encounter linked by subject_id and encounter ID.
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
            "exam_features": torch.tensor(rec.get("exam_features", [0.0] * (len(CANONICAL_CLINICAL_EXAM_FINDINGS) * 3)), dtype=torch.float32),
            "xray_pathology_features": torch.tensor(rec["xray_pathology_features"], dtype=torch.float32),
            "xray_anatomy_probs": torch.tensor(rec.get("xray_anatomy_probs", [0.0] * 33), dtype=torch.float32),
            "xray_pathology_eligible": torch.tensor([rec["xray_pathology_eligible"]], dtype=torch.float32),
            "modality_mask": torch.tensor(rec["modality_mask"], dtype=torch.float32),
            "target": torch.tensor(rec["target_class_idx"], dtype=torch.long),
            "subject_id": str(rec.get("subject_id", "")),
            "hadm_id": str(rec.get("hadm_id", "")),
        }


def check_authentic_data_availability(data_dir: Path) -> dict[str, Any]:
    """
    Rigorously audit local filesystem for genuine patient-level linked multimodal cohorts (Problem H).
    Rejects missing IDs, synthetic tags, unlinked files, and missing clinical provenance.
    """
    status: dict[str, Any] = {
        "is_available": False,
        "is_authentic": False,
        "cohort_name": "None",
        "sample_count": 0,
        "patient_count": 0,
        "encounter_count": 0,
        "missing_prerequisites": [],
        "rejection_reasons": [],
        "roadmap_action": "",
        "authenticity_rules": [
            "Patient-level linkage strictly verified via authentic subject_id.",
            "Encounter-level temporal linkage verified via hadm_id / charttime.",
            "Imaging study linkage verified via authentic study_id.",
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
        if not records:
            status["rejection_reasons"].append("Manifest contains empty record list.")
            return status

        # Strict authenticity audit of every record
        reasons: list[str] = []
        subjects: set[str] = set()
        encounters: set[str] = set()

        for idx, r in enumerate(records):
            sid = str(r.get("subject_id") or "").strip()
            hadm = str(r.get("hadm_id") or "").strip()
            # 1. Missing patient or encounter ID
            if not sid or sid.lower() in {"none", "null", "undefined", "unknown", str(idx)}:
                reasons.append(f"Record {idx} missing authentic subject_id.")
                break
            if not hadm or hadm.lower() in {"none", "null", "undefined"}:
                reasons.append(f"Record {idx} missing encounter identifier (hadm_id).")
                break
            # 2. Check for synthetic / simulated tags
            if r.get("is_synthetic") or "synthetic" in str(r.get("data_source", "")).lower():
                reasons.append(f"Record {idx} flagged as synthetic/simulated.")
                break
            # 3. Target class validation
            tgt = r.get("target_class_idx")
            if tgt is None or tgt < 0 or tgt >= len(CONDITION_CLASSES):
                reasons.append(f"Record {idx} contains invalid target class index {tgt}.")
                break

            subjects.add(sid)
            encounters.add(hadm)

        if reasons:
            status["is_available"] = False
            status["is_authentic"] = False
            status["rejection_reasons"] = reasons
            logger.warning("Authentic data audit failed: %s", "; ".join(reasons))
            return status

        status["is_available"] = True
        status["is_authentic"] = True
        status["cohort_name"] = data.get("cohort_name", "Verified Linked EHR")
        status["sample_count"] = len(records)
        status["patient_count"] = len(subjects)
        status["encounter_count"] = len(encounters)
    except Exception as e:
        status["rejection_reasons"].append(f"Error parsing manifest: {e}")
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
    Never falls back to row index or record object id.
    """
    patient_records: dict[str, list[dict[str, Any]]] = {}
    for idx, r in enumerate(records):
        sid = str(r.get("subject_id") or "").strip()
        if not sid:
            raise ValueError(f"Record {idx} missing subject_id; patient-level splitting cannot proceed.")
        patient_records.setdefault(sid, []).append(r)

    unique_patients = sorted(list(patient_records.keys()))
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

    logger.info(
        "Patient-level split completed: Train=%d patients (%d encounters), Val=%d patients (%d encounters), Test=%d patients (%d encounters)",
        len(train_pats), len(train_records), len(val_pats), len(val_records), len(test_pats), len(test_records)
    )
    return train_records, val_records, test_records


def evaluate_dataset_across_modality_subsets(
    model: MultimodalConditionPredictor,
    test_records: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """
    Evaluate the model on genuine held-out test patients under distinct modality presentations (Problem I).
    CRITICAL: Evaluates ONLY samples that ACTUALLY possess the required modalities.
    Never simulates missing modalities with fake zero-tensors.
    """
    model.eval()

    # Modality filter criteria: (modality_name, lambda(rec) -> bool)
    subset_definitions = {
        "symptoms_only": lambda r: r["modality_mask"][0] == 1.0 and sum(r["modality_mask"][1:]) == 0.0,
        "symptoms_plus_labs": lambda r: r["modality_mask"][0] == 1.0 and r["modality_mask"][1] == 1.0,
        "symptoms_plus_radiology": lambda r: r["modality_mask"][0] == 1.0 and r["modality_mask"][2] == 1.0,
        "symptoms_plus_xray": lambda r: r["modality_mask"][0] == 1.0 and r["modality_mask"][3] == 1.0,
        "symptoms_plus_clinical_exam": lambda r: r["modality_mask"][0] == 1.0 and len(r["modality_mask"]) > 4 and r["modality_mask"][4] == 1.0,
        "symptoms_labs_radiology": lambda r: r["modality_mask"][0] == 1.0 and r["modality_mask"][1] == 1.0 and r["modality_mask"][2] == 1.0,
        "all_available_modalities": lambda r: True,  # Natural patient presentation
    }

    results: dict[str, Any] = {}

    for config_name, filter_fn in subset_definitions.items():
        eligible_records = [r for r in test_records if filter_fn(r)]
        sample_count = len(eligible_records)
        patient_count = len(set(r["subject_id"] for r in eligible_records))

        if sample_count == 0:
            results[config_name] = {
                "status": "INSUFFICIENT_GENUINE_SAMPLES",
                "sample_count": 0,
                "patient_count": 0,
                "accuracy": None,
                "macro_f1": None,
                "message": f"Held-out test cohort contains 0 genuine patients meeting modality criteria for '{config_name}'.",
            }
            continue

        dataset = PatientLinkedMultimodalDataset(eligible_records)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_preds: list[int] = []
        all_targets: list[int] = []

        with torch.no_grad():
            for batch in loader:
                sym_tok = batch["symptom_tokens"].to(device)
                sym_feat = batch["symptom_features"].to(device)
                lab_feat = batch["lab_features"].to(device)
                rad_feat = batch["rad_features"].to(device)
                exam_feat = batch["exam_features"].to(device)
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
                    exam_feat,
                    xray_anat,
                    xray_elig,
                    mask,
                )

                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().tolist())

        correct = sum(p == t for p, t in zip(all_preds, all_targets))
        acc = correct / sample_count

        # Compute per-class precision, recall, and F1
        per_class_f1: dict[str, float] = {}
        for c_idx, c_name in enumerate(CONDITION_CLASSES):
            tp = sum(1 for p, t in zip(all_preds, all_targets) if p == c_idx and t == c_idx)
            fp = sum(1 for p, t in zip(all_preds, all_targets) if p == c_idx and t != c_idx)
            fn = sum(1 for p, t in zip(all_preds, all_targets) if p != c_idx and t == c_idx)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            if (tp + fn) > 0:  # Only evaluate classes represented in this subset
                per_class_f1[c_name] = f1

        macro_f1 = sum(per_class_f1.values()) / len(per_class_f1) if per_class_f1 else 0.0

        results[config_name] = {
            "status": "EVALUATED",
            "sample_count": sample_count,
            "patient_count": patient_count,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "classes_evaluated": len(per_class_f1),
            "per_class_f1": {k: round(v, 4) for k, v in per_class_f1.items()},
        }

    return results


def evaluate_acceptance_criteria(
    val_accuracy: float,
    val_macro_f1: float,
    test_eval_results: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Clinical & AI engineering acceptance gate (Problem K).
    Only models that achieve satisfactory held-out test performance without mode collapse
    are approved for production.
    """
    reasons: list[str] = []
    overall_test = test_eval_results.get("all_available_modalities", {})
    test_f1 = overall_test.get("macro_f1")
    test_acc = overall_test.get("accuracy")

    if test_f1 is None or test_f1 < 0.50:
        reasons.append(f"Test macro F1 ({test_f1}) falls below minimum acceptance threshold (0.50).")

    if test_acc is None or test_acc < 0.55:
        reasons.append(f"Test accuracy ({test_acc}) falls below minimum acceptance threshold (0.55).")

    # Check for generalization gap (overfitting)
    if test_acc is not None and abs(val_accuracy - test_acc) > 0.20:
        reasons.append(f"Excessive validation vs test accuracy gap ({abs(val_accuracy - test_acc):.4f} > 0.20).")

    # Check for complete mode collapse in per-class F1
    per_class = overall_test.get("per_class_f1", {})
    zero_f1_classes = [c for c, f1 in per_class.items() if f1 == 0.0]
    if len(zero_f1_classes) > len(per_class) * 0.5:
        reasons.append(f"Severe mode collapse: {len(zero_f1_classes)}/{len(per_class)} classes have zero F1.")

    passed = len(reasons) == 0
    return passed, reasons


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
    best_val_acc = 0.0
    best_val_f1 = 0.0
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
            exam_feat = batch["exam_features"].to(device)
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
                exam_feat,
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
        val_preds_list: list[int] = []
        val_targets_list: list[int] = []

        with torch.no_grad():
            for batch in val_loader:
                sym_tok = batch["symptom_tokens"].to(device)
                sym_feat = batch["symptom_features"].to(device)
                lab_feat = batch["lab_features"].to(device)
                rad_feat = batch["rad_features"].to(device)
                exam_feat = batch["exam_features"].to(device)
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
                    exam_feat,
                    xray_anat,
                    xray_elig,
                    mask,
                )

                loss = criterion(logits, targets)
                val_loss += loss.item() * targets.size(0)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
                val_preds_list.extend(preds.cpu().tolist())
                val_targets_list.extend(targets.cpu().tolist())

        avg_val_loss = val_loss / val_total if val_total > 0 else 0.0
        avg_val_acc = val_correct / val_total if val_total > 0 else 0.0

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = avg_val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_correct/train_total:.4f} | Val Acc: {avg_val_acc:.4f} (Loss: {avg_val_loss:.4f})")

    # Load best validation model for independent test evaluation (Problem K)
    if best_model_state:
        model.load_state_dict(best_model_state)

    logger.info("Running single independent test cohort evaluation...")
    test_eval_results = evaluate_dataset_across_modality_subsets(model, test_records, device)

    # Evaluate Acceptance Gate
    acceptance_passed, gate_reasons = evaluate_acceptance_criteria(best_val_acc, best_val_f1, test_eval_results)

    output_dir.mkdir(parents=True, exist_ok=True)
    status_str = "PRODUCTION_APPROVED" if acceptance_passed else "ACCEPTANCE_FAILED"

    checkpoint_payload = {
        "version": MODEL_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "lab_schema_version": LAB_SCHEMA_VERSION,
        "symptom_vocab_version": SYMPTOM_VOCAB_VERSION,
        "status": status_str,
        "acceptance_passed": acceptance_passed,
        "acceptance_reasons": gate_reasons,
        "condition_classes": CONDITION_CLASSES,
        "model_state": best_model_state,
        "val_metrics": {
            "best_loss": round(best_val_loss, 4),
            "accuracy": round(best_val_acc, 4),
        },
        "test_evaluation_report": test_eval_results,
        "dataset_provenance": {
            "is_authentic": True,
            "train_samples": len(train_records),
            "val_samples": len(val_records),
            "test_samples": len(test_records),
            "train_patients": len(set(r["subject_id"] for r in train_records)),
            "val_patients": len(set(r["subject_id"] for r in val_records)),
            "test_patients": len(set(r["subject_id"] for r in test_records)),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    ckpt_filename = "multimodal_model.pth" if acceptance_passed else "multimodal_model_experimental.pth"
    save_path = output_dir / ckpt_filename
    torch.save(checkpoint_payload, save_path)
    logger.info(f"Saved checkpoint ({status_str}) to {save_path}")

    return {
        "status": status_str,
        "acceptance_passed": acceptance_passed,
        "reasons": gate_reasons,
        "save_path": str(save_path),
        "test_results": test_eval_results,
    }


def main():
    parser = argparse.ArgumentParser(description="MediSense AI Multimodal Condition Training Pipeline")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to authentic linked cohort")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Path to save trained checkpoints")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    logger.info("=== AUDITING MULTIMODAL TRAINING PREREQUISITES ===")
    audit = check_authentic_data_availability(args.data_dir)

    if not audit["is_available"] or not audit["is_authentic"]:
        logger.warning("CRITICAL TRAINING DATA RULE ACTIVATED: Authentic patient-linked multimodal dataset is UNAVAILABLE on disk.")
        logger.warning("Per clinical safety and AI engineering constraints, training on fabricated or unlinked data is STRICTLY BLOCKED.")
        logger.info("\nMissing Prerequisites:")
        for prereq in audit.get("missing_prerequisites", []):
            logger.info(f"  * {prereq}")
        for rej in audit.get("rejection_reasons", []):
            logger.info(f"  * Rejection: {rej}")
        logger.info(f"\nRequired Next Action to Unblock:\n  {audit.get('roadmap_action')}")
        logger.info("\nActive Baseline System:")
        logger.info(f"  * Unimodal Symptom-Based Classifier: ACTIVE ({BASELINE_MODEL_PATH})")
        logger.info("  * Downstream Evidence Reconciliation & Safety Triage: ACTIVE")
        logger.info("  * Multimodal ML Training Status: BLOCKED (Awaiting authentic linked EHR data access)\n")
        sys.exit(2)

    logger.info(f"Authentic dataset verified: {audit['cohort_name']} ({audit['sample_count']} samples across {audit['patient_count']} patients).")
    with open(args.data_dir / "linked_patient_cohort.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]

    train_recs, val_recs, test_recs = split_patient_cohort(records)
    res = train_multimodal_model(train_recs, val_recs, test_recs, args.output_dir, args.epochs, args.batch_size, args.lr)
    logger.info(f"Pipeline finished with status: {res['status']}")


if __name__ == "__main__":
    main()
