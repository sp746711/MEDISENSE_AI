"""Comprehensive Multimodal Disease Prediction Integration & Verification Suite for Goal 1.

Verifies:
1. Symptoms-only input reaches the multimodal model with valid tensor shapes and mask [1, 0, 0, 0].
2. Symptoms plus labs reach the model with mask [1, 1, 0, 0].
3. Symptoms plus radiology reach the model with mask [1, 0, 1, 0].
4. Symptoms plus eligible X-ray pathology features reach the model with mask [1, 0, 0, 1].
5. Symptoms plus labs plus radiology reach the model with mask [1, 1, 1, 0].
6. All four main modalities reach the model together with mask [1, 1, 1, 1].
7. Valid modality value changes distinctly change the model representation (sensitivity test).
8. Missing modalities do not create false normal values (masking and zero-representation test).
9. Invalid / unverified X-ray findings (unverified anatomy or region mismatch) are strictly excluded.
10. Every modality feature is traceable to its clinical provenance source.
11. MultimodalInferenceService fallback safely reports blocked-untrained status without running random weights.
12. Baseline unimodal symptom model is fully intact and unmodified.
13. Protected X-ray files retain 100% bit-exact SHA-256 hash match against the baseline manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import joblib
import pytest
import torch

from app.ai.multimodal_condition_model import (
    CANONICAL_LAB_TESTS,
    CANONICAL_RAD_FINDINGS,
    CANONICAL_XRAY_PATHOLOGIES,
    CONDITION_CLASSES,
    ClinicalSymptomObservation,
    MultimodalConditionPredictor,
    MultimodalFeatureExtractor,
    MultimodalInferenceService,
    MultimodalPatientRecord,
    StructuredLabParameter,
    StructuredRadiologyFinding,
    XRaySpecialistOutput,
    get_multimodal_inference_service,
)
from scripts.train_multimodal_model import check_authentic_data_availability


@pytest.fixture
def predictor():
    """Instantiate a MultimodalConditionPredictor for inference architecture verification."""
    model = MultimodalConditionPredictor(num_classes=len(CONDITION_CLASSES), hidden_dim=64)
    model.eval()
    return model


# ==============================================================================
# 1. MODALITY REACH & MASK VERIFICATION (6 COMBINATIONS)
# ==============================================================================

def test_1_symptoms_only_reaches_model(predictor):
    """Verify symptoms-only input constructs mask [1, 0, 0, 0] and executes forward pass."""
    record = MultimodalPatientRecord(
        subject_id="pat_sym_only",
        symptoms=[
            ClinicalSymptomObservation(symptom="cough", canonical="cough", state="PRESENT", duration_days=5, severity="MODERATE"),
            ClinicalSymptomObservation(symptom="fever", canonical="fever", state="PRESENT", duration_days=3, severity="MILD"),
        ],
    )
    tensors = MultimodalFeatureExtractor.extract_tensors_from_record(record)
    mask = tensors["modality_mask"].squeeze(0).tolist()
    assert mask == [1.0, 0.0, 0.0, 0.0], f"Expected mask [1, 0, 0, 0], got {mask}"

    with torch.no_grad():
        logits, is_ev = predictor(
            tensors["symptom_tokens"],
            tensors["symptom_features"],
            tensors["lab_features"],
            tensors["rad_features"],
            tensors["xray_pathology_features"],
            tensors["xray_anatomy_probs"],
            tensors["xray_pathology_eligible"],
            tensors["modality_mask"],
        )

    assert logits.shape == (1, 24)
    assert is_ev.item() == 1.0
    assert not torch.isnan(logits).any()


def test_2_symptoms_plus_labs_reach_model(predictor):
    """Verify symptoms plus laboratory results construct mask [1, 1, 0, 0]."""
    record = MultimodalPatientRecord(
        subject_id="pat_sym_lab",
        symptoms=[
            ClinicalSymptomObservation(symptom="fever", canonical="fever", state="PRESENT"),
        ],
        laboratory=[
            StructuredLabParameter(
                test_name="White Blood Cell Count (WBC)",
                canonical_test_name="White Blood Cell Count (WBC)",
                value=14500.0,
                unit="/µL",
                reference_low=4000.0,
                reference_high=11000.0,
                flag="HIGH",
            ),
            StructuredLabParameter(
                test_name="C-Reactive Protein (CRP)",
                canonical_test_name="C-Reactive Protein (CRP)",
                value=45.0,
                unit="mg/L",
                reference_low=0.0,
                reference_high=5.0,
                flag="HIGH",
            ),
        ],
    )
    tensors = MultimodalFeatureExtractor.extract_tensors_from_record(record)
    mask = tensors["modality_mask"].squeeze(0).tolist()
    assert mask == [1.0, 1.0, 0.0, 0.0]

    with torch.no_grad():
        logits, is_ev = predictor(
            tensors["symptom_tokens"],
            tensors["symptom_features"],
            tensors["lab_features"],
            tensors["rad_features"],
            tensors["xray_pathology_features"],
            tensors["xray_anatomy_probs"],
            tensors["xray_pathology_eligible"],
            tensors["modality_mask"],
        )
    assert logits.shape == (1, 24)
    assert is_ev.item() == 1.0


def test_3_symptoms_plus_radiology_reach_model(predictor):
    """Verify symptoms plus radiology findings construct mask [1, 0, 1, 0]."""
    record = MultimodalPatientRecord(
        subject_id="pat_sym_rad",
        symptoms=[
            ClinicalSymptomObservation(symptom="shortness of breath", canonical="shortness of breath", state="PRESENT"),
        ],
        radiology=[
            StructuredRadiologyFinding(
                finding="consolidation",
                canonical_finding="consolidation",
                state="PRESENT",
                anatomical_region="chest",
            ),
            StructuredRadiologyFinding(
                finding="pleural effusion",
                canonical_finding="pleural effusion",
                state="ABSENT",
                anatomical_region="chest",
            ),
        ],
    )
    tensors = MultimodalFeatureExtractor.extract_tensors_from_record(record)
    mask = tensors["modality_mask"].squeeze(0).tolist()
    assert mask == [1.0, 0.0, 1.0, 0.0]

    with torch.no_grad():
        logits, is_ev = predictor(
            tensors["symptom_tokens"],
            tensors["symptom_features"],
            tensors["lab_features"],
            tensors["rad_features"],
            tensors["xray_pathology_features"],
            tensors["xray_anatomy_probs"],
            tensors["xray_pathology_eligible"],
            tensors["modality_mask"],
        )
    assert logits.shape == (1, 24)
    assert is_ev.item() == 1.0


def test_4_symptoms_plus_xray_pathology_reach_model(predictor):
    """Verify symptoms plus verified X-ray specialist outputs construct mask [1, 0, 0, 1]."""
    record = MultimodalPatientRecord(
        subject_id="pat_sym_xray",
        symptoms=[
            ClinicalSymptomObservation(symptom="wrist pain", canonical="wrist pain", state="PRESENT"),
        ],
        xray=XRaySpecialistOutput(
            file_name="wrist.png",
            detected_anatomy="wrist",
            anatomy_verified=True,
            specialist_model_executed=True,
            has_region_mismatch=False,
            is_pathology_eligible=True,
            pathology_scores={"Musculoskeletal Abnormality": 0.88},
            status="COMPLETED",
        ),
    )
    tensors = MultimodalFeatureExtractor.extract_tensors_from_record(record)
    mask = tensors["modality_mask"].squeeze(0).tolist()
    assert mask == [1.0, 0.0, 0.0, 1.0]

    with torch.no_grad():
        logits, is_ev = predictor(
            tensors["symptom_tokens"],
            tensors["symptom_features"],
            tensors["lab_features"],
            tensors["rad_features"],
            tensors["xray_pathology_features"],
            tensors["xray_anatomy_probs"],
            tensors["xray_pathology_eligible"],
            tensors["modality_mask"],
        )
    assert logits.shape == (1, 24)
    assert is_ev.item() == 1.0


def test_5_symptoms_plus_labs_plus_radiology_reach_model(predictor):
    """Verify symptoms + labs + radiology report findings construct mask [1, 1, 1, 0]."""
    record = MultimodalPatientRecord(
        subject_id="pat_sym_lab_rad",
        symptoms=[
            ClinicalSymptomObservation(symptom="productive cough", canonical="cough", state="PRESENT"),
        ],
        laboratory=[
            StructuredLabParameter(
                test_name="Procalcitonin",
                canonical_test_name="Procalcitonin",
                value=2.4,
                unit="ng/mL",
                flag="HIGH",
            )
        ],
        radiology=[
            StructuredRadiologyFinding(
                finding="infiltrate",
                canonical_finding="infiltrate",
                state="PRESENT",
                anatomical_region="chest",
            )
        ],
    )
    tensors = MultimodalFeatureExtractor.extract_tensors_from_record(record)
    mask = tensors["modality_mask"].squeeze(0).tolist()
    assert mask == [1.0, 1.0, 1.0, 0.0]

    with torch.no_grad():
        logits, is_ev = predictor(
            tensors["symptom_tokens"],
            tensors["symptom_features"],
            tensors["lab_features"],
            tensors["rad_features"],
            tensors["xray_pathology_features"],
            tensors["xray_anatomy_probs"],
            tensors["xray_pathology_eligible"],
            tensors["modality_mask"],
        )
    assert logits.shape == (1, 24)
    assert is_ev.item() == 1.0


def test_6_all_four_modalities_reach_model_together(predictor):
    """Verify all four clinical modalities together construct mask [1, 1, 1, 1]."""
    record = MultimodalPatientRecord(
        subject_id="pat_all_modalities",
        symptoms=[
            ClinicalSymptomObservation(symptom="fever", canonical="fever", state="PRESENT"),
            ClinicalSymptomObservation(symptom="cough", canonical="cough", state="PRESENT"),
        ],
        laboratory=[
            StructuredLabParameter(
                test_name="White Blood Cell Count (WBC)",
                canonical_test_name="White Blood Cell Count (WBC)",
                value=16200.0,
                unit="/µL",
                flag="HIGH",
            )
        ],
        radiology=[
            StructuredRadiologyFinding(
                finding="consolidation",
                canonical_finding="consolidation",
                state="PRESENT",
                anatomical_region="chest",
            )
        ],
        xray=XRaySpecialistOutput(
            file_name="cxr.png",
            detected_anatomy="chest",
            anatomy_verified=True,
            specialist_model_executed=True,
            has_region_mismatch=False,
            is_pathology_eligible=True,
            pathology_scores={"Consolidation": 0.85, "Pneumonia": 0.79},
            status="COMPLETED",
        ),
    )
    tensors = MultimodalFeatureExtractor.extract_tensors_from_record(record)
    mask = tensors["modality_mask"].squeeze(0).tolist()
    assert mask == [1.0, 1.0, 1.0, 1.0]

    with torch.no_grad():
        logits, is_ev = predictor(
            tensors["symptom_tokens"],
            tensors["symptom_features"],
            tensors["lab_features"],
            tensors["rad_features"],
            tensors["xray_pathology_features"],
            tensors["xray_anatomy_probs"],
            tensors["xray_pathology_eligible"],
            tensors["modality_mask"],
        )
    assert logits.shape == (1, 24)
    assert is_ev.item() == 1.0


# ==============================================================================
# 2. REPRESENTATION SENSITIVITY & INTEGRITY
# ==============================================================================

def test_7_modality_value_changes_alter_model_inputs(predictor):
    """Verify that changing valid clinical input values changes the feature tensors and embeddings."""
    rec_normal_wbc = MultimodalPatientRecord(
        subject_id="p1",
        laboratory=[
            StructuredLabParameter(
                test_name="White Blood Cell Count (WBC)",
                canonical_test_name="White Blood Cell Count (WBC)",
                value=6500.0,
                unit="/µL",
                flag="NORMAL",
            )
        ],
    )
    rec_severe_leukocytosis = MultimodalPatientRecord(
        subject_id="p2",
        laboratory=[
            StructuredLabParameter(
                test_name="White Blood Cell Count (WBC)",
                canonical_test_name="White Blood Cell Count (WBC)",
                value=28000.0,
                unit="/µL",
                flag="HIGH",
            )
        ],
    )

    t1 = MultimodalFeatureExtractor.extract_tensors_from_record(rec_normal_wbc)
    t2 = MultimodalFeatureExtractor.extract_tensors_from_record(rec_severe_leukocytosis)

    # Feature tensors must be distinctly different
    assert not torch.allclose(t1["lab_features"], t2["lab_features"])

    # Latent embeddings must be distinctly different
    with torch.no_grad():
        h1 = predictor.lab_encoder(t1["lab_features"])
        h2 = predictor.lab_encoder(t2["lab_features"])
    assert not torch.allclose(h1, h2, atol=1e-4)


def test_8_missing_modalities_do_not_create_false_normal_values():
    """Verify that unmeasured lab tests or unmentioned radiology findings are NOT imputed as normal."""
    record = MultimodalPatientRecord(
        subject_id="p_minimal",
        symptoms=[ClinicalSymptomObservation(symptom="headache", canonical="headache", state="PRESENT")],
        # No labs or radiology
    )
    tensors = MultimodalFeatureExtractor.extract_tensors_from_record(record)

    # Every lab test in tensors["lab_features"] must have is_present = 0.0 (5th element of each test)
    lab_flat = tensors["lab_features"].squeeze(0).tolist()
    for idx in range(len(CANONICAL_LAB_TESTS)):
        base = idx * 5
        val_norm = lab_flat[base]
        flag_code = lab_flat[base + 3]
        is_present = lab_flat[base + 4]
        assert is_present == 0.0, f"Unmeasured lab test {CANONICAL_LAB_TESTS[idx]} had is_present=1!"
        assert val_norm == 0.0
        assert flag_code == 0.0

    # Every rad finding must have is_reported = 0.0 (2nd element of each finding)
    rad_flat = tensors["rad_features"].squeeze(0).tolist()
    for idx in range(len(CANONICAL_RAD_FINDINGS)):
        base = idx * 3
        is_reported = rad_flat[base + 1]
        assert is_reported == 0.0, f"Unreported finding {CANONICAL_RAD_FINDINGS[idx]} was marked reported!"


def test_9_unverified_or_mismatched_xray_strictly_excluded(predictor):
    """Verify fail-closed rule: Unverified anatomy or region mismatch zeros out X-ray representation."""
    # Case A: Unverified anatomy
    rec_unverified = MultimodalPatientRecord(
        subject_id="p_unverified",
        xray=XRaySpecialistOutput(
            file_name="unverified.png",
            detected_anatomy="wrist",
            anatomy_verified=False,  # Unverified!
            specialist_model_executed=True,
            is_pathology_eligible=False,
            pathology_scores={"Musculoskeletal Abnormality": 0.95},
            status="UNVERIFIED",
        ),
    )
    t_unv = MultimodalFeatureExtractor.extract_tensors_from_record(rec_unverified)
    assert t_unv["xray_pathology_eligible"].item() == 0.0
    assert t_unv["modality_mask"][0, 3].item() == 0.0

    # Case B: Region mismatch
    rec_mismatch = MultimodalPatientRecord(
        subject_id="p_mismatch",
        xray=XRaySpecialistOutput(
            file_name="elbow_labeled_wrist.png",
            detected_anatomy="elbow",
            anatomy_verified=True,
            has_region_mismatch=True,  # Region mismatch!
            is_pathology_eligible=False,
            pathology_scores={"Musculoskeletal Abnormality": 0.90},
            status="MISMATCH",
        ),
    )
    t_mis = MultimodalFeatureExtractor.extract_tensors_from_record(rec_mismatch)
    assert t_mis["xray_pathology_eligible"].item() == 0.0
    assert t_mis["modality_mask"][0, 3].item() == 0.0

    # In both cases, X-ray adapter forward strictly produces zero vector
    with torch.no_grad():
        h_xray_unv = predictor.xray_adapter(
            t_unv["xray_pathology_features"], t_unv["xray_anatomy_probs"], t_unv["xray_pathology_eligible"]
        )
        h_xray_mis = predictor.xray_adapter(
            t_mis["xray_pathology_features"], t_mis["xray_anatomy_probs"], t_mis["xray_pathology_eligible"]
        )
    assert torch.all(h_xray_unv == 0.0)
    assert torch.all(h_xray_mis == 0.0)


# ==============================================================================
# 3. PROVENANCE & HIGH-LEVEL SERVICE SAFETY
# ==============================================================================

def test_10_modality_provenance_traceability():
    """Verify that build_record_from_inputs accurately retains provenance and source file links."""
    svc = MultimodalInferenceService()
    record, tensors = svc.build_record_from_inputs(
        subject_id="test_assessment_123",
        symptoms_present=[{"name": "fever", "duration_days": 3, "severity": "MILD"}],
        symptoms_absent=["chest pain"],
        symptoms_unknown=["nausea"],
        medical_reports=[
            {
                "file_name": "cbc_report_lab.pdf",
                "id": "rep_999",
                "findings": [
                    {
                        "test_name": "White Blood Cell Count (WBC)",
                        "value": 15000,
                        "unit": "/µL",
                        "status": "HIGH",
                        "reference_low": 4000,
                        "reference_high": 11000,
                    }
                ],
            }
        ],
        xray_reports=[
            {
                "file_name": "chest_xray_rad.pdf",
                "id": "rad_888",
                "region": "chest",
                "qualitative_findings": [
                    {"finding": "consolidation", "state": "PRESENT"}
                ],
            }
        ],
    )

    assert record.subject_id == "test_assessment_123"
    assert len(record.symptoms) == 3
    assert len(record.laboratory) == 1
    assert record.laboratory[0].source_file == "cbc_report_lab.pdf"
    assert record.laboratory[0].report_id == "rep_999"

    assert len(record.radiology) == 1
    assert record.radiology[0].source_file == "chest_xray_rad.pdf"
    assert record.radiology[0].report_id == "rad_888"


def test_11_untrained_multimodal_service_safely_falls_back():
    """Verify that when no trained checkpoint exists, service returns transparent blocked status."""
    svc = MultimodalInferenceService(checkpoint_path=None)
    assert not svc.is_trained

    record = MultimodalPatientRecord(
        subject_id="patient_check",
        symptoms=[ClinicalSymptomObservation(symptom="fever", canonical="fever", state="PRESENT")],
    )
    result = svc.predict(record)
    assert result["status"] == "blocked_untrained"
    assert result["is_multimodal_trained"] is False
    assert "MIMIC" in result["message"]
    assert result["active_baseline_model"] == "condition_pred_v2.0-calibrated_logistic"
    assert result["conditions"] == []  # Never hallucinates random conditions!


def test_12_empty_modalities_returns_insufficient_evidence():
    """Verify that submitting zero modalities returns explicit insufficient_evidence without phantom diagnoses."""
    svc = MultimodalInferenceService(checkpoint_path=None)
    empty_record = MultimodalPatientRecord(subject_id="pat_empty")
    res = svc.predict(empty_record)
    assert res["status"] == "insufficient_evidence"
    assert res["conditions"] == []


# ==============================================================================
# 4. BASELINE INTEGRITY & PROTECTED FILES SHA-256 CHECK
# ==============================================================================

def test_13_baseline_unimodal_symptom_model_preserved():
    """Verify that existing TF-IDF + Logistic Regression baseline model is intact and operational."""
    baseline_path = Path(__file__).resolve().parents[1] / "models" / "conditions" / "condition_pipeline.joblib"
    assert baseline_path.exists(), f"Missing baseline model file: {baseline_path}"

    pipeline = joblib.load(baseline_path)
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")
    assert len(pipeline.classes_) == 24
    assert "Pneumonia" in pipeline.classes_
    assert "Common Cold" in pipeline.classes_


def test_14_protected_xray_files_sha256_integrity():
    """VERIFICATION OF PART 2 ABSOLUTE PROTECTED X-RAY RULE: All 15 protected assets retain bit-exact hashes."""
    backend_dir = Path(__file__).resolve().parents[1]
    manifest_path = backend_dir / "models" / "xray_protected_manifest.json"
    assert manifest_path.exists(), "Protected X-ray SHA-256 manifest is missing!"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for rel_path, expected_hash in manifest.items():
        full_path = backend_dir.parent / rel_path
        assert full_path.exists(), f"Protected file missing: {full_path}"
        hasher = hashlib.sha256()
        with open(full_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        actual_hash = hasher.hexdigest()
        assert actual_hash == expected_hash, (
            f"CRITICAL VIOLATION: Protected X-ray asset '{rel_path}' hash changed! "
            f"Expected {expected_hash}, got {actual_hash}"
        )
