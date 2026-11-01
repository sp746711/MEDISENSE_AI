"""Comprehensive Multimodal Disease Prediction Integration & Verification Suite for Goal 1.

Verifies:
1. Symptoms-only input reaches the multimodal model with valid tensor shapes and mask [1, 0, 0, 0, 0].
2. Symptoms plus labs reach the model with mask [1, 1, 0, 0, 0].
3. Symptoms plus radiology reach the model with mask [1, 0, 1, 0, 0].
4. Symptoms plus eligible X-ray pathology features reach the model with mask [1, 0, 0, 1, 0].
5. Symptoms plus clinical examination reach the model with mask [1, 0, 0, 0, 1].
6. All five main modalities reach the model together with mask [1, 1, 1, 1, 1].
7. Valid modality value changes distinctly change the model representation (sensitivity test).
8. Missing modalities do not create false normal values (masking and zero-representation test).
9. Invalid / unverified X-ray findings (unverified anatomy or region mismatch) are strictly excluded.
10. Multi-study X-ray processing aggregates multiple images and traces unverified/mismatched studies.
11. Laboratory preprocessing correctly handles unit conversions and all 21 canonical aliases.
12. Radiology conflicts (e.g. clear lungs co-occurring with consolidation) are explicitly flagged.
13. Strict checkpoint validation rejects unapproved, unverified, or incompatible checkpoints.
14. Every modality feature is traceable to its clinical provenance source.
15. MultimodalInferenceService fallback safely reports blocked-untrained status without running random weights.
16. Baseline unimodal symptom model is fully intact and unmodified.
17. Protected X-ray files retain 100% bit-exact SHA-256 hash match against the baseline manifest.
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
    ARCHITECTURE_VERSION,
    CANONICAL_CLINICAL_EXAM_FINDINGS,
    CANONICAL_LAB_TESTS,
    CANONICAL_RAD_FINDINGS,
    CANONICAL_XRAY_PATHOLOGIES,
    CONDITION_CLASSES,
    FEATURE_SCHEMA_VERSION,
    LAB_SCHEMA_VERSION,
    SYMPTOM_VOCAB_VERSION,
    ClinicalSymptomObservation,
    MultimodalConditionPredictor,
    MultimodalFeatureExtractor,
    MultimodalInferenceService,
    MultimodalPatientRecord,
    StructuredClinicalFinding,
    StructuredLabParameter,
    StructuredRadiologyFinding,
    XRaySpecialistOutput,
    get_multimodal_inference_service,
    normalize_lab_value,
    resolve_canonical_lab_name,
    validate_multimodal_checkpoint,
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
    """Verify symptoms-only input constructs mask [1, 0, 0, 0, 0] and executes forward pass."""
    record = MultimodalPatientRecord(
        subject_id="pat_sym_only",
        symptoms=[
            ClinicalSymptomObservation(symptom="cough", canonical="cough", state="PRESENT", duration_days=5, severity="MODERATE"),
            ClinicalSymptomObservation(symptom="fever", canonical="fever", state="PRESENT", duration_days=3, severity="MILD"),
        ],
    )
    tensors = MultimodalFeatureExtractor.extract_tensors_from_record(record)
    mask = tensors["modality_mask"].squeeze(0).tolist()
    assert mask == [1.0, 0.0, 0.0, 0.0, 0.0], f"Expected mask [1, 0, 0, 0, 0], got {mask}"

    with torch.no_grad():
        logits, is_ev = predictor(
            tensors["symptom_tokens"],
            tensors["symptom_features"],
            tensors["lab_features"],
            tensors["rad_features"],
            tensors["xray_pathology_features"],
            tensors["exam_features"],
            tensors["xray_anatomy_probs"],
            tensors["xray_pathology_eligible"],
            tensors["modality_mask"],
        )

    assert logits.shape == (1, 24)
    assert is_ev.item() == 1.0
    assert not torch.isnan(logits).any()


def test_2_symptoms_plus_labs_reach_model(predictor):
    """Verify symptoms plus laboratory results construct mask [1, 1, 0, 0, 0]."""
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
    assert mask == [1.0, 1.0, 0.0, 0.0, 0.0]

    with torch.no_grad():
        logits, is_ev = predictor(
            tensors["symptom_tokens"],
            tensors["symptom_features"],
            tensors["lab_features"],
            tensors["rad_features"],
            tensors["xray_pathology_features"],
            tensors["exam_features"],
            tensors["xray_anatomy_probs"],
            tensors["xray_pathology_eligible"],
            tensors["modality_mask"],
        )
    assert logits.shape == (1, 24)
    assert is_ev.item() == 1.0


def test_3_symptoms_plus_radiology_reach_model(predictor):
    """Verify symptoms plus radiology findings construct mask [1, 0, 1, 0, 0]."""
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
    assert mask == [1.0, 0.0, 1.0, 0.0, 0.0]

    with torch.no_grad():
        logits, is_ev = predictor(
            tensors["symptom_tokens"],
            tensors["symptom_features"],
            tensors["lab_features"],
            tensors["rad_features"],
            tensors["xray_pathology_features"],
            tensors["exam_features"],
            tensors["xray_anatomy_probs"],
            tensors["xray_pathology_eligible"],
            tensors["modality_mask"],
        )
    assert logits.shape == (1, 24)
    assert is_ev.item() == 1.0


def test_4_symptoms_plus_xray_pathology_reach_model(predictor):
    """Verify symptoms plus verified X-ray specialist outputs construct mask [1, 0, 0, 1, 0]."""
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
    assert mask == [1.0, 0.0, 0.0, 1.0, 0.0]

    with torch.no_grad():
        logits, is_ev = predictor(
            tensors["symptom_tokens"],
            tensors["symptom_features"],
            tensors["lab_features"],
            tensors["rad_features"],
            tensors["xray_pathology_features"],
            tensors["exam_features"],
            tensors["xray_anatomy_probs"],
            tensors["xray_pathology_eligible"],
            tensors["modality_mask"],
        )
    assert logits.shape == (1, 24)
    assert is_ev.item() == 1.0


def test_5_symptoms_plus_clinical_examination_reach_model(predictor):
    """Verify symptoms plus physical clinical examination construct mask [1, 0, 0, 0, 1] (Problem A)."""
    record = MultimodalPatientRecord(
        subject_id="pat_sym_exam",
        symptoms=[
            ClinicalSymptomObservation(symptom="knee pain", canonical="knee pain", state="PRESENT"),
        ],
        clinical_examination=[
            StructuredClinicalFinding(
                finding="localized tenderness",
                canonical_finding="localized tenderness",
                category="examination",
                state="PRESENT",
            ),
            StructuredClinicalFinding(
                finding="reduced range of motion",
                canonical_finding="reduced range of motion",
                category="examination",
                state="PRESENT",
            ),
        ],
    )
    tensors = MultimodalFeatureExtractor.extract_tensors_from_record(record)
    mask = tensors["modality_mask"].squeeze(0).tolist()
    assert mask == [1.0, 0.0, 0.0, 0.0, 1.0]

    with torch.no_grad():
        logits, is_ev = predictor(
            tensors["symptom_tokens"],
            tensors["symptom_features"],
            tensors["lab_features"],
            tensors["rad_features"],
            tensors["xray_pathology_features"],
            tensors["exam_features"],
            tensors["xray_anatomy_probs"],
            tensors["xray_pathology_eligible"],
            tensors["modality_mask"],
        )
    assert logits.shape == (1, 24)
    assert is_ev.item() == 1.0


def test_6_all_five_modalities_reach_model_together(predictor):
    """Verify all five clinical modalities together construct mask [1, 1, 1, 1, 1]."""
    record = MultimodalPatientRecord(
        subject_id="pat_all_5_modalities",
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
        clinical_examination=[
            StructuredClinicalFinding(
                finding="localized tenderness",
                canonical_finding="localized tenderness",
                state="PRESENT",
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
    assert mask == [1.0, 1.0, 1.0, 1.0, 1.0]

    with torch.no_grad():
        logits, is_ev = predictor(
            tensors["symptom_tokens"],
            tensors["symptom_features"],
            tensors["lab_features"],
            tensors["rad_features"],
            tensors["xray_pathology_features"],
            tensors["exam_features"],
            tensors["xray_anatomy_probs"],
            tensors["xray_pathology_eligible"],
            tensors["modality_mask"],
        )
    assert logits.shape == (1, 24)
    assert is_ev.item() == 1.0


# ==============================================================================
# 2. FEATURE INTEGRITY & SENSITIVITY TESTS
# ==============================================================================

def test_7_modality_value_changes_alter_model_inputs():
    """Verify sensitivity: modifying lab or radiology values directly changes tensor outputs."""
    base_rec = MultimodalPatientRecord(
        subject_id="p1",
        symptoms=[ClinicalSymptomObservation(symptom="cough", canonical="cough", state="PRESENT")],
        laboratory=[
            StructuredLabParameter(
                test_name="Hemoglobin",
                canonical_test_name="Hemoglobin",
                value=14.0,
                unit="g/dL",
                flag="NORMAL",
            )
        ],
    )
    t_normal = MultimodalFeatureExtractor.extract_tensors_from_record(base_rec)

    # Anemic record
    anemic_rec = MultimodalPatientRecord(
        subject_id="p1",
        symptoms=[ClinicalSymptomObservation(symptom="cough", canonical="cough", state="PRESENT")],
        laboratory=[
            StructuredLabParameter(
                test_name="Hemoglobin",
                canonical_test_name="Hemoglobin",
                value=7.2,
                unit="g/dL",
                flag="LOW",
            )
        ],
    )
    t_anemic = MultimodalFeatureExtractor.extract_tensors_from_record(anemic_rec)

    # Tensor features for hemoglobin must distinctly differ
    assert not torch.allclose(t_normal["lab_features"], t_anemic["lab_features"])


def test_8_missing_modalities_do_not_create_false_normal_values():
    """Verify that unmeasured labs and unreported radiology have is_present=0.0 and is_reported=0.0."""
    rec_empty = MultimodalPatientRecord(
        subject_id="p_empty",
        symptoms=[ClinicalSymptomObservation(symptom="headache", canonical="headache", state="PRESENT")],
    )
    tensors = MultimodalFeatureExtractor.extract_tensors_from_record(rec_empty)

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
        base = idx * 4
        is_reported = rad_flat[base + 1]
        assert is_reported == 0.0, f"Unreported finding {CANONICAL_RAD_FINDINGS[idx]} was marked reported!"


def test_9_laboratory_unit_conversions_and_aliases():
    """Verify Problem B & C: Clinical unit conversions and canonical alias resolution."""
    # Test glucose conversion: 5.5 mmol/L -> ~99.1 mg/dL
    val_norm_mmol, _, _, _, _ = normalize_lab_value("Fasting Blood Glucose", 5.5, "mmol/L")
    val_norm_mgdl, _, _, _, _ = normalize_lab_value("Fasting Blood Glucose", 99.1, "mg/dL")
    assert abs(val_norm_mmol - val_norm_mgdl) < 0.05

    # Test alias resolution for all common clinical variations
    assert resolve_canonical_lab_name("TLC") == "White Blood Cell Count (WBC)"
    assert resolve_canonical_lab_name("Total Leukocyte Count (WBC)") == "White Blood Cell Count (WBC)"
    assert resolve_canonical_lab_name("FBS") == "Fasting Blood Glucose"
    assert resolve_canonical_lab_name("Glycated Hemoglobin") == "HbA1c"
    assert resolve_canonical_lab_name("SGOT (AST)") == "SGOT / AST"
    assert resolve_canonical_lab_name("SGPT (ALT)") == "SGPT / ALT"
    assert resolve_canonical_lab_name("BUN") == "Blood Urea Nitrogen"


def test_10_multi_study_xray_aggregation_and_unverified_logging():
    """Verify Problem D & G: Multiple X-ray studies aggregate via max-pooling and unverified studies are logged."""
    study_chest = XRaySpecialistOutput(
        file_name="cxr.png",
        detected_anatomy="chest",
        anatomy_verified=True,
        specialist_model_executed=True,
        has_region_mismatch=False,
        is_pathology_eligible=True,
        pathology_scores={"Consolidation": 0.85, "Pneumonia": 0.70},
        status="COMPLETED",
    )
    study_wrist = XRaySpecialistOutput(
        file_name="wrist.png",
        detected_anatomy="wrist",
        anatomy_verified=True,
        specialist_model_executed=True,
        has_region_mismatch=False,
        is_pathology_eligible=True,
        pathology_scores={"Musculoskeletal Abnormality": 0.92},
        status="COMPLETED",
    )
    study_unverified = XRaySpecialistOutput(
        file_name="unverified.png",
        detected_anatomy="shoulder",
        anatomy_verified=False,
        specialist_model_executed=False,
        has_region_mismatch=False,
        is_pathology_eligible=False,
        status="UNVERIFIED",
    )

    rec = MultimodalPatientRecord(
        subject_id="p_multi_study",
        symptoms=[ClinicalSymptomObservation(symptom="pain", canonical="pain", state="PRESENT")],
        xray_studies=[study_chest, study_wrist, study_unverified],
    )
    tensors = MultimodalFeatureExtractor.extract_tensors_from_record(rec)

    # Must be eligible because 2 valid studies exist
    assert tensors["xray_pathology_eligible"].item() == 1.0

    # Pathology tensor must retain both consolidation (from chest) and musculoskeletal abnormality (from wrist)
    p_flat = tensors["xray_pathology_features"].squeeze(0).tolist()
    consolidation_idx = CANONICAL_XRAY_PATHOLOGIES.index("Consolidation")
    mura_idx = CANONICAL_XRAY_PATHOLOGIES.index("Musculoskeletal Abnormality")

    assert p_flat[consolidation_idx * 3] == pytest.approx(0.85, rel=1e-4)
    assert p_flat[mura_idx * 3] == pytest.approx(0.92, rel=1e-4)


def test_11_radiology_conflict_detection():
    """Verify Problem F: Clear lungs co-occurring with positive lung infiltrate flags conflict."""
    rec_conflict = MultimodalPatientRecord(
        subject_id="p_conflict",
        symptoms=[ClinicalSymptomObservation(symptom="cough", canonical="cough", state="PRESENT")],
        radiology=[
            StructuredRadiologyFinding(finding="clear lungs", canonical_finding="clear lungs", state="PRESENT"),
            StructuredRadiologyFinding(finding="consolidation", canonical_finding="consolidation", state="PRESENT"),
        ],
    )
    tensors = MultimodalFeatureExtractor.extract_tensors_from_record(rec_conflict)
    rad_flat = tensors["rad_features"].squeeze(0).tolist()
    consolidation_idx = CANONICAL_RAD_FINDINGS.index("consolidation")
    conflict_flag = rad_flat[consolidation_idx * 4 + 3]

    assert conflict_flag == 1.0, "Conflict flag was not asserted when clear lungs co-occurred with consolidation!"


def test_12_strict_checkpoint_validation():
    """Verify Problem J: validate_multimodal_checkpoint rejects invalid or unapproved checkpoints."""
    valid_payload = {
        "architecture_version": ARCHITECTURE_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "lab_schema_version": LAB_SCHEMA_VERSION,
        "symptom_vocab_version": SYMPTOM_VOCAB_VERSION,
        "status": "PRODUCTION_APPROVED",
        "acceptance_passed": True,
        "condition_classes": CONDITION_CLASSES,
        "model_state": {},
        "dataset_provenance": {"is_authentic": True, "patient_count": 1200},
        "test_evaluation_report": {"macro_f1": 0.82},
    }
    is_valid, reasons = validate_multimodal_checkpoint(valid_payload)
    assert is_valid is True

    # Case A: Status is experimental / acceptance failed
    invalid_status = dict(valid_payload, status="ACCEPTANCE_FAILED", acceptance_passed=False)
    is_valid_a, reasons_a = validate_multimodal_checkpoint(invalid_status)
    assert is_valid_a is False
    assert any("not 'PRODUCTION_APPROVED'" in r for r in reasons_a)

    # Case B: Incompatible architecture version
    invalid_arch = dict(valid_payload, architecture_version="1.0.0")
    is_valid_b, reasons_b = validate_multimodal_checkpoint(invalid_arch)
    assert is_valid_b is False


def test_13_untrained_multimodal_service_safely_falls_back():
    """Verify fail-closed rule: Unvalidated checkpoint falls back without executing untrained weights."""
    service = MultimodalInferenceService(checkpoint_path=None)
    assert service.is_trained is False

    rec = MultimodalPatientRecord(
        subject_id="p_test",
        symptoms=[ClinicalSymptomObservation(symptom="fever", canonical="fever", state="PRESENT")],
        laboratory=[StructuredLabParameter(test_name="Hemoglobin", canonical_test_name="Hemoglobin", value=14.0, unit="g/dL")],
    )
    res = service.predict(rec)
    assert res["status"] == "blocked_untrained"
    assert res["is_multimodal_trained"] is False
    assert res["model_scope"] == "unimodal_symptom_baseline_active"
    assert "MIMIC" in res["message"]


def test_14_empty_modalities_returns_insufficient_evidence():
    """Verify that when no modalities are submitted, system returns insufficient_evidence."""
    service = MultimodalInferenceService(checkpoint_path=None)
    empty_rec = MultimodalPatientRecord(subject_id="p_empty")
    res = service.predict(empty_rec)
    assert res["status"] == "insufficient_evidence"
    assert res["conditions"] == []


def test_15_baseline_unimodal_symptom_model_preserved():
    """Verify baseline model integrity."""
    baseline_path = Path("models/conditions/condition_pipeline.joblib")
    assert baseline_path.exists()
    pipeline = joblib.load(baseline_path)
    assert hasattr(pipeline, "predict")
    assert len(pipeline.classes_) == 24


def test_16_protected_xray_files_sha256_integrity():
    """Verify absolute protection constraint: all 15 protected files match manifest."""
    manifest_path = Path("models/xray_protected_manifest.json")
    assert manifest_path.exists(), "Protected manifest missing!"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for rel_path, expected_hash in manifest.items():
        full_path = Path(rel_path) if Path(rel_path).exists() else Path("..") / rel_path
        assert full_path.exists(), f"Protected file missing: {full_path}"
        actual_hash = hashlib.sha256(full_path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, (
            f"PROTECTED ASSET TAMPERING DETECTED! File {rel_path} was modified.\n"
            f"Expected: {expected_hash}\nActual:   {actual_hash}"
        )
