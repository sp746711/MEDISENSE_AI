"""Unit and regression tests for Goal 2 Step 5:
Controlled Model Decoupling and Temporal Validation Implementation.

Tests:
1. Track A retains exactly 24 classes.
2. Track B has exactly four independent classes.
3. Track B ICD mapping cannot alter Track A mapping.
4. Primary and secondary diagnosis handling.
5. Valid laboratory timestamps.
6. Labs before the allowed pre-admission window.
7. Labs after discharge.
8. Missing and malformed timestamps.
9. Imaging studies outside the encounter window.
10. Imaging studies without reliable encounter linkage.
11. Unlinked X-ray studies are excluded from encounter-level supervision.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import pytest

from app.ai.multimodal_condition_model import (
    CONDITION_CLASSES,
    INPATIENT_MULTIMODAL_CLASSES,
)
from app.ai.multimodal_dataset_ingestion import (
    AuthenticMultimodalDatasetIngester,
    ICD_TO_CONDITION_MAP,
    INPATIENT_ICD_TO_CONDITION_MAP,
    parse_clinical_timestamp,
)


def test_track_a_retains_exactly_24_classes():
    """1. Track A retains exactly 24 classes, preserving baseline vocabulary and artifact."""
    assert len(CONDITION_CLASSES) == 24, f"Expected 24 classes, got {len(CONDITION_CLASSES)}"

    # Check baseline condition classes JSON artifact on disk
    backend_root = Path(__file__).resolve().parents[1]
    classes_json_path = backend_root / "models" / "conditions" / "condition_classes.json"
    assert classes_json_path.exists(), f"Missing {classes_json_path}"

    with open(classes_json_path, "r", encoding="utf-8") as f:
        disk_classes = json.load(f)

    assert len(disk_classes) == 24
    assert disk_classes == list(CONDITION_CLASSES)

    # Check baseline model pipeline artifact is present
    pipeline_path = backend_root / "models" / "conditions" / "condition_pipeline.joblib"
    assert pipeline_path.exists(), f"Missing {pipeline_path}"


def test_track_b_has_exactly_four_independent_classes():
    """2. Track B has exactly four independent classes, separate from Track A."""
    expected_classes = [
        "Pneumonia",
        "Acute Asthma Exacerbation",
        "Peptic Ulcer Disease",
        "Urinary Tract Infection",
    ]
    assert len(INPATIENT_MULTIMODAL_CLASSES) == 4
    assert list(INPATIENT_MULTIMODAL_CLASSES) == expected_classes

    # Ensure Track B is not identical to Track A
    assert set(INPATIENT_MULTIMODAL_CLASSES) != set(CONDITION_CLASSES)

    # Ensure acute inpatient exacerbation is distinguished from outpatient chronic asthma
    assert "Acute Asthma Exacerbation" in INPATIENT_MULTIMODAL_CLASSES
    assert "Bronchial Asthma" in CONDITION_CLASSES
    assert "Bronchial Asthma" not in INPATIENT_MULTIMODAL_CLASSES


def test_track_b_icd_mapping_cannot_alter_track_a_mapping():
    """3. Track B ICD mapping is separate and cannot alter Track A mapping."""
    # Maps must be separate dict instances
    assert ICD_TO_CONDITION_MAP is not INPATIENT_ICD_TO_CONDITION_MAP

    # Test baseline resolution
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("493.90", taxonomy="baseline") == "Bronchial Asthma"
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("J45.909", taxonomy="baseline") == "Bronchial Asthma"

    # Test inpatient resolution: chronic uncomplicated asthma (493.90 / J45.909) must be rejected
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("493.90", taxonomy="inpatient") is None
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("J45.909", taxonomy="inpatient") is None

    # Acute exacerbation codes (493.92 / J45.901) resolve to Track B class
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("493.92", taxonomy="inpatient") == "Acute Asthma Exacerbation"
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("J45.901", taxonomy="inpatient") == "Acute Asthma Exacerbation"

    # Mutating Track B map must not affect Track A map
    test_key = "MOCK_CODE_999"
    try:
        INPATIENT_ICD_TO_CONDITION_MAP[test_key] = "Pneumonia"
        assert test_key not in ICD_TO_CONDITION_MAP
    finally:
        INPATIENT_ICD_TO_CONDITION_MAP.pop(test_key, None)


def test_primary_and_secondary_diagnosis_handling():
    """4. Primary and secondary diagnosis handling: reject comorbidities without silent conversion."""
    ingester_inpatient = AuthenticMultimodalDatasetIngester(
        data_source_name="MIMIC-IV-TEST",
        target_taxonomy="inpatient",
        enforce_primary_diagnosis=True,
    )

    records = [
        # Record 1: Primary diagnosis (seq_num=1)
        {
            "subject_id": "10002001",
            "hadm_id": "20003001",
            "target_condition": "Pneumonia",
            "seq_num": 1,
            "priority": "PRIMARY",
            "symptoms": ["cough", "fever"],
            "admittime": "2026-03-01T10:00:00Z",
            "charttime": "2026-03-01T12:00:00Z",
        },
        # Record 2: Secondary comorbidity (seq_num=2)
        {
            "subject_id": "10002002",
            "hadm_id": "20003002",
            "target_condition": "Pneumonia",
            "seq_num": 2,
            "priority": "SECONDARY",
            "symptoms": ["cough"],
            "admittime": "2026-03-01T10:00:00Z",
            "charttime": "2026-03-01T12:00:00Z",
        },
    ]

    valid, report = ingester_inpatient.validate_and_ingest_records(records)

    assert len(valid) == 1
    assert valid[0]["subject_id"] == "10002001"
    assert valid[0]["seq_num"] == 1
    assert valid[0]["is_primary_diagnosis"] is True
    assert valid[0]["diagnosis_priority"] == "PRIMARY"

    assert report.rejected_records_count == 1
    assert "secondary_comorbidity_not_primary_target" in report.rejection_reasons


def test_valid_laboratory_timestamps_within_window():
    """5. Valid laboratory timestamps within admittime - 48h <= charttime <= dischtime."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10002003",
        "hadm_id": "20003003",
        "target_condition": "Pneumonia",
        "admittime": "2026-03-02T10:00:00Z",
        "dischtime": "2026-03-06T15:00:00Z",
        "charttime": "2026-03-02T08:00:00Z",  # 2 hours before admission (within 48h window)
        "laboratory": {"wbc": 12.0},
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    assert valid[0]["temporal_linkage_evidence"]["lab_in_window"] is True
    assert report.is_valid is True


def test_labs_before_allowed_preadmission_window():
    """6. Labs before the allowed pre-admission window (> 48 hours prior) are rejected."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10002004",
        "hadm_id": "20003004",
        "target_condition": "Pneumonia",
        "admittime": "2026-03-05T12:00:00Z",
        "charttime": "2026-03-02T10:00:00Z",  # 74 hours before admission (> 48h)
        "laboratory": {"crp": 25.0},
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 0
    assert report.is_valid is False
    assert "lab_precedes_admission_window" in report.rejection_reasons
    assert "temporal_linkage_violation" in report.rejection_reasons


def test_labs_after_discharge():
    """7. Labs charted after patient discharge are strictly rejected."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10002005",
        "hadm_id": "20003005",
        "target_condition": "Pneumonia",
        "admittime": "2026-03-01T10:00:00Z",
        "dischtime": "2026-03-04T12:00:00Z",
        "charttime": "2026-03-05T09:00:00Z",  # 21 hours after discharge
        "laboratory": {"creatinine": 1.2},
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 0
    assert report.is_valid is False
    assert "lab_after_discharge" in report.rejection_reasons
    assert "temporal_linkage_violation" in report.rejection_reasons


def test_missing_and_malformed_timestamps():
    """8. Missing and malformed timestamps are detected and never silently ignored."""
    # Test strict parser
    dt, err = parse_clinical_timestamp("not_a_timestamp")
    assert dt is None
    assert err is not None
    assert "malformed_timestamp" in err

    dt_valid, err_valid = parse_clinical_timestamp("2026-04-10T14:30:00Z")
    assert dt_valid is not None
    assert err_valid is None

    ingester = AuthenticMultimodalDatasetIngester()
    rec_bad_admit = {
        "subject_id": "10002006",
        "hadm_id": "20003006",
        "target_condition": "Pneumonia",
        "admittime": "GARBAGE_DATE_STR",
        "charttime": "2026-04-10T14:30:00Z",
        "laboratory": {"wbc": 10.0},
    }
    valid, report = ingester.validate_and_ingest_records([rec_bad_admit])
    assert len(valid) == 0
    assert "malformed_timestamp" in report.rejection_reasons
    assert "temporal_linkage_violation" in report.rejection_reasons


def test_imaging_studies_outside_encounter_window():
    """9. Imaging studies outside the encounter window (admittime - 24h <= study_time <= dischtime) are rejected."""
    ingester = AuthenticMultimodalDatasetIngester()

    # Case A: Imaging precedes admission by 36 hours (> 24h limit)
    rec_early_img = {
        "subject_id": "10002007",
        "hadm_id": "20003007",
        "study_id": "50001001",
        "target_condition": "Pneumonia",
        "admittime": "2026-03-03T12:00:00Z",
        "dischtime": "2026-03-07T12:00:00Z",
        "study_time": "2026-03-01T12:00:00Z",  # 48 hours prior
        "xray": {"study_id": "50001001", "findings": ["consolidation"]},
    }
    valid_a, report_a = ingester.validate_and_ingest_records([rec_early_img])
    assert len(valid_a) == 0
    assert "imaging_precedes_admission_window" in report_a.rejection_reasons

    # Case B: Imaging after discharge
    rec_late_img = {
        "subject_id": "10002008",
        "hadm_id": "20003008",
        "study_id": "50001002",
        "target_condition": "Pneumonia",
        "admittime": "2026-03-01T12:00:00Z",
        "dischtime": "2026-03-04T12:00:00Z",
        "study_time": "2026-03-05T12:00:00Z",  # 24 hours after discharge
        "xray": {"study_id": "50001002", "findings": ["consolidation"]},
    }
    valid_b, report_b = ingester.validate_and_ingest_records([rec_late_img])
    assert len(valid_b) == 0
    assert "imaging_after_discharge" in report_b.rejection_reasons


def test_imaging_studies_without_reliable_encounter_linkage():
    """10. Imaging studies without reliable encounter linkage are rejected or flagged."""
    # When strict study timestamp is required but missing:
    ingester_strict = AuthenticMultimodalDatasetIngester(require_study_timestamp=True)

    rec_missing_time = {
        "subject_id": "10002009",
        "hadm_id": "20003009",
        "study_id": "50001003",
        "target_condition": "Pneumonia",
        "admittime": "2026-03-01T12:00:00Z",
        "xray": {"study_id": "50001003"},
        # study_time is missing!
    }
    valid, report = ingester_strict.validate_and_ingest_records([rec_missing_time])
    assert len(valid) == 0
    assert "missing_study_timestamp" in report.rejection_reasons


def test_unlinked_xray_studies_excluded_from_supervision():
    """11. Unlinked X-ray studies are excluded from encounter-level supervised training."""
    ingester = AuthenticMultimodalDatasetIngester()

    # Record has valid symptoms and labs, but the X-ray study cannot be reliably linked to this encounter
    rec_unlinked_study = {
        "subject_id": "10002010",
        "hadm_id": "20003010",
        "study_id": "50001004",
        "target_condition": "Pneumonia",
        "admittime": "2026-03-01T12:00:00Z",
        "charttime": "2026-03-01T14:00:00Z",
        "symptoms": ["cough", "fever"],
        "laboratory": {"wbc": 13.0},
        "xray": {"study_id": "50001004"},
        "imaging_study_linked": False,  # Encounter linkage verification failed
    }

    valid, report = ingester.validate_and_ingest_records([rec_unlinked_study])
    assert len(valid) == 1
    record = valid[0]

    # The encounter itself is retained with symptom and lab modalities
    assert record["modality_mask"][0] == 1.0  # symptoms
    assert record["modality_mask"][1] == 1.0  # laboratory
    # But X-ray is excluded from supervision!
    assert record["modality_mask"][3] == 0.0  # X-ray masked out
    assert record["xray_pathology_eligible"] == 0.0
    assert record["imaging_study_linked"] is False
    assert record["temporal_linkage_evidence"]["study_encounter_linked"] is False
