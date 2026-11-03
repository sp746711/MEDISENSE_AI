"""Tests for Authentic Multimodal Dataset Ingestion and Linkage Pipeline (Part 7)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import subprocess
import sys

from app.ai.multimodal_condition_model import CONDITION_CLASSES
from app.ai.multimodal_dataset_ingestion import (
    AuthenticMultimodalDatasetIngester,
    CrosswalkSigningAuthority,
    ICD_TO_CONDITION_MAP,
    IngestionAuditReport,
    create_authenticated_imaging_crosswalk,
)


def test_resolve_condition_label_direct_and_icd():
    # Direct canonical name matching
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("Pneumonia") == "Pneumonia"
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("diabetes") == "diabetes"
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("Bronchial Asthma") == "Bronchial Asthma"

    # ICD-9 and ICD-10 crosswalk
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("486") == "Pneumonia"
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("J18.9") == "Pneumonia"
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("250.00") == "diabetes"
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("E11.9") == "diabetes"
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("493.90") == "Bronchial Asthma"
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("J45.909") == "Bronchial Asthma"
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("401.9") == "Hypertension"
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("I10") == "Hypertension"
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("530.81") == "gastroesophageal reflux disease"
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("K21.9") == "gastroesophageal reflux disease"

    # Unsupported or invalid labels
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("UNKNOWN_DISEASE_XYZ") is None
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label("") is None
    assert AuthenticMultimodalDatasetIngester.resolve_condition_label(None) is None


def test_reject_synthetic_and_placeholder_records():
    ingester = AuthenticMultimodalDatasetIngester(data_source_name="MIMIC-IV-TEST", source_version="v2.2")

    records = [
        # Missing subject_id
        {"hadm_id": "H1001", "target_condition": "Pneumonia", "symptoms": ["cough"]},
        # Synthetic subject_id
        {"subject_id": "synthetic_pt_01", "hadm_id": "H1002", "target_condition": "Pneumonia", "symptoms": ["cough"]},
        # Placeholder mock subject_id
        {"subject_id": "mock_patient", "hadm_id": "H1003", "target_condition": "Pneumonia", "symptoms": ["cough"]},
        # Row index used as subject_id (forbidden shortcut)
        {"subject_id": "3", "hadm_id": "H1004", "target_condition": "Pneumonia", "symptoms": ["cough"]},
        # Synthetic encounter
        {"subject_id": "REAL_PATIENT_01", "hadm_id": "dummy_enc_01", "target_condition": "Pneumonia", "symptoms": ["cough"]},
        # Empty modalities
        {"subject_id": "REAL_PATIENT_02", "hadm_id": "H1006", "target_condition": "Pneumonia"},
        # Valid authentic record
        {
            "subject_id": "10002430",
            "hadm_id": "21400215",
            "study_id": "50412891",
            "target_condition": "Pneumonia",
            "admittime": "2026-01-10T10:00:00Z",
            "charttime": "2026-01-10T12:00:00Z",
            "symptoms": ["cough", "fever", "dyspnea"],
            "symptom_tokens": [1, 2, 3] + [0] * 29,
            "laboratory": {"wbc": 14.5},
            "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                subject_id="10002430",
                hadm_id="21400215",
                study_id="50412891",
                source="MIMIC_IV_CXR_ED_TRANSFER",
                source_record_reference="MIMIC_IV_ED.ED_RADIOLOGY",
                signing_authority=CrosswalkSigningAuthority.get_test_authority(),
            ),
            "xray": {"study_id": "50412891", "findings": ["consolidation"]},
        },
    ]

    valid_records, report = ingester.validate_and_ingest_records(records)

    assert len(valid_records) == 1
    assert valid_records[0]["subject_id"] == "10002430"
    assert valid_records[0]["hadm_id"] == "21400215"
    assert valid_records[0]["target_condition"] == "Pneumonia"
    assert valid_records[0]["target_class_idx"] == CONDITION_CLASSES.index("Pneumonia")
    assert valid_records[0]["modality_mask"] == [1.0, 1.0, 0.0, 1.0, 0.0]

    assert report.is_valid is True
    assert report.valid_linked_records == 1
    assert report.total_raw_records == 7
    assert report.unique_patients_count == 1
    assert report.rejected_records_count == 6


def test_reject_xray_without_authentic_study_id():
    ingester = AuthenticMultimodalDatasetIngester()

    # Record has X-ray data but missing authentic study_id
    rec_bad = {
        "subject_id": "10002430",
        "hadm_id": "21400215",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"findings": ["consolidation"]},
        "study_id": "",  # missing!
    }

    valid_records, report = ingester.validate_and_ingest_records([rec_bad])
    assert len(valid_records) == 0
    assert report.is_valid is False
    assert "xray_missing_authentic_study_id" in report.rejection_reasons


def test_reject_temporal_linkage_violation():
    ingester = AuthenticMultimodalDatasetIngester()

    # Lab charttime precedes admission by 72 hours (> 48h limit)
    rec_stale = {
        "subject_id": "10002430",
        "hadm_id": "21400215",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "admittime": "2026-01-10T12:00:00Z",
        "charttime": "2026-01-05T12:00:00Z",  # 5 days prior
    }

    valid_records, report = ingester.validate_and_ingest_records([rec_stale])
    assert len(valid_records) == 0
    assert report.is_valid is False
    assert "temporal_linkage_violation" in report.rejection_reasons


def test_cli_ingestion_script(tmp_path):
    # Create test input file
    input_file = tmp_path / "raw_cohort.json"
    output_file = tmp_path / "linked_cohort.json"
    report_file = tmp_path / "ingest_report.json"

    data = [
        {
            "subject_id": "10001001",
            "hadm_id": "20002001",
            "study_id": "50003001",
            "icd_code": "J18.9",
            "symptoms": ["fever", "cough"],
            "symptom_tokens": [10, 20] + [0] * 30,
            "laboratory": {"crp": 45.0},
            "radiology": {"consolidation": 1.0},
        },
        {
            "subject_id": "10001002",
            "hadm_id": "20002002",
            "target_condition": "diabetes",
            "symptoms": ["polyuria", "polydipsia"],
            "laboratory": {"fasting_blood_glucose": 180.0},
        },
    ]

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "scripts" / "ingest_authentic_multimodal_data.py"),
        "--input",
        str(input_file),
        "--output",
        str(output_file),
        "--report",
        str(report_file),
        "--source",
        "MIMIC-IV-TEST",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI error: {result.stderr}"

    assert output_file.exists()
    assert report_file.exists()

    with open(output_file, "r", encoding="utf-8") as f:
        saved_records = json.load(f)
    assert len(saved_records) == 2
    assert saved_records[0]["target_condition"] == "Pneumonia"
    assert saved_records[1]["target_condition"] == "diabetes"

    with open(report_file, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["is_valid"] is True
    assert rep["valid_linked_records"] == 2
    assert rep["unique_patients_count"] == 2
