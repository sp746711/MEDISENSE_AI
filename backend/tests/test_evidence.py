"""Tests for the MediSense AI evidence combination engine.

Run from the backend directory:
    python -m pytest .\tests\test_evidence.py -q

These tests validate evidence-state behavior and core evidence aggregation.
They do not validate clinical accuracy or replace clinical review.
"""

from app.ai.evidence_engine import combine_evidence


def test_evidence_insufficient_by_default():
    result = combine_evidence()
    assert result["evidence_state"] == "INSUFFICIENT_EVIDENCE"
    assert result["symptoms"]["present"] == []


def test_evidence_multimodal_combination():
    symptoms = [
        {
            "symptom": "cough",
            "state": "PRESENT",
            "duration": "3 days",
            "domain": "respiratory",
        },
        {
            "symptom": "chest pain",
            "state": "ABSENT",
            "domain": "cardiovascular",
        },
    ]
    reports = [
        {
            "test_name": "Fasting Blood Glucose",
            "value": 135.0,
            "unit": "mg/dL",
            "interpretation": "HIGH",
            "domain": "endocrine",
        }
    ]

    result = combine_evidence(symptoms=symptoms, report_findings=reports)

    assert result["evidence_state"] == "EVIDENCE_PRESENT"
    assert len(result["symptoms"]["present"]) == 1
    assert len(result["symptoms"]["absent"]) == 1
    assert len(result["report"]["abnormal_findings"]) == 1


def test_absent_symptoms_only_are_insufficient():
    result = combine_evidence(
        symptoms=[
            {"symptom": "chest pain", "state": "ABSENT", "domain": "cardiovascular"}
        ]
    )
    assert result["evidence_state"] == "INSUFFICIENT_EVIDENCE"
    assert result["symptoms"]["present"] == []
    assert len(result["symptoms"]["absent"]) == 1


def test_historical_symptoms_only_are_insufficient():
    result = combine_evidence(
        symptoms=[
            {
                "symptom": "headache",
                "state": "PRESENT",
                "type": "HISTORICAL",
                "domain": "neurological",
            }
        ]
    )
    assert result["evidence_state"] == "INSUFFICIENT_EVIDENCE"


def test_empty_uploaded_medical_report_is_insufficient():
    result = combine_evidence(
        medical_reports=[{"file_name": "empty_report.pdf"}],
        medical_report_provided=True,
    )
    assert result["evidence_state"] == "INSUFFICIENT_EVIDENCE"
    assert result["report"]["provided"] is True


def test_unavailable_xray_without_other_findings_is_insufficient():
    result = combine_evidence(
        xray_results=[
            {
                "status": "UNAVAILABLE",
                "prediction": None,
                "finding": "",
            }
        ]
    )
    assert result["evidence_state"] == "INSUFFICIENT_EVIDENCE"
    assert result["xray"]["received"] is True
    assert result["xray"]["status"] == "UNAVAILABLE"


def test_current_present_symptom_counts_as_evidence():
    result = combine_evidence(
        symptoms=[
            {
                "symptom": "cough",
                "state": "PRESENT",
                "type": "CURRENT",
                "domain": "respiratory",
            }
        ]
    )
    assert result["evidence_state"] == "EVIDENCE_PRESENT"
    assert len(result["symptoms"]["present"]) == 1


def test_empty_qualitative_report_record_is_not_evidence():
    result = combine_evidence(
        qualitative_report_findings=[{}],
    )
    assert result["evidence_state"] == "INSUFFICIENT_EVIDENCE"


def test_synthetic_placeholder_narrative_is_not_clinical_evidence():
    result = combine_evidence(
        narrative_report_findings=[
            {"content": "SYNTHETIC TEST DATA - example report, no actual clinical findings"}
        ]
    )
    assert result["evidence_state"] == "INSUFFICIENT_EVIDENCE"
