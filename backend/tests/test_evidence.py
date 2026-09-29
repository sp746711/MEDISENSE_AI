"""Unit tests for Evidence Combination Engine."""

from app.ai.evidence_engine import combine_evidence


def test_evidence_insufficient_by_default():
    result = combine_evidence()
    assert result["evidence_state"] == "INSUFFICIENT_EVIDENCE"
    assert result["symptoms"]["present"] == []


def test_evidence_multimodal_combination():
    symptoms = [
        {"symptom": "cough", "state": "PRESENT", "duration": "3 days", "domain": "respiratory"},
        {"symptom": "chest pain", "state": "ABSENT", "domain": "cardiovascular"},
    ]
    reports = [
        {"test_name": "Fasting Blood Glucose", "value": 135.0, "unit": "mg/dL", "interpretation": "HIGH", "domain": "endocrine"}
    ]
    result = combine_evidence(symptoms=symptoms, report_findings=reports)
    assert result["evidence_state"] == "EVIDENCE_PRESENT"
    assert len(result["symptoms"]["present"]) == 1
    assert len(result["symptoms"]["absent"]) == 1
    assert len(result["report"]["abnormal_findings"]) == 1
    assert any("chest pain" in item for item in ["chest pain"] if "chest pain" not in result["missing"])
