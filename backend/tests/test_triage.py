"""Unit tests for rule-based Triage and Safety Engine."""

from app.ai.triage_engine import apply_triage


def test_triage_does_not_force_pathway_on_insufficient():
    result = apply_triage({"evidence_state": "INSUFFICIENT_EVIDENCE"})
    assert result["pathway"] is None
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["clinically_validated"] is False


def test_triage_emergency_on_chest_pain():
    evidence = {
        "evidence_state": "EVIDENCE_PRESENT",
        "symptoms": {
            "present": [{"symptom": "chest pain", "severity": "severe"}],
            "absent": [],
        },
        "report": {"abnormal_findings": []},
    }
    result = apply_triage(evidence)
    assert result["pathway"] == "EMERGENCY"
    assert any("chest_pain" in r for r in result["rules_triggered"])


def test_triage_consultation_on_abnormal_labs():
    evidence = {
        "evidence_state": "EVIDENCE_PRESENT",
        "symptoms": {
            "present": [{"symptom": "fatigue", "severity": "mild"}],
            "absent": [{"symptom": "chest pain"}],
        },
        "report": {
            "abnormal_findings": [
                {"test_name": "Fasting Blood Glucose", "value": 160.0, "interpretation": "HIGH"}
            ]
        },
    }
    result = apply_triage(evidence)
    assert result["pathway"] == "CONSULTATION"


def test_triage_mild_when_no_red_flags():
    evidence = {
        "evidence_state": "EVIDENCE_PRESENT",
        "symptoms": {
            "present": [{"symptom": "rhinorrhea", "severity": "mild"}],
            "absent": [{"symptom": "chest pain"}, {"symptom": "shortness of breath"}],
        },
        "report": {"abnormal_findings": []},
    }
    result = apply_triage(evidence)
    assert result["pathway"] == "MILD"
