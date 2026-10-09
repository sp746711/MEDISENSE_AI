"""Automated validation tests for exact demo assessment case.

Verifies end-to-end extraction, evidence synthesis, triage, and feedback:
1. "mild chest discomfort when coughing" is not converted to severe chest pain.
2. "no severe breathing difficulty" becomes ABSENT.
3. "no chest pain when resting" becomes ABSENT for chest pain at rest.
4. Positive chest discomfort is NOT removed by negative chest pain at rest.
5. Fever without explicit severity remains UNKNOWN severity.
6. Dry cough without explicit severity remains UNKNOWN severity.
7. Sore throat without explicit severity remains UNKNOWN severity.
8. Headache without explicit severity remains UNKNOWN severity.
9. Tiredness without explicit severity remains UNKNOWN severity.
10. No vomiting becomes ABSENT.
11. No fainting becomes ABSENT.
12. Medical report values are extracted only when actually present.
13. X-ray unavailable status is preserved when the model is unavailable.
14. X-ray findings are never fabricated.
15. Triage does not become EMERGENCY because multiple ordinary symptoms exist.
16. Final feedback does not contain unsupported diagnosis.
"""

from __future__ import annotations

from pathlib import Path

from app.ai.evidence_engine import combine_evidence
from app.ai.report_nlp import structure_report_text
from app.ai.report_ocr import extract_text_from_file
from app.ai.symptom_nlp import extract_symptoms
from app.ai.triage_engine import apply_triage
from app.ai.xray_model import XRayModelService
from app.services.assessment_service import generate_assessment_feedback

EXACT_DEMO_TEXT = """I have had fever for 3 days. My temperature is around 101°F.
I have a dry cough, sore throat, headache and tiredness.
I also feel mild chest discomfort when coughing.
The symptoms started gradually 3 days ago.
I have mild difficulty sleeping because of the cough.
I do not have severe breathing difficulty.
I do not have chest pain when resting.
No vomiting or fainting."""

DEMO_REPORT_PDF = Path("uploads/reports/935059528b22489b8fd2b08942563c40.pdf")


def test_1_mild_chest_discomfort_not_converted_to_severe_chest_pain():
    res = extract_symptoms(EXACT_DEMO_TEXT)
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    assert "chest discomfort" in symptoms
    cd = symptoms["chest discomfort"]
    assert cd["state"] == "PRESENT"
    assert cd["severity"].lower() == "mild"
    assert cd.get("trigger") == "coughing"
    assert "severe" not in cd["severity"].lower()


def test_2_no_severe_breathing_difficulty_becomes_absent():
    res = extract_symptoms(EXACT_DEMO_TEXT)
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    assert "severe breathing difficulty" in symptoms
    sbd = symptoms["severe breathing difficulty"]
    assert sbd["state"] == "ABSENT"
    # Must NOT create shortness of breath PRESENT
    assert symptoms.get("shortness of breath", {}).get("state") != "PRESENT"


def test_3_no_chest_pain_when_resting_becomes_absent():
    res = extract_symptoms(EXACT_DEMO_TEXT)
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    assert "chest pain at rest" in symptoms
    cp_rest = symptoms["chest pain at rest"]
    assert cp_rest["state"] == "ABSENT"


def test_4_positive_chest_discomfort_not_removed_by_negative_chest_pain_at_rest():
    res = extract_symptoms(EXACT_DEMO_TEXT)
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    assert "chest discomfort" in symptoms
    assert symptoms["chest discomfort"]["state"] == "PRESENT"
    assert "chest pain at rest" in symptoms
    assert symptoms["chest pain at rest"]["state"] == "ABSENT"


def test_5_fever_without_explicit_severity_remains_unknown():
    res = extract_symptoms(EXACT_DEMO_TEXT)
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    assert "fever" in symptoms
    fever = symptoms["fever"]
    assert fever["state"] == "PRESENT"
    assert fever["severity"] == "UNKNOWN"
    assert "3 days" in fever["duration"]


def test_6_dry_cough_without_explicit_severity_remains_unknown():
    res = extract_symptoms(EXACT_DEMO_TEXT)
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    # Either canonical cough with type=dry or dry cough
    cough = symptoms.get("cough") or symptoms.get("dry cough")
    assert cough is not None
    assert cough["state"] == "PRESENT"
    assert cough["severity"] == "UNKNOWN"
    if "type" in cough:
        assert cough["type"] == "dry"


def test_7_sore_throat_without_explicit_severity_remains_unknown():
    res = extract_symptoms(EXACT_DEMO_TEXT)
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    assert "sore throat" in symptoms
    st = symptoms["sore throat"]
    assert st["state"] == "PRESENT"
    assert st["severity"] == "UNKNOWN"


def test_8_headache_without_explicit_severity_remains_unknown():
    res = extract_symptoms(EXACT_DEMO_TEXT)
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    assert "headache" in symptoms
    ha = symptoms["headache"]
    assert ha["state"] == "PRESENT"
    assert ha["severity"] == "UNKNOWN"


def test_9_tiredness_without_explicit_severity_remains_unknown():
    res = extract_symptoms(EXACT_DEMO_TEXT)
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    fatigue = symptoms.get("tiredness/fatigue") or symptoms.get("fatigue") or symptoms.get("tiredness")
    assert fatigue is not None
    assert fatigue["state"] == "PRESENT"
    assert fatigue["severity"] == "UNKNOWN"


def test_10_no_vomiting_becomes_absent():
    res = extract_symptoms(EXACT_DEMO_TEXT)
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    assert "vomiting" in symptoms
    assert symptoms["vomiting"]["state"] == "ABSENT"


def test_11_no_fainting_becomes_absent():
    res = extract_symptoms(EXACT_DEMO_TEXT)
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    fainting = symptoms.get("fainting") or symptoms.get("fainting/syncope")
    assert fainting is not None
    assert fainting["state"] == "ABSENT"


def test_12_medical_report_values_extracted_only_when_actually_present():
    if DEMO_REPORT_PDF.exists():
        ocr_res = extract_text_from_file(str(DEMO_REPORT_PDF))
        nlp_res = structure_report_text(ocr_res["extracted_text"], tables=ocr_res.get("tables"))
        findings = nlp_res["findings"]
        assert len(findings) == 8
        tests_extracted = {f["test_name"]: f for f in findings}
        assert "Hemoglobin" in tests_extracted
        assert tests_extracted["Hemoglobin"]["value"] == 13.8
        assert tests_extracted["Hemoglobin"]["interpretation"] == "NORMAL"

        assert "Fasting Blood Glucose" in tests_extracted
        assert tests_extracted["Fasting Blood Glucose"]["value"] == 102.0
        assert tests_extracted["Fasting Blood Glucose"]["interpretation"] == "HIGH"

        assert "White Blood Cell Count (WBC)" in tests_extracted
        assert tests_extracted["White Blood Cell Count (WBC)"]["value"] == 7200.0


def test_13_xray_unavailable_status_preserved_when_model_unavailable():
    srv = XRayModelService()
    # No checkpoint is configured
    analysis = srv.analyze("uploads/xrays/d493d20c2f484383bbfc263a5b180671.png")
    assert analysis["status"] == "unavailable"
    assert analysis["prediction"] is None
    assert "unavailable" in analysis["message"].lower()


def test_14_xray_findings_never_fabricated():
    evidence = combine_evidence(
        symptoms=[],
        report_findings=[],
        xray_result={
            "status": "unavailable",
            "region": "chest",
            "prediction": None,
            "model_version": None,
            "uncertainty": "Model not configured",
            "message": "Interpretation unavailable.",
        },
    )
    xray = evidence["xray"]
    assert xray["status"] == "UNAVAILABLE"
    assert xray["prediction"] is None
    assert xray["model_version"] == "N/A"
    # Never claim normal when unavailable
    assert xray.get("prediction") != "Normal"
    assert xray.get("prediction") != "Pneumonia"


def test_15_triage_does_not_become_emergency_on_ordinary_symptoms():
    sym_res = extract_symptoms(EXACT_DEMO_TEXT)
    report_findings = []
    if DEMO_REPORT_PDF.exists():
        ocr_res = extract_text_from_file(str(DEMO_REPORT_PDF))
        nlp_res = structure_report_text(ocr_res["extracted_text"], tables=ocr_res.get("tables"))
        report_findings = nlp_res["findings"]

    xray_result = {
        "status": "unavailable",
        "region": "chest",
        "prediction": None,
        "model_version": None,
        "uncertainty": "Model not configured",
        "message": "Interpretation unavailable.",
    }

    evidence = combine_evidence(
        symptoms=sym_res["symptoms"],
        report_findings=report_findings,
        xray_result=xray_result,
    )
    triage = apply_triage(evidence)
    assert triage["pathway"] != "EMERGENCY"
    assert triage["pathway"] == "CONSULTATION"
    assert "rule_e1_chest_pain_present" not in triage["rules_triggered"]
    assert "rule_e3_severe_acute_presentation" not in triage["rules_triggered"]


def test_16_final_feedback_does_not_contain_unsupported_diagnosis():
    sym_res = extract_symptoms(EXACT_DEMO_TEXT)
    report_findings = []
    if DEMO_REPORT_PDF.exists():
        ocr_res = extract_text_from_file(str(DEMO_REPORT_PDF))
        nlp_res = structure_report_text(ocr_res["extracted_text"], tables=ocr_res.get("tables"))
        report_findings = nlp_res["findings"]

    xray_result = {
        "status": "unavailable",
        "region": "chest",
        "prediction": None,
        "model_version": None,
        "uncertainty": "Model not configured",
        "message": "Interpretation unavailable.",
    }

    evidence = combine_evidence(
        symptoms=sym_res["symptoms"],
        report_findings=report_findings,
        xray_result=xray_result,
    )
    triage = apply_triage(evidence)
    feedback = generate_assessment_feedback(evidence, triage, specialty="General Physician")

    summary_text = feedback["assessment_summary"]
    # Verify no invented diagnoses
    assert "pneumonia" not in summary_text.lower()
    assert "flu" not in summary_text.lower()
    assert "covid" not in summary_text.lower()
    assert feedback["xray_findings"]["description"] != "X-ray is normal."
    assert feedback["triage_level"]["level"] == "CONSULTATION"
    assert len(feedback["negative_red_flags"]) >= 4
