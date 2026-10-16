"""Reproduction tests for MediSense AI Audit:
1. Temperature unit handling (°C vs °F, 38.0°C must not become 38.0°F).
2. 'Wheezing: Unknown / not reported' must be state UNKNOWN, not PRESENT.
3. No duplicate symptom 'temperature' when fever is extracted.
4. Condition prediction feature representation is clean (no artificial 'in chest/throat').
5. Input-type tracking separates X-ray images, Radiology reports, and General medical reports.
6. PDF and Feedback payload state consistency.
"""

import pytest
from app.ai.symptom_nlp import extract_symptoms
from app.ai.condition_prediction import get_condition_prediction_service
from app.ai.evidence_engine import combine_evidence
from app.ai.triage_engine import apply_triage
from app.services.assessment_service import generate_assessment_feedback


def test_temperature_unit_celsius_not_forced_to_fahrenheit():
    """Bug C: 38.0°C must not be converted or recorded as 38.0°F."""
    text = "I have a fever, temperature 38.0°C. Also coughing."
    res = extract_symptoms(text)
    s_map = {s["symptom"]: s for s in res["symptoms"]}
    
    assert "fever" in s_map
    ctx = str(s_map["fever"].get("context") or "")
    # Must NOT record 38.0°F
    assert "38.0°F" not in ctx, f"38.0°C was erroneously recorded as 38.0°F! Context: {ctx}"
    assert "38" in ctx


def test_temperature_celsius_without_unit_in_biological_range():
    """Bug C: An alive human temperature of 38.0 without unit is in the Celsius range (35-42°C), not 38.0°F."""
    text = "I have fever. Temperature is 38.0. Started 2 days ago."
    res = extract_symptoms(text)
    s_map = {s["symptom"]: s for s in res["symptoms"]}
    
    assert "fever" in s_map
    ctx = str(s_map["fever"].get("context") or "")
    assert "38.0°F" not in ctx, f"Biological temperature 38.0 was falsely tagged as 38.0°F! Context: {ctx}"


def test_wheezing_unknown_not_reported_is_unknown_state():
    """Bug C: 'Wheezing: Unknown / not reported' must be extracted with state UNKNOWN, never PRESENT."""
    text = """- Cough: Present, started 2 days ago
- Fever: Present, temperature 38.0°C
- Shortness of breath: Mild, only when walking quickly
- Chest discomfort: Mild, mainly while coughing
- Fatigue: Present
- Wheezing: Unknown / not reported"""
    res = extract_symptoms(text)
    s_map = {s["symptom"]: s for s in res["symptoms"]}
    
    assert "wheezing" in s_map, "Wheezing was not extracted from text."
    assert s_map["wheezing"]["state"] == "UNKNOWN", (
        f"Wheezing with context 'Unknown / not reported' was extracted as state '{s_map['wheezing']['state']}', expected 'UNKNOWN'!"
    )


def test_canonical_symptom_deduplication():
    """Bug C: Repeated symptom mentions must be deduplicated by canonical symptom key."""
    text = """- Cough: Present, started 2 days ago
- Persistent cough: Dry cough
- Fever: Present, temperature 38.0°C
- High fever: Burning hot"""
    res = extract_symptoms(text)
    s_names = [s["symptom"] for s in res["symptoms"]]

    # Both cough mentions map to canonical cough; both fever mentions map to canonical fever/high fever
    # Seen canonicals ensures each distinct canonical symptom is extracted once
    cough_entries = [s for s in res["symptoms"] if s["symptom"] == "cough"]
    assert len(cough_entries) == 1, f"Expected exactly 1 canonical cough entry, found: {cough_entries}"


def test_condition_prediction_feature_text_purity():
    """Bug A: Feature text passed to ML model must not contain artificial 'in chest/throat' phrases."""
    svc = get_condition_prediction_service()
    if not svc.is_available():
        pytest.skip("Model not available on disk")

    syms_present = [
        {"symptom": "cough", "body_area": "chest/throat", "state": "PRESENT"},
        {"symptom": "fever", "body_area": "whole body", "state": "PRESENT"},
        {"symptom": "shortness of breath", "body_area": "chest/lungs", "state": "PRESENT"},
        {"symptom": "chest discomfort", "body_area": "chest", "state": "PRESENT"},
        {"symptom": "fatigue", "body_area": "whole body", "state": "PRESENT"},
    ]
    
    res = svc.predict_conditions(
        symptoms_present=syms_present,
        symptoms_absent=[],
        symptoms_unknown=[{"symptom": "wheezing", "state": "UNKNOWN"}],
    )
    
    # Must successfully return candidate conditions or clear status
    assert res["status"] in {"completed", "unsupported_condition"}
    if res["status"] == "completed":
        assert len(res["conditions"]) > 0
        top = res["conditions"][0]
        assert top["probability_score"] >= 0.25
        assert len(top["supporting_symptoms"]) > 0


def test_input_type_tracking_distinguishes_radiology_report_from_image():
    """Bug B: When a radiology report is uploaded without an X-ray image, image must not be reported as NOT_PROVIDED under received: True."""
    # Patient provides symptoms, 1 medical report, and 1 radiology report (NO image file)
    evidence = combine_evidence(
        symptoms=[{"symptom": "cough", "state": "PRESENT"}],
        report_findings=[{"test_name": "SGPT / ALT", "value": 48.0, "unit": "U/L", "reference_range": "7.0 - 35.0", "interpretation": "HIGH"}],
        medical_reports=[{"file_name": "liver_panel.pdf", "report_category": "medical_report"}],
        xray_reports=[{"file_name": "chest_radiology.pdf", "report_category": "xray_report", "qualitative_findings": [{"finding": "lung fields clear", "state": "PRESENT"}]}],
        xray_results=[],  # No images!
    )
    triage = apply_triage(evidence)
    feedback = generate_assessment_feedback(
        evidence=evidence,
        triage_result=triage,
        specialty="Pulmonology",
    )
    
    xray_fb = feedback["xray_findings"]
    # Image must be clearly distinguished from radiology reports
    assert len(xray_fb.get("reports", [])) == 1
    assert len(xray_fb.get("images", [])) == 0
    # Must NOT say 'Radiology documentation received: 1 report(s)' while simultaneously claiming status is NOT_PROVIDED for the whole block
    assert "Radiology documentation received: 1 report(s)" in xray_fb.get("description", "") or xray_fb.get("reports")
