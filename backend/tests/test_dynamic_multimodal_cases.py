"""Tests for Dynamic Multimodal Assessment Pipeline in MediSense AI.

Verifies:
TEST CASE A — RESPIRATORY
TEST CASE B — WRIST INJURY
TEST CASE C — HEADACHE
TEST CASE D — REPORT ONLY NARRATIVE
TEST CASE E — X-RAY REGION ROUTING AND PRESERVATION
"""

from __future__ import annotations

import pytest

from app.ai.evidence_engine import combine_evidence
from app.ai.followup_engine import suggest_followups
from app.ai.report_nlp import structure_report_text
from app.ai.specialty_mapper import map_evidence_to_specialty
from app.ai.symptom_nlp import extract_symptoms
from app.ai.triage_engine import apply_triage
from app.ai.xray_model import XRayModelService
from app.ai.xray_router import route_xray
from app.services.assessment_service import generate_assessment_feedback


def test_case_a_respiratory():
    """TEST CASE A — RESPIRATORY Presentation."""
    text = (
        "I have a fever, dry cough, and sore throat for 3 days. "
        "I have a mild headache and tiredness. "
        "I experience mild chest discomfort when coughing. "
        "I do not have severe breathing difficulty or shortness of breath. "
        "I have no chest pain at rest. I have no vomiting or fainting."
    )
    symptom_res = extract_symptoms(text)
    assert symptom_res["status"] == "extracted"
    symptoms = symptom_res["symptoms"]

    by_name = {s["symptom"]: s for s in symptoms}

    # Positive findings preserved
    assert "cough" in by_name and by_name["cough"]["state"] == "PRESENT"
    assert "fever" in by_name and by_name["fever"]["state"] == "PRESENT"
    assert "sore throat" in by_name and by_name["sore throat"]["state"] == "PRESENT"
    assert "chest discomfort" in by_name and by_name["chest discomfort"]["state"] == "PRESENT"
    assert by_name["chest discomfort"].get("trigger") == "coughing"

    # Explicit negatives preserved
    assert "chest pain at rest" in by_name or "chest pain" in by_name
    if "chest pain at rest" in by_name:
        assert by_name["chest pain at rest"]["state"] == "ABSENT"
    assert "severe breathing difficulty" in by_name or "difficulty breathing" in by_name
    for k in ["severe breathing difficulty", "difficulty breathing", "shortness of breath"]:
        if k in by_name:
            assert by_name[k]["state"] == "ABSENT"
    assert "vomiting" in by_name and by_name["vomiting"]["state"] == "ABSENT"
    assert "fainting" in by_name and by_name["fainting"]["state"] == "ABSENT"

    # No unsupported shortness of breath PRESENT
    for k in ["severe breathing difficulty", "difficulty breathing", "shortness of breath"]:
        if k in by_name:
            assert by_name[k]["state"] != "PRESENT"

    evidence = combine_evidence(symptoms=symptoms)
    triage = apply_triage(evidence)
    specialty = map_evidence_to_specialty(evidence)

    assert triage["pathway"] in {"MILD", "CONSULTATION"}
    # Triage does not trigger E1/E2 since chest pain at rest and dyspnea are absent
    assert "rule_e1_chest_pain_present" not in triage.get("rules_triggered", [])
    assert "rule_e2_dyspnea_with_chest_pain" not in triage.get("rules_triggered", [])

    assert specialty["suggested_specialty"] in {"Pulmonology", "Internal Medicine", "General Physician"}


def test_case_b_wrist_injury():
    """TEST CASE B — WRIST INJURY Presentation."""
    text = (
        "I fell onto my right hand 2 days ago. "
        "Since then, I have had significant pain in my right wrist, especially when moving or gripping objects. "
        "My wrist is swollen and tender. I have reduced movement because of the pain. "
        "I do not have numbness or tingling in my fingers. "
        "I have no fever, dizziness, fainting, chest pain, or breathing difficulty."
    )
    symptom_res = extract_symptoms(text)
    symptoms = symptom_res["symptoms"]
    by_name = {s["symptom"]: s for s in symptoms}

    # Positive findings
    assert "fall/trauma" in by_name and by_name["fall/trauma"]["state"] == "PRESENT"
    assert "wrist pain" in by_name and by_name["wrist pain"]["state"] == "PRESENT"
    assert by_name["wrist pain"].get("laterality") == "RIGHT"
    assert by_name["wrist pain"].get("body_area") == "wrist"
    assert by_name["wrist pain"].get("severity") in {"MODERATE", "SEVERE", "moderate", "severe"}

    assert "swelling" in by_name or "wrist swelling" in by_name
    assert "tenderness" in by_name or "wrist tenderness" in by_name
    assert "reduced movement" in by_name and by_name["reduced movement"]["state"] == "PRESENT"

    # Negatives
    assert "numbness" in by_name and by_name["numbness"]["state"] == "ABSENT"
    assert "fever" in by_name and by_name["fever"]["state"] == "ABSENT"
    assert "chest pain" in by_name and by_name["chest pain"]["state"] == "ABSENT"
    assert "difficulty breathing" in by_name and by_name["difficulty breathing"]["state"] == "ABSENT"

    # NO fracture diagnosis from symptoms alone
    assert "fracture" not in by_name

    # X-ray wrist routing
    routing = route_xray("dummy.jpg", declared_region="wrist")
    assert routing["region"] == "wrist"
    assert routing["supported"] is False

    model_service = XRayModelService()
    xray_result = model_service.analyze("dummy.jpg", region="wrist")
    assert xray_result["region"] == "wrist"
    assert xray_result["status"] == "unavailable"
    assert xray_result["prediction"] is None
    assert "chest" not in xray_result["region"].lower()

    # Evidence combination
    evidence = combine_evidence(symptoms=symptoms, xray_result=xray_result)
    assert evidence["xray"]["region"] == "wrist"
    assert evidence["xray"]["status"] == "UNAVAILABLE"
    assert evidence["xray"]["prediction"] is None

    # Verify no "Chest X-ray" in supporting statements for wrist
    for supp in evidence.get("supporting", []):
        assert "Chest X-ray" not in supp

    specialty = map_evidence_to_specialty(evidence)
    assert specialty["suggested_specialty"] == "Orthopedics"

    # Feedback generation
    triage = apply_triage(evidence)
    feedback = generate_assessment_feedback(evidence, triage, specialty["suggested_specialty"])
    assert "Chest X-ray" not in feedback["assessment_summary"]
    assert feedback["xray_findings"]["region"] == "Wrist"
    assert "fracture" not in feedback["assessment_summary"].lower()


def test_case_c_headache():
    """TEST CASE C — HEADACHE Presentation."""
    text = (
        "I have a severe right-sided throbbing headache for 2 days. "
        "I am sensitive to bright light (photophobia) and loud sounds (phonophobia). "
        "I feel nausea. "
        "I do not have vomiting, fever, weakness, numbness, speech difficulty, fainting, or seizures."
    )
    symptom_res = extract_symptoms(text)
    symptoms = symptom_res["symptoms"]
    by_name = {s["symptom"]: s for s in symptoms}

    assert "headache" in by_name and by_name["headache"]["state"] == "PRESENT"
    assert by_name["headache"].get("laterality") == "RIGHT"
    assert by_name["headache"].get("quality") == "THROBBING"
    assert by_name["headache"].get("severity") in {"SEVERE", "severe"}

    assert "photophobia" in by_name and by_name["photophobia"]["state"] == "PRESENT"
    assert "phonophobia" in by_name and by_name["phonophobia"]["state"] == "PRESENT"
    assert "nausea" in by_name and by_name["nausea"]["state"] == "PRESENT"

    assert "vomiting" in by_name and by_name["vomiting"]["state"] == "ABSENT"
    assert "fever" in by_name and by_name["fever"]["state"] == "ABSENT"
    assert "numbness" in by_name and by_name["numbness"]["state"] == "ABSENT"
    assert "speech difficulty" in by_name and by_name["speech difficulty"]["state"] == "ABSENT"
    assert "fainting" in by_name and by_name["fainting"]["state"] == "ABSENT"
    assert "seizure" in by_name and by_name["seizure"]["state"] == "ABSENT"

    # No respiratory symptoms fabricated
    assert "cough" not in by_name
    assert "chest pain" not in by_name

    evidence = combine_evidence(symptoms=symptoms)
    specialty = map_evidence_to_specialty(evidence)
    assert specialty["suggested_specialty"] in {"Neurology", "General Physician"}


def test_case_d_report_only_narrative():
    """TEST CASE D — REPORT ONLY NARRATIVE (No numeric laboratory table)."""
    report_text = """
    CLINICAL REPORT
    Patient: John Doe

    Clinical History:
    Fall onto outstretched right hand 2 days ago during sports activity.

    Physical Examination:
    Localized tenderness and swelling over right wrist. Reduced range of motion due to pain.
    No neurovascular deficit noted in digits.

    Impression:
    Acute right wrist injury following fall. No fracture identified on plain radiography.
    """

    nlp_res = structure_report_text(report_text)
    assert nlp_res["status"] == "structured"
    assert len(nlp_res["findings"]) == 0  # No numeric lab table
    assert len(nlp_res["narrative_findings"]) >= 2
    assert len(nlp_res["qualitative_findings"]) >= 1

    # Explicit fracture check
    fracture_finding = next((q for q in nlp_res["qualitative_findings"] if q["finding"] == "fracture"), None)
    assert fracture_finding is not None
    assert fracture_finding["state"] == "ABSENT"

    evidence = combine_evidence(
        report_provided=True,
        qualitative_report_findings=nlp_res["qualitative_findings"],
        narrative_report_findings=nlp_res["narrative_findings"],
    )

    assert evidence["report"]["provided"] is True
    assert len(evidence["report"]["narrative_findings"]) >= 2
    assert len(evidence["report"]["qualitative_findings"]) >= 1

    triage = apply_triage(evidence)
    feedback = generate_assessment_feedback(evidence, triage, "Orthopedics")

    # MUST NOT say "Medical report not provided"
    assert "Medical laboratory report was not provided" not in feedback["assessment_summary"]
    assert "Medical report provided" in feedback["assessment_summary"]
    assert feedback["medical_report_status"] == "PROVIDED"
    assert "Medical Report: Not provided" not in feedback["medical_report_message"]


def test_case_e_xray_region_routing():
    """TEST CASE E — X-RAY REGION ROUTING AND PRESERVATION."""
    # 1. Wrist
    wrist_route = route_xray("dummy.jpg", declared_region="wrist")
    assert wrist_route["region"] == "wrist"
    assert wrist_route["supported"] is False

    service = XRayModelService()
    wrist_ana = service.analyze("dummy.jpg", region="wrist")
    assert wrist_ana["region"] == "wrist"
    assert wrist_ana["status"] == "unavailable"
    assert wrist_ana["prediction"] is None
    assert "chest" not in wrist_ana["region"].lower()

    # 2. Chest
    chest_route = route_xray("dummy.jpg", declared_region="chest")
    assert chest_route["region"] == "chest"
    assert chest_route["supported"] is True

    # 3. Unknown / Undeclared
    unknown_route = route_xray("dummy.jpg", declared_region=None)
    assert unknown_route["region"] == "unknown"
    assert unknown_route["supported"] is False

    unknown_ana = service.analyze("dummy.jpg", region=None)
    assert unknown_ana["region"] == "unknown"
    assert unknown_ana["region"] != "chest"
    assert unknown_ana["status"] == "unavailable"

    # Evidence engine with unknown
    evidence_unknown = combine_evidence(xray_result=unknown_ana)
    assert evidence_unknown["xray"]["region"] in {"unknown", "not declared"}
    assert evidence_unknown["xray"]["region"] != "chest"


def test_followup_domain_awareness():
    """Verify follow-up engine selects domain-aware questions and skips addressed ones."""
    # Wrist text with all safety checks addressed in user text
    wrist_full_text = (
        "I fell onto my right hand 2 days ago. Since then, I have had significant pain in my right wrist, "
        "especially when moving or gripping objects. My wrist is swollen and tender. I have reduced movement "
        "because of the pain. I do not have numbness or tingling in my fingers. I can move my fingers normally. "
        "I have no fever, dizziness, fainting, chest pain, or breathing difficulty."
    )
    symptoms = extract_symptoms(wrist_full_text)["symptoms"]
    followups = suggest_followups(symptoms=symptoms)
    # Since duration, numbness, movement, chest pain, breathing difficulty, and fever are all addressed:
    assert followups["needed"] is False
    assert len(followups["questions"]) == 0

    # Wrist minimal text: "I fell and my wrist hurts" (missing duration, numbness, movement)
    minimal_wrist = extract_symptoms("I fell and my wrist hurts")["symptoms"]
    wrist_followups = suggest_followups(symptoms=minimal_wrist)
    assert wrist_followups["needed"] is True
    q_targets = [q["target"] for q in wrist_followups["questions"]]
    # Must NOT ask chest pain or breathing difficulty for a wrist injury case
    assert "chest pain" not in q_targets
    assert "difficulty breathing" not in q_targets
    # Should ask numbness/movement/duration
    assert any(t in q_targets for t in ["numbness", "reduced movement", "duration"])
