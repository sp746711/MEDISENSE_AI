"""Comprehensive evaluation suite for MediSense AI assessment feedback pipeline.

Verifies:
1. Positive symptom
2. Explicit negative
3. Unknown finding
4. Negation precision without leakage
5. Duration preservation
6. Severity preservation
7. Laterality preservation
8. Trigger preservation
9. Associated symptom
10. Previous occurrence
11. Normal report value
12. High report value
13. Low report value
14. Missing reference range status = UNKNOWN
15. X-ray available (actual model result)
16. X-ray unavailable (truthful status, never fabricated)
17. Unsupported X-ray region
18. No X-ray provided
19. Evidence merge
20. Contradiction detection
21. Unknown preservation
22. Triage rule triggering
23. Dynamic feedback generation
24. Assessment isolation (Tests A through H)
"""

from __future__ import annotations

import os
from uuid import uuid4
import pytest

from app.ai.evidence_engine import combine_evidence
from app.ai.followup_engine import suggest_followups
from app.ai.report_nlp import structure_report_text
from app.ai.specialty_mapper import map_evidence_to_specialty
from app.ai.symptom_nlp import extract_symptoms
from app.ai.triage_engine import apply_triage
from app.ai.xray_model import XRayModelService
from app.ai.xray_router import route_xray
from app.database.database import SessionLocal, init_db
from app.database.models import Assessment, MedicalReport, Symptom, User, XrayResult
from app.services.assessment_service import generate_assessment_feedback, run_assessment_pipeline
from app.services.pdf_service import generate_assessment_pdf


@pytest.fixture(scope="module")
def db_session():
    init_db()
    session = SessionLocal()
    yield session
    session.close()


# ─────────────────────────────────────────────────────────────
# 1-10: Symptom NLP Requirements
# ─────────────────────────────────────────────────────────────

def test_01_positive_symptom():
    res = extract_symptoms("I have a cough.")
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    assert "cough" in symptoms
    assert symptoms["cough"]["status"] == "PRESENT"
    assert symptoms["cough"]["state"] == "PRESENT"


def test_02_explicit_negative():
    res = extract_symptoms("I do not have fever.")
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    assert "fever" in symptoms
    assert symptoms["fever"]["status"] == "ABSENT"


def test_03_unknown_finding():
    res = extract_symptoms("I have a cough.")
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    # Unmentioned symptom must not be created as PRESENT or ABSENT
    assert "fever" not in symptoms
    assert "shortness of breath" not in symptoms


def test_04_negation_local_and_precise():
    text = "I have mild chest discomfort when coughing. I do not have chest pain when resting."
    res = extract_symptoms(text)
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    assert symptoms["chest discomfort"]["status"] == "PRESENT"
    assert symptoms["chest discomfort"]["severity"].lower() == "mild"
    assert symptoms["chest discomfort"]["trigger"] == "coughing"
    assert symptoms["chest pain at rest"]["status"] == "ABSENT"


def test_05_duration():
    res = extract_symptoms("I have had severe headache for 2 days.")
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    assert "2 days" in symptoms["headache"]["duration"]


def test_06_severity():
    res = extract_symptoms("I have severe headache. I also have mild difficulty sleeping.")
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    assert symptoms["headache"]["severity"].lower() == "severe"
    assert symptoms["sleep difficulty"]["severity"].lower() == "mild"


def test_07_laterality():
    res = extract_symptoms("I have had a severe headache, mainly on the right side of my head.")
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    assert symptoms["headache"]["laterality"] == "RIGHT"


def test_08_trigger():
    res = extract_symptoms("I feel mild chest discomfort when coughing.")
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    assert symptoms["chest discomfort"]["trigger"] == "coughing"


def test_09_associated_symptoms():
    res = extract_symptoms("I feel nauseous and have difficulty concentrating.")
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    assert symptoms["nausea"]["status"] == "PRESENT"
    assert symptoms["concentration difficulty"]["status"] == "PRESENT"


def test_10_previous_occurrence():
    res = extract_symptoms("I have had similar headaches once or twice before.")
    symptoms = {s["symptom"]: s for s in res["symptoms"]}
    assert "previous similar headache" in symptoms
    assert symptoms["previous similar headache"]["status"] == "PRESENT"


# ─────────────────────────────────────────────────────────────
# 11-14: Medical Report Requirements
# ─────────────────────────────────────────────────────────────

def test_11_normal_report_value():
    report_text = "Hemoglobin: 14.5 g/dL (Reference: 13.0 - 17.0)"
    nlp = structure_report_text(report_text)
    f = nlp["findings"][0]
    assert f["test_name"] == "Hemoglobin"
    assert f["value"] == 14.5
    assert f["status"] == "NORMAL"


def test_12_high_report_value():
    report_text = "Fasting Blood Glucose: 165 mg/dL (Reference: 70 - 99)"
    nlp = structure_report_text(report_text)
    f = nlp["findings"][0]
    assert f["test_name"] == "Fasting Blood Glucose"
    assert f["value"] == 165.0
    assert f["status"] == "HIGH"


def test_13_low_report_value():
    report_text = "Hemoglobin: 9.2 g/dL (Reference: 13.0 - 17.0)"
    nlp = structure_report_text(report_text)
    f = nlp["findings"][0]
    assert f["test_name"] == "Hemoglobin"
    assert f["value"] == 9.2
    assert f["status"] == "LOW"


def test_14_missing_reference_range():
    report_text = "Serum Creatinine: 1.1 mg/dL"
    nlp = structure_report_text(report_text)
    f = nlp["findings"][0]
    assert f["test_name"] == "Serum Creatinine"
    assert f["value"] == 1.1
    assert f["status"] == "UNKNOWN"
    assert f["reference_range"] is None


# ─────────────────────────────────────────────────────────────
# 15-18: X-Ray Requirements
# ─────────────────────────────────────────────────────────────

def test_15_xray_available():
    # If checkpoint/model exists and predicts valid class
    mock_xray = {
        "status": "completed",
        "region": "chest",
        "prediction": "Consolidation / Opacity",
        "model_version": "chest_resnet18@model.pt",
    }
    evidence = combine_evidence(xray_result=mock_xray)
    assert evidence["xray"]["status"] == "COMPLETED"
    assert evidence["xray"]["prediction"] == "Consolidation / Opacity"
    assert evidence["xray"]["evidence_status"] == "SUPPORTING"


def test_16_xray_unavailable():
    service = XRayModelService()
    analysis = service.analyze("nonexistent_path.jpg", region="chest")
    assert analysis["status"] == "unavailable"
    assert analysis["prediction"] is None
    assert "unavailable" in analysis["message"].lower()

    evidence = combine_evidence(xray_result=analysis)
    assert evidence["xray"]["status"] == "UNAVAILABLE"
    assert evidence["xray"]["prediction"] is None
    assert evidence["xray"]["evidence_status"] == "UNKNOWN"


def test_17_unsupported_xray():
    routed = route_xray("sample.png", declared_region="knee")
    assert routed["supported"] is False
    assert "unavailable" in routed["message"].lower()


def test_18_no_xray():
    evidence = combine_evidence(symptoms=[], report_findings=[], xray_result=None)
    assert evidence["xray"]["status"] == "NOT_PROVIDED"
    assert evidence["xray"]["received"] is False
    assert evidence["xray"]["evidence_status"] == "NOT_ASSESSED"


# ─────────────────────────────────────────────────────────────
# 19-24: Evidence, Contradiction, Triage & Dynamic Feedback
# ─────────────────────────────────────────────────────────────

def test_19_evidence_merge():
    sym_res = extract_symptoms("I have a dry cough for 3 days.")
    rep_res = structure_report_text("White Blood Cell Count (WBC): 12500 /uL (Reference: 4000 - 11000)")
    evidence = combine_evidence(
        symptoms=sym_res["symptoms"],
        report_findings=rep_res["findings"],
        xray_result=None,
    )
    assert len(evidence["symptoms"]["present"]) == 1
    assert len(evidence["report"]["abnormal_findings"]) == 1
    assert evidence["report"]["provided"] is True
    assert evidence["xray"]["received"] is False


def test_20_contradiction():
    sym_res = extract_symptoms("I do not have a fever. My temperature is 103°F.")
    evidence = combine_evidence(symptoms=sym_res["symptoms"])
    assert len(evidence["contradictory"]) > 0
    assert "Contradiction" in evidence["contradictory"][0]


def test_21_unknown_preservation():
    sym_res = extract_symptoms("I have a headache.")
    evidence = combine_evidence(symptoms=sym_res["symptoms"])
    # Duration was not mentioned -> recorded in missing
    assert any("duration" in m.lower() for m in evidence["missing"])
    # Critical negatives neither confirmed nor denied
    assert any("chest pain" in m.lower() for m in evidence["missing"])


def test_22_triage_rule_triggering():
    # Emergency rule test: extreme glucose
    rep_high_glucose = structure_report_text("Fasting Blood Sugar: 390 mg/dL (Reference: 70 - 100)")
    ev = combine_evidence(report_findings=rep_high_glucose["findings"])
    triage = apply_triage(ev)
    assert triage["pathway"] == "EMERGENCY"
    assert "rule_e4_critical_glycemic_emergency" in triage["rules_triggered"]

    # Mild rule test: single mild symptom, normal labs
    sym_mild = extract_symptoms("I feel mild fatigue.")
    rep_norm = structure_report_text("Hemoglobin: 14.0 g/dL (Reference: 13.0 - 17.0)")
    ev_mild = combine_evidence(symptoms=sym_mild["symptoms"], report_findings=rep_norm["findings"])
    triage_mild = apply_triage(ev_mild)
    assert triage_mild["pathway"] == "MILD"


def test_23_dynamic_feedback():
    # Assessment 1: Headache dominant
    sym1 = extract_symptoms("I have a severe headache with bright light sensitivity for 2 days.")
    ev1 = combine_evidence(symptoms=sym1["symptoms"])
    triage1 = apply_triage(ev1)
    spec1 = map_evidence_to_specialty(ev1)
    fb1 = generate_assessment_feedback(ev1, triage1, spec1["suggested_specialty"])

    # Assessment 2: Abdominal pain dominant
    sym2 = extract_symptoms("I have stomach pain.")
    ev2 = combine_evidence(symptoms=sym2["symptoms"])
    triage2 = apply_triage(ev2)
    spec2 = map_evidence_to_specialty(ev2)
    fb2 = generate_assessment_feedback(ev2, triage2, spec2["suggested_specialty"])

    assert "headache" in fb1["assessment_summary"].lower()
    assert "stomach" not in fb1["assessment_summary"].lower()
    assert "abdominal" not in fb1["assessment_summary"].lower()

    assert "abdominal" in fb2["assessment_summary"].lower() or "stomach" in fb2["assessment_summary"].lower()
    assert "headache" not in fb2["assessment_summary"].lower()

    assert spec1["suggested_specialty"] == "Neurology"
    assert spec2["suggested_specialty"] == "Gastroenterology"
    assert fb1["assessment_summary"] != fb2["assessment_summary"]


# ─────────────────────────────────────────────────────────────
# 25: Assessment Isolation Tests (A through H)
# ─────────────────────────────────────────────────────────────

def test_25_dynamic_assessment_isolation_suite(db_session):
    user = User(
        name="Isolation Test User",
        email=f"iso_{uuid4().hex[:8]}@example.com",
        password_hash="pwdhash",
        state="Maharashtra",
        district="Mumbai",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    results = {}

    # Case A: headache + photophobia
    ass_a = Assessment(user_id=user.user_id, input_types=["symptoms"], status="draft")
    db_session.add(ass_a)
    db_session.commit()
    sym_a = extract_symptoms("I have a severe headache and sensitivity to bright light for 2 days.")
    for s in sym_a["symptoms"]:
        db_session.add(Symptom(assessment_id=ass_a.assessment_id, symptom=s["symptom"], state=s["state"], duration=s["duration"], severity=s["severity"], source="user_input"))
    db_session.commit()
    res_a = run_assessment_pipeline(db_session, ass_a.assessment_id, user.user_id)
    results["A"] = res_a

    # Case B: fever + cough
    ass_b = Assessment(user_id=user.user_id, input_types=["symptoms"], status="draft")
    db_session.add(ass_b)
    db_session.commit()
    sym_b = extract_symptoms("I have had fever and dry cough for 4 days.")
    for s in sym_b["symptoms"]:
        db_session.add(Symptom(assessment_id=ass_b.assessment_id, symptom=s["symptom"], state=s["state"], duration=s["duration"], severity=s["severity"], source="user_input"))
    db_session.commit()
    res_b = run_assessment_pipeline(db_session, ass_b.assessment_id, user.user_id)
    results["B"] = res_b

    # Case C: abdominal pain
    ass_c = Assessment(user_id=user.user_id, input_types=["symptoms"], status="draft")
    db_session.add(ass_c)
    db_session.commit()
    sym_c = extract_symptoms("I have abdominal pain.")
    for s in sym_c["symptoms"]:
        db_session.add(Symptom(assessment_id=ass_c.assessment_id, symptom=s["symptom"], state=s["state"], source="user_input"))
    db_session.commit()
    res_c = run_assessment_pipeline(db_session, ass_c.assessment_id, user.user_id)
    results["C"] = res_c

    # Case D: medical report only
    ass_d = Assessment(user_id=user.user_id, input_types=["medical_report"], status="draft")
    db_session.add(ass_d)
    db_session.commit()
    rep_d = structure_report_text("SGPT / ALT: 145 U/L (Reference: 7 - 56)\nSGOT / AST: 120 U/L (Reference: 10 - 40)")
    db_session.add(MedicalReport(
        assessment_id=ass_d.assessment_id,
        file_name="liver_panel.pdf",
        stored_name="liver_panel.pdf",
        file_type="application/pdf",
        extracted_text="liver report",
        structured_findings={"lab_parameters": rep_d["findings"]},
    ))
    db_session.commit()
    res_d = run_assessment_pipeline(db_session, ass_d.assessment_id, user.user_id)
    results["D"] = res_d

    # Case E: X-ray only
    ass_e = Assessment(user_id=user.user_id, input_types=["xray"], status="draft")
    db_session.add(ass_e)
    db_session.commit()
    db_session.add(XrayResult(
        assessment_id=ass_e.assessment_id,
        region="chest",
        status="completed",
        prediction="Pneumothorax",
        model_version="chest_resnet18@model.pt",
    ))
    db_session.commit()
    res_e = run_assessment_pipeline(db_session, ass_e.assessment_id, user.user_id)
    results["E"] = res_e

    # Case F: symptoms + medical report
    ass_f = Assessment(user_id=user.user_id, input_types=["symptoms", "medical_report"], status="draft")
    db_session.add(ass_f)
    db_session.commit()
    sym_f = extract_symptoms("I have joint pain.")
    for s in sym_f["symptoms"]:
        db_session.add(Symptom(assessment_id=ass_f.assessment_id, symptom=s["symptom"], state=s["state"], source="user_input"))
    rep_f = structure_report_text("Uric Acid: 9.8 mg/dL (Reference: 3.5 - 7.2)")
    db_session.add(MedicalReport(
        assessment_id=ass_f.assessment_id,
        file_name="uric.pdf",
        stored_name="uric.pdf",
        file_type="application/pdf",
        extracted_text="uric acid report",
        structured_findings={"lab_parameters": rep_f["findings"]},
    ))
    db_session.commit()
    res_f = run_assessment_pipeline(db_session, ass_f.assessment_id, user.user_id)
    results["F"] = res_f

    # Case G: symptoms + X-ray
    ass_g = Assessment(user_id=user.user_id, input_types=["symptoms", "xray"], status="draft")
    db_session.add(ass_g)
    db_session.commit()
    sym_g = extract_symptoms("I have a dry cough.")
    for s in sym_g["symptoms"]:
        db_session.add(Symptom(assessment_id=ass_g.assessment_id, symptom=s["symptom"], state=s["state"], source="user_input"))
    db_session.add(XrayResult(
        assessment_id=ass_g.assessment_id,
        region="chest",
        status="unavailable",
        prediction=None,
        message="Model unavailable.",
    ))
    db_session.commit()
    res_g = run_assessment_pipeline(db_session, ass_g.assessment_id, user.user_id)
    results["G"] = res_g

    # Case H: symptoms + medical report + X-ray
    ass_h = Assessment(user_id=user.user_id, input_types=["symptoms", "medical_report", "xray"], status="draft")
    db_session.add(ass_h)
    db_session.commit()
    sym_h = extract_symptoms("I have severe shortness of breath.")
    for s in sym_h["symptoms"]:
        db_session.add(Symptom(assessment_id=ass_h.assessment_id, symptom=s["symptom"], state=s["state"], severity="severe", source="user_input"))
    rep_h = structure_report_text("White Blood Cell Count (WBC): 18000 /uL (Reference: 4000 - 11000)")
    db_session.add(MedicalReport(
        assessment_id=ass_h.assessment_id,
        file_name="cbc.pdf",
        stored_name="cbc.pdf",
        file_type="application/pdf",
        extracted_text="cbc",
        structured_findings={"lab_parameters": rep_h["findings"]},
    ))
    db_session.add(XrayResult(
        assessment_id=ass_h.assessment_id,
        region="chest",
        status="completed",
        prediction="Consolidation / Opacity",
        model_version="chest_resnet18@model.pt",
    ))
    db_session.commit()
    res_h = run_assessment_pipeline(db_session, ass_h.assessment_id, user.user_id)
    results["H"] = res_h

    # VERIFY COMPLETE ISOLATION AND DISTINCT OUTPUTS ACROSS ALL CASES
    summaries = [results[k]["result"]["feedback"]["assessment_summary"] for k in "ABCDEFGH"]
    # Every summary must be unique
    assert len(set(summaries)) == 8

    # Verify specialty differences
    assert results["A"]["specialty"] == "Neurology"
    assert "Pulmonology" in results["B"]["specialty"] or "Internal Medicine" in results["B"]["specialty"]
    assert results["C"]["specialty"] == "Gastroenterology"
    assert "Gastroenterology" in results["D"]["specialty"] or "Hepatology" in results["D"]["specialty"]
    assert "Pulmonology" in results["E"]["specialty"] or "Cardiology" in results["E"]["specialty"]

    # Verify no cross-talk: Case C must not contain headache or cough
    fb_c = results["C"]["result"]["feedback"]["assessment_summary"].lower()
    assert "headache" not in fb_c
    assert "cough" not in fb_c

    # Verify PDF generation for each isolated case
    for k in "ABCDEFGH":
        ass_obj = db_session.get(Assessment, results[k]["result"]["evidence"].get("assessment_id") or results[k]["assessment_id"])
        pdf_res = generate_assessment_pdf(ass_obj.assessment_id, db_session)
        assert pdf_res["status"] == "ok"
        assert os.path.exists(pdf_res["path"])
