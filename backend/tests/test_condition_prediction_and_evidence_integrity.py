
"""Comprehensive test suite for Symptom Uncertainty, Evidence Integrity,

X-Ray Diagnostics, and ML Condition Prediction (Phases 1-10).
"""

from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.ai.condition_prediction import (
    ConditionPredictionService,
    get_condition_prediction_service,
)
from app.ai.evidence_engine import combine_evidence
from app.ai.followup_engine import is_target_addressed, suggest_followups
from app.ai.report_nlp import is_synthetic_text, structure_report_text
from app.ai.symptom_nlp import extract_symptoms
from app.ai.triage_engine import apply_triage
from app.ai.xray_model import XRayModelService
from app.database.database import Base
from app.database.models import Assessment, MedicalReport, Symptom, User, XrayResult
from app.services.assessment_service import (
    generate_assessment_feedback,
    run_assessment_pipeline,
)
from app.services.pdf_service import generate_assessment_pdf


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


# ----------------------------------------------------------------------
# 1. Symptom Classification & Uncertainty (Phase 2)
# ----------------------------------------------------------------------

def test_uncertain_fever_classified_as_unknown():
    """Uncertain fever statements must be classified as UNKNOWN."""

    res1 = extract_symptoms("I am not sure whether I have a fever")
    fever_sym1 = next(
        (s for s in res1["symptoms"] if s["symptom"] == "fever"),
        None,
    )
    assert fever_sym1 is not None, "Fever symptom should be captured"
    assert fever_sym1["state"] == "UNKNOWN"

    res2 = extract_symptoms("I might have a fever")
    fever_sym2 = next(
        (s for s in res2["symptoms"] if s["symptom"] == "fever"),
        None,
    )
    assert fever_sym2 is not None
    assert fever_sym2["state"] == "UNKNOWN"

    res3 = extract_symptoms("Unsure if I have a fever")
    fever_sym3 = next(
        (s for s in res3["symptoms"] if s["symptom"] == "fever"),
        None,
    )
    assert fever_sym3 is not None
    assert fever_sym3["state"] == "UNKNOWN"


def test_explicitly_absent_vomiting_classified_as_absent():
    """Explicitly denied symptoms must be classified as ABSENT."""

    res_vomit = extract_symptoms("I have not vomited")
    vomit_sym = next(
        (s for s in res_vomit["symptoms"] if s["symptom"] == "vomiting"),
        None,
    )
    assert vomit_sym is not None
    assert vomit_sym["state"] == "ABSENT"

    res_fever = extract_symptoms("I do not have a fever")
    fever_sym = next(
        (s for s in res_fever["symptoms"] if s["symptom"] == "fever"),
        None,
    )
    assert fever_sym is not None
    assert fever_sym["state"] == "ABSENT"


def test_explicitly_present_symptoms():
    """An explicitly reported fever must be classified as PRESENT."""

    res = extract_symptoms("I have a fever")
    fever_sym = next(
        (s for s in res["symptoms"] if s["symptom"] == "fever"),
        None,
    )

    assert fever_sym is not None
    assert fever_sym["state"] == "PRESENT"


def test_uncertain_fever_not_treated_as_addressed_in_followups():
    """Uncertain symptoms must trigger clarification rather than assumption."""

    symptoms = [
        {"symptom": "fever", "state": "UNKNOWN", "status": "UNKNOWN"},
        {"symptom": "cough", "state": "PRESENT", "domain": "respiratory"},
    ]

    assert not is_target_addressed(
        "fever",
        symptoms,
        raw_text="I might have a fever",
    )

    followups = suggest_followups(
        symptoms,
        raw_text="I might have a fever",
        max_questions=5,
    )

    assert followups["needed"] is True
    assert any(
        "fever" in q["target"].lower()
        or "fever" in q["question"].lower()
        for q in followups["questions"]
    )


# ----------------------------------------------------------------------
# 2. Evidence Integrity & Medical Report Processing (Phase 3)
# ----------------------------------------------------------------------

def test_missing_crp_never_appears_in_findings_or_reassuring_evidence():
    """CRP must not be fabricated when the report contains only WBC."""

    report_findings = [
        {
            "test_name": "White Blood Cell Count (WBC)",
            "value": 7500.0,
            "unit": "/uL",
            "reference_range": "4000 - 11000",
            "interpretation": "NORMAL",
            "status": "NORMAL",
            "source_file": "cbc_report.pdf",
        }
    ]

    symptoms = [
        {"symptom": "cough", "state": "PRESENT", "domain": "respiratory"}
    ]

    evidence = combine_evidence(
        symptoms=symptoms,
        report_findings=report_findings,
        report_provided=True,
    )

    reassuring = " ".join(evidence.get("reassuring", []))
    supporting = " ".join(evidence.get("supporting", []))

    assert "WBC" in reassuring
    assert "CRP" not in reassuring
    assert "CRP" not in supporting


def test_synthetic_report_instructions_excluded_from_clinical_evidence():
    """Synthetic report instructions must not become clinical findings."""

    demo_text = (
        "Instructions: This is a synthetic demonstration report. "
        "Ensure no actual clinical findings are generated. "
        "Sample test case only. "
        "Impression: Example report with demo placeholder text."
    )

    assert is_synthetic_text(demo_text) is True

    parsed = structure_report_text(demo_text)

    assert parsed.get("is_synthetic") is True
    assert not any(
        q["finding"] == "fracture" and q["state"] == "PRESENT"
        for q in parsed["qualitative_findings"]
    )

    evidence = combine_evidence(
        symptoms=[{"symptom": "wrist pain", "state": "PRESENT"}],
        narrative_report_findings=parsed["narrative_findings"],
        report_provided=True,
    )

    for supporting_item in evidence.get("supporting", []):
        assert "no actual clinical findings" not in supporting_item.lower()
        assert "synthetic demonstration" not in supporting_item.lower()


# ----------------------------------------------------------------------
# 3. Assessment Isolation & Cross-Modal Evidence (Phase 4)
# ----------------------------------------------------------------------

def test_unrelated_xray_excluded_from_respiratory_condition_evidence():
    """A wrist X-ray must not support a respiratory condition prediction."""

    symptoms = [
        {"symptom": "cough", "state": "PRESENT", "domain": "respiratory"}
    ]

    xray_results = [
        {
            "id": str(uuid4()),
            "region": "wrist",
            "status": "completed",
            "prediction": "Normal / Clear",
            "file_name": "wrist.png",
        }
    ]

    evidence = combine_evidence(
        symptoms=symptoms,
        xray_results=xray_results,
    )

    assert not any(
        "wrist" in item.lower()
        for item in evidence.get("supporting", [])
    )

    assert any(
        "wrist" in item.lower()
        for item in evidence.get("separate_or_contextual", [])
    )


# ----------------------------------------------------------------------
# 4. X-Ray Diagnostics (Phase 5)
# ----------------------------------------------------------------------

def test_unsupported_xray_region_produces_honest_unavailable_result(tmp_path):
    """An unsupported anatomical region returns unsupported_region."""

    dummy_img = tmp_path / "clavicle.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

    service = XRayModelService()
    res = service.analyze(str(dummy_img), region="clavicle")

    assert res["status"] == "unavailable"
    assert res["reason_code"] == "unsupported_region"
    assert res["prediction"] is None


def test_missing_xray_checkpoint_produces_honest_unavailable_result(
    tmp_path,
    monkeypatch,
):
    """A missing chest checkpoint returns unavailable without preprocessing."""

    dummy_img = tmp_path / "chest.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

    service = XRayModelService()

    # Simulate a missing checkpoint without relying on local model files.
    # This test verifies the unavailable-result contract rather than
    # testing actual checkpoint discovery or image preprocessing.
    service._loaded = False
    service._load_failure_code = "checkpoint_path_not_found"

    monkeypatch.setattr(
        service,
        "is_available",
        lambda region=None: False,
    )

    res = service.analyze(str(dummy_img), region="chest")

    assert res["status"] == "unavailable"
    assert res["reason_code"] == "checkpoint_path_not_found"
    assert res["prediction"] is None


def test_corrupted_xray_produces_invalid_image_result(tmp_path):
    """A non-existent X-ray image must return invalid_image."""

    non_existent = str(tmp_path / "ghost.png")

    service = XRayModelService()
    res = service.analyze(non_existent, region="chest")

    assert res["status"] == "unavailable"
    assert res["reason_code"] == "invalid_image"


# ----------------------------------------------------------------------
# 5. ML Condition Prediction (Phases 7, 8, 9)
# ----------------------------------------------------------------------

def test_condition_prediction_with_active_symptoms():
    """Predict candidate conditions from active symptoms."""

    svc = get_condition_prediction_service()

    if not svc.is_available():
        pytest.skip("Condition prediction model artifact not found on disk")

    res = svc.predict_conditions(
        symptoms_present=[
            {"symptom": "cough"},
            {"symptom": "difficulty breathing"},
        ],
        symptoms_absent=[{"symptom": "vomiting"}],
        symptoms_unknown=[{"symptom": "fever"}],
    )

    assert res["status"] == "completed"
    assert len(res["conditions"]) > 0

    top_cand = res["conditions"][0]

    assert "condition" in top_cand
    assert "probability_score" in top_cand
    assert top_cand["probability_score"] > 0
    assert "confidence_tier" in top_cand
    assert "disclaimer" in top_cand


def test_model_predictions_never_override_emergency_triage():
    """Emergency triage must remain unchanged regardless of ML predictions."""

    symptoms = [
        {
            "symptom": "chest pain at rest",
            "state": "PRESENT",
            "severity": "SEVERE",
            "domain": "cardiovascular",
        }
    ]

    evidence = combine_evidence(symptoms=symptoms)
    triage_result = apply_triage(evidence)

    assert triage_result["pathway"] == "EMERGENCY"
    assert "rule_e1_chest_pain_present" in triage_result["rules_triggered"]

    svc = get_condition_prediction_service()

    preds = svc.predict_conditions(
        symptoms_present=symptoms,
        symptoms_absent=[],
        symptoms_unknown=[],
    )

    feedback = generate_assessment_feedback(
        evidence=evidence,
        triage_result=triage_result,
        specialty="Cardiology",
        condition_predictions=preds,
    )

    assert feedback["triage_level"]["pathway"] == "EMERGENCY"


# ----------------------------------------------------------------------
# 6. Full Pipeline, Database & PDF Generation Parity (Phase 6 & 10)
# ----------------------------------------------------------------------

def test_full_pipeline_and_pdf_generation(in_memory_db, tmp_path):
    """End-to-end integration: symptoms -> pipeline -> result payload -> PDF."""

    user = User(
        user_id=uuid4(),
        email="patient@test.com",
        password_hash="fake",
        name="John Doe",
        state="Maharashtra",
        district="Mumbai",
    )

    assessment = Assessment(
        assessment_id=uuid4(),
        user_id=user.user_id,
        input_types=["symptoms"],
        status="draft",
    )

    symptom1 = Symptom(
        symptom_id=uuid4(),
        assessment_id=assessment.assessment_id,
        symptom="cough",
        state="PRESENT",
        severity="MILD",
        source="user_input",
    )

    symptom2 = Symptom(
        symptom_id=uuid4(),
        assessment_id=assessment.assessment_id,
        symptom="fever",
        state="UNKNOWN",
        source="user_input",
    )

    in_memory_db.add_all([user, assessment, symptom1, symptom2])
    in_memory_db.commit()

    result = run_assessment_pipeline(
        in_memory_db,
        assessment.assessment_id,
        user.user_id,
    )

    assert result["status"] == "completed"
    assert "feedback" in result["result"]

    feedback = result["result"]["feedback"]

    present_symptoms = [
        s["symptom"] for s in feedback["symptoms_identified"]
    ]

    assert "cough" in present_symptoms
    assert "fever" not in present_symptoms

    pdf_res = generate_assessment_pdf(
        assessment.assessment_id,
        in_memory_db,
    )

    assert pdf_res["status"] == "ok"
    assert Path(pdf_res["path"]).exists()
    assert Path(pdf_res["path"]).stat().st_size > 1000


def test_unknown_symptom_as_string_and_dict_compatibility():
    """Unknown symptoms in dictionary or string format must be handled safely."""

    svc = get_condition_prediction_service()

    if not svc.is_available():
        pytest.skip("Condition prediction model not found")

    res_dict = svc.predict_conditions(
        symptoms_present=[{"symptom": "cough"}],
        symptoms_absent=[{"symptom": "vomiting"}],
        symptoms_unknown=[{"symptom": "fever", "state": "UNKNOWN"}],
    )

    assert res_dict["status"] in {"completed", "unsupported_condition"}
    assert res_dict["status"] != "unavailable"

    res_str = svc.predict_conditions(
        symptoms_present=[{"symptom": "cough"}],
        symptoms_absent=[{"symptom": "vomiting"}],
        symptoms_unknown=["fever"],
    )

    assert res_str["status"] in {"completed", "unsupported_condition"}
    assert res_str["status"] != "unavailable"


def test_outofdomain_symptoms_rejected_cleanly():
    """Symptoms outside the 24 classes must be rejected cleanly."""

    svc = get_condition_prediction_service()

    if not svc.is_available():
        pytest.skip("Condition prediction model not found")

    res_toe = svc.predict_conditions(
        symptoms_present=[
            {
                "symptom": "stubbed toe",
                "body_area": "foot",
                "quality": "throbbing",
            }
        ],
        symptoms_absent=[],
        symptoms_unknown=[],
        raw_text="stubbed toe throbbing bruised nail bleeding",
    )

    assert res_toe["status"] == "unsupported_condition"
    assert len(res_toe["conditions"]) == 0
    assert "rejection_threshold" in res_toe
    assert "not show strong alignment" in res_toe["message"]

    res_tooth = svc.predict_conditions(
        symptoms_present=[
            {
                "symptom": "toothache",
                "body_area": "mouth",
                "quality": "sharp",
            }
        ],
        symptoms_absent=[],
        symptoms_unknown=[],
        raw_text="sharp toothache sensitive to cold ice cream cavity",
    )

    assert res_tooth["status"] == "unsupported_condition"
    assert len(res_tooth["conditions"]) == 0


def test_empty_input_handled_safely():
    """An empty symptom list must return insufficient_evidence."""

    svc = get_condition_prediction_service()
    res = svc.predict_conditions([], [], [])

    assert res["status"] == "insufficient_evidence"
    assert res["conditions"] == []


def test_pipeline_uncertain_fever_end_to_end_does_not_crash_ml(in_memory_db):
    """Unknown fever remains UNKNOWN throughout the assessment pipeline."""

    user = User(
        user_id=uuid4(),
        email="fever_test@medisense.ai",
        password_hash="fake",
        name="Jane Doe",
        state="Karnataka",
        district="Bengaluru",
    )

    assessment = Assessment(
        assessment_id=uuid4(),
        user_id=user.user_id,
        input_types=["symptoms"],
        status="draft",
    )

    s1 = Symptom(
        symptom_id=uuid4(),
        assessment_id=assessment.assessment_id,
        symptom="cough",
        state="PRESENT",
        severity="MILD",
        duration="4 days",
    )

    s2 = Symptom(
        symptom_id=uuid4(),
        assessment_id=assessment.assessment_id,
        symptom="fever",
        state="UNKNOWN",
        severity="UNKNOWN",
        context="not sure if I have a fever",
    )

    s3 = Symptom(
        symptom_id=uuid4(),
        assessment_id=assessment.assessment_id,
        symptom="vomiting",
        state="ABSENT",
        severity="ABSENT",
    )

    in_memory_db.add_all([user, assessment, s1, s2, s3])
    in_memory_db.commit()

    result = run_assessment_pipeline(
        in_memory_db,
        assessment.assessment_id,
        user.user_id,
    )

    assert result["status"] == "completed"

    payload = result["result"]
    assert "condition_predictions" in payload

    cp = payload["condition_predictions"]

    assert cp["status"] in {"completed", "unsupported_condition"}
    assert cp["status"] != "unavailable"

    ev_unknown = [
        s["symptom"] if isinstance(s, dict) else s
        for s in payload["evidence"]["symptoms"]["unknown"]
    ]

    assert "fever" in ev_unknown

    ev_present = [
        s["symptom"]
        for s in payload["evidence"]["symptoms"]["present"]
    ]

    assert "cough" in ev_present
    assert "fever" not in ev_present

    ev_absent = [
        s["symptom"]
        for s in payload["evidence"]["symptoms"]["absent"]
    ]

    assert "vomiting" in ev_absent
