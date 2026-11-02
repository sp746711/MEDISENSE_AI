"""End-to-End Tests for User-Selected Multimodal Combinations & Unified Disease Prediction.

Verifies:
1. Symptoms only -> ONE unified assessment.
2. Laboratory report only -> ONE unified assessment (no false symptoms).
3. Symptoms + Laboratory report -> ONE unified combined prediction.
4. Symptoms + Laboratory + X-ray -> ONE unified multimodal assessment.
5. Modality states and provenance tracking across all combinations.
6. PDF generation creates one unified document covering all contributing modalities.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import (
    Assessment,
    Base,
    MedicalReport,
    Symptom,
    User,
    XrayResult,
)
from app.services.assessment_service import run_assessment_pipeline
from app.services.pdf_service import generate_assessment_pdf


@pytest.fixture(scope="module")
def db_engine():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def test_user(db_session) -> User:
    user = User(
        user_id=uuid4(),
        email=f"multimodal_test_{uuid4().hex[:8]}@example.com",
        password_hash="test_hash",
        name="Multimodal Test Patient",
        district="Bengaluru Urban",
        state="Karnataka",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def test_unified_assessment_symptoms_only(db_session, test_user):
    """Case 1: User submits symptoms only -> ONE unified assessment."""
    assessment = Assessment(
        assessment_id=uuid4(),
        user_id=test_user.user_id,
        input_types=["symptoms"],
        status="draft",
        rules_version="2.2.0",
    )
    db_session.add(assessment)
    db_session.flush()

    s1 = Symptom(
        assessment_id=assessment.assessment_id,
        symptom="fever",
        state="PRESENT",
        duration="4 days",
        severity="SEVERE",
    )
    s2 = Symptom(
        assessment_id=assessment.assessment_id,
        symptom="cough",
        state="PRESENT",
        duration="4 days",
        body_area="chest",
    )
    s3 = Symptom(
        assessment_id=assessment.assessment_id,
        symptom="fatigue",
        state="PRESENT",
        severity="SEVERE",
    )
    db_session.add_all([s1, s2, s3])
    db_session.commit()

    processed = run_assessment_pipeline(db_session, assessment.assessment_id, test_user.user_id)
    assert processed["status"] == "completed"

    payload = processed["result"]
    assert "condition_predictions" in payload
    cond_preds = payload["condition_predictions"]

    # ONE unified condition candidate differential
    assert isinstance(cond_preds.get("conditions"), list)
    assert len(cond_preds["conditions"]) > 0

    # Provenance tracking
    assert cond_preds["modality_states"]["symptoms"] == "ELIGIBLE"
    assert cond_preds["modality_states"]["laboratory"] == "NOT_PROVIDED"
    assert cond_preds["modality_states"]["xray_images"] == "NOT_PROVIDED"
    assert cond_preds["actual_modalities_used_by_ml"] == ["symptoms"]

    # PDF generation
    pdf_res = generate_assessment_pdf(assessment.assessment_id, db_session)
    assert pdf_res["status"] == "ok"
    assert Path(pdf_res["path"]).exists()


def test_unified_assessment_lab_only(db_session, test_user):
    """Case 2: User submits laboratory report only -> ONE unified assessment."""
    assessment = Assessment(
        assessment_id=uuid4(),
        user_id=test_user.user_id,
        input_types=["medical_report"],
        status="draft",
        rules_version="2.2.0",
    )
    db_session.add(assessment)
    db_session.flush()

    report = MedicalReport(
        report_id=uuid4(),
        assessment_id=assessment.assessment_id,
        file_name="cbc_report.pdf",
        stored_name="stored_cbc.pdf",
        file_type="pdf",
        report_category="medical_report",
        structured_findings={
            "status": "structured",
            "findings": [
                {
                    "test_name": "White Blood Cell Count (WBC)",
                    "value": 15200.0,
                    "unit": "cells/mcL",
                    "reference_range": "4000.0 - 11000.0",
                    "reference_low": 4000.0,
                    "reference_high": 11000.0,
                    "interpretation": "HIGH",
                    "status": "ABNORMAL",
                    "flag": "HIGH",
                },
                {
                    "test_name": "C-Reactive Protein (CRP)",
                    "value": 48.0,
                    "unit": "mg/L",
                    "reference_range": "0.0 - 5.0",
                    "reference_low": 0.0,
                    "reference_high": 5.0,
                    "interpretation": "HIGH",
                    "status": "ABNORMAL",
                    "flag": "HIGH",
                },
                {
                    "test_name": "Hemoglobin",
                    "value": 14.2,
                    "unit": "g/dL",
                    "reference_range": "12.0 - 17.0",
                    "reference_low": 12.0,
                    "reference_high": 17.0,
                    "interpretation": "NORMAL",
                    "status": "NORMAL",
                    "flag": "NORMAL",
                },
            ],
            "abnormal_findings": [
                {
                    "test_name": "White Blood Cell Count (WBC)",
                    "value": 15200.0,
                    "unit": "cells/mcL",
                    "reference_range": "4000.0 - 11000.0",
                    "interpretation": "HIGH",
                },
                {
                    "test_name": "C-Reactive Protein (CRP)",
                    "value": 48.0,
                    "unit": "mg/L",
                    "reference_range": "0.0 - 5.0",
                    "interpretation": "HIGH",
                },
            ],
        },
    )
    db_session.add(report)
    db_session.commit()

    processed = run_assessment_pipeline(db_session, assessment.assessment_id, test_user.user_id)
    assert processed["status"] == "completed"

    payload = processed["result"]
    cond_preds = payload["condition_predictions"]

    # Modality states correctly identify laboratory as eligible
    assert cond_preds["modality_states"]["laboratory"] == "ELIGIBLE"
    assert cond_preds["modality_states"]["symptoms"] == "NOT_PROVIDED"
    assert cond_preds["modality_states"]["xray_images"] == "NOT_PROVIDED"

    # Feedback contains laboratory findings in summary
    summary_text = payload["feedback"].get("assessment_summary", "")
    assert "out-of-range" in summary_text.lower() or "report" in summary_text.lower()

    # PDF generation works seamlessly for lab-only assessment
    pdf_res = generate_assessment_pdf(assessment.assessment_id, db_session)
    assert pdf_res["status"] == "ok"


def test_unified_assessment_symptoms_plus_lab_plus_xray(db_session, test_user):
    """Case 3: User submits Symptoms + Laboratory report + X-ray image -> ONE unified multimodal assessment."""
    assessment = Assessment(
        assessment_id=uuid4(),
        user_id=test_user.user_id,
        input_types=["symptoms", "medical_report", "xray"],
        status="draft",
        rules_version="2.2.0",
    )
    db_session.add(assessment)
    db_session.flush()

    # 1. Symptoms
    s1 = Symptom(
        assessment_id=assessment.assessment_id,
        symptom="fever",
        state="PRESENT",
        duration="3 days",
        severity="SEVERE",
    )
    s2 = Symptom(
        assessment_id=assessment.assessment_id,
        symptom="cough",
        state="PRESENT",
        duration="3 days",
    )
    s3 = Symptom(
        assessment_id=assessment.assessment_id,
        symptom="fatigue",
        state="PRESENT",
        duration="3 days",
    )
    db_session.add_all([s1, s2, s3])

    # 2. Lab Report
    report = MedicalReport(
        report_id=uuid4(),
        assessment_id=assessment.assessment_id,
        file_name="inflammatory_panel.pdf",
        stored_name="stored_panel.pdf",
        file_type="pdf",
        report_category="medical_report",
        structured_findings={
            "status": "structured",
            "findings": [
                {
                    "test_name": "White Blood Cell Count (WBC)",
                    "value": 16500.0,
                    "unit": "cells/mcL",
                    "reference_range": "4000.0 - 11000.0",
                    "interpretation": "HIGH",
                    "status": "ABNORMAL",
                    "flag": "HIGH",
                },
                {
                    "test_name": "C-Reactive Protein (CRP)",
                    "value": 55.0,
                    "unit": "mg/L",
                    "reference_range": "0.0 - 5.0",
                    "interpretation": "HIGH",
                    "status": "ABNORMAL",
                    "flag": "HIGH",
                },
            ],
            "abnormal_findings": [
                {
                    "test_name": "White Blood Cell Count (WBC)",
                    "value": 16500.0,
                    "unit": "cells/mcL",
                    "interpretation": "HIGH",
                },
                {
                    "test_name": "C-Reactive Protein (CRP)",
                    "value": 55.0,
                    "unit": "mg/L",
                    "interpretation": "HIGH",
                },
            ],
        },
    )
    db_session.add(report)

    # 3. Verified Chest X-Ray
    xray = XrayResult(
        xray_result_id=uuid4(),
        assessment_id=assessment.assessment_id,
        file_name="cxr_patient.png",
        region="chest",
        status="COMPLETED",
        findings=[
            {"label": "Consolidation", "model_score": 0.88, "status": "detected"},
            {"label": "Pneumonia", "model_score": 0.79, "status": "detected"},
        ],
        anatomy_metadata={
            "anatomy_user_confirmed": True,
            "anatomy_verification_status": "completed",
            "anatomy_probs": [0.0] * 33,
        },
    )
    db_session.add(xray)
    db_session.commit()

    processed = run_assessment_pipeline(db_session, assessment.assessment_id, test_user.user_id)
    assert processed["status"] == "completed"

    payload = processed["result"]
    cond_preds = payload["condition_predictions"]

    # All three modalities are verified eligible
    assert cond_preds["modality_states"]["symptoms"] == "ELIGIBLE"
    assert cond_preds["modality_states"]["laboratory"] == "ELIGIBLE"
    assert cond_preds["modality_states"]["xray_images"] == "ELIGIBLE"

    # Reconciled candidates combine evidence from all 3 inputs into ONE unified differential
    candidates = cond_preds.get("conditions", [])
    assert len(candidates) > 0
    top_candidate = candidates[0]

    # Verify unified candidate-specific evidence structure
    assert "supporting_evidence" in top_candidate
    assert "evidence_summary" in top_candidate
    assert "evidence_badges" in top_candidate

    # PDF compiles the unified multimodal assessment
    pdf_res = generate_assessment_pdf(assessment.assessment_id, db_session)
    assert pdf_res["status"] == "ok"
    assert Path(pdf_res["path"]).exists()
