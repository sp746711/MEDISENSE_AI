"""Unit and integration tests for Assessment Orchestration Pipeline and PDF Generation."""

import os
import pytest
from uuid import uuid4

from app.database.database import SessionLocal, init_db
from app.database.models import Assessment, Symptom, User
from app.services.assessment_service import run_assessment_pipeline
from app.services.pdf_service import generate_assessment_pdf


@pytest.fixture(scope="module")
def db_session():
    init_db()
    session = SessionLocal()
    yield session
    session.close()


def test_assessment_pipeline_and_pdf(db_session):
    # Create test user
    user = User(
        name="Test Patient",
        email=f"patient_{uuid4().hex[:8]}@example.com",
        password_hash="fakehash",
        state="Karnataka",
        district="Bengaluru Urban",
        city="Bengaluru",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    # Create assessment
    assessment = Assessment(
        user_id=user.user_id,
        input_types=["symptoms"],
        status="draft",
    )
    db_session.add(assessment)
    db_session.commit()
    db_session.refresh(assessment)

    # Add symptoms
    s1 = Symptom(
        assessment_id=assessment.assessment_id,
        symptom="cough",
        state="PRESENT",
        duration="4 days",
        body_area="chest",
        source="user_input",
    )
    s2 = Symptom(
        assessment_id=assessment.assessment_id,
        symptom="chest pain",
        state="ABSENT",
        body_area="chest",
        source="user_input",
    )
    db_session.add_all([s1, s2])
    db_session.commit()

    # Run assessment pipeline
    result = run_assessment_pipeline(db_session, assessment.assessment_id, user.user_id)
    assert result["status"] in {"completed", "consultation", "mild", "emergency"}
    assert result["pathway"] in {"EMERGENCY", "CONSULTATION", "MILD"}
    assert result["specialty"] is not None

    # Verify PDF generation
    pdf_res = generate_assessment_pdf(assessment.assessment_id, db_session)
    assert pdf_res["status"] == "ok"
    assert os.path.exists(pdf_res["path"])
    assert pdf_res["path"].endswith(".pdf")
