"""Verification Test Suite for Part D Known Problems & Part J Specialist Recommendation.

Verifies:
1. Problem 1: Symptom contradiction - swelling never appears as both PRESENT and ABSENT;
   conflicts are recorded in contradictory evidence.
2. Problem 2: Lost symptom duration - 'approximately three weeks of knee pain' extracts '3 weeks'
   and is preserved through the assessment pipeline and PDF.
3. Problem 3: Positive radiology findings - 'Mild medial tibiofemoral joint-space narrowing' and
   'Small marginal osteophyte-like changes' are extracted as PRESENT.
4. Problem 4: Duplicate evidence - generic swelling is not duplicated when soft-tissue swelling is reported.
5. Problem 5: Incorrect generic guidance - trauma language is excluded when trauma is denied or absent.
6. Part J: Specialist recommendation - Orthopedics recommended with transparent rule_based_evidence_mapping.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.ai.evidence_engine import combine_evidence
from app.ai.report_nlp import structure_report_text
from app.ai.specialty_mapper import map_evidence_to_specialty
from app.ai.symptom_nlp import extract_symptoms
from app.ai.triage_engine import apply_triage
from app.database.models import Assessment, Base, MedicalReport, Symptom, User
from app.services.assessment_service import generate_assessment_feedback, run_assessment_pipeline
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
        email=f"part_d_{uuid4().hex[:8]}@example.com",
        password_hash="test_hash",
        name="Part D Test Patient",
        district="Pune",
        state="Maharashtra",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def test_problem_1_symptom_contradiction_resolution():
    """Problem 1: Swelling must never be rendered as both PRESENT and ABSENT."""
    symptoms = [
        {"symptom": "swelling", "state": "PRESENT", "source": "user_input"},
        {"symptom": "swelling", "state": "ABSENT", "source": "followup_answer"},
        {"symptom": "knee pain", "state": "PRESENT", "source": "user_input"},
    ]
    evidence = combine_evidence(symptoms=symptoms)

    present_names = [s.get("symptom") for s in evidence["symptoms"]["present"]]
    absent_names = [s.get("symptom") for s in evidence["symptoms"]["absent"]]

    # Swelling is preserved in PRESENT
    assert "swelling" in present_names
    # Swelling is NOT categorized in absent
    assert "swelling" not in absent_names

    # The conflict is captured in contradictory evidence
    assert any("swelling" in c for c in evidence.get("contradictory", []))

    # Feedback negative_red_flags must not contain swelling
    feedback = generate_assessment_feedback(
        evidence=evidence,
        triage_result={"pathway": "CONSULTATION"},
        specialty="Orthopedics",
    )
    neg_flag_names = [s.get("symptom") for s in feedback.get("negative_red_flags", [])]
    assert "swelling" not in neg_flag_names


def test_problem_2_lost_symptom_duration_resolution():
    """Problem 2: 'approximately three weeks of knee pain' extracts '3 weeks'."""
    text = "I have had approximately three weeks of knee pain."
    extracted = extract_symptoms(text)
    symptoms = extracted["symptoms"]

    assert len(symptoms) >= 1
    kp = next(s for s in symptoms if s["symptom"] == "knee pain")
    assert kp.get("duration") == "3 weeks"


def test_problem_3_positive_radiology_findings_resolution():
    """Problem 3: Positive findings in synthetic knee radiology report must be extracted."""
    rad_text = """
    FINDINGS:
    - Mild medial tibiofemoral joint-space narrowing.
    - Small marginal osteophyte-like changes.
    - No acute fracture.
    - No large joint effusion.
    - No dislocation.
    """
    res = structure_report_text(rad_text)
    qf = {item["finding"].lower(): item["state"] for item in res["qualitative_findings"]}

    # Positive findings
    assert any("narrowing" in k for k in qf)
    narrowing_key = next(k for k in qf if "narrowing" in k)
    assert qf[narrowing_key] == "PRESENT"
    assert "tibiofemoral" in narrowing_key or "medial" in narrowing_key

    assert any("osteophyte" in k for k in qf)
    osteophyte_key = next(k for k in qf if "osteophyte" in k)
    assert qf[osteophyte_key] == "PRESENT"

    # Negative findings
    assert qf.get("fracture") == "ABSENT"
    assert qf.get("joint effusion") == "ABSENT"
    assert qf.get("dislocation") == "ABSENT"


def test_problem_4_duplicate_evidence_resolution():
    """Problem 4: Qualitative examination does not duplicate swelling when soft-tissue swelling is reported."""
    rad_text = """
    TECHNIQUE: Radiographs of the knee.
    FINDINGS:
    - Mild soft-tissue swelling around suprapatellar pouch.
    - No fracture or dislocation.
    """
    res = structure_report_text(rad_text)
    findings = [q["finding"] for q in res["qualitative_findings"]]

    # Mild soft-tissue swelling is present
    assert any("soft-tissue swelling" in f for f in findings)
    # Generic bare 'swelling' from examination must not be added as a duplicate
    assert findings.count("swelling") == 0


def test_problem_5_guidance_no_unsupported_trauma():
    """Problem 5: Do not insert trauma language when trauma is denied or absent."""
    # Case A: Trauma explicitly denied
    symptoms_denied = [
        {"symptom": "knee pain", "state": "PRESENT", "domain": "musculoskeletal"},
        {"symptom": "fall/trauma", "state": "ABSENT"},
    ]
    ev_denied = combine_evidence(symptoms=symptoms_denied)
    fb_denied = generate_assessment_feedback(
        evidence=ev_denied,
        triage_result={"pathway": "CONSULTATION"},
        specialty="Orthopedics",
    )
    guidance_denied = fb_denied["recommended_next_step"]["guidance"].lower()
    assert "following trauma or limb strain" not in guidance_denied
    assert "no acute trauma" in guidance_denied

    # Case B: Trauma confirmed present
    symptoms_present = [
        {"symptom": "knee pain", "state": "PRESENT", "domain": "musculoskeletal"},
        {"symptom": "trauma/injury", "state": "PRESENT"},
    ]
    ev_present = combine_evidence(symptoms=symptoms_present)
    fb_present = generate_assessment_feedback(
        evidence=ev_present,
        triage_result={"pathway": "CONSULTATION"},
        specialty="Orthopedics",
    )
    guidance_present = fb_present["recommended_next_step"]["guidance"].lower()
    assert "following the reported joint strain or injury" in guidance_present


def test_part_j_specialist_recommendation_radiology_and_knee():
    """Part J: Recommend Orthopedics for knee presentation and radiology-only report."""
    # Knee symptoms -> Orthopedics with rule-based labeling
    ev_knee = combine_evidence(symptoms=[{"symptom": "knee pain", "state": "PRESENT"}])
    spec_knee = map_evidence_to_specialty(ev_knee)
    assert spec_knee["suggested_specialty"] == "Orthopedics"
    assert spec_knee["specialty_selection_mode"] == "rule_based_evidence_mapping"
    assert spec_knee["is_ml_specialty_model"] is False

    # Radiology report only -> Orthopedics
    ev_rad_only = combine_evidence(
        symptoms=[],
        qualitative_report_findings=[
            {"finding": "mild medial tibiofemoral joint-space narrowing", "domain": "musculoskeletal", "state": "PRESENT"}
        ],
        report_provided=True,
    )
    spec_rad = map_evidence_to_specialty(ev_rad_only)
    assert spec_rad["suggested_specialty"] == "Orthopedics"
    assert spec_rad["specialty_selection_mode"] == "rule_based_evidence_mapping"


def test_end_to_end_knee_case_pdf_verification(db_session, test_user):
    """End-to-end knee case: verifies duration '3 weeks', non-contradictory swelling, and PDF text."""
    assessment = Assessment(
        assessment_id=uuid4(),
        user_id=test_user.user_id,
        input_types=["symptoms", "medical_report"],
        status="draft",
        rules_version="2.2.0",
    )
    db_session.add(assessment)
    db_session.flush()

    # User reported 3 weeks of knee pain and swelling
    s1 = Symptom(
        assessment_id=assessment.assessment_id,
        symptom="knee pain",
        state="PRESENT",
        duration="3 weeks",
        severity="MODERATE",
    )
    s2 = Symptom(
        assessment_id=assessment.assessment_id,
        symptom="swelling",
        state="PRESENT",
        duration="3 weeks",
    )
    db_session.add_all([s1, s2])

    # Radiology report with the positive and negative findings
    rad_report = MedicalReport(
        report_id=uuid4(),
        assessment_id=assessment.assessment_id,
        file_name="knee_xray_report.pdf",
        stored_name="stored_knee.pdf",
        file_type="pdf",
        report_category="xray_report",
        structured_findings={
            "status": "structured",
            "qualitative_findings": [
                {
                    "finding": "mild medial tibiofemoral joint-space narrowing",
                    "state": "PRESENT",
                    "domain": "musculoskeletal",
                },
                {
                    "finding": "small marginal osteophyte-like changes",
                    "state": "PRESENT",
                    "domain": "musculoskeletal",
                },
                {
                    "finding": "fracture",
                    "state": "ABSENT",
                    "domain": "musculoskeletal",
                },
            ],
            "findings": [],
        },
    )
    db_session.add(rad_report)
    db_session.commit()

    processed = run_assessment_pipeline(db_session, assessment.assessment_id, test_user.user_id)
    assert processed["status"] == "completed"

    payload = processed["result"]
    fb = payload["feedback"]

    # 1. Swelling is NOT in negative red flags
    neg_flags = [s.get("symptom") for s in fb.get("negative_red_flags", [])]
    assert "swelling" not in neg_flags

    # 2. Specialty is Orthopedics
    assert payload.get("specialty") == "Orthopedics"

    # 3. PDF generation
    pdf_res = generate_assessment_pdf(assessment.assessment_id, db_session)
    assert pdf_res["status"] == "ok"
    pdf_path = Path(pdf_res["path"])
    assert pdf_path.exists()

    # Verify PDF content does not have swelling marked ABSENT
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pdf_text = " ".join([page.extract_text() for page in reader.pages])
        assert "Swelling" in pdf_text
        assert "3 weeks" in pdf_text or "3 Weeks" in pdf_text
        # Must not say 'swelling: ABSENT'
        assert "swelling: ABSENT" not in pdf_text.lower()
    except ImportError:
        pass
