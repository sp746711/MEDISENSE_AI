"""End-to-End Live Workflow & Pipeline Verification for Right-Knee Case.

Executes the entire backend pipeline on the synthetic right-knee presentation:
1. Symptoms NLP extraction
2. Laboratory Report NLP extraction
3. Radiology Report NLP extraction
4. Follow-up engine dynamic question selection & answering
5. Evidence engine combination
6. Triage engine rule evaluation
7. Specialty mapping
8. Assessment feedback & clinical guidance generation
9. PDF report generation and text inspection
10. Strict assertions for every Goal 1 requirement.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from app.ai.symptom_nlp import extract_symptoms
from app.ai.report_nlp import structure_report_text
from app.ai.followup_engine import suggest_followups, apply_followup_answer
from app.ai.evidence_engine import combine_evidence
from app.ai.triage_engine import apply_triage
from app.ai.specialty_mapper import map_evidence_to_specialty
from app.services.assessment_service import generate_assessment_feedback
from app.services.pdf_service import generate_assessment_pdf


def run_right_knee_e2e_verification():
    # 1. Inputs
    raw_symptom_text = (
        "I have right knee pain, medial knee pain, and swelling for approximately 14 days "
        "following a twisting injury while playing football. Pain severity is 6/10. "
        "Morning stiffness and pain on stairs, bending, and squatting. "
        "Intermittent clicking/catching. No fever. No other joint pain."
    )

    lab_report_text = """
    COMPLETE BLOOD COUNT (CBC) & INFLAMMATORY MARKERS
    Test Name               Result      Unit            Reference Interval   Flag
    Total WBC count         9.8         ×10³/µL         4.5 - 11.0
    Platelet count          276         ×10³/µL         150 - 450
    C-Reactive Protein      12.6        mg/L            0.0 - 5.0            HIGH
    Erythrocyte Sedimentation Rate (ESR)  36  mm/hr     0 - 20               HIGH
    """

    radiology_report_text = """
    CLINICAL HISTORY: 24-year-old male with acute twisting injury to right knee while playing football. Persistent pain and swelling.
    TECHNIQUE: Frontal and lateral radiographs of the right knee.
    FINDINGS:
    - No evidence of acute fracture or dislocation.
    - Joint spaces are preserved.
    - Mild soft-tissue swelling noted around the suprapatellar and medial compartment.
    IMPRESSION:
    1. No acute bony injury or dislocation identified.
    2. Mild periarticular soft-tissue swelling.
    3. Clinical correlation and MRI recommended if meniscal or ligamentous pathology is suspected.
    """

    xray_result_meta = {
        "received": True,
        "region": "knee",
        "status": "UNAVAILABLE",
        "reason_code": "unsupported_region",
        "prediction": None,
        "findings": [],
        "uncertainty": "Automated interpretation for Knee radiography is not configured.",
        "message": "Automated image interpretation for Knee is currently unavailable because no compatible trained model is configured for this region.",
        "file_name": "synthetic_right_knee_xray_AP_lateral.png",
        "images": [{
            "file_name": "synthetic_right_knee_xray_AP_lateral.png",
            "region": "knee",
            "status": "UNAVAILABLE",
            "prediction": None,
            "findings": [],
        }],
    }

    # 2. Symptom extraction
    nlp_res = extract_symptoms(raw_symptom_text)
    symptoms = nlp_res["symptoms"]

    # 3. Lab extraction
    lab_res = structure_report_text(lab_report_text)
    lab_findings = lab_res["findings"]

    # 4. Radiology extraction
    rad_res = structure_report_text(radiology_report_text)
    rad_findings = rad_res["qualitative_findings"]

    # 5. Follow-ups
    followup_res = suggest_followups(symptoms=symptoms, raw_text=raw_symptom_text)
    questions = followup_res.get("questions", [])
    q_ids = [q["id"] for q in questions]

    # Verify no headache questions
    assert "q_sudden_onset" not in q_ids, "Headache onset question must not be asked for knee presentation"

    # Simulate answering relevant limb movement follow-up
    updated_symptoms = symptoms
    if "q_limb_movement" in q_ids:
        updated_symptoms = apply_followup_answer(updated_symptoms, "q_limb_movement", "Reduced movement")

    # 6. Evidence combination
    evidence = combine_evidence(
        symptoms=updated_symptoms,
        report_findings=lab_findings,
        qualitative_report_findings=rad_findings,
        xray_result=xray_result_meta,
        report_provided=True,
    )

    # 7. Triage
    triage = apply_triage(evidence)

    # 8. Specialty
    spec_result = map_evidence_to_specialty(evidence)
    specialty_name = spec_result.get("suggested_specialty") if isinstance(spec_result, dict) else spec_result

    # 9. Assessment feedback
    feedback = generate_assessment_feedback(
        evidence=evidence,
        triage_result=triage,
        specialty=specialty_name,
    )

    # 10. Persist and Generate PDF
    from uuid import uuid4
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database.models import Base, User, Assessment

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        user = User(
            user_id=uuid4(),
            email="knee_patient@example.com",
            password_hash="pw",
            name="Right Knee Patient",
            state="Maharashtra",
            district="Mumbai",
        )
        session.add(user)
        session.commit()

        assessment_uuid = uuid4()
        assessment = Assessment(
            assessment_id=assessment_uuid,
            user_id=user.user_id,
            input_types=["symptoms", "medical_reports", "radiology_reports", "xray"],
            status="completed",
            result_payload={
                "feedback": feedback,
                "triage": triage,
                "evidence": evidence,
                "specialty": specialty_name,
            },
        )
        session.add(assessment)
        session.commit()

        pdf_res = generate_assessment_pdf(assessment_uuid, session)
        assert pdf_res["status"] == "ok"
        pdf_path = pdf_res["path"]
        assert pdf_path and Path(pdf_path).exists()

        pdf_bytes = Path(pdf_path).read_bytes()
        assert len(pdf_bytes) > 1000, "PDF generation failed"

        # Read PDF text using pdfminer or pypdf
        pdf_text = ""
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                pdf_text += page.extract_text() + "\n"
        except Exception:
            try:
                from pdfminer.high_level import extract_text
                import io
                pdf_text = extract_text(io.BytesIO(pdf_bytes))
            except Exception:
                pdf_text = str(pdf_bytes)

        pdf_text_lower = pdf_text.lower()
    finally:
        session.close()

    # CRITICAL ASSERTIONS:
    # A. Numbness must not be PRESENT
    present_symptoms = {s["symptom"].lower(): s for s in evidence["symptoms"]["present"]}
    assert "numbness" not in present_symptoms, "Numbness must not be marked PRESENT in evidence"
    assert "numbness: present" not in pdf_text_lower, "PDF must not report Numbness as PRESENT"

    # B. Knee pain duration and severity preserved
    assert "knee pain" in present_symptoms
    kp = present_symptoms["knee pain"]
    assert kp.get("duration") == "14 days", f"Expected duration '14 days', got {kp.get('duration')}"
    assert str(kp.get("severity")) == "MODERATE", f"Expected severity 'MODERATE', got {kp.get('severity')}"

    # C. Pain must not be contradictory (both PRESENT and ABSENT)
    absent_symptoms = {s["symptom"].lower(): s for s in evidence["symptoms"]["absent"]}
    assert "pain" not in absent_symptoms, "Pain must not be marked ABSENT"
    assert "knee pain" not in absent_symptoms

    # D. Labs: WBC and Platelets preserve scaled multipliers
    lab_map = {item["test_name"]: item for item in evidence["report"]["abnormal_findings"] + evidence["report"]["normal_findings"]}
    wbc = next(v for k, v in lab_map.items() if "White Blood Cell" in k or "WBC" in k)
    assert (wbc.get("value") == pytest.approx(9.8) or wbc.get("numeric_value") == pytest.approx(9.8))
    assert "10" in wbc["unit"] or "³" in wbc["unit"] or "^3" in wbc["unit"], f"Multiplier stripped from WBC unit: {wbc['unit']}"

    plt = next(v for k, v in lab_map.items() if "Platelet" in k)
    assert (plt.get("value") == pytest.approx(276.0) or plt.get("numeric_value") == pytest.approx(276.0))
    assert "10" in plt["unit"] or "³" in plt["unit"] or "^3" in plt["unit"], f"Multiplier stripped from Platelet unit: {plt['unit']}"

    # E. Labs: ESR and CRP extracted and HIGH
    esr = next(v for k, v in lab_map.items() if "Erythrocyte Sedimentation Rate" in k or "ESR" in k)
    assert (esr.get("value") == pytest.approx(36.0) or esr.get("numeric_value") == pytest.approx(36.0))
    assert esr["status"] == "HIGH"

    crp = next(v for k, v in lab_map.items() if "C-Reactive Protein" in k or "CRP" in k)
    assert (crp.get("value") == pytest.approx(12.6) or crp.get("numeric_value") == pytest.approx(12.6))
    assert crp["status"] == "HIGH"

    # F. Radiology: mild soft-tissue swelling PRESENT, fracture and dislocation ABSENT
    rad_map = {f["finding"].lower(): f for f in evidence["report"]["qualitative_findings"]}
    assert rad_map["fracture"]["state"] == "ABSENT"
    assert rad_map["dislocation"]["state"] == "ABSENT"
    assert any("soft-tissue swelling" in k or "soft tissue swelling" in k for k in rad_map)
    st = next(f for k, f in rad_map.items() if "soft" in k and "swelling" in k)
    assert st["state"] == "PRESENT"

    # G. Specialty: Orthopedics
    assert "orthoped" in specialty_name.lower(), f"Expected Orthopedics, got {specialty_name}"

    # H. Guidance: No neurologist or acute headache recommendations
    rec_guidance = feedback.get("recommended_guidance", "").lower()
    rec_action = feedback.get("recommended_action", "").lower()
    summary = feedback.get("assessment_summary", "").lower()
    combined_guidance = f"{rec_guidance} {rec_action} {summary}"

    assert "neurologist" not in combined_guidance, "Guidance must not recommend neurologist for knee case"
    assert "headache" not in combined_guidance, "Guidance must not contain acute headache text for knee case"
    assert "orthoped" in combined_guidance or "joint" in combined_guidance or "musculoskeletal" in combined_guidance

    # I. X-Ray Status: UNAVAILABLE without pathology fabrication
    xray_meta = evidence["xray"]
    assert xray_meta["status"] in {"UNAVAILABLE", "NOT_PROVIDED"}
    assert xray_meta["prediction"] is None, "Pathology must not be fabricated for unavailable X-ray"

    print("ALL RIGHT-KNEE E2E ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    run_right_knee_e2e_verification()
