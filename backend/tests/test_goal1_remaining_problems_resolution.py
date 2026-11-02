"""Comprehensive Verification of Goal 1 Remaining Problems Resolution.

Tests the 5 identified findings:
1. Symptom duration: 3 weeks preserved from 'about three weeks' narrative.
2. Swelling contradiction: Knee swelling (PRESENT) and calf swelling (ABSENT) do not contradict.
3. Positive radiology findings: Mild medial tibiofemoral joint-space narrowing & small marginal osteophytes extracted.
4. Duplicate evidence: Negative findings and swelling evidence deduplicated across ledger.
5. Deterministic triage: CONSULTATION pathway assigned (NOT DETERMINED resolved).
6. Specialist recommendation: Orthopedics (rule_based_evidence_mapping).
7. PDF generation: Generated without contradiction, displaying CONSULTATION and positive findings.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from app.ai.symptom_nlp import extract_symptoms
from app.ai.report_nlp import structure_report_text
from app.ai.evidence_engine import combine_evidence
from app.ai.triage_engine import apply_triage
from app.ai.specialty_mapper import map_evidence_to_specialty
from app.services.assessment_service import generate_assessment_feedback
from app.database.database import SessionLocal
from app.database.models import Assessment, Symptom, MedicalReport, XrayResult
from app.services.pdf_service import generate_assessment_pdf


def test_finding_1_and_2_symptom_duration_and_no_swelling_contradiction():
    """Verify duration '3 weeks' is extracted and calf swelling negation does not contradict knee swelling."""
    text = (
        "I have been having pain in my right knee for about three weeks. It started gradually and I cannot remember one specific injury. "
        "The pain is mostly around the inner side and front of the knee. It gets worse when I walk for a long time, climb stairs, or stand up after sitting. "
        "Sometimes the knee looks a little swollen in the evening. It feels stiff when I first get up in the morning, but usually loosens after 10–15 minutes. "
        "I can still walk, although I avoid putting too much weight on it when it hurts. "
        "I have not had a fever. The knee is not very red or hot. I have not noticed calf swelling, numbness, or foot colour changes. "
        "I have not had a recent fall or major accident. I have not had the same problem in other joints."
    )
    res = extract_symptoms(text)
    symptoms = res["symptoms"]

    # 1. Duration preserved
    pain_symptom = next((s for s in symptoms if s["symptom"] == "pain"), None)
    assert pain_symptom is not None
    assert pain_symptom["duration"] == "3 weeks"

    # 2. Knee swelling vs calf swelling separation
    present_syms = [s for s in symptoms if s["state"] == "PRESENT"]
    absent_syms = [s for s in symptoms if s["state"] == "ABSENT"]

    has_swelling_present = any(s["symptom"] in {"swelling", "knee swelling"} and s.get("body_area") in {"knee", "extremity/joints"} for s in present_syms)
    assert has_swelling_present, "Knee swelling must be PRESENT"

    has_calf_swelling_absent = any(s["symptom"] == "calf swelling" and s.get("body_area") == "calf" for s in absent_syms)
    assert has_calf_swelling_absent, "Calf swelling must be ABSENT and recognized as calf-specific"

    # 3. Evidence engine combination must NOT produce contradiction
    evidence = combine_evidence(symptoms=symptoms)
    contradictions = evidence.get("contradictory", [])
    swelling_contradictions = [c for c in contradictions if "swelling" in c.lower()]
    assert len(swelling_contradictions) == 0, f"Expected 0 swelling contradictions, got: {swelling_contradictions}"


def test_finding_3_positive_and_negative_radiology_findings():
    """Verify positive findings (joint-space narrowing, marginal osteophytes) and negative findings extracted."""
    rad_text = """
    SYNTHETIC TEST DATA - NOT A REAL PATIENT
    Radiology Report - Right Knee X-ray
    Case ID: KNEE-SYN-2026-1101-01
    Study: Right knee AP and lateral views
    FINDINGS:
    - Mild medial tibiofemoral joint-space narrowing.
    - Small marginal osteophyte-like changes.
    - No acute fracture.
    - No large joint effusion.
    - No dislocation.
    IMPRESSION:
    1. Mild medial compartment osteoarthritis features (narrowing and marginal osteophytes).
    2. No acute fracture or dislocation.
    """
    res = structure_report_text(rad_text)
    qf = {item["finding"].lower(): item["state"] for item in res["qualitative_findings"]}

    # Positive findings
    narrowing_keys = [k for k in qf if "narrowing" in k]
    assert len(narrowing_keys) > 0
    assert qf[narrowing_keys[0]] == "PRESENT"
    assert "medial" in narrowing_keys[0] or "tibiofemoral" in narrowing_keys[0]

    osteophyte_keys = [k for k in qf if "osteophyte" in k]
    assert len(osteophyte_keys) > 0
    assert qf[osteophyte_keys[0]] == "PRESENT"

    # Negative findings
    assert qf.get("fracture") == "ABSENT"
    assert qf.get("joint effusion") == "ABSENT"
    assert qf.get("dislocation") == "ABSENT"


def test_finding_4_and_5_deduplication_and_deterministic_triage():
    """Verify evidence ledger deduplication and deterministic triage produces CONSULTATION."""
    symptoms = [
        {"symptom": "pain", "state": "PRESENT", "duration": "3 weeks", "body_area": "knee", "domain": "musculoskeletal"},
        {"symptom": "swelling", "state": "PRESENT", "body_area": "knee", "domain": "musculoskeletal"},
        {"symptom": "reduced movement", "state": "PRESENT", "body_area": "knee", "domain": "musculoskeletal"},
        {"symptom": "calf swelling", "state": "ABSENT", "body_area": "calf", "domain": "musculoskeletal"},
        {"symptom": "fall/trauma", "state": "ABSENT", "domain": "musculoskeletal"},
        {"symptom": "fever", "state": "ABSENT", "domain": "general"},
    ]
    lab_findings = [
        {"test_name": "White Blood Cell Count (WBC)", "value": 8.4, "unit": "10^9/L", "reference_range": "4.0 - 11.0", "status": "NORMAL"},
        {"test_name": "C-Reactive Protein (CRP)", "value": 8.6, "unit": "mg/L", "reference_range": "< 5.0", "status": "HIGH"},
        {"test_name": "Erythrocyte Sedimentation Rate (ESR)", "value": 26.0, "unit": "mm/hr", "reference_range": "0 - 20", "status": "HIGH"},
        {"test_name": "Uric Acid", "value": 5.4, "unit": "mg/dL", "reference_range": "3.5 - 7.2", "status": "NORMAL"},
        {"test_name": "Hemoglobin", "value": 14.1, "unit": "g/dL", "reference_range": "13.0 - 17.0", "status": "NORMAL"},
    ]
    xray_reports = [{
        "file_name": "synthetic_right_knee_radiology_report.pdf",
        "qualitative_findings": [
            {"finding": "mild medial tibiofemoral joint-space narrowing", "state": "PRESENT", "domain": "musculoskeletal", "category": "radiology/joint"},
            {"finding": "small marginal osteophyte-like changes", "state": "PRESENT", "domain": "musculoskeletal", "category": "radiology/bone"},
            {"finding": "fracture", "state": "ABSENT", "domain": "musculoskeletal", "category": "radiology/bone"},
            {"finding": "joint effusion", "state": "ABSENT", "domain": "musculoskeletal", "category": "radiology"},
            {"finding": "dislocation", "state": "ABSENT", "domain": "musculoskeletal", "category": "radiology/bone"},
        ],
    }]

    evidence = combine_evidence(
        symptoms=symptoms,
        report_findings=lab_findings,
        xray_reports=xray_reports,
        medical_report_provided=True,
        radiology_report_provided=True,
    )

    # Deduplication check: no duplicate entries in supporting, reassuring, or relationships
    supp = evidence.get("supporting", [])
    assert len(supp) == len(set(supp)), "Supporting evidence must not have duplicate entries"

    reassuring = evidence.get("reassuring", [])
    assert len(reassuring) == len(set(reassuring)), "Reassuring evidence must not have duplicate entries"

    contra = evidence.get("contradictory", [])
    assert len(contra) == 0, f"Expected 0 contradictions, got: {contra}"

    # Triage engine: must assign CONSULTATION
    triage = apply_triage(evidence)
    assert triage["pathway"] == "CONSULTATION", f"Expected CONSULTATION pathway, got: {triage.get('pathway')}"
    assert "rule_c3_persistent_symptom_duration" in triage.get("rules_triggered", [])
    assert "rule_c1_abnormal_laboratory_findings" in triage.get("rules_triggered", [])

    # Specialty: must recommend Orthopedics
    spec = map_evidence_to_specialty(evidence)
    assert spec["suggested_specialty"] == "Orthopedics"
    assert spec["specialty_selection_mode"] == "rule_based_evidence_mapping"
    assert spec["is_ml_specialty_model"] is False


def test_end_to_end_assessment_and_pdf_generation():
    """Verify live database assessment creation, pipeline run, and PDF generation."""
    import uuid
    from app.database.models import User

    db = SessionLocal()
    try:
        user = db.query(User).first()
        test_user_id = user.user_id if user else uuid.uuid4()

        # Create test assessment
        assessment = Assessment(
            user_id=test_user_id,
            input_types=["symptoms", "medical_report", "radiology_report"],
            status="draft",
        )
        db.add(assessment)
        db.commit()
        db.refresh(assessment)

        # Add symptoms
        raw_symptom_text = (
            "I have been having pain in my right knee for about three weeks. It started gradually and I cannot remember one specific injury. "
            "Sometimes the knee looks a little swollen in the evening. I have not had a fever. I have not noticed calf swelling or numbness. "
            "I have not had a recent fall or accident."
        )
        sym_res = extract_symptoms(raw_symptom_text)
        for s in sym_res["symptoms"]:
            db.add(Symptom(
                assessment_id=assessment.assessment_id,
                symptom=s["symptom"],
                state=s["state"],
                duration=s.get("duration"),
                severity=s.get("severity"),
                body_area=s.get("body_area"),
                context=s.get("context", ""),
                source="user_input",
            ))

        # Add Lab Report
        db.add(MedicalReport(
            assessment_id=assessment.assessment_id,
            file_name="synthetic_lab_report.pdf",
            stored_name="test_synthetic_lab_report.pdf",
            file_type="application/pdf",
            report_category="medical_report",
            extracted_text="CRP: 8.6 mg/L (HIGH), ESR: 26 mm/hr (HIGH), WBC: 8.4 x10^9/L (NORMAL)",
            structured_findings={
                "lab_parameters": [
                    {"test_name": "C-Reactive Protein (CRP)", "value": 8.6, "unit": "mg/L", "reference_range": "< 5.0", "status": "HIGH"},
                    {"test_name": "Erythrocyte Sedimentation Rate (ESR)", "value": 26.0, "unit": "mm/hr", "reference_range": "0 - 20", "status": "HIGH"},
                    {"test_name": "White Blood Cell Count (WBC)", "value": 8.4, "unit": "10^9/L", "reference_range": "4.0 - 11.0", "status": "NORMAL"},
                ]
            },
        ))

        # Add Radiology Report
        db.add(MedicalReport(
            assessment_id=assessment.assessment_id,
            file_name="synthetic_knee_radiology_report.pdf",
            stored_name="test_synthetic_knee_radiology_report.pdf",
            file_type="application/pdf",
            report_category="xray_report",
            extracted_text="FINDINGS: Mild medial tibiofemoral joint-space narrowing. Small marginal osteophyte-like changes. No acute fracture or dislocation.",
            structured_findings={
                "qualitative_findings": [
                    {"finding": "mild medial tibiofemoral joint-space narrowing", "state": "PRESENT", "category": "radiology/joint", "domain": "musculoskeletal"},
                    {"finding": "small marginal osteophyte-like changes", "state": "PRESENT", "category": "radiology/bone", "domain": "musculoskeletal"},
                    {"finding": "fracture", "state": "ABSENT", "category": "radiology/bone", "domain": "musculoskeletal"},
                    {"finding": "dislocation", "state": "ABSENT", "category": "radiology/bone", "domain": "musculoskeletal"},
                ]
            },
        ))
        db.commit()

        # Run assessment pipeline
        from app.services.assessment_service import run_assessment_pipeline
        result = run_assessment_pipeline(db, assessment.assessment_id, test_user_id)
        assert result["pathway"] == "CONSULTATION", f"Expected CONSULTATION, got {result.get('pathway')}"
        assert result["specialty"] == "Orthopedics"

        # Generate PDF
        pdf_res = generate_assessment_pdf(assessment.assessment_id, db)
        assert pdf_res["status"] in {"ok", "success"}
        pdf_path = pdf_res.get("path") or pdf_res.get("file_path")
        assert os.path.exists(pdf_path)

        # Inspect generated PDF text
        from app.ai.report_ocr import extract_text_from_file
        extracted_pdf = extract_text_from_file(pdf_path)
        pdf_text = extracted_pdf.get("extracted_text", "")

        # Strict PDF assertions:
        # 1. Triage is CONSULTATION, NOT 'NOT DETERMINED'
        assert "CONSULTATION" in pdf_text
        assert "NOT DETERMINED" not in pdf_text

        # 2. Specialty is Orthopedics
        assert "Orthopedics" in pdf_text

        # 3. Positive radiology findings are visible
        assert "narrowing" in pdf_text.lower() or "osteophyte" in pdf_text.lower()

        # 4. Swelling is NOT listed as a negative red flag
        # (Negative red flags section must not contain swelling)
        if "Important Negative Findings" in pdf_text:
            neg_sec = pdf_text.split("Important Negative Findings")[1].split("Possible Conditions")[0]
            # Must not list bare swelling as absent
            for line in neg_sec.splitlines():
                if "swelling" in line.lower() and "calf" not in line.lower():
                    pytest.fail(f"Bare swelling found in negative findings: {line}")

        # Clean up test rows
        db.query(Symptom).filter(Symptom.assessment_id == assessment.assessment_id).delete()
        db.query(MedicalReport).filter(MedicalReport.assessment_id == assessment.assessment_id).delete()
        db.query(Assessment).filter(Assessment.assessment_id == assessment.assessment_id).delete()
        db.commit()
    finally:
        db.close()
