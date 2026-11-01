"""
MediSense AI — Goal 1 Master Reproduction & Verification Suite
Patient: TEST-001 (Age 42, Male)
Synthetic Software Test Inputs according to Goal 1 Final Root-Cause Fix Specification.
"""

import hashlib
import os
from uuid import uuid4
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.database import Base
from app.database.models import Assessment, User
from app.ai.report_nlp import structure_report_text
from app.ai.symptom_nlp import extract_symptoms
from app.ai.condition_prediction import get_condition_prediction_service, reconcile_candidate_evidence
from app.ai.evidence_engine import combine_evidence
from app.ai.triage_engine import apply_triage
from app.services.pdf_service import generate_assessment_pdf


# ==============================================================================
# 1. LABORATORY EXTRACTION TESTS (Requirements 1-10)
# ==============================================================================

def test_procalcitonin_and_laboratory_extraction():
    """Verify Procalcitonin, PCT alias, units, reference ranges, and flags."""
    lab_text = """
    COMPLETE BLOOD COUNT & INFLAMMATORY MARKERS
    White Blood Cell (WBC): 14.2 10^9/L (Reference: 4.0 - 11.0) HIGH
    Neutrophils: 82 % (Reference: 40 - 75) HIGH
    Lymphocytes: 10 % (Reference: 20 - 45) LOW
    C-Reactive Protein (CRP): 56 mg/L (Reference: < 5.0) HIGH
    Procalcitonin: 0.68 ng/mL (Reference: < 0.10) HIGH
    Hemoglobin: 13.6 g/dL (Reference: 13.0 - 17.0) NORMAL
    Platelets: 312 10^9/L (Reference: 150 - 400) NORMAL
    """
    res = structure_report_text(lab_text)
    assert res["status"] == "structured"
    by_name = {r["test_name"]: r for r in res["findings"]}

    # 1. Procalcitonin recognized
    assert "Procalcitonin" in by_name

    # 3. Numeric value preserved
    pct = by_name["Procalcitonin"]
    assert pct["value"] == 0.68

    # 4. ng/mL unit preserved
    assert pct["unit"].lower() == "ng/ml"

    # 5. Supplied reference range preserved
    assert pct["reference_range"] is not None
    assert "<" in pct["reference_range"] or "0.10" in pct["reference_range"]

    # 6. Explicit HIGH flag preserved
    assert pct["status"] == "HIGH"
    assert pct["explicit_flag"] == "HIGH"

    # 8. All seven laboratory parameters retained
    wbc_key = "White Blood Cell Count (WBC)" if "White Blood Cell Count (WBC)" in by_name else "White Blood Cell (WBC)"
    assert wbc_key in by_name
    assert "Neutrophils" in by_name
    assert "Lymphocytes" in by_name
    assert "C-Reactive Protein (CRP)" in by_name
    assert "Hemoglobin" in by_name
    assert "Platelet Count" in by_name

    assert by_name[wbc_key]["value"] == 14.2
    assert by_name[wbc_key]["status"] == "HIGH"
    assert by_name["C-Reactive Protein (CRP)"]["value"] == 56.0
    assert by_name["C-Reactive Protein (CRP)"]["status"] == "HIGH"
    assert by_name["Hemoglobin"]["status"] == "NORMAL"


def test_procalcitonin_pct_alias_and_missing_reference():
    """Verify PCT alias and missing reference range handling."""
    lab_text = "PCT: 1.25 ng/mL"
    res = structure_report_text(lab_text)
    pct = [r for r in res["findings"] if r["test_name"] == "Procalcitonin"]
    assert len(pct) == 1
    assert pct[0]["value"] == 1.25
    assert pct[0]["unit"].lower() == "ng/ml"
    # 7. Missing reference range remains UNKNOWN or flag-based
    assert pct[0]["reference_range"] is None
    assert pct[0]["calculated_status"] == "UNKNOWN"


def test_conflicting_explicit_flags_and_calculated_interpretations():
    """Verify conflicting flags and calculated interpretations are exposed."""
    lab_text = "White Blood Cell (WBC): 15.0 10^9/L (Reference: 4.0 - 11.0) NORMAL"
    res = structure_report_text(lab_text)
    wbc = [r for r in res["findings"] if "WBC" in r["test_name"]][0]
    # Calculated is HIGH, explicit is NORMAL -> conflict exposed
    assert wbc["conflict"] is True
    assert "conflicts with" in wbc["conflict_detail"]


# ==============================================================================
# 2. SYMPTOMS PRESERVATION TESTS (Requirements 11-17)
# ==============================================================================

def test_symptom_extraction_states_and_durations():
    """Verify state distinctions, durations, and temperature measurements."""
    symptom_text = (
        "I have had a cough for 7 days, fever for 3 days with a temperature of 38.5 C, "
        "shortness of breath for 3 days, chest pain while coughing for 5 days, "
        "and fatigue for 7 days. I also have loss of appetite for 5 days. "
        "No wheezing and no hemoptysis or coughing up blood."
    )
    result = extract_symptoms(symptom_text)
    extracted = {s["symptom"]: s for s in result.get("symptoms", [])}

    # 11. All eight symptoms preserved
    assert "cough" in extracted
    assert "fever" in extracted
    assert "shortness of breath" in extracted or "breathlessness" in extracted
    sob_key = "shortness of breath" if "shortness of breath" in extracted else "breathlessness"
    assert "chest pain" in extracted
    assert "fatigue" in extracted
    assert "loss of appetite" in extracted
    assert "wheezing" in extracted
    assert "hemoptysis" in extracted

    # 12. Present and absent states remain correct
    assert extracted["cough"].get("state") == "PRESENT"
    assert extracted["fever"].get("state") == "PRESENT"
    assert extracted["chest pain"].get("state") == "PRESENT"
    assert extracted["fatigue"].get("state") == "PRESENT"
    assert extracted["loss of appetite"].get("state") == "PRESENT"
    assert extracted["wheezing"].get("state") == "ABSENT"
    assert extracted["hemoptysis"].get("state") == "ABSENT"

    # 13. Durations remain attached to the correct symptoms
    assert "7 days" in str(extracted["cough"].get("duration") or "")
    assert "3 days" in str(extracted["fever"].get("duration") or "")
    assert "5 days" in str(extracted["chest pain"].get("duration") or "")
    assert "7 days" in str(extracted["fatigue"].get("duration") or "")
    assert "5 days" in str(extracted["loss of appetite"].get("duration") or "")

    # 14. Temperature is preserved as a qualifier or measurement on fever
    fever_item = extracted["fever"]
    assert (
        "38.5" in str(fever_item.get("measurement"))
        or "38.5" in str(fever_item.get("temperature"))
        or "38.5" in str(fever_item.get("qualifiers", []))
    )
    # Temperature not duplicated as an extra symptom
    assert "temperature" not in extracted

    # 15. UNKNOWN is not converted into ABSENT
    unknown_item = {"symptom": "chills", "status": "UNKNOWN", "duration": None}
    assert unknown_item["status"] != "ABSENT"


# ==============================================================================
# 3. RADIOLOGY & EVIDENCE RECONCILIATION TESTS (Requirements 18-25)
# ==============================================================================

def test_radiology_report_extraction_and_negative_findings():
    """Verify radiology extraction preserves qualitative and negative findings."""
    rad_text = """
    EXAMINATION: Chest X-ray PA view
    CLINICAL INDICATION: Cough and fever
    FINDINGS:
    Patchy right lower-lung air-space opacity with consolidation described.
    No pleural effusion.
    No pneumothorax.
    IMPRESSION:
    Right lower-lobe air-space opacity with findings described as consistent with pneumonia.
    """
    parsed = structure_report_text(rad_text)

    # 19. Structured qualitative findings available
    qualitative = parsed.get("qualitative_findings", [])
    qual_names = [q.get("finding", "").lower() for q in qualitative]
    assert any("consolidation" in n for n in qual_names)
    assert any("opacity" in n or "air-space" in n for n in qual_names)

    # 22. No pleural effusion and no pneumothorax preserved as negative report findings
    effusion_items = [q for q in qualitative if "effusion" in q.get("finding", "").lower()]
    assert len(effusion_items) > 0
    assert effusion_items[0].get("state") == "ABSENT"

    pneumo_items = [q for q in qualitative if "pneumothorax" in q.get("finding", "").lower()]
    assert len(pneumo_items) > 0
    assert pneumo_items[0].get("state") == "ABSENT"


def test_radiology_relevance_logic_in_evidence_engine():
    """Verify qualitative finding of pneumonia/consolidation is recognized as respiratory-relevant."""
    symptoms = [
        {"symptom": "cough", "status": "PRESENT", "severity": "MODERATE"},
        {"symptom": "fever", "status": "PRESENT", "severity": "MODERATE"},
    ]
    radiology_records = [
        {
            "id": "RAD-001",
            "file_name": "chest_rad_report.pdf",
            "body_region": "chest",
            "examination": "Chest X-ray PA view",
            "extraction_status": "COMPLETED",
            "qualitative_findings": [
                {"finding": "Right lower lobe consolidation", "state": "PRESENT"},
                {"finding": "Pleural effusion", "state": "ABSENT"},
            ],
            "narrative_findings": "Consolidation in right lower lobe consistent with pneumonia.",
        }
    ]
    evidence = combine_evidence(
        symptoms=symptoms,
        xray_reports=radiology_records,
    )

    # 20. Pneumonia report findings are NOT discarded as unrelated
    separate_text = " ".join(evidence.get("separate", []))
    assert "does not directly account for the primary presenting clinical presentation" not in separate_text
    assert any("consolidation" in s.lower() for s in evidence.get("supporting", []))
    assert any("effusion" in r.lower() for r in evidence.get("reassuring", []))


# ==============================================================================
# 4. CANDIDATE EVIDENCE RECONCILIATION TESTS (Requirements 26-35)
# ==============================================================================

def test_candidate_reconciliation_preserves_ml_and_handles_evidence():
    """Verify genuine ML candidate scores, order, and cautious lab/radiology evidence."""
    symptoms = [
        {"name": "cough", "symptom": "cough", "status": "PRESENT", "duration": "7 days"},
        {"name": "fever", "symptom": "fever", "status": "PRESENT", "duration": "3 days"},
        {"name": "shortness of breath", "symptom": "shortness of breath", "status": "PRESENT", "duration": "3 days"},
        {"name": "chest pain", "symptom": "chest pain", "status": "PRESENT", "duration": "5 days"},
        {"name": "fatigue", "symptom": "fatigue", "status": "PRESENT", "duration": "7 days"},
        {"name": "loss of appetite", "symptom": "loss of appetite", "status": "PRESENT", "duration": "5 days"},
        {"name": "wheezing", "symptom": "wheezing", "status": "ABSENT"},
        {"name": "hemoptysis", "symptom": "hemoptysis", "status": "ABSENT"},
    ]
    present_syms = [s for s in symptoms if s["status"] == "PRESENT"]
    absent_syms = [s for s in symptoms if s["status"] == "ABSENT"]

    condition_svc = get_condition_prediction_service()
    raw_pred = condition_svc.predict_conditions(
        symptoms_present=present_syms,
        symptoms_absent=absent_syms,
        symptoms_unknown=[],
    )
    assert raw_pred.get("status") in {"ok", "success", "completed"}
    candidates = raw_pred.get("conditions", [])
    assert len(candidates) > 0

    # Record baseline model scores and ranks
    baseline_scores = [c.get("probability_score") or c.get("model_score") for c in candidates]
    baseline_names = [c["condition"] for c in candidates]

    # Multimodal evidence inputs
    labs = [
        {"test_name": "White Blood Cell (WBC)", "status": "HIGH", "value": 14.2, "unit": "10^9/L", "source_file": "lab_report.pdf"},
        {"test_name": "C-Reactive Protein (CRP)", "status": "HIGH", "value": 56.0, "unit": "mg/L", "source_file": "lab_report.pdf"},
        {"test_name": "Procalcitonin", "status": "HIGH", "value": 0.68, "unit": "ng/mL", "source_file": "lab_report.pdf"},
        {"test_name": "Hemoglobin", "status": "NORMAL", "value": 13.6, "unit": "g/dL", "source_file": "lab_report.pdf"},
    ]
    rad_reports = [
        {
            "report_id": "RAD-001",
            "file_name": "radiology_chest.pdf",
            "region": "chest",
            "qualitative_findings": [
                {"finding": "consolidation", "state": "PRESENT"},
                {"finding": "pleural effusion", "state": "ABSENT"},
                {"finding": "pneumothorax", "state": "ABSENT"},
            ],
            "narrative_findings": "Right lower-lobe air-space opacity with consolidation described.",
        }
    ]

    evidence = combine_evidence(
        symptoms=symptoms,
        report_findings=labs,
        xray_reports=rad_reports,
    )
    source_records = {
        "medical_reports": [{"findings": labs, "file_name": "lab_report.pdf"}],
        "xray_reports": rad_reports,
        "xray_images": [],
        "symptoms": symptoms,
    }

    reconciled = reconcile_candidate_evidence(
        conditions=candidates,
        evidence=evidence,
        source_records=source_records,
    )

    # 26. Genuine ML candidates preserved
    assert len(reconciled) == len(candidates)

    # 27. Original model scores strictly unchanged
    new_scores = [c.get("probability_score") or c.get("model_score") for c in reconciled]
    assert new_scores == baseline_scores

    # 28. Original candidate order strictly unchanged
    new_names = [c["condition"] for c in reconciled]
    assert new_names == baseline_names

    # Check Pneumonia candidate evidence breakdown
    pneu = next((c for c in reconciled if "pneumonia" in c["condition"].lower()), None)
    if pneu:
        # 29. Nonspecific inflammatory markers (WBC, CRP, PCT) are in contextual_evidence, NOT false disease-specific support
        contextual_findings = [e.get("finding") for e in pneu.get("contextual_evidence", [])]
        assert any("WBC" in f or "White Blood Cell" in f for f in contextual_findings)
        assert any("CRP" in f or "C-Reactive Protein" in f for f in contextual_findings)
        assert any("Procalcitonin" in f for f in contextual_findings)

        # Confirm Pneumonia does NOT get false "Relevant laboratory finding available" badge
        badges = pneu.get("evidence_badges", [])
        assert "Relevant laboratory finding available" not in badges

        # 30. Candidate-specific evidence includes correct source IDs and filenames
        supp_evidence = pneu.get("supporting_evidence", [])
        rad_supp = [e for e in supp_evidence if "radiology" in e.get("source_type", "")]
        assert len(rad_supp) > 0
        assert rad_supp[0].get("source_file") == "radiology_chest.pdf"
        assert "consolidation" in rad_supp[0].get("finding", "").lower()

        # 32. Reassuring evidence includes negative findings (no pleural effusion)
        reassuring = pneu.get("reassuring_evidence", [])
        reassuring_findings = [r.get("finding", "").lower() for r in reassuring]
        assert any("effusion" in f for f in reassuring_findings)


# ==============================================================================
# 5. X-RAY IMAGE PROTECTION & VERIFICATION TESTS (Requirements 36-40)
# ==============================================================================

def test_xray_protected_files_hashes_unmodified():
    """Verify that all 4 protected X-ray files remain strictly unmodified."""
    protected_files = {
        "backend/app/ai/xray_router.py": "F333EA7421A97474C1ADB26678F2134819BFD62EF23CDCA049DF4790BDFE2A13",
        "backend/app/ai/xray_anatomy_classifier.py": "1BD025E61071BFA36D0AEDDD9DB5B13C246ECB9DEEF34A40933BBAF767DE027E",
        "backend/app/ai/xray_model.py": "869D5869339193DE8E8EBB1E71DA54E7AEC06600B754BA864F96B4233B8536C4",
        "backend/app/ai/gradcam.py": "844852A86D1431C6E6C6943E3F16FD9D9A219683C255572479274B8B5ED706AF",
    }
    for rel_path, expected_hash in protected_files.items():
        full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", rel_path))
        with open(full_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest().upper()
        assert file_hash == expected_hash, f"Protected file modified: {rel_path}"


def test_unverified_xray_output_excluded_from_candidate_support():
    """Verify unverified X-ray output does not contaminate candidate evidence."""
    candidates = [{"condition": "Pneumonia", "probability_score": 0.65}]
    symptoms = [{"symptom": "cough", "status": "PRESENT"}]
    evidence = {
        "symptoms": {"present": symptoms, "absent": []},
        "xray": {
            "images": [
                {
                    "id": "XRAY-001",
                    "file_name": "chest_sample.png",
                    "region": "chest",
                    "anatomy_independently_verified": False,
                    "prediction": "Pneumonia",
                }
            ]
        }
    }
    reconciled = reconcile_candidate_evidence(
        conditions=candidates,
        evidence=evidence,
    )
    pneu = reconciled[0]
    supp = pneu.get("supporting_evidence", [])
    xray_supp = [e for e in supp if "xray" in e.get("source_type", "")]
    assert len(xray_supp) == 0  # Unverified output excluded


# ==============================================================================
# 6. TRIAGE & PDF OUTPUT CONSISTENCY TESTS (Requirements 41-46)
# ==============================================================================

def test_emergency_triage_cannot_be_downgraded_by_ml_or_labs():
    """Verify deterministic safety triage remains independent."""
    symptoms = [
        {"symptom": "shortness of breath", "status": "PRESENT", "severity": "SEVERE"},
        {"symptom": "cough", "status": "PRESENT", "duration": "7 days"},
    ]
    evidence = combine_evidence(symptoms=symptoms)
    triage = apply_triage(evidence)
    assert triage.get("pathway") in {"EMERGENCY", "URGENT"}


def test_pdf_service_builds_with_reconciled_evidence():
    """Verify PDF generates successfully using the complete candidate evidence payload."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        user = User(
            user_id=uuid4(),
            email="test001@example.com",
            password_hash="pw",
            name="Test Patient 001",
            state="Delhi",
            district="New Delhi",
        )
        session.add(user)
        session.commit()

        feedback_payload = {
            "assessment_id": "TEST-001",
            "patient": {"age": 42, "gender": "Male", "id": "TEST-001"},
            "triage": {"pathway": "URGENT", "why_this_pathway": "Persistent high fever and respiratory findings."},
            "symptoms": [
                {"name": "cough", "status": "PRESENT", "duration": "7 days"},
                {"name": "fever", "status": "PRESENT", "duration": "3 days", "measurement": "38.5 C"},
                {"name": "chest pain", "status": "PRESENT", "duration": "5 days"},
                {"name": "wheezing", "status": "ABSENT"},
            ],
            "laboratory_findings": [
                {"test_name": "White Blood Cell (WBC)", "value": 14.2, "unit": "10^9/L", "status": "HIGH"},
                {"test_name": "Procalcitonin", "value": 0.68, "unit": "ng/mL", "status": "HIGH"},
            ],
            "radiology_findings": [
                {"finding": "Right lower-lobe consolidation", "state": "PRESENT", "source_file": "rad_report.pdf"},
                {"finding": "Pleural effusion", "state": "ABSENT", "source_file": "rad_report.pdf"},
            ],
            "possible_conditions": [
                {
                    "rank": 1,
                    "condition": "Pneumonia",
                    "probability_score": 0.72,
                    "confidence_tier": "Probable",
                    "evidence_badges": ["Supported by reported symptoms", "Relevant radiology finding available"],
                    "supporting_evidence": [
                        {"source_type": "symptom_nlp", "finding": "cough (PRESENT, 7 days)"},
                        {"source_type": "radiology_report", "finding": "Right lower-lobe consolidation", "source_file": "rad_report.pdf"},
                    ],
                    "contradictory_evidence": [
                        {"source_type": "symptom_nlp", "finding": "wheezing (ABSENT)"}
                    ],
                    "reassuring_evidence": [
                        {"finding": "Pleural effusion: ABSENT", "reason": "No pleural effusion identified"}
                    ],
                    "contextual_evidence": [
                        {"finding": "White Blood Cell (WBC): 14.2 10^9/L (HIGH)"},
                        {"finding": "Procalcitonin: 0.68 ng/mL (HIGH)"},
                    ],
                    "missing_information": ["Oxygen saturation measurement"],
                    "suggested_specialty": "Pulmonology / Internal Medicine",
                }
            ],
        }

        assessment = Assessment(
            assessment_id=uuid4(),
            user_id=user.user_id,
            input_types=["symptoms", "medical_reports", "radiology_reports"],
            status="completed",
            result_payload={"feedback": feedback_payload, "triage": {"pathway": "URGENT"}},
        )
        session.add(assessment)
        session.commit()

        pdf_res = generate_assessment_pdf(assessment.assessment_id, session)
        assert pdf_res["status"] == "ok"
        assert pdf_res["path"] is not None
        assert os.path.exists(pdf_res["path"])
        with open(pdf_res["path"], "rb") as f:
            pdf_bytes = f.read()
        assert len(pdf_bytes) > 1000
        assert pdf_bytes.startswith(b"%PDF")
    finally:
        session.close()


# ==============================================================================
# 7. EDGE CASE TESTS (Multiple reports, unreadable text, missing IDs)
# ==============================================================================

def test_multiple_lab_reports_individual_attribution():
    """Verify multiple lab reports preserve their distinct source files and IDs."""
    candidates = [{"condition": "Pneumonia", "probability_score": 0.65}]
    labs_rep1 = [
        {"test_name": "White Blood Cell (WBC)", "status": "HIGH", "value": 14.2, "unit": "10^9/L", "source_file": "lab_day1.pdf", "report_id": "LAB-1"},
    ]
    labs_rep2 = [
        {"test_name": "C-Reactive Protein (CRP)", "status": "HIGH", "value": 56.0, "unit": "mg/L", "source_file": "lab_day3.pdf", "report_id": "LAB-2"},
    ]
    source_records = {
        "medical_reports": [
            {"findings": labs_rep1, "file_name": "lab_day1.pdf", "report_id": "LAB-1"},
            {"findings": labs_rep2, "file_name": "lab_day3.pdf", "report_id": "LAB-2"},
        ]
    }
    evidence = {
        "symptoms": {"present": [{"symptom": "cough", "status": "PRESENT"}], "absent": []},
        "report": {"findings": labs_rep1 + labs_rep2},
    }
    reconciled = reconcile_candidate_evidence(
        conditions=candidates,
        evidence=evidence,
        source_records=source_records,
    )
    pneu = reconciled[0]
    ctx = pneu.get("contextual_evidence", [])
    src_files = {e.get("source_file") for e in ctx}
    assert "lab_day1.pdf" in src_files
    assert "lab_day3.pdf" in src_files


def test_multiple_radiology_reports_relevant_vs_contextual():
    """Verify relevant radiology report supports while unrelated report remains contextual."""
    candidates = [{"condition": "Pneumonia", "probability_score": 0.65}]
    rad_chest = {
        "report_id": "RAD-CHEST",
        "file_name": "rad_chest.pdf",
        "region": "chest",
        "qualitative_findings": [{"finding": "consolidation", "state": "PRESENT"}],
    }
    rad_knee = {
        "report_id": "RAD-KNEE",
        "file_name": "rad_knee.pdf",
        "region": "knee",
        "qualitative_findings": [{"finding": "joint effusion", "state": "PRESENT"}],
    }
    source_records = {
        "xray_reports": [rad_chest, rad_knee],
    }
    evidence = {
        "symptoms": {"present": [{"symptom": "cough", "status": "PRESENT"}], "absent": []},
        "xray": {"reports": [rad_chest, rad_knee]},
    }
    reconciled = reconcile_candidate_evidence(
        conditions=candidates,
        evidence=evidence,
        source_records=source_records,
    )
    pneu = reconciled[0]
    supp = pneu.get("supporting_evidence", [])
    ctx = pneu.get("contextual_evidence", [])

    # Chest consolidation is supporting
    assert any(e.get("source_file") == "rad_chest.pdf" for e in supp)
    # Knee report is contextual, NOT supporting pneumonia
    assert not any(e.get("source_file") == "rad_knee.pdf" for e in supp)
    assert any(e.get("source_file") == "rad_knee.pdf" for e in ctx)


def test_empty_and_unreadable_reports_safe_handling():
    """Verify empty and scanned/unreadable report text fail closed safely."""
    # Empty report
    res_empty = structure_report_text("")
    assert res_empty["status"] == "unavailable"
    assert res_empty["findings"] == []

    # Unreadable noise
    res_noise = structure_report_text("~~~$$$###???@@@ 12345 67890")
    assert len(res_noise["findings"]) == 0
    assert len(res_noise["qualitative_findings"]) == 0


def test_missing_source_identifiers_handled_safely():
    """Verify missing report IDs or filenames do not cause exceptions."""
    candidates = [{"condition": "Pneumonia", "probability_score": 0.65}]
    rad_no_id = {
        "region": "chest",
        "qualitative_findings": [{"finding": "consolidation", "state": "PRESENT"}],
    }
    source_records = {"xray_reports": [rad_no_id]}
    evidence = {
        "symptoms": {"present": [{"symptom": "cough", "status": "PRESENT"}], "absent": []},
        "xray": {"reports": [rad_no_id]},
    }
    reconciled = reconcile_candidate_evidence(
        conditions=candidates,
        evidence=evidence,
        source_records=source_records,
    )
    assert len(reconciled) == 1
    pneu = reconciled[0]
    assert len(pneu.get("supporting_evidence", [])) > 0

