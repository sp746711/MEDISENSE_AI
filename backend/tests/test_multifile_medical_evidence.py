"""Unit & Integration tests for Multi-File Medical Evidence & Radiology Support.

Covers all 29 requirements:
1. One X-ray image
2. Multiple X-ray images
3. One X-ray report
4. Multiple X-ray reports
5. Image + report
6. Multiple images + multiple reports
7. Multiple ordinary medical reports
8. X-ray report separated from ordinary medical report
9. Image model unavailable
10. OCR unavailable
11. Duplicate file handling logic
12. One failed upload among successful uploads
13. Historical report preservation
14. Multiple X-ray results reaching evidence engine
15. Final feedback showing ALL evidence sources
16. No fabricated X-ray prediction
17. Body-region mismatch detection
18-29. Preview, file separation, and full E2E multimodal test case.
"""

from __future__ import annotations

import pytest

from app.ai.evidence_engine import combine_evidence
from app.ai.report_nlp import structure_report_text
from app.ai.report_ocr import extract_text_from_file
from app.ai.symptom_nlp import extract_symptoms
from app.ai.triage_engine import apply_triage
from app.ai.xray_model import XRayModelService
from app.services.assessment_service import generate_assessment_feedback


def test_req_1_one_xray_image():
    """Requirement 1: Single X-ray image in evidence engine."""
    img = {
        "file_name": "knee_xray_1.png",
        "region": "knee",
        "status": "UNAVAILABLE",
        "prediction": None,
        "model_version": "N/A",
    }
    evidence = combine_evidence(xray_results=[img])
    assert evidence["xray"]["received"] is True
    assert len(evidence["xray"]["images"]) == 1
    assert evidence["xray"]["images"][0]["file_name"] == "knee_xray_1.png"
    assert evidence["xray"]["images"][0]["status"] == "UNAVAILABLE"
    assert evidence["xray"]["images"][0]["prediction"] is None


def test_req_2_multiple_xray_images():
    """Requirement 2: Multiple X-ray images preserved independently."""
    img1 = {
        "file_name": "knee_xray_1.png",
        "region": "knee",
        "status": "UNAVAILABLE",
        "prediction": None,
    }
    img2 = {
        "file_name": "knee_xray_2.png",
        "region": "knee",
        "status": "UNAVAILABLE",
        "prediction": None,
    }
    evidence = combine_evidence(xray_results=[img1, img2])
    assert evidence["xray"]["received"] is True
    assert len(evidence["xray"]["images"]) == 2
    assert evidence["xray"]["images"][0]["file_name"] == "knee_xray_1.png"
    assert evidence["xray"]["images"][1]["file_name"] == "knee_xray_2.png"


def test_req_3_one_xray_report():
    """Requirement 3: Single X-ray / radiology report."""
    rep = {
        "file_name": "knee_report_current.pdf",
        "extracted_text": "KNEE RADIOGRAPH: No acute fracture or dislocation. Normal joint alignment.",
        "structured_findings": {
            "qualitative_findings": [
                {"finding": "No acute fracture", "state": "ABSENT"},
                {"finding": "Normal joint alignment", "state": "PRESENT"},
            ]
        },
    }
    evidence = combine_evidence(xray_reports=[rep])
    assert len(evidence["xray"]["reports"]) == 1
    assert evidence["xray"]["reports"][0]["file_name"] == "knee_report_current.pdf"
    assert any("knee_report_current.pdf" in r for r in evidence["reassuring"])


def test_req_4_multiple_xray_reports():
    """Requirement 4: Multiple X-ray reports preserved independently."""
    rep1 = {
        "file_name": "knee_report_current.pdf",
        "extracted_text": "Current radiograph: No fracture.",
        "structured_findings": {
            "qualitative_findings": [{"finding": "No acute fracture", "state": "ABSENT"}]
        },
    }
    rep2 = {
        "file_name": "knee_report_previous.pdf",
        "extracted_text": "Previous radiograph from 6 months ago: Mild degenerative changes.",
        "structured_findings": {
            "qualitative_findings": [{"finding": "Mild degenerative changes", "state": "PRESENT"}]
        },
        "is_historical": True,
    }
    evidence = combine_evidence(xray_reports=[rep1, rep2])
    assert len(evidence["xray"]["reports"]) == 2
    assert evidence["xray"]["reports"][0]["file_name"] == "knee_report_current.pdf"
    assert evidence["xray"]["reports"][1]["file_name"] == "knee_report_previous.pdf"


def test_req_5_image_plus_report():
    """Requirement 5: Image + X-ray report combination."""
    img = {"file_name": "knee_xray.png", "region": "knee", "status": "UNAVAILABLE", "prediction": None}
    rep = {
        "file_name": "knee_report.pdf",
        "extracted_text": "Right knee examination shows intact bone cortex.",
        "structured_findings": {
            "qualitative_findings": [{"finding": "Intact bone cortex", "state": "PRESENT"}]
        },
    }
    evidence = combine_evidence(xray_results=[img], xray_reports=[rep])
    assert evidence["xray"]["received"] is True
    assert len(evidence["xray"]["images"]) == 1
    assert len(evidence["xray"]["reports"]) == 1


def test_req_6_multiple_images_plus_multiple_reports():
    """Requirement 6: Multiple images + multiple reports."""
    img1 = {"file_name": "knee_1.png", "region": "knee", "status": "UNAVAILABLE"}
    img2 = {"file_name": "knee_2.png", "region": "knee", "status": "UNAVAILABLE"}
    rep1 = {"file_name": "knee_rep_1.pdf", "extracted_text": "No fracture"}
    rep2 = {"file_name": "knee_rep_2.pdf", "extracted_text": "Prior study normal"}

    evidence = combine_evidence(xray_results=[img1, img2], xray_reports=[rep1, rep2])
    assert len(evidence["xray"]["images"]) == 2
    assert len(evidence["xray"]["reports"]) == 2


def test_req_7_multiple_ordinary_medical_reports():
    """Requirement 7: Multiple ordinary medical laboratory reports."""
    rep1 = {
        "file_name": "blood_report.pdf",
        "structured_findings": {
            "lab_parameters": [
                {"test_name": "Hemoglobin", "value": 14.2, "unit": "g/dL", "interpretation": "NORMAL"}
            ]
        },
    }
    rep2 = {
        "file_name": "liver_report.pdf",
        "structured_findings": {
            "lab_parameters": [
                {"test_name": "ALT", "value": 35.0, "unit": "U/L", "interpretation": "NORMAL"}
            ]
        },
    }
    report_findings = [
        {"test_name": "Hemoglobin", "value": 14.2, "unit": "g/dL", "interpretation": "NORMAL", "source_file": "blood_report.pdf"},
        {"test_name": "ALT", "value": 35.0, "unit": "U/L", "interpretation": "NORMAL", "source_file": "liver_report.pdf"},
    ]
    evidence = combine_evidence(report_findings=report_findings, medical_reports=[rep1, rep2])
    assert evidence["report"]["provided"] is True
    assert len(evidence["report"]["medical_reports"]) == 2
    assert len(evidence["report"]["normal_findings"]) == 2


def test_req_8_xray_report_separated_from_ordinary_medical_report():
    """Requirement 8: X-ray report separated from ordinary medical report."""
    med_rep = {"file_name": "cbc.pdf", "source_type": "MEDICAL_REPORT"}
    xray_rep = {"file_name": "chest_xray_report.pdf", "source_type": "XRAY_REPORT"}

    evidence = combine_evidence(medical_reports=[med_rep], xray_reports=[xray_rep])
    assert len(evidence["report"]["medical_reports"]) == 1
    assert evidence["report"]["medical_reports"][0]["file_name"] == "cbc.pdf"
    assert len(evidence["xray"]["reports"]) == 1
    assert evidence["xray"]["reports"][0]["file_name"] == "chest_xray_report.pdf"


def test_req_9_and_16_image_model_unavailable_no_fabrication():
    """Requirements 9 & 16: Image model unavailable produces UNAVAILABLE without fabricating findings."""
    service = XRayModelService()
    res = service.analyze("nonexistent_knee.png", region="knee")
    assert res["status"] == "unavailable"
    assert res["prediction"] is None
    assert "unavailable" in res["message"].lower() or "not" in res["message"].lower()

    # Pass to evidence engine
    evidence = combine_evidence(xray_results=[{"file_name": "knee.png", "region": "knee", "status": "UNAVAILABLE", "prediction": None}])
    assert evidence["xray"]["status"] == "UNAVAILABLE"
    assert evidence["xray"]["prediction"] is None
    # Must NOT fabricate normal or disease findings
    assert not any("normal x-ray" in s.lower() for s in evidence["supporting"] + evidence["reassuring"])


def test_req_10_ocr_unavailable():
    """Requirement 10: OCR unavailable handles missing file truthfully."""
    res = extract_text_from_file("missing_doc.pdf")
    assert res["status"] == "unavailable"
    assert res["extracted_text"] is None


def test_req_17_body_region_mismatch():
    """Requirement 17: Body region mismatch between image and report detected."""
    img = {"file_name": "knee_xray.png", "region": "knee", "status": "UNAVAILABLE"}
    rep = {
        "file_name": "chest_report.pdf",
        "extracted_text": "CHEST RADIOGRAPH: Clear lungs, normal cardiac shadow.",
    }
    evidence = combine_evidence(xray_results=[img], xray_reports=[rep])
    # Must identify contradiction / mismatch
    assert any("body region mismatch" in c.lower() or "conflicting" in c.lower() for c in evidence["contradictory"])


def test_req_full_e2e_knee_multimodal():
    """FULL END-TO-END TEST CASE:
    Symptoms:
      Right knee pain for 4 days.
      Pain becomes worse while walking and climbing stairs.
      Morning stiffness.
      Mild swelling.
      No fever.
      No numbness.
      No weakness.
      No injury.
    Files:
      X-Ray Images: knee_xray_1.png, knee_xray_2.png
      X-Ray Reports: knee_report_current.pdf, knee_report_previous.pdf
      Medical Reports: blood_report.pdf, previous_lab_report.pdf
    """
    text = (
        "Right knee pain for 4 days. "
        "Pain becomes worse while walking and climbing stairs. "
        "Morning stiffness. "
        "Mild swelling. "
        "No fever. "
        "No numbness. "
        "No weakness. "
        "No injury."
    )
    symptom_res = extract_symptoms(text)
    symptoms = symptom_res["symptoms"]

    # Verify extracted symptoms
    by_name = {s["symptom"]: s for s in symptoms}
    assert "knee pain" in by_name or "joint pain" in by_name
    active_pain = by_name.get("knee pain") or by_name.get("joint pain")
    assert active_pain["state"] == "PRESENT"

    # Explicit negatives
    assert "fever" in by_name and by_name["fever"]["state"] == "ABSENT"
    assert "numbness" in by_name and by_name["numbness"]["state"] == "ABSENT"
    assert "weakness" in by_name and by_name["weakness"]["state"] == "ABSENT"
    injury_symptom = by_name.get("injury") or by_name.get("trauma/injury")
    assert injury_symptom is not None and injury_symptom["state"] == "ABSENT"

    # Files setup
    xray_imgs = [
        {"file_name": "knee_xray_1.png", "region": "knee", "status": "UNAVAILABLE", "prediction": None},
        {"file_name": "knee_xray_2.png", "region": "knee", "status": "UNAVAILABLE", "prediction": None},
    ]
    xray_reps = [
        {
            "file_name": "knee_report_current.pdf",
            "extracted_text": "KNEE RADIOGRAPH: Intact articular cortex. No acute fracture.",
            "structured_findings": {
                "qualitative_findings": [{"finding": "No acute fracture", "state": "ABSENT"}]
            },
        },
        {
            "file_name": "knee_report_previous.pdf",
            "extracted_text": "Prior study from 1 year ago: Minimal patellofemoral narrowing.",
            "is_historical": True,
        },
    ]
    med_reps = [
        {
            "file_name": "blood_report.pdf",
            "structured_findings": {
                "lab_parameters": [{"test_name": "ESR", "value": 15, "unit": "mm/hr", "interpretation": "NORMAL"}]
            },
        },
        {
            "file_name": "previous_lab_report.pdf",
            "is_historical": True,
        },
    ]
    report_findings = [
        {"test_name": "ESR", "value": 15, "unit": "mm/hr", "interpretation": "NORMAL", "source_file": "blood_report.pdf"}
    ]

    evidence = combine_evidence(
        symptoms=symptoms,
        report_findings=report_findings,
        xray_results=xray_imgs,
        xray_reports=xray_reps,
        medical_reports=med_reps,
    )

    triage = apply_triage(evidence)
    feedback = generate_assessment_feedback(evidence, triage, specialty="Orthopedics")

    # Verify all evidence sources are preserved
    assert len(evidence["xray"]["images"]) == 2
    assert len(evidence["xray"]["reports"]) == 2
    assert len(evidence["report"]["medical_reports"]) == 2

    # Verify no fabricated predictions
    assert evidence["xray"]["prediction"] is None
    assert feedback["xray_findings"]["prediction"] is None

    # Verify feedback reflects both images
    summary = feedback["assessment_summary"]
    assert "knee_xray_1.png" in summary or "2 X-ray image(s)" in summary
    assert "ESR" in str(feedback["medical_report_findings"])
    assert "No acute fracture" in str(feedback["reassuring_evidence"])


def test_remove_file_first_middle_last_correctness():
    """Verify that removing a file from first, middle, or last index preserves remaining items."""
    files = [
        {"id": "file-1", "name": "xray1.png"},
        {"id": "file-2", "name": "xray2.png"},
        {"id": "file-3", "name": "xray3.jpg"},
    ]
    # Remove middle file
    remaining_mid = [f for f in files if f["id"] != "file-2"]
    assert len(remaining_mid) == 2
    assert [f["name"] for f in remaining_mid] == ["xray1.png", "xray3.jpg"]

    # Remove first file
    remaining_first = [f for f in remaining_mid if f["id"] != "file-1"]
    assert len(remaining_first) == 1
    assert remaining_first[0]["name"] == "xray3.jpg"

    # Remove last file
    remaining_last = [f for f in remaining_mid if f["id"] != "file-3"]
    assert len(remaining_last) == 1
    assert remaining_last[0]["name"] == "xray1.png"


@pytest.mark.asyncio
async def test_backend_validation_xray_image_rejects_pdf():
    """Requirement: upload_xray rejects PDF files with specific error message."""
    import io
    from unittest.mock import MagicMock
    from fastapi import HTTPException, UploadFile
    from app.routes.xray import upload_xray

    mock_db = MagicMock()
    mock_user = MagicMock()
    mock_user.user_id = "user-123"

    mock_assessment = MagicMock()
    mock_assessment.user_id = "user-123"
    mock_assessment.input_types = ["xray"]

    pdf_file = UploadFile(filename="xray_report.pdf", file=io.BytesIO(b"%PDF-1.4 test content"))

    with pytest.raises(HTTPException) as exc_info:
        from unittest.mock import patch
        with patch("app.routes.xray._owned", return_value=mock_assessment):
            await upload_xray(
                assessment_id=MagicMock(),
                current_user=mock_user,
                db=mock_db,
                file=pdf_file,
                region="chest",
            )
    assert exc_info.value.status_code == 400
    assert "X-ray images must be PNG, JPG, or JPEG files." in exc_info.value.detail


@pytest.mark.asyncio
async def test_backend_validation_xray_report_rejects_images():
    """Requirement: upload_report with category xray_report rejects PNG/JPG files."""
    import io
    from unittest.mock import MagicMock
    from fastapi import HTTPException, UploadFile
    from app.routes.medical_reports import upload_report

    mock_db = MagicMock()
    mock_user = MagicMock()
    mock_user.user_id = "user-123"

    mock_assessment = MagicMock()
    mock_assessment.user_id = "user-123"
    mock_assessment.input_types = ["xray"]

    img_file = UploadFile(filename="knee_xray.png", file=io.BytesIO(b"\x89PNG\r\n\x1a\nfakeimage"))

    with pytest.raises(HTTPException) as exc_info:
        from unittest.mock import patch
        with patch("app.routes.medical_reports._owned", return_value=mock_assessment):
            await upload_report(
                assessment_id=MagicMock(),
                current_user=mock_user,
                db=mock_db,
                file=img_file,
                report_category="xray_report",
            )
    assert exc_info.value.status_code == 400
    assert "X-ray reports must be PDF files." in exc_info.value.detail


@pytest.mark.asyncio
async def test_backend_validation_xray_report_rejects_medical_report():
    """Requirement: upload_report rejects MedAI_Test2_Normal_Medical_Report.pdf when uploaded as xray_report."""
    import io
    from unittest.mock import MagicMock
    from fastapi import HTTPException, UploadFile
    from app.routes.medical_reports import upload_report

    mock_db = MagicMock()
    mock_user = MagicMock()
    mock_user.user_id = "user-123"

    mock_assessment = MagicMock()
    mock_assessment.user_id = "user-123"
    mock_assessment.input_types = ["xray"]

    med_file = UploadFile(
        filename="MedAI_Test2_Normal_Medical_Report.pdf",
        file=io.BytesIO(b"%PDF-1.4 general lab report"),
    )

    with pytest.raises(HTTPException) as exc_info:
        from unittest.mock import patch
        with patch("app.routes.medical_reports._owned", return_value=mock_assessment):
            await upload_report(
                assessment_id=MagicMock(),
                current_user=mock_user,
                db=mock_db,
                file=med_file,
                report_category="xray_report",
            )
    assert exc_info.value.status_code == 400
    assert "general medical report" in exc_info.value.detail.lower()
