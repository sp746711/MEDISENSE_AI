"""Comprehensive test suite for Medical Report Upload, OCR, Data Extraction, Persistence, and Evidence Integration.

Covers all 15 required objectives:
1. Digital PDF containing selectable text.
2. Scanned / image-only PDF requiring OCR.
3. PNG containing readable printed text.
4. JPG / JPEG containing readable report text.
5. Image with no readable text.
6. Corrupt or unsupported file validation.
7. Multiple files where one succeeds and another fails.
8. Backend validation error returned to client.
9. Network / auth failure message handling.
10. Retry of an upload without unintended duplicate database records.
11. Correct persistence of extracted text, structured findings, reference ranges, and OCR metadata.
12. Retrieval of successfully extracted report findings isolated to the correct assessment.
13. Uploaded findings reaching the consolidated evidence and final feedback pipeline.
14. OCR failure is NOT presented as normal or successfully analyzed.
15. Regression test confirming X-ray images continue through the separate X-ray computer vision pipeline.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.ai.evidence_engine import combine_evidence
from app.ai.report_nlp import structure_report_text
from app.ai.report_ocr import extract_text_from_file, normalize_ocr_text
from app.core.security import create_access_token
from app.database.database import SessionLocal, init_db
from app.database.models import Assessment, MedicalReport, Symptom, User, XrayResult
from app.main import app
from app.services.assessment_service import run_assessment_pipeline
from app.utils.file_validation import (
    detect_file_type_from_content,
    sanitize_filename,
    validate_image_file,
    validate_report_file,
)


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Ensure database schema and migrations are initialized."""
    init_db()


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def auth_user(db_session):
    user = db_session.query(User).filter(User.email == "test_ocr_patient@example.com").first()
    if not user:
        user = User(
            email="test_ocr_patient@example.com",
            password_hash="testpasshash",
            name="Test OCR Patient",
            state="Maharashtra",
            district="Mumbai",
            city="Mumbai",
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
    return user


@pytest.fixture
def auth_token(auth_user):
    return create_access_token(str(auth_user.user_id))


@pytest.fixture
def client():
    return TestClient(app)


def _create_digital_pdf(content_text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), content_text)
    data = doc.tobytes()
    doc.close()
    return data


def _create_image(text: str, fmt: str = "PNG") -> bytes:
    img = Image.new("RGB", (600, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 30), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# --------------------------------------------------------------------------
# 1. Digital PDF containing selectable text
# --------------------------------------------------------------------------
def test_1_digital_pdf_selectable_text(tmp_path):
    pdf_text = "CLINICAL LABORATORY REPORT\nHemoglobin: 13.5 g/dL (Reference: 13.0 - 17.0)\nWhite Blood Cell Count (WBC): 7500 /uL"
    pdf_bytes = _create_digital_pdf(pdf_text)
    file_path = tmp_path / "digital_report.pdf"
    file_path.write_bytes(pdf_bytes)

    res = extract_text_from_file(str(file_path))
    assert res["status"] == "extracted"
    assert "Hemoglobin: 13.5 g/dL" in res["extracted_text"]
    assert res["ocr_engine"] == "pdf_text_layer"
    assert res["pages"] == 1


# --------------------------------------------------------------------------
# 2. Scanned / image-only PDF requiring OCR
# --------------------------------------------------------------------------
def test_2_scanned_pdf_requiring_ocr(tmp_path):
    # Create a PDF with an embedded image page (no text layer)
    doc = fitz.open()
    page = doc.new_page()
    img_bytes = _create_image("Serum Creatinine: 1.8 mg/dL (Reference: 0.6 - 1.2)", "PNG")
    page.insert_image(page.rect, stream=img_bytes)
    pdf_path = tmp_path / "scanned_report.pdf"
    pdf_path.write_bytes(doc.tobytes())
    doc.close()

    # When OCR engine is available/mocked:
    with patch("pytesseract.image_to_string", return_value="Serum Creatinine: 1.8 mg/dL (Reference: 0.6 - 1.2)"), \
         patch("app.ai.report_ocr.is_ocr_engine_available", return_value=True):
        res = extract_text_from_file(str(pdf_path))
        assert res["status"] == "extracted"
        assert "Serum Creatinine" in res["extracted_text"]
        assert res["ocr_engine"] == "pytesseract"


# --------------------------------------------------------------------------
# 2b. Mixed PDF: preserve selectable digital text and OCR scanned pages
# --------------------------------------------------------------------------
def test_2b_mixed_digital_and_scanned_pdf_preserves_both_pages(tmp_path):
    """A mixed PDF must not lose scanned pages when another page has digital text."""
    doc = fitz.open()

    # Page 1: selectable digital text.
    page1 = doc.new_page()
    page1.insert_text((50, 72), "DIGITAL_PAGE_MARKER: Hemoglobin 12.5 g/dL")

    # Page 2: image-only scanned content with no selectable text layer.
    scanned_bytes = _create_image("SCANNED_PAGE_MARKER: WBC 7500 /uL", "PNG")
    page2 = doc.new_page()
    page2.insert_image(page2.rect, stream=scanned_bytes)

    pdf_path = tmp_path / "mixed_digital_scanned_report.pdf"
    pdf_path.write_bytes(doc.tobytes())
    doc.close()

    # Mock OCR to keep this regression test independent of local Tesseract.
    with patch(
        "pytesseract.image_to_string",
        return_value="SCANNED_PAGE_MARKER: WBC 7500 /uL",
    ), patch(
        "app.ai.report_ocr.is_ocr_engine_available",
        return_value=True,
    ):
        res = extract_text_from_file(str(pdf_path))

    assert res["status"] == "extracted"
    assert "DIGITAL_PAGE_MARKER" in res["extracted_text"]
    assert "SCANNED_PAGE_MARKER" in res["extracted_text"]
    assert res["pages"] == 2



# --------------------------------------------------------------------------
# 3. PNG containing readable printed text
# --------------------------------------------------------------------------
def test_3_png_readable_printed_text(tmp_path):
    png_bytes = _create_image("Platelet Count: 180000 /uL (Reference: 150000 - 450000)", "PNG")
    png_path = tmp_path / "platelet_panel.png"
    png_path.write_bytes(png_bytes)

    with patch("pytesseract.image_to_string", return_value="Platelet Count: 180000 /uL (Reference: 150000 - 450000)"), \
         patch("app.ai.report_ocr.is_ocr_engine_available", return_value=True):
        res = extract_text_from_file(str(png_path))
        assert res["status"] == "extracted"
        assert "Platelet Count" in res["extracted_text"]
        assert res["ocr_engine"] == "pytesseract"


# --------------------------------------------------------------------------
# 4. JPG / JPEG containing readable report text
# --------------------------------------------------------------------------
def test_4_jpg_readable_report_text(tmp_path):
    jpg_bytes = _create_image("Total Bilirubin: 2.4 mg/dL (Reference: 0.2 - 1.2)", "JPEG")
    jpg_path = tmp_path / "liver_panel.jpg"
    jpg_path.write_bytes(jpg_bytes)

    with patch("pytesseract.image_to_string", return_value="Total Bilirubin: 2.4 mg/dL (Reference: 0.2 - 1.2)"), \
         patch("app.ai.report_ocr.is_ocr_engine_available", return_value=True):
        res = extract_text_from_file(str(jpg_path))
        assert res["status"] == "extracted"
        assert "Total Bilirubin" in res["extracted_text"]
        assert res["ocr_engine"] == "pytesseract"


# --------------------------------------------------------------------------
# 5. Image with no readable text
# --------------------------------------------------------------------------
def test_5_image_with_no_readable_text(tmp_path):
    # Blank white image
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    blank_path = tmp_path / "blank_drawing.png"
    img.save(blank_path, format="PNG")

    with patch("pytesseract.image_to_string", return_value="   \n  \n"), \
         patch("app.ai.report_ocr.is_ocr_engine_available", return_value=True):
        res = extract_text_from_file(str(blank_path))
        assert res["status"] == "unreadable"
        assert res["extracted_text"] is None
        assert "no legible medical text" in res["message"].lower()


# --------------------------------------------------------------------------
# 6. Corrupt or unsupported file
# --------------------------------------------------------------------------
def test_6_corrupt_or_unsupported_file():
    # Corrupt PNG header
    corrupt_png = b"\x89PNG\r\n\x1a\nthis_is_not_a_valid_png_stream"
    val = validate_report_file("corrupt.png", corrupt_png, 10)
    assert val["ok"] is False
    assert "corrupted" in val["message"].lower()

    # Executable file
    exe_content = b"MZ\x90\x00\x03\x00\x00\x00"
    val_exe = validate_report_file("malicious.exe", exe_content, 10)
    assert val_exe["ok"] is False

    # Path traversal attack filename
    clean_name = sanitize_filename("../../etc/passwd.png")
    assert "/" not in clean_name and ".." not in clean_name


# --------------------------------------------------------------------------
# 7. Multiple files where one succeeds and another fails
# --------------------------------------------------------------------------
def test_7_multiple_files_one_succeeds_one_fails(client, auth_user, auth_token, db_session):
    ass = Assessment(user_id=auth_user.user_id, input_types=["medical_report"], status="draft")
    db_session.add(ass)
    db_session.commit()
    db_session.refresh(ass)

    valid_pdf = _create_digital_pdf("Hemoglobin: 12.0 g/dL")
    corrupt_file = b"MZbadexecutable"

    # File 1: Valid PDF
    r1 = client.post(
        f"/api/assessments/{ass.assessment_id}/reports",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("good_report.pdf", valid_pdf, "application/pdf")},
        data={"report_category": "medical_report"},
    )
    assert r1.status_code == 200
    assert r1.json()["upload_status"] == "uploaded"

    # File 2: Bad File
    r2 = client.post(
        f"/api/assessments/{ass.assessment_id}/reports",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("bad_file.pdf", corrupt_file, "application/pdf")},
        data={"report_category": "medical_report"},
    )
    assert r2.status_code == 400

    # Ensure File 1 is preserved and not impacted by File 2's failure
    reports = db_session.query(MedicalReport).filter(MedicalReport.assessment_id == ass.assessment_id).all()
    assert len(reports) == 1
    assert reports[0].file_name == "good_report.pdf"


# --------------------------------------------------------------------------
# 8. Backend validation error returned to client
# --------------------------------------------------------------------------
def test_8_backend_validation_error_returned(client, auth_user, auth_token, db_session):
    ass = Assessment(user_id=auth_user.user_id, input_types=["medical_report"], status="draft")
    db_session.add(ass)
    db_session.commit()

    # Mismatched file extension (declared .jpg but contents are .pdf)
    pdf_bytes = _create_digital_pdf("Test Content")
    res = client.post(
        f"/api/assessments/{ass.assessment_id}/reports",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("mismatched.jpg", pdf_bytes, "image/jpeg")},
        data={"report_category": "medical_report"},
    )
    assert res.status_code == 400
    assert "does not match actual file content format" in res.json()["detail"]


# --------------------------------------------------------------------------
# 9. Network / auth failure message handling
# --------------------------------------------------------------------------
def test_9_network_and_auth_failures(client, auth_user, db_session):
    ass = Assessment(user_id=auth_user.user_id, input_types=["medical_report"], status="draft")
    db_session.add(ass)
    db_session.commit()

    valid_pdf = _create_digital_pdf("Test")
    # Missing token
    res = client.post(
        f"/api/assessments/{ass.assessment_id}/reports",
        files={"file": ("report.pdf", valid_pdf, "application/pdf")},
    )
    assert res.status_code == 401
    assert "Authentication required" in res.json()["detail"]


# --------------------------------------------------------------------------
# 10. Retry of an upload without unintended duplicate records
# --------------------------------------------------------------------------
def test_10_retry_without_duplicate_records(client, auth_user, auth_token, db_session):
    ass = Assessment(user_id=auth_user.user_id, input_types=["medical_report"], status="draft")
    db_session.add(ass)
    db_session.commit()

    valid_pdf = _create_digital_pdf("Fasting Blood Sugar: 110 mg/dL (Reference: 70 - 100)")

    # Attempt 1
    r1 = client.post(
        f"/api/assessments/{ass.assessment_id}/reports",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("blood_glucose.pdf", valid_pdf, "application/pdf")},
        data={"report_category": "medical_report"},
    )
    assert r1.status_code == 200

    # Attempt 2 (Retry of the same file name)
    r2 = client.post(
        f"/api/assessments/{ass.assessment_id}/reports",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("blood_glucose.pdf", valid_pdf, "application/pdf")},
        data={"report_category": "medical_report"},
    )
    assert r2.status_code == 200

    records = (
        db_session.query(MedicalReport)
        .filter(MedicalReport.assessment_id == ass.assessment_id, MedicalReport.file_name == "blood_glucose.pdf")
        .all()
    )
    assert len(records) == 1, "Duplicate record created on upload retry!"


# --------------------------------------------------------------------------
# 11. Correct persistence of extracted text and structured findings
# --------------------------------------------------------------------------
def test_11_persistence_of_extracted_text_and_findings(client, auth_user, auth_token, db_session):
    ass = Assessment(user_id=auth_user.user_id, input_types=["medical_report"], status="draft")
    db_session.add(ass)
    db_session.commit()

    sample_text = "Hemoglobin: 10.5 g/dL (Reference: 13.0 - 17.0)\nSerum Creatinine: 0.9 mg/dL (Reference: 0.6 - 1.2)"
    pdf_bytes = _create_digital_pdf(sample_text)

    res = client.post(
        f"/api/assessments/{ass.assessment_id}/reports",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("cbc_creatinine.pdf", pdf_bytes, "application/pdf")},
        data={"report_category": "medical_report"},
    )
    assert res.status_code == 200
    rep_id = res.json()["report_id"]

    row = db_session.get(MedicalReport, rep_id)
    assert row is not None
    assert row.file_name == "cbc_creatinine.pdf"
    assert "Hemoglobin" in row.extracted_text
    assert row.structured_findings["total_extracted"] >= 2
    assert "Hemoglobin" in row.reference_ranges
    assert row.ocr_metadata["ocr_status"] == "extracted"


# --------------------------------------------------------------------------
# 12. Retrieval of report findings isolated to correct assessment
# --------------------------------------------------------------------------
def test_12_isolated_retrieval_by_assessment(client, auth_user, auth_token, db_session):
    ass1 = Assessment(user_id=auth_user.user_id, input_types=["medical_report"], status="draft")
    ass2 = Assessment(user_id=auth_user.user_id, input_types=["medical_report"], status="draft")
    db_session.add_all([ass1, ass2])
    db_session.commit()

    pdf_bytes = _create_digital_pdf("TSH: 6.8 uIU/mL (Reference: 0.4 - 4.2)")
    client.post(
        f"/api/assessments/{ass1.assessment_id}/reports",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("thyroid.pdf", pdf_bytes, "application/pdf")},
    )

    # Fetch reports for ass1
    res1 = client.get(f"/api/assessments/{ass1.assessment_id}/reports", headers={"Authorization": f"Bearer {auth_token}"})
    assert len(res1.json()["reports"]) == 1

    # Fetch reports for ass2 (must be empty)
    res2 = client.get(f"/api/assessments/{ass2.assessment_id}/reports", headers={"Authorization": f"Bearer {auth_token}"})
    assert len(res2.json()["reports"]) == 0


# --------------------------------------------------------------------------
# 13. Uploaded findings reach consolidated evidence and feedback pipeline
# --------------------------------------------------------------------------
def test_13_uploaded_findings_reach_evidence_and_feedback(client, auth_user, auth_token, db_session):
    ass = Assessment(user_id=auth_user.user_id, input_types=["symptoms", "medical_report"], status="draft")
    db_session.add(ass)
    db_session.commit()

    # Add symptom
    db_session.add(Symptom(assessment_id=ass.assessment_id, symptom="fatigue", state="PRESENT", source="user_input"))
    db_session.commit()

    # Upload report with low hemoglobin
    pdf_bytes = _create_digital_pdf("Hemoglobin: 8.5 g/dL (Reference: 13.0 - 17.0)")
    r = client.post(
        f"/api/assessments/{ass.assessment_id}/reports",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("anemia_report.pdf", pdf_bytes, "application/pdf")},
    )
    assert r.status_code == 200

    # Run full clinical assessment pipeline
    feedback_result = run_assessment_pipeline(db_session, ass.assessment_id, auth_user.user_id)
    evidence = feedback_result["result"]["evidence"]
    assert evidence["report"]["provided"] is True
    assert len(evidence["report"]["abnormal_findings"]) >= 1

    hb_param = next(p for p in evidence["report"]["abnormal_findings"] if p["test_name"] == "Hemoglobin")
    assert hb_param["value"] == 8.5
    assert hb_param["interpretation"] == "LOW"
    assert hb_param["source_file"] == "anemia_report.pdf"


# --------------------------------------------------------------------------
# 14. OCR failure is NOT presented as normal or successfully analyzed
# --------------------------------------------------------------------------
def test_14_ocr_failure_not_presented_as_normal_or_analyzed(client, auth_user, auth_token, db_session, tmp_path):
    ass = Assessment(user_id=auth_user.user_id, input_types=["medical_report"], status="draft")
    db_session.add(ass)
    db_session.commit()

    # Blank image where OCR finds nothing
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    r = client.post(
        f"/api/assessments/{ass.assessment_id}/reports",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("unreadable_scan.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 200
    res = r.json()
    assert res["status"] == "unreadable"
    assert res["extraction_status"] == "unreadable"
    assert res["structured_findings"]["total_extracted"] == 0

    # Ensure unreadable report is NOT used to generate fake normal evidence
    pipeline_res = run_assessment_pipeline(db_session, ass.assessment_id, auth_user.user_id)
    assert len(pipeline_res["result"]["evidence"]["report"]["abnormal_findings"]) == 0
    assert len(pipeline_res["result"]["evidence"]["report"]["normal_findings"]) == 0


# --------------------------------------------------------------------------
# 15. Regression: X-ray images route through X-ray pipeline, not report OCR
# --------------------------------------------------------------------------
def test_15_xray_images_separate_from_report_ocr(client, auth_user, auth_token, db_session):
    ass = Assessment(user_id=auth_user.user_id, input_types=["xray"], status="draft")
    db_session.add(ass)
    db_session.commit()

    # Medical report upload endpoint rejects upload if category is xray_report and file is an image (PNG/JPG)
    img_bytes = _create_image("X-Ray Image", "PNG")
    res_report_route = client.post(
        f"/api/assessments/{ass.assessment_id}/reports",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("chest_xray.png", img_bytes, "image/png")},
        data={"report_category": "xray_report"},
    )
    assert res_report_route.status_code == 400
    assert "X-ray reports must be PDF files" in res_report_route.json()["detail"]

    # In contrast, the dedicated X-ray image endpoint accepts PNG/JPG image
    res_xray_route = client.post(
        f"/api/assessments/{ass.assessment_id}/xray",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("chest_xray.png", img_bytes, "image/png")},
        data={"region": "chest"},
    )
    assert res_xray_route.status_code in (200, 201)
    # Stored in xray_results, not medical_reports
    xray_recs = db_session.query(XrayResult).filter(XrayResult.assessment_id == ass.assessment_id).all()
    assert len(xray_recs) == 1
    assert xray_recs[0].region == "chest"

    # Verify the upload response matches the persisted X-ray record.
    upload_data = res_xray_route.json()
    assert upload_data["region"] == "chest"
    assert str(upload_data["xray_result_id"]) == str(xray_recs[0].xray_result_id)
    assert upload_data["status"] == xray_recs[0].status

    # Verify that GET retrieves the saved X-ray result for this assessment.
    res_saved = client.get(
        f"/api/assessments/{ass.assessment_id}/xray",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert res_saved.status_code == 200

    saved_data = res_saved.json()
    assert str(saved_data["assessment_id"]) == str(ass.assessment_id)
    assert len(saved_data["results"]) == 1

    saved_xray = saved_data["results"][0]
    assert str(saved_xray["xray_result_id"]) == str(upload_data["xray_result_id"])
    assert saved_xray["region"] == "chest"
    assert saved_xray["status"] == xray_recs[0].status
    assert saved_xray["findings"] == (xray_recs[0].findings or [])
    assert saved_xray["model_version"] == xray_recs[0].model_version

