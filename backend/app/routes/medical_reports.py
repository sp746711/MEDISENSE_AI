"""Medical report upload routes — OCR/NLP in Stage 6."""

import uuid
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.dependencies import CurrentUser, DbSession
from app.database.models import Assessment, MedicalReport
from app.utils.file_validation import validate_report_file

router = APIRouter(prefix="/assessments", tags=["Medical Reports"])


@router.post("/{assessment_id}/reports")
async def upload_report(
    assessment_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
    file: UploadFile = File(...),
    report_category: str = Form("medical_report"),
) -> dict:
    assessment = _owned(db, assessment_id, current_user.user_id)
    category = (report_category or "medical_report").strip().lower()
    if category not in {"medical_report", "xray_report"}:
        category = "medical_report"

    input_types = assessment.input_types or []
    if category == "xray_report":
        if "xray" not in input_types and "medical_report" not in input_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This assessment was not configured for X-ray or medical report input.",
            )
    else:
        if "medical_report" not in input_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This assessment was not configured for medical report input.",
            )

    fname_lower = (file.filename or "").lower()
    if category == "xray_report":
        if not fname_lower.endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="X-ray reports must be PDF files.",
            )
        # Prevent normal medical reports from being falsely categorized as xray reports
        if any(marker in fname_lower for marker in ["medical_report", "medical-report", "blood_report", "lab_report", "cbc", "liver_report", "kidney_report"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This file appears to be a general medical report, not an X-ray / radiology report.",
            )

    settings = get_settings()
    content = await file.read()
    validation = validate_report_file(file.filename or "", content, settings.max_upload_size_mb)
    if not validation["ok"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=validation["message"])

    reports_dir = settings.upload_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}{validation['extension']}"
    dest = reports_dir / stored_name
    dest.write_bytes(content)

    from app.ai.report_ocr import extract_text_from_file
    from app.ai.report_nlp import structure_report_text
    from app.database.models import utcnow

    try:
        ocr_result = extract_text_from_file(str(dest))
    except Exception as exc:
        ocr_result = {
            "status": "error",
            "extracted_text": None,
            "tables": [],
            "message": f"OCR processing error: {exc}",
            "ocr_version": "error",
            "ocr_engine": "none",
        }

    extracted_text = ocr_result.get("extracted_text")

    try:
        nlp_result = structure_report_text(extracted_text, tables=ocr_result.get("tables", []))
    except Exception as exc:
        nlp_result = {
            "status": "error",
            "findings": [],
            "qualitative_findings": [],
            "narrative_findings": [],
            "message": f"NLP parsing error: {exc}",
        }

    structured_findings = nlp_result.get("findings", [])
    qualitative_findings = nlp_result.get("qualitative_findings", [])
    narrative_findings = nlp_result.get("narrative_findings", [])
    total_findings_count = len(structured_findings) + len(qualitative_findings) + len(narrative_findings)
    combined_findings = {
        "lab_parameters": structured_findings,
        "qualitative_findings": qualitative_findings,
        "narrative_findings": narrative_findings,
        "total_extracted": total_findings_count,
    }

    clean_file_name = Path(validation.get("filename") or file.filename or "report").name

    # Determine status model
    if extracted_text and total_findings_count > 0:
        overall_status = "extracted"
        extraction_status = "successful"
    elif extracted_text and total_findings_count == 0:
        overall_status = "extraction_incomplete"
        extraction_status = "incomplete"
    else:
        overall_status = "unreadable"
        extraction_status = "unreadable"

    ocr_metadata = {
        "ocr_status": ocr_result.get("status"),
        "ocr_version": ocr_result.get("ocr_version"),
        "ocr_engine": ocr_result.get("ocr_engine"),
        "nlp_status": nlp_result.get("status"),
        "pages": ocr_result.get("pages", 1),
        "message": ocr_result.get("message"),
        "extraction_status": extraction_status,
    }

    # Deduplicate: if this exact filename was previously uploaded for this assessment (e.g. retry), update existing row
    existing_report = (
        db.query(MedicalReport)
        .filter(
            MedicalReport.assessment_id == assessment.assessment_id,
            MedicalReport.file_name == clean_file_name,
        )
        .first()
    )

    if existing_report:
        report = existing_report
        report.stored_name = stored_name
        report.file_type = validation["file_type"]
        report.report_category = category
        report.extracted_text = extracted_text
        report.structured_findings = combined_findings
        report.reference_ranges = {
            f["test_name"]: f.get("reference_range")
            for f in structured_findings
            if f.get("reference_range")
        }
        report.ocr_metadata = ocr_metadata
        report.created_at = utcnow()
    else:
        report = MedicalReport(
            assessment_id=assessment.assessment_id,
            file_name=clean_file_name,
            stored_name=stored_name,
            file_type=validation["file_type"],
            report_category=category,
            extracted_text=extracted_text,
            structured_findings=combined_findings,
            reference_ranges={
                f["test_name"]: f.get("reference_range")
                for f in structured_findings
                if f.get("reference_range")
            },
            ocr_metadata=ocr_metadata,
        )
        db.add(report)

    if assessment.status == "draft":
        assessment.status = "inputs_received"
    db.add(assessment)
    db.commit()
    db.refresh(report)

    msg = ocr_result.get("message") or nlp_result.get("message") or "Medical report successfully processed."

    return {
        "status": overall_status,
        "upload_status": "uploaded",
        "extraction_status": extraction_status,
        "report_id": str(report.report_id),
        "file_name": report.file_name,
        "report_category": report.report_category,
        "message": msg,
        "ocr_metadata": ocr_metadata,
        "structured_findings": combined_findings,
        "extracted_text_snippet": (extracted_text[:200] + "...") if extracted_text else None,
    }


@router.get("/{assessment_id}/reports")
def list_reports(
    assessment_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    _owned(db, assessment_id, current_user.user_id)
    rows = (
        db.query(MedicalReport)
        .filter(MedicalReport.assessment_id == assessment_id)
        .all()
    )
    return {
        "assessment_id": str(assessment_id),
        "reports": [
            {
                "report_id": str(r.report_id),
                "file_name": r.file_name,
                "file_type": r.file_type,
                "report_category": r.report_category,
                "created_at": r.created_at.isoformat(),
                "has_extracted_text": bool(r.extracted_text),
                "structured_findings": r.structured_findings,
            }
            for r in rows
        ],
    }


def _owned(db: DbSession, assessment_id: UUID, user_id: UUID) -> Assessment:
    assessment = db.get(Assessment, assessment_id)
    if assessment is None or assessment.user_id != user_id:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return assessment
