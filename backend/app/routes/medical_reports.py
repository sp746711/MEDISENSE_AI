"""Medical report upload routes — OCR/NLP in Stage 6.

Medical reports and radiology-report PDFs are extracted as text. X-ray image
inference belongs to the dedicated X-ray route and is never run on these PDFs.
"""

import uuid
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.dependencies import CurrentUser, DbSession
from app.database.models import Assessment, MedicalReport
from app.utils.file_validation import validate_report_file

router = APIRouter(prefix="/assessments", tags=["Medical Reports"])
_ALLOWED_CATEGORIES = {"medical_report", "xray_report"}
_READ_CHUNK_SIZE = 1024 * 1024


async def _read_upload_with_limit(file: UploadFile, max_bytes: int) -> bytes:
    """Read at most max_bytes + 1 bytes so oversized uploads are rejected early."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(min(_READ_CHUNK_SIZE, max_bytes + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File exceeds the maximum upload size of {max_bytes // (1024 * 1024)} MB.",
            )
    return b"".join(chunks)


def _owned(db: DbSession, assessment_id: UUID, user_id: UUID) -> Assessment:
    assessment = db.get(Assessment, assessment_id)
    if assessment is None or assessment.user_id != user_id:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return assessment


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
    if category not in _ALLOWED_CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="report_category must be 'medical_report' or 'xray_report'.",
        )

    input_types = assessment.input_types or []
    if category == "xray_report":
        if "xray" not in input_types and "medical_report" not in input_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This assessment was not configured for X-ray or medical report input.",
            )
    elif "medical_report" not in input_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This assessment was not configured for medical report input.",
        )

    filename = Path(file.filename or "report").name
    if category == "xray_report" and Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-ray reports must be PDF files.",
        )

    settings = get_settings()
    max_bytes = int(settings.max_upload_size_mb) * 1024 * 1024
    try:
        content = await _read_upload_with_limit(file, max_bytes)
    finally:
        await file.close()

    validation = validate_report_file(filename, content, settings.max_upload_size_mb)
    if not validation.get("ok"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation.get("message") or "Invalid report file.",
        )
    if category == "xray_report" and str(validation.get("extension", "")).lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-ray reports must be PDF files.",
        )

    reports_dir = Path(settings.upload_path) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}{validation['extension']}"
    dest = reports_dir / stored_name
    dest.write_bytes(content)

    # Local imports avoid making OCR/NLP initialization a requirement for route import.
    from app.ai.report_ocr import extract_text_from_file
    from app.ai.report_nlp import structure_report_text
    from app.database.models import utcnow

    try:
        try:
            ocr_result = extract_text_from_file(str(dest)) or {}
        except Exception:
            # Do not expose filesystem paths or internal exception details to clients.
            ocr_result = {
                "status": "error",
                "extracted_text": None,
                "tables": [],
                "message": "The report could not be read by the text extraction service.",
                "ocr_version": "error",
                "ocr_engine": "none",
            }

        extracted_text = ocr_result.get("extracted_text")
        if not isinstance(extracted_text, str) or not extracted_text.strip():
            extracted_text = None

        try:
            nlp_result = structure_report_text(
                extracted_text, tables=ocr_result.get("tables", []) or []
            ) or {}
        except Exception:
            nlp_result = {
                "status": "error",
                "findings": [],
                "qualitative_findings": [],
                "narrative_findings": [],
                "message": "Text was extracted, but structured finding analysis failed.",
            }

        structured_findings = nlp_result.get("findings", []) or []
        qualitative_findings = nlp_result.get("qualitative_findings", []) or []
        narrative_findings = nlp_result.get("narrative_findings", []) or []
        # Guard against unexpected NLP response shapes rather than crashing during len().
        if not all(isinstance(items, list) for items in (
            structured_findings, qualitative_findings, narrative_findings
        )):
            structured_findings = []
            qualitative_findings = []
            narrative_findings = []
            nlp_result = {
                **nlp_result,
                "status": "error",
                "message": "The extracted text could not be converted into a valid findings structure.",
            }

        total_findings_count = (
            len(structured_findings) + len(qualitative_findings) + len(narrative_findings)
        )
        combined_findings = {
            "lab_parameters": structured_findings,
            "qualitative_findings": qualitative_findings,
            "narrative_findings": narrative_findings,
            "total_extracted": total_findings_count,
        }

        if extracted_text and total_findings_count > 0:
            overall_status = "extracted"
            extraction_status = "successful"
        elif extracted_text:
            overall_status = "extraction_incomplete"
            extraction_status = "incomplete"
        else:
            overall_status = "unreadable"
            extraction_status = "unreadable"

        ocr_metadata = {
            "ocr_status": ocr_result.get("status", "unknown"),
            "ocr_version": ocr_result.get("ocr_version"),
            "ocr_engine": ocr_result.get("ocr_engine"),
            "nlp_status": nlp_result.get("status", "unknown"),
            "pages": ocr_result.get("pages", 1),
            "message": ocr_result.get("message") or nlp_result.get("message"),
            "extraction_status": extraction_status,
        }

        clean_file_name = Path(validation.get("filename") or filename).name
        existing_report = (
            db.query(MedicalReport)
            .filter(
                MedicalReport.assessment_id == assessment.assessment_id,
                MedicalReport.file_name == clean_file_name,
            )
            .first()
        )
        previous_stored_name = existing_report.stored_name if existing_report else None

        reference_ranges = {
            str(f.get("test_name")): f.get("reference_range")
            for f in structured_findings
            if isinstance(f, dict) and f.get("test_name") and f.get("reference_range")
        }

        if existing_report:
            report = existing_report
            report.stored_name = stored_name
            report.file_type = validation["file_type"]
            report.report_category = category
            report.extracted_text = extracted_text
            report.structured_findings = combined_findings
            report.reference_ranges = reference_ranges
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
                reference_ranges=reference_ranges,
                ocr_metadata=ocr_metadata,
            )
            db.add(report)

        if assessment.status == "draft":
            assessment.status = "inputs_received"
        db.add(assessment)
        db.commit()
        db.refresh(report)

        # Remove the replaced upload only after the database commit succeeds.
        if previous_stored_name and previous_stored_name != stored_name:
            old_path = reports_dir / Path(previous_stored_name).name
            try:
                old_path.unlink(missing_ok=True)
            except OSError:
                # A stale file is preferable to reporting a failed successful upload.
                pass

        msg = (
            ocr_metadata.get("message")
            or nlp_result.get("message")
            or "Report uploaded; extracted findings are shown below when available."
        )
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
    except HTTPException:
        db.rollback()
        dest.unlink(missing_ok=True)
        raise
    except Exception as exc:
        db.rollback()
        dest.unlink(missing_ok=True)
        # Keep server-side diagnostics while returning a controlled response.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The report upload could not be saved. Please retry.",
        ) from exc


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
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "has_extracted_text": bool(r.extracted_text),
                "structured_findings": r.structured_findings,
                "ocr_metadata": r.ocr_metadata,
            }
            for r in rows
        ],
    }
